# macOS ScreenCaptureKit 到 MPS/CoreML 零拷贝实现调查报告

## 结论

macOS 原生技术栈支持一条不经过 CPU `numpy` 的捕获到推理路线：

```text
ScreenCaptureKit
-> CVPixelBuffer / IOSurface
-> CVMetalTextureCacheCreateTextureFromImage
-> MTLTexture
-> Metal / MPS preprocess
-> MPSGraph 或 CoreML inference
```

但本项目当前没有实现这条链路。当前实现仍是：

```text
ScreenCaptureKit / CoreGraphics
-> CPU 可读 buffer
-> numpy
-> torch tensor
-> .to("mps")
-> PyTorch MPS 推理
```

也就是说，目前 macOS 是 MPS GPU 推理，不是捕获到推理的 GPU 零拷贝。

## 项目现状

相关文件：

```text
src/capture/backends/macos_screencapturekit.py
src/capture/backends/macos_coregraphics.py
src/stereo_runtime/providers/apple/pytorch_mps.py
```

当前 `macos_screencapturekit.py` 使用 PyObjC 调 ScreenCaptureKit，但帧处理走：

```text
CMSampleBuffer
-> CVPixelBufferGetBaseAddress
-> numpy ndarray
```

`macos_coregraphics.py` 也是 CPU / numpy 路线。

Apple provider 目前只有 PyTorch MPS provider：

```text
pytorch_mps.py
```

它把模型和 tensor 放到 `mps` 设备上执行，但输入来自前面 CPU/numpy 管线后再 `.to("mps")`。

当前没有看到真正的 CoreML depth provider，也没有看到 `MTLTexture`、`CVMetalTextureCache`、`MPSGraph`、`MLModel` 原生桥接实现。

## 每段技术成熟度

### ScreenCaptureKit 捕获

成熟。ScreenCaptureKit 是 Apple 官方屏幕/窗口捕获 API，用于替代更老的截图方式。它输出的 sample buffer 可以携带 `CVPixelBuffer`。

当前项目已经通过 PyObjC 加载并使用 ScreenCaptureKit，因此“Python 调用 macOS 捕获 API”这一步已经成立。

### CVPixelBuffer / IOSurface

成熟。`CVPixelBuffer` 可以由 IOSurface-backed memory 支撑，是视频、图像和 GPU 互操作的基础对象。

对于零拷贝路线，关键不是读取 `CVPixelBufferGetBaseAddress`，而是避免 lock base address 后读 CPU 内存，改为把 pixel buffer 转成 Metal texture。

### CVPixelBuffer 到 MTLTexture

成熟。官方推荐入口是：

```text
CVMetalTextureCacheCreateTextureFromImage
```

它可以从 `CVPixelBuffer` 创建 `CVMetalTexture`，再取出 `MTLTexture`。

目标链路：

```text
CVPixelBuffer
-> CVMetalTextureCacheCreateTextureFromImage
-> CVMetalTextureGetTexture
-> MTLTexture
```

这一步仍需要原生生命周期管理：texture cache、pixel format、plane index、width/height、frame 生命周期。

### Metal / MPS 预处理

成熟。模型输入通常不是屏幕原始 BGRA texture，而是 resize、normalize、layout 后的 tensor。推荐用 Metal compute shader 或 MPS/MPSGraph 做：

```text
BGRA/RGBA texture sample
resize
normalize
NHWC/NCHW layout transform
FP16/FP32 tensor output
```

这一步不是 CPU 拷贝，但通常会有一次 GPU shader/compute 写入。应称为“零 CPU 回读”，不要宣称“完全无 GPU 拷贝”。

### CoreML / MPS 推理

成熟，但工程路径不同：

```text
CoreML
= 高层模型运行时，适合部署 .mlmodel/.mlpackage

MPS / MPSGraph
= 更低层 Metal GPU 计算图/算子路线，控制更强

PyTorch MPS
= Python/PyTorch 生态路线，易集成，但不等于捕获零拷贝
```

如果目标是最少改动，PyTorch MPS 继续保留即可。如果目标是捕获 texture 到推理零 CPU 回读，CoreML 或 MPSGraph 需要原生 bridge。

## Python 调用方式

Python 已经有成熟方式调用 macOS Objective-C framework：

```text
PyObjC
```

项目当前已经使用 PyObjC：

```python
objc.loadBundle("ScreenCaptureKit", ...)
import ScreenCaptureKit as SCK
```

因此可以继续用 Python 调原生 API。但完整零拷贝 pipeline 不建议全写在 Python/PyObjC 里，原因是：

- Metal object 生命周期复杂。
- command queue / command buffer / fence 同步需要严谨管理。
- shader dispatch 和 texture/tensor binding 用 PyObjC 写会很笨重。
- 性能路径不应频繁跨 Python/Objective-C 边界。

推荐形态：

```text
Python 薄调用
-> Objective-C++ / Swift / Rust native bridge
-> bridge 内部管理 ScreenCaptureKit + Metal + MPS/CoreML
```

Python 层只暴露类似：

```python
bridge.start_capture(...)
depth = bridge.next_depth(timeout_ms=...)
bridge.stop()
```

## 推荐实现路线

## 验证前提

这条路线可以先在 Windows 上做方案设计和接口拆分，但不能完整验证。真正验证至少需要一台 macOS 机器，因为以下环节都依赖本机 Apple framework、权限和 GPU runtime：

```text
ScreenCaptureKit 权限和 run loop
CVPixelBuffer / IOSurface 生命周期
MTLTexture 引用和释放时机
Metal command queue / command buffer 同步
CoreML / MPSGraph 在目标芯片上的算子和性能
打包、签名、macOS 版本兼容
```

如果目标是可用的零 CPU 回读 demo，建议用 Apple Silicon Mac 验证。Intel Mac 可以做部分 API 验证，但性能、ANE、CoreML 后端行为和最终目标不同。

### 里程碑 1：保留现状，标注非零拷贝

目标：避免误导。

工作项：

- 文档明确当前 macOS MPS 是 GPU 推理，不是捕获零拷贝。
- 日志或 provider info 可标记：`capture_zero_copy=False`。
- 不改现有 PyTorch MPS 路线。

### 里程碑 2：PyObjC 验证 CVPixelBuffer 到 MTLTexture

目标：证明捕获帧能不读 CPU base address，转换成 `MTLTexture`。

工作项：

- 从 ScreenCaptureKit callback 拿 `CMSampleBuffer`。
- 获取 `CVPixelBuffer`。
- 创建 `CVMetalTextureCache`。
- 调 `CVMetalTextureCacheCreateTextureFromImage`。
- 取 `MTLTexture`。
- 输出 texture width/height/pixel format 诊断。

该阶段只做验证，不接模型。

### 里程碑 3：Metal GPU 预处理 demo

目标：把 `MTLTexture` 转成模型输入 buffer。

工作项：

- 写最小 Metal compute shader。
- 输入 BGRA/RGBA texture。
- 输出 FP16/FP32 tensor buffer。
- 实现 resize / normalize / layout transform 中的最小可用子集。
- 与 CPU 预处理结果比较误差。

### 里程碑 4：CoreML 或 MPSGraph 推理桥

目标：原生端完成推理，Python 只拿结果。

可选方案：

```text
A. CoreML：转换模型到 mlpackage，使用 CoreML runtime
B. MPSGraph：手写/加载计算图，控制更强但工作量更大
C. 继续 PyTorch MPS：最省事，但 texture 零拷贝输入很难完整闭环
```

推荐先走 CoreML，因为 Apple 生态部署更成熟。

第一阶段固定使用 Distill Depth / Distill-Any-Depth Base，不把 InfiniDepth 放进首版范围。

```text
Distill Depth / Distill-Any-Depth Base
-> PyTorch / ONNX
-> CoreML mlpackage
-> CoreML 推理验证
```

原因：

- Distill Depth 类模型结构更常规，PyTorch -> ONNX -> CoreML 成功率更高。
- InfiniDepth 结构更复杂，包含 DINOv3 相关块、implicit decoder、动态 shape 风险，ONNX 阶段已经出现 TracerWarning，CoreML 转换风险更高。
- 零拷贝链路先验证捕获、Metal preprocess、CoreML runtime 是否闭环；模型难度不应一开始成为主阻塞。

这里的“Distill Depth 更简单”只解决模型转换难度，不等于零拷贝 pipeline 已经完成。完整闭环还需要把捕获帧保持在 `CVPixelBuffer` / `MTLTexture` 路径里，并在原生层完成预处理、推理和输出同步。

InfiniDepth 转 CoreML 可能单独卡几天。若 CoreML 不支持关键算子，才考虑 MPSGraph/Metal 自定义推理。

### 里程碑 5：封装 native bridge

目标：提供项目可调用的最小 Python API。

建议 API：

```python
class MacOSMetalDepthBridge:
    def start(self, target): ...
    def next_depth(self, timeout_ms: int): ...
    def stop(self): ...
```

桥内部管理：

```text
ScreenCaptureKit stream
CVMetalTextureCache
MTLDevice / MTLCommandQueue
Metal preprocess pipeline
CoreML / MPSGraph model
同步和资源池
```

## Distill Depth 时间表

目标是先做一个 Distill Depth 版 macOS 原生零 CPU 回读原型：

```text
ScreenCaptureKit
-> CVPixelBuffer / IOSurface
-> MTLTexture
-> Metal preprocess
-> CoreML Distill Depth
-> depth output
-> Python bridge
```

| 阶段 | 目标 | 交付物 | 预估 |
|---|---|---|---|
| 0 | 准备 Mac 验证环境 | 可运行项目、捕获权限、Xcode/Command Line Tools、Python 环境 | 0.5 - 1 天 |
| 1 | Distill Depth 转 CoreML | `.mlpackage`，固定输入尺寸，单张图片输出 depth | 1 - 2 天 |
| 2 | CoreML 单帧推理接入 | 原生或 Python 测试入口，输入图片得到 depth buffer | 1 - 2 天 |
| 3 | ScreenCaptureKit 原生捕获 | 不读 CPU base address，拿到 `CVPixelBuffer` / `IOSurface` | 1 - 2 天 |
| 4 | `CVPixelBuffer` 转 `MTLTexture` | `CVMetalTextureCacheCreateTextureFromImage` 验证，输出 texture 诊断 | 1 - 2 天 |
| 5 | Metal 预处理 | resize、normalize、layout，输出 CoreML 可吃的输入 buffer | 2 - 4 天 |
| 6 | Capture -> Metal -> CoreML 闭环 | 实时帧直接进入 CoreML，输出 depth，避免 CPU 图像回读 | 2 - 4 天 |
| 7 | Python bridge 接入项目 | `bridge.start()` / `bridge.next_depth()` / `bridge.stop()`，接入现有 runtime | 1 - 2 天 |
| 8 | 稳定性整理 | 资源生命周期、同步、错误处理、权限提示、打包说明 | 1 - 3 天 |

合计：`10.5 - 22` 个 AI 工作日。这个估算假设有真实 Apple Silicon Mac 可以连续验证，并且 Distill Depth 转 CoreML 没有关键算子阻塞。

最短可演示版本可以压缩到 `5 - 8` 个 AI 工作日，但只建议作为 demo：

```text
固定显示器捕获
固定输入尺寸
固定模型
只输出 depth 诊断图
不完整接入现有 GUI / OpenXR pipeline
```

InfiniDepth 不建议放进第一阶段。它的 CoreML 转换风险应单独评估，可能额外消耗数天到更久，取决于算子、动态 shape 和精度差异。

## 不建议现在做的事

- 不建议把完整 Metal/MPS/CoreML pipeline 全写成 PyObjC Python 代码。
- 不建议把 PyTorch MPS 路线称为零拷贝。
- 不建议为了“未来零拷贝”重写当前可运行的 macOS capture provider。
- 不建议在没有模型转换验证前承诺 CoreML 性能。
- 不建议在第一阶段用 InfiniDepth 验证 CoreML；先用 Distill Depth / Distill-Any-Depth Base 跑通链路。

## 命名建议

当前路线：

```text
pytorch_mps
```

未来原生零 CPU 回读路线可命名为：

```text
macos_metal_coreml
macos_screencapturekit_metal
macos_metal_mpsgraph
```

不要叫 `xpu`，`XPU` 在项目里是 Intel PyTorch XPU，不是 macOS。

## 风险

- ScreenCaptureKit 权限和窗口/显示器捕获策略会影响可用性。
- CVPixelBuffer pixel format 可能和模型预处理期望不同。
- CoreML 模型转换可能遇到算子、动态 shape、精度问题；InfiniDepth 风险高于 Distill Depth。
- MPSGraph/Metal 路线开发成本高，但控制力最好。
- 原生 bridge 打包要处理 macOS 架构、签名、framework 链接。

## 媒体文件播放的 Apple 路径

本节针对本地 HEVC Main 10 / P010 HDR 文件。Apple 在 [WWDC22: Display HDR video in EDR with AVFoundation and Metal](https://developer.apple.com/videos/play/wwdc2022/110565/) 建议尽可能使用较高层的媒体框架，让框架管理播放和时序；需要逐帧处理时，示例用 `AVPlayerItemVideoOutput` 按播放时间取 `CVPixelBuffer`，再通过 Core Video 和 Metal 做处理。该结构适合 Desktop2Stereo 的“解码、深度推理、合成立体帧”流程。

Apple 的示例通过 `CVMetalTextureCacheCreateTextureFromImage` 将 `CVPixelBuffer` 映射为 Metal texture，并要求 Metal command buffer 完成前保持 `CVMetalTexture` 存活。结合 [CVPixelBuffer](https://developer.apple.com/documentation/corevideo/cvpixelbuffer) 和 [CVMetalTextureCache](https://developer.apple.com/documentation/corevideo/cvmetaltexturecache)，目标媒体路径应保留解码 surface，避免锁定 CPU base address、逐帧复制到 NumPy，再上传到 MPS。这里承诺的是避免 CPU 图像回读；Metal 颜色转换、缩放和模型输入写入仍会产生 GPU 工作。

如果必须直接控制解码，可用 [VideoToolbox `VTDecompressionSessionCreate`](https://developer.apple.com/documentation/videotoolbox/vtdecompressionsessioncreate(allocator:formatdescription:decoderspecification:imagebufferattributes:outputcallback:decompressionsessionout:))，并用 [`VTIsHardwareDecodeSupported`](https://developer.apple.com/documentation/videotoolbox/vtishardwaredecodesupported(_:)) 探测当前系统对 HEVC 的硬解能力。该探测只表示 codec 级支持，不保证当前机器、Main 10 profile、分辨率、帧率都能由硬件解码；运行时必须验证真实解码路径，并保留软件回退。优先先做 `AVPlayerItemVideoOutput` 原型，只有它无法满足取帧或同步需求时再下沉到 VideoToolbox。

HDR 输入应从解码到 Metal 预处理都保留其色彩语义。Apple 提供的 [`kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`](https://developer.apple.com/documentation/corevideo/kcvpixelformattype_420ypcbcr10biplanarvideorange) 对应 10-bit bi-planar video-range YCbCr；WWDC 示例也明确使用 10-bit 配合 PQ/HLG transfer function。Desktop2Stereo 应读取并传递实际的 primaries、transfer function 和 range，在 Metal 中执行 P010 range 展开、BT.2020/PQ 转换及所需缩放，输出与当前模型输入一致的浮点数据。不要为了追 FPS 提前转成 8-bit RGB、跳过 PQ 转换，或更改模型输入尺寸和精度。

`AVPlayerItemVideoOutput` 的 item time 可用于按 PTS 取新帧；显示 link 用于跟随呈现时钟。OpenXR 应继续按目标呈现时间消费到期帧，处理赶不上时复用最新有效立体帧，而不是让视频时钟随推理延迟变慢。分别统计解码、预处理、推理、合成、唯一内容帧率和显示提交帧率，才能确认 1080p 内容帧率持续达到 30 FPS。

Core ML 的 [`MLComputeUnits`](https://developer.apple.com/documentation/coreml/mlcomputeunits) 可约束模型允许使用的计算单元；`cpuAndNeuralEngine` 明确包含 CPU 且不含 GPU，所以它不是“无 CPU 处理”保证。可把 Core ML 与 MPS 并发作为真机 A/B 变量，但 Apple 的 API 文档不保证特定模型实际落在哪个单元，也不保证更快。若严格要求视频像素数据不经 CPU，必须避免 CPU pixel-buffer 映射；推理计算单元仍需根据模型支持、实际调度和设备 profile 单独验证。

### 对当前实现的影响

- 当前 FFmpeg raw P010 pipe 会把整帧字节交给 Python，再进入 NumPy/MPS；仅打开 FFmpeg 的 VideoToolbox 解码不会去掉这条 CPU 管线。
- AVFoundation/VideoToolbox surface + `CVMetalTextureCache` + Metal shader 是适配当前 macOS 的方向，但现有 PyTorch MPS provider 接收 tensor，并没有把 `MTLTexture` 直接作为模型输入的接口。完整零 CPU 回读需要原生 Swift/Objective-C++ bridge，并将预处理、模型调用和资源同步留在同一 Metal/Core ML 路径中。
- Apple API 让这条路线可实现，但不等于 30 FPS 保证。HEVC Main 10/4K 硬解能力、Core ML 模型转换和精度、GPU/ANE 调度、OpenXR 合成仍需在目标 Apple Silicon 设备上分别测量。

## 参考资料

- ScreenCaptureKit: https://developer.apple.com/documentation/screencapturekit
- AVPlayerItemVideoOutput: https://developer.apple.com/documentation/avfoundation/avplayeritemvideooutput
- VideoToolbox hardware decode capability: https://developer.apple.com/documentation/videotoolbox/vtishardwaredecodesupported(_:)
- Core Video 10-bit bi-planar video-range pixel format: https://developer.apple.com/documentation/corevideo/kcvpixelformattype_420ypcbcr10biplanarvideorange
- Core ML compute units: https://developer.apple.com/documentation/coreml/mlcomputeunits
- WWDC22 EDR video and Metal workflow: https://developer.apple.com/videos/play/wwdc2022/110565/
- CoreVideo CVMetalTextureCache: https://developer.apple.com/documentation/corevideo/cvmetaltexturecache
- Metal Performance Shaders MPSImage: https://developer.apple.com/documentation/metalperformanceshaders/mpsimage
- Core ML: https://developer.apple.com/documentation/coreml
- PyObjC: https://pyobjc.readthedocs.io/
