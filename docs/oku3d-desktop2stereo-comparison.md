# Oku3D、Video Stereo Converter 与 Desktop2Stereo 对比报告

日期：2026-11-14

## 结论摘要

Desktop2Stereo 不应把 Oku3D 或 Video Stereo Converter 的算法直接搬进来。两者适合参考的是“播放器/工作流如何组织用户体验”，而不是替代 Desktop2Stereo 现有的实时 GPU 立体链路。

Desktop2Stereo 当前优势是：

- 捕获、媒体、深度推理、立体合成和 Vulkan/OpenXR 呈现处于同一个实时运行链路。
- CUDA/ROCm 均支持，模型和合成路径以 GPU 常驻为主。
- 支持本地屏幕、窗口、媒体文件，以及 OpenXR、网络和 Moonlight 输出。
- 已经具备动态分辨率、深度时间状态、边缘重建和本地/XR 统一呈现代码。

Oku3D 值得借鉴的是：

- 播放列表是独立的一等信息架构，不只是“打开一个文件”。
- 当前播放项与用户选择项分离。
- 文件/文件夹导入、排序、拖放、保存列表、失败项汇总和空状态都明确可见。
- 桌面侧栏和 VR 空间面板使用同一逻辑状态。

Video Stereo Converter 值得借鉴的是：

- 处理阶段边界清晰：抽帧、深度、SBS、分块编码、音视频拼接各自独立、可恢复。
- 参数具有明确的用户语义：`max_disparity`、`convergence`、`depth_gamma`、`edge_softness`、`artifact_smoothing`、`super_sampling`、`sharpen`。
- 对失败、缺失深度帧、磁盘空间和恢复点有明确处理。

## 证据边界

### Oku3D

目标：`C:\Program Files\Oku3D\oku3d.exe`

已观察到：

- 签名 Windows x64 应用，版本 4.0.0。
- SHA-256：`80e88f2caa42f3f9c4bcfa2399daac9d62aec8c4fd9b1e0ba2ad84da3d0e8655`。
- 打包字符串与 `CHANGELOG.md` 证明播放列表面板、添加/移除/拖放、排序、自动前进、播放列表格式和 VR 侧面板行为存在。
- 当前环境没有可用的 Ghidra/IDA provider，因此没有把静态字符串或产品说明外推为已验证的深度算法实现细节。

### Video Stereo Converter

目标：公开源码 `https://github.com/jens-duttke/Video-Stereo-Converter`。

已核实源码事实：

- `frame_extractor.py` 用 FFmpeg 抽帧到 PNG。
- `depth_map_generator.py` 使用 ONNX Runtime 和 DepthPro ONNX 模型，默认输入 1536×1536，CPU NumPy/OpenCV 预处理后转成 float16。
- `helper/stereo_core.py` 使用 PyTorch 深度归一化、gamma、Gaussian 深度软化、`scatter_` 前向 splatting、CPU OpenCV bilateral filter 和 Telea inpaint，再裁剪成左右视图并横向拼接。
- `chunk_generator.py`、`video_concatenator.py` 通过分块编码和音视频拼接输出文件。
- README 明确说明该仓库是离线批处理工作流，会产生大量中间帧和磁盘占用。

### Desktop2Stereo

通过当前仓库源码核实的链路：

- 媒体文件通过 FFmpeg/PyNvVideoCodec 解码；NVIDIA HDR 已改为 PyNvVideoCodec `NATIVE` P010 显存帧，AMD 使用 AMF 硬解后上传 GPU。
- HDR 预处理在 GPU Triton 路径完成 YUV、BT.2020 → BT.709、Hable tone-map 和可选的 UI 缩放。
- 深度模型支持 TensorRT、MIGraphX、CUDA/ROCm PyTorch 等后端。
- 立体合成沿用现有 DIBR/深度重建路径，不依赖磁盘中间 PNG。
- Local Viewer、OpenXR、网络输出共享运行时结果和媒体控制状态。
- HDR PyNvVideoCodec NATIVE P010 的 DLPack 视图在 RTX 2070 上报告类似字节步长的 `(W, 2)` stride；现在先按底层 row-major storage 重解释为 `(W, 1)` 再进入 Triton，否则会出现重复/紫色绿色串色，已用 4K HDR 样片确认修复。

## Oku3D 的 3D 运行时观察

Oku3D 的安装包和 changelog 还提供了以下与 3D 处理有关的静态证据：

- 使用 Rust、`wgpu`/`naga`、FFmpeg、D3D11/D3D12、DirectML 和 OpenXR 相关依赖，说明它和 Desktop2Stereo 一样属于实时 GPU 播放器，而不是离线抽帧脚本。
- 二进制中的模块名包含 `depth_preprocess.rs`、`inference_region.rs`、`temporal_stabilize.rs`、`edge_dilation.rs`、`edge_inset.rs`、`fine_structure.rs`、`line_probe.rs`、`motion.rs`。这些名字表明深度处理包含推理前后清理、边缘/细结构处理和时间稳定阶段，不能简化为“一次深度推理后直接 warp”。
- 立体呈现相关字符串包含 `Mesh Warp`、`grid-vertex`、`grid-index`、`warp-uniform-left`、`warp-uniform-right`、`stereo-split-pipeline`、`carried-depth-chain`、`depth-perspective-pipeline` 和左右眼 sharpen。最保守的结论是：Oku3D 使用 GPU 网格/逐眼 warp 进行立体化；精确网格尺寸、位移公式和遮挡填补仍未验证。
- Oku3D 文档将 Depth/3D intensity 定义为画面宽度的百分比（0–2%），Convergence 最大约画面宽度的 20%，并提供 Depth Gamma、逐帧/显示帧限制和 Motion-Sync 3D。Motion-Sync 会在文件/流播放中延迟画面和声音以等待匹配深度，在屏幕捕获设备上关闭。
- 它有 `HDR passthrough` 和 BT.709 → BT.2020 相关字符串，OpenXR 被当作普通亮度输出而非 HDR 显示路径。HDR 的精确 tone-map 和显示色彩管理仍未从二进制中解出。

因此，可比较的结论是：Oku3D 的核心优势在于“实时播放器 + 独立深度 worker + 深度清理/时间稳定 + GPU 网格 warp + HDR-aware 呈现”，而不是某个可以直接复制的滤镜或 shader 常量。

## Oku3D 播放列表设计观察

Oku3D 的播放列表是一个独立面板/状态，不是播放按钮旁边的临时下拉框：

- 有序队列，且区分当前播放条目与用户选择条目。
- 支持文件、文件夹、拖放、文件夹递归、扫描进度和跳过项统计。
- 支持添加、删除选中项、拖动排序、排序菜单、随机播放和保存。
- 支持上一项、下一项和自动前进；重播会停在当前片段开头，不跨到上一项。
- 列表为空时有明确空状态；失败项和列表读取失败有可恢复提示。
- 播放列表格式包括 M3U/M3U8、PLS、XSPF、WPL、ASX、MPC-HC 和 PotPlayer；CUE 可读为曲目列表。
- 桌面端可把播放列表显示为侧栏、分离窗口或屏幕旁面板；VR 中对应为可移动的空间侧面板。

## 与 Desktop2Stereo 的差异

| 维度 | Oku3D | Video Stereo Converter | Desktop2Stereo 当前实现 |
|---|---|---|---|
| 使用方式 | 桌面播放器/实时播放 | 离线批处理脚本 | 实时捕获与实时立体运行时 |
| 播放列表 | 一等面板、可选择、排序、保存 | 不适用 | 已加入文件夹队列、当前项、上一/下一项和自动前进 |
| 深度来源 | 应用内部模型/实时处理，具体静态实现未完全验证 | DepthPro ONNX，1536×1536 | TensorRT/MIGraphX/PyTorch 等运行时深度后端 |
| 立体方法 | 播放器内实时转换，细节未完全验证 | 前向 splatting + CPU 后处理 | 现有 DIBR/深度重建和 GPU 合成链 |
| 中间文件 | 强调无需大量中间帧 | PNG 帧、深度图、SBS 图、分块视频 | 不落盘，帧在运行时队列中流动 |
| HDR | 产品说明支持 HDR | 主要按图像/视频工作流处理 | GPU P010 → BT.2020 → BT.709/Hable |
| 输出 | 本地播放、SBS/TB/anaglyph、VR 等 | 导出 SBS 视频文件 | Local Viewer、OpenXR、RTMP/Moonlight/MJPEG |
| 交互重点 | 播放器、队列、侧面板 | 参数调试和可恢复批处理 | 实时源切换、Local/XR 输出、播放控制 |

## 可安全借鉴的设计

1. **保留一等队列模型**
   - 队列条目、当前播放索引和用户选择是三个独立概念。
   - 选择文件后自动把同目录的视频/图片作为候选队列，是一个低复杂度、可预测的默认行为。

2. **文件夹导入可视化**
   - 扫描文件夹时显示扫描状态、有效项数和跳过项数。
   - 当前实现采用后台线程扫描，避免阻塞播放和 Vulkan 呈现。

3. **当前项与选择项分离**
   - 当前项使用主色高亮和播放图标。
   - 选择项只用于打开、删除或后续批量操作，不自动改变播放位置。

4. **失败恢复**
   - 单个不可读文件不应导致整个队列崩溃。
   - 显示跳过数量和可重新选择目录的入口。
   - 播放下一项前检查路径仍存在。

5. **桌面与 XR 共用状态**
   - 不在 Local Viewer 和 OpenXR 各维护一份队列。
   - 两者都通过 session/revision 媒体控制文件和媒体状态快照同步。

6. **参数语义保持稳定**
   - 可以借鉴 VSC 的 `convergence`、`depth_gamma`、`edge_softness`、`artifact_smoothing` 命名和用户解释，但不能把它的 CPU 前向 splatting 替换掉 Desktop2Stereo 现有 GPU DIBR 链。

## 不应直接照搬的部分

- 不引入 VSC 的 PNG 帧、深度图和 SBS 中间目录作为实时播放路径。
- 不把 VSC 的 CPU `scatter_`/OpenCV bilateral/inpaint 放到实时 4K 热路径。
- 不因 Oku3D 产品有某种参数就假定 Desktop2Stereo 当前模型输出语义相同。
- 不复制 Oku3D 的品牌、图标、面板文案或专有资源。
- 不把播放列表保存格式扩展排在实时播放稳定性之前。

## 对当前实现的建议

本轮已经落地：

- 媒体播放器模式的文件浏览器不再调用系统文件对话框。
- 文件浏览器增加最近打开列表，支持单项打开和删除。
- 文件浏览器支持目录、视频、图片、ISO 和原盘目录识别。
- 同一目录媒体自动组成播放列表，支持上一项、下一项、清空和结束后自动前进。
- Local Viewer 与 OpenXR 共用媒体状态和播放控制。
- 播放条与浏览器面板采用半透明深色面板，并遵循现有 Inter/色彩令牌。

后续最值得补的两项：

1. 增加显式的“选择项”与批量移除/排序操作。
2. 增加可选的工作流/队列持久化，而不是默认自动恢复整个队列。
