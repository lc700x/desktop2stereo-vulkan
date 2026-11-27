# Desktop2Stereo 各运行模式完整性测试报告

- 日期：2026-11-28（Asia/Singapore）
- 仓库：`F:\desktop2stereo-vulkan`
- 测试 GPU：AMD Radeon RX 9060 XT，ROCm / MIGraphX
- 头显：Meta Quest 2，Horizon OS Android 14 / SDK 34
- OpenXR Runtime：Virtual Desktop `VirtualDesktopXR`
- 测试视频：4K 120 FPS HDR10《文字大战》，应用按 60 FPS 播放
- 测试原则：不使用模拟器结果替代 Quest 实机结论；失败或阻塞项不标记为通过。

## 1. 总结

| 模式 | 结论 | 关键结果 |
|---|---|---|
| Local Viewer | PASS | 4K 输入，真实呈现；最后 5 秒约 `27.8 FPS`，输出打包尺寸 `1920x960` |
| OpenXR Link / 单 SBS Projection | PASS | VDXR 实机正常；最后 5 秒 `59.8 FPS`；`xr_projection_p95=3.21 ms`，`xr_frame_p95=16.42 ms` |
| MJPEG Streamer | PASS | Quest Browser 通过 LAN 播放；左右两半分别为绿/洋红，内容不同 |
| RTMP Streamer / WebRTC | PASS | Quest Browser 建立 WebRTC `peer connection`，播放 H.264；媒体链路可用 |
| Moonlight Streamer | FAIL / BLOCKED | Sunshine、桥接和 AMF 编码可工作（约 `56 FPS`），但 Moonlight XR 实机画面为黑；真实 D2S runtime 在自定义端口下无法被 Moonlight XR 正常连接 |
| OpenXR Screen Quad | FAIL | 生产代码硬编码禁用；强制打开后内部计数为 active，但 Quest 实机整屏黑，`presents=0`、`rc=3` |

补洞设置已在真实 Quest 的 Projection 运行中确认：

```text
rt_fill=none
rt_hole_fill=none(0/0.00)
rt_no_fill=used
```

## 2. 自动化回归

### 2.1 清晰度、菜单和 OpenXR 定向回归

```text
703 passed, 1 skipped
```

覆盖：

- 屏幕超采样策略
- OpenXR Vulkan
- GUI 状态本地化
- 桌面设置菜单
- XR 设置菜单
- Overlay 纹理

### 2.2 五种模式专项回归

```text
1323 passed, 38 skipped
```

覆盖 Local Viewer、OpenXR、MJPEG、RTMP、Moonlight、运行时路由、输出尺寸、串流校准、音频检测、Vulkan 输出能力等。

### 2.3 全量测试

```text
3232 passed, 88 skipped, 13 failed
```

13 个失败与此前基线一致，集中在 macOS Metal/CoreML 和 POSIX 安装脚本：

- `test_adapter_coreml_plumbing`
- `test_host_frame_packer`（8 个参数化用例）
- `test_host_frame_packer`（1 个 native half-SBS 用例）
- `test_macos_coreml_io`
- `test_macos_metal_viewer`
- `test_windows_standalone_installer` 中的 POSIX 安装脚本断言

这些失败未指向本轮 Windows Vulkan/OpenXR 改动，但仍是跨平台发布前必须清零的回归项。

88 个跳过项主要来自：

- macOS：Quartz、ScreenCaptureKit、AVFoundation、Cocoa、Metal/MPS、PyObjC
- NVIDIA CUDA/NVENC 专属路径（当前推理环境为 ROCm）
- HIP/OpenGL interop 不可用
- macOS runtime 修复和 Unix 特权符号链接
- 一个非 Windows 回退测试
- 一个 OpenXR Vulkan 适配器不一致场景

## 3. Local Viewer

真实 9060 XT 运行结果：

- 输入：`3840x1920`
- 输出眼尺寸：`1920x960`
- Half-SBS 打包尺寸：`1920x960`
- `captures=1142`
- `presents=715`
- 最后 5 秒约 `27.8 FPS`
- `rc=0`

功能结论：PASS。画面、打包和呈现链路工作正常；性能明显低于 OpenXR 单 SBS 路径，Local Viewer 的最终呈现与 GPU 等待链仍值得继续优化。

## 4. OpenXR Link

### 4.1 单 SBS Projection（真实 Quest 2 / VDXR）

运行参数：

```text
D2S_OPENXR_PROJECTION_SBS_SINGLE_SWAPCHAIN=1
D2S_OPENXR_SCREEN_OVERSAMPLE=1.5
```

实机结果：

- Runtime：`VirtualDesktopXR`
- Projection：`array_size=1`
- Swapchain：`5238x2857`
- 单眼区域：`2619x2857`
- 最后 5 秒呈现：`59.8 FPS`
- `xr_projection_p50=2.01 ms`
- `xr_projection_p95=3.21 ms`
- `xr_frame_p50=13.91 ms`
- `xr_frame_p95=16.42 ms`
- `screen_new=60.0`
- `captures=1114`
- `presents=1101`
- `rc=0`

### 4.2 左右眼不同

Quest 实机截图按左右眼半屏裁切后的差异：

- 分辨率：`3664x1920`
- 左眼/右眼各 `1832x1920`
- 平均绝对像素差 RGB：`[2.121, 5.174, 9.386]`
- 差异区域覆盖主体屏幕和空间背景

同时，单 SBS 帧缓冲左右半边差异：

- 单眼尺寸：`2619x2857`
- 平均绝对像素差 RGB：`[1.174, 2.824, 4.380]`

结论：左右眼不是简单复制，具有明确的立体视差。

### 4.3 逐眼 Projection

逐眼 Projection 路径已进入初始化，但两次真实 Quest 验收都被 OpenXR 会话焦点阻塞：

```text
OpenXR session is not focused
xrEndFrame state=2/3
presents=0
```

这是本轮实机运行顺序造成的会话焦点问题，不能据此判定逐眼路径损坏。此前模拟器与自动化测试已经覆盖该路径；正式发布前仍需在 Quest 前台重新做一次干净的逐眼 Projection 实机测试。

## 5. OpenXR Screen Quad

### 5.1 生产开关行为

设置：

```text
D2S_OPENXR_SCREEN_QUAD_REPROJECTION=1
```

实际输出：

```text
[OpenXRViewer] Screen Quad Reprojection disabled:
Virtual Desktop stereo Quad swapchain is unsupported; using Projection swapchain
screen_quad_requested=0
screen_quad_active=0
```

因此当前环境变量不能打开 Quad。源码常量为 `False`，此前的判断属实。

### 5.2 强制诊断覆盖

仅为诊断，在测试进程内把常量覆盖为 `True`，并使用左红/右绿 Quad 眼诊断：

```text
D2S_OPENXR_SCREEN_QUAD_REPROJECTION=1
D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC=1
```

内部结果：

- `screen_quad_requested=1`
- `screen_quad_active=1`
- `screen_quad_new≈48.6 FPS`
- `xr_projection_p95≈2.27 ms`
- 无内部 `quad_failed`
- 无内部 `quad_fallback`

Quest 实机结果：

- ADB 截图整屏全黑
- 环境、控制器和屏幕全部不可见
- `presents=0`
- `rc=3`

结论：当前 VDXR/Quest 2 组合不能接受现有的 stereo Quad swapchain 提交方式。硬编码禁用是正确的保护，不应为了启用开关而直接删除。需要先实现真正的运行时能力探测，或者改为 VDXR 可接受的单层纹理/图层布局，并通过至少一帧可见性验证后再启用。

## 6. MJPEG Streamer

测试方法：

- 主机启动 `MjpegDirectSbsOutput`
- 生成左右不同色块的 SBS 帧
- Quest 2 使用系统 Browser 打开 LAN URL

Quest 实机结果：

- 首页 HTTP `200`
- `/stream.mjpg` 返回 JPEG
- 画面尺寸 `640x360`
- 左半平均 RGB 约 `[253.8, 0.0, 0.2]`
- 右半平均 RGB 约 `[0.2, 0.0, 253.8]`
- Quest Browser 显示绿/洋红两半

结论：PASS。

## 7. RTMP Streamer / WebRTC

测试方法：

- 主机启动 MediaMTX + FFmpeg
- 默认 WebRTC 协议
- 合成 `1280x720@30` H.264 视频
- Quest 2 Browser 打开 MediaMTX 播放页

实机结果：

- MediaMTX WebRTC 会话创建
- `peer connection established`
- Quest 读取 `1 track (H264)`
- Quest Browser 正常显示四段绿/洋红测试图
- 另一次 RTSP 探测读到 H.264 `640x360@30`
- 音频测试读到 `Opus 48000 Hz 2ch`；由于系统输出静音，音频内容为静音，但轨道和 WASAPI loopback 链路存在

结论：PASS。浏览器消费端和音视频封装链路可用。

## 8. Moonlight Streamer

### 8.1 主机编码与桥接

合成 SBS 帧测试：

- Moonlight XR 成功发现并连接 Sunshine
- 协商：`1280x720@60`、HEVC、约 `11.8 Mbps`、AMD AMF
- D2S encoded-frame bridge connected
- 主机侧约 `56 FPS` encoded/sent，`dropped=0`
- 接收端出现 `D2SF` 帧头、IDR 标记和有效负载

结论：Sunshine、D2S bridge、AMF 编码和帧传输均在工作。

### 8.2 Quest 显示结果

Moonlight XR 连接后实际画面为黑屏；主机仍持续编码和发送，因此不是“没有帧”问题，而是 Quest 客户端显示/解码或协议兼容问题。

### 8.3 真实 D2S runtime 测试

真实 Moonlight 模式启动后：

- D2S Sunshine 正常启动
- 因本机已有 ApolloService 占用默认端口，D2S 自动选择自定义端口族
- Moonlight XR 能发现主机，但连接自定义端口失败
- 手动输入 `host:port` 被客户端拒绝
- 使用 `adb reverse` 把默认端口映射到自定义端口后，Moonlight XR 仍无法稳定完成连接

结论：FAIL / BLOCKED。需要解决：

1. Moonlight XR 对自定义 Sunshine 端口族的兼容性。
2. 黑屏是解码、surface 提交还是协议元数据问题。
3. 真实 D2S 视频在 Moonlight XR 上的端到端画面验收。

## 9. 发现的问题与优先级

### P0：Quad 不能启用

- 内部 active 但 Quest 全黑
- 生产硬编码禁用必须保留
- 需要能力探测或更换图层方案

### P1：Moonlight XR 黑屏

- 编码与发送正常
- 客户端不显示画面
- 需要确认解码输出、HEVC NAL 封装、时间戳和 surface 格式

### P1：Moonlight 自定义端口不可用

- 默认端口被其他 Sunshine/Apollo 服务占用时，Moonlight XR 无法连接 D2S 的备用端口族
- GUI 需要提示端口冲突，或自动采用客户端可发现的端口策略

### P2：Local Viewer 4K 呈现性能

- 同机 OpenXR 单 SBS 可接近 60 FPS
- Local Viewer 最后 5 秒约 27.8 FPS
- 继续检查 Local Viewer 的 GPU wait、present pacing 和输出打包

## 10. 证据文件

Quest 实机截图：

- `runtime-diagnostics/quest2-during-vdxr-projection.png`
- `runtime-diagnostics/quest2-during-vdxr-projection-left.png`
- `runtime-diagnostics/quest2-during-vdxr-projection-right.png`
- `runtime-diagnostics/quest2-during-vdxr-quad-eye-diag.png`
- `runtime-diagnostics/quest2-mjpeg-browser-2.png`
- `runtime-diagnostics/quest2-webrtc-browser.png`
- `runtime-diagnostics/quest2-moonlightxr-streaming-2.png`

运行数据和日志：

- `src/desktop2stereo/runtime-diagnostics/quest2-vdxr-projection-sbs.json`
- `runtime-diagnostics/quest2-vdxr-projection-sbs.log`
- `src/desktop2stereo/runtime-diagnostics/quest2-vdxr-projection-stages.json`
- `runtime-diagnostics/quest2-vdxr-projection-stages.log`
- `src/desktop2stereo/runtime-diagnostics/quest2-vdxr-quad-request.json`
- `runtime-diagnostics/quest2-vdxr-quad-request.log`
- `src/desktop2stereo/runtime-diagnostics/quest2-vdxr-quad-forced-eye-diag.json`
- `runtime-diagnostics/quest2-vdxr-quad-forced-eye-diag.log`
- `runtime-diagnostics/mode-completeness-full-pytest.log`
