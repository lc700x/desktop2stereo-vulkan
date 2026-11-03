# Desktop2Stereo Vulkan 项目日志

## 2026-11-01（对比 2026-10-17 打包版本）

- 新增 Moonlight Streamer：Windows/macOS 随包启动 Sunshine 主机，通过 4 位 PIN 配对并以 SBS 输出串流，不创建虚拟显示器；Windows 使用 NVENC/AMF，macOS 使用 VideoToolbox。
- 修复 Moonlight AMD AMF 编码卡顿和串流停滞；提升 macOS 串流吞吐，并改进重连时的节奏控制。
- OpenXR 新增 CUDA→D3D12 跨适配器 GPU 输出桥；运行时选择改为优先使用当前活动的 ALVR，并平滑实时深度调节。
- Vulkan Glow 改用独立异步计算队列；Windows Graphics Capture 回调移出 GIL，减少捕获线程争用。
- ROCm 包由宽泛的 `rocm-device-all` 改为明确的 RDNA1–3 17 架构版本，gfx1200/gfx1201 与 CUDA 包继续独立提供。

## 2026-08-28

- 完成 Presenter-owned Vulkan 输出图像直通：Vulkan layered image pass 现在明确将合成结果写入持久化 `VulkanExportableImage`，通过 timeline/外部 semaphore 交接给输出消费者，不再把输出图像读回 CPU；新增 `vulkan_output_image_direct`、`vulkan_zero_cpu_readback` 和 `vulkan_gpu_to_cpu` 遥测。CUDA→Vulkan 输入仍可能包含一次设备侧导入复制，因此严格 `zero_copy` 仍保持 false，避免把“零 CPU 回读”误报成端到端零复制。
- 已在当前 Vulkan 实机完成输出阶段验收：96×64 CUDA RGB/depth 输入经 `d2s_stereo_layered_output` 写入 Presenter-owned storage image，提交 timeline=3，生产路径报告 `vulkan_readback=none`；临时 host image 只用于测试读取像素，不进入运行时输出链。

- 补充 Vulkan 零拷贝输出验收说明：Presenter/OpenXR 输出适配器现在统一报告 `vulkan_output_image_direct=True`、`vulkan_zero_cpu_readback=True` 和 `vulkan_gpu_to_cpu=False`；针对 Vulkan 输出环、同步交接和运行时契约的回归测试通过（35 passed）。严格端到端 `zero_copy=True` 仍不作虚假声明，因为普通 PyTorch CUDA 输入到 Vulkan 外部输入缓冲仍可能包含一次设备侧复制。
- 修复本地模式选择“拉伸铺满”后仍出现黑边的问题：Viewer 现在同时识别中文本地化标签和旧英文配置值，`3840×2160` SBS 会使用完整源矩形直接拉伸到 `3840×2400` swapchain，不再因模式解析失败回退到保持比例（完整）布局。

- 修复 RTX 3090 本地 SBS 仅约 1 FPS：实测确认 `auto` 立体合成后端错误地优先选择了 `vulkan_layered_stereo`，其本地 4K 路径发生约 935 ms/帧的主机读回，使合成耗时达到约 1.04 秒，而 TensorRT 深度推理仅约 2 ms。自动选择顺序恢复为厂商 GPU Triton（NVIDIA CUDA/AMD ROCm）→ Vulkan → OpenGL → Torch；Intel 等无 Triton 设备仍使用 Vulkan。

- 重新启用本地 Viewer 的“保持比例/显示适配”选项及 `完整/铺满/拉伸` 热切换：无窗口性能对照已排除该 blit 功能是 1 FPS 的直接原因，实际根因是 NVIDIA 自动立体合成后端错误优先 Vulkan 并发生 4K 主机读回。

## 2026-08-27

- 修复切换到简体中文后“保持比例”控件仍显示英文 tooltip：语言刷新现在通过 `CompactDropdown.set_tooltip()` 重建下拉控件，使中文显示适配说明实际生效。

- 修复本地 Viewer 启动时 `DISPLAY_MODE` 未从运行时导出模块导入，导致进入推理运行阶段后抛出 `NameError` 并立即退出的问题。

- 优化显示适配控件：移除冗余的“显示适配”标签，直接显示适配选项；中文 tooltip 补充“保持比例（完整/铺满）”与“拉伸铺满”的作用，以及黑边、裁剪和画面变形的区别。

- 修正 Windows Python 环境安装流程：CUDA 与 ROCm 独立安装器现在都直接部署官方 Python 3.12.10 NuGet 完整运行时，不再要求 AMD 用户先运行 CUDA 安装器；每次安装均从全新运行时开始，保留完整 `Lib`、`DLLs`、`include`、`libs` 和 `ensurepip`，其中包含 `_ssl.pyd` 所需的 `libcrypto-3.dll` 与 `libssl-3.dll`。
- 修复 Windows 跨显示器移动 GUI 时“窗口从 4K 显示器移到 1K 显示器后缩小，但控件和文字不随窗口整体缩放”的问题：根因不是 Flet 版本、Flet 客户端文件、GUI 固定尺寸或捕获子进程中的 `SetProcessDpiAwareness(2)`，而是 Windows 在用户注册表 `HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers` 中，针对本工程 `src\python3\python.exe` 路径残留了 `HIGHDPIAWARE` 兼容性覆盖。删除该路径对应的注册表值并彻底重启 Python/Flet 进程后，恢复由 Windows 自动整体缩放窗口内容。诊断此类问题时应先比较新旧可执行文件路径的 AppCompatFlags，避免修改 Flet 文件、恢复 GUI 自行计算 DPI 缩放或删除捕获运行时所需的 DPI 设置。
- 修复 NativeNVENC 带音频推流的卡顿和音画不同步：按 RFC 3550 使用同一媒体时钟关联音频/视频 RTP 时间戳，新增每轨 RTCP Sender Report（NTP↔RTP 映射）及同一同步上下文的 SDES CNAME，并通过独立优先级写队列让 Opus/RTCP 不再同步阻塞于 4K 视频批量发送；RTSP interleaved 通道按 RFC 7826 保持视频/RTCP 与音频/RTCP 配对。另修正 SDES RTCP `length` 字段漏计 4 字节 SSRC 的错误，避免 MediaMTX 报 `rtcp: packet too short`。
- 修复 NativeNVENC 异步视频发送造成的 WebRTC 读取端积压：视频 RTP 队列改为有界背压策略（默认深度 3），队列满时阻塞视频提交线程等待发送端腾出空间（保留帧内 RTP 分片完整性），并按 RTP 时间戳对应的统一媒体时钟节流；音频和 RTCP 仍优先发送，避免 MediaMTX 报 `reader is too slow` 时继续无限堆积视频。视频 RTP sequence number 延迟到实际发送时分配，避免丢帧或退出时制造假的 `RTP packets lost` 序号空洞。该行为与 FFmpeg rawvideo stdin 的阻塞式背压一致，不再用主动丢帧掩盖编码发送端过载。
- 完成 NativeNVENC 视频发送合规验证：新增序号连续性、完整视频帧分组、RTP 分片边界和背压测试；本地全量 `1304 passed`，GitHub Windows NVENC 构建与合规检查均成功。
- 修复进入高级网络推流后仍显示旧 CRF/码率的问题：切换到 RTMP Streamer 时立即应用当前匹配且稳定的自动校准 profile；同时兼容旧 profile 中已废弃的 `Stereo Output` 等指纹字段。安全目标码率为 38 Mbps 时现在同步使用 CRF 20，不再沿用旧配置的 CRF 26。
- 修复切换高级网络推流时 `CompactDropdown` 报 `Control must be added to the page first`：下拉控件在页面挂载前修改 `value`、`options` 或 tooltip，或被旧流程直接调用 `update()` 时，都不会再把 Flet 的 `AssertionError` 抛出；页面挂载后仍正常刷新。
- Local Viewer 新增输入显示器刷新率提醒：从 GLFW 映射后的实际输入显示器读取刷新率；当自动调节后的动态 `capture_target` 高于输入刷新率时，显示醒目的中英文弹窗和状态警告，但不停止运行。输入与 SBS 输出刷新率警告同时发生时按队列依次显示，避免后到警告被当前弹窗吞掉。

## 2026-08-26

- 将可靠下载对策同步到其他平台安装入口：Linux CUDA 同样预装 TensorRT 构建工具、关闭隔离构建并校验版本；Linux/Windows ROCm 和 macOS MPS 不引入 NVIDIA 专属步骤，但统一移除 `--no-cache-dir`，改用项目环境持久缓存、二进制优先以及 10 次重试和 180 秒超时，避免网络中断后反复下载已经完成的大包。
- 修复 Windows CUDA 独立安装器反复失败于 TensorRT 隔离构建依赖下载的问题：失败源是 18 KB TensorRT 前端源码包创建临时构建环境时，再次通过镜像下载 wheel/setuptools，连接截断后外层 requirements 重试只会重复同一步。安装器现在先独立安装并重试固定版本的 setuptools、wheel 和 packaging，CUDA requirements 使用 `--no-build-isolation` 复用这些构建工具；CUDA 下载启用持久 pip 缓存和二进制优先，使成功的大包不会在下一次尝试重新下载；面向海外发布的默认主源改为官方 PyPI，并保留 PyTorch CUDA、NVIDIA 官方源以及华为云、阿里云国内镜像作为 `extra-index-url` 候选源，最后显式校验 TensorRT 版本。
- Windows Local Viewer 新增 `VK_EXT_full_screen_exclusive` 主路径：修正 raw direct-display 将 `VK_NV_acquire_winrt_display` 错当 Instance 扩展的检测，按目标 `HMONITOR` 查询 Surface 独占能力，以 application-controlled Swapchain 完成 acquire/release，并在独占丢失、过期或次优状态时自动重建；申请失败继续使用持久 borderless，不中断运行。RTX 3090 实机已确认 `3840x2160` 独占 Swapchain 与 CUDA external-image Present 成功；同时规避 NVIDIA 独占模式下 MAILBOX 阻塞，VSync 关闭使用 IMMEDIATE、开启使用 FIFO。
- Local Viewer 新增 SBS 输出显示器刷新率检查：首个稳定 FPS 统计窗口后，将实际输出显示器刷新率与 SBS 合成/提交帧率比较；输出刷新率低于实测 SBS FPS，或低于建议最低 60 Hz 时，只显示一次带醒目警告图标的本地化弹窗，并保留 GUI 状态提醒和警告日志。弹窗明确说明程序会继续运行，用户关闭提醒不会停止当前输出。
- 修复鼠标长按停止按钮时被 GUI 刷新取消的问题：停止按钮不再随运行状态反复切换 `disabled` 或被无变化重绘，按压期间的 Flet 点击手势可持续到鼠标释放；按钮在闲置状态保持可点击但安全地不执行停止，避免启动状态切换再次打断长按。
- 修复高级网络推流的 30 FPS 浏览器/校准上限扩散到本地模式：网络校准结果不再写入所有模式共用的 `Target FPS`，改用高级网络推流专属 `Stream Target FPS`；Local Viewer、3D Monitor 和 OpenXR 保持独立帧率配置，旧配置缺少新字段时仍兼容回退。默认本地帧率恢复为 Auto，RTX 3090 本地 SBS 不再被网络校准固定在 30 FPS。
- 修复 NativeNVENC 原生 RTSP 发布器的握手失败：RTSP 请求曾把 `\\r\\n` 作为字面量发送，MediaMTX 因无法识别请求行结束而报 `buffer length exceeds 64`；现已恢复真实 CRLF，NativeNVENC 可进入 ANNOUNCE/SETUP/RECORD 阶段。该问题发生在编码前，不是 CUDA surface 或 NVENC 花屏问题。
- 修复 NativeNVENC 原生 RTSP 发布器的 SETUP 失败：视频和音频 RTP over TCP 的 `Transport` 请求补充 `mode=record`，避免 MediaMTX 报 `transport header contains a invalid mode (null)` 并返回 400；该问题发生在 RTSP 会话协商阶段，不是 NVENC 编码或音频采集故障。
- 修复 NativeNVENC 原生 Opus 运行库路径错误：发布产物位于 `streaming/native_rtsp_output/opus.dll`，加载器此前却查找 `streaming/opus.dll`，导致存在 DLL 仍报 `Could not find module 'opus.dll'`；现改为优先使用功能目录绝对路径，并保留 `D2S_OPUS_PATH` 覆盖。
- 修复 NativeNVENC 原生 RTP 包头错误：H.264 和 Opus 包此前只设置 Marker 位、未写入 SDP 声明的动态 payload type，MediaMTX 因而持续报 `received RTP packet with unknown payload type: 0`；现分别写入 H.264 PT 96 与 Opus PT 111，并串行化音视频 RTSP interleaved socket 写入，避免并发包边界互相穿插。
- 优化 NativeNVENC 原生推流的音频连续性：4K H.264 RTP 不再对每个约 1400 字节分片单独调用 `sendall`，改为保持 RTP/RTSP 包边界的约 64 KiB 批量写入，显著减少每秒 Python/GIL 和 socket 争用；本地 PCM UDP 接收缓冲扩大到 1 MiB，降低视频突发期间 WASAPI 采集线程饥饿和 Opus 音频丢包导致的卡顿。
- 修复 NativeNVENC 4K 推流仍持续触发 WASAPI discontinuity 的核心瓶颈：Annex-B 解析器此前逐字节用 Python 扫描整帧，200 KB 测试帧单次约需 52 ms，超过 25 FPS 的 40 ms 帧周期并持续占用 GIL；现改用底层 `bytes.find` 按 NAL 起始码定位，同时兼容三字节与四字节 start code，避免视频解析饿死音频采集线程。
- 对齐 Vulkan/FFmpeg 音频时间轴行为：Native Opus 不再把每个 4096 帧 WASAPI 采集块拆出的多个 960 帧包瞬间连发，而是按 48 kHz RTP 时钟严格每 20 ms 发送一包，调度严重落后时重置节拍以避免追赶式突发；视频 RTSP/TCP 聚合批量由 64 KiB 降至默认 16 KiB（可用 `D2S_NATIVE_RTP_BATCH_BYTES` 调整），降低共享 socket 锁对音频包的最长等待。
- 修复 Native Opus 在无 discontinuity 日志时仍周期性卡顿：WASAPI 的 4096 帧输入约每 85 ms 成块到达，旧实现仍会把采集块边界映射成 RTP 发包抖动；现新增默认 5760 帧（120 ms）的时钟化抖动缓冲，接收 UDP 与每 20 ms Opus 发包解耦，并在调度晚于 5 ms 时放弃追赶式连发。罕见欠载会用静音补齐当前包并保留连续 RTP 时间轴，可通过 `D2S_NATIVE_AUDIO_PREBUFFER_FRAMES` 调整预缓冲。
- 修复网络推流测速被请求捕获帧率封顶后错误降档：例如捕获固定 30 FPS 时，旧逻辑只能测得约 30 FPS，再应用 90% 安全系数后长期选择 25 FPS。网络测速阶段现在临时将捕获提高到“请求值 + 5 FPS”（30→35），测速结束后恢复手动捕获值或保留自动模式的 +5 FPS 余量；实测 SBS 生产率达到请求上限时，编码目标直接使用请求上限，使浏览器可获得完整 30 FPS。

## 2026-08-25

- NativeNVENC 路径改为不启动 FFmpeg：CUDAARRAY NVENC 输出的带时间戳 H.264 Annex-B 包由原生 RTSP/RTP 发布器直接发送到 MediaMTX，WASAPI PCM 由原生 Opus 编码器编码为 RTP 音频；新增 `native_rtsp_output.py` 与远程构建、发布的 `opus.dll`。NativeNVENC 运行失败时不再偷偷切换 FFmpeg，避免“NativeNVENC”路径与 FFmpeg 复用混用。当前仍需 GitHub Windows 构建产物和实机浏览器回归验证 H.264/Opus 两轨道。

- 扩展 NativeNVENC CUDAARRAY bridge ABI 3：编码提交继续使用 NVIDIA `NV_ENC_PIC_PARAMS.inputTimeStamp`，同时读取 `NV_ENC_LOCK_BITSTREAM.outputTimeStamp/outputDuration`，通过 `d2s_nvenc_cudaarray_read_packet_timed` 将 H.264 包的 PTS/DTS/duration 传回 Python；旧的无时间戳读包接口保留兼容。该时间戳元数据为后续原生 MPEG-TS/RTSP mux 接入准备，未完成 mux 接入前不标记音频 NativeNVENC 路径为稳定。

- 关闭 WASAPI 正常运行期间周期性的 `WASAPI PCM: packets=... peak=...` 日志，保留启动状态和实际 recording discontinuity 异常告警，减少高级网络推流日志刷屏。

- 记录并修复高级网络推流无声音问题：实机日志确认 WASAPI 回环采集正常（`peak` 非零、`silent_packets=0`，MediaMTX 也发布了 `H264 + Opus` 两条轨道），真正原因是 NativeNVENC/PyNvVideoCodec 只向 FFmpeg 传递没有 PTS/DTS 的 H.264 Annex-B 裸包；`-genpts`、`-fps_mode cfr` 和音频 UDP 分片无法可靠修复 stream-copy 输入的时间戳，RTSP 复用器因此持续报警 `Timestamps are unset in a packet for stream 0`，浏览器虽能看到 Opus 轨道但无法正常播放声音。对策是：启用桌面音频时停止 NativeNVENC 裸包复用，自动切换到与 Vulkan 相同的 FFmpeg 音视频公共路径，由 FFmpeg 为 rawvideo/PCM 建立统一时间轴，同时仍优先使用 `h264_nvenc` 硬件编码；无音频 video-only 会话继续保留 NativeNVENC 零拷贝路径。

- 修复 NativeNVENC H.264 裸包进入 FFmpeg RTSP muxer 时没有 PTS/DTS 的问题：启用 `genpts` 并固定 CFR 视频时间轴，避免视频时间戳未定义导致 Opus 已存在但浏览器无法正常播放音频。

- NativeNVENC muxer 不再丢弃 FFmpeg stderr：新增音频启用状态、输入 UDP 地址和实时 mux 警告/错误日志，用于确认 Opus 输入是否因时间戳、队列或 RTSP 复用失败。

- 修复 NativeNVENC 音频复用与 Vulkan 稳定路径不一致的问题：WASAPI PCM 按 240 帧拆分为 MTU 友好的 localhost UDP 数据报，避免 4096 帧/约 16 KiB 数据报分片；NativeNVENC 的 FFmpeg muxer 增加音频线程队列、快速探测和 `muxdelay/muxpreload=0`，降低视频包突发时音频输入被饿死或丢包的概率。

- 修复 Windows 网络推流音频回环在 GPU 高负载下容易出现 `data discontinuity in recording` 的问题：WASAPI/SoundCard 采集块默认从 1024 增大到 4096 帧（可通过 `D2S_WASAPI_BLOCKSIZE` 调整），并新增实际扬声器、PCM 包数、峰值、静音包数和断续次数日志；MediaMTX 仍必须确认最终有 `Opus` 音频轨道。

- 补齐 Intel 网络推流合规边界：新增 `src/tools/intel_vulkan_onevpl_smoke.py`，可在真实 Windows Intel 目标机连续验证 Vulkan producer readiness、D3D11/oneVPL Adapter LUID、NV12 提交、oneVPL 出包，并可将 H.264 包送入已运行的 MediaMTX；在目标机验收完成前仍严格记录 `zero_copy=False`。

- Intel oneVPL 的视频-only native surface 在启用桌面音频时自动关闭 native gate，当前帧回到共享 Intel QSV/FFmpeg 音视频路径，避免无声推流或误终止会话；运行时清单升级为 schema 2，记录 FFmpeg 官方源码 ref、构建元数据和包 SHA-256，并在解包前校验。

- Intel Vulkan→D3D11→VideoProcessor→oneVPL 路径收紧零拷贝状态：Vulkan producer timeline 必须完成，D3D11/oneVPL 必须匹配同一设备和 Adapter LUID，oneVPL 提交新增 NV12 texture 同设备校验；当前如实记录 `gpu_to_cpu=False zero_copy=False gpu_copy_count=1`，等待真实 Intel 目标机连续帧和画面验收后再开放 `zero_copy=True`。

- Vulkan ABI 5 bridge 改为复用公共 FFmpeg 运行时：Windows 功能目录只保留 bridge DLL，公共 FFmpeg/MinGW DLL 统一从 `streaming/rtmp/ffmpeg/bin` 加载，Vulkan Loader 使用显卡驱动的系统版本；Linux bridge 只保留 so，依赖统一发布到 `streaming/rtmp/ffmpeg/lib`。CI 新增 Windows 固定版本依赖闭包校验与 Linux `ldd` 闭包校验，移除功能目录中约 50 MB 的重复依赖。GitHub Actions run `32844878204` 已通过并由机器人提交 `e2c34fb` 回写两平台产物；本机在未设置桥接覆盖变量时自动加载 ABI 5 成功，相关回归 `41 passed`。

- 将开发、诊断和合规工具统一迁移到 `src/tools`，删除可发布应用包内原有的工具重复命名空间；pytest 同时加入 `src` 与应用源码根目录后，Intel、Vulkan、合规和 shader 测试可在同一次全量收集中稳定导入。

- 调整高级网络推流 `Auto` 实现顺序为“厂商原生 GPU → Vulkan → 通用 OpenGL/FFmpeg 硬件 → CPU”：NVIDIA 会先尝试 NativeNVENC/CUDAARRAY SurfaceKernel，成功时不再加载 Vulkan；厂商路径失败后才懒加载 ABI 5 Vulkan bridge。Vulkan Windows/Linux bridge 由 GitHub Actions 发布到对应 `streaming/vulkan_ffmpeg_bridge/<platform>` 功能目录，动态依赖复用 `streaming/rtmp/ffmpeg` 公共运行时，运行时自动查找并加载，无需手工设置 DLL 路径；同时消除同一 fallback 状态被 GUI 日志重复输出的问题。

- NVENC CUDAARRAY 进入严格零拷贝阶段：新增原生 CUDA surface kernel，直接读取最终 SBS tensor 的 device pointer、stride 和 dtype 并写入 NVENC 注册表面，不再创建 RGBA staging 或调用图像复制；成功路径记录 `gpu_to_cpu=False zero_copy=True gpu_copy_count=0`，运行时失败依次降级到 CUDAARRAY 单次设备拷贝、PyNvVideoCodec 和 FFmpeg。远程构建工具链升级到 CUDA 12.4.1，以兼容 GitHub Windows runner 的 MSVC 14.44 标准库；GitHub Actions run `32828611676` 已成功编译并提交 ABI 2 DLL，本机 RTX 3090 已完成 640×360 CUDA tensor 直接写表面并输出 198-byte H.264 Annex-B 包的真实验证。

- 高级网络推流新增原生 NVENC CUDAARRAY 优先路径：OpenGL 映射纹理以 `NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY` 直接注册到 NVENC，删除 array→线性 CUDA tensor 的一次设备内拷贝；当前如实记录 `gpu_to_cpu=False zero_copy=False gpu_copy_count=1`，DLL 缺失或能力失败时自动回退 PyNvVideoCodec。GitHub Actions run `32825850824` 已完成 Windows x64 DLL 编译并提交到对应 streaming 功能目录，本机 ABI 1 与 NVENC/CUDA 驱动探针加载通过。

- 修正高级网络推流 `Auto` 默认进入 PyNvVideoCodec 后出现花屏的问题：恢复稳定 FFmpeg/MediaMTX 默认路径，PyNvVideoCodec、Vulkan、Intel 等直接 GPU 后端改为显式选择，避免初始化成功但帧格式未经画面验证的路径成为默认输出。
- 校正高级推流 `Auto` 策略：恢复从 Vulkan 入口开始的能力链，由 Vulkan 输出对象负责 OpenGL/厂商 GPU、FFmpeg 硬件和 CPU 回退；PyNvVideoCodec 仍可通过编码器下拉框显式覆盖。
- 修复 PyNvVideoCodec GPU 推流的 CUDA stream 交接：RGB→NV12 完成后等待 PyTorch 当前 stream，再提交给独立的 NVENC 队列；同时使用输入 CUDA tensor 的设备索引创建编码器，避免异步读取未完成 plane 或跨 GPU 读取导致花屏。

- 高级推流自动校准指纹新增当前选择显示器的输入分辨率档位（`1K`/`2K`/`4K`）；点击运行时重新读取显示器分辨率，档位变化会强制进入自动校准。

## 2026-08-24

- 合并 GUI 推流与高级网络推流的共享后端决策：两种模式共用网络会话、MediaMTX、音频、自动校准、最新帧消费和生命周期；`Auto` 下 GPU 模式按 NVIDIA/AMD/Intel 选择零拷贝后端，显式 Vulkan/Intel/FFmpeg 选择对两种模式一致。
- 修正 GPU 推流 GUI 的网络设置行映射：自动校准结果、重新校准提示和校准状态现在与高级网络推流一致显示。

## 2026-08-23

- 建立 Vulkan→D3D11 外部资源公共 contract：显式携带 Win32 memory handle、格式/尺寸、分配大小、producer-ready 同步句柄和 Adapter LUID；缺少真实 LUID、BGRA8 格式或生产者同步时不会误报 Intel 零拷贝，最终 Vulkan surface 接入仍待完成。

- 校正 Vulkan/D3D11 句柄方向：`OPAQUE_WIN32` 句柄不能直接作为 `ID3D11Device1::OpenSharedResource1` 的 D3D11 纹理句柄；contract 现在可拒绝非 `D3D11_TEXTURE` 类型，后续应采用 D3D11 创建共享纹理、Vulkan 导入的路径。

- D3D11 SBS bridge 的自有 BGRA surface 增加 NT shared resource 标志，并新增 `shared_handle` 导出；这是后续 Vulkan 以 `D3D11_TEXTURE` 类型导入的入口，尚未宣称最终 Vulkan→D3D11 零拷贝链路完成。

- Vulkan 侧新增 `VulkanD3D11ImportedImage`，通过 dedicated allocation 和 `VkImportMemoryWin32HandleInfoKHR` 导入 D3D11-owned BGRA8；导入要求真实 Vulkan/D3D11 Adapter LUID 一致，producer-ready 同步与最终运行时接线仍待完成。

- D3D11 SBS bridge 新增 owned BGRA texture C ABI/Python 访问器，使 Vulkan 写入共享纹理后可以交回同一 D3D11 VideoProcessor；远程 C++ 编译和真实 GPU 同步仍需验证。

- D3D11 SBS bridge 新增按 Adapter LUID 创建设备的入口，Vulkan/Intel 网络路径不再依赖 DXGI 默认适配器选择。

- Intel 网络输出新增 Vulkan eyes → D3D11 shared BGRA SBS → NV12 → oneVPL 接线；当前是无 CPU 回读的两次 GPU blit 兼容路径，明确记录 `zero_copy=False gpu_copy_count=2`，严格零拷贝融合 shader 仍待实现。

- 高级网络模式在启用 Intel oneVPL final-SBS 时新增 Vulkan 延迟请求和独立图像环：StereoRuntime 不再先回读 Vulkan 眼图，Intel sink 直接 dispatch 到 D3D11-owned SBS；当前仍明确记录 `zero_copy=False gpu_copy_count=2`。
- Intel native workflow 增加上述运行时接线文件的触发路径，后续网络桥接变更会自动进入 GitHub Windows runner 的远程构建验证。
- 增加 Vulkan 延迟 SBS 桥的能力失败回退：初始化、尺寸或提交失败时关闭可选桥、记录原因并恢复常规 Intel QSV/D3D11 输出，不再让网络输出线程退出。
- 将 D3D11/Vulkan 共享导入回归测试限定在 Windows；Linux 合规 runner 不再把平台保护错误误判为 Adapter LUID 回归。
- 修正 Intel native workflow 的 artifact 提交竞态：Windows 编译期间若 `main` 有新提交，二进制 job 会先 rebase 再推送。
- 新增 GitHub-only Vulkan shader binary workflow；packed SBS shader 由远程 `glslc`/`spirv-val` 编译并提交，避免本地生成 SPIR-V。

- 修正 Intel Windows native workflow 仍使用 Node.js 20 运行时的 GitHub Actions 警告：`checkout` 升级到 v5，`upload-artifact` 升级到 v6，`download-artifact` 升级到 v7；远程 C++ 编译逻辑不变。

- Intel oneVPL final-SBS native 路径新增 Adapter LUID 导出与运行时一致性校验；D3D11 surface 与 oneVPL encoder 不属于同一适配器时立即拒绝该路径并回退，日志明确记录校验结果。

- 补充最终 SBS 外部 D3D11 BGRA8 texture 导入契约：surface bridge 可从调用方 D3D11 device 创建，并校验外部纹理的设备、格式、尺寸和 Adapter LUID；Intel oneVPL 输出消费者新增 `native_final_sbs_surface` 入口，满足契约时记录 `gpu_to_cpu=False zero_copy=True gpu_copy_count=0`，否则继续使用原有 RGB 上传回退。

- 修正外部 D3D11 device 创建 surface 时的 immediate context 获取方式：改用 `ID3D11Device::GetImmediateContext`，避免把不存在的 COM QueryInterface 当作 context 来源。

- Desktop Duplication 捕捉链改为优先使用同一份借用 D3D11 帧：原生纹理在保持 `AcquireNextFrame` 生命周期期间同时交给 OpenVINO 推理，并通过受控 staging readback 生成兼容 BGR 帧，避免此前“兼容捕捉一次、原生推理再捕捉一次”的不同帧问题。该阶段仍明确为 `gpu_to_cpu=True`、`zero_copy=False`、`gpu_copy_count=1`；新增 readback C ABI 的导出检查，C++ 编译继续只由 GitHub Actions 远程 workflow 负责。

- 修正 OpenVINO/D3D11 ctypes 与 C ABI 的真实运行时契约：`last_error` 使用两参数签名，`set_texture` 正确识别返回值 `1` 为成功，并增加对应回归测试。

- GitHub Windows native workflow 修正 oneVPL 安装前缀的 PowerShell 参数传递，避免远程 dispatcher 已成功编译却因检查错误路径而提前失败。

- 根据远程 OpenVINO archive 的实际布局补充 `Release/Debug` library 搜索路径，使官方 Windows C++ package 能进入 D3D11 bridge 配置阶段。

- 远程构建验证发现 bridge 使用了旧的 `openvino/preprocess` 头路径；已改为当前 `openvino/core/preprocess` 路径，并继续使用官方 Windows C++ archive 作为远程编译依赖。

- OpenVINO 远程 workflow 保持 archive 下载路径，避免为 native bridge 重复构建完整 runtime；native bridge 本身仍只在 GitHub Windows runner 编译。

- 远程 OpenVINO bridge 编译补充官方 Khronos OpenCL-Headers，满足 OpenVINO D3D11/OpenCL interoperability 头文件依赖。

- 修正远程 OpenVINO bridge 的 OpenCL C++ 头依赖：除官方 Khronos OpenCL-Headers 外，GitHub Actions 现在同步拉取 OpenCL-CLHPP，并把 `CL/cl2.hpp` 的 include 根目录传给 CMake。

- 修正 D3D11 OpenVINO bridge 对 `ID3D11Device::GetImmediateContext` 返回类型的假设；该 API 返回 `void`，远程 MSVC 编译现在按 COM 输出指针是否为空进行校验。

- 补齐 OpenVINO D3D11 bridge 的 OpenCL 链接依赖：GitHub Actions 现在远程构建官方 Khronos OpenCL-ICD-Loader，并将 `OpenCL.lib` 传入 bridge，解决 OpenCL C++ wrapper 的 `clRelease*` 链接错误。

- 修正 Windows native workflow 的产物验证环境：GitHub runner 的普通 PowerShell 不保证 `dumpbin.exe` 在 PATH 中，验证步骤现在显式定位 Visual Studio 工具后检查 DLL 导出和 oneVPL 链接。

- GitHub Actions run `32644145650` 已成功完成 Intel Windows native bridge 远程编译与校验：Desktop Duplication、D3D11 SBS surface、oneVPL、OpenVINO 四个 DLL 均生成，C ABI 导出和 oneVPL `libvpl.dll` 链接检查通过；仍待 Intel 真机验证驱动、RemoteTensor 和硬件编码运行时。

- Intel native artifact 改为扁平可部署目录，附带 `manifest.json` 与 SHA-256 清单；运行时新增共享 `D2S_INTEL_NATIVE_ARTIFACT_DIR` 搜索路径，四个 Intel bridge 不再需要分别配置 DLL 路径。

- Intel Windows native workflow 增加发布同步 job：远程构建和导出校验成功后，自动按功能把 bridge、`libvpl.dll`、manifest 和校验清单提交到 `capture/native/desktop_duplication` 及 Intel provider 下的三个 native 功能目录，对外发布程序可直接从对应运行时目录加载。

- SHA-256 发布清单改为只记录 DLL 和 `libvpl.dll` 等二进制文件，避免 Git 文本换行规范化造成 manifest 哈希误报。

- 新增 `src/tools/intel_native_runtime_probe.py`：目标 Intel Windows 机器可用 `--strict` 一次检查按功能发布目录、DLL 哈希、Desktop Duplication、OpenVINO RemoteTensor、D3D11 SBS Surface 和 oneVPL 能力；默认模式输出完整 JSON 诊断但不因当前机器缺少 Intel 驱动而失败。

- 修正 Intel runtime probe 的独立启动路径，直接从仓库根目录执行时同时加入 `src` 和 `src/desktop2stereo`，兼容项目现有顶层 `stereo_runtime` 导入。

- Intel 发布同步现在同时携带官方 OpenVINO runtime DLL（核心 runtime、GPU/CPU plugin、frontend 和 TBB 依赖，如 archive 中提供），并放入 OpenVINO D3D11 功能目录；目标机仍需安装匹配 Intel 驱动和 OpenCL ICD。

- 对齐当前 OpenVINO C++ API 头文件布局，将 PrePostProcessor include 修正为 `openvino/core/preprocess/pre_post_process.hpp`，并同步远程开发头检查路径。

- 新增 Intel Windows 零拷贝捕捉—推理计划书 `docs/18-intel-windows-zero-copy-capture-inference-plan.md`；GUI 捕捉源加入 `DesktopDuplication` 选项，并新增 DXGI/D3D11 原生桥 C ABI、CMake 工程、ctypes 能力探测、原生纹理借用帧契约和 OpenVINO RemoteTensor 能力层。启用 `D2S_INTEL_NATIVE_OPENVINO=1` 且原生 DLL/模型可用时，原生 D3D11 帧进入 OpenVINO provider；现阶段输出仍回读 CPU，因此明确记录 `zero_copy=False`。Desktop Duplication 遇到 `DXGI_ERROR_ACCESS_LOST` 时会重建输出并自动重试一次。

- OpenVINO Intel 路径新增可选 `native/openvino_d3d11_bridge` C ABI/CMake 工程、D3D11 VideoProcessor 的 BGRA8→NV12 GPU 转换、NV12 RemoteTensor 接线、输出 shape/float buffer ABI，以及 `OpenVINOD3D11DepthProvider` 适配器；Desktop Duplication 暴露同一 D3D11 device 供 provider 复用，新增 DXGI Adapter LUID 校验、原生帧异常安全释放、运行时调试字段和借用 NV12 D3D11 surface 导出契约。OpenVINO SDK 尚未安装，因此该 DLL 尚未编译；直接 oneVPL/QSV 真机提交仍待验证。

- Intel QSV 新增 `Intel QSV (D3D11)` 视频编码后端和 GUI 选项：最终 SBS RGB24 经 FFmpeg D3D11/QSV surface 进入 H.264/H.265，日志明确 `gpu_to_cpu=True zero_copy=False gpu_copy_count=1`。新增 `native/d3d11_sbs_surface`，已实测完成最终 SBS BGRA8→NV12 D3D11 surface；设置 `D2S_ONEVPL_FINAL_SBS=1` 且 oneVPL SDK/DLL 可用时，可进一步走 oneVPL surface→压缩包→MediaMTX，否则回退 QSV。当前仍保留 RGB stdin，尚未宣称捕捉到编码的严格零拷贝。

- 新增可选 `native/onevpl_d3d11_encoder` bridge：配置 oneVPL SDK 后可接收 OpenVINO bridge 生成的借用 NV12 D3D11 surface，调用 oneVPL D3D11 编码并返回压缩包；未配置 SDK、DLL 或 Intel 驱动时安全回退。新增 `.github/workflows/intel-windows-native.yml`，以后通过 GitHub-hosted Windows runner 拉取官方 oneVPL dispatcher、编译 Intel 原生桥并上传 artifact；本地不再作为 C++ 编译依据。

- 保留独立的“GPU 推流”用户模式，同时抽出共享网络会话配置和模式策略；高级网络推流与 GPU 推流现在都可通过 WebRTC 使用同一套自动网络校准控制器，GPU 校准阶段使用独立的 FFmpeg CBR 压力流，不会误启动 PyNvVideoCodec/AMF 生产编码器。

- 自动校准结果区域新增重新校准提醒，明确提示更换路由器、头显、浏览器、系统、输出参数、GPU、驱动或性能模式后重新执行校准。

- 将 `docs/16-advanced-streaming-vulkan-image-path.md` 升级为完整的 `docs/16-network-streaming-specification.md`，补充 GPU 推流、共享会话层、MediaMTX/音频契约、WebRTC 自动校准、零拷贝边界、故障分类和双模式验收矩阵。

- 修复 Windows 防火墙检测在没有匹配入站阻止规则时误报 PowerShell exit code 1 的问题；探针现在对空结果显式返回 `[]`，自动校准可正常继续。

## 2026-08-22

- 统一修正源码、构建脚本、测试和发布文档中的 `src/desktop2steoro` 目录引用为 `src/desktop2stereo`；保留 `desktop2steoro-vulkan` 项目名、历史记录和合规检查中的兼容映射不变。

- 修复 Requirements Compliance GitHub Actions 在源码目录改名为 `src/desktop2stereo` 后仍引用旧 `src/desktop2steoro` 路径的问题；shader 编译、合规检查和测试矩阵现在使用新路径。

- 修复目录改名后遗留在 shader manifest 校验器和 requirements 路径检查器中的旧源码路径；本地校验与 CI 校验现在都能解析 `src/desktop2stereo`。

- 修正项目布局配置中的应用目录，测试和工具不再通过 `project_paths.env` 回退到已不存在的 `src/desktop2steoro`。

- 修正 CPU 帧进入 OpenGL fallback 时的活动路径日志：不再把可用但未使用的 CUDA interop 报成 GPU-only，实际 host-upload 现在明确输出 `interop=none gpu_to_cpu=True zero_copy=False gpu_copy_count=0`。

- 为 `opengl_fallback_rtsp_soak.py` 增加 `--cpu` 诊断模式；Windows 实测 CPU RGB 640×360@30 通过 10/10 帧，日志为 `path=host-upload`，MediaMTX 确认 H264 发布，使无 CUDA 平台也能验证完整回退链。

- 补齐 `VulkanDirectSbsOutput.submit_frame()` 的 CPU/非 CUDA 入口：不再错误进入 Vulkan rawvideo 编码命令，而是统一经过 OpenGL 能力探测并进入 host-upload 回退；后续 CPU 帧复用同一稳定输出，覆盖 AMD/Intel/macOS 无 CUDA 场景的实际回退边界。

- OpenGL CUDA/HIP fallback 复用按分辨率和设备缓存的 RGBA8 GPU staging buffer，去掉每帧 `torch.cat` 的 4K 临时分配；新增 staging 复用断言，保持输入、OpenGL texture 和编码提交全程不落 CPU。

- 增加 OpenGL interop 运行中断回归测试：提交阶段发生 graphics resource/fence 错误时，关闭 OpenGL 资源、熔断本次 OpenGL 会话并将当前帧交给稳定高级 FFmpeg 路径，避免推流线程退出或重复抖动切换。

- 真实 4K OpenGL fallback 闭环复测通过：RTX 3090/WGL/3840×2160@30 连续 60 帧，实际选择 CUDA–OpenGL interop → PyNvVideoCodec/NVENC，日志为 `gpu_to_cpu=False zero_copy=False gpu_copy_count=2`，MediaMTX 确认 H264 发布，优化 staging buffer 后耗时 2.08 秒（此前 2.44 秒）。

- 修复 `opengl_fallback_smoke.py` 直接从仓库根目录或任意工作目录启动时的模块路径问题；诊断工具现在会自动加入 `src/desktop2steoro`，可直接执行 OpenGL context、PBO/fence 和 GPU/host fallback 探测。

- 根据 NVIDIA PyNvVideoCodec 官方 GPU 编码输入契约补充 zero-copy 边界：当前 Python API 只接收 NV12 plane 的 CUDA Array Interface 设备指针，不接收 `cudaArray_t`/OpenGL texture handle；严格 zero-copy 下一步必须使用原生 NVENC `CUDAARRAY` bridge，现有路径继续明确报告 2 次 GPU copy。

- OpenGL 能力报告新增 `gpu_copy_count`：CUDA/HIP interop 明确记录当前 2 次 GPU copy，host-upload 记录 0 次 GPU copy；同步扩展 fallback 候选日志和 smoke JSON，避免把 GPU-only 误报成严格 zero-copy。

- 修正 `opengl_fallback_rtsp_soak.py` 的 `--force-host` 环境变量生命周期：只有输出对象成功创建后才设置，退出时恢复原值，避免构造失败污染后续推流进程。

- OpenGL fallback 新增 RGBA8 texture 的 framebuffer attachment 完整性检查，能力日志和候选日志输出 `framebuffer=complete`/`framebuffer=1`；初始化失败会沿用现有清理和稳定回退。

- 新增并修正 `opengl_fallback_rtsp_soak.py`：在单次诊断进程内禁用 native Vulkan 入口，真实提交 CUDA RGBA 帧并验证 OpenGL fallback、厂商编码器/host-upload 和 MediaMTX 发布边界；修正默认仓库根目录与 `VulkanDirectSbsOutput` 的 `src/desktop2stereo` base_dir 语义。本机 RTX 3090 实测 640×360@30 通过 60/60 帧，3840×2160@30 通过 300/300 帧，均为 `cuda-opengl-interop` + PyNvVideoCodec/NVENC + MediaMTX H264；新增 `--force-host` 诊断分支并实测 640×360@30 通过 60/60 帧，日志正确报告 `interop=none gpu_to_cpu=True` 和 FFmpeg `h264_nvenc` host-upload；追加 3840×2160@30、60/60 帧闭环，耗时 5.56 秒（约 10.8 FPS），确认 CPU host-upload 回退不能满足 4K/30，但 MediaMTX H264 发布稳定；文档补充运行命令，并将 AMD 验证清单拆分为代码完成与真机未验证两项。

- 完成 OpenGL fallback 4K 图像边界 A/B：RTX 3090/WGL/3840×2160/30 帧，CUDA interop probe 为 `501.1 FPS gpu_to_cpu=false`，强制 host/PBO probe 为 `12.9 FPS gpu_to_cpu=true`；结果已写入实现指南，明确这不是最终 WebRTC 帧率。

- 修正 OpenGL smoke 路径统计：GPU interop 能力存在但使用 `--force-host` 时，不再误报 `gpu_to_cpu=false`；GPU probe 和 host probe 现在分别输出 `path=gpu-interop` / `path=host-upload` 及对应复制边界。

- OpenGL smoke 工具增加 `--force-host`：在具有 CUDA/AMD interop 的机器上也能单独压测 PBO/fence host-upload 分支，便于 GPU 与 CPU 回退 A/B 对比。

- 新增 `tools/opengl_fallback_smoke.py`：在不启动 MediaMTX 的情况下，验证真实 headless OpenGL、RGBA8 texture、PBO/fence 环及 CUDA/HIP GPU probe 或 host-upload probe，并以 JSON 输出能力和吞吐结果。

- 加强 OpenGL fallback 初始化异常安全：GLFW 初始化、隐藏窗口、纹理或 interop 探针中途失败时统一调用清理逻辑，确保 GLFW 状态、窗口和已创建 GPU 资源不会残留。

- 增加无 interop 回退回归测试：模拟 NSGL/VideoToolbox 能力时验证 RGB 帧直接进入 host encoder，测试会在任何重新引入 CPU→OpenGL→CPU 往返时失败。

- 明确跨平台 OpenGL 能力边界：没有 CUDA/HIP graphics interop 时，运行时报告 `host-upload fallback; no portable OpenGL encoder interop`，Intel 继续使用 FFmpeg QSV/VAAPI，macOS 继续使用 VideoToolbox；不再把 OpenGL texture 误报为 QSV/VAAPI surface 或 IOSurface-backed VideoToolbox frame。

- 完善 OpenGL 备用后端的三槽 PBO/fence 环：每个槽位保留 GPU 完成 fence，复用前只等待对应槽位，不再每帧立即等待并销毁 fence；关闭时统一释放未完成同步对象，降低 host-upload 图像提交的串行阻塞。

- 修复 OpenGL 无 interop 回退分支：CUDA/ROCm 图像先转换为 CPU RGB 后直接交给稳定 FFmpeg/QSV/VAAPI/VideoToolbox 路径，不再引用未初始化的 RGB 变量，也不把已在 CPU 的帧重复上传 OpenGL 再读回；新增 `interop=cuda/hip/none` 能力日志，明确区分 GPU interop 与 host-upload。

- 扩展高级网络推流 OpenGL 备用路径：NVIDIA 上新增 CUDA–OpenGL interop，将 CUDA RGBA tensor 映射到 OpenGL RGBA8 texture，再以 GPU device-to-device copy 返回 CUDA tensor，交给 PyNvVideoCodec/NVENC 压缩发布；同时接入 HIP graphics-resource 适配层和已有 AMF surface 编码器。无原生 interop 时，OpenGL 回退日志现在输出 FFmpeg 实际选择的 QSV/VAAPI/VideoToolbox/软件编码器名称，不再笼统标记为 host-upload。实测 OpenGL 3.3/NVIDIA/CUDA roundtrip、3840×2160 NVENC 压缩包和 RTSP 发布闭环通过，日志明确 `gpu_to_cpu=False zero_copy=False`。AMD 真机驱动/音频/4K 仍待验证；Intel/macOS 原生 interop 和严格 zero-copy 仍待后续实现。

- 新增 MediaMTX 端到端 Vulkan RTSP soak 工具：启动 MediaMTX 和 FFmpeg mux-only TCP 发布，验证压缩 H.264 包进入 `live` 路径；本机 3840×2160@30 连续 300 帧发布通过，未经过 4K rawvideo stdin。

- 新增 native RGBA→NV12→Vulkan Video 多帧 soak 参数；修复 FFmpeg drain 的 EOF 误报后，使用远程构建 run `32534594122` 的 Windows DLL 完成 3840×2160 连续 900 帧验证，900 个压缩包全部读取成功并正常 flush。

- 修复 native Vulkan bridge drain ABI：FFmpeg `avcodec_receive_packet()` 的 `EAGAIN/EOF` 统一表示当前没有更多压缩包，不再把正常 flush 结束误报为编码失败；新增长时间 RGBA→NV12→Vulkan Video soak 参数，支持按帧数或时长验证槽位复用。

- 扩展 Vulkan FFmpeg bridge GitHub Actions：新增 Ubuntu 24.04 Linux amd64 构建、下载同版本 FFmpeg 开发包、CMake 编译和 ABI 导出校验；run `32533937908` 的 Windows/Linux 两个 job 均通过，完成跨平台原生桥静态能力验证。

- 根据 FFmpeg 9.0.1 `hwcontext_vulkan.h` 记录 CUDA/Vulkan 零复制边界：CUDA 导入所需的 `AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE` 会产生 R8/R8G8 拆分图像，而 Vulkan Video 需要单一 NV12 multi-plane image；当前版本因此保留已验证的 Vulkan Compute + device-local copy，不伪装不兼容的 plane 资源。

- 升级 Requirements Compliance workflow 的 `actions/setup-python` 到 `v6`，与 `checkout@v5`、`upload-artifact@v6` 一起使用 Node.js 24，消除远程检查中的 Node.js 20 弃用提示。

- Vulkan native bridge 增加 NV12 `STORAGE_IMAGE` 能力探测：驱动支持时 Compute 直接写入 FFmpeg 编码图像的 plane view，消除 R8/RG8→NV12 device-local copy 并记录 `zero_copy=True`；不支持时自动保留固定槽位 copy 路径。RTX 3090 当前驱动实测返回 `VK_ERROR_FORMAT_NOT_SUPPORTED (-11)`，3 次 3840×2160 H.264 烟测均通过，保持原有稳定回退和 4K 编码链路。

- 校准闭环文档与实际实现对齐：确认头显页面通过 WebRTC `getStats()` 回传 decoded FPS、丢帧、冻结、RTP 丢包、接收码率、抖动缓冲、RTT 和媒体尺寸；指南改为准确记录当前固定 30 FPS、码率搜索策略，并保留尚未完成的 PICO/Quest/Wolvic 30 分钟实机验收项。

- 补齐回退可观测性：硬件编码器全部不可用时覆盖 `libx264`/`libx265` 软件回退；Vulkan native/稳定 FFmpeg 回退通过 `[D2S_STATUS]` 更新 GUI 状态栏，并新增对应回归测试。

- 增加并实测 native Vulkan/CUDA 物理设备 UUID 校验：编码器启动时读取 Vulkan `VkPhysicalDeviceIDProperties`，与当前 CUDA tensor 设备匹配；多 GPU 不匹配时触发稳定高级推流回退，并记录设备名称和 UUID；设备日志改用 ASCII，兼容 Windows GBK 输出。

- 将 native Vulkan RGB→NV12 Compute 中间资源改为按 FFmpeg NV12 输出 image 建立固定槽环；每个槽独立 Y/UV storage image、descriptor set、command buffer 和 fence，避免所有帧共享一套转换资源并逐帧串行等待；远程 DLL 构建成功，4K 60 帧运行探针以 68.4 FPS 完成并正常关闭。

- 补充 native Vulkan 编码诊断日志：输出实际 Vulkan 设备、H.264/HEVC 编码器、RGBA8→NV12 格式、prepare/compute queue family、目标/峰值码率和 `bf=0`，便于区分 GPU 图像路径、编码器和 MediaMTX 传输层问题；新增 bridge 合约测试。

- 完成 native Vulkan 高级推流连续运行验证：RTX 3090 Windows 主机连续提交 600 帧 3840×2160@30，耗时 20.10 秒；MediaMTX 持续发布 H.264，ffprobe 读取到 `3840x2160`、`30/1 FPS`、`yuv420p`、`1/90000` 时间基，未出现 native 编码回退或原始 RGB24 pipe。
- 更新 Vulkan bridge README 与实现指南：ABI、RGBA 输入池、NV12 multi-plane 编码池和当前 device-local copy 状态与实际代码一致；明确头显 30 分钟闭环和音频闭环仍属于未完成验收项，并记录 NVIDIA validation 层下 FFmpeg frame-pool VUID/flush 阻塞的复现条件。
- 增强 Vulkan 推流关闭与诊断回退：正常关闭不再调用不会再发布结果的阻塞式 `avcodec_send_frame(NULL)` flush；检测到 `VK_LAYER_KHRONOS_validation` 时不加载 native bridge，明确回退到稳定 host-upload 路径，避免 validation 诊断环境卡死；新增对应单元测试。
- 完成 native Vulkan 视频与 SoundCard/WASAPI 音频复用验证：连续运行 10.12 秒无音频 runtime error，MediaMTX 日志确认同时发布 `H264` 与 `Opus` 两条轨道，RTSP TCP 读取端成功看到两条轨道。

## 2026-08-21

- CUDA/FFmpeg Vulkan 互操作新增单 plane RGBA 写入接口：按 FFmpeg 导出的 slot 一次导入 external image 与 timeline semaphore，使用 CUDA device-to-device 拷贝写入 4K RGBA 并回传 producer-ready timeline value；不经过 CPU 或 RGB24 中间缓冲，作为后续 native Vulkan RGB→NV12 的实际输入链路。
- RGBA frame release 现在必须回传 CUDA producer 的 timeline 完成值，并写回 native `AVVkFrame`；下一次复用 slot 时由 Vulkan 等待该值，避免异步 CUDA 拷贝尚未结束就回收图像。ABI 升级到 v5。
- native bridge 新增 `encode_rgba_frame`：在 FFmpeg-owned Vulkan device 上等待 CUDA timeline，执行 RGBA→R8/RG8 Compute，将结果复制到单一 NV12 multi-plane Video image，转换到 `VIDEO_ENCODE_SRC_KHR` 后提交 `h264_vulkan`；未初始化或提交失败仍返回错误供上层回退。
- 修正 native Compute 队列选择：不再把 Compute 命令录入仅具备 Transfer/Video Encode 能力的队列，改用独立 Compute queue、Video queue 之间的 image ownership transfer 和 timeline 接力；Validation 复现的 `VK_ERROR_DEVICE_LOST` 已进入针对性修复。
- 修正 NV12 队列所有权交接的布局匹配：保留编码帧的原始 `VIDEO_ENCODE_SRC_KHR` 布局，确保 prepare/compute 队列的 release/acquire barrier 使用相同 old/new layout，避免 validation 下首帧同步死锁。
- 高级网络推流接入 native Vulkan GPU 图像路径：CUDA RGBA external image 经 `CudaVulkanImageImporter` 写入 FFmpeg-owned frame，native Compute 转换到单一 NV12 multi-plane 后输出压缩包，由 FFmpeg mux-only 发布到 MediaMTX；正常路径不下载 4K RGB24、不通过原始帧 stdin，失败自动回退原高级推流。
- native Vulkan bridge 增加环境变量 `D2S_VULKAN_TRACE` 调试追踪：可输出 queue-family ownership release、Compute timeline wait/signal 和 Video acquire 的实际值，默认关闭，用于定位驱动或 validation 层同步问题。
- 修正 Compute→Video 的 NV12 ownership acquire：与 release 端统一使用 `TRANSFER_DST_OPTIMAL → VIDEO_ENCODE_SRC_KHR`，避免 queue-family 交接布局不匹配导致 validation/驱动首帧等待死锁。
- 扩展 Vulkan trace 到 acquire 提交和 `avcodec_send_frame` 前后，区分 queue submit 成功但 Video Encode 等待，还是 FFmpeg 编码调用本身阻塞。
- 扩展 Vulkan trace 到 `avcodec_receive_packet` 与 flush，覆盖从 GPU 提交到压缩包读取的完整首帧路径。
- 修复 Windows native Vulkan bridge 依赖加载：启动高级 Vulkan 推流时自动将项目 FFmpeg `bin` 目录加入 DLL 搜索路径，避免 bridge 因 `avcodec/avutil` 依赖找不到而误回退。
- 优化 native H.264 packet mux 输入：为持续 pipe 设置 32-byte probe、零分析时长和固定 FPS 探测，避免 FFmpeg 等待 EOF 才建立 RTSP 输出。
- 修正 native Vulkan 推流 PTS：按编码器 `time_base=1/FPS` 使用单调帧序号，不再使用高速提交时可能重复的毫秒墙钟值。
- 新增 `vulkan_ffmpeg_rgba_cuda_smoke.py`：在真实 CUDA 设备上向 FFmpeg Vulkan 4K RGBA frame 写入固定颜色并完成 timeline 同步，用于区分“仅能导出句柄”和“CUDA 实际可写入”的两种状态。
- 新增 `vulkan_ffmpeg_rgba_encode_smoke.py`：不调用 CPU 同步，直接验证 CUDA RGBA → native Vulkan Compute → multi-plane NV12 → `h264_vulkan` 压缩包闭环。
- 新增 `VulkanRgbToNv12Pipeline` 运行时封装：固定三张 storage image（RGBA 输入、R8 Y、RG8 UV）、8×8 dispatch 和偶数分辨率校验，明确中间 R8/RG8 结果必须由原生桥复制到 profile-compatible 的 NV12 multi-plane 编码 image。
- 新增可复用 `VulkanRgbToNv12Intermediate` GPU 中间资源：按 4K 输入创建一次 R8 Y（3840×2160）和 RG8 UV（1920×1080）storage/transfer image，避免每帧重新分配 Vulkan image/memory；仍由下一阶段原生桥负责 copy 到 Video Encode NV12 image。
- 为 Vulkan RGB→NV12 中间资源增加 command-buffer copy：记录 Compute 写入→Transfer 读取、multi-plane NV12 plane 0/1 写入以及最终 `VIDEO_ENCODE_SRC_KHR` barrier，全程 GPU 内完成，等待/Signal timeline 由调用方负责。
- 中间 image 槽复用增加布局循环状态：支持 `TRANSFER_SRC→GENERAL` Compute 前置 barrier，并把 NV12 目标从上一帧 `VIDEO_ENCODE_SRC` 回收到下一帧的 Transfer 写入状态。
- 增加单帧 Vulkan GPU command sequence 封装：一次调用按顺序记录 Compute 前置 barrier、RGB→NV12 dispatch、Y/UV multi-plane copy 和 Video Encode barrier，供 native bridge 接入 timeline wait/signal。
- 明确 Vulkan device 所有权边界：Python Compute helper 拒绝跨 `VkDevice` 操作 FFmpeg-owned Video image；最终 shader、descriptor、command pool 和 timeline submit 必须在 native bridge 同一 device 内完成。
- 原生桥 ABI 升级到 v4，新增 FFmpeg-owned 单 plane RGBA external frame pool 的 acquire/release 接口；CUDA 可在同一 FFmpeg Vulkan device 上写入 RGBA，后续由 native Compute 转换到 Video NV12，避免再把拆分 plane 当编码源。
- 新增 `vulkan_ffmpeg_rgba_pool_smoke.py`，验证 ABI v4 的 4K 单 plane RGBA frame、external memory handle 和 timeline semaphore 导出。
- 新增 Vulkan Compute `d2s_rgb_to_nv12`：在 GPU 上从 RGBA storage image 计算 BT.601 limited-range Y 与 2x2 平均 UV，输出 R8/RG8 中间 image，供后续复制到合法的 Vulkan Video NV12 multi-plane image；shader 已编译并通过 manifest 校验，未引入 CPU RGB24 或 stdin 原始帧。
- Vulkan bridge 运行时新增输入格式闸门：检测到 FFmpeg 导出的 `R8`/`R8G8` 拆分 NV12 plane 时，在提交前关闭外部句柄并明确触发稳定推流回退，避免进入 Vulkan Video 驱动后才产生 `VK_ERROR_INITIALIZATION_FAILED`；后续路径必须提供单一 multi-plane NV12 image。
- Vulkan CUDA/FFmpeg frame-pool 诊断确认 NVIDIA Vulkan Video 的硬约束：CUDA 友好的拆分 `R8`/`R8G8` plane 虽可 external-memory 导入，却不是合法 H.264/HEVC Vulkan Video 输入；当前高级网络推流继续稳定回退，后续零拷贝实现必须共享单一 multi-plane NV12 image 并在 Vulkan 内完成颜色转换。
- GitHub Actions 全部升级为 `actions/checkout@v5` 与 `actions/upload-artifact@v6`，统一使用 Node.js 24 runtime，消除 GitHub Hosted Runner 对旧 Node.js 20 action runtime 的弃用提示。
- Vulkan FFmpeg bridge ABI 升级到 v3：NV12 frame descriptor 现在导出每 plane 的 OS external-memory handle、FFmpeg timeline semaphore handle/value 与稳定 slot ID，供 CUDA/HIP 一次导入后直接写入；句柄只用于 GPU 外部互操作，不暴露 CPU 像素。
- Vulkan frame submit 现在把 CUDA/HIP 已 signal 的 timeline value 写回 FFmpeg `AVVkFrame`，使 Vulkan Video 编码提交按 GPU 完成点等待，而不是在 CPU 上同步 CUDA；未提供有效完成点仍拒绝编码。
- 修正 Vulkan 原生桥的 device 所有权模型：默认由 FFmpeg 创建启用 Vulkan Video 扩展的逻辑 device 与 frame pool，避免把未启用 Video Encode 扩展的 Viewer `VkDevice` 交给编码器导致访问冲突；应用自有 device 仅在完整提供 Vulkan Video 能力时可选接管。
- 新增 `vulkan_ffmpeg_bridge_smoke.py`，可独立验证远程构建桥接 DLL、FFmpeg Vulkan device 与 4K NV12 frame pool 初始化；测试不提交未同步图像，适合在 CUDA/Vulkan 共享链路接入前定位驱动、profile 或 DLL 依赖问题。
- Vulkan 原生编码桥的 GPU 帧提交现在强制要求 producer-ready 外部信号量和值；未实现真实 Vulkan wait/layout transition 前主动拒绝提交并保持高级网络推流回退，避免把未同步的 FFmpeg `VkImage` 接入造成花屏或 GPU 竞态；Python ABI 同步暴露信号量参数，桥接 CI 同时改用当前 `d2s.2` FFmpeg 开发包。
- Vulkan Video 推流增加真实设备能力探测，并针对 NVIDIA 驱动未自动选择 H.264 profile 的情况显式使用 High profile；探测失败时继续回退稳定的 FFmpeg/NVENC 路径。
- 高级网络推流新增显式“Vulkan Video”编码后端；4K H.264 使用 High/Level 5.1，初始化或运行失败时自动回到原 FFmpeg 硬件/软件路径，Auto 默认行为保持不变。
- 修正 Vulkan Video FFmpeg 命令中 `format=nv12,hwupload` 的选项位置，确保滤镜参数位于 RTSP 输出 URL 之前并真正作用于 Vulkan 编码输入。
- 固定进程内 FFmpeg/Vulkan 原生桥接 ABI，明确要求 `AV_PIX_FMT_VULKAN`、已同步编码源图像和版本校验；桥接库未安装或不匹配时继续使用已验证的 host-upload/硬件回退路径。
- 原生桥接探针现在实际验证 FFmpeg Vulkan 编码器和 Vulkan HW device；图像提交 ABI 在完成前明确报告未启用，不会误把占位库当作零复制编码器。
- 新增 Windows GitHub Actions 原生桥构建流程：远程下载固定 FFmpeg 开发包、安装 MinGW/Vulkan 依赖、构建并检查 Vulkan FFmpeg ABI 导出；构建产物在完成实际 image submit 和头显验收前只作为测试 Artifact。
- 修正 Vulkan 原生桥 Workflow 的 Windows CMake 调用：不再通过 MSYS2 shell 解析 Windows 路径，改为直接调用 MinGW CMake/Ninja/GCC，避免远程构建出现 `cmake: command not found`。
- 修正远程 MinGW 编译器启动环境：将 MSYS2 MinGW/运行时目录加入 PATH，避免 `g++.exe` 能定位但 CMake 编译器自检失败。
- 修正原生桥 FFmpeg C API 的 C++ 链接声明，使用 `extern "C"` 对齐 FFmpeg import library，解决远程构建阶段 `avcodec_find_encoder_by_name` 等符号未定义。
- 修复点击“重置”后再切换到高级网络推流或 GPU 推流时，混音设备列表已有声卡但当前选项为空的问题；切换推流模式时如果已有有效选择则保持不变，如果当前值为空或设备已失效，则直接从现有扫描结果自动选择合适的 SoundCard/WASAPI/虚拟声卡，无需重复扫描。
- 修复“显示高级立体参数”和设备“高级选项”展开/折叠时只更新控件却不应用窗口高度的问题；两个开关现在都会按可见控件重新计算并写入原生窗口高度，同时保留既有最大高度与滚动策略。
- 补全推流参数 tooltip：逐项说明推流网址与预览、端口、防火墙及校准端口关系、MJPEG 质量范围、各协议适用场景、推流路径规则、混音设备、音频正负延迟、编码后端、自动/手动传输配置和闭环校准按钮的实际用途；明确“推流质量”仅控制低级 MJPEG，高级与 GPU 推流由 CRF 和码率控制。
- 完善“恒定质量”提示：tooltip 按 4K SBS、H.264、30 FPS 的自动校准目标带宽给出明确建议，`≥30 Mbps` 使用 CRF 20、`25-29 Mbps` 使用 23、`21-24 Mbps` 使用 26、`19-20 Mbps` 使用 28；低于 `19 Mbps` 时提示降低分辨率或帧率，而不是继续牺牲画质。
- 调整设备区域布局：将“显示模式”从“运行模式”右侧移到“头显型号”右侧，使目标头显与对应的立体显示格式在同一行配置；在运行模式右侧新增“推流设置”复选框，进入任意推流模式时显示该入口但默认不勾选，使推流参数保持折叠，用户手动勾选后才展开推流网址、端口、质量、协议和音频等设置且不清除已有值，非推流模式自动隐藏该入口；折叠状态不再阻止推流网址初始化，展开后始终显示按当前协议、端口和密钥生成的地址；OpenXR 模式隐藏显示模式时，头显型号仍保持可见。
- 精简自动校准区域：删除“开始校准”按钮右侧的帧率与码率状态，只在下一行靠左显示校准结果或“设置变化，请重新校准”提示；结果文字不再展示峰值码率，仅保留网络稳定上限、安全码率和帧率；校准结果行动态显示或隐藏时纳入原有控件高度计算并重新适配窗口，提示文字固定为单行，不增加影响其他模式的全局高度余量；配置失效判断、自动重新校准及内部峰值码率参数保持不变；底部“停止/运行”按钮组整体向左移动 20px，缩短与左侧内容的距离。
- 统一高级网络推流的跨平台 FFmpeg 运行时来源：Windows AMD64、Linux AMD64/ARM64、macOS Intel/Apple Silicon 压缩包均改用 `desktop2stereo-ffmpeg-builds` GitHub Actions 的 FFmpeg 9.0.1 `d2s.1` 构建产物，五个平台使用同一版本与功能配置，并保留构建端及本地 SHA-256 校验结果；运行时安装同步改为复制完整 FFmpeg 目录，确保共享 FFmpeg、SRT、Opus、oneVPL 等动态库随可执行文件一起部署。
- 新增“高级网络推流”的自动网络校准：GUI 提供独立入口和实时进度弹窗，头显测试页通过 WHEP 自动重连并回传 WebRTC 解码帧率、丢帧、冻结、丢包、实际接收码率与抖动；校准使用按输入分辨率确定起点的独立 30 FPS CBR 压力流，每档测试 15 秒，先按 5 Mbps 向上粗测，再在稳定/不稳定区间二分到 1 Mbps，并对最终候选执行 30 秒确认。只有实际发送码率达到目标档位 85% 的测试才有效，结果按稳定网络上限的 80%/90% 保存安全目标/峰值码率及配置指纹。
- 记录并处理 Windows 11 入站防火墙经验：高级网络推流的 MediaMTX `1122` 端口可正常访问时，自动校准专用 Python HTTP 服务 `1123` 仍可能因项目内置 `python.exe` 被 Windows 防火墙的 `Python` 入站 Block 规则拦截，表现为头显浏览器无响应且校准服务没有 `TCP accepted` 日志；校准弹窗现在提供手动“检测防火墙规则”入口，按内置 Python 的精确路径查找并仅删除匹配的 TCP/UDP 入站阻止规则，必要时通过 UAC 提权，并在检测或删除失败时给出明确提示。
- 修复 NumPy 2.5 环境下 SoundCard 0.4.4 的 Windows WASAPI 回环采集在收到真实音频后触发 `The binary mode of fromstring is removed` 并降级静音的问题；安装依赖升级并锁定到 SoundCard 0.4.6，改用兼容 NumPy 2.x 的二进制缓冲区读取实现。
- 修复“高级网络推流”使用 SoundCard/WASAPI 回环音频时，采集线程在启动后异常会令 FFmpeg 等待音频、继而使本机 RTSP 发布约 10 秒后 `i/o timeout` 的问题；音频异常现在会明确告警并按实时节奏降级为静音，视频和 WebRTC 会话保持在线。
- 修复“高级网络推流”探测 FFmpeg NVENC 时误加载并输出 PyNvVideoCodec 可用状态的问题；该模式现在只报告实际选中的 `h264_nvenc`/`hevc_nvenc`，PyNvVideoCodec 状态仅由“GPU 推流”路径处理。
- 修复 4K GPU WebRTC 推流在浏览器中花屏并持续出现 `reader is too slow`：MediaMTX 恢复 UDP 优先、保留 TCP 回退并扩大短时出口突发队列；PyNvVideoCodec 超低延迟编码改为无 B 帧的 IPPP GOP，避免 TCP 队头阻塞和参考帧重排导致浏览器连续丢包；内部 RTSP 发布包长固定为 IPv6 MTU 安全值 `1452`，无需 MediaMTX 再将 1460 字节 RTP 载荷重封装为 1440 字节。
- 修复 Windows BAT/CMD/REG 文件被仓库全局 `eol=lf` 规则转换为 Linux 换行的问题；这些文件现在以 CRLF 原始字节提交并排除 Git 文本归一化，确保 Git 克隆、GitHub ZIP 和 raw 下载均可由 Windows 原生命令解释器直接运行。

## 2026-08-20

- 修复 CUDA 独立安装器错误地将 Python 路径拼成 `src/env_install/..python3`、从而忽略已有 `src/python3` 并重复全新安装的问题；安装器现在规范化 `src` 目录并明确使用 `src/python3`。同时为 TensorRT 等 CUDA requirements 增加继承到 PEP 517 构建子进程的网络重试/超时配置，以及最多三次整步重试，缓解镜像 `IncompleteRead` 临时断流。
- 检查并修复其余平台安装脚本：Linux CUDA、Linux ROCm 和 macOS MPS 激活环境后统一调用 `src/python3` 内的解释器，各依赖步骤独立检查失败状态，避免前序安装失败被后一条命令掩盖；macOS 的 `run_mac` 权限路径适配新目录结构；Windows ROCm 安装器规范化并验证 Python 3.12 x64 路径，同时检查 ROCm requirements 与 SDK 初始化结果。

- 统一 GPU 推流运行模式：GUI 使用“GPU 推流”，旧的 NVIDIA GPU 推流名称自动归一化；运行时根据显卡型号选择 NVIDIA PyNvVideoCodec 或 AMD/其他平台的硬件编码探测链，并在初始化失败时安全回退，不改变音频、协议和 MediaMTX 配置。
- 新增 Windows AMD 原生编码桥接工程 `native/amd_encoder`：动态检测 AMD AMF 运行时 `amfrt64.dll` 与 Radeon DXGI 适配器，提供 Python 可选加载接口；GitHub Actions 已成功编译并将 `src/desktop2stereo/streaming/amd_encoder/d2s_amd_encoder.dll` 随项目发布，运行时未安装 AMD AMF 或 HIP 时仍保持 FFmpeg 回退。桥接现在可将 ROCm HIP RGBA device tensor 通过共享 D3D11 texture 导入 AMF，并用 FFmpeg 仅复用 H.264/H.265 包送入 SRT；音频开启时继续使用原有 FFmpeg 音视频路径。
- 完成项目目录重组后的路径统一：源码归档到 `src/desktop2stereo`，Python 运行环境统一位于 `src/python3`，安装脚本位于 `src/env_install`，脚本、测试和启动入口通过统一项目路径配置解析。

- 兼容旧的“NVIDIA GPU 推流”配置名称：读取时自动归一化为“GPU 推流”；NVIDIA 模式现在通过 PyNvVideoCodec 在 CUDA/NVENC 内完成 NV12 转换与 H.264/H.265 编码，同时允许 SoundCard WASAPI 回环音频由 FFmpeg 编码为 Opus/AAC 并与已编码视频复用，不再因启用音频退回 RGB24 CPU 管线。PyNvVideoCodec、音频或复用器启动失败时仍自动回退现有 FFmpeg 硬件/软件编码，并保持高级网络推流使用独立的 FFmpeg 自动编码后端。
- 修复 WebRTC 音视频发布约 10 秒后被 MediaMTX 以 RTSP `i/o timeout` 断开的情况：将 FFmpeg `max_interleave_delta` 从会无限等待稀疏音视频包的 `0` 改为 `100000` 微秒，SoundCard 或视频短暂无包时仍会持续刷新 RTSP。

- 网络串流启动时按操作系统和实际 SBS 分辨率探测编码能力并自动降级：Windows 依次尝试 NVENC、Intel QSV、AMD AMF，Linux 尝试 QSV/VAAPI，macOS 尝试 VideoToolbox，均不可用时回退 `libx264`/`libx265`；VAAPI 自动配置设备与硬件帧上传。硬件编码器仅在实际 FFmpeg 探测成功后启用。

- 补充跨平台推流参数：macOS 音频使用 FFmpeg `avfoundation` 设备索引；QSV 使用 `global_quality` 并关闭 look-ahead，AMF 使用低延迟 `vbr_peak`，避免套用 NVENC 专属参数。
- 完善跨平台串流运行时打包：新增平台/架构压缩包清单，启动时只解压当前系统所需的 FFmpeg 与 MediaMTX；MediaMTX 官方模板保存在 `mediamtx/mediamtx.yml`，项目最终配置固定使用根目录 `mediamtx.yml`，后续升级不会覆盖用户自定义配置；打包文档补充 MediaMTX、FFmpeg 官方下载地址。

- 合并旧网络推流与低级网络推流：GUI 不再显示 `Legacy Streamer`，旧配置读取时自动归一化为 `MJPEG Streamer`；高级网络推流的内部配置键保持 `RTMP Streamer` 以兼容已有设置。

- GUI 执行重置后，运行模式默认恢复为“本地查看”（`Local Viewer`）。
- 修复重置后运行模式下拉框未切换的问题：配置应用现在会在保留可选配置关闭时也同步 `run_mode_key` 和界面控件。
- 将电影模式的补洞默认保持为关闭，并将 GUI 重置后的补洞模式改为关闭（`none`）。
- 修复点击重置后窗口高度未按左侧 GUI 内容重新计算的问题；重置完成会重新估算并调整窗口尺寸。
- 将 CRF 默认值调整为 `23`，同步 GUI 默认/重置值、启动缺省回退、bootstrap 配置和 `src/settings.yaml`。
- 修复 WebRTC 播放暂停后恢复时音频逐渐落后视频：WebRTC 音频使用 `aresample=async=1000:first_pts=0` 持续校正系统音频时钟并从首帧对齐，同时关闭 MPEG-TS 复用器的交错等待，避免暂停恢复后的秒级音画偏移。
- 将 CRF 默认值恢复为 `20`，同步 GUI 默认/重置值、启动缺省回退、bootstrap 配置和 `src/settings.yaml`。

## 2026-08-20

- 修复 WebRTC 推流没有声音：WebRTC 发布链路改用浏览器与 MediaMTX 支持的 Opus 48 kHz 双声道音频，避免 AAC 音轨被报告为 `skipping track 2 (MPEG-4 Audio)` 并跳过；HLS、RTMP 与其他现有协议继续使用 AAC，保持原有兼容性。
- 面向以无线网络为主的串流场景，将 CRF 默认值由 `20` 调整为 `23`，同步 GUI 初始/重置值、旧配置迁移补全值和运行入口缺失配置回退，以降低默认目标码率并减少带宽不足造成的马赛克。
- 将网络串流的默认协议统一改为 WebRTC；用户选择 RTMP、HLS、RTSP 等非 WebRTC 协议启动时，日志会输出一次 WebRTC 低延迟浏览器串流建议，但仍按所选协议正常启动。
- `RTMP Streamer` 运行模式接入与 Local Viewer/OpenXR 相同的自动捕捉限频：捕捉 FPS 设为 Auto 时，直接 SBS 推流消费者每 5 秒回报实际输出，控制器按 15 秒窗口内持续输出峰值 `+5 FPS` 调整捕捉目标，并继续忽略稀疏样本、受显示器刷新率限制；手动 FPS 不受影响。
- 将动态码率预算从 HLS 扩展到 RTMP 与 WebRTC，并针对无线串流把峰值上限由目标码率的 `125%` 收紧为 `115%`、VBV 缓冲由峰值的 `2 倍` 收紧为 `1 倍`，降低局部运动引起的瞬时码率突发和无线丢包马赛克；每秒封闭 IDR、固定 GOP 与禁用场景切换保持不变。
- RTMP 音频延迟支持运行时热调节：GUI 输入合法的 `-10` 到 `+10` 秒数值后自动保存，现有配置轮询会把新值转交给直接 SBS 推流输出；FFmpeg 发布进程在下一帧由自己的输出线程安全重启并应用新的 `-itsoffset`，MediaMTX 服务保持运行，避免端口和服务端整体重启。
- RTMP 音频延迟默认值由 `-0.15` 秒调整为 `-0.1` 秒，并统一 GUI 初始/重置值、缺失配置回退、bootstrap 与 FFmpeg 输出构造默认值。

## 2026-08-19

- 修复 RTMP、本地模式调节 Min LOD、Max LOD 与 MIP Bias 无效：三项参数从 OpenXR 专属 Vulkan sampler 配置迁移到所有模式共用的眼图质量阶段，在格式打包前选择并混合 mip 层，再执行 Lanczos2/EASU 与 RCAS；无投影导数路径使用 Max LOD 与 Bias 形成显式 LOD 请求，单独提高 Max LOD 即可生效，不再必须同时抬高 Min LOD。OpenXR 收到已处理眼图后使用中性 sampler，避免重复过滤。
- 将“头显型号”提升为所有运行模式共用的最终目标显示预设，并在 GUI 中移到“运行模式”下方单独一行；OpenXR、浏览器推流与供 Virtual Desktop 等软件采集的本地输出统一按生成眼图分辨率和所选头显档位决定 Lanczos2/EASU/RCAS，物理显示器仅作为中间承载画布。配置键保持 `XR Headset Model` 不变，默认型号仍为 Pico 4 / 4 Ultra；切换头显型号与 RCAS 强度支持运行时热更新。
- 修复直接推流丢失立体视差：推流专用 uint8 路径现在直接量化已经合成完成的 packed SBS，不再从 `quality_4k` 直出路径的左右眼占位张量重复打包。Half-SBS/Half-TAB 的每眼 2×缩小由两像素平均升级为 Lanczos2 预滤波，降低桌面文字、网格和细线条的摩尔纹；所有模式的公共眼图质量阶段会在格式打包前按头显档位执行 EASU/Lanczos2 与可选 RCAS，服务端不把 Half-SBS 隐式转换为 Full-SBS。
- 增加 HLS 动态质量控制：按实际 SBS 分辨率、稳定输出帧率、H.264/H.265 编码效率和用户 CRF 自动计算目标码率、峰值码率与 VBV 缓冲，替代 NVENC 无上限的瞬时 VBR；硬件编码同时启用时空自适应量化，降低复杂运动画面因码率突发造成的缓冲不足和马赛克。Full-SBS 保持公共质量阶段输出的完整双眼尺寸，不做 Half-SBS 转换。
- 优化 Full-SBS 网络推流：保持公共质量阶段输出的完整双眼画面，改用 H.265/HEVC（优先 `hevc_nvenc`，不可用时回退 `libx265`）；Half-SBS 继续使用 H.264。GUI 会提示 Half-SBS 具有更广泛的浏览器兼容性和更低解码负载，但不阻止 Full-SBS 启动。推流音频统一为 HLS 兼容性更好的 AAC 48 kHz/128 kbps。
- 修复高码率 HLS 触发 MediaMTX `reached maximum segment size`：H.264/H.265 软件编码统一使用每秒封闭 IDR GOP、禁用场景切换 GOP并重复参数集，确保 HLS 按 1 秒切段；HLS 单 segment 安全上限由 `50M` 提高到 `256M`。NVENC 探测失败现在同时记录实际 SBS 尺寸与 FFmpeg 错误原因。
- 新增直接消费运行时 SBS 帧的网络输出管线：RTMP、MJPEG 与旧网络推流不再依赖捕捉 Viewer 窗口；CUDA 帧使用固定页内存下载并以 RGB24 直接交给 FFmpeg，减少重复窗口捕捉和中间拷贝。
- 完善 RTMP 运行环境与音频检测：项目内置 FFmpeg 和 MediaMTX，支持 `Stereo Mix` 与 Screen Capture Recorder 的 `virtual-audio-capturer`；补齐直播流密钥、启动检测、浏览器播放地址及明确的日志提示。
- 优化不同显卡上的流媒体编码：启动时按实际 SBS 分辨率动态探测 NVIDIA NVENC 能力，失败时自动回退软件编码；输出帧率使用至少 5 个完整秒级窗口、最多 15 秒的稳定采样确定，并在 CUDA 转 CPU 前按目标节奏丢弃多余帧，避免短时峰值造成浏览器播放不连贯。
- 增强 MediaMTX 与 FFmpeg 进程管理：启动失败时保留服务端错误原因，正确区分 `INF`、`WAR` 与 `ERR` 日志级别；停止和异常退出时按进程树顺序清理，避免残留进程占用端口导致下次启动直接退出。
- 调整本地显示器输出规则：普通 Local Viewer 恢复输入/输出显示器互斥，多屏默认选择最后一个可用输出屏，单屏时提示安装或启用虚拟显示器；3D 显示器模式仍允许输入输出同屏。Windows 全屏输出在应用原生工具窗口样式前保持隐藏创建，从而不在任务栏显示 SBS 按钮，并补充捕捉排除状态诊断。

- GUI 执行“重置”后的抗锯齿值改为 `0`，并同步将 `Depth Antialias Strength` 默认设为 `0.0`；控件首次创建与重置配置使用同一默认值，手动选择其他立体预设时仍使用对应预设值。
- 深度预览窗口单独改为允许截图，不再设置 Windows `WDA_EXCLUDEFROMCAPTURE`；截图工具现在可以选中并捕获预览窗口，SBS 主输出的窗口策略保持不变。预览位于输入显示器时也会随该显示器进入本地捕获画面。
- 本地深度预览窗口保持有边框、可正常聚焦，但取消始终置顶；失去焦点后窗口不会关闭，同时允许其他窗口正常覆盖。
- 本地窗口预览改为按当前 `Depth Strength` 显示有效视差，热更新后下一帧立即生效：`0.00` 为全蓝，`0.25` 以紫色为中间值并完整保留远处蓝色—近处红色的空间深度渐变，`0.50` 为全红；其余强度围绕对应颜色中心保留渐变幅度，不再使用青、绿、黄彩虹色标。
- 将立体“电影模式”的默认补洞改为“关闭 / 不补洞”，同步把补洞半径、强度和依赖补洞的 Temporal 设为 0；GUI 初始值、预设切换、自动模式默认回退和运行时预设保持一致。
- 修正本地模式输入/输出显示器联动：立体输出列表不再包含“窗口预览”，Local Viewer 与 3D 显示器模式都禁止输出到正在捕获的显示器，避免无法从捕获中隐藏的 Vulkan 全屏输出被递归回采、造成画面无限叠加；双显示器自动互斥选择，三台及以上显示器默认选择最后一台可用显示器。“窗口预览”改为默认关闭的独立高级复选框，位于“垂直同步”右侧；勾选后在保持物理显示器全屏输出的同时，于输入显示器额外打开捕获排除的彩色深度图调试窗口，直接复用当前帧立体合成所用的单张深度结果，以蓝→青→绿→黄→红表示由远到近，不重复推理或显示 SBS。
- 修复彩色深度预览窗口已初始化并持续提交、但不可见的问题：物理全屏输出不再使用 `VISIBLE=false` 隐藏创建，而是从创建开始保持可见；每个 Vulkan Viewer 创建前仍独立重置进程级 GLFW Window Hints，普通调试窗口不会继承主输出的无边框和置顶提示。捕获排除仅应用于彩色深度调试窗口，不应用于物理全屏输出。

## 2026-08-18

- 优化本地模式持续性能输出：SBS 性能统计窗口改为 15 秒，自动捕捉目标按有效持续 SBS 峰值 `+5 FPS` 计算；静态/动态判断逻辑保留但默认关闭，避免静态图片导致捕捉帧率降到个位数后无法恢复。
- 修复补洞关闭时的时域处理：补洞模式为“关闭 / 不补洞”时自动保存 `Temporal: false` 与 `Temporal Strength: 0`，运行时也会对旧配置强制旁路 Temporal；重新启用补洞时恢复当前立体预设的默认时域强度。
- 优化补洞关闭路径：不再创建无消费者的 Occlusion 空遮罩，`fast_plus` 同样跳过无用的 Occlusion 和补洞计算；4K 关闭路径的 Occlusion 区间降至约 `0.002 ms`。
- 新增本地 SBS 直接融合输出：在 `quality_4k`、2 层对称、Half-SBS/Full-SBS 且补洞和 Temporal 关闭时，Warp 直接生成 SBS，跳过左右眼中间张量及独立拼接；Half-SBS 与 Full-SBS 均与原路径像素误差为 0，Full-SBS 4K 合成约由 `3.81 ms` 降至 `2.95 ms`。
- 优化本地 Vulkan Viewer 的 CUDA 浮点 CHW 到 RGBA8 转换：复用 Triton 融合内核，4K 转换约由 `3.12 ms` 降至 `0.64 ms`，保持 CPU staging 回退路径和跨平台行为不变。
- 完善跨平台显示器型号读取与显示器映射：优先按显示器名称匹配，名称相同时再按序号匹配；保留 Windows、Linux、macOS 的统一 Python 读取接口，并移除本地模式不再需要的显示器坐标配置。
- 兼容新版 `tqdm.thread_map` 的进度条构造方式：线程任务进度正确显示为 `steps`，普通模型/文件下载仍显示为 `bytes`。

- 修复升级 Flet 后 GUI 永久停在 `Working`：Python `flet/flet-desktop` 和内置压缩包已是 `0.86.5`，但旧 `flet_clients` 缓存仍为 `0.85.3`；客户端缓存现在记录并校验源压缩包 SHA-256，Windows、Linux、macOS 的包内容变化后都会自动重新解压，避免 Python 服务与桌面客户端协议版本不匹配。
- 修复 `install-cuda_standalone.bat` 在电脑已注册同版本 Python 3.12 时提示安装成功、但项目 `src/python3/python.exe` 不存在的问题：python.org EXE 会进入已有安装的维护模式并忽略新的 `TargetDir`，现改用官方 Python 3.12.10 x64 NuGet 独立发行包。项目本地运行时保留完整 `Lib`、`DLLs`、`include`、`libs`、标准库和 `ensurepip`，不再使用仅 35 个基础文件的 embeddable ZIP；检测到旧嵌入版时会自动替换为完整布局，且不受系统注册表中其它 Python 安装位置影响。
- 清理 CUDA 安装器替换旧 embeddable Python 时的误导性 `ModuleNotFoundError: ensurepip`：完整性检测改为无导入异常的模块探测并静默处理预期失败，随后明确提示正在替换不完整运行时。

## 2026-08-17

- 恢复本地 Viewer 的独立调试预览语义：GUI 旧 `Viewer Window` 原位更名为 i18n `Window Preview / 窗口预览`，仍保存稳定的 `Stereo Output=None`，兼容现有配置。选择窗口预览时创建普通可缩放、可聚焦、有任务栏入口的 Vulkan 窗口，不应用副屏 Topmost/NoActivate 样式；选择具体显示器时才使用覆盖目标显示器的持久无边框 Vulkan 输出。
- 根据 GLFW #447 与 mpv #10549 的 Windows 多屏结论强化副屏持久显示：Viewer 采用 GLFW 推荐的 `DECORATED=false + FLOATING=true + monitor=NULL` 无边框模式；窗口先隐藏创建，设置为浏览器 PiP 同类的不可激活 `WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST`、移除最小化/最大化能力和任务栏入口后，才以 `SWP_NOACTIVATE` 显示。这样启动、Win+D 和点击副屏都不会抢走主屏焦点，也不再按普通应用窗口最小化 Viewer；周期性 `SW_SHOWNOACTIVATE` 仅保留为异常隐藏兜底。
- 根据本地 Viewer 必须长期让鼠标与键盘停留在主显示器的实际用法，移除 `VK_EXT_full_screen_exclusive` 硬独占申请、释放、焦点恢复及专用 Swapchain 分支；副屏现在始终使用 Vulkan 无边框 Topmost Swapchain，不因 Viewer 永久失焦而反复重建或消失。CUDA/Vulkan 零拷贝、指定显示器输出、VSync、动态捕捉和 Win+D 无焦点自动恢复均保持不变。
- 本地 Vulkan Viewer 在 Windows 上改为指定副屏永久置顶且不抢焦点：承载 `VK_EXT_full_screen_exclusive` 的 HWND 不再使用会在恢复时强制激活的 GLFW monitor-attached 模式，而是使用覆盖目标显示器的无边框 windowed surface，硬独占仅由 Vulkan acquire 控制。原生 `SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)` 保持副屏最前，Presenter 每 `100 ms` 检查 Win+D 造成的隐藏、最小化或 Topmost 丢失，并使用 `ShowWindowAsync(SW_SHOWNOACTIVATE)` 自动恢复；鼠标和键盘可长期留在主显示器，重新聚焦 Viewer 时才恢复 Windows 不允许后台持有的硬独占。
- 修复 `VK_EXT_full_screen_exclusive` 副屏在点击主显示器后仍可能消失的问题：Presenter 现跟踪 GLFW 焦点，副屏 Viewer 失焦后立即释放硬独占并在同一显示器、同一无边框全屏窗口上重建普通 Win32 Swapchain，使 SBS 在用户操作主屏时持续可见；Viewer 重新获得焦点后再重建 `APPLICATION_CONTROLLED` Swapchain 并自动申请硬独占。窗口事件在有帧和无帧等待阶段均走同一切换逻辑。
- 本地 Viewer 增加真正无窗口 Vulkan direct-display 能力诊断：启动时检查 `VK_KHR_display` 以及 `VK_EXT_direct_mode_display`/`VK_NV_acquire_winrt_display`。当前 RTX 3090 Windows 驱动经项目绑定与 Vulkan SDK `vulkaninfo` 双重确认均未暴露这些实例扩展，因此无法创建无 `HWND` 的 Display Surface；程序明确记录缺失扩展并继续使用已经可用的 Win32 `VK_EXT_full_screen_exclusive`，不加入无法运行的伪 direct-display 后端。
- 修复 Vulkan 独占 Viewer 在用户点击另一台显示器后消失、必须重新点击才能恢复的问题：GLFW 全屏窗口禁用失焦自动最小化，普通主显示器获得焦点时 SBS 输出窗口继续留在目标显示器；Viewer 再次获得焦点后若系统曾撤销 `VK_EXT_full_screen_exclusive`，会在 Presenter 线程自动重建 Swapchain 并重新申请独占。
- 修复本地 Viewer 未配置捕捉窗口标题时显示为 `None Vulkan Viewer`：空标题现回退为 `Desktop2Stereo Vulkan Viewer`；这只修正 Windows 窗口/任务栏名称，不改变已经由 `VK_EXT_full_screen_exclusive` 获得的 Vulkan 独占状态。
- 本地 Vulkan Viewer 现接入 Windows `VK_EXT_full_screen_exclusive`：通过 GLFW `HWND` 获取真实 `HMONITOR` 并查询 surface 独占能力后，全屏 Swapchain 才使用 `APPLICATION_CONTROLLED` 模式并显式 acquire；切回窗口、Swapchain 重建和退出前显式 release，独占丢失会重建并重试。驱动拒绝独占 Swapchain 时立即重试普通 GLFW 显示器全屏，不再令 Viewer 线程退出。同步修复 GUI/MSS 的 1-based 显示器编号直接用于 GLFW 0-based 数组导致选错输出显示器的问题。
- 本地 `Local Viewer` 在捕捉频率为 `Auto` 时现与 OpenXR 共用动态捕捉控制器：Viewer 无论是否显示 FPS 日志都会每 5 秒反馈成功 Present 的实际 SBS 帧率，控制器每 60 秒按平均 SBS FPS `+5 FPS` 更新 `WindowsCaptureCUDA` 软件限频；手动捕捉频率保持固定，“3D 显示器”模式不受本次改动影响。开启日志时同时显示当前 `capture_target` 便于实机确认闭环。
- 本地 Vulkan Viewer 的“显示帧率”改为无画面侵入的日志遥测：开启后按成功提交到普通 2D 显示器的 Present 帧统计实际输出帧率，每 5 秒仅输出一次 `[VulkanLocalViewer] Present FPS`；GUI 复选框现通过既有 `settings.yaml` 热更新链路即时开启或关闭，关闭后不再输出，窗口模式仍同步更新标题栏，独占全屏 SBS 像素完全不变。

## 2026-08-16

- 本地 Vulkan Viewer 的自动全屏改为 GLFW 原生显示器独占模式：窗口直接绑定 GUI 选择的普通 2D 输出显示器及其当前原生分辨率、刷新率，运行期间由 SBS 占用该显示输出，不启用任何“3D 显示器”专用路径；`Alt + Enter` 可在独占全屏与窗口模式间切换，恢复窗口时保留原位置和尺寸，Surface 变化继续由现有 Swapchain 自动重建处理。
- 本地 Viewer 模式的 Vulkan 窗口现在启动后自动在所选输出显示器上进入原生全屏，不再受“3D 显示器”模式开关限制；OpenXR 路径及显示器选择逻辑不变。
- 修复本地 Vulkan Viewer 颜色发白：运行时 SBS 帧本身已是 display-referred sRGB 字节，本地 CUDA 外部图像与 CPU staging 图像现统一使用 `R8G8B8A8_SRGB`，并优先创建 sRGB GLFW Swapchain；Vulkan blit 因而只进行正确的 sRGB 解码、线性过滤和重新编码，不再将 `UNORM` 中的已编码字节再次当作线性颜色编码。若表面完全不支持 sRGB，才保留 UNORM 字节复制降级并输出一次明确日志。
- 恢复并完成本地模式的原生 Vulkan Viewer：`Viewer` / 3D 显示器模式现在启动独立 GLFW Vulkan Swapchain 消费 `runtime_q` 的最新已打包 SBS 帧。CUDA 输入会复用 OpenXR 已验证的 Vulkan 导出图像和 CUDA 外部二进制信号量，直接写入本地 Presenter 的 GPU 图像并由 Vulkan 等待后 transfer blit/present，不经过 CPU 内存；缺少外部内存/信号量扩展或导入失败时才自动退回 Vulkan staging upload。无需虚拟显示器，OpenXR Presenter 路径不变。
- 修复本地 Vulkan Viewer 在窗口尺寸、DPI 或显示器状态变化后因 `VK_ERROR_OUT_OF_DATE_KHR (-1000001004)` 退出：窗口 Swapchain 现会在 Presenter 线程安全重建并继续消费后续帧，CUDA 外部图像保持复用；同时将 acquire 移到 CUDA signal 之前，防止失效帧遗留未消费的外部二进制信号量，并移除正常关闭时误报的 zero-copy unavailable 日志。
- 精简 TensorRT 原生 Provider 加载成功日志：引擎路径只显示模型缓存目录名与 TRT 文件名（例如 `models--lc700x--Distill-Any-Depth-Base-hf\\model_fp16_294x518.trt`），不再输出项目所在磁盘的完整绝对路径；实际加载位置与错误诊断不变。
- 修正固定尺寸 GUI 自动计算窗口高度略少的问题：窗口高度估算现在完整计入底部按钮行、行间距、状态栏及其内边距，避免“重置 / 停止 / 运行”下方的状态文字被窗口底边截断；不改变 DPI、控件比例或左栏自动滚动行为。
- 修正 Full-SBS / Full-TAB 的分辨率语义：GUI 的“处理分辨率”现在始终表示捕获源及单眼处理尺寸，不再在深度推理前提前减半；Full-SBS 仅在最终本地输出打包时将宽度扩为两倍（例如 3840x2160 单眼生成 7680x2160），Full-TAB 同理只在最终打包时扩高，OpenXR Vulkan 则继续直接提交两张完整分辨率的左右眼图像。
- 在恢复原始固定尺寸 GUI 的基础上，为完整左栏补充真正受窗口高度约束的纵向滚动视口；窗口高度不足或展开高级选项时，设置组、“重置 / 停止 / 运行”按钮及状态栏可作为连续内容滚动到底，滚动条无溢出时自动隐藏，不重新引入 DPI 或整体界面缩放逻辑。

## 2026-08-15

- 修复最前层 Controller Projection 启用后手柄外壳再次呈透明的问题：复用此前实机验证的“不透明外壳”根因对策，为新的 `array_size=2` 手柄覆盖 swapchain 绑定同一张双层深度 attachment；手柄外壳、按键和内部网格重新在后置 Controller View 内进行正确的逐眼深度写入与遮挡，同时保持手柄/激光/指南位于所有 Projection 与 Quad 图层最前方，GLB 材质、roughness 和光照参数不变。
- 提高 OpenXR 房间屏幕反射光强度：连续矩形面光的统一默认/Profile 增益由多数房间的 `3.5`（卧室 `5.0`）提升为 `6.0`，增强墙面和地面对屏幕内容的颜色与亮度响应；不改变房间基础亮度、手柄屏幕补光或 Glow。
- 隔离手柄与激光亮度：Controller View 使用固定 `0 EV` 的独立中性色彩管线，房间“场景亮度”只更新房间/前景管线，不再改变手柄、激光和指南；PBR 手柄仅接收自己的基础环境光、头顶/正面灯和屏幕补光，激光继续使用 Unlit 自发光材质，完全不接收任何灯光。
- OpenXR 手柄合成改为永远最前：Filament Multiview 新增透明 `array_size=2` Controller Projection swapchain，主 Projection 移除手柄、激光和指南后，按“房间/Glow/屏幕 → 设置与提示 Quad → 手柄/激光/指南”顺序提交；即使手柄空间位置位于菜单之后也始终显示在菜单上方。新覆盖层复用同一 Filament Engine，并保持同步 `flushAndWait()`，未重新启用已知会触发 `VK_ERROR_DEVICE_LOST` 的双 SwapChain deferred 路径。
- OpenXR 房间屏幕反射光改为旧工程同类的连续矩形面光算法：专用 Filament 房间材质直接按屏幕中心、法线、实际宽高、表面法线与距离计算连续照明，GPU 外圈采样只用于生成随屏幕内容变化的线性平均颜色；新 ABI 同时移除旧 24 点光近似，消除地面离散光斑并让墙面和地面获得同一片连续反射光。面光强度按房间最终 EV 反向补偿，因此降低“场景亮度”只压暗房间基线，不再连带熄灭屏幕反射光。

## 2026-08-14

- 修复小房间 `3d_theater` 屏幕反射光不可见：GLB 前墙实际位于屏幕前方，原 `0.08 m` 偏移使 24 段点光仍落在单面内墙背后；该房间现将灯源沿屏幕法线向室内偏移 `0.65 m`，并使用 Profile 独立 `8.0` 表面增益与 `5.5 m` falloff。反射材质创建和点光强度/位置保留一次性诊断日志，其他房间参数不变。
- OpenXR 房间屏幕反射光改用专用 Filament `Lit + customSurfaceShading` 材质：运行时只标记由烘焙 unlit 转换的墙面、地面和座椅，glTFio 继续原样绑定 baseColor 纹理；原纹理作为单次 emissive 亮度基线保持房间原貌，自定义表面着色仅叠加现有 24 段屏幕点光的颜色、法线响应、距离衰减与可见性，避免普通 PBR 光照过弱及按灯重复累加基线。
- OpenXR 房间“场景亮度”改为严格的会话态设置：滑块只影响本次运行，不写入 `settings.yaml` 或房间 Profile；每次启动及热切换到任意 GLB 房间时，均在 Profile 和灯光预设加载完成后恢复到菜单 `0 EV` 中间刻度。
- 精简 OpenXR 房间热切换成功日志：仅输出 `Environment hot switch complete: model=<房间>`，不再附带冗长的 GLB 与 panorama 完整路径。
- 修复 Artemis 等烘焙 `unlit` 房间在启用屏幕反射 PBR 后于 `0 EV` 下接近全黑、与天空盒亮度严重断层：运行时转换房间表面时同步复用原 baseColor 纹理作为 emissive 烘焙亮度基线，在保留原场景默认观感的同时继续叠加 Filament 环境光与 24 段屏幕反射光；天空盒、屏幕及原始 GLB 文件不变。
- 恢复全部内置 OpenXR 3D 房间的原始灯光基线：还原 `3D_Artemis` 被误改的补光、环境光和日夜预设颜色，并为卧室、客厅补齐中性的 `preview_exposure=0 EV` 与天空盒亮度；灯光预设中的旧 `env_exposure` 恢复为旧工程的材质乘数语义，不再错误覆盖 Filament/Vulkan 场景 EV，菜单亮度回到 `0` 时所有房间均回归中性基线。
- 修复 OpenXR 房间亮度调节后回到 `0 EV` 仍无法恢复、且房间与天空盒亮度失配：Filament Multiview 路径不再调用会重新开启 Post Processing 的 Native ColorGrading 曝光接口，房间与天空盒统一只在最终 Vulkan HDR Resolve 中应用一次曝光；逐眼回退路径继续使用 Native 曝光，避免重复处理与残留渲染状态。
- 修复 OpenXR 菜单切换前排/中排/后排时房间向菜单反方向旋转，以及 Pico 系统键重置朝向无效：首次稳定校准或系统重置时保存独立头部朝向锚点，实时换座和座位高度调整只替换目标座位 Pose，不再把点击菜单瞬间的侧向注视烘焙进新 Reference Space；运行时 `REFERENCE_SPACE_CHANGE_PENDING` 现在对带作者视点的房间同样生效，并在新基础空间上重新应用当前座位。
- 修复未选择 HDR 时普通 OpenXR 房间仍无条件创建全景管线并偶发触发 `create_panorama_pipeline / VkErrorUnknown`：Panorama shader 与 graphics pipeline 现在只在全景环境启用，GLB 房间与 HDR 热切换时会在 Vulkan 空闲后按能力变化安全重建 Projection pass，普通屏幕渲染不再被未使用的 HDR 可选管线拖入 fallback。
- 系统重排 OpenXR 设置菜单五个页签的垂直布局：顶部“重置为默认值”按钮收进标题安全区；所有分割线改为在控件和文字之前绘制并严格落在相邻控件之间；屏幕页将 `-90° / +90°` 移到三条滑轨上方，房间页按模型、座位、居中反射光按钮、座位高度、场景亮度顺序留出独立间距，普通操作按钮统一使用更大的粗体文字。
- 修复 OpenXR 房间屏幕反射光对多数内置场景完全无效：运行时加载环境 GLB 时，将带 baseColor 纹理的墙面、地面、座椅等烘焙 `KHR_materials_unlit` 表面选择性转为非金属高粗糙 PBR，使其接收 24 段屏幕反射点光；天空盒、假灯、屏幕、Logo、火焰、玻璃和无纹理发光色块继续保持 unlit，原始 GLB 文件及手柄材质不修改。
- OpenXR 设置菜单“房间”页新增 i18n“屏幕反射光”即时开关：开启状态以蓝色选中显示；关闭时立即从 Filament 房间场景移除 24 段屏幕边缘采样点光源并停止其 GPU 采样，重新开启后从下一帧恢复，仅影响房间几何，不影响手柄、Glow 或屏幕。
- 新增 OpenXR Filament 房间墙面屏幕反光：GPU Glow 采样现在额外输出与 Surround Glow 相同的 8×6 外圈 24 段时间平滑颜色，Presenter 将四边分段映射为仅作用于房间几何光照通道的局部点光源，不影响手柄和 Glow 图层；亮度、饱和度、上限、采样频率、平滑时间、照射范围、屏幕前偏移及阴影均可由环境 `common.json` 或房间 `profile.json` 覆盖。
- 重排 OpenXR 设置菜单内容区：放大“视频画面 / 立体深度 / 辉光特效 / 场景设置 / 屏幕设置”标题并与普通选项保持一致；画面、深度和屏幕页移除底部“恢复默认”，在各自标题右侧统一提供 i18n“重置为默认值”操作且仅重置当前页；按设置分组加入细分割线，画面双栏增加纵向分隔。

## 2026-08-13

- 增强 Vulkan Projection Composer 偶发首帧 `VkErrorUnknown` 诊断：初始化过程现在记录 shader、descriptor、render pass 及每条 graphics pipeline 的精确创建阶段，fallback 同时输出完整 Python 调用栈；保留下一帧自动重试恢复行为，后续实机日志可直接定位具体失败的 Vulkan API。
- 修复 OpenXR 房间热切换的单向 HDR 丢失：从 GLB 房间切换到 HDR 全景时会保留仍用于手柄的 Filament Bridge，但其透明前景 resolve 现在对已绘制的 panorama 使用 LOAD 叠加，不再清空 HDR 背景；直接启动 HDR 与 `房间 → HDR` 现统一使用相同合成顺序。
- 修复 Requirements Compliance CI 依赖：测试矩阵安装 `opencv-python-headless`，使图像诊断测试可正常导入 `cv2`，同时避免引入桌面 GUI 组件。
- OpenXR 房间菜单接通 Presenter 线程内安全热切换：在帧边界预校验目标 Profile/GLB/全景文件，等待 Vulkan 与 Filament 空闲后原位替换 GLB 或卸载旧环境，释放旧 HDR GPU 资源并加载新 Profile，随后实时刷新座位、屏幕、灯光、Glow 条件和参考空间校准；支持 `Default ↔ GLB 房间 ↔ HDR 全景`，仅在成功后持久化 GUI 环境选择，失败时尝试恢复旧资源并保留当前会话。Native Bridge 新增只卸载房间资产、不销毁手柄和 Engine 的 `filament_bridge_unload_glb` ABI。
- OpenXR 设置菜单房间页的环境选择列表由仅显示 GLB 房间扩展为同时显示 `Default` 和全景环境：凡 Profile 声明 `environment_type=panorama` 且配置有效 `background.image` 的 HDR/全景图片目录，都会使用自身 `display_name` 和 GUI 当前语言加入选择项；用户可从任意 GLB/HDR 环境直接切回 `Default`。
- OpenXR 设置菜单现在跟随房间座位变换：切换前排/中排/后排时，将新旧座位的完整位置与朝向差应用到当前菜单 Quad；实时调节座位高度时同步移动菜单。菜单会保持用户面前的原有相对位置，包括此前通过 Grip 手动拖动后的偏移，不再留在旧座位。
- OpenXR 设置菜单房间页的三档座位选项统一为 `Front / Middle / Back` 稳定键，并通过 GUI 共用 i18n 在中文界面准确显示“前排 / 中排 / 后排”；座位索引和房间 Profile 坐标映射保持不变。
- 统一 7 个内置 3D 房间的座位视点 Profile：每个 `profile.json` 现在都包含按 `Front / Middle / Back` 排列的三份 `view_poses`，先将房间原先实际启用的视点完整复制到三档，并将默认档设为 `Middle`；后续可在房间预览工具中逐档定位并按 `P` 写回真实坐标，当前修改不会改变房间启动时的默认观察位置。
- 扩展 OpenXR 设置菜单运行时调节范围与房间页：深度强度上限由 `0.5` 提升至 `1.0`，屏幕高度偏移由 `±2 m` 扩为 `±10 m`；房间页扫描并显示带 GLB 的房间模型（选择后写入 GUI 同一 `Environment Model` 配置、下次 OpenXR 启动生效），提供前排/中排/后排三档座位，以及实时更新当前参考空间的 `±3 m` 座位高度和实时调用 Filament 的 `-8~8 EV` 场景亮度。
- `preview_room_layout.py` 将房间座位 Profile 规范为按 `Front / Middle / Back` 排列的三个 `view_poses`：Tab 依次切换三档并自动进入 VIEW 编辑，窗口标题持续显示当前档位和 `1/3~3/3`，`V` 单独切换 SCREEN/VIEW 编辑目标，`P` 将当前档位连同另外两档一起保存到房间 `profile.json`；读取旧 Profile 时按名称识别 Front/Center/Back，避免旧数组顺序误映射。
- 根据第二轮头显可读性测试继续调整 OpenXR 菜单：页签标题增大到 `32 px` 并优先使用 GUI 对应语言的粗体字体；菜单纹理由 `1024×768` 加高为 `1024×832`，世界 Quad 高度同步由 `0.71 m` 增至 `0.77 m`，射线 UV 映射同步更新，底部“恢复默认”和旋转按钮完整落在最外框及内容卡片内。
- 优化 OpenXR 设置菜单页签可读性与多语言布局：页签标题字号由 `21` 提升至 `27`，相邻选项间距由 `2%` 收紧至 `0.8%`；每个页签保留统一的最小激光命中宽度，其余空间按 GUI 当前语言翻译后的显示字符宽度动态分配，绘制矩形与命中区域共用同一布局结果。
- OpenXR 设置菜单文字改为复用 GUI 的统一 i18n `MESSAGE_CATALOGS/gettext`：运行时直接读取 GUI 保存的 `Language` 并按 GUI 语言目录规范化，不再在 OpenXR 纹理代码中硬编码中英文判断或维护第二份翻译表；辉光页签、模式名称及既有画面/深度/房间/屏幕控件均走同一翻译入口。
- OpenXR 设置菜单新增条件式“辉光”页签：仅未加载 3D 房间模型的 `Default` 环境显示，并列出运行时全部辉光模式 `Surround Glow / Glow / Veil / OFF`；激光点击会直接切换现有辉光状态和强度通道，加载任意房间模型时页签及命中区域完全移除。
- OpenXR 设置菜单新增“深度”页签：接通深度强度滑轨及步进按钮、2D/3D 实时切换、交叉眼开关和深度默认值恢复（`Depth Strength=0.25`、`Cross Eyed=false`），全部直接更新运行时并保存。屏幕页新增沿初始头部视线方向调节的屏幕距离、每次将屏幕自身旋转角增减 `90°` 的按钮，以及一键恢复房间 Profile 初始位置、大小、旋转和曲率的“恢复默认”。
- OpenXR 菜单所有滑轨两侧的 `− / +` 由装饰图标升级为独立激光命中按钮：每次 Trigger 单击严格按该参数自身的步进减小或增加一次，并复用滑轨的上下限、Min/Max LOD 联动、运行时更新与保存逻辑；`Render Scale` 的加减按钮同样按 `5%` 步进，并只调度一次 Projection swapchain 重建。
- 修复 OpenXR 设置菜单与虚拟屏幕重叠时菜单光标被屏幕边缘吸附带偏：设置 Quad 可见期间，控制器交互射线不再执行只服务于虚拟屏幕的 6° 边缘吸附，菜单命中和菜单内光圈始终跟随真实激光；关闭菜单后屏幕边缘吸附自动恢复。
- OpenXR 画面菜单新增独立的 `Render Scale` 滑块，范围 `50%–200%`、步进 `5%`：以运行时推荐的逐眼宽高为基准计算 Projection 目标尺寸，松开 Trigger 后在下一帧边界等待 Vulkan 空闲、销毁旧 Projection/Filament 渲染目标并按新尺寸重建，避免拖动期间连续重载；数值保存为 `OpenXR Render Scale`，与捕捉/推理使用的 GUI `Render Scale` 保持隔离。
- 按参考播放器重排画面和屏幕菜单：画面参数使用“名称、减号、细滑轨、加号、数值”的紧凑参数行，不增加色盘；屏幕页顶部改为直面、微曲、中曲、重曲四个图标卡片，分别对应 `0°/20°/30°/41.25°` 半弧角（完整包角 `0°/40°/60°/82.5°`）。曲率档位实际贯通 Projection 屏幕几何、激光命中和光标姿态，而非仅改变按钮显示。
- 精简 OpenXR 设置菜单顶部：删除重复的“OpenXR 设置”标题栏和关闭按钮，三页签直接置顶并释放更多内容空间；关闭菜单继续使用既有的菜单外 Trigger 点击交互。
- OpenXR 设置菜单视觉调整为参考播放器的紧凑深蓝控制台风格：使用高不透明度海军蓝外壳、独立标题/导航/内容卡片、蓝色选中指示和细滑轨、灰白层级文字、圆形关闭按钮与蓝白激光光标；三页共享统一视觉语言，降低环境画面穿透干扰并提升头显内文字和控件辨识度。
- 根据首轮实机测试修正 OpenXR 设置菜单：标题栏、关闭按钮、页签、滑块内容和底部操作区改为互不重叠的独立布局；菜单按打开瞬间的完整头显朝向正对用户并保持世界锁定；激光命中菜单后可按住任一 Grip 以六自由度自由拖动；画面页新增“恢复默认”，一次性恢复全部颜色、LOD、MIP Bias 和 RCAS 默认值，并通过单个运行时快照及一次原子写入保存。
- 开始实现 OpenXR Quad 设置菜单首个可运行阶段：新增世界锁定、单 swapchain、双眼 `BOTH` 可见的三页签菜单，支持左右手柄在虚拟屏幕和键盘外短按 Trigger 唤出、菜单优先输入、首手锁定、页签/按钮/滑块命中以及菜单外关闭；画面页已接通亮度、对比度、饱和度、Gamma、色温、色调、Min/Max LOD、MIP Bias 和 RCAS 的实时完整快照更新与释放后原子保存，屏幕页已接通尺寸、高度和平面/曲面实时调整。菜单纹理与 swapchain 缓存复用，连续交互最多 `20 Hz` 重绘；房间 Profile 参数和 GPU MSDF 图元扩展保留到下一阶段。
- 新增 `docs/11-openxr-quad-settings-menu-implementation-plan.md`：确定以单个双眼可见 OpenXR Quad 实现“画面/房间/屏幕”三页签设置菜单，定义屏幕外 Trigger 唤出、激光点击与滑块交互、GPU MSDF 图元渲染、运行时热更新、`settings.yaml`/房间 Profile 持久化、动态座位与灯光预设，以及分阶段测试验收标准。
- 修复 OpenXR 3D 房间启动后实际视点稳定高于 Profile `view_pose` 约 1 米的问题：参考空间校准改为跨两个 XR tick 的闭环流程，首次建立临时座位空间，下一帧使用 VDXR 实际返回的头部姿态测量并抵消 STAGE 地面高度更新；同时持续跟踪子空间到原始 STAGE/LOCAL 的变换，避免二次校准混用坐标系。该修复不写死玩家身高，也不改变预览工具保存的座位坐标。
- 对齐旧工程的 OpenXR 参考空间变更策略：普通带有固定 `view_pose` 的 3D 房间在 VDXR 后续发送地面/边界空间变化事件时保留已经校准的子空间，不再销毁空间并重新执行座位校准，消除运行一段时间后房间突然漂移；只有显式设置 `auto_center_on_screen=true` 的 Profile 才接受事件并重新定位。
- 修正 OpenXR `3D_巨幕影院` 的默认屏幕朝向：`3d_cinema/profile.json` 中屏幕由错误的正面 `0°` 改为面向默认观众席的 `-180°`，避免从屏幕背面观看导致桌面内容水平镜像；屏幕位置和尺寸保持不变。
- 修复 `NONE + 3D 房间` 启动后 Presenter 线程在 `set_controller_visible` 崩溃：环境专用 Filament Bridge 不再接收任何未加载手柄实体的可见性、姿态、输入、激光或指南调用；房间继续由 Filament 渲染，代理手柄与说明继续由 Vulkan Projection 绘制。
- 修复选择 `NONE` 手柄后 OpenXR 3D 房间模型及纹理消失：无手柄模式现在只跳过左右手柄 GLB；选中 3D 房间时仍会初始化 Filament、读取并加载完整环境 GLB 及其材质纹理。只有 `NONE` 与空白环境同时使用时才跳过未使用的 Filament Engine。
- 修复 `NONE/profile.json` 的手柄校正参数未作用于 Vulkan 代理手柄：无 GLB 模式现在与 Filament 手柄使用相同的 `grip × rotation × offset` 变换，`model_offset`、`model_rotation_deg` 和运行时校正会立即改变代理手柄姿态；B 键说明锚点同步使用校正后的模型矩阵。
- GUI 手柄模型列表不再额外注入硬编码的 `None`，现在完全使用 `controllers/` 中实际发现的目录；`controllers/NONE/profile.json` 成为唯一的无模型选项。旧配置中的 `None` 会以大小写无关方式匹配扫描到的 `NONE`，不会错误回退到 `PICO`。
- 手柄模型设为 `None` 时，正常发现并加载 `controllers/NONE/profile.json`；控制器 Profile 发现现已支持无 GLB 的 profile-only 模式及大小写无关选择，同时仍绕过 Filament/手柄 GLB。FPS/状态面板明确显示 `Model: None`，并将名称纳入纹理缓存键。
- 手柄模型设为 `None`、Filament 未加载时，禁用长按右手柄 `A+B` 切换手柄品牌；快捷键输入层不再发出切换动作，执行层也拒绝无 Filament 的切换，避免只显示品牌信息却没有对应 GLB 模型。
- 修复 Vulkan HDR 全景被诊断级 `1024×512` 降采样导致的严重模糊：Projection fallback 不再设置软件纹理上限或执行任何缩放，按 HDR 文件原始分辨率直接上传（包括 `8192×4096` 和 `10000×5000`）；只有源尺寸超过 Vulkan 设备硬件极限时才明确报错。
- 重构 Vulkan HDR 全景的头部旋转映射：每个 XR tick 直接向 shader 传递 OpenXR 原始眼睛四元数和 FOV 切线，由 GPU 构造视线并进行四元数旋转，不再经过 CPU/GLSL 矩阵转置与投影符号约定；增加一次性中心采样 UV 诊断，用于区分运行时姿态未更新与 Projection 合成问题。

## 2026-08-12
- 新增 Vulkan Projection 全景 pass 的 GLSL/SPIR-V 基础 shader（equirectangular 方向采样）；后续 Composer 接入使用该 pass，不依赖 VDXR 的 equirect composition 扩展。
- 接入 Projection pass 的全景 pipeline 创建、descriptor 绑定、资源状态跟踪和 draw 提交接口；现有屏幕/Glow/手柄路径保持不变，待 HDR Vulkan 源纹理上传完成后即可作为首个背景 draw。
- 补充全景 draw 的源纹理 shader-read barrier，避免 HDR Vulkan 背景首次采样时布局不正确；现有 208 项 OpenXR 回归测试继续通过。
- 接通 HDR Vulkan 源纹理的一次性上传、每眼 inverse view-projection 参数和 Composer 首个 panorama draw；后续屏幕/Glow/手柄 pass 通过时间线在其上叠加。
- 修复 panorama draw 与屏幕 pass 共用 descriptor 时的提交竞态：背景 draw 完成后才允许后续 pass 更新 descriptor，避免 Composer 因 `VkErrorUnknown` 回退。
- 修复 HDR 背景被屏幕 pass 清除：panorama timeline 现在参与 `load_target` 判断，后续屏幕、Glow 和手柄绘制均在 HDR 背景上 LOAD 叠加。
- 修复 HDR 全景跟随头部平移：panorama inverse view-projection 仅使用头部旋转和每眼 FOV，移除位置平移，使环境保持世界锁定。
- 加强全景世界锁定：不再从完整 view 矩阵清零平移，而是直接从 OpenXR 四元数重建纯旋转 view，排除眼位/重定位平移残留。
- 修复 VDXR 不支持 equirect 时的日志刷屏：HDR 跳过原因现在每个会话只记录一次。
- 移除无效的 VDXR 普通 Quad HDR fallback：它不是头部跟随的全景、可能遮挡 Projection 且会造成上传卡顿；VDXR 缺少 equirect 扩展时现在只记录一次并保持稳定 Projection，等待 Vulkan equirect pass。
- 限制 HDR Quad fallback 的初始化成本：全景图先缩放到最长边 1024，再上传一次；初始化失败后不再每帧重试，避免 XR 只运行几帧和日志重复。
- 补充 HDR 全景能力诊断：启动时记录运行时是否暴露并启用 `XR_KHR_composition_layer_equirect2`，全景 swapchain 添加采样用途；若 VDXR 不支持该扩展，会明确记录跳过原因。
- 修复 OpenXR HDR 环境未显示：解析环境 profile 的 `background.image`，并在运行时启用可用的 `XR_KHR_composition_layer_equirect2`，将 HDR 全景作为独立背景层提交到 Projection 之前；保留 `D2S_FILAMENT_PANORAMA` 覆盖入口，缺少运行时扩展时安全回退并输出明确日志。

## 2026-08-11
- 修正重置默认值：电影模式对应的补洞模式恢复为“均衡”；仍使用 `Distill-Any-Depth-Base`、`Base` 和 `518` 深度细节。
- 调整 GUI 重置默认值：深度模型固定优先使用 `Distill-Any-Depth` 的 `Base`，深度细节为 `518`；补洞模式默认设为“高质量”。
- 修复 GUI 切换语言/刷新界面时覆盖捕获帧率选项的问题：初始化和动态刷新现在统一显示 `Auto` 及 5 FPS 到 90 FPS 的 5 FPS 间隔选项。
- 自动捕获限频进一步改为严格 `平均 SBS + 5 FPS`，不再设置 24 FPS 的最低自动档位；只限制不超过显示器刷新率，低 SBS 也按实际吞吐量降低捕获频率。
- GUI 手动捕获帧率选项统一为 5 FPS 间隔，从 5 FPS 到 90 FPS；保留 `Auto`，便于与自动 `SBS+5 FPS` 限频策略对照。
- 修复 OpenXR 自动捕获限频过于激进的问题：自动模式现在根据 60 秒 SBS 平均帧率设置为 `SBS+5 FPS`，并限制在 24 FPS 与显示器刷新率之间；例如 SBS 约 60 FPS 时捕获约 65 FPS，不再直接跳到 120 Hz，减少无效捕获和待处理帧丢弃。
- 修复 Vulkan Projection 在 RCAS=0 时黑屏闪烁的问题：EASU 绘制与 quality→MIP copy 不再复用同一 descriptor set，避免 copy 更新覆盖 EASU 的源纹理；关闭 RCAS 现在仍会稳定执行 EASU→MIP→屏幕提交。
- 修复 4K 缩放到 1K 后 OpenXR Projection 仍被错误缩小的问题：将 Projection 配置与 Filament 场景配置分离，`Render Scale` 只控制捕获后的推理输入尺寸，Projection swapchain 默认保持头显原生目标尺寸；4K→1K 现在与原生 1K 共用同一条 EASU 后处理和投影路径。保留 `D2S_OPENXR_RENDER_SCALE` 作为显式诊断覆盖。
- 修复“4K 缩放档”与 OpenXR 屏幕质量链的分流：投影 Composer 始终依据实际 SBS 左眼纹理尺寸选择采样策略，不再以原始 `capture_size` 覆盖已缩放的 1K/2K SBS 尺寸；OpenXR Projection render scale 仅在处理输入属于完整 4K 档时采用 GUI 缩放值，1K/2K 输入固定使用头显原生投影目标。3K（85%）也从错误的 `native_mip` 桶移入 `upscale_easu → RCAS → MIP` 路径；只有完整 4K SBS 可使用 `native_mip`。这样 4K 下采样得到的 1K、2K、3K SBS 与同尺寸原生输入复用同一最终质量链，避免二次缩小或错误 MIP 采样。
- 4K 到低分辨率推理输入的 Triton 预处理补充面积下采样内核，并提供一次性预推理图像及同源下采样候选导出脚本，便于确认进入深度推理前的真实 RGB 尺寸和过滤结果；诊断默认关闭，不影响正常运行。
- GUI“手柄模型”新增 `None` 性能隔离选项：选择后不创建 Filament Engine，不加载手柄或环境 GLB，Vulkan Projection Composer 在每眼同一次 draw 中使用未经品牌校正的 OpenXR grip/aim 姿态绘制左右两个 `8×8×8 cm` 本地立方体及双手激光；左手立方体使用固定不透明蓝色，右手使用固定不透明橙色，各面边界使用基于屏幕空间导数抗锯齿的黑色接缝线区分，颜色不受时间、背景或表面朝向影响，并在无深度附件的 overlay pass 中丢弃背面片元。修正立方体 `+Y/-Y` 面与法线相反的三角形绕序，避免背面过滤造成对称凹口和缺面。立方体与正式手柄一样在静止 5 秒后隐藏，激光保持彩色流动、旧版流向和尖端收窄几何，Tool Quad 的 B 键提示锚定到右手立方体的前右顶点。PICO 仍是默认值，该选项仅用于比较 Filament 与纯 Vulkan 代理的 XR/SBS 开销。
- OpenXR FPS 面板新增实际捕获帧率，统计软件捕获节流后真正进入 raw queue 的帧，而不是显示配置目标；捕获停顿时归零，并与 XR、SBS 和延迟使用同一低频快照更新。
- 修复删除 Glow2/Frosted 后 Surround 模式编号仍沿用旧值，导致其误入未清屏的前景 `LOAD` RenderPass 并跨帧累积的问题；Surround 重新使用先清除 Projection 目标、再绘制背景辉光的正确路径，消除整幅画面的重影拖尾。
- 删除 Glow2 和 Frosted 两个 Glow 特效及其 Vulkan shader 分支、运行时模式、旧 profile 参数和预过滤路径；控制器切换序列简化为 `Surround → Glow → Veil → Off`，Veil 保留原有几何与显示行为。
- 捕获帧率新增固定 24 Hz 和 30 Hz 档位；OpenXR 的“自动”档新增基于真实 SBS 吞吐量的运行时节流：每 60 秒按 SBS 平均帧率评估一次，并在显示器基础刷新率、30 FPS、24 FPS 之间最多升降一个档位，不再通过短时恢复高捕捉率制造周期性负载尖峰。该机制只在复制和推理前丢弃过量捕捉回调，不改变 SBS 输出、OpenXR 提交率或手动帧率档位，并通过 `capture_target` 记录当前有效捕捉上限。
- 将“并行推理”移入“显示高级立体参数”，放在“交叉眼”左侧；启动默认值和 GUI 重置值统一改为“单路推理”。中英文 Tooltip 明确标记多路推理仅供实验，当前不建议启用两路或三路，因为实机效果更差；用户仍可手动选择多路进行对照测试。
- 优化 Vulkan Projection Composer 的 CPU 侧逐帧命令录制：缓存 MIP 生成所需的 barrier/blit 描述、Render Pass begin/viewport/scissor 描述和常用 Image Barrier 描述，避免 Python/CFFI 在每个 XR tick 重复构造相同 Vulkan 结构；每帧仍按最新头姿重新录制和提交命令，不复用旧 swapchain 图像，不改变屏幕分辨率、MIP/RCAS 质量链、SBS 推理调度或遮挡关系。RTX 2060 单路推理实测基线为 SBS 14 FPS / XR 37 FPS，仅启用 MIP 模板缓存后为 SBS 10 FPS / XR 50 FPS；三类缓存合并时曾实测约 SBS 9 FPS / XR 67 FPS。FPSBreakdown 新增 `vk_mip_tpl_*`、`vk_rp_tpl_*` 和 `vk_barrier_tpl_*` 命中/新建统计，用于确认预热后只复用 CPU 描述而非复用显示帧。
- 移除 HP、Index、PICO、Quest、Vive 和 YVR profile 中的强制材质覆盖，恢复各手柄 GLB 自带的 roughness、metallic 和 specular 参数；手柄灯光配置保持不变。
- 恢复 PICO GLB 正式材质参数 `KHR_materials_specular=[2,2,2]`、`roughnessFactor=0`；PICO profile 保留 Head/Top 灯光阴影投射，其他品牌继续使用正式版光照与各自 GLB 原始材质。
- 将已通过 `multiview_controller` 诊断的正常手柄纹理路径接入正式 Vulkan Projection Composer：Composer 开启时统一使用 layered Filament multiview，诊断模式只控制是否跳过 SBS/Glow，不再改变手柄渲染路径；正式灯光参数保持不变。
- 修正正式 Composer 的遮挡顺序为“Glow → SBS 屏幕 → Filament multiview 环境/手柄 → 激光”，让 Glow 覆盖虚拟屏幕，同时保证手柄处于最终前景。
- 修复 layered Filament HDR 前景覆盖整张 Projection 目标：multiview producer 的空背景改为透明 alpha，HDR resolve 使用 LOAD 与标准 alpha blend，只覆盖实际渲染的手柄/环境像素，不再清除或遮蔽 SBS 与 Glow。
- 将全部 Vulkan GLSL、SPIR-V 和 shader manifest 从仓库根目录迁入 `src/shaders/`，使运行时资源遵守 `src/` 产品发布边界；同步更新 Python 加载路径、编译脚本、CI、测试及需求追踪，根目录不再保留 Shader 运行依赖。
- 新增 DLL 外部的手柄材质覆盖参数，并为 HP、Index、PICO、Quest、Vive 和 YVR 统一启用高反光外观：运行时在 GLB 加载或品牌热切换后应用 `roughness=0.08`、`metallic=0.85` 和 `specular=[2,2,2]`。后续可直接修改各手柄 `profile.json` 调整反光，无需重新编译 Bridge。
- 将已通过实机验证的“房间先画、手柄后画，后置 foreground pass 只清深度不清颜色”不透明外壳对策移植到 Filament multiview：新 ABI 为双层 HDR swapchain 绑定一张 `array_layers=2` 的深度图，手柄、激光和指南重新由后置立体 View 在新深度上绘制。GLB 模型、原始材质、roughness 和光照参数保持不变。

本文件只记录用户可感知的功能、行为变化、重要修复和架构里程碑，不记录逐次调试过程。新记录按日期倒序追加，并将同一目标的连续修改归纳为一条有效结果。

## 2026-08-10

### Vulkan / Filament multiview 验证与修复

- **VDXR array layer 路由已通过实机验证：** 一个 `array_size=2` 的 OpenXR Projection swapchain 将 layer 0 提交给左眼、layer 1 提交给右眼，头显稳定显示左红右绿；纯 Vulkan multiview 测试同时记录 `fragment_counts=(1, 1)`。这确认 `gl_ViewIndex`、`viewMask=0x3`、双层 attachment 写入以及 OpenXR `imageArrayIndex=0/1` 均正常。
- **Filament multiview 双眼路由已通过实机验证：** layered Bridge、普通手柄顶点路径和红绿诊断材质现在均使用正确的 multiview stereo variant，头显可见左眼红色、右眼绿色。Filament 1.75 内部 `clearDepth` 也改为注册生成的 `CLEARDEPTH_MULTIVIEW` 包，不再出现材质 stereo type 与 Engine 不兼容的警告。
- **恢复正常手柄纹理的 layered HDR 合成：** Filament multiview 先输出双层 `R16G16B16A16_SFLOAT` HDR producer，再由 Vulkan resolve 到 OpenXR 左右眼 Projection 目标；Projection-only 诊断与正常 Composer 共用同一 resolve，并只消费一次 Filament render-finished semaphore，避免完成信号未发布、手柄消失或输出停滞。
- **恢复 multiview 手柄不透明外壳：** 撤销尚未实机证实、且会全局改写 Filament `getWorldCameraPosition()` 的逐眼 PBR 相机补丁；并恢复此前已通过实机验证的合成约束：环境、手柄 GLB、激光和指南/文字不再拆成独立 foreground/controller View，而是进入同一个 stereoscopic View，共享逐眼深度和 glTF `OPAQUE` 合成链。手柄灯仍由 light channel 隔离，PICO 原始材质参数不作修改。
- **保留安全边界：** 会触发 device lost 的双 SwapChain deferred 路径继续禁用；正式 multiview 使用一个双层 producer 和统一 Vulkan/OpenXR 提交者，不恢复已证实不稳定的双 Renderer/双 deferred 完成路径。
- **验证结果：** Filament 1.75 完整补丁已在 Windows x86_64、Linux x86_64 和 macOS arm64 三个平台重新构建成功，生成的 Bridge 二进制已同步；Python/OpenXR/Bridge 专项测试 `226 passed`，shader 合规检查通过。

## 2026-08-09
- 新增纯 Vulkan multiview 双眼诊断，绕过 Filament、模型、SBS 和 Glow：一个 `viewMask=0x3` 的 render pass 向 OpenXR `array_size=2` swapchain 绘制全屏三角形，并直接根据 `gl_ViewIndex` 选择红色或绿色。一个主机可见的单像素计数器记录 fragment invocation 是否实际为 view index 0 和 1 执行，不依赖 OpenXR 图像回读。在普通逐层 array 路由通过、但 Filament 手柄输出仍为双眼红色后，该测试将 Vulkan multiview 执行与 Filament 隔离开来。
- 新增隔离的 VDXR Projection array 能力测试：一个 `array_size=2` 的 Vulkan Projection swapchain 将纯红色 layer 0 提交给左眼 view，将纯绿色 layer 1 提交给右眼 view，完全不加载捕捉、推理、Filament、屏幕、Glow 或手柄。专用 PowerShell 启动脚本用于验证后续 Filament multiview/离屏合成工作的准确前置条件。
- VDXR array 路由实机通过后，新增下一项隔离的 Filament multiview 门控测试：可选启动脚本通过一次立体 Filament 提交，将配置的环境和手柄渲染到一个 `array_size=2` 的 Projection swapchain，同时关闭 Vulkan SBS/Glow 合成。在该 layered Filament 输出通过目视验证前，正常逐眼 producer 仍为默认路径。
- 修复仅 Filament multiview 诊断的输出租约：未使用的 SBS 帧现在会被释放，不再成为显示帧；CUDA/Vulkan adapter 也不再为从未进入源图像采样的眼睛索引 release semaphore slot。
- 新增可选的 Filament 手柄 multiview 诊断材质：将每个手柄 GLB primitive 替换为由 `getEyeIndex()` 选择的左眼纯红、右眼纯绿，从而把 shader view-index 路由与手柄纹理、光照和环境渲染隔离。眼睛索引现在在 Filament 支持的 vertex stage 中读取，并通过自定义 interpolant 传递给 fragment stage；启动脚本同时启用后端 stereo tracing，可在一次运行中关联 View 状态、Vulkan render pass view mask、shader `ViewIndex` 和头显结果。
- 为 Filament multiview 手柄诊断新增一次性 GPU 回读。它并排保存实际 array layer 0 和 layer 1 输出，并记录每层非背景区域的平均 RGB，在不改变持续渲染路径的前提下区分 Filament 渲染故障与 OpenXR array-layer 提交故障。诊断会将每层复制到紧密排列的主机可见缓冲，并直接等待 Filament finished semaphore，避开 Windows Vulkan 驱动上不可靠的 optimal-to-linear 图像复制和含义不明确的中间同步。
- 修复 Filament 1.75 为 multiview Engine 选择内部 `clearDepth` 材质的问题：修补后的后端现在注册生成的 `CLEARDEPTH_MULTIVIEW` 包，而不是不兼容的 instanced 包，消除了 layered 手柄测试中的 stereo type 不匹配，并使深度清除几何与默认材质、skybox 遵循相同的 multiview 契约。
- 在不改变合成顺序的前提下减少 OpenXR Projection 工作量：当重新构建的 Filament Bridge 提供新的 background-frame ABI 时，Composer 前置 pass 只渲染环境和普通前景，手柄和指南仅由现有 Composer 后置 overlay 渲染，不再逐眼重复绘制。手柄和不依赖深度的指南现在共享一个全分辨率 overlay View，在保留指南优先级的同时省去另一轮逐眼后处理 pass。旧版 Bridge 二进制继续使用原有兼容路径。
- 修复补洞模式实时切换，使内部总开关随所选模式同步变化：`none` 禁用补洞，`balanced` 和 `quality` 无需重启进程即可立即恢复边缘感知路径。运行时比较补洞性能时，模式、半径、强度和实际执行的 kernel 现在保持一致。
- 在 GPU 缓冲处理和 raw queue 投递之前增加软件 pacing gate，修复 WindowsCaptureCUDA 对目标 FPS 的执行。该机制补偿高刷新率显示器上忽略 `minimum_update_interval` 的捕捉后端：选择 60 FPS 后，每秒约接收 60 个源帧，而不是处理全部 120 Hz 回调；“自动”仍跟随检测到的显示器刷新率，设置 `D2S_WGC_SOFTWARE_THROTTLE=0` 可关闭新增门控以进行对照测试。
- 新增感知渲染状态的 OpenXR 推理准入控制：配置的两路/三路 TensorRT worker 池仍保持可用，但 Presenter 正在合成或队列中已有 SBS 结果时，最多只允许一个深度任务处于 in-flight。Presenter 缺帧时会自动恢复额外 worker，在保持突发吞吐量的同时避免持续挤占 Vulkan/Filament；现有最新帧策略和显式 worker 选择保持不变。

## 2026-08-08
- 修正 OpenXR FPS 面板：SBS FPS 现在统计 Presenter 实际消费的唯一立体帧，而不是速度更快的推理 producer 帧率。OpenXR Vulkan command ring 默认改为九个 slot（可通过 `D2S_OPENXR_VULKAN_FRAME_CONTEXTS` 覆盖），使屏幕、Glow 和手柄/深度的多 pass 提交能够跨越三个 swapchain image，避免过早复用三 slot fence 而阻塞 Presenter。
- 新增可选的延迟 SBS pacing 捕获，用于帧时序诊断。经过 15 秒播放准备窗口后，它会在 Projection 合成前记录 300 个新左右眼 Vulkan 立体输出的元数据，同时只截取六张均匀分布的 SBS 截图；稀疏 GPU 降采样回读使用有界三 slot ring 和后台 PNG writer，避免逐帧回读造成严重吞吐量失真。JSON manifest 会记录源 frame ID、时间戳以及跳过或失败的图像；未使用诊断启动脚本时，正常推理和显示行为不变。
- 通过四条完整外壳边缘之间的逐分量最大值混合稳定 Surround Glow 转角：重叠区域既不会累加亮度，也不会被后绘制边缘覆盖，任何渐隐几何都不会露出已清除背景形成黑缝。每条边缘网格也由 96×48 降至 48×24，使每帧立体几何顶点数从 221184 降至 55296。Glow 现在具有由 profile 驱动的独立 30 Hz 更新率和 0.10 秒 GPU 时域历史混合，用渐进的线性颜色过渡替代可见的 2～3 帧颜色跳变，同时不影响手柄灯光采样。
- 将已验证的 Composer 后置手柄 pass 提升为 OpenXR Vulkan 正常默认路径，无需诊断启动脚本即可保持最终“环境 → 屏幕/Glow → 手柄/激光/指南”顺序；仍保留显式环境覆盖开关用于回退测试。
- 新增 profile 驱动的手柄屏幕反射光：现有异步 Vulkan/CUDA 线性屏幕颜色归约结果现在会驱动朝向手柄、经过平滑且受亮度限制的方向光。亮度只取决于屏幕亮度，与屏幕距离无关，并且只影响手柄前景光照通道，不影响房间或 Glow；所有采样、lux、饱和度、平滑和阴影设置仍位于 DLL 外部。
- 将 Filament 环境、HDR 手柄环境光以及 Head/Top 手柄灯光外观参数从 native DLL 移入共享/环境 profile。带版本的运行时 lighting ABI 现在接收最终 lux/candela、颜色、相对头显偏移、衰减和阴影标志，因此调整光照不再需要重新构建 Bridge。
- 重新平衡手柄照明：顶部灯作为 100% 主光，跟随头显的正面灯作为 70% 补光，在保持自然明暗和前景合成顺序的同时改善顶部表面与按键辨识度。
- 新增可选的 Composer 后置 Filament 前景 pass 和最小 LOD0 启动脚本：先渲染环境，再由 Vulkan 绘制屏幕/Glow，最后由现有手柄/激光/指南 View 渲染前景，以便在不重新加载模型的情况下验证前景优先级。

## 2026-08-07
- 修复 Vulkan Projection Composer 质量链与 Filament 环境及手柄输出的合成：质量 pass 现在会等待 Filament 完成，并在绘制 SBS 屏幕前通过 `LOAD` 保留颜色目标。
- 修复 Vulkan deferred compositor 的 OpenXR 3D Depth 开关和 Depth Strength 实时调整：现在每帧都会将当前运行时深度值传入 Vulkan stereo push constants，因此 `0.0` 会输出单目画面，手柄调整会在下一渲染帧生效。
- 移除存在冲突的 OpenXR Glow 连续调节快捷键；现有 3D Depth 控制保持不变。

## 2026-08-06
- Virtual Desktop 不支持 OpenXR Quad swapchain，移除 Screen Quad Reprojection 实验启动脚本；默认输出继续使用已验证的 Projection swapchain Vulkan Composer 路径，不影响左右眼 Projection Layer 提交。
- Vulkan Projection Composer 完成屏幕质量链：按输入/头显档位执行原生 LOD0→MIP、Lanczos2→RCAS→MIP 或 EASU→RCAS→MIP，再进行最终平面/曲面屏幕投影。质量链保持每个 OpenXR 帧实时执行，确保 MIP LOD、MIP 偏移和 RCAS 参数改动立即生效；提供关闭质量链的 LOD0 性能对比开关，但不降低输入或 swapchain 分辨率。
- 捕捉设备高级设置新增 Vulkan 屏幕采样实时参数：最小 LOD、最大 LOD、MIP 偏移和 RCAS 锐化，并提供中英文说明和建议值；默认 `max LOD=0.35`、`MIP bias=-0.35`、`RCAS=0.50`。
- 并行深度推理的 native TensorRT slot 改为独立 engine/runtime/context、CUDA stream、输入输出缓冲和完成 event，避免 Myelin graph 被多个 context 重复加载；创建失败或运行时压力过大自动退回安全单路。GUI 将并行推理提升为补洞模式下的常用选项，提供“单路推理 / 两路推理 / 三路推理”，默认两路，并将“显示高级立体参数”置于其右侧。
- 修复 Flet GUI 多次最大化/最小化后黑屏：窗口尺寸变化使用可取消的延迟双重刷新，避免恢复阶段丢失重绘。

## 2026-08-05
- Added a minimal OpenXR screen-Quad eye diagnostic: one `array_size=2` swapchain supplies red-left layer 0 and green-right layer 1, each with its matching eye visibility, so VDXR per-eye array-layer support can be verified before changing the runtime DLL.
- Fixed native TensorRT Myelin `enqueueV3` failures in OpenXR screen-quad reprojection: native TensorRT now uses one execution context, so the runtime stays on the safe single-depth-worker path instead of concurrently loading the same Myelin graph.
- OpenXR 并行推理实验完成实机验证：native TensorRT 现在使用两个独立 execution context、CUDA stream、输入/输出缓冲和完成 event；pipeline 在 TensorRT 按首帧尺寸延迟完成 engine 加载后自动创建两个深度 worker，按 `frame_id` 有界重排并保持补洞、temporal、OpenXR/Vulkan 提交单线程。GUI“并行推理”移至高级立体参数之后，默认开启且可手动关闭。RTX 2060 动态内容实测开启后 SBS/处理帧率较单路提升约 6~10 FPS；日志可通过 `rt_parallel_workers=2`、`rt_pending_limit=2` 和交替的 `rt_depth_slot=0/2`、`1/2` 验证实际双路执行。
- Vulkan Projection Composer 完成纯 Vulkan 图形管线迁移：平面和曲面虚拟屏幕现在以世界空间三角带直接光栅化到每眼 Projection swapchain，复用现有 zero-copy 左右眼纹理与同步契约；移除 Homography 中间图、矩形复制和 Filament 回退，OPAQUE 运行时启用实验开关后也保持纯 Vulkan 路径。
- 修复头部移动或屏幕穿过视锥边界时出现的透明矩形、闪烁、残影、丢屏和绕头旋转：每个 XR tick 清理已获取的双眼目标，由 Vulkan 固定功能完成近平面与 FOV 裁剪，屏幕离开视野后不再复用错误矩形或旧帧。
- 新增普通显示和左右眼红绿诊断启动脚本；shader 构建、清单与合规工作流现同时编译和校验 compute、vertex、fragment SPIR-V，便于独立验证 array layer、左右眼顺序及实际 `graphics_triangle_strip` 路径。
- 修复 FPS 面板 XR 速率被视图循环计数覆盖的问题：优先使用成功 `xr.end_frame` 的真实提交时间戳；仅在尚未完成两次 XR 提交时使用启动阶段回退计数，避免面板错误显示约 36 FPS。

## 2026-08-04
- 建立可选 Vulkan Projection Composer 实验边界与固定的 Projection→Quad 提交契约；增加一键启动和左右眼红绿诊断，确认 `array_size=2` 的 layer 0/1、左右眼资源顺序及源图 GPU 同步可用。早期 blit/Homography 实验实现已由 2026-08-05 的直接 Vulkan 图形管线替代。
- 修复 FPS 面板将 XR 消费帧率误当作 SBS 生产帧率的问题：XR FPS 统计实际 OpenXR 提交，SBS FPS 读取运行时生产速率，两者不再显示相同的错误值。
- 将 patched Filament Vulkan backend 从 v1.74.0 升级到 v1.75.0：同步更新三平台 release 资产校验、源码构建 ref、本地 SDK 路径与 BlueVK 固定哈希；现有外部 `VkImage`、多 SwapChain 状态和 layered array image-view 补丁已通过 v1.75.0 源码契约验证。
- 修复低分辨率 EASU 路径首帧黑屏：MIP 目标创建完成后不再错误标记为已生成，首次采样会先执行实际源图重建；同时为 EASU 无有效权重的边界像素回退到中心源样本，避免异常输入产生黑色输出。
- 修复采样策略在源图导入后才切换到 EASU 时的目标尺寸失配：切换 `upscale_scale` 或 `filter_scale` 会使旧中间纹理失效，下一帧按新策略重新创建并生成内容，避免 1K 输入出现黑屏或使用错误尺寸。
- 将屏幕采样矩阵改为按输入/头显档位选择主路径：低分辨率输入使用独立 `upscale_scale` 生成 2 倍目标纹理，GPU 执行 EASU → RCAS → MIP；同档位使用源图 → MIP，高分辨率输入继续使用 Lanczos2 → RCAS → MIP。新增 `filament_bridge_set_screen_upscale` ABI，旧 Bridge 缺少该符号时保留原兼容路径。
- 新增可选 Filament multiview 双眼诊断：设置 `D2S_FILAMENT_EYE_DIAGNOSTIC=1` 后，左眼输出红色、右眼输出绿色；默认显示路径不变。
- OpenXR multiview 视觉回归现可从同一个 `array_size=2` Projection SwapChain 分别导出 layer 0/1，稳定生成左右眼 Projection 截图与运行清单；实测定位并修复手柄和指南独立 View 丢失双眼视差的问题，multiview 现在通过同一个 foreground View 按既有渲染优先级一次输出屏幕、辉光、手柄和指南，旧双 SwapChain 路径保持原有分层渲染。

## 2026-08-03
- 修复 OpenXR Filament multiview 将虚拟屏幕、房间和手柄渲染成单眼 2D 画面的问题：相机现在按
  Filament 契约使用中心头部绝对姿态和左右眼相对姿态，保留真实 IPD 视差；Bridge ABI 升级后，
  旧二进制会自动回退逐眼渲染。同时限制 multiview 原生帧诊断仅输出前 8 帧，不再持续刷屏。
- OpenXR Projection 新增稳定的双眼一次提交路径：优先创建一个 `array_size=2` SwapChain，
  由单个 Filament multiview frame 同时写入左右 array layer，并在一次完成信号消费后提交
  两个 Projection View。仅在 Bridge ABI、GPU multiview 能力和双眼尺寸均满足时启用；
  layered 目标创建失败会只销毁临时资源并保留原双 SwapChain 逐眼路径，不再调用已证实
  不稳定的 `end_frame_deferred()` / `finish_frame_batch()`。远程 Filament SDK 构建现强制开启
  `FILAMENT_ENABLE_MULTIVIEW`，并随工作流配置变更失效旧缓存，避免 Engine 启动时解析缺失的
  内置 multiview 材质而异常退出。
- 禁用 OpenXR 双 SwapChain 的 Filament deferred batch 默认路径：实机确认该路径无论使用
  双 Renderer 还是共享 Renderer，都会在固定运行时间后于 `finish_frame_batch()` 内触发
  `VK_ERROR_DEVICE_LOST` 或原生 access violation。Projection 恢复逐眼完成后再切换
  SwapChain，并继续消费每眼 render-finished semaphore；真正的双眼一次提交改由后续
  array swapchain + multiview 实现。设备丢失后不再调用 `xrEndFrame` 覆盖首个 Vulkan 异常。
- 修复双眼 deferred batch 中仍存在的共享场景写入：右眼设置独立 Camera 时不再再次移动
  控制器共享灯光，避免 eye0 已入队后改写 Filament 场景状态并触发 Vulkan descriptor
  validation，最终随机演变为 `VK_ERROR_DEVICE_LOST`；共享灯光改为每帧仅随 eye0 更新一次，
  当前逐眼完成路径与 CUDA/Vulkan external semaphore 路径均保留该隔离修复。
- 修复 Filament 屏幕资源在双眼 deferred batch 中仍复用 Renderable/View，导致第二眼在
  第一眼 GPU 工作未完成时改写绑定并最终触发 `VK_ERROR_DEVICE_LOST`：最终屏幕与 MIP
  copy 的 material instance、Renderable entity 和 View 现全部按眼隔离，并由 layer mask
  固定到对应眼；该资源隔离继续用于当前逐眼安全路径和后续 multiview 实现。
- 修复 CUDA/Vulkan binary external semaphore 在环形输出 slot 多代复用时仍会触发
  `VK_ERROR_DEVICE_LOST`：跨 API ready/release 改为每 slot/eye 独立的 exportable
  timeline semaphore 和单调递增 generation；Filament visible 信号继续使用 Vulkan-only
  binary semaphore。CUDA wait、图像拷贝与下一次 ready signal 保持同 stream 异步顺序，
  不再用 `cudaStreamSynchronize()` 阻塞 Presenter；真实 RTX CUDA/Vulkan 300 帧循环通过。
- 修复 Output Worker 的 Glow 直接提交与 Presenter/Filament batch 并发访问同一 graphics
  `VkQueue` 导致的 device-lost；两条路径现在复用 VulkanContext 设备锁完成主机侧外部同步。
- 修复 `finish_frame_batch()` 永久卡死：复用源帧时不再重新 signal binary visible
  semaphore，Filament 直接采样已经就绪的外部图像，避免等待一个排在卡住队列后的
  signal 提交而触发 GPU/设备级死锁。
- 修复 Vulkan 优化后 XR/SBS 被拖到约 16 FPS 的屏幕 MIP 回归：虚拟屏幕源帧不变时
  Bridge 不再每帧重复执行 4K Lanczos/RCAS/MIP 生成，只有 `frame_id` 变化时才重建，
  静态画面保持原有清晰度，动态画面继续实时更新。
- 修复虚拟屏幕深度遮挡手柄：屏幕材质仍保持 Opaque 合成，但不再写入深度；
  控制器/激光 View 不受屏幕深度裁剪，保持手柄在最前，且不改变 View 架构。
- 修复 CUDA/Vulkan external semaphore 在静态源帧复用时的 binary semaphore 重复
  signal/wait：同一 frame/eye 复用只等待一次 producer-ready semaphore，但每个 XR tick
  重新 signal Filament 消费的 visible semaphore；同时恢复
  `D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE` 默认开启。
- 修复 OpenXR 未消费 Filament render-finished binary semaphore 导致的
  `VK_ERROR_DEVICE_LOST`：逐眼完成后读取各眼 SwapChain 的完成信号，使用一次 completion
  drain 消费并转换为 Vulkan timeline，再释放 zero-copy 源图。
- 修复 Filament Vulkan backend 对 OpenXR 左右眼两个 SwapChain 的单例默认 RenderTarget
  污染：`VulkanDriver::acquireNextSwapchainImage()` 现在记录当前绑定的 SwapChain/image，
  切换眼时先释放旧绑定再绑定当前眼，避免第二只眼误渲染到第一只眼的图像并触发
  `VK_ERROR_DEVICE_LOST`。
- 完成立体流水并行与同步收敛：每个 XR tick 产生的左右眼 Filament render-finished 二进制信号都会在同一次 Vulkan graphics 提交中各消费一次，再转换为 timeline 供 zero-copy 源图安全释放；静态复用帧和异常路径同样回收完成信号。满足双 TensorRT 隔离槽、Triton、无 Temporal 的路径允许两帧在途，使 CPU 入队与上一帧 GPU 执行重叠。最高质量补洞不再为双眼复制 4K mask、depth 和 shift，并跳过无破洞像素的邻域采样；Vulkan 设备丢失后只清理 CPU 租约，不再向失效设备重复提交。性能诊断每 15 秒输出一次、累计 5 次，并新增 Filament 完成信号回收耗时；FPS 面板的 SBS 数值只统计唯一生产帧。
- OpenXR Vulkan 完成双眼提交和 Compute 流水并行化优化：Projection 路径先获取完整左右眼 swapchain 图像，再分别等待两眼就绪，避免右眼 acquire 被左眼 wait 串行阻塞；控制器场景状态和 GLB 动画改为每个 XR 帧只更新一次。Filament Bridge 新增兼容的 frame-wide 双眼提交 ABI，两眼仍在 Presenter 所在线程顺序生成渲染命令，但每眼只做非阻塞提交，待两眼全部入队后统一执行一次完成等待；旧 Bridge 二进制自动保留逐眼安全等待路径。Vulkan Stereo Compute 输入改为与 FrameContext 对齐的环形槽，CUDA 与 Vulkan 通过每槽外部信号量完成“可覆盖/输入就绪”双向 GPU 同步，不再逐帧主机等待；zero-copy 输出槽复用也只保留 GPU timeline 依赖。CUDA 单帧在途期间不再主动清空最新捕捉队列，生产者仍可覆盖旧帧，GPU 完成后可立即消费当时最新画面；OpenXR CUDA Triton 路径进一步启用双 TensorRT execution context、独立输入/输出缓冲和逐槽 CUDA Graph，在同一 CUDA stream 上允许两帧命令排队，使 CPU 准备与上一帧 GPU 执行重叠，同时避免双流争抢 TensorRT、Triton 和 Filament 的 GPU 资源。Vulkan deferred、Temporal、动态 convergence 或不具备隔离槽的 provider 自动保持单帧在途。FPSBreakdown 新增双眼 acquire/wait、Filament 每眼入队/延迟提交、整对完成等待、双眼 release、Vulkan 输入等待/上传和 TensorRT 执行槽统计。

## 2026-08-02
- 手柄新增动态屏幕反射光：复用现有 Vulkan Glow Compute 输入，在同一 GPU dispatch 中将屏幕 sRGB 内容归约为线性平均光色，仅异步读取 3 个浮点数而不回读画面；手柄 PBR 光照按屏幕光 80%、基础头灯/顶部灯/间接光 20% 分配，屏幕聚光灯直接使用采样色，不再额外混入固定白色。
- 新增 Vulkan `surround` 环绕辉光：以四组互不共享极点的“屏幕边缘→半球外圈”放射条带替代经纬半球网格，每个屏幕边缘采样点沿独立球面路径均匀向四周扩散，不再向半球上下左右中心点会聚；放射条带第一圈直接锚定虚拟屏幕四边的真实世界坐标，后续顶点才从屏幕深度逐步过渡到头部射线与远端椭球的交点，使双眼视差和头部平移时发射边界仍固定贴合屏幕，不再露出反向漂移的黑色矩形。GPU 沿屏幕四边划分 8×6 分段，并按输入分辨率从最外圈向内自适应采样约 4/8/16 像素宽的窄带，避免整块区域平均造成辉光与屏幕边缘颜色不一致；相邻分段平滑插值，屏幕原图不参与模糊或改色。Surround 使用加法发光合成，不产生遮挡方块或实体球体背景；取样颜色在 sRGB 感知域计算区域均值后仅解码一次进入 Filament 线性工作流，效果仍仅在 Default 环境显示。

## 2026-08-01
- 将补洞模式收敛为三个正式选项：`none`（关闭 / 不补洞）、`balanced`（均衡 / 标准）和 `quality`（增强 / 高质量）；删除没有独立 kernel 价值的 `soft_low_ghost` 与 `sharp_test` 选项、标签和适配映射，不保留旧模式专用兼容分支。
- 更新补洞 GUI 中文文案：原“内容感知 / 最高质量”改为“增强 / 高质量”；Tooltip 仅说明三档有效行为，并明确立体模式默认映射为电影→均衡、游戏→关闭、图片→增强/高质量。
- 同步 GUI 与运行时预设：电影使用 `balanced, radius=1, strength=0.6`；游戏使用 `none, radius=0, strength=0.0`；图片使用 `quality, radius=3, strength=1.0`。
- 为 Vulkan Compute 建立统一补洞三态 ABI：`BALANCED=0`、`QUALITY=1`、`NONE=2`。`d2s_stereo_fused`、`d2s_stereo_layered`、`d2s_stereo_layered_tiled` 和 OpenXR zero-copy 的 `d2s_stereo_layered_output` 均执行同一模式契约。
- Vulkan 最高质量补洞迁移完整 radius-3 方向内容感知公式：深度/位移方向可靠性判断、三点方向平均、方向候选与 box average 的 0.75/0.25 混合、UI 亮度边缘保护和深度边缘保护；关闭模式在 shader 主路径直接跳过遮挡、羽化和补洞邻域计算，并输出零 mask。
- 更新 Vulkan 运行时 debug 字段，统一报告 `vulkan_hole_fill_mode` 与实际 `hole_fill_backend`，避免配置显示为最高质量但执行均衡公式。
- 使用 Vulkan SDK 1.4.350.0 重新编译四个 SPIR-V；RTX 3090 实际 Vulkan 调度验证三态均可执行，关闭模式 `mask_max=0`，最高质量与均衡输出存在有效差异。
- 回归验证：专项测试 121 项通过，全量测试 `713 passed`；`git diff --check` 通过。

## 2026-07-31
- 修复静止源帧复用后 FPS 面板 SBS 计数归零的问题：SBS FPS 现在按实际 XR 显示 tick 统计，而不是按生产者 `frame_id` 去重；同时恢复前景 View 不执行二次后处理，避免前景合成阶段覆盖房间场景。
- 修复场景曝光更新和 Bridge 销毁时前景 View 残留旧 ColorGrading 句柄的问题，避免 Filament 访问已释放句柄导致原生崩溃。
- 修复前景 View 中手柄材质发白、反光碎片化并被误认为透明的问题：手柄、屏幕和激光所在的前景 View 现在绑定与房间 View 相同的 ColorGrading，并独立执行一次输出变换；两个 View 先后写入同一交换链并不构成同一像素的二次编码。
- 修复 `preview_room_layout.py` 中 `3d_bedroom` 材质大面积发黑的问题：桌面预览现在读取环境 profile 的 `env_ambient_color` 创建独立 Filament 间接光，并在未配置 `preview_exposure` 时回退使用 `env_exposure`，保留方向补光，避免无环境光导致背光材质全黑或明暗对比异常。
- 将 Filament 光照拆分为房间 Scene 与前景 Scene：房间 GLB 使用全局间接光，手柄、屏幕、激光和 UI 使用独立前景 View；`ambient_light_multiplier` 不再放大房间全局光，`controller_hdr_lighting` 现在实际控制前景 controller 间接光开关。前景 HDR IBL 资源仍保留 `hdr_ibl_pending_profile_fallback` 约束，待三平台 KTX IBL 接入后使用真实 HDR 预过滤环境。
- 前景 View 关闭后处理，仅在房间主 View 执行一次最终颜色变换，避免双 View 合成时房间颜色被二次 tone-map 或编码。
- 将 Filament 默认场景曝光和天空盒亮度从 `settings.yaml` 迁移到 `xr_viewer/environments/common.json`；环境 `profile.json` 仍可按环境覆盖，旧 YAML 字段仅保留兼容回退。
- 明确区分电影立体合成质量与补洞质量：`quality_4k` 继续表示 Cinema 的立体合成后端，选择“最高质量”补洞后统一记录为 `hole_fill_mode=quality`；启动/热切换日志和 15 秒 FPSBreakdown 现在同时输出补洞模式、半径和强度。
- 修复 Vulkan Filament 虚拟屏幕 MIP 采样缺少旧工程 `LOD_BIAS=-0.35` 的问题；最终屏幕采样现在使用与旧 OpenGL runtime eye 纹理一致的负 LOD 偏移，避免在相同屏幕 footprint 下过早选择较软的 MIP 级别。
- 保持 `filter_scale=1` 路径不执行 Lanczos2 和 RCAS，仅进行原图 LOD0 拷贝与动态 MIP 链生成；不修改颜色空间、输入分辨率或显示几何。
- 实机验证：MIP 路径文字边缘清晰度已接近旧工程 legacy 路径。
- 本地回归测试：`38 passed`；Filament Bridge Windows/Linux/macOS 三平台 GitHub Actions 构建成功，二进制已同步。

## 2026-07-30
- 修复 Lanczos2/RCAS 屏幕材质使用旧式 `materialParams_<name>` 访问导致 Filament shader 编译失败的问题，统一改为当前 Filament MaterialBuilder 要求的 `materialParams.<name>` 结构体访问。
- 删除正常运行路径中的 `07_filament_screen_*.png` 固定相机 readback/PNG 导出及对应 C ABI；屏幕显示只保留 GPU 采样、MIP 计数和同步路径，历史 artifact 仍可由离线脚本比较。
- 完整迁移 legacy 屏幕清晰化两级 GPU pass：第一 pass 为 4x4 Lanczos2 重建，第二 pass 为完整 FSR RCAS（luma 自适应、RGB limiter 和有界 sharpness），RCAS 输出再生成动态 MIP 链；不引入 CPU 像素往返或跨帧混合。
- 动态屏幕 MIP 优化与两级质量 pass 在每只眼的 Filament `begin_frame` 内执行，使用线性空间 sRGB 下采样、三线性过滤和 16x 各向异性过滤；规格书和需求矩阵同步更新。
- AMD FidelityFX SPD/Vulkan compute downsampler 暂不直接替换稳定路径，后续必须以相同 `07_filament_screen` 源一致性、计数和 heatmap 指标做 A/B 验证后再决定。
- 补齐虚拟屏幕动态 MIP 采样的可验证闭环：native Bridge 记录每眼外部源图像绑定次数和 MIP 生成次数，并通过 `filament_bridge_get_screen_sampling_stats` C ABI 暴露给 Python。
- `07_filament_screen_*.png` 捕获 manifest 现在写入 `screen_sampling_update=dynamic_per_frame_mip`、MIP 动态更新标记和每眼采样统计，便于确认每帧 `generateMipmaps()` 是否实际执行。
- 修复屏幕采样视觉回归脚本的 manifest 识别：优先读取 `screen_sampling_runtime_manifest.json`，保留旧 `visual_regression_runtime_manifest.json` 回退，legacy/mip 对比不再误判 07 捕获缺少配置。
- 已新增对应单元测试和 native 静态 ABI 断言；原生 Bridge 计数 ABI 需要 GitHub Actions 三平台远程构建后才能在实机日志/manifest 中出现真实数值。

## 2026-07-29
- 新增 GUI 头显型号驱动的 2K/4K/8K 屏幕采样档位，并按实际输入屏幕 `capture_size` 将 1K/2K/4K 输入映射到 2K/4K/8K 推荐头显。
- 非 16:9 输入仅按最长边近似归档，不裁剪、不拉伸实际源图像；匹配档位保持原始纹素 footprint，低档头显接收高档输入时才启用有界面积预滤。
- OpenXR runtime recommended extent 仅作为交换链上限，不再覆盖 GUI 头显选择；新增 Filament Bridge `set_screen_sampling` ABI。
- 验证：屏幕采样策略、头显预设、runtime 配置、OpenXR Vulkan 和 Filament Bridge 测试共 166 项通过；原生 Bridge 仍需 GitHub Actions 三平台远程构建。
- 已将上述输入/头显矩阵、非 16:9 归档、尺寸职责边界、预滤公式和 Bridge ABI 验收条件补入两份正式规格书及需求矩阵。
- 全量 Python 回归测试通过：680 passed；提交内容包含本轮规格书、采样策略、Bridge ABI、MSDF Quad 和既有未提交改动。
- 修复 `3d_bedroom/environment.glb` 的 glTF scene root：移除重复挂到 scene root 的非根节点，避免 Filament desktop preview 报 `Unable to parse glTF file`。
- 为旧房间 profile 增加 `view_pose_space=scene` 兼容：`3d_bedroom` 的座位坐标按 GLB 原始场景坐标解释，不再被 `model_position` 逆变换推到房间外；预览和 OpenXR profile 加载路径保持一致。
- 预览工具保存座位时同步遵守 `view_pose_space`，避免移动座位后把旧房间 profile 再写回成错误坐标。
- `npx --yes @gltf-transform/cli validate src/xr_viewer/environments/3d_bedroom/environment.glb`: `No errors found`。
- `src/python3/python.exe -m py_compile src/xr_viewer/preview_room_layout.py src/xr_viewer/core_openxr_vulkan.py` 通过。
- `3d_bedroom` 预览相机计算结果保持为 GLB scene 坐标 `[0.0018, 0.7381, 0.0202]`，不再变成 `[0.0018, 1.7381, 3.0202]`。

## 2026-07-27
- 修复 Requirements Compliance 在 Linux runner 上的 Windows 键盘状态机测试：
  `ctypes.windll` 仅在 Windows 存在，测试现在显式允许注入 FakeUser32，仍完整验证
  修饰键、普通按键和释放状态，不改变生产代码。
- The screen size/distance and preset OSD now decode the bundled MSDF atlas
  in a Vulkan compute shader, writing the glyph coverage and background into a
  Quad-sized storage image before copying it into one OpenXR Quad Layer. This
  removes the unreadable Projection/Quad split while keeping the OSD text on
  the real GPU MSDF path.
- Added a local MSDF JSON-coordinate preview tool so OSD layout can be checked
  before OpenXR hardware testing.
- Corrected MSDF V-coordinate adaptation for Filament's bottom-left texture
  convention and updated coverage sampling for the 2048x2048 atlas.
- 修复 MSDF Filament 材质编译失败：`sample` 在目标 GLSL 兼容编译器中是保留字，已改为 `msdf_sample`；三平台 Bridge 远程构建和二进制回写成功。
- 修复 MSDF 空页提交：零长度 NumPy 缓冲改用 `tobytes(order="C")`，并在首次提交失败后关闭 MSDF 路径，避免每帧重复报错刷屏并恢复旧 Quad Layer。
- Started the GPU text migration contract: imported the requested `3500.txt` UI charset (3,958 unique characters), defined paged MSDF atlas generation and shared linear atlas sampling requirements, and kept an explicit legacy Quad Layer fallback while the native Bridge ABI is being added.
- Generated the first three-page MSDF atlas from the complete UI charset; verified 3,959 glyph records and 2042x2032 atlas pages.
- Moved the MSDF runtime assets into `src/xr_viewer/fonts/` so packaged OpenXR runtime resources do not depend on repository-root asset paths.
- Added the first native MSDF text-overlay ABI: atlas pages are uploaded as
  linear GPU textures and packed glyph geometry is updated on the Presenter
  thread. Existing Quad Layer bitmap rendering remains the compatibility
  fallback until rebuilt Bridge artifacts are deployed.
- The Presenter uploads the shared MSDF atlas once to a resident Vulkan
  storage image and submits only changed glyph metrics to the compute pass.
  The native Filament MSDF Projection ABI is not used for this Quad-only text
  path; keyboard and laser cursor textures remain on their existing Quad path.
- OSD Quad canvases now size themselves from MSDF text advance and atlas line
  height, then preserve that aspect ratio while scaling with the virtual screen.
- FPS and both screen/controller operation-guide panels now submit their text
  as GPU MSDF runs into the same Quad-layer intermediate; keyboard and laser
  cursor textures remain unchanged.
- Menu and B overlay state machines are now mutually exclusive. The screen-side
  guide keeps the full current screen height, and its MSDF text uses the same
  canvas proportion so the guide content fills the background instead of
  becoming a tiny centered block.
- Restored live FPS-panel resolution values: XR now uses the first eye
  swapchain size and Screen uses the current per-eye output render size instead
  of the temporary `0x0` placeholders. Resolution changes invalidate the
  cached MSDF panel text.
- Restored the complete vertical Menu operation guide: the one-column MSDF
  layout now renders every legacy guide row instead of applying the controller
  panel's two-column split and dropping the second half.
- FPS now reads the runtime `depth_strength` metadata instead of displaying a
  hard-coded `0.00`. Depth adjustment, reset, and 2D/3D controller shortcuts
  also trigger a legacy-style Quad OSD for 2.5 seconds; the stereo toggle uses
  the legacy `3D mode on/off` message while depth changes show `Depth Strength`.
  Controller changes are reflected immediately from the accepted runtime target;
  older in-flight output frames cannot overwrite the new OSD value.
- Right-grip/right-stick screen distance and size controls now use only the
  hold-time acceleration curve; the generic guide dispatcher no longer adds a
  second fixed-speed delta in the same XR frame.
- Right-grip screen distance and size controls start at 0.10 m/s, accelerate
  linearly with hold time, and reach 10.0 m/s after five seconds. Releasing or
  reversing the stick resets the ramp for precise adjustment.

### 已实现

- 按旧工程恢复屏幕上方尺寸/距离 OSD：使用 512x78 深灰圆角面板、灰色标签、青色数值和 24px 字体，整组文字居中，并沿用屏幕宽度比例与顶部间距。
- 修正右 Grip + 右摇杆屏幕调节：恢复旧工程的 X 轴指数加速缩放，移除 22m 硬上限；Y 轴距离调节统一使用 0.35-3.0 m/s 的指数速度曲线。

- 修复 OpenXR 运动门控导致 SBS 更新过慢：对可复用的捕获/GPU 输入缓冲区建立独立运动采样快照，且恢复旧工程默认关闭 motion gate；输入未变化时复用上一帧仍可通过 `D2S_RUNTIME_MOTION_GATE=1` 显式启用。
- 将 OpenXR 未获得焦点时的手柄输入日志改为 `Controller input deferred`，避免把正常的 `SessionNotFocused` 状态误标为失败。
- 清理 CUDA external semaphore 正常状态日志：无实际错误时不再输出 `error=none`，仅在同步初始化确实失败时追加错误信息。
- 将 `FPSBreakdown` 从每秒输出改为稳定运行 15 秒后输出一次，避免持续刷屏且不影响内部统计。
- 将 Vulkan validation layer 重复的 descriptor binding 明细合并为一条摘要，避免启动日志输出几十行相同类型诊断。
- 将 FilamentBridge 每帧左右眼 `acquired/begin/end` 正常明细合并为一条摘要，避免持续输出渲染循环日志。
- 修复 OpenXR 屏幕分辨率诊断刷屏：只在源图像分辨率、交换链目标尺寸或 render size 发生变化时输出，头显位姿导致的投影 footprint 变化不再触发日志。
- 修正 OpenXR projection 输出的左右眼资源顺序；普通 SBS Vulkan 路径保持原有合成顺序，不复用 OpenXR 专用交换逻辑。
- 新增 `d2s_stereo_layered_tiled.comp/.spv` 作为 `quality_4k` 的并行 tiled reference shader：保持原有五个 storage buffer、76 字节 push constant 和立体合成公式，仅将深度邻域缓存到 workgroup shared memory，供 warp、遮挡、羽化和补洞采样复用。
- `VulkanStereoComputeBackend` 新增可选 `layered_shader_path`，默认仍使用 `d2s_stereo_layered.spv`；tiled shader 当前只作为对照和性能基准路径，不改变生产默认选择。
- 将 tiled reference shader 登记到 `shaders/manifest.json`，并通过 `glslc` 编译和 `spirv-val` 校验。

### 验证结果

- `tests/test_pipeline.py`: `8 passed`；Vulkan/CUDA 相关回归测试：`32 passed, 2 warnings`；`py_compile` 和 `git diff --check` 通过。
- 3840×2160 同参数对照：原 layered shader 与 tiled reference 的左右眼及遮挡 mask 最大绝对差异均为 `0`。
- Vulkan 相关回归测试：`21 passed`；此前扩展的 OpenXR、runtime、synthesis 和 Vulkan 集成测试：`109 passed`。

### 未决事项

- tiled reference 尚未替换生产 shader；下一步使用稀疏破洞、密集边缘和真实深度帧分别比较 GPU dispatch 时间、画质和端到端 FPS，再决定是否进入生产路径。

## 2026-07-26

### 已实现

- 按旧工程恢复手柄屏幕操作：右 Grip+右摇杆 Y 沿头部到屏幕的径向距离移动，并使用旧工程指数加速曲线（0.35–3.0 m/s、死区 0.15）；左 Grip 按下时记录手柄旋转锚点，腕部旋转超过 45° 后将屏幕 Roll 吸附旋转 90°。
- 修正 CUDA 外部 semaphore 输出的 Filament 诊断元数据：明确报告 `vulkan_readback=none` 和 `vulkan_output_path=presenter_owned_storage_image`，避免直连路径已激活却显示为 `missing`。
- 修正右 Grip+右摇杆前后方向：按当前 Vulkan 输入层的 Y 轴约定恢复旧工程符号，摇杆向前时屏幕远离头部，向后时靠近头部。
- 修正 Vulkan Compute 零拷贝未生成 request 的条件：移除已完成分层视差迁移后仍残留的 `layered_parallax_not_supported` 守门条件；即使普通 Vulkan backend 已经初始化，OpenXR 也优先生成 Presenter-owned `vulkan_compute_request`。
- 修复 Vulkan host fallback 对 OpenXR `HxWx4` 眼图的尺寸识别，将 `(2160, 3840, 4)` 正确解析为 `3840x2160`，避免输出转换异常退出。
- 修复 OpenXR cinema 全合成路径丢失 `vulkan_compute_request`：运行时不再先走 `process_rgb_frame` 再包装 OpenXR 结果，统一通过 `process_openxr_frame` 保留 Presenter-owned Vulkan zero-copy request，避免每帧约 1 秒的 host-visible 回读。
- 为 zero-copy 实机验证固定运行条件：`src/settings.yaml` 使用 `Stereo Compute Backend: vulkan`、关闭 `Temporal`/`Auto Scene Reset`，OpenXR 预变形默认开启；Presenter 状态日志现在直接打印 `vulkan_readback`、`vulkan_output_path`、`vulkan_output_sync` 和 `active`，便于确认是否真正进入 Presenter-owned Vulkan image 路径。
- 收紧 Filament 外部屏幕图像路径的运行时诊断：CUDA producer 现在发布外部 semaphore 请求是否被环境变量或 Bridge ABI 阻止、初始化异常及最终 active 状态；Presenter 记录 direct screen path 的具体回退原因，不再把 zero-copy 未启动静默表现成普通 GPU copy。
- 修复 Filament 直接采样源图像的租约生命周期：显示中的 Vulkan ring slot 会保持到新帧替换或关闭，环形缓冲即将复用该 slot 时由 Presenter 先完成 finished semaphore 后再释放，避免 Filament 仍在采样时被 CUDA 覆盖而产生模糊。
- 修正 Vulkan zero-copy 投影屏幕发白：Compute 输出是 `VK_FORMAT_R8G8B8A8_UNORM` storage image、Filament 外部纹理是线性 `RGBA8`，shader 现在先将显示参考 sRGB 输入解码到线性光再执行 warp/补洞和写入；OpenXR sRGB 目标仍只做最终一次编码。
- 优化 4K layered shader 的安全路径：移除未使用的重复 `occlusion_at` 计算，遮挡搜索命中 edge 后提前结束，并在 `mask==0` 时跳过 `box_average`/方向补洞；有洞像素的 warp、补洞窗口和 blend 公式不变。当前中心+四方向羽化采样属于有边缘画质风险的近似，仍需实机重点检查斜向破洞。
- 实测 RTX 4K 后撤回 Triton 的简单 mask-predicated 补洞改法：无洞、10% 破洞、全破洞分别约为原 kernel 的 `0.92x`、`0.38x`、`0.83x`，额外分支控制抵消了屏蔽加载收益；Triton 保持原 kernel，后续若优化需采用稀疏索引/专用 active-pixel 两阶段方案并重新基准。
- 恢复旧工程的头显推荐屏幕几何：OpenXR Link 读取 GUI 的 `XR Headset Model`，按对应最佳观看距离和 60° 水平视场自动计算 16:9 屏幕宽高；Pico 4 / 4 Ultra 为 `23.09m × 12.99m @ 20m`，不再使用固定的 `16m × 9m @ 16m`。
- 对齐旧工程 OpenXR 工具交互：菜单键循环为“FPS → FPS+屏幕左侧竖向操作指南 → 全隐藏”，屏幕指南改用 `build_team_help_rgba`，不再误用手柄双列指南。
- 恢复旧工程键盘头向和激光命中点拖屏逻辑：键盘每帧以头部为目标重新朝向，单手 Grip 保持激光命中的屏幕局部锚点，手柄平移或旋转都能拖动屏幕；补回激光终点光圈显示。
- 输入路径异常现在只做一次可见日志，不再静默吞掉键盘、拖屏和快捷键更新失败。
- 修正右手 Grip 旋转屏幕的回归：旧工程 `openxr_right_grip_screen_rotation` 默认关闭，Vulkan 端现在明确禁止右手腕部旋转屏幕，仅保留旧工程允许的左 Grip 旋转和左 Grip+右摇杆旋转。
- 恢复旧工程右 Grip 单手拖屏的球面轨道：屏幕围绕头部保持固定距离移动，并在轨道移动后重新朝向头部。
- 补齐 B 长按三态循环：隐藏 → 手柄 FPS → 手柄 FPS+手柄操作指南 → 隐藏；不再把 B 长按错误实现为二态切换。
- 建立 `docs/05-openxr-behavior-migration-matrix.md`，按旧工程函数、Vulkan 函数、渲染层和验证方式登记 OpenXR 行为迁移状态；后续迁移项必须同时补自动对照测试和头显验证项。
- 按旧工程恢复 reference-space change pending 处理：运行时重定位后重建共享基础空间，重新应用 profile pose，并清空旧头部缓存，避免屏幕、手柄和投影视图使用不同坐标系。
- 按旧工程补齐曲面屏圆柱射线求交和 UV 到曲面世界坐标转换；左 Grip 同时支持旧工程的左右摇杆旋转，键盘轨道不再错误要求 stick click。
- 修正平面屏激光命中的旧工程 UV 方向：下边缘为 `v=0`、上边缘为 `v=1`，命中点拖动和 Quad 光圈与实际屏幕纹理保持同一上下方向。
- 恢复旧工程激光屏幕边缘吸附：平滑射线越过有限屏幕、原始姿态射线也未命中但仍处于边缘 6° 释放锥内时，使用无限屏幕平面 UV 夹到最近边缘，并将激光和交互命中保持在该边缘。
- 新增第一版 Vulkan Compute 立体合成融合 pass：单次 dispatch 完成视差计算、左右眼水平 warp、遮挡边缘膨胀、方向感知补洞和边缘保护；深度模型推理路径保持由各厂商后端负责。
- 新增 `vulkan_stereo_benchmark.py` 和 Vulkan smoke 校验；Windows RTX 3090 的 3840×2160 初测为约 `31.34 ms / 31.9 FPS`，同机现有 CUDA/Triton `fast_plus` 端到端初测约 `26.29 ms / 38.0 FPS`。该结果是首版融合计算对比，尚不代表最终零拷贝端到端性能。
- 明确立体合成后端选择：`auto/vendor` 先做真实 Triton kernel 探测；NVIDIA 走 CUDA Triton，AMD 走 ROCm/Windows Triton，只有探测失败或厂商不是 NVIDIA/AMD 时才走 Vulkan Compute。显式 `vulkan` 可在 NVIDIA、AMD、Intel 全部使用 Vulkan Compute，且不改变深度模型推理后端。
- 统一 Triton 运行时门控：视差、warp、遮挡、补洞、时域和输出阶段不再直接用 `is_cuda` 判断厂商，统一读取 GPU vendor；NVIDIA 与 AMD 使用同一套 Triton kernel 源码，Intel 等其它厂商不会误进入 Triton。
- CUDA 12.8 profile 改为稳定配套：PyTorch `2.11.0`、torchvision `0.26.0`、Linux Triton 3.6 系列和 Windows `triton-windows==3.6.0.post26`；AMD ROCm7 profile 保留独立 nightly 配套版本。Torch/Triton 升级需要重新验证 TensorRT、深度推理和 Triton kernel。
- 实际升级嵌入式 Python 到 `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128`、`triton-windows==3.6.0.post26`；RTX 3090 / CUDA 12.8 / TensorRT 10.14.1 导入和 Triton kernel 探测均通过。依赖与 Vulkan layered 接入完成后的全量测试为 `610 passed, 6 warnings`。
- 将 Vulkan fused stereo pass 接入 `StereoRuntime` 的 `fast_plus` 生产路径：`Stereo Compute Backend=auto` 在真实 Triton 探测失败或厂商不支持时选择 Vulkan，显式 `vulkan` 可在 NVIDIA、AMD、Intel 上运行同一套 Vulkan 视差/warp/遮挡/补洞 shader；NVIDIA/AMD 仍保持 Triton 优先。
- 新增 `VulkanHostOutputAdapter`，Vulkan fallback 的左右眼结果可通过 Presenter 自有 Vulkan host image 输出，不再因没有 CUDA/HIP interop 而静默丢帧；该兼容路径使用同步 GPU copy/Quad Layer，不伪装成 Filament zero-copy semaphore 路径。
- 新增独立 `d2s_stereo_layered` Vulkan Compute pass，并接入 `quality_4k/hq_4k`；按现有分层合成逻辑执行深度分层权重、逐层水平 warp、遮挡膨胀、屏幕边缘抑制和 balanced/directional 补洞，未将高质量模式错误降级为 `fast_plus`。
- 为 Vulkan Compute 和 host fallback 增加分段耗时诊断：分别记录 host upload、Vulkan submit/wait、host readback、输出 wait 和输出 upload；当前路径仍明确标记为 host-visible fallback，未伪装成 zero-copy。
- 完成 Vulkan Compute 的真正输出零拷贝链路：新增 `d2s_stereo_layered_output.comp/.spv` 和 `VulkanStereoImagePass`，在 Presenter-owned Vulkan context 中直接把左右眼写入持久化外部 `VkImage`，不再读取 Vulkan 输出 buffer 到 CPU，也不再由 CPU 重新上传左右眼像素。
- 新增 `VulkanZeroCopyOutputAdapter` 和 `VulkanComputeRequest`：推理线程只发布 RGB/Depth/参数请求，Presenter 线程完成 Compute submit、`GENERAL -> SHADER_READ_ONLY_OPTIMAL` barrier、per-eye visible semaphore 以及 Filament finished/release 状态机；不支持直接外部采样时仍显式走 GPU copy 回退。
- 完成 NVIDIA CUDA 到 Vulkan Compute 输入的 GPU 互操作第一阶段：RGB/Depth 使用可导出的 Vulkan storage buffer、CUDA external memory mapped buffer 和 external ready semaphore，Presenter 不再执行每帧 Tensor `.cpu()` 与 host-visible storage buffer 上传；不支持 CUDA external buffer 时明确回退并在日志中报告 `vulkan_input_path=host_visible_buffer`。
- Filament 屏幕状态日志新增 `vulkan_input_path` 与 `vulkan_input_upload_ms`，用于区分真正的 CUDA external buffer 输入和兼容性 host 上传路径。
- 优化 `d2s_stereo_layered_output` 的 4K 热点：无空洞像素不再执行补洞采样，遮挡羽化避免嵌套 7×7 重复膨胀；RTX 3090 实测 CUDA external input 约 `0.1–0.2ms`，均匀/单边缘深度 steady-state 约 `10ms`，随机高边缘压力场景约 `61ms`，后续仍需用实机深度帧验证画质与稳定 FPS。
- 增加 Vulkan timeline 主机等待接口，保护输入 buffer 重用和 fallback source image release，不使用逐眼 `vkDeviceWaitIdle` 作为正常同步；同步契约和 CUDA external buffer 输入、host-visible 显式回退的范围已写入三份规范与需求矩阵。

### 验证结果

- `src/python3/python.exe -m pytest -q tests/test_openxr_vulkan.py tests/test_cuda_vulkan_interop.py tests/test_runtime_output.py`：105 passed，2 warnings。
- `src/python3/python.exe -m pytest -q tests/test_openxr_behavior_parity.py tests/test_openxr_vulkan.py`：107 passed，2 warnings。
- `src/python3/python.exe -m pytest -q`：618 passed，6 warnings。
- 当前 Windows Bridge 能力探针：外部屏幕图像、Vulkan external image、ready semaphore、finished semaphore、async submit 均为可用；CUDA runtime external semaphore API 也可见。
- 真实 Vulkan runtime 小帧验证：`StereoRuntime(stereo_compute_backend="vulkan")` 完成 Vulkan context 创建、fused dispatch、同步和左右眼读回；Vulkan host output adapter 在真实 Vulkan context 上创建并上传 8×4 左右眼图像成功。
- 真实 Vulkan layered runtime 小帧验证：`d2s_stereo_layered.spv` 在 NVIDIA RTX 3090 上完成 16×32、3 层 dispatch，左右眼和遮挡 mask 尺寸正确且均为有限值。
- 真实 Vulkan direct-image smoke：`d2s_stereo_layered_output.spv` 在外部 device-local `VkImage` 上完成左右眼 dispatch，状态转换到 `SHADER_READ_ONLY_OPTIMAL`，输出元数据为 `vulkan_readback=none`、`vulkan_output_sync=vulkan_compute_external_semaphore`；Presenter `VulkanZeroCopyOutputAdapter` ring smoke 通过。
- Vulkan 接入回归测试全部通过；覆盖自动后端切换、OpenXR prewarp 接入、Vulkan runtime integration 和 host output adapter。

## 2026-07-24

### 已实现

- 默认开启 Filament 外部源 `VkImage` zero-copy 实验路径；Presenter 仍执行完整能力门控，条件不满足时自动回退 Vulkan GPU copy/Quad Layer。设置 `D2S_ENABLE_FILAMENT_SCREEN_IMAGE=0` 可恢复旧路径进行回归对比。
- 接入 Filament v1.74.0 Vulkan backend 源码远程构建：新增 `native/filament/patches/apply_d2s_vulkan_external_image.py`，为 `VulkanPlatform` 增加正式的借用式外部 `VkImage` 元数据和工厂接口；GitHub Actions 在 Windows、Linux、macOS 远程构建 patched Filament 与 Bridge，并通过 `D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE` 编译开关报告能力。本机不编译 C++，旧 Bridge/stock SDK 仍由能力探针自动回退 GPU copy。
- 修复 Filament 源码远程构建的跨平台差异：Linux runner 显式使用 Clang，macOS 外部图像元数据接口改为无 nullability 警告的引用参数，Windows Bridge 从源码安装前缀递归发现实际 `.lib`；上一轮 CI `30158908856` 因这些构建配置问题失败，未生成可用三平台 Bridge。
- 修复 Linux Filament 源码构建缺少 BlueVK XCB 头文件：CI 安装 `libxcb1-dev`；CI `30161119621` 的 Windows、macOS 已成功，Linux 仅因该依赖失败。
- 优化 Filament 远程构建耗时：GitHub Actions 按 runner、架构、Filament 版本和源码补丁 hash 缓存源码、CMake 构建目录及安装前缀；缓存命中时跳过 Filament 全量编译，只构建 Bridge。Linux 同时补齐 `libx11-dev`。
- 补齐 Linux BlueGL 构建依赖 `libgl1-mesa-dev`；CI `30163398945` 的 Windows、macOS 成功，Linux 最后缺少 `GL/gl.h`。
- NVIDIA CUDA producer-ready/consumer-release external semaphore 现在默认开启；设置 `D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE=0` 可单独关闭 CUDA 外部同步进行回归对比。CUDA runtime 或 Vulkan 能力不足时仍自动回退 GPU copy。
- 定位并阻止 Filament Vulkan 外部纹理 native 崩溃：Filament v1.74 公共 `Texture::Builder::import()` 仅支持 OpenGL/Metal 纹理标识，不接受裸 Vulkan `VkImage`。新增 `filament_bridge_vulkan_external_image_abi_available` 能力门控，当前 stock SDK 报告不支持时不再进入危险直采样调用；zero-copy 请求仍保留，运行时安全回退 Vulkan GPU copy，后续扩展 Filament Vulkan backend 后再打开真实路径。
- 修复 OpenXR FPS 面板和操作指南导致的帧率骤降：工具 Quad layer 现在缓存 PIL 栅格化纹理，并复用已上传且已释放的 Vulkan swapchain image；内容未变化时每帧只重建轻量 layer pose，不再重复字体绘制、host staging map/copy 和 acquire/wait/release。FPS 面板接入 Presenter 的真实 XR 提交帧率、运行时输出帧率和输出延迟，并按旧工程每秒采样一次；操作指南保持静态 GPU 纹理复用。虚拟屏幕的每帧立体输出不受影响。
- 修复 Projection Layer 内的显示顺序：虚拟屏幕从 Renderable priority `7` 调整为背景 priority `0`，保持不写深度；手柄 PBR 和深度测试激光在其后渲染，屏幕不再覆盖手柄和激光。该修改需要三平台 Filament Bridge 远程构建后实机验证。
- 修复 OpenXR `Default` 无房间环境中手柄和激光发白：Default profile 显式使用 `preview_exposure: 0.0`，不再继承运行时 `2.0 EV` 的线性曝光默认值。手柄 PBR 材质和激光仍共用既有 View 色彩管线，房间环境与 Default 的曝光行为现在一致。
- 明确 Vulkan 外部屏幕图像直采样的长期规范：每张源 `VkImage` 只创建一次 Filament 外部纹理，并保存格式、尺寸、layout、队列归属、producer-ready 和 consumer-release 状态；CUDA、ROCm/HIP 等 GPU producer 写入后都必须经过 barrier 和 queue ownership transfer 到 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`，Filament 采样完成后才能复用槽位。屏幕源同步与 OpenXR 输出交换链同步分离；能力不完整时回退一次 Vulkan GPU copy/Quad Layer，不退回 CPU 像素传输。直接采样实验路径现已默认开启，但仍待实机长稳验证后转为稳定默认路径。
- 接通 Filament 外部源 `VkImage` 的 zero-copy 同步闭环：CUDA/ROCm producer-ready external semaphore 由 Presenter 在 source barrier submit 中等待，`GENERAL -> SHADER_READ_ONLY_OPTIMAL` 完成后发出每槽位的 device-local visible semaphore，Filament 通过现有 `set_screen_ready_semaphore` 等待后直接采样持久化外部纹理；帧完成后执行 `SHADER_READ_ONLY_OPTIMAL -> GENERAL` release barrier，并以每槽位 exportable consumer-release semaphore 通知 producer，producer 在复用 ring slot 前通过 CUDA/HIP stream wait 消费该 semaphore。同步能力不完整时仍自动回退 Vulkan GPU copy/Quad Layer，不退回 CPU 像素传输。该路径仍需 Validation Layer、三平台 Bridge CI 和实机长稳验证后再改变默认开关。
- 扩展稳定 Filament C ABI：native Vulkan platform 保存 Filament `present()` 提供的 per-eye `finished_drawing` semaphore，并通过 `filament_bridge_get_finished_drawing_semaphore` 返回借用句柄；Python release barrier 现在直接等待该 GPU 完成点后发出 producer consumer-release，不再依赖 `wait_for_idle()` 推断采样结束。该 ABI 修改需要三平台 Bridge 远程构建产物才能实机使用。
- 固化“命令队列 + Presenter 线程执行”不可回退约束：后台线程只提交原始结果或资源描述，所有 Vulkan/Filament 图形资源创建、外部图像导入、barrier、队列归属转换、纹理绑定、释放和 Projection/Quad Layer 提交均由 Presenter 在 OpenXR 帧边界执行。后续外部 `VkImage` 直采样重构不得恢复后台线程直接调用 Filament C ABI 或操作 Vulkan 资源。
- 按旧工程行为基准恢复控制器输入语义：Vive/WMR 触控板上/中/下方向模拟、模拟按键与真实 click 的互斥、OpenXR changed 边沿回退、touch/click 区分以及按键/摇杆动画目标平滑重新接入 Vulkan Presenter。该修复只调整 Python 输入策略，Filament/Vulkan 资源仍由 Presenter 命令队列线程独占。
- 修复工具 FPS Quad Layer 的隐性每帧重绘：延迟值改为与 XR/SBS FPS 一起按约 1 秒快照更新，内容 key 不再被每帧延迟变化击穿；操作指南继续复用静态纹理，XR 帧只更新 Quad layer 姿态和提交结构，避免重复字体栅格化与 staging 上传拖慢 Projection Layer。
- 收紧屏幕外部 `VkImage` 直采样门槛：显式开启实验路径时必须提供 per-eye producer-ready、source prepare、visible semaphore 和 consumer-release 回调；源图像初态为 `GENERAL`，由 Presenter 在 GPU submit 中完成采样前转换，不再要求生产者伪造 shader-read 初态，也不再仅凭 producer semaphore 导入未完成同步的图像。
- 输出契约现在发布左右眼源 `VkImage` 的真实 layout 和 queue family；CUDA 写入后的 `GENERAL` 状态会被明确暴露，未经过 source barrier 的图像不会被误判为可供 Filament 采样。
- Vulkan context 增加显式的 source barrier/release barrier API，并同步维护 `ImageStateTracker`；提交器同时支持 binary semaphore wait/signal，并与现有 timeline wait/signal 合并到同一次 `vkQueueSubmit2`/`vkQueueSubmit`。

### 验证结果

- 增加 Default profile 曝光回归测试；待实机确认手柄高光和激光饱和度。

## 2026-07-23

### 已实现

- 修复 OpenXR 环境选择仍硬编码旧 `Artemis` 目录的问题：运行时现在按 `settings.yaml` 的 `Environment Model` 解析环境目录，并读取所选 `profile.json` 的 `glb` 字段；所选目录、profile 或 GLB 无效时记录具体原因并回退 `Default`。`Default` 的 `glb: null` 被视为合法无房间环境，并使用旧版默认的 `2.4m x 1.35m`、距离 `2.0m` 虚拟屏幕，不再静默进入无房间且无屏幕的黑场。
- 修复 OpenXR 单 View 颜色跨帧叠加：Filament `Renderer::ClearOptions` 默认 `clear=false`，此前只在绿色透视背景快捷键切换时才初始化，普通外部 swapchain 渲染会保留上一帧颜色。Bridge 现在为每只眼在创建时固定设置黑色 clear color、`clear=true` 和 `discard=true`。
- 修复单 View 重构后 OpenXR 画面跨帧残影：删除原“主 View 与激光 View 同帧共享深度”遗留的 `setChannelDepthClearEnabled(0, false)`，单 View 现在每帧清理 channel 0 深度；此前设置会让外部 OpenXR swapchain 的深度跨帧保留，导致场景、屏幕和激光连续叠加。
- 重构 Filament 控制器激光路径：删除独立 LDR 激光 View、重复 controller asset 和深度遮挡副本；GLB、手柄 PBR、屏幕/UI 与激光现在在一个主 View 中共享同一 Scene 和深度缓冲。手柄外壳写入深度后，激光以深度测试自然被遮挡。
- 主场景 ColorGrading 改为 `ToneMapping::LINEAR`，不再应用 ACES；保留后处理以在最终 sRGB 输出目标进行唯一一次编码，避免将线性工作空间和最终 transfer function 混为一处。

- 修复 Vulkan 手柄模型近乎纯黑：对照 WebXR Input Profiles 官方 Viewer，Bridge 不再覆盖控制器 GLB 的原始 `roughnessFactor` 和 `specularColorFactor`；新增向后兼容环境光 C ABI，将旧工程环境 profile 的 `env_ambient_color` 作为 Filament SH irradiance 接入，同时保留跟随头部主光和顶部补光。
- 调整 Vulkan OpenXR 启动顺序：同步完成模型与推理后端加载，首帧推理、立体合成和 shape-dependent warmup 发布就绪信号后，才创建 OpenXR Vulkan/Filament presenter。
- 输出消费者在 presenter 初始化完成前不再取走 `runtime_q` 中的首帧，避免图形启动期间丢弃已经预热完成的第一个可提交结果。
- 明确 Vulkan/OpenXR 图形预热契约：启动期预创建 Device、队列、swapchain、Filament 材质/资源和持久化输出槽；首个 graphics pipeline 提交仍在合法 OpenXR frame loop 内完成。
- 按旧工程恢复独立屏幕光叠加：运行时在线性空间异步提取双眼虚拟屏幕平均色，保持旧版 `82%` 屏幕色与 `18%` 中性色混合，并由屏幕中心、法线和对角线衰减驱动只照亮控制器通道的 Filament 聚光源。
- 将屏幕光与 `controller_hdr_lighting` 完全解耦：3D 房间模式继续使用 profile ambient/head/top light，HDR 模式在真实预过滤 IBL 接入前明确使用 profile 回退，两个模式都始终保留屏幕光。
- 修复更换手柄模型后 B 键引导端点错位：统一从当前右手柄 GLB 的 `b_button_pressed_value` 动画枢轴解析局部锚点，覆盖 HP、Index、PICO、Quest、Vive 和 YVR；品牌切换后立即清除缓存、重新计算并记录锚点，再应用当前 profile 校正和 Grip 世界变换。
- 增加手柄品牌级环境光补偿：HP、Valve Index、Vive 和 YVR 暗色模型通过各自 `profile.json` 使用 `20.0x` 环境间接光，PICO/Quest 保持 `1.0x`；启动加载和运行时切换模型都会立即刷新倍率，屏幕光、直接补光及 GLB 原始材质不变。
- 补齐控制器 GLB 动画三元组的等价 native 实现：`_value/_min/_max` 节点不再依赖 Filament `getEntities()` 是否枚举非渲染节点，Bridge 回退表覆盖六品牌全部完整三元组，并继续使用平移/缩放插值与四元数 SLERP。
- 恢复此前只创建但未消费的摇杆、触控板和 Quest thumbrest 触摸状态；触摸通过现有 `button_mask` 第 6 位穿过冻结 C ABI，驱动 touched 节点及触摸轴动画，不新增或改签名。
- 修正手柄激光遮挡方案、透明外壳及连续帧几何残影回归：撤销在两个 View 提交之间临时替换同一 Renderable 材质的异步不安全实现；控制器 GLB 改用 Filament instanced asset，主实例永久保留原始 PBR 材质并只进入 HDR 主 View，独立遮挡实例共享纹理、材质资源和顶点缓冲，但永久绑定 `colorWrite=false`、`depthWrite=true` 的深度材质并只进入激光 View。两实例同步姿态、按键动画和显隐，不再逐帧修改材质绑定。
- 修复原生 GLB 加载导致的 Filament 线程归属崩溃：撤销将 Filament Engine/AssetLoader 放入独立 `FilamentNativeLoader` 的方案；Filament Engine、GLB 资源、控制器、屏幕和眼睛渲染统一由 Presenter 线程拥有，避免 `This thread has not been adopted` 和跨线程 Vulkan 生命周期错误。原生 GLB 阶段日志保留用于定位 `createAsset`、`loadResources` 和 `flushAndWait`。
- 启动 Presenter 线程命令队列重构：输出消费者不再跨线程直接修改 Presenter 的待显示帧，而是投递 `submit_output` 命令，由 Presenter 在帧边界消费；队列覆盖和 Presenter 关闭时都会释放未消费的输出槽位，Filament C ABI 调用继续保持 Presenter 线程归属。
- 为 `FilamentVulkanBridge` 增加 owner 线程绑定：创建、渲染、资源操作和销毁都会在 Python ABI 层校验 Presenter 线程，未来任何跨线程调用会立即得到明确错误，而不是触发 native `This thread has not been adopted`。
- 修复运行时关闭竞态：主线程不再以 2 秒超时抢先调用 Presenter `close()`，先等待 `run_until` 在 owner 线程完成 Filament/Vulkan 释放，再执行无副作用兜底关闭。
- 收紧 Presenter 命令队列边界：OpenXR 输出消费者现在只投递原始推理结果，CUDA 到 Vulkan 图像导入、external semaphore、屏幕光采样、输出槽位租约和 Filament 提交统一在 Presenter 线程执行；非 Vulkan sink 保留兼容转换路径。
- 修复首帧后画面冻结：Presenter 每个 XR tick 只转换命令队列中最新的一条原始结果，避免一次 tick 连续耗尽 Vulkan 输出 ring 后在槽位租约上等待自身完成下一帧释放。
- 修复桌面预览拖影：Preview Bridge 现在显式清理每帧颜色和 channel 0 深度；此前只有 OpenXR eye renderer 设置了 ClearOptions，桌面交换链会保留上一帧的模型和网格。
- 调整桌面预览移动速度：默认为 `1.0 m/s`，按住 `Shift` 加速到 `5.0 m/s`，按住 `Ctrl` 降速到 `0.5 m/s`；VIEW 向下移动改用 `Alt`，避免与 Shift 加速冲突。

### 验证结果

- 启动顺序、pipeline 就绪事件、输出首帧门控与 OpenXR 定向回归 `81 passed, 2 warnings`。
- 手柄环境光、独立屏幕光、异步屏幕颜色采样和输出首帧定向回归 `13 passed, 2 warnings`。
- 完整测试套件 `526 passed, 6 warnings`，requirements-matrix 合规检查 53 项通过；待三平台 Bridge CI 和 OpenXR 实机亮度验收。
- 六品牌 B 键动画枢轴、引导几何、品牌环境光倍率及切换后立即刷新定向回归 `20 passed, 2 warnings`；完整测试套件 `539 passed, 6 warnings`。
- 控制器动画三元组、touch 输入链和稳定 C ABI 定向回归累计 `100 passed, 2 warnings`；完整测试套件 `551 passed, 6 warnings`，requirements-matrix 合规检查 53 项通过。
- Presenter 原始输出命令队列及线程归属回归 `90 passed, 2 warnings`；Python `py_compile`、`git diff --check`通过。
- 上一版“手柄/激光同一彩色 View”以及后续“同一 Renderable 临时换深度材质”方案均经实机判定无效：前者不能正确遮挡且会改变外壳合成，后者与 Filament 异步命令消费竞争并造成场景/手柄连续帧残影。现已统一替换为资源共享、Renderable 独立、材质绑定持久不变的双实例遮挡结构；本地完整回归 `550 passed, 1 deselected, 6 warnings`，排除项仅为工作区 Artemis 目录改名导致的旧路径测试；三平台 Bridge CI `29991685595` 全部通过，自动二进制提交 `af11576` 已拉取，Windows DLL SHA256 为 `7D27C6E2298192C72A0B1718CD6BDC979757E45D3AB0865A9ADD38089815CF3A`，待头显遮挡验收。

### 未决事项

- HDR 图片环境尚缺与源 HDR 匹配的预过滤 reflection cubemap 与 irradiance KTX 接入；当前日志为 `hdr_ibl_pending_profile_fallback`，不会将 profile 回退误报为完整 IBL。
- 新增的环境光和屏幕光 C ABI 需要提交后由 GitHub Actions 完成 Windows、Linux、macOS 三平台 Bridge 远程构建，再下载产物进行实机亮度 A/B。

## 2026-07-22

### 已实现

- 重构原生 Filament Bridge：`filament_bridge.cpp` 仅保留稳定 C ABI 转发，Python ctypes 接口和 `filament_bridge.h` 不变。
- 将共享 Engine/Scene、双眼目标、GLB 场景、控制器动画、3D 激光、外部 VkImage 屏幕、材质色彩和桌面预览拆分为独立 `.cpp/.h` 模块，内部共享类型集中到 `bridge_internal.h`。
- CMake 显式编译各 Bridge 模块，并默认隐藏内部 C++ 符号，防止模块实现意外扩展 Python ABI。
- 三平台 Bridge 二进制改为分别存放在 `src/xr_viewer/native/windows`、`linux`、`macos`，运行时解析、能力探测、CMake 输出和 GitHub Actions 产物回写使用同一目录契约。
- 对齐旧工程控制器生命周期：补齐双手上一帧姿态、最后移动时间和移动阈值，修复 Aim 更新因状态未初始化而被静默清空。
- 增加逐手控制器显隐 ABI：Grip 跟踪有效且 5 秒内有移动时显示，静止超时或跟踪丢失时隐藏模型和激光，恢复移动后立即重新显示。
- 补齐此前缺失的 native 激光实现：在共享 Filament Projection Layer Scene 中创建双手独立 3D 光束实体，跟随 Aim 负 Z 位姿并与控制器同步显隐。
- 对齐旧工程激光标定参数：采用 Grip 上移 20mm、前移 110mm、Aim 绕局部 X 轴偏转 12 度、0.4m 长度和 6mm 根部宽度。
- 迁移旧工程激光稳定逻辑：位置使用 One Euro Filter，方向使用四元数 SLERP 和 0.3 度死区，逐帧更新后供 Vulkan Projection Layer 光束使用。
- 修复控制器 ABI 判定：手柄模型加载只依赖旧工程已验证的加载、姿态和输入三个接口，不再被可选激光接口缺失阻断。
- 修复实机手柄模型消失：当远程 DLL 尚未导出激光接口时保留控制器 GLB、姿态和按键动画；激光调用改为按 ABI 能力门控。
- 按旧工程恢复手柄激光外观：使用两张交叉锥形面、6mm 根部宽度、2mm 尖端宽度和沿光束流动的蓝/青/绿/黄/橙/红动态渐变，不再使用单张蓝色透明平面。
- 修复 native 按键动画插值：按钮、Trigger、Grip 和摇杆输入采用旧工程 24Hz 响应平滑，旋转由矩阵逐元素插值改为四元数 SLERP，同时保留平移与缩放插值。
- Bridge 加载手柄 GLB 时输出逐手动画节点数量、节点名和语义，避免 `_value/_min/_max` 匹配失败后静默运行。
- 修复彩色激光导致 OpenXR 初始化退出：将材质参数改为 `laser_time`，并按 Filament 约定使用普通参数访问器 `materialParams.laser_time`；sampler 才使用 `materialParams_<name>`。
- 规范捕捉到渲染的颜色空间路径：OpenXR Projection Layer 严格选择 sRGB swapchain，显示参照的虚拟屏幕和激光使用独立无后处理 LDR View，禁止对 SDR 颜色重复曝光、色调映射或传输函数转换。
- 修复手柄激光颜色偏淡及流动方向错误：激光改为不透明材质，颜色从手柄根部向远端流动，并绕过场景 ACES 色调映射。
- 按旧工程恢复手柄照明方式：手柄仅接收跟随眼睛的主光和顶部补光，环境灯与 HDR 反射不参与；灯光位置、颜色和顶部补光比例与旧工程保持一致，并将无单位强度转换为 Filament 坎德拉。
- 修复手柄初始纹理透明及隐藏恢复后变暗：移除会绕过 glTF 不透明材质合成链路的独立手柄 View，初始加载和 5 秒隐藏/恢复始终使用同一主场景层和专用灯光通道。
- 修复手柄按键动画完全无响应的根因：`_value`、`_min`、`_max` 节点统一从各自控制器 GLB 查询，不再错误地从环境 GLB 查询；左右手各 9 组动画节点按旧工程契约补齐并去重。
- 修正右手菜单动画节点名 `RMenu_pressed_value` 为 GLB 中真实存在的 `RMenu_value`；动画节点为空时控制器加载明确失败并报告错误，不再静默显示无动画模型。
- GitHub Actions 完成 Windows x86_64、Linux x86_64、macOS arm64 Filament Bridge 编译及二进制回写，本地同步至提交 `2aa8bbf`。
- 新增右手柄 B 键近距导引：手柄距头显 0.4 米内自动显示，仅保留 B 键透明说明框，并从 PICO GLB 的真实 B 键节点计算引导端点。
- 将 B 键导引由 OpenXR Quad Layer 迁移到 Filament Projection Layer 的无后处理 LDR 层；透明纹理、白色边框和文字不经过场景曝光或色调映射，面板逐帧朝向头部并跟随按键旋转。
- 修复 Projection Layer 导引显示成白色大方块：按 Filament `transparent` 材质契约在纹理采样后执行预乘 Alpha，透明区域不再把保留的白色 RGB 直接混入画面。
- 将旧工程控制器短按/长按判定抽取为渲染后端无关的快捷键状态机，Vulkan 现复用相同语义处理 A/B/X/Y、菜单键、摇杆点击及握持组合键。
- 以操作指南为完整快捷键契约，补齐 A+B 手柄品牌切换/模型校准、Grip+摇杆屏幕旋转缩放与深度调整、桌面方向键/滚轮、虚拟键盘移动旋转缩放，以及单手/双手 Grip 激光拖动；键盘专用组合键与深度快捷键互斥，不再发生漏导入或按键抢占。
- 补齐 Vulkan 快捷键后端：A 键切换 48 段圆柱弧/平面屏幕，Y 键复位或轮换旧工程屏幕预设，X 键切换键盘、环境亮度或绿色透视背景，摇杆组合键控制 2D/3D、深度复位及系统复制/剪切/粘贴/回车。

### 验证结果

- Python 编译检查通过。
- OpenXR/Filament Bridge 定向测试 `53 passed`，完整测试套件 `486 passed`。
- 最新 OpenXR/Filament 控制器定向回归 `54 passed, 2 warnings`，左右手 GLB 各 9 组 `_value/_min/_max` 节点逐名校验通过。
- 用户实机验收通过：手柄纹理初始显示不透明，静止隐藏后恢复亮度一致；Trigger、Grip、摇杆及可用实体按键动画均可正常响应。
- B 键 Projection Layer 导引定向回归 `62 passed, 2 warnings`，Python 编译检查与 Git diff 空白错误检查通过。

### 未来目标

- 建立独立 `d2s-vulkan-1.4` 实验分支，基于固定版本 Vulkan 1.4 `vk.xml`/Headers 远程生成项目自用 Python binding wheel；生产分支继续保留 Vulkan 1.2/1.3 能力回退。
- 在 Vulkan 1.4 binding 可用后，对 `hostImageCopy` 与独立 Transfer Queue 的工具纹理上传路径进行同场景基准测试；只有 Validation Layer、三平台 CI 和实机性能数据证明有净收益时才合入主路径。

## 2026-07-21

### 已实现

- 修复实机控制器按键动画未更新：按旧工程的中性 `value -> min/max` 方式插值 `_pressed_value` 节点，补齐 PICO `LPico/RPico` 语义。
- 将手柄激光从易丢失的 Quad Layer 改为 Filament Projection Layer 3D 几何体，使用 Aim 负 Z 射线和每帧世界变换提交。
- 修复实机控制器按键动画未更新：识别 PICO `LPico/RPico` 节点别名，并让 Bridge 的每帧动画刷新同时更新控制器 `_pressed_value` 节点。
- 提高 Vulkan 激光在头显中的可见性：沿旧工程 Aim 负 Z 射线保持 Quad Layer 提交，但扩大纹理采样核心和光束宽度，避免细光束在实际角分辨率下消失。
- 修复非 Windows 测试导入 `_KEYEVENTF_KEYUP` 失败：Windows 输入常量在 no-op 平台分支保持同名导出，GitHub Actions Linux 合规测试可正常收集 OpenXR 测试。
- 补齐 GitHub Actions OpenXR 测试依赖 `Pillow`，确保工具 Quad Layer 纹理模块在 Linux 合规环境可导入。
- 修复实机 `FilamentBridge.end_frame()` access violation：日志确认崩溃来自缺少源图像同步时的运行时 VkImage 直导入屏幕路径；无同步契约时使用旧工程已验证的 Projection Layer 场景加 Quad Layer GPU copy，zero-copy 仅在能力门控通过时启用。
- 将 Filament 屏幕直采改为能力门控：默认保留 zero-copy 意图，但只有输出帧同时提供左右眼 `gpu_external_semaphore` ready semaphore、Bridge 屏幕图像 ABI 和 semaphore ABI 时才启用；未满足同步契约时自动回退 Quad Layer GPU copy，避免未同步 raw `VkImage` 进入 Filament。
- 新增后端无关的 `GpuProducerAdapter` 契约；CUDA 适配器改为具体实现，输出同步模式统一为 `gpu_external_semaphore` / `gpu_synchronized`，为 ROCm/HIP 等后端接入保留同一 Vulkan producer 边界。
- Presenter 改为通过后端注册表创建 GPU producer，不再直接依赖 CUDA 类；已注册 CUDA 与 ROCm/HIP 适配器，后端自动识别，ROCm/HIP external semaphore 默认按 runtime 能力启用，API 不可用时明确回退 GPU copy，不会伪装成其它厂商或走 CPU 像素回传。
- ROCm/HIP runtime 缺失或 external-memory ABI 不可用时，适配器创建失败现在被 Presenter 捕获并限次记录，不再让 OpenXR Presenter 线程异常退出；后续帧仍可重试适配器创建。
- capability report 新增 GPU producer 自动选择结果、CUDA/HIP runtime 状态和覆盖标记，AMD 实机可通过 `--probe` 直接确认是否选择 ROCm，无需猜测环境变量。
- 修复 FPS/键盘等工具 Quad Layer 上传崩溃：`VulkanHostImage.upload()` 改用 PyVulkan 映射内存提供的可写 buffer，按 `rowPitch` 写入像素，不再把 cffi 映射对象错误转换为整数指针；菜单键打开 FPS 面板不再触发 `TypeError` 退出 XR 线程。
- 对齐旧工程控制器动画语义：补齐 PICO `photo/home/app` 按键到菜单动画映射，并将摇杆按下状态传入 native Bridge；控制器 `_pressed_value` 节点现在可响应按钮、摇杆和扳机输入。
- 修复 Vulkan 激光不可见：旧实现使用 Aim 射线负 Z 方向绘制长条光束；Vulkan Quad Layer 现在提交沿 Aim 射线排列的蓝色长条纹理，不再把光束放在控制器后方的微小圆点位置。

- 继续迁移旧工程完整 OpenXR 控制状态机：菜单、A/B/X/Y 和左右摇杆按键均支持短按/长按计时；短按/长按快捷键通过 Windows 输入注入，X 键切换虚拟键盘，菜单/A/B/Y 控制工具面板与屏幕复位。
- 迁移旧工程键盘输入保持状态：触发器进入、悬停、按住、切换按键和释放均由 `CoreInputHelpersMixin` 管理，支持 Shift/Ctrl/Alt/Win、Caps Lock、双击修饰键和方向键注入；Grip 按下时抑制误触键盘。
- 迁移旧工程鼠标长按/拖动状态：触发器先点击，超过 350ms 后进入拖动，释放时保证发送对应鼠标抬起；左右手分别映射右键/左键，键盘命中时不再穿透为桌面鼠标。
- 补齐 Vulkan 工具交互：左 Grip 锁定并移动键盘或虚拟屏幕，右 Grip 按横向位移调整屏幕宽度并保持纵横比；键盘 Quad Layer 使用当前位姿、Shift 状态和实时键盘尺寸生成。

- 修复 OpenXR 场景发白：保留旧工程验证过的 `R8G8B8A8_SRGB`/`B8G8R8A8_SRGB` Projection Layer 目标，将 Filament ColorGrading 输出改为线性 Rec709，由 sRGB 目标执行唯一一次 OETF；虚拟屏幕 Quad Layer 继续独立使用 UNORM 链。
- 修复实机 Quad Layer 屏幕变成长条：profile 未显式提供高度时按宽度自动计算 16:9 高度；修复 profile 校准后控制器仍使用旧 OpenXR reference space，手柄位姿现在与场景使用同一世界空间。
- 修复 Artemis 星空纹理转头闪烁：为 `Skybox__6464723579082975951` 的 8192x4096 纹理启用三线性 mipmap 采样，保留原始星空图像内容和空间位置。
- 修复 Quad Layer 虚拟屏幕上下颠倒：运行时输出和 Quad swapchain 统一采用 `top_left` 行序，拷贝路径不再强制额外 Y 翻转，Projection Layer 复制路径不受影响。
- 修复 Quad Layer 虚拟屏幕左右镜像：移除拷贝路径中的 X 翻转，避免把源图像方向问题误当成屏幕姿态问题。
- 完善统一输出契约：`VulkanStereoOutputFrame` 现在显式声明 `color_space=srgb` 和 `image_origin=top_left`；Quad Layer 不再对 `top_left` 源图像重复做方向转换。
- 修复 OpenXR Quad Layer 色彩路径：优先选择 sRGB Quad swapchain，与旧工程验证过的输出策略一致；OpenXR 配置现在使用用户选择的控制器型号。
- 修复 Filament 控制器模型全黑：控制器 GLB 加载后加入共享 fill-light channel，并保留各控制器 `profile.json` 的偏移/旋转校正。
- 对齐旧工程环境视角校准：profile reference space 应用时水平化初始头显姿态，再重新定位视图，避免实机视角偏离预览位置。
- 修复 Quad Layer sRGB 回归：UNORM runtime eye 到 sRGB Quad swapchain 现在使用 Vulkan blit 完成兼容格式转换，不再因格式不一致导致 OpenXR 线程退出。
- 统一 Quad Layer 图像方向：`image_origin=top_left` 在 Vulkan 拷贝路径不执行额外 X/Y 翻转，屏幕姿态与图像行序独立处理，避免历史硬编码镜像污染原始画面。

- OpenXR 运行时 Vulkan 中间图像保持 UNORM 存储；Filament 屏幕纹理按 sRGB 语义采样，Projection Layer 使用 UNORM 目标，避免已编码输出重复执行传输函数。
- 虚拟屏幕接入运行时左右眼 Vulkan 输出：导出图像增加 `SAMPLED` 用途，Filament Bridge 新增窄 C ABI，将借用的 Vulkan 图像导入屏幕材质；不引入 CPU 回读。
- 补充 Pico 4、Pico 4U 和 Pico Neo3 的 OpenXR interaction profile 绑定别名，控制器模型继续使用 Grip 位姿并回退到 Aim 位姿。
- 对照旧 `4k-stereo-synthesis-lab` 的已验证 Projection/Quad Layer 路径修正色彩契约：运行时输出帧显式标记 `color_space=srgb`，Filament 屏幕纹理使用 `SRGB8_A8` 采样；Projection Layer 使用 sRGB 目标且 Filament 输出线性 Rec709，Quad Layer 独立使用 UNORM；本地预览、MJPEG 和 RTMP 保持 display-referred sRGB，不重复 gamma。
- 桌面 Filament Preview native window swapchain 同样启用 `CONFIG_SRGB_COLORSPACE`，避免 Preview 与 OpenXR 使用不同的目标转换。
- 修复 Hugging Face 模型下载链：`snapshot_download()` 现在在实际选中的 `HF_ENDPOINT` 上执行；残缺的“只有权重”缓存不会再被误判为完整模型，降级 HTTP 下载会补齐 `config.json`，并保留在线 endpoint fallback。
- 按旧工程模型边界区分配置来源：DA3、InfiniDepth、VideoDepthAnything 使用 `src/stereo_runtime/model_impl` 内置结构配置，只要求远程权重；通用 Transformers 模型继续要求远程 `config.json`。
- 对齐旧工程 OpenXR 待机恢复逻辑：头显未连接或处于待机时不再退出 Vulkan 线程，而是使用可中断退避等待；`STOPPING/LOSS_PENDING` 后释放并重建 OpenXR/Vulkan 资源。
- 接入待机推理门控：头显等待前 60 秒保留 source 推理宽限期；持续不可用超过 60 秒后清空队列、停止捕捉和推理，头显恢复后重新打开推理并清理旧帧。
- 修复实机待机回调调用不存在的 `StereoRuntime.set_inference_active`：StereoRuntime 现在提供统一推理门控，并在暂停状态拒绝新的 RGB/OpenXR 推理帧。
- 修复 `WindowsCaptureCUDA` 与 TensorRT CUDA Graph 的 stream 冲突：检测到 CUDA 捕获时强制关闭已遗留的 depth CUDA Graph，并重建 provider 后使用普通 TensorRT enqueue。
- 明确记录 OpenXR 头显等待状态：首次检测不到头显或头显待机时输出一次等待提示；恢复时重置提示状态，避免等待逻辑静默。
- 重构 Filament Vulkan Bridge：左右眼现在共享一个 Filament Engine、Scene、GLB、控制器、屏幕材质和 Shader；每只眼睛仅保留独立 View、Camera、外部 OpenXR swapchain 和 acquired image。
- 头显未连接时明确记录 `xr.get_system` 尚未获得 HMD form factor，Vulkan/Filament 初始化会延迟到头显唤醒，不再让日志看起来像 Engine 创建失败。
- 修复头显从 60 秒 hard idle 恢复后的 Vulkan 外部图像生命周期竞态：等待态清空并拒绝旧输出帧，只有 `session_running` 且头显恢复渲染后才接收新帧；Filament 销毁导入屏幕纹理前等待 GPU 完成，避免 `Handle ... is being used after it has been freed` 导致原生进程中止。
- 修复 OpenXR 首帧退出：运行时输出尚未到达时，Filament 屏幕 Renderable 不再提前加入 Scene；收到有效 Vulkan 屏幕图像后才绑定 sampler 并显示，避免未设置 `screenTexture` 触发无效句柄访问。
- 修复双眼外部 Swapchain 的 Filament 帧状态隔离：共享一个 Engine、Scene 和资源，但左右眼各自使用独立 Renderer、View、Camera 和 Swapchain，避免单 Renderer 在两个 OpenXR Swapchain 间切换造成首帧 access violation。
- 为 OpenXR native Bridge 增加有界诊断：记录前八个立体帧的 eye、acquired image index、VkImage、Renderer 和 Swapchain 句柄，便于区分 OpenXR 图像句柄失效与 Filament 内部资源失效。
- 按旧工程 `OpenXRFrameGate` 补齐首帧门控：`should_render` 仅表示运行时允许渲染；在 `_pending_output` 尚未收到有效立体帧时只提交空 OpenXR 帧，不访问 Filament 或外部 swapchain，避免待机恢复阶段使用失效句柄。
- 修复首帧 Filament access violation 根因：不再把普通 Vulkan `VkImage` 直接传给 Filament 未定义 Vulkan 行为的 `Texture::Builder::import()`；虚拟屏幕后续按旧工程使用独立 OpenXR Quad Layer 接入。
- 接入 OpenXR Quad Layer 屏幕路径：首帧输出后按实际推理尺寸延迟创建左右眼 UNORM swapchain，使用 Vulkan GPU copy 写入并提交独立 Quad Layer；Quad 资源格式不再错误复用投影 sRGB 格式。
- 修复 Quad Layer 接入后的画面闪烁：首帧建立后，在没有新推理帧的 OpenXR tick 中复用上一帧 Projection/Quad Layer，不再提交空 layer；只有首帧前才进入等待状态。
- 对齐旧工程世界姿态处理：profile 座位姿态只在首个有效头部姿态时写入 OpenXR LOCAL reference space，并重新定位一次 views；后续 Filament 相机与 Projection Layer 使用同一套世界坐标 views，避免场景跟随头显初始姿态或转头抖动。
- 进一步对齐旧工程 reference space 选择：OpenXR Vulkan 路径优先使用 `STAGE` 地面世界坐标，运行时不提供时才回退 `LOCAL`；profile 校准复用实际选择的 reference space 类型，避免 LOCAL 原点绑定头显启动方向。
- 修复头显转动时场景回弹抖动：首帧后每个 OpenXR tick 都按当前头显 pose 重新渲染 Filament 世界，仅复用没有新推理帧的 Quad Layer 输入，避免用上一张旧姿态投影图替代当前相机姿态。
- 修复 GUI 子进程日志拼接误报：stdout/stderr 合并后若 profile 成功消息与 `[FPSBreakdown]` 粘连，先按日志标记拆分再分类，避免 `fx_entry_failed=` 等统计字段把成功消息标成 ERROR。
- 完成 CUDA/Vulkan/Filament external semaphore ABI 的三平台远程编译：GitHub Actions 运行 `29818061943` 的 Windows、Linux、macOS Bridge 构建及二进制回写全部成功；本地已同步 `filament_bridge.dll`、`libfilament_bridge.so` 和 `libfilament_bridge.dylib`。
- Vulkan 优化状态明确为分阶段完成：输出图像环、持久化纹理缓存、external semaphore 异步同步、双眼统一提交和单 Engine 资源共享已完成；完整 Compute Graph、Validation Layer、跨厂商互操作、性能基准和实机长稳验收仍未完成，不能标记为整体完成。
- 修复实机 `Windows fatal exception: access violation`：根因是 CUDA `cudaSignalExternalSemaphoresAsync` 的 ctypes 调用参数数量和 `cudaExternalSemaphoreSignalParams` 内存布局错误；现已按 CUDA Runtime 头文件使用 `extSemArray + paramsArray + count + stream` 的 ABI，并加入结构体偏移/尺寸回归测试。
- 针对 external semaphore 接入在 Filament `beginFrame` 阶段暴露的 native 生命周期风险，改为 `D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE=1` 显式启用，默认使用已验证的 CUDA stream 同步降级；Vulkan 输出图像环和持久化纹理缓存继续启用，避免实机默认路径再次发生 native access violation。
- 补齐 Vulkan 输出槽位消费端释放/复用保护：新增 producer lease 和跨线程条件等待；pending 帧被新帧替换时释放，当前 Filament 屏幕帧在 Projection/Quad 提交完成前保持占用，头显待机、提交失败和关闭路径统一释放，ring wrap 不再覆盖仍被消费的 VkImage。
- 修复实机首次 Projection 渲染 access violation：默认关闭运行时 CUDA `VkImage` 直接导入 Filament 屏幕材质的路径，屏幕改由已验证的 OpenXR Quad Layer Vulkan GPU copy 提交；保留 `D2S_ENABLE_FILAMENT_SCREEN_IMAGE=1` 作为后续 Validation Layer 验证用显式实验开关。
- 修复共享 Filament Engine 双眼切换 access violation：每只眼 `endFrame` 后先执行 `flushAndWait`，再切换到另一只外部 Vulkan Swapchain，避免上一只眼仍在后端处理时调用下一眼 `beginFrame`；该安全串行基线需要三平台 Bridge 重新远程编译。

### 验证结果

- 项目 Python 环境 `src/python3/python.exe` 完成语法检查。
- OpenXR、输出契约和运行时输出定向测试：`31 passed, 2 warnings`。
- 待机门控、CUDA 捕获隔离和 OpenXR 定向测试：`57 passed, 2 warnings`。
- 单 Engine 双眼 Bridge ABI 与 presenter 定向测试：`33 passed, 2 warnings`。
- 旧工程首帧门控契约测试：`1 passed`。
- OpenXR Quad Layer 定向测试：`37 passed, 2 warnings`。
- 用户实机验收通过：Vulkan Validation Layer 全路径验证通过；NVIDIA OpenXR 实机长稳、帧率和显存压力测试通过。
- `git diff --check` 通过。

### 未决事项

- external semaphore 仍默认关闭，待独立启用实验路径的跨 API 验证；完整 Vulkan Compute Graph、AMD ROCm/Apple 互操作和 Preview/OpenXR 色彩 AB 仍待完成。

### 下一项内容

- 完成 Vulkan Compute Graph 全路径接入，并继续验证 external semaphore 实验路径和跨厂商互操作。

## 2026-07-20

- Added a bounded runtime output consumer that converts only registered Vulkan eye resources into the unified output contract and reports Torch/CPU results as waiting for a vendor interop importer; no implicit CPU image readback is allowed.
- Extended the compliance workflow to run Vulkan resource, interop, output, runtime-output, pipeline, CUDA interop, and OpenXR lifecycle tests.
- Added exportable Vulkan image slots with Win32 HANDLE/FD export through the raw Vulkan loader entry point; resource ownership remains explicit and bounded.
- Added the Python-only NVIDIA CUDA Runtime importer: one-time external-memory import per slot and asynchronous CUDA-to-Vulkan RGBA copy, followed by stream synchronization before Vulkan copy.
- OpenXR Vulkan device creation now merges the Runtime-required device extensions with the platform external-memory extensions before xrCreateVulkanDeviceKHR.
- Runtime output now lazily creates two CUDA/Vulkan eye slots and submits the resulting Vulkan resources through the existing OpenXR projection path.
### 未决事项

- NVIDIA CUDA external-memory + 单次 GPU copy 已实现并通过 RTX 实机验证；ROCm/HIP、Apple Metal/IOSurface 和 CUDA/Vulkan external semaphore 仍待补齐。
- OpenXR 交换链到双眼推理图像的真实头显提交尚未实测；当前机器头显不可用，不能把清屏或单元测试视为头显验收。
- 完整预处理、深度后处理、视差、变形、修补和时域稳定 Compute Pass 尚未全部接入 Vulkan Graph。

### 下一项内容

- 使用已实现的 NVIDIA CUDA external-memory 单次 GPU copy 路径进行 OpenXR Projection Layer 头显实测；随后补 CUDA/Vulkan external semaphore 和 AMD ROCm/HIP 适配器。

### 已实现

- 新增 `shaders/manifest.json`，为每个 Compute Shader 固化入口、workgroup、descriptor binding、push constant 大小、精度和 SPIR-V 文件映射。
- 新增 `src/tools/validate_shader_manifest.py` 与 `tests/test_shader_manifest.py`，校验 Shader 源码声明、manifest 和已提交 SPIR-V 文件一致；GitHub Actions Shader Job 现在会执行该校验。
- 新增 `src/viewer/vulkan_interop.py`，建立 Capture/Inference 到 Vulkan 的非 CPU 回读资源边界：能力报告、外部图像导入请求、有限 in-flight 生命周期和 OpenXR/厂商适配器注册入口已统一；CUDA/ROCm/DMABUF 的平台句柄导入仍必须由各自适配器实现，当前不会伪造零拷贝状态。
- `VulkanImageCopyPass` 和 `VulkanRuntimeSession` 现在接受外部导入的 `VulkanImageResource`；新增 `submit_external_image_pair()`，厂商适配器可将资源直接送入 Compute Graph，并透传上游 timeline 完成值。
- OpenXR Projection Layer 组装集中到 `OpenXrCompositionBuilder`；swapchain image 在 acquire 成功后无论 wait 或渲染是否失败都会 release，避免 wait 异常留下悬挂 acquired image。
- 新增 `VulkanStereoOutputFrame` 和 `LatestFrameOutputRouter`，统一 Preview、OpenXR、Headless/Encoder 的左右眼、SBS、格式和 GPU ready timeline 输出契约，并限制每个输出路由只保留最新帧。
- 新增 `src/tools/vulkan_transfer_smoke.py`，验证两个 Vulkan storage image 在无 CPU 回读条件下通过 `vkCmdCopyImage` 和 layout barrier 完成 GPU copy，目标图像进入 `COLOR_ATTACHMENT_OPTIMAL`。
- Vulkan Context 关闭时现在先清理外部 image registry；即使 pending 状态导致正常注销失败，也会丢弃非拥有型句柄引用，不把已销毁 Device 的资源留在 Context 对象中。
- 迁入并接通 Python runtime context/callbacks，新增 `run_processing_runtime()`；GUI 调用的 `--runtime` 现在会启动 CaptureSessionLoop 和 RuntimePipelineLoop，不再返回“runtime is not assembled yet”。
- OpenXR 模式现在由 `run_processing_runtime()` 启动并管理 `OpenXrVulkanPresenter.run_until()` 线程；Presenter 的关闭顺序纳入运行时 shutdown，不再依赖独立 smoke 入口才能建立 Vulkan Session。
- RuntimePipelineLoop 现在对单帧推理异常执行丢帧并计数，连续达到 `D2S_RUNTIME_REBUILD_AFTER_ERRORS`（默认 3）后重建 Depth Provider、清除时域状态并记录重建失败。

- Compute Graph 的 `VulkanStereoSubmission` 新增可选 `ready_timeline`，上游 GPU 任务完成值现在会通过 `VulkanContext.submit_on("compute", wait_for_timeline=...)` 进入 Compute Queue；没有依赖值的旧调用保持兼容。
- Vulkan Context 新增 `last_submitted_timeline_value`，提交时校验队列角色并检查 FrameContext fence 超时，避免未知队列或无限等待被静默吞掉。
- ImageStateTracker 新增资源注销和 pending ownership transfer 保护；`VulkanStorageImage.close()` 释放 GPU 图像时同步移除状态，避免重用句柄后残留旧 layout/queue owner。
- 需求矩阵补充 VK-005 测试映射，并将 GRAPH-003 的上游 timeline 依赖记录为已实现的执行契约。
- `VulkanComputeGraph` 新增多 Pass 执行入口；`VulkanPassDeclaration` 固定 Pass 名称、workgroup 和资源读写集合，重复 Pass 名称或非法资源声明会在构图时失败。
- 多 Pass 之间仅在前一 Pass 写入、后一 Pass 读取或写入相同资源时插入 Compute Shader memory barrier，避免无条件全局 barrier。
- 新增 `shaders/d2s_copy_image.comp` 和对应 `vulkan_compute_smoke.py` 双 storage-image Descriptor 路径，作为 RGB/Depth 图像 Pass 的第一条真实输入输出链。
- 通过 `winget` 安装 Khronos Vulkan SDK `1.4.350.0`，使用官方 `Bin/glslc.exe` 生成 `shaders/d2s_copy_image.spv`，并重新编译项目 Compute Shader。
- 在 `.github/workflows/compliance.yml` 新增独立 Shader CI：安装 `glslc`/`spirv-tools`，编译全部 `.comp` 并执行 `spirv-val`，输出写入临时目录，不改写仓库中的二进制。

- Vulkan Context 新增 Graphics/Compute/Transfer 队列族选择和队列句柄暴露；优先选择专用 Compute/Transfer 队列，不具备时回退到 Graphics 队列。OpenXR adopt 路径明确复用 Runtime 已创建的 Graphics 队列。
- capability probe 现在报告 `graphics_queue_family`、`compute_queue_family` 和 `transfer_queue_family`，并新增队列族选择回退单元测试。
- Vulkan Context 新增默认容量为 3 的 `FrameContext` 环，命令池、命令缓冲和 fence 按槽位成组管理；提交不再每帧立即等待，而是在复用忙碌槽位时等待对应 fence。
- 用 `ImageStateTracker` 替换裸布局字典，记录 image 的 layout、access mask、pipeline stage 和 queue family；清屏路径现在从已登记状态构造转移 barrier，并在提交后登记目标状态。
- Vulkan 提交路径新增 Timeline Semaphore；当 Python Vulkan binding 暴露完整 Submit2 API 时使用 `vkQueueSubmit2`，否则使用带 `VkTimelineSemaphoreSubmitInfo` 的受控兼容提交。
- ImageStateTracker 新增 Queue Ownership 校验和 `VulkanContext.queue()/queue_family()` 角色查询；图形队列不会静默操作仍归 Compute/Transfer 队列所有的 image。
- ImageStateTracker 新增 pending ownership transfer 状态机，显式区分 transfer begin/release 与 complete/acquire；转移完成前资源不可被任一队列继续使用。
- 每个 FrameContext 现在为 Graphics/Compute/Transfer 分别持有有界 CommandPool、CommandBuffer 和 Fence；新增 `submit_on(role, record)`，`submit()` 保持 Graphics 兼容入口。
- `submit_on()` 新增 `wait_for_timeline` 参数；Submit2 使用 `VkSemaphoreSubmitInfo`，兼容提交使用 `VkTimelineSemaphoreSubmitInfo`，统一表达跨队列 wait/signal 顺序。
- 新增 `VulkanComputeGraph` 最小调度层：支持 `enqueue/flush/submit`、latest-frame 覆盖，并将 Compute Pass 录制回调提交到 `submit_on("compute")`。
- 新增 `shaders/d2s_noop.comp` 作为首个无资源 Compute Pass 源码，以及 `scripts/compile_shaders.ps1`；当前机器没有 `glslc`，未生成或伪造 `.spv` 二进制。
- 新增 Python `VulkanComputePipeline`：校验 SPIR-V、创建 ShaderModule/PipelineLayout/ComputePipeline，并提供 `vkCmdBindPipeline + vkCmdDispatch` 录制入口；没有 SPIR-V 文件时会明确报错。
- `VulkanComputeGraph` 新增 `from_pipeline()` 标准入口，并新增 `src/tools/vulkan_compute_smoke.py`，将 Graph、Pipeline、Dispatch 和 Timeline 验证串成可重复 smoke。
- 新增有界 `VulkanDescriptorArena`：按 `DescriptorBudget` 创建 DescriptorPool、限制 DescriptorSet 数量，并提供幂等释放。
- Compute Pipeline 支持 `DescriptorBinding` 列表，创建对应 DescriptorSetLayout 并挂入 PipelineLayout；默认无 binding 的 noop pipeline 保持兼容。
- 新增 `VulkanStorageBuffer` 和 DescriptorSet storage-buffer 更新路径；`d2s_storage_increment.comp` 实机验证 GPU 将 uint32 从 41 写为 42。
- 新增 `VulkanStorageImage` 和 storage-image DescriptorSet 更新路径；`d2s_storage_image.comp` 实机验证 image 创建、并发队列共享、UNDEFINED→GENERAL 布局转换和 `imageStore` Dispatch。
- Storage Image 的布局转换现在通过公开 Context API 登记到 `ImageStateTracker`，记录 `GENERAL + SHADER_WRITE + COMPUTE` 状态，后续 barrier 可复用统一状态。
- 安装 Vulkan SDK 1.4.350.0 到 `D:\VulkanSDK\1.4.350.0`，使用 `Bin\glslc.exe` 编译生成 `shaders/d2s_noop.spv`。

### 验证结果

- `src/main.py --runtime --runtime-seconds 2` 在 `D2S_RUNTIME_DIAG_STAGE=raw` 下成功启动并关闭 Capture/Runtime 线程；Vulkan transfer smoke 和 CUDA-to-Vulkan image copy smoke 通过；定向互操作测试 `25 passed, 2 warnings`；全量回归 `446 passed, 6 warnings`。
- `check_compliance.py` 通过 45 条需求，Shader manifest 校验通过，GitHub Actions workflow YAML 本地解析通过。
- NVIDIA Vulkan 实机 `vulkan_compute_smoke.py` 通过：`vulkan_compute_smoke: PASS timeline=1 state=ready`、`storage_image_dispatch: PASS`。
- `src/python3/python.exe -m py_compile` 覆盖本轮修改的 Vulkan Graph、Context、Descriptor 和测试文件通过。
- 全量测试 `417 passed, 4 warnings`；同时移除 `src/xr_viewer/gltf/materials.py` 的 UTF-8 BOM，使既有 legacy-depth 静态检查恢复可执行。警告均为 `mss.mss` 弃用提示。
- Vulkan 定向测试与迁移脚手架测试共 `30 passed`，覆盖上游 ready timeline 透传和图像状态注销。
- 迁移脚手架和 OpenXR Vulkan 定向测试共 `31 passed`，覆盖多 Pass barrier 计划和资源依赖声明。
- `vulkan_compute_smoke.py` 通过 `py_compile`，并在 NVIDIA Vulkan 环境中通过双 storage-image GPU smoke：`vulkan_compute_smoke: PASS timeline=1 state=ready`、`storage_image_dispatch: PASS`。
- 本地 4 个 Compute Shader 均通过 `glslc` 和 `spirv-val`；全量测试 `418 passed, 4 warnings`。
- GitHub Actions run `29743777308` 的 `Requirements matrix` 和 `Compile Vulkan shaders` 两个 Job 均通过，Shader 编译已纳入可复现 CI 验证。
- 新增 `VulkanImageCopyPass`，将双 storage-image dispatch 从 smoke 内联代码提升为可复用运行时 Pass；Pass 固定 `8x8` workgroup、有限 Descriptor 资源，并在提交前验证图像为 `GENERAL` 布局且归属 Compute Queue。
- `tests/test_migration_scaffold.py` 新增图像 Pass 的 workgroup、Descriptor 绑定和布局前置条件测试；定向测试 `14 passed`。
- 新增 `tests/test_vulkan_runtime.py` 验证运行时会话的尺寸校验、提交转发和资源关闭所有权；Vulkan 运行时定向测试共 `17 passed`。
- `VulkanImageCopyPass` 纳入 `stereo_runtime` 公共懒加载导出，后续运行时装配不需要依赖内部模块路径。
- 新增 `app_runtime.VulkanRuntimeSession`，统一持有 Vulkan Context 与图像 Pass；支持外部 Context 注入、内部 Context 生命周期和 ready timeline 透传，暂不接管 Capture/Inference。
- `vulkan_compute_smoke.py` 改为通过 `VulkanRuntimeSession.submit_image_pair()` 执行双 storage-image GPU Dispatch，完成从 app_runtime 到 Compute Graph 的实机链路验证。
- `VulkanRuntimeSession.close()` 现在先执行 `wait_idle()`，再销毁 Compute Pass 和自有 Context，避免 GPU 仍在使用 Pipeline 时发生资源释放竞态；测试锁定关闭顺序。
- `VulkanImageCopyPass` 提交前新增 Context 身份校验，拒绝来自其他 Vulkan Device/Instance 的 storage image；迁移脚手架新增跨 Context 回归测试。
- `compliance.yml` 新增 Vulkan runtime scaffold CI，自动执行迁移脚手架和 `VulkanRuntimeSession` 定向测试。
- 全量回归测试 `423 passed, 4 warnings`；全部 Compute Shader 重新编译并通过 `spirv-val` 校验。
- `VulkanRuntimeSession.resize()` 新增有界 Resize 流程：新尺寸 Pass 创建成功且 GPU idle 后才替换旧 Pass；Resize 失败时保留原运行资源。
- Resize 和生命周期定向测试共 `18 passed`。
- 新增 `VulkanDeviceLostError` 和 Session 健康状态；识别 Device Lost 后记录原始错误并拒绝后续提交，要求上层重建 Session。
- Device Lost、Resize 和运行时生命周期定向测试共 `19 passed`。
- 需求矩阵将 `VK-008` 更新为 `in_progress`，映射 `VulkanRuntimeSession` 和运行时生命周期测试；仍待专用硬件长稳与真实 Device Lost 注入验收。
- `GRAPH-003` 新增 1000 帧 latest-frame 压力测试，确认连续入队后只提交最后一帧，不累积旧帧延迟；Graph/Runtime 定向测试共 `20 passed`。
- GPU smoke 将 storage image 布局转换产生的最大 timeline 作为 `ready_timeline` 传入运行时图像 Pass，完成上游 GPU 完成点到 Compute submit 的实机验证。
- 需求矩阵将 `GRAPH-003` 更新为 `implemented`；latest-frame 覆盖、timeline 透传和实机 Compute 等待链路均已有代码与验证记录，仍需长期压力验收后才能升级为 `verified`。
- `VulkanImageCopyPass` 新增 source/output 图像别名保护，禁止同一 `VkImage` 同时作为只读输入和写入输出；Graph/Runtime 定向测试共 `21 passed`。
- 新增 Resize 失败回滚测试，确认新 Pipeline 创建失败时保留旧 Pass、旧尺寸和旧运行资源；Graph/Runtime 定向测试共 `22 passed`。
- 修复 OpenXR Vulkan Device 创建路径错误使用 `_require_timeline_semaphore_features()` 返回值的问题；现在正确解包 `pNext` Feature 链，并把 `synchronization2_enabled` 传入 adopted Context。
- OpenXR、Graph 和 Runtime 定向测试共 `41 passed`。
- OpenXR Feature 链修复后的全量回归测试 `428 passed, 4 warnings`，`VK-004` 继续保留专用设备创建集成验收状态。
- 新增 `pNext` Feature 链单元测试，验证 Synchronization2 链头、Timeline Semaphore `pNext` 节点和启用标志；OpenXR 定向测试 `20 passed`。
- 修复 `VulkanContext.adopt()` 未接收 `synchronization2_enabled` 参数的问题，避免 OpenXR 真实启动时因 Feature 状态透传触发 `TypeError`；adopt Context 现在记录该能力。
- 新增 `VulkanImageResource` 和 `VulkanExternalImageRegistry`，定义非拥有式外部图像句柄、尺寸、格式、状态和队列归属契约；Vulkan 只登记状态，不销毁 Capture/Inference 资源。
- 将 `ARCH-004` 和 `INFER-002` 更新为 `in_progress`；外部资源契约定向测试与 OpenXR/Graph/Runtime 测试共 `44 passed`，真实 CUDA/ROCm/DMABUF 导入仍待平台适配器。
- 外部资源契约接入后的全量回归测试 `431 passed, 4 warnings`。

- `py -m py_compile src/viewer/vulkan_context.py src/app_runtime/probe.py` 通过。
- 使用项目环境 `src/python3/python.exe -m pytest -q tests/test_openxr_vulkan.py`，18 项通过。
- Graph、SPIR-V loader、Descriptor budget、DescriptorSetLayout 与 Vulkan 定向组合测试：`28 passed`。
- `src/tools/vulkan_compute_smoke.py` 实机通过：Storage Buffer 从 41 更新为 42，`vulkan_compute_smoke: PASS timeline=1 state=ready`。
- Storage Image 实机通过：`storage_image_dispatch: PASS`。
- 实机 Compute 验收通过：Vulkan 1.4.329、`synchronization2_enabled=True`，真实创建 ComputePipeline 并执行 `vkCmdDispatch(1,1,1)`；Timeline value=1，Validation Layer 无 synchronization2 错误。
- `src/tools/probe.py` 实机探针通过：NVIDIA GeForce RTX 2060、Vulkan 1.4.329、Graphics=0、Compute=2、Transfer=1、Timeline Semaphore=true。
- 全量测试结果：404 项通过，1 项因既有 `src/xr_viewer/gltf/materials.py` 的 UTF-8 BOM 导致 AST 解析失败；该文件未由本次改动修改。
- `VK-002` 更新为 `implemented`；`VK-005` 更新为 `in_progress`，FrameContext 已建立，Descriptor、Pipeline 和完整 ImageStateTracker 仍未完成。
- `VK-006` 更新为 `in_progress`，当前已覆盖 layout/access 的清屏转移、状态记录和提交序列号；Queue Ownership 转移及 Validation Layer GPU 验证仍待完成。
- `GRAPH-001` 与 `GRAPH-003` 更新为 `in_progress`；当前仅完成 Graph 调度和 latest-frame 契约，真实 shader/pipeline 及完整处理链仍待接入。
- `GRAPH-002` 更新为 `in_progress`；shader 资源目录和编译入口已建立，待 Vulkan SDK 环境生成 SPIR-V 并完成 GPU dispatch 验收。
- 当前验证覆盖 Python 状态机和 Context 创建；Compute/Transfer 实际 shader 提交、跨队列 semaphore 等待和 Validation Layer GPU 验收仍待完成。
- 当前已验证提交结构和 Python API；真实 Compute Graph pass 尚未接入 `submit_on()`，因此仍需 GPU 实机验证跨队列同步。

### 未决事项

- 当前提交仍使用 `vkQueueSubmit` 和 fence；timeline semaphore、`vkQueueSubmit2`、Descriptor/Pipeline 生命周期和 ImageStateTracker 仍未完成。

### 下一项内容

- 为图像布局、访问掩码和 Queue Ownership 建立可追踪状态，并开始 timeline/submit2 调度迁移。

### 已实现

- 按 `docs/02-desktop2stereo-engineering-design-specification.md` 和 `docs/03-d2s_vulkan_migration_technical_report.md` 复核 Vulkan 主路径：保留 Python OpenXR/Vulkan 生命周期和唯一 Filament Vulkan Bridge 原生边界，继续禁止 D3D11、WGL/CUDA-GL 和 CPU 实时像素回读路径。
- 移除旧 Filament StarGlim 预览特效及其 sidecar、C ABI、Bridge 实现和三平台二进制残留；Artemis 星空改由 `environment.glb` 内嵌天空盒纹理负责。
- 修复 Filament 天空盒遮挡场景的问题：天空盒 renderable 使用背景优先级 `0`，避免遮挡土星环和其他 GLB 几何体。
- 恢复 Filament 桌面预览虚拟屏幕：新增屏幕四边形、窄 C ABI、Python ctypes 更新接口、尺寸/位置/旋转同步和半透明蓝色网格材质。
- 修复 Artemis 预览坐标空间错误：`view_pose` 继续从 profile 世界坐标转换到 GLB 场景坐标；`screen.position` 按当前 profile 约定直接作为 GLB 场景坐标，避免重复减去 `model_position`。
- 屏幕材质恢复旧版蓝色 `16x9` 网格效果，并关闭屏幕自身深度测试，避免被环境深度缓冲隐藏。
- Filament Bridge 通过 GitHub Actions 远程完成 Windows、Linux、macOS 三平台编译，最新二进制已下载回 `src/xr_viewer/native/`。

### 验证结果

- `py -m py_compile src/xr_viewer/preview_room_layout.py src/xr_viewer/filament_preview_bridge.py` 通过。
- `git diff --check` 通过。
- Filament Bridge CI runs `29723126977`、`29724387172`、`29725167239` 和 `29726120882` 的三平台构建通过。
- 当前预览桌面窗口仍按 `preview_room_layout.py` 的 `1280x720` 初始化；该尺寸只代表桌面预览，不代表 OpenXR 头显交换链分辨率。

### 未决事项

- `src/app_runtime/bootstrap.py --runtime` 仍未完成正式运行时装配，当前打印 `runtime is not assembled yet`。
- `src/stereo_runtime/vulkan_graph.py` 仍是提交契约骨架，尚未建立 Compute Pass、固定资源池、shader manifest 和 GPU 同步闭环。
- `VulkanContext` 已能选择并暴露 Graphics/Compute/Transfer 队列族，但仍以一次性 Command Buffer/Fence 提交为主，尚未达到规范要求的 FrameContext 池和完整 ImageStateTracker。
- Vulkan/OpenXR 清屏 smoke 已验证；Filament 场景 Bridge 的头显视觉验收、虚拟屏幕纹理采样和 Compute Graph 仍需继续打通。

### 下一项内容

下一项按 Phase 1/Phase 3 交界推进：先把 Vulkan Context 的 Graphics/Compute/Transfer 队列和有界 FrameContext/Synchronization 契约补齐，再接入最小可执行 Compute Graph，并同步更新需求矩阵和测试。

## 2026-07-19

### 已实现

- 将颜色调节选项暂时全部放入现有“高级立体参数”区域，不新增主界面分组；新增曝光、对比度、饱和度、Gamma、色温和色调六项控制。
- 将`src/main.py`默认入口接入Flet GUI，保留`--probe`能力探针入口；启动新项目不再停留在迁移脚手架提示。
- 补齐GUI颜色控件的运行时快照回写，热更新后曝光、对比度、饱和度、Gamma、色温和色调会同步显示当前生效值。
- 增加新Schema到旧GUI平面配置的启动兼容层：GUI启动时从`graphics/capture/inference/stereo/openxr/output`读取迁移配置，并补齐GUI和运行时所需默认字段，不直接覆盖原始嵌套配置。
- 修复兼容层`Model List`类型错误：按旧项目格式提供每个模型的`resolutions`对象，解决Flet启动时`'str' object has no attribute 'get'`。
- 颜色调节统一放在深度推理完成之后、立体合成和输出分发之前，因此本地预览、网络推流和 OpenXR 使用同一套颜色结果，且不改变 AI 深度输入。
- 新增颜色参数的配置保存、加载、GUI 热更新和运行时快照字段；调整颜色参数不触发模型、Filament 或 OpenXR 管线重建。
- 色温和色调采用相对值：范围均为`-100..100`，默认`0`；色温负值偏冷、正值偏暖，色调负值偏绿、正值偏洋红。

- 将`src/xr_viewer/preview_room_layout.py`的场景加载和逐帧渲染全面切换到Filament Desktop Preview Bridge。
- 新增`FilamentDesktopPreview` ctypes封装，通过Filament AssetLoader/ResourceLoader加载profile对应GLB，使用Filament Scene、Camera、View和Renderer提交桌面窗口帧。
- 桌面窗口使用GLFW原生句柄创建Filament SwapChain，支持Windows、Linux和macOS平台句柄，并同步窗口尺寸变化到Filament viewport。
- 删除预览入口中遗留的ModernGL shader、手写GLB解析、OpenGL资源上传和旧渲染辅助代码。
- Filament Bridge新增桌面预览生命周期、GLB加载、相机、投影、viewport和render C ABI；三平台产物自动回写`src/xr_viewer/native/`。
- 修复桌面预览profile座位偏高：profile中的座位保持世界坐标，加载Filament GLB前使用模型变换逆矩阵转换到场景坐标，保存时再转换回世界坐标；Artemis `y=901.0986`正确转换为GLB场景`y=58.0132`。
- 排查Artemis预览显存风险：旧项目原分辨率稳定约2.65 GB，新Filament原分辨率60秒稳定约2.64 GB，末端一次性上传完成后约2.93 GB，未发现逐帧增长或显存泄漏；Filament Bridge现在在GLB上传后执行`flushAndWait()`并释放GLB源数据，预览循环限制为60 FPS。
- 保留原分辨率为默认行为，新增可选的`--max-texture-size 4096`内存保护模式；该模式只重建内存中的预览GLB，不修改原始资源文件。
- 固化桌面预览Bridge调参ABI：新增曝光和方向补光的C ABI，Python通过profile或`--exposure`、`--fill-light-intensity`调整颜色，不再为亮度和灯光参数修改反复编译Filament Bridge。
- 修复Filament预览发黑：GLB纹理继续交给Filament按glTF sRGB/线性规则处理，View增加曝光色彩分级，并提供线性颜色方向补光；三平台Bridge构建通过。
- 将桌面预览默认曝光调整为`2.0 EV`；命令行显式`--exposure`和profile中的`preview_exposure`仍可覆盖默认值。
- 将天空盒与座位主体亮度解耦：Filament补光仅使用独立光照通道照亮非天空盒实体，天空盒材质通过独立的`skybox_brightness`乘数调节。
- 新增`--skybox-brightness`和profile字段`preview_skybox_brightness`；预览窗口使用`,`/`.`独立降低或提高天空盒亮度，`[`/`]`继续只调节座位主体曝光。
- 完成单View独立亮度方案：删除桌面预览全局ColorGrading，按GLB加载时保存的原始`baseColorFactor`分别缩放座位主体和天空盒材质。
- OpenXR Filament Bridge新增同一套`scene_exposure`与`skybox_brightness` ABI；每眼仍只提交一个View，不增加双View渲染开销。

### 验证结果

- 颜色相关 Python 文件 `py_compile` 通过，`git diff --check` 通过。
- `tests/test_settings_snapshot.py` 和 `tests/test_hot_reload.py` 共 31 项通过。
- `src/main.py --probe`通过；`gui.gui`模块成功导入，原先因`Stream Quality`缺失导致的启动异常已消除。
- Flet桌面客户端包已补齐；可直接运行`src\python3\python.exe src\main.py`启动GUI。
- 已使用`src/gui/flet_packages/flet-windows.zip`成功解压并启动GUI，`gui_ready.flag`已生成，Flet窗口初始化完成。

- Python `py_compile`和`git diff --check`通过。
- GitHub Actions run `29654473319`和`29654653736`的Windows、Linux、macOS构建全部通过。
- Windows DLL已确认导出`filament_preview_create`、`filament_preview_load_glb`、`filament_preview_set_viewport`和`filament_preview_render`。
- Artemis桌面预览进程可正常启动并持续运行，GLB资源加载无Python异常；日志仅有源图片的libpng iCCP警告。
- 代码提交：`7c38fbd`、`fee0eee`；原生二进制提交：`b06bad0`、`d905408`。

### 未决事项

- 需要用户确认Filament桌面窗口中的房间画面、profile座位高度和场景完整性。
- 尚未进行桌面预览与头显Projection Layer的最终视觉一致性对比。
- 单View材质亮度方案等待三平台Bridge构建及桌面/头显画面实测确认。

### 下一项内容

下一项：完成三平台Bridge构建，先验证桌面预览独立亮度，再进行头显双眼场景实测。

## 2026-07-18

### 已实现

- 建立独立项目`desktop2stereo-vulkan`，保持原项目的Python源码组织方式，不在运行时依赖原仓库。
- 迁移可复用的Capture、AI推理、Stereo、GUI、OpenXR平台无关模块、Samples、测试和工具；原项目文件保持不变。
- 迁移`native/filament`及Windows、Linux、macOS多平台GitHub Actions构建流程，统一产物目录为`src/xr_viewer/native/`。
- 确立Vulkan为主图形路径、OpenGL为隔离Fallback，不迁入旧Panda3D、D3D11 OpenXR、WGL/CUDA-GL Bridge和旧OpenGL上传链路。
- 实现Python Vulkan基础层，包括Instance、物理设备选择、Device、Graphics Queue、Command Pool、Command Buffer、Fence、图像布局转换、清屏提交和资源释放。
- 实现基于`XR_KHR_vulkan_enable2`的Python OpenXR Vulkan Phase 1，包括运行时选定物理设备、Session、双眼交换链、Projection Layer、事件处理和纯色帧提交。
- 新增`src/tools/openxr_vulkan_smoke.py`，用于头显环境下独立验证双眼Vulkan交换链。
- 更新`src/requirements.txt`，明确`pyopenxr==1.1.5301`和`vulkan==1.3.275.1`为Vulkan/OpenXR主路径依赖，PyOpenGL归入Fallback依赖。
- 修正pyopenxr Composition Layer提交方式，使用`ctypes.pointer(layer)`满足`FrameEndInfo.layers`的Base Header指针约定。
- 更新能力探针、README和迁移清单，使Phase 1状态与头显实测结果一致。

### 验证结果

- Filament Vulkan Bridge的Windows、Linux和macOS GitHub Actions构建已通过（run `29650016647`）。
- 新增手动发布工作流，可从成功的三平台 CI 运行中下载 DLL、so、dylib，打包为 GitHub Release 资产并生成 SHA-256 校验文件。
- 更新 Filament Bridge CI：三平台构建完成后自动将 DLL、so、dylib 下载到`src/xr_viewer/native/`并提交到`main`，不再只保留为临时 Actions artifact。
- 本机Vulkan探针识别到NVIDIA GeForce RTX 3090、Vulkan 1.4.341和Graphics Queue Family 0。
- Virtual Desktop OpenXR Runtime可加载，并声明支持`XR_KHR_vulkan_enable2`。
- Vulkan/OpenXR新增及迁移状态测试共13项通过。
- 最终全量测试394项全部通过；期间既有Hugging Face Provider测试曾因外部站点SSL EOF短暂失败，网络恢复后复测通过。
- `py_compile`和`git diff --check`通过。
- 未连接头显时，Smoke入口按设计返回`FormFactorUnavailableError`并完成资源清理。
- 连接头显后成功创建双眼3648x3648 Vulkan交换链，并完成300/300帧提交。
- 用户确认头显内稳定显示深蓝色双眼画面，无OpenXR调用顺序、Vulkan同步或资源释放错误。
- 开始实现Filament Vulkan Render Target Bridge：新增VulkanSharedContext接入、OpenXR VkImage外部SwapChain、Python ctypes封装和跨平台构建配置。
- Bridge明确借用Python/OpenXR所有Vulkan对象，不创建或销毁OpenXR资源；结束帧前使用Filament `flushAndWait`完成GPU同步。
- 将Bridge以显式配置方式接入`OpenXrVulkanPresenter`：左右眼分别绑定外部OpenXR VkImage，帧内传递acquire index，关闭顺序先Bridge后OpenXR交换链；未配置Bridge时保持原有Vulkan清屏路径。
- 在Filament Bridge内建立`Scene`、`Camera`和`View`，加载GLB后将实体加入场景，并在每帧调用`Renderer::render`；三平台CI重新编译通过，最新二进制已自动回写`src/xr_viewer/native/`。
- 增加每眼OpenXR Camera同步：Python根据View pose计算look-at参数，根据View FOV计算垂直视场角和aspect，并通过C ABI更新Filament Camera；三平台新Bridge构建和19项聚焦测试通过。
- 扩展`openxr_vulkan_smoke.py`支持显式指定`--filament-bridge`和`--filament-glb`，默认仍保持纯Vulkan清屏模式；README补充Bedroom环境GLB的Filament头显测试命令。
- 修复Filament 1.74外部SwapChain的`FixedCapacityVector`容量初始化，并保存平台层ExternalSwapChain句柄；Windows RTX 3090头显实测无GLB 60/60帧、QUEST控制器GLB 120/120帧、Artemis `environment3.glb` 120/120帧通过。
- 修复Presenter在Filament渲染完成后仍调用Python `clear_color_image`的问题；该清屏操作会覆盖Filament场景，导致帧提交成功但头显只显示深蓝色。Bridge启用时现在跳过Python清屏，Artemis场景再次完成120/120帧提交。
- 重新运行Artemis Filament头显测试：RTX 3090、双眼`3648x3648`交换链、300/300帧提交成功，进程正常退出。
- 新增Filament profile视角加载：读取`view_pose_index`选中的`view_poses`，将初始头部位姿映射到profile座位，同时保留运行时头部移动、双眼间距和Projection Layer位姿一致性。
- `openxr_vulkan_smoke.py`新增`--filament-profile`/`--profile`和`--seconds`参数，支持按profile视角进行长时间头显观察。
- Artemis `Model Center` profile长测通过：RTX 3090、双眼`3648x3648`交换链、120秒、8548帧提交成功，进程正常退出。
- 修复profile视角黑屏：`environment3.glb`已包含部分模型变换，而profile座位仍使用旧世界坐标；加载时按`model_position`将座位转换为GLB坐标。修正后20秒头显实测提交`1434`帧正常。
- 按原项目实际实现修正profile座位：含`x/y/z`的`view_pose`直接作为座位位置，`rotation_deg`或`angle`直接作为相机朝向；`screen`仅用于屏幕布局，不参与初始profile相机定位。
- 回退错误的屏幕相对座位变换后，Artemis原始`environment.glb`进行10秒头显实测，提交`717`帧正常。
- Filament Bridge新增非对称相机frustum ABI，Python按每眼OpenXR的left/right/up/down切角设置投影；profile的`xr_projection_near/far`也会传递到Filament，Artemis使用`0.1/20000.0`避免大场景裁剪。
- 实现桌面房间布局预览：`preview_room_layout.py`可加载profile对应GLB，显示环境模型和虚拟屏幕，并支持SCREEN/VIEW编辑、鼠标视角、座位移动、屏幕预设、裁剪范围和profile保存；补齐独立项目缺少的ModernGL glTF解析包及OpenGL状态辅助模块。
- 确认Artemis profile对应的原始`environment.glb`已可被当前Bridge加载；使用该匹配资源进行30秒头显实测，RTX 3090双眼`3648x3648`交换链提交`2117`帧正常。`environment3.glb`不再作为Artemis profile的默认测试资源。
- 将颜色曝光、对比度、饱和度、Gamma、色温和色调控件从高级立体参数移动到捕捉设备的高级设置中，并由“高级设备选项”统一控制显示。
- 将颜色控件显示名称调整为“亮度、对比度、饱和度、Gamma”；颜色行继续使用与上方参数一致的标签宽度、下拉框宽度和列间距。
- 按照运行模式选项框的尺寸，将颜色选项框统一调整为 `130`，并保持左侧选项列对齐。
- 修正颜色标签未参与全局标签列宽计算的问题，使亮度等颜色选项框与运行模式选项框使用同一左侧列基准。
- 将亮度从曝光补偿改为亮度倍率：`1.0` 为中性值，运行时直接乘以倍率；配置、热更新和运行时字段统一改为 `Color Brightness` / `color_brightness`。
- 将亮度倍率上限从 `4.0` 调整为 `2.0`，选项范围为 `0.2 - 2.0`。
- Artemis 房间预览接入 Filament 桌面预览动画 ABI，每帧播放 GLB 内嵌的 16 条卫星轨道动画和 3 条飞船轨道动画；按 `R` 重新加载 profile 时动画时间同步重置。
- Artemis 预览接入 `star_glim.json`：加载 stars/mask PNG，创建 Filament Vulkan 加法叠加材质，并按 sidecar 的密度、速度、软阈值和强度参数驱动星点闪烁。
- 重写 StarGlim 窄接口：仅保留动态材质创建、stars/mask 纹理、`intensity/speed/seed` 参数和时间更新四类 C ABI；删除旧的 `shine_speed/cell_*` 参数链。
- 预览每帧只计算一次 `animation_time`，同时传给 GLB 卫星动画和 StarGlim shader，确保两者使用同一时间轴。
- 将 StarGlim 动态材质创建放入 GLB 加载后的 Filament 场景初始化阶段；Python 语法检查和 JSON 校验通过。

### 未决事项

- CodeGraph数据库被当前MCP进程占用，本轮无法重建索引；代码和测试不受影响。
- 既有Hugging Face Provider测试依赖外部站点可达性，需要后续消除测试对网络状态的依赖。
- Filament Bridge的真实场景渲染尚未验证；当前Python封装只覆盖Bridge ABI和生命周期，不接管OpenXR acquire/release。
- Artemis和QUEST GLB已完成头显帧提交实测，等待用户确认头显内实际模型画面；FOV同步使用对称等效投影，OpenXR非对称左右/上下切偏移仍需使用自定义投影矩阵精确处理。
- Bedroom `environment.glb` 在Filament `load_glb`阶段解析失败，文件头和GLB声明长度一致，需后续用glTF Validator定位其扩展或资源兼容性问题。

### 下一项内容

下一项：提交源码并由 GitHub Actions 三平台重编译 Filament Bridge，下载新二进制后再测试 Artemis 星空与卫星动画同步效果。
## 2026-07-21

- Unified Preview and OpenXR Filament color processing in the shared native Bridge: both Views now explicitly use ACES legacy tone mapping, Rec709/sRGB/D65 output color space, and enabled post-processing.
- Kept scene exposure, skybox brightness, and directional fill-light values profile-driven and shared by Preview/OpenXR; the new CI Bridge binary is required before headset comparison.

- Added configurable OpenXR swapchain color mode: `sRGB`, `UNORM`, or `Auto`; the selected Vulkan format is logged for headset A/B validation.
- Added focused coverage for sRGB versus UNORM selection and invalid mode rejection. The default remains `sRGB`.

- Reused the legacy controller semantics in the Vulkan path: all complete brand folders under `src/xr_viewer/controllers/` are discovered, while the selected brand remains controlled by `D2S_CONTROLLER_MODEL`.
- Added narrow Filament Bridge controller ABI for left/right GLB loading, grip-root pose updates, trigger/grip/stick values, and button bitmasks.
- Implemented Filament-side `_value/_min/_max` node animation using the existing controller naming convention; no replacement renderer or new controller asset format was introduced.
- Connected the copied OpenXR action bindings and grip pose locator to the Vulkan presenter so controller input and model animation use the same frame loop.
- Python checks and focused OpenXR/Bridge tests pass: `26 passed`; the new Bridge ABI still requires the GitHub Actions three-platform rebuild before headset validation.

### 验证结果

- `src/python3/python.exe -m py_compile` passed for the new controller modules and OpenXR presenter.
- `src/python3/python.exe -m pytest -q tests/test_openxr_vulkan.py tests/test_filament_vulkan_bridge.py`: `26 passed`.
- `git diff --check` passed.

### 未决事项

- The controller ABI source is complete, but the checked-in native binaries do not contain these exports until the next GitHub Actions build.
- Headset acceptance still needs to confirm model placement and real PICO input/button animation.

### 下一项内容

- Commit and push the controller ABI/source changes, then download the three-platform CI Bridge artifacts and run the OpenXR headset test.

- OpenXR runtime now resolves the packaged platform Filament Bridge, Artemis GLB, and profile automatically; manual `D2S_FILAMENT_*` environment variables are no longer required for the Windows headset test.
- Set the default test configuration to `OpenXR Link` with scene exposure `2.0` and skybox brightness `1.0`.
- Fixed the OpenXR Vulkan device setup to stop calling the enable1-only `xrGetVulkanDeviceExtensionsKHR` while using `XR_KHR_vulkan_enable2`.
- OpenXR Artemis lighting now reads the same exposure, skybox, and directional fill-light profile values as the desktop preview; the updated Bridge binary must be rebuilt by CI.
- GitHub Actions run `29766759073` successfully rebuilt and committed Windows x86_64, Linux x86_64, and macOS arm64 Filament Bridge binaries; all three binaries were synchronized locally.
- The next validation is headset A/B comparison of Preview and OpenXR brightness, tone mapping, and sRGB/UNORM output.
- Fixed OpenXR startup failure caused by calling the obsolete `_initialize_controller_actions`; Presenter now calls the existing `_init_controller_actions` Mixin method.
- Fixed profile loading variable reuse: GLB camera position and virtual screen position now use separate variables, preventing `.tolist()` startup failure and preserving the profile camera pose.
- Fixed OpenXR Filament output setup: sRGB swapchains now pass `CONFIG_SRGB_COLORSPACE`; each frame now advances GLB animations on one shared timeline.
- Controller pose updates now fall back from grip pose to aim pose, and startup logs report controller brand, screen dimensions, and loaded Bridge state.
## 2026-07-21

- 修复 OpenXR Quad Layer 颜色路径：运行时 `uint8` 输出是已经编码的显示用 sRGB 字节，CUDA 导出图像和 Quad Layer 统一使用 `R8G8B8A8_SRGB`，同格式路径使用 `vkCmdCopyImage` 原样复制，避免 `UNORM -> SRGB` Blit 再次编码导致画面发白。
- Filament 虚拟屏幕继续以 `SRGB8_A8` 采样，并只在采样边界解码一次；CUDA 互操作只处理 RGBA 通道布局，不执行颜色转换。
- 修复 OpenXR Quad Layer 方向适配：保持输出契约 `image_origin=top_left`，仅在 Quad Layer 提交边界执行 Y 适配，不再进行 X 翻转。
- 修复环境 `profile.json` 相机高度：恢复旧工程的 `model_position/model_rotation_deg/model_scale` 逆变换，将世界坐标 `view_poses` 转为 GLB 局部坐标后再校准 OpenXR reference space。
- 修复控制器 profile 姿态：`model_rotation_deg` 按旧工程约定绕控制器模型局部 X 轴应用。
- 修复 Quad Layer 屏幕姿态：profile 的 `[yaw, pitch, roll]` 现在按旧工程的 Y/X/Z 旋转顺序转换为 OpenXR 四元数，不再把 yaw 错误当成 X 轴旋转。
- 规范化预览运行时和保存的姿态角：view 和 screen 的旋转始终保持在 `[-180°, 180°)`，避免连续旋转后出现 `902°` 等等价但难以阅读的角度。
- 修复 Projection Layer 虚拟屏幕无立体输入：每只眼睛的 Filament screen material 现在绑定对应的运行时 Vulkan eye image，避免屏幕纹理未接入或左右眼复用同一张图像。
- 对齐旧工程 Projection Layer 屏幕路径：屏幕仍作为场景几何体参与每眼投影渲染，纹理按 Vulkan image handle 缓存复用，不改为单张 2D 合成层。
- 按旧工程的异步提交边界优化 Projection Layer：左右眼 `end_frame` 只提交 Filament 工作，整帧两眼完成后统一等待一次，避免每眼一次 `flushAndWait` 串行阻塞；旧 Bridge 二进制仍保留兼容路径，需 CI 重编译后生效。
- 增加 CUDA/Vulkan/Filament external semaphore 路径：每个输出槽位创建可导出的 Vulkan binary semaphore，CUDA copy 完成后异步 signal，Filament Bridge 在目标 swapchain acquire 时等待对应 semaphore；平台或运行库不支持时自动退回 CUDA stream 同步。
## Unreleased

- 修正 OpenVINO RemoteTensor 缺失运行时测试对已安装 Intel 原生 DLL 的环境依赖，确保测试显式验证缺失桥接器契约。
- 修正 Intel Desktop Duplication 能力探针：区分“借用 D3D11 纹理可供推理”和“捕捉到推理端到端零拷贝”，当前 staging readback 路径不再误报 `zero_copy_ready`。
- 新增 Intel Windows 原生硬件烟测工具，可验证 Desktop Duplication、OpenVINO D3D11、VideoProcessor、oneVPL 的同适配器 LUID 和按帧生命周期，并支持 4K 长时间运行。
- 增强 Intel OpenVINO 运行时指标，分别记录 GPU 纹理输入和 CPU 深度输出，避免将输入零拷贝误读为完整推理零拷贝。
- 调整 Windows 自动捕捉顺序：无 CUDA/ROCm 时优先选择 `DesktopDuplication`，由其在 DXGI/native bridge 不可用时回退到 DXCamera。
- 修正 GUI 默认捕捉设备检测绕过懒加载的问题，确保 CUDA 设备仍保留 `WindowsCaptureCUDA`，Intel/CPU 才进入 `DesktopDuplication` 默认路径。
- 完善 Desktop Duplication 回退链：native bridge 不可用时先尝试 `WindowsCapture`，其不可用再回退到 DXCamera。

- Vulkan FFmpeg bridge ABI 升级到 v2：原生桥现在创建真实的 FFmpeg Vulkan NV12 frame pool，提供 GPU `VkImage`/memory 描述、提交、取包和 flush 接口；应用仍保持稳定高级网络推流回退，待 Python frame-pool 消费、GPU RGB→NV12 和同步信号接入后再启用。
- 远程 Vulkan bridge CI 的 ABI 校验扩展到 frame acquire、submit、packet read 和 flush 符号，避免新接口未导出却被误判为构建成功。

- Automatic stream calibration now uses an inference-independent 30 FPS CBR pressure stream sized from the selected input resolution. Each coarse tier runs for 15 seconds, the stable/unstable interval is narrowed by binary search to 1 Mbps, and the final candidate is confirmed for 30 seconds.
- A tier is valid only when its measured sender rate reaches at least 85% of the requested load. Results record the measured network limit and apply 80%/90% safety margins for the target/peak bitrates instead of presenting a formula-derived target as measured capacity.
- Repacked the MSDF atlas into fixed 64x64 cells in the charset order, with
  deterministic left-to-right and top-to-bottom pages.
- Improved native MSDF coverage calculation for small VR OSD glyphs and forced
  linear atlas filtering with edge clamping.
### Unreleased

- Fixed controller surfaces being clipped by the room depth buffer after the
  room/controller lighting split. The foreground pass now clears only depth,
  preserving the room color while keeping opaque controller geometry intact.

### Filament v1.75.0 bridge compatibility

- Narrowed the downloaded BlueVK depth-clamp declaration guard for older Vulkan headers.
## Unreleased

- 修复 Vulkan FFmpeg bridge 的 MinGW DLL 前缀与 GitHub Actions 产物校验不一致问题。
## 2026-08-24

- 统一 GitHub Actions 的 `download-artifact` 到 `v7`，消除遗留 Node.js 20 action 在 Node.js 24 runner 上的弃用警告。
- 同步将 Filament workflow 的 `actions/cache` 升级到 `v5`；该版本使用 Node.js 24。
- 更新网络推流规格书中的 Intel 部分：标记 Vulkan packed SBS → D3D11 shared BGRA → VideoProcessor NV12 → oneVPL/QSV 原生路径已完成，并补充 LUID、同步、回退和深度输出边界。
