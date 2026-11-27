# VDXR / Quest 2 OpenGL 与 Vulkan 性能对比及立体 Quad 可行性报告

**文档状态**：实测报告

**测试日期**：2026-11-24

**测试设备**：AMD Radeon RX 9060 XT、Meta Quest 2、Virtual Desktop / VDXR

**代码边界**：本轮不修改 Desktop2Stereo 源码，不修改用户 `settings.yaml`、Guardian、Proximity 或面部检测状态

**关联计划**：[08 Vulkan 流水优化计划](08-openxr-vulkan-pipeline-performance-optimization-plan.md)、[11 Quad 设置菜单计划](11-openxr-quad-settings-menu-implementation-plan.md)

## 1. 结论摘要

**合成呈现率不差。**

在当前 VDXR 会话里，OpenGL 与 Vulkan 的合成呈现率都稳定在约 72 Hz。OpenGL 稳定阶段 11 个 5 秒窗口加权为 `71.84 Hz`；Vulkan/PyTorch 的 13 个窗口均值为 `71.50 Hz`、中位数为 `71.80 Hz`；Vulkan/MIGraphX 的 13 个窗口均值为 `71.35 Hz`、中位数为 `71.80 Hz`。Vulkan 均值受首个新内容窗口的 `67.50/65.80 Hz` 拉低，其余窗口约为 71.6–72.0 Hz。三条路径都紧跟 VDXR 的 72 Hz 刷新节拍，未出现持续性 36 Hz 锁帧。

**问题不在 Presenter。**

“Vulkan 帧率差”目前不能归因于 Vulkan Presenter 或 `xrEndFrame`。Vulkan 的两轮日志均显示 `xr_path=vulkan_composer`、`vk_composer_active=1`、`xr_should≈xr_input≈xr_present≈72 Hz`、`xr_no_render=0`。投影合成的 CPU 耗时 p95 只有 `2.64 ms`（PyTorch）或 `3.11 ms`（MIGraphX）。

**差异来自新内容生产。**

相同 Vulkan 合成路径下，PyTorch/ROCm 的 `raw/runtime/screen_new` 约 `41.8 / 41.8 / 39.0 FPS`；启用 MIGraphX 后约 `51.4 / 51.4 / 48.6 FPS`。PyTorch 的 `frame_exec` 约 `17.2 ms`、`capture_ready` 约 `29.1 ms`；MIGraphX 对应约 `3.6 ms`、`10.7 ms`。因此当前最值得优化的是捕获到推理、深度上采样和立体合成这段上游链路。

**OpenGL 的新内容吞吐仍待同口径证明。**

OpenGL 轮次确实持续 72 Hz 呈现，但其 `fresh_input` 只来自临时注入的 `GLMeasure` 采样器，仓库中没有该计数器定义。稳定窗口累计约 `2.7 次/秒`，与 Vulkan 的 `screen_new≈39–49 FPS` 不是同口径，不能据此得出“OpenGL 新内容也比 Vulkan 快”。要回答这一点，必须先让两条路径输出同一组 capture/runtime/screen-new 指标。

**VDXR 左右立体 Quad 已实机通过。**

旧路径测试时同一 Quad 区域在左右眼显示了不同的画面内容，用户已确认“Quad 左右不同画面测试通过”。这与“菜单使用单个 `eyeVisibility=BOTH` Quad”是两个不同需求：菜单仍可用双眼共享的单个 Quad；需要每眼独立内容时，应实现独立的左/右 Quad 内容路径。

**`layer_count=0` 是遥测异常。**

同一 Vulkan 轮次中 `vk_composer_frame` 持续增长、`xr_present_fps≈72`、用户确认头显有画面，且截图显示左右眼内容不同。该字段与实际提交/显示结果冲突，需在后续遥测中单独排查；在此之前不应把 `layer_count=0` 当作黑屏或空帧证据。

## 2. 测试环境与口径

| 项目 | 配置 |
|---|---|
| GPU | AMD Radeon RX 9060 XT |
| 头显 / 运行时 | Meta Quest 2 / Virtual Desktop OpenXR（VDXR） |
| Python 环境 | `F:\desktop2stereo-vulkan\src\python3\python.exe`（ROCm 7） |
| 捕获 | `WindowsCaptureROCm`，Monitor 1，桌面 `3840×2160` |
| 模型 | `Distill-Any-Depth-Small`，深度分辨率 336，FP16 |
| 立体设置 | `quality_4k`、Half-SBS、Render Scale 100%、XR Render 1.0、深度强度 0.48、收敛 0.5 |
| 隔离覆盖 | `D2S_CONTROLLER_MODEL=none`、`D2S_OPENXR_DISABLE_TOOL_QUADS=1`、`D2S_OPENXR_GLOW_MODE=off` |
| 测试时长 | 每轮 70 秒；首次运行包含推理、立体 kernel、OpenXR 初始化，性能统计剔除启动后的前 20 秒 |
| 采集 | `D2S_FPS_BREAKDOWN=1`、`D2S_FPS_BREAKDOWN_INTERVAL=5`、`D2S_FPS_BREAKDOWN_LIMIT=0` |

OpenGL 轮次使用旧 OpenXR 路径（日志明确为 `XrInstance created (OpenGL)`、`XrSession created (OpenGL)`），但通过新版 ROCm Python 环境运行，因此依赖与驱动环境接近；D2S 新 Vulkan Presenter 使用了不同的图形 API 与合成实现。该对比是“实际产品路径对比”，不是只替换 OpenXR graphics binding 的严格微基准。

所有测试都使用同一台机器、同一张桌面捕获、同一模型与深度分辨率。OpenGL 日志中的 `Using CUDA device: AMD Radeon RX 9060 XT` 来自 ROCm 版 PyTorch 沿用的 `cuda` 设备命名，不代表启用了 NVIDIA CUDA 路径。

## 3. 实测数据

### 3.1 合成呈现率

| 路径 | 统计窗口 | 呈现 FPS 均值 | 呈现 FPS 中位数 | 范围 |
|---|---:|---:|---:|---:|
| OpenGL / PyTorch-ROCm | 11 × 5 s | 71.84 | 71.80 | 71.60–72.00 |
| Vulkan / PyTorch-ROCm | 13 × 5 s | 71.50 | 71.80 | 67.50–72.00 |
| Vulkan / MIGraphX | 13 × 5 s | 71.35 | 71.80 | 65.80–72.00 |

OpenGL 的最终汇总为 `70.1 s / 4131` 个 `xrEndFrame`，但这包含启动阶段的空帧；不能用 `4131 ÷ 70.1 = 58.9` 代表稳定阶段刷新率。其稳定窗口的 `end_fps` 与 `nonempty_fps` 均为 `71.6–72.0`。Vulkan 的 `xr_present_fps` 是窗口速率，因此启动窗口单列，不参与稳定统计。

### 3.2 Vulkan 上游吞吐与耗时

Vulkan 稳定窗口的字段含义如下：`cap/raw/runtime` 分别是捕获、原始取帧和运行管线完成速率；`screen_new` 是提交到屏幕合成的新内容速率；`frame_exec` 是应用帧工作耗时；`capture_ready` 是捕获到输出可用的端到端指标；`xr_projection_total` 是当前 Presenter 内投影合成耗时。

| 指标 | Vulkan / PyTorch-ROCm | Vulkan / MIGraphX | 观察 |
|---|---:|---:|---|
| `cap` FPS | 48.96 | 51.48 | MIGraphX 轮次捕获节拍更高 |
| `raw` / `runtime` FPS | 41.80 / 41.80 | 51.41 / 51.41 | PyTorch 是当前主要吞吐限制 |
| `screen_new` FPS | 38.95 | 48.64 | 新内容到达屏幕的真实速率 |
| `screen_reuse` FPS | 30.01 | 22.55 | 合成器重放上一张有效 GPU 帧 |
| `overwrite` / 5 s | 7.20 | 0.08 | PyTorch 有更多捕获帧被最新帧覆盖 |
| `rt_total_p50` | 16.61 ms | 2.35 ms | PyTorch 推理链明显更重 |
| `rt_total_p95` | 21.57 ms | 7.46 ms | 尾延迟差距更大 |
| `rt_model` | 8.64 ms | 0.99 ms | 深度模型推理差值 |
| `frame_exec` | 17.20 ms | 3.63 ms | 应用处理帧工作量 |
| `capture_ready` | 29.08 ms | 10.74 ms | 捕获至可提交输出 |
| `xr_projection_total` | 1.74 ms | 1.82 ms | Vulkan 合成本身几乎相同 |
| `xr_projection_p95` | 2.64 ms | 3.11 ms | 仍远低于 13.9 ms 的 72 Hz 帧预算 |

表中 p95 是 13 个 5 秒窗口各自所报 p95 的均值，不是把所有原始帧合并后的单一全局分位数；报告中因此只把它用于阶段级瓶颈判断，不用于尾部延迟承诺。

这些数字说明：在当前配置下，Vulkan 合成器已经能每 13.9 ms 刷新周期稳定提交；只有约 39–49 FPS 的“新内容”进入它，其余刷新周期复用上一张 GPU 有效帧。用户看到的流畅度差异更可能来自新内容速率、帧间节奏和深度/立体处理尾延迟，而不是 `xrEndFrame` 本身。

### 3.3 OpenGL 数据限制

OpenGL 轮次只有以下有效指标：

- `wait_fps = should_render_fps = end_fps = nonempty_fps ≈ 71.8–72.0`；
- `layers_per_end = 1.00`；
- `final nonempty = 4131`；
- `fresh_input` 累计 152 次，稳定阶段约 147 次。

该轮次没有输出 D2S Vulkan 的 `screen_new`、`capture_ready`、`frame_exec_p95`、`rt_total_p95` 字段；`fresh_input` 的临时采样逻辑是测试时注入、当前仓库中没有定义。因此不能将 `fresh_input` 等同于 Vulkan 的 `screen_new`，也不能用它推断 OpenGL 的推理吞吐。

同时，OpenGL 日志出现两次 MIOpen `Invalid elapsed time`，发生在启动/卷积求解阶段；其后呈现仍稳定到 72 Hz。该警告应记为 OpenGL 路径的启动噪声，不应作为 Vulkan 性能问题的证据。

### 3.4 视觉与合成验证

- Vulkan 头显截图：[`quest-vulkan-20261124-visible-during/screen.png`](../runtime-diagnostics/quest-vulkan-20261124-visible-during/screen.png)。
- OpenGL 头显截图：[`quest-gl-openxr-20261124/screen.png`](../runtime-diagnostics/quest-gl-openxr-20261124/screen.png)。
- 截图显示 Quest 2 双眼画面可见；Vulkan 截图中左右眼视角/内容存在差异，符合立体输出预期。
- Vulkan 日志 `layer_count=0.0` 与 `vk_composer_active=1`、`vk_composer_frame` 递增、`xr_present_fps≈72` 及可见截图不一致。该字段亟需校正，当前只能记为“未解释的遥测异常”。

## 4. 瓶颈判断

### 4.1 已排除的瓶颈

- **不是 VDXR 的刷新上限问题**：两条路径都能到约 72 Hz。
- **不是 Vulkan Projection/Composer 的提交路径瓶颈**：`xr_projection_total` 约 1.7–1.8 ms，p95 约 2.6–3.1 ms。
- **不是 Quad 开销主导**：在 `D2S_OPENXR_DISABLE_TOOL_QUADS=1` 的隔离测试中，`quad_total≈0.6 ms`，不能解释上游 17–29 ms 的 PyTorch 帧耗时。
- **不是缺少有效 GPU 输出**：`vk_composer_fallback=0`、`screen_new` 非零、`screen_reuse` 持续存在。

### 4.2 高概率瓶颈

按当前日志证据排序：

1. **PyTorch/ROCm 深度推理与预处理**：`rt_model` 与 `rt_pre` 合计构成 `rt_total_p50` 的主要部分，且 `rt_total_p95` 明显高于中位数。
2. **捕获到运行管线的等待与覆盖**：`capture_ready≈29 ms`、每个 5 秒窗口有约 1–16 个新的捕获帧被覆盖，说明生产速度低于捕获速度。
3. **新内容帧节奏**：`screen_new` 低于 `xr_present`，合成器必须复用 GPU 帧；这不影响 72 Hz 刷新，但会造成动态内容看起来不够连续。
4. **需要进一步确认的捕获/色彩转换阶段**：当前只有阶段级耗时，尚无 Vulkan timestamp query 区分 capture copy、BGR resize/normalize、模型、深度后处理、warp/composite 和外部资源交接。

### 4.3 不能直接下的结论

- 不能说“Vulkan 的 72 Hz 呈现比 OpenGL 差”；现有证据相反，至少不差。
- 不能说“OpenGL 的新内容一定更流畅”；`fresh_input` 口径不匹配，且 OpenGL 没有同一组上游耗时字段。
- 不能把 `layer_count=0` 当成空帧；实际画面与提交计数均否定该解释。
- 不能直接承诺 MIGraphX 在所有画质/场景下都优于 PyTorch；本轮只证明同一模型、同一深度分辨率、同一 70 秒窗口下的明显吞吐差异。

## 5. 改进计划

### 阶段 A：补齐同口径基线（优先）

不修改 D2S 源码，先统一测试口径：

1. 在同一会话中运行 OpenGL/PyTorch 和 Vulkan/PyTorch 各 70 秒，使用相同的 `D2S_CONTROLLER_MODEL=none`、`D2S_OPENXR_DISABLE_TOOL_QUADS=1` 和 `D2S_OPENXR_GLOW_MODE=off`。
2. 两个路径都记录可比较的 `screen_new`、`screen_reuse`、`screen_replay`、`capture_ready_p95`、`frame_exec_p95`、`rt_total_p95` 和 `xr_present_fps`。
3. 如 OpenGL 路径无法输出这些字段，先补充仅用于诊断的采样器；这是文档/测量层改动，不改变渲染行为。
4. 对每个路径保留完整日志、头显截图和具体后端字段，禁止只保留一个“FPS”汇总。

验收标准：在同一输入、同一模型、同一深度分辨率下，能明确区分“合成呈现率”和“新内容吞吐率”，并给出两条路径的 p50/p95 耗时与覆盖/丢帧计数。

### 阶段 B：先消除上游尾延迟

优先级从高到低：

1. **推理后端选择**：将 `migraphx_rocm` 与 `pytorch_rocm` 做成受控 A/B；保留当前画质和深度分辨率，不把后端切换和画质改变混在一次测试里。
2. **捕获/色彩转换与推理重叠**：检查 `capture_ready` 中捕获、RGB 转换、resize/normalize 的串行段；优先验证 capture pipeline 与推理是否已有独立资源/队列可并行。
3. **深度后处理与立体合成融合**：在 `rt_total` 中分离模型、深度上采样、warp/composite；先测每个子阶段 p95，再决定是否融合，不直接引入新的采样算法。
4. **GPU 时间戳**：为立体 Compute 的 warp/occlusion/fill、输出转换和 Vulkan Projection 增加 Vulkan timestamp query，替代纯 CPU 墙钟推断。
5. **背压和覆盖策略**：验证 `overwrite` 是否来自捕获过快而非 Presenter 阻塞；如果是，调整捕获 pacing，而不是让 Presenter 等待。

### 阶段 C：只在确认收益后优化 Vulkan 合成

当前 `xr_projection_total` 已低于 3.2 ms p95，不应先做复杂的 multiview 大改。若阶段 B 完成后呈现仍无法维持目标，再依次评估：

- 双眼 transition/输出转换合并；
- Projection Composer 的 layered draw/dispatch；
- `array_size=2` Projection SwapChain 或 multiview。

任何阶段都必须保留双 SwapChain/Presenter 回退，并通过 10 分钟设备稳定性、双眼视觉回归和真实头显 A/B。

## 6. Quad 左右立体结论与后续接入

### 6.1 已证实

- VDXR 能接受 Quad 内容，并在实机上显示到头显。
- 旧测试中左右眼看到了不同的 Quad 内容，用户确认“Quad 左右不同画面测试通过”。
- 因此不应再把 VDXR 当作“不支持立体 Quad”的运行时；问题只剩 D2S 如何正确产生、提交和回收每眼内容。

### 6.2 两种 Quad 必须分开设计

| 用途 | 推荐 OpenXR 表示 | 内容来源 | 回退 |
|---|---|---|---|
| 设置菜单/提示 | 单个 `array_size=1` + `eyeVisibility=BOTH` Quad | 一张双眼共享的 MSDF/UI 纹理 | 普通单 Quad |
| 左右眼立体屏幕 | 左、右眼各自的内容路径 | 左/右眼独立图像或分层纹理 | 双眼共享纹理（仅非立体用途） |

菜单不需要左右眼不同画面；立体屏幕需要。不能为了菜单引入第二套 UI 纹理，也不能把单 Quad 的 `BOTH` 当作立体内容已经完成。

### 6.3 推荐实现顺序

1. 先实现“左 Quad + 右 Quad”的独立 swapchain 路径，分别设置 `eyeVisibility=LEFT` 和 `eyeVisibility=RIGHT`，两者使用相同 pose、尺寸和姿态；这是最直接、最容易在 VDXR 验证立体内容的方案。
2. 若运行时稳定且需要减少提交开销，再评估 `array_size=2` 的 layered Quad swapchain；它必须与普通双 Quad 做画面、性能和同步 A/B。
3. 每眼内容必须来自对应的左/右 GPU 输出，不能把同一张 SBS 纹理用不同 UV 假装成双眼内容；首次接入时用可区分的测试图案确认左右眼不会交换。
4. 菜单和屏幕使用不同资源所有权与生命周期；关闭菜单只停止菜单 Quad，不影响立体屏幕 Quad。

### 6.4 Quad 验收

- 左右眼分别显示可区分的图案，交换左右眼时能明确看出错误。
- 头显中无重影、无双目内容交换、无单眼黑屏。
- 画面姿态与物理尺寸在双眼中一致，转头时没有相对漂移。
- 连续 10 分钟无 device lost、无 swapchain 获取失败、无资源回收错误。
- `xrEndFrame` 仍保持目标刷新率；Quad 更新不阻塞 Presenter 的投影合成。
- 菜单 Quad 与立体屏幕 Quad 可同时存在，层顺序稳定且互不覆盖错误。

## 7. 证据文件

- OpenGL 实测日志：[`runtime-diagnostics/rx9060xt-vdxr-20261124-gl-openxr-verified.log`](../runtime-diagnostics/rx9060xt-vdxr-20261124-gl-openxr-verified.log)
- Vulkan/PyTorch 实测日志：[`runtime-diagnostics/rx9060xt-vdxr-20261124-vulkan-openxr-visible-pytorch.log`](../runtime-diagnostics/rx9060xt-vdxr-20261124-vulkan-openxr-visible-pytorch.log)
- Vulkan/MIGraphX 实测日志：[`runtime-diagnostics/rx9060xt-vdxr-20261124-vulkan-openxr-visible.log`](../runtime-diagnostics/rx9060xt-vdxr-20261124-vulkan-openxr-visible.log)
- Vulkan 可见头显截图：[`runtime-diagnostics/quest-vulkan-20261124-visible-during/screen.png`](../runtime-diagnostics/quest-vulkan-20261124-visible-during/screen.png)
- OpenGL 头显截图：[`runtime-diagnostics/quest-gl-openxr-20261124/screen.png`](../runtime-diagnostics/quest-gl-openxr-20261124/screen.png)

## 8. 限制

- 本轮没有修改 D2S 源码，没有把 OpenGL 和 Vulkan 的所有阶段统一成同一套 instrumentation。
- 每条路径只有一轮 70 秒实际运行；它足以定位当前主要瓶颈，但还不足以估计机器状态或后台负载引起的运行间波动。
- OpenGL 的 `fresh_input` 不是当前仓库中可复现的正式指标，报告中未把它换算为内容 FPS。
- Vulkan/PyTorch 和 Vulkan/MIGraphX 是不同推理后端对比；它们能说明 Vulkan 合成路径不是瓶颈，但不能单独量化 OpenGL 推理实现与 Vulkan 推理实现的差异。
- 截图和用户确认证明画面可见、Quad 左右内容不同；它们不等同于完整的 10 分钟稳定性验收。
