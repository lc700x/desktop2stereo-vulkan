# OpenXR Quad 设置菜单实施计划

**更新时间**：2026-11-24

**关联实测**：[23 号 VDXR 性能与立体 Quad 报告](23-vdxr-quest2-opengl-vulkan-performance-and-stereo-quad-report.md)

## 1. 目标与交互入口

在 OpenXR Vulkan 路径新增一个 `array_size=1`、`eyeVisibility=BOTH` 的设置菜单 Quad。菜单采用单面板三页签：

- **画面**：完整颜色参数、LOD 与 RCAS。
- **房间**：动态座位选择、座位高度、灯光预设和场景亮度。
- **屏幕**：屏幕宽度、垂直位置和平面/曲面切换。

菜单关闭时，任一手 Trigger 在激光未命中虚拟屏幕和键盘的区域完成一次短按后打开菜单。菜单打开时放在头部前方约 `1.1 m`、视线下方 `0.12 m`，按打开瞬间的水平头部朝向生成姿态，之后锁定世界空间，不持续跟随头部。

菜单打开后 Trigger 输入优先交给菜单：

- 命中页签、按钮或开关时执行对应操作。
- 命中滑块时按住拖动，释放时保存。
- 命中菜单外时关闭菜单，不向桌面发送鼠标事件。
- 两手同时操作时锁定最先按下 Trigger 的手，另一只手忽略到释放。

沿用现有 Trigger 阈值：`0.7` 按下、`0.3` 释放、`0.35 s` 长按。菜单关闭且激光命中虚拟屏幕时，左右 Trigger 现有桌面左键/右键语义保持不变。

## 2. Quad 渲染与层顺序

最终合成固定为三段画家顺序：主 Projection（房间、Glow、虚拟屏幕）→ 设置/提示 Quad → 透明双眼 Controller Projection（手柄、激光、操作指南）。Controller Projection 使用 `array_size=2` 的 multiview swapchain 并带源 Alpha 混合标志，永远最后提交，因此手柄、激光和指南始终显示在全部画面及菜单最前方。该设计明确不提供手柄与菜单之间的真实深度遮挡；即使手柄空间位置在菜单之后，仍按产品要求显示在菜单上方。

Controller Projection 复用同一个 Filament Engine 和现有 multiview 相机，只增加同步 `flushAndWait()` 的透明输出目标；不得重新启用已经触发过 `VK_ERROR_DEVICE_LOST` 的双 SwapChain deferred 提交路径。

扩展现有 GPU MSDF Quad 渲染协议，在文字之外支持：

- 圆角矩形和面板背景；
- 普通、悬停、选中和禁用按钮；
- 滑轨、填充进度和滑块手柄；
- 页签指示和菜单内激光光标。

文字和图元在一次 Compute dispatch 中绘制，避免拖动滑块时由 CPU 重画并上传整张位图。菜单画布固定约为 `1024×768`，物理尺寸约 `0.95×0.71 m`，使用深色半透明背景、黄色选中状态和蓝白激光光标。

菜单内容未变化时复用当前 Swapchain 图像，仅在悬停、页签或参数变化时重绘；连续滑块更新限制为最高 `20 Hz`。关闭菜单时停止提交该 Quad Layer，重新打开时复用既有 Swapchain 和 GPU 资源。

**界面 Quad 与立体屏幕 Quad 的边界**：本节的 `eyeVisibility=BOTH` Quad 只用于设置菜单、提示和激光光标，它不需要左右眼不同内容。需要每眼不同画面的立体屏幕必须使用独立的左右眼内容路径；单一 `BOTH` Quad 不能替代立体屏幕。2026-11-24 已在 Quest 2 / VDXR 上确认 Quad 左右不同画面可行，后续只需验证 D2S 的资源接入与生命周期。

## 3. 画面页

画面页完整复用桌面 GUI 的参数范围与步进：

| 参数 | 范围 | 步进 |
|---|---:|---:|
| 亮度 | `0.2–2.0` | `0.1` |
| 对比度 | `0.5–2.0` | `0.1` |
| 饱和度 | `0.0–2.0` | `0.1` |
| Gamma | `0.5–2.0` | `0.1` |
| 色温 | `-100–100` | `10` |
| 色调 | `-100–100` | `10` |
| Min LOD | `0.0–2.0` | `0.05` |
| Max LOD | `0.0–2.0` | `0.05` |
| MIP Bias | `-1.5–0.0` | `0.05` |
| RCAS | `0.0–1.0` | `0.05` |

Min/Max LOD 始终保持 `Min LOD ≤ Max LOD`。拖动时通过新增的菜单设置回调更新完整、版本化的 `RuntimeSettingsSnapshot`，同步作用于推理与 Vulkan Projection；释放滑块时原子合并写入 `settings.yaml`。

当前运行中的桌面 GUI 不新增反向 IPC。GUI 在下次启动或重新加载设置时读取菜单保存的新值。

## 4. 房间页

### 4.1 座位选择与高度

从当前房间 Profile 的 `view_poses` 动态生成座位按钮，显示实际名称，不硬编码“前排/中排/后排”：

- 影院 Profile 可显示 Front、Center、Back；
- 客厅 Profile 可显示 Sofa Left、Sofa Center、Sofa Right；
- 只有一个座位时只显示当前座位和高度调节。

选择座位时更新 `view_pose_index`，复用现有 Profile 世界/场景坐标转换生成目标头部变换，并执行一次用户主动触发的闭环参考空间校准。完成后继续保持普通房间忽略 VDXR 自动参考空间重定位事件，避免运行期间漂移。

座位高度以当前座位原始 Y 为中心调节 `±1.0 m`，步进 `0.02 m`，写回对应 `view_poses[index].y`。

### 4.2 场景灯光

动态显示 Profile 中实际存在的 `lighting_presets`。选择预设时更新 `lighting_preset_index` 并立即应用到 Filament。

新增可选 Profile 字段：

```json
{
  "lighting_exposure_offset_ev": 0.0
}
```

该值范围为 `-4.0～+4.0 EV`、步进 `0.1 EV`，叠加在当前灯光预设的基础曝光上，不修改 Day/Night 等预设本身。

无 `lighting_presets` 的房间隐藏预设按钮，但仍可显示场景亮度滑块。

## 5. 屏幕页

- **屏幕宽度**：以 Profile 初始宽度为基准，范围为 `0.25–2.0 倍`；同步调整物理高度并保持当前宽高比。
- **垂直位置**：调整 `screen.position[1]`，范围为加载 Profile 时初始位置的 `±2.0 m`，步进 `0.02 m`。
- **平面/曲面**：更新 `screen.curved`；仅当 `screen.allow_curve=true` 时可操作，否则显示禁用状态。
- 每个连续参数提供恢复按钮，恢复本次加载 Profile 时记录的初始值。

屏幕修改立即同步到 Presenter 的屏幕几何、Glow、键盘锚点和激光命中计算。释放滑块或切换开关后原子写回当前 `profile.json`。

若屏幕页后续选择 Quad 呈现虚拟屏幕，必须把“立体内容”作为独立契约处理：

- 左、右眼分别读取对应眼的 GPU 图像，禁止用同一张单眼/SBS 纹理冒充双眼内容。
- 首版优先使用左/右独立 Quad swapchain（`eyeVisibility=LEFT/RIGHT`），共享同一世界姿态与物理尺寸。
- 双眼内容必须支持一次性左右交换诊断，确保 OpenXR 的 eye index 映射不会反转。
- 若后续采用 layered `array_size=2` Quad，保留独立双 Quad 回退，并在同一 pose、同一输入下做视觉与性能 A/B。
- 立体屏幕 Quad 与设置菜单 Quad 使用独立的资源所有权、更新时钟和生命周期；菜单关闭不能影响屏幕内容。

## 6. 数据流与内部接口

新增独立菜单状态与布局模块，负责：

- 三页签状态；
- 控件矩形、数值格式化和禁用状态；
- Quad UV 命中测试；
- Trigger 点击/拖动状态机；
- 参数变化事件和重绘脏标记。

`VulkanMsdfQuadRequest` 增加图元集合；MSDF Compute shader 增加图元缓冲，并同步更新 shader manifest 和 SPIR-V。

Presenter 新增统一菜单设置回调：

- 画面参数交给 `RuntimeCallbacks`，原子修改 `settings.yaml` 后复用现有 `StereoHotReloader` 生成完整快照，同时更新推理和 Projection。
- 房间、灯光和屏幕参数由 Presenter 更新当前运行状态，并通过临时文件加 `os.replace()` 原子保存当前 Profile。

菜单文本跟随现有中英文语言设置。Profile 缺少某项能力时隐藏或禁用对应控件，不制造通用房间参数。

## 7. 测试计划

### 7.1 单元测试

- 三页签布局、按钮/滑块 UV 命中、悬停和拖动。
- 左右 Trigger 均能打开和操作菜单。
- 菜单交互不会发送桌面鼠标事件；菜单外单击只关闭菜单。
- 双手同时按下时只接受首个操作手。
- 所有画面参数的范围、步进、LOD 联动约束和完整快照热更新。
- 不同 Profile 的动态座位名称、单座位降级、无灯光预设和禁止曲面状态。
- 座位切换后只执行用户主动闭环校准，不重新启用运行时漂移。
- `settings.yaml` 与 Profile 原子保存以及重新加载结果一致。

### 7.2 渲染与性能测试

- 菜单使用单 Quad、`eyeVisibility=BOTH`，左右眼位置和物理尺寸一致。
- 立体屏幕使用左右眼独立内容；若采用左/右独立 Quad，分别设置 `eyeVisibility=LEFT` 与 `eyeVisibility=RIGHT`，并验证左右眼不会交换。
- 若评估 `array_size=2` layered Quad，必须与双 Quad 做画面、同步和性能 A/B；单 Quad `BOTH` 只用于非立体 UI。
- 验证 `主 Projection → Quad → Controller Projection` 的稳定层顺序、Swapchain 复用和关闭时停止提交。
- 验证 GPU MSDF 图元容量、shader manifest 和 SPIR-V 构建。
- 菜单未变化时不重复上传；滑块连续更新不超过 `20 Hz`。
- 确认菜单开启后不会显著降低 OpenXR Presentation FPS。

### 7.3 实机验收

- 屏幕外 Trigger 短按稳定弹出菜单，菜单打开后世界锁定且不随转头漂移。
- 菜单双眼显示一致，不被虚拟屏幕、Glow 或房间遮挡。
- 左右立体 Quad 显示不同内容，左右交换测试能明确发现；单眼黑屏、重影或姿态不一致均判失败。
- 菜单 Quad 与立体屏幕 Quad 同时存在时层顺序和资源回收正确，连续 10 分钟无 device lost。
- 颜色、LOD 和 RCAS 实时可见，重启后保留。
- 座位、座位高度、灯光和屏幕调整即时生效，重新启动同一房间后保持。
- 菜单关闭后，原有桌面鼠标、键盘、手柄快捷键和屏幕拖动行为不变。

## 8. 首版边界

- 仅实现 OpenXR Vulkan 路径，不为桌面预览窗口增加菜单。
- 菜单首版保持单个 `eyeVisibility=BOTH` Quad；左右眼独立内容的立体屏幕是独立能力，不以菜单 Quad 的完成作为前置条件。
- 所有菜单修改均持久化。
- 当前运行中的 GUI 不实时同步；下次加载 `settings.yaml` 时显示新值。
- 首版不提供任意颜色选择器、环境模型切换、媒体播放控制或手柄模型配置。
