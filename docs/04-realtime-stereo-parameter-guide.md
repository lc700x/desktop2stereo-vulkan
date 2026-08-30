# 实时立体参数测试与视觉回归指南

本文是 Desktop2Stereo 当前有效的立体参数测试指南。它用于通过固定样本、视觉回归和人工评分，找出 `traditional_fastest`、`cinema`、`game_low_latency`、`still_image_hq` 四类立体模式的推荐参数。

运行时语义以 `docs/01-Realtime-2d-to-3d-specification.md` 为准；工程设计以 `docs/02-desktop2stereo-engineering-design-specification.md` 为准。本文只描述测试方法、视觉判定和参数 sweep，不再保留旧 IPD / Stereo Scale / Max Shift Ratio 调参链。

## 当前参数模型

当前 normalized-depth 路径使用软件视差预算，而不是物理瞳距公式。

```text
depth_response = depth - convergence
disparity_px = depth_response * max_disparity_px * depth_strength
left_shift_px = +disparity_px / 2
right_shift_px = -disparity_px / 2
```

字段分工：

| 参数 | 职责 | GUI / 配置 |
|---|---|---|
| `Stereo Preset` | 一组内容模式预设，决定后端、预算、深度强度、temporal、hole fill 等默认组合 | Traditional / Cinema / Game / Image |
| `Parallax Budget Preset` | 根据 `render_size` 解析 `max_disparity_px` 的档位预算 | comfort / standard / strong / extreme |
| `Depth Strength` | 用户连续调节实际立体强度的 gain | Soft=0.20 / Standard=0.25 / Enhanced=0.30，也可直接输入数值 |
| `Convergence` | 零视差/汇聚平面 | 默认 0.0 |
| `Dynamic Convergence Strength` | 动态会聚强度；0.00 表示关闭，使用静态 `Convergence` | 默认 0.00 |
| `Depth Separation Preset` | 前景/中景/背景视差倍率的组合预设 | default / standard / strong / weak |
| `Render Scale` | 4K 级输入的稳定缩放档位，影响 `render_size`，进而影响预算解析 | 4K / 100%, 3K / 85%, 2K / 75%, 1K / 50% |

不得再把以下旧字段作为当前测试变量：

```text
IPD
Stereo Scale
Max Shift Ratio
ipd_mm
stereo_scale
max_shift_ratio
```

这些字段属于旧实现经验，不参与当前参数优化。本文可以用它们做历史对比，但不能把它们写成当前推荐调参动作。

## Parallax Budget 与 IPD 的区别

`Parallax Budget` 是当前 Desktop2Stereo 的软件视差预算，也就是在当前 `render_size` 下允许左右眼分离的像素上限。它不是物理双眼瞳距，也不是现实世界单位。

| 项目 | Parallax Budget | IPD |
|---|---|---|
| 含义 | 软件合成里的最大视差像素预算 | 人的双眼瞳距，通常约 58-72 mm |
| 单位 | 像素，最终解析为 `max_disparity_px` | 毫米或米 |
| 依赖 | `render_size`、短边分辨率、宽高比保护、预算档位 | 真实相机内参、真实深度 Z、显示/观察几何 |
| 当前用途 | 当前有效，用于限制 4K/3K/2K/1K 下的安全视差范围 | 不作为当前 normalized-depth 路径的调参变量 |
| 用户感觉 | 决定“这个模式最大能拉开多少”的安全上限 | 旧实现里常被误用成全局分离倍率 |
| 测试方式 | 可通过 visual regression、`max_disparity_px`、边缘伪影稳定复现 | 没有真实 metric depth 时无法按物理公式验证 |

当前深度模型输出的是 normalized depth，不提供真实物理距离、相机焦距、传感器尺寸和观察者位置。因此不能用真实 IPD 公式直接计算左右眼图像。旧参数里所谓 IPD 的实际观感更接近“把左右眼拉开一点”的全局倍率，而不是真正的物理瞳距模拟。

当前实现把这个概念拆清楚：

```text
max_disparity_px = resolve(render_size, parallax_budget_preset)
base_disparity_px = depth_response(depth, convergence) * max_disparity_px
actual_disparity_px = base_disparity_px * depth_strength
per_eye_shift_px = actual_disparity_px / 2
```

所以：

- 想平滑调节“立体强弱”，优先调 `Depth Strength`。
- 想切换“安全上限/模式风格”，调 `Parallax Budget Preset`。
- 想移动零视差平面，调 `Convergence`。
- 不要再用历史 IPD 名义解释当前视差强度；当前 UI 也不应再暴露 IPD 作为立体参数。

## 当前立体参数详细说明

### 核心视差参数

| 参数 | GUI 位置 / 字段 | 当前选项或范围 | 默认 / 预设 | 作用 | 调参建议 |
|---|---|---|---|---|---|
| Stereo Mode | `Stereo Preset` | Traditional / Fastest, Cinema, Game / Low Latency, Image / High Quality | `cinema` | 内容模式预设，联动后端、视差预算、深度强度、temporal、边缘和补洞默认值 | 先选模式，再做小范围微调；不要把它和所有高级项做全量笛卡尔积 |
| Synthetic View | `Stereo Quality` / `Synthetic View` | fast, fast_plus, quality_4k, hq_4k | 由 Stereo Mode 派生 | 立体合成后端质量/速度档位，GUI 已隐藏为内部派生值 | 普通用户不直接调；若模式效果不对，优先修 preset 映射 |
| Parallax Budget | `Parallax Budget Preset` | Comfort, Standard, Strong, Extreme | `standard` | 按 `render_size` 解析 `max_disparity_px`，限定最大软件视差预算 | 游戏/OpenXR 优先 Comfort；电影 Standard；图片 Strong；Extreme 仅调试/导出 |
| Max Disparity Px | `max_disparity_px` | 结构化运行时字段，可为空 | 空值时由预算解析 | 已解析的像素上限；用于测试、debug、OpenXR/direct shader 对齐 | 不作为 GUI 常规输入；视觉回归 metadata 必须记录 |
| Depth Strength | `Depth Strength` | 0.00-0.50，0.05 步进 | GUI 默认 0.25 | 连续深度强度 gain，直接乘到实际视差上 | 这是用户最直观的“立体强弱”滑杆；太平就升，顶眼/撕裂/重影就降 |
| Depth Quick | `Depth Quick` | Soft, Standard, Enhanced | Standard | 快捷深度强度预设 | Soft=0.20、Standard=0.25、Enhanced=0.30；它应同步/代表 Depth Strength |
| Convergence | `Convergence` | -0.50 到 1.00，0.05 步进 | 0.00 | 零视差/汇聚平面，改变哪些深度落在屏幕平面附近 | 不是强度滑杆；近景太顶眼可适当提高，整体太靠后可降低 |
| Dynamic Convergence | `Dynamic Convergence Strength` | 0.00-1.00，0.10 步进 | 0.00 | 按当前深度图自动微调零视差平面；0.00 关闭，静态 `Convergence` 生效；大于 0.00 时动态会聚生效 | 默认保持 0.00；视频前后景变化很大时再试 0.30-0.50；游戏/低延迟优先关闭 |
| Depth Separation | `Depth Separation Preset` | Default, Standard, Strong, Weak | Cinema=Standard, Game=Weak, Image=Strong | 一键设置 `Foreground Pop` / `Midground Pop` / `Background Pop` | 先用模式默认；人物太顶眼或背景过强时再切 Weak；图片需要层次可用 Strong |
| Render Scale | `Render Scale` | 4K / 100%, 3K / 85%, 2K / 75%, 1K / 50% | 4K / 100% | 对 4K 级输入按比例缩放并保持输入宽高比；影响 `render_size` 与预算解析 | 不每帧动态改变；只在用户切档、OpenXR scale 或输入尺寸稳定跨档时重算 |

### 预算解析参考

`Parallax Budget Preset` 当前按短边分辨率解析。宽高比超过 2:1 时有保护系数，避免超宽或超高画面被拉得过猛。

| Preset | 720p 短边 | 1080p 短边 | 1440p 短边 | 2160p 短边 | 适用场景 |
|---|---:|---:|---:|---:|---|
| comfort | 24 px | 32 px | 48 px | 64 px | 游戏、OpenXR、快速交互、舒适优先 |
| standard | 36 px | 48 px | 64 px | 96 px | 默认电影/传统观看 |
| strong | 48 px | 64 px | 88 px | 128 px | 图片、高质量静态内容 |
| extreme | 64 px | 80 px | 112 px | 160 px | 调试、演示、导出，不作为普通默认 |

### 深度与后处理参数

| 参数 | GUI / 字段 | 当前选项或范围 | 默认 / 预设 | 作用 | 调参建议 |
|---|---|---|---|---|---|
| Depth Model | `Depth Model` | 模型家族 + size | 模型列表首项 | 选择深度估计模型 | 参数视觉回归要固定模型；换模型后必须重新评分 |
| Depth Resolution | `Depth Resolution` | 随模型提供的分辨率选项 | 518 | 深度推理输入宽度，影响深度细节和速度 | 低延迟用较低值，高质量图片可升高；不要把它当作输出分辨率 |
| Depth Pop | `Depth Pop` | -0.9 到 5.0 | 0.0 | 居中深度曲线调整，用于增强或压缩深度层次 | 默认 0；图片可小幅提高，文字/UI 弯曲或前景过强时降低 |
| Foreground Pop | `Foreground Pop` | 0.50-1.60，0.05 步进 | Cinema=1.15 | 增强/减弱近处物体的位移，主要影响人物、手、桌面前景 | 前景太顶眼就降；人物层次不足可小幅升 |
| Midground Pop | `Midground Pop` | 0.50-1.60，0.05 步进 | Cinema=1.05 | 增强/减弱画面主体层的位移，主要影响角色、车辆、常见焦点区域 | 主体层次不足可小幅升；主体边缘不稳就降 |
| Background Pop | `Background Pop` | 0.50-1.60，0.05 步进 | Cinema=1.05 | 增强/减弱远处背景的位移，主要影响天空、墙面、远景建筑 | 背景漂浮或墙面凸起就降；风景图背景太平可小幅升 |
| Anti-aliasing | `Anti-aliasing` / `Depth Antialias Strength` | 0-10 | 1 | 深度图平滑/抗锯齿强度 | 游戏和传统低；电影中等；图片可略高；过高会糊边 |

### 边缘、遮挡与补洞参数

| 参数 | GUI / 字段 | 当前选项或范围 | 默认 / 预设 | 作用 | 调参建议 |
|---|---|---|---|---|---|
| Edge Threshold | `Edge Threshold` | 0.00-0.10 | 0.04 | 深度边缘检测敏感度 | 裂缝多可降低；误检多、轮廓发胖可升高 |
| Edge Dilation | `Edge Dilation` | 0-4 | 2 | 扩张遮挡/补洞边缘区域 | 裂缝多可升；轮廓变胖或发糊就降 |
| Mask Feather | `Mask Feather Radius` | 0-5 | 3 | 羽化 occlusion fill mask | 降低硬边和重复纹理；过高会软化前景边缘 |
| Hole Fill Mode | `Hole Fill Mode` | Balanced, Soft / Low Ghost, Sharp / High Detail, Content Aware / Highest Quality | balanced | 遮挡空洞填充策略 | Balanced 默认；Soft 降重影；Sharp 用于对比细节；Quality 慢但适合图片 |
| Hole Fill Radius | `hole_fill_radius` | 结构化字段，通常由 mode 派生 | balanced=3 | 补洞采样半径 | GUI 不单独暴露；随 Hole Fill Mode 记录到 metadata |
| Hole Fill Strength | `hole_fill_strength` | 结构化字段，通常由 mode 派生 | balanced=1.0 | 补洞混合强度 | GUI 不单独暴露；用于工程验证和回归记录 |
| Screen Edge Mask Suppression | `screen_edge_mask_suppression` | 结构化字段 | 0 | 屏幕边缘遮罩抑制 | 只在边缘特殊伪影检查中使用 |

### 时域稳定参数

| 参数 | GUI / 字段 | 当前选项或范围 | 默认 / 预设 | 作用 | 调参建议 |
|---|---|---|---|---|---|
| Temporal | `Temporal` | bool，由强度推导 | true | 是否启用跨帧稳定 | `Temporal Strength` 为 0 时视为关闭 |
| Temporal Strength | `Temporal Strength` | 0.0-1.0 | 0.25 | 跨帧平滑强度 | 视频可从 0.25 开始；游戏默认 0.0 或很低；拖影就降 |
| Auto Scene Reset | `Auto Scene Reset` | bool，由阈值推导 | true | 场景切换时重置 temporal history | 快切视频必须开启；静态图无意义 |
| Scene Threshold | `Scene Reset Threshold` | 0.00, 0.12, 0.18, 0.22, 0.28, 0.35 | 0.22 | 场景变化检测阈值 | 拖影/残影时降低；频繁闪断时升高 |

### 输出与展示参数

| 参数 | GUI / 字段 | 当前选项或范围 | 默认 / 预设 | 作用 | 调参建议 |
|---|---|---|---|---|---|
| Display Mode | `Display Mode` | Half-SBS, Full-SBS, Half-TAB, Full-TAB, Depth Map, Anaglyph, Interleaved, Mono, Leia | Half-SBS | 输出封装格式 | 按目标设备选择；视觉回归默认记录 SBS 与左右眼单图 |
| Cross Eyed | `Cross Eyed` | true / false | false | 交换左右眼给交叉眼观看 | 只用于特定观看方式；不用于修复深度方向问题 |
| Anaglyph | `Anaglyph Method` | red_cyan, green_magenta, amber_blue | red_cyan | Anaglyph 输出的颜色对 | 只在 Display Mode 为 Anaglyph 时影响画面 |
| Fill 16:9 | `Fill 16:9` | true / false | true | 本地/显示路径的画面填充策略 | 只影响展示构图，不应改变深度语义 |
| Fix Viewer Aspect | `Fix Viewer Aspect` | true / false | false | viewer 宽高比固定策略 | 用于本地 viewer 显示，不作为立体强度参数 |
| VSync | `VSync` | true / false | false | 本地 viewer 垂直同步 | 降低撕裂但可能增加延迟 |
| Target FPS | `Target FPS` | Auto, 60, 72, 80, 90, 120 | Auto | 捕获/显示帧率目标 | OpenXR/游戏按设备刷新率；视觉回归记录实际 timing |
| screen_roll | `screen_roll` | OpenXR runtime 字段 | 0.0 | OpenXR 屏幕手动旋转角，影响展示方向和视差方向 | 只在 OpenXR 展示路径生效；不改变源画面语义和深度模型理解 |

### 运行与加速参数

| 参数 | GUI / 字段 | 当前选项或范围 | 默认 / 预设 | 作用 | 调参建议 |
|---|---|---|---|---|---|
| Run Mode | `Run Mode` | Local Viewer, Basic Streaming, Advanced Streaming, 3D Monitor, OpenXR Link | OpenXR Link | 选择输出运行链路 | 旧配置中的 `Legacy Streamer` 会自动迁移为 低级网络推流；合规检查要分别覆盖本地、OpenXR、推流路径公式一致性 |
| Computing Device | `Computing Device` | 设备列表 | 0 | 深度推理设备 | 不改变立体公式，只影响性能/后端可用性 |
| FP16 | `FP16` | true / false | false | 半精度推理/加速开关 | 可能影响速度和少量数值稳定性；不应改变参数语义 |
| TensorRT / MIGraphX / CoreML / OpenVINO | 加速开关 | 平台相关 | false | 模型推理加速 | 改后要做 provider 级验证；视觉参数不应因加速后端改变 |
| Render Align | `Render Align` | 1, 8, 16, 32 | 8 | 输出纹理尺寸对齐 | 工程兼容参数；不作为用户立体强度调节 |
| Render Min Side / Pixel Cap | `Render Min Dimension` / `Render Max Pixels` | 高级结构化字段 | 480 / 8294400 | 历史 dynamic/fixed 相关保护字段 | 当前 GUI 运行策略固定为 scaled；后续清理时再收敛 |

## 立体模式组合

当前 GUI 中的立体模式本身就是预设组合。视觉回归应按以下组合测试，而不是把后端、预算和强度完全拆散成任意笛卡尔积。

| 模式 | 后端 | 视差预算 | Depth Strength | Depth Quick | 前后分离 | 目标 |
|---|---|---|---:|---|---|---|
| 传统 / 旧算法 | `fast` | `standard` | 0.25 | Standard | default | 旧算法电影观看路径，速度快但不能过度保守 |
| 电影 / 新算法路线 | `quality_4k` | `standard` | 0.25 | Standard | standard | 默认观影模式，平衡质量、稳定性和舒适度 |
| 游戏 | `fast_plus` | `comfort` | 0.20 | Soft | weak | 低延迟、低拖影、HUD 稳定、降低眩晕 |
| 图片 | `hq_4k` | `strong` | 0.30 | Enhanced | strong | 静态图质量优先，更强层次和更完整边缘处理 |

`extreme` 只用于调试、演示或导出，不作为普通模式默认组合。

## 视觉回归目标

视觉回归不是为了让单张图片看起来最立体，而是找出对真实内容稳定、舒适、可复现的默认参数。

必须同时检查：

- 立体感是否足够。
- 左右眼是否容易融合。
- 前景边缘是否撕裂、空洞、重复纹理。
- 字幕、HUD、GUI 文字是否漂浮、弯曲或双边。
- 快速运动是否有 temporal lag。
- OpenXR 旋转/roll 后视差方向是否跟随展示方向。
- 4K / 3K / 2K / 1K render scale 下预算是否稳定。

## 样本集要求

不要用临时挑选的一张 4K 图片决定默认参数。必须建立固定 manifest。

最小样本集：

| 类别 | 样本 | 主要检查 |
|---|---|---|
| cinema_face_closeup | 人像近景、头发、肩膀、衣领 | 人物边缘、脸部深度、前景不顶眼 |
| cinema_dark_scene | 暗场、局部高光 | depth 噪声、暗部错误凸起、闪烁 |
| cinema_subtitle | 字幕/视频 UI | 字幕不漂浮，中线不异常 |
| game_hud_motion | HUD、小地图、准星、快速运动 | 低延迟、无明显拖影、HUD 稳定 |
| game_menu_overlay | 3D 背景上的菜单/背包/弹窗 | UI 平面不被错误立体化 |
| image_portrait_single | 单张人像或主体图 | 静态 HQ 层次、边缘质量 |
| image_landscape | 风景、天空、海面、墙面 | 低纹理区域不被错误拉出 |
| image_ui_document | 桌面、设置页、文档页 | 文字稳定、屏幕不弯曲 |
| image_thumbnail_grid | 图片搜索/缩略图网格 | 网格不整体凹凸，缩略图边界稳定 |

图片模式必须包含正样本和负样本。自然照片是正样本；桌面、文档、缩略图、纯色背景是负样本。否则 `still_image_hq` 很容易在照片上好看，但在真实桌面里产生漂浮文字和弯曲屏幕。

## manifest 格式

推荐 manifest：

```json
{
  "samples": [
    {
      "id": "cinema_face_closeup",
      "path": "cinema/face_closeup.png",
      "category": "cinema",
      "expected_preset": "cinema",
      "checks": ["face_edges", "hair_occlusion", "subtitle_safe"]
    },
    {
      "id": "game_hud_motion",
      "path": "game/hud_motion.png",
      "category": "game",
      "expected_preset": "game_low_latency",
      "checks": ["hud_edges", "no_temporal_lag", "no_softening"]
    },
    {
      "id": "image_portrait_single",
      "path": "image/portrait_single.png",
      "category": "image_natural",
      "expected_preset": "still_image_hq",
      "checks": ["subject_edges", "natural_depth", "background_not_overpulled"]
    },
    {
      "id": "image_thumbnail_grid",
      "path": "image/thumbnail_grid.png",
      "category": "image_thumbnail_grid",
      "expected_preset": "still_image_hq",
      "checks": ["thumbnail_boundaries_flat", "text_stable", "no_curved_screen"]
    }
  ]
}
```

图片文件不建议直接提交到仓库。优先使用本地私有样本目录或宿主产品测试集。

## 推荐输出目录

```text
outputs/visual_preset_matrix/<timestamp>/
  summary.json
  summary.md
  cinema/<sample_id>/
  game_low_latency/<sample_id>/
  still_image_hq/<sample_id>/
  traditional_fastest/<sample_id>/
```

每个样本至少输出：

```text
contact_sheet_labeled.png
left_eye.png
right_eye.png
sbs.png
depth_map.png
occlusion_mask.png
absdiff.png
metadata.json
```

`metadata.json` 必须记录：

```text
sample_id
stereo_preset
backend
render_size
parallax_budget_preset
max_disparity_px
depth_strength
convergence
dynamic_convergence_strength
depth_pop
depth_separation_preset
foreground_pop
midground_pop
background_pop
depth_response
hole_fill_mode
edge_threshold
edge_dilation
mask_feather_radius
temporal_enabled
temporal_strength
output_format
provider_info
timing
```

## 自动测试入口

推荐入口：

```powershell
.\python3\python.exe -B scripts\tools\generate_preset_visual_matrix.py `
  --manifest samples\visual_preset_manifest.json `
  --depth-backend tensorrt_native `
  --out-dir outputs\visual_preset_matrix
```

只验证 manifest 和输出结构，不跑模型：

```powershell
.\python3\python.exe -B scripts\tools\generate_preset_visual_matrix.py `
  --manifest samples\visual_preset_manifest.json `
  --depth-backend luma `
  --out-dir outputs\visual_preset_matrix `
  --dry-run
```

如果工具入口尚未覆盖某个新字段，应先补工具，不要临时手工改参数再截图。视觉回归必须可复现。

## 参数 sweep 策略

调参必须先固定一个 preset，再围绕该 preset 做小范围 sweep。不要一次同时改动后端、预算、深度强度、temporal 和 hole fill。

### 第一阶段：确认模式组合

固定当前推荐组合：

```text
traditional_fastest: fast + standard + depth_strength=0.25 + depth_separation=default
cinema: quality_4k + standard + depth_strength=0.25 + depth_separation=standard
game_low_latency: fast_plus + comfort + depth_strength=0.20 + depth_separation=weak
still_image_hq: hq_4k + strong + depth_strength=0.30 + depth_separation=strong
```

每个样本都跑四个 preset，确认 expected preset 是否是最稳选择，同时观察其它 preset 暴露出的风险。

### 第二阶段：Depth Strength

只改 `Depth Strength`：

```text
traditional_fastest: 0.20 / 0.25 / 0.30
cinema: 0.20 / 0.25 / 0.30
game_low_latency: 0.15 / 0.20 / 0.25
still_image_hq: 0.25 / 0.30 / 0.35
```

判定：

- 太低：整体太平，主体层次不足。
- 合适：5-10 秒内自然融合，前景有层次但不顶眼。
- 太高：难融合、边缘撕裂、空洞增多、字幕漂浮、OpenXR 不适。

### 第三阶段：Parallax Budget

只改 `Parallax Budget Preset`：

```text
comfort / standard / strong
```

`extreme` 只在调试组测试，不进入普通默认候选。

判定：

- `comfort`：优先舒适和低伪影，适合游戏、OpenXR、快速交互。
- `standard`：默认观影范围，适合传统和电影。
- `strong`：更强静态图层次，适合图片。
- `extreme`：只用于暴露边界问题，不用于默认值。

### 第四阶段：Convergence 与 Dynamic Convergence

先只改静态 `Convergence`：

```text
-0.25 / 0.00 / 0.25 / 0.50
```

再只改 `Dynamic Convergence Strength`：

```text
0.00 / 0.30 / 0.50
```

判定：

- `Dynamic Convergence Strength = 0.00` 时关闭动态会聚，静态 `Convergence` 生效。
- 大于 0.00 时启用动态会聚，按当前深度图自动微调零视差平面。
- 背景跑到前面：先确认眼序，再调 convergence。
- 近景太顶眼：提高 convergence、降低 dynamic convergence strength，或降低 depth strength。
- 整体太往后：降低 convergence。
- 视频主体前后变化大时可试 0.30-0.50；游戏/低延迟默认保持 0.00。

`Convergence` 和动态会聚都不是强度滑杆，不能用它们替代 `Depth Strength`。

### 第五阶段：分层视差、边缘与补洞

只在前四步稳定后再调：

```text
depth_separation_preset: default / standard / strong / weak
depth_pop: -0.2 / 0.0 / 0.2 / 0.4
foreground_pop: 0.90 / 1.00 / 1.15 / 1.25
midground_pop: 1.00 / 1.05 / 1.10
background_pop: 0.85 / 1.00 / 1.05
edge_threshold: 0.02 / 0.04 / 0.06
edge_dilation: 0 / 1 / 2 / 3
mask_feather_radius: 0 / 1 / 2 / 3 / 4
hole_fill_mode: balanced / soft_low_ghost / sharp_test / quality
depth_antialias_strength: 0.0 / 1.0 / 2.0
```

判定：

- 边缘裂缝多：降低 edge_threshold，适度提高 edge_dilation。
- 边缘发糊：降低 edge_dilation、mask feather 或 depth antialias。
- 重复纹理明显：提高 mask feather，或从 sharp_test 改 balanced / soft_low_ghost。
- 人物轮廓软化：降低 depth antialias 和 mask feather。
- 天空/墙面错误凸起：降低 background_pop、depth_pop 或 depth_strength。

## 模式判定标准

### Traditional / Fast

目标：旧算法电影观看路径，速度快，立体感不能明显弱于电影模式。

推荐默认：

```text
backend=fast
parallax_budget_preset=standard
depth_strength=0.25
depth_separation=default
dynamic_convergence_strength=0.00
temporal_strength=0.0
hole_fill_mode=balanced
```

通过标准：

- 立体感接近 cinema 的基础层次。
- 边缘伪影可接受。
- 不因默认 comfort 导致整体过平。

### Cinema

目标：默认观影模式。

推荐默认：

```text
backend=quality_4k
parallax_budget_preset=standard
depth_strength=0.25
depth_separation=standard
foreground_pop=1.15
midground_pop=1.05
background_pop=1.05
dynamic_convergence_strength=0.00
temporal_strength=0.25
hole_fill_mode=balanced
```

通过标准：

- 人像、字幕、暗场稳定。
- temporal 不产生明显拖影。
- 边缘比 traditional 更干净。

### Game / Low Latency

目标：低延迟和舒适优先。

推荐默认：

```text
backend=fast_plus
parallax_budget_preset=comfort
depth_strength=0.20
depth_separation=weak
foreground_pop=1.15
midground_pop=1.05
background_pop=0.85
dynamic_convergence_strength=0.00
temporal_strength=0.0
hole_fill_mode=soft_low_ghost
```

通过标准：

- HUD、准星、菜单文字稳定。
- 鼠标快速移动不拖影。
- 边缘不明显软化。
- OpenXR 或高帧率场景不产生明显不适。

### Still Image / HQ

目标：静态图质量优先。

推荐默认：

```text
backend=hq_4k
parallax_budget_preset=strong
depth_strength=0.30
depth_separation=strong
foreground_pop=1.25
midground_pop=1.10
background_pop=1.00
dynamic_convergence_strength=0.00
temporal_strength=0.0
hole_fill_mode=quality
```

通过标准：

- 正样本主体层次清楚，边缘更干净。
- 负样本文字、UI、缩略图不漂浮、不弯曲。
- 低纹理自然照片不被错误全图平面化，也不出现径向凹凸。

## 评分表

每个样本/参数组合记录人工判定：

| 字段 | 含义 |
|---|---|
| `pass` | 没有明显肉眼问题，可以作为默认候选 |
| `warn` | 可接受但有轻微问题，只适合特定内容或高级选项 |
| `fail` | 撕裂、空洞、重复纹理、文字漂浮、屏幕弯曲、明显不适 |

建议评分维度：

```text
stereo_strength: 1-5
comfort: 1-5
edge_quality: 1-5
text_stability: 1-5
temporal_stability: 1-5
hole_fill_artifacts: 1-5
```

默认参数候选必须满足：

```text
comfort >= 4
edge_quality >= 3
text_stability >= 4 for UI/text samples
temporal_stability >= 4 for video/game samples
no fail on expected preset
```

## 常见问题对照

| 现象 | 优先检查 / 调整 |
|---|---|
| 整体太平 | 提高 Depth Strength；若仍不足，检查 Parallax Budget 是否过于保守 |
| 近景太顶眼、难融合 | 降低 Depth Strength；必要时从 strong 降到 standard/comfort |
| 背景跑到前面 | 先确认左右眼序，再调 Convergence |
| 字幕/GUI 漂浮 | 降低 Depth Strength，降低 Parallax Budget，或对 UI 样本使用更保守 preset |
| 边缘裂缝、空洞明显 | 降低视差强度，调 edge_threshold / edge_dilation / hole_fill_mode |
| 边界发糊、前景轮廓变胖 | 降低 edge_dilation、mask_feather_radius、depth_antialias_strength |
| 快速运动拖影 | 降低 Temporal Strength，确认 scene reset 生效 |
| OpenXR 手动旋转屏幕后方向不对 | 检查 `screen_roll` 是否进入 shader uniform，OpenGL/D3D11 是否同语义 |

## 验证命令

修改参数默认值、GUI 映射或视觉回归工具后，至少运行：

```powershell
.\python3\python.exe -B -m pytest tests\test_gui_config.py tests\test_presets.py tests\test_parallax.py tests\test_runtime.py
.\python3\python.exe -B -m py_compile src\gui\config_mgr.py src\stereo_runtime\presets.py src\stereo_runtime\baseline_shift.py
```

若改动 OpenXR shader 或 direct 路径，还必须运行：

```powershell
.\python3\python.exe -B -m pytest tests\test_openxr_runtime.py tests\test_runtime_openxr.py
```

若改动推流或 output packing，还必须运行：

```powershell
.\python3\python.exe -B -m pytest tests\test_legacy_sbs.py tests\test_mjpeg_streamer.py tests\test_synthesis.py
```

最终默认值不能只凭单次测试决定。必须保存 manifest、summary、contact sheet、metadata 和人工评分，便于下次参数调整时回归对比。
