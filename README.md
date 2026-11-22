# Desktop2Stereo Vulkan

Desktop2Stereo的Python Vulkan迁移项目。项目保持原工程的`src/`目录和Python编程方式，默认图形主路径迁移为Vulkan，OpenGL仅作为隔离Fallback。

## 架构原则

- Capture、AI推理、立体合成调度、Vulkan、OpenXR、GUI和输出均使用Python源码实现。
- 保留已验证的WindowsCaptureCUDA、WindowsCaptureROCm、TensorRT、ROCm/MIGraphX和Triton优化代码。
- GPU实时路径不进行CPU NumPy/PIL逐帧往返。
- Filament DLL Bridge是唯一允许的项目自有C/C++组件。
- 不迁入Panda3D、D3D11 OpenXR、WGL/CUDA-GL Bridge和旧OpenGL上传器。

## 当前状态

仓库已完成第一批复制迁移和目标目录初始化。Capture、Inference、Stereo、GUI、资产及平台无关OpenXR代码已迁入。Phase 1已实现Python Vulkan Device/Queue/同步基础层，以及基于`XR_KHR_vulkan_enable2`的OpenXR Vulkan Session、双眼交换链和纯色Projection Layer；已在Virtual Desktop OpenXR Runtime和RTX 3090上完成300帧头显实测，双眼稳定显示。Filament Vulkan Render Target Bridge的C ABI、Vulkan Platform和Python ctypes封装已实现，正在进行SDK CI编译与运行时联调。

## 启动

```powershell
src\python3\python.exe src\desktop2stereo\main.py
```

默认启动进入新的 Flet GUI；颜色、立体参数和运行配置由 GUI 写入`src/desktop2stereo/settings.yaml`。能力检查仍可使用：

```powershell
src\python3\python.exe src\desktop2stereo\main.py --probe
```

GUI 的“运行”按钮已使用独立的`--runtime`子进程入口，不会递归打开第二个 GUI；完整捕捉、推理和输出运行时仍按迁移计划继续装配。

连接并唤醒头显、确认目标OpenXR Runtime处于活动状态后，可执行Phase 1双眼纯色帧实测：

```powershell
src\python3\python.exe src\tools\openxr_vulkan_smoke.py --frames 300
```

该入口只验证OpenXR Vulkan会话与交换链闭环，不代表Filament场景渲染已经接入。

启用Filament GLB场景渲染时，显式指定当前平台Bridge和GLB资源：

```powershell
src\python3\python.exe src\tools\openxr_vulkan_smoke.py `
  --seconds 120 `
  --filament-bridge src\desktop2stereo\xr_viewer\native\windows\filament_bridge.dll `
  --filament-glb src\desktop2stereo\xr_viewer\environments\Artemis\environment.glb `
  --filament-profile src\desktop2stereo\xr_viewer\environments\Artemis\profile.json
```

`--filament-profile`会把profile中选中的座位映射到初始头部位置，同时保留后续头部移动和双眼视差；`--seconds`用于按时间进行长测。Linux使用`libfilament_bridge.so`，macOS使用`libfilament_bridge.dylib`。该模式需要连接并唤醒头显。

## 文档

- [项目日志](changelog.md)
- [实时系统规格](docs/01-Realtime-2d-to-3d-specification.md)
- [工程设计规范](docs/02-desktop2stereo-engineering-design-specification.md)
- [Vulkan迁移技术报告](docs/03-d2s_vulkan_migration_technical_report.md)
- [迁移清单](docs/MIGRATION_MANIFEST.md)
- [跨平台串流运行时打包](docs/streaming-runtime-packaging.md)
