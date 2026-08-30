import gettext
from types import MappingProxyType


DEFAULT_LOCALE = "EN"

MESSAGE_CATALOGS = {
    "EN": {
        "Monitor": "Mointor",
        "Window": "Window",
        "Refresh": "Refresh",
        "Show FPS": "Show FPS",
        "Debug Mode": "Debug Mode",
        "Convergence:": "Convergence:",
        "Dynamic Convergence:": "Dynamic Convergence:",
        "Display Mode:": "Display Mode:",
        "Depth Model:": "Depth Model:",
        "Depth Strength:": "Depth Strength:",
        "Depth Quick:": "Depth Quick:",
        "Soft": "Soft",
        "Standard": "Standard",
        "Enhanced": "Enhanced",
        "Stereo Mode:": "Stereo Mode:",
        "Traditional / Fastest": "Traditional / Fastest",
        "Cinema": "Cinema / Balance",
        "Game / Low Latency": "Game / Low Latency",
        "Image  / High Quality": "Image  / High Quality",
        "Debug / Export": "Debug / Export",
        "Synthetic View:": "Synthetic View:",
        "Parallax Budget:": "Parallax Budget:",
        "Depth Separation:": "Depth Separation:",
        "separation_default": "Default",
        "separation_standard": "Standard",
        "separation_strong": "Strong",
        "separation_weak": "Weak",
        "comfort": "Comfort",
        "standard": "Standard",
        "strong": "Strong",
        "extreme": "Extreme",
        "fast": "Lowest",
        "fast_plus": "Medium",
        "quality_4k": "High",
        "hq_4k": "Highest",
        "Temporal Strength:": "Temporal Strength:",
        "Temporal": "Temporal",
        "Scene Threshold:": "Scene Threshold:",
        "Auto Scene Reset": "Auto Scene Reset",
        "Edge Dilation:": "Edge Dilation:",
        "Mask Feather:": "Mask Feather:",
        "Edge Threshold:": "Edge Threshold:",
        "Hole Fill Mode:": "Hole Fill:",
        "Off / No Fill": "Off / No Fill",
        "Balanced": "Balanced",
        "Balanced / Standard": "Balanced / Standard",
        "Content Aware / Highest Quality": "Content Aware / Highest Quality",
        "On": "On",
        "Anaglyph:": "Anaglyph:",
        "Color Brightness:": "Brightness:",
        "Color Contrast:": "Contrast:",
        "Color Saturation:": "Saturation:",
    "Color Gamma:": "Gamma:",
    "Color Temperature:": "Color Temperature:",
    "Color Tint:": "Color Tint:",
    "Projection Min LOD:": "Min LOD:",
    "Projection Max LOD:": "Max LOD:",
    "Projection MIP Bias:": "MIP Bias:",
    "Projection RCAS:": "RCAS:",
    "Projection Min LOD tooltip": "Sets the lowest shared-output mip level for every mode. Recommended: 0.00.",
    "Projection Max LOD tooltip": "Caps the highest shared-output mip level for every mode. Recommended: 0.35.",
    "Projection MIP Bias tooltip": "Biases shared output sampling toward sharper mip levels. Recommended: -0.35.",
    "Projection RCAS tooltip": "Applies RCAS after shared mip filtering and scaling. Recommended: 0.50.",
        "Cross Eyed": "Cross Eyed",
        "Picture": "Picture",
        "Depth": "Depth",
        "Glow": "Glow",
        "Room": "Room",
        "Screen": "Screen",
        "Render Scale": "Render Scale",
        "Brightness": "Brightness",
        "Contrast": "Contrast",
        "Saturation": "Saturation",
        "Gamma": "Gamma",
        "Temperature": "Temperature",
        "Tint": "Tint",
        "Min LOD": "Min LOD",
        "Max LOD": "Max LOD",
        "MIP Bias": "MIP Bias",
        "RCAS": "RCAS",
        "Reset to default values": "Reset to default values",
        "Depth strength": "Depth strength",
        "2D / 3D": "2D / 3D",
        "Cross eyed": "Cross eyed",
        "Previous seat": "Previous seat",
        "Next seat": "Next seat",
        "Front": "Front",
        "Middle": "Middle",
        "Back": "Back",
        "Seat height": "Seat height",
        "Scene brightness": "Scene brightness",
        "Screen reflection light": "Screen reflection light",
        "Screen size": "Screen size",
        "Screen height": "Screen height",
        "Screen distance": "Screen distance",
        "Flat": "Flat",
        "Subtle": "Subtle",
        "Medium": "Medium",
        "Deep": "Deep",
        "Surround Glow": "Surround Glow",
        "Veil": "Veil",
        "OFF": "OFF",
        "Video appearance": "Video appearance",
        "Stereo depth": "Stereo depth",
        "Glow effects": "Glow effects",
        "Scene controls": "Scene controls",
        "Screen geometry": "Screen geometry",
        "Advanced Stereo": "Advanced Stereo",
        "Advanced Device Options": "Advanced Options",
        "Depth Resolution:": "Depth Resolution:",
        "Anti-aliasing:": "Anti-aliasing:",
        "Depth Pop:": "Depth Pop:",
        "Foreground Pop:": "Foreground Pop:",
        "Midground Pop:": "Midground Pop:",
        "Background Pop:": "Background Pop:",
        "FP16": "FP16",
        "Inference Acceleration:": "Acceleration:",
        "Recompile TensorRT": "Recompile TensorRT",
        "Recompile CoreML": "Recompile CoreML",
        "Recompile OpenVINO": "Recompile OpenVINO",
        "Stop": "Stop",
        "Computing Device:": "Computing Device:",
        "Reset": "Reset",
        "Run": "Run",
        "Set Language:": "Set Language:",
        "Error": "Error",
        "Warning": "Warning",
        "Display refresh warning": "Display refresh warning",
        "Input display refresh warning": "Input display refresh warning",
        "display_refresh_warning_body": "The SBS output display is running at {refresh_hz} Hz, below the measured {sbs_fps:.1f} FPS or the recommended 60 Hz minimum. Increase the display refresh rate in Windows or the GPU control panel.",
        "input_refresh_warning_body": "The input display is running at {refresh_hz} Hz, below the dynamic capture target of {capture_target} FPS. Increase the input display refresh rate or lower the capture target manually.",
        "display_refresh_warning_continuing": "Desktop2Stereo will continue running; this warning does not stop output.",
        "Saved": "Run Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "PyYAML not installed, cannot save YAML file.",
        "Settings saved to settings.yaml": "Settings saved to settings.yaml",
        "Failed to save settings.yaml:": "Failed to save settings.yaml:",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.",
        "Loaded settings.yaml at startup": "Loaded settings.yaml at startup",
        "Running": "Running... (Hold ESC 3s to Stop)",
        "Stopped": "Stopped.",
        "Countdown": "Settings saved to settings.yaml, starting...",
        "A thread already running!": "A thread already running!",
        "No windows found": "No windows found",
        "err_refresh_window": "Failed to refresh window list: {}",
        "Selected input window:": "Selected input window:",
        "Selected input monitor:": "Selected input monitor:",
        "Run Mode:": "Run Mode:",
        "Stream Settings": "Stream Settings",
        "Local Viewer": "Local Viewer",
        "MJPEG Streamer": "Basic Streaming",
        "RTMP Streamer": "Advanced Streaming",
        "Intel QSV (D3D11)": "Intel QSV (D3D11)",
        "full_sbs_stream_advisory": "Full-SBS will be streamed unchanged with H.265/HEVC. Half-SBS is recommended for wider browser compatibility and lower decoding load.",
        "Stream Protocol:": "Stream Protocol:",
        "Stream Key": "Stream Key:",
        "Stereo Mix": "Stereo Mix:",
        "CRF": "CRF:",
        "Audio Delay": "Audio Delay (s):",
        "Lossless Scaling Support": "Frame Generation",
        "3D Monitor": "3D Monitor",
        "OpenXR Link": "OpenXR Link",
        "Headset Model:": "Headset Model:",
        "XR Preview Window": "XR Preview Window",
        "VSync": "VSync",
        "Capture FPS:": "Capture FPS:",
        "Render Policy:": "Render Policy:",
        "Render Scale:": "4K Render Scale:",
        "Render Fixed Size:": "Fixed Size:",
        "Render Pixel Cap:": "Pixel Cap:",
        "Render Min Side:": "Min Side:",
        "Render Align:": "Align:",
        "Native": "Native",
        "Scaled": "Scaled",
        "Fixed": "Fixed",
        "Dynamic": "Dynamic",
        "Upscaler:": "Upscaler:",
        "Upscaler Sharpness:": "Sharpness:",
        "Auto": "Auto",
        "Off": "Off",
        "Streamer Port:": "Streamer Port:",
        "Streamer URL": "Streamer URL:",
       "Preview":"Preview",
        "Stream Quality:": "Stream Quality:",
        "Transmission Profile:": "Transmission Profile:",
        "Auto Calibration": "Auto Calibration",
        "Manual": "Manual",
        "Start Calibration": "Start Calibration",
        "Not calibrated": "Not calibrated",
        "Automatic Network and Performance Calibration": "Automatic Network and Performance Calibration",
        "Cancel Calibration": "Cancel Calibration",
        "Detect Firewall Rules": "Detect Firewall Rules",
        "Close": "Close",
        "calibration_open_headset_url": "Open this address in the headset browser and keep the page visible:",
        "calibration_waiting": "Waiting for the headset to open the test page...",
        "calibration_settling": "Stabilizing the new bitrate...",
        "calibration_testing": "Testing {fps} FPS end-to-end...",
        "calibration_reconnecting": "Switching profile; the headset page is reconnecting...",
        "calibration_complete": "Calibration complete",
        "calibration_requires_advanced": "Automatic calibration requires Advanced Network Streaming with WebRTC.",
        "calibration_requires_webrtc": "Automatic calibration requires WebRTC.",
        "calibration_port_unavailable": "Automatic calibration needs the port immediately after the WebRTC port.",
        "calibration_required_before_run": "Automatic calibration is required before starting because no current calibration profile is available.",
        "calibration_profile_stale": "Settings changed, please recalibrate",
        "calibration_bandwidth_insufficient": "Bandwidth is insufficient for the current resolution; lower the resolution and recalibrate.",
        "calibration_receiver_metrics": "decode {decoded:.1f} FPS · bitrate {bitrate:.1f} Mbps · dropped {dropped} · freeze {freeze} · lost {lost} · jitter {jitter:.1f} ms",
        "calibration_firewall_remove_failed": "Windows Firewall is blocking the bundled Python inbound connection ({protocol}), and the rule could not be removed. Run Desktop2Stereo as administrator and try again.",
        "calibration_firewall_probe_failed": "Windows Firewall detection failed, so automatic calibration cannot safely continue: {error}",
        "calibration_firewall_manual_hint": "If the headset does not respond when opening port {port}, click Detect Firewall Rules.",
        "calibration_firewall_checking": "Checking Windows Firewall rules...",
        "calibration_firewall_no_blocks": "No inbound block rules were found for the bundled Python.",
        "calibration_firewall_removed": "Matching Windows Firewall block rules were removed. Open the headset page again.",
        "calibration_profile_summary": "{fps} FPS · {target} Mbps",
        "calibration_result_stable": "Stable network limit: {network_max} Mbps, safe bitrate: {safe_target} Mbps, {fps} FPS.",
        "calibration_result_limited": "Network calibration failed at {network_max} Mbps; lower the resolution and recalibrate.",
        "calibration_applied": "Calibration applied: {fps} FPS, {target} Mbps",
        "calibration_result": "Stable network limit: {network_max} Mbps · safe bitrate: {target} Mbps · peak {peak} Mbps · {fps} FPS",
        "calibration_recalibrate_hint": "Also recalibrate after:\n• Changing the router, Wi-Fi band, or headset position.\n• Changing the Pico, Quest, or other headset.\n• Changing the headset browser, or switching Wolvic between Gecko and Chromium.\n• A major browser or headset system update.\n• Changing output resolution, codec, or quality settings.\n• Changing the PC GPU, driver, or performance mode.",
        "Host": "Host:",
        "Invalid port number (1-65535)": "Invalid port number (must be between 1-65535)",
        "Invalid port number": "Port must be a number",
        "Please select a window before running in Window capture mode": "Please select a window before running in Window capture mode",
        "Local Viewer requires a second display": "Local Viewer requires at least two displays. Install or enable a virtual display, refresh the display list, and try again.",
        "Selected input display is unavailable": "The selected input display is disconnected or unavailable. Refresh the display list and select an available input display.",
        "Selected stereo output display is unavailable": "The selected stereo output display is disconnected or unavailable. Refresh the display list and select an available output display.",
        "The selected window no longer exists. Please refresh and select a valid window.": "The selected window no longer exists. Please refresh and select a valid window.",
        "Failed to stop process on exit:": "Failed to stop process on exit:",
        "Failed to stop process:": "Failed to stop process:",
        "Failed to run process:": "Failed to run process:",
        "Failed to load settings.yaml:": "Failed to load settings.yaml:",
        "Opening URL in browser": "Opening URL in browser",
        "Controller:": "Controller:",
        "Environment:": "Room:",
        "Capture Tool:": "Capture Tool:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "Fix Aspect",
        "Display Fit:": "Display Fit:",
        "Display Fit Contain": "Keep Ratio (Complete)",
        "Display Fit Cover": "Keep Ratio (Fill)",
        "Display Fit Stretch": "Stretch to Fill",
        "Stereo Output:": "Stereo Output:",
        "Window Preview": "Window Preview",
        "Theme:": "Theme:",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "Parallel Inference": "Parallel Inference",
        "Single Inference": "Single Inference",
        "Dual Inference": "Dual Inference",
        "Triple Inference": "Triple Inference",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "tooltip_window": "Select a window to capture",
        "tooltip_depth_model": "Depth model family. Recommended order: Distill-Any-Depth first as the general-purpose default; InfiniDepth second when its model family or weights are preferred. Choose Small/Base/Large within a family: larger variants usually improve detail but require more VRAM and inference time. InfiniDepth uses its own preprocessing and normally targets 512 input resolution with FP32.",
        "tooltip_model_size": "Model size within the selected family. Small is fastest and uses the least VRAM; Base is the recommended quality/speed balance; Large gives more depth detail but increases VRAM use and latency. Change size only when the matching model weights are installed.",
        "tooltip_depth_res": "Depth-map inference resolution. 518 is the recommended Distill-Any-Depth detail setting; lower values reduce inference time and VRAM use but can lose thin structures and edge detail. InfiniDepth normally uses 512 because its preprocessing patch size is different.",
        "tooltip_convergence": "Zero-parallax screen plane. Raise it in 0.05 steps when foreground objects pop out too much or show ghosting; lower it when the whole scene feels too flat or sits behind the screen.",
        "tooltip_dynamic_convergence_strength": "Dynamic convergence strength. 0.00 keeps manual Convergence; values above 0.00 enable automatic convergence and follow the measured depth target.",
        "tooltip_depth_strength": "Overall stereo depth intensity. Range is 0.00-0.50 in 0.05 steps. Use Standard / 0.25 as the baseline; raise it only when the scene feels flat, and lower it when foreground objects show ghosts, edges tear, or viewing feels uncomfortable.",
        "tooltip_depth_quick": "Quick fixed depth presets for everyday use: Soft, Standard, or Enhanced",
        "tooltip_stereo_preset": "Stereo preset. Traditional is fastest, Cinema uses quality_4k, Game uses fast_plus, and Image uses hq_4k; selecting a preset loads its advanced parameters.",
        "tooltip_stereo_quality": "Internal stereo synthesis quality selected by Stereo Mode. Fast modes use the low-latency path; quality_4k/hq_4k preserve more detail and improve 4K structure but cost more GPU time and memory. This is normally controlled by the preset rather than edited independently.",
        "tooltip_parallax_budget": "Maximum stereo parallax budget resolved from render size. Comfort is safest, Standard is the default, Strong and Extreme increase separation.",
        "tooltip_depth_separation": "Preset for Foreground, Midground, and Background Pop. Default uses 1.00/1.00/1.00; Standard uses 1.15/1.05/1.05; Strong uses 1.25/1.10/1.00; Weak uses 1.15/1.05/0.85.",
        "tooltip_max_shift": "Maximum horizontal shift as a ratio of image width; higher values increase stereo separation",
        "tooltip_temporal_strength": "Temporal smoothing strength for stereo output; higher values reduce flicker but can add lag",
        "tooltip_temporal": "Enable temporal stabilization between frames",
        "tooltip_scene_reset": "Scene-change threshold for resetting temporal history; lower values reset more often",
        "tooltip_auto_scene_reset": "Automatically reset temporal state when a scene cut is detected",
        "tooltip_edge_dilation": "Expands detected depth edges for occlusion handling",
        "tooltip_mask_feather": "Softens the occlusion fill mask; higher values reduce hard edge artifacts",
        "tooltip_edge_threshold": "Depth edge sensitivity; lower values detect more edges",
        "tooltip_hole_fill_mode": "Occlusion fill preset: Off / No Fill disables hole filling and sets Temporal Strength to 0; Balanced / Standard keeps the realtime speed-detail balance; Content Aware / Highest Quality uses directional content-aware fill and is much slower. Defaults: Cinema = Off / No Fill; Game = Off / No Fill; Image = Content Aware / Highest Quality.",
        "tooltip_anaglyph": "Color pair used when Display Mode is Anaglyph",
        "tooltip_cross_eyed": "Swap left and right eyes for cross-eyed viewing",
        "tooltip_parallel_inference": "Pipelines depth inference with SBS synthesis. Two workers are recommended; three workers use more VRAM and may not improve throughput.",
        "tooltip_advanced_stereo": "Show expert stereo/runtime parameters. Leave off for the simplified everyday UI.",
        "tooltip_advanced_device_options": "Show capture color controls, frame pacing, preview windows, and image enhancement controls.",
        "tooltip_xr_preview": "Show the desktop XR preview window while running OpenXR Link.",
        "tooltip_depth_pop": "Centered depth curve: output = 0.5 + sign(depth - 0.5) * abs(depth - 0.5) ** (1 / (1 + Depth Pop)). Use 0 for no change.",
        "tooltip_foreground_pop": "Increase or reduce parallax shift for nearby objects, mainly people, hands, and tabletop foreground.",
        "tooltip_midground_pop": "Increase or reduce parallax shift for the main subject layer, mainly characters, vehicles, and common focus areas.",
        "tooltip_background_pop": "Increase or reduce parallax shift for distant background, mainly sky, walls, and far buildings.",
        "tooltip_antialiasing": "Depth-map smoothing level. Higher values reduce jagged depth edges and flicker, but can soften fine geometry; keep low for games/realtime, raise when object edges shimmer or produce broken stereo borders.",
        "tooltip_device": "Inference device",
        "tooltip_capture_tool": "Capture backend. WindowsCaptureCUDA/ROCm uses the vendor GPU capture path when available; DesktopDuplication is the stable Windows desktop-duplication path; WindowsCapture is the Windows capture fallback; DXCamera is the DirectX fallback. On macOS, ScreenCaptureKit is the modern capture path and Quartz is the compatibility fallback. GPU paths can reduce CPU copies but require a matching driver and device.",
        "tooltip_run_mode": "Output mode. Local Viewer shows the result in a desktop window; 3D Monitor sends stereo output to a physical stereo display (Windows); OpenXR Link sends it to an OpenXR headset; Basic Network Streaming (MJPEG) favors simple, broadly compatible JPEG streaming; Advanced Network Streaming publishes compressed H.264/H.265 with WebRTC/RTSP/RTMP and automatically chooses GPU, Vulkan, hardware-FFmpeg, or CPU fallback paths.",
        "tooltip_display_mode": "Stereo layout. Half-SBS packs left/right views side by side at half horizontal eye resolution and is the most compatible network format; Full-SBS keeps full horizontal eye resolution but doubles the output width and decoding load; Half-TAB and Full-TAB are the equivalent top/bottom layouts. Depth Map shows the inferred depth for diagnosis; Anaglyph combines eyes into color channels for red/cyan glasses; Interleaved targets compatible line/column-interleaved displays; Mono shows one eye as a normal 2D image; Leia targets Leia light-field displays.",
        "tooltip_xr_headset": "Target headset preset shared by all output modes. It provides the target display resolution used by the screen sampling policy.",
        "tooltip_vsync": "Synchronize the local viewer to the display refresh rate",
        "tooltip_window_preview": "Open an additional standard effective-disparity debug window that stays open when unfocused and can be captured by screenshot tools. Depth Strength 0.00 is blue, 0.25 preserves the full far-blue / mid-purple / near-red depth gradient, and 0.50 is red. Physical-display output remains active.",
        "tooltip_target_fps": "Capture pacing target. Manual values stay fixed. In Auto mode, every 15 seconds the peak sustained SBS output sets capture to SBS + 5 FPS. If that dynamic target exceeds the input display refresh rate, Desktop2Stereo shows a warning without stopping. Sparse output windows are ignored.",
        "tooltip_render_policy": "Runtime render-size policy is fixed to 4K scale tiers.",
        "tooltip_render_scale": "Scale tier used only for 4K-class input. Output keeps the input aspect ratio; smaller input keeps its native size.",
        "tooltip_render_fixed_size": "Output size used when Render Policy is Fixed.",
        "tooltip_render_max_pixels": "Maximum output pixel count used by Dynamic render sizing.",
        "tooltip_render_min_dimension": "Minimum short-side dimension used by Dynamic render sizing.",
        "tooltip_render_align": "Output width and height alignment in pixels for runtime texture compatibility.",
        "tooltip_ctrl_model": "Controller model",
        "tooltip_env_model": "Room environment model",
        "tooltip_capture_mode": "Capture source. Monitor captures the selected full display and is the most predictable/high-throughput option; Window captures only the selected window and follows its client rectangle, but may be affected by minimized, occluded, protected, or resized windows.",
        "tooltip_monitor": "Input monitor used by Monitor capture mode. Select the display containing the desktop or application to convert; this is independent from the stereo output monitor.",
        "tooltip_stereo_monitor": "Physical display used for stereo output in 3D Monitor mode. Choose the monitor connected to the stereo panel; network, OpenXR, and local-window modes do not use this target.",
        "tooltip_display_fit": "How packed stereo is mapped to the local output display; changes apply while running. Keep Ratio (Complete) preserves every input pixel and adds bars when aspect ratios differ. Keep Ratio (Fill) preserves geometry and fills the display by centrally cropping both eyes identically. Stretch to Fill keeps all content and removes bars but distorts geometry when aspect ratios differ.",
        "tooltip_lang": "Interface language",
        "tooltip_theme": "Color theme",
        "tooltip_stream_settings": "Show or hide the streaming parameter panel. Hiding it does not clear or disable saved streaming settings.",
        "tooltip_stream_url": "Playback address generated from this PC's LAN IP, protocol, port, and stream key. Click the address to copy it.",
        "tooltip_stream_preview": "Open the generated HTTP playback address in the default browser. RTMP and RTSP require a compatible player, so browser preview is unavailable for them.",
        "tooltip_stream_quality": "JPEG quality for Basic Network Streaming (MJPEG): 50-100. Higher values preserve more detail but increase bandwidth and CPU work; lower values reduce bandwidth but show more blockiness. Advanced Network Streaming uses CRF and bitrate controls instead.",
        "tooltip_stream_proto": "Network protocol. WebRTC is recommended for low-latency headset browsers and supports automatic calibration; HLS/HLS M3U8 works in ordinary browsers but adds substantial latency; RTSP suits compatible low-latency players; RTMP is mainly for publishing to streaming servers and is not a browser playback format.",
        "tooltip_audio": "Desktop loopback/mix device sent with the video. Select the active system-output device; an empty or unavailable device produces video without captured desktop audio.",
        "tooltip_stream_port": "Listening port used by the streaming service. It must be free and allowed through the firewall. WebRTC automatic calibration additionally uses the next port (stream port + 1).",
        "tooltip_stream_key": "Stream path name, default 'live'. It becomes the final path segment of the playback URL and may contain only letters, digits, underscores, and hyphens (up to 64 characters).",
        "tooltip_crf": "For Advanced Network Streaming automatic calibration, this value is set from the calibrated safe target bitrate: 30 Mbps or higher -> CRF 20; 25-29 Mbps -> CRF 23; 21-24 Mbps -> CRF 26; 19-20 Mbps -> CRF 28; below 19 Mbps -> CRF 30. Lower CRF gives higher quality and requires more bitrate. Manual changes are used until the next automatic calibration.",
        "tooltip_audio_delay": "Audio timestamp offset in seconds (-10 to 10). Positive values delay audio; negative values advance audio. Adjust when sound leads or trails the picture.",
        "tooltip_video_backend": "Streaming encoder backend: Auto prefers available NVIDIA/AMD/Intel GPU paths and falls back through Vulkan, FFmpeg hardware, and software encoding; explicit choices select one backend and retain its fallback behavior.",
        "tooltip_stream_calibration_mode": "Transmission profile: Auto Calibration applies the saved end-to-end FPS and safe bitrate result; Manual ignores the saved calibration bitrate and uses the normal dynamic bitrate calculation.",
        "tooltip_stream_calibration_start": "Run a complete PC-to-headset WebRTC test. It measures encoding, LAN transmission, browser reception, and decoding, then saves a safe FPS and bitrate profile. Available only for Advanced Network Streaming with WebRTC.",
        "tooltip_nvfruc": "NVIDIA NvFRUC 2x frame generation. Requires an NVIDIA GPU with NVOFA support; GTX 1630 and standard GTX 1650 models using TU117 are not supported. RTX 2060 or newer is recommended. The program measures the complete pipeline, reserves performance headroom, and limits the effective output FPS when needed. It runs inside Desktop2Stereo without a separate window and adds about one base-frame of latency.",
        "err_crf": "CRF must be between 0-51",
        "err_audio_delay": "Audio Delay must be between -10 and 10",
        "err_stream_key": "Stream Key can only contain letters, digits, underscore, hyphen, max 64 chars",
        "err_start_failed": "Start failed: {}",
        "esc_stop": "Hold ESC 3s — stopping!",
        "exited_with_code": "Exited with code {}",
        "failed_save_yaml": "Failed to save YAML: {}",
        "stereo_parameters_saved": "Stereo parameters saved",
        "invalid_url_scheme": "Invalid URL scheme: {}",
        "error_preview": "Failed to preview: {}",
        "url_copied": "URL copied to clipboard",
        "Log panel title": "Run Log",
        "Log panel running title": "Running - live log",
        "Log panel error title": "Issue detected - check logs",
        "Clear logs": "Clear",
        "Hide log panel link": "Hide log <-",
        "Show log panel link": "Show log ->",
        "Report issue": "Report bug",
        "Open log file": "Open log",
        "Opening log file": "Opening log file",
        "Bug report copied to clipboard!": "Bug report copied to clipboard!",
        "Starting Desktop2Stereo...": "Starting Desktop2Stereo {}...",
        "Runtime stopped": "Stopped",
        "Downloading model...": "Downloading AI model...",
        "Exporting ONNX...": "Exporting ONNX file...",
        "Building TensorRT engine...": "Building TensorRT engine (this may take a while)...",
        "Error occurred": "Error occurred",
        "Preparing Flet package...": "Preparing Flet desktop client...",
        "Startup preparation complete": "Startup preparation complete.",
        "Startup preparation failed: {}": "Startup preparation failed: {}",
    },
    "CN": {
        "Monitor": "输入屏幕",
        "Window": "输入窗口",
        "Refresh": "刷新",
        "Show FPS": "显示帧率",
        "Debug Mode": "调试模式",
        "Convergence:": "会聚位置:",
        "Dynamic Convergence:": "动态会聚:",
        "Display Mode:": "显示模式:",
        "Depth Model:": "深度模型:",
        "Depth Strength:": "深度强度:",
        "Depth Quick:": "深度选项:",
        "Soft": "柔和",
        "Standard": "标准",
        "Enhanced": "增强",
        "Stereo Mode:": "立体模式:",
        "Traditional / Fastest": "传统 / 速度快",
        "Cinema": "电影 / 偏均衡",
        "Game / Low Latency": "游戏 / 低延迟",
        "Image  / High Quality": "图片 / 高质量",
        "Debug / Export": "调试 / 导出",
        "Synthetic View:": "立体质量:",
        "Parallax Budget:": "视差预算:",
        "Depth Separation:": "前后分离：",
        "separation_default": "默认",
        "separation_standard": "标准",
        "separation_strong": "增强",
        "separation_weak": "减弱",
        "comfort": "舒适",
        "standard": "标准",
        "strong": "强",
        "extreme": "极强",
        "fast": "最低",
        "fast_plus": "中等",
        "quality_4k": "较高",
        "hq_4k": "最高",
        "Temporal Strength:": "时域强度:",
        "Temporal": "时域稳定",
        "Scene Threshold:": "场景阈值:",
        "Auto Scene Reset": "自动场景重置",
        "Edge Dilation:": "边缘扩张:",
        "Mask Feather:": "遮罩羽化:",
        "Edge Threshold:": "边缘阈值:",
        "Hole Fill Mode:": "补洞模式:",
        "Off / No Fill": "关闭 / 不补洞",
        "Balanced": "均衡",
        "Balanced / Standard": "均衡 / 标准",
        "Content Aware / Highest Quality": "增强 / 高质量",
        "On": "开启",
        "Anaglyph:": "红蓝模式:",
        "Color Brightness:": "亮度:",
        "Color Contrast:": "对比度:",
        "Color Saturation:": "饱和度:",
    "Color Gamma:": "Gamma:",
    "Color Temperature:": "色温:",
    "Color Tint:": "色调:",
    "Projection Min LOD:": "最小 LOD:",
    "Projection Max LOD:": "最大 LOD:",
    "Projection MIP Bias:": "MIP 偏移:",
    "Projection RCAS:": "RCAS:",
    "Projection Min LOD tooltip": "限制所有模式统一输出阶段使用的最低 MIP 层级。建议值：0.00。",
    "Projection Max LOD tooltip": "限制所有模式统一输出阶段使用的最高 MIP 层级。建议值：0.35。",
    "Projection MIP Bias tooltip": "使统一输出采样偏向更清晰的 MIP 层级。建议值：-0.35。",
    "Projection RCAS tooltip": "在统一 MIP 过滤和缩放后应用 RCAS 锐化。建议值：0.50。",
        "Cross Eyed": "交叉眼",
        "Picture": "画面",
        "Depth": "深度",
        "Glow": "辉光",
        "Room": "房间",
        "Screen": "屏幕",
        "Render Scale": "渲染比例",
        "Brightness": "亮度",
        "Contrast": "对比度",
        "Saturation": "饱和度",
        "Gamma": "Gamma",
        "Temperature": "色温",
        "Tint": "色调",
        "Min LOD": "最小 LOD",
        "Max LOD": "最大 LOD",
        "MIP Bias": "MIP 偏移",
        "RCAS": "RCAS 锐化",
        "Reset to default values": "重置为默认值",
        "Depth strength": "深度强度",
        "2D / 3D": "2D / 3D切换",
        "Cross eyed": "交叉眼",
        "Previous seat": "上一个座位",
        "Next seat": "下一个座位",
        "Front": "前排",
        "Middle": "中排",
        "Back": "后排",
        "Seat height": "座位高度",
        "Scene brightness": "场景亮度",
        "Screen reflection light": "屏幕反射光",
        "Screen size": "屏幕大小",
        "Screen height": "屏幕高度",
        "Screen distance": "屏幕距离",
        "Flat": "直面",
        "Subtle": "微曲",
        "Medium": "中曲",
        "Deep": "重曲",
        "Surround Glow": "环绕辉光",
        "Veil": "光幕",
        "OFF": "关闭",
        "Video appearance": "视频画面",
        "Stereo depth": "立体深度",
        "Glow effects": "辉光特效",
        "Scene controls": "场景设置",
        "Screen geometry": "屏幕设置",
        "Advanced Stereo": "显示高级立体参数",
        "Advanced Device Options": "高级选项",
        "Depth Resolution:": "深度细节:",
        "Anti-aliasing:": "抗锯齿值:",
        "Depth Pop:": "深度弹出:",
        "Foreground Pop:": "前景视差:",
        "Midground Pop:": "中景视差:",
        "Background Pop:": "背景视差:",
        "FP16": "FP16",
        "Inference Acceleration:": "推理加速:",
        "Recompile TensorRT": "重译TensorRT",
        "Recompile CoreML": "重译CoreML",
        "Recompile OpenVINO": "重译OpenVINO",
        "Stop": "停止",
        "Computing Device:": "计算设备:",
        "Reset": "重置",
        "Run": "运行",
        "Set Language:": "设置语言:",
        "Error": "错误",
        "Warning": "警告",
        "Display refresh warning": "输出显示器刷新率过低",
        "Input display refresh warning": "输入显示器刷新率低于捕获目标",
        "display_refresh_warning_body": "SBS 输出显示器当前仅 {refresh_hz} Hz，低于实测 {sbs_fps:.1f} FPS 或建议最低 60 Hz。请在 Windows 显示设置或显卡控制面板中提高输出显示器刷新率。",
        "input_refresh_warning_body": "输入显示器当前仅 {refresh_hz} Hz，低于动态捕获目标 {capture_target} FPS。请提高输入显示器刷新率，或手动降低捕获帧率。",
        "display_refresh_warning_continuing": "Desktop2Stereo 将继续运行；此提醒不会停止当前输出。",
        "Saved": "运行Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "未安装PyYAML，无法保存YAML文件。",
        "Settings saved to settings.yaml": "设置已保存到 settings.yaml",
        "Failed to save settings.yaml:": "保存 settings.yaml 失败：",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "无法获取显示器列表。\n回退到索引1和2。",
        "Loaded settings.yaml at startup": "启动时已加载 settings.yaml",
        "Running": "运行中...（长按ESC 3秒停止）",
        "Stopped": "已停止。",
        "Countdown": "设置已保存到 settings.yaml，启动中...",
        "A thread already running!": "一个进程已经运行！",
        "No windows found": "未找到窗口",
        "err_refresh_window": "刷新窗口列表失败：{}",
        "Selected input window:": "已选择输入窗口:",
        "Selected input monitor:": "已选择输入显示器 :",
        "Run Mode:": "运行模式:",
        "Stream Settings": "推流设置",
        "Local Viewer": "本地查看",
        "MJPEG Streamer": "低级网络推流",
        "RTMP Streamer": "高级网络推流",
        "Intel QSV (D3D11)": "Intel QSV（D3D11）",
        "full_sbs_stream_advisory": "Full-SBS 将保持原尺寸并使用 H.265/HEVC 推流；如需更广泛的浏览器兼容性和更低的解码负载，建议选择 Half-SBS。",
        "Stream Protocol:": "流协议:",
        "Stream Key": "推流密钥:",
        "Stereo Mix": "混音设备:",
        "CRF": "恒定质量:",
        "Audio Delay": "音频延迟 (秒):",
        "system": "系统",
        "blue": "蓝色",
        "green": "绿色",
        "red": "红色",
        "purple": "紫色",
        "orange": "橙色",
        "teal": "青色",
        "pink": "粉色",
        "grey": "灰色",
        "Lossless Scaling Support": "NvFRUC 补帧",
        "3D Monitor": "3D显示器",
        "OpenXR Link": "OpenXR串流",
        "Headset Model:": "头显型号:",
        "VSync": "垂直同步",
        "Capture FPS:": "捕获帧率:",
        "Render Policy:": "渲染策略:",
        "Render Scale:": "4K缩放档:",
        "Render Fixed Size:": "固定尺寸:",
        "Render Pixel Cap:": "像素上限:",
        "Render Min Side:": "最短边:",
        "Render Align:": "尺寸对齐:",
        "Native": "原生",
        "Scaled": "缩放",
        "Fixed": "固定",
        "Dynamic": "动态",
        "Upscaler:": "画面增强:",
        "Upscaler Sharpness:": "增强锐度:",
        "Auto": "自动",
        "Off": "关闭",
        "Streamer Port:": "推流端口:",
        "Streamer URL": "推流网址:",
        "Preview": "预览",
        "Stream Quality:": "推流质量:",
        "Transmission Profile:": "传输配置:",
        "Auto Calibration": "自动校准",
        "Manual": "手动设置",
        "Start Calibration": "开始校准",
        "Not calibrated": "尚未校准",
        "Automatic Network and Performance Calibration": "自动网络与性能校准",
        "Cancel Calibration": "取消校准",
        "Detect Firewall Rules": "检测防火墙规则",
        "Close": "关闭",
        "calibration_open_headset_url": "请在头显浏览器中打开以下地址，并保持页面位于前台：",
        "calibration_waiting": "正在等待头显打开测试页面……",
        "calibration_settling": "正在等待新码率稳定……",
        "calibration_testing": "正在进行 {fps} FPS 端到端测试……",
        "calibration_reconnecting": "正在切换发送档位，头显页面将自动重连……",
        "calibration_complete": "校准完成",
        "calibration_requires_advanced": "自动校准需要使用高级网络推流，并选择 WebRTC。",
        "calibration_requires_webrtc": "自动校准需要选择 WebRTC 协议。",
        "calibration_port_unavailable": "自动校准需要使用 WebRTC 端口之后的一个端口，请降低推流端口。",
        "calibration_required_before_run": "当前没有有效的自动校准结果，或设置已经变化，启动前必须先完成自动校准。",
        "calibration_profile_stale": "设置变化，请重新校准",
        "calibration_bandwidth_insufficient": "当前网络带宽不足以支持当前分辨率，请降低分辨率后重新校准。",
        "calibration_receiver_metrics": "解码 {decoded:.1f} FPS · 码率 {bitrate:.1f} Mbps · 丢帧 {dropped} · 冻结 {freeze} · 丢包 {lost} · 抖动 {jitter:.1f} ms",
        "calibration_firewall_remove_failed": "Windows 防火墙正在阻止项目内置 Python 的入站连接（{protocol}），但阻止规则删除失败。请以管理员身份运行 Desktop2Stereo 后重试。",
        "calibration_firewall_probe_failed": "Windows 防火墙检测失败，自动校准无法安全继续：{error}",
        "calibration_firewall_manual_hint": "如果头显访问 {port} 端口页面无响应，请点击“检测防火墙规则”。",
        "calibration_firewall_checking": "正在检测 Windows 防火墙规则……",
        "calibration_firewall_no_blocks": "未发现针对项目内置 Python 的入站阻止规则。",
        "calibration_firewall_removed": "已删除匹配的 Windows 防火墙阻止规则，请重新打开头显页面。",
        "calibration_profile_summary": "{fps} FPS · {target} Mbps",
        "calibration_result_stable": "网络稳定上限：{network_max} Mbps，安全码率：{safe_target} Mbps，帧率 {fps} FPS。",
        "calibration_result_limited": "网络校准在 {network_max} Mbps 未通过，请降低分辨率后重新校准。",
        "calibration_applied": "已应用校准结果：{fps} FPS，{target} Mbps",
        "calibration_result": "网络稳定上限：{network_max} Mbps · 安全码率：{target} Mbps · 峰值 {peak} Mbps · {fps} FPS",
        "calibration_recalibrate_hint": "以下情况也建议重新校准：\n• 更换路由器、Wi-Fi 频段或头显连接位置。\n• 更换 Pico、Quest 或其他头显。\n• 更换头显浏览器，或者 Wolvic 在 Gecko 与 Chromium 版本之间切换。\n• 浏览器或头显系统完成较大版本升级。\n• 改变输出分辨率、编码格式或画质设置。\n• 电脑 GPU、驱动或性能模式发生变化。",
        "Host": "主机:",
        "Invalid port number (1-65535)": "端口号无效 (必须介于1-65535之间)",
        "Invalid port number": "端口必须是数字",
        "Please select a window before running in Window capture mode": "请在窗口捕获模式下选择一个窗口再运行",
        "Local Viewer requires a second display": "本地查看模式至少需要两个显示屏。请安装或打开虚拟显示屏，刷新显示屏列表后重试。",
        "Selected input display is unavailable": "已选择的输入显示器已断开或当前不可用。请刷新显示器列表并重新选择可用的输入显示器。",
        "Selected stereo output display is unavailable": "已选择的立体输出显示器已断开或当前不可用。请刷新显示器列表并重新选择可用的输出显示器。",
        "The selected window no longer exists. Please refresh and select a valid window.": "所选窗口已不存在。请刷新并选择一个有效的窗口。",
        "Failed to stop process on exit:": "退出时停止进程失败：",
        "Failed to stop process:": "停止进程失败：",
        "Failed to run process:": "运行进程失败：",
        "Failed to load settings.yaml:": "加载 settings.yaml 失败：",
        "Opening URL in browser": "正在浏览器中打开网址",
        "Controller:": "手柄模型：",
        "Environment:": "房间模型：",
        "Capture Tool:": "捕获工具:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "锁定比例",
        "Display Fit:": "显示适配:",
        "Display Fit Contain": "保持比例（完整）",
        "Display Fit Cover": "保持比例（铺满）",
        "Display Fit Stretch": "拉伸铺满",
        "Stereo Output:": "立体输出:",
        "Window Preview": "窗口预览",
        "Theme:": "主题颜色:",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "Parallel Inference": "并行推理",
        "Single Inference": "单路推理",
        "Dual Inference": "两路推理",
        "Triple Inference": "三路推理",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "tooltip_window": "选择要捕获的窗口",
        "tooltip_depth_model": "深度模型家族。推荐顺序：首先使用 Distill-Any-Depth 作为通用默认模型；其次选择 InfiniDepth，适合已经准备好对应模型权重或偏好该模型家族的场景。每个家族都可选择 Small/Base/Large：更大的模型通常细节更好，但显存占用和推理耗时更高。InfiniDepth 使用独立预处理，通常配合 512 输入分辨率和 FP32。",
        "tooltip_model_size": "当前模型家族内的模型大小。Small 速度最快、显存占用最低；Base 是质量和速度的推荐平衡；Large 深度细节更多，但显存和延迟更高。只有安装了对应权重时才切换大小。",
        "tooltip_depth_res": "深度图推理分辨率。Distill-Any-Depth 推荐使用 518，以获得更稳定的深度边缘、细小结构和立体细节；降低数值可减少推理耗时和显存占用，但更容易丢失轮廓层次。InfiniDepth 通常使用 512，因为它的预处理 patch 尺寸不同。",
        "tooltip_convergence": "零视差屏幕平面。前景太突出或出现重影时，每次提高 0.05；画面整体太平、都贴在屏幕后方时，每次降低 0.05。",
        "tooltip_dynamic_convergence_strength": "动态会聚强度。0.00 表示使用手动会聚位置；大于 0.00 时启用动态会聚并跟随测得的深度目标。",
        "tooltip_depth_strength": "整体立体深度强度。范围 0.00-0.50，步进 0.05；建议以标准档 0.25 为基准，画面太平时再上调，前景重影、边缘撕裂或观看不舒服时下调。",
        "tooltip_depth_quick": "给普通用户使用的固定深度档位：柔和、标准、增强",
        "tooltip_stereo_preset": "立体预设模式。传统速度快，电影使用 quality_4k，游戏使用 fast_plus，图片使用 hq_4k；选择模式会加载对应高级参数。",
        "tooltip_stereo_quality": "由立体模式选择的内部立体合成质量。速度模式使用低延迟路径；quality_4k/hq_4k 会保留更多细节、改善 4K 结构，但需要更多 GPU 时间和显存。通常由预设自动控制，不建议单独修改。",
        "tooltip_parallax_budget": "根据渲染尺寸解析最大视差预算。舒适最稳，标准为默认，强和极强会增加立体分离。",
        "tooltip_depth_separation": "一键设置前景/中景/背景视差：默认为 1.00/1.00/1.00，标准为 1.15/1.05/1.05，增强为 1.25/1.10/1.00，减弱为 1.15/1.05/0.85。",
        "tooltip_max_shift": "水平位移占画面宽度的比例；越高立体分离越强",
        "tooltip_temporal_strength": "时域平滑强度；越高越稳定，但可能增加拖影或延迟",
        "tooltip_temporal": "启用帧间时域稳定",
        "tooltip_scene_reset": "场景变化重置阈值；越低越容易触发重置",
        "tooltip_auto_scene_reset": "检测到场景切换时自动重置时域历史",
        "tooltip_edge_dilation": "扩张深度边缘区域，用于遮挡和补洞处理",
        "tooltip_mask_feather": "柔化遮挡补洞遮罩；数值越高越能减轻硬边重影",
        "tooltip_edge_threshold": "深度边缘检测敏感度；越低检测到的边缘越多",
        "tooltip_hole_fill_mode": "遮挡补洞预设：关闭 / 不补洞会禁用所有补洞算法，并将时域强度设为 0；均衡 / 标准保留实时速度和细节折中；增强 / 高质量使用方向内容感知补洞，速度会明显变慢。默认对应：电影模式 = 关闭 / 不补洞；游戏模式 = 关闭 / 不补洞；图片模式 = 增强 / 高质量。",
        "tooltip_anaglyph": "显示模式为红蓝/补色时使用的颜色组合",
        "tooltip_cross_eyed": "交换左右眼，用于交叉眼观看",
        "tooltip_parallel_inference": "让深度推理与 SBS 合成流水并行。推荐两路；三路会占用更多显存，且不一定继续提升吞吐量。",
        "tooltip_advanced_stereo": "显示专家级立体和运行时参数；普通使用建议保持关闭。",
        "tooltip_advanced_device_options": "显示捕捉设备颜色调节、捕获帧率、画面预览窗口、本地垂直同步和画面增强选项。",
        "tooltip_xr_preview": "运行 OpenXR Link 时显示桌面 XR 画面预览窗口。",
        "tooltip_depth_pop": "居中深度曲线：output = 0.5 + sign(depth - 0.5) * abs(depth - 0.5) ** (1 / (1 + Depth Pop))。0 表示不改变深度曲线。",
        "tooltip_foreground_pop": "增强/减弱近处物体的位移，主要影响人物、手、桌面前景。",
        "tooltip_midground_pop": "增强/减弱画面主体层的位移，主要影响角色、车辆、常见焦点区域。",
        "tooltip_background_pop": "增强/减弱远处背景的位移，主要影响天空、墙面、远景建筑。",
        "tooltip_antialiasing": "深度图平滑级别。数值越高越能减少深度边缘锯齿和闪烁，但会软化细节；游戏和实时观看保持较低，物体边缘闪烁或立体边界破碎时再上调。",
        "tooltip_device": "计算设备",
        "tooltip_capture_tool": "捕获后端。WindowsCaptureCUDA/ROCm 在可用时使用厂商 GPU 捕获路径；DesktopDuplication 是稳定的 Windows 桌面复制路径；WindowsCapture 是 Windows 捕获回退；DXCamera 是 DirectX 回退。macOS 优先使用 ScreenCaptureKit，Quartz 用于兼容回退。GPU 路径可减少 CPU 复制，但要求匹配的驱动和设备；不可用时应选择兼容回退。",
        "tooltip_run_mode": "运行模式。本地查看：在桌面窗口显示结果；3D Monitor：Windows 下输出到物理立体显示器；OpenXR Link：输出到 OpenXR 头显；低级网络推流（MJPEG）：使用简单、兼容性广的 JPEG 推流；高级网络推流：通过 WebRTC/RTSP/RTMP 发布 H.264/H.265，并自动按能力选择 GPU、Vulkan、FFmpeg 硬件或 CPU 回退路径。",
        "tooltip_display_mode": "立体布局。Half-SBS：左右眼并排，每只眼使用一半水平分辨率，网络兼容性最好；Full-SBS：保留完整水平眼分辨率，但输出宽度和解码负载约翻倍；Half-TAB 和 Full-TAB 是对应的上下排列布局。Depth Map 用于检查深度结果；Anaglyph 将左右眼编码到颜色通道，配合红青眼镜；Interleaved 用于兼容行/列交错立体屏；Mono 只显示单眼画面作为普通 2D；Leia 用于 Leia 光场显示器。",
        "tooltip_xr_headset": "所有输出模式共用的目标头显预设，用于向画面采样策略提供目标显示分辨率。",
        "tooltip_vsync": "将本地查看窗口同步到显示器刷新率，关闭可用于帧率对比测试",
        "tooltip_window_preview": "额外打开允许截图的普通有效视差调试窗口，失去焦点后不会关闭，但可以被其他窗口覆盖：深度强度 0.00 显示蓝色，0.25 保留远处蓝色—中间紫色—近处红色的完整深度渐变，0.50 显示红色；物理显示器全屏输出保持不变。",
        "tooltip_target_fps": "捕捉帧率节流目标。手动档位保持固定；自动模式每 15 秒按持续输出窗口内的最高 SBS 帧率评估性能，捕捉帧率设置为 SBS + 5 FPS。动态目标高于输入显示器刷新率时会弹窗提醒，但不会停止运行；输出帧数不足的稀疏窗口不会参与校准。",
        "tooltip_render_policy": "运行时渲染尺寸策略固定为 4K 缩放档位。",
        "tooltip_render_scale": "仅 4K 级输入使用的缩放档位；输出保持输入宽高比，低于 4K 的输入保持原生尺寸。",
        "tooltip_render_fixed_size": "渲染策略为固定时使用的输出尺寸。",
        "tooltip_render_max_pixels": "动态渲染尺寸使用的最大输出像素数。",
        "tooltip_render_min_dimension": "动态渲染尺寸使用的最短边下限。",
        "tooltip_render_align": "运行时纹理兼容所需的输出宽高像素对齐。",
        "tooltip_ctrl_model": "手柄型号",
        "tooltip_env_model": "房间环境模型",
        "tooltip_capture_mode": "捕获来源。屏幕模式捕获选定显示器的完整画面，吞吐最稳定；窗口模式只捕获选定窗口的客户区，会受到窗口最小化、被遮挡、受保护内容和窗口尺寸变化影响。",
        "tooltip_monitor": "屏幕捕获模式使用的输入显示器。选择包含桌面或应用程序的显示器；它与 3D Monitor 的立体输出显示器相互独立。",
        "tooltip_stereo_monitor": "3D Monitor 模式使用的物理立体输出显示器。选择连接立体屏的显示器；网络推流、OpenXR 和本地窗口模式不会使用此目标。",
        "tooltip_display_fit": "保持比例：控制打包立体画面如何映射到本地输出显示器，运行中切换即可生效。保持比例（完整）：保留全部画面，输入与输出比例不同时产生黑边；保持比例（铺满）：保持几何比例，对左右眼执行相同的中央裁剪后铺满；拉伸铺满：保留全部内容并消除黑边，但比例不同时会造成画面变形。",
        "tooltip_lang": "界面语言",
        "tooltip_theme": "主题颜色",
        "tooltip_stream_settings": "显示或隐藏推流参数设置框。隐藏只折叠界面，不会清除或停用已经保存的推流参数。",
        "tooltip_stream_url": "根据本机局域网 IP、推流协议、端口和推流密钥自动生成的播放地址；单击地址可复制。",
        "tooltip_stream_preview": "使用默认浏览器打开生成的 HTTP 播放地址。RTMP 和 RTSP 需要兼容播放器，因此这两种协议不提供浏览器预览。",
        "tooltip_stream_quality": "低级网络推流（MJPEG）的 JPEG 质量，范围 50-100。数值越高细节越好，但带宽和 CPU 占用更高；数值越低更省带宽，但块状伪影更多。高级网络推流使用 CRF 和码率控制，不受此选项影响。",
        "tooltip_stream_proto": "网络协议。WebRTC 推荐用于低延迟头显浏览器，并支持自动校准；HLS/HLS M3U8 可在普通浏览器播放，但延迟明显更高；RTSP 适合兼容的低延迟播放器；RTMP 主要用于发布到直播服务器，不是浏览器播放格式。",
        "tooltip_audio": "随视频发送的桌面回环/混音设备。请选择当前系统输出对应的设备；留空或设备不可用时只推送视频，不采集桌面声音。",
        "tooltip_stream_port": "推流服务监听端口，必须未被占用并允许通过防火墙。高级网络推流的 WebRTC 自动校准还会使用下一个端口（推流端口 + 1）。",
        "tooltip_stream_key": "推流路径名称，默认 live，会作为播放网址的最后一级路径；只能使用字母、数字、下划线和连字符，最长 64 个字符。",
        "tooltip_crf": "高级网络推流自动校准后，会根据校准得到的安全目标码率自动设置：30 Mbps 及以上建议 CRF 20；25-29 Mbps 建议 CRF 23；21-24 Mbps 建议 CRF 26；19-20 Mbps 建议 CRF 28；低于 19 Mbps 不建议继续提高 CRF，自动设置为 CRF 30。CRF 越低画质越高、所需码率越大；手动修改会一直生效，直到下一次自动校准。",
        "tooltip_audio_delay": "音频时间戳偏移，范围 -10 到 10 秒。正数延后音频，负数提前音频；声音领先画面时增大，声音落后画面时减小。",
        "tooltip_video_backend": "推流编码后端：自动模式优先选择可用的 NVIDIA/AMD/Intel GPU 路径，并依次回退到 Vulkan、FFmpeg 硬件和软件编码；显式选择会固定后端并保留其回退行为。",
        "tooltip_stream_calibration_mode": "传输配置：自动校准会应用已保存的端到端 FPS 和安全码率结果；手动设置会忽略校准码率，改用程序的常规动态码率计算。",
        "tooltip_stream_calibration_start": "运行一次电脑到头显浏览器的完整 WebRTC 闭环测试，检查编码、局域网传输、浏览器接收和解码能力，并保存安全帧率与码率；仅支持高级网络推流的 WebRTC 协议。",
        "tooltip_nvfruc": "NVIDIA NvFRUC 2 倍补帧。需要具备 NVOFA 的 NVIDIA 显卡；采用 TU117 芯片的 GTX 1630 和普通 GTX 1650 不支持。推荐 RTX 2060 或更高型号。程序会检测完整处理链路、预留性能余量，并在需要时限制实际输出帧率。补帧在 Desktop2Stereo 进程内完成，不需要单独窗口，但会增加约一个基础帧的延迟。",
        "err_crf": "CRF 必须是 0-51 之间的整数",
        "err_audio_delay": "Audio Delay 必须是 -10 到 10 之间的数值",
        "err_stream_key": "Stream Key 只能包含字母、数字、下划线和连字符，最长 64 字符",
        "err_start_failed": "启动失败: {}",
        "esc_stop": "长按ESC 3秒停止",
        "exited_with_code": "退出码 {}",
        "failed_save_yaml": "保存 YAML 失败: {}",
        "stereo_parameters_saved": "立体参数已保存",
        "invalid_url_scheme": "无效 URL 协议: {}",
        "error_preview": "打开浏览器失败: {}",
        "url_copied": "已复制网址到剪贴板",
        "Log panel title": "运行日志",
        "Log panel running title": "运行中 - 实时日志",
        "Log panel error title": "检测到异常 - 请查看日志",
        "Clear logs": "清空",
        "Hide log panel link": "隐藏log<-",
        "Show log panel link": "显示log->",
        "Report issue": "反馈bug",
        "Open log file": "打开日志",
        "Opening log file": "正在打开日志",
        "Bug report copied to clipboard!": "异常反馈信息已复制到剪贴板！",
        "Starting Desktop2Stereo...": "正在启动 Desktop2Stereo {}...",
        "Runtime stopped": "已停止",
        "Downloading model...": "正在下载 AI 模型...",
        "Exporting ONNX...": "正在导出 ONNX 文件...",
        "Building TensorRT engine...": "正在编译 TensorRT 引擎（可能需要较长时间）...",
        "Error occurred": "出现异常",
        "Preparing Flet package...": "正在准备 Flet 桌面客户端...",
        "Startup preparation complete": "启动准备已完成。",
        "Startup preparation failed: {}": "启动准备失败: {}",
    }
}

LOCALE_ALIASES = MappingProxyType({
    "EN": "EN",
    "EN_US": "EN",
    "EN-US": "EN",
    "CN": "CN",
    "ZH": "CN",
    "ZH_CN": "CN",
    "ZH-CN": "CN",
    "ZH_HANS": "CN",
    "ZH-HANS": "CN",
})

SUPPORTED_LOCALES = tuple(MESSAGE_CATALOGS.keys())
UI_MESSAGES = MESSAGE_CATALOGS
UI_TEXTS = UI_MESSAGES


class CatalogTranslation(gettext.NullTranslations):
    def __init__(self, catalog, fallback=None):
        super().__init__()
        self._catalog = catalog
        self._fallback = fallback

    def gettext(self, message):
        if message in self._catalog:
            return self._catalog[message]
        if self._fallback is not None:
            return self._fallback.gettext(message)
        return message

    def ngettext(self, msgid1, msgid2, n):
        message = msgid1 if n == 1 else msgid2
        return self.gettext(message)


_EN_TRANSLATION = CatalogTranslation(MESSAGE_CATALOGS["EN"])
_LOCALE_TRANSLATIONS = {
    lang: CatalogTranslation(catalog, fallback=_EN_TRANSLATION if lang != "EN" else None)
    for lang, catalog in MESSAGE_CATALOGS.items()
}


def normalize_locale(locale):
    key = str(locale or DEFAULT_LOCALE).replace(" ", "_").upper()
    return LOCALE_ALIASES.get(key, key if key in MESSAGE_CATALOGS else DEFAULT_LOCALE)


def is_supported_locale(locale):
    return normalize_locale(locale) in MESSAGE_CATALOGS


def get_translation(locale=DEFAULT_LOCALE):
    return _LOCALE_TRANSLATIONS[normalize_locale(locale)]


def get_messages(locale=DEFAULT_LOCALE):
    return MESSAGE_CATALOGS[normalize_locale(locale)]


def gettext_for(locale, message):
    return get_translation(locale).gettext(message)

STEREO_QUALITY_KEYS = ("fast", "fast_plus", "quality_4k", "hq_4k")
PARALLAX_BUDGET_KEYS = ("comfort", "standard", "strong", "extreme")
DEPTH_SEPARATION_KEYS = ("default", "standard", "strong", "weak")
DEPTH_SEPARATION_LABELS = {
    "default": "separation_default",
    "standard": "separation_standard",
    "strong": "separation_strong",
    "weak": "separation_weak",
}
PARALLEL_INFERENCE_LABELS = {
    1: "Single Inference",
    2: "Dual Inference",
    3: "Triple Inference",
}


def parallel_inference_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[label] for label in PARALLEL_INFERENCE_LABELS.values()]


def parallel_inference_to_display(workers, locale=DEFAULT_LOCALE):
    try:
        workers = int(workers)
    except (TypeError, ValueError):
        workers = 1
    label = PARALLEL_INFERENCE_LABELS.get(workers, PARALLEL_INFERENCE_LABELS[1])
    return get_messages(locale).get(label, label)


def display_to_parallel_inference_workers(value):
    text = str(value or "")
    for workers, label in PARALLEL_INFERENCE_LABELS.items():
        if text == label:
            return workers
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(label):
                return workers
    return 1


def stereo_quality_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[key] for key in STEREO_QUALITY_KEYS]


def stereo_quality_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "quality_4k")
    messages = get_messages(locale)
    return messages.get(key, messages["quality_4k"])

def display_to_stereo_quality(value):
    text = str(value or "")
    for key in STEREO_QUALITY_KEYS:
        if text == key:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(key):
                return key
    return "quality_4k"


def parallax_budget_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[key] for key in PARALLAX_BUDGET_KEYS]


def parallax_budget_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "standard")
    messages = get_messages(locale)
    return messages.get(key, messages.get("standard", "Standard"))


def display_to_parallax_budget(value):
    text = str(value or "")
    for key in PARALLAX_BUDGET_KEYS:
        if text == key:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(key):
                return key
    return "standard"


def depth_separation_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[DEPTH_SEPARATION_LABELS[key]] for key in DEPTH_SEPARATION_KEYS]


def depth_separation_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "standard")
    messages = get_messages(locale)
    label = DEPTH_SEPARATION_LABELS.get(key, DEPTH_SEPARATION_LABELS["standard"])
    return messages.get(label, label)


def display_to_depth_separation(value):
    text = str(value or "")
    for key, label in DEPTH_SEPARATION_LABELS.items():
        if text == key or text == label:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(label):
                return key
    return "standard"


HOLE_FILL_MODE_KEYS = ("none", "balanced", "quality")
HOLE_FILL_MODE_LABELS = {
    "none": "Off / No Fill",
    "balanced": "Balanced / Standard",
    "quality": "Content Aware / Highest Quality",
}
HOLE_FILL_MODE_LEGACY_LABELS = {
    "none": ("Off", "Disabled", "None", "No Fill"),
    "balanced": ("Balanced",),
    "quality": ("Quality", "Content Aware", "Directional"),
}


def hole_fill_mode_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[HOLE_FILL_MODE_LABELS[key]] for key in HOLE_FILL_MODE_KEYS]


def hole_fill_mode_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "balanced")
    messages = get_messages(locale)
    label = HOLE_FILL_MODE_LABELS.get(key, HOLE_FILL_MODE_LABELS["balanced"])
    return messages.get(label, label)


def display_to_hole_fill_mode(value):
    text = str(value or "")
    for key, label in HOLE_FILL_MODE_LABELS.items():
        labels = (label, *HOLE_FILL_MODE_LEGACY_LABELS.get(key, ()))
        if text == key or text in labels:
            return key
        for locale in SUPPORTED_LOCALES:
            messages = get_messages(locale)
            if any(text == messages.get(candidate) for candidate in labels):
                return key
    return "balanced"
