@echo off
chcp 65001>nul
cls
title Desktop2Stereo
set "SRC_DIR=%~dp0"
set "APP_DIR=%SRC_DIR%desktop2stereo"
set "PYTHON_EXE=%SRC_DIR%python3_cuda\python.exe"
set "LOG_DIR=%APP_DIR%\logs"
set "GUI_READY_FILE=%LOG_DIR%\gui_ready.flag"
set "LAUNCH_STDOUT=%LOG_DIR%\launcher_stdout.log"
set "LAUNCH_STDERR=%LOG_DIR%\launcher_stderr.log"
set "APP_LOG=%LOG_DIR%\desktop2stereo.log"
set "GUI_PID_FILE=%LOG_DIR%\gui.pid"
set "GUI_LOCK_FILE=%LOG_DIR%\gui.instance.lock"
set "GUI_ACTIVATION_FILE=%LOG_DIR%\gui.activate.request"

rem Formal launcher must not inherit diagnostic overrides from a reused shell.
set "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC="
set "D2S_FILAMENT_EYE_DIAGNOSTIC="
set "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK="
set "D2S_FILAMENT_MULTIVIEW="
set "D2S_FILAMENT_SHADER_DUMP_DIR="
set "D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC="
set "D2S_FILAMENT_PROJECTION_ONLY="
set "D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC="
set "D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC="
set "D2S_OPENXR_DISABLE_TOOL_QUADS="
set "D2S_OPENXR_VULKAN_MULTIVIEW_EYE_DIAGNOSTIC="
set "D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC="
set "D2S_VULKAN_PROJECTION_QUALITY_CHAIN="
set "D2S_ROCM_DISABLE_OPENXR_OVERLAYS="
set "D2S_ROCM_ENABLE_OPENXR_OVERLAYS="
set "D2S_CONTROLLER_MODEL="
set "D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE="

if not exist "%PYTHON_EXE%" (
    echo [Error] [EN] Python not found at %PYTHON_EXE%
    echo [Error] [CN] 初始化 Python 环境 ...失败，未找到 %PYTHON_EXE%
    pause
    exit /b 1
)

if not exist "%APP_DIR%\gui\gui.py" (
    echo [Error] [EN] %APP_DIR%\gui\gui.py not found.
    echo [Error] [CN] 初始化 Python 环境 ...失败，未找到 %APP_DIR%\gui\gui.py
    pause
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "PYTHONPATH=%APP_DIR%"
"%PYTHON_EXE%" -c "import sys; from gui.paths import GUI_LOCK_FILE, GUI_PID_FILE, GUI_ACTIVATION_FILE; from gui.single_instance import GUIInstanceLock, request_gui_activation; lock=GUIInstanceLock.acquire(GUI_LOCK_FILE, GUI_PID_FILE, mutex_scope='GUI'); (lock.close(), sys.exit(0)) if lock else (request_gui_activation(GUI_ACTIVATION_FILE), sys.exit(75))"
if "%ERRORLEVEL%"=="75" (
    echo [Desktop2Stereo] GUI is already running; requested it to come to the front.
    exit /b 0
)

if exist "%GUI_READY_FILE%" del /f /q "%GUI_READY_FILE%" >nul 2>nul
if exist "%LAUNCH_STDOUT%" del /f /q "%LAUNCH_STDOUT%" >nul 2>nul
if exist "%LAUNCH_STDERR%" del /f /q "%LAUNCH_STDERR%" >nul 2>nul
if exist "%APP_LOG%" type nul > "%APP_LOG%"

echo [1/2] [EN] Starting Desktop2Stereo GUI first ...
echo       [CN] 优先显示 Desktop2Stereo GUI ...
echo [2/2] [EN] Waiting for GUI ready signal. Startup details continue in the GUI log panel.
echo       [CN] 正在等待 GUI 就绪标志。首次运行的编译和加载过程将在 GUI 右侧日志窗口显示。
echo       [EN] This CMD window will close automatically after the GUI reports ready.
echo       [CN] 收到 GUI 就绪标志后，此 CMD 窗口会自动关闭。

set "PYTHON_EXE=%PYTHON_EXE%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:PYTHONPATH='%APP_DIR%'; $env:PYTHON_EXE='%PYTHON_EXE%'; $p = Start-Process -FilePath '%PYTHON_EXE%' -ArgumentList '-m','gui' -WorkingDirectory '%APP_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LAUNCH_STDOUT%' -RedirectStandardError '%LAUNCH_STDERR%' -PassThru; $deadline = (Get-Date).AddSeconds(60); while ((Get-Date) -lt $deadline) { if (Test-Path -LiteralPath '%GUI_READY_FILE%') { exit 0 }; if ($p.HasExited) { $p.Refresh(); if ($p.ExitCode -eq 75) { exit 0 }; exit 1 }; Start-Sleep -Milliseconds 250 }; exit 2"
if errorlevel 2 goto launch_timeout
if errorlevel 1 goto launch_failed
exit /b 0

:launch_timeout
echo.
echo [Error] [EN] Desktop2Stereo GUI did not report ready within 60 seconds. This CMD window will stay open.
echo [Error] [CN] Desktop2Stereo GUI 未在 60 秒内回传就绪标志，CMD 窗口将保留用于查看状态。
echo.
goto show_logs

:launch_failed
echo.
echo [Error] [EN] Desktop2Stereo GUI failed before reporting ready. This CMD window will stay open.
echo [Error] [CN] Desktop2Stereo GUI 在回传就绪标志前失败，CMD 窗口将保留用于查看错误。
echo.
goto show_logs

:show_logs
echo [Hint] [EN] Check the messages below, then run this BAT again after fixing the issue.
echo [Hint] [CN] 请先查看下面的错误信息，修复后再次运行本 BAT。
echo.
if exist "%LAUNCH_STDERR%" (
    echo ===== launcher_stderr.log =====
    type "%LAUNCH_STDERR%"
)
pause
exit /b 1
