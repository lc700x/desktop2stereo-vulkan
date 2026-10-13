
@echo off
setlocal
echo --- Desktop2Stereo Installer (With ROCm7 for AMD GPU/APUs.) ---
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON_ROOT=%PROJECT_ROOT%\python3"
set "PYTHONPATH=%PYTHON_ROOT%"
set "PYTHONNOUSERSITE=1"
set "PIP_TARGET="
set "PIP_PREFIX="
set "PIP_USER="
set "PYTHON_EXE=%PYTHON_ROOT%\python.exe"
set "PYTHON_VERSION=3.12.10"
set "PYTHON_STAGING=%SCRIPT_DIR%python3.installing"
set "PYTHON_ARCHIVE=%TEMP%\python-%PYTHON_VERSION%-nuget.zip"
set "PYTHON_URL=https://www.nuget.org/api/v2/package/python/%PYTHON_VERSION%"
set "PIP_RETRIES=10"
set "PIP_TIMEOUT=180"
if defined LOCALAPPDATA (
    set "D2S_PIP_CACHE=%LOCALAPPDATA%\Desktop2Stereo\pip-cache"
) else (
set "D2S_PIP_CACHE=%TEMP%\Desktop2Stereo-pip-cache"
)

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import importlib.util, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 and importlib.util.find_spec('ensurepip') is not None else 1)" >nul 2>nul
    if not errorlevel 1 goto python_ready
)

echo - Installing fresh Python %PYTHON_VERSION% x64 runtime
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ARCHIVE%'"
if errorlevel 1 (
    echo Failed to download Python %PYTHON_VERSION% x64
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $staging=[IO.Path]::GetFullPath('%PYTHON_STAGING%'); if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }; Expand-Archive -LiteralPath '%PYTHON_ARCHIVE%' -DestinationPath $staging -Force; $runtime=Join-Path $staging 'tools'; & (Join-Path $runtime 'python.exe') -c 'import ensurepip, ssl, sqlite3, sys; raise SystemExit(0 if sys.version_info[:3] == (3, 12, 10) and sys.maxsize.bit_length() == 63 else 1)'; if ($LASTEXITCODE -ne 0) { throw 'Complete Python runtime validation failed' }; $target=[IO.Path]::GetFullPath('%PYTHON_ROOT%'); if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }; Move-Item -LiteralPath $runtime -Destination $target; Remove-Item -LiteralPath $staging -Recurse -Force"
if errorlevel 1 (
    echo Failed to install Python %PYTHON_VERSION% x64
    pause
    exit /b 1
)
if exist "%PYTHON_ARCHIVE%" del /f /q "%PYTHON_ARCHIVE%" >nul 2>nul
:python_ready

REM --- Get AMD GPUs sorted by AdapterRAM (largest first) ---
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object Name -like '*AMD*' | Sort-Object AdapterRAM -Descending | ForEach-Object { $_.Name }"`) do (
    set "FULL_GPU_NAME=%%i"
    goto :ProcessGPU
)


echo ERROR: No AMD GPU detected.
pause
exit /b 1

:ProcessGPU
echo Detected GPU: %FULL_GPU_NAME%




:InstallDependencies
echo - Setting up the virtual environment
REM Set paths
set "VIRTUAL_ENV=%PYTHON_ROOT%"
set "REQUIREMENTS_FILE=%SCRIPT_DIR%requirements-rocm7.txt"

echo.
echo Installing requirements from: %REQUIREMENTS_FILE%
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%" --cache-dir "%D2S_PIP_CACHE%" --prefer-binary --no-user --no-warn-script-location --retries %PIP_RETRIES% --timeout %PIP_TIMEOUT%
if errorlevel 1 (
    echo Failed to install ROCm requirements
    pause
    exit /b 1
)
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --cache-dir "%D2S_PIP_CACHE%" --prefer-binary --no-user --no-warn-script-location --retries %PIP_RETRIES% --timeout %PIP_TIMEOUT%
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m rocm_sdk init
if errorlevel 1 (
    echo Failed to initialize the ROCm SDK
    pause
    exit /b 1
)
rem Keep WindowsCaptureROCm frame pacing inside the bundled Python runtime.
"%PYTHON_EXE%" "%SCRIPT_DIR%apply-wc-rocm-patches.py"
if errorlevel 1 (
    echo Failed to apply bundled wc_rocm capture pacing patch
    pause
    exit /b 1
)
rem Portable triton JIT: patch the bundled triton to compile hip_utils
rem against the project-local msvcstub instead of the Windows SDK/MSVC.
"%PYTHON_EXE%" "%SCRIPT_DIR%apply-triton-portable-patches.py" "%PYTHON_ROOT%\Lib\site-packages"
if errorlevel 1 (
    echo Failed to apply triton portable patches
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "from pathlib import Path; import sys,triton; Path(triton.__file__).resolve().relative_to(Path(sys.prefix).resolve())" >nul 2>nul
if errorlevel 1 (
    echo Triton is not installed in the bundled Python runtime
    pause
    exit /b 1
)
echo Python environment deployed successfully.

echo The bundled AMD Triton JIT uses the bundled ROCm clang and project-local runtime stubs; Visual Studio Build Tools are not required for this path.

pause
exit /b 0
