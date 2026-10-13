@echo off
echo --- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---
echo - Setting up the virtual environment

@REM Install or reuse the project-local Python 3.12 x64 runtime.
@REM The official NuGet package provides a complete relocatable runtime layout.
Set "PYTHON_VERSION=3.12.10"
Set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do Set "PROJECT_ROOT=%%~fI"
Set "PYTHON_ROOT=%PROJECT_ROOT%\python3"
Set "PYTHON_STAGING=%SCRIPT_DIR%python3.installing"
Set "PYTHON_ARCHIVE=%TEMP%\python-%PYTHON_VERSION%-nuget.zip"
Set "PYTHON_URL=https://www.nuget.org/api/v2/package/python/%PYTHON_VERSION%"

if exist "%PYTHON_ROOT%\python.exe" (
    "%PYTHON_ROOT%\python.exe" -c "import importlib.util, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 and importlib.util.find_spec('ensurepip') is not None else 1)" >nul 2>nul
    if not errorlevel 1 goto python_ready
)

echo - Installing Python %PYTHON_VERSION% x64 to %PYTHON_ROOT%
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

if not exist "%PYTHON_ROOT%\python.exe" (
    echo Python installation completed but python.exe was not found at %PYTHON_ROOT%\python.exe
    pause
    exit /b 1
)
:python_ready

@REM Set paths
Set "PYTHON_EXE=%PYTHON_ROOT%\python.exe"
Set "PIP_RETRIES=10"
Set "PIP_TIMEOUT=180"
if defined LOCALAPPDATA (
    Set "D2S_PIP_CACHE=%LOCALAPPDATA%\Desktop2Stereo\pip-cache"
) else (
    Set "D2S_PIP_CACHE=%TEMP%\Desktop2Stereo-pip-cache"
)

@REM Bootstrap pip from the complete runtime without another network download.
"%PYTHON_EXE%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo - Installing pip into the local Python runtime
    "%PYTHON_EXE%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo Failed to install pip
        pause
        exit /b 1
    )
)

@REM Update pip
echo - Updating the pip package
"%PYTHON_EXE%" -m pip install --upgrade pip -r "%SCRIPT_DIR%requirements-pip-options.txt" --cache-dir "%D2S_PIP_CACHE%" --prefer-binary --no-warn-script-location --retries %PIP_RETRIES% --timeout %PIP_TIMEOUT%
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM TensorRT is distributed as a small source front-end package. Its isolated
@REM build otherwise downloads wheel/setuptools again and mirror truncation is not
@REM recoverable by retrying the outer requirements command. Prepare the official
@REM build prerequisites once and reuse pip's persistent download cache.
echo.
echo - Preparing Python package build tools
call :install_build_requirements
if errorlevel 1 (
    echo - Build tools download failed; retrying 1/2
    timeout /t 3 /nobreak >nul
    call :install_build_requirements
)
if errorlevel 1 (
    echo - Build tools download failed; retrying 2/2
    timeout /t 3 /nobreak >nul
    call :install_build_requirements
)
if errorlevel 1 (
    echo Failed to prepare Python package build tools after 3 attempts
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
call :install_cuda_requirements
if errorlevel 1 (
    echo - CUDA requirements download failed; retrying 1/2
    timeout /t 3 /nobreak >nul
    call :install_cuda_requirements
)
if errorlevel 1 (
    echo - CUDA requirements download failed; retrying 2/2
    timeout /t 3 /nobreak >nul
    call :install_cuda_requirements
)
if errorlevel 1 (
    echo Failed to install CUDA requirements after 3 attempts
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "import tensorrt as trt; raise SystemExit(0 if trt.__version__ == '10.14.1.48.post1' else 1)"
if errorlevel 1 (
    echo TensorRT installation validation failed
    pause
    exit /b 1
)
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --cache-dir "%D2S_PIP_CACHE%" --prefer-binary --no-warn-script-location --retries %PIP_RETRIES% --timeout %PIP_TIMEOUT%
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_DIR%apply-wc-cuda-patches.py"
if errorlevel 1 (
    echo Failed to configure CUDA capture pacing
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause
exit /b 0

:install_build_requirements
"%PYTHON_EXE%" -m pip install "setuptools==78.1.0" "wheel>=0.45,<1" "packaging>=24,<27" -r "%SCRIPT_DIR%requirements-pip-options.txt" --cache-dir "%D2S_PIP_CACHE%" --prefer-binary --no-warn-script-location --retries %PIP_RETRIES% --timeout %PIP_TIMEOUT%
exit /b %errorlevel%

:install_cuda_requirements
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements-cuda-legacy.txt" --cache-dir "%D2S_PIP_CACHE%" --prefer-binary --no-build-isolation --no-warn-script-location --retries %PIP_RETRIES% --timeout %PIP_TIMEOUT%
exit /b %errorlevel%
