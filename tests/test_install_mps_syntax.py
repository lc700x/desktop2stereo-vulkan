"""macOS installer scripts must remain valid bash (ported MoltenVK block)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _bash_usable() -> bool:
    if shutil.which("bash") is None:
        return False
    try:
        return subprocess.run(
            ["bash", "--version"],
            capture_output=True,
            timeout=10,
        ).returncode == 0
    except Exception:
        return False


_BASH_USABLE = _bash_usable()


@pytest.mark.skipif(not _BASH_USABLE, reason="bash is not runnable in this environment")
def test_install_mps_shell_syntax() -> None:
    shutil.which("bash") is not None
    for name in ("install-mps", "install-mps0"):
        script = ROOT / "src" / "env_install" / name
        assert script.is_file()
        result = subprocess.run(
            ["bash", "-n"],
            input=script.read_text(encoding="utf-8").encode("utf-8"),
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr


def test_install_mps_contains_moltenvk_install_step() -> None:
    script = (ROOT / "src" / "env_install" / "install-mps").read_text(encoding="utf-8")
    assert "brew install molten-vk vulkan-loader" in script
    assert "brew list --formula vulkan-loader" in script


def test_install_mps_rebuilds_a_verified_python312_mps_environment() -> None:
    script = (ROOT / "src" / "env_install" / "install-mps").read_text(encoding="utf-8")
    assert "find_supported_python" in script
    assert "brew install python@3.12" in script
    assert 'STAGING_ENV="$PROJECT_ROOT/python3.staging.$$"' in script
    assert 'requirements-mps.txt' in script
    assert "verify_macos_runtime" in script
    assert "import coremltools" in script
    assert "mlx" not in script.lower()
    assert 'BACKUP_ENV="$VIRTUAL_ENV.backup.' in script
    assert "chmod -R 777" not in script


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.skipif(not _BASH_USABLE, reason="bash is not runnable in this environment")
def test_run_mac_does_not_invoke_global_pkill(tmp_path: Path) -> None:
    package = tmp_path / "package"
    python_dir = package / "python3" / "bin"
    gui_dir = package / "desktop2stereo" / "gui"
    fake_bin = tmp_path / "bin"
    for path in (python_dir, gui_dir, fake_bin):
        path.mkdir(parents=True, exist_ok=True)

    (package / "run_mac").write_text((ROOT / "src" / "run_mac").read_text(encoding="utf-8"), encoding="utf-8")
    (package / "run_mac").chmod(0o755)
    (gui_dir / "gui.py").write_text("", encoding="utf-8")
    marker = tmp_path / "pkill-was-called"
    _write_executable(
        python_dir / "python",
        """#!/bin/bash
mkdir -p "$(dirname "$0")/../../desktop2stereo/logs"
touch "$(dirname "$0")/../../desktop2stereo/logs/gui_ready.flag"
""",
    )
    _write_executable(fake_bin / "pkill", f'#!/bin/bash\ntouch "{marker}"\n')

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    result = subprocess.run(["bash", str(package / "run_mac")], env=env, capture_output=True, text=True, timeout=15)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (package / "desktop2stereo" / "logs" / "gui_ready.flag").is_file()
    assert not marker.exists()


@pytest.mark.skipif(not _BASH_USABLE, reason="bash is not runnable in this environment")
def test_run_mac_refuses_a_second_gui_from_the_same_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    real_python_root = tmp_path / "real-python"
    python_exe = real_python_root / "bin" / "python"
    logs = package / "desktop2stereo" / "logs"
    package.mkdir(parents=True, exist_ok=True)
    python_exe.parent.mkdir(parents=True)
    (package / "python3").symlink_to(real_python_root, target_is_directory=True)
    (package / "desktop2stereo" / "gui").mkdir(parents=True)
    logs.mkdir(parents=True)

    (package / "run_mac").write_text((ROOT / "src" / "run_mac").read_text(encoding="utf-8"), encoding="utf-8")
    (package / "run_mac").chmod(0o755)
    (package / "desktop2stereo" / "gui" / "gui.py").write_text("", encoding="utf-8")
    old_pid_file = tmp_path / "old.pid"
    _write_executable(
        python_exe,
        """#!/bin/bash
root="$(cd "$(dirname "$0")/../../desktop2stereo" && pwd)"
if [ -n "$D2S_GUI_ACTIVATION_FILE" ]; then
  printf 'activated\\n' > "$D2S_GUI_ACTIVATION_FILE"
  exit 0
fi
if [ -n "$D2S_FAKE_STAY" ]; then
  echo "$$" > "$D2S_OLD_PID_FILE"
  while true; do sleep 1; done
fi
mkdir -p "$root/logs"
touch "$root/logs/gui_ready.flag"
""",
    )
    old_env = dict(os.environ)
    old_env.update({"D2S_FAKE_STAY": "1", "D2S_OLD_PID_FILE": str(old_pid_file)})
    old_gui = subprocess.Popen(
        [str(python_exe), "-m", "gui"],
        env=old_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if old_pid_file.is_file():
                break
            subprocess.run(["sleep", "0.1"], check=False)
        assert old_pid_file.is_file()
        old_pid = int(old_pid_file.read_text(encoding="utf-8").strip())
        (logs / "gui.pid").write_text(str(old_pid), encoding="utf-8")

        result = subprocess.run(
            ["bash", str(package / "run_mac")],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "GUI is already running" in result.stdout
        assert old_gui.poll() is None
        assert (logs / "gui.pid").read_text(encoding="utf-8").strip() == str(old_pid)
        assert (logs / "gui.activate.request").is_file()
    finally:
        if old_gui.poll() is None:
            old_gui.terminate()
            old_gui.wait(timeout=5)


@pytest.mark.skipif(not _BASH_USABLE, reason="bash is not runnable in this environment")
def test_install_mps_installs_missing_python312_before_building_environment(tmp_path: Path) -> None:
    package = tmp_path / "install-package"
    env_dir = package / "env_install"
    fake_bin = tmp_path / "install-bin"
    env_dir.mkdir(parents=True)
    fake_bin.mkdir()

    installer = (ROOT / "src" / "env_install" / "install-mps").read_text(encoding="utf-8")
    (env_dir / "install-mps").write_text(installer, encoding="utf-8")
    (package / "run_mac").write_text("", encoding="utf-8")
    python_body = """#!/bin/bash
set -e
if [ "$1" = "-c" ]; then
  echo 3.12
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
  chmod +x "$3/bin/python"
  exit 0
fi
exit 0
"""
    python_path = fake_bin / "fake-python"
    _write_executable(python_path, python_body)
    marker = tmp_path / "brew-installed-python"
    brew_body = f"""#!/bin/bash
if [ "$1" = "install" ] && [ "$2" = "python@3.12" ]; then
  touch "{marker}"
  cp "{python_path}" "{fake_bin / 'python3.12'}"
  chmod +x "{fake_bin / 'python3.12'}"
fi
exit 0
"""
    _write_executable(fake_bin / "brew", brew_body)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    result = subprocess.run(
        ["bash", str(env_dir / "install-mps")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists()
    assert (package / "python3" / "bin" / "python").is_file()


@pytest.mark.skipif(not _BASH_USABLE, reason="bash is not runnable in this environment")
def test_install_mps_treats_duplicate_gui_exit_code_as_success(tmp_path: Path) -> None:
    package = tmp_path / "package"
    python_dir = package / "python3" / "bin"
    gui_dir = package / "desktop2stereo" / "gui"
    for path in (python_dir, gui_dir):
        path.mkdir(parents=True, exist_ok=True)

    (package / "run_mac").write_text((ROOT / "src" / "run_mac").read_text(encoding="utf-8"), encoding="utf-8")
    (package / "run_mac").chmod(0o755)
    (gui_dir / "gui.py").write_text("", encoding="utf-8")
    _write_executable(python_dir / "python", "#!/bin/bash\nexit 75\n")

    result = subprocess.run(
        ["bash", str(package / "run_mac")],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Existing GUI was requested to come to the front" in result.stdout
    assert not (package / "desktop2stereo" / "logs" / "gui.pid").exists()
