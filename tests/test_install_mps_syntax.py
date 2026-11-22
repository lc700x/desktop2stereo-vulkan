"""macOS installer scripts must remain valid bash (ported MoltenVK block)."""

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


def test_install_mps_rebuilds_a_verified_python312_environment() -> None:
    script = (ROOT / "src" / "env_install" / "install-mps").read_text(encoding="utf-8")
    assert 'PYTHON_CMD="python3.12"' in script
    assert "brew install python@3.12" in script
    assert 'STAGING_ENV="$PROJECT_ROOT/python3.staging.$$"' in script
    assert 'requirements-mps.txt' in script
    assert "verify_macos_runtime" in script
    assert 'BACKUP_ENV="$VIRTUAL_ENV.backup.' in script
    assert "chmod -R 777" not in script
