"""macOS installer scripts must remain valid bash (ported MoltenVK block)."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_install_mps_shell_syntax() -> None:
    shutil.which("bash") is not None
    for name in ("install-mps", "install-mps0"):
        script = ROOT / "src" / "env_install" / name
        assert script.is_file()
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_install_mps_contains_moltenvk_install_step() -> None:
    script = (ROOT / "src" / "env_install" / "install-mps").read_text(encoding="utf-8")
    assert "brew install molten-vk vulkan-loader" in script
    assert "brew list --formula vulkan-loader" in script
