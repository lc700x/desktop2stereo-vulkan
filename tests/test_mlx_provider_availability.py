from types import SimpleNamespace

import pytest

from stereo_runtime.providers.apple import mlx_depth


def test_missing_mlx_error_preserves_import_reason_and_setup_command(monkeypatch) -> None:
    import_error = ModuleNotFoundError("No module named 'mlx'")

    def fail_import(_module_name):
        raise import_error

    monkeypatch.setattr(mlx_depth.sys, "platform", "darwin")
    monkeypatch.setattr(mlx_depth, "mx", None)
    monkeypatch.setattr(mlx_depth, "nn", None)
    monkeypatch.setattr(mlx_depth.importlib, "import_module", fail_import)

    assert not mlx_depth.is_mlx_available()
    with pytest.raises(RuntimeError) as exc_info:
        mlx_depth._require_mlx()

    message = str(exc_info.value)
    assert "MLX import failed (ModuleNotFoundError: No module named 'mlx')" in message
    assert "src/env_install/install-mps" in message
    assert exc_info.value.__cause__ is import_error


def test_metal_probe_error_is_preserved(monkeypatch) -> None:
    probe_error = RuntimeError("Metal initialization failed")

    def fail_metal_probe():
        raise probe_error

    fake_mlx = SimpleNamespace(
        gpu=object(),
        is_available=lambda _device: True,
        metal=SimpleNamespace(is_available=fail_metal_probe),
    )
    monkeypatch.setattr(mlx_depth.sys, "platform", "darwin")
    monkeypatch.setattr(mlx_depth, "mx", fake_mlx)
    monkeypatch.setattr(mlx_depth, "nn", object())

    assert not mlx_depth.is_mlx_available()
    with pytest.raises(RuntimeError, match="Metal initialization failed") as exc_info:
        mlx_depth._require_mlx()

    assert "src/env_install/install-mps" in str(exc_info.value)
    assert exc_info.value.__cause__ is probe_error


def test_mlx_dtype_defaults_to_fp16_and_accepts_fp32_override(monkeypatch) -> None:
    fake_mlx = SimpleNamespace(float16="fp16", float32="fp32", bfloat16="bf16")
    monkeypatch.setattr(mlx_depth, "mx", fake_mlx)
    monkeypatch.delenv("D2S_MLX_DTYPE", raising=False)

    assert mlx_depth._mlx_dtype() == "fp16"

    monkeypatch.setenv("D2S_MLX_DTYPE", "float32")
    assert mlx_depth._mlx_dtype() == "fp32"
