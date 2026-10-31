import runpy
import sys
import types
from pathlib import Path


def _device_module():
    path = Path(__file__).resolve().parents[1] / "src/desktop2stereo/utils/device.py"
    return runpy.run_path(str(path))["get_device"]


def _torch_stub(*, cuda_count=2):
    return types.SimpleNamespace(
        device=lambda value: value,
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda_count > 0,
            device_count=lambda: cuda_count,
            get_device_name=lambda index: f"GPU {index}",
        ),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        ),
        xpu=types.SimpleNamespace(is_available=lambda: False),
    )


def test_selected_nvidia_device_keeps_its_cuda_ordinal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda_count=2))
    monkeypatch.delitem(sys.modules, "torch_directml", raising=False)

    device, description = _device_module()(1)

    assert device == "cuda:1"
    assert description == "Using CUDA device: GPU 1"


def test_cuda_ordinal_accounts_for_directml_devices_in_ui_order(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda_count=2))
    directml = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        device=lambda index: f"directml:{index}",
        device_name=lambda index: f"DirectML GPU {index}",
    )
    monkeypatch.setitem(sys.modules, "torch_directml", directml)

    device, description = _device_module()(3)

    assert device == "cuda:1"
    assert description == "Using CUDA device: GPU 1"
