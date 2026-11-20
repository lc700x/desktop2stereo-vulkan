from __future__ import annotations

import torch

from desktop2stereo.stereo_runtime.providers.intel.openvino_native_depth import (
    OpenVINOD3D11DepthProvider,
)
class _FakeNativeSession:
    adapter_luid = 0x100

    def set_texture(self, *_args):
        raise AssertionError("adapter mismatch should reject before set_texture")


class _FakeResource:
    texture = 1
    width = 64
    height = 64
    adapter_luid = 0x200
    resource_kind = "d3d11_texture"


class _WorkingNativeSession:
    adapter_luid = 0x100

    def set_texture(self, *_args):
        return None

    def nv12_surface(self):
        return 3, 64, 64

    def infer_output(self):
        return (1, 1, 2, 2), (0.0, 1.0, 2.0, 3.0)


class _NonfiniteNativeSession(_WorkingNativeSession):
    def infer_output(self):
        return (1, 1, 2, 2), (1.0, float("nan"), float("inf"), 4.0)


class _MatchingResource(_FakeResource):
    adapter_luid = 0x100


from desktop2stereo.stereo_runtime.providers.intel.openvino_d3d11_native import (
    OpenVINOD3D11Session,
    probe_openvino_d3d11_bridge,
)


class _FakeBridgeLibrary:
    def __init__(self) -> None:
        self.last_error_args = None

    def d2s_openvino_d3d11_set_texture(self, *_args):
        return 1

    def d2s_openvino_d3d11_last_error(self, *args):
        self.last_error_args = args
        return 0

    def d2s_openvino_d3d11_destroy(self, _handle):
        return None


def test_native_session_matches_c_abi_success_values() -> None:
    session = OpenVINOD3D11Session.__new__(OpenVINOD3D11Session)
    session._library = _FakeBridgeLibrary()
    session._handle = 1
    session.set_texture("input", 2, 64, 64)
    assert session._library.last_error_args is None


def test_native_session_last_error_uses_two_argument_c_abi() -> None:
    session = OpenVINOD3D11Session.__new__(OpenVINOD3D11Session)
    session._library = _FakeBridgeLibrary()
    session._handle = 1
    assert session.last_error() == "native bridge operation failed"
    assert len(session._library.last_error_args) == 2


def test_native_bridge_probe_is_safe_without_dll() -> None:
    result = probe_openvino_d3d11_bridge()
    assert result["backend"] == "openvino_d3d11_remote_tensor"
    assert isinstance(result["available"], bool)


def test_native_depth_provider_requires_complete_bridge() -> None:
    result = probe_openvino_d3d11_bridge()
    if result["available"] and result.get("nv12_surface") and result.get("bgra_to_nv12"):
        return
    try:
        OpenVINOD3D11DepthProvider(model_path="missing.xml", d3d11_device=0)
    except RuntimeError as exc:
        assert "bridge" in str(exc).lower() or "conversion" in str(exc).lower()
    else:
        raise AssertionError("native provider unexpectedly initialized without complete bridge")


def test_native_depth_provider_rejects_adapter_luid_mismatch() -> None:
    provider = OpenVINOD3D11DepthProvider.__new__(OpenVINOD3D11DepthProvider)
    provider.session = _FakeNativeSession()
    try:
        provider.predict_native(_FakeResource())
    except RuntimeError as exc:
        assert "adapter LUID mismatch" in str(exc)
    else:
        raise AssertionError("adapter mismatch was not rejected")


def test_native_depth_provider_reports_shared_nv12_surface() -> None:
    provider = OpenVINOD3D11DepthProvider.__new__(OpenVINOD3D11DepthProvider)
    provider.session = _WorkingNativeSession()
    result = provider.predict_native(_MatchingResource())
    assert tuple(result.depth.shape) == (1, 1, 2, 2)
    assert result.cuda_timing_events["nv12_surface_ready"] is True
    assert result.cuda_timing_events["nv12_surface_width"] == 64
    assert result.cuda_timing_events["nv12_surface_height"] == 64
    assert result.cuda_timing_events["input_zero_copy"] is True
    assert result.cuda_timing_events["input_gpu_to_cpu"] is False
    assert result.cuda_timing_events["output_device"] == "cpu"
    assert result.cuda_timing_events["output_gpu_to_cpu"] is True
    assert result.cuda_timing_events["output_zero_copy"] is False
    assert result.cuda_timing_events["zero_copy"] is False


def test_native_depth_provider_uses_declared_z_depth_semantics_once() -> None:
    provider = OpenVINOD3D11DepthProvider.__new__(OpenVINOD3D11DepthProvider)
    provider.session = _WorkingNativeSession()
    provider.output_semantics = "relative_z_depth"
    result = provider.predict_native(_MatchingResource())

    assert result.raw_output_semantics == "relative_z_depth"
    assert result.dibr_output_semantics == "relative_inverse_depth"
    assert torch.isfinite(result.depth).all().item()
    assert result.depth[0, 0, 0, 0] > result.depth[0, 0, 1, 1]


def test_native_depth_provider_sanitizes_nonfinite_model_output() -> None:
    provider = OpenVINOD3D11DepthProvider.__new__(OpenVINOD3D11DepthProvider)
    provider.session = _NonfiniteNativeSession()
    provider.output_semantics = "relative_inverse_depth"
    result = provider.predict_native(_MatchingResource())

    assert result.finite_depth is True
    assert result.nonfinite_count == 2
    assert torch.isfinite(result.depth).all().item()


def test_session_requires_native_bridge() -> None:
    result = probe_openvino_d3d11_bridge()
    if result["available"]:
        return
    try:
        OpenVINOD3D11Session("missing.xml", 0)
    except RuntimeError as exc:
        assert "bridge DLL" in str(exc)
    else:
        raise AssertionError("session unexpectedly initialized without native bridge")
