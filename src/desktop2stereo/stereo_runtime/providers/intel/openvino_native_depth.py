"""OpenVINO D3D11 depth adapter for a borrowed Desktop Duplication frame.

The adapter is deliberately separate from the tensor-only provider factory:
creating it requires the capture-owned ID3D11Device and an OpenVINO IR model.
It never turns a missing native bridge into a silent CPU path.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from stereo_runtime.depth_provider import (
    DEFAULT_DEPTH_OUTPUT_SEMANTICS,
    DepthOutputSemantics,
    DepthProfileResult,
    DepthProviderInfo,
    _convert_depth_output_to_dibr_response,
    _normalize_depth_canonical,
)

from .openvino_d3d11_native import OpenVINOD3D11Session, probe_openvino_d3d11_bridge


class OpenVINOD3D11DepthProvider:
    """Run an OpenVINO depth IR directly from a borrowed D3D11 texture."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        d3d11_device: int,
        depth_resolution: int = 518,
        output_semantics: DepthOutputSemantics = DEFAULT_DEPTH_OUTPUT_SEMANTICS,
    ) -> None:
        capability = probe_openvino_d3d11_bridge()
        if not capability.get("available"):
            raise RuntimeError(str(capability.get("reason")))
        if not capability.get("nv12_surface") or not capability.get("bgra_to_nv12"):
            raise RuntimeError(str(capability.get("reason")))
        self.model_path = str(model_path)
        self.depth_resolution = int(depth_resolution)
        if output_semantics not in {
            "relative_inverse_depth",
            "raw_disparity",
            "relative_z_depth",
            "metric_z_depth",
        }:
            raise ValueError(f"unsupported OpenVINO output semantics: {output_semantics!r}")
        self.output_semantics = output_semantics
        self.session = OpenVINOD3D11Session(self.model_path, d3d11_device)
        self.info = DepthProviderInfo(
            provider="OpenVINOD3D11Session",
            model_name=Path(self.model_path).stem,
            model_id=self.model_path,
            depth_resolution=self.depth_resolution,
            cache_dir=str(Path(self.model_path).parent),
            load_mode="native",
            depth_backend="openvino_d3d11_remote",
            runtime="openvino-gpu-d3d11",
            execution_provider="OpenVINO GPU RemoteTensor",
            fallback_reason=None,
            output_device="cpu",
            raw_output_semantics=self.output_semantics,
        )

    def load(self):
        return self

    def predict_native(self, resource) -> DepthProfileResult:
        texture = getattr(resource, "texture", None)
        if texture is None:
            raise TypeError("native OpenVINO provider requires a borrowed D3D11 texture frame")
        resource_luid = int(getattr(resource, "adapter_luid", 0))
        session_luid = int(getattr(self.session, "adapter_luid", 0))
        if not resource_luid or not session_luid:
            raise RuntimeError(
                "D3D11 adapter LUID is required for OpenVINO RemoteTensor sharing"
            )
        if resource_luid != session_luid:
            raise RuntimeError(
                "D3D11 adapter LUID mismatch: "
                f"capture=0x{resource_luid:016x}, provider=0x{session_luid:016x}"
            )
        resource_format = str(getattr(resource, "format", "BGRA8") or "BGRA8").upper()
        if resource_format not in {"BGRA8", "B8G8R8A8_UNORM"}:
            raise RuntimeError(
                "OpenVINO D3D11 RemoteTensor requires BGRA8 input, "
                f"got {resource_format}"
            )
        start = time.perf_counter()
        self.session.set_texture(
            "",
            texture,
            int(resource.width),
            int(resource.height),
        )
        _nv12_texture, nv12_width, nv12_height = self.session.nv12_surface()
        preprocess_ms = (time.perf_counter() - start) * 1000.0
        infer_start = time.perf_counter()
        shape, values = self.session.infer_output()
        model_ms = (time.perf_counter() - infer_start) * 1000.0
        postprocess_start = time.perf_counter()
        raw_depth = torch.tensor(values, dtype=torch.float32).reshape(shape)
        nonfinite_count = int((~torch.isfinite(raw_depth)).sum().item())
        output_semantics = getattr(
            self,
            "output_semantics",
            DEFAULT_DEPTH_OUTPUT_SEMANTICS,
        )
        depth = _normalize_depth_canonical(
            _convert_depth_output_to_dibr_response(raw_depth, output_semantics)
        )
        return DepthProfileResult(
            depth=depth,
            preprocess_ms=preprocess_ms,
            model_ms=model_ms,
            postprocess_ms=(time.perf_counter() - postprocess_start) * 1000.0,
            finite_depth=bool(torch.isfinite(depth).all().item()),
            nonfinite_count=nonfinite_count,
            raw_output_semantics=output_semantics,
            cuda_timing_events={
                "input_resource_kind": getattr(resource, "resource_kind", "d3d11_texture"),
                "capture_gpu": True,
                "input_gpu_to_cpu": False,
                "input_zero_copy": True,
                "gpu_to_cpu": True,
                "output_device": "cpu",
                "output_gpu_to_cpu": True,
                "output_zero_copy": False,
                "zero_copy": False,
                "adapter_luid": resource_luid,
                "provider_adapter_luid": session_luid,
                "adapter_match": bool(
                    not resource_luid or not session_luid or resource_luid == session_luid
                ),
                "nv12_surface_ready": True,
                "nv12_surface_width": nv12_width,
                "nv12_surface_height": nv12_height,
            },
        )

    def close(self) -> None:
        self.session.close()


__all__ = ["OpenVINOD3D11DepthProvider"]
