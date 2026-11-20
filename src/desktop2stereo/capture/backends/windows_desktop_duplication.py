"""Desktop Duplication capture integration point.

The real DXGI/D3D11 bridge is intentionally optional until the native extension
is built. When it is unavailable, this backend uses the existing DXCamera
compatibility path and marks the frame as host-backed; it never reports
zero-copy.
"""

from __future__ import annotations

import logging
import os
import platform

from .desktop_duplication_native import NativeDesktopDuplication, probe as probe_native
from stereo_runtime.providers.intel.openvino_native_depth import (
    OpenVINOD3D11DepthProvider,
)
from stereo_runtime.providers.intel.onevpl_d3d11_encoder import (
    probe_onevpl_d3d11,
)
from stereo_runtime.providers.intel.d3d11_sbs_surface import (
    probe_d3d11_sbs_surface,
)
from stereo_runtime.providers.intel.openvino_remote import (
    probe_openvino_remote_tensor,
)
from .windows_dxcamera import DesktopGrabber as _CompatDesktopGrabber

logger = logging.getLogger(__name__)


class DesktopDuplicationUnavailable(RuntimeError):
    """Raised when the native DXGI Desktop Duplication bridge is unavailable."""


def probe() -> dict:
    """Return capture and Intel RemoteTensor capabilities without starting capture."""
    result = probe_native()
    inference = probe_openvino_remote_tensor()
    encoder = probe_onevpl_d3d11()
    final_surface = probe_d3d11_sbs_surface()
    result.update(
        {
            "inference_backend": "openvino_d3d11_remote_tensor",
            "encoder_backend": encoder["backend"],
            "encoder_available": encoder["available"],
            "encoder_reason": encoder.get("reason"),
            "final_sbs_surface_backend": final_surface["backend"],
            "final_sbs_surface_available": final_surface["available"],
            "final_sbs_surface_reason": final_surface.get("reason"),
            "inference_available": inference.zero_copy_ready,
            "native_inference_zero_copy_ready": (
                result.get("available", False) and inference.zero_copy_ready
            ),
            "inference_reason": inference.reason,
            # The current Desktop Duplication consumer intentionally performs
            # one staging readback to preserve the existing CPU-frame contract.
            # Do not promote the borrowed-texture inference capability to an
            # end-to-end capture zero-copy claim.
            "zero_copy_ready": False,
            "zero_copy_reason": (
                "native capture-to-inference texture is available, but the "
                "compatibility output still performs a staging readback"
            ),
        }
    )
    return result


class DesktopGrabber:
    """Desktop Duplication-compatible source with an explicit safe fallback."""

    backend_name = "desktop_duplication"

    def __init__(
        self,
        output_resolution=1080,
        fps=60,
        window_title=None,
        capture_mode="Monitor",
        monitor_index=1,
    ):
        if platform.system() != "Windows":
            raise DesktopDuplicationUnavailable(
                "Desktop Duplication requires Windows DXGI 1.2"
            )
        self._native = None
        self._native_depth_provider = None
        self._last_native_depth_profile = None
        self.capture_mode = capture_mode
        self.scaled_height = output_resolution
        self._monitor_index = int(monitor_index)
        self._native_probe = probe()
        self._compat = _CompatDesktopGrabber(
            output_resolution=output_resolution,
            fps=fps,
            window_title=window_title,
            capture_mode=capture_mode,
            monitor_index=monitor_index,
        )
        if self._native_probe.get("available"):
            logger.info(
                "[DesktopDuplication] native DXGI bridge is available; "
                "D3D11 texture consumer capability=%s, inference_zero_copy=%s",
                self._native_probe.get("texture_consumer", True),
                self._native_probe.get("native_inference_zero_copy_ready", False),
            )
        if (
            str(os.environ.get("D2S_INTEL_NATIVE_OPENVINO", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
            and os.environ.get("D2S_OPENVINO_MODEL")
            and self._native_probe.get("native_inference_zero_copy_ready")
        ):
            try:
                self._native_depth_provider = self.create_native_depth_provider(
                    os.environ["D2S_OPENVINO_MODEL"]
                )
                logger.info(
                    "[DesktopDuplication] native OpenVINO depth path enabled "
                    "for the capture adapter"
                )
            except Exception as exc:
                logger.warning(
                    "[DesktopDuplication] native OpenVINO path unavailable; "
                    "using compatibility depth path: %s",
                    exc,
                )
        if self._native_depth_provider is not None:
            logger.info(
                "[DesktopDuplication] native D3D11 capture-to-inference is active; "
                "same borrowed frame also feeds compatibility RGB via one readback; "
                "native output_gpu_to_cpu=True zero_copy=False gpu_copy_count=1"
            )
        elif self._native_probe.get("available") and capture_mode == "Monitor":
            logger.info(
                "[DesktopDuplication] native single-frame capture is active; "
                "BGRA readback is shared by downstream compatibility consumers; "
                "output_gpu_to_cpu=True zero_copy=False gpu_copy_count=1"
            )
        else:
            logger.warning(
                "[DesktopDuplication] using DXCamera compatibility capture "
                "(gpu_to_cpu=True, zero_copy=False); native_reason=%s",
                self._native_probe.get("reason"),
            )

    @property
    def zero_copy(self):
        return False

    @property
    def native_probe(self):
        return dict(self._native_probe)

    def acquire_native_frame(self):
        """Acquire a borrowed D3D11 texture for a native consumer bridge."""
        if not self._native_probe.get("available"):
            raise DesktopDuplicationUnavailable(self._native_probe.get("reason"))
        if self._native is None:
            output_index = max(0, self._monitor_index - 1)
            self._native = NativeDesktopDuplication(output_index=output_index)
        return self._native.acquire_frame()

    def infer_native_frame(self, provider):
        """Acquire one borrowed texture, infer, and always release the frame."""
        frame = self.acquire_native_frame()
        if frame is None:
            return None
        try:
            predict_native = getattr(provider, "predict_native", None)
            if not callable(predict_native):
                raise TypeError("native provider must implement predict_native(resource)")
            return predict_native(frame)
        finally:
            frame.release()

    def create_native_depth_provider(self, model_path, depth_resolution=518):
        """Create the optional provider on the same D3D11 adapter as capture."""
        if not self._native_probe.get("available"):
            raise DesktopDuplicationUnavailable(self._native_probe.get("reason"))
        if self._native is None:
            output_index = max(0, self._monitor_index - 1)
            self._native = NativeDesktopDuplication(output_index=output_index)
        device = self._native.device
        if not device:
            raise DesktopDuplicationUnavailable("native Desktop Duplication device handle is unavailable")
        output_semantics = os.environ.get(
            "D2S_OPENVINO_OUTPUT_SEMANTICS",
            "relative_inverse_depth",
        ).strip().lower()
        return OpenVINOD3D11DepthProvider(
            model_path=model_path,
            d3d11_device=device,
            depth_resolution=depth_resolution,
            output_semantics=output_semantics,
        )

    def grab(self):
        self._last_native_depth_profile = None
        if self._native_probe.get("available") and self.capture_mode == "Monitor":
            frame = None
            try:
                frame = self.acquire_native_frame()
                if frame is None:
                    raise RuntimeError("Desktop Duplication frame timed out")
                if self._native_depth_provider is not None:
                    predict_native = getattr(self._native_depth_provider, "predict_native", None)
                    if not callable(predict_native):
                        raise TypeError("native provider must implement predict_native(resource)")
                    self._last_native_depth_profile = predict_native(frame)
                bgra = frame.readback_bgra()
                frame_raw = bgra[..., :3][:, :, ::-1].copy()
                return frame_raw, self.scaled_height
            except Exception as exc:
                logger.warning(
                    "[DesktopDuplication] native single-frame capture failed; "
                    "falling back to DXCamera compatibility capture: %s",
                    exc,
                )
                self._last_native_depth_profile = None
            finally:
                if frame is not None:
                    frame.release()

        frame_raw, size = self._compat.grab()
        if self._native_depth_provider is not None:
            try:
                self._last_native_depth_profile = self.infer_native_frame(
                    self._native_depth_provider
                )
            except Exception as exc:
                logger.warning(
                    "[DesktopDuplication] native OpenVINO inference failed; "
                    "disabling native path and continuing with compatibility depth: %s",
                    exc,
                )
                self._native_depth_provider.close()
                self._native_depth_provider = None
        return frame_raw, size

    def pop_native_depth_profile(self):
        profile = self._last_native_depth_profile
        self._last_native_depth_profile = None
        return profile

    def stop(self):
        if self._native_depth_provider is not None:
            self._native_depth_provider.close()
            self._native_depth_provider = None
        if self._native is not None:
            self._native.close()
            self._native = None
        self._compat.stop()
