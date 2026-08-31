import platform


OS_NAME = platform.system()


def resolve_capture_tool(raw_value, os_name=OS_NAME):
    """Pick the OS- and device-specific capture backend when settings use auto/none."""
    if raw_value and raw_value != "none":
        normalized = str(raw_value).strip()
        if os_name == "Windows":
            # Fall back across the vendor-specific GPU capture tools when the
            # configured tool does not match the detected device backend (e.g.
            # the GUI saved WindowsCaptureCUDA but this machine is AMD ROCm).
            try:
                import torch

                is_rocm = bool(getattr(torch.version, "hip", None))
                is_cuda = bool(torch.cuda.is_available()) and not is_rocm
            except Exception:
                is_rocm = False
                is_cuda = False
            if normalized == "WindowsCaptureCUDA" and is_rocm:
                return "WindowsCaptureROCm"
            if normalized == "WindowsCaptureROCm" and is_cuda:
                return "WindowsCaptureCUDA"
        return raw_value
    if os_name == "Windows":
        try:
            import torch

            if torch.cuda.is_available():
                if getattr(torch.version, "hip", None) is not None:
                    return "WindowsCaptureROCm"
                return "WindowsCaptureCUDA"
        except Exception:
            pass
        try:
            import torch_directml

            if torch_directml.is_available() and torch_directml.device_count() > 0:
                return "DXCamera"
        except Exception:
            pass
        # WindowsCapture is the generic Windows capture default. The
        # capture factory still falls back to Desktop Duplication/DXCamera
        # when that backend is unavailable.
        return "WindowsCapture"
    if os_name == "Darwin":
        return "ScreenCaptureKit"
    return "DXCamera"


_resolve_capture_tool = resolve_capture_tool
