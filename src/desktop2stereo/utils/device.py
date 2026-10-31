def get_device(index=0):
    import torch

    try:
        index = max(0, int(index))
        directml_count = 0
        try:
            import torch_directml

            if torch_directml.is_available():
                directml_count = int(torch_directml.device_count())
                if index < directml_count:
                    return (
                        torch_directml.device(index),
                        f"Using DirectML device: {torch_directml.device_name(index)}",
                    )
        except ImportError:
            pass
        index -= directml_count
        if torch.cuda.is_available():
            cuda_count = int(torch.cuda.device_count())
            if index < cuda_count:
                return (
                    torch.device(f"cuda:{index}"),
                    f"Using CUDA device: {torch.cuda.get_device_name(index)}",
                )
            index -= cuda_count
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            if index == 0:
                return torch.device("mps"), "Using Apple Silicon (MPS) device"
            index -= 1
        if torch.xpu.is_available():
            xpu_count = int(torch.xpu.device_count())
            if index < xpu_count:
                return (
                    torch.device(f"xpu:{index}"),
                    f"Using XPU device: {torch.xpu.get_device_name(index)}",
                )
        return torch.device("cpu"), "Using CPU device"
    except Exception:  # noqa: BLE001 - accelerator discovery retains the CPU fallback.
        return torch.device("cpu"), "Using CPU device"
