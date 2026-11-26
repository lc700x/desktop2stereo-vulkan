from stereo_runtime.depth_settings import resolve_depth_settings


def _settings(**overrides):
    settings = {
        "Depth Model": "Distill-Any-Depth-Base",
        "Model List": {},
        "Depth Resolution": 518,
        "Computing Device": "mps",
        "FP16": True,
        "Depth Pop": 5,
        "Anti-aliasing": 1,
        "torch.compile": False,
        "TensorRT": False,
        "Recompile TensorRT": False,
        "MIGraphX": False,
        "Recompile MIGraphX": False,
        "CoreML": True,
        "Recompile CoreML": False,
        "OpenVINO": False,
        "Recompile OpenVINO": False,
    }
    settings.update(overrides)
    return settings


def test_resolve_depth_settings_keeps_fp16_for_macos_coreml(monkeypatch):
    monkeypatch.setattr(
        "stereo_runtime.depth_settings.model_name_mapping",
        lambda: {"Distill-Any-Depth-Base": "lc700x/Distill-Any-Depth-Base-hf"},
    )

    depth_settings = resolve_depth_settings(_settings(), os_name="Darwin")

    assert depth_settings.fp16 is True


def test_resolve_depth_settings_keeps_fp16_for_cuda(monkeypatch):
    monkeypatch.setattr(
        "stereo_runtime.depth_settings.model_name_mapping",
        lambda: {"Distill-Any-Depth-Base": "lc700x/Distill-Any-Depth-Base-hf"},
    )

    depth_settings = resolve_depth_settings(
        _settings(**{"Computing Device": "cuda:0", "CoreML": False}),
        os_name="Windows",
    )

    assert depth_settings.fp16 is True


def test_resolve_depth_settings_uses_depth_pop(monkeypatch):
    monkeypatch.setattr(
        "stereo_runtime.depth_settings.model_name_mapping",
        lambda: {"Distill-Any-Depth-Base": "lc700x/Distill-Any-Depth-Base-hf"},
    )

    depth_settings = resolve_depth_settings(_settings(**{"Depth Pop": 7}), os_name="Windows")

    assert depth_settings.foreground_scale == 0.7
