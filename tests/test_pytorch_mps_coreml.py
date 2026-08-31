"""CoreML provider plumbing tests (ported macOS behavior)."""

from stereo_runtime.depth_provider import (
    DepthProviderConfig,
    create_depth_provider,
)
from stereo_runtime.providers.apple.pytorch_mps import (
    CoreMLEngine,
    GenericAutoDepthMpsProvider,
    create_pytorch_mps_provider,
    is_coreml_available,
)


def test_is_coreml_available_returns_bool() -> None:
    assert isinstance(is_coreml_available(), bool)


def test_generic_provider_accepts_coreml_flags() -> None:
    provider = GenericAutoDepthMpsProvider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
        recompile_coreml=False,
    )
    assert provider.use_coreml is True
    assert provider.recompile_coreml is False
    assert CoreMLEngine.__module__ == "stereo_runtime.providers.apple.pytorch_mps"


def test_create_pytorch_mps_provider_forwards_coreml_flags() -> None:
    provider = create_pytorch_mps_provider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        depth_upsample="bilinear",
        depth_upsample_edge_strength=0.35,
        use_coreml=True,
        recompile_coreml=True,
    )
    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.use_coreml is True
    assert provider.recompile_coreml is True


def test_depth_provider_config_carries_coreml_fields() -> None:
    cfg = DepthProviderConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
        recompile_coreml=False,
    )
    assert cfg.use_coreml is True
    assert cfg.recompile_coreml is False


def test_create_depth_provider_mps_backend_returns_mps_provider() -> None:
    cfg = DepthProviderConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=False,
    )
    provider = create_depth_provider(cfg)
    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.use_coreml is False
