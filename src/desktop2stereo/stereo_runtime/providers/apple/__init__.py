"""Apple CoreML and MPS depth providers."""

from .mlx_depth import MlxDepthProvider, create_mlx_depth_provider, is_mlx_available
from .pytorch_mps import (
    DistillAnyDepthBaseMps,
    GenericAutoDepthMpsProvider,
    GenericTorchMpsDepthProvider,
    TorchMpsDepthProvider,
    create_pytorch_mps_provider,
    is_mps_torch_available,
)

__all__ = [
    "TorchMpsDepthProvider",
    "GenericTorchMpsDepthProvider",
    "DistillAnyDepthBaseMps",
    "GenericAutoDepthMpsProvider",
    "create_pytorch_mps_provider",
    "is_mps_torch_available",
    "MlxDepthProvider",
    "create_mlx_depth_provider",
    "is_mlx_available",
]
