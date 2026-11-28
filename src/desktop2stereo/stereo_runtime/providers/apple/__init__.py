"""Apple CoreML and MPS depth providers."""

from .coreml_families import (
    CoreMLFamilyAccelerationError,
    create_coreml_family_provider,
)
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
    "CoreMLFamilyAccelerationError",
    "create_coreml_family_provider",
    "create_pytorch_mps_provider",
    "is_mps_torch_available",
]
