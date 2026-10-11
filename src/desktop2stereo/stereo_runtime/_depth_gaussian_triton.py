from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised by the Torch fallback
    triton = None
    tl = None

from .triton_runtime import triton_runtime_available


if triton is not None:

    @triton.jit
    def _gaussian_1_kernel(
        source,
        output,
        width: tl.constexpr,
        height: tl.constexpr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offset < total
        plane = width * height
        batch = offset // plane
        pixel = offset % plane
        y = pixel // width
        x = pixel % width
        result = tl.full((BLOCK,), 0.0, tl.float32)

        for dy in tl.static_range(-3, 4):
            yy = tl.minimum(tl.maximum(y + dy, 0), height - 1)
            wy = tl.where(
                tl.abs(dy) == 3,
                0.07015933,
                tl.where(
                    tl.abs(dy) == 2,
                    0.13107488,
                    tl.where(tl.abs(dy) == 1, 0.19071282, 0.21610594),
                ),
            )
            for dx in tl.static_range(-3, 4):
                xx = tl.minimum(tl.maximum(x + dx, 0), width - 1)
                wx = tl.where(
                    tl.abs(dx) == 3,
                    0.07015933,
                    tl.where(
                        tl.abs(dx) == 2,
                        0.13107488,
                        tl.where(tl.abs(dx) == 1, 0.19071282, 0.21610594),
                    ),
                )
                sample = tl.load(
                    source + batch * plane + yy * width + xx,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                result += sample * wx * wy

        tl.store(output + offset, tl.minimum(tl.maximum(result, 0.0), 1.0), mask=active)


def gaussian_1(depth: torch.Tensor) -> torch.Tensor:
    if (
        triton is None
        or not isinstance(depth, torch.Tensor)
        or not depth.is_cuda
        or depth.dtype != torch.float32
        or not depth.is_contiguous()
        or not triton_runtime_available(depth.device)
    ):
        raise RuntimeError("Triton depth Gaussian is unavailable")
    batch, _, height, width = map(int, depth.shape)
    output = torch.empty_like(depth)
    total = batch * height * width
    block = 256
    _gaussian_1_kernel[(triton.cdiv(total, block),)](
        depth, output, width, height, total, block
    )
    return output
