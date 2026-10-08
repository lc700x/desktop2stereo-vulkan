"""Stage-by-stage visual regression for the stereo display pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .depth_postprocess import postprocess_depth
from .io import load_depth, load_rgb, save_depth, save_rgb
from .openxr_visual_regression import compare_tensors, diff_heatmap, make_depth_proxy_from_rgb
from .output import make_sbs
from .parallax import resolve_parallax_budget
from .synthesis import StereoConfig, synthesize_stereo
from .vulkan_stereo_pass import (
    VulkanLayeredStereoParams,
    resolve_vulkan_hole_fill_mode,
    resolve_vulkan_hole_fill_parameters,
)


@dataclass(frozen=True)
class StageVisualRegressionConfig:
    depth_strength: float = 0.25
    convergence: float = 0.0
    max_disparity_px: float | None = None
    parallax_preset: str = "standard"
    layers: int = 2
    symmetric: bool = True
    occlusion: bool = True
    edge_threshold: float = 0.04
    edge_dilation: int = 2
    mask_feather_radius: int = 3
    hole_fill_radius: int = 1
    hole_fill_strength: float = 0.6
    hole_fill_mode: str = "balanced"
    foreground_scale: float = 1.0
    midground_scale: float = 1.0
    background_scale: float = 1.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_eye_pair(output_dir: Path, prefix: str, left: torch.Tensor, right: torch.Tensor) -> None:
    save_rgb(left, output_dir / f"{prefix}_left_eye.png")
    save_rgb(right, output_dir / f"{prefix}_right_eye.png")
    save_rgb(
        make_sbs(left, right, "full_sbs", fused=False),
        output_dir / f"{prefix}_full_sbs.png",
    )
    save_rgb(
        make_sbs(left, right, "half_sbs", fused=False),
        output_dir / f"{prefix}_half_sbs.png",
    )


def _write_contact_sheet(output_dir: Path) -> None:
    from PIL import Image, ImageDraw

    paths = sorted(output_dir.glob("*.png"))
    if not paths:
        return
    tile_width, tile_height = 480, 300
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = None
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_width - 20, tile_height - 45), Image.Resampling.LANCZOS)
            x = (index % columns) * tile_width + (tile_width - image.width) // 2
            y = (index // columns) * tile_height + 8
            sheet.paste(image, (x, y))
            draw.text(
                ((index % columns) * tile_width + 10, (index // columns) * tile_height + tile_height - 30),
                path.stem,
                fill="white",
            )
        finally:
            if image is not None:
                image.close()
    sheet.save(output_dir / "visual_regression_contact_sheet.png")


def _load_reference_pair(reference_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    directory = Path(reference_dir)
    candidates = (
        (directory / "cuda_left_eye.png", directory / "cuda_right_eye.png"),
        (directory / "left_eye.png", directory / "right_eye.png"),
    )
    for left_path, right_path in candidates:
        if left_path.is_file() and right_path.is_file():
            return load_rgb(left_path), load_rgb(right_path)
    raise FileNotFoundError(
        f"CUDA reference directory must contain left_eye.png/right_eye.png: {directory}"
    )


def _rgba_to_tensor(rgba: Any) -> torch.Tensor:
    linear = torch.from_numpy(rgba[..., :3].copy()).permute(2, 0, 1).float() / 255.0
    value = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear.clamp_min(0.0031308), 1.0 / 2.4) - 0.055,
    )
    return value.unsqueeze(0)


def _run_vulkan_output_image_stage(
    rgb: torch.Tensor,
    prepared_depth: torch.Tensor,
    params: VulkanLayeredStereoParams,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run and read back the same storage-image shader used by zero-copy output."""
    from viewer.vulkan_context import VulkanContext, VulkanContextConfig
    from viewer.vulkan_descriptors import VulkanStorageImage
    from viewer.vulkan_resources import VulkanHostImage
    from .vulkan_backend import VulkanStereoImageComputeBackend

    context = VulkanContext.create(VulkanContextConfig(frame_context_count=3))
    left_image = right_image = left_readback = right_readback = None
    try:
        vk = context.vk
        output_usage = (
            vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        )
        left_image = VulkanStorageImage(
            context,
            int(rgb.shape[-1]),
            int(rgb.shape[-2]),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            usage=output_usage,
        )
        right_image = VulkanStorageImage(
            context,
            int(rgb.shape[-1]),
            int(rgb.shape[-2]),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            usage=output_usage,
        )
        left_ready = left_image.transition_to_general(role="compute")
        right_ready = right_image.transition_to_general(role="compute")
        context.wait_for_timeline(max(left_ready, right_ready))
        left_readback = VulkanHostImage(
            context,
            int(rgb.shape[-1]),
            int(rgb.shape[-2]),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            label="visual-regression-readback-left",
        )
        right_readback = VulkanHostImage(
            context,
            int(rgb.shape[-1]),
            int(rgb.shape[-2]),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            label="visual-regression-readback-right",
        )
        with VulkanStereoImageComputeBackend(context) as backend:
            timeline, debug = backend.submit_to_images(
                rgb,
                prepared_depth,
                left_image,
                right_image,
                params=params,
            )
        left_copy_timeline = context.copy_image(
            left_image,
            left_readback.resource,
            wait_for_timeline=timeline,
        )
        right_copy_timeline = context.copy_image(
            right_image,
            right_readback.resource,
            wait_for_timeline=timeline,
        )
        context.wait_for_timeline(max(left_copy_timeline, right_copy_timeline))
        left = _rgba_to_tensor(left_readback.read_rgba())
        right = _rgba_to_tensor(right_readback.read_rgba())
        debug = dict(debug)
        debug["visual_regression_shader"] = "d2s_stereo_layered_output"
        debug["visual_regression_readback"] = "temporary_host_image"
        return left, right, debug
    finally:
        for resource in (left_readback, right_readback, left_image, right_image):
            if resource is not None:
                resource.close()
        context.close()


def run_stage_visual_regression(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    output_dir: str | Path,
    config: StageVisualRegressionConfig = StageVisualRegressionConfig(),
    run_cuda: bool = True,
    cuda_device: str = "cuda",
    cuda_reference_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Save deterministic images for every inspectable stereo stage.

    The Vulkan stage uses the existing host-visible Vulkan Compute adapter so
    the shader output can be read back for diagnosis. Production zero-copy
    output is not changed by this helper.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rgb = rgb.detach().float().clamp(0.0, 1.0).contiguous()
    depth = depth.detach().float().clamp(0.0, 1.0).contiguous()
    if rgb.ndim != 4 or tuple(rgb.shape[:2]) != (1, 3):
        raise ValueError(f"rgb must have shape [1,3,H,W], got {tuple(rgb.shape)}")
    if depth.ndim != 4 or tuple(depth.shape[:2]) != (1, 1):
        raise ValueError(f"depth must have shape [1,1,H,W], got {tuple(depth.shape)}")

    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    if tuple(depth.shape[-2:]) != (height, width):
        depth = torch.nn.functional.interpolate(
            depth, size=(height, width), mode="bilinear", align_corners=False
        )
    prepared_depth = postprocess_depth(depth, depth_pop=0.0, antialias_strength=0.0)
    budget = resolve_parallax_budget(
        render_width=width,
        render_height=height,
        preset=config.parallax_preset,
        convergence=config.convergence,
        max_disparity_px=config.max_disparity_px,
    )
    resolved_max_disparity = float(budget.max_disparity_px)

    save_rgb(rgb, output / "00_capture_rgb.png")
    save_depth(depth, output / "01_raw_depth.png")
    save_depth(prepared_depth, output / "02_prepared_depth.png")

    vulkan_hole_fill_mode = resolve_vulkan_hole_fill_mode(
        config.hole_fill_mode,
        config.hole_fill_mode,
    )
    fill_radius, fill_strength = resolve_vulkan_hole_fill_parameters(
        vulkan_hole_fill_mode,
        fill_radius=config.hole_fill_radius,
        fill_strength=config.hole_fill_strength,
    )
    vulkan_params = VulkanLayeredStereoParams(
        depth_strength=max(0.0, float(config.depth_strength)),
        max_disparity_px=resolved_max_disparity,
        convergence=float(config.convergence),
        edge_threshold=float(config.edge_threshold),
        fill_strength=fill_strength,
        fill_radius=fill_radius,
        mask_feather_radius=max(0, min(3, int(config.mask_feather_radius))),
        symmetric=bool(config.symmetric),
        layers=max(1, min(4, int(config.layers))),
        softness=0.08,
        foreground_scale=max(0.0, float(config.foreground_scale)),
        midground_scale=max(0.0, float(config.midground_scale)),
        background_scale=max(0.0, float(config.background_scale)),
        edge_dilation=max(0, min(3, int(config.edge_dilation))),
        screen_edge_suppression=0,
        hole_fill_mode=vulkan_hole_fill_mode,
        occlusion_enabled=bool(config.occlusion),
    )

    stages: dict[str, Any] = {
        "config": asdict(config),
        "resolved_max_disparity_px": resolved_max_disparity,
        "input_shape": list(rgb.shape),
    }

    vulkan_mask: torch.Tensor | None
    try:
        vulkan_left, vulkan_right, vulkan_debug = _run_vulkan_output_image_stage(
            rgb, prepared_depth, vulkan_params
        )
        vulkan_mask = None
    except Exception as exc:
        from .vulkan_backend import VulkanStereoComputeBackend

        with VulkanStereoComputeBackend() as vulkan_backend:
            vulkan_left, vulkan_right, vulkan_mask, vulkan_debug = vulkan_backend.submit_layered_frame(
                rgb,
                prepared_depth,
                params=vulkan_params,
            )
        vulkan_debug = dict(vulkan_debug)
        vulkan_debug["visual_regression_shader"] = "d2s_stereo_layered"
        vulkan_debug["visual_regression_readback"] = "host_visible_storage_buffer"
        vulkan_debug["visual_regression_exact_output_error"] = f"{type(exc).__name__}: {exc}"
    _save_eye_pair(output, "03_vulkan", vulkan_left, vulkan_right)
    if vulkan_mask is not None:
        save_depth(vulkan_mask, output / "03_vulkan_occlusion_mask.png")
    else:
        stages["vulkan_occlusion_mask"] = {
            "skipped": "production output-image shader does not expose a mask image"
        }
    stages["vulkan"] = _json_safe(vulkan_debug)

    cuda_left: torch.Tensor | None = None
    cuda_right: torch.Tensor | None = None
    cuda_debug: dict[str, Any] | None = None
    if cuda_reference_dir is not None:
        cuda_left, cuda_right = _load_reference_pair(cuda_reference_dir)
        _save_eye_pair(output, "04_cuda_reference", cuda_left, cuda_right)
        stages["cuda_reference"] = {"source": str(cuda_reference_dir)}
    elif run_cuda and torch.cuda.is_available():
        device = torch.device(cuda_device)
        cuda_config = StereoConfig(
            backend="quality_4k",
            layers=int(config.layers),
            occlusion=bool(config.occlusion),
            symmetric=bool(config.symmetric),
            temporal=False,
            output_format="half_sbs",
            debug_output=True,
            depth_strength=float(config.depth_strength),
            convergence=float(config.convergence),
            max_disparity_px=resolved_max_disparity,
            parallax_preset=config.parallax_preset,
            foreground_shift_scale=float(config.foreground_scale),
            midground_shift_scale=float(config.midground_scale),
            background_shift_scale=float(config.background_scale),
            edge_dilation=int(config.edge_dilation),
            edge_threshold=float(config.edge_threshold),
            mask_feather_radius=int(config.mask_feather_radius),
            hole_fill_mode=str(config.hole_fill_mode),
            hole_fill_radius=int(config.hole_fill_radius),
            hole_fill_strength=float(config.hole_fill_strength),
            fused=True,
        )
        cuda_result = synthesize_stereo(
            rgb.to(device), depth.to(device), config=cuda_config
        )
        cuda_left = cuda_result.left_eye.detach().float().cpu()
        cuda_right = cuda_result.right_eye.detach().float().cpu()
        cuda_debug = cuda_result.debug_info
        _save_eye_pair(output, "04_cuda", cuda_left, cuda_right)
        cuda_mask = cuda_debug.get("occlusion_mask")
        if isinstance(cuda_mask, torch.Tensor):
            save_depth(cuda_mask.detach().float().cpu(), output / "04_cuda_occlusion_mask.png")
        stages["cuda"] = _json_safe(cuda_debug)
    else:
        stages["cuda"] = {"skipped": "CUDA unavailable or disabled"}

    if cuda_left is not None and cuda_right is not None:
        stages["comparison"] = {
            "left": compare_tensors(vulkan_left, cuda_left),
            "right": compare_tensors(vulkan_right, cuda_right),
        }
        save_rgb(
            diff_heatmap(vulkan_left, cuda_left),
            output / "05_diff_vulkan_vs_cuda_left_heatmap.png",
        )
        save_rgb(
            diff_heatmap(vulkan_right, cuda_right),
            output / "05_diff_vulkan_vs_cuda_right_heatmap.png",
        )
    else:
        stages["comparison"] = {"skipped": "CUDA reference unavailable"}

    _write_contact_sheet(output)
    manifest_path = output / "visual_regression_manifest.json"
    manifest_path.write_text(
        json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stages


def run_stage_visual_regression_from_paths(
    *,
    rgb_path: str | Path,
    depth_path: str | Path | None,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path) if depth_path is not None else make_depth_proxy_from_rgb(rgb)
    return run_stage_visual_regression(rgb=rgb, depth=depth, output_dir=output_dir, **kwargs)
