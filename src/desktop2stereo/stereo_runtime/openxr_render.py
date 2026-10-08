from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from .baseline_shift import ShiftParams, compute_shift_px, shift_debug_info, warp_horizontal
from .output import ensure_bchw, match_depth

PaddingMode = Literal["zeros", "border", "reflection"]
OpenXROutputMode = Literal["auto", "rgb_depth", "full_synthesis_eyes"]


@dataclass(frozen=True)
class OpenXRRenderConfig:
    depth_strength: float = 2.0
    convergence: float | torch.Tensor = 0.0
    max_disparity_px: float | None = None
    parallax_preset: str = "standard"
    output_mode: OpenXROutputMode = "auto"
    foreground_shift_scale: float = 1.0
    midground_shift_scale: float = 1.0
    background_shift_scale: float = 1.0
    screen_roll: float = 0.0
    # A reflected sample creates a mirrored copy of objects at the screen
    # boundary when disparity pushes a pixel outside the source image.  Clamp
    # to the edge instead; this keeps the final eye image free of double edges.
    padding_mode: PaddingMode = "border"
    vulkan_projection_min_lod: float = 0.0
    vulkan_projection_max_lod: float = 0.35
    vulkan_projection_mip_lod_bias: float = -0.35
    vulkan_projection_rcas_sharpness: float = 0.5


@dataclass(frozen=True)
class OpenXRScreenPose:
    width_m: float = 2.4
    height_m: float = 1.35
    distance_m: float = 2.0
    pan_x_m: float = 0.0
    pan_y_m: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class OpenXRFov:
    angle_left: float
    angle_right: float
    angle_up: float
    angle_down: float


@dataclass(frozen=True)
class OpenXREyeView:
    eye_index: int
    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    fov: OpenXRFov


@dataclass(frozen=True)
class OpenXRStereoResult:
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    debug_info: dict[str, torch.Tensor | float | int | str] = field(default_factory=dict)


def is_pyopenxr_available() -> bool:
    try:
        import xr  # noqa: F401
    except Exception:
        return False
    return True


def pyopenxr_module_name() -> str:
    return "xr"


def render_openxr_stereo(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: OpenXRRenderConfig | None = None,
) -> OpenXRStereoResult:
    config = config or OpenXRRenderConfig()
    # Compute the depth match, parallax shift, and RGB float conversion ONCE and
    # share them across both eyes.  Previously each eye re-ran match_depth (a real
    # F.interpolate when the depth model output is smaller than the RGB frame) and
    # compute_shift_px, plus a third match/shift for the debug field -- 3x depth
    # interpolation and 5x shift per frame.  The two eyes differ only by eye_sign
    # (a sign flip applied inside warp_horizontal), so the matched depth and base
    # shift are identical and can be reused.
    rgb_bchw = ensure_bchw(rgb, name="rgb").float()
    _, _, h, w = rgb_bchw.shape
    depth_matched = match_depth(depth, h, w, rgb=rgb_bchw, edge_aware=True)
    params = _shift_params(config)
    base_shift = compute_shift_px(depth_matched, w, params)
    left = _render_eye_from_matched(rgb_bchw, depth_matched, eye_sign=-1.0, config=config, base_shift=base_shift)
    right = _render_eye_from_matched(rgb_bchw, depth_matched, eye_sign=1.0, config=config, base_shift=base_shift)
    return OpenXRStereoResult(
        left_eye=left,
        right_eye=right,
        debug_info={
            "backend": "openxr_roll_adaptive_grid_sample",
            "screen_roll": float(config.screen_roll),
            "shift_px": base_shift,
            **shift_debug_info(depth_matched, w, params),
        },
    )


def render_openxr_eye(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    eye_sign: float,
    config: OpenXRRenderConfig | None = None,
) -> torch.Tensor:
    config = config or OpenXRRenderConfig()
    rgb = ensure_bchw(rgb, name="rgb").float()
    _, _, h, w = rgb.shape
    depth_matched = match_depth(depth, h, w, rgb=rgb, edge_aware=True)
    return _render_eye_from_matched(rgb, depth_matched, eye_sign=eye_sign, config=config)


def _render_eye_from_matched(
    rgb: torch.Tensor,
    depth_matched: torch.Tensor,
    *,
    eye_sign: float,
    config: OpenXRRenderConfig,
    base_shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp a single eye from a pre-matched depth map.

    `rgb` must already be BCHW float and `depth_matched` already resized to
    rgb's (h, w); both are computed once per frame and shared between eyes.
    """
    b, _, h, w = rgb.shape
    if base_shift is None:
        base_shift = compute_shift_px(depth_matched, w, _shift_params(config))

    if config.screen_roll == 0.0:
        return warp_horizontal(
            rgb,
            base_shift,
            eye_sign=float(eye_sign),
            padding_mode=config.padding_mode,
        )

    shift_px = base_shift * float(eye_sign)
    yy, xx = _base_grid_components(h, w, rgb.device, rgb.dtype)
    cos_r = math.cos(config.screen_roll)
    sin_r = math.sin(config.screen_roll)
    shift_x = (2.0 * shift_px.squeeze(1) * cos_r) / max(w - 1, 1)
    shift_y = (2.0 * shift_px.squeeze(1) * sin_r) / max(h - 1, 1)
    grid = torch.stack((xx.unsqueeze(0) + shift_x, yy.unsqueeze(0) + shift_y), dim=-1)
    grid = grid.expand(b, h, w, 2)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode=config.padding_mode, align_corners=True)


def build_openxr_eye_mvp(
    eye_view: OpenXREyeView,
    screen_pose: OpenXRScreenPose | None = None,
    *,
    clip_space: Literal["opengl", "d3d"] = "opengl",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    view = pose_to_view_matrix(
        eye_view.position_xyz,
        eye_view.orientation_xyzw,
        device=device,
        dtype=dtype,
    )
    projection = fov_to_projection_matrix(
        eye_view.fov.angle_left,
        eye_view.fov.angle_right,
        eye_view.fov.angle_up,
        eye_view.fov.angle_down,
        clip_space=clip_space,
        device=device,
        dtype=dtype,
    )
    model = build_openxr_screen_model_matrix(screen_pose, device=device, dtype=dtype)
    return projection @ view @ model


def build_openxr_screen_model_matrix(
    pose: OpenXRScreenPose | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    pose = pose or OpenXRScreenPose()
    scale = torch.diag(
        torch.tensor([pose.width_m * 0.5, pose.height_m * 0.5, 1.0, 1.0], device=device, dtype=dtype)
    )
    rotation = euler_to_matrix(pose.yaw, pose.pitch, pose.roll, device=device, dtype=dtype)
    translation = torch.eye(4, device=device, dtype=dtype)
    translation[0, 3] = pose.pan_x_m
    translation[1, 3] = pose.pan_y_m
    translation[2, 3] = -pose.distance_m
    return translation @ rotation @ scale


def pose_to_view_matrix(
    position_xyz: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rotation = quaternion_xyzw_to_matrix(orientation_xyzw, device=device, dtype=dtype)[:3, :3]
    rt = rotation.transpose(0, 1)
    position = torch.tensor(position_xyz, device=device, dtype=dtype)
    view = torch.eye(4, device=device, dtype=dtype)
    view[:3, :3] = rt
    view[:3, 3] = -(rt @ position)
    return view


def fov_to_projection_matrix(
    angle_left: float,
    angle_right: float,
    angle_up: float,
    angle_down: float,
    *,
    near: float = 0.05,
    far: float = 100.0,
    clip_space: Literal["opengl", "d3d"] = "opengl",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    left = math.tan(angle_left) * near
    right = math.tan(angle_right) * near
    top = math.tan(angle_up) * near
    bottom = math.tan(angle_down) * near
    eps = 1e-6
    if abs(right - left) < eps:
        right += eps
    if abs(top - bottom) < eps:
        top += eps

    projection = torch.zeros((4, 4), device=device, dtype=dtype)
    projection[0, 0] = 2.0 * near / (right - left)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 1] = 2.0 * near / (top - bottom)
    projection[1, 2] = (top + bottom) / (top - bottom)
    if clip_space == "d3d":
        projection[2, 2] = -far / (far - near)
        projection[2, 3] = -(far * near) / (far - near)
    else:
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -2.0 * far * near / (far - near)
    projection[3, 2] = -1.0
    return projection


def euler_to_matrix(
    yaw: float,
    pitch: float,
    roll: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rot_y = torch.tensor(
        [[cy, 0.0, sy, 0.0], [0.0, 1.0, 0.0, 0.0], [-sy, 0.0, cy, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    rot_x = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, cp, -sp, 0.0], [0.0, sp, cp, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    rot_z = torch.tensor(
        [[cr, -sr, 0.0, 0.0], [sr, cr, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return rot_y @ rot_x @ rot_z


def quaternion_xyzw_to_matrix(
    orientation_xyzw: tuple[float, float, float, float],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x, y, z, w = orientation_xyzw
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0.0],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0.0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _shift_params(config: OpenXRRenderConfig) -> ShiftParams:
    return ShiftParams(
        depth_strength=config.depth_strength,
        convergence=config.convergence,
        max_disparity_px=config.max_disparity_px,
        parallax_preset=config.parallax_preset,
        foreground_shift_scale=config.foreground_shift_scale,
        midground_shift_scale=config.midground_shift_scale,
        background_shift_scale=config.background_shift_scale,
    )


def _base_grid_components(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx
