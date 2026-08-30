"""
Streaming aspect ratio and fit-mode handling, consistent with local viewer.

This module reuses the pure-python helpers from viewer.vulkan_local_viewer
to ensure identical letterbox/crop/stretch behavior between local and streaming paths.
"""

from __future__ import annotations

from typing import Any, Tuple

# Pure-math helpers from local viewer (no Vulkan deps)
# These are copied here to avoid import cycles and keep streaming self-contained.
# If local viewer changes, update here accordingly.


def normalize_display_fit_mode(value: str | None) -> str:
    """Normalize display fit mode to canonical form."""
    normalized = str(value or "contain").strip().casefold().replace("-", "_")
    aliases = {
        "contain": "contain",
        "complete": "contain",
        "fit": "contain",
        "keep ratio (complete)": "contain",
        "\u4fdd\u6301\u6bd4\u4f8a\uff08\u5b8c\u6574\uff09": "contain",
        "\u4fdd\u6301\u6bd4\u4f8a(\u5b8c\u6574)": "contain",
        "cover": "cover",
        "fill": "cover",
        "keep ratio (fill)": "cover",
        "\u4fdd\u6301\u6bd4\u4f8a\uff08\u94fa\u6ee1\uff09": "cover",
        "\u4fdd\u6301\u6bd4\u4f8a(\u94fa\u6ee1)": "cover",
        "stretch": "stretch",
        "stretch to fill": "stretch",
        "\u62c9\u4f38\u94fa\u6ee1": "stretch",
    }
    return aliases.get(normalized, "contain")


def _aspect_close(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Return whether two sizes have the same aspect within rounding tolerance."""
    aw, ah = a
    bw, bh = b
    if min(aw, ah, bw, bh) <= 0:
        return False
    return abs(aw * bh - ah * bw) <= max(2, (aw * bh) // 100)


def frame_hw(frame: Any) -> Tuple[int, int]:
    """Return (h, w) for CHW/HWC torch or numpy frames."""
    shape = tuple(int(value) for value in getattr(frame, "shape", ()))
    if len(shape) == 4:
        if shape[-1] in (1, 3, 4):
            return shape[-3], shape[-2]  # B,H,W,C
        return shape[-2], shape[-1]  # B,C,H,W
    if len(shape) == 3:
        if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
            return shape[-2], shape[-1]  # C,H,W
        return shape[0], shape[1]  # H,W,C
    return 0, 0


def _input_eye_size(
    source: Tuple[int, int],
    display_mode: str | None,
    input_size: Tuple[int, int] | None,
) -> Tuple[int, int]:
    """Return the per-eye capture size for the given packed source.

    The packed ``source`` is un-packed according to ``display_mode`` (half
    modes keep the packed aspect, full modes halve the packed SBS width / TAB
    height). This is the ground truth of the current frame, so it is always
    preferred: a stale ``input_size`` captured at startup would otherwise make
    the per-eye fit non-uniform and distort the eye content. ``input_size`` is
    kept only as an advisory cross-check for aspect-close sizes.
    """
    sw, sh = source
    packed_mode = str(display_mode or "").strip().casefold().replace("_", "-")
    if packed_mode == "full-sbs":
        eye_size = max(1, sw // 2), sh
    elif packed_mode == "full-tab":
        eye_size = sw, max(1, sh // 2)
    else:
        eye_size = sw, sh
    if input_size is not None:
        try:
            iw, ih = int(input_size[0]), int(input_size[1])
        except (TypeError, ValueError):
            return eye_size
        if iw > 0 and ih > 0 and _aspect_close((iw, ih), eye_size):
            return iw, ih
    return eye_size


def pad_eye_to_16_9(eye_size: Tuple[int, int]) -> Tuple[int, int]:
    """Contain-pad one eye to a 16:9 canvas (legacy ``fill_16_9`` semantics).

    Wider than 16:9 grows the height (top/bottom bars); narrower than 16:9
    grows the width (left/right bars). Only the canvas width is even-aligned;
    the height is kept exactly as derived so it still matches the source frame
    height. Truncating the height (e.g. 801 -> 800) made the per-eye fit
    non-uniform and distorted the eye content for odd-height inputs.
    """
    iw, ih = eye_size
    if min(iw, ih) <= 0:
        return max(1, iw), max(1, ih)
    if iw * 9 > ih * 16:
        pw, ph = iw, max(1, round(iw * 9 / 16))
    else:
        pw, ph = max(1, round(ih * 16 / 9)), ih
    pw -= pw % 2
    return max(1, pw), max(1, ph)


def input_needs_16_9_canvas(eye_size: Tuple[int, int]) -> bool:
    """Return whether contain-padding the input eye to 16:9 changes its size."""
    iw, ih = eye_size
    if min(iw, ih) <= 0:
        return False
    return pad_eye_to_16_9((iw, ih)) != (iw, ih)


def transport_canvas_size(
    source: Tuple[int, int],
    fit_mode: str,
    input_size: Tuple[int, int] | None = None,
    display_mode: str | None = None,
) -> Tuple[int, int]:
    """
    Return the transport canvas size for a fit mode, mirroring local viewer.

    contain (Keep Ratio Complete): each eye is contain-padded to a 16:9 canvas
    first, exactly like the legacy ``fill_16_9`` flag in ``make_sbs_core``,
    then the padded eyes are packed. Half modes therefore produce a 16:9
    frame; Full-SBS / Full-TAB produce a 2x16:9 / 16:9x2 frame. Inputs whose
    aspect ratio is already 16:9 are passed through unchanged. Per-eye
    placement inside the canvas is handled by presentation_blit_regions.
    cover/stretch (Keep Ratio Fill / Stretch to Fill): keep the original
    input aspect, i.e. the packed source size unchanged.
    """
    sw, sh = source
    if min(sw, sh) <= 0:
        return source
    if normalize_display_fit_mode(fit_mode) != "contain":
        return source
    iw, ih = _input_eye_size(source, display_mode, input_size)
    padded = pad_eye_to_16_9((iw, ih))
    if _aspect_close(padded, (iw, ih)):
        # The input aspect ratio is already 16:9; keep the packed size.
        return source
    pw, ph = padded
    packed_mode = str(display_mode or "").strip().casefold().replace("_", "-")
    if packed_mode == "full-sbs":
        return pw * 2, ph
    if packed_mode == "full-tab":
        return pw, ph * 2
    # Half-SBS / Half-TAB / mono / unknown: the final frame is the padded eye.
    return pw, ph


def fit_rect(source: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Return an aspect-correct, centered destination rectangle (x, y, w, h)."""
    sw, sh = source
    tw, th = target
    if min(sw, sh, tw, th) <= 0:
        return 0, 0, 0, 0
    scale = min(tw / sw, th / sh)
    width, height = max(1, round(sw * scale)), max(1, round(sh * scale))
    return (tw - width) // 2, (th - height) // 2, width, height


def _cover_crop_rect(
    source: Tuple[int, int], target_aspect: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Return a centered source crop matching the requested aspect ratio (x, y, w, h)."""
    sw, sh = source
    tw, th = target_aspect
    if min(sw, sh, tw, th) <= 0:
        return 0, 0, max(0, sw), max(0, sh)
    if sw * th > sh * tw:
        width = max(1, min(sw, round(sh * tw / th)))
        return (sw - width) // 2, 0, width, sh
    height = max(1, min(sh, round(sw * th / tw)))
    return 0, (sh - height) // 2, sw, height


def presentation_blit_regions(
    source: Tuple[int, int],
    target: Tuple[int, int],
    fit_mode: str,
    display_mode: str = "Half-SBS",
    input_size: Tuple[int, int] | None = None,
) -> Tuple[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]], ...]:
    """
    Resolve source/destination blits without mixing the two packed eyes.

    Returns a tuple of 2 regions for SBS/TAB modes: ((src_x0, src_y0, src_x1, src_y1), (dst_x0, dst_y0, dst_x1, dst_y1))
    For non-packed modes, returns single region.

    Args:
        source: (sw, sh) packed SBS size from runtime
        target: (tw, th) transport size (EncoderProfile.resize_size or output canvas)
        fit_mode: "contain" | "cover" | "stretch"
        display_mode: "Half-SBS" | "Full-SBS" | "Half-TAB" | "Full-TAB"
        input_size: (tex_w, tex_h) original capture WxH before packing (for dynamic eye ratio)
    """
    sw, sh = source
    tw, th = target
    if min(sw, sh, tw, th) <= 0:
        return ()
    mode = normalize_display_fit_mode(fit_mode)
    full_source = (0, 0, sw, sh)
    full_target = (0, 0, tw, th)
    packed_mode = str(display_mode or "").strip().casefold().replace("_", "-")

    if packed_mode in {"half-sbs", "full-sbs", "half-tab", "full-tab"}:
        is_sbs = packed_mode.endswith("sbs")
        is_half = packed_mode.startswith("half-")
        if is_sbs:
            source_split = sw // 2
            encoded_eye_size = (source_split, sh)
            logical_eye_size = (source_split * (2 if is_half else 1), sh)
            source_origins = ((0, 0), (source_split, 0))
        else:
            source_split = sh // 2
            encoded_eye_size = (sw, source_split)
            logical_eye_size = (sw, source_split * (2 if is_half else 1))
            source_origins = ((0, 0), (0, source_split))

        if mode == "stretch":
            half_w, half_h = (tw // 2, th) if is_sbs else (tw, th // 2)
            if is_sbs:
                destinations = ((0, 0, half_w, th), (half_w, 0, tw, th))
            else:
                destinations = ((0, 0, tw, half_h), (0, half_h, tw, th))
            regions = []
            for (ox, oy), dest in zip(source_origins, destinations):
                ew, eh = encoded_eye_size
                regions.append(((ox, oy, ox + ew, oy + eh), dest))
            return tuple(regions)

        if mode == "contain":
            half_w, half_h = (tw // 2, th) if is_sbs else (tw, th // 2)
            # Fit the ENCODED (packed) eye into each half. The destination rect
            # then always has the same aspect as the source crop, so the blit
            # scales uniformly and the eye content can never be distorted --
            # even when a startup-captured input_size goes stale (window or
            # monitor resized mid-stream). With the transport canvas derived
            # from the packed frame, this is exactly the legacy fill_16_9
            # pad-then-place geometry (identity 1:1 placement + bars).
            x, y, w, h = fit_rect(encoded_eye_size, (half_w, half_h))
            if is_sbs:
                left_dest = (x, y, x + w, y + h)
                right_dest = (half_w + x, y, half_w + x + w, y + h)
                destinations = (left_dest, right_dest)
            else:
                top_dest = (x, y, x + w, y + h)
                bottom_dest = (x, half_h + y, x + w, half_h + y + h)
                destinations = (top_dest, bottom_dest)
            regions = []
            for (ox, oy), dest in zip(source_origins, destinations):
                ew, eh = encoded_eye_size
                regions.append(((ox, oy, ox + ew, oy + eh), dest))
            return tuple(regions)

        if mode == "cover":
            half_w, half_h = (tw // 2, th) if is_sbs else (tw, th // 2)
            if input_size is not None:
                iw, ih = input_size
                if is_sbs:
                    eye_size = (max(1, iw // 2), ih) if is_half else (iw, ih)
                else:
                    eye_size = (iw, max(1, ih // 2)) if is_half else (iw, ih)
            else:
                eye_size = encoded_eye_size
            cx_eye, cy_eye, cw_eye, ch_eye = _cover_crop_rect(
                eye_size, (half_w, half_h)
            )
            # Map eye crop to packed eye coords (encoded is stretched eye)
            if eye_size != encoded_eye_size:
                sx = encoded_eye_size[0] / eye_size[0] if eye_size[0] else 1.0
                sy = encoded_eye_size[1] / eye_size[1] if eye_size[1] else 1.0
                crop_x = int(round(cx_eye * sx))
                crop_y = int(round(cy_eye * sy))
                crop_w = int(round(cw_eye * sx))
                crop_h = int(round(ch_eye * sy))
            else:
                crop_x, crop_y, crop_w, crop_h = cx_eye, cy_eye, cw_eye, ch_eye
            if is_sbs:
                destinations = ((0, 0, half_w, th), (half_w, 0, tw, th))
            else:
                destinations = ((0, 0, tw, half_h), (0, half_h, tw, th))
            regions = []
            for (ox, oy), dest in zip(source_origins, destinations):
                regions.append((
                    (ox + crop_x, oy + crop_y, ox + crop_x + crop_w, oy + crop_y + crop_h),
                    dest,
                ))
            return tuple(regions)

        # fallback
        crop_x, crop_y = 0, 0
        crop_w, crop_h = encoded_eye_size
        target_box = full_target
        tx0, ty0, tx1, ty1 = target_box
        if is_sbs:
            target_split = tx0 + (tx1 - tx0) // 2
            destination_regions = (
                (tx0, ty0, target_split, ty1),
                (target_split, ty0, tx1, ty1),
            )
        else:
            target_split = ty0 + (ty1 - ty0) // 2
            destination_regions = (
                (tx0, ty0, tx1, target_split),
                (tx0, target_split, tx1, ty1),
            )
        regions = []
        for (origin_x, origin_y), destination_rect in zip(
            source_origins, destination_regions
        ):
            regions.append((
                (
                    origin_x + crop_x,
                    origin_y + crop_y,
                    origin_x + crop_x + crop_w,
                    origin_y + crop_y + crop_h,
                ),
                destination_rect,
            ))
        return tuple(regions)

    if mode == "stretch":
        return ((full_source, full_target),)
    if mode == "contain":
        x, y, width, height = fit_rect(source, target)
        return ((full_source, (x, y, x + width, y + height)),)
    crop_x, crop_y, crop_w, crop_h = _cover_crop_rect(source, target)
    return (((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h), full_target),)


def apply_aspect_on_cpu(
    frame: "np.ndarray",
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    fit_mode: str,
    display_mode: str,
    input_size: Tuple[int, int] | None = None,
) -> "np.ndarray":
    """
    CPU fallback: apply aspect-ratio processing using cv2.
    Returns correctly sized frame ready for JPEG encode.
    """
    import cv2
    import numpy as np

    regions = presentation_blit_regions(
        source_size, target_size, fit_mode, display_mode, input_size
    )
    if not regions:
        return frame

    tw, th = target_size
    is_packed = str(display_mode or "").strip().casefold().replace("_", "-") in {
        "half-sbs", "full-sbs", "half-tab", "full-tab"
    }

    if is_packed and len(regions) == 2:
        # Per-eye blit
        output = np.zeros((th, tw, 3), dtype=np.uint8)
        for (src_rect, dst_rect) in regions:
            sx0, sy0, sx1, sy1 = src_rect
            dx0, dy0, dx1, dy1 = dst_rect
            src_crop = frame[sy0:sy1, sx0:sx1]
            if src_crop.size == 0:
                continue
            dw, dh = dx1 - dx0, dy1 - dy0
            if src_crop.shape[1] != dw or src_crop.shape[0] != dh:
                interp = cv2.INTER_AREA if (src_crop.shape[1] > dw or src_crop.shape[0] > dh) else cv2.INTER_LINEAR
                src_crop = cv2.resize(src_crop, (dw, dh), interpolation=interp)
            output[dy0:dy1, dx0:dx1] = src_crop
        return np.ascontiguousarray(output)
    else:
        # Single region
        (src_rect, dst_rect), = regions
        sx0, sy0, sx1, sy1 = src_rect
        dx0, dy0, dx1, dy1 = dst_rect
        src_crop = frame[sy0:sy1, sx0:sx1]
        dw, dh = dx1 - dx0, dy1 - dy0
        if src_crop.shape[1] != dw or src_crop.shape[0] != dh:
            interp = cv2.INTER_AREA if (src_crop.shape[1] > dw or src_crop.shape[0] > dh) else cv2.INTER_LINEAR
            src_crop = cv2.resize(src_crop, (dw, dh), interpolation=interp)
        output = np.zeros((th, tw, 3), dtype=np.uint8)
        output[dy0:dy1, dx0:dx1] = src_crop
        return np.ascontiguousarray(output)


def apply_aspect_on_gpu(
    frame: "torch.Tensor",
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    fit_mode: str,
    display_mode: str,
    input_size: Tuple[int, int] | None = None,
) -> "torch.Tensor":
    """
    GPU path: apply aspect-ratio processing using torch on CUDA.
    frame: CUDA tensor, expected shape [H, W, 3] or [1, 3, H, W] (uint8 or float32 0..1)
    Returns CUDA tensor [H_t, W_t, 3] uint8.
    """
    import torch
    import torch.nn.functional as F

    # Normalize to [H, W, 3] uint8 on CUDA
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"unexpected frame shape {tuple(frame.shape)}")
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        # CHW -> HWC
        frame = frame.permute(1, 2, 0)
    h, w, c = frame.shape
    if c == 1:
        frame = frame.expand(-1, -1, 3)
    elif c == 4:
        frame = frame[..., :3]
    elif c != 3:
        raise ValueError(f"unexpected channel count {c}")
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    frame = frame.contiguous()

    regions = presentation_blit_regions(
        source_size, target_size, fit_mode, display_mode, input_size
    )
    if not regions:
        return frame

    tw, th = target_size
    is_packed = str(display_mode or "").strip().casefold().replace("_", "-") in {
        "half-sbs", "full-sbs", "half-tab", "full-tab"
    }

    if is_packed and len(regions) == 2:
        # Per-eye blit
        output = torch.zeros((th, tw, 3), dtype=torch.uint8, device=frame.device)
        for (src_rect, dst_rect) in regions:
            sx0, sy0, sx1, sy1 = src_rect
            dx0, dy0, dx1, dy1 = dst_rect
            src_crop = frame[sy0:sy1, sx0:sx1]
            if src_crop.numel() == 0:
                continue
            dw, dh = dx1 - dx0, dy1 - dy0
            if src_crop.shape[1] != dw or src_crop.shape[0] != dh:
                # Resize on GPU
                src_crop = src_crop.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1, 3, H, W]
                mode = "area" if (src_crop.shape[3] > dw or src_crop.shape[2] > dh) else "bilinear"
                if mode == "area":
                    src_crop = F.interpolate(src_crop, size=(dh, dw), mode=mode)
                else:
                    src_crop = F.interpolate(src_crop, size=(dh, dw), mode=mode, align_corners=False)
                src_crop = (src_crop.clamp(0, 1) * 255).to(torch.uint8)[0].permute(1, 2, 0)
            output[dy0:dy1, dx0:dx1] = src_crop
        return output.contiguous()
    else:
        (src_rect, dst_rect), = regions
        sx0, sy0, sx1, sy1 = src_rect
        dx0, dy0, dx1, dy1 = dst_rect
        src_crop = frame[sy0:sy1, sx0:sx1]
        dw, dh = dx1 - dx0, dy1 - dy0
        if src_crop.shape[1] != dw or src_crop.shape[0] != dh:
            src_crop = src_crop.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mode = "area" if (src_crop.shape[3] > dw or src_crop.shape[2] > dh) else "bilinear"
            if mode == "area":
                src_crop = F.interpolate(src_crop, size=(dh, dw), mode=mode)
            else:
                src_crop = F.interpolate(src_crop, size=(dh, dw), mode=mode, align_corners=False)
            src_crop = (src_crop.clamp(0, 1) * 255).to(torch.uint8)[0].permute(1, 2, 0)
        output = torch.zeros((th, tw, 3), dtype=torch.uint8, device=frame.device)
        output[dy0:dy1, dx0:dx1] = src_crop
        return output.contiguous()