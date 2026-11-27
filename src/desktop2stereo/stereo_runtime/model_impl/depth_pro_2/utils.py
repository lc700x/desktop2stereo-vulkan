#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps

try:
    import pillow_heif
except ImportError:
    pillow_heif = None
else:
    pillow_heif.register_heif_opener()


def default_device() -> torch.device:
    """CUDA if it is available, then MPS, then the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fpx_from_f35(width: float, height: float, f_mm: float) -> float:
    """Convert a 35mm-equivalent focal length in mm to pixels for a given image size."""
    return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)


def upright(frame: Any) -> np.ndarray:
    """A decoded frame as an RGB ``(H, W, 3)`` uint8 array, turned upright per its rotation."""
    image = frame.to_ndarray(format="rgb24")
    if frame.rotation:
        image = np.ascontiguousarray(np.rot90(image, frame.rotation // 90))
    return image


def get_supported_image_extensions() -> tuple[str, ...]:
    """Return the image file extensions that :func:`load_rgb` can read."""
    # Pillow only writes PDF and PALM, recognizes MPEG video without decoding it, and needs
    # Ghostscript for EPS and a registered handler for the BUFR, GRIB, HDF5 and WMF stubs. PFM
    # and FITS hold float and scientific data, not photos.
    unreadable = ("PDF", "PALM", "MPEG", "EPS", "BUFR", "GRIB", "HDF5", "WMF", "FITS")
    extensions = {
        ext.lower()
        for ext, fmt in Image.registered_extensions().items()
        if fmt not in unreadable and ext != ".pfm"
    }
    return tuple(sorted(extensions))


def load_rgb(path: Path | str) -> tuple[np.ndarray, float | None]:
    """Load an image as RGB, upright per its EXIF orientation.

    Returns:
        The ``(H, W, 3)`` uint8 image, and the focal length in pixels derived from the
        35mm-equivalent EXIF tag, or ``None`` if the image does not carry one.
    """
    img_pil = Image.open(path)
    f_35mm = img_pil.getexif().get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.FocalLengthIn35mmFilm)
    ImageOps.exif_transpose(img_pil, in_place=True)

    # Pillow clips 16-bit values to 8 bits when converting, so scale them first.
    if img_pil.mode.startswith("I;16"):
        img_pil = Image.fromarray((np.array(img_pil) >> 8).astype(np.uint8))
    elif img_pil.mode in ("I", "F"):
        raise ValueError(f"Unsupported image mode {img_pil.mode} in {path}.")
    img = np.array(img_pil.convert("RGB"))

    f_px = None
    if isinstance(f_35mm, numbers.Real) and f_35mm > 0:
        f_px = fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
    return img, f_px
