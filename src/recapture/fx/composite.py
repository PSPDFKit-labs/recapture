"""Background layer: a photo (or a plain color) placed under the rendered page."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .blur import gaussian_blur

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_x = np.arange(256, dtype=np.float64) / 255.0
_SRGB_TO_LINEAR = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def background_layer(image: np.ndarray | None, cfg: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
    """Linear RGB (H, W, 3) background with mean luminance 1 (scaled later)."""
    w, h = size
    if image is None:
        color = np.asarray(cfg["color"], dtype=np.float32)
        return np.broadcast_to(color / max(float(color @ _LUMA), 1e-6), (h, w, 3)).copy()

    img = np.rot90(image[..., :3], k=int(cfg["quarter_turns"]) % 4)
    if cfg["flip"]:
        img = img[:, ::-1]
    ih, iw = img.shape[:2]
    # Crop the largest region with the frame's aspect, shrunk by `zoom`.
    scale = min(iw / w, ih / h) / max(float(cfg["zoom"]), 1.0)
    cw, ch = w * scale, h * scale
    cx = np.clip(float(cfg["center"][0]) * iw, cw / 2, iw - cw / 2)
    cy = np.clip(float(cfg["center"][1]) * ih, ch / 2, ih - ch / 2)
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    pil = Image.fromarray(np.ascontiguousarray(img)).resize((w, h), Image.Resampling.BICUBIC, box=box)
    lin = _SRGB_TO_LINEAR[np.asarray(pil)]

    sat = float(cfg["saturation"])
    if sat != 1.0:
        lum = (lin @ _LUMA)[..., None]
        lin = np.maximum(lum + sat * (lin - lum), 0.0)
    if cfg["blur_px"] > 0:
        lin = gaussian_blur(lin, float(cfg["blur_px"]))
    return lin / max(float((lin @ _LUMA).mean()), 1e-6)


def composite(page: np.ndarray, alpha: np.ndarray, background: np.ndarray, brightness: float,
              light: float | np.ndarray) -> np.ndarray:
    """Premultiplied page over the background ("normal" blend mode).

    `light` is the light arriving at the background: one number (the paper
    white, for a flat layer) or an (H, W, 3) buffer from the renderer's table
    pass, with the lamp's falloff, color and shadows. `brightness` is the
    background's reflectance relative to the paper, so both layers stay in a
    believable ratio whatever the light.
    """
    bg = background * (np.float32(brightness) * np.asarray(light, dtype=np.float32))
    return page + bg * (1.0 - alpha)[..., None]
