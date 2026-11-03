"""Background layer and 2D camera effects on the rendered linear image.

First the background photo goes under the page, like a layer in an image
editor. Then the camera effects run on both layers together, so they share
one camera look. Order follows a phone camera: optics (defocus, motion blur, vignetting),
then the sensor (noise), then the image processor (white balance, auto
exposure, tone mapping, sRGB, sharpening). JPEG happens when the file is
written. No effect moves pixels, so the labels stay valid. With the
background off, the result keeps the page's alpha (a transparent PNG).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .blur import gaussian_blur, motion_blur
from .composite import background_layer, composite

_LUT_SIZE = 1 << 16
_x = np.linspace(0.0, 1.0, _LUT_SIZE)
_SRGB_LUT = (np.where(_x <= 0.0031308, 12.92 * _x, 1.055 * np.power(_x, 1.0 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def linear_to_srgb8(color: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Linear float RGB -> sRGB uint8, through a 16-bit lookup table."""
    scale = np.float32(exposure * (_LUT_SIZE - 1))
    idx = np.clip(color * scale + np.float32(0.5), 0, _LUT_SIZE - 1).astype(np.uint16)
    return _SRGB_LUT[idx]


def filmic(x: np.ndarray) -> np.ndarray:
    """ACES fit by K. Narkowicz: soft shoulder, rich contrast, 1.0 maps to about 0.8."""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def filmic_inverse(y: float) -> float:
    lo, hi = 0.0, 64.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if filmic(np.array(mid)) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_fx(color: np.ndarray, alpha: np.ndarray, label: np.ndarray, fx: dict[str, Any],
             focus_depth: float | None, bg_image: np.ndarray | None, bg_cfg: dict[str, Any],
             bg_layer: np.ndarray | None = None, table: np.ndarray | None = None,
             desk: bool = False) -> np.ndarray:
    """Premultiplied page render + label buffer + background -> final sRGB uint8 image.

    With `background.enabled` false there is no background layer: the result
    is RGBA, transparent around the page. `bg_layer` is a precomputed
    `background_layer(...)`, for callers that cache it. `table` is the
    renderer's table buffer (light arriving at the table, and its depth); with
    it, the background is lit by the same lamp as the page and shows shadows.
    With `desk`, `table` holds the light the desk reflects: it is the background.
    """
    h, w = alpha.shape
    page = label[..., 2] > 0.5
    if page.sum() < 100:
        page = np.ones(page.shape, dtype=bool)
    # Paper white: a high percentile of the fully covered page pixels.
    solid = (alpha > 0.99) & page
    lum = (color[::4, ::4] @ _LUMA)[solid[::4, ::4]]
    paper_white = float(np.percentile(lum, 97)) if lum.size else 1.0
    transparent = not bg_cfg.get("enabled", True)
    if transparent:
        # Carry alpha with the premultiplied color, so the blurs soften the page edge too.
        img = np.concatenate([color, alpha[..., None]], axis=2).astype(np.float32)
    elif desk:
        img = composite(np.ascontiguousarray(color, dtype=np.float32), alpha, table[..., :3], 1.0, 1.0)
    else:
        bg = background_layer(bg_image, bg_cfg, (w, h)) if bg_layer is None else bg_layer
        light = paper_white if table is None else table[..., :3]
        img = composite(np.ascontiguousarray(color, dtype=np.float32), alpha, bg, float(bg_cfg["brightness"]), light)
    if not fx["enabled"]:
        return _with_alpha(linear_to_srgb8(unpremultiply(img)), img) if transparent else linear_to_srgb8(img)
    rng = np.random.default_rng(int(fx["seed"]))

    # Optics.
    d = fx["defocus"]
    if d["enabled"] and focus_depth and d["strength_px"] > 0:
        # A flat background layer has no depth: treat it as in focus. A table has its own.
        far = np.float32(focus_depth) if table is None or transparent else table[..., 3]
        # Desk items have their depth in the label buffer, in front of the desk.
        depth = np.where(page, label[..., 3], np.minimum(far, label[..., 3]) if desk else far)
        img = defocus(img, depth, focus_depth, float(d["strength_px"]))
    m = fx["motion_blur"]
    if m["enabled"] and m["length_px"] > 0.5:
        img = motion_blur(img, float(m["length_px"]), float(m["angle_deg"]))
    rgba = None
    if transparent:
        # The rest works on plain color; the alpha goes back on at the end.
        rgba, img = img, unpremultiply(img)
    if fx["vignette"] > 0:
        img *= vignette(img.shape[:2], round(float(fx["vignette"]), 3))[..., None]

    # Image processor: white balance and exposure set the sensor level the
    # noise is measured at, so compute them first and apply them around it.
    gain = white_balance_gain(img, page, float(fx["white_balance"])) * np.asarray(fx["tint"], np.float32)
    img *= gain
    exposure = 2.0 ** float(fx["exposure_ev"])
    ae = fx["auto_exposure"]
    if ae["enabled"]:
        lum = (img[::4, ::4] @ _LUMA)[page[::4, ::4]]
        level = float(np.percentile(lum, float(ae["percentile"]))) if lum.size else 1.0
        target = filmic_inverse(float(ae["target"])) if fx["tonemap"] == "filmic" else float(ae["target"])
        exposure *= target / max(level, 1e-6)
    img *= np.float32(exposure)

    # Sensor noise, in linear light at the exposed level.
    n = fx["noise"]
    if n["enabled"]:
        img = add_noise(img, float(n["shot"]), float(n["read"]), float(n["chroma"]), float(n["grain_px"]), rng)

    if fx["tonemap"] == "filmic":
        img = filmic(img)
    out = linear_to_srgb8(img)

    s = fx["sharpen"]
    if s["enabled"] and s["amount"] > 0:
        pil = Image.fromarray(out).filter(ImageFilter.UnsharpMask(
            radius=float(s["radius_px"]), percent=int(100 * float(s["amount"])), threshold=0))
        out = np.asarray(pil)
    return out if rgba is None else _with_alpha(out, rgba)


def unpremultiply(rgba: np.ndarray) -> np.ndarray:
    """Premultiplied (H, W, 4) -> plain color (H, W, 3).

    Fully transparent pixels take the color of the nearby page, so a later
    sharpen or blur does not pull a dark fringe into the page edge.
    """
    a = rgba[..., 3:]
    out = rgba[..., :3] / np.maximum(a, np.float32(1e-4))
    hole = a[..., 0] < 0.02
    if hole.any():
        fill = gaussian_blur(rgba, 6.0)
        out[hole] = (fill[..., :3] / np.maximum(fill[..., 3:], np.float32(1e-4)))[hole]
    return out


def _with_alpha(rgb8: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    a = (np.clip(rgba[..., 3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.concatenate([rgb8, a[..., None]], axis=2)


def defocus(img: np.ndarray, depth: np.ndarray, focus: float, strength_px: float) -> np.ndarray:
    """Depth-dependent blur. The blur circle grows with |1 - focus / depth|.

    `strength_px` is the blur radius for a point at twice the focus distance.
    The image is blurred at a few fixed levels and each pixel blends the two
    levels around its own blur size.
    """
    coc = strength_px * np.abs(1.0 - focus / np.maximum(depth, 1e-3)) / 0.5
    sigma = np.clip(0.5 * coc, 0.0, 4.0 * strength_px).astype(np.float32)
    top = float(sigma.max())
    if top < 0.3:
        return img
    levels = [0.0]
    while levels[-1] < top:
        levels.append(max(0.5, levels[-1] * 2.0))
    stack = [img] + [gaussian_blur(img, s) for s in levels[1:]]
    out = np.zeros_like(img)
    lv = np.asarray(levels, dtype=np.float32)
    pos = np.interp(sigma, lv, np.arange(len(lv))).astype(np.float32)
    for i, layer in enumerate(stack):
        w = np.clip(1.0 - np.abs(pos - i), 0.0, 1.0)
        if w.any():
            out += layer * w[..., None]
    return out


def vignette(shape: tuple[int, int], strength: float) -> np.ndarray:
    return 1.0 - strength * _vignette_falloff(shape)


@lru_cache(maxsize=8)
def _vignette_falloff(shape: tuple[int, int]) -> np.ndarray:
    """Corner falloff for a frame size; the strength changes per photo, this does not."""
    h, w = shape
    y = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / (0.5 * np.hypot(w, h))
    x = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / (0.5 * np.hypot(w, h))
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    return np.clip(r2, 0.0, 1.0) ** 1.5


def white_balance_gain(img: np.ndarray, page: np.ndarray, strength: float) -> np.ndarray:
    """Gray-world on the page: the paper is white, so pull its mean to neutral."""
    if strength <= 0:
        return np.ones(3, np.float32)
    mean = img[::4, ::4][page[::4, ::4]].mean(axis=0) if page.any() else img.mean(axis=(0, 1))
    mean = np.maximum(mean, 1e-6)
    gain = (mean @ _LUMA) / mean
    return (gain ** strength).astype(np.float32)


def add_noise(img: np.ndarray, shot: float, read: float, chroma: float, grain_px: float,
              rng: np.random.Generator) -> np.ndarray:
    """Poisson-Gaussian sensor noise: variance = shot * signal + read^2.

    Phones denoise color more than brightness, so most noise is shared by
    the three channels; `chroma` sets the per-channel part. `grain_px`
    blurs the noise into the soft clumps left by denoising.
    """
    h, w, _ = img.shape
    noise = rng.standard_normal((h, w, 1), dtype=np.float32)
    if chroma > 0:
        # Color noise at half resolution: cheaper, and phone color noise is blotchy anyway.
        col = rng.standard_normal(((h + 1) // 2, (w + 1) // 2, 3), dtype=np.float32) * np.float32(2.0 * chroma)
        noise = noise + col.repeat(2, axis=0).repeat(2, axis=1)[:h, :w]
        noise /= np.float32(np.sqrt(1.0 + 4.0 * chroma * chroma))
    if grain_px > 0.3:
        noise = gaussian_blur(noise, grain_px)
        noise /= max(float(noise.std()), 1e-6)
    sigma = np.sqrt(np.maximum(shot * img, 0.0) + read * read)
    return np.maximum(img + noise * sigma, 0.0)
