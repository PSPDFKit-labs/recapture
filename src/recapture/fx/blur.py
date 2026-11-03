"""Float blurs in numpy, with Pillow resizing for large radii."""

from __future__ import annotations

import numpy as np
from PIL import Image

_DIRECT_MAX_SIGMA = 2.0


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of an (H, W) or (H, W, C) float32 image, edges repeated.

    Small sigmas use direct separable taps. Larger ones shrink the image,
    blur it, and enlarge it back, which costs about the same at any radius.
    """
    if sigma < 0.3:
        return img
    if sigma <= _DIRECT_MAX_SIGMA:
        return _separable(_separable(img, sigma, 0), sigma, 1)
    f = int(sigma / 1.25)
    h, w = img.shape[:2]
    small_size = (max(1, round(w / f)), max(1, round(h / f)))
    small = _resize(img, small_size, Image.Resampling.BOX)
    # A box shrink by f already blurs by about f / sqrt(12) in full pixels.
    rest = np.sqrt(max(sigma * sigma - f * f / 12.0, 0.25)) / f
    small = gaussian_blur(small, rest)
    return _resize(small, (w, h), Image.Resampling.BICUBIC)


def motion_blur(img: np.ndarray, length: float, angle_deg: float) -> np.ndarray:
    """Average of copies shifted along a line of `length` px."""
    n = max(2, int(np.ceil(length)) + 1)
    a = np.radians(angle_deg)
    out = np.zeros_like(img)
    for t in np.linspace(-0.5 * length, 0.5 * length, n):
        dx, dy = int(round(t * np.cos(a))), int(round(t * np.sin(a)))
        out += _shift(img, dx, dy)
    return out / np.float32(n)


def _separable(img: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    r = int(np.ceil(2.5 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="edge")
    n = img.shape[axis]
    out = np.zeros_like(img)
    window = [slice(None)] * img.ndim
    for i, kv in enumerate(k):
        window[axis] = slice(i, i + n)
        out += kv * p[tuple(window)]  # a view, no copy
    return out


def _resize(img: np.ndarray, size: tuple[int, int], method) -> np.ndarray:
    """Resize each channel as a Pillow float ('F') image."""
    if img.ndim == 2:
        return np.asarray(Image.fromarray(np.ascontiguousarray(img, dtype=np.float32)).resize(size, method), dtype=np.float32)
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32)).resize(size, method),
                        dtype=np.float32) for c in range(img.shape[2])]
    return np.stack(chans, axis=2)


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift with edge repeat (no wrap-around)."""
    h, w = img.shape[:2]
    ys = np.clip(np.arange(h) - dy, 0, h - 1)
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    return img[ys][:, xs]
