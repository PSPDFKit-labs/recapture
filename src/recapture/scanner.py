"""Scan mode: a page from a printer, then on a flatbed scanner.

There is no 3D scene. First the printer marks the page (`printer`, at the
document's own resolution, every effect off at 0):

    ink_bleed       ink spreads a little into the paper around each stroke
    broken_strokes  a printer low on toner: small gaps inside the ink
    ghost           a faint shifted copy of the ink, like a double strike
    toner_streaks   thin broken lines along the paper feed (a dirty drum)
    toner_specks    black dots of loose toner

Then the scanner (`scanner`), in order:

    skew      the page lies a little crooked on the glass
    optics    the lens and sensor soften the image slightly
    raster    the page is sampled at the scan resolution (dpi)
    bold      strokes thicken and small holes fill in (an over-dark scan)
    edges     dark bands where the scanner sees past the paper, with a
              ragged shadow at the paper edge and clouds of dots beside it
    noise     sensor noise
    streaks   thin lines along the scan direction (dust on the sensor bar)
    dust      small dark specks (dust or hairs on the glass)
    color     color, grayscale, or 1-bit black and white

The skew and the resolution change are one 3x3 matrix, so the labels follow
exactly. All random draws come from `scanner.seed`. The output's own JPEG
compression comes from `output.format` and `output.jpeg_quality`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .crop import warp_image, warp_labels
from .fx.blur import gaussian_blur
from .labels import SourceBox
from .scene.mesh import page_dimensions

COLOR_MODES = ("color", "gray", "bw")
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_LID = (255, 255, 255)  # the scanner lid, seen where the skewed page leaves the glass area


def scan_document(doc_rgba: np.ndarray, boxes: list[SourceBox], cfg: dict[str, Any],
                  page_size_cm: list[float], printer: dict[str, Any] | None = None) -> tuple[
        np.ndarray, tuple[int, int], list, list[dict[str, Any]]]:
    """Clean document (RGBA uint8) -> scanned page (RGB uint8), its size, the
    page corners and the boxes in its pixels. `cfg` is the resolved `scanner`
    section and `printer` the resolved `printer` section (None = a clean print);
    `page_size_cm` (page.size_cm) gives the paper size for the dpi."""
    seed = int(cfg["seed"])

    def rng(k: int) -> np.random.Generator:
        return np.random.default_rng([seed, k])

    img = _on_white(doc_rgba)
    if printer is not None:
        img = print_marks(img, printer, rng)
    h, w = img.shape[:2]
    w_cm, _ = page_dimensions(page_size_cm, (w, h))
    scale = float(cfg["dpi"]) / 2.54 * w_cm / w  # output pixels per source pixel
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    m = _skew_scale_matrix(w, h, size, float(cfg["skew_deg"]))

    # Optics plus the anti-alias filter the resampling needs, in source pixels.
    sigma = np.hypot(float(cfg["blur_px"]), 0.5 * max(1.0 / scale - 1.0, 0.0)) / scale
    img = gaussian_blur(img, sigma)
    out = warp_image(np.clip(img, 0, 255).astype(np.uint8), m, size, fill=_LID).astype(np.float32)

    sh, sw = out.shape[:2]
    if cfg["bold"] > 0:
        out = _bold(out, float(cfg["bold"]))
    if any(float(v) > 0 for v in cfg["dark_edges"]):
        out = _dark_edges(out, [float(v) for v in cfg["dark_edges"]], int(cfg["edge_dots"]), rng(4))
    if cfg["noise"] > 0:
        out += rng(1).normal(0.0, float(cfg["noise"]), (sh, sw, 1)).astype(np.float32)
    if cfg["streaks"] > 0:
        out = _streaks(out, int(cfg["streaks"]), rng(2))
    if cfg["dust"] > 0:
        out = _dust(out, int(cfg["dust"]), rng(3))
    out = _color(out, cfg["color"], float(cfg["bw_threshold"]))

    corners = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
    start = [{"id": b.id, "label": b.label, "quad": b.quad.tolist(), "corners": b.quad.tolist(), "visible": 1.0,
              "truncated": False, **b.extra} for b in boxes]
    labels, page = warp_labels(start, corners, m, size)
    return np.clip(out, 0, 255).astype(np.uint8), size, page, labels


# ---- printer (float32 RGB, 0..255, document pixels) -----------------------------------


def print_marks(img: np.ndarray, p: dict[str, Any], rng) -> np.ndarray:
    """The printer's marks on the page. `rng(k)` gives a seeded generator per step."""
    if p["ink_bleed"] > 0:
        a = float(np.clip(p["ink_bleed"], 0, 1))
        img = np.minimum(img, img * (1 - a) + gaussian_blur(img, 1.0) * a)
    if p["broken_strokes"] > 0:
        mask = (_gray(img) < 160) & (rng(10).random(img.shape[:2]) < float(p["broken_strokes"]))
        img = img.copy()
        img[mask] = 250.0
    if p["ghost"] > 0:
        r = rng(11)
        ink = np.clip((150.0 - _gray(img)) / 150.0, 0, 1)
        shifted = np.roll(ink, (int(r.integers(-2, 3)), int(r.integers(2, 5))), axis=(0, 1))
        img = img * (1.0 - shifted[..., None] * float(np.clip(p["ghost"], 0, 1)))
    if p["toner_streaks"] > 0 or p["toner_specks"] > 0:
        img = _toner(img, int(p["toner_streaks"]), int(p["toner_specks"]), rng(12))
    return img


def _toner(img: np.ndarray, streaks: int, specks: int, rng: np.random.Generator) -> np.ndarray:
    """A dirty drum: thin streaks along the feed that come and go, and black toner dots."""
    h, w = img.shape[:2]
    mark = np.zeros((h, w), np.float32)
    for _ in range(streaks):
        x, width = int(rng.integers(0, w)), int(rng.integers(1, 5))
        # Broken along its length: a smooth random profile, cut where it is low.
        prof = gaussian_blur(rng.random(h // 8 + 2).astype(np.float32)[:, None], 2.0)[:, 0]
        prof = np.interp(np.arange(h) / 8.0, np.arange(len(prof)), prof)
        mark[:, x:x + width] = np.maximum(mark[:, x:x + width], np.clip((prof - 0.55) * 4, 0, 1)[:, None] * 0.35)
    if specks > 0:
        ys, xs = rng.integers(0, h, specks), rng.integers(0, w, specks)
        mark[ys, xs] = np.maximum(mark[ys, xs], rng.uniform(0.5, 1.0, specks).astype(np.float32))
    mark = np.maximum(mark, gaussian_blur(mark, 0.7) * 1.5)  # dots a pixel or two wide
    return img * (1.0 - np.clip(mark, 0, 0.95)[..., None])


def _gray(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255) @ _LUMA


# ---- scanner -----------------------------------------------------------------------------


def _on_white(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    if rgba.shape[2] == 4:
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb * a + 255.0 * (1.0 - a)
    return rgb


def _skew_scale_matrix(w: int, h: int, size: tuple[int, int], skew_deg: float) -> np.ndarray:
    """Source pixels -> scan pixels: turn the page about its center by `skew_deg`
    and scale it to the scan size."""
    a = np.radians(skew_deg)
    c, s = np.cos(a), np.sin(a)
    sx, sy = size[0] / w, size[1] / h
    to_center = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1]])
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    back = np.array([[sx, 0, size[0] / 2.0], [0, sy, size[1] / 2.0], [0, 0, 1]])
    return back @ turn @ to_center


def _bold(img: np.ndarray, strength: float) -> np.ndarray:
    """Thicker strokes, as in a scan or copy made too dark. `strength` 0..1;
    1 thickens each stroke by about one scan pixel and starts to close the
    smallest holes."""
    ink = 1.0 - np.clip(img, 0, 255) / 255.0
    spread = gaussian_blur(ink, 0.8) * (1.0 + float(np.clip(strength, 0.0, 1.0)))
    return 255.0 * (1.0 - np.clip(np.maximum(ink, spread), 0.0, 1.0))


def _dark_edges(img: np.ndarray, widths: list[float], dots: int, rng: np.random.Generator) -> np.ndarray:
    """Dark bands along the sides where the scanner sees past the paper (an open
    or dark lid). `widths` (left, right, top, bottom) are parts of the page width.
    The paper edge is ragged, it casts a soft shadow inward, and loose dots
    scatter beside the band and thin out toward the page."""
    h, w = img.shape[:2]
    dark = np.zeros((h, w), np.float32)
    dots_mask = np.zeros((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for side, width in zip(("left", "right", "top", "bottom"), widths):
        if width <= 0:
            continue
        across = xs if side == "left" else (w - 1 - xs) if side == "right" else ys if side == "top" else (h - 1 - ys)
        along_n = h if side in ("left", "right") else w
        d = width * w
        # The paper edge wanders: a slow wave plus a little jitter along its length.
        wave = gaussian_blur(rng.normal(0, 1, along_n // 16 + 4).astype(np.float32)[:, None], 2.0)[:, 0]
        wave = np.interp(np.arange(along_n) / 16.0, np.arange(len(wave)), wave) * 0.25 * d
        edge = d + wave + rng.normal(0, 0.6, along_n).astype(np.float32)
        edge = edge[:, None] if side in ("left", "right") else edge[None, :]
        dist = across - edge  # < 0 inside the band, > 0 on the paper
        band = np.clip(0.5 - dist, 0.0, 1.0)  # one pixel of anti-aliasing at the edge
        shade = np.exp(-np.maximum(dist, 0.0) / max(0.15 * d, 1.5)) * 0.55  # the shadow on the paper
        dark = np.maximum(dark, np.maximum(band * 0.9, shade))
        if dots > 0:
            # Dots fall off with distance from the band: most within one band width of it.
            n = int(dots * width / max(sum(widths), 1e-6))
            t = rng.exponential(0.6 * d, n)
            a = rng.integers(0, along_n, n)
            if side == "left":
                py, px = a, (d + t)
            elif side == "right":
                py, px = a, (w - 1 - d - t)
            elif side == "top":
                py, px = (d + t), a
            else:
                py, px = (h - 1 - d - t), a
            py = np.clip(py, 0, h - 1).astype(int)
            px = np.clip(px, 0, w - 1).astype(int)
            dots_mask[py, px] = np.maximum(dots_mask[py, px], rng.uniform(0.6, 1.0, n).astype(np.float32))
    if dots > 0:
        dots_mask = np.maximum(dots_mask, gaussian_blur(dots_mask, 0.5) * 1.8)
        dark = np.maximum(dark, np.clip(dots_mask, 0, 1))
    # The shadow is grainy: a copier breaks a soft gray into specks.
    grain = rng.normal(0, 0.12, (h, w)).astype(np.float32) * (dark > 0.05) * (dark < 0.85)
    dark = np.clip(dark + grain, 0.0, 0.95)
    return img * (1.0 - dark[..., None]) + 18.0 * dark[..., None]


def _streaks(img: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Thin lines down the whole page, along the scan direction: dust on the sensor
    bar. Most are faintly dark, some faintly light."""
    h, w = img.shape[:2]
    for _ in range(count):
        x = int(rng.integers(0, w))
        width = int(rng.integers(1, 3))
        strength = float(rng.uniform(0.04, 0.15)) * (1 if rng.random() < 0.8 else -0.6)
        img[:, x:x + width] *= 1.0 - strength
    return img


def _dust(img: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Small dark specks and short hairs on the glass."""
    h, w = img.shape[:2]
    mark = np.zeros((h, w), np.float32)
    for _ in range(count):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.8:  # a speck
            mark[y, x] = float(rng.uniform(0.5, 1.0))
        else:  # a short curved hair
            n = int(rng.integers(8, 40))
            ang = float(rng.uniform(0, np.pi)) + np.cumsum(rng.normal(0, 0.15, n))
            xs = np.clip(x + np.cumsum(np.cos(ang)), 0, w - 1).astype(int)
            ys = np.clip(y + np.cumsum(np.sin(ang)), 0, h - 1).astype(int)
            mark[ys, xs] = np.maximum(mark[ys, xs], float(rng.uniform(0.3, 0.7)))
    mark = np.maximum(mark, gaussian_blur(mark, 0.6) * 2.0)
    return img * (1.0 - np.clip(mark, 0.0, 0.9)[..., None])


def _color(img: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    """The scan's color mode. "bw" is a fixed threshold, as scanners do it."""
    if mode == "color":
        return img
    gray = (np.clip(img, 0, 255) @ _LUMA)[..., None]
    if mode == "gray":
        return np.repeat(gray, 3, axis=2)
    return np.repeat(np.where(gray < threshold, 0.0, 255.0), 3, axis=2).astype(np.float32)

