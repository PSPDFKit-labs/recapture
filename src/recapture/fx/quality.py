"""A cheap phone camera: the faults that make a photo less clear.

`fx.quality` runs from 1 (a good camera, no change) to 0 (a cheap one).
Below 1 it adds, in the order a phone makes them:

- a small sensor: the image is captured at a lower resolution and scaled
  back up, so fine text goes soft (at 0: 35 % of the output resolution);
- color fringes: red and blue are scaled a little apart from the center,
  as a cheap lens does (at 0: 2 px at the image corners);
- oversharpening: the phone's image processing puts bright and dark halos
  along edges to hide the softness;
- JPEG compression: blocks and ringing near text, with half-resolution color.

Every step is fixed by `quality` (no random draws), so the same config
always gives the same photo.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

MIN_SENSOR = 0.35  # sensor resolution at quality 0, as a part of the output
MAX_FRINGE_PX = 2.0  # color fringe at the image corners at quality 0
MAX_HALO = 90  # unsharp mask strength (percent) at quality 0
JPEG_BEST, JPEG_WORST = 95, 45


def cheap_camera(image: np.ndarray, quality: float, pixel_scale: float = 1.0) -> np.ndarray:
    """`image` is the final sRGB uint8 photo, RGB or RGBA. `pixel_scale` scales
    the effects measured in pixels (the editor's smaller preview)."""
    loss = 1.0 - float(np.clip(quality, 0.0, 1.0))
    if loss <= 0.0:
        return image
    rgba = image.shape[2] == 4
    im = Image.fromarray(image)
    alpha = im.getchannel("A") if rgba else None
    rgb = im.convert("RGB")
    w, h = rgb.size

    fringe = MAX_FRINGE_PX * loss * pixel_scale
    if fringe > 0.05:
        rgb = _fringe(rgb, fringe)

    sensor = 1.0 - (1.0 - MIN_SENSOR) * loss
    sw, sh = max(8, round(w * sensor)), max(8, round(h * sensor))
    if (sw, sh) != (w, h):
        small = rgb.resize((sw, sh), Image.Resampling.BOX)
        halo = round(MAX_HALO * loss)
        if halo > 0:  # the phone sharpens at its own resolution
            small = small.filter(ImageFilter.UnsharpMask(radius=1.2, percent=halo, threshold=2))
        rgb = small.resize((w, h), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=round(JPEG_BEST - (JPEG_BEST - JPEG_WORST) * loss),
             subsampling=2 if loss > 0.2 else 0)
    rgb = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    if alpha is not None:
        rgb.putalpha(alpha)
    return np.asarray(rgb)


def _fringe(rgb: Image.Image, px: float) -> Image.Image:
    """Lateral chromatic aberration: red a little larger, blue a little smaller."""
    w, h = rgb.size
    cx, cy = w / 2.0, h / 2.0
    k = px / np.hypot(cx, cy)
    r, g, b = rgb.split()

    def scale(ch: Image.Image, s: float) -> Image.Image:
        # Output pixel (x, y) reads input (cx + (x - cx) / s, cy + (y - cy) / s).
        a = 1.0 / s
        return ch.transform((w, h), Image.Transform.AFFINE, (a, 0, cx - cx * a, 0, a, cy - cy * a),
                            Image.Resampling.BILINEAR)

    return Image.merge("RGB", (scale(r, 1.0 + k), g, scale(b, 1.0 - k)))
