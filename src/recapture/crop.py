"""Scanner-app crop: cut the page out of the photo and make it a rectangle.

Like a phone scanner app, this finds the four page corners in the photo and
maps them to the corners of the output with one perspective transform
(a homography). The page in the result is flat only as far as four corners
can make it: a curl or a crumple stays, and where the page edge bows in, a
bit of desk shows.

A homography maps straight lines to straight lines, so a label quad maps to
a quad that still holds every pixel of its box.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .labels.quad import convex_hull, enclosing_quad, order_quad
from .labels.transform import UV_CORNERS, project_uv
from .scene import Scene


def crop_size(scene: Scene, crop: dict[str, Any]) -> tuple[int, int]:
    """Output size: the page's shape, with the long side of the camera image."""
    if crop["size_px"] is not None:
        return int(crop["size_px"][0]), int(crop["size_px"][1])
    long = max(scene.camera.width, scene.camera.height)
    w, h = scene.page.width_cm, scene.page.height_cm
    return (long, max(1, round(long * h / w))) if w >= h else (max(1, round(long * w / h)), long)


def crop_transform(scene: Scene, crop: dict[str, Any], seed: int) -> tuple[np.ndarray, tuple[int, int]] | None:
    """The homography from photo pixels to the cropped image, and the crop size.
    None when a page corner is behind the camera."""
    corners, w = project_uv(scene, UV_CORNERS)  # top-left, top-right, bottom-right, bottom-left
    if np.any(w <= scene.camera.near):
        return None
    # The app finds the corners only to a few pixels: move each one a little.
    side = min(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[0]))
    error = float(crop["corner_error"]) * side
    if error > 0:
        corners = corners + np.random.default_rng(seed).uniform(-error, error, (4, 2))
    size = crop_size(scene, crop)
    m = float(crop["margin"]) / (1.0 + 2.0 * float(crop["margin"]))  # page fraction -> output fraction
    sw, sh = size
    target = np.array([[m * sw, m * sh], [(1 - m) * sw, m * sh], [(1 - m) * sw, (1 - m) * sh], [m * sw, (1 - m) * sh]])
    return homography(corners, target), size


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 3x3 matrix that maps the four `src` points onto the four `dst` points."""
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(a, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.concatenate([p, np.ones((len(p), 1))], 1) @ h.T
    return q[:, :2] / q[:, 2:3]


def warp_image(image: np.ndarray, h: np.ndarray, size: tuple[int, int], fill=None) -> np.ndarray:
    """Resample the photo into the crop. Pixels outside the photo are `fill`,
    or black (transparent with the background off)."""
    # PIL maps each output pixel center (x + 0.5) through the matrix and reads the
    # input in the same continuous coordinates as the labels, so the plain inverse
    # is exact (checked at 0.25x to 4x).
    m = np.linalg.inv(h)
    m /= m[2, 2]
    im = Image.fromarray(image)
    out = im.transform(size, Image.Transform.PERSPECTIVE, tuple(m.ravel()[:8]), Image.Resampling.BICUBIC,
                       fillcolor=fill)
    return np.asarray(out)


def warp_labels(boxes: list[dict[str, Any]], page_corners: list | None, h: np.ndarray,
                size: tuple[int, int]) -> tuple[list[dict[str, Any]], list | None]:
    """Map the label quads and corners into the crop. A quad that leaves the
    crop is cut at its edge, refitted and marked truncated."""
    out = []
    for box in boxes:
        box = dict(box)
        corners = None if box["corners"] is None else apply_h(h, box["corners"])
        if box["quad"] is not None:
            quad = apply_h(h, box["quad"])
            clipped = _clip_to_rect(quad, size)
            full_area = abs(_area(quad))
            if len(clipped) < 3 or abs(_area(clipped)) < 1e-6:
                box.update(quad=None, visible=0.0, truncated=True)
            else:
                if len(clipped) != 4 or not np.allclose(clipped, quad):
                    box["visible"] = round(float(box["visible"]) * abs(_area(clipped)) / max(full_area, 1e-9), 4)
                    box["truncated"] = True
                    first = corners[0] if corners is not None else clipped[0]
                    quad = order_quad(enclosing_quad(convex_hull(clipped)), first)
                box["quad"] = _round(quad)
        box["corners"] = None if corners is None else _round(corners)
        out.append(box)
    return out, None if page_corners is None else _round(apply_h(h, page_corners))


def _clip_to_rect(poly: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon to [0, w] x [0, h]."""
    w, h = size
    pts = [p for p in poly]
    for axis, limit, keep_below in ((0, 0.0, False), (0, float(w), True), (1, 0.0, False), (1, float(h), True)):
        if not pts:
            break
        res = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            cin = cur[axis] <= limit if keep_below else cur[axis] >= limit
            pin = prev[axis] <= limit if keep_below else prev[axis] >= limit
            if cin != pin:
                t = (limit - prev[axis]) / (cur[axis] - prev[axis])
                res.append(prev + t * (cur - prev))
            if cin:
                res.append(cur)
        pts = res
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _round(p: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in np.asarray(p).reshape(-1, 2)]
