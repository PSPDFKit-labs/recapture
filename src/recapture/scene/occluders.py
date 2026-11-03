"""Shadow casters: flat shapes above the page that the camera does not see.

Each shape is an alpha mask on a rectangle of `width` x `length` cm. The
mask's y axis runs from the far end (y = 0) toward the photographer
(y = length), so an arm reaches toward the camera. `anchor` is the point
that sits at `position_cm`, or whose shadow lands on the page `target`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PX_PER_CM = 8.0


@dataclass(frozen=True)
class Shape:
    width: float
    length: float
    anchor: tuple[float, float]


SHAPES = {
    "phone": Shape(8.0, 16.0, (4.0, 8.0)),
    "hand_phone": Shape(18.0, 70.0, (9.0, 8.5)),
    "arm": Shape(14.0, 70.0, (7.0, 8.0)),
    "blob": Shape(24.0, 24.0, (12.0, 12.0)),
}


@dataclass(frozen=True)
class Occluder:
    shape: str
    corners: np.ndarray  # (4, 3) world cm: mask (0,0), (W,0), (W,L), (0,L)


def place_occluder(shape_name: str, target: np.ndarray, light: np.ndarray, camera: np.ndarray,
                   height: float, scale: float, rotation_offset_deg: float) -> Occluder:
    """Put the shape on the ray from `target` to the light, at `height`.

    Its arm (mask +y) points toward the camera, as if the photographer's hand
    blocks the light.
    """
    h = min(height, 0.8 * light[2])
    t = (h - target[2]) / max(light[2] - target[2], 1e-6)
    return place_occluder_at(shape_name, target + t * (light - target), camera, scale, rotation_offset_deg)


def place_occluder_at(shape_name: str, anchor: np.ndarray, camera: np.ndarray, scale: float,
                      rotation_offset_deg: float) -> Occluder:
    """Put the shape's anchor at a fixed world point. The arm points toward the camera."""
    shape = SHAPES[shape_name]
    anchor = np.asarray(anchor, dtype=np.float64)
    toward = camera[:2] - anchor[:2]
    if np.linalg.norm(toward) < 1e-6:
        toward = np.array([0.0, -1.0])
    a = np.arctan2(toward[1], toward[0]) + np.radians(rotation_offset_deg)
    ay = np.array([np.cos(a), np.sin(a), 0.0])  # mask +y
    ax = np.array([np.sin(a), -np.cos(a), 0.0])  # mask +x, to the right of +y seen from above
    ox, oy = shape.anchor
    corners = []
    for mx, my in ((0, 0), (shape.width, 0), (shape.width, shape.length), (0, shape.length)):
        corners.append(anchor + scale * ((mx - ox) * ax + (my - oy) * ay))
    return Occluder(shape=shape_name, corners=np.array(corners))


@lru_cache(maxsize=8)
def occluder_mask(shape_name: str) -> np.ndarray:
    """Alpha mask (rows along length, cols along width), float32 in [0, 1]."""
    s = SHAPES[shape_name]
    w, h = int(s.width * PX_PER_CM), int(s.length * PX_PER_CM)
    x, y = np.meshgrid((np.arange(w) + 0.5) / PX_PER_CM, (np.arange(h) + 0.5) / PX_PER_CM)
    if shape_name == "phone":
        d = _rounded_rect(x, y, (4.0, 8.0), (3.6, 7.6), 1.0)
    elif shape_name == "hand_phone":
        phone = _rounded_rect(x, y, (9.0, 8.5), (3.6, 7.6), 1.0)
        palm = _ellipse(x, y, (9.0, 17.0), (5.5, 6.5))
        thumb = _capsule(x, y, (4.5, 15.0), (3.5, 9.0), 1.3)
        fingers = _capsule(x, y, (13.5, 16.0), (13.2, 9.5), 1.5)
        arm = _capsule(x, y, (9.0, 20.0), (9.5, 75.0), 4.2)
        d = np.minimum.reduce([phone, palm, thumb, fingers, arm])
    elif shape_name == "arm":
        hand = _ellipse(x, y, (7.0, 9.0), (4.5, 6.0))
        arm = _capsule(x, y, (7.0, 12.0), (7.5, 75.0), 4.2)
        d = np.minimum(hand, arm)
    else:  # blob: an ellipse with a wavy edge
        ang = np.arctan2(y - 12.0, x - 12.0)
        wave = 1.0 + 0.08 * np.sin(3 * ang + 0.7) + 0.05 * np.sin(5 * ang + 2.1)
        d = np.hypot((x - 12.0) / 9.0, (y - 12.0) / 6.5) * 6.5 - 6.5 * wave
    # Signed distance in cm -> antialiased alpha.
    return np.clip(0.5 - d * PX_PER_CM, 0.0, 1.0).astype(np.float32)


def _rounded_rect(x, y, center, half, r):
    qx = np.abs(x - center[0]) - (half[0] - r)
    qy = np.abs(y - center[1]) - (half[1] - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def _ellipse(x, y, center, radii):
    k = np.hypot((x - center[0]) / radii[0], (y - center[1]) / radii[1])
    return (k - 1.0) * min(radii)


def _capsule(x, y, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa = np.stack([x - a[0], y - a[1]], -1)
    ba = b - a
    t = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - t[..., None] * ba, axis=-1) - r
