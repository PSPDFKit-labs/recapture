"""Move source boxes into the rendered photo.

The label pass gives, for each photo pixel, the source (u, v) that the
camera sees there, or nothing. A box's enclosing quad is fitted around all
photo pixels whose (u, v) falls inside the box. Parts hidden by a fold or
cut off by the frame have no pixels, so they drop out by themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..scene.scene import Scene
from .boxes import SourceBox
from .quad import convex_hull, enclosing_quad, order_quad, points_in_convex_polygon

_GRID = 64  # buckets per side for the source-space pixel index
_AREA_STEPS = 16  # sub-quads per side when computing the expected box area
_PAD = 1.0  # px around a box pixel center on a cut edge, see _cut_edge_points
_OUTLINE_STEPS = 32  # min exact outline samples per box edge, see _visible_outline
_OUTLINE_MAX_STEPS = 4096
_SMOOTH_RATIO = 2.0  # max UV step growth between neighbor pixels on a smooth surface

UV_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


class PixelIndex:
    """Visible photo pixels, bucketed by their source position for fast lookup."""

    def __init__(self, uvbuf: np.ndarray, src_w: int, src_h: int):
        self.src_w, self.src_h = src_w, src_h
        h, w = uvbuf.shape[:2]
        flat = uvbuf.reshape(-1, 4)
        idx = np.flatnonzero(flat[:, 2] > 0.5)
        sx = flat[idx, 0].astype(np.float64) * src_w
        sy = flat[idx, 1].astype(np.float64) * src_h
        # 16-bit keys let numpy use a linear-time radix sort.
        key = (self._bucket(sy, src_h) * _GRID + self._bucket(sx, src_w)).astype(np.uint16)
        order = np.argsort(key, kind="stable")
        idx = idx[order]
        self.ys, self.xs = np.divmod(idx, w)
        self.sx, self.sy = sx[order], sy[order]
        self.starts = np.searchsorted(key[order], np.arange(_GRID * _GRID + 1))

    @staticmethod
    def _bucket(v: np.ndarray, size: int) -> np.ndarray:
        return np.clip((v / size * _GRID).astype(np.int64), 0, _GRID - 1)

    def candidates(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        bx0, bx1 = self._bucket(np.array([x0, x1]), self.src_w)
        by0, by1 = self._bucket(np.array([y0, y1]), self.src_h)
        parts = [np.arange(self.starts[by * _GRID + bx0], self.starts[by * _GRID + bx1 + 1])
                 for by in range(by0, by1 + 1)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def pixels_in(self, box_quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        (x0, y0), (x1, y1) = box_quad.min(0), box_quad.max(0)
        idx = self.candidates(x0, y0, x1, y1)
        inside = points_in_convex_polygon(np.stack([self.sx[idx], self.sy[idx]], 1), box_quad)
        idx = idx[inside]
        return self.xs[idx], self.ys[idx]


def project_uv(scene: Scene, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return scene.camera.project(scene.page.sample(uv))


def transform_boxes(boxes: list[SourceBox], uvbuf: np.ndarray, scene: Scene,
                    src_size: tuple[int, int]) -> list[dict[str, Any]]:
    src_w, src_h = src_size
    cam = scene.camera
    index = PixelIndex(uvbuf, src_w, src_h)
    scale = np.array([src_w, src_h], dtype=np.float64)
    out = []
    for box in boxes:
        uv_quad = box.quad / scale
        corners_px, corners_w = project_uv(scene, uv_quad)
        corners = None if np.any(corners_w <= cam.near) else _round(corners_px)

        xs, ys = index.pixels_in(box.quad)
        quad = None
        if len(xs):
            outline_px = _visible_outline(scene, uv_quad, uvbuf)
            cut_px = _cut_edge_points(xs, ys, uvbuf)
            pts = np.concatenate([outline_px, cut_px])
            if len(pts) == 0:  # cannot happen when xs is non-empty; keep the pixels as a fallback
                pts = _pixel_squares(xs, ys, cam.width, cam.height)
            quad = _round(order_quad(enclosing_quad(convex_hull(pts)), corners_px[0]))

        expected, truncated = _expected_area(scene, uv_quad)
        visible = min(1.0, len(xs) / expected) if expected > 0 else 0.0
        out.append({
            "id": box.id,
            "label": box.label,
            "quad": quad,
            "corners": corners,
            "visible": round(visible, 4),
            "truncated": truncated,
            **box.extra,
        })
    return out


def page_corners(scene: Scene) -> list[list[float]] | None:
    px, w = project_uv(scene, UV_CORNERS)
    return None if np.any(w <= scene.camera.near) else _round(px)


# How the enclosing quad is built
# -------------------------------
# The visible part of a box is bounded by two kinds of edges:
#   1. the box's own edges, where the camera sees them;
#   2. cut edges, where something else ends the visible part: a fold that
#      covers the box, the page edge, or the frame edge.
# Kind 1 is known exactly from the mesh and the camera (_visible_outline).
# Kind 2 is only known to pixel accuracy (_cut_edge_points), so those pixels
# get a safety pad. On a flat page with the box in frame there are no cut
# edges, and the quad equals the exact projected box.


def _visible_outline(scene: Scene, uv_quad: np.ndarray, uvbuf: np.ndarray) -> np.ndarray:
    """Exact projected points on the box outline that the camera can see.

    The outline is sampled about twice per photo pixel, so a crumple crease
    cannot bend the edge between two samples by a visible amount.

    A point counts as visible when a nearby page pixel shows (almost) the
    same source (u, v). If a fold covers the point, the nearby pixels show
    other parts of the page. A point judged visible by mistake only makes
    the quad larger, never smaller.
    """
    cam = scene.camera
    coarse_px, _ = project_uv(scene, _outline_uv(uv_quad, [_OUTLINE_STEPS] * 4))
    per_edge = coarse_px.reshape(4, _OUTLINE_STEPS, 2)
    nxt = np.concatenate([per_edge[:, 1:], np.roll(per_edge[:, :1], -1, axis=0)], axis=1)
    lengths = np.linalg.norm(nxt - per_edge, axis=2).sum(axis=1)
    steps = np.clip(np.ceil(2.0 * np.nan_to_num(lengths, posinf=0.0)), _OUTLINE_STEPS, _OUTLINE_MAX_STEPS)
    uv = _outline_uv(uv_quad, steps.astype(int))
    px, w = project_uv(scene, uv)
    ok = w > cam.near

    # Local tolerance from the projection Jacobian d(px)/d(uv). The smallest
    # singular value gives the direction where one pixel spans the most UV.
    eps = 1e-4
    px_u, _ = project_uv(scene, uv + [eps, 0.0])
    px_v, _ = project_uv(scene, uv + [0.0, eps])
    a, c = (px_u - px) / eps, (px_v - px) / eps  # Jacobian columns
    # Closed form for the smaller singular value of a 2x2 matrix.
    fro = (a * a).sum(1) + (c * c).sum(1)
    det = a[:, 0] * c[:, 1] - a[:, 1] * c[:, 0]
    sigma_min = np.sqrt(np.maximum(0.5 * (fro - np.sqrt(np.maximum(fro * fro - 4 * det * det, 0.0))), 0.0))
    tol = 2.5 / np.maximum(sigma_min, 1e-9)

    h, w_img = uvbuf.shape[:2]
    ix, iy = np.floor(px[:, 0]).astype(np.int64), np.floor(px[:, 1]).astype(np.int64)
    visible = np.zeros(len(uv), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = ix + dx, iy + dy
            inside = ok & (x >= 0) & (y >= 0) & (x < w_img) & (y < h)
            lab = uvbuf[np.clip(y, 0, h - 1), np.clip(x, 0, w_img - 1)]
            near = np.linalg.norm(lab[:, :2] - uv, axis=1) <= tol
            visible |= inside & (lab[:, 2] > 0.5) & near
    return np.clip(px[visible], 0.0, [cam.width, cam.height])


def _outline_uv(uv_quad: np.ndarray, steps: list[int]) -> np.ndarray:
    parts = []
    for k, n in enumerate(steps):
        t = np.linspace(0.0, 1.0, int(n), endpoint=False)[:, None]
        parts.append(uv_quad[k] + t * (uv_quad[(k + 1) % 4] - uv_quad[k]))
    return np.concatenate(parts)


def _cut_edge_points(xs: np.ndarray, ys: np.ndarray, uvbuf: np.ndarray) -> np.ndarray:
    """Padded points around box pixels that sit on a cut edge.

    Take a box pixel p whose neighbor n (one of 4) is not a box pixel. If the
    surface continues smoothly from p into n, the edge between them is the
    box's own edge, which _visible_outline covers exactly. Smooth means the
    UV step from p to n is close to the UV step from the pixel behind p to p.
    Anything else is a cut: n is outside the frame, shows the ground or the
    back of the page, or jumps to another part of the page (a fold). The
    true cut lies between the two pixel centers, so a +-1 px square around
    p's center contains it.
    """
    h, w = uvbuf.shape[:2]
    x0, y0 = int(xs.min()) - 1, int(ys.min()) - 1
    mask = np.zeros((int(ys.max()) - y0 + 2, int(xs.max()) - x0 + 2), dtype=bool)
    mask[ys - y0, xs - x0] = True
    on_cut = np.zeros_like(mask)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor_in = np.zeros_like(mask)
        src = mask[max(dy, 0):mask.shape[0] + min(dy, 0), max(dx, 0):mask.shape[1] + min(dx, 0)]
        neighbor_in[max(-dy, 0):mask.shape[0] + min(-dy, 0), max(-dx, 0):mask.shape[1] + min(-dx, 0)] = src
        cy, cx = np.nonzero(mask & ~neighbor_in)
        if not len(cy):
            continue
        gx, gy = cx + x0, cy + y0
        lab_p = uvbuf[gy, gx]
        lab_n, n_ok = _lookup(uvbuf, gx + dx, gy + dy)
        lab_b, b_ok = _lookup(uvbuf, gx - dx, gy - dy)
        step_in = np.linalg.norm(lab_p[:, :2] - lab_b[:, :2], axis=1)
        step_out = np.linalg.norm(lab_n[:, :2] - lab_p[:, :2], axis=1)
        smooth = n_ok & b_ok & (step_out <= _SMOOTH_RATIO * step_in + 1e-9)
        on_cut[cy[~smooth], cx[~smooth]] = True
    cy, cx = np.nonzero(on_cut)
    return _pixel_squares(cx + x0, cy + y0, w, h)


def _lookup(uvbuf: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label values at (x, y), and whether each is an in-frame page pixel."""
    h, w = uvbuf.shape[:2]
    in_frame = (x >= 0) & (y >= 0) & (x < w) & (y < h)
    lab = uvbuf[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    return lab, in_frame & (lab[:, 2] > 0.5)


def _pixel_squares(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    """Corners of a +-_PAD square around each pixel center, clipped to the frame."""
    cx, cy = xs + 0.5, ys + 0.5
    pts = np.concatenate([
        np.stack([cx - _PAD, cy - _PAD], 1), np.stack([cx + _PAD, cy - _PAD], 1),
        np.stack([cx + _PAD, cy + _PAD], 1), np.stack([cx - _PAD, cy + _PAD], 1),
    ])
    return np.clip(pts, 0.0, [width, height])


def _expected_area(scene: Scene, uv_quad: np.ndarray) -> tuple[float, bool]:
    """Projected area of the box on the mesh, with no occlusion and no frame clip.

    Also returns True when part of the box falls outside the frame.
    """
    n = _AREA_STEPS
    a, b = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    a, b = a.ravel()[:, None], b.ravel()[:, None]
    tl, tr, br, bl = uv_quad
    uv = (1 - b) * ((1 - a) * tl + a * tr) + b * ((1 - a) * bl + a * br)
    px, w = project_uv(scene, uv)
    cam = scene.camera
    behind = w <= cam.near
    off = (px[:, 0] < 0) | (px[:, 1] < 0) | (px[:, 0] > cam.width) | (px[:, 1] > cam.height)
    truncated = bool(np.any(behind | off))
    g = px.reshape(n + 1, n + 1, 2)
    p00, p10, p11, p01 = g[:-1, :-1], g[:-1, 1:], g[1:, 1:], g[1:, :-1]
    cells = np.stack([p00, p10, p11, p01], 2)
    x, y = cells[..., 0], cells[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, 2) - np.roll(x, -1, 2) * y, axis=2))
    ok = ~behind.reshape(n + 1, n + 1)
    ok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    return float(area[ok].sum()), truncated


def _round(pts: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]
