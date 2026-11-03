"""Page deformations. Pure functions on page-local positions (cm).

Page-local frame: the flat page is centered on the origin in the z = 0
plane, x points right, y points to the top edge, z points up (the printed
side faces +z). "Material" coordinates name a point on the paper itself and
never change: mx = u * W (right), my = v * H (down from the top edge).

Order: crumple, folds, curl, edge lift, then the page rests on z = 0.
Crumple goes first because it is defined on the flat sheet. Folds and curl
then move the crumpled sheet as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .mesh import PageMesh
from .paper_scan import CrumpleScan, sample_height, scan_info

_SLOPE_PER_STRENGTH = 0.15  # procedural RMS facet slope (tan of tilt) at strength 1


@dataclass(frozen=True)
class Creases:
    """The crease network of a crumpled page, in page-local cm (x right, y up)."""
    center: np.ndarray  # (K, 2)
    direction: np.ndarray  # (K, 2) unit vector along the crease
    half_length: np.ndarray  # (K,)
    width: np.ndarray  # (K,) half-width of the tent profile
    slope: np.ndarray  # (K,) tan of the facet tilt; > 0 ridge, < 0 valley


def deform_page(mesh: PageMesh, cfg: dict[str, Any]) -> PageMesh:
    """Apply the resolved `deform` config section to a flat page mesh.

    Everything is real geometry: the crumple moves mesh vertices, and the
    renderer lights the surface from its own normals.
    """
    w, h = mesh.width_cm, mesh.height_cm
    mat = mesh.uv.astype(np.float64) * [w, h]
    pos = mesh.pos.astype(np.float64)

    crumple = cfg["crumple"]
    strength = float(crumple["strength"])
    if strength <= 0:
        pass  # flat paper
    elif crumple["mode"] == "scan":
        pos = pos.copy()
        pos[:, 2] += sample_height(make_scan(crumple), pos[:, :2])
    elif crumple["creases"] > 0:
        creases = make_creases(w, h, crumple)
        pos = apply_crumple(pos, creases, strength * float(crumple["waviness_cm"]), int(crumple["seed"]), w, h)
    return mesh.with_pos(_shape(mesh, pos, mat, cfg))


def _shape(mesh: PageMesh, pos: np.ndarray, mat: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Folds, curl and edge lift, then rest the page on z = 0."""
    w, h = mesh.width_cm, mesh.height_cm
    for fold in cfg["folds"]:
        pos = apply_fold(mesh.with_pos(pos), fold)
    curl = cfg["curl"]
    if curl["x_curvature"]:
        pos = bend(pos, axis=0, curvature=float(curl["x_curvature"]), center=(curl["x_center"] - 0.5) * w)
    if curl["y_curvature"]:
        pos = bend(pos, axis=1, curvature=float(curl["y_curvature"]), center=(0.5 - curl["y_center"]) * h)
    if any(cfg["edge_lift_cm"]):
        pos = apply_edge_lift(pos, mat, w, h, cfg["edge_lift_cm"], float(cfg["edge_lift_width_cm"]))
    pos = pos.copy()
    pos[:, 2] -= pos[:, 2].min()
    return pos


def make_scan(p: dict[str, Any]) -> CrumpleScan:
    s = p["scan"]
    rng = np.random.default_rng(int(p["seed"]) + 7)
    offset = rng.uniform(0, 1, 2) if s["offset"] is None else np.asarray(s["offset"], dtype=np.float64)
    turns = int(rng.integers(4)) if s["quarter_turns"] is None else int(s["quarter_turns"])
    info = scan_info(str(s["directory"]))  # per-scan tuning, see paper_scan.py
    return CrumpleScan(directory=str(s["directory"]), tile_cm=float(s["tile_cm"]) * info["tile_scale"],
                       depth_cm=min(float(p["strength"]), info["max_strength"]) * float(s["depth_cm"]) * info["depth_scale"],
                       offset=offset, quarter_turns=turns, smooth_px=info["smooth_px"])


# ---- crumple ---------------------------------------------------------------


def make_creases(w: float, h: float, p: dict[str, Any]) -> Creases:
    """Random crease segments, scaled so the RMS facet slope matches `strength`.

    Each crease is a straight segment with a tent profile across it: height
    rises linearly from 0 at distance `width` to a sharp ridge (or valley)
    on the line, and fades out toward the segment ends. Overlapping tents
    give flat facets with sharp edges.
    """
    rng = np.random.default_rng(int(p["seed"]))
    k = int(p["creases"])
    diag = float(np.hypot(w, h))
    w_lo, w_hi = (float(v) for v in p["width_cm"])
    l_lo, l_hi = (float(v) for v in p["length"])
    center = rng.uniform([-0.5 * w, -0.5 * h], [0.5 * w, 0.5 * h], size=(k, 2))  # creases lie on the paper
    angle = rng.uniform(0, np.pi, size=k)
    creases = Creases(
        center=center,
        direction=np.stack([np.cos(angle), np.sin(angle)], 1),
        half_length=0.5 * diag * rng.uniform(l_lo, l_hi, size=k),
        width=rng.uniform(w_lo, w_hi, size=k),
        slope=rng.uniform(0.3, 1.0, size=k) * rng.choice([-1.0, 1.0], size=k),
    )
    # Overlapping creases add up, so the raw slope depends on the count.
    # Measure the RMS slope on a grid and rescale to the target.
    n = 96
    gx, gy = np.meshgrid(np.linspace(-w / 2, w / 2, n), np.linspace(-h / 2, h / 2, int(n * h / w)))
    dz, _ = crease_field(np.stack([gx.ravel(), gy.ravel()], 1), creases, with_shrink=False)
    dz = dz.reshape(gx.shape)
    sx = np.diff(dz, axis=1) / (w / (n - 1))
    sy = np.diff(dz, axis=0) / (h / (gx.shape[0] - 1))
    rms = float(np.sqrt(0.5 * (np.mean(sx * sx) + np.mean(sy * sy))))
    if rms > 1e-9:
        creases = replace(creases, slope=creases.slope * (_SLOPE_PER_STRENGTH * float(p["strength"]) / rms))
    return creases


def crease_field(xy: np.ndarray, c: Creases, with_shrink: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Height (N,) and in-plane shrink (N, 2) of the crease network at points xy.

    A tilted facet takes less room than a flat one, so each crease pulls the
    paper toward its line by (1 - cos(tilt)) per unit of distance. This keeps
    the paper close to its true size and bends text lines where they cross
    a crease, like on real crumpled paper.
    """
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    dz = np.zeros(len(x), np.float32)
    dx = np.zeros(len(x), np.float32)
    dy = np.zeros(len(x), np.float32)
    # Sort points by x once, so each crease only visits the points inside
    # its own bounding box (creases are short compared with the page).
    order = np.argsort(x, kind="stable")
    xs_sorted = x[order]
    for (cx, cy), (tx, ty), half_len, width, slope in zip(c.center, c.direction, c.half_length, c.width, c.slope):
        mx, my = -ty, tx
        reach_x = abs(tx) * half_len + abs(mx) * width
        reach_y = abs(ty) * half_len + abs(my) * width
        lo, hi = np.searchsorted(xs_sorted, [cx - reach_x, cx + reach_x])
        cand = order[lo:hi]
        cand = cand[np.abs(y[cand] - cy) <= reach_y]
        along = np.abs((x[cand] - cx) * tx + (y[cand] - cy) * ty)
        keep = along < half_len  # the taper is 0 outside the strip
        idx = cand[keep]
        if not len(idx):
            continue
        along = along[keep]
        across = (x[idx] - cx) * mx + (y[idx] - cy) * my
        taper = 1.0 - _smoothstep(0.5 * half_len, half_len, along)
        dist = np.minimum(np.abs(across), width)
        dz[idx] += slope * (width - dist) * taper
        if with_shrink:
            shrink = (1.0 - 1.0 / np.sqrt(1.0 + slope * slope)) * dist * taper * np.sign(across)
            dx[idx] -= shrink * mx
            dy[idx] -= shrink * my
    return dz.astype(np.float64), np.stack([dx, dy], 1).astype(np.float64)


def apply_crumple(flat: np.ndarray, creases: Creases, waviness_cm: float, seed: int,
                  w: float, h: float) -> np.ndarray:
    dz, dxy = crease_field(flat[:, :2], creases)
    if waviness_cm > 0:
        rng = np.random.default_rng(seed + 1)
        diag = float(np.hypot(w, h))
        for _ in range(4):
            k = rng.normal(size=2)
            k *= rng.uniform(1.0, 3.0) * 2 * np.pi / diag / np.linalg.norm(k)
            dz += 0.5 * waviness_cm * np.sin(flat[:, :2] @ k + rng.uniform(0, 2 * np.pi))
    out = flat.copy()
    out[:, :2] += dxy
    out[:, 2] += dz
    return out


# ---- folds -----------------------------------------------------------------


def apply_fold(mesh: PageMesh, fold: dict[str, Any]) -> np.ndarray:
    """Rotate the part of the page on one side of a fold line.

    The line passes through `position` (page fractions, x right, y down) with
    direction `direction_deg` (0 = parallel to the top edge). The moving side
    is the one to the right of the direction; add 180 degrees to swap sides.
    A positive `angle_deg` lifts the moving side toward the printed face. The
    crease is rounded over a strip of `crease_radius_cm * angle` width.

    The axis is the current 3D position of the line, so folds stack. For
    folds that cross, this is exact for the first fold and close for small
    angles of the later ones.
    """
    w, h = mesh.width_cm, mesh.height_cm
    theta = np.radians(float(fold["angle_deg"]))
    if abs(theta) < 1e-6:
        return mesh.pos.astype(np.float64)
    a = np.radians(float(fold["direction_deg"]))
    t = np.array([np.cos(a), np.sin(a)])  # material frame: x right, y down
    m = np.array([-t[1], t[0]])
    c = np.asarray(fold["position"], dtype=np.float64) * [w, h]

    ends = _clip_line(c, t, w, h)
    if ends is None:
        return mesh.pos.astype(np.float64)
    to_uv = np.array([w, h])
    p0, p1 = mesh.sample(np.stack(ends) / to_uv)
    origin = mesh.sample((c / to_uv)[None])[0]
    e = p1 - p0
    if np.linalg.norm(e) < 1e-9:
        return mesh.pos.astype(np.float64)
    e /= np.linalg.norm(e)

    # In-surface direction toward the moving side, and the surface normal.
    probe = np.clip(c + 0.5 * m, [0, 0], [w, h])
    perp = mesh.sample((probe / to_uv)[None])[0] - origin
    perp -= (perp @ e) * e
    normal = _sample_normal(mesh, c / to_uv)
    if np.linalg.norm(perp) < 1e-9:
        perp = np.cross(normal, e)
    perp /= np.linalg.norm(perp)
    # The axis runs along the fold line. Its sign makes a positive angle turn
    # `perp` toward `normal`: (perp x normal) x perp = normal.
    side = np.cross(perp, normal) @ e
    if abs(side) < 1e-9:
        return mesh.pos.astype(np.float64)
    axis = e * np.sign(side)

    mat = mesh.uv.astype(np.float64) * [w, h]
    d = (mat - c) @ m
    strip = max(float(fold["crease_radius_cm"]) * abs(theta), 1e-6)
    phi = theta * np.clip(d / strip, 0.0, 1.0)
    pos = mesh.pos.astype(np.float64)
    moving = d > 0
    pos[moving] = origin + _rotate(pos[moving] - origin, axis, phi[moving])
    return pos


def _clip_line(c: np.ndarray, t: np.ndarray, w: float, h: float):
    """End points of the line c + s*t inside the page rectangle, or None."""
    lo, hi = -np.inf, np.inf
    for k, size in ((0, w), (1, h)):
        if abs(t[k]) < 1e-12:
            if not 0 <= c[k] <= size:
                return None
            continue
        s0, s1 = (0 - c[k]) / t[k], (size - c[k]) / t[k]
        lo, hi = max(lo, min(s0, s1)), min(hi, max(s0, s1))
    if hi - lo < 1e-6:
        return None
    inset = 1e-3 * (hi - lo)
    return c + (lo + inset) * t, c + (hi - inset) * t


def _sample_normal(mesh: PageMesh, uv: np.ndarray) -> np.ndarray:
    """Surface normal at one UV point, from nearby surface points."""
    e = 2.0 / max(mesh.nu, mesh.nv)
    pts = mesh.sample(np.array([[uv[0] - e, uv[1]], [uv[0] + e, uv[1]], [uv[0], uv[1] - e], [uv[0], uv[1] + e]]))
    n = np.cross(pts[3] - pts[2], pts[1] - pts[0])  # v x u points up, as in PageMesh.normals
    return n / max(np.linalg.norm(n), 1e-12)


def _rotate(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of each row of v about `axis` by its own angle."""
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * cos + np.cross(axis, v) * sin + np.outer(v @ axis, axis) * (1.0 - cos)


# ---- curl and edge lift ----------------------------------------------------


def bend(pos: np.ndarray, axis: int, curvature: float, center: float) -> np.ndarray:
    """Wrap the page around a cylinder of radius 1 / curvature.

    axis 0 bends along x (the left and right edges move), axis 1 along y.
    Positive curvature curls the edges up. The bend keeps lengths on the
    page surface, so the paper does not stretch.
    """
    r0 = 1.0 / curvature
    out = pos.copy()
    s = pos[:, axis] - center
    phi = s * curvature
    r = r0 - pos[:, 2]
    out[:, axis] = center + r * np.sin(phi)
    out[:, 2] = r0 - r * np.cos(phi)
    return out


def apply_edge_lift(pos: np.ndarray, mat: np.ndarray, w: float, h: float,
                    lift_cm: list[float], band_cm: float) -> np.ndarray:
    """Raise the page edges (top, right, bottom, left) with a smooth curve."""
    dist = [mat[:, 1], w - mat[:, 0], h - mat[:, 1], mat[:, 0]]
    out = pos.copy()
    band = max(band_cm, 1e-6)
    for lift, d in zip(lift_cm, dist):
        if lift:
            f = np.clip(1.0 - d / band, 0.0, 1.0)
            out[:, 2] += float(lift) * f * f
    return out


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
