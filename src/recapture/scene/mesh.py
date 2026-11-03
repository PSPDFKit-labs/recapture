"""Page mesh: a regular grid in UV space.

UV never changes after the grid is built. u runs left to right and v runs
top to bottom of the source image, so (u * src_w, v * src_h) is a source
pixel position. Deformations move only `pos`, which keeps the link between
a photo pixel and a source pixel exact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PageMesh:
    nu: int  # vertices along u
    nv: int  # vertices along v
    uv: np.ndarray  # (N, 2) float32
    pos: np.ndarray  # (N, 3) float32, world cm
    indices: np.ndarray  # (M, 3) uint32
    width_cm: float
    height_cm: float

    def with_pos(self, pos: np.ndarray) -> "PageMesh":
        return replace(self, pos=np.ascontiguousarray(pos, dtype=np.float32))

    def normals(self) -> np.ndarray:
        """Vertex normals from the grid: the cross product of the position
        differences along v and along u (central differences inside, one-sided
        at the border). Same result as averaging the triangle normals around
        each vertex, at a fraction of the cost."""
        g = self.pos.astype(np.float32).reshape(self.nv, self.nu, 3)
        du = np.empty_like(g)
        du[:, 1:-1] = g[:, 2:] - g[:, :-2]
        du[:, 0] = g[:, 1] - g[:, 0]
        du[:, -1] = g[:, -1] - g[:, -2]
        dv = np.empty_like(g)
        dv[1:-1] = g[2:] - g[:-2]
        dv[0] = g[1] - g[0]
        dv[-1] = g[-1] - g[-2]
        # u runs to page right (+x), v to page bottom (-y): dv x du points up.
        n = np.cross(dv, du).reshape(-1, 3)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return n

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """World positions at UV points, interpolated on the mesh triangles."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        fu = np.clip(uv[:, 0], 0.0, 1.0) * (self.nu - 1)
        fv = np.clip(uv[:, 1], 0.0, 1.0) * (self.nv - 1)
        i = np.minimum(fu.astype(np.int64), self.nu - 2)
        j = np.minimum(fv.astype(np.int64), self.nv - 2)
        s = (fu - i)[:, None]
        t = (fv - j)[:, None]
        p = self.pos.astype(np.float64)
        p00 = p[j * self.nu + i]
        p10 = p[j * self.nu + i + 1]
        p01 = p[(j + 1) * self.nu + i]
        p11 = p[(j + 1) * self.nu + i + 1]
        # Triangles match `grid_indices`: (00, 10, 11) when s >= t, else (00, 11, 01).
        upper = p00 + s * (p10 - p00) + t * (p11 - p10)
        lower = p00 + t * (p01 - p00) + s * (p11 - p01)
        return np.where(s >= t, upper, lower)

    def boundary_uv(self, n: int = 64) -> np.ndarray:
        """UV points along the page border, clockwise from top-left."""
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        z, o = np.zeros_like(t), np.ones_like(t)
        return np.concatenate([
            np.stack([t, z], 1), np.stack([o, t], 1),
            np.stack([1 - t, o], 1), np.stack([z, 1 - t], 1),
        ])


def page_dimensions(size_cm: list[float], image_size_px: tuple[int, int]) -> tuple[float, float]:
    """The long side keeps the configured length, the short side follows the image."""
    long_side = float(max(size_cm))
    w_px, h_px = image_size_px
    if w_px >= h_px:
        return long_side, long_side * h_px / w_px
    return long_side * w_px / h_px, long_side


def grid_indices(nu: int, nv: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1))
    v00 = (j * nu + i).ravel()
    v10, v01 = v00 + 1, v00 + nu
    v11 = v01 + 1
    # v grows toward -y in world space, so this order is counter-clockwise
    # seen from above and the face normals point up (+z).
    tris = np.concatenate([np.stack([v00, v11, v10], 1), np.stack([v00, v01, v11], 1)])
    return tris.astype(np.uint32)


def make_page_mesh(width_cm: float, height_cm: float, resolution: int) -> PageMesh:
    """A flat page centered on the origin, lying in the z = 0 plane."""
    long_n = max(int(resolution), 2)
    if width_cm >= height_cm:
        nu, nv = long_n, max(2, round(long_n * height_cm / width_cm))
    else:
        nu, nv = max(2, round(long_n * width_cm / height_cm)), long_n
    u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    pos = np.stack([(uv[:, 0] - 0.5) * width_cm, (0.5 - uv[:, 1]) * height_cm, np.zeros(len(uv))], 1)
    return PageMesh(
        nu=nu, nv=nv,
        uv=uv.astype(np.float32), pos=pos.astype(np.float32),
        indices=grid_indices(nu, nv),
        width_cm=width_cm, height_cm=height_cm,
    )


def place_page(mesh: PageMesh, position_cm: list[float], rotation_deg: float, lift_cm: float) -> PageMesh:
    a = np.radians(rotation_deg)
    c, s = np.cos(a), np.sin(a)
    p = mesh.pos.astype(np.float64)
    x = c * p[:, 0] - s * p[:, 1] + position_cm[0]
    y = s * p[:, 0] + c * p[:, 1] + position_cm[1]
    z = np.maximum(p[:, 2], 0.0) + lift_cm
    return mesh.with_pos(np.stack([x, y, z], 1))
