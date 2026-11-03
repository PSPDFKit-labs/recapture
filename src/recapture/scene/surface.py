"""Surface detail for the page: crease occlusion and the paper grain texture."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .mesh import PageMesh


# Neighborhood radii in cm for the occlusion estimate. Small radii catch
# sharp crease valleys, large ones catch broad dips.
_AO_RADII_CM = (0.15, 0.5, 1.5)


def crease_occlusion(mesh: PageMesh, normals: np.ndarray) -> np.ndarray:
    """Ambient occlusion per vertex from the geometry, 1 = open.

    For each radius, compare a vertex with the mean position of its
    neighborhood. If the neighborhood sits above the vertex (along the
    normal), the vertex is in a valley and sees less of the room. The height
    difference over the radius approximates how much of the sky it loses.
    """
    nv, nu = mesh.nv, mesh.nu
    p = mesh.pos.astype(np.float32).reshape(nv, nu, 3)
    n = normals.astype(np.float32).reshape(nv, nu, 3)
    cell = mesh.width_cm / max(nu - 1, 1)
    ao = np.ones((nv, nu), np.float32)
    for radius in _AO_RADII_CM:
        r = max(1, int(round(radius / cell)))
        avg = _box_blur(_box_blur(p, r, 0), r, 1)
        rise = ((avg - p) * n).sum(axis=2)  # > 0: neighbors above this vertex
        ao *= 1.0 - 0.5 * np.clip(rise / (r * cell), 0.0, 1.0)
    return ao.ravel()


def _box_blur(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Mean over a window of 2r+1 along `axis`, edges repeated."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


@lru_cache(maxsize=2)
def grain_texture(size: int = 512, seed: int = 12345) -> np.ndarray:
    """Tileable paper grain, (size, size, 4) float32; B = grain, zero mean, unit std.

    Fiber-scale brightness variation only. The shape of the paper comes from
    the geometry, never from this texture.
    """
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.sqrt(fx * fx + fy * fy)

    def band(center: float, width: float) -> np.ndarray:
        noise = np.fft.fft2(rng.normal(size=(size, size)))
        out = np.real(np.fft.ifft2(noise * np.exp(-0.5 * ((f - center) / width) ** 2)))
        return out / (out.std() + 1e-12)

    grain = band(0.25, 0.12) * 0.6 + band(0.06, 0.03) * 0.4
    grain /= grain.std() + 1e-12
    out = np.zeros((size, size, 4), np.float32)
    out[..., 2] = grain
    return out
