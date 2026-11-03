"""Crumple shape from the height map of a scanned paper material.

A scan folder holds a file named like `<name>_Displacement.jpg` (the
ambientCG layout). Only the height is used: it moves the mesh, and the
renderer lights that geometry.

An optional `info.json` in the scan folder tunes one scan to match the
others, since scans differ in crease size and roughness:
    tile_scale   multiplies scan.tile_cm (2 = creases twice as large)
    depth_scale  multiplies the height (0.5 = half as deep at any strength)
    max_strength caps deform.crumple.strength for this scan
    smooth_px    blurs the height map by this many map pixels first,
                 to drop a fine paper-fiber grain and keep the creases

Mapping from page-local cm p (x right, y up) to the tiling scan:
    st = R(quarter_turns) p / tile_cm + offset,  texel = (st.x, -st.y)
so the image's up direction lies along R^T (0, 1) on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import json

import numpy as np
from PIL import Image

_SUFFIXES = {"height": "_Displacement"}
INFO_DEFAULTS = {"tile_scale": 1.0, "depth_scale": 1.0, "smooth_px": 0.0, "max_strength": float("inf")}


@dataclass(frozen=True)
class CrumpleScan:
    directory: str
    tile_cm: float
    depth_cm: float  # height range of the displacement map
    offset: np.ndarray  # (2,) tile fractions
    quarter_turns: int
    smooth_px: float = 0.0  # blur of the height map, in map pixels

    def path(self, kind: str) -> str:
        return scan_files(self.directory)[kind]

    def rotation(self) -> np.ndarray:
        k = self.quarter_turns % 4
        c, s = [1, 0, -1, 0][k], [0, 1, 0, -1][k]
        return np.array([[c, -s], [s, c]], dtype=np.float64)


@lru_cache(maxsize=16)
def scan_files(directory: str) -> dict[str, str]:
    d = Path(directory)
    out = {}
    for kind, suffix in _SUFFIXES.items():
        hits = sorted(p for p in d.iterdir() if p.stem.endswith(suffix) and p.suffix.lower() in (".jpg", ".png"))
        if not hits:
            raise FileNotFoundError(f"{directory}: no *{suffix}.jpg/png file (needed for the {kind} map)")
        out[kind] = str(hits[0])
    return out


def is_scan_dir(directory: str | Path) -> bool:
    try:
        scan_files(str(directory))
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


@lru_cache(maxsize=16)
def scan_info(directory: str) -> dict[str, float]:
    """The scan's info.json settings, with defaults for missing ones."""
    out = dict(INFO_DEFAULTS)
    p = Path(directory) / "info.json"
    if p.exists():
        data = json.loads(p.read_text())
        out.update({k: float(data[k]) for k in INFO_DEFAULTS if k in data})
    return out


@lru_cache(maxsize=8)
def height_map(path: str, smooth_px: float) -> np.ndarray:
    """The height map in [0, 1], blurred with wrap-around so the tile stays seamless."""
    h = load_map(path, gray=True).astype(np.float32) / 255.0
    if smooth_px <= 0:
        return h
    fy = np.fft.fftfreq(h.shape[0])[:, None]
    fx = np.fft.rfftfreq(h.shape[1])[None, :]
    gain = np.exp(-2.0 * (np.pi * smooth_px) ** 2 * (fx ** 2 + fy ** 2))  # a Gaussian of sigma smooth_px
    return np.fft.irfft2(np.fft.rfft2(h) * gain, s=h.shape).astype(np.float32)


@lru_cache(maxsize=8)
def load_map(path: str, gray: bool) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L" if gray else "RGB"))


def sample_height(scan: CrumpleScan, p: np.ndarray) -> np.ndarray:
    """Height in cm at page points p (N, 2), centered on 0."""
    hmap = height_map(scan.path("height"), float(scan.smooth_px))
    st = p @ scan.rotation().T / scan.tile_cm + scan.offset
    hh, ww = hmap.shape
    x = np.mod(st[:, 0], 1.0) * ww - 0.5
    y = np.mod(-st[:, 1], 1.0) * hh - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    x0, y0 = x0 % ww, y0 % hh
    x1, y1 = (x0 + 1) % ww, (y0 + 1) % hh
    h = (hmap[y0, x0] * (1 - fx) * (1 - fy) + hmap[y0, x1] * fx * (1 - fy)
         + hmap[y1, x0] * (1 - fx) * fy + hmap[y1, x1] * fx * fy)
    return (h - float(hmap.mean())).astype(np.float64) * scan.depth_cm
