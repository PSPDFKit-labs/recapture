"""The CC0 assets: where they live and how to get them.

    recapture fetch-assets

downloads them once per machine (about 105 MB; 85 MB kept):

    polyhaven/textures/<name>/albedo.jpg   surfaces for the flat background layer
    polyhaven/desks/<name>/albedo.jpg      desk tops, with info.json (real-world size)
    ambientcg/<Paper00N>/...Displacement   crumpled-paper height maps, info.json tunes one
    desks_plain/<name>/albedo.jpg          plain laminate desks, made here (no download)

Poly Haven and ambientCG publish under CC0, so the files can go into any
dataset, commercial ones included, with no attribution.

The asset folder is, in order: $RECAPTURE_ASSETS; `assets/` in the repo
when this runs from a clone; ~/.cache/recapture/assets. Configs name files
in it as "@assets/<path>", so a config works in any folder.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

PREFIX = "@assets/"
DOWNLOAD_MB = 105  # what a first fetch downloads, about
KEPT_MB = 85  # what it keeps on disk, about

POLYHAVEN = "https://api.polyhaven.com"
_HEADERS = {"User-Agent": "recapture-asset-fetch/0.1"}  # the APIs refuse the default Python agent

# Surfaces for the flat background layer.
BACKGROUNDS = [
    "weathered_brown_planks", "wood_table_001", "wood_table_worn", "old_planks_02",
    "laminate_floor_02", "dark_wood", "denim_fabric", "concrete_floor_worn_001", "marble_01",
]
# Desk tops. The Poly Haven concretes look like floors, so they are left out.
DESKS = [
    # wood
    "oak_veneer_01", "white_oak_veneer", "walnut_veneer", "grey_oak_veneer_01", "black_oak_veneer",
    "white_maple_veneer", "plywood", "cherry_veneer", "washed_grey_oak_veneer", "wood_table_001",
    "american_walnut_veneer", "ash_veneer", "teak_veneer", "smoked_walnut_veneer", "wood_table",
    "wood_table_worn", "stained_pine", "coated_pine", "fine_grained_wood", "white_planks_clean",
    # stone, linoleum
    "marble_01", "linoleum_brown",
    # leather desk pads, tablecloths
    "brown_leather", "leather_white", "rough_linen", "gingham_check",
]
# ambientCG crumple scans; only the height map is kept. A scan with settings
# gets an info.json that tunes it (see scene/paper_scan.py).
SCANS = {
    "Paper003": None,
    # Packing paper with a fine fiber grain: larger, smoothed and capped so it reads as office paper.
    "Paper005": {"tile_scale": 2.5, "depth_scale": 1.5, "smooth_px": 2.0, "max_strength": 1.2},
}
# Plain laminate desks, made here. name: (sRGB color, mottle, speck, brushed)
PLAIN_DESKS = {
    "white_laminate": ((236, 236, 232), 0.015, 0.025, False),
    "warm_white_laminate": ((232, 227, 216), 0.015, 0.03, False),
    "light_grey_laminate": ((200, 201, 200), 0.02, 0.03, False),
    "grey_laminate": ((150, 152, 153), 0.025, 0.04, False),
    "anthracite_laminate": ((66, 68, 71), 0.03, 0.05, False),
    "black_laminate": ((34, 34, 36), 0.04, 0.06, False),
    "beige_laminate": ((214, 200, 176), 0.02, 0.035, False),
    "brushed_grey": ((168, 170, 172), 0.02, 0.02, True),
    "brushed_black": ((44, 45, 47), 0.03, 0.03, True),
}
PARTS = ("textures", "desks", "scans", "plain")


def asset_home() -> Path:
    """The asset folder (see the module docstring)."""
    env = os.environ.get("RECAPTURE_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if (repo / "pyproject.toml").is_file() and (repo / "src" / "recapture").is_dir():  # running from a clone
        return repo / "assets"
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "recapture" / "assets"


def resolve(item: str | Path, base_dir: Path) -> Path:
    """A config path: "@assets/..." in the asset folder, otherwise relative to the config file."""
    s = str(item)
    if s.startswith(PREFIX):
        return asset_home() / s[len(PREFIX):]
    p = Path(s).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def missing_hint(path: Path) -> str:
    """Added to "not found" errors for files in the asset folder."""
    try:
        path.resolve().relative_to(asset_home().resolve())
    except ValueError:
        return ""
    return " (run `recapture fetch-assets` to download the assets)"


def fetch(home: Path | None = None, parts: tuple[str, ...] = PARTS, res: str = "2k",
          log: Callable[[str], None] = print) -> tuple[int, int]:
    """Download what is missing. Returns (bytes downloaded, failures). Files
    already there are skipped, so this is safe to run again."""
    home = home or asset_home()
    total, failed = 0, 0

    def step(name: str, fn: Callable[[], int]) -> None:
        nonlocal total, failed
        try:
            total += fn()
        except Exception as e:  # report and go on with the next file
            failed += 1
            log(f"  {name}: FAILED {e}")

    if "textures" in parts:
        for name in BACKGROUNDS:
            step(name, lambda name=name: _polyhaven(name, res, home / "polyhaven" / "textures" / name, log, info=False))
    if "desks" in parts:
        for name in DESKS:
            step(name, lambda name=name: _polyhaven(name, res, home / "polyhaven" / "desks" / name, log, info=True))
    if "scans" in parts:
        for name, info in SCANS.items():
            step(name, lambda name=name, info=info: _scan(name, res.upper(), home / "ambientcg" / name, info, log))
    if "plain" in parts:
        step("plain desks", lambda: _plain_desks(home / "desks_plain", log))
    return total, failed


# ---- downloads ---------------------------------------------------------------------------


def _read(url: str, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=300) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))  # the APIs time out now and then
    raise AssertionError("unreachable")


def _polyhaven(name: str, res: str, folder: Path, log, info: bool) -> int:
    """One Poly Haven texture's albedo, and with `info` its real-world size."""
    target = folder / "albedo.jpg"
    if target.exists() and (not info or (folder / "info.json").exists()):
        return 0
    entry = json.loads(_read(f"{POLYHAVEN}/files/{name}"))["Diffuse"][res]["jpg"]
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    if not (target.exists() and target.stat().st_size == entry["size"]):
        log(f"  {folder.parent.name}/{name}  {entry['size'] / 1e6:.1f} MB")
        target.write_bytes(_read(entry["url"]))
        n = entry["size"]
    if info:
        meta = json.loads(_read(f"{POLYHAVEN}/info/{name}"))
        w_mm, h_mm = meta.get("dimensions") or [2000.0, 2000.0]
        (folder / "info.json").write_text(json.dumps({"name": name, "size_cm": [w_mm / 10, h_mm / 10], "license": "CC0",
                                                      "source": f"https://polyhaven.com/a/{name}"}))
    return n


def _scan(name: str, res: str, folder: Path, info: dict | None, log) -> int:
    """One ambientCG material's height map (the rest of the pack is dropped)."""
    target = folder / f"{name}_{res}-JPG_Displacement.jpg"
    n = 0
    if not target.exists():
        log(f"  ambientcg/{name} ({res} pack)")
        data = _read(f"https://ambientcg.com/get?file={name}_{res}-JPG.zip")
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.namelist():
                if member.endswith("_Displacement.jpg") and "/" not in member:
                    target.write_bytes(z.read(member))
        n = len(data)
    if info is not None and not (folder / "info.json").exists():
        (folder / "info.json").write_text(json.dumps(info, indent=1))
    return n


# ---- plain desks (generated) ---------------------------------------------------------------

_PLAIN_PX = 1024
_PLAIN_CM = 60.0  # one copy of the texture covers 60 x 60 cm


def _plain_desks(folder: Path, log) -> int:
    """Plain laminate desks. A flat color looks fake, so each gets the faint
    mottling and fine speck of real laminate, some a light brushed grain. The
    noise is made in the frequency domain, so every texture tiles without a seam."""
    made = 0
    for k, (name, (color, mottle, speck, brushed)) in enumerate(PLAIN_DESKS.items()):
        out = folder / name
        if (out / "albedo.jpg").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(plain_desk(color, mottle, speck, brushed, seed=k)).save(out / "albedo.jpg", quality=95)
        (out / "info.json").write_text(json.dumps({"name": name, "size_cm": [_PLAIN_CM, _PLAIN_CM],
                                                   "license": "generated", "source": "recapture.assets"}))
        made += 1
    if made:
        log(f"  desks_plain: {made} made")
    return 0


def plain_desk(color, mottle: float, speck: float, brushed: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lin = (np.asarray(color, np.float64) / 255.0) ** 2.2
    shade = 1.0 + mottle * _noise(rng, _PLAIN_PX, 0.01) + speck * _noise(rng, _PLAIN_PX, 0.35)
    if brushed:  # fine streaks along x
        shade += 0.03 * _noise(rng, _PLAIN_PX, 0.3, stretch=(25.0, 1.0))
    out = np.clip(lin[None, None] * shade[..., None], 0.0, 1.0) ** (1 / 2.2)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _noise(rng: np.random.Generator, size: int, cutoff: float, stretch: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Seamless noise with unit spread: white noise low-passed in the frequency domain."""
    f = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(f * stretch[1], f * stretch[0], indexing="ij")
    gain = np.exp(-(fx ** 2 + fy ** 2) / (2 * cutoff ** 2))
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * gain))
    return (n - n.mean()) / (n.std() + 1e-12)
