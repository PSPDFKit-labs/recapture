"""Config defaults, loading, validation and resolution.

A user config is merged onto DEFAULTS. Keys that are not in DEFAULTS are
errors, so typos fail early. `resolve_config` turns a config with random
specs into a plain config with sampled values.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import missing_hint
from ..assets import resolve as resolve_asset
from .values import SpecError, check_tree, is_random, resolve_at, resolve_from, spec_kind, stream

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Units: centimeters, degrees. World: ground is z = 0, +z points up,
# the top edge of the source document faces +y.
DEFAULTS: dict[str, Any] = {
    "version": 1,
    "seed": 0,
    # Written into each image's .config.json: the Recapture version that made it.
    # Ignored when a config is read.
    "recapture_version": None,
    # "photo": a phone photo of the page (3D page, lamp, camera, desk).
    # "scan": a printed page through a scanner, copier or fax (2D, see `scanner`).
    "mode": "photo",
    "page": {
        "size_cm": [21.0, 29.7],
        "position_cm": [0.0, 0.0],
        "rotation_deg": 0.0,
        "lift_cm": 0.03,
        "mesh_resolution": 1024,  # vertices along the long side; the crumple lives in this mesh
    },
    "material": {
        "tint": [1.0, 0.99, 0.97],  # paper color multiplier
        "back_color": [0.85, 0.85, 0.83],  # the unprinted back side
        "roughness": 0.6,  # sheen width, 0..1
        "specular": 0.04,  # sheen strength at normal view (Fresnel F0)
        "crease_shading": 0.6,  # darkening in crease valleys (from the geometry), 0..1
        "grain": 0.03,  # paper fiber brightness variation
        "grain_tile_cm": 10.0,
        # Plastic card look (card-look). corner_radius_cm > 0 rounds the page corners (ID cards: about 0.32).
        "corner_radius_cm": 0.0,
        # Laminate glare: the mirror image of a broad ceiling or window light on a glossy surface, Fresnel-weighted.
        # strength 0 = off; about 10 to 40 washes out a patch of a laminated card. elevation/azimuth: where the
        # light is (90 = straight up); size_deg: its angular radius; softness: 0 hard edge .. 1 soft.
        # shape: "disk" (a round light) or "panel" (a rectangular ceiling panel or window, aspect = width / height,
        # turned by rotation_deg). waviness: how far the laminate's slight unevenness bends the glare's edge, 0..1.
        "glare": {"strength": 0.0, "elevation_deg": 75.0, "azimuth_deg": 0.0, "size_deg": 15.0, "softness": 0.5,
                  "shape": "disk", "aspect": 1.6, "rotation_deg": 0.0, "waviness": 0.0},
        # A faint mirror image of the room on a glossy laminate, over the whole card, 0 = off, about 1 = clearly glossy.
        "room_reflection": 0.0,
        # Laminate wear: fine scratches and smudges that show where the glare falls, 0 = a new card, 1 = worn.
        "wear": 0.0,
        # Holographic laminate: strength of a rainbow sheen that shifts with the view angle, in a fine pattern.
        "hologram": 0.0,
    },
    # A 2D layer under the rendered page, added before the camera effects,
    # like a layer in an image editor. The page casts nothing onto it.
    "background": {
        # False: no background at all. The photo is a PNG, transparent around
        # the page. Camera effects still apply to the page.
        "enabled": True,
        # A file, a directory (searched recursively), or a list of them.
        # None gives a plain color.
        "images": None,
        "color": [0.35, 0.30, 0.25],  # linear RGB, used when images is None
        "zoom": 1.0,  # >= 1 crops into the image; 1 = the image just covers the frame
        "center": [0.5, 0.5],  # crop center, image fractions
        "quarter_turns": 0,  # rotate by 90 degrees this many times
        "flip": False,  # mirror left-right
        "brightness": 0.35,  # mean brightness relative to the paper white
        "saturation": 1.0,
        "blur_px": 0.0,
        # The background is a table under the page: the lamp lights it, with
        # its falloff and color, and the page and hand shadows fall on it.
        # False: a flat picture with one brightness, no light and no shadows.
        "lit": True,
        "contact_shadow": 0.6,  # darkening where the page touches or nearly touches the table, 0..1
        "contact_radius_cm": 1.0,  # how far that darkening reaches
    },
    # Shapes that cast shadows but are not seen by the camera. A list, or
    # {"repeat": n, "item": {...}}. See OCCLUDER_DEFAULTS.
    "occluders": [],
    # A desk under the page instead of a background picture: a desk-top
    # texture lit by the lamp, and items beside the page (a keyboard, a
    # mousepad, a pen, other paper). Needs background.enabled.
    "desk": {
        "enabled": False,
        # A desk texture folder (albedo.jpg + info.json from
        # `recapture fetch-assets`), a folder of them, or a list. None = plain `color`.
        "surface": None,
        "color": [0.6, 0.6, 0.58],  # linear RGB, for a plain desk
        "texture_offset": None,  # [u, v] texture fractions; None = from the crumple seed
        "texture_rotation_deg": 0.0,
        # Images printed on "paper" items (a file, a folder or a list). None = blank paper.
        "papers": None,
        # A list of items, or {"repeat": n, "item": {...}}. See DESK_ITEM_DEFAULTS.
        "items": [],
    },
    "light": {
        # Where the lamp is, seen from the page center. Low elevations rake
        # across the page and show the crumple most; overhead hides it.
        # Azimuth: 0 = from the page's right, 90 = from its top edge,
        # counterclockwise seen from above.
        "elevation_deg": 25.0,
        "azimuth_deg": 135.0,
        "distance_cm": 60.0,
        "position_cm": None,  # [x, y, z] places the lamp directly and overrides the angles
        "intensity": 0.9,
        "size_cm": 3.0,  # lamp size; larger = softer shadows
        "color_temp_k": 5000.0,
        "ambient": 0.12,  # room light level; more room light shows the crumple less
        "ambient_color": None,  # linear RGB; None = the lamp's color
        # Room light comes mostly from this direction (like a window or a
        # ceiling). ambient_focus 0 = even from all sides, 1 = strongly
        # from one side, so facets in shadow still differ in brightness.
        "ambient_direction": [0.0, 0.0, 1.0],
        "ambient_focus": 0.5,
        "wrap": 0.0,
    },
    "camera": {
        "position_cm": [0.0, -10.0, 34.0],
        "target_cm": [0.0, 0.0, 0.0],
        "roll_deg": 0.0,
        "focal_mm": 26.0,
        "output_px": [1536, 2048],
        "frame_coverage": [0.2, 0.95],
        "frame_margin": 0.02,
        # True: frame the page like someone who tries to photograph only the
        # page. position_cm and target_cm then set only the view angle; the
        # camera moves in or out and aims so the page fills the frame, turned
        # so the page edges run along the image edges (roll_deg adds to that).
        # frame_coverage does not apply.
        "fit_page": False,
        "fit_margin": 0.03,  # desk left on the tightest side, as a part of the short image side
        "max_tries": 50,
    },
    "deform": {
        "crumple": {
            # Where the crumple shape comes from. Either way it moves the mesh:
            # "procedural": a random crease network.
            # "scan": the height map of a scanned crumpled-paper material.
            "mode": "procedural",
            # How crumpled the page is: 0 = flat, 1 = a normal crumple, 2 = heavy.
            # It scales the crumple of both modes.
            "strength": 0.0,
            "scan": {
                # A scan folder (with a *_Displacement map), a folder of scan
                # folders, or a list. None = no scan.
                "directory": None,
                "tile_cm": 22.0,  # page length covered by one copy of the scan
                "depth_cm": 0.6,  # height range of the displacement map at strength 1
                "offset": None,  # [u, v] tile fractions; None = from the crumple seed
                "quarter_turns": None,  # None = from the crumple seed
            },
            "creases": 300,  # number of crease segments; more = smaller facets
            "width_cm": [0.3, 2.5],  # range for the half-width of each crease
            "length": [0.05, 0.35],  # range for crease length, as a part of the page diagonal
            "waviness_cm": 0.0,  # procedural: soft large-scale waves at strength 1
            "seed": None,  # None = picked from the job seed
        },
        # A list of fold items, or {"repeat": n, "item": {...}}. See FOLD_DEFAULTS.
        "folds": [],
        "curl": {
            # 1 / radius in cm. 0 = flat, 0.02 = radius 50 cm. Positive lifts the edges.
            "x_curvature": 0.0,
            "x_center": 0.5,  # page fraction, from the left edge
            "y_curvature": 0.0,
            "y_center": 0.5,  # page fraction, from the top edge
        },
        "edge_lift_cm": [0.0, 0.0, 0.0, 0.0],  # top, right, bottom, left
        "edge_lift_width_cm": 3.0,
    },
    "fx": {
        "enabled": True,  # False skips every effect below (plain clip + sRGB)
        # "gpu": run the effects on the GPU (fast). "cpu": the numpy version.
        # They agree within a level or two of 255; a transparent background always uses "cpu".
        "backend": "gpu",
        "seed": None,  # None = picked from the job seed
        # Scale the image so this percentile of page brightness lands at
        # `target` after tone mapping, like a phone's auto exposure.
        "auto_exposure": {"enabled": True, "percentile": 97.0, "target": 0.86},
        "exposure_ev": 0.0,  # extra exposure in stops, after auto exposure
        "white_balance": 0.7,  # 0 keeps the light's color cast, 1 removes it
        "tint": [1.0, 1.0, 1.0],  # per-channel gain after white balance
        "defocus": {"enabled": True, "strength_px": 1.5, "focus": [0.5, 0.5]},  # blur at 2x the focus distance
        "motion_blur": {"enabled": False, "length_px": 2.0, "angle_deg": 0.0},
        "vignette": 0.25,  # corner darkening, 0..1
        "noise": {"enabled": True, "shot": 0.002, "read": 0.0005, "chroma": 0.3, "grain_px": 0.6},
        "tonemap": "filmic",  # "filmic" or "clip"
        "sharpen": {"enabled": True, "amount": 0.4, "radius_px": 1.2},
        # Camera quality: 1 = a good camera (no change), 0 = a cheap phone.
        # Lower values add a low-resolution sensor, color fringes, sharpening
        # halos and JPEG blocks (see fx/quality.py).
        "quality": 1.0,
    },
    # Scanner-app crop: the output is the page alone, cut out of the photo
    # at its four corners and pulled to a rectangle (like a phone scanner
    # app). The camera then frames the page itself (see camera.fit_page).
    "crop": {
        "enabled": False,
        # Space around the page, as a part of the page size. Negative cuts into the page.
        "margin": 0.0,
        # How far off each corner the app finds it, as a part of the page's short side.
        "corner_error": 0.0,
        "size_px": None,  # [width, height]; None = the page's shape, long side from camera.output_px
        "seed": None,  # None = picked from the job seed
    },
    # Scan mode only (mode "scan"): the printer's marks on the page, in document
    # pixels. Every effect is off at 0. See scanner.py.
    "printer": {
        "ink_bleed": 0.0,  # 0..1: ink spreads into the paper around each stroke
        "broken_strokes": 0.0,  # part of the ink pixels left out (low toner), e.g. 0.01 to 0.04
        "ghost": 0.0,  # 0..1: strength of a faint shifted copy of the ink (double strike)
        "toner_streaks": 0,  # thin broken lines along the paper feed (a dirty drum)
        "toner_specks": 0,  # black dots of loose toner
    },
    # Scan mode only (mode "scan"): a printed page on a flatbed scanner. See scanner.py.
    "scanner": {
        "dpi": 200,  # scan resolution; the output size is the paper size (page.size_cm) at this dpi
        "color": "color",  # "color", "gray", or "bw" (1-bit black and white)
        "bw_threshold": 150,  # "bw": gray levels below this become black
        "skew_deg": 0.0,  # the page lies this crooked on the glass
        "blur_px": 0.6,  # lens and sensor softness, in scan pixels
        "bold": 0.0,  # 0..1: strokes thicken (an over-dark scan); 1 = about one scan pixel; 0 = off
        # Dark bands where the scanner sees past the paper, as parts of the page width:
        # [left, right, top, bottom]. 0 = none on that side.
        "dark_edges": [0.0, 0.0, 0.0, 0.0],
        "edge_dots": 0,  # loose dots scattered beside the dark bands
        "noise": 2.0,  # sensor noise, standard deviation in gray levels
        "streaks": 0,  # thin lines along the scan direction (dust on the sensor bar)
        "dust": 0,  # dark specks and short hairs on the glass
        "seed": None,  # None = picked from the job seed
    },
    "output": {
        "format": "jpg",
        "jpeg_quality": 92,
    },
}

FOLD_DEFAULTS: dict[str, Any] = {
    "position": [0.5, 0.5],  # a point on the fold line, page fractions (x right, y down)
    "direction_deg": 0.0,  # 0 = parallel to the top edge. +180 moves the other side.
    "angle_deg": 20.0,  # positive lifts the moving side toward the printed face
    "crease_radius_cm": 0.15,
}

DESK_KINDS = ("keyboard", "mousepad", "phone", "mug", "pen", "pencil", "sticky_notes", "paper")

DESK_ITEM_DEFAULTS: dict[str, Any] = {
    "kind": "keyboard",  # keyboard, mousepad, phone, mug, pen, pencil, sticky_notes or paper
    "side": "right",  # the page edge it lies beside: "left", "right", "top" or "bottom"
    "along": 0.5,  # where along that edge, page fractions
    "gap_cm": 3.0,  # distance from the page edge
    "rotation_deg": 0.0,  # turn, relative to the page
    "color": None,  # linear RGB; None = picked from the item's usual colors
    "image": None,  # paper only: the printed image; None = one of desk.papers
}

OCCLUDER_DEFAULTS: dict[str, Any] = {
    "shape": "hand_phone",  # "hand_phone", "arm", "phone" or "blob"
    # [x, y, z] world cm of the hand itself. It stays there when the lamp
    # moves, and its shadow falls where the lamp throws it. None = aim the
    # shadow at `target` instead (the hand sits on the line to the lamp).
    "position_cm": None,
    "target": [0.5, 0.5],  # page point (fractions) where the shadow center falls
    "height_cm": 20.0,  # height of the shape above the ground (with `target`)
    "scale": 1.0,
    # The arm points toward the camera; this turns it further.
    "rotation_offset_deg": 0.0,
}

# Dicts under these paths hold free-form keys (none yet).
_OPEN_PATHS: set[str] = set()


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        user = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from e
    return build_config(user, base_dir=path.parent)


def build_config(user: dict[str, Any], base_dir: str | Path = ".") -> dict[str, Any]:
    """Merge `user` onto the defaults, check it, and expand file paths."""
    if not isinstance(user, dict):
        raise ConfigError("config root must be a JSON object")
    _check_keys(user, DEFAULTS, "$")
    cfg = _merge(copy.deepcopy(DEFAULTS), user)
    try:
        check_tree(cfg)
    except SpecError as e:
        raise ConfigError(str(e)) from e
    tries = cfg["camera"]["max_tries"]
    if not (isinstance(tries, int) and not isinstance(tries, bool) and tries >= 1):
        raise ConfigError("camera.max_tries: must be a fixed integer >= 1")
    cfg["deform"]["folds"] = _list_items(cfg["deform"]["folds"], FOLD_DEFAULTS, "deform.folds")
    cfg["occluders"] = _list_items(cfg["occluders"], OCCLUDER_DEFAULTS, "occluders")
    desk = cfg["desk"]
    desk["items"] = _list_items(desk["items"], DESK_ITEM_DEFAULTS, "desk.items")
    desk["surface"] = _expand_desks(desk["surface"], Path(base_dir))
    if desk["papers"] is not None and not spec_kind(desk["papers"]):
        papers = _expand_images(desk["papers"], Path(base_dir))
        desk["papers"] = papers["choice"] if isinstance(papers, dict) else [papers]
    cfg["background"]["images"] = _expand_images(cfg["background"]["images"], Path(base_dir))
    scan = cfg["deform"]["crumple"]["scan"]
    scan["directory"] = _expand_scans(scan["directory"], Path(base_dir))

    return cfg


# The values the frame check depends on: where the camera is and where the page lies. When a sample fails the
# check, a retry draws only these again, so the desk, the lamp, the crumple and the items stay (pipeline.sample_scene).
FRAME_PATHS: tuple[tuple[str, ...], ...] = (("camera",), ("page", "position_cm"), ("page", "rotation_deg"))
# Folds change the page's outline too, and some fold sets fit no camera (two crossing folds can push a flap past
# the page). Late retries draw them again as well; early ones keep them, so the folds do not jump needlessly.
FOLD_PATHS: tuple[tuple[str, ...], ...] = (("deform", "folds"),)


def frame_is_random(cfg: dict[str, Any]) -> bool:
    """True when a retry can change the frame check's result: a value in FRAME_PATHS or FOLD_PATHS is random."""
    return any(is_random(_at(cfg, path)) for path in FRAME_PATHS + FOLD_PATHS)


def resolve_config(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sample every random spec. The result holds plain values only."""
    return resolve_with_base(cfg, int(rng.integers(0, 2**63 - 1)))


def redraw_frame(resolved: dict[str, Any], cfg: dict[str, Any], base: int, attempt: int,
                 folds: bool = False) -> dict[str, Any]:
    """`resolved` (attempt 0 of `base`) with the values in FRAME_PATHS, and with `folds` those in FOLD_PATHS,
    drawn from retry `attempt`'s own streams. Everything else is shared with `resolved`, not copied."""
    retry = int(stream(base, ("retry", attempt)).integers(0, 2**63 - 1))
    out = dict(resolved)
    for path in FRAME_PATHS + (FOLD_PATHS if folds else ()):
        node = out
        for key in path[:-1]:
            node[key] = dict(node[key])
            node = node[key]
        node[path[-1]] = resolve_at(_at(cfg, path), retry, path)
    _check_resolved(out)
    return out


def resolve_with_base(cfg: dict[str, Any], base: int) -> dict[str, Any]:
    """`resolve_config` with its streams keyed by `base`."""
    out = resolve_from(cfg, base)
    for name, section in (("crumple", out["deform"]["crumple"]), ("fx", out["fx"]), ("crop", out["crop"]),
                          ("scanner", out["scanner"])):
        if section["seed"] is None:  # its own stream too, so other settings leave it alone
            section["seed"] = int(stream(base, ("seed", name)).integers(0, 2**31 - 1))
    papers = out["desk"]["papers"] or []
    for i, item in enumerate(out["desk"]["items"]):
        if item["kind"] == "paper" and item["image"] is None and papers:
            item["image"] = papers[int(stream(base, ("desk", "paper", i)).integers(len(papers)))]
    _check_resolved(out)
    return out


def _at(cfg: Any, path: tuple) -> Any:
    for key in path:
        cfg = cfg[key]
    return cfg


def _list_items(items: Any, defaults: dict[str, Any], name: str) -> Any:
    """Check list items (a list, or a repeat spec) and fill in missing keys."""
    def item(f: Any, path: str) -> dict:
        if not isinstance(f, dict) or spec_kind(f):
            raise ConfigError(f"{path}: each item must be an object")
        _check_keys(f, defaults, path)
        return _merge(copy.deepcopy(defaults), f)

    if isinstance(items, list):
        return [item(f, f"$.{name}[{i}]") for i, f in enumerate(items)]
    if spec_kind(items) == "repeat":
        return {"repeat": items["repeat"], "item": item(items["item"], f"$.{name}.item")}
    raise ConfigError(f"{name}: must be a list or {{\"repeat\": n, \"item\": {{...}}}}")


def _merge(base: dict, user: dict) -> dict:
    for k, v in user.items():
        if isinstance(v, dict) and not spec_kind(v) and isinstance(base.get(k), dict):
            base[k] = _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _check_keys(user: dict, defaults: dict, path: str) -> None:
    for k, v in user.items():
        p = f"{path}.{k}"
        if k not in defaults:
            known = ", ".join(sorted(defaults))
            raise ConfigError(f"{p}: unknown key. Known keys here: {known}")
        section = isinstance(defaults[k], dict) and not spec_kind(defaults[k]) and p not in _OPEN_PATHS
        if section and not isinstance(v, dict):
            known = ", ".join(sorted(defaults[k]))
            raise ConfigError(f"{p}: must be an object with these keys: {known}")
        if section and not spec_kind(v):
            _check_keys(v, defaults[k], p)


def _expand_images(images: Any, base_dir: Path) -> Any:
    """Turn a directory or list of files into a choice spec of absolute paths."""
    if images is None or spec_kind(images):
        return images
    paths = images if isinstance(images, list) else [images]
    files: list[str] = []
    for item in paths:
        p = resolve_asset(item, base_dir)
        if p.is_dir():
            files += sorted(str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            files.append(str(p))
        else:
            raise ConfigError(f"background.images: not found: {p}{missing_hint(p)}")
    if not files:
        raise ConfigError("background.images: no image files found")
    return files[0] if len(files) == 1 else {"choice": files}


def _expand_scans(directory: Any, base_dir: Path) -> Any:
    """Turn scan folders into a choice spec of absolute scan folder paths."""
    if directory is None or spec_kind(directory):
        return directory
    from ..scene.paper_scan import is_scan_dir

    found: list[str] = []
    for item in directory if isinstance(directory, list) else [directory]:
        p = resolve_asset(item, base_dir)
        if is_scan_dir(p):
            found.append(str(p))
        elif p.is_dir():
            found += sorted(str(c) for c in p.iterdir() if c.is_dir() and is_scan_dir(c))
        else:
            raise ConfigError(f"deform.crumple.scan.directory: not found: {p}{missing_hint(p)}")
    if not found:
        raise ConfigError("deform.crumple.scan.directory: no scan folders found")
    return found[0] if len(found) == 1 else {"choice": found}


def _expand_desks(surface: Any, base_dir: Path) -> Any:
    """Turn desk texture folders into a choice spec of absolute folder paths."""
    if surface is None or spec_kind(surface):
        return surface
    found: list[str] = []
    for item in surface if isinstance(surface, list) else [surface]:
        p = resolve_asset(item, base_dir)
        if (p / "albedo.jpg").exists():
            found.append(str(p))
        elif p.is_dir():
            found += sorted(str(c) for c in p.iterdir() if (c / "albedo.jpg").exists())
        else:
            raise ConfigError(f"desk.surface: not found: {p}{missing_hint(p)}")
    if not found:
        raise ConfigError("desk.surface: no desk texture folders (with albedo.jpg) found")
    return found[0] if len(found) == 1 else {"choice": found}


def _check_resolved(cfg: dict[str, Any]) -> None:
    def vec(path: str, value: Any, n: int) -> None:
        if not (isinstance(value, list) and len(value) == n):
            raise ConfigError(f"{path}: needs a list of {n} numbers, got {value!r}")

    page, cam, light = cfg["page"], cfg["camera"], cfg["light"]
    vec("page.size_cm", page["size_cm"], 2)
    vec("page.position_cm", page["position_cm"], 2)
    vec("camera.position_cm", cam["position_cm"], 3)
    vec("camera.target_cm", cam["target_cm"], 3)
    vec("camera.output_px", cam["output_px"], 2)
    vec("camera.frame_coverage", cam["frame_coverage"], 2)
    if light["position_cm"] is not None:
        vec("light.position_cm", light["position_cm"], 3)
        if light["position_cm"][2] <= 0:
            raise ConfigError("light.position_cm: z must be above the page (> 0)")
    elif not 1.0 <= light["elevation_deg"] <= 90.0:
        raise ConfigError("light.elevation_deg: must be in [1, 90]")
    if light["ambient_color"] is not None:
        vec("light.ambient_color", light["ambient_color"], 3)
    vec("light.ambient_direction", light["ambient_direction"], 3)
    if np.linalg.norm(light["ambient_direction"]) < 1e-6:
        raise ConfigError("light.ambient_direction: must not be zero")
    if min(page["size_cm"]) <= 0:
        raise ConfigError("page.size_cm: must be positive")
    if int(page["mesh_resolution"]) < 2:
        raise ConfigError("page.mesh_resolution: must be >= 2")
    if cam["position_cm"][2] <= 0:
        raise ConfigError("camera.position_cm: z must be above the ground (> 0)")
    if not 0 <= cfg["fx"]["quality"] <= 1:
        raise ConfigError("fx.quality: must be in [0, 1]")
    if cfg["mode"] not in ("photo", "scan"):
        raise ConfigError("mode: must be 'photo' or 'scan'")
    pr = cfg["printer"]
    if min(pr["ink_bleed"], pr["broken_strokes"], pr["ghost"], pr["toner_streaks"], pr["toner_specks"]) < 0:
        raise ConfigError("printer: every value must be >= 0")
    if pr["ink_bleed"] > 1 or pr["ghost"] > 1 or pr["broken_strokes"] > 0.5:
        raise ConfigError("printer: ink_bleed and ghost must be <= 1, broken_strokes <= 0.5")
    sc = cfg["scanner"]
    if not 30 <= sc["dpi"] <= 1200:
        raise ConfigError("scanner.dpi: must be in [30, 1200]")
    if sc["color"] not in ("color", "gray", "bw"):
        raise ConfigError("scanner.color: must be 'color', 'gray' or 'bw'")
    if sc["bold"] > 1:
        raise ConfigError("scanner.bold: must be in [0, 1]")
    vec("scanner.dark_edges", sc["dark_edges"], 4)
    if min(sc["dark_edges"]) < 0 or max(sc["dark_edges"]) > 0.3:
        raise ConfigError("scanner.dark_edges: each side must be in [0, 0.3]")
    if sc["blur_px"] < 0 or sc["noise"] < 0 or sc["streaks"] < 0 or sc["dust"] < 0 or sc["bold"] < 0 \
            or sc["edge_dots"] < 0:
        raise ConfigError("scanner: blur_px, bold, noise, streaks, dust and edge_dots must be >= 0")
    crop = cfg["crop"]
    if not -0.2 <= crop["margin"] <= 0.5:
        raise ConfigError("crop.margin: must be in [-0.2, 0.5]")
    if not 0 <= crop["corner_error"] <= 0.1:
        raise ConfigError("crop.corner_error: must be in [0, 0.1]")
    if crop["size_px"] is not None:
        vec("crop.size_px", crop["size_px"], 2)
        if min(crop["size_px"]) < 16:
            raise ConfigError("crop.size_px: must be at least 16 px per side")
    if not 0 <= cam["fit_margin"] < 0.45:
        raise ConfigError("camera.fit_margin: must be in [0, 0.45)")
    if not 1 <= cam["focal_mm"] <= 1000:
        raise ConfigError("camera.focal_mm: must be in [1, 1000]")
    crumple = cfg["deform"]["crumple"]
    if crumple["strength"] < 0:
        raise ConfigError("deform.crumple.strength: must be >= 0")
    if crumple["mode"] not in ("procedural", "scan"):
        raise ConfigError("deform.crumple.mode: must be 'procedural' or 'scan'")
    if crumple["mode"] == "scan" and crumple["scan"]["directory"] is None:
        raise ConfigError("deform.crumple.scan.directory: needed when mode is 'scan'")
    vec("deform.crumple.width_cm", cfg["deform"]["crumple"]["width_cm"], 2)
    vec("deform.crumple.length", cfg["deform"]["crumple"]["length"], 2)
    vec("deform.edge_lift_cm", cfg["deform"]["edge_lift_cm"], 4)
    if min(cfg["deform"]["crumple"]["width_cm"]) <= 0:
        raise ConfigError("deform.crumple.width_cm: must be positive")
    for i, fold in enumerate(cfg["deform"]["folds"]):
        vec(f"deform.folds[{i}].position", fold["position"], 2)
        if not -170 <= fold["angle_deg"] <= 170:
            raise ConfigError(f"deform.folds[{i}].angle_deg: must be in [-170, 170]")
    for k in ("x_curvature", "y_curvature"):
        if abs(cfg["deform"]["curl"][k]) > 0.2:
            raise ConfigError(f"deform.curl.{k}: must be in [-0.2, 0.2] (radius 5 cm or more)")
    for i, occ in enumerate(cfg["occluders"]):
        if occ["shape"] not in ("hand_phone", "arm", "phone", "blob"):
            raise ConfigError(f"occluders[{i}].shape: must be hand_phone, arm, phone or blob")
        vec(f"occluders[{i}].target", occ["target"], 2)
        if occ["position_cm"] is not None:
            vec(f"occluders[{i}].position_cm", occ["position_cm"], 3)
            if occ["position_cm"][2] <= 0:
                raise ConfigError(f"occluders[{i}].position_cm: z must be above the page (> 0)")
        if occ["height_cm"] <= 0:
            raise ConfigError(f"occluders[{i}].height_cm: must be > 0")
    for i, item in enumerate(cfg["desk"]["items"]):
        if item["kind"] not in DESK_KINDS:
            raise ConfigError(f"desk.items[{i}].kind: must be one of {', '.join(DESK_KINDS)}")
        if item["side"] not in ("left", "right", "top", "bottom"):
            raise ConfigError(f"desk.items[{i}].side: must be left, right, top or bottom")
    if cfg["fx"]["backend"] not in ("gpu", "cpu"):
        raise ConfigError("fx.backend: must be 'gpu' or 'cpu'")
    if cfg["material"]["glare"]["shape"] not in ("disk", "panel"):
        raise ConfigError("material.glare.shape: must be 'disk' or 'panel'")
    if cfg["fx"]["tonemap"] not in ("filmic", "clip"):
        raise ConfigError("fx.tonemap: must be 'filmic' or 'clip'")
    if cfg["output"]["format"] not in ("jpg", "png"):
        raise ConfigError("output.format: must be 'jpg' or 'png'")
    if not cfg["background"]["enabled"] and cfg["output"]["format"] != "png":
        raise ConfigError("background.enabled is false: set output.format to png (JPEG has no transparency)")
