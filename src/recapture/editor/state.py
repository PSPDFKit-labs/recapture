"""Config editing helpers for the editor. Pure functions, no UI.

The editor keeps the user's own config (only the keys they set) and shows
it merged onto the defaults. Saving writes the user's config, so the file
stays short and readable.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from ..config.schema import DEFAULTS, DESK_ITEM_DEFAULTS, FOLD_DEFAULTS, OCCLUDER_DEFAULTS
from ..config.values import spec_kind

Path = tuple  # of str keys and int indices

ITEM_DEFAULTS = {"folds": FOLD_DEFAULTS, "occluders": OCCLUDER_DEFAULTS, "items": DESK_ITEM_DEFAULTS}

# Keys whose string values come from a fixed set.
ENUMS: dict[str, list] = {
    "mode": ["procedural", "scan"],
    "shape": ["hand_phone", "arm", "phone", "blob"],
    "tonemap": ["filmic", "clip"],
    "format": ["jpg", "png"],
    "kind": ["keyboard", "mousepad", "phone", "mug", "pen", "pencil", "sticky_notes", "paper"],
    "side": ["left", "right", "top", "bottom"],
}
# Keys whose name means something else at one place in the tree.
PATH_ENUMS: dict[tuple, list] = {("mode",): ["photo", "scan"]}


def read_mode(mode: Any) -> tuple[str, float]:
    """The editor's view of `mode`: ("photo" | "scan" | "mix" | "other", share of scans).
    "mix" is a choice between photo and scan, with or without weights."""
    if mode in ("photo", "scan"):
        return mode, 1.0 if mode == "scan" else 0.0
    if isinstance(mode, dict) and set(mode) <= {"choice", "weights"} and sorted(mode["choice"]) == ["photo", "scan"]:
        weights = mode.get("weights") or [1.0, 1.0]
        total = float(sum(weights))
        return "mix", float(weights[mode["choice"].index("scan")]) / total if total > 0 else 0.5
    return "other", 0.5


def mix_mode(scan_share: float) -> dict[str, Any]:
    """A `mode` that makes `scan_share` of the images scans and the rest photos."""
    s = round(min(max(float(scan_share), 0.0), 1.0), 3)
    return {"choice": ["photo", "scan"], "weights": [round(1.0 - s, 3), s]}


def enum_items(path: tuple, key: str) -> list | None:
    """The allowed strings for the value at `path`, or None for free text."""
    return PATH_ENUMS.get(tuple(path)) or ENUMS.get(key)


def merged(user: dict[str, Any]) -> dict[str, Any]:
    """The user's config on top of the defaults, without expanding paths."""
    return _merge(copy.deepcopy(DEFAULTS), user)


# The keys the editor's Paper / Card switch sets. Their values come from the photo and card example configs.
LOOK_PATHS: list[tuple] = [
    ("page", "size_cm"), ("page", "lift_cm"),
    ("material", "tint"), ("material", "roughness"), ("material", "specular"), ("material", "grain"),
    ("material", "corner_radius_cm"), ("material", "glare"), ("material", "room_reflection"),
    ("material", "wear"), ("material", "hologram"),
    ("deform", "crumple", "strength"), ("deform", "crumple", "waviness_cm"), ("deform", "folds"),
    ("deform", "curl"), ("deform", "edge_lift_cm"),
    ("camera", "fit_page"), ("camera", "output_px"),
]


@lru_cache(maxsize=None)
def _example(name: str) -> str:
    return (resources.files("recapture") / "examples" / f"{name}.json").read_text()


def set_look(user: dict[str, Any], look: str, memory: dict[str, dict]) -> None:
    """Switch the config to the paper or the card look ("paper" | "card").

    The current values of LOOK_PATHS are kept in `memory` under the other
    look, so switching back restores them. A look not seen before takes its
    values from the photo or the card example."""
    other = "paper" if look == "card" else "card"
    current = merged(user)
    memory[other] = {path: copy.deepcopy(get(current, path)) for path in LOOK_PATHS}
    source = memory.get(look)
    if source is None:
        example = merged(json.loads(_example("card" if look == "card" else "photo")))
        source = {path: get(example, path) for path in LOOK_PATHS}
    for path in LOOK_PATHS:
        set_value(user, path, copy.deepcopy(source[path]))


def card_look(cfg: dict[str, Any]) -> bool:
    """True when a merged config turns on any card-look option (fixed above 0, or a random range)."""
    m = cfg["material"]
    values = [m["corner_radius_cm"], m["glare"]["strength"], m["hologram"], m["room_reflection"], m["wear"]]
    return any(spec_kind(v) or (isinstance(v, (int, float)) and v > 0) for v in values)


def get(cfg: Any, path: Path, default: Any = None) -> Any:
    node = cfg
    for key in path:
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return default
    return node


def set_value(user: dict[str, Any], path: Path, value: Any) -> None:
    """Set a value in the user's config, creating parents from the defaults.

    A random spec on the way is replaced, except a repeat spec: its "repeat"
    count and "item" can be set like keys of a plain object."""
    node: Any = user
    for i, key in enumerate(path[:-1]):
        nxt = path[i + 1]
        if isinstance(node, dict):
            kind = spec_kind(node[key]) if key in node else None
            if key not in node or not isinstance(node[key], (dict, list)) or (kind and kind != "repeat"):
                # Start from the default so sibling keys and list lengths stay right.
                base = get(DEFAULTS, path[: i + 1])
                # Dicts merge onto the defaults later, so they can start empty.
                node[key] = (copy.deepcopy(base) if isinstance(base, list) else []) if isinstance(nxt, int) else {}
            node = node[key]
        else:  # list
            node = node[key]
    last = path[-1]
    if isinstance(node, list) and isinstance(last, int):
        while len(node) <= last:
            node.append(None)
    node[last] = value


def to_random(value: Any) -> Any:
    """Turn a fixed value into a random spec around it."""
    if isinstance(value, bool):
        return {"choice": [False, True]}
    if isinstance(value, int):
        d = max(1, abs(value) // 5)
        return {"randint": [value - d, value + d]}
    if isinstance(value, float):
        d = max(abs(value) * 0.2, 0.1)
        return {"uniform": [round(value - d, 4), round(value + d, 4)]}
    if isinstance(value, str):
        return {"choice": [value]}
    return value


def to_fixed(spec: Any) -> Any:
    """A fixed value that stands for a random spec (its middle or first choice)."""
    kind = spec_kind(spec)
    if kind == "uniform":
        lo, hi = spec["uniform"]
        return round((lo + hi) / 2, 4)
    if kind == "randint":
        lo, hi = spec["randint"]
        return int((lo + hi) // 2)
    if kind == "normal":
        return spec["normal"][0]
    if kind == "choice":
        return copy.deepcopy(spec["choice"][0])
    if kind == "repeat":
        n = to_fixed(spec["repeat"]) if spec_kind(spec["repeat"]) else spec["repeat"]
        return [copy.deepcopy(spec["item"]) for _ in range(int(n))]
    return spec


def strip_defaults(user: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value equals the default (for a short saved file)."""
    def walk(node: Any, default: Any) -> Any:
        if isinstance(node, dict) and not spec_kind(node) and isinstance(default, dict):
            out = {}
            for k, v in node.items():
                if k in default:
                    w = walk(v, default[k])
                    if w is not _SAME:
                        out[k] = w
                else:
                    out[k] = v
            return out if out else _SAME
        return _SAME if node == default else node

    result = walk(user, DEFAULTS)
    return {} if result is _SAME else result


_SAME = object()


def _merge(base: dict, user: dict) -> dict:
    for k, v in user.items():
        if isinstance(v, dict) and not spec_kind(v) and isinstance(base.get(k), dict):
            base[k] = _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def format_config(obj: Any, indent: int = 0) -> str:
    """JSON with short values on one line, like the hand-written configs."""
    import json

    flat = json.dumps(obj, separators=(", ", ": "))
    if not isinstance(obj, dict) or spec_kind(obj) or len(flat) + indent <= 90 and not _has_dict_value(obj):
        if len(flat) + indent <= 100 or not isinstance(obj, (dict, list)):
            return flat
    pad = " " * (indent + 2)
    if isinstance(obj, list):
        items = [format_config(v, indent + 2) for v in obj]
        return "[\n" + ",\n".join(pad + i for i in items) + "\n" + " " * indent + "]"
    items = [f"{pad}{json.dumps(k)}: {format_config(v, indent + 2)}" for k, v in obj.items()]
    return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"


def _has_dict_value(obj: dict) -> bool:
    return any(isinstance(v, dict) and not spec_kind(v) for v in obj.values())


PATH_KEYS = [("background", "images"), ("deform", "crumple", "scan", "directory"), ("desk", "surface"),
             ("desk", "papers")]


def rebase_paths(user: dict[str, Any], old_base: Any, new_base: Any) -> dict[str, Any]:
    """Rewrite relative file paths so they stay valid from a new folder."""
    import os
    from pathlib import Path as FsPath

    out = copy.deepcopy(user)

    def fix(p: Any) -> Any:
        if isinstance(p, str) and not os.path.isabs(p) and not p.startswith("@assets/"):
            absolute = (FsPath(old_base) / p).resolve()
            return os.path.relpath(absolute, FsPath(new_base).resolve())
        if isinstance(p, list):
            return [fix(v) for v in p]
        if spec_kind(p) == "choice":
            return {**p, "choice": [fix(v) for v in p["choice"]]}
        return p

    for path in PATH_KEYS:
        value = get(out, path, _SAME)
        if value is not _SAME and value is not None:
            set_value(out, path, fix(value))
    return out


def orbit_lamp(azimuth: float, elevation: float, dx: float, dy: float) -> tuple[float, float]:
    """Lamp angles after a mouse drag of (dx, dy) pixels."""
    return (azimuth - dx * 0.5) % 360.0, min(max(elevation - dy * 0.3, 3.0), 89.0)


def orbit_camera(eye: list[float], target: list[float], dx: float, dy: float,
                 wheel: float) -> list[float]:
    """Camera position after a drag (orbit around the target) and a scroll (distance)."""
    import math

    v = [e - t for e, t in zip(eye, target)]
    r = math.sqrt(sum(c * c for c in v))
    az = math.atan2(v[1], v[0]) - math.radians(dx * 0.4)
    el = math.asin(max(-1.0, min(1.0, v[2] / r))) + math.radians(dy * 0.3)
    el = min(max(el, math.radians(10)), math.radians(89))
    r = min(max(r * (1.0 - 0.08 * wheel), 10.0), 300.0)
    return [t + r * c for t, c in zip(target, (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))]
