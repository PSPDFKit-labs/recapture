"""Value specs: a config value is either a plain value or a random spec.

Spec forms:
    {"uniform": [lo, hi]}                      float in [lo, hi)
    {"randint": [lo, hi]}                      int in [lo, hi], both ends included
    {"normal": [mean, std], "clip": [lo, hi]}  "clip" is optional
    {"choice": [a, b, ...], "weights": [...]}  "weights" is optional
    {"repeat": n, "item": template}            a list of n items; each item
                                               is resolved on its own

A chosen value can itself hold specs, so they are resolved again.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

SPEC_KEYS = {
    "uniform": {"uniform"},
    "randint": {"randint"},
    "normal": {"normal", "clip"},
    "choice": {"choice", "weights"},
    "repeat": {"repeat", "item"},
}


class SpecError(ValueError):
    pass


def spec_kind(value: Any) -> str | None:
    """Return the spec kind if `value` is a random spec, else None."""
    if not isinstance(value, dict):
        return None
    keys = set(value)
    for kind, allowed in SPEC_KEYS.items():
        if kind in keys and keys <= allowed:
            return kind
    return None


def check_spec(value: Any, path: str) -> None:
    kind = spec_kind(value)
    if kind is None:
        return
    arg = value[kind]
    if kind in ("uniform", "randint", "normal"):
        if not (isinstance(arg, list) and len(arg) == 2 and all(_is_number(a) for a in arg)):
            raise SpecError(f"{path}: '{kind}' needs a list of 2 numbers, got {arg!r}")
        if kind != "normal" and arg[0] > arg[1]:
            raise SpecError(f"{path}: '{kind}' low end {arg[0]} is above high end {arg[1]}")
        if kind == "normal" and arg[1] < 0:
            raise SpecError(f"{path}: 'normal' std must be >= 0")
        clip = value.get("clip")
        if clip is not None and not (isinstance(clip, list) and len(clip) == 2 and clip[0] <= clip[1]):
            raise SpecError(f"{path}: 'clip' needs [lo, hi] with lo <= hi")
    elif kind == "repeat":
        if "item" not in value:
            raise SpecError(f"{path}: 'repeat' needs an 'item' template")
        check_tree(arg, f"{path}.repeat")
        check_tree(value["item"], f"{path}.item")
    elif kind == "choice":
        if not (isinstance(arg, list) and arg):
            raise SpecError(f"{path}: 'choice' needs a non-empty list")
        weights = value.get("weights")
        if weights is not None:
            if len(weights) != len(arg) or any(w < 0 for w in weights) or sum(weights) <= 0:
                raise SpecError(f"{path}: 'weights' must match 'choice' in length and be positive")
        for i, item in enumerate(arg):
            check_tree(item, f"{path}.choice[{i}]")


def check_tree(value: Any, path: str = "$") -> None:
    """Check every spec inside a nested value."""
    if spec_kind(value):
        check_spec(value, path)
    elif isinstance(value, dict):
        for k, v in value.items():
            check_tree(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            check_tree(v, f"{path}[{i}]")


def resolve(value: Any, rng: np.random.Generator) -> Any:
    """Replace every spec inside `value` with a sampled plain value.

    Each spec draws from its own random stream, keyed by its place in the
    tree. So making one value fixed, or changing its range, leaves every
    other drawn value as it was.
    """
    return resolve_from(value, int(rng.integers(0, 2**63 - 1)))


def resolve_from(value: Any, base: int) -> Any:
    """`resolve` with the streams keyed by `base`."""
    return _resolve(value, base, ())


def resolve_at(value: Any, base: int, path: tuple) -> Any:
    """`resolve_from` for the part of a config at `path` (its streams keep their full keys)."""
    return _resolve(value, base, tuple(path))


def stream(base: int, path: tuple) -> np.random.Generator:
    """The random stream for the value at `path`."""
    key = int.from_bytes(hashlib.blake2b(repr(path).encode(), digest_size=8).digest(), "little")
    return np.random.default_rng([base, key])


def _resolve(value: Any, base: int, path: tuple) -> Any:
    kind = spec_kind(value)
    if kind in ("uniform", "randint", "normal"):
        rng = stream(base, path)
        if kind == "uniform":
            lo, hi = value["uniform"]
            return float(rng.uniform(lo, hi))
        if kind == "randint":
            lo, hi = value["randint"]
            return int(rng.integers(int(lo), int(hi) + 1))
        mean, std = value["normal"]
        x = float(rng.normal(mean, std))
        clip = value.get("clip")
        return float(np.clip(x, clip[0], clip[1])) if clip else x
    if kind == "repeat":
        n = _resolve(value["repeat"], base, path + ("repeat",))
        if not isinstance(n, (int, float)) or n < 0:
            raise SpecError(f"'repeat' count must be a number >= 0, got {n!r}")
        return [_resolve(value["item"], base, path + (i,)) for i in range(int(n))]
    if kind == "choice":
        items = value["choice"]
        weights = value.get("weights")
        p = None if weights is None else np.asarray(weights, float) / float(sum(weights))
        k = int(stream(base, path).choice(len(items), p=p))
        return _resolve(items[k], base, path + ("choice",))
    if isinstance(value, dict):
        return {k: _resolve(v, base, path + (k,)) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, base, path + (i,)) for i, v in enumerate(value)]
    return value


def is_random(value: Any) -> bool:
    """True if `value` holds at least one spec."""
    if spec_kind(value):
        return True
    if isinstance(value, dict):
        return any(is_random(v) for v in value.values())
    if isinstance(value, list):
        return any(is_random(v) for v in value)
    return False


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)
