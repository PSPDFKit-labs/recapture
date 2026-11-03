"""Sidecar box files.

Input format (one JSON file next to each source image):
    {"boxes": [
        {"id": "p1", "label": "paragraph", "bbox": [x_min, y_min, x_max, y_max]},
        {"id": "f1", "label": "figure", "quad": [[x, y], [x, y], [x, y], [x, y]]}
    ]}
Coordinates are source image pixels, origin at the top-left corner.
A quad lists its corners as top-left, top-right, bottom-right, bottom-left.

A document-analysis API response works too: an object with
`output.elements`, each with 4 corner `points` (or `bounds`), an `id`, a
`role` or `type` (the label), and `text`. Coordinates are scaled from the
element's `page` size to the image size when they differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class BoxFileError(ValueError):
    pass


@dataclass(frozen=True)
class SourceBox:
    id: str
    label: str
    quad: np.ndarray  # (4, 2) source pixels: TL, TR, BR, BL
    extra: dict[str, Any]  # other keys, copied to the output as they are


def sidecar_path(image_path: str | Path) -> Path:
    return Path(image_path).with_suffix(".json")


def load_boxes(path: str | Path, image_size: tuple[int, int] | None = None) -> list[SourceBox]:
    """Boxes from a sidecar file. `image_size` (w, h) scales API responses
    whose page size differs from the image."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise BoxFileError(f"{path}: invalid JSON: {e}") from e
    if isinstance(data, dict) and isinstance(data.get("output"), dict) and "elements" in data["output"]:
        return _api_elements(path, data["output"]["elements"], image_size)
    items = data.get("boxes") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise BoxFileError(f"{path}: needs an object with a 'boxes' list")
    boxes = []
    for n, item in enumerate(items):
        where = f"{path}: boxes[{n}]"
        if "bbox" in item:
            x0, y0, x1, y1 = (float(v) for v in item["bbox"])
            if x1 <= x0 or y1 <= y0:
                raise BoxFileError(f"{where}: bbox must be [x_min, y_min, x_max, y_max] with max > min")
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        elif "quad" in item:
            quad = np.asarray(item["quad"], dtype=np.float64)
            if quad.shape != (4, 2):
                raise BoxFileError(f"{where}: quad must be 4 [x, y] points")
        else:
            raise BoxFileError(f"{where}: needs 'bbox' or 'quad'")
        extra = {k: v for k, v in item.items() if k not in ("id", "label", "bbox", "quad")}
        boxes.append(SourceBox(id=str(item.get("id", n)), label=str(item.get("label", "")), quad=quad, extra=extra))
    return boxes


_API_KEEP = ("type", "role", "text", "readingOrder", "confidence")


def _api_elements(path: Path, elements: list, image_size: tuple[int, int] | None) -> list[SourceBox]:
    """Layout elements of a document-analysis API response, one box each."""
    pages = {e.get("page", {}).get("pageIndex", 0) for e in elements}
    if len(pages) > 1:
        raise BoxFileError(f"{path}: elements from {len(pages)} pages; one image needs the elements of one page")
    boxes = []
    for n, e in enumerate(elements):
        where = f"{path}: output.elements[{n}]"
        if "points" in e and len(e["points"]) == 4:
            quad = np.asarray(e["points"], dtype=np.float64)
        elif "bounds" in e:
            b = e["bounds"]
            x, y, w, h = (float(b[k]) for k in ("x", "y", "width", "height"))
            quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        else:
            raise BoxFileError(f"{where}: needs 'points' or 'bounds'")
        page = e.get("page") or {}
        if image_size and page.get("width") and page.get("height"):
            quad = quad * [image_size[0] / float(page["width"]), image_size[1] / float(page["height"])]
        label = e.get("role") or e.get("type") or ""
        extra = {k: e[k] for k in _API_KEEP if k in e}
        boxes.append(SourceBox(id=str(e.get("id", n)), label=str(label), quad=quad, extra=extra))
    return boxes
