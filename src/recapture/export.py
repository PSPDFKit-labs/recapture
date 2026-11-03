"""Turn rendered label files into training formats.

    recapture export dataset/ -o export/ --format coco --val 0.1

Reads every `<name>.json` label file that has its image next to it, splits
the images into train, val and test by source document (all variants of one
document land in the same split, so a model never sees near-copies of its
test pages), and writes one of:

    coco      images/<split>/..., annotations/instances_<split>.json with a
              4-point polygon, a bounding box and the area per box
    yolo-obb  images/<split>/..., labels/<split>/<name>.txt with
              "class x1 y1 x2 y2 x3 y3 x4 y4" (0..1), and data.yaml

Boxes with no quad (cut out of the frame or hidden) are left out.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMATS = ("coco", "yolo-obb")
SPLITS = ("train", "val", "test")
UNLABELED = "unlabeled"


@dataclass
class Item:
    image: Path
    labels: dict[str, Any]

    @property
    def source(self) -> str:
        return str(self.labels.get("source") or self.image.stem)


def find_items(root: Path) -> list[Item]:
    """Label files with their image, in a stable order."""
    items = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".config.json"):
            continue
        try:
            labels = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(labels, dict) or "boxes" not in labels or not labels.get("image"):
            continue
        image = path.parent / labels["image"]
        if image.is_file():
            items.append(Item(image=image, labels=labels))
    return items


def split_of(source: str, val: float, test: float, seed: int) -> str:
    """The split a source document goes to: fixed per document and seed."""
    h = int.from_bytes(hashlib.blake2b(f"{seed}|{source}".encode(), digest_size=8).digest(), "big") / 2.0 ** 64
    return "test" if h < test else "val" if h < test + val else "train"


def export(root: Path, out: Path, fmt: str, val: float = 0.1, test: float = 0.0, seed: int = 0,
           classes: list[str] | None = None, link: bool = False) -> dict[str, Any]:
    """Write the dataset in `fmt`. Returns counts per split and the class list."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    if not 0 <= val < 1 or not 0 <= test < 1 or val + test >= 1:
        raise ValueError("val and test must be parts of the data, together below 1")
    items = find_items(root)
    if not items:
        raise ValueError(f"no label files with images found in {root}")
    names = classes or sorted({_label(b) for it in items for b in it.labels["boxes"] if b.get("quad")})
    index = {n: i for i, n in enumerate(names)}
    by_split: dict[str, list[Item]] = {s: [] for s in SPLITS}
    for it in items:
        by_split[split_of(it.source, val, test, seed)].append(it)

    counts: dict[str, dict[str, int]] = {}
    for split, group in by_split.items():
        if not group:
            continue
        img_dir = out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        names_used: set[str] = set()
        coco: dict[str, Any] = {"images": [], "annotations": [],
                                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
        n_boxes = skipped = 0
        for k, it in enumerate(group):
            name = _unique_name(it, root, names_used)
            _place(it.image, img_dir / name, link)
            w, h = (int(v) for v in it.labels["size"])
            lines = []
            for b in it.labels["boxes"]:
                quad, label = b.get("quad"), _label(b)
                if not quad or label not in index:
                    skipped += 1
                    continue
                n_boxes += 1
                if fmt == "yolo-obb":
                    pts = " ".join(f"{min(max(x / w, 0.0), 1.0):.6f} {min(max(y / h, 0.0), 1.0):.6f}" for x, y in quad)
                    lines.append(f"{index[label]} {pts}")
                else:
                    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
                    x0, y0 = max(min(xs), 0.0), max(min(ys), 0.0)
                    x1, y1 = min(max(xs), float(w)), min(max(ys), float(h))
                    coco["annotations"].append({
                        "id": len(coco["annotations"]) + 1, "image_id": k + 1, "category_id": index[label] + 1,
                        "segmentation": [[round(float(v), 2) for p in quad for v in p]],
                        "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                        "area": round(_polygon_area(quad), 2), "iscrowd": 0,
                        "attributes": {"box_id": b.get("id"), "visible": b.get("visible"),
                                       "truncated": b.get("truncated")},
                    })
            if fmt == "yolo-obb":
                lbl_dir = out / "labels" / split
                lbl_dir.mkdir(parents=True, exist_ok=True)
                (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            else:
                coco["images"].append({"id": k + 1, "file_name": name, "width": w, "height": h,
                                       "source": it.source})
        if fmt == "coco":
            ann = out / "annotations"
            ann.mkdir(parents=True, exist_ok=True)
            (ann / f"instances_{split}.json").write_text(json.dumps(coco))
        counts[split] = {"images": len(group), "boxes": n_boxes, "skipped_boxes": skipped,
                         "documents": len({it.source for it in group})}
    if fmt == "yolo-obb":
        yaml = [f"path: {out.resolve()}"] + [f"{s}: images/{s}" for s in counts] + ["names:"]
        yaml += [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
        (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    return {"splits": counts, "classes": names}


def _label(box: dict[str, Any]) -> str:
    return str(box.get("label") or UNLABELED)


def _unique_name(it: Item, root: Path, used: set[str]) -> str:
    """The image's file name, prefixed with its folder when two would clash."""
    name = it.image.name
    if name in used:
        rel = it.image.relative_to(root)
        name = "_".join(rel.parts)
    used.add(name)
    return name


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        dst.unlink()
    if link:
        try:
            dst.hardlink_to(src)
            return
        except OSError:  # another disk: copy instead
            pass
    shutil.copy2(src, dst)


def _polygon_area(quad: list[list[float]]) -> float:
    s = 0.0
    for (x0, y0), (x1, y1) in zip(quad, quad[1:] + quad[:1]):
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0
