"""A quick look at a config: a few renders on one contact sheet, plus a text summary.

    recapture preview my.json -i docs/ -n 6 -o preview.jpg

Each tile is one image with its label quads drawn in red and a caption
(document, variant, photo or scan, size, boxes kept). The summary says the
same in text, so an agent that cannot see images can still check the run.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .pipeline import Job, Pipeline, job_seed

TILE_W = 420


def preview(cfg: dict[str, Any], docs: list[Path], count: int, out: Path, seed: int,
            log=print) -> list[dict[str, Any]]:
    """Render `count` images (cycling through `docs`, then variants) and write the sheet to `out`."""
    pipe = Pipeline(cfg)
    rows: list[dict[str, Any]] = []
    tiles: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="recapture-preview-") as tmp:
        for k in range(count):
            doc, variant = docs[k % len(docs)], k // len(docs)
            job = Job(source=doc, out_dir=Path(tmp), variant=variant, seed=job_seed(seed, doc.name, variant))
            t = time.perf_counter()
            res = pipe.run(job)
            seconds = time.perf_counter() - t
            boxes = res.labels["boxes"]
            kept = sum(1 for b in boxes if b.get("quad"))
            row = {"doc": doc.name, "variant": variant, "mode": res.resolved["mode"],
                   "size": res.labels["size"], "boxes": len(boxes), "kept": kept, "seconds": round(seconds, 2)}
            rows.append(row)
            log(f"{doc.name} v{variant}: {row['mode']}, {row['size'][0]}x{row['size'][1]}, "
                f"boxes {kept}/{len(boxes)}, {seconds:.2f} s")
            tiles.append(_tile(res.image_path, boxes, f"{doc.name} v{variant} · {row['mode']} · "
                                                      f"{kept}/{len(boxes)} boxes"))
    _sheet(tiles).save(out, quality=90)
    lost = sum(r["boxes"] - r["kept"] for r in rows)
    modes = {m: sum(r["mode"] == m for r in rows) for m in sorted({r["mode"] for r in rows})}
    log(f"{len(rows)} images ({', '.join(f'{n} {m}' for m, n in modes.items())}), "
        f"{lost} boxes lost, {sum(r['seconds'] for r in rows) / max(len(rows), 1):.2f} s per image; sheet: {out}")
    return rows


def _tile(path: Path, boxes: list[dict[str, Any]], caption: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    width = max(2, im.size[0] // 400)
    for b in boxes:
        if b.get("quad"):
            d.polygon([tuple(p) for p in b["quad"]], outline=(230, 30, 30), width=width)
    scale = TILE_W / im.size[0]
    im = im.resize((TILE_W, max(1, round(im.size[1] * scale))), Image.Resampling.LANCZOS)
    bar = Image.new("RGB", (TILE_W, 26), (255, 255, 210))
    ImageDraw.Draw(bar).text((6, 5), caption, fill=(0, 0, 0), font_size=13)
    out = Image.new("RGB", (TILE_W, im.size[1] + bar.size[1]), (255, 255, 255))
    out.paste(bar, (0, 0))
    out.paste(im, (0, bar.size[1]))
    return out


def _sheet(tiles: list[Image.Image], columns: int = 3, gap: int = 8) -> Image.Image:
    columns = min(columns, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    row_h = [max(t.size[1] for t in tiles[r * columns:(r + 1) * columns]) for r in range(rows)]
    sheet = Image.new("RGB", (columns * TILE_W + (columns - 1) * gap, sum(row_h) + (rows - 1) * gap), (255, 255, 255))
    y = 0
    for r in range(rows):
        for c, t in enumerate(tiles[r * columns:(r + 1) * columns]):
            sheet.paste(t, (c * (TILE_W + gap), y))
        y += row_h[r] + gap
    return sheet

