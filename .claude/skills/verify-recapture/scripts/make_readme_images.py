"""Make the README header image (docs/header.jpg) from the fixture page.

    .venv/bin/python .claude/skills/verify-recapture/scripts/make_readme_images.py [--out docs/header.jpg]
        [--photo IMAGE] [--scan IMAGE]

Three panels on a light background: the clean page with its boxes filled, an arrow, then a photo and a scan
with the quads from their label files. One color per label.

With no --photo or --scan, it renders them with the working tree's source into a run folder:
- photo: the photo example with a higher camera (31 to 34 cm, so the desk shows), crumple 0.5 to 0.9,
  3 or 4 desk items, less defocus and no motion blur; seed 7, variant 10.
- scan: the scan example, seed 7, variant 2 (gray, 0.6 deg skew).
--photo and --scan take a rendered image instead; its label file sits next to it with the same stem.

The README's editor screenshot (docs/editor.jpg) comes from editor_shot.py:

    editor_shot.py photo --open Paper Lamp Camera --name readme-editor

then a resize to 1504 px wide and JPEG quality 92.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

PHOTO_SEED, PHOTO_VARIANT = 7, 10
SCAN_SEED, SCAN_VARIANT = 7, 2

W, H = 2400, 1040
BG = (245, 244, 240)
INK = (40, 40, 44)
PANEL_TOP, PANEL_H = 70, 860
CAPTION_Y = 985
COLORS = {
    "title": (220, 50, 47),
    "text": (38, 110, 220),
    "table": (140, 70, 190),
    "chart": (40, 160, 70),
    "caption": (240, 140, 20),
    "footer": (110, 110, 110),
}
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]
SS = 2  # draw boxes at twice the panel size, then shrink, for smooth lines


def photo_config(dest: Path) -> Path:
    cfg = json.loads((lib.root() / "src" / "recapture" / "examples" / "photo.json").read_text())
    cfg["camera"]["position_cm"][2] = {"uniform": [31, 34]}
    cfg["deform"]["crumple"]["strength"] = {"uniform": [0.5, 0.9]}
    cfg["desk"]["items"]["repeat"] = {"randint": [3, 4]}
    cfg["fx"]["defocus"]["strength_px"] = {"uniform": [0.3, 1.2]}
    cfg["fx"]["motion_blur"]["enabled"] = False
    path = dest / "photo_readme.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def render(config: Path, pages: Path, out: Path, seed: int, variant: int) -> Path:
    """Render variants 0..variant (each variant's seed depends only on seed, page and variant)."""
    res = subprocess.run([lib.python(), "-m", "recapture.cli", "render", "-i", str(pages), "-c", str(config),
                          "-o", str(out), "-n", str(variant + 1), "--seed", str(seed), "--format", "png"],
                         env=lib.env_for(lib.root()), capture_output=True, text=True)
    if res.returncode:
        sys.exit(res.stdout + res.stderr)
    return out / f"page_{variant:03d}.png"


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def fit(img: Image.Image, height: int) -> Image.Image:
    return img.convert("RGB").resize((round(img.width * height / img.height), height), Image.LANCZOS)


def clean_panel(page: Path, boxes: list[dict]) -> Image.Image:
    img = Image.open(page).convert("RGBA")
    s = PANEL_H * SS / img.height
    img = img.resize((round(img.width * s), PANEL_H * SS), Image.LANCZOS)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for b in boxes:
        x0, y0, x1, y1 = (v * s for v in b["bbox"])
        c = COLORS.get(b["label"], (0, 0, 0))
        d.rectangle((x0, y0, x1, y1), fill=c + (46,), outline=c + (255,), width=2 * SS)
    img = Image.alpha_composite(img, over)
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS).convert("RGB")


def quad_panel(image: Path) -> Image.Image:
    lab = json.loads(image.with_suffix(".json").read_text())
    img = Image.open(image).convert("RGB")
    s = PANEL_H * SS / img.height
    img = img.resize((round(img.width * s), PANEL_H * SS), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for b in lab["boxes"]:
        pts = [(x * s, y * s) for x, y in b["quad"]]
        d.line(pts + pts[:1], fill=COLORS.get(b["label"], (0, 0, 0)), width=2 * SS, joint="curve")
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


def compose(panels: list[Image.Image], captions: list[str]) -> Image.Image:
    left, photo, scan = panels
    arrow_w, gap = 150, 60
    total = left.width + arrow_w + photo.width + gap + scan.width
    x0 = (W - total) // 2
    xs = [x0, x0 + left.width + arrow_w, x0 + left.width + arrow_w + photo.width + gap]

    canvas = Image.new("RGB", (W, H), BG)
    shadow = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shadow)
    for x, p in zip(xs, panels):
        sd.rectangle((x + 4, PANEL_TOP + 10, x + p.width + 4, PANEL_TOP + p.height + 10), fill=70)
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), canvas, shadow)
    for x, p in zip(xs, panels):
        canvas.paste(p, (x, PANEL_TOP))

    d = ImageDraw.Draw(canvas)
    ay = PANEL_TOP + PANEL_H // 2
    ax0, ax1 = xs[0] + left.width + 38, xs[1] - 38
    d.line((ax0, ay, ax1 - 12, ay), fill=INK, width=8)
    d.polygon([(ax1, ay), (ax1 - 30, ay - 22), (ax1 - 30, ay + 22)], fill=INK)

    f = font(40)
    for x, p, text in zip(xs, panels, captions):
        tw = d.textlength(text, font=f)
        d.text((x + (p.width - tw) / 2, CAPTION_Y), text, font=f, fill=INK, anchor="ls")
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=lib.root() / "docs" / "header.jpg")
    ap.add_argument("--photo", type=Path, help="a rendered photo (its label file next to it)")
    ap.add_argument("--scan", type=Path, help="a rendered scan (its label file next to it)")
    ap.add_argument("--assets", type=Path, help="asset folder (default: $RECAPTURE_ASSETS or assets/)")
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()
    if args.assets:
        os.environ["RECAPTURE_ASSETS"] = str(args.assets.resolve())

    page = lib.FIXTURES / "page.png"
    boxes = json.loads(page.with_suffix(".json").read_text())["boxes"]
    photo, scan = args.photo, args.scan
    if photo is None or scan is None:
        run = lib.run_dir("readme-images")
        pages = lib.stage_pages(run / "pages", [page])
        if photo is None:
            photo = render(photo_config(run), pages, run / "photo", PHOTO_SEED, PHOTO_VARIANT)
        if scan is None:
            example = lib.root() / "src" / "recapture" / "examples" / "scan.json"
            scan = render(example, pages, run / "scan", SCAN_SEED, SCAN_VARIANT)
        print(run)

    for img in (photo, scan):
        kept = len(json.loads(img.with_suffix(".json").read_text())["boxes"])
        if kept != len(boxes):
            print(f"warning: {img.name} keeps {kept} of {len(boxes)} boxes", file=sys.stderr)

    panels = [clean_panel(page, boxes), quad_panel(photo), quad_panel(scan)]
    out = compose(panels, ["Clean page + boxes", "Photo mode", "Scan mode"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, quality=args.quality, optimize=True, progressive=True)
    print(args.out, f"{args.out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
