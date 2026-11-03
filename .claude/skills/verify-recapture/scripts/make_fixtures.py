"""Draw the fixture pages and their box files (run once; the outputs are committed).

    .venv/bin/python .claude/skills/verify-recapture/scripts/make_fixtures.py

page.png: an A4 page with a heading, paragraphs, a table and a bar chart.
card.png: a fictional ID-1 card with a photo, fields and an MRZ.
Pillow's built-in font keeps them the same on any machine.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "fixtures"


def font(size):
    return ImageFont.load_default(size=size)


def page():
    w, h = 1240, 1754
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    boxes = []

    def box(label, xy):
        boxes.append({"id": f"b{len(boxes)}", "label": label, "bbox": [int(v) for v in xy]})

    d.rectangle((90, 60, 1150, 72), fill=(30, 70, 140))
    d.text((90, 90), "Quarterly report", font=font(56), fill=(20, 20, 20))
    box("title", (88, 88, 560, 152))
    rng = np.random.default_rng(3)
    words = "the of and to in is was for on that with as by at from this be are an have it".split()
    y = 190
    for _ in range(3):
        top = y
        for _ in range(int(rng.integers(4, 7))):
            line = " ".join(rng.choice(words, size=int(rng.integers(11, 15))))
            d.text((90, y), line.capitalize(), font=font(26), fill=(40, 40, 40))
            y += 36
        box("text", (88, top - 2, 1150, y))
        y += 34
    # A table: 5 rows, 4 columns.
    top = y
    for r in range(6):
        d.line((90, y + r * 48, 1150, y + r * 48), fill=(80, 80, 80), width=2)
        for c, x in enumerate((100, 400, 660, 920)):
            if r < 5:
                d.text((x, y + r * 48 + 12), "Region" if r == 0 and c == 0 else f"{int(rng.integers(10, 999))}",
                       font=font(24), fill=(30, 30, 30))
    box("table", (88, top - 2, 1152, top + 5 * 48 + 2))
    y = top + 5 * 48 + 60
    # A bar chart.
    top = y
    d.line((140, y + 420, 1100, y + 420), fill=(60, 60, 60), width=3)
    for i in range(8):
        bh = int(rng.integers(60, 400))
        x = 180 + i * 115
        d.rectangle((x, y + 420 - bh, x + 70, y + 420), fill=(40, 120, 60) if i % 2 else (30, 70, 140))
    box("chart", (130, top, 1110, top + 430))
    d.text((90, top + 450), "Figure 1: sales per region", font=font(24), fill=(60, 60, 60))
    box("caption", (88, top + 448, 440, top + 480))
    d.text((90, h - 90), "Page 4", font=font(22), fill=(110, 110, 110))
    box("footer", (88, h - 92, 170, h - 64))
    return img, boxes


def card():
    w, h = 1712, 1080  # ID-1 at about 508 dpi
    img = Image.new("RGB", (w, h), (226, 236, 244))
    d = ImageDraw.Draw(img)
    boxes = []

    def box(label, xy):
        boxes.append({"id": f"b{len(boxes)}", "label": label, "bbox": [int(v) for v in xy]})

    for i in range(0, w, 14):
        d.arc([i - 600, 200, i + 600, 1400], 200, 340, fill=(205, 220, 235), width=2)
    d.rectangle([0, 0, w, 130], fill=(24, 64, 120))
    d.text((60, 32), "UTOPIA  IDENTITY CARD", font=font(62), fill="white")
    box("header", (50, 25, 900, 105))
    d.text((1350, 42), "SPECIMEN", font=font(44), fill=(255, 210, 90))
    d.rectangle([70, 190, 470, 690], fill=(180, 190, 205), outline=(120, 130, 150), width=3)
    d.ellipse([170, 250, 370, 470], fill=(150, 160, 178))
    d.pieslice([110, 450, 430, 790], 180, 360, fill=(150, 160, 178))
    box("photo", (70, 190, 470, 690))
    y = 190
    for label, value in [("Surname", "ERIKSSON"), ("Given names", "ANNA MARIA"), ("Date of birth", "12 08 1974"),
                         ("Document no.", "D23145890"), ("Expiry", "15 04 2031")]:
        d.text((540, y), label, font=font(30), fill=(70, 80, 100))
        d.text((540, y + 36), value, font=font(44), fill=(20, 25, 35))
        box("field", (535, y - 4, 545 + 12 + 26 * len(value), y + 92))
        y += 104
    d.rectangle([0, 760, w, h], fill=(240, 244, 248))
    for i, line in enumerate(["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F3104159UTO<<<<<<<<<<<6",
                              "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]):
        d.text((60, 790 + i * 92), line, font=font(58), fill=(15, 15, 20))
    box("mrz", (50, 785, 1560, 1055))
    return img, boxes


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, make in (("page", page), ("card", card)):
        img, boxes = make()
        img.save(OUT / f"{name}.png", optimize=True)
        (OUT / f"{name}.json").write_text(json.dumps({"boxes": boxes}, indent=1) + "\n")
        print(f"{name}.png: {img.size[0]}x{img.size[1]}, {len(boxes)} boxes")
