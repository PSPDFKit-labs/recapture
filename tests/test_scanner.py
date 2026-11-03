"""Scan mode: a flatbed scanner pass, with labels that follow the skew and the dpi."""

import numpy as np
import pytest

from recapture.config import build_config, resolve_config
from recapture.labels import SourceBox
from recapture.labels.quad import points_in_convex_polygon
from recapture.scanner import scan_document

W, H = 600, 849  # A4 shape
PAGE = [21.0, 29.7]


def _doc_and_boxes(rng):
    doc = np.full((H, W, 4), 255, np.uint8)
    boxes = []
    for k in range(6):
        x0, y0 = int(rng.integers(20, W - 200)), int(rng.integers(20, H - 100))
        x1, y1 = x0 + int(rng.integers(30, 180)), y0 + int(rng.integers(10, 80))
        doc[y0:y1, x0:x1, :3] = 0
        boxes.append(SourceBox(id=str(k), label="b", quad=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float),
                               extra={}))
    return doc, boxes


def _scanner(seed, **over):
    cfg = build_config({"mode": "scan", "scanner": over})
    return resolve_config(cfg, np.random.default_rng(seed))["scanner"]


def _printer(**over):
    return {"ink_bleed": 0.0, "broken_strokes": 0.0, "ghost": 0.0, "toner_streaks": 0, "toner_specks": 0, **over}


def test_same_seed_same_page():
    doc, boxes = _doc_and_boxes(np.random.default_rng(0))
    over = dict(skew_deg=1.0, noise=3.0, streaks=2, dust=30)
    a = scan_document(doc, boxes, _scanner(3, **over), PAGE)
    b = scan_document(doc, boxes, _scanner(3, **over), PAGE)
    assert np.array_equal(a[0], b[0]) and a[3] == b[3]


@pytest.mark.parametrize("dpi", [100, 150, 300])
def test_output_size_follows_the_dpi(dpi):
    doc, boxes = _doc_and_boxes(np.random.default_rng(1))
    image, size, corners, _ = scan_document(doc, boxes, _scanner(1, dpi=dpi), PAGE)
    assert image.shape[1::-1] == size
    w_cm, h_cm = 29.7 * W / H, 29.7  # the long side keeps page.size_cm, the short side follows the image
    assert abs(size[0] - w_cm / 2.54 * dpi) <= 1 and abs(size[1] - h_cm / 2.54 * dpi) <= 1
    assert np.allclose(corners, [[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], atol=0.02)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dpi", [120, 300])
def test_labels_hold_their_ink(seed, dpi):
    """No noise or dust, so every dark pixel is box ink: each must lie in its box's quad."""
    doc, boxes = _doc_and_boxes(np.random.default_rng(100 + seed))
    skew = float(np.random.default_rng(seed).uniform(-4, 4))
    image, size, _, labels = scan_document(doc, boxes, _scanner(seed, dpi=dpi, skew_deg=skew, noise=0.0), PAGE)
    ys, xs = np.nonzero(image.astype(float).mean(axis=2) < 128)
    pts = np.stack([xs + 0.5, ys + 0.5], 1)
    inside = np.zeros(len(pts), bool)
    for b in labels:
        if b["quad"]:
            inside |= points_in_convex_polygon(pts, np.array(b["quad"]), tol=1.0)
    assert inside.mean() > 0.999


def test_color_modes():
    doc, boxes = _doc_and_boxes(np.random.default_rng(2))
    doc[..., 0] = np.where(doc[..., 0] > 0, 200, 0)  # a tinted page, so color differs from gray
    gray, *_ = scan_document(doc, boxes, _scanner(2, color="gray", noise=0.0), PAGE)
    bw, *_ = scan_document(doc, boxes, _scanner(2, color="bw", noise=2.0), PAGE)
    assert np.array_equal(gray[..., 0], gray[..., 1]) and np.array_equal(gray[..., 1], gray[..., 2])
    assert set(np.unique(bw)) <= {0, 255}


def test_printer_marks():
    doc, boxes = _doc_and_boxes(np.random.default_rng(3))
    sc = _scanner(3, noise=0.0, blur_px=0.0, dpi=2.54 * H / 29.7)  # one scan pixel per document pixel
    clean, *_ = scan_document(doc, boxes, sc, PAGE, _printer())
    assert np.array_equal(clean, scan_document(doc, boxes, sc, PAGE)[0])  # all zero = no printer step
    ink = clean.mean(axis=2) < 128
    broken, *_ = scan_document(doc, boxes, sc, PAGE, _printer(broken_strokes=0.05))
    assert (broken.mean(axis=2) < 128).sum() < ink.sum()  # gaps in the ink
    specks, *_ = scan_document(doc, boxes, sc, PAGE, _printer(toner_specks=400))
    assert ((specks.mean(axis=2) < 200) & ~ink).sum() > 50  # dark dots on the paper
    ghost, *_ = scan_document(doc, boxes, sc, PAGE, _printer(ghost=0.3))
    assert ((ghost.mean(axis=2) < 250) & ~ink).sum() > ((clean.mean(axis=2) < 250) & ~ink).sum()
    a = scan_document(doc, boxes, sc, PAGE, _printer(toner_streaks=3, toner_specks=100, ink_bleed=0.5))
    b = scan_document(doc, boxes, sc, PAGE, _printer(toner_streaks=3, toner_specks=100, ink_bleed=0.5))
    assert np.array_equal(a[0], b[0])


def test_bold_thickens_but_keeps_text():
    doc, boxes = _doc_and_boxes(np.random.default_rng(4))
    doc[100:104, 50:300, :3] = 0  # a thin line, 4 px
    ink = []
    for bold in (0.0, 0.5, 1.0):
        img, *_ = scan_document(doc, boxes, _scanner(4, dpi=2.54 * H / 29.7, noise=0.0, blur_px=0.0, bold=bold), PAGE)
        ink.append((img.mean(axis=2) < 128).sum())
    assert ink[0] < ink[1] < ink[2] < ink[0] * 1.3  # thicker, by a pixel or so, not blobs


def test_dark_edges_and_dots_stay_at_the_sides():
    doc, boxes = _doc_and_boxes(np.random.default_rng(5))
    doc[..., :3] = 255  # a blank page: every dark pixel comes from the edges
    img, size, _, labels = scan_document(doc, [], _scanner(5, noise=0.0, dark_edges=[0.05, 0.03, 0.0, 0.0],
                                                           edge_dots=3000), PAGE)
    dark = img.mean(axis=2) < 128
    w = size[0]
    assert dark[:, : int(0.03 * w)].mean() > 0.9 and dark[:, -int(0.015 * w):].mean() > 0.9  # the bands
    assert dark[:, int(0.35 * w): int(0.65 * w)].mean() < 0.001  # the middle stays clean
    near = dark[:, int(0.07 * w): int(0.15 * w)].mean()
    far = dark[:, int(0.25 * w): int(0.33 * w)].mean()
    assert near > far  # the dots thin out toward the page
