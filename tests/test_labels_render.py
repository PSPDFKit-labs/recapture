"""Label check on real renders.

For each box, the page is rendered with only that box filled black on
white, and once more as a blank white page with the same scene. A pixel is
box ink where the box render is much darker than the blank render; shading
(folds, creases, the back of the paper) is the same in both. Every box ink
pixel must lie inside the box's enclosing quad.
"""

import numpy as np
import pytest

from recapture.config import build_config
from recapture.labels import SourceBox, transform_boxes
from recapture.labels.quad import points_in_convex_polygon
from recapture.pipeline import sample_scene

SEEDS = range(24)
SRC_W, SRC_H = 1000, 1400


@pytest.fixture(scope="module")
def renderer():
    from recapture.render import Renderer
    try:
        r = Renderer()
    except Exception as e:  # no GL available on this machine
        pytest.skip(f"no OpenGL context: {e}")
    yield r
    r.release()


def _boxes(rng):
    boxes = []
    y = 60
    while y < SRC_H - 120:
        h = min(int(rng.integers(10, 160)), SRC_H - 20 - y)
        x0 = int(rng.integers(20, 300))
        x1 = int(rng.integers(x0 + 10, SRC_W - 20))
        boxes.append(SourceBox(id=str(len(boxes)), label="b", quad=np.array(
            [[x0, y], [x1, y], [x1, y + h], [x0, y + h]], float), extra={}))
        y += h + int(rng.integers(20, 60))
    return boxes


def _page_with(box):
    img = np.full((SRC_H, SRC_W, 4), 255, np.uint8)
    (x0, y0), (x1, y1) = box.quad[0].astype(int), box.quad[2].astype(int)
    img[y0:y1, x0:x1, :3] = 0
    return img


DEFORM = {
    "crumple": {"strength": {"uniform": [0.0, 2.0]}, "creases": {"randint": [10, 80]},
                "waviness_cm": {"uniform": [0.0, 0.6]}},
    "folds": {"repeat": {"randint": [0, 3]}, "item": {
        "position": [{"uniform": [0.2, 0.8]}, {"uniform": [0.2, 0.8]}],
        "direction_deg": {"uniform": [0, 360]},
        "angle_deg": {"uniform": [-60, 60]},
    }},
    "curl": {"x_curvature": {"uniform": [-0.03, 0.03]}, "y_curvature": {"uniform": [-0.02, 0.02]}},
    "edge_lift_cm": [{"uniform": [0, 2]}, 0, {"uniform": [0, 2]}, 0],
}


def _luminance(out):
    return out.color @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _config(deform=None):
    return build_config({
        "deform": deform or {},
        "fx": {"enabled": False},
        # The label code does not depend on mesh density; a smaller mesh keeps the tests fast.
        "page": {"mesh_resolution": 256, "position_cm": [{"uniform": [-3, 3]}, {"uniform": [-3, 3]}],
                 "rotation_deg": {"uniform": [-40, 40]}},
        "light": {"intensity": 0.0, "ambient": 1.0, "ambient_color": [1, 1, 1]},
        "camera": {
            "position_cm": [{"uniform": [-15, 15]}, {"uniform": [-30, 5]}, {"uniform": [25, 50]}],
            "target_cm": [{"uniform": [-4, 4]}, {"uniform": [-4, 4]}, 0],
            "roll_deg": {"uniform": [-30, 30]},
            "focal_mm": {"uniform": [22, 40]},
            "output_px": [600, 800],
            # Allow the page to leave the frame, so some boxes are truncated.
            "frame_coverage": [0.05, 10.0],
            "frame_margin": -0.25,
        },
    })


@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "deformed"])
@pytest.mark.parametrize("seed", SEEDS)
def test_box_pixels_lie_inside_quads(renderer, seed, deformed):
    rng = np.random.default_rng(seed)
    boxes = _boxes(rng)
    _, scene = sample_scene(_config(DEFORM if deformed else None), (SRC_W, SRC_H), rng)
    out = renderer.render(scene, _page_with(boxes[0]))
    labels = transform_boxes(boxes, out.label, scene, (SRC_W, SRC_H))

    blank = _luminance(renderer.render(scene, np.full((SRC_H, SRC_W, 4), 255, np.uint8)))
    for box, lab in zip(boxes, labels):
        out = renderer.render(scene, _page_with(box))
        ink = _luminance(out)
        # Mostly covered by the box: at least half covered by the page, and
        # less than a third of the blank brightness.
        ys, xs = np.nonzero((out.alpha >= 0.5) & (blank > 0.02) & (ink < blank / 3))
        if lab["quad"] is None:
            assert len(xs) == 0 or not _center_in_box(out.label[ys, xs], box).any(), \
                f"box {box.id}: pixels visible but no quad"
            continue
        quad = np.array(lab["quad"])
        centers = np.stack([xs + 0.5, ys + 0.5], 1)
        outside = _distance_outside(centers, quad)
        # The promise: a pixel whose center shows the box is inside its quad.
        shows_box = _center_in_box(out.label[ys, xs], box)
        assert (outside[shows_box] <= 1e-3).all(), \
            f"box {box.id}: {int((outside[shows_box] > 1e-3).sum())} box pixels outside its quad"
        # A pixel that shows the box only in part (an edge seen almost
        # side-on) can sit just outside, by less than one pixel.
        assert outside.max(initial=0.0) < 1.0, f"box {box.id}: ink {outside.max():.2f} px outside its quad"


def _center_in_box(label: np.ndarray, box) -> np.ndarray:
    src = label[:, :2] * [SRC_W, SRC_H]
    (x0, y0), (x1, y1) = box.quad[0], box.quad[2]
    return (label[:, 2] > 0.5) & (src[:, 0] >= x0) & (src[:, 0] <= x1) & (src[:, 1] >= y0) & (src[:, 1] <= y1)


def _distance_outside(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """How far each point lies outside the convex quad, in px (0 inside)."""
    q = quad if _signed_area(quad) > 0 else quad[::-1]
    d = np.zeros(len(points))
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        n = np.array([b[1] - a[1], a[0] - b[0]]) / np.linalg.norm(b - a)  # outward normal (y down, clockwise)
        d = np.maximum(d, (points - a) @ n)
    return d


def _signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


@pytest.mark.parametrize("seed", SEEDS)
def test_flat_page_quad_matches_projected_corners(renderer, seed):
    rng = np.random.default_rng(seed)
    boxes = _boxes(rng)
    _, scene = sample_scene(_config(), (SRC_W, SRC_H), rng)
    out = renderer.render(scene, _page_with(boxes[0]))
    labels = transform_boxes(boxes, out.label, scene, (SRC_W, SRC_H))
    for lab in labels:
        if lab["truncated"] or lab["quad"] is None:
            continue
        quad, corners = np.array(lab["quad"]), np.array(lab["corners"])
        # A box that is fully in frame on a flat page has no cut edges, so
        # its quad is the exact projected box.
        assert np.abs(quad - corners).max() < 0.05, lab
        # Output coordinates are rounded to 0.01 px.
        assert points_in_convex_polygon(corners, quad, tol=0.02).all(), lab
        assert lab["visible"] > 0.95, lab
