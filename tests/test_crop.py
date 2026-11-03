"""crop: the scanner-app crop maps the page corners to the image corners."""

import numpy as np
import pytest

from recapture.config import build_config
from recapture.crop import apply_h, crop_transform, homography, warp_image, warp_labels
from recapture.labels import page_corners
from recapture.pipeline import sample_scene


def _scene(seed, **crop):
    cfg = build_config({
        "page": {"rotation_deg": {"uniform": [-15, 15]}},
        "deform": {"crumple": {"strength": 0.0}},
        "camera": {"position_cm": [{"uniform": [-4, 4]}, {"uniform": [-8, 2]}, {"uniform": [25, 50]}],
                   "output_px": [600, 800]},
        "crop": {"enabled": True, **crop},
    })
    return sample_scene(cfg, (1000, 1400), np.random.default_rng(seed))


def test_homography_maps_the_points():
    src = np.array([[10, 20], [300, 40], [280, 400], [30, 380]], float)
    dst = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], float)
    assert np.allclose(apply_h(homography(src, dst), src), dst, atol=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_page_corners_go_to_the_image_corners(seed):
    resolved, scene = _scene(seed)
    h, size = crop_transform(scene, resolved["crop"], 0)
    assert size[1] == 800 and abs(size[0] / size[1] - 1000 / 1400) < 0.01
    corners = apply_h(h, page_corners(scene))
    assert np.allclose(corners, [[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], atol=0.01)


def test_warp_moves_a_dot_where_the_homography_says():
    image = np.zeros((200, 300, 3), np.uint8)
    image[99:102, 149:152] = 255  # a dot at (150.5, 100.5)
    src = np.array([[50, 30], [260, 50], [250, 180], [40, 170]], float)
    h = homography(src, np.array([[0, 0], [400, 0], [400, 300], [0, 300]], float))
    out = warp_image(image, h, (400, 300)).astype(float).sum(axis=2)
    ys, xs = np.nonzero(out > out.max() * 0.5)
    expect = apply_h(h, [[150.5, 100.5]])[0]
    assert abs(xs.mean() + 0.5 - expect[0]) < 0.35 and abs(ys.mean() + 0.5 - expect[1]) < 0.35


def test_a_quad_that_leaves_the_crop_is_cut_and_marked():
    h = np.eye(3)
    boxes = [{"id": "a", "label": "x", "quad": [[-50, 10], [50, 10], [50, 30], [-50, 30]],
              "corners": [[-50, 10], [50, 10], [50, 30], [-50, 30]], "visible": 1.0, "truncated": False},
             {"id": "b", "label": "x", "quad": [[10, 10], [20, 10], [20, 20], [10, 20]],
              "corners": [[10, 10], [20, 10], [20, 20], [10, 20]], "visible": 1.0, "truncated": False}]
    out, _ = warp_labels(boxes, None, h, (100, 100))
    q = np.array(out[0]["quad"])
    assert q[:, 0].min() >= -1e-6 and out[0]["truncated"] and out[0]["visible"] == pytest.approx(0.5)
    assert out[1]["quad"] == boxes[1]["quad"] and not out[1]["truncated"]
