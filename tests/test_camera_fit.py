"""camera.fit_page: the page fills the frame with only a small margin."""

import numpy as np
import pytest

from recapture.config import build_config
from recapture.pipeline import sample_scene


def _edge_angles(px: np.ndarray) -> np.ndarray:
    """Angle of each page side on screen, left over after the nearest quarter turn."""
    n = len(px) // 4
    corners = px[::n][:4]
    d = np.roll(corners, -1, axis=0) - corners
    a = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return (a + 45.0) % 90.0 - 45.0


@pytest.mark.parametrize("seed", range(8))
def test_fit_page_fills_the_frame(seed):
    margin = 0.03
    cfg = build_config({
        "page": {"rotation_deg": {"uniform": [-15, 15]}, "position_cm": [{"uniform": [-5, 5]}, {"uniform": [-5, 5]}]},
        "deform": {"crumple": {"strength": 0.0}},
        "camera": {"position_cm": [{"uniform": [-4, 4]}, {"uniform": [-6, 2]}, {"uniform": [20, 60]}],
                   "target_cm": [{"uniform": [-5, 5]}, {"uniform": [-5, 5]}, 0],
                   "focal_mm": {"uniform": [18, 40]}, "roll_deg": 0.0, "output_px": [600, 800],
                   "fit_page": True, "fit_margin": margin},
        "background": {"enabled": False}, "output": {"format": "png"},
    })
    _, scene = sample_scene(cfg, (1000, 1400), np.random.default_rng(seed))
    cam = scene.camera
    px, _ = cam.project(scene.page.sample(scene.page.boundary_uv(64)))
    lo, hi = px.min(axis=0), px.max(axis=0)
    gaps = np.array([lo[0], lo[1], cam.width - hi[0], cam.height - hi[1]])
    m = margin * min(cam.width, cam.height)
    assert gaps.min() >= m - 1.0  # the whole page is in frame
    assert gaps.min() <= m * 1.1 + 1.0  # and tight on one side
    assert np.all(np.abs(_edge_angles(px)) < 8.0)  # the page is turned square to the frame
