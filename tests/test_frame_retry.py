"""Frame-check retries draw only the camera and the page placement again (and the folds late), so the rest of
a sample stays. No GL needed."""

import json
from importlib import resources

import numpy as np
import pytest

import recapture.pipeline as pl
from recapture.config import build_config, resolve_config
from recapture.config.schema import FOLD_PATHS, FRAME_PATHS, redraw_frame, resolve_with_base

DOC = (1240, 1754)


def _photo():
    return build_config(json.loads((resources.files("recapture") / "examples" / "photo.json").read_text()))


def _tries(monkeypatch):
    seen = []
    orig = pl.redraw_frame

    def spy(first, cfg, base, attempt, folds=False):
        seen.append((attempt, folds))
        return orig(first, cfg, base, attempt, folds)

    monkeypatch.setattr(pl, "redraw_frame", spy)
    return seen


def test_the_first_try_is_the_plain_sample():
    cfg = _photo()
    assert resolve_config(cfg, np.random.default_rng(7)) == resolve_with_base(
        cfg, int(np.random.default_rng(7).integers(0, 2**63 - 1)))


@pytest.mark.parametrize("folds", [False, True])
def test_a_retry_changes_only_the_frame_values(folds):
    cfg = _photo()
    first = resolve_with_base(cfg, 123)
    retry = redraw_frame(first, cfg, 123, 5, folds=folds)
    changed = FRAME_PATHS + (FOLD_PATHS if folds else ())
    assert retry["camera"] != first["camera"]
    for key in first:
        if key not in ("camera", "page") and not (folds and key == "deform"):
            assert retry[key] == first[key], key  # desk, lamp, background, effects...
    assert retry["page"]["size_cm"] == first["page"]["size_cm"]
    if not folds:
        assert retry["deform"] == first["deform"]
    else:
        assert retry["deform"]["crumple"] == first["deform"]["crumple"]
    assert ("camera",) in changed


def test_fixed_camera_page_and_folds_take_one_try(monkeypatch):
    seen = _tries(monkeypatch)
    user = json.loads((resources.files("recapture") / "examples" / "photo.json").read_text())
    user["camera"].update(position_cm=[0, 0, 60], target_cm=[0, 0, 0], roll_deg=0)  # too far: never fits
    user["page"].update(position_cm=[0, 0], rotation_deg=0)
    user["deform"]["folds"] = []
    with pytest.raises(Exception, match="in 1 tries"):
        pl.sample_scene(build_config(user), DOC, np.random.default_rng(0))
    assert seen == []


def test_late_retries_redraw_the_folds(monkeypatch):
    # Two crossing folds can push a flap past the page, and then no camera fits; late retries draw new folds.
    seen = _tries(monkeypatch)
    cfg = _photo()
    pl.sample_scene(cfg, DOC, np.random.default_rng(pl.job_seed(4242, "page.png", 3)))
    tries = int(cfg["camera"]["max_tries"])
    assert seen and all(f == (a >= tries // 2) for a, f in seen)
    assert seen[-1][1]  # this sample needed the fold stage to fit
