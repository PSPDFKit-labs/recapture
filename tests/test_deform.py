import numpy as np
import pytest

from recapture.config import ConfigError, build_config, resolve_config
from recapture.scene.deform import deform_page
from recapture.scene.mesh import make_page_mesh

W, H = 21.0, 29.7


def _deform(deform, res=128):
    mesh = make_page_mesh(W, H, res)
    cfg = resolve_config(build_config({"deform": deform}), np.random.default_rng(0))
    return mesh, deform_page(mesh, cfg["deform"])


def _stretch(mesh, out):
    """Largest relative change of a grid edge length."""
    def lengths(p):
        g = p.astype(np.float64).reshape(mesh.nv, mesh.nu, 3)
        return np.concatenate([np.linalg.norm(np.diff(g, axis=a), axis=2).ravel() for a in (0, 1)])
    return np.abs(lengths(out.pos) / lengths(mesh.pos) - 1).max()


def test_flat_config_keeps_page_flat():
    mesh, out = _deform({})
    np.testing.assert_allclose(out.pos, mesh.pos, atol=1e-6)


def _scan_dir(tmp_path):
    from PIL import Image
    d = tmp_path / "Scan"
    d.mkdir()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)).save(d / "Scan_Displacement.png")
    return str(d)


@pytest.mark.parametrize("mode", ["procedural", "scan"])
def test_crumple_strength_zero_is_flat_and_strength_scales(mode, tmp_path):
    crumple = {"mode": mode, "waviness_cm": 0.5, "seed": 3, "scan": {"directory": _scan_dir(tmp_path)}}
    mesh, flat = _deform({"crumple": {**crumple, "strength": 0.0}})
    np.testing.assert_allclose(flat.pos, mesh.pos, atol=1e-6)
    heights = [np.ptp(_deform({"crumple": {**crumple, "strength": k}})[1].pos[:, 2]) for k in (0.5, 1.0, 2.0)]
    assert 0 < heights[0] < heights[1] < heights[2]


@pytest.mark.parametrize("fold, lift", [
    ({"position": [0.5, 0.5], "angle_deg": 30}, 0.5 * H * np.sin(np.radians(30))),
    ({"position": [0.5, 0.5], "direction_deg": 90, "angle_deg": 30}, 0.5 * W * np.sin(np.radians(30))),
])
def test_fold_lifts_moving_side_without_stretch(fold, lift):
    mesh, out = _deform({"folds": [fold]})
    assert out.pos[:, 2].max() == pytest.approx(lift, rel=0.02)
    assert out.pos[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    # The crease strip is narrower than one grid cell, so a grid edge that
    # spans it becomes a chord of the bend: 1 - cos(angle / 2) shorter.
    # All other edges keep their length.
    chord = 1 - np.cos(np.radians(fold["angle_deg"]) / 2)
    assert _stretch(mesh, out) < chord + 0.005


def test_negative_fold_rests_on_ground():
    _, out = _deform({"folds": [{"position": [0.5, 0.5], "angle_deg": -30}]})
    assert out.pos[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert out.pos[:, 2].max() > 1.0


def test_curl_keeps_lengths():
    mesh, out = _deform({"curl": {"x_curvature": 0.05}})
    assert _stretch(mesh, out) < 1e-3
    assert out.pos[:, 2].max() > 1.0


def test_crumple_is_deterministic_and_shrinks_page():
    d = {"crumple": {"strength": 1.6, "creases": 40, "seed": 5}}
    _, a = _deform(d)
    _, b = _deform(d)
    np.testing.assert_array_equal(a.pos, b.pos)
    # Crumpled paper takes up less room than flat paper.
    g = a.pos.reshape(a.nv, a.nu, 3)
    edge = np.concatenate([g[0, :, :2], g[1:, -1, :2], g[-1, -2::-1, :2], g[-2:0:-1, 0, :2]])
    x, y = edge[:, 0], edge[:, 1]
    footprint = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    assert footprint < W * H
    assert a.pos[:, 2].max() > 0.3


def test_crumple_seed_is_filled_from_job_rng():
    cfg = build_config({"deform": {"crumple": {"strength": 1.0}}})
    s1 = resolve_config(cfg, np.random.default_rng(1))["deform"]["crumple"]["seed"]
    s2 = resolve_config(cfg, np.random.default_rng(2))["deform"]["crumple"]["seed"]
    assert isinstance(s1, int) and s1 != s2


def test_repeat_folds_get_their_own_values():
    cfg = build_config({"deform": {"folds": {"repeat": 3, "item": {"angle_deg": {"uniform": [-40, 40]}}}}})
    folds = resolve_config(cfg, np.random.default_rng(0))["deform"]["folds"]
    assert len(folds) == 3
    assert len({f["angle_deg"] for f in folds}) == 3
    assert all(f["crease_radius_cm"] == 0.15 for f in folds)  # defaults filled in


def test_unknown_fold_key_is_an_error():
    with pytest.raises(ConfigError, match="folds"):
        build_config({"deform": {"folds": [{"angel_deg": 10}]}})


@pytest.mark.parametrize("kind", ["keyboard", "mousepad", "phone", "mug", "pen", "pencil", "sticky_notes", "paper"])
def test_desk_items_build_and_sit_on_the_desk(kind):
    from recapture.config.schema import DESK_ITEM_DEFAULTS
    from recapture.scene.desk import build_items

    item = {**DESK_ITEM_DEFAULTS, "kind": kind, "side": "right", "gap_cm": 2.0}
    meshes = build_items([item], np.zeros(3), 0.0, (21.0, 29.7), seed=3, load_image=lambda p: None)
    assert meshes
    pos = np.concatenate([m.pos for m in meshes])
    assert pos[:, 2].min() > -0.05  # nothing sinks into the desk
    assert pos[:, 0].min() > 21.0 / 2 + 2.0 - 0.6  # beside the page, not on it
    for m in meshes:
        assert np.isfinite(m.normal).all() and len(m.indices) % 3 == 0
        # Seen from above, the surfaces face up more than down (wound the right way round).
        assert (m.normal[:, 2] > 0.2).sum() >= (m.normal[:, 2] < -0.2).sum()


def test_scan_info_scales_and_caps(tmp_path):
    import json

    from PIL import Image

    from recapture.scene.deform import make_scan
    from recapture.scene.paper_scan import height_map, scan_info

    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(tmp_path / "X_Displacement.png")
    p = {"strength": 2.0, "seed": 1, "scan": {"directory": str(tmp_path), "tile_cm": 10.0, "depth_cm": 0.5,
                                               "offset": [0, 0], "quarter_turns": 0}}
    plain = make_scan(p)
    assert plain.tile_cm == 10.0 and plain.depth_cm == 1.0 and plain.smooth_px == 0.0

    (tmp_path / "info.json").write_text(json.dumps({"tile_scale": 2, "depth_scale": 3, "max_strength": 0.5,
                                                    "smooth_px": 4}))
    scan_info.cache_clear()
    tuned = make_scan(p)
    assert tuned.tile_cm == 20.0 and tuned.depth_cm == 0.5 * 0.5 * 3 and tuned.smooth_px == 4.0

    raw = height_map(str(tmp_path / "X_Displacement.png"), 0.0)
    smooth = height_map(str(tmp_path / "X_Displacement.png"), 4.0)
    assert smooth.std() < raw.std() * 0.5 and abs(smooth.mean() - raw.mean()) < 1e-3
    # Wrap-around blur: the left and right edges still meet without a step.
    assert abs(float(smooth[:, 0].mean() - smooth[:, -1].mean())) < 0.05
