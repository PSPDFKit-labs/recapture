import json

import pytest

from recapture.config import build_config
from recapture.editor import state


def test_set_value_creates_parents_and_keeps_siblings_default():
    user = {}
    state.set_value(user, ("light", "elevation_deg"), 20.0)
    state.set_value(user, ("camera", "position_cm", 2), 30.0)
    assert user == {"light": {"elevation_deg": 20.0}, "camera": {"position_cm": [0.0, -10.0, 30.0]}}
    merged = state.merged(user)
    assert merged["light"]["azimuth_deg"] == 135.0  # sibling keeps its default
    build_config(user)  # still a valid config


def test_random_and_fixed_round_trip():
    assert state.to_random(10.0) == {"uniform": [8.0, 12.0]}
    assert state.to_fixed({"uniform": [8.0, 12.0]}) == 10.0
    assert state.to_random(5) == {"randint": [4, 6]}
    assert state.to_fixed({"randint": [4, 6]}) == 5
    assert state.to_random(True) == {"choice": [False, True]}
    assert state.to_fixed({"repeat": 2, "item": {"angle_deg": 5}}) == [{"angle_deg": 5}, {"angle_deg": 5}]


def test_strip_defaults_keeps_only_changes():
    user = {"light": {"elevation_deg": 25.0, "azimuth_deg": 90.0}, "fx": {"enabled": True}}
    assert state.strip_defaults(user) == {"light": {"azimuth_deg": 90.0}}


def test_format_config_is_valid_json():
    cfg = json.load(open("configs/phone_photo.json"))
    assert json.loads(state.format_config(cfg)) == cfg


def test_rebase_paths(tmp_path):
    user = {"background": {"images": "../assets/bg"}, "deform": {"crumple": {"scan": {"directory": "/abs/scan"}}}}
    out = state.rebase_paths(user, tmp_path / "configs", tmp_path / "other" / "deeper")
    assert out["background"]["images"] == "../../assets/bg"
    assert out["deform"]["crumple"]["scan"]["directory"] == "/abs/scan"


def test_lamp_orbit_wraps_and_clamps():
    az, el = state.orbit_lamp(10.0, 80.0, dx=40.0, dy=-100.0)
    assert az == pytest.approx(350.0)
    assert el == 89.0


def test_camera_orbit_keeps_distance_and_zooms():
    eye, target = [0.0, -10.0, 34.0], [0.0, 0.0, 0.0]
    moved = state.orbit_camera(eye, target, dx=30.0, dy=0.0, wheel=0.0)
    dist = lambda p: sum((a - b) ** 2 for a, b in zip(p, target)) ** 0.5
    assert dist(moved) == pytest.approx(dist(eye))
    closer = state.orbit_camera(eye, target, dx=0.0, dy=0.0, wheel=1.0)
    assert dist(closer) == pytest.approx(dist(eye) * 0.92)


def test_lamp_angles_round_trip():
    import numpy as np
    from recapture.config import build_config, resolve_config
    from recapture.editor.view3d import lamp_angles
    from recapture.scene.scene import lamp_position

    lc = resolve_config(build_config({"light": {"elevation_deg": 25.0, "azimuth_deg": 300.0, "distance_cm": 70.0}}),
                        np.random.default_rng(0))["light"]
    center = np.array([2.0, -1.0, 0.0])
    az, el, d = lamp_angles(lamp_position(lc, center), center)
    assert (az, el, d) == pytest.approx((300.0, 25.0, 70.0))


def test_fixed_hand_stays_put_when_the_lamp_moves():
    import numpy as np
    from recapture.config import build_config
    from recapture.config.schema import resolve_config
    from recapture.editor.view3d import occluder_anchor
    from recapture.scene.scene import build_scene

    anchors = []
    for az in (30.0, 200.0):
        cfg = build_config({"occluders": [{"shape": "hand_phone", "position_cm": [4.0, -6.0, 18.0]}],
                            "light": {"azimuth_deg": az}})
        sc = build_scene(resolve_config(cfg, np.random.default_rng(0)), (2480, 3508), mesh_resolution=16)
        occ = sc.occluders[0]
        anchors.append(occluder_anchor(occ.shape, occ.corners))
    assert anchors[0] == pytest.approx([4.0, -6.0, 18.0])
    assert anchors[1] == pytest.approx([4.0, -6.0, 18.0])


def test_set_value_edits_inside_a_repeat_spec():
    user = {"deform": {"folds": {"repeat": {"choice": [0, 1]}, "item": {"angle_deg": 10}}}}
    state.set_value(user, ("deform", "folds", "repeat"), 3)
    assert user["deform"]["folds"] == {"repeat": 3, "item": {"angle_deg": 10}}


def test_orbit_camera_view_looks_at_pivot():
    import numpy as np
    from recapture.editor.view3d import OrbitCamera

    cam = OrbitCamera(pivot=np.array([1.0, 2.0, 0.0]))
    eye_space = cam.view() @ np.array([1.0, 2.0, 0.0, 1.0])
    assert eye_space[:2] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert eye_space[2] == pytest.approx(-cam.distance)


def test_moving_the_lamp_pins_hands_that_aim_at_the_page():
    import numpy as np
    pytest.importorskip("imgui_bundle")
    from recapture.editor import app
    from recapture.editor.preview import Preview
    from recapture.render import Renderer

    try:
        renderer = Renderer()
    except Exception as e:
        pytest.skip(f"no OpenGL context: {e}")
    ed = app.Editor(None, None)
    ed.preview = Preview(renderer)
    ed.preview.add_image(app.SAMPLE_DOC, app._sample_page())
    ed._upload_photo = lambda: None  # no window to show it in
    ed.user = {"occluders": [{"shape": "blob", "target": [0.5, 0.5], "height_cm": 15}]}
    ed._rebuild(app.FAST)
    before = ed._objects(ed.frame)["hand0"]
    ed._move("lamp", np.array([-30.0, 20.0, 25.0]), ed.frame)
    ed._rebuild(app.FAST)
    assert ed.user["occluders"][0]["position_cm"] == pytest.approx(before, abs=0.01)
    assert ed._objects(ed.frame)["hand0"] == pytest.approx(before, abs=0.01)
    renderer.release()


def test_random_box_keeps_the_range_and_the_drawn_value():
    pytest.importorskip("imgui_bundle")
    from types import SimpleNamespace
    from recapture.editor import app

    ed = app.Editor(None, None)
    s = app.Slider("Crumple", ("deform", "crumple", "strength"), 0.0, 3.0)
    ed.set(s.path, 0.7)
    ed.toggle_random(s, 0.7, True)
    ed.set(s.path, {"uniform": [1.05, 1.95]})  # the user drags the range
    ed.frame = SimpleNamespace(resolved={"deform": {"crumple": {"strength": 1.4321}}})  # this photo drew 1.4321
    ed.toggle_random(s, {"uniform": [1.05, 1.95]}, False)
    assert state.get(ed.user, s.path) == 1.4321  # the photo does not change
    ed.toggle_random(s, 1.4321, True)
    assert state.get(ed.user, s.path) == {"uniform": [1.05, 1.95]}


def test_flat_page_button_flattens_a_random_config():
    import numpy as np
    pytest.importorskip("imgui_bundle")
    from pathlib import Path
    from recapture.config import build_config
    from recapture.config.schema import resolve_config
    from recapture.editor import app

    ed = app.Editor(None, None)
    ed.user = json.loads(Path("configs/phone_photo.json").read_text())
    ed.flat_page()
    deform = resolve_config(build_config(ed.user, base_dir=Path("configs")), np.random.default_rng(5))["deform"]
    assert deform["crumple"]["strength"] == 0 and deform["folds"] == []
    assert deform["edge_lift_cm"] == [0, 0, 0, 0]
    assert deform["curl"]["x_curvature"] == 0 and deform["curl"]["y_curvature"] == 0


def test_edge_lift_slider_sets_all_four_edges():
    pytest.importorskip("imgui_bundle")
    from recapture.editor import app

    ed = app.Editor(None, None)
    s = next(s for _, group in app.SLIDER_GROUPS for s in group if s.label == "Edge lift (cm)")
    ed.set_slider(s, {"uniform": [0.0, 1.0]})
    assert ed.user["deform"]["edge_lift_cm"] == [{"uniform": [0.0, 1.0]}] * 4


def test_mode_mix_round_trip():
    assert state.read_mode("photo") == ("photo", 0.0)
    assert state.read_mode("scan") == ("scan", 1.0)
    assert state.read_mode(state.mix_mode(0.3)) == ("mix", pytest.approx(0.3))
    assert state.read_mode({"choice": ["scan", "photo"]}) == ("mix", 0.5)
    assert state.read_mode({"choice": ["photo", "scan"], "weights": [3, 1]}) == ("mix", 0.25)
    assert state.read_mode({"choice": ["photo"]})[0] == "other"
