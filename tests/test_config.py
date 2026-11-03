import json

import numpy as np
import pytest

from recapture.config import ConfigError, build_config, load_config, resolve, resolve_config


def test_resolve_is_deterministic_per_seed():
    spec = {"a": {"uniform": [0, 1]}, "b": [{"randint": [1, 6]}, {"normal": [0, 1], "clip": [-1, 1]}],
            "c": {"choice": ["x", {"uniform": [5, 6]}], "weights": [1, 3]}}
    one = resolve(spec, np.random.default_rng(7))
    two = resolve(spec, np.random.default_rng(7))
    assert one == two
    assert 0 <= one["a"] < 1
    assert one["b"][0] in range(1, 7)
    assert -1 <= one["b"][1] <= 1


def test_choice_resolves_nested_specs():
    out = [resolve({"choice": [{"uniform": [5, 6]}]}, np.random.default_rng(s)) for s in range(10)]
    assert all(5 <= v < 6 for v in out)


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="camera.fov"):
        build_config({"camera": {"fov": 30}})


def test_bad_spec_is_an_error():
    with pytest.raises(ConfigError, match="uniform"):
        build_config({"light": {"intensity": {"uniform": [2, 1]}}})


def test_resolved_config_has_no_specs():
    cfg = build_config({"light": {"intensity": {"uniform": [0.5, 1.0]}}})
    out = resolve_config(cfg, np.random.default_rng(0))
    assert isinstance(out["light"]["intensity"], float)


def test_background_directory_becomes_choice(tmp_path):
    bg = tmp_path / "bg"
    bg.mkdir()
    for name in ("a.jpg", "b.png", "notes.txt"):
        (bg / name).write_bytes(b"")
    (tmp_path / "cfg.json").write_text(json.dumps({"background": {"images": "bg"}}))
    cfg = load_config(tmp_path / "cfg.json")
    choices = cfg["background"]["images"]["choice"]
    assert [p.rsplit("/", 1)[1] for p in choices] == ["a.jpg", "b.png"]


def test_fixing_one_value_leaves_the_other_draws_alone():
    user = {"deform": {"crumple": {"strength": {"uniform": [1.0, 2.0]}}},
            "light": {"azimuth_deg": {"uniform": [0, 360]}}, "camera": {"roll_deg": {"uniform": [-8, 8]}}}
    a = resolve_config(build_config(user), np.random.default_rng(3))
    user["deform"]["crumple"]["strength"] = 1.5
    b = resolve_config(build_config(user), np.random.default_rng(3))
    assert b["deform"]["crumple"]["strength"] == 1.5
    assert a["light"]["azimuth_deg"] == b["light"]["azimuth_deg"]
    assert a["camera"]["roll_deg"] == b["camera"]["roll_deg"]
    assert a["deform"]["crumple"]["seed"] == b["deform"]["crumple"]["seed"]
