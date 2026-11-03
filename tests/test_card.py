"""The card look: rounded corners, laminate glare and hologram (material keys, off by default)."""

import json
from importlib import resources

import numpy as np
import pytest

from recapture.config import build_config
from recapture.pipeline import sample_scene

SRC_W, SRC_H = 856, 540  # an ID-1 card's proportions


@pytest.fixture(scope="module")
def renderer():
    from recapture.render import Renderer
    try:
        r = Renderer()
    except Exception as e:  # no GL available on this machine
        pytest.skip(f"no OpenGL context: {e}")
    yield r
    r.release()


def _config(material=None):
    return build_config({
        "fx": {"enabled": False},
        "page": {"size_cm": [8.56, 5.4], "mesh_resolution": 128, "lift_cm": 0.076},
        "deform": {"crumple": {"strength": 0.0}, "folds": [], "curl": {"x_curvature": 0.0, "y_curvature": 0.0}},
        "material": material or {},
        "light": {"intensity": 0.6, "ambient": 0.4, "elevation_deg": 60.0},
        # Straight down: the mirror direction of the view is straight up, where the glare light sits in these tests.
        "camera": {"position_cm": [0.0, 0.0, 14.0], "target_cm": [0.0, 0.0, 0.0], "roll_deg": 0.0,
                   "output_px": [640, 480], "frame_coverage": [0.05, 1.0]},
    })


def _page():
    img = np.full((SRC_H, SRC_W, 4), 255, np.uint8)
    img[:, :, :3] = 200
    return img


def _render(renderer, material, seed=0):
    _, scene = sample_scene(_config(material), (SRC_W, SRC_H), np.random.default_rng(seed))
    return renderer.render(scene, _page())


def test_new_keys_default_to_the_paper_look(renderer):
    plain = _render(renderer, None)
    zeros = _render(renderer, {"corner_radius_cm": 0.0, "glare": {"strength": 0.0, "waviness": 0.0}, "hologram": 0.0,
                               "room_reflection": 0.0, "wear": 0.0})
    assert np.array_equal(plain.color, zeros.color)
    assert np.array_equal(plain.alpha, zeros.alpha)


def test_rounded_corners_clear_the_corner_pixels(renderer):
    square = _render(renderer, None)
    round_ = _render(renderer, {"corner_radius_cm": 0.8})
    u, v, mask = square.label[..., 0], square.label[..., 1], square.label[..., 2] > 0.5
    # Pixels of the square page within 0.15 cm of a corner lie outside a 0.8 cm rounded corner.
    du, dv = 0.15 / 8.56, 0.15 / 5.4
    corner = mask & ((u < du) | (u > 1 - du)) & ((v < dv) | (v > 1 - dv))
    assert corner.sum() > 20
    assert (round_.alpha[corner] < 0.5).all()
    # Away from the corners the card is unchanged.
    middle = mask & (u > 0.2) & (u < 0.8) & (v > 0.2) & (v < 0.8)
    assert np.array_equal(square.color[middle], round_.color[middle])


def test_table_under_a_cut_corner_is_not_in_the_card_shadow(renderer):
    # The shadow and contact-shadow passes must cut the same corners as the color pass: the table seen through a
    # cut corner was black when they still drew a square card.
    square = _render(renderer, None)
    round_ = _render(renderer, {"corner_radius_cm": 0.8})
    u, v, mask = square.label[..., 0], square.label[..., 1], square.label[..., 2] > 0.5
    du, dv = 0.15 / 8.56, 0.15 / 5.4
    corner = mask & ((u < du) | (u > 1 - du)) & ((v < dv) | (v > 1 - dv))
    light = round_.table[..., :3].mean(axis=-1)
    away = ~mask & (round_.alpha < 0.01)
    assert light[corner].mean() > 0.5 * np.median(light[away])
    # and the label pass agrees with what the camera sees: no page label in a cut corner
    assert (round_.label[..., 2][corner] < 0.5).all()


@pytest.mark.parametrize("seed", range(6))
def test_glare_brightens_and_stays_finite(renderer, seed):
    plain = _render(renderer, {"roughness": 0.15}, seed)
    glare = _render(renderer, {"roughness": 0.15, "hologram": 0.3,
                               "glare": {"strength": 30.0, "elevation_deg": 90.0, "size_deg": 15.0, "softness": 0.5}}, seed)
    assert np.isfinite(glare.color).all()
    on = glare.alpha > 0.5
    assert glare.color[on].mean() > plain.color[on].mean() + 0.05  # a washed-out patch, never a black frame


@pytest.mark.parametrize("seed", range(4))
def test_panel_glare_room_and_wear_brighten_and_stay_finite(renderer, seed):
    plain = _render(renderer, {"roughness": 0.15}, seed)
    lam = _render(renderer, {"roughness": 0.15, "room_reflection": 0.5, "wear": 1.0,
                             "glare": {"strength": 10.0, "elevation_deg": 90.0, "size_deg": 12.0, "shape": "panel",
                                       "aspect": 2.0, "rotation_deg": 30.0, "waviness": 1.0}}, seed)
    assert np.isfinite(lam.color).all()
    on = lam.alpha > 0.5
    assert lam.color[on].mean() > plain.color[on].mean() + 0.02


def test_unknown_glare_shape_is_rejected():
    from recapture.config import ConfigError
    from recapture.config.schema import resolve_config
    with pytest.raises(ConfigError, match="glare.shape"):
        resolve_config(_config({"glare": {"shape": "star"}}), np.random.default_rng(0))


def test_a_plain_value_for_a_section_is_rejected():
    from recapture.config import ConfigError
    with pytest.raises(ConfigError, match="material.glare: must be an object"):
        build_config({"material": {"glare": 20}})


def test_editor_opens_the_card_sliders_only_for_card_configs():
    from recapture.editor import state
    card = json.loads((resources.files("recapture") / "examples" / "card.json").read_text())
    photo = json.loads((resources.files("recapture") / "examples" / "photo.json").read_text())
    assert state.card_look(state.merged(card))
    assert not state.card_look(state.merged(photo))


def test_look_switch_turns_paper_into_a_card_and_back():
    import copy
    from recapture.editor import state
    photo = json.loads((resources.files("recapture") / "examples" / "photo.json").read_text())
    user, memory = copy.deepcopy(photo), {}
    state.set_look(user, "card", memory)
    card = state.merged(user)
    assert state.card_look(card)
    assert card["deform"]["crumple"]["strength"] == 0 and card["deform"]["folds"] == []
    assert card["page"]["size_cm"] == [8.56, 5.4] and card["camera"]["fit_page"] is True
    built = build_config(user)  # still a valid config, and the rest is untouched
    assert built["light"] == build_config(photo)["light"] and built["desk"] == build_config(photo)["desk"]
    state.set_look(user, "paper", memory)
    assert state.merged(user) == state.merged(photo)  # back to exactly the paper values
    assert state.strip_defaults(user) == state.strip_defaults(photo)  # and the saved file matches


def test_look_switch_works_on_an_empty_config():
    from recapture.editor import state
    user = {}
    state.set_look(user, "card", {})
    assert state.card_look(state.merged(user))
    build_config(user)
    state.set_look(user, "paper", {})
    assert not state.card_look(state.merged(user))
    build_config(user)


def test_card_example_builds():
    cfg = json.loads((resources.files("recapture") / "examples" / "card.json").read_text())
    built = build_config(cfg)
    assert built["material"]["corner_radius_cm"] > 0
    assert built["deform"]["folds"] == []
