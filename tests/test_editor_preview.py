"""The editor's full render must be the same image the CLI makes for that variant."""

import json

import numpy as np
import pytest
from PIL import Image

from recapture.config import build_config


@pytest.fixture(scope="module")
def renderer():
    from recapture.render import Renderer
    try:
        r = Renderer()
    except Exception as e:  # no GL available on this machine
        pytest.skip(f"no OpenGL context: {e}")
    yield r
    r.release()


def test_full_render_matches_cli(tmp_path, renderer):
    from recapture.editor.preview import Preview
    from recapture.labels import load_boxes
    from recapture.pipeline import Job, Pipeline, job_seed

    doc = tmp_path / "page.png"
    page = np.full((700, 500, 3), 255, np.uint8)
    page[100:160, 60:440] = 30
    Image.fromarray(page).save(doc)
    (tmp_path / "page.json").write_text(json.dumps({"boxes": [{"id": "a", "label": "p", "bbox": [60, 100, 440, 160]}]}))
    user = {
        "page": {"mesh_resolution": 128},
        "deform": {"crumple": {"strength": {"uniform": [0.4, 1.6]}}},
        "light": {"elevation_deg": {"uniform": [20, 60]}, "azimuth_deg": {"uniform": [0, 360]}},
        "camera": {"output_px": [300, 400]},
        "output": {"format": "png"},
    }
    for variant in range(2):
        cli = Pipeline(build_config(user, base_dir=tmp_path), renderer).run(
            Job(source=doc, out_dir=tmp_path / "out", variant=variant, seed=job_seed(0, doc.name, variant)))
        frame = Preview(renderer).render(user, tmp_path, str(doc), load_boxes(tmp_path / "page.json"),
                                         variant, None, None, True)
        assert np.array_equal(np.asarray(Image.open(cli.image_path)), frame.image)
        assert cli.labels["boxes"] == frame.labels


def test_background_off_gives_a_transparent_png(tmp_path, renderer):
    from recapture.pipeline import Job, Pipeline

    doc = tmp_path / "page.png"
    Image.fromarray(np.full((700, 500, 3), 255, np.uint8)).save(doc)
    cfg = build_config({"page": {"mesh_resolution": 64}, "camera": {"output_px": [300, 400]},
                        "background": {"enabled": False}, "output": {"format": "png"}})
    out = Pipeline(cfg, renderer).run(Job(source=doc, out_dir=tmp_path / "out", variant=0, seed=1))
    img = np.asarray(Image.open(out.image_path))
    assert img.shape[2] == 4
    alpha = img[..., 3]
    assert alpha[0, 0] == 0 and alpha[200, 150] == 255  # a corner shows no page; the center does
    assert img[200, 150, :3].mean() > 150  # the page keeps its color


def test_background_off_needs_png():
    from recapture.config import ConfigError, resolve_config
    with pytest.raises(ConfigError, match="png"):
        resolve_config(build_config({"background": {"enabled": False}}), np.random.default_rng(0))


def test_cached_page_renders_the_same(renderer):
    import dataclasses
    from recapture.config import resolve_config
    from recapture.scene import PageCache, build_scene

    cfg = resolve_config(build_config({"deform": {"crumple": {"strength": 1.0}}, "camera": {"output_px": [200, 260]}}),
                         np.random.default_rng(0))
    cache = PageCache()
    a = build_scene(cfg, (500, 700), mesh_resolution=64, page_cache=cache)
    b = build_scene(cfg, (500, 700), mesh_resolution=64, page_cache=cache)
    assert a.page is b.page  # built once
    doc = np.full((700, 500, 4), 255, np.uint8)
    first = renderer.render(a, doc, doc_key="d").color
    moved = dataclasses.replace(b, light=dataclasses.replace(b.light, position=b.light.position + [5.0, 0, 0]))
    renderer.render(moved, doc, doc_key="d")
    assert np.array_equal(renderer.render(b, doc, doc_key="d").color, first)


def test_table_gets_the_page_shadow_and_lamp_color(renderer):
    from recapture.config import resolve_config
    from recapture.scene import build_scene

    # Lamp low on the left; the page's right edge is lifted 3 cm.
    cfg = resolve_config(build_config({
        "deform": {"edge_lift_cm": [0, 3.0, 0, 0]},
        "light": {"elevation_deg": 20, "azimuth_deg": 180, "distance_cm": 70, "color_temp_k": 3000},
        "camera": {"position_cm": [0, 0, 70], "output_px": [400, 400], "frame_coverage": [0.05, 0.95]}}),
        np.random.default_rng(0))
    scene = build_scene(cfg, (2100, 2970), mesh_resolution=64)
    out = renderer.render(scene, np.full((2970, 2100, 4), 255, np.uint8))
    assert out.table is not None
    light = out.table[..., :3].sum(axis=2)
    u, front = scene.camera.project(np.array([[-16.0, 0.0, 0.0], [16.0, 0.0, 0.0]]))
    (lx, ly), (rx, ry) = np.round(u).astype(int)
    assert light[ry, rx] < 0.6 * light[ly, lx]  # in the page's shadow, right of the page
    r, g, b = out.table[ly, lx, :3]
    assert r > b  # a 3000 K lamp lights the table warm


def test_flat_background_has_no_table(renderer):
    from recapture.config import resolve_config
    from recapture.scene import build_scene

    cfg = resolve_config(build_config({"background": {"lit": False}, "camera": {"output_px": [200, 260]}}),
                         np.random.default_rng(0))
    scene = build_scene(cfg, (500, 700), mesh_resolution=32)
    assert scene.table is None
    assert renderer.render(scene, np.full((700, 500, 4), 255, np.uint8)).table is None


def test_desk_items_beside_the_page_keep_labels_exact(tmp_path, renderer):
    """Desk items lie beside the page: they show in the photo, cast shadows,
    and the labels still match a render without them."""
    from recapture.config import resolve_config
    from recapture.scene import build_scene

    base = {"camera": {"position_cm": [0, 0, 60], "output_px": [400, 400], "frame_coverage": [0.05, 0.95]},
            "light": {"elevation_deg": 30, "azimuth_deg": 0}}
    items = [{"kind": "keyboard", "side": "left", "gap_cm": 2}, {"kind": "mousepad", "side": "right", "gap_cm": 1},
             {"kind": "pen", "side": "top"}, {"kind": "paper", "side": "bottom", "gap_cm": 1}]
    doc = np.full((2970, 2100, 4), 255, np.uint8)
    plain = resolve_config(build_config({**base, "desk": {"enabled": True}}), np.random.default_rng(0))
    full = resolve_config(build_config({**base, "desk": {"enabled": True, "items": items}}), np.random.default_rng(0))
    a, b = (build_scene(c, (2100, 2970), mesh_resolution=32) for c in (plain, full))
    assert len(b.props) >= 4 and b.table.desk  # items are made of one or more parts
    out_a, out_b = renderer.render(a, doc), renderer.render(b, doc)
    assert np.array_equal(out_a.label[..., 2] > 0.5, out_b.label[..., 2] > 0.5)  # the page is not covered
    left = scene_px(b, [-25.0, 0.0, 0.0])
    assert out_b.color[left[1], left[0]].sum() != pytest.approx(out_a.color[left[1], left[0]].sum())


def scene_px(scene, p):
    px, _ = scene.camera.project(np.array([p]))
    return np.round(px[0]).astype(int)


def test_desk_item_config_is_checked():
    from recapture.config import ConfigError, resolve_config
    with pytest.raises(ConfigError, match="kind"):
        resolve_config(build_config({"desk": {"items": [{"kind": "lamp"}]}}), np.random.default_rng(0))
