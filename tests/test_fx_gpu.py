"""The GPU camera effects must give the same look as the numpy version."""

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


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    d = tmp_path_factory.mktemp("doc")
    page = np.full((700, 500, 3), 250, np.uint8)
    page[80:120, 40:460] = 20  # a title bar
    for y in range(160, 600, 22):
        page[y:y + 8, 40:420] = 60  # text lines
    path = d / "page.png"
    Image.fromarray(page).save(path)
    return path


def _render(renderer, doc, tmp_path, backend, fx=None, extra=None, variant=0):
    from recapture.pipeline import Job, Pipeline

    user = {"page": {"mesh_resolution": 128}, "deform": {"crumple": {"strength": 1.0}},
            "camera": {"output_px": [384, 512], "position_cm": [3, -8, 30]},
            "fx": {"backend": backend, "defocus": {"strength_px": 2.5}, **(fx or {})},
            "output": {"format": "png"}, **(extra or {})}
    out = Pipeline(build_config(user), renderer).run(Job(source=doc, out_dir=tmp_path / backend, variant=variant, seed=3))
    return np.asarray(Image.open(out.image_path)).astype(np.int16)


@pytest.mark.parametrize("extra", [{}, {"desk": {"enabled": True, "items": [{"kind": "keyboard", "side": "left"}]}},
                                   {"background": {"lit": False}}])
def test_gpu_effects_match_numpy_without_noise(renderer, doc, tmp_path, extra):
    fx = {"noise": {"enabled": False}}
    cpu = _render(renderer, doc, tmp_path, "cpu", fx, extra)
    gpu = _render(renderer, doc, tmp_path, "gpu", fx, extra)
    d = np.abs(cpu - gpu)
    assert d.mean() < 0.5
    assert np.percentile(d, 99) <= 6


def test_gpu_effects_match_numpy_exactly_without_blurs(renderer, doc, tmp_path):
    fx = {"noise": {"enabled": False}, "defocus": {"enabled": False}, "sharpen": {"enabled": False}}
    d = np.abs(_render(renderer, doc, tmp_path, "cpu", fx) - _render(renderer, doc, tmp_path, "gpu", fx))
    assert d.max() <= 1


def test_gpu_noise_has_the_same_strength_and_grain(renderer, doc, tmp_path):
    def noise(backend):
        off = _render(renderer, doc, tmp_path, backend, {"noise": {"enabled": False}, "sharpen": {"enabled": False}})
        on = _render(renderer, doc, tmp_path, backend, {"noise": {"shot": 0.004, "read": 0.001},
                                                        "sharpen": {"enabled": False}})
        n = (on - off).astype(np.float32).mean(axis=2)
        return n.std(), np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]

    (std_c, corr_c), (std_g, corr_g) = noise("cpu"), noise("gpu")
    assert std_g == pytest.approx(std_c, rel=0.05)
    assert corr_g == pytest.approx(corr_c, abs=0.03)
