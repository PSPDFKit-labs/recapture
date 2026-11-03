"""fx.quality: a cheap phone camera makes the photo less clear."""

import numpy as np

from recapture.fx.quality import cheap_camera


def _text_like(h=400, w=300):
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), 235, np.uint8)
    for y in range(20, h - 20, 12):  # thin dark lines like small text
        xs = rng.integers(10, w - 60)
        img[y:y + 2, xs:xs + 50] = 30
    return img


def _error(a, b):
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


def test_quality_one_changes_nothing():
    img = _text_like()
    assert cheap_camera(img, 1.0) is img


def test_lower_quality_loses_more_detail():
    img = _text_like()
    e = [_error(cheap_camera(img, q), img) for q in (1.0, 0.6, 0.0)]
    assert e[0] < e[1] < e[2]


def test_size_and_alpha_stay():
    img = np.concatenate([_text_like(), np.full((400, 300, 1), 200, np.uint8)], axis=2)
    out = cheap_camera(img, 0.3)
    assert out.shape == img.shape and np.array_equal(out[..., 3], img[..., 3])


def test_same_input_same_output():
    img = _text_like()
    assert np.array_equal(cheap_camera(img, 0.4), cheap_camera(img, 0.4))
