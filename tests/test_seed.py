"""A page's seed comes from its path inside the input folder. Scan mode, so no GL is needed."""

import json

import numpy as np
from PIL import Image

from recapture.cli import main
from recapture.pipeline import job_seed


def _page(path, shade):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((350, 250, 3), shade, np.uint8)).save(path)


def _render(tmp_path, *extra):
    cfg = tmp_path / "scan.json"
    cfg.write_text(json.dumps({"mode": "scan", "scanner": {"skew_deg": {"uniform": [-2, 2]}}}))
    out = tmp_path / "out"
    assert main(["render", "-i", str(tmp_path / "in"), "-c", str(cfg), "-o", str(out), "-n", "2", "--seed", "7",
                 *extra]) == 0
    return out


def _sample(path):
    return json.loads(path.read_text())


def test_same_name_in_two_folders_gets_two_samples(tmp_path):
    _page(tmp_path / "in" / "doc_a" / "page.png", 200)
    _page(tmp_path / "in" / "doc_b" / "page.png", 120)
    out = _render(tmp_path, "--recursive")
    for v in ("000", "001"):
        a, b = (_sample(out / d / f"page_{v}.config.json") for d in ("doc_a", "doc_b"))
        assert a["seed"] != b["seed"]
        assert a["scanner"]["skew_deg"] != b["scanner"]["skew_deg"]
    assert _sample(out / "doc_a" / "page_000.config.json")["seed"] == job_seed(7, "doc_a/page.png", 0)


def test_a_flat_folder_keeps_the_file_name_seed(tmp_path):
    _page(tmp_path / "in" / "invoice.png", 200)
    out = _render(tmp_path)
    assert _sample(out / "invoice_001.config.json")["seed"] == job_seed(7, "invoice.png", 1)
