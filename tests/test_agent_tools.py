"""The parts that let the tool run outside the repo: the asset folder, @assets
paths, the example configs, `init` and `export`."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recapture import assets
from recapture.cli import main
from recapture.config import ConfigError, load_config
from recapture.editor import state
from recapture.export import export, split_of


def test_asset_home_order(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAPTURE_ASSETS", str(tmp_path / "a"))
    assert assets.asset_home() == (tmp_path / "a").resolve()
    monkeypatch.delenv("RECAPTURE_ASSETS")
    # This test runs from the clone, so the repo's own assets/ folder is used.
    assert assets.asset_home() == Path(__file__).resolve().parents[1] / "assets"


def test_paths_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAPTURE_ASSETS", str(tmp_path / "home"))
    assert assets.resolve("@assets/ambientcg", tmp_path / "cfg") == (tmp_path / "home" / "ambientcg").resolve()
    assert assets.resolve("../x", tmp_path / "cfg") == (tmp_path / "x").resolve()  # relative to the config, as before
    assert assets.resolve("/abs/y", tmp_path) == Path("/abs/y")


def test_missing_asset_says_how_to_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAPTURE_ASSETS", str(tmp_path / "empty"))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"deform": {"crumple": {"scan": {"directory": "@assets/ambientcg"}}}}))
    with pytest.raises(ConfigError, match="recapture fetch-assets"):
        load_config(cfg)


def test_plain_desks_are_made_without_a_download(tmp_path):
    total, failed = assets.fetch(tmp_path, parts=("plain",))
    assert (total, failed) == (0, 0)
    made = sorted(p.name for p in (tmp_path / "desks_plain").iterdir())
    assert made == sorted(assets.PLAIN_DESKS)
    assert json.loads((tmp_path / "desks_plain" / "white_laminate" / "info.json").read_text())["size_cm"] == [60.0, 60.0]


@pytest.mark.parametrize("name", ["photo", "scan", "mix"])
def test_init_writes_the_example(tmp_path, name):
    out = tmp_path / "my.json"
    assert main(["init", str(out), "--from", name]) == 0
    cfg = json.loads(out.read_text())
    assert cfg["mode"] == {"photo": "photo", "scan": "scan"}.get(name, cfg["mode"])
    assert main(["init", str(out), "--from", name]) == 1  # no overwrite without --force
    if (assets.asset_home() / "polyhaven" / "desks").is_dir():  # assets fetched: the example loads as is
        load_config(out)


def test_saving_keeps_asset_paths():
    user = {"desk": {"surface": ["@assets/polyhaven/desks", "../assets/desks_plain"]}}
    out = state.rebase_paths(user, Path("/a/configs"), Path("/a/other/deeper"))
    assert out["desk"]["surface"] == ["@assets/polyhaven/desks", "../../assets/desks_plain"]


def _dataset(root: Path, docs: int, variants: int) -> None:
    root.mkdir(parents=True)
    for d in range(docs):
        for v in range(variants):
            stem = f"doc{d}_{v:03d}"
            Image.fromarray(np.full((50, 40, 3), 255, np.uint8)).save(root / f"{stem}.jpg")
            boxes = [{"id": "a", "label": "text", "quad": [[4, 5], [30, 5], [30, 20], [4, 20]]},
                     {"id": "b", "label": "figure", "quad": None},
                     {"id": "c", "label": "", "quad": [[1, 30], [39, 30], [39, 49], [1, 49]]}]
            (root / f"{stem}.json").write_text(json.dumps({"source": f"doc{d}.png", "image": f"{stem}.jpg",
                                                           "size": [40, 50], "boxes": boxes}))
            (root / f"{stem}.config.json").write_text("{}")


def test_export_coco_and_yolo(tmp_path):
    _dataset(tmp_path / "ds", docs=6, variants=3)
    res = export(tmp_path / "ds", tmp_path / "coco", "coco", val=0.3, seed=1)
    assert res["classes"] == ["text", "unlabeled"]  # the box with no quad adds no class
    total = sum(n["images"] for n in res["splits"].values())
    assert total == 18 and sum(n["boxes"] for n in res["splits"].values()) == 36
    train = json.loads((tmp_path / "coco" / "annotations" / "instances_train.json").read_text())
    a = train["annotations"][0]
    assert a["segmentation"] == [[4, 5, 30, 5, 30, 20, 4, 20]] and a["bbox"] == [4, 5, 26, 15] and a["area"] == 390
    # All variants of one document are in the same split.
    for split, n in res["splits"].items():
        assert n["images"] == 3 * n["documents"]

    res = export(tmp_path / "ds", tmp_path / "yolo", "yolo-obb", val=0.3, seed=1)
    line = (tmp_path / "yolo" / "labels" / "train").glob("*.txt").__next__().read_text().splitlines()[0]
    cls, *pts = line.split()
    assert cls == "0" and [float(v) for v in pts[:2]] == [0.1, 0.1]
    assert "names:" in (tmp_path / "yolo" / "data.yaml").read_text()


def test_split_is_fixed_per_document():
    assert split_of("a.png", 0.2, 0.1, 3) == split_of("a.png", 0.2, 0.1, 3)
    shares = [split_of(f"d{k}.png", 0.2, 0.1, 0) for k in range(2000)]
    assert abs(shares.count("val") / 2000 - 0.2) < 0.04 and abs(shares.count("test") / 2000 - 0.1) < 0.03
