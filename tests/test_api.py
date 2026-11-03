"""The public Python API: the names in `recapture.__all__`, and the README example, in scan mode (no GL)."""

import json
import subprocess
import sys

import numpy as np
from PIL import Image

import recapture
from recapture.cli import main


def test_public_names_resolve():
    for name in recapture.__all__:
        assert getattr(recapture, name) is not None, name


def test_import_stays_light():
    # `import recapture` must not pull in the renderer: scan-only users may have no OpenGL.
    code = "import sys, recapture; print('recapture.render' in sys.modules, 'moderngl' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]


def test_python_api_matches_the_cli(tmp_path):
    from recapture import Job, Pipeline, build_config, job_seed

    (tmp_path / "in").mkdir()
    page = tmp_path / "in" / "page.png"
    Image.fromarray(np.full((350, 250, 3), 200, np.uint8)).save(page)
    user = {"mode": "scan", "scanner": {"skew_deg": {"uniform": [-2, 2]}}}
    cfg_path = tmp_path / "scan.json"
    cfg_path.write_text(json.dumps(user))

    assert main(["render", "-i", str(tmp_path / "in"), "-c", str(cfg_path), "-o", str(tmp_path / "cli"),
                 "-n", "2", "--seed", "7"]) == 0
    pipe = Pipeline(build_config(user))
    for v in range(2):
        result = pipe.run(Job(source=page, out_dir=tmp_path / "py", variant=v, seed=job_seed(7, "page.png", v)))
        cli_image = tmp_path / "cli" / result.image_path.name
        assert result.image_path.read_bytes() == cli_image.read_bytes()
        assert json.loads(cli_image.with_suffix(".json").read_text()) == result.labels
