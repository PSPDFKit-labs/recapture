# Recapture

Python 3.12 package and CLI that turns clean page images into phone photos and scans, with exact layout labels. `README.md` covers use; this file covers working on the code.

## Read first

- [`GLOSSARY.md`](GLOSSARY.md): the project's terms. Name things in code, docs, commits and messages with these words, and skip the ones listed under _Avoid_.
- [`docs/design.md`](docs/design.md): how photo mode works inside (scene, render passes, label math, editor, timings). Its **Decisions** section lists choices that look odd without their reason and ideas already tried and dropped. Read it before you change an area, and say so when a change goes against one. Scan mode lives in `src/recapture/scanner.py` and its docstrings.

## Layout

```
src/recapture/
  config/    config keys, defaults, random values (pure)
  scene/     page mesh, deformations, crumple scans, desk, shadow casters (pure)
  labels/    box files, box transforms, quad fitting (pure)
  render/    ModernGL passes (GPU)
  fx/        camera effects, numpy and GPU paths
  scanner.py scan mode
  crop.py    scanner-app crop
  pipeline.py one job: page, boxes, config, seed -> image, label file, sample
  cli.py     every `recapture` command
  editor/    the editor window
  examples/  the photo, scan and mix example configs
skills/recapture/  the agent skill; .claude-plugin/ makes the repo its plugin
```

Pure modules touch neither the GPU nor the disk. Keep I/O in `render/`, `pipeline.py` and `cli.py`.

## Commands

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev,editor]'
.venv/bin/recapture fetch-assets
.venv/bin/pytest
```

Use `.venv/bin/python`. Photo tests need OpenGL; scan tests need none.

## Rules

- **Reproducible output.** The same config, seed and version give the same image, byte for byte. A change that is not about the look must leave photo outputs unchanged: prove it with the `verify-recapture` skill (`.claude/skills/verify-recapture/`), which compares renders against a base commit and times the editor. Draw every new random value through its own stream (`resolve` or `stream` in `config/values.py`); one shared generator would change every existing dataset.
- **Exact labels.** Every pixel whose center shows a box lies inside its quad. A change to geometry, crop or scan transforms runs `tests/test_labels_render.py`, `tests/test_crop.py` and `tests/test_scanner.py`.
- **Configs belong to users.** Existing configs keep working across versions. A new config key gets a default that keeps the old look, a line in `DEFAULTS` with a comment (`recapture defaults` prints them), and an editor slider when users will change it often.
- **Public Python API.** The names in `recapture.__all__` (`src/recapture/__init__.py`) are as stable as the CLI. Change them only with a CHANGELOG entry, and keep the README section "Use from Python" and `tests/test_api.py` in step.
- **CLI and skill together.** `skills/recapture/` tells agents how to use the CLI. When a command, flag or output changes, update `SKILL.md`, its `references/` and the README in the same change.
- **Release.** The version is in `pyproject.toml`, `.claude-plugin/plugin.json` and `VERSION` in `skills/recapture/scripts/install.sh`; `tests/test_release.py` checks they match. Bump all three, then tag `vX.Y.Z` and push the tag. Also set the pinned tag in the README's agent prompt, and the minimum version in `SKILL.md` step 1 when users must update.
