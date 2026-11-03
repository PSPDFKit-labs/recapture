# Contributing to Recapture

This file tells you how to set up the code, test a change and send it. [AGENTS.md](AGENTS.md) has the full rules, and [GLOSSARY.md](GLOSSARY.md) has the words to use in code, docs and commits. Read [docs/design.md](docs/design.md) before you change how photo mode works. Its **Decisions** section gives the reason for choices that look odd, and lists ideas that were tried and dropped.

## Set up

You need [git](https://git-scm.com) and [uv](https://docs.astral.sh/uv/). uv brings its own Python 3.12.

Fork the repo on GitHub, then clone your fork and add this repo as `upstream`:

```bash
git clone https://github.com/<you>/recapture.git
cd recapture
git remote add upstream https://github.com/PSPDFKit-labs/recapture.git
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev,editor]'
.venv/bin/recapture fetch-assets
```

`fetch-assets` downloads about 105 MB of CC0 textures into `assets/`. Git ignores that folder. Use `.venv/bin/python` and `.venv/bin/recapture` for all work on the repo.

## Run the tests

```bash
.venv/bin/pytest
```

Photo tests need OpenGL. A Mac or a Linux desktop has it. A Linux server needs EGL. Scan tests need no GPU, so you can run them anywhere:

```bash
.venv/bin/pytest tests/test_scanner.py
```

## Rules for a change

AGENTS.md explains each rule. In short:

- **Reproducible output.** The same config, seed and version give the same image, byte for byte. If your change is not about the look, photo outputs must not change. Draw each new random value from its own stream (`resolve` or `stream` in `config/values.py`).
- **Exact labels.** Every pixel whose center shows a box lies inside its quad. A change to geometry, the scanner-app crop or scan transforms must pass `tests/test_labels_render.py`, `tests/test_crop.py` and `tests/test_scanner.py`.
- **Configs keep working.** A config belongs to its user. A new config key gets a default that keeps the old look, a line with a comment in `DEFAULTS`, and an editor slider when users will change it often.
- **CLI and skill change together.** When a command, a flag or an output changes, update `skills/recapture/SKILL.md`, its `references/` and the README in the same pull request.

Keep pure modules (`config/`, `scene/`, `labels/`) free of GPU and disk access.

## Prove the change

The `verify-recapture` skill in [.claude/skills/verify-recapture/SKILL.md](.claude/skills/verify-recapture/SKILL.md) compares your working tree with a base commit. A coding agent can run it for you, or you can run the scripts yourself. Compare against `upstream/main`, the latest code of this repo. Fetch it first:

```bash
git fetch upstream
.venv/bin/python .claude/skills/verify-recapture/scripts/compare.py --base upstream/main --name <what-changed>
.venv/bin/python .claude/skills/verify-recapture/scripts/bench.py --base upstream/main --name <what-changed>
.venv/bin/python .claude/skills/verify-recapture/scripts/editor_shot.py photo --open Paper Card --name <what-changed>
```

- `compare.py` renders the example configs on two fixture pages with the base and with your tree, and compares the images and label files byte for byte. If your change should not alter the look, every file must match. If it should, add `--expect-change` and look at the side-by-side sheets in `diff/`.
- `bench.py` times scene sampling and editor drags. Run it when you change sampling, scene building, rendering or the editor loop.
- `editor_shot.py` opens the editor and saves a screenshot. Run it when you change the editor.

Each run writes its results to `.verify-runs/recapture/runs/`. Git ignores that folder. `bench.py` and `editor_shot.py` need OpenGL, and `editor_shot.py` opens a real window.

## Open a pull request

- Keep it small, with one topic. Two unrelated fixes go in two pull requests.
- Write the title and description with the words from GLOSSARY.md.
- Put the evidence in the description: the `compare.py` result, before and after images when the look changes, `bench.py` numbers when speed can change, and the test count. If you skipped a step, say which one and why.
- Say what can break, and which outputs change for existing users.

The pull request template asks for these items.

## Report a bug or ask for a feature

Use the issue templates. For a bug, give the output of `recapture --version`, your OS and GPU, the command, the config and a sample image if you can share one. Report security problems in private, as [SECURITY.md](SECURITY.md) tells you. Everyone in the project follows the [code of conduct](CODE_OF_CONDUCT.md).

## License

Recapture is under the [Apache License 2.0](LICENSE). When you send a contribution, you agree that it is under the same license.
