---
name: verify-recapture
description: Prove that a Recapture change does what it should and nothing else, against a base commit. Renders the examples and fixture pages with the base and the working tree and compares the images and label files byte for byte, times sampling and editor drags, and screenshots the editor. Use before you call a change done, and when asked to verify, check for regressions, compare against main, benchmark, or show the editor.
---

# Verify a Recapture change

Run from the checkout under test with its `.venv`, after `git fetch origin`, so `origin/main` is current. In a fork, where `origin` is the fork, fetch `upstream` and pass `--base upstream/main`. The scripts build the base commit in a worktree under `.verify-runs/recapture/base/`, run each tree with its own source, and share the checkout's assets (`assets/`, or `$RECAPTURE_ASSETS`). Photo renders, `bench.py` and `editor_shot.py` need OpenGL; `editor_shot.py` opens a real window.

## 1. Output: compare.py

```bash
.venv/bin/python .claude/skills/verify-recapture/scripts/compare.py --base origin/main --name <what-changed>
```

Renders `photo`, `mix`, `scan` and `card` (when the tree has it), 3 variants, seed 4242, on `fixtures/page.png`, and `fixtures/card.png` for card configs. Add a user's config with `--configs photo configs/phone_photo.json`. An example missing in the base shows as "new in head".

- **A change that is not about the look** (refactor, speed, editor, docs): every image and label file is identical. Any difference fails the run. Find its cause before going on. A sample file (`.config.json`) may differ when the change adds keys at their defaults; the summary lists them.
- **A change that is meant to alter the look:** add `--expect-change`. A failed render still fails the run. Open the sheets in `diff/` (base and head side by side, with label quads), show them to the user, and say which images changed and why. Images the change should leave alone must still be identical.

## 2. Speed: bench.py

```bash
.venv/bin/python .claude/skills/verify-recapture/scripts/bench.py --base origin/main --name <what-changed>
```

Times `sample_scene` over 40 variants, and the editor's fast preview over a 60-frame camera drag and a lamp drag. Both drags end with the camera where an A4 page covers too little of the frame. It also counts how often the desk changes during a drag. Fails when the head's 95th percentile is over 1.5 times the base's, the head changes the desk more often, or more samples find no camera that fits (`no fit`). Run it for any change to sampling, scene building, rendering or the editor loop, and report the numbers for base and head.

## 3. Editor: editor_shot.py

```bash
.venv/bin/python .claude/skills/verify-recapture/scripts/editor_shot.py photo --open Paper Card --name <what-changed>
```

Opens the editor on a config and the fixture page, waits for the preview, saves `editor.png` and closes. `--open` keeps only the named left-panel sections open, so the ones a change touches fit on one screen. Look at the screenshot before you describe the editor to anyone. A screenshot proves layout, and the tests and `compare.py` prove values.

## 4. Tests

```bash
.venv/bin/pytest
```

A change to geometry, the crop or scan transforms also has to pass `tests/test_labels_render.py`, `tests/test_crop.py` and `tests/test_scanner.py`. These are in the full run.

## Evidence and report

Each run writes `.verify-runs/recapture/runs/<UTC time>-<name>/`, which git ignores: `summary.json` (base and head commits, a dirty flag, results, `ok`), the renders under `base/` and `head/`, and `diff/` sheets. Report the run folders, the head commit and whether it was dirty, the test count, and the result of each step you ran. A step you skipped goes in the report too, with the reason.

`fixtures/` holds the two pages and their box files. `scripts/make_fixtures.py` draws them with Pillow's built-in font. Run it again only to change the fixtures, and commit the result.

Remove a base worktree you no longer need with `git worktree remove .verify-runs/recapture/base/<sha>`.
