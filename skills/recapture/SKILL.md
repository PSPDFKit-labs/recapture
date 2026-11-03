---
name: recapture
description: Degrade documents: turn clean document or page images into realistic phone photos and flatbed scans, with exact layout labels, for layout-detection training data. Use when asked to degrade documents or pages, to make photos or scans of documents, to build or augment a document image dataset, or to create or edit a Recapture config.
---

# Recapture

Recapture renders each clean page image as a phone photo (a 3D page on a desk, lit, photographed) or as a flatbed scan (printer marks, scanner artifacts), and moves the page's layout boxes into every output image as exact 4-point quads. The `recapture` CLI does the work; this skill runs it.

Run every step in the user's project folder. Each step ends on its **done** line; reach it before moving on.

## 1. Setup

Run `recapture --version`.

- It prints `recapture 0.1.0` or newer: skip to the asset check.
- Missing or older: ask the user once, then run `bash ${CLAUDE_SKILL_DIR}/scripts/install.sh` (Claude Code fills in `${CLAUDE_SKILL_DIR}`; in other agents use `scripts/install.sh` in this skill's folder). Add `--editor` when the user wants the editor window. The script needs `uv`, `git` and network access to github.com; when it stops, show the user its message and stop.

The assets (textures and crumple scans) are checked in step 3: `recapture validate` prints OK only when every asset the config names is on disk. A "not found ... run `recapture fetch-assets`" error means they are missing: tell the user it is about 105 MB of CC0 textures (Poly Haven, ambientCG), get a yes, run `recapture fetch-assets --yes`, then validate again.

**Done:** `recapture --version` prints the version.

## 2. Inputs

The user gives a folder of page images (PNG or JPG, one page per image). Boxes come from a JSON file next to each image with the same stem (`page.png` → `page.json`); [`references/boxes.md`](references/boxes.md) has both accepted formats. A page with no box file still renders, with no labels.

Look at one box file of each kind: different formats, or different naming in the files, give different class names (`Text` from the API format next to `paragraph` from the simple format). Tell the user when the label names differ between files; they show up as separate classes in step 6.

**Done:** you know the image folder, how many pages it holds, which pages have box files, and which label names they use. Step 4 confirms the box files load: its summary prints `boxes N/N` per image, and a broken box file stops it with the file's name.

## 3. Config

Start from an example, then change only what the user asks:

```bash
recapture init <name>.json --from mix     # one of: photo, scan, mix (photos and 30 % scans), card
```

`mix` makes only paper: phone photos and scans. `card` turns every image into a laminated plastic card, so use it only when the user's images are cards (ID, driving licence, bank card) or the user asks for card photos. For pages, pick `photo`, `scan` or `mix`.

Edit the JSON to match the request, using [`references/config.md`](references/config.md) for the keys and for common requests ("more scans", "no hand shadows", "black-and-white scans"). Write ranges as `{"uniform": [lo, hi]}`, `{"randint": [lo, hi]}` or `{"choice": [...], "weights": [...]}`. Keep asset paths as `@assets/...`. Then run `recapture validate <name>.json`. A user who wants sliders can run `recapture edit <name>.json` (needs `--editor` at install).

Shares like "30 % scans" are a chance per image, so a run of a few dozen images drifts from them; tell the user when the run is small.

Tell the user the config's path and list each change you made against the example, so they can review it or open it with `recapture edit <name>.json`.

**Done:** `recapture validate` prints OK, every change the user asked for is in the file, and the user has the path and your list of changes.

## 4. Preview

```bash
recapture preview <name>.json -i <images> -n 6 -o preview.jpg
```

Look at `preview.jpg` yourself (label quads drawn in red) and read the summary lines. Then put the sheet in front of the user: show it inline if your interface can display images; otherwise open it on their screen with `open preview.jpg` (macOS) or `xdg-open preview.jpg` (Linux), and give its full path. Ask whether the look is right only after the user can see it. Change the config and preview again until they agree.

**Done:** the user approved the look, and every box the summary reports lost (`boxes 15/17`) is one the image does not show: cut off by the frame in a photo, or hidden under a fold.

## 5. Render

```bash
recapture render -i <images> -c <name>.json -o <dataset> -n <variants> -j <workers>
```

- `-n` is images per page. Size the run with the user: pages x variants.
- `-j`: photos take about 1.6 GB of memory per worker, scans about 0.3 GB. Read the machine's memory and CPU count (`sysctl -n hw.memsize hw.ncpu` on macOS, `free -g; nproc` on Linux) and pick the smaller of: CPU count, memory / 1.6 GB, and images / 10 (a small run gains nothing from many workers).
- Photo mode needs OpenGL. On a Linux server without a display it can fail to make a GL context; scans need no GPU. Report the error instead of retrying.
- A stopped run continues with the same command plus `--skip-existing`.

Each output image gets `<stem>_<variant>.jpg` with a 3-digit variant (`invoice_000.jpg`), its labels `invoice_000.json` and its drawn settings `invoice_000.config.json`. Same config, seed and version give the same images.

**Done:** the progress line reports `failed: 0`, or every failure is reported to the user with its message.

## 6. Export

```bash
recapture export <dataset> -o <export> --format coco --val 0.1
```

`coco` (polygons, boxes, areas) or `yolo-obb` (Ultralytics oriented boxes with `data.yaml`); ask which the user trains with. The split is by source document, so every variant of a page lands in one split, and a split needs at least one whole document: with 10 pages, `--val 0.1` gives about one. `--test` adds a test split. `--classes A B C` fixes the class order (and drops other labels).

**Done:** you reported to the user: images and boxes per split, any requested split that came out empty, the class list (and any names that look like the same class), and the export path.
