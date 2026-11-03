# Changelog

All notable changes to Recapture are in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The same config, seed and version give the same image. A line under **Output changes** tells you which images a new version makes in a different way.

## [Unreleased]

## [0.1.0] - 2026-11-10

First public release, under the Apache License 2.0.

### Added

- Photo mode: the page as a 3D mesh with crumple from crumple scans, folds, curl and edge lift, on a desk with desk items, lit by one lamp, with optional shadow casters, and taken by a virtual phone camera. Camera effects run on the GPU, and `fx.quality` gives the look of a cheap phone.
- Card look for laminated plastic cards, such as ID and bank cards: rounded corners (`material.corner_radius_cm`), laminate glare (`material.glare`), a hologram sheen (`material.hologram`), a room reflection (`material.room_reflection`) and wear (`material.wear`). All are off by default.
- `camera.fit_page` frames only the page, and the scanner-app crop (`crop`) cuts a photo to the page and straightens it.
- `background.enabled: false` gives a PNG that is transparent around the page.
- Scan mode: printer marks (ink bleed, broken strokes, ghost, toner streaks and specks), then a virtual flatbed scanner (skew, blur, bold, noise, streaks, dust, dark edges, edge dots, and color, gray or black-and-white). Scans need no GPU.
- Mix configs, whose mode is a choice between photo and scan.
- Exact labels: each box's quad holds every pixel whose center shows the box. Box files in the simple format or the API format.
- The CLI: `render`, `preview`, `export` (COCO and YOLO-OBB, split by page), `init`, `edit`, `validate`, `defaults`, `sample` and `fetch-assets`. `render --recursive` seeds each page by its path inside the input folder.
- The Python API: `Pipeline`, `Job`, `JobResult`, `job_seed`, `load_config`, `build_config`, `ConfigError` and `fetch_assets`, importable from `recapture`. The same config and seed give the same images as the CLI.
- The `photo`, `scan`, `mix` and `card` example configs.
- The editor: a 3D scene view with gizmos for the lamp, the camera and the shadow casters, sliders for every value, random ranges, a Paper/Card Look menu, and a live preview with label quads.
- The asset folder and `@assets/` paths, so one config works on any machine.
- The `recapture` agent skill, and the repo as a Claude Code plugin and marketplace.
- The `verify-recapture` skill for contributors. It compares renders of a change with a base commit byte for byte, times scene sampling and editor drags, and takes an editor screenshot.

[Unreleased]: https://github.com/PSPDFKit-labs/recapture/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PSPDFKit-labs/recapture/releases/tag/v0.1.0
