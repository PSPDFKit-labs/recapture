"""Open the editor on a config and a page, let it render, save a screenshot and close it.

    .venv/bin/python .claude/skills/verify-recapture/scripts/editor_shot.py CONFIG [--page PAGE]
        [--name NAME] [--open Paper Card ...]

--open keeps only these left-panel sections open (the rest start closed), so the sections a change touches
fit in one screenshot. The editor runs the working tree's source. Needs a screen (it opens a real window).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="example name or config path")
    ap.add_argument("--page", type=Path, help="page image (default: the fixture that fits the config)")
    ap.add_argument("--open", nargs="+", help="left-panel sections to keep open")
    ap.add_argument("--name", default="editor")
    args = ap.parse_args()

    sys.path.insert(0, str(lib.root() / "src"))
    from imgui_bundle import imgui

    from recapture.editor import app

    if args.open:
        keep = set(args.open)
        section = app.Editor._section

        def only_some(self, name, *rest):
            if name in keep:
                return section(self, name, *rest)
            return imgui.collapsing_header(f"{name}###section-{name}", imgui.TreeNodeFlags_.none)

        app.Editor._section = only_some
    run = lib.run_dir(args.name)
    config = lib.example_path(lib.root(), args.config)
    page = args.page or lib.fixture_for(args.config)
    shot = run / "editor.png"
    app.run(config, page, screenshot=shot)
    lib.write_summary(run, {"head": lib.head_info(), "config": str(config), "page": str(page), "screenshot": str(shot)})
    print(shot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
