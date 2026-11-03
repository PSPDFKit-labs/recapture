"""Render the same configs, pages and seeds with a base commit and with the working tree, and compare.

    .venv/bin/python .claude/skills/verify-recapture/scripts/compare.py [--base origin/main] [--name NAME]
        [--configs photo mix scan configs/my.json] [-n 3] [--seed 4242] [--expect-change]

Images and label files must be byte-identical. A sample file (.config.json) may differ (new keys at their
defaults); the summary lists it but it does not fail the run. For every image that differs, diff/ holds the
base and head side by side with their label quads. Exit 1 when anything differs, unless --expect-change
(a change that is meant to alter the look: then the diff sheets are the evidence to review).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

DEFAULT_CONFIGS = ["photo", "mix", "scan", "card"]


def quads(img: Image.Image, label_file: Path) -> Image.Image:
    img = img.convert("RGB")
    if label_file.exists():
        d = ImageDraw.Draw(img)
        for box in json.loads(label_file.read_text()).get("boxes", []):
            if box.get("quad"):
                d.polygon([tuple(p) for p in box["quad"]], outline=(255, 40, 40), width=3)
    return img


def side_by_side(base: Path, head: Path, out: Path) -> None:
    a = quads(Image.open(base), base.with_suffix(".json"))
    b = quads(Image.open(head), head.with_suffix(".json"))
    h = 900
    a = a.resize((round(a.width * h / a.height), h))
    b = b.resize((round(b.width * h / b.height), h))
    sheet = Image.new("RGB", (a.width + b.width + 30, h + 40), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), f"base: {base.name}", fill=(0, 0, 0))
    d.text((a.width + 20, 10), "head", fill=(0, 0, 0))
    sheet.paste(a, (10, 30))
    sheet.paste(b, (a.width + 20, 30))
    sheet.save(out, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="commit, branch or tag to compare against (default origin/main)")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="example names or config paths")
    ap.add_argument("-n", type=int, default=3, help="variants per page")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--name", default="compare")
    ap.add_argument("--expect-change", action="store_true", help="differences are intended: exit 0, keep the sheets")
    args = ap.parse_args()

    base, base_sha = lib.base_tree(args.base)
    head = lib.root()
    run = lib.run_dir(args.name)
    summary = {"base": {"ref": args.base, "sha": base_sha}, "head": lib.head_info(), "run": str(run),
               "n": args.n, "seed": args.seed, "configs": {}}
    failed = broken = False  # broken: a render failed, which --expect-change never excuses
    for config in args.configs:
        name = Path(config).stem
        head_cfg, base_cfg = lib.example_path(head, config), lib.example_path(base, config)
        if head_cfg is None:
            summary["configs"][name] = {"status": "missing in head"}
            continue
        pages = lib.stage_pages(run / "pages" / name, [lib.fixture_for(config)])
        entry = {"head": lib.render(head, head_cfg, pages, run / "head" / name, args.n, args.seed)}
        if base_cfg is None:
            entry["status"] = "new in head (no base to compare)" if entry["head"]["ok"] else "render failed"
            broken |= not entry["head"]["ok"]
            summary["configs"][name] = entry
            continue
        entry["base"] = lib.render(base, base_cfg, pages, run / "base" / name, args.n, args.seed)
        if not (entry["head"]["ok"] and entry["base"]["ok"]):
            entry["status"] = "render failed"
            broken = True
            summary["configs"][name] = entry
            continue
        same, differ, samples = [], [], []
        for f in sorted((run / "base" / name).iterdir()):
            g = run / "head" / name / f.name
            if f.name.endswith(".config.json"):
                if not g.exists() or g.read_bytes() != f.read_bytes():
                    samples.append(f.name)
                continue
            if g.exists() and g.read_bytes() == f.read_bytes():
                same.append(f.name)
            else:
                differ.append(f.name)
                if f.suffix in (".jpg", ".png") and g.exists():
                    (run / "diff").mkdir(exist_ok=True)
                    side_by_side(f, g, run / "diff" / f"{name}_{f.stem}.jpg")
        entry.update(identical=len(same), differ=differ, sample_files_differ=samples,
                     status="identical" if not differ else "differs")
        failed |= bool(differ)
        summary["configs"][name] = entry

    summary["ok"] = not broken and (not failed or args.expect_change)
    lib.write_summary(run, summary)
    for name, e in summary["configs"].items():
        line = f"{name:>14}: {e['status']}"
        if "identical" in e:
            line += f", {e['identical']} identical, {len(e['differ'])} differ"
        if "base" in e and "head" in e:
            line += f", render {e['base']['seconds']} s base / {e['head']['seconds']} s head"
        print(line)
        for side in ("base", "head"):
            if side in e and not e[side]["ok"]:
                print(f"{'':>16}{side}: {e[side]['tail'][-1]}")
    print(f"{'ok' if summary['ok'] else 'FAILED'}: {run}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
