"""Time sampling and editor drags with a base commit and with the working tree.

    .venv/bin/python .claude/skills/verify-recapture/scripts/bench.py [--base origin/main] [--name NAME]
        [--config photo] [--variants 40] [--frames 60]

Three measurements, each run in a fresh process per tree:
- sample: `sample_scene` for N variants of the config on the A4 fixture (the frame check and its retries).
- camera drag: the editor's fast preview while the camera moves in a sweep that leaves the frame in its
  second half, like a user dragging it too high. Also counts how often the desk changes between frames.
- lamp drag: the lamp moves while the camera stays where the page does not fit.
Reports milliseconds (median, 95th percentile, max). Exit 1 when the head's p95 is more than 1.5 times the
base's in any measurement, the head changes the desk more often during a drag, or more samples find no fit.
Needs OpenGL, like the photo tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402


def inner(config: Path, page: Path, variants: int, frames: int) -> dict:
    """Runs inside the tree under test (PYTHONPATH points at its source)."""
    import time

    import numpy as np

    from recapture.config import load_config
    from recapture.editor import state
    from recapture.editor.preview import Preview
    from recapture.pipeline import sample_scene
    from recapture.render import Renderer

    def stats(ms):
        a = np.array(ms)
        return {"p50": round(float(np.median(a)), 1), "p95": round(float(np.percentile(a, 95)), 1),
                "max": round(float(a.max()), 1)}

    cfg = load_config(config)
    size = (1240, 1754)
    ms, failed = [], 0
    for v in range(variants):
        t = time.perf_counter()
        try:
            sample_scene(cfg, size, np.random.default_rng(1000 + v))
        except Exception:
            failed += 1
        ms.append((time.perf_counter() - t) * 1000)
    out = {"sample": {**stats(ms), "failed": failed}}

    user = json.loads(config.read_text())
    user["mode"] = "photo"
    preview = Preview(Renderer())
    preview.render(user, config.parent, str(page), [], 0, 640, 160, False)  # warm caches and shaders

    def drag(move):
        ms, desks, warnings = [], [], 0
        for i in range(frames):
            move(i / max(frames - 1, 1))
            t = time.perf_counter()
            frame = preview.render(user, config.parent, str(page), [], 0, 640, 160, False)
            ms.append((time.perf_counter() - t) * 1000)
            desks.append(str(frame.resolved["desk"]["surface"]))
            warnings += bool(frame.warning)
        changes = sum(a != b for a, b in zip(desks, desks[1:]))
        return {**stats(ms), "desk_changes": changes, "frames_out_of_frame": warnings}

    def camera(s):  # left to right at 30 cm, then up to 50 cm, where an A4 page covers too little
        z = 30.0 if s < 0.5 else 30.0 + 40.0 * (s - 0.5)
        state.set_value(user, ("camera", "position_cm"), [round(-5 + 10 * min(s * 2, 1), 2), -3.0, round(z, 2)])
        state.set_value(user, ("camera", "target_cm"), [0.0, 0.0, 0.0])

    out["camera_drag"] = drag(camera)
    state.set_value(user, ("camera", "position_cm"), [0.0, 0.0, 50.0])

    def lamp(s):
        state.set_value(user, ("light", "azimuth_deg"), round(120 + 100 * s, 2))
        state.set_value(user, ("light", "elevation_deg"), round(30 + 15 * s, 2))

    out["lamp_drag"] = drag(lamp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--config", default="photo", help="example name or config path")
    ap.add_argument("--variants", type=int, default=40)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--name", default="bench")
    ap.add_argument("--inner", nargs=2, metavar=("CONFIG", "PAGE"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.inner:
        print(json.dumps(inner(Path(args.inner[0]), Path(args.inner[1]), args.variants, args.frames)))
        return 0

    base, base_sha = lib.base_tree(args.base)
    head = lib.root()
    run = lib.run_dir(args.name)
    page = lib.FIXTURES / "page.png"
    results, failures = {}, []
    for side, tree in (("base", base), ("head", head)):
        config = lib.example_path(tree, args.config)
        res = subprocess.run([lib.python(), __file__, "--inner", str(config), str(page),
                              "--variants", str(args.variants), "--frames", str(args.frames)],
                             env=lib.env_for(tree), capture_output=True, text=True)
        if res.returncode != 0:
            failures.append(f"{side}: {res.stderr.strip().splitlines()[-1:]}")
            continue
        results[side] = json.loads(res.stdout.strip().splitlines()[-1])

    slower = []
    if len(results) == 2:
        for key in results["head"]:
            b, h = results["base"][key], results["head"][key]
            if h["p95"] > 1.5 * b["p95"]:
                slower.append(f"{key}: p95 {b['p95']} -> {h['p95']} ms")
            if h.get("desk_changes", 0) > b.get("desk_changes", 0):
                slower.append(f"{key}: desk changes {b['desk_changes']} -> {h['desk_changes']}")
            if h.get("failed", 0) > b.get("failed", 0):
                slower.append(f"{key}: samples with no fit {b['failed']} -> {h['failed']}")
    ok = not failures and not slower
    lib.write_summary(run, {"base": {"ref": args.base, "sha": base_sha}, "head": lib.head_info(),
                            "config": args.config, "results": results, "failures": failures,
                            "regressions": slower, "ok": ok})
    for key in ("sample", "camera_drag", "lamp_drag"):
        row = [f"{key:>12}"]
        for side in ("base", "head"):
            r = results.get(side, {}).get(key)
            if r:
                extra = f", desk changes {r['desk_changes']}" if "desk_changes" in r else f", no fit {r['failed']}"
                row.append(f"{side} p50 {r['p50']} / p95 {r['p95']} / max {r['max']} ms{extra}")
        print("  |  ".join(row))
    for line in failures + slower:
        print("  " + line)
    print(f"{'ok' if ok else 'FAILED'}: {run}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
