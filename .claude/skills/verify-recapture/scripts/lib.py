"""Shared helpers: the checkout, base worktrees, run folders, renders in a chosen source tree."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
FIXTURES = SKILL / "fixtures"


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd or root(), check=True, capture_output=True, text=True).stdout.strip()


def root() -> Path:
    """The checkout being verified: $VERIFY_ROOT, or the git top level of this skill."""
    if os.environ.get("VERIFY_ROOT"):
        return Path(os.environ["VERIFY_ROOT"]).resolve()
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=SKILL, check=True, capture_output=True, text=True)
    return Path(out.stdout.strip())


def python() -> str:
    """The checkout's .venv Python (every tree runs with it; PYTHONPATH picks the source)."""
    exe = root() / ".venv" / "bin" / "python"
    return str(exe) if exe.exists() else sys.executable


def head_info() -> dict:
    return {"sha": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def base_tree(ref: str) -> tuple[Path, str]:
    """A detached worktree of `ref` under .verify-runs/recapture/base/<sha>, made once and reused."""
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    path = root() / ".verify-runs" / "recapture" / "base" / sha[:12]
    if not (path / "src" / "recapture").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(path), sha)
    return path, sha


def run_dir(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root() / ".verify-runs" / "recapture" / "runs" / f"{stamp}-{name}"
    path.mkdir(parents=True)
    return path


def env_for(tree: Path) -> dict:
    """Run the CLI from `tree`'s source, with the assets of the checkout being verified."""
    env = dict(os.environ, PYTHONPATH=str(tree / "src"))
    assets = root() / "assets"
    if "RECAPTURE_ASSETS" not in env and assets.is_dir():
        env["RECAPTURE_ASSETS"] = str(assets)
    return env


def example_path(tree: Path, name_or_path: str) -> Path | None:
    """An example name (`photo`) in `tree`, or a config path (the same file for every tree)."""
    p = Path(name_or_path)
    if p.suffix == ".json":
        return p.resolve()
    ex = tree / "src" / "recapture" / "examples" / f"{name_or_path}.json"
    return ex if ex.exists() else None


def fixture_for(config: str) -> Path:
    """The card fixture for card configs, the A4 page for the rest."""
    return FIXTURES / ("card.png" if "card" in Path(config).stem else "page.png")


def stage_pages(dest: Path, pages: list[Path]) -> Path:
    """Copy page images and their box files into `dest` (the CLI takes a folder)."""
    dest.mkdir(parents=True, exist_ok=True)
    for page in pages:
        shutil.copy2(page, dest / page.name)
        box = page.with_suffix(".json")
        if box.exists():
            shutil.copy2(box, dest / box.name)
    return dest


def render(tree: Path, config: Path, pages: Path, out: Path, n: int, seed: int) -> dict:
    """`recapture render` with `tree`'s source. Returns the time and the CLI's last lines."""
    t = time.perf_counter()
    res = subprocess.run([python(), "-m", "recapture.cli", "render", "-i", str(pages), "-c", str(config),
                          "-o", str(out), "-n", str(n), "--seed", str(seed)],
                         env=env_for(tree), capture_output=True, text=True)
    return {"ok": res.returncode == 0, "seconds": round(time.perf_counter() - t, 2),
            "tail": (res.stdout + res.stderr).strip().splitlines()[-3:]}


def write_summary(path: Path, data: dict) -> None:
    (path / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
