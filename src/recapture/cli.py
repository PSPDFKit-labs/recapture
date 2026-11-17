"""Command line: recapture render | preview | export | init | fetch-assets | defaults | validate | sample | edit."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__
from .config import ConfigError, build_config, load_config
from .config.schema import IMAGE_SUFFIXES
from .export import FORMATS
from .pipeline import Job, Pipeline, job_seed, sample_scene

EXAMPLES = ("photo", "scan", "mix", "card")

_PIPELINE: Pipeline | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recapture", description="Turn clean document images into phone photos and scans, with exact layout labels.")
    parser.add_argument("--version", action="version", version=f"recapture {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render images")
    r.add_argument("--input", "-i", required=True, type=Path, help="an image file or a directory of images")
    r.add_argument("--config", "-c", type=Path, help="config JSON (default: built-in defaults)")
    r.add_argument("--out", "-o", required=True, type=Path, help="output directory")
    r.add_argument("--variants", "-n", type=int, default=1, help="renders per source image")
    r.add_argument("--seed", type=int, help="base seed (default: the config 'seed')")
    r.add_argument("--workers", "-j", type=int, default=1, help="parallel worker processes")
    r.add_argument("--format", choices=["jpg", "png"], help="override output.format")
    r.add_argument("--recursive", action="store_true", help="search sub-directories for images")
    r.add_argument("--skip-existing", action="store_true", help="skip jobs whose label file exists")

    p = sub.add_parser("preview", help="render a few images onto one contact sheet, with a text summary")
    p.add_argument("config", type=Path, help="config JSON")
    p.add_argument("--input", "-i", required=True, type=Path, help="an image file or a directory of images")
    p.add_argument("--count", "-n", type=int, default=6, help="images on the sheet")
    p.add_argument("--out", "-o", type=Path, default=Path("preview.jpg"), help="sheet file (default: preview.jpg)")
    p.add_argument("--seed", type=int, help="base seed (default: the config 'seed')")

    x = sub.add_parser("export", help="turn rendered label files into a training format")
    x.add_argument("dataset", type=Path, help="the output folder of `recapture render`")
    x.add_argument("--out", "-o", required=True, type=Path, help="export folder")
    x.add_argument("--format", "-f", choices=FORMATS, default="coco")
    x.add_argument("--val", type=float, default=0.1, help="part of the documents for validation (default 0.1)")
    x.add_argument("--test", type=float, default=0.0, help="part of the documents for testing (default 0)")
    x.add_argument("--seed", type=int, default=0, help="seed of the split")
    x.add_argument("--classes", nargs="+", help="class names in order (default: every label found, sorted)")
    x.add_argument("--link", action="store_true", help="hard-link the images instead of copying them")

    sp = sub.add_parser("split-pages", help="cut a document's API response into one box file per page")
    sp.add_argument("response", type=Path, help="the API response JSON of the whole document")
    sp.add_argument("--out", "-o", required=True, type=Path, help="folder for the page box files")
    sp.add_argument("--name", help="file name before the page number (default: the response's name)")

    i = sub.add_parser("init", help="write a starting config from an example")
    i.add_argument("config", type=Path, help="config file to write")
    i.add_argument("--from", dest="example", choices=EXAMPLES, default="mix",
                   help="photo (phone photos), scan (flatbed scans) or mix (both; default)")
    i.add_argument("--force", action="store_true", help="overwrite an existing file")

    f = sub.add_parser("fetch-assets", help="download the CC0 assets once (about 105 MB)")
    f.add_argument("--yes", "-y", action="store_true", help="do not ask before downloading")
    f.add_argument("--dest", type=Path, help="asset folder (default: $RECAPTURE_ASSETS, the repo's assets/ "
                                             "in a clone, or ~/.cache/recapture/assets)")
    f.add_argument("--res", default="2k", help="texture resolution: 1k, 2k or 4k (default 2k)")

    sub.add_parser("defaults", help="print every config key with its default and what it does")

    v = sub.add_parser("validate", help="check a config file")
    v.add_argument("config", type=Path)

    e = sub.add_parser("edit", help="open the config editor")
    e.add_argument("config", type=Path, nargs="?", help="config JSON to open")
    e.add_argument("--doc", type=Path, help="document image to preview")

    s = sub.add_parser("sample", help="print one resolved config")
    s.add_argument("config", type=Path)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--doc-size", type=int, nargs=2, default=[2480, 3508], metavar=("W", "H"))

    args = parser.parse_args(argv)
    try:
        if args.cmd == "render":
            return _render(args)
        if args.cmd == "edit":
            try:
                from .editor.app import run as run_editor
            except ImportError as err:
                print(f"the editor needs extra packages: pip install 'recapture[editor]' ({err})", file=sys.stderr)
                return 1
            run_editor(args.config, args.doc)
            return 0
        if args.cmd == "preview":
            return _preview(args)
        if args.cmd == "export":
            from .export import export

            result = export(args.dataset, args.out, args.format, args.val, args.test, args.seed, args.classes, args.link)
            for split, n in result["splits"].items():
                print(f"{split}: {n['images']} images from {n['documents']} documents, {n['boxes']} boxes"
                      + (f" ({n['skipped_boxes']} without a quad or class left out)" if n["skipped_boxes"] else ""))
            print(f"classes: {', '.join(result['classes'])}\nwritten to {args.out}")
            sys.stdout.flush()  # the summary first, then any warning
            for split, part in (("val", args.val), ("test", args.test)):
                if part > 0 and split not in result["splits"]:
                    docs = sum(n["documents"] for n in result["splits"].values())
                    print(f"warning: no {split} split: {docs} documents is too few for --{split} {part} "
                          "(the split counts whole documents)", file=sys.stderr)
            return 0
        if args.cmd == "split-pages":
            return _split_pages(args)
        if args.cmd == "init":
            return _init(args)
        if args.cmd == "fetch-assets":
            return _fetch_assets(args)
        if args.cmd == "defaults":
            print(defaults_text())
            return 0
        if args.cmd == "validate":
            load_config(args.config)
            print(f"{args.config}: OK")
            return 0
        if args.cmd == "sample":
            cfg = load_config(args.config)
            resolved, _ = sample_scene(cfg, tuple(args.doc_size), np.random.default_rng(args.seed))
            print(json.dumps(resolved, indent=1))
            return 0
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:  # bad export arguments or an empty dataset
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1


def defaults_text() -> str:
    """The commented default blocks of config/schema.py, as written there (Python
    literals: None is null, True and False are true and false in JSON)."""
    from .config import schema

    src = Path(schema.__file__).read_text()
    start = src.index("DEFAULTS: dict[str, Any] = {")
    end = src.index("}\n", src.index("OCCLUDER_DEFAULTS: dict[str, Any] = {")) + 2
    return src[start:end]


def _preview(args) -> int:
    from .sheet import preview

    cfg = load_config(args.config)
    docs = _find_images(args.input, False)
    if not docs:
        print(f"no images found in {args.input}", file=sys.stderr)
        return 1
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    preview(cfg, docs, max(1, args.count), args.out, seed)
    return 0


def _split_pages(args) -> int:
    from .labels import api_page_size, split_api_pages

    data = json.loads(args.response.read_text())
    parts = split_api_pages(data)
    # Name pages the way `pdftoppm -png doc.pdf <out>/<name>` does: <name>-1, or <name>-01 from 10 pages on.
    count = max([(data.get("metrics") or {}).get("pagesProcessed") or 0, *(i + 1 for i in parts)])
    name = args.name or args.response.stem
    args.out.mkdir(parents=True, exist_ok=True)
    for i, part in parts.items():
        (args.out / f"{name}-{i + 1:0{len(str(count))}d}.json").write_text(json.dumps(part, indent=1))
    print(f"wrote {len(parts)} box files for {count} pages into {args.out}. Render the pages next to them with:\n"
          f"  pdftoppm -r 200 -png <document>.pdf '{args.out / name}'")
    sizes = {s for s in map(api_page_size, parts.values()) if s}
    if sizes:
        print("Coordinates are for " + ", ".join(f"{w}x{h}" for w, h in sorted(sizes)) + " pages; any resolution "
              "with that shape works.")
    return 0


def _init(args) -> int:
    if args.config.exists() and not args.force:
        print(f"{args.config} exists; pass --force to overwrite it", file=sys.stderr)
        return 1
    text = (resources.files("recapture") / "examples" / f"{args.example}.json").read_text()
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(text)
    print(f"wrote {args.config} (from the {args.example} example). Edit it by hand, or with "
          f"`recapture edit {args.config}`; check it with `recapture preview {args.config} -i <docs>`.")
    return 0


def _fetch_assets(args) -> int:
    from .assets import DOWNLOAD_MB, KEPT_MB, asset_home, fetch

    home = args.dest or asset_home()
    print(f"CC0 assets (Poly Haven, ambientCG): about {DOWNLOAD_MB} MB to download, {KEPT_MB} MB kept, into {home}.\n"
          "Files already there are skipped.")
    if not args.yes:
        if not sys.stdin.isatty():
            print("pass --yes to download without asking", file=sys.stderr)
            return 1
        if input("Download? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    total, failed = fetch(home, res=args.res)
    print(f"done, {total / 1e6:.1f} MB downloaded" + (f", {failed} failed (run it again to retry)" if failed else ""))
    if args.dest:
        print(f"set RECAPTURE_ASSETS={home} so recapture finds them there")
    return 1 if failed else 0


def _render(args) -> int:
    cfg = load_config(args.config) if args.config else build_config({})
    if args.format:
        cfg["output"]["format"] = args.format
    base_seed = args.seed if args.seed is not None else int(cfg["seed"])
    sources = _find_images(args.input, args.recursive)
    if not sources:
        print(f"no images found in {args.input}", file=sys.stderr)
        return 1

    root = args.input if args.input.is_dir() else args.input.parent
    jobs = []
    for src in sources:
        out_dir = args.out / src.parent.relative_to(root)
        source = src.relative_to(root).as_posix()  # the seed tells doc_a/page.png from doc_b/page.png
        for v in range(args.variants):
            job = Job(source=src, out_dir=out_dir, variant=v, seed=job_seed(base_seed, source, v))
            if args.skip_existing and (out_dir / f"{job.stem}.json").exists():
                continue
            jobs.append(job)
    print(f"{len(sources)} source(s), {len(jobs)} job(s), {args.workers} worker(s)", file=sys.stderr)

    start = time.perf_counter()
    failed = 0
    done = 0
    if args.workers <= 1:
        _init_worker(cfg)
        results = map(_run_job, jobs)
    else:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(cfg,))
        results = pool.imap_unordered(_run_job, jobs)
    for ok, message in results:
        done += 1
        if not ok:
            failed += 1
            print(f"\nFAILED {message}", file=sys.stderr)
        rate = done / (time.perf_counter() - start)
        print(f"\r{done}/{len(jobs)}  {rate:.1f} img/s  failed: {failed}", end="", file=sys.stderr)
    if args.workers > 1:
        pool.close()
        pool.join()
    print(file=sys.stderr)
    return 1 if failed else 0


def _find_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _init_worker(cfg) -> None:
    global _PIPELINE
    _PIPELINE = Pipeline(cfg)


def _run_job(job: Job) -> tuple[bool, str]:
    try:
        _PIPELINE.run(job)
        return True, job.stem
    except Exception as e:  # report and continue with the other jobs
        detail = str(e) if isinstance(e, ConfigError) else traceback.format_exc()
        return False, f"{job.source} variant {job.variant}: {detail}"


if __name__ == "__main__":
    sys.exit(main())
