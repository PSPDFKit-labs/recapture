"""One job: (source image, boxes, config, seed) -> photo, labels, resolved config."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import __version__
from .config import ConfigError, is_random, resolve, resolve_config
from .config.schema import frame_is_random, redraw_frame, resolve_with_base
from .crop import crop_transform, warp_image, warp_labels
from .fx import apply_fx
from .fx.composite import background_layer
from .fx.quality import cheap_camera
from .labels import load_boxes, page_corners, sidecar_path, transform_boxes
from .render import Renderer, RenderOutput
from .scene import PageCache, Scene, build_scene, frame_check, quick_frame_check

Image.MAX_IMAGE_PIXELS = 200_000_000


@dataclass(frozen=True)
class Job:
    """One image to make: the page `source`, the folder `out_dir`, and which `variant` with which `seed`.

    Give the seed from `job_seed` to get the same image `recapture render` makes for that page and variant.
    Boxes are read from the box file next to `source` (`page.json` for `page.png`) when there is one."""

    source: Path
    out_dir: Path
    variant: int
    seed: int

    @property
    def stem(self) -> str:
        return f"{self.source.stem}_{self.variant:03d}"


@dataclass
class JobResult:
    """What `Pipeline.run` made: the image path, the labels written next to it, the resolved config (every
    random value drawn) and the time each stage took, in seconds."""

    job: Job
    image_path: Path
    labels: dict[str, Any]
    resolved: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


def job_seed(base_seed: int, source: str, variant: int) -> int:
    """Stable seed per (base seed, page, variant). Same inputs, same seed.

    `source` is the page's path inside the input folder, with "/" between parts: `doc_a/page_001.png`. For a
    page at the top of the folder that is its file name, so flat folders keep the seeds they always had, and
    two pages with the same name in different folders get different photos."""
    h = hashlib.blake2b(f"{base_seed}|{source}|{variant}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") >> 1


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


_PREVIEW_MESH = 64


def sample_scene(cfg: dict[str, Any], doc_size: tuple[int, int], rng: np.random.Generator,
                 page_cache: PageCache | None = None) -> tuple[dict, Scene]:
    """Resolve the config until the page fits the frame rules.

    A retry draws only the camera and the page placement again (FRAME_PATHS), and from the second half of the
    tries the folds too (FOLD_PATHS); the desk, the lamp, the crumple and the items keep the first draw. With
    those values fixed, a retry cannot help: one try."""
    base = int(rng.integers(0, 2**63 - 1))
    tries = int(cfg["camera"]["max_tries"]) if frame_is_random(cfg) else 1
    reason = ""
    first = resolve_with_base(cfg, base)
    for attempt in range(max(tries, 1)):
        resolved = redraw_frame(first, cfg, base, attempt, folds=attempt >= tries // 2) if attempt else first
        coverage, margin = resolved["camera"]["frame_coverage"], float(resolved["camera"]["frame_margin"])
        if resolved["camera"]["fit_page"] or resolved["crop"]["enabled"]:  # the camera fits the page to the frame
            coverage, margin = [0.0, 1.0], 0.0
        # Test the sample on a coarse page first; the dense page is slow to build.
        ok, reason = quick_frame_check(resolved, doc_size, _PREVIEW_MESH, coverage, margin, page_cache)
        if not ok:
            continue
        scene = build_scene(resolved, doc_size, page_cache=page_cache)
        ok, reason = frame_check(scene, coverage, margin)
        if ok:
            return resolved, scene
    raise ConfigError(
        f"no sample passed the frame check in {tries} tries (last: {reason}). "
        "Move the camera ranges or widen camera.frame_coverage."
    )


def uses_gpu_effects(fx: dict[str, Any], bg_cfg: dict[str, Any]) -> bool:
    """The GPU runs the camera effects, except for a transparent background."""
    return fx["backend"] == "gpu" and bool(bg_cfg["enabled"])


def camera_effects(renderer: Renderer, scene: Scene, out: RenderOutput, fx: dict[str, Any], bg_cfg: dict[str, Any],
                   bg_image: np.ndarray | None, layer_for=None, pixel_scale: float = 1.0) -> np.ndarray:
    """The final photo: the render over its background, through the camera
    effects. `layer_for(size)` may give a cached background layer.
    `pixel_scale` scales effects measured in pixels (a smaller preview)."""
    h, w = out.label.shape[:2]
    desk = scene.table is not None and scene.table.desk
    layer = None
    if bg_cfg["enabled"] and not desk:
        layer = layer_for((w, h)) if layer_for is not None else background_layer(bg_image, bg_cfg, (w, h))
    focus = focus_depth(scene, fx)
    if uses_gpu_effects(fx, bg_cfg):
        mode = "desk" if desk else "lit" if scene.table is not None else "layer"
        tex = out.textures
        image = renderer.effects.apply(tex["color"], tex["label"], out.label, tex["table"], fx, focus, bg_cfg, layer,
                                       mode)
    else:
        image = apply_fx(out.color, out.alpha, out.label, fx, focus, bg_image, bg_cfg, bg_layer=layer,
                         table=out.table, desk=desk)
    if fx["enabled"]:
        image = cheap_camera(image, fx["quality"], pixel_scale)
    return image


def scan_crop(image: np.ndarray, scene: Scene, crop: dict[str, Any], corners: list | None,
              boxes: list[dict[str, Any]]) -> tuple[np.ndarray, tuple[int, int], list | None, list[dict[str, Any]]]:
    """With crop.enabled, cut the page out of the photo like a scanner app and
    map the labels with it. Returns the image, its size, the page corners and the boxes."""
    size = (image.shape[1], image.shape[0])
    if not crop["enabled"]:
        return image, size, corners, boxes
    found = crop_transform(scene, crop, int(crop["seed"]))
    if found is None:
        raise ConfigError("crop: a page corner is behind the camera")
    h, size = found
    boxes, corners = warp_labels(boxes, corners, h, size)
    return warp_image(image, h, size), size, corners, boxes


def focus_depth(scene: Scene, fx: dict[str, Any]) -> float | None:
    """Camera depth (cm) of the page point the camera focuses on."""
    uv = np.asarray(fx["defocus"]["focus"], dtype=np.float64)[None]
    _, w = scene.camera.project(scene.page.sample(uv))
    return float(w[0]) if w[0] > scene.camera.near else None


def pick_mode(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """The config with `mode` fixed for this job. A random mode (a dataset of
    photos and scans) is drawn from its own stream, so a fixed mode gives the
    same samples as before."""
    if not is_random(cfg["mode"]):
        return cfg
    out = dict(cfg)
    out["mode"] = resolve(cfg["mode"], np.random.default_rng([seed, 0x5CA2]))
    return out


def scan_sample(cfg: dict[str, Any], doc: np.ndarray, boxes: list, rng: np.random.Generator,
                max_size: int | None = None) -> tuple[dict[str, Any], np.ndarray, tuple[int, int], list,
                                                      list[dict[str, Any]]]:
    """Scan mode: resolve the config once (no frame to check) and scan the page.
    `max_size` lowers the dpi so the long side stays within it (a preview)."""
    from .scanner import scan_document
    from .scene.mesh import page_dimensions

    resolved = resolve_config(cfg, rng)
    if max_size:
        long_px = float(resolved["scanner"]["dpi"]) / 2.54 * max(
            page_dimensions(resolved["page"]["size_cm"], (doc.shape[1], doc.shape[0])))
        if long_px > max_size:
            scanner = dict(resolved["scanner"])
            scanner["dpi"] = float(scanner["dpi"]) * max_size / long_px
            resolved = {**resolved, "scanner": scanner}
    image, size, corners, labels = scan_document(doc, boxes, resolved["scanner"], resolved["page"]["size_cm"],
                                                 resolved["printer"])
    return resolved, image, size, corners, labels


class Pipeline:
    """Makes images from one config. Build it once and call `run` for each job.

    `cfg` is a config from `load_config` or `build_config`. Photo mode makes an OpenGL context on the first
    photo job, so it needs a GPU or EGL; scan mode needs none. One Pipeline is not safe to share across
    threads; use one per process."""

    def __init__(self, cfg: dict[str, Any], renderer: Renderer | None = None):
        self.cfg = cfg
        self._renderer = renderer
        self._bg_cache: tuple[str, np.ndarray] | None = None

    @property
    def renderer(self) -> Renderer:
        """Made on first use: scan mode needs no OpenGL."""
        if self._renderer is None:
            self._renderer = Renderer()
        return self._renderer

    def run(self, job: Job) -> JobResult:
        """Make one image, write it with its label file and resolved config into `job.out_dir`."""
        t = {"start": time.perf_counter()}
        doc = load_rgba(job.source)
        src_size = (doc.shape[1], doc.shape[0])
        boxes_file = sidecar_path(job.source)
        boxes = load_boxes(boxes_file, src_size) if boxes_file.exists() else []
        t["load"] = time.perf_counter()

        rng = np.random.default_rng(job.seed)
        cfg = pick_mode(self.cfg, job.seed)
        if cfg["mode"] == "scan":
            resolved, image, size, corners, boxes_out = scan_sample(cfg, doc, boxes, rng)
            resolved["seed"] = job.seed
            t["scan"] = time.perf_counter()
            labels = {"source": job.source.name, "image": "", "size": list(size), "page_corners": corners,
                      "boxes": boxes_out}
            image_path = self._write(job, image, labels, resolved)
            t["write"] = time.perf_counter()
            keys = list(t)
            timings = {k: round(t[k] - t[p], 4) for p, k in zip(keys, keys[1:])}
            return JobResult(job=job, image_path=image_path, labels=labels, resolved=resolved, timings=timings)

        resolved, scene = sample_scene(cfg, src_size, rng)
        resolved["seed"] = job.seed
        bg_path = resolved["background"]["images"]
        bg = self._background(bg_path) if resolved["background"]["enabled"] else None
        t["scene"] = time.perf_counter()

        fx, bg_cfg = resolved["fx"], resolved["background"]
        out: RenderOutput = self.renderer.render(scene, doc, read_color=not uses_gpu_effects(fx, bg_cfg))
        t["render"] = time.perf_counter()

        image = camera_effects(self.renderer, scene, out, fx, bg_cfg, bg)
        t["fx"] = time.perf_counter()

        image, size, corners, boxes_out = scan_crop(image, scene, resolved["crop"], page_corners(scene),
                                                    transform_boxes(boxes, out.label, scene, src_size))
        labels = {
            "source": job.source.name,
            "image": "",
            "size": list(size),
            "page_corners": corners,
            "boxes": boxes_out,
        }
        t["labels"] = time.perf_counter()

        image_path = self._write(job, image, labels, resolved)
        t["write"] = time.perf_counter()
        keys = list(t)
        timings = {k: round(t[k] - t[p], 4) for p, k in zip(keys, keys[1:])}
        return JobResult(job=job, image_path=image_path, labels=labels, resolved=resolved, timings=timings)

    def _background(self, path: str | None) -> np.ndarray | None:
        if path is None:
            return None
        if self._bg_cache is None or self._bg_cache[0] != path:
            self._bg_cache = (path, load_rgba(path))
        return self._bg_cache[1]

    def _write(self, job: Job, image: np.ndarray, labels: dict, resolved: dict) -> Path:
        return write_outputs(job.out_dir, job.stem, image, labels, resolved)


def write_outputs(out_dir: Path, stem: str, image: np.ndarray, labels: dict, resolved: dict) -> Path:
    """Write the photo, its labels and the resolved config. Returns the photo path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = resolved["output"]
    ext = "jpg" if out["format"] == "jpg" else "png"
    image_path = out_dir / f"{stem}.{ext}"
    pil = Image.fromarray(image)
    if ext == "jpg":
        pil.save(image_path, quality=int(out["jpeg_quality"]), subsampling=0)
    else:
        pil.save(image_path, compress_level=3)
    labels["image"] = image_path.name
    resolved = {**resolved, "recapture_version": __version__}
    (out_dir / f"{stem}.json").write_text(json.dumps(labels, indent=1))
    (out_dir / f"{stem}.config.json").write_text(json.dumps(resolved, indent=1))
    return image_path
