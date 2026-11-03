"""Renders the current config for the editor: fast previews and full renders."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ConfigError, build_config, resolve_config
from ..fx.composite import background_layer
from ..labels import SourceBox, page_corners, transform_boxes
from ..pipeline import (camera_effects, job_seed, load_rgba, pick_mode, sample_scene, scan_crop, scan_sample,
                        uses_gpu_effects)
from ..render import Renderer
from ..scene import PageCache, Scene, build_scene


@dataclass
class Frame:
    image: np.ndarray  # (H, W, 3) uint8, final look; (H, W, 4) with the background off
    label: np.ndarray  # (H, W, 4) label buffer of the photo (before any scanner-app crop)
    labels: list[dict[str, Any]]  # transformed boxes, in this frame's pixels
    page_corners: list | None
    resolved: dict[str, Any]
    scene: Scene | None  # None in scan mode: no 3D scene
    warning: str = ""  # e.g. the page leaves the frame (the CLI would resample)


# Effect settings measured in pixels; they scale with the preview size.
_PIXEL_KEYS = [("fx", "defocus", "strength_px"), ("fx", "motion_blur", "length_px"),
               ("fx", "noise", "grain_px"), ("fx", "sharpen", "radius_px"), ("background", "blur_px")]


class Preview:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer
        self._images: dict[str, np.ndarray] = {}
        self._pinned: dict[str, np.ndarray] = {}
        # Dragging the lamp, the camera or a hand keeps the page and the
        # background, so both are reused instead of built again.
        self.page_cache = PageCache()
        self._bg: tuple[str, np.ndarray] | None = None

    def add_image(self, key: str, rgba: np.ndarray) -> None:
        """Register an in-memory image (kept until replaced)."""
        self._pinned[key] = rgba

    def image(self, path: str | None) -> np.ndarray | None:
        if path is None:
            return None
        if path in self._pinned:
            return self._pinned[path]
        if path not in self._images:
            if len(self._images) > 16:
                self._images.clear()
            self._images[path] = load_rgba(path)
        return self._images[path]

    def render(self, user: dict[str, Any], base_dir: Path, doc_path: str, boxes: list[SourceBox],
               variant: int, max_size: int | None, mesh: int | None, effects: bool) -> Frame:
        """Render one sample of the config.

        `variant` picks the same sample the CLI makes for that variant, so a
        full render here matches `recapture render`. `max_size` and `mesh` make
        a cheaper preview: smaller image, coarser page.
        """
        cfg = build_config(user, base_dir=base_dir)
        doc = self.image(doc_path)
        src_size = (doc.shape[1], doc.shape[0])
        seed = job_seed(int(cfg["seed"]), Path(doc_path).name, variant)
        scale = 1.0
        if max_size:
            w, h = (int(v) for v in cfg["camera"]["output_px"])
            scale = min(1.0, max_size / max(w, h))
            cfg["camera"]["output_px"] = [max(16, round(w * scale)), max(16, round(h * scale))]
        if mesh:
            cfg["page"]["mesh_resolution"] = mesh
        cfg = pick_mode(cfg, seed)
        if cfg["mode"] == "scan":
            return self._scan(cfg, doc, boxes, seed, max_size)
        warning = ""
        try:
            resolved, scene = sample_scene(cfg, src_size, np.random.default_rng(seed), self.page_cache)
        except ConfigError as e:
            if "frame check" not in str(e):
                raise
            # Show the scene anyway; the CLI would reject this sample.
            warning = str(e).split(" (last: ")[-1].split(")")[0]
            resolved = resolve_config(cfg, np.random.default_rng(seed))
            scene = build_scene(resolved, src_size, page_cache=self.page_cache)
        resolved["seed"] = seed

        fx = copy.deepcopy(resolved["fx"])
        bg_cfg = copy.deepcopy(resolved["background"])
        if not effects:
            fx["enabled"] = False
        for section, *keys in _PIXEL_KEYS:
            node = fx if section == "fx" else bg_cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = float(node[keys[-1]]) * scale
        out = self.renderer.render(scene, doc, doc_key=doc_path, read_color=not uses_gpu_effects(fx, bg_cfg))
        image = camera_effects(self.renderer, scene, out, fx, bg_cfg, self.image(bg_cfg["images"]),
                               lambda size: self._background(bg_cfg, (size[1], size[0]), scene), scale)
        labels = transform_boxes(boxes, out.label, scene, src_size) if boxes else []
        crop = copy.deepcopy(resolved["crop"])
        if crop["size_px"] is not None:
            crop["size_px"] = [max(16, round(v * scale)) for v in crop["size_px"]]
        image, _, corners, labels = scan_crop(image, scene, crop, page_corners(scene), labels)
        return Frame(image=image, label=out.label, labels=labels, page_corners=corners,
                     resolved=resolved, scene=scene, warning=warning)

    def _scan(self, cfg: dict[str, Any], doc: np.ndarray, boxes: list[SourceBox], seed: int,
              max_size: int | None) -> Frame:
        """Scan mode. A preview scans at a lower dpi, so it comes out at most `max_size` pixels."""
        resolved, image, _, corners, labels = scan_sample(cfg, doc, boxes, np.random.default_rng(seed),
                                                          max_size=max_size)
        resolved["seed"] = seed
        return Frame(image=image, label=np.zeros((1, 1, 4), np.float32), labels=labels if boxes else [],
                     page_corners=corners, resolved=resolved, scene=None)

    def _background(self, bg_cfg: dict[str, Any], shape: tuple[int, int], scene: Scene) -> np.ndarray:
        key = json.dumps([bg_cfg, list(shape)], sort_keys=True)
        if self._bg is None or self._bg[0] != key:
            img, size = self.image(bg_cfg["images"]), (shape[1], shape[0])
            layer = background_layer(img, bg_cfg, size)
            self._bg = (key, layer)
        return self._bg[1]
