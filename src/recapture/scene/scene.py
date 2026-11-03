"""Scene description built from a resolved config. Pure data, no GPU."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import Camera, fit_camera, make_camera
from .deform import deform_page
from .surface import crease_occlusion
from .mesh import PageMesh, make_page_mesh, page_dimensions, place_page
from .desk import DeskMesh, build_items
from .occluders import Occluder, place_occluder, place_occluder_at


@dataclass(frozen=True)
class Light:
    position: np.ndarray  # (3,) cm
    color: np.ndarray  # (3,) linear RGB, luminance 1
    intensity: float  # irradiance factor at `ref_distance`
    ref_distance: float  # light to page center, cm
    size: float  # cm, sets shadow softness
    ambient: float
    ambient_color: np.ndarray
    ambient_direction: np.ndarray  # unit vector
    ambient_focus: float
    wrap: float


@dataclass(frozen=True)
class Material:
    tint: np.ndarray
    back_color: np.ndarray
    roughness: float
    specular: float
    crease_shading: float  # darkening along valley creases, 0..1
    grain: float
    grain_tile_cm: float
    grain_offset: np.ndarray  # (2,) tile fractions, so pages differ
    corner_radius_cm: float = 0.0
    glare_strength: float = 0.0
    glare_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    glare_cos: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))  # (outer, inner) cosines
    hologram: float = 0.0
    glare_panel: bool = False  # a rectangular panel or window instead of a round cone
    glare_u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))  # panel axes
    glare_v: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    glare_half: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))  # tangents of the half angles
    glare_edge: float = 0.0
    waviness: float = 0.0
    room_reflection: float = 0.0
    wear: float = 0.0


def _card_material(mc: dict) -> dict:
    """card-look: rounded corners, laminate glare direction and cone, hologram strength."""
    g = mc["glare"]
    el, az = np.radians(float(g["elevation_deg"])), np.radians(float(g["azimuth_deg"]))
    size = np.radians(max(0.5, float(g["size_deg"])))
    soft = float(np.clip(g["softness"], 0.0, 1.0))
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    # Panel axes: across and along the panel, turned by rotation_deg about its direction.
    a = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    rot = np.radians(float(g["rotation_deg"]))
    u, v = np.cos(rot) * a + np.sin(rot) * b, -np.sin(rot) * a + np.cos(rot) * b
    half = np.tan(min(size, np.radians(80.0)))
    return dict(
        glare_panel=g["shape"] == "panel",
        glare_u=u, glare_v=v,
        glare_half=np.array([half * max(0.1, float(g["aspect"])), half]),
        glare_edge=0.02 + 0.2 * soft + 0.4 * float(np.clip(mc["roughness"], 0.0, 1.0)),
        waviness=max(0.0, float(g["waviness"])),
        room_reflection=max(0.0, float(mc["room_reflection"])),
        wear=float(np.clip(mc["wear"], 0.0, 1.0)),
        corner_radius_cm=max(0.0, float(mc["corner_radius_cm"])),
        glare_strength=max(0.0, float(g["strength"])),
        glare_dir=d,
        glare_cos=np.array([np.cos(size * (1.0 + soft)), np.cos(size * (1.0 - 0.9 * soft))]),
        hologram=max(0.0, float(mc["hologram"])),
    )


@dataclass(frozen=True)
class Table:
    """The table under the page (the plane z = 0). It gets the lamp and the
    shadows; the background picture is its color."""
    contact: float  # contact shadow strength, 0..1
    contact_radius_cm: float
    albedo: np.ndarray  # linear RGB, for the editor's 3D view
    # A desk: the table pass gives the light reflected by the desk top (desk mode).
    desk: bool = False
    desk_image: np.ndarray | None = None  # RGBA uint8 desk texture; None = plain desk_color
    desk_size_cm: tuple[float, float] = (100.0, 100.0)  # one copy of the texture on the desk
    desk_offset: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2))
    desk_rotation_deg: float = 0.0
    desk_color: np.ndarray = dataclasses.field(default_factory=lambda: np.full(3, 0.6))


@dataclass(frozen=True)
class Scene:
    page: PageMesh
    page_normals: np.ndarray  # (N, 3) from the final geometry
    page_ao: np.ndarray  # (N,) crease occlusion from the geometry, 1 = open
    page_center: np.ndarray
    material: Material
    light: Light
    occluders: list[Occluder]
    camera: Camera
    table: Table | None = None  # None: no table (a flat or no background)
    props: list[DeskMesh] = dataclasses.field(default_factory=list)  # desk items beside the page


class _Recent:
    """The last `size` values built by key."""

    def __init__(self, size: int):
        self.size = size
        self._items: dict[str, Any] = {}

    def get(self, key: str, build) -> Any:
        if key in self._items:
            self._items[key] = self._items.pop(key)  # most recent last
        else:
            self._items[key] = build()
            while len(self._items) > self.size:
                self._items.pop(next(iter(self._items)))
        return self._items[key]


class PageCache(_Recent):
    """Keeps the last few built pages. A page depends only on the page and
    deform settings, so moving the lamp, the camera or a hand can reuse it
    (the editor does this while you drag). `shapes` keeps the coarse bent
    pages the frame check tries, apart, so a retry with new folds cannot
    push out the pages."""

    def __init__(self, size: int = 4):
        super().__init__(size)
        self.shapes = _Recent(8)


def build_scene(cfg: dict[str, Any], doc_size_px: tuple[int, int], mesh_resolution: int | None = None,
                page_cache: PageCache | None = None, props: bool = True) -> Scene:
    """`cfg` must be resolved (no random specs left). `props` false leaves
    out the desk items.

    `mesh_resolution` overrides the config, e.g. for a cheap preview.
    """
    pc, lc, cc, mc = cfg["page"], cfg["light"], cfg["camera"], cfg["material"]
    resolution = int(mesh_resolution or pc["mesh_resolution"])
    position, rotation, lift = pc["position_cm"], float(pc["rotation_deg"]), float(pc["lift_cm"])

    def build_page():
        w_cm, h_cm = page_dimensions(pc["size_cm"], doc_size_px)
        shaped = deform_page(make_page_mesh(w_cm, h_cm, resolution), cfg["deform"])
        page = place_page(shaped, position, rotation, lift)
        normals = page.normals()
        return page, normals, crease_occlusion(page, normals)  # placing is rigid, so the AO is the same

    if page_cache is None:
        page, normals, ao = build_page()
    else:
        key = json.dumps([pc, cfg["deform"], list(doc_size_px), resolution], sort_keys=True)
        page, normals, ao = page_cache.get(key, build_page)
    material = Material(
        tint=np.asarray(mc["tint"], dtype=np.float64),
        back_color=np.asarray(mc["back_color"], dtype=np.float64),
        roughness=float(np.clip(mc["roughness"], 0.05, 1.0)),
        specular=float(mc["specular"]),
        crease_shading=float(mc["crease_shading"]),
        grain=float(mc["grain"]),
        grain_tile_cm=float(mc["grain_tile_cm"]),
        grain_offset=np.random.default_rng(int(cfg["deform"]["crumple"]["seed"])).uniform(0, 1, 2),
        **_card_material(mc),
    )
    center = np.array([position[0], position[1], 0.0])

    bc = cfg["background"]
    light_pos = lamp_position(lc, center)
    color = kelvin_to_rgb(float(lc["color_temp_k"]))
    light = Light(
        position=light_pos,
        color=color,
        intensity=float(lc["intensity"]),
        ref_distance=float(np.linalg.norm(light_pos - center)),
        size=max(float(lc["size_cm"]), 0.0),
        ambient=float(lc["ambient"]),
        ambient_color=color if lc["ambient_color"] is None else np.asarray(lc["ambient_color"], dtype=np.float64),
        ambient_direction=np.asarray(lc["ambient_direction"], dtype=np.float64)
        / np.linalg.norm(lc["ambient_direction"]),
        ambient_focus=float(np.clip(lc["ambient_focus"], 0.0, 1.0)),
        wrap=float(lc["wrap"]),
    )
    camera = scene_camera(cfg, page)
    occluders = [
        place_occluder_at(o["shape"], np.asarray(o["position_cm"], dtype=np.float64), camera.position,
                          float(o["scale"]), float(o["rotation_offset_deg"]))
        if o["position_cm"] is not None else
        place_occluder(o["shape"], page.sample(np.asarray(o["target"], dtype=np.float64)[None])[0],
                       light_pos, camera.position, float(o["height_cm"]), float(o["scale"]),
                       float(o["rotation_offset_deg"]))
        for o in cfg["occluders"]
    ]
    dc = cfg["desk"]
    desk = bool(bc["enabled"] and dc["enabled"])
    desk_items: list[DeskMesh] = []
    table = None
    if bc["enabled"] and (bc["lit"] or desk):
        color = np.asarray(bc["color"], dtype=np.float64)
        extra: dict[str, Any] = {}
        if desk:
            seed = int(cfg["deform"]["crumple"]["seed"])
            items = dc["items"] if props else []
            desk_items = build_items(items, center, float(pc["rotation_deg"]), (page.width_cm, page.height_cm),
                                seed, _small_image)
            image, size = _desk_texture(dc["surface"]) if dc["surface"] else (None, (100.0, 100.0))
            offset = dc["texture_offset"]
            extra = dict(desk=True, desk_image=image, desk_size_cm=size, desk_rotation_deg=float(dc["texture_rotation_deg"]),
                         desk_color=np.asarray(dc["color"], dtype=np.float64),
                         desk_offset=np.random.default_rng(seed + 13).uniform(0, 1, 2) if offset is None
                         else np.asarray(offset, dtype=np.float64))
        table = Table(contact=float(np.clip(bc["contact_shadow"], 0.0, 1.0)),
                      contact_radius_cm=max(float(bc["contact_radius_cm"]), 0.05),
                      albedo=float(bc["brightness"]) * 0.8 * color / max(float(color.mean()), 1e-6), **extra)
    return Scene(page=page, page_normals=normals, table=table, props=desk_items,
                 page_ao=(1.0 - float(mc["crease_shading"]) * (1.0 - ao)).astype(np.float32),
                 page_center=center, material=material,
                 light=light, occluders=occluders, camera=camera)


def scene_camera(cfg: dict[str, Any], page: PageMesh) -> Camera:
    """The photo camera of a resolved config, for the placed `page` (fit_page and the crop aim at it)."""
    cc = cfg["camera"]
    rotation = float(cfg["page"]["rotation_deg"])
    w_px, h_px = (int(v) for v in cc["output_px"])
    if cc["fit_page"] or cfg["crop"]["enabled"]:
        margin = float(cc["fit_margin"])
        if cfg["crop"]["enabled"]:  # keep the crop's margin and corner error inside the photo
            margin += max(float(cfg["crop"]["margin"]), 0.0) + float(cfg["crop"]["corner_error"])
        axis = np.array([np.cos(np.radians(rotation)), np.sin(np.radians(rotation)), 0.0])
        return fit_camera(page.sample(page.boundary_uv(64)), cc["position_cm"], cc["target_cm"],
                          float(cc["roll_deg"]), float(cc["focal_mm"]), w_px, h_px, min(margin, 0.4), axis)
    return make_camera(cc["position_cm"], cc["target_cm"], float(cc["roll_deg"]), float(cc["focal_mm"]), w_px, h_px)


def quick_frame_check(cfg: dict[str, Any], doc_size_px: tuple[int, int], resolution: int, coverage: list[float],
                      margin: float, page_cache: PageCache | None = None) -> tuple[bool, str]:
    """`frame_check` on a coarse page, without building a scene. The bent page depends only on its size and
    the deformations, so it is cached apart from where it lies: trying another placement or camera only moves
    it. The placed page is the same as `build_scene` makes at this resolution."""
    pc = cfg["page"]

    def shape():
        w_cm, h_cm = page_dimensions(pc["size_cm"], doc_size_px)
        return deform_page(make_page_mesh(w_cm, h_cm, resolution), cfg["deform"])

    if page_cache is None:
        shaped = shape()
    else:
        key = json.dumps(["shape", pc["size_cm"], cfg["deform"], list(doc_size_px), resolution], sort_keys=True)
        shaped = page_cache.shapes.get(key, shape)
    page = place_page(shaped, pc["position_cm"], float(pc["rotation_deg"]), float(pc["lift_cm"]))
    return _frame_check(page, scene_camera(cfg, page), coverage, margin)


@lru_cache(maxsize=16)
def _desk_texture(folder: str) -> tuple[np.ndarray, tuple[float, float]]:
    """A desk texture (RGBA uint8) and the desk size one copy of it covers (cm)."""
    image = np.asarray(Image.open(Path(folder) / "albedo.jpg").convert("RGBA"))
    info = Path(folder) / "info.json"
    size = tuple(json.loads(info.read_text())["size_cm"]) if info.exists() else (100.0, 100.0)
    return image, size


@lru_cache(maxsize=32)
def _small_image(path: str) -> np.ndarray:
    """An image printed on a desk item, at most 1024 px: only part of it shows."""
    im = Image.open(path).convert("RGBA")
    im.thumbnail((1024, 1024))
    return np.asarray(im)


def lamp_position(lc: dict[str, Any], page_center: np.ndarray) -> np.ndarray:
    """Lamp position from `position_cm`, or from elevation, azimuth and distance."""
    if lc["position_cm"] is not None:
        return np.asarray(lc["position_cm"], dtype=np.float64)
    el, az = np.radians(float(lc["elevation_deg"])), np.radians(float(lc["azimuth_deg"]))
    d = float(lc["distance_cm"])
    return page_center + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def frame_check(scene: Scene, coverage: list[float], margin: float) -> tuple[bool, str]:
    """Check that the whole page is in frame and covers a set part of it."""
    return _frame_check(scene.page, scene.camera, coverage, margin)


def _frame_check(page: PageMesh, cam: Camera, coverage: list[float], margin: float) -> tuple[bool, str]:
    pts = page.sample(page.boundary_uv(64))
    px, w = cam.project(pts)
    if np.any(w <= cam.near):
        return False, "page is behind or too close to the camera"
    m = margin * min(cam.width, cam.height)
    if px[:, 0].min() < m or px[:, 1].min() < m or px[:, 0].max() > cam.width - m or px[:, 1].max() > cam.height - m:
        return False, "page is not fully inside the frame"
    area = abs(polygon_area(px)) / (cam.width * cam.height)
    if not coverage[0] <= area <= coverage[1]:
        return False, f"page covers {area:.2f} of the frame, outside {coverage}"
    return True, f"coverage {area:.2f}"


def polygon_area(pts: np.ndarray) -> float:
    """Signed shoelace area. Positive means clockwise on screen (y down)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Approximate black-body color as linear RGB, scaled to luminance 1."""
    t = np.clip(kelvin, 1000.0, 40000.0) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    srgb = np.clip(np.array([r, g, b]) / 255.0, 0.0, 1.0)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return lin / float(lin @ np.array([0.2126, 0.7152, 0.0722]))
