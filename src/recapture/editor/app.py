"""Recapture editor.

Center: a 3D scene view, like Blender or Unity. Orbit, pan and zoom around
the page; click the lamp, the camera, its aim point or the hand shadow, and
move it with the arrows.
Right: the photo the camera takes, live.
Left: files and a short list of sliders. Each slider can become a random range.

Run with `recapture edit [config.json] [--doc page.png]`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
import traceback
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from imgui_bundle import hello_imgui, imgui, imguizmo, immapp
from imgui_bundle import portable_file_dialogs as pfd
from PIL import Image, ImageDraw, ImageFont

from ..assets import asset_home
from ..config import ConfigError, build_config
from ..config.values import SpecError, check_tree, spec_kind
from ..labels import load_boxes, sidecar_path
from ..pipeline import write_outputs
from ..render import Renderer
from ..scene.occluders import place_occluder
from . import state
from .advanced import AdvancedPanel
from .preview import Frame, Preview
from .view3d import OrbitCamera, frustum_lines, lamp_angles, occluder_anchor, project

gizmo = imguizmo.im_guizmo

# The folds the Folds slider adds: across the page at a third or half, a random angle.
RANDOM_FOLD = {"position": [0.5, {"choice": [0.333, 0.5, 0.667]}], "direction_deg": {"choice": [0, 180]},
               "angle_deg": {"uniform": [-20, 20]}}

# The desk textures `recapture fetch-assets` downloads, and the items the Desk box adds.
_ASSETS = asset_home()
DESK_TEXTURES = ["polyhaven/desks", "desks_plain"]  # in the asset folder: downloaded, generated
RANDOM_DESK_ITEMS = {
    "repeat": {"randint": [1, 3]},
    "item": {"kind": {"choice": ["keyboard", "mousepad", "phone", "mug", "pen", "pencil", "sticky_notes", "paper"],
                      "weights": [0.22, 0.12, 0.13, 0.11, 0.11, 0.09, 0.11, 0.11]},
             "side": {"choice": ["left", "right", "top", "bottom"]}, "along": {"uniform": [0.1, 0.9]},
             "gap_cm": {"uniform": [0.5, 5.0]}, "rotation_deg": {"normal": [0, 8], "clip": [-25, 25]}},
}

FAST = {"max_size": 640, "mesh": 160, "effects": False}  # while something moves
GOOD = {"max_size": 1024, "mesh": 512, "effects": True}  # when it stops
SAMPLE_DOC = "<sample page>"
LEFT_W, RIGHT_W = 360, 430
PICK_RADIUS = 14.0
VIEW_BG = (0.23, 0.235, 0.25)  # the 3D view's backdrop, display colors

COL_LAMP = imgui.IM_COL32(255, 210, 80, 255)
COL_CAM = imgui.IM_COL32(120, 190, 255, 255)
COL_CAM_DIM = imgui.IM_COL32(120, 190, 255, 110)
COL_HAND = imgui.IM_COL32(255, 120, 200, 255)
COL_GRID = imgui.IM_COL32(255, 255, 255, 28)
COL_SEL = imgui.IM_COL32(255, 255, 255, 255)


@dataclasses.dataclass
class Slider:
    label: str
    path: tuple
    lo: float
    hi: float
    fmt: str = "%.2f"
    integer: bool = False
    help: str = ""
    also: tuple = ()  # more paths that get the same value (e.g. all four page edges)
    count_of: tuple | None = None  # the drawn value is the length of this resolved list


LAMP_PLACE = {("light", "elevation_deg"), ("light", "azimuth_deg"), ("light", "distance_cm")}

# The quick panel. The crumple shape, folds and the hand shadow have their own controls.
SLIDER_GROUPS: list[tuple[str, list[Slider]]] = [
    ("Paper", [
        Slider("Crumple", ("deform", "crumple", "strength"), 0.0, 3.0,
               help="0 = flat paper, 1 = a normal crumple, 2 or more = heavy."),
        Slider("Folds", ("deform", "folds", "repeat"), 0, 4, "%d", integer=True, help="Number of folds.",
               count_of=("deform", "folds")),
        Slider("Curl across", ("deform", "curl", "x_curvature"), -0.04, 0.04, "%.3f",
               help="Bend across the page (left to right). 0 = flat."),
        Slider("Curl along", ("deform", "curl", "y_curvature"), -0.04, 0.04, "%.3f",
               help="Bend along the page (top to bottom). 0 = flat."),
        Slider("Edge lift (cm)", ("deform", "edge_lift_cm", 0), 0.0, 3.0,
               help="How far the page edges lift off the table. A range: each edge draws its own value.",
               also=(("deform", "edge_lift_cm", 1), ("deform", "edge_lift_cm", 2), ("deform", "edge_lift_cm", 3))),
    ]),
    ("Lamp", [
        Slider("Height angle", ("light", "elevation_deg"), 3.0, 89.0, "%.0f",
               help="Lamp angle above the table, degrees. Low = long shadows, crumples stand out more."),
        Slider("Direction", ("light", "azimuth_deg"), 0.0, 360.0, "%.0f",
               help="Side the light comes from, degrees around the page. 0 = from the right, 90 = from the top."),
        Slider("Distance (cm)", ("light", "distance_cm"), 15.0, 150.0, "%.0f",
               help="Lamp distance from the page center. Closer = brighter near the lamp, sharper fall-off."),
        Slider("Brightness", ("light", "intensity"), 0.0, 2.0),
        Slider("Softness (cm)", ("light", "size_cm"), 0.2, 15.0, "%.1f", help="Lamp size: larger gives softer shadows."),
        Slider("Color (K)", ("light", "color_temp_k"), 2500.0, 8000.0, "%.0f", help="Low = warm, high = cool."),
        Slider("Room light", ("light", "ambient"), 0.0, 0.8, help="Light from the rest of the room."),
    ]),
    ("Background", [
        Slider("Brightness", ("background", "brightness"), 0.05, 1.0, help="Relative to the paper white."),
        Slider("Blur (px)", ("background", "blur_px"), 0.0, 6.0, "%.1f"),
        Slider("Zoom", ("background", "zoom"), 1.0, 3.0),
    ]),
    ("Camera", [
        Slider("X (cm)", ("camera", "position_cm", 0), -40.0, 40.0, "%.1f",
               help="Phone position across the page. 0 = above the page center, + = to the right."),
        Slider("Y (cm)", ("camera", "position_cm", 1), -50.0, 30.0, "%.1f",
               help="Phone position along the page. - = toward the bottom edge (where you stand)."),
        Slider("Height", ("camera", "position_cm", 2), 10.0, 90.0, "%.1f", help="Phone height above the table, cm."),
        Slider("Aim X (cm)", ("camera", "target_cm", 0), -20.0, 20.0, "%.1f",
               help="Where the phone points, across the page."),
        Slider("Aim Y (cm)", ("camera", "target_cm", 1), -20.0, 20.0, "%.1f",
               help="Where the phone points, along the page."),
        Slider("Roll", ("camera", "roll_deg"), -45.0, 45.0, "%.1f", help="Phone turned around its view axis, degrees."),
        Slider("Lens (mm)", ("camera", "focal_mm"), 13.0, 77.0, "%.0f",
               help="35 mm equivalent focal length. 26 = a phone's main camera, 13 = ultra wide, 77 = tele."),
        Slider("Crop margin", ("crop", "margin"), -0.05, 0.1, "%.3f",
               help="Space around the page, as a part of the page size. Negative cuts into the page."),
        Slider("Corner error", ("crop", "corner_error"), 0.0, 0.03, "%.3f",
               help="How far off each corner the scanner app finds it, as a part of the page's short side."),
    ]),
    ("Camera look", [
        Slider("Focus blur (px)", ("fx", "defocus", "strength_px"), 0.0, 5.0, "%.1f"),
        Slider("Camera quality", ("fx", "quality"), 0.0, 1.0,
               help="1 = a good camera. Lower = a cheap phone: a low-resolution sensor, color fringes, "
                    "sharpening halos and JPEG blocks."),
        Slider("Noise", ("fx", "noise", "shot"), 0.0, 0.012, "%.4f"),
        Slider("Vignette", ("fx", "vignette"), 0.0, 0.7),
        Slider("Sharpen", ("fx", "sharpen", "amount"), 0.0, 1.5),
        Slider("Exposure (EV)", ("fx", "exposure_ev"), -1.5, 1.5),
        Slider("JPEG quality", ("output", "jpeg_quality"), 30, 100, "%d", integer=True),
    ]),
    ("Card", [
        Slider("Corner radius (cm)", ("material", "corner_radius_cm"), 0.0, 1.0, "%.2f",
               help="Rounded page corners. An ID or bank card: about 0.32. 0 = square paper corners."),
        Slider("Glossiness", ("material", "roughness"), 0.05, 1.0,
               help="Surface roughness: low = a glossy laminate with sharp highlights, 0.6 = paper."),
        Slider("Glare", ("material", "glare", "strength"), 0.0, 60.0, "%.1f",
               help="The mirror image of a ceiling or window light on a laminated card. 0 = off; "
                    "10 to 40 washes out a patch."),
        Slider("Glare height", ("material", "glare", "elevation_deg"), 0.0, 90.0, "%.0f",
               help="Where that light is, degrees above the table. 90 = straight up."),
        Slider("Glare size", ("material", "glare", "size_deg"), 1.0, 45.0, "%.0f",
               help="Angular radius of the light: larger = a bigger glare patch."),
        Slider("Glare waviness", ("material", "glare", "waviness"), 0.0, 1.0,
               help="How far the laminate's slight unevenness bends the glare's edges. 0 = ruler-straight."),
        Slider("Room reflection", ("material", "room_reflection"), 0.0, 1.5,
               help="A faint mirror image of the room over the whole card. 0 = off; about 0.4 = clearly glossy."),
        Slider("Wear", ("material", "wear"), 0.0, 1.0,
               help="Fine scratches and smudges that show where the glare falls. 0 = a new card."),
        Slider("Hologram", ("material", "hologram"), 0.0, 1.0,
               help="A rainbow sheen that shifts with the view angle, as on ID-card laminates. 0 = off."),
    ]),
]

# Scan mode's panel.
PRINTER_SLIDERS = [
    Slider("Ink bleed", ("printer", "ink_bleed"), 0.0, 1.0, help="Ink spreads into the paper around each stroke."),
    Slider("Broken strokes", ("printer", "broken_strokes"), 0.0, 0.08, "%.3f",
           help="Part of the ink left out, like a printer low on toner."),
    Slider("Ghost echo", ("printer", "ghost"), 0.0, 0.5, help="A faint shifted copy of the ink, like a double strike."),
    Slider("Toner streaks", ("printer", "toner_streaks"), 0, 8, "%d", integer=True,
           help="Thin broken lines along the paper feed, from a dirty drum."),
    Slider("Toner specks", ("printer", "toner_specks"), 0, 2000, "%d", integer=True, help="Black dots of loose toner."),
]
SCAN_SLIDERS = [
    Slider("Resolution (dpi)", ("scanner", "dpi"), 75, 600, "%d", integer=True,
           help="Scan resolution. The output size is the paper size at this dpi."),
    Slider("Skew", ("scanner", "skew_deg"), -5.0, 5.0, "%.2f", help="How crooked the page lies on the glass, degrees."),
    Slider("Softness (px)", ("scanner", "blur_px"), 0.0, 2.0, help="Lens and sensor blur, in scan pixels."),
    Slider("Bold", ("scanner", "bold"), 0.0, 1.0, help="Strokes thicken, like a scan or copy made too dark. "
           "1 = about one scan pixel thicker."),
    Slider("Dark edges", ("scanner", "dark_edges", 0), 0.0, 0.1, "%.3f",
           help="Dark bands at the left and right where the scanner sees past the paper, as a part of the page "
                "width. Advanced settings has each side on its own.",
           also=(("scanner", "dark_edges", 1),)),
    Slider("Edge dots", ("scanner", "edge_dots"), 0, 20000, "%d", integer=True,
           help="Loose dots scattered beside the dark bands."),
    Slider("Noise", ("scanner", "noise"), 0.0, 10.0, "%.1f", help="Sensor noise, in gray levels."),
    Slider("Streaks", ("scanner", "streaks"), 0, 10, "%d", integer=True,
           help="Thin lines down the page: dust on the sensor bar."),
    Slider("Dust", ("scanner", "dust"), 0, 200, "%d", integer=True, help="Specks and short hairs on the glass."),
]


class Editor:
    def __init__(self, config_path: Path | None, doc_path: Path | None):
        self.user: dict[str, Any] = {}
        self.config_path: Path | None = None
        self.base_dir = Path.cwd()
        self.doc_path = SAMPLE_DOC
        self.boxes = []
        self.variant = 0
        self.show_boxes = True
        self.show_advanced = False
        self.status = ""
        self.error = ""
        self.unsaved = False
        self.frame: Frame | None = None
        self.full_frame: Frame | None = None
        self.mix_share = 0.3  # the scan share the Mode menu's "Mix" starts with
        self.dirty = True  # the config changed: rebuild the scene and the photo
        self.view_dirty = True  # only the 3D view moved: render the scene view again
        self.quality_pending = False
        self.selected: str | None = None
        self.orbiting = False  # an orbit or pan drag is in progress
        self.orbit = OrbitCamera(pivot=np.zeros(3))
        self.render_ms = 0.0
        self.ctx: moderngl.Context | None = None
        self.preview: Preview | None = None
        self.photo_tex: moderngl.Texture | None = None
        self.view_tex: moderngl.Texture | None = None
        self.view_size = (0, 0)
        self.advanced = AdvancedPanel(self)
        # Per slider: the last range and the last fixed value, so the random box can switch back.
        self.slider_memory: dict[tuple, dict[str, Any]] = {}
        self.look_memory: dict[str, dict] = {}  # the values the Paper / Card switch restores
        self._pending = (config_path, doc_path)

    # ---- setup and files ---------------------------------------------------------

    def _init_gl(self) -> None:
        self.ctx = moderngl.create_context()
        self.preview = Preview(Renderer(ctx=self.ctx))
        self.preview.add_image(SAMPLE_DOC, _sample_page())
        self.ctx.screen.use()
        config_path, doc_path = self._pending
        if config_path:
            self.load_config(config_path)
        if doc_path:
            self.load_doc(doc_path)

    def load_config(self, path: Path) -> None:
        try:
            user = json.loads(Path(path).read_text())
            build_config(user, base_dir=Path(path).parent)
        except (OSError, json.JSONDecodeError, ConfigError) as e:
            self.error = f"Could not load {path}: {e}"
            return
        self.user, self.config_path, self.base_dir = user, Path(path), Path(path).parent
        self.advanced.json_buffers.clear()
        self.slider_memory.clear()
        self.look_memory.clear()
        self.unsaved = False
        self.status = f"Loaded {Path(path).name}"
        self.changed(unsaved=False)

    def save_config(self, path: Path) -> None:
        user = state.rebase_paths(self.user, self.base_dir, Path(path).parent)
        Path(path).write_text(state.format_config(state.strip_defaults(user)) + "\n")
        self.user, self.base_dir, self.config_path = user, Path(path).parent, Path(path)
        self.unsaved = False
        self.status = f"Saved {Path(path).name}"

    def load_doc(self, path: Path) -> None:
        self.doc_path = str(Path(path).resolve())
        side = sidecar_path(path)
        try:
            doc = self.preview.image(self.doc_path) if self.preview else None
            size = (doc.shape[1], doc.shape[0]) if doc is not None else None
            self.boxes = load_boxes(side, size) if side.exists() else []
        except ValueError as e:
            self.boxes, self.error = [], str(e)
        self.status = f"{Path(path).name}: {len(self.boxes)} layout boxes"
        self.changed(unsaved=False)

    def changed(self, unsaved: bool = True) -> None:
        self.dirty = True
        self.unsaved = self.unsaved or unsaved
        self.full_frame = None

    def set(self, path: tuple, value: Any) -> None:
        state.set_value(self.user, path, value)
        self.changed()

    # ---- frame -------------------------------------------------------------------

    def gui(self) -> None:
        if self.ctx is None:
            self._init_gl()
        # Cheap previews while something is dragged: a gizmo arrow or a slider.
        moving = (gizmo.is_using_any() or imgui.is_any_item_active()) and imgui.is_mouse_down(imgui.MouseButton_.left)
        if self.dirty:
            self._rebuild(FAST if moving else GOOD)
            self.dirty, self.quality_pending, self.view_dirty = False, moving, True
        elif self.quality_pending and not moving:
            self._rebuild(GOOD)
            self.quality_pending, self.view_dirty = False, True

        gizmo.begin_frame()
        avail = imgui.get_content_region_avail()
        imgui.begin_child("left", imgui.ImVec2(LEFT_W, avail.y), imgui.ChildFlags_.borders)
        self._left_panel()
        imgui.end_child()
        imgui.same_line()
        center_w = avail.x - LEFT_W - RIGHT_W - 16
        imgui.begin_child("scene", imgui.ImVec2(center_w, avail.y), imgui.ChildFlags_.borders,
                          imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse)
        self._scene_view()
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("right", imgui.ImVec2(RIGHT_W, avail.y), imgui.ChildFlags_.borders)
        self._photo_panel()
        imgui.end_child()

        if self.show_advanced:
            imgui.set_next_window_size(imgui.ImVec2(520, 760), imgui.Cond_.first_use_ever)
            _, self.show_advanced = imgui.begin("Advanced settings", self.show_advanced)
            self.advanced.draw()
            imgui.end()

    def _rebuild(self, quality: dict[str, Any]) -> None:
        t = time.perf_counter()
        try:
            check_tree(self.user)
            first = self.frame is None or self.frame.scene is None  # also the first photo after a scan
            self.frame = self.preview.render(self.user, self.base_dir, self.doc_path, self.boxes, self.variant,
                                             quality["max_size"], quality["mesh"], quality["effects"])
            self.error = ""
            if first and self.frame.scene is not None:
                self._frame_all(self.frame)
                self.view_dirty = True
        except (ConfigError, SpecError, ValueError, KeyError, TypeError) as e:
            self.error = str(e)
        except Exception:
            self.error = traceback.format_exc(limit=3)
        self.render_ms = (time.perf_counter() - t) * 1000
        self._upload_photo()

    def _upload_photo(self) -> None:
        frame = self.full_frame or self.frame
        if frame is not None:
            img = frame.image
            if img.shape[2] == 4:  # transparent: show it on a checkerboard
                img = _on_checkerboard(img)
            self.photo_tex = _upload(self.ctx, self.photo_tex, img)
        self.ctx.screen.use()

    # ---- left panel ------------------------------------------------------------------

    def _section(self, name: str) -> bool:
        """A collapsible section header, open at start. Returns whether it is open."""
        return imgui.collapsing_header(f"{name}###section-{name}", imgui.TreeNodeFlags_.default_open)

    def _left_panel(self) -> None:
        if self._section("Files"):
            self._files()

        merged = state.merged(self.user)
        if self._section("Sample"):
            if imgui.button("New random sample##left"):
                self.variant += 1
                self.changed(unsaved=False)
            imgui.set_item_tooltip("Draw new values for every slider with a ticked box (and a new crumple pattern).")
            imgui.same_line()
            imgui.text_disabled(f"variant {self.variant}")
        photo, scan = self._mode_switch(merged)
        card = self._look_switch(merged) if photo else False
        for group, sliders in SLIDER_GROUPS if photo else []:
            if group == "Card" and not card:
                continue
            # A card keeps the page-shape sliders it uses (the curl) under its own name.
            if self._section("Card shape" if card and group == "Paper" else group):
                self._group(group, sliders, merged, card)
            if group == "Lamp" and self._section("Hand shadows"):
                self._hand_controls()
        if scan:
            if self._section("Printer"):
                for s in PRINTER_SLIDERS:
                    self._slider(s, merged)
            if self._section("Scanner"):
                self._scan_controls(merged)

        if self._section("More"):
            if imgui.button("Advanced settings..."):
                self.show_advanced = True
            imgui.same_line()
            if imgui.button("Reset all"):
                self.user = {}
                self.advanced.json_buffers.clear()
                self.slider_memory.clear()
                self.changed()
            imgui.text_disabled("Ticked box: each photo draws a value from the range.")

        if self.error:
            imgui.separator_text("Problem")
            imgui.text_colored(imgui.ImVec4(1, 0.45, 0.45, 1), self.error)

    def _mode_switch(self, merged: dict[str, Any]) -> tuple[bool, bool]:
        """Photo, scan or a mix of both. Returns whether the photo and the scan controls apply."""
        kind, share = state.read_mode(merged["mode"])
        if kind == "other":
            imgui.text_disabled("Mode: a custom random spec (see advanced settings)")
            return True, True
        kinds = ["photo", "scan", "mix"]
        imgui.set_next_item_width(LEFT_W - 150)
        ch, i = imgui.combo("Mode", kinds.index(kind), ["Phone photo (3D)", "Scan (2D)", "Mix photo and scan"])
        imgui.set_item_tooltip("Phone photo: the page on a desk, lit and photographed. Scan: a printed page through "
                               "a flatbed scanner. Mix: each image draws one of the two, so one run makes both.")
        if ch and kinds[i] != kind:
            self.set(("mode",), state.mix_mode(self.mix_share) if kinds[i] == "mix" else kinds[i])
            kind = kinds[i]
        if kind == "mix":
            self.mix_share = share
            imgui.set_next_item_width(LEFT_W - 150)
            ch, pct = imgui.slider_int("Scans (%)", round(share * 100), 0, 100)
            imgui.set_item_tooltip("The share of images that come out as scans. The rest are phone photos.")
            if ch:
                self.mix_share = pct / 100.0
                self.set(("mode",), state.mix_mode(self.mix_share))
            if self.frame is not None:
                imgui.text_disabled(f"This variant: {'scan' if self.frame.scene is None else 'phone photo'}")
        return kind != "scan", kind != "photo"

    def _look_switch(self, merged: dict[str, Any]) -> bool:
        """Paper or a plastic card. Returns whether the config has the card look."""
        card = state.card_look(merged)
        imgui.set_next_item_width(LEFT_W - 150)
        ch, i = imgui.combo("Look", int(card), ["Paper", "Plastic card"])
        imgui.set_item_tooltip("Paper: a sheet that crumples and folds. Plastic card: an ID, driving licence or bank "
                               "card, rigid and glossy, with rounded corners and glare. Switching back restores "
                               "the values you had.")
        if ch and i != int(card):
            self.set_look("card" if i else "paper")
            card = bool(i)
        return card

    def set_look(self, look: str) -> None:
        state.set_look(self.user, look, self.look_memory)
        self.changed()

    def _scan_controls(self, merged: dict[str, Any]) -> None:
        sc = merged["scanner"]
        color = sc["color"]
        if spec_kind(color):
            drawn = self.frame.resolved["scanner"]["color"] if self.frame is not None else "?"
            imgui.text_disabled(f"Color mode: random (this page: {drawn})")
        else:
            modes = ["color", "gray", "bw"]
            imgui.set_next_item_width(LEFT_W - 150)
            ch, i = imgui.combo("Color mode", modes.index(color), ["Color", "Grayscale", "Black and white (1-bit)"])
            if ch:
                self.set(("scanner", "color"), modes[i])
        for s in SCAN_SLIDERS:
            self._slider(s, merged)
        if color == "bw":
            self._slider(Slider("B/W threshold", ("scanner", "bw_threshold"), 60, 220, "%d", integer=True,
                                help="Gray levels below this become black."), merged)

    def _files(self) -> None:
        doc = Path(self.doc_path).name if self.doc_path != SAMPLE_DOC else "sample page"
        if imgui.button("Open document..."):
            res = pfd.open_file("Open document", str(Path.cwd()),
                                ["Images", "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp"]).result()
            if res:
                self.load_doc(Path(res[0]))
        imgui.same_line()
        imgui.text(doc)
        if imgui.button("Open config..."):
            res = pfd.open_file("Open config", str(Path.cwd()), ["Config", "*.json"]).result()
            if res:
                self.load_config(Path(res[0]))
        imgui.same_line()
        if imgui.button("Save") and self.config_path:
            self.save_config(self.config_path)
        imgui.same_line()
        if imgui.button("Save as..."):
            res = pfd.save_file("Save config", str(self.config_path or Path.cwd() / "config.json"),
                                ["Config", "*.json"]).result()
            if res:
                self.save_config(Path(res if res.endswith(".json") else res + ".json"))
        name = self.config_path.name if self.config_path else "new config"
        imgui.text_disabled(name + ("  (unsaved)" if self.unsaved else ""))

    def _group(self, group: str, sliders: list[Slider], merged: dict[str, Any], card: bool = False) -> None:
        if group == "Paper" and not card:
            if imgui.button("Flat page"):
                self.flat_page()
            imgui.set_item_tooltip("Crumple, folds, curl and edge lift all to 0.")
            self._crumple_shape(merged)
        if group == "Background" and not self._background_switch(merged):
            return
        placed = group == "Lamp" and merged["light"]["position_cm"] is not None
        if placed:
            imgui.text_disabled("The lamp has a fixed position (light.position_cm).")
            if imgui.button("Place it by angles"):
                self.set(("light", "position_cm"), None)
            imgui.set_item_tooltip("Use the height angle, direction and distance below instead.")
        crop = False
        if group == "Camera":
            imgui.text_disabled("Ticked box: a range; each photo draws its own camera.")
            crop = self._crop_switch(merged)
        for s in sliders:
            if s.path[0] == "crop" and not crop:
                continue
            if card and s.path[0] == "deform" and s.path[1] != "curl" and _flat(merged, s.path[1]):
                continue  # a card does not crumple, fold or lift; a value someone set still shows
            if placed and s.path in LAMP_PLACE:
                continue
            if s.path == ("deform", "folds", "repeat"):
                self._folds(s, merged)
            else:
                self._slider(s, merged)

    def _slider(self, s: Slider, merged: dict[str, Any]) -> None:
        value = state.get(merged, s.path)
        imgui.push_id(s.label + str(s.path))
        kind = spec_kind(value)
        ranged = kind in ("uniform", "randint")
        ch, random_new = imgui.checkbox("##random", kind is not None)
        imgui.set_item_tooltip("Random: each photo draws a value from the range. "
                               "Unticked: every photo uses the one value.")
        if ch:
            self.toggle_random(s, value, random_new)
            imgui.pop_id()
            return
        imgui.same_line()
        imgui.set_next_item_width(LEFT_W - 165)
        if ranged:
            lo, hi = value[kind]
            # Show the value this sample drew from the range.
            drawn = self._drawn(s)
            label = s.label if not isinstance(drawn, (int, float)) else f"{s.label}: {s.fmt % drawn}###{s.label}"
            if s.integer:
                ch, v = imgui.slider_int2(label, [int(lo), int(hi)], int(s.lo), int(s.hi))
            else:
                ch, v = imgui.slider_float2(label, [float(lo), float(hi)], s.lo, s.hi, s.fmt)
            if ch:
                self.set_slider(s, {kind: sorted(v)})
        elif kind:  # another kind of random spec: edit it in the advanced panel
            drawn = self._drawn(s)
            shown = "" if not isinstance(drawn, (int, float)) else f", this photo: {s.fmt % drawn}"
            imgui.text_disabled(f"{s.label}: random ({kind}){shown}")
        else:
            current = value if isinstance(value, (int, float)) else s.lo
            if s.integer:
                ch, v = imgui.slider_int(s.label, int(current), int(s.lo), int(s.hi))
            else:
                ch, v = imgui.slider_float(s.label, float(current), s.lo, s.hi, s.fmt)
            if ch:
                self.set_slider(s, v if s.integer else round(v, 4))
        if s.help:
            imgui.set_item_tooltip(s.help)
        imgui.pop_id()

    def toggle_random(self, s: Slider, value: Any, on: bool) -> None:
        """The random box: switch between one value and a range.

        Ticking brings back the last range set on this slider. Unticking
        keeps the value the current photo drew, so the photo does not jump.
        """
        memory = self.slider_memory.setdefault(s.path, {})
        if on:
            self.set_slider(s, memory.get("range") or self._range_around(s, value))
            return
        memory["range"] = value
        drawn = self._drawn(s)
        if not isinstance(drawn, (int, float)):
            drawn = state.to_fixed(value)
        self.set_slider(s, int(drawn) if s.integer else round(float(drawn), 4))

    def _drawn(self, s: Slider) -> Any:
        """The value the current photo drew for this slider."""
        if self.frame is None:
            return None
        if s.count_of is not None:
            items = state.get(self.frame.resolved, s.count_of)
            return len(items) if isinstance(items, list) else None
        return state.get(self.frame.resolved, s.path)

    def set_slider(self, s: Slider, value: Any) -> None:
        for path in (s.path, *s.also):
            state.set_value(self.user, path, copy.deepcopy(value))
        self.changed()

    def flat_page(self) -> None:
        """Everything that bends the page to 0."""
        for path in (("deform", "crumple", "strength"), ("deform", "curl", "x_curvature"),
                     ("deform", "curl", "y_curvature")):
            state.set_value(self.user, path, 0.0)
        state.set_value(self.user, ("deform", "edge_lift_cm"), [0.0, 0.0, 0.0, 0.0])
        state.set_value(self.user, ("deform", "folds"), [])
        self.changed()

    @staticmethod
    def _range_around(s: Slider, value: Any) -> dict[str, list]:
        v = float(value) if isinstance(value, (int, float)) else (s.lo + s.hi) / 2
        span = (s.hi - s.lo) * 0.15
        lo, hi = max(s.lo, v - span), min(s.hi, v + span)
        return {"randint": [int(lo), int(hi)]} if s.integer else {"uniform": [round(lo, 4), round(hi, 4)]}

    def _background_switch(self, merged: dict[str, Any]) -> bool:
        """The background on/off box. Returns whether the background is on."""
        on = merged["background"]["enabled"]
        if spec_kind(on):
            imgui.text_disabled("Background: random on/off (see advanced settings)")
            return True
        ch, on = imgui.checkbox("Background", bool(on))
        imgui.set_item_tooltip("Off: only the page, on a transparent PNG.")
        if ch:
            self.set(("background", "enabled"), on)
            if not on:
                self.set(("output", "format"), "png")  # JPEG cannot be transparent
        if not on:
            return False
        desk = merged["desk"]["enabled"]
        if spec_kind(desk):
            imgui.text_disabled("Desk: random on/off (see advanced settings)")
            return True
        imgui.same_line()
        ch, desk = imgui.checkbox("Desk", bool(desk))
        imgui.set_item_tooltip("A desk under the page, with a keyboard, a mousepad, a pen or other paper beside it. "
                               "Off: the background picture.")
        if ch:
            self.set(("desk", "enabled"), desk)
            folders = [f"@assets/{p}" for p in DESK_TEXTURES if (_ASSETS / p).is_dir()]
            if desk and merged["desk"]["surface"] is None and folders:
                self.set(("desk", "surface"), folders if len(folders) > 1 else folders[0])
            if desk and not merged["desk"]["items"]:
                self.set(("desk", "items"), dict(RANDOM_DESK_ITEMS))
        if desk:
            items = merged["desk"]["items"]
            if spec_kind(items) == "repeat":
                self._slider(Slider("Desk items", ("desk", "items", "repeat"), 0, 4, "%d", integer=True,
                                    help="How many items lie beside the page. Advanced settings has the rest."), merged)
            return False  # the picture sliders do not apply
        return True

    def _crop_switch(self, merged: dict[str, Any]) -> bool:
        """The scanner-app crop box. Returns whether the crop sliders apply."""
        crop = merged["crop"]["enabled"]
        if spec_kind(crop):
            imgui.text_disabled("Scan crop: random on/off (see advanced settings)")
            return True
        ch, crop = imgui.checkbox("Scan crop (page only)", bool(crop))
        imgui.set_item_tooltip("Like a phone scanner app: the four page corners are pulled to the image corners, "
                               "so the whole image is the page. Curl and crumples stay. The camera frames the "
                               "page itself: X, Y, Height and Aim set only the view angle.")
        if ch:
            self.set(("crop", "enabled"), crop)
        return bool(crop)

    def _crumple_shape(self, merged: dict[str, Any]) -> None:
        mode = merged["deform"]["crumple"]["mode"]
        if spec_kind(mode):
            imgui.text_disabled("Crumple shape: random (see advanced settings)")
            return
        modes = ["scan", "procedural"]
        imgui.set_next_item_width(LEFT_W - 150)
        ch, i = imgui.combo("Crumple shape", modes.index(mode), ["Scanned paper", "Generated creases"])
        if ch:
            self.set(("deform", "crumple", "mode"), modes[i])
        imgui.set_item_tooltip("Where the crumple shape comes from. Both bend the page geometry.")

    def _folds(self, s: Slider, merged: dict[str, Any]) -> None:
        folds = merged["deform"]["folds"]
        if spec_kind(folds) == "repeat":
            self._slider(s, merged)
            return
        # A list of folds: the slider turns it into n random folds.
        imgui.dummy(imgui.ImVec2(imgui.get_frame_height(), 0))
        imgui.same_line()
        imgui.set_next_item_width(LEFT_W - 150)
        ch, n = imgui.slider_int(s.label, len(folds), int(s.lo), int(s.hi))
        if ch:
            self.set(("deform", "folds"), {"repeat": n, "item": dict(RANDOM_FOLD)})
        imgui.set_item_tooltip(s.help)

    def _hand_controls(self) -> None:
        occ = self.user.get("occluders")
        if spec_kind(occ) == "repeat":
            scene = self.frame.scene if self.frame else None
            if scene is None:  # a scan in a photo-and-scan mix
                imgui.text_wrapped("Random shadows. This variant is a scan; pick a photo variant to see them.")
                return
            n = len(scene.occluders)
            imgui.text_wrapped(f"Random shadows: {n} in this sample. Each aims at a point on the page, "
                               "so it follows the lamp.")
            if imgui.button("Fix them in place"):
                self.user["occluders"] = self._fixed_hands()
                self.changed()
            imgui.set_item_tooltip("Turn this sample's shadows into fixed ones you can move. "
                                   "They stop following the lamp, and stop being random.")
            imgui.same_line()
            if imgui.button("Remove all"):
                self.set(("occluders",), [])
                self.selected = None
            return
        items = occ if isinstance(occ, list) else []
        shapes = ["hand_phone", "arm", "phone", "blob"]
        for k, item in enumerate(items):
            imgui.push_id(f"hand{k}")
            if imgui.button("x"):
                del items[k]
                self.selected = None
                self.changed()
                imgui.pop_id()
                break
            imgui.set_item_tooltip("Remove this shadow")
            imgui.same_line()
            sel = self.selected == f"hand{k}"
            if imgui.selectable(f"{k + 1}", sel, 0, imgui.ImVec2(18, 0))[0]:
                self.selected = f"hand{k}"
            imgui.set_item_tooltip("Select it in the 3D view")
            cur = item.get("shape", "hand_phone")
            if not spec_kind(cur):
                imgui.same_line()
                imgui.set_next_item_width(115)
                ch, i = imgui.combo("##shape", shapes.index(cur), ["hand + phone", "arm", "phone", "blob"])
                if ch:
                    item["shape"] = shapes[i]
                    self.changed()
            scale = item.get("scale", 1.0)
            if not spec_kind(scale):
                imgui.same_line()
                imgui.set_next_item_width(-40)
                ch, v = imgui.slider_float("Size", float(scale), 0.5, 1.5, "%.2f")
                if ch:
                    item["scale"] = round(v, 3)
                    self.changed()
            imgui.pop_id()
        if imgui.button("Add hand shadow"):
            self._add_hand()
        imgui.set_item_tooltip("A hand above the page. Click it in the 3D view and drag its arrows.")

    def _add_hand(self) -> None:
        """Add a hand above the page center, on the line to the lamp, at a fixed place."""
        item: dict[str, Any] = {"shape": "hand_phone"}
        if self.frame is not None and self.frame.scene is not None:
            sc = self.frame.scene
            occ = place_occluder("hand_phone", sc.page_center, sc.light.position, sc.camera.position, 20.0, 1.0, 0.0)
            item["position_cm"] = [round(float(v), 2) for v in occluder_anchor(occ.shape, occ.corners)]
        occ = self.user.get("occluders")
        items = occ if isinstance(occ, list) else []
        self.set(("occluders",), [*items, item])
        self.selected = f"hand{len(items)}"

    def _fixed_hands(self) -> list[dict[str, Any]]:
        """This sample's shadow casters as fixed items, each at its current place."""
        if self.frame is None or self.frame.scene is None:
            return []
        out = []
        for o, occ in zip(self.frame.resolved["occluders"], self.frame.scene.occluders):
            out.append({"shape": o["shape"], "scale": o["scale"], "rotation_offset_deg": o["rotation_offset_deg"],
                        "position_cm": [round(float(v), 2) for v in occluder_anchor(occ.shape, occ.corners)]})
        return out

    # ---- 3D scene view ----------------------------------------------------------------

    def _scene_view(self) -> None:
        frame = self.frame
        if frame is not None and frame.scene is None:
            imgui.text_disabled("Scan mode has no 3D scene: the page goes straight through the scanner. "
                                "The result is on the right.")
            return
        imgui.text_disabled("Drag: orbit   Shift-drag: pan   Scroll: zoom   F: frame the page   "
                            "Click the lamp, camera, aim point or hand, then drag its arrows")
        avail = imgui.get_content_region_avail()
        w, h = int(avail.x), int(avail.y)
        if frame is None or w < 50 or h < 50:
            return
        if (w, h) != self.view_size:
            self.view_size, self.view_dirty = (w, h), True
        if self.view_dirty:
            self._render_view(frame, w, h)
            self.view_dirty = False

        pos = imgui.get_cursor_screen_pos()
        # Only reserve the space. An interactive item here (a button) would
        # sit under the mouse and ImGuizmo then ignores clicks on its arrows.
        imgui.dummy(imgui.ImVec2(w, h))
        io = imgui.get_io()
        inside = pos.x <= io.mouse_pos.x < pos.x + w and pos.y <= io.mouse_pos.y < pos.y + h
        hovered = inside and imgui.is_window_hovered()
        draw = imgui.get_window_draw_list()
        draw.add_image(imgui.ImTextureRef(self.view_tex.glo), pos, imgui.ImVec2(pos.x + w, pos.y + h),
                       imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))  # GL rows run bottom to top

        view, proj = self.orbit.view(), self.orbit.proj(w / h)
        vp = proj @ view
        objects = self._objects(frame)
        self._draw_overlays(draw, frame, vp, pos, w, h, objects)

        # The gizmo goes first: when the mouse is on it, it owns the drag.
        gizmo.set_drawlist(draw)
        gizmo.set_rect(pos.x, pos.y, w, h)
        gizmo.set_orthographic(False)
        if self.selected in objects:
            m = gizmo.Matrix16(list(_translation(objects[self.selected]).T.ravel()))
            if gizmo.manipulate(gizmo.Matrix16(list(view.T.ravel())), gizmo.Matrix16(list(proj.T.ravel())),
                                gizmo.OPERATION.translate, gizmo.MODE.world, m):
                self._move(self.selected, np.asarray(m.values, dtype=np.float64)[12:15].copy(), frame)
        on_gizmo = gizmo.is_over() or gizmo.is_using()

        left, right = imgui.MouseButton_.left, imgui.MouseButton_.right
        if hovered and not on_gizmo and imgui.is_mouse_clicked(left):
            mouse = np.array([io.mouse_pos.x - pos.x, io.mouse_pos.y - pos.y])
            self.selected = self._pick(objects, vp, w, h, mouse)
            self.orbiting = self.selected is None  # a click on empty space starts an orbit drag
        if hovered and imgui.is_mouse_clicked(right):
            self.orbiting = True
        if not (imgui.is_mouse_down(left) or imgui.is_mouse_down(right)):
            self.orbiting = False
        if self.orbiting and not gizmo.is_using() and (io.mouse_delta.x or io.mouse_delta.y):
            if io.key_shift:
                self.orbit.pan(io.mouse_delta.x, io.mouse_delta.y, h)
            else:
                self.orbit.orbit(io.mouse_delta.x, io.mouse_delta.y)
            self.view_dirty = True
        if hovered and io.mouse_wheel:
            self.orbit.zoom(io.mouse_wheel)
            self.view_dirty = True
        if hovered and imgui.is_key_pressed(imgui.Key.f):
            self._frame_all(frame)
            self.view_dirty = True

    def _frame_all(self, frame: Frame) -> None:
        """Aim the orbit view at the page and zoom out until every object fits."""
        pts = np.array(list(self._objects(frame).values()))
        center = frame.scene.page_center
        # Aim between the table and the highest object, then fit them all with a margin.
        pivot = center + np.array([0.0, 0.0, 0.35 * float(pts[:, 2].max())])
        reach = max(float(np.linalg.norm(pts - pivot, axis=1).max()),
                    0.5 * float(np.hypot(frame.scene.page.width_cm, frame.scene.page.height_cm)))
        w, h = self.view_size if self.view_size[1] else (4, 3)
        tan_v = np.tan(np.radians(self.orbit.fov_deg) / 2)
        tan_fit = min(tan_v, tan_v * w / h)  # the narrower of the two view angles
        self.orbit.pivot = pivot
        self.orbit.distance = reach / tan_fit * 1.15

    def _render_view(self, frame: Frame, w: int, h: int) -> None:
        """The page seen from the orbit camera, lit by the scene lamp."""
        scale = min(1.0, 1100 / max(w, h))
        rw, rh = max(32, int(w * scale)), max(32, int(h * scale))
        scene = dataclasses.replace(frame.scene, camera=self.orbit.camera(rw, rh))
        # Stays on the GPU: no read back, no upload.
        self.view_tex = self.preview.renderer.render_display(scene, self.preview.image(self.doc_path),
                                                             self.doc_path, VIEW_BG)
        self.ctx.screen.use()

    def _objects(self, frame: Frame) -> dict[str, np.ndarray]:
        scene = frame.scene
        objs = {"lamp": scene.light.position, "camera": scene.camera.position,
                "aim": np.asarray(frame.resolved["camera"]["target_cm"], float)}
        for k, occ in enumerate(scene.occluders):
            objs[f"hand{k}"] = occluder_anchor(occ.shape, occ.corners)
        return objs

    def _draw_overlays(self, draw, frame: Frame, vp, pos, w, h, objects) -> None:
        def pt(p):
            px, front = project(vp, np.asarray(p)[None], w, h)
            return imgui.ImVec2(pos.x + px[0, 0], pos.y + px[0, 1]), bool(front[0])

        def line(a, b, col, thick=1.5):
            (pa, fa), (pb, fb) = pt(a), pt(b)
            if fa and fb:
                draw.add_line(pa, pb, col, thick)

        c = frame.scene.page_center
        for k in range(-10, 11):  # table grid, 5 cm cells
            line(c + [k * 5, -50, 0], c + [k * 5, 50, 0], COL_GRID, 1.0)
            line(c + [-50, k * 5, 0], c + [50, k * 5, 0], COL_GRID, 1.0)

        cam = frame.scene.camera
        for a, b in frustum_lines(cam, 6.0):  # a small camera icon; the triangle marks the photo's top
            line(a, b, COL_CAM, 1.5)
        line(cam.position, objects["aim"], COL_CAM_DIM, 1.0)
        line(objects["lamp"], c, COL_LAMP, 1.0)
        for occ in frame.scene.occluders:
            q = occ.corners
            for i in range(4):
                line(q[i], q[(i + 1) % 4], COL_HAND, 1.5)

        labels = {"lamp": ("Lamp", COL_LAMP), "camera": ("Camera", COL_CAM), "aim": ("Aim", COL_CAM)}
        many = len(frame.scene.occluders) > 1
        for name, p in objects.items():
            sp, front = pt(p)
            if not front:
                continue
            if name.startswith("hand"):
                text, col = (f"Hand {int(name[4:]) + 1}" if many else "Hand"), COL_HAND
            else:
                text, col = labels[name]
            r = 9.0 if name in ("lamp", "camera") else 6.0
            draw.add_circle_filled(sp, r, col)
            if name == self.selected:
                draw.add_circle(sp, r + 4, COL_SEL, 0, 2.0)
            draw.add_text(imgui.ImVec2(sp.x + r + 4, sp.y - 8), col, text)

    def _pick(self, objects, vp, w, h, mouse) -> str | None:
        best, best_d = None, PICK_RADIUS
        for name, p in objects.items():
            px, front = project(vp, np.asarray(p)[None], w, h)
            d = float(np.linalg.norm(px[0] - mouse))
            if front[0] and d < best_d:
                best, best_d = name, d
        return best

    def _move(self, name: str, p: np.ndarray, frame: Frame) -> None:
        """Write a moved object back into the config as fixed values."""
        scene, resolved = frame.scene, frame.resolved
        if name == "lamp":
            # Hands that aim at the page would follow the lamp. Pin them first.
            occ = self.user.get("occluders")
            if isinstance(occ, list) and any(o.get("position_cm") is None for o in occ):
                fixed = self._fixed_hands()
                for o, f in zip(occ, fixed):
                    if o.get("position_cm") is None:
                        o["position_cm"] = f["position_cm"]
            p[2] = max(p[2], 1.0)
            az, el, d = lamp_angles(p, scene.page_center)
            state.set_value(self.user, ("light", "position_cm"), None)
            for key, v in (("azimuth_deg", az), ("elevation_deg", el), ("distance_cm", d)):
                state.set_value(self.user, ("light", key), round(v, 2))
        elif name == "camera":
            p[2] = max(p[2], 2.0)
            state.set_value(self.user, ("camera", "position_cm"), [round(float(v), 2) for v in p])
            state.set_value(self.user, ("camera", "target_cm"),
                            [round(float(v), 2) for v in resolved["camera"]["target_cm"]])
        elif name == "aim":
            state.set_value(self.user, ("camera", "target_cm"), [round(float(v), 2) for v in p])
            state.set_value(self.user, ("camera", "position_cm"),
                            [round(float(v), 2) for v in resolved["camera"]["position_cm"]])
        elif name.startswith("hand"):
            p[2] = max(p[2], 0.5)
            if not isinstance(self.user.get("occluders"), list):  # random shadows: fix this sample's first
                self.user["occluders"] = self._fixed_hands()
            self.user["occluders"][int(name[4:])]["position_cm"] = [round(float(v), 2) for v in p]
        self.changed()

    # ---- photo panel -------------------------------------------------------------------

    def _photo_panel(self) -> None:
        imgui.separator_text("Photo")
        frame = self.full_frame or self.frame
        if frame is not None and self.photo_tex is not None:
            ih, iw = frame.image.shape[:2]
            w = imgui.get_content_region_avail().x
            pos = imgui.get_cursor_screen_pos()
            imgui.image(imgui.ImTextureRef(self.photo_tex.glo), imgui.ImVec2(w, w * ih / iw))
            if self.show_boxes:
                s = w / iw
                draw = imgui.get_window_draw_list()
                for box in frame.labels:
                    if box["quad"]:
                        pts = [imgui.ImVec2(pos.x + x * s, pos.y + y * s) for x, y in box["quad"]]
                        draw.add_polyline(pts, imgui.IM_COL32(255, 60, 60, 230), 1.5, imgui.ImDrawFlags_.closed.value)
            imgui.text_disabled("full size" if self.full_frame else f"preview, {self.render_ms:.0f} ms")
            if frame.warning:
                imgui.text_colored(imgui.ImVec4(1, 0.75, 0.3, 1), "The page leaves the frame; the CLI would redraw this sample.")
        _, self.show_boxes = imgui.checkbox(f"Show {len(self.boxes)} layout boxes", self.show_boxes)

        imgui.separator_text("Sample")
        ch, v = imgui.input_int("Variant", self.variant)
        if ch:
            self.variant = max(0, v)
            self.changed(unsaved=False)
        imgui.set_item_tooltip("Random ranges are drawn per variant. Variant n is the same photo "
                               "`recapture render` makes as variant n.")
        if imgui.button("New random sample"):
            self.variant += 1
            self.changed(unsaved=False)

        imgui.separator_text("Output")
        if imgui.button("Render full size"):
            t = time.perf_counter()
            try:
                self.full_frame = self.preview.render(self.user, self.base_dir, self.doc_path, self.boxes,
                                                      self.variant, None, None, True)
                self.status = f"Full render: {(time.perf_counter() - t) * 1000:.0f} ms"
            except Exception as e:
                self.error = str(e)
            self._upload_photo()
        if self.full_frame is not None:
            imgui.same_line()
            if imgui.button("Save..."):
                res = pfd.select_folder("Save the photo to", str(Path.cwd())).result()
                if res:
                    stem = f"{Path(self.doc_path).stem.strip('<>').replace(' ', '_')}_{self.variant:03d}"
                    labels = {"source": Path(self.doc_path).name, "image": "",
                              "size": list(self.full_frame.image.shape[1::-1]),
                              "page_corners": self.full_frame.page_corners, "boxes": self.full_frame.labels}
                    out = write_outputs(Path(res), stem, self.full_frame.image, labels, self.full_frame.resolved)
                    self.status = f"Saved {out.name}"

        if self.frame is not None and self.frame.scene is None:
            r = self.frame.resolved["scanner"]
            imgui.separator_text("This sample")
            imgui.text(f"Scan: {r['dpi']:.0f} dpi, {r['color']}, skew {r['skew_deg']:.2f} deg")
        elif self.frame is not None:
            r = self.frame.resolved
            imgui.separator_text("This sample")
            lt = r["light"]
            imgui.text(f"Lamp: {lt['elevation_deg']:.0f} deg up, {lt['azimuth_deg']:.0f} deg around")
            imgui.text(f"Crumple: {r['deform']['crumple']['mode']}, {len(r['deform']['folds'])} folds")
        if self.status:
            imgui.text_wrapped(self.status)


def _flat(merged: dict[str, Any], key: str) -> bool:
    """True when the crumple, the folds or the edge lift (`key`) leave the page flat."""
    d = merged["deform"]
    if key == "crumple":
        return d["crumple"]["strength"] == 0
    if key == "folds":
        return d["folds"] == []
    return all(v == 0 for v in d["edge_lift_cm"])


def _translation(p: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = p
    return m


def _upload(ctx: moderngl.Context, tex: moderngl.Texture | None, img: np.ndarray) -> moderngl.Texture:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if tex is None or tex.size != (w, h):
        if tex is not None:
            tex.release()
        tex = ctx.texture((w, h), 3, img.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    else:
        tex.write(img.tobytes())
    return tex


def _on_checkerboard(rgba: np.ndarray, cell: int = 12) -> np.ndarray:
    h, w = rgba.shape[:2]
    y, x = np.mgrid[:h, :w]
    board = np.where(((x // cell) + (y // cell)) % 2 == 0, 200, 150).astype(np.float32)[..., None]
    a = rgba[..., 3:].astype(np.float32) / 255.0
    return (rgba[..., :3] * a + board * (1.0 - a) + 0.5).astype(np.uint8)


def _sample_page() -> np.ndarray:
    """A simple built-in page, so the editor works before a document is opened."""
    w, h = 1240, 1754
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        title = ImageFont.truetype("Helvetica", 64)
        body = ImageFont.truetype("Helvetica", 30)
    except OSError:
        title = body = ImageFont.load_default()
    d.text((110, 120), "Recapture", font=title, fill=(20, 20, 20))
    d.text((110, 210), "Open a document to replace this sample page.", font=body, fill=(60, 60, 60))
    rng = np.random.default_rng(1)
    y = 320
    while y < h - 200:
        for _ in range(int(rng.integers(3, 8))):
            d.rectangle((110, y, 110 + int(rng.integers(600, 1020)), y + 14), fill=(70, 70, 70))
            y += 34
        y += 40
    return np.asarray(img.convert("RGBA"))


def run(config_path: Path | None = None, doc_path: Path | None = None,
        screenshot: Path | None = None, setup=None) -> None:
    """Open the editor. With `screenshot`, draw a few frames, save the window
    image and exit (used to check the UI without a person at the screen).
    `setup(editor)` runs once after the GL context exists."""
    editor = Editor(config_path, doc_path)
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Recapture"
    params.app_window_params.window_geometry.size = (1700, 1000)
    params.fps_idling.enable_idling = screenshot is None
    # Keep the window layout file out of the folder the editor is started from.
    params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    params.ini_filename = "recapture/imgui.ini"
    frames = {"n": 0}

    def gui() -> None:
        editor.gui()
        if frames["n"] == 1 and setup is not None:
            setup(editor)
        frames["n"] += 1
        if screenshot is not None and frames["n"] > 14 and not editor.dirty and not editor.quality_pending:
            hello_imgui.get_runner_params().app_shall_exit = True

    params.callbacks.show_gui = gui
    immapp.run(params)
    if screenshot is not None:
        Image.fromarray(np.asarray(hello_imgui.final_app_window_screenshot())).save(screenshot)
