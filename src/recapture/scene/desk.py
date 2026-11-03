"""Desk items around the page: keyboard, mousepad, phone, mug,
pen, pencil, sticky notes, other paper.

Each item is built here from a few smooth shapes (rounded extrusions,
lathed profiles, swept tubes), with a color per vertex and, where it
helps, a small texture (key labels, fabric, a printed page). Items sit on
the desk (z = 0) beside the page, never on it: `side` picks the page edge,
`along` the place along that edge, `gap_cm` the distance from it. Most of
them reach past the frame, so the photo shows only a part, like the edge
of a keyboard. An item that would cut into one placed before it is left out.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

KINDS = ("keyboard", "mousepad", "phone", "mug", "pen", "pencil", "sticky_notes", "paper")
SIDES = ("left", "right", "top", "bottom")

# Linear RGB palettes per item (the main color); a config `color` overrides them.
_PALETTE = {
    "keyboard": [(0.02, 0.02, 0.022), (0.75, 0.75, 0.74), (0.32, 0.33, 0.35)],
    "mousepad": [(0.02, 0.02, 0.022), (0.035, 0.04, 0.06), (0.10, 0.10, 0.10), (0.05, 0.02, 0.02)],
    "phone": [(0.01, 0.01, 0.012), (0.55, 0.55, 0.56), (0.05, 0.08, 0.16), (0.45, 0.38, 0.30)],
    "mug": [(0.80, 0.80, 0.78), (0.02, 0.02, 0.025), (0.05, 0.12, 0.35), (0.45, 0.04, 0.03), (0.55, 0.45, 0.30)],
    "pen": [(0.02, 0.03, 0.12), (0.01, 0.01, 0.01), (0.55, 0.02, 0.02), (0.75, 0.75, 0.72)],
    "pencil": [(0.85, 0.55, 0.04), (0.04, 0.10, 0.04), (0.60, 0.05, 0.05)],
    "sticky_notes": [(0.95, 0.80, 0.18), (0.95, 0.45, 0.55), (0.40, 0.75, 0.90), (0.55, 0.85, 0.40)],
    "paper": [(1.0, 1.0, 0.98)],
}
_METAL = (0.55, 0.55, 0.53)


@dataclass(frozen=True)
class DeskMesh:
    pos: np.ndarray  # (N, 3) float32, world cm
    normal: np.ndarray  # (N, 3) float32
    color: np.ndarray  # (N, 3) float32, linear RGB albedo
    uv: np.ndarray  # (N, 2) float32
    indices: np.ndarray  # (M,) uint32
    roughness: float
    image: np.ndarray | None = None  # RGBA uint8, multiplied by the color (labels, fabric, print)


def build_items(items: list[dict[str, Any]], page_center: np.ndarray, page_rotation_deg: float,
                page_size: tuple[float, float], seed: int, load_image) -> list[DeskMesh]:
    """Desk meshes for the resolved `desk.items`. `load_image(path)` gives RGBA for paper items."""
    rng = np.random.default_rng(seed + 11)
    out: list[DeskMesh] = []
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for it in items:
        kind = it["kind"]
        pal = _PALETTE[kind]
        color = np.asarray(it["color"] if it["color"] is not None else pal[rng.integers(len(pal))], dtype=np.float64)
        if kind == "paper":
            parts, size = _paper(color, load_image(it["image"]) if it["image"] else None)
        else:
            parts, size = _BUILDERS[kind](color, rng)
        placed = [_place(m, size, it, page_center, page_rotation_deg, page_size) for m in parts]
        pts = np.concatenate([m.pos[:, :2] for m in placed])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if any(np.all(lo < h2 + 0.5) and np.all(l2 - 0.5 < hi) for l2, h2 in boxes):
            continue  # it would cut into an item already on the desk
        boxes.append((lo, hi))
        out += placed
    return out


def _place(mesh: DeskMesh, size: tuple[float, float], it: dict, center: np.ndarray, page_rot: float,
           page_size: tuple[float, float]) -> DeskMesh:
    """Put an item (built around the origin) beside the page edge `side`."""
    w, h = page_size
    a = np.radians(float(it["rotation_deg"]))
    # Extent of the rotated item along the page axes, for the gap.
    ex = 0.5 * (abs(np.cos(a)) * size[0] + abs(np.sin(a)) * size[1])
    ey = 0.5 * (abs(np.sin(a)) * size[0] + abs(np.cos(a)) * size[1])
    along, gap = float(it["along"]) - 0.5, float(it["gap_cm"])
    local = {
        "right": (w / 2 + gap + ex, -along * h),
        "left": (-(w / 2 + gap + ex), -along * h),
        "top": (along * w, h / 2 + gap + ey),
        "bottom": (along * w, -(h / 2 + gap + ey)),
    }[it["side"]]
    pr = np.radians(page_rot)
    rot_page = np.array([[np.cos(pr), -np.sin(pr)], [np.sin(pr), np.cos(pr)]])
    offset = rot_page @ np.asarray(local)
    t = pr + a
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    pos = mesh.pos @ rot.T + np.array([center[0] + offset[0], center[1] + offset[1], 0.0])
    return DeskMesh(pos=pos.astype(np.float32), normal=(mesh.normal @ rot.T).astype(np.float32), color=mesh.color,
                    uv=mesh.uv, indices=mesh.indices, roughness=mesh.roughness, image=mesh.image)


# ---- mesh tools -------------------------------------------------------------------

class _Mesh:
    """Vertices and triangles. Normals are averaged over the triangles that
    share a vertex (smooth), unless a vertex was given its own (flat).
    Triangles are counterclockwise seen from outside."""

    def __init__(self):
        self.pos: list = []
        self.color: list = []
        self.uv: list = []
        self.normal: list = []  # None = smooth
        self.tris: list[int] = []

    def vertex(self, p, color, uv=(0.0, 0.0), normal=None) -> int:
        self.pos.append(np.asarray(p, dtype=np.float64))
        self.color.append(np.broadcast_to(np.asarray(color, dtype=np.float64), (3,)))
        self.uv.append(np.asarray(uv, dtype=np.float64))
        self.normal.append(None if normal is None else np.asarray(normal, dtype=np.float64))
        return len(self.pos) - 1

    def tri(self, a: int, b: int, c: int) -> None:
        self.tris += [a, b, c]

    def quad(self, a: int, b: int, c: int, d: int) -> None:
        self.tris += [a, b, c, a, c, d]

    def flat_quad(self, corners, color, uv=((0, 0), (1, 0), (1, 1), (0, 1))) -> None:
        corners = [np.asarray(c, dtype=np.float64) for c in corners]
        n = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        n /= max(np.linalg.norm(n), 1e-12)
        ids = [self.vertex(c, color, t, n) for c, t in zip(corners, uv)]
        self.quad(*ids)

    def transform(self, rot: np.ndarray, shift=(0.0, 0.0, 0.0), start: int = 0) -> None:
        """Rotate, then move, the vertices added since `start`."""
        for k in range(start, len(self.pos)):
            self.pos[k] = rot @ self.pos[k] + np.asarray(shift, dtype=np.float64)
            if self.normal[k] is not None:
                self.normal[k] = rot @ self.normal[k]

    def mesh(self, roughness: float, image: np.ndarray | None = None) -> DeskMesh:
        pos = np.array(self.pos)
        tri = np.array(self.tris, dtype=np.int64).reshape(-1, 3)
        acc = np.zeros_like(pos)
        face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        for k in range(3):
            np.add.at(acc, tri[:, k], face)
        for k, n in enumerate(self.normal):
            if n is not None:
                acc[k] = n
        acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
        return DeskMesh(pos=pos.astype(np.float32), normal=acc.astype(np.float32),
                        color=np.array(self.color, np.float32), uv=np.array(self.uv, np.float32),
                        indices=tri.astype(np.uint32).ravel(), roughness=roughness, image=image)


def _rounded_rect(w: float, d: float, r: float, seg: int = 5) -> np.ndarray:
    """Outline of a rounded rectangle centered on the origin, counterclockwise."""
    r = float(np.clip(r, 1e-3, 0.5 * min(w, d) - 1e-4))
    out = []
    for cx, cy, a0 in ((w / 2 - r, -d / 2 + r, -90), (w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90),
                       (-w / 2 + r, -d / 2 + r, 180)):
        for k in range(seg + 1):
            a = np.radians(a0 + 90 * k / seg)
            out.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(out)


def _extrude(m: _Mesh, w: float, d: float, r: float, profile, color, top_color=None, top_uv=None,
             seg: int = 5) -> None:
    """A rounded block: rings of the outline at each (inset, z) of `profile`,
    bottom to top, joined smoothly, and a flat top. `color` may be a
    function of the profile index (for darker gaps at the bottom).
    `top_uv(x, y)` maps the top face to texture coordinates."""
    rings = []
    for i, (inset, z) in enumerate(profile):
        c = color(i) if callable(color) else color
        ring = _rounded_rect(w - 2 * inset, d - 2 * inset, r - inset, seg)
        rings.append([m.vertex((x, y, z), c) for x, y in ring])
    n = len(rings[0])
    for lo, hi in zip(rings, rings[1:]):
        for k in range(n):
            m.quad(lo[k], lo[(k + 1) % n], hi[(k + 1) % n], hi[k])
    inset, z = profile[-1]
    tc = top_color if top_color is not None else (color(len(profile) - 1) if callable(color) else color)
    ring = _rounded_rect(w - 2 * inset, d - 2 * inset, r - inset, seg)
    up = (0.0, 0.0, 1.0)
    uv_of = top_uv or (lambda x, y: (0.0, 0.0))
    center = m.vertex((0, 0, z), tc, uv_of(0.0, 0.0), up)
    ids = [m.vertex((x, y, z), tc, uv_of(x, y), up) for x, y in ring]
    for k in range(n):
        m.tri(center, ids[k], ids[(k + 1) % n])


def _lathe(m: _Mesh, profile, color, seg: int = 32, cap: bool = False) -> None:
    """Revolve (radius, z) points around the z axis, in profile order:
    faces look to the right of the profile direction, so a profile that
    climbs the outside looks outward. `cap` closes the last ring with a
    flat disk facing along the profile's last step."""
    rings = []
    angles = np.linspace(0.0, 2 * np.pi, seg, endpoint=False)
    for i, (r, z) in enumerate(profile):
        c = color(i) if callable(color) else color
        rings.append([m.vertex((r * np.cos(a), r * np.sin(a), z), c) for a in angles])
    for lo, hi in zip(rings, rings[1:]):
        for k in range(seg):
            m.quad(lo[k], lo[(k + 1) % seg], hi[(k + 1) % seg], hi[k])
    if cap:
        r, z = profile[-1]
        up = 1.0 if profile[-1][1] >= profile[-2][1] else -1.0
        c = color(len(profile) - 1) if callable(color) else color
        center = m.vertex((0.0, 0.0, z), c, normal=(0.0, 0.0, up))
        last = [m.vertex(m.pos[k], c, normal=(0.0, 0.0, up)) for k in rings[-1]]
        for k in range(seg):
            if up > 0:
                m.tri(center, last[k], last[(k + 1) % seg])
            else:
                m.tri(center, last[(k + 1) % seg], last[k])


_Z_TO_X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # lays a lathed shape along x


def _srgb8(linear) -> tuple[int, int, int]:
    c = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    s = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return tuple(int(round(v * 255)) for v in s)


# ---- keyboard -----------------------------------------------------------------------

# A compact keyboard: rows of (label, width in key units). Labels print on the keys.
_KEY_ROWS = [
    [("esc", 1.4)] + [(f"F{k}", 1.0) for k in range(1, 13)] + [("", 1.0)],
    [("`", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1),
     ("-", 1), ("=", 1), ("delete", 1.5)],
    [("tab", 1.5), ("Q", 1), ("W", 1), ("E", 1), ("R", 1), ("T", 1), ("Y", 1), ("U", 1), ("I", 1), ("O", 1),
     ("P", 1), ("[", 1), ("]", 1), ("\\", 1)],
    [("caps", 1.75), ("A", 1), ("S", 1), ("D", 1), ("F", 1), ("G", 1), ("H", 1), ("J", 1), ("K", 1), ("L", 1),
     (";", 1), ("'", 1), ("return", 1.75)],
    [("shift", 2.25), ("Z", 1), ("X", 1), ("C", 1), ("V", 1), ("B", 1), ("N", 1), ("M", 1), (",", 1), (".", 1),
     ("/", 1), ("shift", 2.25)],
    [("fn", 1), ("ctrl", 1), ("opt", 1), ("cmd", 1.25), ("", 5.0), ("cmd", 1.25), ("opt", 1), ("<", 1), ("v", 1),
     (">", 1)],
]
_UNIT = 1.85  # cm per key unit
_ROW_UNITS = 14.5
_CELL, _COLS = 96, 12  # key label texture layout


@lru_cache(maxsize=8)
def _key_atlas(key_rgb: tuple[int, int, int], ink_rgb: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, int]]:
    """One texture with every key label, on the key color. Cell 0 is blank."""
    from PIL import Image, ImageDraw, ImageFont

    labels = sorted({lab for row in _KEY_ROWS for lab, _ in row if lab})
    rows = (len(labels) + 1 + _COLS - 1) // _COLS
    img = Image.new("RGBA", (_COLS * _CELL, rows * _CELL), key_rgb + (255,))
    draw = ImageDraw.Draw(img)
    big, small = ImageFont.load_default(size=40), ImageFont.load_default(size=21)
    index = {"": 0}
    for k, lab in enumerate(labels, start=1):
        x, y = (k % _COLS) * _CELL, (k // _COLS) * _CELL
        font = big if len(lab) == 1 else small
        box = draw.textbbox((0, 0), lab, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        if len(lab) == 1:  # characters in the middle, words at the bottom left, like on a laptop
            pos = (x + (_CELL - tw) / 2 - box[0], y + (_CELL - th) / 2 - box[1])
        else:
            pos = (x + 12 - box[0], y + _CELL - th - 14 - box[1])
        draw.text(pos, lab, font=font, fill=ink_rgb + (255,))
        index[lab] = k
    return np.asarray(img), index


def _keyboard(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A low-profile compact keyboard: a rounded wedge base and sculpted
    keycaps with printed labels. It depends only on its color, so each
    color is built once."""
    return _keyboard_of(tuple(float(v) for v in color))


@lru_cache(maxsize=16)
def _keyboard_of(color_t: tuple[float, float, float]) -> tuple[list[DeskMesh], tuple[float, float]]:
    color = np.asarray(color_t)
    dark = float(color.mean()) < 0.2
    base_color = np.asarray((0.03, 0.03, 0.033) if dark else _METAL)
    ink = (0.6, 0.6, 0.6) if dark else (0.04, 0.04, 0.04)
    atlas, index = _key_atlas(_srgb8(color), _srgb8(ink))
    rows_n = atlas.shape[0] // _CELL

    width = _ROW_UNITS * _UNIT + 1.4
    depth = 5.6 * _UNIT + 1.6
    base_h = 0.55
    base = _Mesh()
    _extrude(base, width, depth, 0.9, [(0.0, 0.0), (0.0, base_h - 0.12), (0.06, base_h - 0.03), (0.14, base_h)],
             base_color)
    caps = _Mesh()
    gap, cap_h = 0.28, 0.42
    ao = lambda i: np.full(3, (0.5, 0.8, 0.95, 1.0)[min(i, 3)])  # noqa: E731  darker toward the gaps
    y = depth / 2 - 0.85
    for r, row in enumerate(_KEY_ROWS):
        h = _UNIT * (0.6 if r == 0 else 1.0)
        stretch = _ROW_UNITS / sum(wu for _, wu in row)
        x = -width / 2 + 0.7
        for lab, wu in row:
            kw = wu * _UNIT * stretch
            cw, ch = kw - gap, h - gap
            k = index.get(lab, 0)
            u0, v0 = (k % _COLS) / _COLS, (k // _COLS) / rows_n
            top_w, top_h = cw - 0.24, ch - 0.24

            def top_uv(px, py, u0=u0, v0=v0, top_w=top_w, top_h=top_h):
                # Wide keys keep a square label area at their left.
                if top_w > top_h * 1.3:
                    fx = float(np.clip((px + top_w / 2) / max(top_h, 1e-3), 0.0, 1.0))
                else:
                    fx = px / top_w + 0.5
                fy = 0.5 - py / top_h
                return (u0 + fx / _COLS, v0 + fy / rows_n)

            start = len(caps.pos)
            _extrude(caps, cw, ch, 0.22, [(0.0, 0.0), (0.03, 0.12), (0.10, cap_h - 0.06), (0.12, cap_h)], ao,
                     top_color=np.ones(3), top_uv=top_uv, seg=3)
            caps.transform(np.eye(3), (x + kw / 2, y - h / 2, base_h - 0.05), start)
            x += kw
        y -= h
    # The back stands higher than the front, like most keyboards.
    tilt = np.radians(1.6)
    rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    for mesh in (base, caps):
        mesh.transform(rot, (0.0, 0.0, np.sin(tilt) * depth / 2))
    return [base.mesh(roughness=0.5 if dark else 0.35), caps.mesh(roughness=0.55, image=atlas)], (width, depth)


# ---- mousepad --------------------------------------------------------

def _mousepad(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A thin pad with rounded corners, a stitched rim and a woven top."""
    w, d, h = 26.0, 21.0, 0.3
    m = _Mesh()
    rim = color * 0.55
    _extrude(m, w, d, 1.2, [(0.0, 0.0), (0.0, h - 0.06), (0.05, h)], lambda i: rim if i < 2 else color,
             top_color=color, top_uv=lambda x, y: (x / w + 0.5, 0.5 - y / d), seg=8)
    return [m.mesh(roughness=0.95, image=_fabric(int(rng.integers(4))))], (w, d)


@lru_cache(maxsize=4)
def _fabric(seed: int, size: int = 512) -> np.ndarray:
    """A fine woven texture around light gray (the item color multiplies it)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size].astype(np.float32)
    weave = 0.5 + 0.5 * np.sin(x * 2.2) * np.sin(y * 2.2)
    coarse = rng.normal(0.0, 1.0, (size // 8, size // 8)).astype(np.float32)
    noise = np.asarray(Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC))
    v = 0.8 + 0.12 * weave + 0.06 * noise
    img = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([img, img, img, np.full_like(img, 255)])


# ---- phone ----------------------------------------------------------------------------

def _phone(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A phone lying face up (black glass screen) or face down (camera bump)."""
    w, d, h, r = 7.2, 14.7, 0.8, 0.95
    body = _Mesh()
    _extrude(body, w, d, r, [(0.0, 0.0), (0.0, 0.1), (0.02, h - 0.12), (0.12, h)], color, seg=8)
    parts = [body.mesh(roughness=0.3)]
    if rng.random() < 0.6:  # face up: the screen, black glass with a thin border
        glass = _Mesh()
        _extrude(glass, w - 0.3, d - 0.3, r - 0.15, [(0.0, h - 0.01), (0.0, h + 0.005)], (0.008, 0.008, 0.01), seg=8)
        parts.append(glass.mesh(roughness=0.06))
    else:  # face down: the camera bump with three lenses
        bump = _Mesh()
        _extrude(bump, 3.2, 3.2, 0.8, [(0.0, h - 0.01), (0.0, h + 0.12), (0.06, h + 0.16)], color * 0.8, seg=6)
        bump.transform(np.eye(3), (-w / 2 + 2.0, d / 2 - 2.0, 0.0))
        lens = _Mesh()
        for cx, cy in ((-0.7, 0.7), (0.7, 0.7), (-0.7, -0.7)):
            s = len(lens.pos)
            _lathe(lens, [(0.55, h + 0.15), (0.55, h + 0.28), (0.45, h + 0.3)], (0.01, 0.01, 0.012), seg=20, cap=True)
            lens.transform(np.eye(3), (-w / 2 + 2.0 + cx, d / 2 - 2.0 + cy, 0.0), s)
        parts += [bump.mesh(roughness=0.25), lens.mesh(roughness=0.08)]
    return parts, (w, d)


# ---- mug ------------------------------------------------------------------------------

def _mug(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A ceramic mug with coffee in it and a handle."""
    r0, r1, hgt, wall = 3.7, 4.0, 9.5, 0.35
    m = _Mesh()
    # Outside up, over the rounded lip, inside down.
    profile = [(r0 - 0.3, 0.0), (r0, 0.25), (r0 + 0.1, 2.0), (r1, hgt - 0.3), (r1 - 0.05, hgt),
               (r1 - wall + 0.05, hgt), (r1 - wall, hgt - 0.3), (r0 - wall + 0.1, 1.0)]
    _lathe(m, profile, color, seg=40)
    coffee = _Mesh()
    level = hgt - rng.uniform(1.2, 3.0)
    inner = r0 - wall + 0.1 + (r1 - r0) * (level - 1.0) / (hgt - 1.3)
    _lathe(coffee, [(inner, level), (0.0, level)], (0.03, 0.012, 0.005), seg=40)
    # Handle: a tube swept along a half ring on the +x side.
    handle = _Mesh()
    ring_r, tube_r, zc, cx0 = 2.5, 0.45, hgt * 0.55, r1 - 0.35
    n_a, n_t = 20, 12
    rows = []
    for i in range(n_a + 1):
        a = -np.pi / 2 + np.pi * i / n_a
        cx, cz = cx0 + ring_r * 0.75 * np.cos(a), zc + ring_r * np.sin(a)
        nx, nz = np.cos(a), np.sin(a)  # away from the ring center
        rows.append([handle.vertex((cx + tube_r * np.cos(t) * nx, tube_r * np.sin(t), cz + tube_r * np.cos(t) * nz),
                                   color) for t in np.linspace(0, 2 * np.pi, n_t, endpoint=False)])
    for i in range(n_a):
        for j in range(n_t):
            handle.quad(rows[i][j], rows[i + 1][j], rows[i + 1][(j + 1) % n_t], rows[i][(j + 1) % n_t])
    parts = [m.mesh(roughness=0.22), coffee.mesh(roughness=0.12), _outward(handle.mesh(roughness=0.22), (cx0, 0, zc))]
    return parts, (2 * r1 + ring_r * 0.75 + 0.6, 2 * r1)


def _outward(mesh: DeskMesh, center) -> DeskMesh:
    """Flip a tube's faces if they point toward the center of its ring."""
    d = mesh.pos - np.asarray(center, dtype=np.float32)
    if float((mesh.normal * d).sum()) >= 0:
        return mesh
    tris = mesh.indices.reshape(-1, 3)[:, ::-1].ravel()
    return DeskMesh(mesh.pos, -mesh.normal, mesh.color, mesh.uv, tris, mesh.roughness, mesh.image)


# ---- pen and pencil -----------------------------------------------------------------

def _pen(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A ballpoint pen lying on the desk: round body, metal tip, a clip on its side."""
    r = 0.45
    body = _Mesh()
    _lathe(body, [(0.0, -7.0), (0.25, -6.95), (r, -6.7), (r, 3.2), (r * 0.92, 3.4)], color, seg=24)
    tip = _Mesh()
    _lathe(tip, [(r * 0.9, 3.35), (r * 0.6, 5.6), (0.18, 6.8), (0.06, 7.0), (0.0, 7.05)], _METAL, seg=24)
    parts = []
    for mesh, rough in ((body, 0.35), (tip, 0.2)):
        mesh.transform(_Z_TO_X, (0.0, 0.0, r))
        parts.append(mesh.mesh(roughness=rough))
    clip = _Mesh()
    _extrude(clip, 3.8, 0.28, 0.12, [(0.0, 0.0), (0.0, 0.18)], _METAL, seg=2)
    clip.transform(np.eye(3), (-3.9, 0.0, 2 * r - 0.02))
    parts.append(clip.mesh(roughness=0.25))
    return parts, (14.2, 1.0)


def _pencil(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A hexagonal pencil: painted body, sharpened wood, graphite, metal band, eraser."""
    r, length = 0.37, 15.0
    body = _Mesh()
    hexagon = [(r * np.cos(np.pi / 6 + k * np.pi / 3), r * np.sin(np.pi / 6 + k * np.pi / 3)) for k in range(6)]
    for k in range(6):  # flat painted faces
        (x0, y0), (x1, y1) = hexagon[k], hexagon[(k + 1) % 6]
        body.flat_quad([(x0, y0, -length / 2), (x1, y1, -length / 2), (x1, y1, length / 2), (x0, y0, length / 2)],
                       color)
    inner = r * np.cos(np.pi / 6)  # the hexagon rests on a flat side: round parts stay within this radius
    wood = _Mesh()
    _lathe(wood, [(inner, length / 2), (0.12, length / 2 + 1.5)], (0.75, 0.52, 0.30), seg=18)
    lead = _Mesh()
    _lathe(lead, [(0.12, length / 2 + 1.5), (0.02, length / 2 + 1.85), (0.0, length / 2 + 1.87)], (0.04, 0.04, 0.045),
           seg=18)
    band = _Mesh()
    _lathe(band, [(inner, -length / 2 - 1.1), (inner, -length / 2 + 0.05)], _METAL, seg=18)
    eraser = _Mesh()
    _lathe(eraser, [(0.0, -length / 2 - 1.92), (inner * 0.7, -length / 2 - 1.9), (inner * 0.97, -length / 2 - 1.8),
                    (inner * 0.97, -length / 2 - 1.1)], (0.85, 0.45, 0.45), seg=18)
    parts = []
    for mesh, rough in ((body, 0.4), (wood, 0.8), (lead, 0.3), (band, 0.25), (eraser, 0.9)):
        mesh.transform(_Z_TO_X, (0.0, 0.0, r * np.cos(np.pi / 6)))
        parts.append(mesh.mesh(roughness=rough))
    return parts, (length + 3.8, 0.8)


# ---- sticky notes and paper -----------------------------------------------------------

def _sticky_notes(color: np.ndarray, rng: np.random.Generator) -> tuple[list[DeskMesh], tuple[float, float]]:
    """A pad of sticky notes: a stack with faint sheet lines and a top sheet curling up."""
    s, h = 7.6, 0.9
    stack = _Mesh()
    sheets = 9
    prof = [(0.0, 0.0)] + [(0.0, h * k / sheets) for k in range(1, sheets + 1)]
    _extrude(stack, s, s, 0.08, prof, lambda i: color * (0.8 if i % 2 else 0.9), top_color=color, seg=2)
    top = _Mesh()
    lift = rng.uniform(0.1, 0.5)
    n = 8
    ids = []
    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            x, y = -s / 2 + s * j / n, -s / 2 + s * i / n
            z = h + 0.02 + lift * max(0.0, (y + s / 2) / s - 0.6) ** 2 * 6.0  # the far end curls up
            row.append(top.vertex((x, y, z), color))
        ids.append(row)
    for i in range(n):
        for j in range(n):
            top.quad(ids[i][j], ids[i][j + 1], ids[i + 1][j + 1], ids[i + 1][j])
    return [stack.mesh(roughness=0.85), top.mesh(roughness=0.85)], (s, s)


def _paper(color: np.ndarray, image: np.ndarray | None) -> tuple[list[DeskMesh], tuple[float, float]]:
    m = _Mesh()
    w, h, z = 21.0, 29.7, 0.02
    m.flat_quad([(-w / 2, -h / 2, z), (w / 2, -h / 2, z), (w / 2, h / 2, z), (-w / 2, h / 2, z)], color,
                uv=((0, 1), (1, 1), (1, 0), (0, 0)))
    return [m.mesh(roughness=0.8, image=image)], (w, h)


_BUILDERS = {"keyboard": _keyboard, "mousepad": _mousepad, "phone": _phone, "mug": _mug,
             "pen": _pen, "pencil": _pencil, "sticky_notes": _sticky_notes}
