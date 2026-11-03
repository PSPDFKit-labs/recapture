"""Pinhole camera with OpenGL conventions.

Pixel coordinates: origin at the top-left corner of the image, x to the
right, y down. Pixel (i, j) covers [i, i+1] x [j, j+1], so its center is at
(i + 0.5, j + 0.5). The renderer flips GL rows to match this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FULL_FRAME_DIAGONAL_MM = 43.267  # 36 x 24 mm sensor, for 35 mm equivalent focal length


@dataclass(frozen=True)
class Camera:
    position: np.ndarray  # (3,)
    view: np.ndarray  # (4, 4) world -> eye
    proj: np.ndarray  # (4, 4) eye -> clip
    width: int
    height: int
    near: float
    far: float

    @property
    def view_proj(self) -> np.ndarray:
        return self.proj @ self.view

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixel coords (N, 2) and clip w (N,). w <= 0 is behind the camera."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        clip = np.concatenate([p, np.ones((len(p), 1))], 1) @ self.view_proj.T
        w = clip[:, 3]
        safe_w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip[:, :2] / safe_w[:, None]
        px = np.stack([(ndc[:, 0] + 1.0) * 0.5 * self.width, (1.0 - ndc[:, 1]) * 0.5 * self.height], 1)
        return px, w

    def ground_point(self, px, z: float = 0.0) -> np.ndarray:
        """The world point at height `z` seen at pixel `px`."""
        x, y = float(px[0]), float(px[1])
        ndc = np.array([2.0 * x / self.width - 1.0, 1.0 - 2.0 * y / self.height])
        inv = np.linalg.inv(self.view_proj)
        a, b = (inv @ np.array([*ndc, d, 1.0]) for d in (-1.0, 1.0))
        a, b = a[:3] / a[3], b[:3] / b[3]
        t = (z - a[2]) / (b[2] - a[2]) if abs(b[2] - a[2]) > 1e-12 else 0.0
        return a + t * (b - a)


def look_at(eye: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # Up hint between +y and +z: a camera that looks down shows the page top
    # (+y) at the top of the image, and a tilted camera keeps the horizon level.
    up_hint = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    if roll_deg:
        a = np.radians(roll_deg)
        right, up = np.cos(a) * right + np.sin(a) * up, -np.sin(a) * right + np.cos(a) * up
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(tan_half_x: float, tan_half_y: float, near: float, far: float) -> np.ndarray:
    m = np.zeros((4, 4))
    m[0, 0] = 1.0 / tan_half_x
    m[1, 1] = 1.0 / tan_half_y
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -2.0 * far * near / (far - near)
    m[3, 2] = -1.0
    return m


def make_camera(position, target, roll_deg: float, focal_mm: float, width: int, height: int,
                near: float = 1.0, far: float = 5000.0) -> Camera:
    eye = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if np.linalg.norm(tgt - eye) < 1e-6:
        raise ValueError("camera position and target are the same point")
    # A 35 mm equivalent focal length sets the diagonal field of view.
    tan_half_diag = FULL_FRAME_DIAGONAL_MM / (2.0 * focal_mm)
    diag = np.hypot(width, height)
    proj = perspective(tan_half_diag * width / diag, tan_half_diag * height / diag, near, far)
    return Camera(position=eye, view=look_at(eye, tgt, roll_deg), proj=proj,
                  width=int(width), height=int(height), near=near, far=far)


def fit_camera(points: np.ndarray, position, target, roll_deg: float, focal_mm: float, width: int, height: int,
               margin: float, page_axis=None) -> Camera:
    """A camera that frames `points` (the page border) as tightly as `margin` allows.

    The view direction comes from `position` and `target`: it sets how much
    the photo is tilted. The camera then moves along that direction and to
    the side until the points fill the frame. `margin` is the space left on
    the tightest side, as a part of the short image side. With `page_axis`
    (the page's x axis in the world), the camera first turns so the page
    edges line up with the image edges; `roll_deg` is added after that.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    direction = np.asarray(position, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        raise ValueError("camera position and target are the same point")
    direction /= dist
    aim = np.array([*pts[:, :2].mean(axis=0), float(np.asarray(target, dtype=np.float64)[2])])

    def camera(roll: float) -> Camera:
        return make_camera(aim + direction * dist, aim, roll, focal_mm, width, height)

    roll = 0.0
    if page_axis is not None:
        roll = _align_roll(camera, aim, np.asarray(page_axis, dtype=np.float64))
    roll += roll_deg
    m = margin * min(width, height)
    room = np.array([width - 2 * m, height - 2 * m], dtype=np.float64)
    if np.any(room <= 1):
        raise ValueError("camera.fit_margin leaves no room for the page")
    # Size on screen goes as 1 / distance, so a few steps converge.
    for _ in range(8):
        cam = camera(roll)
        px, w = cam.project(pts)
        if np.any(w <= cam.near):
            dist *= 2.0
            continue
        lo, hi = px.min(axis=0), px.max(axis=0)
        dist *= float(np.max((hi - lo) / room))
        aim = cam.ground_point((lo + hi) / 2, aim[2])  # center the page
    # The last step can leave the page a hair too large: step back until it fits.
    for _ in range(50):
        cam = camera(roll)
        px, w = cam.project(pts)
        if np.all(w > cam.near) and px.min() >= m - 0.5 and np.all(px.max(axis=0) <= [width - m + 0.5, height - m + 0.5]):
            break
        dist *= 1.005
    return cam


def _align_roll(camera, aim: np.ndarray, axis: np.ndarray) -> float:
    """The roll that turns the page edges (seen at `aim`) to the nearest image edges.

    Seen from the side, the page's two axes are not square on screen, so this
    splits the difference between them.
    """
    axes = np.stack([axis, np.cross([0.0, 0.0, 1.0], axis)])

    def angle(roll: float) -> float:
        px, _ = camera(roll).project(np.concatenate([aim[None], aim + 5.0 * axes]))
        d = px[1:] - px[0]
        a = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        left = (a + 45.0) % 90.0 - 45.0  # left over after the nearest quarter turn
        return float(np.degrees(np.arctan2(np.sin(np.radians(left * 4)).sum(), np.cos(np.radians(left * 4)).sum())) / 4)

    off = angle(0.0)
    # The roll sign depends on the axes; take the one that cancels the angle.
    return min((off, -off), key=lambda r: abs(angle(r)))
