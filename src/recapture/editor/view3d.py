"""3D scene view math: the editor's orbit camera, and turning moved objects
back into config values. Pure numpy, no UI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..scene.camera import Camera, perspective


@dataclass
class OrbitCamera:
    """A free camera that orbits a pivot point, z up (like Blender's viewport)."""
    pivot: np.ndarray
    distance: float = 75.0
    yaw: float = -90.0  # degrees; -90 looks from -y toward +y, like the default photo camera
    pitch: float = 35.0
    fov_deg: float = 40.0

    @property
    def eye(self) -> np.ndarray:
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return self.pivot + self.distance * np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])

    def view(self) -> np.ndarray:
        return z_up_look_at(self.eye, self.pivot)

    def proj(self, aspect: float) -> np.ndarray:
        t = math.tan(math.radians(self.fov_deg) / 2)
        return perspective(t * aspect, t, 1.0, 3000.0)

    def camera(self, width: int, height: int) -> Camera:
        """A renderer camera for this view."""
        return Camera(position=self.eye, view=self.view(), proj=self.proj(width / height),
                      width=width, height=height, near=1.0, far=3000.0)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.4
        self.pitch = float(np.clip(self.pitch + dy * 0.4, -5.0, 89.0))

    def pan(self, dx: float, dy: float, height_px: float) -> None:
        v = self.view()
        right, up = v[0, :3], v[1, :3]
        per_px = 2 * self.distance * math.tan(math.radians(self.fov_deg) / 2) / max(height_px, 1.0)
        self.pivot = self.pivot - right * dx * per_px + up * dy * per_px

    def zoom(self, wheel: float) -> None:
        self.distance = float(np.clip(self.distance * (1.0 - 0.1 * wheel), 5.0, 500.0))


def z_up_look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def project(view_proj: np.ndarray, points: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """World points -> pixel coords in a (width, height) viewport, and a 'in front' mask."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    clip = np.concatenate([p, np.ones((len(p), 1))], 1) @ view_proj.T
    w = clip[:, 3]
    front = w > 1e-3
    ndc = clip[:, :2] / np.where(front, w, 1.0)[:, None]
    px = np.stack([(ndc[:, 0] + 1) * 0.5 * width, (1 - ndc[:, 1]) * 0.5 * height], 1)
    return px, front


def frustum_lines(cam: Camera, depth: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Line segments of a camera's viewing pyramid, `depth` cm long."""
    inv = np.linalg.inv(cam.view)
    eye = cam.position
    tx, ty = 1.0 / cam.proj[0, 0], 1.0 / cam.proj[1, 1]
    corners = [inv[:3, :3] @ np.array([sx * tx * depth, sy * ty * depth, -depth]) + eye
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    lines = [(eye, c) for c in corners]
    lines += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    up = inv[:3, :3] @ np.array([0.0, ty * depth * 1.25, -depth]) + eye  # marks the top of the photo
    lines += [(corners[2], up), (up, corners[3])]
    return lines


def lamp_angles(position: np.ndarray, page_center: np.ndarray) -> tuple[float, float, float]:
    """(azimuth_deg, elevation_deg, distance_cm) of a lamp position seen from the page center."""
    v = np.asarray(position, float) - page_center
    d = float(np.linalg.norm(v))
    el = math.degrees(math.asin(np.clip(v[2] / max(d, 1e-9), -1.0, 1.0)))
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, float(np.clip(el, 1.0, 90.0)), d


def occluder_anchor(shape_name: str, corners: np.ndarray) -> np.ndarray:
    """World position of a shadow caster's anchor (the point aimed at the page)."""
    from ..scene.occluders import SHAPES

    s = SHAPES[shape_name]
    fx, fy = s.anchor[0] / s.width, s.anchor[1] / s.length
    return corners[0] + fx * (corners[1] - corners[0]) + fy * (corners[3] - corners[0])
