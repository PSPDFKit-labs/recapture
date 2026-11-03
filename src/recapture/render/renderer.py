"""ModernGL renderer: shadow pass, color pass (MSAA), label pass.

Only the page is drawn. The background stays transparent and is added later
as a 2D layer (see fx.composite), so nothing casts onto it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import moderngl
import numpy as np

from ..scene.camera import look_at, perspective
from ..scene.occluders import occluder_mask
from ..scene.scene import Scene
from ..scene.surface import grain_texture
from . import shaders

GL_SRGB8_ALPHA8 = 0x8C43


@dataclass
class RenderOutput:
    color: np.ndarray | None  # (H, W, 3) float32, linear RGB premultiplied by alpha, row 0 = top
    alpha: np.ndarray | None  # (H, W) float32 page coverage, antialiased (None: left on the GPU)
    label: np.ndarray  # (H, W, 4) float32: u, v, page mask, view depth (cm)
    table: np.ndarray | None = None  # (H, W, 4) float32: light arriving at the table (RGB), view depth (cm)
    # The same buffers on the GPU, valid until the next render: color, label, table.
    textures: dict | None = None


def create_context() -> moderngl.Context:
    """Standalone context. Uses EGL on Linux when there is no display."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return moderngl.create_standalone_context(require=330, backend="egl")
    return moderngl.create_standalone_context(require=330)


def _cover_radius(scene: Scene) -> float:
    """Radius around the page center that the shadow map covers."""
    return 0.75 * float(np.hypot(scene.page.width_cm, scene.page.height_cm)) + 5.0


def _view_radius(scene: Scene) -> float:
    """Radius around the page center that covers the table the camera sees,
    for the table's own, wider shadow map."""
    cam = scene.camera
    inv = np.linalg.inv(cam.view)
    tx, ty = 1.0 / cam.proj[0, 0], 1.0 / cam.proj[1, 1]
    reach = _cover_radius(scene)
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        d = inv[:3, :3] @ np.array([sx * tx, sy * ty, -1.0])
        t = -cam.position[2] / d[2] if d[2] < -1e-6 else np.inf  # where the corner ray meets the table
        hit = cam.position + min(t, 400.0) * d
        reach = max(reach, float(np.linalg.norm(hit[:2] - scene.page_center[:2])))
    return min(reach * 1.05, 150.0)


_HEIGHT_SIZE = 512
_TABLE_HALF_CM = 1500.0  # the table reaches past anything the camera sees
_DISPLAY_TABLE_HALF_CM = 60.0  # in the editor's 3D view: a table you can see around


class Renderer:
    def __init__(self, ctx: moderngl.Context | None = None, shadow_size: int = 2048, samples: int = 4):
        self.ctx = ctx or create_context()
        self.shadow_size = shadow_size
        self.samples = min(samples, self.ctx.max_samples)
        prog = self.ctx.program
        self.shadow_prog = prog(vertex_shader=shaders.SHADOW_VS, fragment_shader=shaders.SHADOW_FS)
        self.color_prog = prog(vertex_shader=shaders.SCENE_VS, fragment_shader=shaders.COLOR_FS)
        self.label_prog = prog(vertex_shader=shaders.SCENE_VS, fragment_shader=shaders.LABEL_FS)
        self.table_prog = prog(vertex_shader=shaders.TABLE_VS, fragment_shader=shaders.TABLE_FS)
        self.prop_prog = prog(vertex_shader=shaders.PROP_VS, fragment_shader=shaders.PROP_FS)
        self.height_prog = prog(vertex_shader=shaders.HEIGHT_VS, fragment_shader=shaders.HEIGHT_FS)
        self.height_tex = self.ctx.texture((_HEIGHT_SIZE, _HEIGHT_SIZE), 1, dtype="f4")
        self.height_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.height_tex.repeat_x = self.height_tex.repeat_y = False
        self.height_fbo = self.ctx.framebuffer(
            color_attachments=[self.height_tex],
            depth_attachment=self.ctx.depth_renderbuffer((_HEIGHT_SIZE, _HEIGHT_SIZE)),
        )

        self.shadow_tex = self.ctx.texture((shadow_size, shadow_size), 1, dtype="f4")
        self.shadow_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = self.ctx.framebuffer(
            color_attachments=[self.shadow_tex],
            depth_attachment=self.ctx.depth_renderbuffer((shadow_size, shadow_size)),
        )

        gr = grain_texture()
        self.grain_tex = self.ctx.texture((gr.shape[1], gr.shape[0]), 4, gr.tobytes(), dtype="f4")
        self.grain_tex.build_mipmaps()
        self.grain_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.grain_tex.anisotropy = 16.0

        self._targets: dict[tuple[int, int], tuple] = {}
        self._page_cache: tuple | None = None
        self._doc_cache: tuple | None = None
        self._display: tuple | None = None
        self._light_vp, self._shadow_tan = np.eye(4), 1.0
        self._height_rect = np.array([0.0, 0.0, 1.0, 1.0])
        self._props: list[dict] = []
        self._effects = None
        self._desk_cache: tuple | None = None
        self._masks: dict[str, moderngl.Texture] = {}
        self._blank = self._texture(np.full((1, 1, 4), 255, np.uint8), mipmaps=False)

    # ---- public -------------------------------------------------------

    def render(self, scene: Scene, doc_rgba: np.ndarray, doc_key: str | None = None,
               read_color: bool = True) -> RenderOutput:
        """Render the page. `doc_key` names the document; with it, the texture
        stays on the GPU for the next call with the same key. With
        `read_color` false, the color and table buffers stay on the GPU
        (`RenderOutput.textures`) for the GPU camera effects."""
        ctx = self.ctx
        cam = scene.camera
        page_vao = self._page_vaos(scene)
        doc_tex = self._doc_texture(doc_rgba, doc_key)
        self._props = self._prop_vaos(scene)
        try:
            self._shadow_pass(scene)
            ms_fbo, resolve_fbo, label_fbo = self._target(cam.width, cam.height)
            ms_fbo.use()
            ms_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            self._color_pass(scene, doc_tex, page_vao, display=False)
            ctx.copy_framebuffer(resolve_fbo, ms_fbo)
            rgba = self._read(resolve_fbo, 4) if read_color else None
            table = self._table_pass(scene, label_fbo.size, read_color) if scene.table is not None else None

            # Label pass: no MSAA, so each pixel holds the value at its center.
            label_fbo.use()
            label_fbo.clear(0.0, 0.0, 0.0, cam.far, depth=1.0)
            self._set(self.label_prog, u_view_proj=cam.view_proj, u_view=cam.view, u_kind=1, **self._corner(scene))
            page_vao["label"].render()
            self._set(self.label_prog, u_kind=0)  # desk items hide the page but are not it
            for prop in self._props:
                prop["label"].render()
            label = self._read(label_fbo, 4)
        finally:
            if doc_key is None:
                doc_tex.release()
            self._release_props()
        textures = {"color": resolve_fbo.color_attachments[0], "label": label_fbo.color_attachments[0],
                    "table": self._table_target(label_fbo.size).color_attachments[0] if scene.table is not None else None}
        if rgba is None:
            return RenderOutput(color=None, alpha=None, label=label, table=None, textures=textures)
        return RenderOutput(color=rgba[..., :3], alpha=np.clip(rgba[..., 3], 0.0, 1.0), label=label, table=table,
                            textures=textures)

    def render_display(self, scene: Scene, doc_rgba: np.ndarray, doc_key: str | None,
                       clear_rgb: tuple[float, float, float]) -> moderngl.Texture:
        """Render for the screen: display colors (gamma 2.2) on a solid color,
        left on the GPU in a texture. Row 0 of the texture is the bottom."""
        cam = scene.camera
        page_vao = self._page_vaos(scene)
        doc_tex = self._doc_texture(doc_rgba, doc_key)
        self._props = self._prop_vaos(scene)
        try:
            self._shadow_pass(scene)
            ms_fbo, out_fbo, out_tex = self._display_target(cam.width, cam.height)
            ms_fbo.use()
            ms_fbo.clear(*clear_rgb, 1.0, depth=1.0)
            if scene.table is not None:
                self._height_pass(scene)
                self._shadow_pass(scene, _DISPLAY_TABLE_HALF_CM * 1.45)
                ms_fbo.use()
                self._draw_table(scene, display=True)
                self._shadow_pass(scene)
                ms_fbo.use()
            self._color_pass(scene, doc_tex, page_vao, display=True)
            self.ctx.copy_framebuffer(out_fbo, ms_fbo)
        finally:
            if doc_key is None:
                doc_tex.release()
            self._release_props()
        return out_tex

    def _shadow_pass(self, scene: Scene, radius: float | None = None) -> None:
        """Shadow map around the page center. The default radius covers the
        page closely (sharp crumple shadows); the table passes a wider one."""
        ctx = self.ctx
        page_vao = self._page_vaos(scene)
        light_vp, self._shadow_tan = self._light_frustum(scene, radius or _cover_radius(scene))
        self._light_vp = light_vp
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.DEPTH_TEST)
        self.shadow_fbo.use()
        self.shadow_fbo.clear(1e9, 0.0, 0.0, 0.0, depth=1.0)
        self._set(self.shadow_prog, u_light_vp=light_vp, u_light_pos=scene.light.position, u_mask=0,
                  **self._corner(scene))
        self._blank.use(0)
        page_vao["shadow"].render()
        self._set(self.shadow_prog, u_corner_radius=0.0)
        occ_vaos, occ_res = self._occluder_vaos(scene)
        for shape, vao in occ_vaos:
            self._mask(shape).use(0)
            vao.render()
        for res in occ_res:
            res.release()
        self._blank.use(0)
        for prop in self._props:
            prop["shadow"].render()

    @staticmethod
    def _corner(scene: Scene) -> dict:
        """Uniforms of the rounded card corners (card look); radius 0 keeps square corners."""
        return dict(u_corner_radius=scene.material.corner_radius_cm,
                    u_page_size=np.array([scene.page.width_cm, scene.page.height_cm]))

    def _lighting(self, scene: Scene) -> dict:
        """Uniforms shared by the page and the table: camera, lamp, room light, shadows."""
        cam, lt = scene.camera, scene.light
        return dict(
            u_view_proj=cam.view_proj, u_view=cam.view, u_camera_pos=cam.position, u_shadow=1,
            u_light_vp=self._light_vp, u_shadow_tan=self._shadow_tan, u_shadow_texel=1.0 / self.shadow_size,
            u_light_pos=lt.position, u_light_color=lt.color, u_light_intensity=lt.intensity,
            u_light_ref=lt.ref_distance, u_light_size=lt.size, u_ambient=lt.ambient,
            u_ambient_color=lt.ambient_color, u_ambient_dir=lt.ambient_direction,
            u_ambient_focus=lt.ambient_focus, u_wrap=lt.wrap,
        )

    def _color_pass(self, scene: Scene, doc_tex: moderngl.Texture, page_vao, display: bool) -> None:
        mat = scene.material
        self._set(
            self.color_prog, **self._lighting(scene),
            u_doc=0, u_grain_tex=2, u_display=display,
            u_page_size=np.array([scene.page.width_cm, scene.page.height_cm]),
            u_grain_tile=mat.grain_tile_cm, u_grain_offset=mat.grain_offset,
            u_grain=mat.grain, u_paper_tint=mat.tint,
            u_back_color=mat.back_color, u_roughness=mat.roughness, u_specular=mat.specular,
            u_corner_radius=mat.corner_radius_cm, u_glare=mat.glare_strength, u_glare_dir=mat.glare_dir,
            u_glare_cos=mat.glare_cos, u_hologram=mat.hologram, u_glare_panel=int(mat.glare_panel),
            u_glare_u=mat.glare_u, u_glare_v=mat.glare_v, u_glare_half=mat.glare_half,
            u_glare_edge=mat.glare_edge, u_waviness=mat.waviness, u_room=mat.room_reflection, u_wear=mat.wear,
        )
        doc_tex.use(0)
        self.shadow_tex.use(1)
        self.grain_tex.use(2)
        page_vao["color"].render()
        if self._props:
            self._set(self.prop_prog, **self._lighting(scene), u_display=display, u_image=5, u_specular=0.04)
            for prop in self._props:
                self._set(self.prop_prog, u_roughness=prop["roughness"], u_has_image=prop["image"] is not None)
                (prop["image"] or self._blank).use(5)
                prop["color"].render()

    def _height_pass(self, scene: Scene) -> None:
        """Page height seen from above, over the page footprint plus a margin."""
        pos = np.concatenate([scene.page.pos] + [p.pos for p in scene.props])
        margin = 2.0 * scene.table.contact_radius_cm
        c = scene.page_center[:2]
        lo = np.maximum(pos[:, :2].min(axis=0) - margin, c - 75.0)
        size = np.maximum(np.minimum(pos[:, :2].max(axis=0) + margin, c + 75.0) - lo, 1e-3)
        top = float(pos[:, 2].max()) + 1.0
        proj = np.eye(4)
        proj[0, 0], proj[0, 3] = 2.0 / size[0], -1.0 - 2.0 * lo[0] / size[0]
        proj[1, 1], proj[1, 3] = 2.0 / size[1], -1.0 - 2.0 * lo[1] / size[1]
        proj[2, 2], proj[2, 3] = 2.0 / top, -1.0  # higher = farther, so the depth test keeps the lowest layer
        self._height_rect = np.array([lo[0], lo[1], size[0], size[1]])
        self.height_fbo.use()
        self.height_fbo.clear(1e3, 0.0, 0.0, 0.0, depth=1.0)
        self._set(self.height_prog, u_proj=proj, **self._corner(scene))
        self._page_vaos(scene)["height"].render()
        self._set(self.height_prog, u_corner_radius=0.0)
        for prop in self._props:
            prop["height"].render()

    def _draw_table(self, scene: Scene, display: bool) -> None:
        t = scene.table
        half = _DISPLAY_TABLE_HALF_CM if display else _TABLE_HALF_CM
        c = scene.page_center
        quad = np.array([[c[0] - half, c[1] - half, 0], [c[0] + half, c[1] - half, 0],
                         [c[0] + half, c[1] + half, 0], [c[0] - half, c[1] + half, 0]], dtype=np.float32)
        vbo = self.ctx.buffer(quad[[0, 1, 2, 0, 2, 3]].tobytes())
        vao = self.ctx.vertex_array(self.table_prog, [(vbo, "3f", "in_pos")])
        self._set(self.table_prog, **self._lighting(scene), u_height=3, u_height_rect=self._height_rect,
                  u_contact=t.contact, u_contact_radius=t.contact_radius_cm, u_table_albedo=t.albedo,
                  u_display=display, u_desk=t.desk, u_desk_tex=t.desk_image is not None,
                  u_desk_map=4, u_desk_size=np.asarray(t.desk_size_cm, float), u_desk_offset=t.desk_offset,
                  u_desk_rot=float(np.radians(t.desk_rotation_deg)), u_desk_color=t.desk_color)
        self.shadow_tex.use(1)
        self.height_tex.use(3)
        (self._desk_texture(t.desk_image) if t.desk_image is not None else self._blank).use(4)
        vao.render()
        vao.release()
        vbo.release()

    def _prop_vaos(self, scene: Scene) -> list[dict]:
        """Vertex buffers of the desk items, for one render call."""
        out = []
        for prop in scene.props:
            data = np.concatenate([prop.pos, prop.normal, prop.color, prop.uv], axis=1).astype(np.float32)
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(prop.indices.astype(np.uint32).tobytes())

            def vao(prog, fmt, *names):
                return self.ctx.vertex_array(prog, [(vbo, fmt, *names)], index_buffer=ibo, index_element_size=4)

            item = {
                "color": vao(self.prop_prog, "3f 3f 3f 2f", "in_pos", "in_normal", "in_color", "in_uv"),
                "shadow": vao(self.shadow_prog, "3f 24x 2f", "in_pos", "in_uv"),
                "label": vao(self.label_prog, "3f 24x 2f", "in_pos", "in_uv"),
                "height": vao(self.height_prog, "3f 32x", "in_pos"),
                "roughness": prop.roughness,
                "image": self._texture(prop.image) if prop.image is not None else None,
            }
            item["_res"] = [item["color"], item["shadow"], item["label"], item["height"], vbo, ibo]
            if item["image"] is not None:
                item["_res"].append(item["image"])
            out.append(item)
        return out

    def _release_props(self) -> None:
        for prop in self._props:
            for res in prop["_res"]:
                res.release()
        self._props = []

    def _desk_texture(self, image: np.ndarray) -> moderngl.Texture:
        """The desk texture, kept on the GPU while the same image is used."""
        if self._desk_cache is None or self._desk_cache[0] is not image:
            if self._desk_cache is not None:
                self._desk_cache[1].release()
            tex = self._texture(image)
            tex.repeat_x = tex.repeat_y = True
            self._desk_cache = (image, tex)
        return self._desk_cache[1]

    def _float_texture(self, size: tuple[int, int], dtype: str) -> moderngl.Texture:
        tex = self.ctx.texture(size, 4, dtype=dtype)
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = tex.repeat_y = False
        return tex

    def _table_pass(self, scene: Scene, size: tuple[int, int], read: bool = True) -> np.ndarray | None:
        """Light arriving at the table, per pixel, and the table's depth."""
        self._height_pass(scene)
        self._shadow_pass(scene, _view_radius(scene))
        fbo = self._table_target(size)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, scene.camera.far)
        self.ctx.disable(moderngl.DEPTH_TEST)  # one flat quad: nothing to sort
        self._draw_table(scene, display=False)
        self.ctx.enable(moderngl.DEPTH_TEST)
        if not read:
            return None
        w, h = fbo.size
        raw = fbo.read(components=4, dtype="f2")
        return np.frombuffer(raw, dtype=np.float16).reshape(h, w, 4)[::-1].astype(np.float32)

    def _table_target(self, size: tuple[int, int]) -> moderngl.Framebuffer:
        key = ("table", size)
        if key not in self._targets:
            self._targets[key] = self.ctx.framebuffer(color_attachments=[self._float_texture(size, "f2")])
        return self._targets[key]

    @property
    def effects(self):
        """The GPU camera effects, on this renderer's context."""
        if self._effects is None:
            from ..fx.gpu import GpuFx

            self._effects = GpuFx(self.ctx)
        return self._effects

    def release(self) -> None:
        self.ctx.release()

    # ---- helpers ------------------------------------------------------

    def _page_vaos(self, scene: Scene):
        """The page's vertex buffers, kept while the same page is drawn again
        (the editor redraws one page from many cameras)."""
        c = self._page_cache
        if c is not None and c[0] is scene.page and c[1] is scene.page_normals and c[2] == scene.material.crease_shading:
            return c[3]
        if c is not None:
            for res in c[4]:
                res.release()
        mesh = scene.page
        data = np.concatenate([mesh.pos, mesh.uv, scene.page_normals, scene.page_ao[:, None]], 1)
        vaos, res = self._vaos(data.astype(np.float32), mesh.indices)
        # Holding the arrays keeps their identity unique while they are cached.
        self._page_cache = (scene.page, scene.page_normals, scene.material.crease_shading, vaos, res)
        return vaos

    def _doc_texture(self, rgba: np.ndarray, key: str | None) -> moderngl.Texture:
        if key is None:
            return self._texture(rgba)
        if self._doc_cache is None or self._doc_cache[0] != key or self._doc_cache[1] is not rgba:
            if self._doc_cache is not None:
                self._doc_cache[2].release()
            self._doc_cache = (key, rgba, self._texture(rgba))
        return self._doc_cache[2]

    def _display_target(self, w: int, h: int):
        if self._display is None or self._display[0] != (w, h):
            if self._display is not None:
                for res in self._display[1:]:
                    res.release()
            ctx = self.ctx
            ms_fbo = ctx.framebuffer(
                color_attachments=[ctx.renderbuffer((w, h), 4, samples=self.samples)],
                depth_attachment=ctx.depth_renderbuffer((w, h), samples=self.samples),
            )
            tex = ctx.texture((w, h), 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            out_fbo = ctx.framebuffer(color_attachments=[tex])
            self._display = ((w, h), ms_fbo, out_fbo, tex)
        return self._display[1:]

    def _occluder_vaos(self, scene: Scene):
        vaos, res = [], []
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        for occ in scene.occluders:
            data = np.concatenate([occ.corners.astype(np.float32), uv], 1)
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32).tobytes())
            vao = self.ctx.vertex_array(self.shadow_prog, [(vbo, "3f 2f", "in_pos", "in_uv")],
                                        index_buffer=ibo, index_element_size=4)
            vaos.append((occ.shape, vao))
            res += [vao, vbo, ibo]
        return vaos, res

    def _vaos(self, data: np.ndarray, indices: np.ndarray):
        ctx = self.ctx
        vbo = ctx.buffer(np.ascontiguousarray(data, dtype=np.float32).tobytes())
        ibo = ctx.buffer(np.ascontiguousarray(indices, dtype=np.uint32).tobytes())

        def vao(prog, fmt, *names):
            return ctx.vertex_array(prog, [(vbo, fmt, *names)], index_buffer=ibo, index_element_size=4)

        vaos = {
            "shadow": vao(self.shadow_prog, "3f 2f 16x", "in_pos", "in_uv"),
            "color": vao(self.color_prog, "3f 2f 3f 1f", "in_pos", "in_uv", "in_normal", "in_ao"),
            "label": vao(self.label_prog, "3f 2f 16x", "in_pos", "in_uv"),
            "height": vao(self.height_prog, "3f 2f 16x", "in_pos", "in_uv"),
        }
        return vaos, [*vaos.values(), vbo, ibo]

    def _texture(self, rgba: np.ndarray, mipmaps: bool = True) -> moderngl.Texture:
        img = np.ascontiguousarray(rgba, dtype=np.uint8)
        h, w = img.shape[:2]
        # sRGB storage: the GPU turns texels into linear values before filtering.
        tex = self.ctx.texture((w, h), 4, img.tobytes(), internal_format=GL_SRGB8_ALPHA8)
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 16.0
        tex.repeat_x = tex.repeat_y = False
        return tex

    def _mask(self, shape: str) -> moderngl.Texture:
        if shape not in self._masks:
            m = occluder_mask(shape)
            tex = self.ctx.texture((m.shape[1], m.shape[0]), 1, np.ascontiguousarray(m).tobytes(), dtype="f4")
            tex.repeat_x = tex.repeat_y = False
            self._masks[shape] = tex
        return self._masks[shape]

    def _light_frustum(self, scene: Scene, radius: float) -> tuple[np.ndarray, float]:
        light = scene.light.position
        center = scene.page_center
        dist = float(np.linalg.norm(light - center))
        tan_half = min(radius / dist, np.tan(np.radians(80.0)))
        near = max(0.5, 0.02 * dist)
        proj = perspective(tan_half, tan_half, near, dist * 4.0)
        return proj @ look_at(light, center), tan_half

    def _target(self, w: int, h: int):
        key = (w, h)
        if key not in self._targets:
            ctx = self.ctx
            ms_fbo = ctx.framebuffer(
                color_attachments=[ctx.renderbuffer((w, h), 4, samples=self.samples, dtype="f4")],
                depth_attachment=ctx.depth_renderbuffer((w, h), samples=self.samples),
            )
            # Textures, not renderbuffers: the GPU camera effects read them in place.
            resolve_fbo = ctx.framebuffer(color_attachments=[self._float_texture((w, h), "f4")])
            label_fbo = ctx.framebuffer(
                color_attachments=[self._float_texture((w, h), "f4")],
                depth_attachment=ctx.depth_renderbuffer((w, h)),
            )
            self._targets[key] = (ms_fbo, resolve_fbo, label_fbo)
        return self._targets[key]

    @staticmethod
    def _read(fbo: moderngl.Framebuffer, components: int) -> np.ndarray:
        w, h = fbo.size
        raw = fbo.read(components=components, dtype="f4")
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w, components)[::-1].copy()

    @staticmethod
    def _set(prog: moderngl.Program, **values) -> None:
        for name, value in values.items():
            if name not in prog:  # the GLSL compiler drops unused uniforms
                continue
            u = prog[name]
            if isinstance(value, np.ndarray) and value.ndim == 2:
                u.write(value.T.astype(np.float32).tobytes())  # GLSL wants column-major
            elif isinstance(value, np.ndarray):
                u.value = tuple(float(v) for v in value)
            elif isinstance(value, bool):
                u.value = int(value)
            else:
                u.value = value
