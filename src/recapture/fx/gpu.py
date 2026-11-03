"""The camera effects of `apply_fx`, on the GPU.

Same steps, same order, same math as the numpy version: composite over the
background, depth blur, motion blur, vignette, white balance and auto
exposure, sensor noise, filmic tone curve, sRGB, unsharp mask. The page
render stays on the GPU; only a 1-in-4 pixel sample (for the white balance
and exposure statistics, computed exactly as on the CPU) and the final
8-bit image come back.

The result is not equal to the numpy version bit for bit: the GPU blurs
are exact Gaussians where numpy shrinks the image for wide blurs, the
noise uses other random numbers with the same statistics, and rounding
differs a little. tests/test_fx_gpu.py checks that both agree within a
small tolerance.
"""

from __future__ import annotations

from typing import Any

import moderngl
import numpy as np

from . import _LUMA, filmic_inverse

_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Texture row 0 is the bottom of the photo; numpy row 0 is the top. Passes
# work in texel coordinates and flip y only where the direction matters.

_COMPOSITE = """
#version 330
uniform sampler2D u_color;   // premultiplied page color, alpha = coverage
uniform sampler2D u_bg;      // background layer (layer and lit modes)
uniform sampler2D u_table;   // table pass: light (lit) or reflected light (desk)
uniform int u_mode;          // 0 flat layer, 1 lit table, 2 desk
uniform float u_brightness;
uniform float u_paper_white;
out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 c = texelFetch(u_color, p, 0);
    vec3 bg;
    if (u_mode == 2) {
        bg = texelFetch(u_table, p, 0).rgb;
    } else {
        vec3 layer = texelFetch(u_bg, p, 0).rgb * u_brightness;
        bg = u_mode == 0 ? layer * u_paper_white : layer * texelFetch(u_table, p, 0).rgb;
    }
    f = vec4(c.rgb + bg * (1.0 - clamp(c.a, 0.0, 1.0)), 1.0);
}
"""

# Separable Gaussian with the same kernel as fx.blur._separable (radius
# ceil(2.5 sigma), edges repeated).
_BLUR = """
#version 330
uniform sampler2D u_src;
uniform ivec2 u_dir;
uniform float u_sigma;
out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_src, 0);
    int r = int(ceil(2.5 * u_sigma));
    vec4 acc = vec4(0.0);
    float total = 0.0;
    for (int i = -r; i <= r; ++i) {
        float w = exp(-0.5 * (float(i) / u_sigma) * (float(i) / u_sigma));
        ivec2 q = clamp(p + u_dir * i, ivec2(0), size - 1);
        acc += w * texelFetch(u_src, q, 0);
        total += w;
    }
    f = acc / total;
}
"""

# Depth blur: add one blur level, weighted by how close each pixel's blur
# size is to that level (fx.defocus).
_DEFOCUS_ADD = """
#version 330
uniform sampler2D u_acc;
uniform sampler2D u_level;
uniform sampler2D u_label;   // .z page mask, .w depth
uniform sampler2D u_table;   // .w table depth
uniform int u_far_mode;      // 0 focus depth, 1 table depth, 2 min(table, label) (desk)
uniform bool u_all_page;     // too little page: every pixel counts as page
uniform float u_focus;
uniform float u_strength;
uniform float u_levels[12];
uniform int u_count;
uniform int u_index;
out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 lab = texelFetch(u_label, p, 0);
    float depth;
    if (u_all_page || lab.z > 0.5) {
        depth = lab.w;
    } else if (u_far_mode == 0) {
        depth = u_focus;
    } else {
        float far = texelFetch(u_table, p, 0).w;
        depth = u_far_mode == 2 ? min(far, lab.w) : far;
    }
    float coc = u_strength * abs(1.0 - u_focus / max(depth, 1e-3)) / 0.5;
    float sigma = clamp(0.5 * coc, 0.0, 4.0 * u_strength);
    // Piecewise-linear position of sigma in the level list (np.interp).
    float pos = float(u_count - 1);
    if (sigma <= u_levels[0]) {
        pos = 0.0;
    } else {
        for (int i = 0; i < u_count - 1; ++i) {
            if (sigma <= u_levels[i + 1]) {
                pos = float(i) + (sigma - u_levels[i]) / (u_levels[i + 1] - u_levels[i]);
                break;
            }
        }
    }
    float w = clamp(1.0 - abs(pos - float(u_index)), 0.0, 1.0);
    f = texelFetch(u_acc, p, 0) + w * texelFetch(u_level, p, 0);
}
"""

_MOTION = """
#version 330
uniform sampler2D u_src;
uniform ivec2 u_shifts[32];  // (dx, dy) in numpy axes (y down)
uniform int u_count;
out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_src, 0);
    vec4 acc = vec4(0.0);
    for (int i = 0; i < u_count; ++i) {
        ivec2 s = u_shifts[i];
        ivec2 q = clamp(p - ivec2(s.x, -s.y), ivec2(0), size - 1);
        acc += texelFetch(u_src, q, 0);
    }
    f = acc / float(u_count);
}
"""

_VIGNETTE = """
#version 330
uniform sampler2D u_src;
uniform float u_strength;
out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 size = vec2(textureSize(u_src, 0));
    float half_diag = 0.5 * length(size);
    vec2 d = (vec2(p) - (size - 1.0) * 0.5) / half_diag;
    float v = 1.0 - u_strength * pow(clamp(dot(d, d), 0.0, 1.0), 1.5);
    f = texelFetch(u_src, p, 0) * vec4(vec3(v), 1.0);
}
"""

# Every 4th pixel from the top-left corner, like img[::4, ::4] in numpy.
_SAMPLE = """
#version 330
uniform sampler2D u_src;
out vec4 f;
void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);  // row 0 of the sample = the top row
    ivec2 size = textureSize(u_src, 0);
    f = texelFetch(u_src, ivec2(4 * q.x, size.y - 1 - 4 * q.y), 0);
}
"""

# Unit noise per channel: luma per pixel plus color per 2x2 block (fx.add_noise).
_NOISE = """
#version 330
uniform uint u_seed;
uniform float u_chroma;
out vec4 f;
uint hash(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v.x ^ v.y ^ v.z;
}
float gauss(uvec3 v) {
    float u1 = (float(hash(v) >> 8) + 0.5) / 16777216.0;
    float u2 = (float(hash(v + uvec3(0u, 0u, 0x9E3779B9u)) >> 8) + 0.5) / 16777216.0;
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
void main() {
    uvec2 p = uvec2(gl_FragCoord.xy);
    float l = gauss(uvec3(p, u_seed));
    vec3 n = vec3(l);
    if (u_chroma > 0.0) {
        uvec2 b = p / 2u;
        vec3 c = vec3(gauss(uvec3(b, u_seed + 1u)), gauss(uvec3(b, u_seed + 2u)), gauss(uvec3(b, u_seed + 3u)));
        n = (n + c * 2.0 * u_chroma) / sqrt(1.0 + 4.0 * u_chroma * u_chroma);
    }
    f = vec4(n, 1.0);
}
"""

# Gain, exposure, noise, tone curve, sRGB, 8 bits (the input of the sharpen).
_FINISH = """
#version 330
uniform sampler2D u_src;
uniform sampler2D u_noise;
uniform bool u_effects;
uniform vec3 u_gain;
uniform float u_exposure;
uniform bool u_noise_on;
uniform float u_noise_scale;  // 1 / std of the (blurred) noise
uniform float u_shot;
uniform float u_read;
uniform bool u_filmic;
out vec4 f;
vec3 srgb(vec3 x) {
    x = clamp(x, 0.0, 1.0);
    return mix(12.92 * x, 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), x));
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 c = texelFetch(u_src, p, 0).rgb;
    if (u_effects) {
        c *= u_gain * u_exposure;
        if (u_noise_on) {
            vec3 n = texelFetch(u_noise, p, 0).rgb * u_noise_scale;
            vec3 sigma = sqrt(max(u_shot * c, 0.0) + u_read * u_read);
            c = max(c + n * sigma, 0.0);
        }
        if (u_filmic) c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    }
    f = vec4(floor(srgb(c) * 255.0 + 0.5) / 255.0, 1.0);
}
"""

# Pillow's UnsharpMask on 8-bit values: out = in + (in - blurred) * percent / 100.
_SHARPEN = """
#version 330
uniform sampler2D u_src;
uniform sampler2D u_blur;
uniform float u_percent;
out vec4 f;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 s = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    vec3 b = floor(texelFetch(u_blur, p, 0).rgb * 255.0 + 0.5);
    vec3 o = clamp(s + trunc((s - b) * u_percent / 100.0), 0.0, 255.0);
    f = vec4(o / 255.0, 1.0);
}
"""


class GpuFx:
    """Runs the camera effects on the renderer's GPU context."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        prog = lambda fs: ctx.program(vertex_shader=_VS, fragment_shader=fs)  # noqa: E731
        self.p_composite = prog(_COMPOSITE)
        self.p_blur = prog(_BLUR)
        self.p_defocus = prog(_DEFOCUS_ADD)
        self.p_motion = prog(_MOTION)
        self.p_vignette = prog(_VIGNETTE)
        self.p_sample = prog(_SAMPLE)
        self.p_noise = prog(_NOISE)
        self.p_finish = prog(_FINISH)
        self.p_sharpen = prog(_SHARPEN)
        self.vao = ctx.vertex_array(self.p_composite, [])
        self._vaos = {id(p): ctx.vertex_array(p, []) for p in (self.p_composite, self.p_blur, self.p_defocus,
                                                                  self.p_motion, self.p_vignette, self.p_sample,
                                                                  self.p_noise, self.p_finish, self.p_sharpen)}
        self._size: tuple[int, int] | None = None
        self._bg_tex: moderngl.Texture | None = None
        self._dummy = ctx.texture((1, 1), 4, dtype="f4")

    # ---- targets ---------------------------------------------------------

    def _targets(self, w: int, h: int) -> None:
        if self._size == (w, h):
            return
        for t in getattr(self, "_all", []):
            t.release()
        ctx = self.ctx

        def tex(dtype="f4"):
            t = ctx.texture((w, h), 4, dtype=dtype)
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.repeat_x = t.repeat_y = False
            return t

        self.t = {k: tex() for k in ("a", "b", "c", "acc0", "acc1", "noise", "noise_tmp")}
        self.t8 = {k: tex("f1") for k in ("srgb", "blur", "blur_tmp", "out")}
        self.fbo = {k: ctx.framebuffer(color_attachments=[v]) for k, v in {**self.t, **self.t8}.items()}
        sw, sh = (w + 3) // 4, (h + 3) // 4
        self.t_sample = ctx.texture((sw, sh), 4, dtype="f4")
        self.fbo_sample = ctx.framebuffer(color_attachments=[self.t_sample])
        self._all = [*self.t.values(), *self.t8.values(), *self.fbo.values(), self.t_sample, self.fbo_sample]
        self._size = (w, h)

    def _run(self, prog, target: str | moderngl.Framebuffer, textures: dict[str, moderngl.Texture], **uniforms) -> None:
        fbo = self.fbo[target] if isinstance(target, str) else target
        fbo.use()
        for unit, (name, tex) in enumerate(textures.items()):
            tex.use(unit)
            if name in prog:
                prog[name].value = unit
        for name, value in uniforms.items():
            if name not in prog:
                continue
            u = prog[name]
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
                u.value = [tuple(v) for v in value]
            else:
                u.value = value
        self._vaos[id(prog)].render(moderngl.TRIANGLES, vertices=3)

    def _blur(self, src: moderngl.Texture, sigma: float, tmp: str, out: str) -> None:
        self._run(self.p_blur, tmp, {"u_src": src}, u_dir=(1, 0), u_sigma=float(sigma))
        self._run(self.p_blur, out, {"u_src": self.t[tmp] if tmp in self.t else self.t8[tmp]},
                  u_dir=(0, 1), u_sigma=float(sigma))

    def sample(self, src: moderngl.Texture) -> np.ndarray:
        """img[::4, ::4] of a full-size texture, as numpy (top row first)."""
        self._run(self.p_sample, self.fbo_sample, {"u_src": src})
        w, h = self.t_sample.size
        return np.frombuffer(self.fbo_sample.read(components=4, dtype="f4"), np.float32).reshape(h, w, 4)

    # ---- the effects -------------------------------------------------------

    def apply(self, color: moderngl.Texture, label: moderngl.Texture, label_np: np.ndarray,
              table: moderngl.Texture | None, fx: dict[str, Any], focus_depth: float | None,
              bg_cfg: dict[str, Any], bg_layer: np.ndarray | None, mode: str) -> np.ndarray:
        """`mode`: "layer", "lit" or "desk" (see apply_fx). Returns
        the final sRGB uint8 image (H, W, 3), top row first."""
        ctx = self.ctx
        w, h = color.size
        self._targets(w, h)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)

        # Statistics, as in apply_fx: page mask from the label buffer, paper white from the page render.
        page = label_np[..., 2] > 0.5
        all_page = bool(page.sum() < 100)
        if all_page:
            page = np.ones(page.shape, dtype=bool)
        page4 = page[::4, ::4]
        col4 = self.sample(color)
        solid = (col4[..., 3] > 0.99) & page4
        lum = (col4[..., :3] @ _LUMA)[solid]
        paper_white = float(np.percentile(lum, 97)) if lum.size else 1.0

        # Composite.
        modes = {"layer": 0, "lit": 1, "desk": 2}
        tex = {"u_color": color, "u_bg": self._upload_bg(bg_layer, w, h) if bg_layer is not None else self._dummy,
               "u_table": table if table is not None else self._dummy}
        self._run(self.p_composite, "a", tex, u_mode=modes[mode], u_brightness=float(bg_cfg["brightness"]),
                  u_paper_white=paper_white)
        cur = "a"

        if fx["enabled"]:
            d = fx["defocus"]
            if d["enabled"] and focus_depth and d["strength_px"] > 0:
                cur = self._defocus(cur, label, table, mode, all_page, float(focus_depth), float(d["strength_px"]))
            m = fx["motion_blur"]
            if m["enabled"] and m["length_px"] > 0.5:
                nxt = "b" if cur != "b" else "c"
                self._run(self.p_motion, nxt, {"u_src": self.t[cur]}, **_motion_uniforms(float(m["length_px"]),
                                                                                           float(m["angle_deg"])))
                cur = nxt
            if fx["vignette"] > 0:
                nxt = "b" if cur != "b" else "c"
                self._run(self.p_vignette, nxt, {"u_src": self.t[cur]}, u_strength=round(float(fx["vignette"]), 3))
                cur = nxt

            # White balance and auto exposure, from the 1-in-4 sample (exactly the numpy math).
            img4 = self.sample(self.t[cur])[..., :3]
            gain = _wb_gain(img4, page4, float(fx["white_balance"])) * np.asarray(fx["tint"], np.float32)
            exposure = 2.0 ** float(fx["exposure_ev"])
            ae = fx["auto_exposure"]
            if ae["enabled"]:
                lum = ((img4 * gain) @ _LUMA)[page4]
                level = float(np.percentile(lum, float(ae["percentile"]))) if lum.size else 1.0
                target = filmic_inverse(float(ae["target"])) if fx["tonemap"] == "filmic" else float(ae["target"])
                exposure *= target / max(level, 1e-6)

            n = fx["noise"]
            noise_on = bool(n["enabled"])
            scale = 1.0
            if noise_on:
                chroma, grain = float(n["chroma"]), float(n["grain_px"])
                seed = int(fx["seed"]) & 0x7FFFFFFF
                self._run(self.p_noise, "noise", {}, u_seed=seed, u_chroma=chroma)
                if grain > 0.3:
                    self._blur(self.t["noise"], grain, "noise_tmp", "c" if cur != "c" else "b")
                    noise_tex = self.t["c" if cur != "c" else "b"]
                    scale = 1.0 / _blurred_noise_std(grain, chroma)
                else:
                    noise_tex = self.t["noise"]
            else:
                noise_tex = self._dummy
            self._run(self.p_finish, "srgb", {"u_src": self.t[cur], "u_noise": noise_tex}, u_effects=True,
                      u_gain=tuple(float(v) for v in gain), u_exposure=float(exposure), u_noise_on=noise_on,
                      u_noise_scale=float(scale), u_shot=float(n["shot"]), u_read=float(n["read"]),
                      u_filmic=fx["tonemap"] == "filmic")
            s = fx["sharpen"]
            if s["enabled"] and s["amount"] > 0:
                self._blur(self.t8["srgb"], float(s["radius_px"]), "blur_tmp", "blur")
                self._run(self.p_sharpen, "out", {"u_src": self.t8["srgb"], "u_blur": self.t8["blur"]},
                          u_percent=float(int(100 * float(s["amount"]))))
                final = "out"
            else:
                final = "srgb"
        else:
            self._run(self.p_finish, "srgb", {"u_src": self.t[cur], "u_noise": self._dummy}, u_effects=False)
            final = "srgb"

        raw = self.fbo[final].read(components=3, dtype="f1")
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3)[::-1].copy()

    def _defocus(self, cur: str, label, table, mode: str, all_page: bool, focus: float, strength: float) -> str:
        top = 4.0 * strength  # the largest blur any pixel can get
        levels = [0.0]
        while levels[-1] < top:
            levels.append(max(0.5, levels[-1] * 2.0))
        far_mode = 0 if table is None or mode == "layer" else (2 if mode == "desk" else 1)
        common = dict(u_far_mode=far_mode, u_all_page=all_page, u_focus=focus, u_strength=strength,
                      u_levels=levels + [0.0] * (12 - len(levels)), u_count=len(levels))
        src = self.t[cur]
        acc_in, acc_out = "acc0", "acc1"
        self.fbo[acc_in].clear(0.0, 0.0, 0.0, 0.0)
        level_name = "b" if cur != "b" else "c"
        tmp_name = "c" if level_name == "b" else "b"
        if tmp_name == cur:
            tmp_name = "noise_tmp"
        for i, sigma in enumerate(levels):
            if i == 0:
                level = src
            else:
                self._blur(src, sigma, tmp_name, level_name)
                level = self.t[level_name]
            self._run(self.p_defocus, acc_out, {"u_acc": self.t[acc_in], "u_level": level, "u_label": label,
                                                "u_table": table if table is not None else self._dummy},
                      u_index=i, **common)
            acc_in, acc_out = acc_out, acc_in
        return acc_in

    def _upload_bg(self, layer: np.ndarray, w: int, h: int) -> moderngl.Texture:
        data = np.ascontiguousarray(layer[::-1], dtype=np.float32)
        if self._bg_tex is None or self._bg_tex.size != (w, h):
            if self._bg_tex is not None:
                self._bg_tex.release()
            self._bg_tex = self.ctx.texture((w, h), 3, dtype="f4")
            self._bg_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._bg_tex.write(data.tobytes())
        return self._bg_tex


def _wb_gain(img4: np.ndarray, page4: np.ndarray, strength: float) -> np.ndarray:
    """fx.white_balance_gain on an already 1-in-4 sampled image."""
    if strength <= 0:
        return np.ones(3, np.float32)
    mean = img4[page4].mean(axis=0) if page4.any() else img4.reshape(-1, 3).mean(axis=0)
    mean = np.maximum(mean, 1e-6)
    return ((mean @ _LUMA) / mean) ** strength


def _motion_uniforms(length: float, angle_deg: float) -> dict:
    n = max(2, int(np.ceil(length)) + 1)
    a = np.radians(angle_deg)
    shifts = [(int(round(t * np.cos(a))), int(round(t * np.sin(a)))) for t in np.linspace(-0.5 * length, 0.5 * length, n)]
    shifts = shifts[:32]
    return {"u_shifts": shifts + [(0, 0)] * (32 - len(shifts)), "u_count": len(shifts)}


def _blurred_noise_std(sigma: float, chroma: float) -> float:
    """Expected std of the noise field after the grain blur (numpy measures it).

    Luma noise is independent per pixel; color noise is shared by 2x2
    blocks. Both have unit variance before the blur.
    """
    r = int(np.ceil(2.5 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    luma = float((k * k).sum()) ** 2
    # Block noise: sum the kernel over each 2-pixel block, for both block alignments.
    block = []
    for phase in (0, 1):
        idx = (np.arange(-r, r + 1) + phase) // 2
        sums = np.bincount(idx - idx.min(), weights=k)
        block.append(float((sums * sums).sum()))
    color = float(np.mean(block)) ** 2
    c2 = 4.0 * chroma * chroma
    return float(np.sqrt((luma + c2 * color) / (1.0 + c2)))
