"""The full settings tree, for everything the quick panel does not cover."""

from __future__ import annotations

import json
from typing import Any

from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from ..config.values import spec_kind
from . import state

SECTIONS = ["page", "deform", "material", "light", "occluders", "background", "camera", "fx", "output"]


class AdvancedPanel:
    """Edits `editor.user`; calls `editor.changed()` after each edit."""

    def __init__(self, editor):
        self.editor = editor
        self.json_buffers: dict[tuple, str] = {}

    @property
    def user(self) -> dict[str, Any]:
        return self.editor.user

    @property
    def base_dir(self):
        return self.editor.base_dir

    def _changed(self) -> None:
        self.editor.changed()

    def draw(self) -> None:
        imgui.text_wrapped("[R] switches a value between fixed and a random range; [R*] marks a random one. "
                           "Other values take JSON: press Enter to apply.")
        merged = state.merged(self.user)
        changed, seed = imgui.input_int("seed", int(merged.get("seed", 0)))
        if changed:
            state.set_value(self.user, ("seed",), seed)
            self._changed()
        for section in SECTIONS:
            if imgui.collapsing_header(section):
                imgui.push_id(section)
                if section == "occluders":
                    self._list_editor(("occluders",), merged["occluders"], "occluders")
                else:
                    self._dict_editor((section,), merged[section])
                imgui.pop_id()

    def _dict_editor(self, path: tuple, node: dict) -> None:
        for key, value in node.items():
            p = path + (key,)
            if key == "folds":
                if imgui.tree_node("folds"):
                    self._list_editor(p, value, "folds")
                    imgui.tree_pop()
            elif isinstance(value, dict) and not spec_kind(value):
                if imgui.tree_node(key):
                    self._dict_editor(p, value)
                    imgui.tree_pop()
            else:
                self._leaf(key, p, value)

    def _list_editor(self, path: tuple, value: Any, kind: str) -> None:
        defaults = state.ITEM_DEFAULTS[kind]
        if spec_kind(value) == "repeat":
            imgui.text("Random count of items:")
            self._leaf("count", path + ("repeat",), value["repeat"])
            if imgui.small_button("Use a fixed list"):
                state.set_value(self.user, path, state.to_fixed(value))
                self._changed()
            if imgui.tree_node("item (each gets its own random values)"):
                self._dict_editor(path + ("item",), state._merge(dict(defaults), value["item"]))
                imgui.tree_pop()
            return
        items = value if isinstance(value, list) else []
        for i, item in enumerate(items):
            imgui.push_id(i)
            open_ = imgui.tree_node(f"{kind[:-1]} {i + 1}")
            imgui.same_line()
            if imgui.small_button("remove"):
                state.set_value(self.user, path, items[:i] + items[i + 1:])
                self._changed()
                if open_:
                    imgui.tree_pop()
                imgui.pop_id()
                return
            if open_:
                self._dict_editor(path + (i,), state._merge(dict(defaults), item))
                imgui.tree_pop()
            imgui.pop_id()
        if imgui.small_button(f"add {kind[:-1]}"):
            state.set_value(self.user, path, items + [{}])
            self._changed()
        imgui.same_line()
        if imgui.small_button("random count"):
            state.set_value(self.user, path, {"repeat": {"randint": [0, max(1, len(items))]},
                                              "item": items[0] if items else {}})
            self._changed()

    def _leaf(self, key: str, path: tuple, value: Any) -> None:
        imgui.push_id(key)
        kind = spec_kind(value)
        fixed_like = isinstance(value, (bool, int, float, str)) and value is not None
        if kind in ("uniform", "randint", "normal", "choice"):
            if imgui.small_button("R*"):
                self._set(path, state.to_fixed(value))
            imgui.set_item_tooltip("Random. Click to make it fixed.")
            imgui.same_line()
        elif fixed_like:
            if imgui.small_button("R "):
                self._set(path, state.to_random(value))
            imgui.set_item_tooltip("Fixed. Click to make it a random range.")
            imgui.same_line()
        else:
            imgui.dummy(imgui.ImVec2(imgui.calc_text_size("R*").x + 8, 0))
            imgui.same_line()

        imgui.set_next_item_width(200)
        if kind == "uniform":
            ch, v = imgui.drag_float2(key, list(map(float, value["uniform"])), _speed(value["uniform"]), format="%.3f")
            if ch:
                self._set(path, {"uniform": sorted([round(v[0], 4), round(v[1], 4)])})
        elif kind == "randint":
            ch, v = imgui.drag_int2(key, list(map(int, value["randint"])))
            if ch:
                self._set(path, {"randint": sorted(v)})
        elif kind == "normal":
            ch, v = imgui.drag_float2(f"{key} (mean, std)", list(map(float, value["normal"])), _speed(value["normal"]))
            if ch:
                self._set(path, {**value, "normal": [v[0], max(0.0, v[1])]})
        elif isinstance(value, bool):
            ch, v = imgui.checkbox(key, value)
            if ch:
                self._set(path, v)
        elif isinstance(value, int):
            ch, v = imgui.drag_int(key, value)
            if ch:
                self._set(path, v)
        elif isinstance(value, float):
            ch, v = imgui.drag_float(key, value, _speed([value]), format="%.3f")
            if ch:
                self._set(path, round(v, 4))
        elif isinstance(value, str) and state.enum_items(path, key):
            items = state.enum_items(path, key)
            idx = items.index(value) if value in items else 0
            ch, i = imgui.combo(key, idx, items)
            if ch:
                self._set(path, items[i])
        elif isinstance(value, list) and value and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                       or spec_kind(v) for v in value) and len(value) <= 4:
            imgui.text(key)
            imgui.indent()
            for i, v in enumerate(value):
                self._leaf("xyzw"[i] if len(value) <= 3 else str(i), path + (i,), v)
            imgui.unindent()
        else:
            self._json_field(key, path, value)
        imgui.pop_id()

    def _json_field(self, key: str, path: tuple, value: Any) -> None:
        """Fallback editor: the value as JSON text. Enter applies it."""
        buf = self.json_buffers.get(path)
        if buf is None:
            buf = json.dumps(value)
        ch, buf = imgui.input_text(key, buf, imgui.InputTextFlags_.enter_returns_true)
        self.json_buffers[path] = buf
        if ch:
            try:
                self._set(path, json.loads(buf))
                self.json_buffers.pop(path, None)
            except json.JSONDecodeError as e:
                self.editor.error = f"{'.'.join(map(str, path))}: not valid JSON ({e.msg})"
        if path[-1] in ("images", "directory"):
            imgui.same_line()
            if imgui.small_button("..."):
                if path[-1] == "images":
                    res = pfd.select_folder("Background images folder", str(self.base_dir)).result()
                else:
                    res = pfd.select_folder("Scan folder", str(self.base_dir)).result()
                if res:
                    self._set(path, res)
                    self.json_buffers.pop(path, None)

    def _set(self, path: tuple, value: Any) -> None:
        state.set_value(self.user, path, value)
        self._changed()



def _speed(values: list) -> float:
    m = max((abs(float(v)) for v in values), default=1.0)
    return max(m * 0.005, 0.001)


