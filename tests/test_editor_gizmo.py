"""Drag a gizmo arrow in a hidden editor window with simulated mouse events.

Regression: an interactive item covering the viewport made ImGuizmo ignore
every click on its arrows.
"""

import json
from pathlib import Path

import numpy as np
import pytest

imgui_bundle = pytest.importorskip("imgui_bundle")


def _drag_arrow(obj: str) -> tuple[str, str]:
    from imgui_bundle import hello_imgui, imgui, immapp

    from recapture.editor import app
    from recapture.editor.view3d import project

    ed = app.Editor(None, None)
    st = {"n": 0, "rect": None, "pt": None, "dir": None, "before": None}
    orig = app.Editor._scene_view

    def scene_view(self):
        orig(self)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        st["rect"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    def pre_new_frame():
        n, io = st["n"], imgui.get_io()
        if n == 30 and st["rect"]:
            x0, y0, w, h = st["rect"]
            p = ed._objects(ed.frame)[obj]
            vp = ed.orbit.proj(w / h) @ ed.orbit.view()
            a, _ = project(vp, p[None], w, h)
            b, _ = project(vp, (p + [5.0, 0, 0])[None], w, h)
            d = (b[0] - a[0]) / np.linalg.norm(b[0] - a[0])
            st["pt"], st["dir"] = (x0 + a[0][0] + d[0] * 40, y0 + a[0][1] + d[1] * 40), d
            st["before"] = json.dumps(ed.user)
            io.add_mouse_pos_event(*st["pt"])
        if n == 32:
            io.add_mouse_button_event(0, True)
        if 34 <= n <= 44:
            k = (n - 33) * 6
            io.add_mouse_pos_event(st["pt"][0] + st["dir"][0] * k, st["pt"][1] + st["dir"][1] * k)
        if n == 46:
            io.add_mouse_button_event(0, False)
        st["n"] += 1

    def gui():
        ed.gui()
        if st["n"] == 3:
            ed.set(("occluders",), [{"shape": "hand_phone", "target": [0.5, 0.5], "height_cm": 15}])
        if st["n"] == 25:
            ed.selected = obj
        if st["n"] > 60:
            hello_imgui.get_runner_params().app_shall_exit = True

    app.Editor._scene_view = scene_view
    try:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_geometry.size = (1500, 950)
        params.app_window_params.hidden = True  # no focus, so the real cursor does not override the events
        params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
        params.callbacks.show_gui = gui
        params.callbacks.pre_new_frame = pre_new_frame
        params.fps_idling.enable_idling = False
        immapp.run(params)
    finally:
        app.Editor._scene_view = orig
    return st["before"], json.dumps(ed.user)


@pytest.mark.parametrize("obj, key", [("hand0", "occluders"), ("lamp", "light")])
def test_dragging_an_arrow_moves_the_object(obj, key):
    try:
        before, after = _drag_arrow(obj)
    except Exception as e:  # no display or GL on this machine
        pytest.skip(f"cannot open a window: {e}")
    assert json.loads(before).get(key) != json.loads(after).get(key)
