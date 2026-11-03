# Recapture config

A config is one JSON object. Keys left out take the built-in default; `recapture validate` names any unknown key and the keys allowed there. Any number can be a random range drawn per image: `{"uniform": [lo, hi]}`, `{"randint": [lo, hi]}`, `{"normal": [mean, std]}`, or `{"choice": [a, b, ...], "weights": [...]}`. `recapture defaults` prints every key with its default and a comment on what it does (Python literals: `None` is `null`).

Paths: `"@assets/<path>"` is the shared asset folder (fill it with `recapture fetch-assets`); any other relative path is relative to the config file.

## Mode

`"mode"`: `"photo"`, `"scan"`, or both as `{"choice": ["photo", "scan"], "weights": [0.7, 0.3]}` (the weight of `"scan"` is the share of scans). A photo uses every section below except `printer` and `scanner`; a scan uses only `page.size_cm`, `printer`, `scanner` and `output`.

## Photo sections

| Section | Keys that matter |
|---|---|
| `deform.crumple` | `strength` (0 flat, 1 normal, 2 heavy), `mode` (`"scan"` = real crumpled-paper height maps, the realistic choice), `scan.directory` (`"@assets/ambientcg"`), `scan.tile_cm` (smaller = smaller, sharper creases; 10 to 30) |
| `deform` | `folds` (`{"repeat": n, "item": {...}}`), `curl.x_curvature` / `curl.y_curvature` (0 flat, about ±0.015), `edge_lift_cm` (4 numbers: top, right, bottom, left) |
| `material` | `roughness` (0.6 paper, about 0.15 a glossy laminate), `corner_radius_cm` (0 square; 0.32 an ID card), `glare` (`strength` 0 off, 3 to 15 for a panel, 10 to 40 washes out a patch; `elevation_deg`, `azimuth_deg`, `size_deg`, `softness`; `shape` `disk` or `panel`, `aspect`, `rotation_deg`, `waviness` 0..1), `room_reflection` (0 off, about 0.4 glossy), `wear` (0 new .. 1 scratched and smudged), `hologram` (0 off, 0.1 to 0.4) |
| `light` | `elevation_deg` (low = long shadows, crumples stand out), `azimuth_deg` (0 to 360), `distance_cm`, `intensity`, `color_temp_k` (3000 warm to 6500 cool), `size_cm` (bigger = softer shadows), `ambient` |
| `occluders` | hand or phone shadows: `{"repeat": n, "item": {"shape": "hand_phone" \| "arm" \| "phone" \| "blob", ...}}`; `[]` for none |
| `desk` | `enabled`, `surface` (`["@assets/polyhaven/desks", "@assets/desks_plain"]`), `items` (`{"repeat": n, "item": {"kind": ...}}`; kinds: keyboard, mousepad, phone, mug, pen, pencil, sticky_notes, paper) |
| `background` | ignored while `desk.enabled` is true; used when the desk is off: `enabled` (false = transparent PNG around the page; needs `output.format: "png"`), `images` (`"@assets/polyhaven/textures"`), `brightness` |
| `camera` | `position_cm` [x, y, height], `target_cm`, `roll_deg`, `focal_mm` (26 = phone main camera, 13 = ultra wide), `output_px` [w, h] |
| `crop` | `enabled`: a scanner-app crop, so the whole image is the page; `margin`, `corner_error` |
| `fx` | `quality` (1 good camera, 0 cheap phone), `defocus.strength_px`, `noise.shot`, `motion_blur.enabled`, `vignette`, `exposure_ev`, `white_balance` |

## Scan sections

| Section | Keys (each effect off at 0) |
|---|---|
| `printer` | `ink_bleed` (0 to 1), `broken_strokes` (part of ink left out, 0.01 to 0.04), `ghost` (double strike, 0 to 0.3), `toner_streaks` (count), `toner_specks` (count, up to about 800) |
| `scanner` | `dpi` (150, 200, 300; sets the output size), `color` (`"color"`, `"gray"`, `"bw"`), `bw_threshold`, `skew_deg`, `blur_px`, `bold` (0 to 1, thicker strokes), `noise`, `streaks`, `dust`, `dark_edges` [left, right, top, bottom] (parts of the page width, up to about 0.05), `edge_dots` |

## Common requests

| The user says | Change |
|---|---|
| "more scans" / "30 % scans" | `mode` → `{"choice": ["photo", "scan"], "weights": [0.7, 0.3]}` |
| "only photos" / "only scans" | `mode` → `"photo"` / `"scan"` |
| "flat pages" / "no crumple" | `deform.crumple.strength` → 0, `deform.folds` → `[]`, curl to 0, `edge_lift_cm` → `[0, 0, 0, 0]` |
| "an ID card" / "a plastic card" / "laminated" | start from `recapture init <name>.json --from card`; or on any config: `deform.crumple.strength` → 0, `deform.folds` → `[]`, `material.corner_radius_cm` → 0.32, `material.roughness` → `{"uniform": [0.1, 0.25]}`, `material.glare.strength` → `{"uniform": [3, 15]}`, `material.glare.shape` → `"panel"`, `material.room_reflection` → `{"uniform": [0.2, 0.6]}`, `material.wear` → `{"uniform": [0.3, 1.0]}`, `material.hologram` → `{"uniform": [0.1, 0.35]}`, `page.size_cm` → the card's size (ID-1: `[8.56, 5.4]`), `camera.fit_page` → true |
| "harder crumple" | `deform.crumple.strength` → `{"uniform": [1.2, 2.0]}`, `scan.tile_cm` → `{"uniform": [10, 16]}` |
| "no hand shadows" / "no shadows on the page" | `occluders` → `[]`; desk items also shade the page under a low lamp, so for no shadows at all also set `desk.items` → `[]` or `light.elevation_deg` → `{"uniform": [40, 70]}` |
| "no desk items" | `desk.items` → `[]` |
| "no keyboards" | remove `"keyboard"` from `desk.items.item.kind.choice` and its weight |
| "plain background" / "transparent" | `desk.enabled` → false, then `background.images` → null (flat color) or `background.enabled` → false with `output.format` → `"png"` |
| "worse phone" / "blurrier" | `fx.quality` → `{"uniform": [0.2, 0.7]}`, raise `fx.defocus.strength_px` |
| "page fills the image" | `crop.enabled` → true |
| "black-and-white scans" | `scanner.color` → `"bw"` (replacing the example's choice of color, gray and bw), `scanner.bw_threshold` → `{"randint": [150, 175]}` (higher = darker, bolder text) |
| "dirty or old scans" | raise `printer.toner_specks`, `scanner.dust`, `scanner.dark_edges`, `scanner.edge_dots`; `scanner.bold` 0.3 to 0.8 |
| "clean scans" | every `printer` value → 0; `scanner.dust`, `streaks`, `edge_dots` → 0; `dark_edges` → `[0, 0, 0, 0]` |
| "bigger or smaller images" | photos: `camera.output_px`; scans: `scanner.dpi` |
| "PNG output" | `output.format` → `"png"` |

After each change, run `recapture validate`, then `recapture preview` to show the effect.
