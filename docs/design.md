# Recapture: design

Status: phases 1 to 4 done, 2026-10-12

## Goal

Recapture turns a clean document image into a realistic phone photo of that document. It also moves the layout boxes of the document to their new place in the photo. The output is ML training data.

Reference look: a crumpled printed page on a wooden bench, one strong side light, a hand shadow across the page, phone camera artifacts.

## Scope of v1

- A 3D page mesh, one lamp, room light, one camera.
- Page deformation as real geometry: crumple, folds, curl, edge lift.
- Shadow casters that the camera does not see (hand, arm, phone, blob).
- A background photo, added afterwards as a 2D layer under the page.
- 2D camera effects. Each effect has its own switch.
- Every parameter can have a fixed value or a random range.
- A CLI for batch work and a desktop editor to make configs.
- Box labels: an enclosing quad and the projected corners for each input box.

Not in v1: more than one lamp, a page held in a hand, handwriting or stamps.

## Stack

| Part | Choice | Reason |
|---|---|---|
| Language | Python 3.12+ | Fits ML pipelines. |
| Renderer | ModernGL (OpenGL 3.3 core) | Small and fast. Runs on macOS and on headless Linux through EGL. |
| Image ops | numpy, Pillow | Camera effects, file I/O, geometry. OpenCV was dropped: it adds 120 MB. |
| Editor | imgui-bundle + glfw | Immediate-mode UI with 3D gizmos. Optional install. |

Install size: the CLI venv is about 43 MB. Assets (desk and background textures, paper scans and generated plain desks) are another 85 MB, fetched by `recapture fetch-assets` or `recapture.fetch_assets()`.

## Decisions

Choices that look odd without their reason, and ideas that were tried and dropped. They are recorded so they do not come back without a new reason. A change that goes against one says so.

- **Path tracing (Mitsuba 3, Metal GPU).** Dropped. It took about 4 s per image at 1536×2048 and still did not look real enough to justify the cost.
- **Crumple as shading only** (per-pixel crease slopes, or a scanned normal map). Dropped. The crumple is now real mesh geometry, and the lamp lights the actual surface. This gives the most convincing creases and lets creases cast small shadows on the paper.
- **Ground plane with the background as a 3D texture.** Dropped for a 2D layer, later replaced by the lit table (see Lit table).
- **One random stream per config value.** A value's random numbers come from a stream keyed by the seed and the value's path in the config (`config/values.py`, `stream`). With one shared generator, adding a key or changing one range would shift every later draw and change every image in a dataset. The mode of a mix config draws from its own stream too, so a config with a fixed mode gives the same images as before modes existed. New random values must draw through `resolve` or `stream`.
- **One crumple scan per photo, never blended.** Each photo picks one of the listed scans. Blending two height maps looked artificial. Paper005 (packing paper) gets larger tiles, smoothing and a strength cap from its `info.json`, so it reads as office paper.
- **Scan mode is 2D.** Scans skip the 3D scene: printer marks at page resolution, one 3×3 matrix for the skew and the dpi scale, then the scanner artifacts (`scanner.py`). The same matrix maps the labels, so they stay exact, and scans need no GPU. Scans have no crumple: a flatbed presses the page flat, and crumple on a scan looked fake. The book-scan look (dark edges, edge dots, bold) is kept gentle.
- **Desks instead of background photos.** See Desks.
- **Assets outside the package.** The CC0 textures and crumple scans (about 85 MB) stay out of git and out of the Python package. `recapture fetch-assets` downloads them once per machine, and configs name them as `@assets/...`, so one config works in a clone, in a `uv tool` install and on another machine.
- **A frame-check retry redraws the frame, not the scene.** A sample that fails the frame check draws only the camera and the page placement again (`FRAME_PATHS` in `config/schema.py`), and from the second half of `max_tries` the folds too (`FOLD_PATHS`: two crossing folds can push a flap past the page, and then no camera fits). The desk, the lamp, the crumple and the items keep the first draw. An earlier version drew the whole sample again on a retry: dragging the camera in the editor changed the desk whenever a different try passed, and a camera stuck where the page never fits ran 50 full tries per frame (about 550 ms). A sample that passes on its first try is the same as before; one that needed a retry is not. The check itself moves a cached coarse page (`quick_frame_check`) instead of building a scene, and a config whose camera, placement and folds are fixed takes one try.
- **A page's seed comes from its path inside the input folder.** `job_seed` hashes the base seed, `doc_a/page_001.png` and the variant. Up to 0.1.x it hashed the file name only, so in a recursive run every document's `page_001.png` got the same desk, lamp, crumple and camera. A page at the top of the input folder has its file name as its path, so flat folders keep their seeds and their images. The editor and `preview` see one file, not an input folder, so they match `render` for pages at the top of the folder.
- **Export splits by page.** Every variant of one page goes to the same split (`export.py`, `split_of`). A split by image would put near-copies of test pages into train and inflate test scores. The cost: a split needs at least one whole page.

## Architecture

```
recapture/
  config/      schema, value specs, sampling with a seed               (pure)
  scene/       page mesh, deformations, paper scans, shadow casters    (pure)
  labels/      sidecar I/O, box transforms, quad fitting                (pure)
  render/      ModernGL context, shaders, passes, readback              (GPU)
  fx/          background layer, blurs, camera effects                  (pure, numpy)
  editor/      config editing helpers (pure), preview, imgui app
  pipeline.py  one job = (source, boxes, config, seed) -> outputs
  cli.py       batch runner, worker pool, `recapture edit`
```

The pure modules hold all the logic and do not touch the GPU or the disk. `render/` and `cli.py` do the I/O. One job is a pure function of (source image, boxes, config, seed).

## Scene model

Units are centimeters. The page rests on the plane z = 0, and +z points up. The top edge of the source document faces +y.

### Page

- The physical size comes from the config (default A4, 21 × 29.7 cm). If the image aspect ratio does not match, the long side keeps the configured size.
- The mesh is a regular grid with 1024 vertices on the long side (about 0.3 mm per cell, close to one output pixel). The crumple lives in this mesh, so it must be dense.
- Each vertex stores its source UV. Deformations change only the position. The UV never changes, so the label transforms are exact.
- Normals come from the grid (central differences), 13 ms for the full mesh.

### Lit table

The background is a table: the plane z = 0 under the page (`background.lit`, on by default). A table pass draws it with the page's lighting code (`TABLE_FS` in `render/shaders.py`). It gets the lamp's color and falloff, the room light, and the shadows of the page and the hands. It uses its own shadow map, wider than the page's, which covers the table the camera sees (`_view_radius`), so long hand shadows do not end in straight cuts. A top-down height pass of the page gives a contact shadow: room light the page blocks where it lies close above the table (`contact_shadow`, `contact_radius_cm`). The pass writes the light arriving at the table and its depth. The background picture is the table's color: `composite()` multiplies it by that light and by `brightness` (reflectance relative to the paper), and defocus uses the table's real depth. The label pass still draws only the page, so labels do not change. Cost: about 5 ms of GPU and one half-float readback.

This replaces the earlier rule that nothing casts onto the background. Research for the next step showed that a shadow that crosses from the page onto the table is the strongest cue that the page lies in the scene. `background.lit: false` keeps the old flat layer.

### Desks

`desk.enabled` turns the table into a desk (`scene/desk.py`). The table pass samples a desk texture at its real-world size (from the texture's `info.json`), with a random offset and an optional quarter turn, and outputs the light the desk reflects. That becomes the background directly. Items are small meshes built in code from rounded extrusions, lathed profiles and swept tubes, with smooth normals, a color per vertex and, where it helps, a texture: a compact keyboard (a real key layout, sculpted keycaps, labels drawn with Pillow's font into a texture, a wedge base), a mousepad with a woven top, a phone face up (black glass) or face down (camera bump), a ceramic mug with coffee and a handle, a pen, a hexagonal pencil, a sticky-note pad with a curling top sheet, and paper with a printed image. Free CC0 3D models of these items are rare; building them in code keeps sizes, colors and placement random and the install small. A mouse was tried and dropped: it did not look real. Each one lies beside a page edge, outside the page. An item that would cut into an earlier one is dropped. The items go into the shadow pass (they shade the page and the desk), the height pass (contact shadow), the MSAA color pass (with the page's lighting code, `PROP_FS`), and the label pass as non-page geometry, so a tilted camera that sees an item in front of the page edge still gets exact labels. Defocus uses their depth.

Desks replaced stock background photos. Those were staged product shots with styled light and props, so a page placed into them never looked like a casual phone photo. A calibrated-photo mode (MoGe-2 found each photo's camera and table plane) was built for them and later removed with them.

### Background off

`background.enabled: false` drops the background layer. The camera effects run on the page with its alpha (the blurs soften the alpha too), and the output is an RGBA PNG, transparent around the page. JPEG has no alpha, so this setting requires `output.format: png`.

### Deformations

They apply in this order: crumple, folds, curl, edge lift. Then the page drops so its lowest point touches z = 0. Code: `scene/deform.py`.

1. **Crumple.** `strength` sets how much: 0 is a flat page, 1 a normal crumple, 2 a heavy one. It scales both shape sources, which are real geometry:
   - `procedural`: random short crease segments. Each crease has a tent profile across it (a sharp ridge or valley on the line, flat at distance `width`) and fades out toward its ends. Overlapping tents form flat facets with sharp edges. The heights are rescaled so the RMS facet slope is 0.15 × `strength`. Each crease also pulls the paper toward its line by (1 − cos of the tilt), so the page shrinks a little and text bends at creases. Crease centers lie on the paper.
   - `scan`: the height map of a scanned crumpled-paper material (ambientCG Paper003, CC0) moves the mesh by up to `strength` × `depth_cm`. Each page gets a random tile offset, a quarter turn and a tile size, so pages differ.
2. **Folds.** Each fold is a straight line in page coordinates. The part on one side rotates about the current 3D position of that line, with a crease rounded over about 1 mm. Folds stack. Known limit: large crossing folds can make one flap pass through another part of the page.
3. **Curl.** A cylinder bend along x and/or y, set as curvature (1 / radius). It keeps paper lengths.
4. **Edge lift.** Each page edge can rise with a quadratic curve over a band.

`deform.folds` is a list of folds or `{"repeat": n, "item": {...}}`, where each repeated item gets its own random values.

### Card look

A plastic card (an ID, driving licence or bank card) is the page with three material options, all off by default so paper looks as before:

- **Rounded corners** (`material.corner_radius_cm`): the color, shadow, height and label passes all drop the page fragments outside a rounded rectangle in page centimeters (`_CORNER` in `render/shaders.py`). ISO/IEC 7810 ID-1 cards use 0.318 cm. All four passes must agree: with the corners cut only in the color pass, the shadow and contact-shadow passes still saw a square card, and the table seen through each cut corner lay in a full shadow and came out black. Casters and desk items draw with radius 0.
- **Laminate glare** (`material.glare`): a glossy laminate mirrors the room. One small lamp gives a highlight only where its mirror angle meets the camera, so a card under a ceiling light or beside a window needs the light's mirror image: a direction (`elevation_deg`, `azimuth_deg`), an angular radius (`size_deg`) and an edge `softness`. The page shader adds `strength × Fresnel(N, V) × cone(reflect(-V, N))` in the ambient light's color. It is analytic rather than an environment map: one cone per image is enough to wash out a patch of text, which is what the glare must do to the data, and it costs no texture.
- **Panel glare, waviness, room reflection and wear.** A round, smooth glare on a clean card reads as a game render. With `glare.shape` `"panel"`, the mirrored light is a rectangle (a ceiling panel or a window, `aspect`, `rotation_deg`), brighter in its middle, with a faintly uneven diffuser; its edge blurs with `roughness`. `glare.waviness` tilts the normal used for reflections by a degree or two of smooth noise at the centimeter scale, so the edges bend like on a real laminate. `material.room_reflection` adds a dim mirror image of the room over the whole card: lighter overhead, with large soft patches, and darker where the view points back at the photographer. `material.wear` adds fine scratches that light up only near the glare, and smudges that dull it into a haze. All of it is analytic in the page shader, seeded by the page's grain offset, and runs only when the laminate is on.
- **Hologram** (`material.hologram`): many ID laminates carry a diffraction pattern. A rainbow tint whose hue follows the reflection direction and the page position, in a fine wavy pattern, is added mostly inside the glare cone.

Rigidity comes from the existing deformation keys: a card config sets the crumple, folds and edge lift to 0 and keeps only a slight curl, and `page.lift_cm` gives the card's thickness. `recapture init --from card` writes such a config. Scan mode is 2D and ignores the three options.

The laminate term clamps its Fresnel base to [0, 1]. Float rounding can put N·V a hair above 1, `pow` of the slightly negative base is NaN in GLSL, and one NaN pixel turned the whole frame black through the auto exposure. A non-finite laminate value is also dropped.

### Light

- One lamp, placed by `elevation_deg` (height above the table), `azimuth_deg` and `distance_cm`, or directly by `position_cm`. Low elevations (12° to 35°) rake across the page and show the crumple most. Overhead light hides it. Room light (`ambient`, default 0.12) fills the shaded side of each crease, so more of it hides the crumple.
- Lamp size sets the shadow softness (PCSS soft shadows). Falloff is inverse square.
- Room light (`ambient`) has a main direction and a focus. Surfaces that face away from it still get 30% from bounced light. Its color follows the lamp unless set.

### Shadow casters

Flat alpha-mask shapes (hand with phone, arm, phone, blob). Each one sits at a fixed world point (`position_cm`), so it stays put when the lamp moves. Without `position_cm` it sits on the ray from a target point on the page to the lamp, at a set height, which suits random configs. Its arm points toward the camera. They cast shadows onto the page only. The camera does not see them.

### Camera

- Position, look-at target, roll, focal length (35 mm equivalent, default 26 mm like a phone main camera), output size.
- Frame check: the page must be inside the frame with a margin and cover a set part of it. A failed sample draws a new camera and page placement (later also new folds), up to `max_tries` (see Decisions). Each try is first tested on a coarse 64-vertex page, so failed tries cost little.

## Rendering

Three passes per job. Only the page is drawn; the background stays transparent.

1. **Shadow pass.** Distance from the lamp into one 2048² perspective map aimed at the page. The page and the shadow casters go in.
2. **Color pass.** 4× MSAA, linear color, transparent background. The output is premultiplied color plus antialiased alpha.
   - Normals from the geometry, so the creases are lit by the lamp.
   - Crease occlusion per vertex from the geometry (valleys see less room light).
   - Diffuse (Lambert, `wrap` 0 by default so creases keep a hard light edge), a weak GGX sheen, paper grain as a small brightness variation.
   - The document texture has mipmaps and anisotropic filtering.
3. **Label pass.** No MSAA, float32. Each pixel stores (u, v, front-face mask, depth).

## Background layer and camera effects

Code: `fx/`. Order:

1. **Background layer.** A photo (random crop, zoom, quarter turn, flip, saturation, blur) or a plain color. Its mean brightness is set relative to the paper white, then the page goes on top ("normal" blend with the page alpha).
2. Defocus blur from the label depth and a focus point on the page (the background counts as in focus).
3. Motion blur (short line, random angle).
4. Vignetting.
5. White balance (gray-world on the paper), tint, auto exposure (a percentile of the paper brightness lands at a target after tone mapping), exposure bias.
6. Sensor noise: Poisson-Gaussian in linear light, color noise at half resolution.
7. Filmic tone mapping (ACES fit) and sRGB.
8. Sharpening (unsharp mask), then JPEG when the file is written.

No effect moves pixels, so the labels stay valid.

## Config

JSON. Each parameter is a plain value or a random spec:

```json
{"uniform": [30, 60]}
{"randint": [80, 95]}
{"normal": [45, 5], "clip": [30, 60]}
{"choice": ["scan", "procedural"], "weights": [0.7, 0.3]}
{"repeat": {"randint": [0, 2]}, "item": {"angle_deg": {"uniform": [-25, 25]}}}
```

`configs/phone_photo.json` is a full example with random ranges for every part. `DEFAULTS` in `config/schema.py` lists every key. The CLI writes the resolved config next to each output, with every random value replaced by its sample, so an image can be made again.

## Labels

### Input: one sidecar JSON per image

`doc_001.png` goes with `doc_001.json`:

```json
{"boxes": [
  {"id": "p1", "label": "paragraph", "bbox": [x_min, y_min, x_max, y_max]},
  {"id": "f1", "label": "figure", "quad": [[x, y], [x, y], [x, y], [x, y]]}
]}
```

### Output: one JSON per render

```json
{
  "source": "doc_001.png",
  "image": "doc_001_0003.jpg",
  "size": [1536, 2048],
  "page_corners": [[x, y], [x, y], [x, y], [x, y]],
  "boxes": [{"id": "p1", "label": "paragraph",
             "quad": [[x, y], [x, y], [x, y], [x, y]],
             "corners": [[x, y], [x, y], [x, y], [x, y]],
             "visible": 0.97, "truncated": false}]
}
```

- `quad`: the enclosing quad, clockwise, starting nearest the box's top-left corner. `null` when no part is visible.
- `corners`: the 4 source corners projected into the photo.
- `visible`: the part of the box area the camera sees. `truncated`: part of the box is outside the frame.

### How the quad is computed

The visible part of a box has two kinds of edges. Its own edges are known exactly: the outline is sampled in UV about twice per photo pixel, moved through the mesh and the camera, and kept where the label pass shows it visible. Cut edges (a fold covering the box, the page edge, the frame edge) are known to pixel accuracy, so each box pixel on a cut edge adds a ±1 px square. A pixel is on a cut edge when the UV jump to its outside neighbor is more than twice the step from the pixel behind it. A greedy edge removal fits a 4-point polygon around the convex hull; each step only adds area. On a flat page the quad equals the exact projected box.

### What is guaranteed

Every pixel whose center shows the box lies inside its quad. A pixel that shows the box only in part, at an edge seen almost side-on, can lie up to 1 px outside. Self-intersecting pages (large crossing folds) can break this at the line where the paper passes through itself.

### Label check

`tests/test_labels_render.py` renders each box alone (black on white) and the blank page in the same scene. Box ink is where the box render is much darker than the blank one. The test checks the guarantee above over 24 random scenes, flat and deformed. A larger stress run (2044 boxes, 200 deformed scenes, 1024 mesh) gave one partial pixel 0.67 px outside, within the guarantee.

## CLI

```bash
recapture render --input docs/ --config configs/phone_photo.json --out out/ --variants 5 --workers 8
```

`recapture validate config.json` checks a config. `recapture sample config.json --seed 7` prints a resolved config. Workers are separate processes, each with its own GL context; on Linux the context uses EGL with no display.

## Editor

`recapture edit [config] [--doc page.png]`, built on imgui-bundle. ModernGL attaches to the window's OpenGL context, so renders go straight to the screen.

- **Preview.** The same code path as the CLI (`editor/preview.py`), with a smaller image and a coarser page while you drag (640 px, 160-vertex mesh, no camera effects), and the full look after you let go (1024 px, 512-vertex mesh, effects on). A slider drag counts as a drag too. Built pages are cached by their page and deform settings (`scene.PageCache`), and the background layer by its settings, so moving the lamp, the camera or a hand only renders again (about 26 ms per drag frame). Pixel-based effect sizes scale with the preview size. A full-size render is the exact CLI image for that variant; `tests/test_editor_preview.py` checks this pixel for pixel.
- **Layout.** A 3D scene view in the center, the photo on the right, sliders on the left. A first version exposed the config tree directly and was too hard to use; it was replaced by this layout, modeled on Blender and Unity.
- **Scene view.** The page is rendered by the same renderer from a free orbit camera (`editor/view3d.py`), lit by the scene lamp. It stays on the GPU (`Renderer.render_display`: display colors into a texture, no read back), and the renderer keeps the page buffers and the document texture between frames, so an orbit frame costs about 2 ms. Overlays drawn with ImGui show a table grid, the lamp, a camera icon with its aim point, and the hand-shadow shape. Click an object to select it; ImGuizmo translate arrows move it. Moves are written back as fixed config values: lamp → azimuth, elevation, distance; camera → position (aim kept); aim → target; hand → its world position (`position_cm`), so moving the lamp later moves only the shadow. Before a lamp move, hands that aim at a page point are pinned at their current place. Random shadow casters (a `repeat` rule) stay random until **Fix them in place** or a drag turns this sample's casters into a fixed list.
- **Sliders.** A short list of the settings that matter most. A checkbox turns a slider into a two-handle range (a uniform or randint spec). **Advanced settings…** opens the full tree over the merged config (`editor/advanced.py`), where any value can be fixed, random, or edited as JSON.
- **Files.** The editor keeps only the user's own keys. Saving strips values equal to the defaults, writes compact JSON, and rebases relative paths to the new folder.
- **Frame check.** If a sample fails the frame check, the preview still shows it, with a note that the CLI would draw a new sample.

## Performance

Apple M4 Pro, `configs/phone_photo.json` (desk, 17 layout boxes), output 1536×2048 JPEG, median of 16:

| Stage | ms |
|---|---|
| Load | 5 |
| Scene (dense mesh, deformations, occlusion, desk items) | 250 |
| Render (passes on the GPU) | 44 |
| Camera effects on the GPU (`fx.backend: "gpu"`) | 27 |
| Labels (17 boxes) | 134 |
| Write | 12 |
| **Total, 1 worker** | **about 465** |

The numpy camera effects (`fx.backend: "cpu"`) take about 290 ms instead, 0.79 s per image in total. Batch of 96 images with 8 workers: 9.0 img/s on the GPU path, 6.3 img/s on the CPU path. More than 8 workers is slower on this machine (10 performance cores, one GPU).

The GPU effects (`fx/gpu.py`) run the same steps in the same order as `fx.apply_fx`. The render stays on the GPU; only a 1-in-4 pixel sample (for the white balance and auto exposure, computed in numpy exactly as before) and the final 8-bit image come back. Against the numpy version, with the noise off: mean difference 0.1 to 0.2 of 255; without depth blur and sharpening, at most 1 level everywhere. The rest comes from the depth blur (numpy shrinks the image for wide blurs, the GPU uses an exact Gaussian) along hard edges, and from Pillow's approximate Gaussian in the unsharp mask. The noise uses other random numbers with the same statistics (strength, color share and grain size match to 0.01). `tests/test_fx_gpu.py` checks these bounds. A transparent background always uses the numpy path.

## Phases

1. **Core path.** Config, page mesh, lamp with shadows, camera, label pass, labels, CLI. Done.
2. **Deformation.** Folds, crumple, curl, edge lift; label check on deformed pages. Done.
3. **Look.** Page-only render, crumple as dense geometry (procedural or scan), background as a 2D layer, shadow casters, camera effects, angle-based lamp. Done.
4. **Editor.** Live preview, drag tools, settings with fixed/random switches, save/load, full render matching the CLI. Done.
5. **Scale.** EGL on Linux, worker pool tuning, benchmark report.
