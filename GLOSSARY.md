# Recapture

Recapture turns clean page images into phone photos and scans, and moves each page's layout boxes into every new image. The output is training data for layout detection.

## Inputs

**Page**:
A clean image of one document page, PNG or JPG. One image holds one page.
_Avoid_: document, doc, input image

**Box file**:
The JSON file next to a page with the same stem (`invoice.png`, `invoice.json`). It holds the page's boxes in the simple format or the API format.
_Avoid_: sidecar, annotation file

**Box**:
One layout region on a page, given as a `bbox` or a 4-point `quad` in page pixels, with a label.
_Avoid_: element, region, annotation

**Label**:
The class name of a box, such as `paragraph` or `Text`. In the API format it comes from `role`, or else `type`.
_Avoid_: class, category, role

**API format**:
A response from the parse endpoint of Nutrient's DWS Data Extraction API (https://www.nutrient.io/guides/dws-data-extraction/parsing/), used as a box file as it is. Recapture reads its `output.elements`.
_Avoid_: document-analysis response

## Modes

**Mode**:
How one image is made: `photo` or `scan`. Each image has one mode.
_Avoid_: 3D mode, 2D mode

**Photo**:
An image made in photo mode: the page as a 3D mesh on a desk, lit by a lamp and taken by a virtual phone camera.
_Avoid_: capture, render (for the result)

**Scan**:
An image made in scan mode: the page printed, then passed through a virtual flatbed scanner. Flat, with no crumple.
_Avoid_: degradation, 2D degradation

**Mix**:
A config whose mode is a choice between photo and scan, so one run gives both.

## Photo scene

**Scene**:
Everything one photo is drawn from: the page mesh, the desk, the lamp, the shadow casters and the camera.

**Deformation**:
A change to the page's shape. The four kinds apply in this order: crumple, folds, curl, edge lift.
_Avoid_: distortion, warp

**Crumple**:
Small creases over the whole page, made as real mesh geometry. Its shape comes from a crumple scan or from procedural creases.
_Avoid_: wrinkle, crinkle

**Crumple scan**:
The height map of a scanned sheet of crumpled paper (ambientCG Paper003 or Paper005). Each photo uses one, picked at random. Two are never mixed.
_Avoid_: paper texture, wrinkle map

**Fold**:
A straight line across the page where one side turns over the other.

**Card look**:
A page rendered as a laminated plastic card: rounded corners, a glossy surface, laminate glare and an optional hologram sheen (`material.corner_radius_cm`, `material.glare`, `material.hologram`). The `card` example config sets it up.
_Avoid_: plastic mode, ID mode

**Laminate glare**:
The mirror image of a ceiling or window light on a glossy card, washing out a patch of it (`material.glare`). With `shape` `"panel"` it is a rectangle with bent edges; `material.wear` adds scratches and smudges in it.
_Avoid_: reflection, flare

**Desk**:
The textured surface under the page in a photo, at its real-world size, with desk items beside the page.
_Avoid_: table, background photo

**Desk item**:
An object beside the page that casts shadows: keyboard, mousepad, phone, mug, pen, pencil, sticky notes or paper. It never covers the page.
_Avoid_: prop

**Background**:
Everything around the page in a photo. It is a desk, a flat surface, or transparent when `background.enabled` is false.

**Lamp**:
The one light in a scene. It casts every shadow.
_Avoid_: light source, sun

**Room light**:
The soft light from all around (`light.ambient`). More of it fills crease shadows and hides the crumple.
_Avoid_: ambient occlusion

**Shadow caster**:
A flat shape (hand with phone, arm, phone, blob) that casts a shadow on the page. The camera does not see it. Config key: `occluders`.
_Avoid_: occluder, hand

**Camera**:
The virtual phone camera: its position, aim point, roll and focal length.

**Frame check**:
The test that the page lies inside the camera frame and covers a set share of it. A random sample that fails is drawn again.

**Camera effects**:
The 2D steps after rendering: focus blur, motion blur, vignette, white balance, exposure, noise, sharpening and JPEG (`fx`).
_Avoid_: post-processing, filters

**Scanner-app crop**:
A photo cut to the page and straightened, like a phone scanner app (`crop`). The page fills the whole image.
_Avoid_: fit page, dewarp

## Scan steps

**Printer marks**:
Print flaws added at page resolution before scanning: ink bleed, broken strokes, ghost, toner streaks and toner specks.
_Avoid_: printer artifacts

**Scanner artifacts**:
Flaws the scanner adds: skew, blur, bold, noise, streaks, dust, dark edges, edge dots and the color mode.

**Dark edges**:
Dark bands at the page sides where the scanner lid shows past the paper. Edge dots are the specks in those bands.
_Avoid_: border, shadow

## Configs and randomness

**Config**:
A JSON file that sets every value Recapture uses, each as a fixed value or a random value. It belongs to the user; library updates never change it.
_Avoid_: preset, settings file

**Example config**:
One of the three configs `recapture init` copies from: `photo`, `scan` or `mix`.
_Avoid_: preset, template

**Defaults**:
The value of every config key when a config leaves it out (`DEFAULTS` in `config/schema.py`, printed by `recapture defaults`).

**Random value**:
A config value that draws a new value for each image: `uniform`, `randint`, `normal`, `choice` or `repeat`.
_Avoid_: distribution, spec (in user-facing text)

**Sample**:
One set of drawn values for one image. Each image writes its sample to `<stem>_<variant>.config.json`.
_Avoid_: resolved config, draw

**Variant**:
The number of one image made from a page: `-n 5` makes variants 0 to 4. It is the 3-digit suffix in output names.
_Avoid_: copy, augmentation

**Seed**:
The base number that, with the page's path inside the input folder (its file name at the top level) and the variant, fixes an image's sample. Same config, seed and version give the same image.

**Stream**:
The random number source of one config value, keyed by the value's path in the config. Changing one range leaves every other drawn value alone.

## Outputs

**Label file**:
The JSON next to each output image (`invoice_000.json`) with `source`, `image`, `size`, `page_corners` and the boxes.
_Avoid_: annotation

**Quad**:
In a label file, the smallest 4-point shape around the visible part of a box, clockwise from the corner nearest its top-left. `null` when none of the box shows.
_Avoid_: polygon, bbox (for outputs)

**Corners**:
A box's 4 source corners projected into the output image. They match the quad on a flat page.

**Page corners**:
The 4 corners of the whole page in the output image.

**Preview sheet**:
The contact sheet `recapture preview` writes: a few images with their quads drawn in red.
_Avoid_: contact sheet, thumbnail grid

**Dataset**:
The folder `recapture render` fills with images, label files and samples.

**Export**:
The dataset written as COCO or YOLO-OBB, split into train, val and test.

**Split**:
Train, val or test. All variants of one page go to the same split.

## Distribution

**Assets**:
The CC0 desk textures, background textures and crumple scans, downloaded once per machine by `recapture fetch-assets`.
_Avoid_: resources, data files

**Asset folder**:
Where the assets live: `$RECAPTURE_ASSETS`, else `assets/` in a clone, else `~/.cache/recapture/assets`.

**@assets path**:
A config path that starts with `@assets/` and points into the asset folder, so a config works on any machine.

**Skill**:
`skills/recapture/`: the instructions an agent follows to install the CLI and make a dataset. The repo is also its Claude Code plugin and marketplace.
