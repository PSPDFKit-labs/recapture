# Box files

One JSON file next to each page image, same stem: `invoice.png` → `invoice.json`. Coordinates are pixels of that image, origin at the top-left, x right, y down. Either format below works; Recapture detects which.

## Simple format

```json
{"boxes": [
  {"id": "p1", "label": "paragraph", "bbox": [120, 340, 980, 460]},
  {"id": "f1", "label": "figure", "quad": [[100, 500], [900, 510], [895, 900], [95, 890]]}
]}
```

- Each box needs `bbox` (`[x_min, y_min, x_max, y_max]`, max > min) or `quad` (4 `[x, y]` corners: top-left, top-right, bottom-right, bottom-left).
- `id` (default: the position in the list) and `label` (the class, default `""`) are optional.
- Any other key, such as `text`, is copied unchanged into each output box.

## Document-analysis API response

An object with `output.elements`, as the [Nutrient Parse API](https://www.nutrient.io/guides/dws-data-extraction/parsing/) returns it. Per element:

- `points` (4 `[x, y]` corners) or `bounds` (`{"x", "y", "width", "height"}`). When an element has both, `points` is used.
- `role`, or else `type`, becomes the label.
- `page.width` and `page.height`, when they differ from the image size, scale the coordinates to the image.
- `type`, `role`, `text`, `readingOrder` and `confidence` are copied into the output boxes.

All elements must come from one page: one image is one page. A response for a whole document holds every page; cut it with:

```bash
recapture split-pages <response>.json -o <pages folder>
pdftoppm -r 200 -png <document>.pdf <pages folder>/<response name>
```

`split-pages` writes `<name>-1.json`, `<name>-2.json`, … (`-01` from 10 pages on), the names `pdftoppm` gives the page images. `--name` changes the part before the number. Any resolution works. Without Poppler, render the pages with another tool and name each image like its box file.

## Output labels

Every rendered image gets `<stem>_<variant>.json`: `source`, `image`, `size`, `page_corners`, and `boxes`, each with `id`, `label`, `quad` (the 4 corners in that image's pixels; `null` when the box is out of frame or hidden), `corners`, `visible`, `truncated`, plus the copied keys.
