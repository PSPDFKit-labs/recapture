"""`recapture split-pages`: an API response for a whole document becomes one box file per page. No GL needed."""

import json

import numpy as np
import pytest

from recapture.cli import main
from recapture.labels import BoxFileError, load_boxes, split_api_pages

PAGE = {"width": 1653, "height": 2339}


def _element(n, page_index, x, y, w=400, h=200, kind="table"):
    return {"id": f"e{n}", "type": kind, "readingOrder": n, "confidence": 0.9,
            "page": {**PAGE, "pageIndex": page_index, "pageNumber": page_index + 1},
            "bounds": {"x": x, "y": y, "width": w, "height": h},
            "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]}


def _response():
    # Pages 0 and 2 have elements, page 1 has none: a blank page between two pages with tables.
    return {"configuration": {"mode": "agentic"},
            "output": {"elements": [_element(0, 0, 100, 50), _element(1, 2, 80, 300, kind="text"),
                                    _element(2, 0, 100, 900)], "markdown": "..."}}


def test_split_keeps_each_page_s_elements_in_order():
    parts = split_api_pages(_response())
    assert list(parts) == [0, 2]
    assert [e["id"] for e in parts[0]["output"]["elements"]] == ["e0", "e2"]
    assert [e["id"] for e in parts[2]["output"]["elements"]] == ["e1"]


def test_split_rejects_a_file_that_is_not_a_response():
    with pytest.raises(BoxFileError, match="output.elements"):
        split_api_pages({"boxes": []})


def test_the_whole_response_fails_and_points_to_split_pages(tmp_path):
    whole = tmp_path / "doc.json"
    whole.write_text(json.dumps(_response()))
    with pytest.raises(BoxFileError, match="split-pages"):
        load_boxes(whole, (1653, 2339))


def test_cli_writes_page_box_files_that_load(tmp_path):
    response = tmp_path / "doc.json"
    response.write_text(json.dumps(_response()))
    out = tmp_path / "pages"
    assert main(["split-pages", str(response), "-o", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["doc-1.json", "doc-3.json"]  # page 2 has no elements
    boxes = load_boxes(out / "doc-1.json", (1653, 2339))
    assert [b.label for b in boxes] == ["table", "table"]
    np.testing.assert_allclose(boxes[0].quad[0], [100, 50])
    # Half-size page images get half-size boxes: the box file keeps the page size the coordinates refer to.
    half = load_boxes(out / "doc-1.json", (826, 1169))
    np.testing.assert_allclose(half[0].quad[2], [250, 125], atol=0.3)


def test_names_pad_like_pdftoppm_from_ten_pages(tmp_path):
    data = _response()
    data["metrics"] = {"pagesProcessed": 12}  # pdftoppm writes doc-01.png ... doc-12.png
    response = tmp_path / "scan 7.json"
    response.write_text(json.dumps(data))
    out = tmp_path / "pages"
    assert main(["split-pages", str(response), "-o", str(out), "--name", "doc"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["doc-01.json", "doc-03.json"]
