from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from cascadeforge import editor
from cascadeforge.sanitize import sanitize_image


class FakeResponse:
    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise editor.requests.HTTPError("temporary")

    def json(self):
        return self.payload


def test_request_json_retries(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse({"ok": True}, fail=len(calls) == 1)

    monkeypatch.setattr(editor.requests, "request", fake_request)
    monkeypatch.setattr(editor.time, "sleep", lambda _: None)
    assert editor._request_json("GET", "https://example.invalid") == {"ok": True}
    assert len(calls) == 2


def test_extract_nested_url_before_top_level_task_id():
    payload = {
        "id": "task-1",
        "status": "success",
        "result": {"data": [{"url": "https://example.invalid/result.jpg"}]},
    }
    assert editor._extract_image_reference(payload) == (
        "url",
        "https://example.invalid/result.jpg",
    )
    assert editor._extract_task_id(payload) == "task-1"


def test_extract_nested_base64_before_top_level_task_id():
    payload = {
        "id": "task-2",
        "status": "success",
        "result": {"data": [{"b64_json": "YWJj"}]},
    }
    assert editor._extract_image_reference(payload) == ("b64_json", "YWJj")
    assert editor._extract_task_id(payload) == "task-2"


def test_extract_task_only_and_legacy_shallow_url():
    assert editor._extract_image_reference({"id": "task-3"}) is None
    assert editor._extract_task_id({"id": "task-3"}) == "task-3"
    assert editor._extract_image_reference({"data": [{"url": "https://example.invalid/a"}]}) == (
        "url",
        "https://example.invalid/a",
    )


def test_download_reference_decodes_base64(tmp_path):
    output = tmp_path / "result.bin"
    editor._download_reference(("b64_json", "YWJj"), output)
    assert output.read_bytes() == b"abc"


def test_build_prompt_contains_all_rounds():
    data = {f"ROUND_{index}": {"long": f"edit-{index}"} for index in range(1, 5)}
    prompt = editor.build_prompt_from_json(data)
    assert all(f"edit-{index}" in prompt for index in range(1, 5))
    assert "序号、数字、标签" in prompt


def test_build_prompt_removes_candidate_ids():
    data = {
        f"ROUND_{index}": {"long": f"将编号{index}左侧目标改成蓝色"}
        for index in range(1, 5)
    }
    prompt = editor.build_prompt_from_json(data)
    assert "编号" not in prompt
    assert "ID" not in prompt


def test_normalize_grid_builds_strict_incremental_sequence(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    for name in ("IMAGE_2", "MASK", "SELECTION"):
        (root / name).mkdir(parents=True)
    digest = "sample"
    Image.new("RGB", (4, 4), (20, 20, 20)).save(root / "IMAGE_2" / f"{digest}.jpg", quality=100)

    # Legacy masks are cumulative; each round introduces one additional pixel.
    targets = [(0, 0), (1, 0), (0, 1), (1, 1)]
    mask_grid = Image.new("RGBA", (8, 8), (20, 20, 20, 255))
    cumulative = []
    for position in targets:
        cumulative.append(position)
        tile = Image.new("RGBA", (4, 4), (20, 20, 20, 255))
        alpha = tile.getchannel("A")
        for x, y in cumulative:
            alpha.putpixel((x, y), 0)
        tile.putalpha(alpha)
        index = len(cumulative) - 1
        mask_grid.paste(tile, ((index % 2) * 4, (index // 2) * 4))
    mask_grid.save(root / "MASK" / f"{digest}_MASK.png")

    generated = Image.new("RGB", (8, 8), "white")
    colors = [(220, 0, 0), (0, 220, 0), (0, 0, 220), (220, 180, 0)]
    for index, color in enumerate(colors):
        generated.paste(Image.new("RGB", (4, 4), color), ((index % 2) * 4, (index // 2) * 4))
    generated_path = tmp_path / "generated.png"
    output_path = tmp_path / "normalized.png"
    generated.save(generated_path)

    upload_mask = editor.independent_upload_mask(root, digest)
    upload_alpha = np.asarray(Image.open(upload_mask).getchannel("A")) < 128
    assert [int(upload_alpha[y0:y1, x0:x1].sum()) for x0, y0, x1, y1 in editor._quadrant_boxes(8, 8)] == [1, 1, 1, 1]

    editor.normalize_grid(root, digest, generated_path, output_path)
    normalized = np.asarray(Image.open(output_path).convert("RGB"))
    original = np.asarray(Image.open(root / "IMAGE_2" / f"{digest}.jpg").convert("RGB"))
    frames = [normalized[0:4, 0:4], normalized[0:4, 4:8], normalized[4:8, 0:4], normalized[4:8, 4:8]]
    for round_index, frame in enumerate(frames):
        for target_index, (x, y) in enumerate(targets):
            expected = colors[target_index] if target_index <= round_index else tuple(original[y, x])
            assert np.allclose(frame[y, x], expected, atol=10)
        assert np.allclose(frame[3, 3], original[3, 3], atol=10)


def test_sanitize_image_drops_exif(tmp_path):
    source = tmp_path / "source.jpg"
    destination = tmp_path / "public.jpg"
    exif = Image.Exif()
    exif[0x010E] = "private-note"
    Image.new("RGB", (32, 32), "blue").save(source, exif=exif)
    sanitize_image(source, destination)
    with Image.open(destination) as image:
        assert not image.getexif()
