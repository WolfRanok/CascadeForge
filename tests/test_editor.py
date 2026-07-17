import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cascadeforge import editor
from cascadeforge.sanitize import sanitize_image


class FakeResponse:
    def __init__(self, payload, fail=False, status_code=200, content=b""):
        self.payload = payload
        self.fail = fail
        self.status_code = status_code
        self.text = ""
        self.content = content

    def raise_for_status(self):
        if self.fail:
            raise editor.requests.HTTPError("temporary")

    def json(self):
        return self.payload


def test_request_json_retries(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(
            {"message": "temporary"} if len(calls) == 1 else {"ok": True},
            status_code=500 if len(calls) == 1 else 200,
        )

    monkeypatch.setattr(editor.requests, "request", fake_request)
    monkeypatch.setattr(editor.time, "sleep", lambda _: None)
    assert editor._request_json("GET", "https://example.invalid") == {"ok": True}
    assert len(calls) == 2


def test_request_json_reports_quota_without_retry(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(
            {"code": "quota_not_enough", "message": "user quota is not enough"},
            status_code=403,
        )

    monkeypatch.setattr(editor.requests, "request", fake_request)
    with pytest.raises(editor.TransportError, match="账户额度不足"):
        editor._request_json("POST", "https://example.invalid")
    assert len(calls) == 1


def test_extract_nested_result_url_with_top_level_task_id():
    payload = {
        "id": "task-1",
        "status": "success",
        "result": {"data": [{"url": "https://example.invalid/result.jpg"}]},
    }
    assert editor._extract_reference(payload) == (
        "https://example.invalid/result.jpg",
        "task-1",
    )


def test_extract_legacy_shallow_url_and_task_only_response():
    assert editor._extract_reference({"data": [{"url": "https://example.invalid/a.jpg"}]}) == (
        "https://example.invalid/a.jpg",
        None,
    )
    assert editor._extract_reference({"id": "task-2"}) == (None, "task-2")


def test_build_prompt_contains_all_rounds():
    data = {f"ROUND_{index}": {"long": f"edit-{index}"} for index in range(1, 5)}
    prompt = editor.build_prompt_from_json(data)
    assert all(f"edit-{index}" in prompt for index in range(1, 5))
    assert "四格是独立任务" in prompt
    assert "右上局部近景：只编辑透明 Mask 内的目标：edit-2" in prompt
    assert "右下局部近景：只编辑透明 Mask 内的目标：edit-4" in prompt
    assert "透明 Mask 是唯一目标位置" in prompt
    assert "忽略指令中可能不准确的位置词" in prompt
    assert "Mask 外和周围上下文保持原图" in prompt
    assert "禁止编号、文字" in prompt
    assert "跨象限修改" in prompt


def _make_legacy_mask_root(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    for name in ("IMAGE_2", "IMAGE_2X4", "MASK", "SELECTION"):
        (root / name).mkdir(parents=True)
    digest = "sample"
    original = Image.new("RGB", (4, 4), (20, 20, 20))
    original.save(root / "IMAGE_2" / f"{digest}.jpg", quality=100)
    grid = Image.new("RGB", (8, 8), (20, 20, 20))
    grid.save(root / "IMAGE_2X4" / f"{digest}_IMAGE.jpg", quality=100)
    targets = [(0, 0), (3, 0), (0, 3), (3, 3)]
    mask_grid = Image.new("RGBA", (8, 8), (20, 20, 20, 255))
    cumulative = []
    for index in range(4):
        cumulative.append(targets[index])
        alpha = Image.new("L", (4, 4), 255)
        for position in cumulative:
            alpha.putpixel(position, 0)
        tile = Image.new("RGBA", (4, 4), (20, 20, 20, 255))
        tile.putalpha(alpha)
        mask_grid.paste(tile, ((index % 2) * 4, (index // 2) * 4))
    mask_grid.save(root / "MASK" / f"{digest}_MASK.png")
    return root, digest, targets


def _make_independent_crop_root(tmp_path):
    root = tmp_path / "CROP_IMAGE_MASK"
    for name in ("IMAGE_2", "MASK", "SELECTION"):
        (root / name).mkdir(parents=True)
    digest = "crop"
    Image.new("RGB", (20, 20), (20, 20, 20)).save(root / "IMAGE_2" / f"{digest}.jpg")
    targets = [(2, 2), (17, 2), (2, 17), (17, 17)]
    mask_grid = Image.new("RGBA", (40, 40), (20, 20, 20, 255))
    for index, (x, y) in enumerate(targets):
        tile = Image.new("RGBA", (20, 20), (20, 20, 20, 255))
        tile.putpixel((x, y), (20, 20, 20, 0))
        mask_grid.paste(tile, ((index % 2) * 20, (index // 2) * 20))
    mask_grid.save(root / "MASK" / f"{digest}_MASK.png")
    (root / "SELECTION" / f"{digest}_selection.json").write_text(
        json.dumps({"selection_mode": editor.SELECTION_MODE}), encoding="utf-8"
    )
    return root, digest, targets


def test_crop_layout_has_shared_context_boxes_and_keeps_targets_inside(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "MIN_CROP_PADDING", 2)
    monkeypatch.setattr(editor, "CROP_PADDING_RATIO", 0.2)
    root, digest, targets = _make_independent_crop_root(tmp_path)
    tile_size, boxes, masks = editor._crop_layout(root, digest)
    assert len(set(boxes)) == 4
    assert all((right - left, bottom - top) == tile_size for left, top, right, bottom in boxes)
    for mask, (left, top, right, bottom), (x, y) in zip(masks, boxes, targets):
        assert left <= x < right and top <= y < bottom
        assert right <= 20 and bottom <= 20
    assert tile_size[0] < 20 and tile_size[1] < 20


def test_materialize_edit_inputs_creates_cropped_image_and_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "MIN_CROP_PADDING", 2)
    monkeypatch.setattr(editor, "CROP_PADDING_RATIO", 0.2)
    root, digest, _ = _make_independent_crop_root(tmp_path)
    image_path, mask_path = editor.materialize_edit_inputs(root, digest)
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        assert image.size == mask.size
        assert image.size[0] % 2 == 0 and image.size[1] % 2 == 0
        alpha = np.asarray(mask.getchannel("A")) < 128
        quadrants = [alpha[0:alpha.shape[0] // 2, 0:alpha.shape[1] // 2],
                     alpha[0:alpha.shape[0] // 2, alpha.shape[1] // 2:],
                     alpha[alpha.shape[0] // 2:, 0:alpha.shape[1] // 2],
                     alpha[alpha.shape[0] // 2:, alpha.shape[1] // 2:]]
        assert [int(item.sum()) for item in quadrants] == [1, 1, 1, 1]


def test_crop_paste_uses_soft_boundary_and_full_target(tmp_path):
    target = np.zeros((20, 30), dtype=bool)
    target[0, 0] = True
    alpha = editor._crop_blend_alpha((30, 20), target)
    assert alpha[0, 0] == 1.0
    assert alpha[0, 10] == 0.0
    assert alpha[10, 15] == 1.0
    assert 0.0 < alpha[1, 15] < 1.0


def test_materialize_upload_mask_converts_legacy_cumulative_masks(tmp_path):
    root, digest, _ = _make_legacy_mask_root(tmp_path)
    upload_mask = editor.materialize_upload_mask(root, digest)
    alpha = np.asarray(Image.open(upload_mask).getchannel("A")) < 128
    boxes = editor._quadrant_boxes(8, 8)
    counts = [int(alpha[y0:y1, x0:x1].sum()) for x0, y0, x1, y1 in boxes]
    assert counts == [1, 1, 1, 1]


def test_materialize_upload_mask_rejects_old_three_target_global_mask(tmp_path):
    root, digest, _ = _make_legacy_mask_root(tmp_path)
    mask_path = root / "MASK" / f"{digest}_MASK.png"
    with Image.open(mask_path) as source:
        mask_grid = source.convert("RGBA")
    mask_grid.paste(Image.new("RGBA", (4, 4), (20, 20, 20, 0)), (4, 4))
    mask_grid.save(mask_path)
    (root / "SELECTION" / f"{digest}_selection.json").write_text(
        json.dumps({"selection_mode": editor.LEGACY_GLOBAL_MODE}), encoding="utf-8"
    )

    with pytest.raises(editor.TransportError, match="process_vl_ac.py"):
        editor.materialize_upload_mask(root, digest)


def test_remote_mask_verification_checks_uploaded_alpha(tmp_path, monkeypatch):
    root, digest, _ = _make_legacy_mask_root(tmp_path)
    upload_mask = editor.materialize_upload_mask(root, digest)
    monkeypatch.setattr(
        editor.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({}, content=upload_mask.read_bytes()),
    )
    stats = editor.verify_remote_mask("https://example.invalid/mask.png", upload_mask)
    assert stats["size"] == [8, 8]
    assert stats["transparent_pixels"] == 4


def test_compose_accumulates_four_target_edits_from_previous_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "MIN_CROP_PADDING", 2)
    monkeypatch.setattr(editor, "CROP_PADDING_RATIO", 0.2)
    root, digest, targets = _make_independent_crop_root(tmp_path)
    generated = Image.new("RGB", (200, 200), (200, 0, 0))
    colors = [(240, 0, 0), (0, 240, 0), (0, 0, 240), (240, 240, 0)]
    for index, color in enumerate(colors):
        generated.paste(
            Image.new("RGB", (100, 100), color), ((index % 2) * 100, (index // 2) * 100)
        )
    raw_path = tmp_path / "raw.png"
    output_path = tmp_path / "output.png"
    generated.save(raw_path)

    passed, metrics = editor.compose_and_measure(root, digest, raw_path, output_path)

    assert passed and all(item["passed"] for item in metrics)
    result = np.asarray(Image.open(output_path).convert("RGB"))
    frames = [result[0:100, 0:100], result[0:100, 100:200],
              result[100:200, 0:100], result[100:200, 100:200]]
    _, crop_boxes, _ = editor._crop_layout(root, digest)
    for round_index, frame in enumerate(frames):
        for target_index, (x, y) in enumerate(targets):
            expected = colors[target_index] if target_index <= round_index else (20, 20, 20)
            assert np.allclose(frame[y * 5, x * 5], expected, atol=25)
        # The full current patch is pasted, including its surrounding context.
        left, top, _, _ = crop_boxes[round_index]
        assert np.allclose(
            frame[int((top + 1.5) * 5), int((left + 1.5) * 5)],
            colors[round_index],
            atol=25,
        )
        # The outer crop edge remains the previous frame, preventing a hard seam.
        assert np.allclose(
            frame[int((top + 0.5) * 5), int((left + 1.5) * 5)],
            (20, 20, 20),
            atol=25,
        )
        assert np.allclose(frame[50, 50], (20, 20, 20), atol=25)


def test_quality_gate_rejects_invisible_target_edits(tmp_path):
    root, digest, _ = _make_legacy_mask_root(tmp_path)
    raw_path = tmp_path / "raw.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (8, 8), (20, 20, 20)).save(raw_path)
    passed, metrics = editor.compose_and_measure(root, digest, raw_path, output_path)
    assert not passed
    assert all(not item["passed"] for item in metrics)


def test_rejected_task_is_terminal_without_api_call(tmp_path, monkeypatch):
    root = tmp_path / "IMAGE_MASK"
    quality = root / ".cascadeforge" / "quality" / "sample.json"
    quality.parent.mkdir(parents=True)
    quality.write_text(
        '{"version":"four-target-cropped-v1","passed":false}', encoding="utf-8"
    )
    monkeypatch.setattr(
        editor,
        "_request_json",
        lambda *args, **kwargs: pytest.fail("rejected task must not call API"),
    )
    result = editor.process_one("sample", root, object())
    assert result["status"] == "rejected"


def test_old_quality_sidecar_does_not_block_cropped_pipeline(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    quality = root / ".cascadeforge" / "quality" / "sample.json"
    quality.parent.mkdir(parents=True)
    quality.write_text(
        '{"version":"four-target-independent-v1","passed":false}', encoding="utf-8"
    )
    assert not editor._is_rejected(root, "sample")


def test_old_download_cache_cannot_reuse_raw_full_image_result(tmp_path):
    raw_path = tmp_path / "sample_raw.jpg"
    cache_path = tmp_path / "sample_url.json"
    raw_path.write_bytes(b"old-raw")
    cache_path.write_text(
        json.dumps(
            {
                "url": "https://example.invalid/old.jpg",
                "pipeline_version": "four-target-independent-v1",
            }
        ),
        encoding="utf-8",
    )
    assert editor._load_current_download_cache(cache_path, raw_path) is None
    assert not raw_path.exists()


def test_current_download_cache_keeps_resumable_raw_result(tmp_path):
    raw_path = tmp_path / "sample_raw.jpg"
    cache_path = tmp_path / "sample_url.json"
    raw_path.write_bytes(b"current-raw")
    cache_path.write_text(
        json.dumps(
            {
                "url": "https://example.invalid/current.jpg",
                "pipeline_version": editor.PIPELINE_VERSION,
            }
        ),
        encoding="utf-8",
    )
    assert editor._load_current_download_cache(cache_path, raw_path).endswith("current.jpg")
    assert raw_path.exists()


def test_sanitize_image_drops_exif(tmp_path):
    from PIL import Image

    source = tmp_path / "source.jpg"
    destination = tmp_path / "public.jpg"
    exif = Image.Exif()
    exif[0x010E] = "private-note"
    Image.new("RGB", (32, 32), "blue").save(source, exif=exif)
    sanitize_image(source, destination)
    with Image.open(destination) as image:
        assert not image.getexif()
