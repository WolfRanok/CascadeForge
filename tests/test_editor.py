from pathlib import Path

import pytest

from cascadeforge import editor
from cascadeforge.sanitize import sanitize_image


class FakeResponse:
    def __init__(self, payload, fail=False, status_code=200):
        self.payload = payload
        self.fail = fail
        self.status_code = status_code
        self.text = ""

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
    assert "复制左上结果，仅新增 edit-2" in prompt
    assert "复制右上结果，仅新增 edit-3" in prompt
    assert "变化必须明显可见" in prompt
    assert "高对比颜色、明显材质或清晰图案" in prompt
    assert "整图天气、昼夜、季节" in prompt
    assert "禁止编号、文字" in prompt
    assert "跨象限修改" in prompt
    assert "硬性要求" not in prompt


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
