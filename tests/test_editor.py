from pathlib import Path

import pytest

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


def test_build_prompt_contains_all_rounds():
    data = {f"ROUND_{index}": {"long": f"edit-{index}"} for index in range(1, 5)}
    prompt = editor.build_prompt_from_json(data)
    assert all(f"edit-{index}" in prompt for index in range(1, 5))


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
