"""Helpers for preparing small public examples without leaking local metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def sanitize_image(source: Path, destination: Path, max_side: int = 1600) -> None:
    """Remove EXIF and optionally resize a public image; visual redaction is manual."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Re-encoding from RGB drops EXIF/GPS blocks carried by the source file.
        image.save(destination, "JPEG", quality=92, exif=b"")


def sanitize_metadata(source: Path, destination: Path, public_name: str) -> dict[str, Any]:
    """Keep model-relevant fields while replacing paths and source identifiers."""
    data = json.loads(source.read_text(encoding="utf-8"))
    image = data.get("image", {})
    public = {
        "image": {
            "image_id": public_name,
            "file_name": f"{public_name}.jpg",
            "width": image.get("width"),
            "height": image.get("height"),
        },
        "annotations": data.get("annotations", []),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    return public
