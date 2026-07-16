"""Split each edited 2x2 grid into cumulative round images."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def split_grid(edited_path: Path, output_dir: Path, source_path: Path | None = None):
    digest = edited_path.name.split("_gpt_edited", 1)[0]
    try:
        with Image.open(edited_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if width < 2 or height < 2 or width % 2 or height % 2:
            return False, digest, f"不是偶数尺寸的 2x2 四宫格：{image.size}"
        destination = output_dir / digest
        destination.mkdir(parents=True, exist_ok=True)
        boxes = {
            "ROUND_1.jpg": (0, 0, width // 2, height // 2),
            "ROUND_2.jpg": (width // 2, 0, width, height // 2),
            "ROUND_3.jpg": (0, height // 2, width // 2, height),
            "ROUND_4.jpg": (width // 2, height // 2, width, height),
        }
        for name, box in boxes.items():
            image.crop(box).save(destination / name, "JPEG", quality=95)
        if source_path and source_path.exists():
            with Image.open(source_path) as original:
                original.convert("RGB").save(destination / "original.jpg", "JPEG", quality=95)
        return True, digest, "完成"
    except Exception as exc:
        return False, digest, str(exc)


def run_organize(
    input_dir: Path, output_dir: Path, source_dir: Path | None = None, workers: int = 8
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    edited_files = sorted(input_dir.glob("*_gpt_edited.jpg"))
    if not edited_files:
        print(f"[错误] {input_dir} 中没有 *_gpt_edited.jpg")
        return 1
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = []
        for edited_path in edited_files:
            digest = edited_path.name.split("_gpt_edited", 1)[0]
            source_path = source_dir / f"{digest}.jpg" if source_dir else None
            futures.append(pool.submit(split_grid, edited_path, output_dir, source_path))
        for future in as_completed(futures):
            ok, digest, message = future.result()
            print(f"[{'OK' if ok else 'ERR'}] {digest}: {message}")
            failures += not ok
    return 1 if failures else 0
