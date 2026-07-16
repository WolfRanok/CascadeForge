"""Split each edited 2x2 grid into cumulative round images."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

from PIL import Image


def split_grid(
    edited_path: Path,
    output_dir: Path,
    source_path: Path | None = None,
    prompt_source: Path | None = None,
    prompt_output_dir: Path | None = None,
):
    digest = edited_path.name.split("_gpt_edited", 1)[0]
    if prompt_source is not None and not prompt_source.exists():
        return False, digest, "缺少对应提示词 JSON，已跳过整个任务"
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
        if prompt_source is not None and prompt_output_dir is not None:
            prompt_output_dir.mkdir(parents=True, exist_ok=True)
            # Published prompt files use a neutral digest-only name.
            shutil.copy2(prompt_source, prompt_output_dir / f"{digest}.json")
        return True, digest, "完成"
    except Exception as exc:
        # Do not leave a partial result directory when its paired prompt failed.
        shutil.rmtree(destination, ignore_errors=True)
        return False, digest, str(exc)


def _sync_prompt_directory(
    prompt_dir: Path, edited_files: list[Path], json_dir: Path
) -> None:
    """Remove stale generated prompt copies without touching source JSON files."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        f"{path.name.split('_gpt_edited', 1)[0]}.json"
        for path in edited_files
        if (json_dir / f"{path.name.split('_gpt_edited', 1)[0]}_JSON_gpt.json").exists()
    }
    # The directory is dedicated to published prompt copies, so all JSON files
    # there are synchronized against the current formal result set.
    for path in prompt_dir.glob("*.json"):
        if path.name not in expected:
            path.unlink(missing_ok=True)


def run_organize(
    input_dir: Path,
    output_dir: Path,
    source_dir: Path | None = None,
    workers: int = 8,
    prompt_dir: Path | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    edited_files = sorted(input_dir.glob("*_gpt_edited.jpg"))
    if not edited_files:
        print(f"[错误] {input_dir} 中没有 *_gpt_edited.jpg")
        return 1
    json_dir = input_dir.parent / "JSON"
    prompt_dir = prompt_dir or output_dir.parent / "提示词"
    _sync_prompt_directory(prompt_dir, edited_files, json_dir)
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = []
        for edited_path in edited_files:
            digest = edited_path.name.split("_gpt_edited", 1)[0]
            source_path = source_dir / f"{digest}.jpg" if source_dir else None
            prompt_source = json_dir / f"{digest}_JSON_gpt.json"
            futures.append(
                pool.submit(
                    split_grid,
                    edited_path,
                    output_dir,
                    source_path,
                    prompt_source,
                    prompt_dir,
                )
            )
        for future in as_completed(futures):
            ok, digest, message = future.result()
            print(f"[{'OK' if ok else 'ERR'}] {digest}: {message}")
            failures += not ok
    return 1 if failures else 0
