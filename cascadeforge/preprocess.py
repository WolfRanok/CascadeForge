"""Convert SA-1B-style annotations into editable object candidates."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

MAX_MASK_SIZE = 6 * 1024 * 1024
MIN_SHORT_EDGE = 1024
MAX_CROP_RATE = 0.05
SUPPORTED_RATIOS = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "2:1": 2.0,
    "1:2": 0.5,
    "21:9": 21 / 9,
    "9:21": 9 / 21,
    "3:2": 1.5,
    "2:3": 2 / 3,
    "4:3": 4 / 3,
    "3:4": 0.75,
    "5:4": 1.25,
    "4:5": 0.8,
    "1:1": 1.0,
}


def crop_size(width: int, height: int) -> tuple[int, int, str] | None:
    ratio = width / height
    best: tuple[float, int, int, str] | None = None
    for name, target in SUPPORTED_RATIOS.items():
        if abs(ratio - target) / target <= 0.001:
            return width - width % 2, height - height % 2, name
        new_width, new_height = (
            (int(height * target), height) if ratio > target else (width, int(width / target))
        )
        new_width -= new_width % 2
        new_height -= new_height % 2
        if min(new_width, new_height) < MIN_SHORT_EDGE:
            continue
        crop_rate = (width * height - new_width * new_height) / (width * height)
        if crop_rate <= MAX_CROP_RATE and (best is None or crop_rate < best[0]):
            best = (crop_rate, new_width, new_height, name)
    return None if best is None else (best[1], best[2], best[3])


def decode_rle(segmentation: dict) -> np.ndarray:
    if not isinstance(segmentation, dict) or not {"counts", "size"} <= segmentation.keys():
        raise ValueError("无效的 COCO RLE")
    rle = {"size": list(segmentation["size"]), "counts": segmentation["counts"]}
    array = mask_utils.decode(rle)
    if array.ndim == 3:
        array = array[..., 0]
    return (array > 0).astype(np.uint8)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def intersection_over_min(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    return float(intersection / max(1, min(left.sum(), right.sum())))


def candidate_score(annotation: dict, width: int, height: int) -> float:
    x, y, box_width, box_height = annotation["bbox"]
    area_ratio = annotation["area"] / (width * height)
    touches = sum(
        (x <= 1, y <= 1, x + box_width >= width - 2, y + box_height >= height - 2)
    )
    quality = min(1.0, float(annotation.get("predicted_iou", 0.0))) * min(
        1.0, float(annotation.get("stability_score", 0.0))
    )
    return area_ratio * quality * (0.35 if touches >= 2 else 1.0)


def select_candidates(
    annotations: list[dict], masks: list[np.ndarray], width: int, height: int
) -> list[dict]:
    items: list[dict] = []
    for annotation_index, (annotation, mask) in enumerate(zip(annotations, masks)):
        area = int(mask.sum())
        if area <= 0:
            continue
        ys, xs = np.where(mask > 0)
        x, y = int(xs.min()), int(ys.min())
        box_width, box_height = int(xs.max() - x + 1), int(ys.max() - y + 1)
        area_ratio = area / (width * height)
        touches = sum(
            (x <= 1, y <= 1, x + box_width >= width - 2, y + box_height >= height - 2)
        )
        if area_ratio < 0.002 or area_ratio > 0.45 or touches >= 3:
            continue
        adjusted = {**annotation, "area": area, "bbox": [x, y, box_width, box_height]}
        small_height = max(1, int(256 * height / width))
        small_mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (256, small_height), Image.Resampling.NEAREST
            )
        ) > 128
        items.append(
            {
                "candidate_id": len(items),
                "annotation_index": annotation_index,
                "annotation_id": annotation.get("id"),
                "area": area,
                "bbox": [x, y, box_width, box_height],
                "score": candidate_score(adjusted, width, height),
                "mask": mask,
                "small_mask": small_mask,
            }
        )

    items.sort(key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    for item in items:
        # Reject nested or near-duplicate masks so the VLM sees distinct objects.
        if any(
            iou(item["small_mask"], old["small_mask"]) >= 0.65
            or intersection_over_min(item["small_mask"], old["small_mask"]) >= 0.85
            for old in kept
        ):
            continue
        kept.append(item)
        if len(kept) >= 32:
            break
    for index, item in enumerate(kept):
        item["candidate_id"] = index
    return kept


def make_contact_sheet(image: Image.Image, candidates: list[dict]) -> Image.Image:
    thumb_width, thumb_height, columns = 256, 256, 4
    rows = max(1, (len(candidates) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 24)), "#202020")
    draw = ImageDraw.Draw(sheet)
    image_array = np.asarray(image)
    for index, candidate in enumerate(candidates):
        mask = candidate["mask"].astype(bool)
        ys, xs = np.where(mask)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            crop = image_array[y0:y1, x0:x1, :3]
            crop_mask = mask[y0:y1, x0:x1]
            tile = Image.fromarray(np.where(crop_mask[..., None], crop, 255).astype(np.uint8))
            tile.thumbnail((thumb_width - 8, thumb_height - 8), Image.Resampling.LANCZOS)
        else:
            tile = Image.new("RGB", (1, 1), "white")
        x = (index % columns) * thumb_width + (thumb_width - tile.width) // 2
        y = (index // columns) * (thumb_height + 24) + (thumb_height - tile.height) // 2
        sheet.paste(tile, (x, y))
        draw.text(
            ((index % columns) * thumb_width + 8, (index // columns) * (thumb_height + 24) + thumb_height),
            f"ID {candidate['candidate_id']}",
            fill="white",
        )
    return sheet


def process_image(path: Path, output_dir: Path) -> tuple[str, str, str]:
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return path.stem, "skip", "缺少同名 JSON"
    image_dir = output_dir / "IMAGE_2"
    grid_dir = output_dir / "IMAGE_2X4"
    candidate_dir = output_dir / "CANDIDATES"
    try:
        digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        expected = [
            candidate_dir / f"{digest}_meta.json",
            candidate_dir / f"{digest}_masks.npz",
            image_dir / f"{digest}.jpg",
            grid_dir / f"{digest}_IMAGE.jpg",
        ]
        if all(item.exists() for item in expected):
            return digest, "skip", "已完成预处理"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        target = crop_size(width, height)
        if target is None:
            return path.stem, "skip", "5% 裁剪范围内没有支持的宽高比"
        new_width, new_height, ratio = target
        left, top = (width - new_width) // 2, (height - new_height) // 2
        image = image.crop((left, top, left + new_width, top + new_height))

        annotations: list[dict] = []
        masks: list[np.ndarray] = []
        for annotation in data.get("annotations", []):
            # Filter inexpensive metadata before decoding large RLE arrays.
            area_ratio = float(annotation.get("area", 0)) / max(1, width * height)
            x, y, box_width, box_height = annotation.get("bbox", [0, 0, 0, 0])
            touches = sum(
                (x <= 1, y <= 1, x + box_width >= width - 2, y + box_height >= height - 2)
            )
            if area_ratio < 0.0015 or area_ratio > 0.50 or touches >= 3:
                continue
            if float(annotation.get("predicted_iou", 1.0)) < 0.90:
                continue
            if float(annotation.get("stability_score", 1.0)) < 0.95:
                continue
            mask = decode_rle(annotation.get("segmentation"))
            if mask.shape != (height, width):
                continue
            annotations.append(annotation)
            masks.append(mask[top : top + new_height, left : left + new_width])

        candidates = select_candidates(annotations, masks, new_width, new_height)
        if len(candidates) < 3:
            return path.stem, "skip", f"只有 {len(candidates)} 个有效候选"

        # Estimate the cumulative mask payload before creating all output files.
        combined = np.logical_or.reduce([candidate["mask"] for candidate in candidates])
        alpha = (1 - combined.astype(np.uint8)) * 255
        rgba = np.dstack((np.asarray(image), alpha)).astype(np.uint8)
        mask_test = Image.new("RGBA", (new_width * 2, new_height * 2), (0, 0, 0, 255))
        tile = Image.fromarray(rgba, "RGBA")
        for position in ((0, 0), (new_width, 0), (0, new_height), (new_width, new_height)):
            mask_test.paste(tile, position)
        buffer = io.BytesIO()
        mask_test.save(buffer, format="PNG", optimize=True)
        if buffer.tell() > MAX_MASK_SIZE and min(new_width, new_height) > MIN_SHORT_EDGE:
            scale = MIN_SHORT_EDGE / min(new_width, new_height)
            final_width, final_height = int(new_width * scale) & ~1, int(new_height * scale) & ~1
            image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
            for candidate in candidates:
                candidate["mask"] = np.asarray(
                    Image.fromarray(candidate["mask"].astype(np.uint8) * 255).resize(
                        (final_width, final_height), Image.Resampling.NEAREST
                    )
                ) > 128
            new_width, new_height = final_width, final_height

        for directory in (image_dir, grid_dir, candidate_dir):
            directory.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / f"{digest}.jpg", "JPEG", quality=95)
        grid = Image.new("RGB", (new_width * 2, new_height * 2))
        for position in ((0, 0), (new_width, 0), (0, new_height), (new_width, new_height)):
            grid.paste(image, position)
        grid.save(grid_dir / f"{digest}_IMAGE.jpg", "JPEG", quality=95)
        make_contact_sheet(image, candidates).save(
            candidate_dir / f"{digest}_candidates.jpg", "JPEG", quality=90
        )
        np.savez_compressed(
            candidate_dir / f"{digest}_masks.npz",
            **{str(item["candidate_id"]): item["mask"].astype(np.uint8) for item in candidates},
        )
        public_candidates = [
            {key: value for key, value in item.items() if key not in {"mask", "small_mask"}}
            for item in candidates
        ]
        # Store only a basename to avoid leaking the contributor's local filesystem path.
        metadata = {
            "md5": digest,
            "source": path.name,
            "ratio": ratio,
            "size": [new_width, new_height],
            "candidates": public_candidates,
        }
        (candidate_dir / f"{digest}_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return digest, "ok", f"{new_width}x{new_height}，{len(candidates)} 个候选"
    except Exception as exc:
        return path.stem, "error", str(exc)[:300]


def run_preprocess(input_dir: Path, output_dir: Path, workers: int = 4) -> int:
    files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    if not files:
        print(f"[错误] 没有在 {input_dir} 中找到 JPG 图片")
        return 1
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(process_image, path, output_dir) for path in files]
        for future in as_completed(futures):
            key, status, message = future.result()
            print(f"[{status.upper()}] {key}: {message}")
            failures += status == "error"
    return 1 if failures else 0
