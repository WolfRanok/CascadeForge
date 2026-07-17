"""Use a vision-language model to select four independent edit targets."""

from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw

from .config import load_config

MAX_ATTEMPTS = 3
MAX_PAIR_IOU = 0.15
MAX_INTERSECTION_OVER_MIN = 0.30
MAX_MASK_SIZE = 6 * 1024 * 1024
MIN_CENTER_DISTANCE_RATIO = 0.12
# Four independent target masks remove cumulative reasoning from the edit model.
SELECTION_MODE = "four-target-independent-v2"

PROMPT = """你会看到两张图：第一张是原始场景，第二张是自动分割候选物体的编号白底图。
请选择恰好四个完整、独立、适合视觉编辑且空间距离较远的实体。

要求：
1. 只能返回候选图中实际存在的四个不同 ID。
2. 不选择天空、地面、水面、墙体、整片人群等背景或群组。
3. 四个目标不得重叠或相邻，应尽量分布在画面的不同位置，便于准确定位。
4. ROUND_1 至 ROUND_4 各编辑一个新目标。修改必须明显、高对比、容易识别。
5. 先准确描述目标在原图中的当前颜色、材质、尺寸、纹理或类别，再提出明显不同的新状态。
6. 禁止无效编辑，例如“蓝色物体改成蓝色”“白花改成白花”；新状态必须不同。
7. 四轮 short 均不超过 10 个汉字，long 均为 15–30 个汉字。
8. long 使用“位置和当前外观明确的目标，清晰变化”的单句，不写候选编号。
9. 优先使用颜色、材质、尺寸、纹理或类别变化，四轮尽量采用不同变化类型。
10. 禁止火焰、着火、轻微改变、略微调整、适当装饰、整体天气或整体色调变化。

合格示例：
- 左上部的白色蘑菇，尺寸明显变大一倍
- 画面中间的鲜绿色叶片，变成红色枫叶
- 右下角的棕色枯叶，变成半透明蓝色玻璃叶片
- 右上部的灰色石块，变成明亮黄色陶瓷石块

严格输出 JSON，不要输出解释或 Markdown：
{
  "selected_ids": [1, 2, 3, 4],
  "ROUND_1": {"short": "...", "long": "..."},
  "ROUND_2": {"short": "...", "long": "..."},
  "ROUND_3": {"short": "...", "long": "..."},
  "ROUND_4": {"short": "...", "long": "..."}
}
"""


def image_data_url(path: Path, max_side: int = 1536, quality: int = 88) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS
            )
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return (
        float(intersection / max(1, union)),
        float(intersection / max(1, min(left.sum(), right.sum()))),
    )


def center_distance_ratio(left: np.ndarray, right: np.ndarray) -> float:
    """Return centroid distance normalized by the image diagonal."""
    left_y, left_x = np.where(left)
    right_y, right_x = np.where(right)
    if not len(left_x) or not len(right_x):
        return 0.0
    distance = np.hypot(left_x.mean() - right_x.mean(), left_y.mean() - right_y.mean())
    diagonal = np.hypot(left.shape[1], left.shape[0])
    return float(distance / max(1.0, diagonal))


def validate_response(data: dict, masks: np.lib.npyio.NpzFile, candidate_meta: list[dict]):
    ids = data.get("selected_ids")
    if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("selected_ids 必须包含四个不同 ID")
    ids = [str(int(value)) for value in ids]
    if any(candidate_id not in masks.files for candidate_id in ids):
        raise ValueError("模型选择了不存在的候选 ID")
    selected_masks = [masks[candidate_id].astype(bool) for candidate_id in ids]
    for left_index in range(4):
        for right_index in range(left_index + 1, 4):
            pair_iou, iom = overlap(selected_masks[left_index], selected_masks[right_index])
            if pair_iou > MAX_PAIR_IOU or iom > MAX_INTERSECTION_OVER_MIN:
                raise ValueError(f"候选 {ids[left_index]} 与 {ids[right_index]} 重叠过多")
            distance = center_distance_ratio(
                selected_masks[left_index], selected_masks[right_index]
            )
            if distance < MIN_CENTER_DISTANCE_RATIO:
                raise ValueError(
                    f"候选 {ids[left_index]} 与 {ids[right_index]} 距离过近"
                )
    for round_index in range(1, 5):
        value = data.get(f"ROUND_{round_index}")
        if not isinstance(value, dict) or not all(
            isinstance(value.get(key), str) and value[key].strip() for key in ("short", "long")
        ):
            raise ValueError(f"ROUND_{round_index} 格式无效")

    # Larger targets are edited first to make cumulative changes easier to preserve.
    area_by_id = {str(item["candidate_id"]): int(item["area"]) for item in candidate_meta}
    ordered = sorted(ids, key=lambda candidate_id: area_by_id[candidate_id], reverse=True)
    original_rounds = {ids[index]: data[f"ROUND_{index + 1}"] for index in range(4)}
    rounds = {
        f"ROUND_{index + 1}": original_rounds[candidate_id]
        for index, candidate_id in enumerate(ordered)
    }
    return ordered, rounds


def make_outputs(
    output_root: Path,
    digest: str,
    selected_ids: list[str],
    rounds: dict,
    metadata: dict,
    masks: np.lib.npyio.NpzFile,
) -> None:
    image_path = output_root / "IMAGE_2" / f"{digest}.jpg"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    array = np.asarray(image)
    selected = [masks[candidate_id].astype(bool) for candidate_id in selected_ids]
    edit_regions = selected

    mask_grid = Image.new("RGBA", (image.width * 2, image.height * 2), (0, 0, 0, 255))
    positions = ((0, 0), (image.width, 0), (0, image.height), (image.width, image.height))
    for region, position in zip(edit_regions, positions):
        alpha = np.where(region, 0, 255).astype(np.uint8)
        mask_grid.paste(Image.fromarray(np.dstack((array, alpha)).astype(np.uint8), "RGBA"), position)
    buffer = io.BytesIO()
    mask_grid.save(buffer, "PNG", optimize=True)
    if buffer.tell() > MAX_MASK_SIZE:
        raise ValueError(f"独立 Mask 为 {buffer.tell() / 1024 / 1024:.1f} MB，超过 6 MB")

    object_grid = Image.new("RGB", (image.width * 2, image.height * 2), "white")
    for index, (mask, position) in enumerate(zip(selected, positions), 1):
        tile = Image.fromarray(np.where(mask[..., None], array, 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((8, 8, 82, 48), fill="black")
        draw.text((18, 15), f"#{index}", fill="white")
        object_grid.paste(tile, position)

    json_dir, mask_dir = output_root / "JSON", output_root / "MASK"
    object_dir, selection_dir = output_root / "OBJECT", output_root / "SELECTION"
    for directory in (json_dir, mask_dir, object_dir, selection_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (json_dir / f"{digest}_JSON_gpt.json").write_text(
        json.dumps(rounds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (mask_dir / f"{digest}_MASK.png").write_bytes(buffer.getvalue())
    object_grid.save(object_dir / f"{digest}_OBJECT.jpg", "JPEG", quality=92)
    candidate_map = {str(item["candidate_id"]): item for item in metadata["candidates"]}
    selection = {
        "md5": digest,
        "selection_mode": SELECTION_MODE,
        "selected_ids": [int(value) for value in selected_ids],
        "selected": [candidate_map[value] for value in selected_ids],
        "rounds": rounds,
    }
    (selection_dir / f"{digest}_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def is_current_selection(path: Path) -> bool:
    """Only the current independent-mask selection is safe to reuse."""
    if not path.exists():
        return False
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return selection.get("selection_mode") == SELECTION_MODE


def process_one(meta_path: Path, output_root: Path, client: OpenAI, model: str):
    digest = meta_path.name.removesuffix("_meta.json")
    json_path = output_root / "JSON" / f"{digest}_JSON_gpt.json"
    mask_path = output_root / "MASK" / f"{digest}_MASK.png"
    selection_path = output_root / "SELECTION" / f"{digest}_selection.json"
    # Do not detach an existing final image from the metadata that produced it.
    if (output_root / "EDITED_4K" / f"{digest}_gpt_edited.jpg").exists():
        return True, digest, "跳过已有正式结果"
    if json_path.exists() and mask_path.exists() and is_current_selection(selection_path):
        return True, digest, "跳过已有结果"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        masks = np.load(output_root / "CANDIDATES" / f"{digest}_masks.npz")
        error_hint = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = PROMPT + (f"\n上一次输出无效：{error_hint}。请重新选择。" if error_hint else "")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url(output_root / "IMAGE_2" / f"{digest}.jpg")
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url(
                                        output_root / "CANDIDATES" / f"{digest}_candidates.jpg", 2048
                                    )
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            try:
                data = json.loads(response.choices[0].message.content)
                selected, rounds = validate_response(data, masks, metadata["candidates"])
                make_outputs(output_root, digest, selected, rounds, metadata, masks)
                return True, digest, f"成功（第 {attempt} 次尝试）"
            except Exception as exc:
                error_hint = str(exc)
        return False, digest, f"连续 {MAX_ATTEMPTS} 次选择无效：{error_hint}"
    except Exception as exc:
        return False, digest, str(exc)[:300]


def pending_metadata(output_root: Path, attempted: set[str]) -> list[Path]:
    """Find new, incomplete metadata files that were not attempted in this run."""
    pending: list[Path] = []
    for meta_path in sorted((output_root / "CANDIDATES").glob("*_meta.json")):
        digest = meta_path.name.removesuffix("_meta.json")
        if digest in attempted:
            continue
        json_path = output_root / "JSON" / f"{digest}_JSON_gpt.json"
        mask_path = output_root / "MASK" / f"{digest}_MASK.png"
        selection_path = output_root / "SELECTION" / f"{digest}_selection.json"
        if json_path.exists() and mask_path.exists() and is_current_selection(selection_path):
            continue
        # Completed legacy outputs remain paired with their original metadata.
        if (output_root / "EDITED_4K" / f"{digest}_gpt_edited.jpg").exists():
            continue
        pending.append(meta_path)
    return pending


def run_selection(output_root: Path, config_path: Path | None, concurrency: int = 8) -> int:
    config = load_config(config_path)
    if not config.vision.api_key:
        print("[错误] 请设置 GPT_API_KEY，或在本地配置文件中提供 moliapi.api_key")
        return 2
    candidate_dir = output_root / "CANDIDATES"
    metadata_files = sorted(candidate_dir.glob("*_meta.json"))
    if not metadata_files:
        print(f"[错误] {candidate_dir} 中没有候选元数据")
        return 1
    client = OpenAI(api_key=config.vision.api_key, base_url=config.vision.base_url)
    attempted: set[str] = set()
    total_successes = 0
    total_failures = 0
    round_number = 0
    idle_scans = 0

    while True:
        pending = pending_metadata(output_root, attempted)
        if not pending:
            idle_scans += 1
            if idle_scans >= 2:
                print(
                    f"[完成] 共处理 {round_number} 轮：成功 {total_successes}，"
                    f"失败 {total_failures}"
                )
                return 1 if total_failures else 0
            print("[INFO] 暂无新增候选，3 秒后再次确认")
            time.sleep(3)
            continue

        idle_scans = 0
        round_number += 1
        # Mark before dispatch so a failed item cannot loop forever in this run.
        attempted.update(path.name.removesuffix("_meta.json") for path in pending)
        print(f"[INFO] 第 {round_number} 轮发现 {len(pending)} 个待处理任务")
        round_successes = 0
        round_failures = 0
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = pool.map(
                lambda path: process_one(path, output_root, client, config.vision.model), pending
            )
            for ok, digest, message in results:
                print(f"[{'成功' if ok else '错误'}] {digest}: {message}")
                round_successes += bool(ok)
                round_failures += not ok
        total_successes += round_successes
        total_failures += round_failures
        print(
            f"[INFO] 第 {round_number} 轮完成：成功 {round_successes}，失败 {round_failures}；"
            "正在检查新增内容"
        )
