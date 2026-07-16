"""Vendor-neutral image editing transport with optional OSS staging."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import numpy as np
from PIL import Image

from .config import AppConfig, load_config

SUCCESS_STATUSES = {"completed", "succeeded", "success"}
FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}
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
    "3:4": 3 / 4,
    "5:4": 5 / 4,
    "4:5": 4 / 5,
    "1:1": 1.0,
}
NORMALIZATION_VERSION = "independent-v2"


def build_prompt_from_json(data: dict[str, Any]) -> str:
    rounds = []
    for index in range(1, 5):
        text = str(data.get(f"ROUND_{index}", {}).get("long", "")).strip()
        # Candidate IDs only exist in the selection contact sheet, not in the edit input.
        text = re.sub(
            r"(?:编号|ID|候选)\s*[0-9一二三四五六七八九十]+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"第[0-9一二三四五六七八九十]+轮", "", text)
        rounds.append(text.strip(" ，,：:。"))
    quadrants = ("左上", "右上", "左下", "右下")
    tasks = "\n".join(
        f"{quadrant}象限任务：只修改 Mask 指定的单个目标——{instruction}。除该目标外，整张象限必须保持原样。"
        for quadrant, instruction in zip(quadrants, rounds)
    )
    return f"""这是同一张原始场景复制成的 2×2 四宫格输入图。四个象限是四个彼此独立的单目标编辑任务，将在同一次请求中生成。

最高优先级硬约束：
- 透明 Mask 是唯一允许修改的区域；Mask 外的每个像素、目标、背景、光影、构图和文字必须保持原样。
- 绝对禁止添加或绘制任何序号、数字、标签、角标、边框、分隔线、水印、说明文字、UI 元素或新的物体。
- 不要把“左上、右上、左下、右下、象限、任务、目标”等控制文字画入图片。
- 不要把一个象限的修改传播到其他象限；不要替其他象限提前执行任务。
- 保持原图的相机视角、透视、比例、轮廓、姿态、位置和背景不变。

{tasks}

输出必须是干净的四宫格图像，不要输出解释，不要在图像上留下任何编辑标记。"""


class TransportError(RuntimeError):
    """Raised when an upload, submit, poll, or download operation fails."""


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120,
    retries: int = 4,
    **kwargs: Any,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TransportError("API 返回不是 JSON 对象")
            return payload
        except (requests.RequestException, ValueError, TransportError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                # Exponential backoff avoids hammering a rate-limited endpoint.
                time.sleep(min(3.0 * (2**attempt), 30.0))
    raise TransportError(f"请求失败：{last_error}") from last_error


def _extract_reference(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract an image reference from the several response shapes used by providers.

    Some deployments wrap results as ``result.data[0]`` and may return image
    bytes as ``b64_json`` instead of a downloadable URL.  Normalising those
    variants here keeps polling and upload handling consistent.
    """
    def walk(node: Any) -> tuple[str | None, str | None]:
        if isinstance(node, dict):
            for key in ("url", "image_url"):
                value = node.get(key)
                if isinstance(value, str) and value:
                    return value, None
            encoded = node.get("b64_json")
            if isinstance(encoded, str) and encoded:
                # Use a data URI so the downstream downloader can persist it.
                return f"data:image/png;base64,{encoded}", None
            for key in ("task_id", "taskId", "id"):
                value = node.get(key)
                if isinstance(value, (str, int)) and value:
                    task = str(value)
                    nested_url, _ = walk(node.get("data"))
                    return nested_url, task
            for value in node.values():
                found_url, found_task = walk(value)
                if found_url or found_task:
                    return found_url, found_task
        elif isinstance(node, list):
            for value in node:
                found_url, found_task = walk(value)
                if found_url or found_task:
                    return found_url, found_task
        return None, None

    return walk(payload)


def _quadrant_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    return (
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height),
    )


def _mask_targets(input_root: Path, digest: str) -> list[np.ndarray]:
    """Load independent masks and convert legacy cumulative masks by differencing."""
    mask_path = input_root / "MASK" / f"{digest}_MASK.png"
    with Image.open(mask_path) as source:
        alpha = np.asarray(source.convert("RGBA").getchannel("A")) < 128
    height, width = alpha.shape
    quadrants = [alpha[y0:y1, x0:x1] for x0, y0, x1, y1 in _quadrant_boxes(width, height)]
    selection_path = input_root / "SELECTION" / f"{digest}_selection.json"
    mode = ""
    if selection_path.exists():
        try:
            mode = json.loads(selection_path.read_text(encoding="utf-8")).get("mask_mode", "")
        except (OSError, json.JSONDecodeError):
            mode = ""
    if mode == NORMALIZATION_VERSION:
        return [mask.astype(bool) for mask in quadrants]

    # Old files stored cumulative masks. A monotonic alpha mask is enough to
    # identify that format, while non-monotonic masks remain independent.
    cumulative = all(
        not np.logical_and(quadrants[index], np.logical_not(quadrants[index + 1])).any()
        for index in range(3)
    )
    if not cumulative:
        return [mask.astype(bool) for mask in quadrants]
    targets: list[np.ndarray] = []
    used = np.zeros_like(quadrants[0], dtype=bool)
    for mask in quadrants:
        current = np.logical_and(mask, np.logical_not(used))
        targets.append(current)
        used = np.logical_or(used, mask)
    return targets


def independent_upload_mask(input_root: Path, digest: str) -> Path:
    """Materialize a v2 mask for the API, including when local data is legacy v1."""
    destination = input_root / ".cascadeforge" / "upload_masks" / f"{digest}_MASK.png"
    targets = _mask_targets(input_root, digest)
    source_path = input_root / "IMAGE_2" / f"{digest}.jpg"
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    height, width = targets[0].shape
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    grid = Image.new("RGBA", (width * 2, height * 2), (0, 0, 0, 255))
    for target, box in zip(targets, _quadrant_boxes(width * 2, height * 2)):
        alpha = np.where(target, 0, 255).astype(np.uint8)
        tile = Image.fromarray(np.dstack((image_array, alpha)).astype(np.uint8), "RGBA")
        grid.paste(tile, (box[0], box[1]))
    buffer = io.BytesIO()
    grid.save(buffer, "PNG", optimize=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(buffer.getvalue())
    return destination


def normalize_grid(
    input_root: Path, digest: str, generated_path: Path, destination_path: Path | None = None
) -> None:
    """Compose one-target generations and restore every protected pixel locally."""
    source_path = input_root / "IMAGE_2" / f"{digest}.jpg"
    if not source_path.exists():
        raise TransportError(f"缺少标准化原图：{source_path}")
    with Image.open(generated_path) as generated_source, Image.open(source_path) as original_source:
        generated = generated_source.convert("RGB")
        original = original_source.convert("RGB")
        width, height = generated.size
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise TransportError(f"编辑结果不是偶数尺寸四宫格：{generated.size}")
        quadrant_size = (width // 2, height // 2)
        original = original.resize(quadrant_size, Image.Resampling.LANCZOS)
        original_array = np.asarray(original).copy()
        generated_array = np.asarray(generated)
        targets = _mask_targets(input_root, digest)
        target_masks = [
            np.asarray(
                Image.fromarray(target.astype(np.uint8) * 255).resize(
                    quadrant_size, Image.Resampling.NEAREST
                )
            )
            > 128
            for target in targets
        ]
        frames: list[np.ndarray] = []
        previous = original_array.copy()
        for target, box in zip(target_masks, _quadrant_boxes(width, height)):
            x0, y0, x1, y1 = box
            model_frame = generated_array[y0:y1, x0:x1]
            current = previous.copy()
            current[target] = model_frame[target]
            frames.append(current)
            previous = current
        normalized = Image.new("RGB", (width, height))
        for frame, box in zip(frames, _quadrant_boxes(width, height)):
            normalized.paste(Image.fromarray(frame, "RGB"), (box[0], box[1]))
        buffer = io.BytesIO()
        normalized.save(buffer, "JPEG", quality=95, subsampling=0)
    (destination_path or generated_path).write_bytes(buffer.getvalue())


def _normalization_sidecar(input_root: Path, digest: str) -> Path:
    return input_root / ".cascadeforge" / "normalized" / f"{digest}.json"


def ensure_normalized(input_root: Path, digest: str, output_path: Path) -> str:
    """Repair an existing output once, while avoiding repeated JPEG recompression."""
    sidecar = _normalization_sidecar(input_root, digest)
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if sidecar.exists():
        try:
            state = json.loads(sidecar.read_text(encoding="utf-8"))
            if state.get("version") == NORMALIZATION_VERSION and state.get("output_sha256") == output_hash:
                return "skip"
        except (OSError, json.JSONDecodeError):
            pass
    raw_backup = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")
    if not raw_backup.exists():
        # Preserve the provider response before repairing legacy outputs in place.
        raw_backup.write_bytes(output_path.read_bytes())
    normalize_grid(input_root, digest, output_path)
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({"version": NORMALIZATION_VERSION, "output_sha256": output_hash}, indent=2),
        encoding="utf-8",
    )
    return "repaired"


def _oss_url(path: Path, config: AppConfig, method: str) -> str:
    oss = config.oss
    if not oss.enabled:
        raise TransportError("OSS 配置不完整，无法上传")
    object_name = f"{oss.path_prefix.rstrip('/')}/{path.name}"
    resource = f"/{oss.bucket_name}/{object_name}"
    expires = int(time.time()) + oss.sign_expires
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    canonical = f"{method}\n\n{content_type if method == 'PUT' else ''}\n{expires}\n{resource}"
    signature = base64.b64encode(
        hmac.new(oss.access_key_secret.encode(), canonical.encode(), hashlib.sha1).digest()
    ).decode()
    endpoint = oss.endpoint.replace("https://", f"https://{oss.bucket_name}.")
    return (
        f"{endpoint}/{object_name}?OSSAccessKeyId={quote(oss.access_key_id)}"
        f"&Expires={expires}&Signature={quote(signature)}"
    )


def _upload(path: Path, config: AppConfig) -> str:
    if config.oss.enabled:
        upload_url = _oss_url(path, config, "PUT")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as stream:
            response = requests.put(
                upload_url, data=stream, headers={"Content-Type": content_type}, timeout=300
            )
        response.raise_for_status()
        # The PUT signature cannot be reused by the image API, so return a GET signature.
        return _oss_url(path, config, "GET")
    if not config.edit.api_key:
        raise TransportError("未配置 TOAPIS_API_KEY，且 OSS 配置不可用")
    with path.open("rb") as stream:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        response = requests.post(
            config.edit.upload_url,
            headers={"Authorization": f"Bearer {config.edit.api_key}"},
            files={"file": (path.name, stream, content_type)},
            timeout=120,
        )
    response.raise_for_status()
    url, _ = _extract_reference(response.json())
    if not url:
        raise TransportError("上传 API 未返回图片 URL")
    return url


def _download(url: str, output: Path) -> None:
    if url.startswith("data:") and ";base64," in url:
        # Providers can return b64_json for completed tasks; decode locally.
        encoded = url.split(";base64,", 1)[1]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(encoded))
        return
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)


def _poll(task_id: str, config: AppConfig) -> str:
    if not config.edit.api_key:
        raise TransportError("未配置 TOAPIS_API_KEY")
    status_url = f"{config.edit.base_url.rstrip('/')}/v1/images/generations/{task_id}"
    headers = {"Authorization": f"Bearer {config.edit.api_key}"}
    started = time.monotonic()
    while time.monotonic() - started < 600:
        payload = _request_json("GET", status_url, headers=headers, timeout=120)
        status = str(payload.get("status", "")).lower()
        if status in SUCCESS_STATUSES:
            url, _ = _extract_reference(payload)
            if url:
                return url
            raise TransportError("任务成功但没有返回图片 URL")
        if status in FAILED_STATUSES:
            raise TransportError(f"编辑任务失败：{payload}")
        time.sleep(5)
    raise TransportError("编辑任务轮询超时")


def process_one(digest: str, input_root: Path, config: AppConfig, model: str = "gpt") -> dict[str, Any]:
    json_path = input_root / "JSON" / f"{digest}_JSON_gpt.json"
    image_path = input_root / "IMAGE_2X4" / f"{digest}_IMAGE.jpg"
    mask_path = input_root / "MASK" / f"{digest}_MASK.png"
    output_path = input_root / "EDITED_4K" / f"{digest}_{model}_edited.jpg"
    raw_path = input_root / "EDITED_4K" / f"{digest}_{model}_edited_raw.jpg"
    download_json = input_root / "DOWNLOAD_JSON" / f"{digest}_{model}_url.json"
    result: dict[str, Any] = {"md5": digest, "model": model, "status": "error"}
    if output_path.exists():
        try:
            result["status"] = ensure_normalized(input_root, digest, output_path)
            return result
        except Exception as exc:
            result["error"] = f"已有结果归一化失败：{exc}"
            return result
    if not json_path.exists() or not image_path.exists() or not mask_path.exists():
        result["error"] = "缺少提示词 JSON、四宫格原图或编辑 Mask"
        return result
    try:
        if raw_path.exists():
            normalize_grid(input_root, digest, raw_path, output_path)
            output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
            sidecar = _normalization_sidecar(input_root, digest)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(
                json.dumps(
                    {"version": NORMALIZATION_VERSION, "output_sha256": output_hash}, indent=2
                ),
                encoding="utf-8",
            )
            result["status"] = "success"
            result["output"] = str(output_path)
            return result
        # Reuse a saved URL when a previous run completed the remote task.
        if download_json.exists():
            cached = json.loads(download_json.read_text(encoding="utf-8"))
            url = cached.get("url")
        else:
            url = None
        if not url:
            image_url = _upload(image_path, config)
            mask_url = _upload(independent_upload_mask(input_root, digest), config)
            prompt = build_prompt_from_json(json.loads(json_path.read_text(encoding="utf-8")))
            with Image.open(image_path) as source:
                width, height = source.size
            ratio = min(SUPPORTED_RATIOS, key=lambda name: abs(width / height - SUPPORTED_RATIOS[name]))
            headers = {
                "Authorization": f"Bearer {config.edit.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config.edit.model,
                "prompt": prompt,
                "size": ratio,
                "resolution": config.edit.resolution,
                "response_format": config.edit.response_format,
                "n": 1,
                "image_urls": [image_url],
                "mask_url": mask_url,
            }
            response = _request_json("POST", config.edit.api_url, headers=headers, json=payload)
            url, task_id = _extract_reference(response)
            if not url and task_id:
                url = _poll(str(task_id), config)
            if not url:
                raise TransportError("编辑 API 未返回 URL 或任务 ID")
            download_json.parent.mkdir(parents=True, exist_ok=True)
            download_json.write_text(
                json.dumps({"url": url, "md5": digest, "model": model}, indent=2), encoding="utf-8"
            )
        if not raw_path.exists():
            _download(url, raw_path)
        # Keep the raw provider response locally, but expose only the
        # deterministic, Mask-constrained result under the old output name.
        normalize_grid(input_root, digest, raw_path, output_path)
        raw_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
        sidecar = _normalization_sidecar(input_root, digest)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps({"version": NORMALIZATION_VERSION, "output_sha256": raw_hash}, indent=2),
            encoding="utf-8",
        )
        result["status"] = "success"
        result["output"] = str(output_path)
    except Exception as exc:
        result["error"] = str(exc)[:500]
    return result


def run_editor(input_root: Path, config_path: Path | None, concurrency: int = 4, watch: bool = False) -> int:
    config = load_config(config_path)
    while True:
        json_dir = input_root / "JSON"
        all_digests = sorted(path.name.removesuffix("_JSON_gpt.json") for path in json_dir.glob("*_JSON_gpt.json"))
        results_dir = input_root / "EDITED_4K"
        results_dir.mkdir(parents=True, exist_ok=True)
        # Existing provider outputs are repaired locally before deciding
        # whether a credentialed API request is needed.
        existing = [digest for digest in all_digests if (results_dir / f"{digest}_gpt_edited.jpg").exists()]
        for digest in existing:
            repaired = process_one(digest, input_root, config)
            if repaired["status"] == "repaired":
                print(f"[REPAIRED] {digest}: 已按 Mask 重新合成")
            elif repaired["status"] == "error":
                print(f"[ERROR] {digest}: {repaired.get('error', '')}")
        pending = [digest for digest in all_digests if digest not in existing]
        recoverable = [
            digest
            for digest in pending
            if (results_dir / f"{digest}_gpt_edited_raw.jpg").exists()
        ]
        for digest in recoverable:
            recovered = process_one(digest, input_root, config)
            print(f"[{recovered['status'].upper()}] {digest}: 从本地原始返回图恢复")
        pending = [digest for digest in pending if digest not in recoverable]
        if not pending:
            if watch:
                print("[INFO] 暂无待处理任务，20 秒后重试；Ctrl+C 退出")
                time.sleep(20)
                continue
            print("[INFO] 没有待处理任务")
            return 0
        if not config.edit.api_key and not config.oss.enabled:
            print("[错误] 请配置 TOAPIS_API_KEY，或提供完整 OSS 配置")
            return 2
        failures = 0
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for result in pool.map(lambda digest: process_one(digest, input_root, config), pending):
                print(f"[{result['status'].upper()}] {result['md5']}: {result.get('error', '')}")
                failures += result["status"] == "error"
        if not watch:
            return 1 if failures else 0
