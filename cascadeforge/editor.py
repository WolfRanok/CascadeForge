"""Vendor-neutral image editing transport with optional OSS staging."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import mimetypes
import os
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
PIPELINE_VERSION = "independent-global-v1"
MIN_MEAN_DIFFERENCE = 18.0
MIN_CHANGED_RATIO = 0.25
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


def build_prompt_from_json(data: dict[str, Any]) -> str:
    rounds = [data.get(f"ROUND_{index}", {}).get("long", "") for index in range(1, 5)]
    return f"""这是同一张原图组成的 2×2 四宫格。四格是独立任务。

左上：只编辑透明 Mask 内的物体：{rounds[0]}
右上：只编辑透明 Mask 内的物体：{rounds[1]}
左下：只编辑透明 Mask 内的物体：{rounds[2]}
右下：执行整图变换：{rounds[3]}

规则：
1. 前三格互不依赖，每格只完成自己的一项编辑。
2. 透明 Mask 是唯一目标位置；忽略指令中可能不准确的位置词。
3. Mask 内变化必须明显，Mask 外保持原图。
4. 保持物体轮廓、姿态、位置和数量不变。
5. 右下可改变整图天气、昼夜、季节、光照、氛围和色调。
6. 禁止编号、文字、标签、边框、水印、UI 和跨象限修改。"""


class TransportError(RuntimeError):
    """Raised when an upload, submit, poll, or download operation fails."""


def _http_error(response: requests.Response) -> TransportError:
    """Convert provider errors into actionable messages without exposing secrets."""
    code = ""
    message = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            code = str(payload.get("code", ""))
            message = str(payload.get("message", ""))
            error = payload.get("error")
            if isinstance(error, dict):
                code = code or str(error.get("code", ""))
                message = message or str(error.get("message", ""))
            elif isinstance(error, str):
                message = message or error
    except ValueError:
        message = response.text[:300].strip()
    if code == "quota_not_enough" or "quota" in message.lower():
        return TransportError("ToAPIs 账户额度不足，请充值或补充额度后重试")
    detail = f"：{message}" if message else ""
    return TransportError(f"API 请求被拒绝（HTTP {response.status_code}）{detail}")


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
            if response.status_code >= 400:
                error = _http_error(response)
                # Authentication, permission, quota, and invalid requests should
                # not be repeated; retries cannot change these provider decisions.
                if response.status_code < 500 and response.status_code != 429:
                    raise error
                raise requests.HTTPError(str(error), response=response)
            payload = response.json()
            if not isinstance(payload, dict):
                raise TransportError("API 返回不是 JSON 对象")
            return payload
        except TransportError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                # Exponential backoff avoids hammering a rate-limited endpoint.
                time.sleep(min(3.0 * (2**attempt), 30.0))
    raise TransportError(f"请求失败：{last_error}") from last_error


def _extract_image_url(node: Any) -> str | None:
    """Search the complete response for an image URL before reading task IDs."""
    if isinstance(node, dict):
        for key in ("url", "image_url"):
            value = node.get(key)
            if isinstance(value, str) and value:
                return value
        for value in node.values():
            url = _extract_image_url(value)
            if url:
                return url
    elif isinstance(node, list):
        for value in node:
            url = _extract_image_url(value)
            if url:
                return url
    return None


def _extract_task_id(payload: dict[str, Any]) -> str | None:
    """Read the task ID independently so it cannot hide a nested result URL."""
    for key in ("task_id", "taskId", "id"):
        value = payload.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    for value in payload.values():
        if isinstance(value, dict):
            task_id = _extract_task_id(value)
            if task_id:
                return task_id
    return None


def _extract_reference(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return both fields after independently inspecting the whole response."""
    return _extract_image_url(payload), _extract_task_id(payload)


def _quadrant_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    return (
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height),
    )


def _target_masks(input_root: Path, digest: str) -> list[np.ndarray]:
    """Load v3 independent masks or derive them from a legacy cumulative grid."""
    mask_path = input_root / "MASK" / f"{digest}_MASK.png"
    with Image.open(mask_path) as source:
        alpha = np.asarray(source.convert("RGBA").getchannel("A")) < 128
    height, width = alpha.shape
    quadrants = [
        alpha[y0:y1, x0:x1] for x0, y0, x1, y1 in _quadrant_boxes(width, height)
    ]
    selection_path = input_root / "SELECTION" / f"{digest}_selection.json"
    mode = ""
    if selection_path.exists():
        try:
            mode = json.loads(selection_path.read_text(encoding="utf-8")).get(
                "selection_mode", ""
            )
        except (OSError, json.JSONDecodeError):
            pass
    if mode == "three-target-global-v3-independent":
        return [mask.astype(bool) for mask in quadrants[:3]]

    # Legacy grids accumulated prior targets; adjacent differences recover
    # the three single-target regions without another segmentation call.
    targets: list[np.ndarray] = []
    used = np.zeros_like(quadrants[0], dtype=bool)
    for mask in quadrants[:3]:
        target = np.logical_and(mask, np.logical_not(used))
        targets.append(target)
        used = np.logical_or(used, mask)
    return targets


def materialize_upload_mask(input_root: Path, digest: str) -> Path:
    """Build and validate the independent four-quadrant mask sent to the API."""
    image_path = input_root / "IMAGE_2X4" / f"{digest}_IMAGE.jpg"
    source_mask = input_root / "MASK" / f"{digest}_MASK.png"
    with Image.open(image_path) as image_source, Image.open(source_mask) as mask_source:
        image = image_source.convert("RGB")
        if mask_source.mode != "RGBA":
            raise TransportError("Mask 必须是带 Alpha 通道的 RGBA PNG")
        if mask_source.size != image.size:
            raise TransportError(
                f"Mask 尺寸 {mask_source.size} 与四宫格原图尺寸 {image.size} 不一致"
            )
    targets = _target_masks(input_root, digest)
    if len(targets) != 3 or any(not target.any() for target in targets):
        raise TransportError("前三个象限必须各自包含一个非空目标 Mask")
    height, width = targets[0].shape
    if image.size != (width * 2, height * 2):
        raise TransportError("目标 Mask 象限尺寸与四宫格原图不一致")

    regions = [*targets, np.ones_like(targets[0], dtype=bool)]
    image_array = np.asarray(image)
    output = Image.new("RGBA", image.size, (0, 0, 0, 255))
    counts: list[int] = []
    for region, box in zip(regions, _quadrant_boxes(*image.size)):
        x0, y0, x1, y1 = box
        alpha = np.where(region, 0, 255).astype(np.uint8)
        tile = np.dstack((image_array[y0:y1, x0:x1], alpha)).astype(np.uint8)
        output.paste(Image.fromarray(tile, "RGBA"), (x0, y0))
        counts.append(int(region.sum()))
    if counts[3] != width * height:
        raise TransportError("第四象限必须为全图可编辑 Mask")

    destination = input_root / ".cascadeforge" / "upload_masks" / f"{digest}_MASK.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination, "PNG", optimize=True)
    return destination


def verify_remote_mask(url: str, local_path: Path) -> dict[str, Any]:
    """Confirm that the exact public URL exposes an equivalent RGBA mask."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    try:
        with Image.open(io.BytesIO(response.content)) as remote_source, Image.open(
            local_path
        ) as local_source:
            if remote_source.mode != "RGBA" or remote_source.size != local_source.size:
                raise TransportError("远程 Mask 的模式或尺寸与本地文件不一致")
            size = local_source.size
            remote_alpha = np.asarray(remote_source.getchannel("A"))
            local_alpha = np.asarray(local_source.getchannel("A"))
    except (OSError, ValueError) as exc:
        raise TransportError(f"远程 Mask 不是有效 RGBA PNG：{exc}") from exc
    if not np.array_equal(remote_alpha, local_alpha):
        raise TransportError("远程 Mask 的 Alpha 数据与本地文件不一致")
    values, counts = np.unique(remote_alpha, return_counts=True)
    if set(int(value) for value in values) - {0, 255}:
        raise TransportError("远程 Mask Alpha 只能包含 0 和 255")
    return {
        "size": list(size),
        "transparent_pixels": int(counts[values.tolist().index(0)]) if 0 in values else 0,
        "opaque_pixels": int(counts[values.tolist().index(255)]) if 255 in values else 0,
    }


def _quality_path(input_root: Path, digest: str) -> Path:
    return input_root / ".cascadeforge" / "quality" / f"{digest}.json"


def _is_rejected(input_root: Path, digest: str) -> bool:
    path = _quality_path(input_root, digest)
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("version") == PIPELINE_VERSION and state.get("passed") is False


def compose_and_measure(
    input_root: Path, digest: str, raw_path: Path, destination: Path
) -> tuple[bool, list[dict[str, Any]]]:
    """Accumulate three independent edits and enforce every protected pixel locally."""
    original_path = input_root / "IMAGE_2" / f"{digest}.jpg"
    with Image.open(raw_path) as raw_source, Image.open(original_path) as original_source:
        generated = raw_source.convert("RGB")
        width, height = generated.size
        if width % 2 or height % 2 or width < 2 or height < 2:
            raise TransportError(f"编辑结果不是有效四宫格尺寸：{generated.size}")
        quadrant_size = (width // 2, height // 2)
        original = original_source.convert("RGB").resize(
            quadrant_size, Image.Resampling.LANCZOS
        )
        original_array = np.asarray(original).copy()
        generated_array = np.asarray(generated)
        masks = [
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    quadrant_size, Image.Resampling.NEAREST
                )
            )
            > 128
            for mask in _target_masks(input_root, digest)
        ]

        frames: list[np.ndarray] = []
        metrics: list[dict[str, Any]] = []
        previous = original_array.copy()
        for index, (mask, box) in enumerate(
            zip(masks, _quadrant_boxes(width, height)), 1
        ):
            x0, y0, x1, y1 = box
            model_frame = generated_array[y0:y1, x0:x1]
            pixel_difference = np.abs(
                model_frame.astype(np.int16) - original_array.astype(np.int16)
            ).mean(axis=2)
            mean_difference = float(pixel_difference[mask].mean())
            changed_ratio = float((pixel_difference[mask] >= MIN_MEAN_DIFFERENCE).mean())
            passed = (
                mean_difference >= MIN_MEAN_DIFFERENCE
                and changed_ratio >= MIN_CHANGED_RATIO
            )
            metrics.append(
                {
                    "round": index,
                    "target_pixels": int(mask.sum()),
                    "mean_difference": round(mean_difference, 3),
                    "changed_ratio": round(changed_ratio, 4),
                    "passed": passed,
                }
            )
            current = previous.copy()
            current[mask] = model_frame[mask]
            frames.append(current)
            previous = current

        # Use the model's global fourth result, then restore all three edited
        # targets so the global pass cannot erase prior object changes.
        x0, y0, x1, y1 = _quadrant_boxes(width, height)[3]
        global_frame = generated_array[y0:y1, x0:x1].copy()
        target_union = np.logical_or.reduce(masks)
        global_frame[target_union] = frames[2][target_union]
        frames.append(global_frame)

        composed = Image.new("RGB", generated.size)
        for frame, box in zip(frames, _quadrant_boxes(width, height)):
            composed.paste(Image.fromarray(frame, "RGB"), (box[0], box[1]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        composed.save(destination, "JPEG", quality=95, subsampling=0)
    return all(metric["passed"] for metric in metrics), metrics


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
    rejected_path = input_root / "EDITED_4K" / "REJECTED" / f"{digest}_{model}_edited.jpg"
    download_json = input_root / "DOWNLOAD_JSON" / f"{digest}_{model}_url.json"
    result: dict[str, Any] = {"md5": digest, "model": model, "status": "error"}
    if output_path.exists():
        result["status"] = "skip"
        return result
    if _is_rejected(input_root, digest):
        result["status"] = "rejected"
        result["error"] = "质量门禁未通过；删除本地 quality sidecar 后可人工重试"
        return result
    if not json_path.exists() or not image_path.exists() or not mask_path.exists():
        result["error"] = "缺少提示词 JSON、四宫格原图或累计 Mask"
        return result
    try:
        url = None
        if download_json.exists():
            cached = json.loads(download_json.read_text(encoding="utf-8"))
            if cached.get("pipeline_version") == PIPELINE_VERSION:
                url = cached.get("url")
        if not raw_path.exists() and not url:
            upload_mask = materialize_upload_mask(input_root, digest)
            image_url = _upload(image_path, config)
            mask_url = _upload(upload_mask, config)
            mask_stats = verify_remote_mask(mask_url, upload_mask)
            print(
                f"[MASK-OK] {digest}: {mask_stats['size'][0]}x{mask_stats['size'][1]}，"
                f"透明像素 {mask_stats['transparent_pixels']}"
            )
            prompt = build_prompt_from_json(json.loads(json_path.read_text(encoding="utf-8")))
            with Image.open(image_path) as source:
                width, height = source.size
            ratio = min(
                SUPPORTED_RATIOS,
                key=lambda name: abs(width / height - SUPPORTED_RATIOS[name]),
            )
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
                json.dumps(
                    {
                        "url": url,
                        "md5": digest,
                        "model": model,
                        "pipeline_version": PIPELINE_VERSION,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not raw_path.exists():
            if not url:
                raise TransportError("缺少可恢复的图片 URL")
            _download(url, raw_path)

        candidate_path = input_root / ".cascadeforge" / "candidates" / f"{digest}.jpg"
        passed, metrics = compose_and_measure(input_root, digest, raw_path, candidate_path)
        quality_path = _quality_path(input_root, digest)
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": PIPELINE_VERSION,
            "passed": passed,
            "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "metrics": metrics,
        }
        quality_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if not passed:
            rejected_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_path.write_bytes(candidate_path.read_bytes())
            result["status"] = "rejected"
            failed_rounds = [str(item["round"]) for item in metrics if not item["passed"]]
            result["error"] = f"质量门禁未通过：ROUND_{'、'.join(failed_rounds)}"
            return result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(candidate_path.read_bytes())
        result["status"] = "success"
        result["output"] = str(output_path)
    except Exception as exc:
        result["error"] = str(exc)[:500]
    return result


def run_editor(input_root: Path, config_path: Path | None, concurrency: int = 4, watch: bool = False) -> int:
    config = load_config(config_path)
    if not config.edit.api_key and not config.oss.enabled:
        print("[错误] 请配置 TOAPIS_API_KEY，或提供完整 OSS 配置")
        return 2
    while True:
        json_dir = input_root / "JSON"
        pending = sorted(path.name.removesuffix("_JSON_gpt.json") for path in json_dir.glob("*_JSON_gpt.json"))
        results_dir = input_root / "EDITED_4K"
        rejected = [digest for digest in pending if _is_rejected(input_root, digest)]
        pending = [
            digest
            for digest in pending
            if not (results_dir / f"{digest}_gpt_edited.jpg").exists()
            and not _is_rejected(input_root, digest)
        ]
        if not pending:
            if rejected:
                print(f"[WARN] {len(rejected)} 个任务被质量门禁拒绝")
            if watch:
                print("[INFO] 暂无待处理任务，20 秒后重试；Ctrl+C 退出")
                time.sleep(20)
                continue
            print("[INFO] 没有待处理任务")
            return 1 if rejected else 0
        failures = 0
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for result in pool.map(lambda digest: process_one(digest, input_root, config), pending):
                print(f"[{result['status'].upper()}] {result['md5']}: {result.get('error', '')}")
                failures += result["status"] in {"error", "rejected"}
        if not watch:
            return 1 if failures else 0
