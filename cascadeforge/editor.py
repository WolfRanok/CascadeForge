"""Vendor-neutral image editing transport with optional OSS staging."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
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


def build_prompt_from_json(data: dict[str, Any]) -> str:
    rounds = [data.get(f"ROUND_{index}", {}).get("long", "") for index in range(1, 5)]
    return f"""这是同一张原图组成的 2×2 四宫格。一次生成四个结果。

左上：原图 + {rounds[0]}，仅执行这一项。
右上：复制左上结果，仅新增 {rounds[1]}。
左下：复制右上结果，仅新增 {rounds[2]}。
右下：复制左下结果，执行整图变换：{rounds[3]}。

规则：
1. 每格先完整继承上一格，再执行本格唯一新增操作。
2. 前三格只修改 Mask 指定物体，变化必须明显可见。
3. 优先使用高对比颜色、明显材质或清晰图案，禁止细微变化。
4. 保持物体轮廓、姿态、位置、数量和场景构图不变。
5. 第四格可改变整图天气、昼夜、季节、光照、氛围和色调。
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
    download_json = input_root / "DOWNLOAD_JSON" / f"{digest}_{model}_url.json"
    result: dict[str, Any] = {"md5": digest, "model": model, "status": "error"}
    if output_path.exists():
        result["status"] = "skip"
        return result
    if not json_path.exists() or not image_path.exists() or not mask_path.exists():
        result["error"] = "缺少提示词 JSON、四宫格原图或累计 Mask"
        return result
    try:
        # Reuse a saved URL when a previous run completed the remote task.
        if download_json.exists():
            cached = json.loads(download_json.read_text(encoding="utf-8"))
            url = cached.get("url")
        else:
            url = None
        if not url:
            image_url = _upload(image_path, config)
            mask_url = _upload(mask_path, config)
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
        _download(url, output_path)
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
        pending = [digest for digest in pending if not (results_dir / f"{digest}_gpt_edited.jpg").exists()]
        if not pending:
            if watch:
                print("[INFO] 暂无待处理任务，20 秒后重试；Ctrl+C 退出")
                time.sleep(20)
                continue
            print("[INFO] 没有待处理任务")
            return 0
        failures = 0
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for result in pool.map(lambda digest: process_one(digest, input_root, config), pending):
                print(f"[{result['status'].upper()}] {result['md5']}: {result.get('error', '')}")
                failures += result["status"] == "error"
        if not watch:
            return 1 if failures else 0
