"""Configuration loading without committing credentials to source control."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    return data


def _value(env_name: str, section: dict[str, Any], key: str, default: Any = None) -> Any:
    """Environment variables intentionally override local configuration."""
    return os.getenv(env_name, section.get(key, default))


@dataclass(frozen=True)
class VisionConfig:
    api_key: str | None
    base_url: str
    model: str


@dataclass(frozen=True)
class EditConfig:
    api_key: str | None
    base_url: str
    api_url: str
    upload_url: str
    model: str
    resolution: str
    response_format: str


@dataclass(frozen=True)
class OssConfig:
    access_key_id: str | None
    access_key_secret: str | None
    bucket_name: str | None
    endpoint: str
    path_prefix: str
    sign_expires: int

    @property
    def enabled(self) -> bool:
        return bool(self.access_key_id and self.access_key_secret and self.bucket_name)


@dataclass(frozen=True)
class AppConfig:
    vision: VisionConfig
    edit: EditConfig
    oss: OssConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    # Loading .env is convenient locally; existing process variables still take precedence.
    load_dotenv(override=False)
    raw = _read_json(Path(path) if path else None)
    vision = raw.get("moliapi", {})
    edit = raw.get("toapis", {})
    oss = raw.get("oss", {})
    base_url = _value("TOAPIS_BASE_URL", edit, "base_url", "https://toapis.com")
    return AppConfig(
        vision=VisionConfig(
            api_key=_value("GPT_API_KEY", vision, "api_key"),
            base_url=_value("GPT_BASE_URL", vision, "base_url", "https://api.openai.com/v1"),
            model=_value("GPT_MODEL", vision, "model", "gpt-4.1-mini"),
        ),
        edit=EditConfig(
            api_key=_value("TOAPIS_API_KEY", edit, "api_key"),
            base_url=base_url,
            api_url=_value("TOAPIS_API_URL", edit, "api_url", f"{base_url}/v1/images/generations"),
            upload_url=_value("TOAPIS_UPLOAD_URL", edit, "upload_url", f"{base_url}/v1/uploads/images"),
            model=_value("TOAPIS_MODEL", edit, "model", "gpt-image-2"),
            resolution=_value("TOAPIS_RESOLUTION", edit, "resolution", "4k"),
            response_format=_value("TOAPIS_RESPONSE_FORMAT", edit, "response_format", "url"),
        ),
        oss=OssConfig(
            access_key_id=_value("OSS_ACCESS_KEY_ID", oss, "access_key_id"),
            access_key_secret=_value("OSS_ACCESS_KEY_SECRET", oss, "access_key_secret"),
            bucket_name=_value("OSS_BUCKET_NAME", oss, "bucket_name"),
            endpoint=_value("OSS_ENDPOINT", oss, "endpoint", "https://oss-cn-hangzhou.aliyuncs.com"),
            path_prefix=_value("OSS_PATH_PREFIX", oss, "path_prefix", "CASCADEFORGE_EDIT"),
            sign_expires=int(_value("OSS_SIGN_EXPIRES", oss, "sign_expires", 3600)),
        ),
    )
