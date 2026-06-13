# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

__all__ = [
    "LLM_USAGE_PATH_ENV",
    "LLM_USAGE_PROCESS_ENV",
    "LLM_USAGE_RUN_ID_ENV",
    "UsageTrackedChatModel",
    "build_usage_summary",
    "configure_usage_tracking",
    "disable_usage_tracking",
    "extract_usage_from_langchain_response",
    "normalize_usage",
    "normalize_usage_stage",
    "record_langchain_usage",
    "record_llm_usage",
    "scrub_usage_tracking_env",
    "write_usage_summary",
]


LLM_USAGE_PATH_ENV = "EMBODICHAIN_LLM_USAGE_PATH"
LLM_USAGE_RUN_ID_ENV = "EMBODICHAIN_LLM_USAGE_RUN_ID"
LLM_USAGE_PROCESS_ENV = "EMBODICHAIN_LLM_USAGE_PROCESS"

_USAGE_ENV_KEYS = {
    LLM_USAGE_PATH_ENV,
    LLM_USAGE_RUN_ID_ENV,
    LLM_USAGE_PROCESS_ENV,
}
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
)


class UsageTrackedChatModel:
    """Proxy a LangChain chat model and record usage after each invoke call."""

    def __init__(
        self,
        inner: Any,
        *,
        stage: str | None,
        provider: str = "langchain_openai",
    ) -> None:
        self._inner = inner
        self._usage_stage = normalize_usage_stage(stage or "chat")
        self._usage_provider = provider

    def invoke(self, *args, **kwargs):
        response = self._inner.invoke(*args, **kwargs)
        record_langchain_usage(
            response,
            stage=self._usage_stage,
            provider=self._usage_provider,
            model=_model_name_from_chat_model(self._inner),
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def configure_usage_tracking(
    *,
    usage_path: str | Path,
    run_id: str,
    process_name: str,
    reset: bool = False,
) -> Path:
    """Configure process-local environment variables for LLM usage logging."""
    path = Path(usage_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.write_text("", encoding="utf-8")
    os.environ[LLM_USAGE_PATH_ENV] = path.as_posix()
    os.environ[LLM_USAGE_RUN_ID_ENV] = str(run_id)
    os.environ[LLM_USAGE_PROCESS_ENV] = str(process_name)
    return path


def disable_usage_tracking() -> None:
    """Disable process-local EmbodiChain LLM usage logging."""
    for key in _USAGE_ENV_KEYS:
        os.environ.pop(key, None)


def scrub_usage_tracking_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment copy without EmbodiChain LLM usage variables."""
    cleaned = dict(os.environ if env is None else env)
    for key in _USAGE_ENV_KEYS:
        cleaned.pop(key, None)
    return cleaned


def normalize_usage_stage(stage: str) -> str:
    """Normalize a human-readable usage stage into a compact identifier."""
    value = str(stage or "unknown").strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or "unknown"


def normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int | None]:
    """Normalize OpenAI and LangChain token usage shapes."""
    if not isinstance(usage, Mapping):
        return {field: None for field in _TOKEN_FIELDS}

    input_tokens = _first_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _first_int(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    prompt_details = _mapping_value(usage, "prompt_tokens_details")
    input_details = _mapping_value(usage, "input_token_details")
    completion_details = _mapping_value(usage, "completion_tokens_details")
    output_details = _mapping_value(usage, "output_token_details")

    cached_tokens = _first_int(usage, "cached_tokens", "cache_read")
    if cached_tokens is None:
        cached_tokens = _first_int(
            prompt_details,
            "cached_tokens",
            "cache_read",
        )
    if cached_tokens is None:
        cached_tokens = _first_int(input_details, "cached_tokens", "cache_read")

    reasoning_tokens = _first_int(usage, "reasoning_tokens", "reasoning")
    if reasoning_tokens is None:
        reasoning_tokens = _first_int(
            completion_details,
            "reasoning_tokens",
            "reasoning",
        )
    if reasoning_tokens is None:
        reasoning_tokens = _first_int(
            output_details,
            "reasoning_tokens",
            "reasoning",
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def extract_usage_from_langchain_response(
    response: Any,
) -> tuple[dict[str, int | None], dict[str, Any]]:
    """Extract usage fields and lightweight metadata from a LangChain response."""
    metadata = _mapping_value_from_object(response, "response_metadata")
    usage = _mapping_value_from_object(response, "usage_metadata")
    if not usage:
        usage = _mapping_value(metadata, "token_usage")

    usage_values = normalize_usage(usage)
    response_metadata = {
        "model": _string_value(metadata, "model_name", "model"),
        "request_id": _string_value(metadata, "id", "request_id"),
        "finish_reason": _finish_reason(metadata),
        "raw_usage": _json_safe(usage) if isinstance(usage, Mapping) else None,
    }
    return usage_values, response_metadata


def record_langchain_usage(
    response: Any,
    *,
    stage: str,
    provider: str = "langchain_openai",
    model: str | None = None,
) -> None:
    """Record usage from a LangChain response if usage logging is enabled."""
    usage, metadata = extract_usage_from_langchain_response(response)
    record_llm_usage(
        stage=stage,
        provider=provider,
        model=metadata.get("model") or model,
        usage=usage,
        request_id=metadata.get("request_id"),
        finish_reason=metadata.get("finish_reason"),
        raw_usage=metadata.get("raw_usage"),
    )


def record_llm_usage(
    *,
    stage: str,
    provider: str,
    model: str | None,
    usage: Mapping[str, Any] | None,
    request_id: str | None = None,
    finish_reason: str | None = None,
    raw_usage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Append one LLM usage record to the configured JSONL file."""
    usage_path = os.getenv(LLM_USAGE_PATH_ENV)
    if not usage_path:
        return

    usage_values = normalize_usage(usage)
    usage_available = any(usage_values[field] is not None for field in _TOKEN_FIELDS)
    record: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": os.getenv(LLM_USAGE_RUN_ID_ENV),
        "process": os.getenv(LLM_USAGE_PROCESS_ENV),
        "pid": os.getpid(),
        "stage": normalize_usage_stage(stage),
        "provider": provider,
        "model": model,
        "usage_available": usage_available,
        "request_id": request_id,
        "finish_reason": finish_reason,
    }
    record.update(usage_values)
    if raw_usage is not None:
        record["raw_usage"] = _json_safe(raw_usage)
    if metadata:
        record["metadata"] = _json_safe(metadata)

    path = Path(usage_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_usage_summary(usage_path: str | Path) -> dict[str, Any]:
    """Build aggregate token usage totals from a JSONL usage file."""
    path = Path(usage_path).expanduser().resolve()
    records = _read_usage_records(path)
    summary: dict[str, Any] = {
        "usage_path": path.as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": os.getenv(LLM_USAGE_RUN_ID_ENV),
        "total": _empty_bucket(),
        "by_stage": {},
        "by_model": {},
        "by_process": {},
    }

    for record in records:
        _add_record(summary["total"], record)
        _add_grouped_record(summary["by_stage"], record.get("stage"), record)
        _add_grouped_record(summary["by_model"], record.get("model"), record)
        _add_grouped_record(summary["by_process"], record.get("process"), record)

    return summary


def write_usage_summary(
    *,
    usage_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    """Write a JSON token usage summary and return it."""
    summary = build_usage_summary(usage_path)
    path = Path(summary_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_usage_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _empty_bucket() -> dict[str, int]:
    bucket = {
        "calls": 0,
        "calls_with_usage": 0,
    }
    for field in _TOKEN_FIELDS:
        bucket[field] = 0
    return bucket


def _add_grouped_record(
    groups: dict[str, dict[str, int]],
    key: Any,
    record: Mapping[str, Any],
) -> None:
    group_key = str(key or "unknown")
    bucket = groups.setdefault(group_key, _empty_bucket())
    _add_record(bucket, record)


def _add_record(bucket: dict[str, int], record: Mapping[str, Any]) -> None:
    bucket["calls"] += 1
    if record.get("usage_available"):
        bucket["calls_with_usage"] += 1
    for field in _TOKEN_FIELDS:
        value = record.get(field)
        if isinstance(value, int):
            bucket[field] += value


def _model_name_from_chat_model(model: Any) -> str | None:
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return None


def _mapping_value_from_object(value: Any, attr_name: str) -> Mapping[str, Any]:
    attr = getattr(value, attr_name, None)
    return attr if isinstance(attr, Mapping) else {}


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key) if isinstance(mapping, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _first_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _string_value(mapping: Mapping[str, Any], *keys: str) -> str | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _finish_reason(metadata: Mapping[str, Any]) -> str | None:
    reason = _string_value(metadata, "finish_reason")
    if reason:
        return reason
    response_metadata = (
        metadata.get("response_metadata") if isinstance(metadata, Mapping) else None
    )
    if isinstance(response_metadata, Mapping):
        return _string_value(response_metadata, "finish_reason")
    return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
