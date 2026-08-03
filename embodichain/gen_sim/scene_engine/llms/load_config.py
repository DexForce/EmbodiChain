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

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_LLM_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "scene_engine_config.json"
)


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible VLM connection settings."""

    api_key: str
    model: str
    base_url: str
    default_query: dict[str, Any]
    max_attempts: int


def load_llm_config(config_path: str | Path | None = None) -> LLMConfig:
    """Load LLM settings from JSON, with ``OPENAI_*`` overrides."""
    resolved_config_path = Path(config_path or DEFAULT_LLM_CONFIG_PATH).expanduser()
    resolved_config_path = resolved_config_path.resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"LLM config not found: {resolved_config_path}")

    try:
        raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM config is not valid JSON: {resolved_config_path}"
        ) from exc

    llm_config = raw_config.get("llm", {}).get("openai_compatible", {})
    if not isinstance(llm_config, dict):
        raise ValueError("LLM config key llm.openai_compatible must be an object.")

    api_key = os.getenv("OPENAI_API_KEY") or llm_config.get("api_key", "")
    model = os.getenv("OPENAI_MODEL") or llm_config.get("model", "")
    base_url = os.getenv("OPENAI_BASE_URL") or llm_config.get("base_url", "")
    default_query = llm_config.get("default_query", {})
    max_attempts = os.getenv("OPENAI_MAX_ATTEMPTS") or llm_config.get("max_attempts", 3)

    if not isinstance(default_query, dict):
        raise ValueError("LLM config key default_query must be an object.")
    missing = [
        key
        for key, value in {
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
        }.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise ValueError(f"Missing required LLM config keys: {missing}")

    try:
        parsed_max_attempts = int(max_attempts)
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM config key max_attempts must be an integer.") from exc
    if parsed_max_attempts < 1:
        raise ValueError("LLM config key max_attempts must be at least 1.")

    return LLMConfig(
        api_key=api_key.strip(),
        model=model.strip(),
        base_url=base_url.rstrip("/"),
        default_query=default_query,
        max_attempts=parsed_max_attempts,
    )
