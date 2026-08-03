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
from typing import Any

from embodichain.gen_sim.env import load_gen_sim_env


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible VLM connection settings."""

    api_key: str
    model: str
    base_url: str
    default_query: dict[str, Any]
    max_attempts: int


def load_llm_config() -> LLMConfig:
    """Load LLM settings from the shared ``.env`` and process environment."""
    load_gen_sim_env()
    try:
        default_query = json.loads(os.getenv("SCENE_ENGINE_OPENAI_DEFAULT_QUERY", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "SCENE_ENGINE_OPENAI_DEFAULT_QUERY must be valid JSON."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    max_attempts = os.getenv("OPENAI_MAX_ATTEMPTS", "3")

    if not isinstance(default_query, dict):
        raise ValueError("SCENE_ENGINE_OPENAI_DEFAULT_QUERY must be a JSON object.")
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
        raise ValueError("OPENAI_MAX_ATTEMPTS must be an integer.") from exc
    if parsed_max_attempts < 1:
        raise ValueError("OPENAI_MAX_ATTEMPTS must be at least 1.")

    return LLMConfig(
        api_key=api_key.strip(),
        model=model.strip(),
        base_url=base_url.rstrip("/"),
        default_query=default_query,
        max_attempts=parsed_max_attempts,
    )
