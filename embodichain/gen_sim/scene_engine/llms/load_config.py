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
from typing import Any

from embodichain.gen_sim.scene_engine.configs.environment import (
    read_scene_engine_env_values,
)


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible VLM connection settings."""

    api_key: str
    model: str
    base_url: str
    default_query: dict[str, Any]
    max_attempts: int


def load_llm_config() -> LLMConfig:
    """Load the required OpenAI-compatible LLM settings from ``gen_sim/.env``."""
    values = read_scene_engine_env_values(
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "SCENE_ENGINE_OPENAI_DEFAULT_QUERY",
        "OPENAI_MAX_ATTEMPTS",
    )
    try:
        default_query = json.loads(values["SCENE_ENGINE_OPENAI_DEFAULT_QUERY"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "SCENE_ENGINE_OPENAI_DEFAULT_QUERY must contain a JSON object."
        ) from exc

    if not isinstance(default_query, dict):
        raise ValueError("SCENE_ENGINE_OPENAI_DEFAULT_QUERY must be a JSON object.")
    missing = [
        key
        for key, value in {
            "OPENAI_API_KEY": values["OPENAI_API_KEY"],
            "OPENAI_MODEL": values["OPENAI_MODEL"],
            "OPENAI_BASE_URL": values["OPENAI_BASE_URL"],
        }.items()
        if not value.strip()
    ]
    if missing:
        raise ValueError(f"Missing required LLM config keys: {missing}")

    try:
        parsed_max_attempts = int(values["OPENAI_MAX_ATTEMPTS"])
    except (TypeError, ValueError) as exc:
        raise ValueError("OPENAI_MAX_ATTEMPTS must be an integer.") from exc
    if parsed_max_attempts < 1:
        raise ValueError("OPENAI_MAX_ATTEMPTS must be at least 1.")

    return LLMConfig(
        api_key=values["OPENAI_API_KEY"].strip(),
        model=values["OPENAI_MODEL"].strip(),
        base_url=values["OPENAI_BASE_URL"].rstrip("/"),
        default_query=default_query,
        max_attempts=parsed_max_attempts,
    )
