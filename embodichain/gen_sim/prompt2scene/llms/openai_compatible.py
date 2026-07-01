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

import json
import os
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from embodichain.gen_sim.prompt2scene.llms.config import OpenAICompatibleLLMCfg

__all__ = ["DEFAULT_LLM_CONFIG_PATH", "build_chat_model", "load_llm_config"]

DEFAULT_LLM_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "llm_config.json"
)


def load_llm_config(config_path: Path | None = None) -> OpenAICompatibleLLMCfg:
    """Load the prompt2scene OpenAI-compatible LLM config.

    Args:
        config_path: Optional path to the LLM config JSON file.

    Returns:
        Parsed OpenAI-compatible LLM config.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required config fields are missing.
    """
    config_path = config_path or DEFAULT_LLM_CONFIG_PATH
    config_path = config_path.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: dict[str, Any] = json.load(f)

    cfg = raw_cfg.get("llm", {}).get("openai_compatible", {})
    api_key = os.getenv("OPENAI_API_KEY") or cfg.get("api_key", "")
    model = os.getenv("OPENAI_MODEL") or cfg.get("model", "")
    base_url = os.getenv("OPENAI_BASE_URL") or cfg.get("base_url", "")
    default_query = cfg.get("default_query", {})
    max_attempts = _load_positive_int(
        os.getenv("OPENAI_MAX_ATTEMPTS") or cfg.get("max_attempts", 3),
        key="max_attempts",
    )

    if base_url:
        base_url = base_url.rstrip("/")

    missing = [
        name
        for name, value in {
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required LLM config keys: {missing}")

    if not isinstance(default_query, dict):
        raise ValueError("LLM config key default_query must be a dict.")

    return OpenAICompatibleLLMCfg(
        api_key=api_key,
        model=model,
        base_url=base_url,
        default_query=default_query,
        max_attempts=max_attempts,
    )


def _load_positive_int(value: object, *, key: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM config key {key} must be an integer.") from exc
    if parsed < 1:
        raise ValueError(f"LLM config key {key} must be >= 1.")
    return parsed


def build_chat_model(cfg: OpenAICompatibleLLMCfg) -> Any:
    """Build a LangChain OpenAI-compatible chat model."""
    kwargs: dict[str, Any] = {
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "temperature": 0,
    }
    if cfg.default_query:
        kwargs["default_query"] = cfg.default_query

    return ChatOpenAI(**kwargs)
