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

import os
from collections.abc import Mapping
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.utils.llm_config import (
    DEFAULT_LLM_MODEL,
    get_openai_compatible_llm_config,
)
from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
    UsageTrackedChatModel,
)

__all__ = [
    "DEFAULT_LLM_MODEL",
    "apply_proxy_env",
    "create_chat_openai",
    "create_openai_client",
    "get_openai_compatible_llm_config",
]


def apply_proxy_env(proxy_url: str | None) -> None:
    """Apply an optional proxy URL for OpenAI-compatible clients."""
    if not proxy_url:
        return
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url


def _resolve_llm_config(
    *,
    config: Mapping[str, Any] | None,
    required: bool,
    require_base_url: bool,
) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    return get_openai_compatible_llm_config(
        required=required,
        require_base_url=require_base_url,
    )


def create_openai_client(
    *,
    config: Mapping[str, Any] | None = None,
    required: bool = True,
    require_base_url: bool = False,
):
    """Create the shared OpenAI-compatible SDK client used by gen-sim MLLM calls."""
    from openai import OpenAI

    cfg = _resolve_llm_config(
        config=config,
        required=required,
        require_base_url=require_base_url,
    )
    apply_proxy_env(cfg.get("proxy_url"))

    kwargs: dict[str, Any] = {
        "api_key": cfg["api_key"],
        "default_query": cfg.get("default_query") or None,
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def create_chat_openai(
    *,
    temperature: float = 0.0,
    model: str | None = None,
    config: Mapping[str, Any] | None = None,
    required: bool = True,
    usage_stage: str | None = None,
):
    """Create the shared LangChain OpenAI-compatible chat client for agents."""
    from langchain_openai import ChatOpenAI

    cfg = _resolve_llm_config(
        config=config,
        required=required,
        require_base_url=False,
    )
    apply_proxy_env(cfg.get("proxy_url"))

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "model": model or cfg.get("model") or DEFAULT_LLM_MODEL,
        "api_key": cfg["api_key"],
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    return UsageTrackedChatModel(
        ChatOpenAI(**kwargs),
        stage=usage_stage,
    )
