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

__all__ = [
    "DEFAULT_ANTHROPIC_MAX_TOKENS",
    "DEFAULT_API_KEY",
    "DEFAULT_DISTURBANCE_PROB",
    "DEFAULT_EMBODICHAIN_ROOT",
    "DEFAULT_HEADLESS",
    "DEFAULT_HTTP_PROXY",
    "DEFAULT_HTTPS_PROXY",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_URL",
    "DEFAULT_NUM_TRIALS",
    "DEFAULT_PRECOMPILE_RECOVERY_ACTIONS",
    "DEFAULT_PRECOMPILE_RECOVERY_MONITORS",
    "DEFAULT_SIM_DEVICE",
    "get_api_key",
    "get_default_headless",
    "get_default_sim_device",
    "get_llm_model",
    "get_llm_url",
    "get_openai_compatible_api_key",
    "get_openai_compatible_base_url",
    "get_openai_compatible_model",
]

DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_LLM_MODEL = "mimo-v2.5"
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192
DEFAULT_PRECOMPILE_RECOVERY_ACTIONS = False
DEFAULT_PRECOMPILE_RECOVERY_MONITORS = False
DEFAULT_API_KEY = "tp-cigd9h4eh33v79adk5wz77y9o9rngsvcrw527wxiic5jdeqq"
DEFAULT_LLM_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_HTTPS_PROXY = "http://127.0.0.1:7897"
DEFAULT_HTTP_PROXY = "http://127.0.0.1:7897"
DEFAULT_EMBODICHAIN_ROOT = "/home/dex/桌面/EmbodiChain/origin/EmbodiChain"
DEFAULT_SIM_DEVICE = "cpu"
DEFAULT_HEADLESS = True
DEFAULT_DISTURBANCE_PROB = 0.05
DEFAULT_NUM_TRIALS = 20


def _env(key: str, default: str) -> str:
    return os.getenv(key) or default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_api_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("API_KEY")
        or DEFAULT_API_KEY
    )


def get_llm_model() -> str:
    return _env("LLM_MODEL", DEFAULT_LLM_MODEL)


def get_llm_url() -> str:
    return _env("LLM_URL", DEFAULT_LLM_URL)


def get_openai_compatible_api_key() -> str:
    return os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or DEFAULT_API_KEY


def get_openai_compatible_model() -> str:
    return os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL


def get_openai_compatible_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_URL") or DEFAULT_LLM_URL


def get_default_sim_device() -> str:
    return _env("SIM_DEVICE", DEFAULT_SIM_DEVICE)


def get_default_headless() -> bool:
    return _env_bool("HEADLESS", DEFAULT_HEADLESS)
