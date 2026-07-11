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

__all__ = [
    "DEFAULT_LLM_MODEL",
    "ACTION_PIPELINE_LLM_ENV_PATH",
    "GEN_CONFIG_PATH",
    "LLM_ENV_PATH",
    "LEGACY_LLM_ENV_PATH",
    "SIMREADY_LLM_ENV_PATH",
    "get_openai_compatible_llm_config",
    "load_local_env_values",
]

DEFAULT_LLM_MODEL = os.getenv("EMBODICHAIN_DEFAULT_LLM_MODEL", "gpt-4o")
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        parent
        for parent in CONFIG_DIR.parents
        if (parent / "setup.py").exists() and (parent / "embodichain").exists()
    ),
    CONFIG_DIR.parents[3],
)
GEN_CONFIG_PATH = (
    PROJECT_ROOT / "embodichain/gen_sim/simready_pipeline/configs/gen_config.json"
)
LLM_ENV_PATH = PROJECT_ROOT / ".env"
SIMREADY_LLM_ENV_PATH = (
    PROJECT_ROOT / "embodichain/gen_sim/simready_pipeline/configs/.env"
)
ACTION_PIPELINE_LLM_ENV_PATH = CONFIG_DIR / ".env"
LEGACY_LLM_ENV_PATH = SIMREADY_LLM_ENV_PATH


def _load_env_file(path: Path | None = None) -> dict[str, str]:
    """Read local KEY=VALUE credentials without overriding shell variables."""
    path = path or LLM_ENV_PATH
    if not path.exists():
        return {}

    env_values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            env_values[key] = value
    return env_values


def _load_env_files(paths: tuple[Path, ...] | None = None) -> dict[str, str]:
    """Read local env files, with later paths taking precedence."""
    env_values: dict[str, str] = {}
    for path in paths or (
        LLM_ENV_PATH,
        ACTION_PIPELINE_LLM_ENV_PATH,
        SIMREADY_LLM_ENV_PATH,
    ):
        env_values.update(_load_env_file(path))
    return env_values


def load_local_env_values() -> dict[str, str]:
    """Load the action pipeline's local environment files.

    Returned values do not include process environment variables. Callers must
    let explicit shell variables take precedence when resolving individual keys.
    """
    return _load_env_files()


def _get_first_value(
    local_env: dict[str, str],
    *names: str,
    default: str | None = None,
) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
        value = local_env.get(name)
        if value:
            return value
    return default


def _load_gen_config(path: Path | None = None) -> dict[str, Any]:
    path = path or GEN_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"gen_config.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    return dict(raw_cfg.get("llm", {}).get("openai_compatible", {}))


def get_openai_compatible_llm_config(
    *,
    required: bool = False,
    require_base_url: bool = False,
    default_model: str = DEFAULT_LLM_MODEL,
) -> dict[str, Any]:
    """Return shared OpenAI-compatible LLM config for agents and gen-sim."""
    local_env = load_local_env_values()
    try:
        json_cfg = _load_gen_config()
    except FileNotFoundError:
        json_cfg = {}

    cfg = {
        "api_key": _get_first_value(local_env, "OPENAI_API_KEY")
        or json_cfg.get("api_key", ""),
        "model": _get_first_value(local_env, "OPENAI_MODEL", "LLM_MODEL")
        or json_cfg.get("model")
        or default_model,
        "base_url": _get_first_value(
            local_env,
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "LLM_URL",
        )
        or json_cfg.get("base_url", ""),
        "default_query": json_cfg.get("default_query", {}) or {},
        "proxy_url": _get_first_value(
            local_env,
            "EMBODICHAIN_LLM_PROXY",
            "LLM_PROXY_URL",
        )
        or json_cfg.get("proxy_url", ""),
    }

    if cfg["base_url"]:
        cfg["base_url"] = cfg["base_url"].rstrip("/")

    if required:
        required_keys = ["api_key", "model"]
        if require_base_url:
            required_keys.append("base_url")
        missing = [key for key in required_keys if not cfg.get(key)]
        if missing:
            raise ValueError(
                f"Missing required LLM config keys: {missing}. "
                "Set them in shell environment variables, "
                f"{LLM_ENV_PATH}, {ACTION_PIPELINE_LLM_ENV_PATH}, "
                f"{SIMREADY_LLM_ENV_PATH}, or {GEN_CONFIG_PATH}."
            )

    return cfg
