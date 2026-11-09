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

"""Compatibility-only configuration readers; never import business owners."""

from __future__ import annotations
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
from typing import Any
from embodichain.gen_sim.environment import _read_gen_sim_env_values

__all__: list[str] = []
_GEN_SIM_DIR = Path(__file__).resolve().parents[1]
_GEN_SIM_ENV_PATH = _GEN_SIM_DIR / ".env"
_GEN_CONFIG_PATH = _GEN_SIM_DIR / "asset_engine/configs/gen_config.json"


def read_scene_engine_env_values(*keys: str) -> dict[str, str]:
    return _read_gen_sim_env_values(
        *keys, env_path=_GEN_SIM_ENV_PATH, owner="Scene Engine"
    )


def read_asset_engine_env_values(*keys: str) -> dict[str, str]:
    aliases = {
        key: key.replace(
            "ASSET_ENGINE_ARTICULATED_GENERATION_",
            "SCENE_ENGINE_ARTICULATED_GENERATION_",
        )
        for key in keys
    }
    return _read_gen_sim_env_values(
        *keys, env_path=_GEN_SIM_ENV_PATH, owner="Asset Engine", aliases=aliases
    )


def instruction_model(model: str | None) -> str | None:
    return _model_name(model, _load_local_env(), _load_llm_config())


def _load_llm_config() -> dict[str, Any]:
    if _GEN_CONFIG_PATH.is_file():
        raw = json.loads(_GEN_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            llm = raw.get("llm", {})
            if isinstance(llm, Mapping):
                configured = llm.get("openai_compatible", {})
                if isinstance(configured, Mapping):
                    return dict(configured)
    return {}


def _model_name(
    explicit: str | None, local_env: Mapping[str, str], config: Mapping[str, Any]
) -> str | None:
    return (
        (explicit.strip() if isinstance(explicit, str) else "")
        or _first_env_value(
            local_env,
            "TASK_ENGINE_LLM_MODEL",
            "ACTION_ENGINE_LLM_MODEL",
            "OPENAI_MODEL",
            "LLM_MODEL",
        )
        or str(config.get("model", "")).strip()
        or None
    )


def _load_local_env() -> dict[str, str]:
    """Read Task Engine model configuration without mutating the environment."""
    return _load_env_file(_GEN_SIM_ENV_PATH)


def _load_llm_settings(*, model: str | None) -> dict[str, Any]:
    local_env = _load_local_env()
    config = _load_llm_config()
    api_key, base_url = _resolve_transport_settings(local_env, config)
    selected_model = _model_name(model, local_env, config)
    default_query = config.get("default_query", {}) or {}
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for Task Engine interpretation. Set it "
            f"in the process environment or {_GEN_SIM_ENV_PATH}."
        )
    if not selected_model:
        raise ValueError(
            "A text LLM model is required through model=, TASK_ENGINE_LLM_MODEL, "
            f"OPENAI_MODEL, or {_GEN_CONFIG_PATH}."
        )
    if not isinstance(default_query, Mapping):
        raise ValueError("LLM default_query must be a mapping.")
    return {
        "api_key": api_key,
        "model": selected_model,
        "base_url": base_url,
        "default_query": dict(default_query),
    }


def _resolve_transport_settings(
    local_env: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve an API key and endpoint from one configuration source."""
    transports = (
        (
            _mapping_value(os.environ, "OPENAI_API_KEY"),
            _mapping_value(
                os.environ,
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "LLM_URL",
            ),
        ),
        (
            _mapping_value(local_env, "OPENAI_API_KEY"),
            _mapping_value(
                local_env,
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "LLM_URL",
            ),
        ),
        (
            _mapping_value(config, "api_key"),
            _mapping_value(config, "base_url"),
        ),
    )
    for api_key, base_url in transports:
        if api_key and base_url:
            return api_key, base_url.rstrip("/")
    for api_key, base_url in transports:
        if api_key:
            return api_key, base_url.rstrip("/")
    return "", ""


def _mapping_value(source: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid dotenv key at {path}:{line_number}.")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _first_env_value(local_env: Mapping[str, str], *names: str) -> str | None:
    for source in (os.environ, local_env):
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def simready_settings() -> dict[str, Any]:
    raw = json.loads(_GEN_CONFIG_PATH.read_text())
    cfg = dict(raw.get("llm", {}).get("openai_compatible", {}))
    for name, env in (
        ("api_key", "OPENAI_API_KEY"),
        ("model", "OPENAI_MODEL"),
        ("base_url", "OPENAI_BASE_URL"),
    ):
        cfg[name] = os.getenv(env) or cfg.get(name, "")
    if any(not cfg.get(key) for key in ("api_key", "model", "base_url")):
        raise ValueError("Missing required SimReady api_key, model or base_url")
    return cfg
