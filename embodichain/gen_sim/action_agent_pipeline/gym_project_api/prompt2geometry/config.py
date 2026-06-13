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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.utils.llm_config import (
    get_openai_compatible_llm_config,
)

__all__ = ["Prompt2GeometryConfig", "load_prompt2geometry_config"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass(frozen=True)
class Prompt2GeometryConfig:
    """Prompt2Geometry runtime configuration."""

    zimage_base_url: str
    sam3_base_url: str
    sam3d_base_url: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_timeout_s: float


def load_prompt2geometry_config(
    config_path: Path | None = None,
) -> Prompt2GeometryConfig:
    """Load prompt2geometry config from a local JSON file and environment."""
    path = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt2Geometry config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    services = _mapping(raw.get("services"), "services")
    llm = _mapping(
        _mapping(raw.get("llm"), "llm").get("openai_compatible"),
        "llm.openai_compatible",
    )
    shared_llm = get_openai_compatible_llm_config(
        required=False,
        require_base_url=False,
    )

    return Prompt2GeometryConfig(
        zimage_base_url=_env_or_config(
            "PROMPT2GEOMETRY_ZIMAGE_BASE_URL",
            _service_base_url(services, "zimage"),
        ),
        sam3_base_url=_env_or_config(
            "PROMPT2GEOMETRY_SAM3_BASE_URL",
            _service_base_url(services, "sam3"),
        ),
        sam3d_base_url=_env_or_config(
            "PROMPT2GEOMETRY_SAM3D_BASE_URL",
            _service_base_url(services, "sam3d"),
        ),
        llm_api_key=_env_or_config(
            "PROMPT2GEOMETRY_LLM_API_KEY",
            str(shared_llm.get("api_key") or llm.get("api_key", "")),
        ),
        llm_model=_env_or_config(
            "PROMPT2GEOMETRY_LLM_MODEL",
            str(shared_llm.get("model") or llm.get("model", "")),
        ),
        llm_base_url=_env_or_config(
            "PROMPT2GEOMETRY_LLM_BASE_URL",
            str(shared_llm.get("base_url") or llm.get("base_url", "")),
        ).rstrip("/"),
        llm_timeout_s=float(
            os.getenv("PROMPT2GEOMETRY_LLM_TIMEOUT_S")
            or llm.get("timeout_s", 120.0)
        ),
    )


def _service_base_url(services: dict[str, Any], name: str) -> str:
    section = _mapping(services.get(name), f"services.{name}")
    return str(section.get("base_url", "")).rstrip("/")


def _env_or_config(env_name: str, config_value: str) -> str:
    return str(os.getenv(env_name) or config_value).strip()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Prompt2Geometry config key {name} must be an object.")
    return value
