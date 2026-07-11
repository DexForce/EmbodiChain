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

"""Runtime configuration adapters for the in-repository Prompt2Scene stage."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import urlparse

from embodichain.gen_sim.action_agent_pipeline.utils.llm_config import (
    get_openai_compatible_llm_config,
    load_local_env_values,
)

if TYPE_CHECKING:
    from embodichain.gen_sim.prompt2scene.llms.config import OpenAICompatibleLLMCfg

__all__ = [
    "build_prompt2scene_llm_config",
    "use_prompt2scene_client_config",
    "write_prompt2scene_client_config",
]

DEFAULT_PROMPT2SCENE_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "prompt2scene/configs"
)
DEFAULT_PROMPT2SCENE_LLM_CONFIG_PATH = (
    DEFAULT_PROMPT2SCENE_CONFIG_DIR / "llm_config.json"
)
DEFAULT_PROMPT2SCENE_CLIENT_CONFIG_PATH = (
    DEFAULT_PROMPT2SCENE_CONFIG_DIR / "client_config.json"
)
_SERVICE_ENV_NAMES = {
    "sam3_segmentation": (
        "PROMPT2SCENE_SAM3_SEGMENTATION_BASE_URL",
        "PROMPT2SCENE_SAM3_SEGMENTATION_PORT",
    ),
    "sam3d_generation": (
        "PROMPT2SCENE_SAM3D_GENERATION_BASE_URL",
        "PROMPT2SCENE_SAM3D_GENERATION_PORT",
    ),
    "zimage": (
        "PROMPT2SCENE_ZIMAGE_BASE_URL",
        "PROMPT2SCENE_ZIMAGE_PORT",
    ),
}
_SERVICE_HOST_ENV_NAME = "PROMPT2SCENE_SERVICE_HOST"
_CLIENT_CONFIG_LOCK = threading.RLock()


def build_prompt2scene_llm_config(
    config_path: Path | None = None,
) -> OpenAICompatibleLLMCfg:
    """Build Prompt2Scene's LLM config from the shared action-pipeline config."""
    from embodichain.gen_sim.prompt2scene.llms.config import OpenAICompatibleLLMCfg

    settings = _load_prompt2scene_llm_settings(config_path)
    shared_cfg = get_openai_compatible_llm_config(
        required=True,
        require_base_url=True,
    )
    return OpenAICompatibleLLMCfg(
        api_key=str(shared_cfg["api_key"]),
        model=str(shared_cfg["model"]),
        base_url=str(shared_cfg["base_url"]),
        default_query=settings["default_query"],
        max_attempts=settings["max_attempts"],
    )


def write_prompt2scene_client_config(
    output_root: Path,
    *,
    source_config_path: Path | None = None,
) -> Path:
    """Write Prompt2Scene's resolved service configuration under the output root."""
    source_path = (
        (source_config_path or DEFAULT_PROMPT2SCENE_CLIENT_CONFIG_PATH)
        .expanduser()
        .resolve()
    )
    raw_config = _load_json_object(source_path, "Prompt2Scene client config")
    local_env = load_local_env_values()

    for section_name in _SERVICE_ENV_NAMES:
        section = raw_config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(
                f"Prompt2Scene client config section {section_name!r} must be an object."
            )
        section["base_url"] = _resolve_service_base_url(section_name, local_env)

    runtime_dir = output_root.expanduser().resolve() / ".action_agent_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / "prompt2scene_client_config.json"
    runtime_path.write_text(
        json.dumps(raw_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return runtime_path


@contextmanager
def use_prompt2scene_client_config(config_path: Path) -> Iterator[None]:
    """Temporarily direct Prompt2Scene clients to a generated config file."""
    from embodichain.gen_sim.prompt2scene.agent_tools.clients import config

    resolved_path = config_path.expanduser().resolve()
    with _CLIENT_CONFIG_LOCK:
        previous_path = config.DEFAULT_CLIENT_CONFIG_PATH
        config.DEFAULT_CLIENT_CONFIG_PATH = resolved_path
        try:
            yield
        finally:
            config.DEFAULT_CLIENT_CONFIG_PATH = previous_path


def _load_prompt2scene_llm_settings(config_path: Path | None) -> dict[str, Any]:
    path = (config_path or DEFAULT_PROMPT2SCENE_LLM_CONFIG_PATH).expanduser().resolve()
    raw_config = _load_json_object(path, "Prompt2Scene LLM config")
    llm = raw_config.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("Prompt2Scene LLM config key 'llm' must be an object.")
    settings = llm.get("openai_compatible")
    if not isinstance(settings, dict):
        raise ValueError(
            "Prompt2Scene LLM config key 'llm.openai_compatible' must be an object."
        )
    default_query = settings.get("default_query", {})
    if not isinstance(default_query, dict):
        raise ValueError(
            "Prompt2Scene LLM config key 'default_query' must be an object."
        )
    return {
        "default_query": default_query,
        "max_attempts": _load_positive_int(settings.get("max_attempts", 3)),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    raw_config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return raw_config


def _resolve_service_base_url(
    section_name: str,
    local_env: dict[str, str],
) -> str:
    base_url_env_name, port_env_name = _SERVICE_ENV_NAMES[section_name]
    base_url = _environment_value(base_url_env_name, local_env)
    if base_url:
        return _validate_base_url(base_url, base_url_env_name)

    host = _environment_value(_SERVICE_HOST_ENV_NAME, local_env)
    port = _environment_value(port_env_name, local_env)
    missing = [
        name
        for name, value in (
            (_SERVICE_HOST_ENV_NAME, host),
            (port_env_name, port),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Prompt2Scene service {section_name!r} is not configured. Set "
            f"{base_url_env_name} or set {', '.join(missing)}."
        )
    if not str(port).isdigit():
        raise ValueError(
            f"Environment variable {port_env_name} must be an integer port."
        )
    return _validate_base_url(f"http://{host}:{port}", section_name)


def _environment_value(name: str, local_env: dict[str, str]) -> str:
    return str(os.getenv(name) or local_env.get(name, "")).strip()


def _validate_base_url(value: str, source: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{source} must be an absolute HTTP(S) URL: {value!r}")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{source} contains an invalid port: {value!r}") from exc
    return normalized


def _load_positive_int(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prompt2Scene LLM config max_attempts must be an integer."
        ) from exc
    if parsed < 1:
        raise ValueError("Prompt2Scene LLM config max_attempts must be >= 1.")
    return parsed
