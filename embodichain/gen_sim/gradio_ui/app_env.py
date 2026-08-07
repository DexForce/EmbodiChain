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

"""Environment-backed deployment settings for the Gradio application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from embodichain.gen_sim.env import get_embodichain_root, load_gen_sim_env

__all__ = [
    "ARTICRAFT_CONDA_ENV",
    "ARTICRAFT_OUTPUT_ROOT",
    "ARTICRAFT_REPOSITORY_URL",
    "ARTICRAFT_ROOT",
    "ARTICRAFT_VISER_PORT",
    "DIRECT_NO_PROXY_VALUE",
    "EMBODICHAIN_ROOT",
    "PROXY_ENV_KEYS",
    "SCENE_ENGINE_VISER_PORT",
    "SERVER_NAME",
    "SERVER_PORT",
    "SIMREADY_OPENAI_API_KEY",
    "SIMREADY_OPENAI_BASE_URL",
    "SIMREADY_OPENAI_MODEL",
    "configure_direct_network_env",
    "configure_simready_llm_env",
]

load_gen_sim_env()

APP_ROOT = Path(__file__).resolve().parent
DEBUG_ENGINE_ROOT = APP_ROOT / ".debug_engine"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
)
DIRECT_NO_PROXY_VALUE = "*"


def _getenv(name: str, default: str) -> str:
    """Read a non-empty shared ``.env`` value, falling back to ``default``."""
    return os.environ.get(name) or default


# The repository root must follow this checkout, not a machine-specific .env
# value. Its path is shared with child processes through their working
# directory, so deriving it once here keeps every Debug workflow relocatable.
EMBODICHAIN_ROOT = get_embodichain_root()
ARTICRAFT_ROOT = Path(
    _getenv("ARTICRAFT_ROOT", str(APP_ROOT / ".articraft"))
).expanduser()
ARTICRAFT_REPOSITORY_URL = _getenv(
    "ARTICRAFT_REPOSITORY_URL", "https://github.com/mattzh72/articraft.git"
)
ARTICRAFT_CONDA_ENV = _getenv("ARTICRAFT_CONDA_ENV", "articraft")
ARTICRAFT_OUTPUT_ROOT = Path(
    _getenv("ARTICRAFT_OUTPUT_ROOT", str(DEBUG_ENGINE_ROOT / "articraft"))
).expanduser()
SCENE_ENGINE_VISER_PORT = int(_getenv("SCENE_ENGINE_VISER_PORT", "8080"))
ARTICRAFT_VISER_PORT = int(_getenv("ARTICRAFT_VISER_PORT", "8081"))
SERVER_NAME = _getenv("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(_getenv("GRADIO_SERVER_PORT", "7860"))
SIMREADY_OPENAI_API_KEY = _getenv("SIMREADY_OPENAI_API_KEY", "")
SIMREADY_OPENAI_MODEL = _getenv("SIMREADY_OPENAI_MODEL", "")
SIMREADY_OPENAI_BASE_URL = _getenv("SIMREADY_OPENAI_BASE_URL", "")


def configure_direct_network_env(env: Any = None) -> None:
    """Disable proxy inheritance for local pipeline and Gradio processes."""
    if env is None:
        env = os.environ
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = DIRECT_NO_PROXY_VALUE
    env["no_proxy"] = DIRECT_NO_PROXY_VALUE
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def configure_simready_llm_env(env: Any = None) -> None:
    """Map app-level SimReady settings to the upstream CLI's environment."""
    if env is None:
        env = os.environ
    configured_values = {
        "OPENAI_API_KEY": SIMREADY_OPENAI_API_KEY,
        "OPENAI_MODEL": SIMREADY_OPENAI_MODEL,
        "OPENAI_BASE_URL": SIMREADY_OPENAI_BASE_URL,
    }
    for key, value in configured_values.items():
        if value:
            env[key] = value
