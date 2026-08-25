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
    "ACTION_ENGINE_VISER_PORT",
    "ARTICULATION_SERVER_BASE_URL",
    "ARTICULATION_SERVER_POLL_INTERVAL_S",
    "ARTICULATION_SERVER_TASK_TIMEOUT_S",
    "ARTICULATION_SERVER_TIMEOUT_S",
    "ARTICRAFT_CONDA_ENV",
    "ARTICRAFT_OUTPUT_ROOT",
    "ARTICRAFT_REPOSITORY_URL",
    "ARTICRAFT_ROOT",
    "DIRECT_NO_PROXY_VALUE",
    "EMBODICHAIN_ROOT",
    "GRADIO_AUTH_PASSWORD",
    "GRADIO_AUTH_USERNAME",
    "PROXY_ENV_KEYS",
    "SCENE_ENGINE_VISER_PORT",
    "SERVER_NAME",
    "SERVER_PORT",
    "SIMREADY_OPENAI_API_KEY",
    "SIMREADY_OPENAI_BASE_URL",
    "SIMREADY_OPENAI_MODEL",
    "build_gradio_allowed_paths",
    "build_gradio_blocked_paths",
    "configure_direct_network_env",
    "configure_simready_llm_env",
    "get_inherited_network_env",
    "get_gradio_auth",
    "validate_gradio_artifact_root",
]

load_gen_sim_env()

APP_ROOT = Path(__file__).resolve().parent
GEN_SIM_ROOT = APP_ROOT / ".gen_sim"
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
_NETWORK_ENV_KEYS = (*PROXY_ENV_KEYS, "NO_PROXY", "no_proxy")
_INHERITED_NETWORK_ENV = {
    key: value for key in _NETWORK_ENV_KEYS if (value := os.environ.get(key))
}


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
    "ARTICRAFT_REPOSITORY_URL", "https://github.com/XuanchaoPENG/Articraft.git"
)
ARTICRAFT_CONDA_ENV = _getenv("ARTICRAFT_CONDA_ENV", "articraft")
ARTICRAFT_OUTPUT_ROOT = Path(
    _getenv("ARTICRAFT_OUTPUT_ROOT", str(GEN_SIM_ROOT / "articraft"))
).expanduser()
ARTICULATION_SERVER_BASE_URL = _getenv("ARTICULATION_SERVER_BASE_URL", "")
ARTICULATION_SERVER_TIMEOUT_S = float(_getenv("ARTICULATION_SERVER_TIMEOUT_S", "30"))
ARTICULATION_SERVER_TASK_TIMEOUT_S = float(
    _getenv("ARTICULATION_SERVER_TASK_TIMEOUT_S", "7200")
)
ARTICULATION_SERVER_POLL_INTERVAL_S = float(
    _getenv("ARTICULATION_SERVER_POLL_INTERVAL_S", "1")
)
SCENE_ENGINE_VISER_PORT = int(_getenv("SCENE_ENGINE_VISER_PORT", "8080"))
ACTION_ENGINE_VISER_PORT = int(_getenv("ACTION_ENGINE_VISER_PORT", "8082"))
SERVER_NAME = _getenv("GRADIO_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(_getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_AUTH_USERNAME = _getenv("GRADIO_AUTH_USERNAME", "")
GRADIO_AUTH_PASSWORD = _getenv("GRADIO_AUTH_PASSWORD", "")
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


def get_inherited_network_env() -> dict[str, str]:
    """Return the network environment captured before Gradio forces direct access.

    Articulation's Codex process uses this snapshot because it may require the
    launching shell's proxy configuration to reach its remote model provider.
    Other GenSim pipelines continue to use :func:`configure_direct_network_env`.

    Returns:
        A copy of the inherited proxy and proxy-bypass variables.
    """
    return _INHERITED_NETWORK_ENV.copy()


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


def get_gradio_auth(
    server_name: str = SERVER_NAME,
    username: str = GRADIO_AUTH_USERNAME,
    password: str = GRADIO_AUTH_PASSWORD,
) -> tuple[str, str] | None:
    """Validate deployment exposure and return Gradio credentials.

    Args:
        server_name: Interface address used by the Gradio server.
        username: Optional HTTP basic-auth username.
        password: Optional HTTP basic-auth password.

    Returns:
        A ``(username, password)`` tuple, or ``None`` for a local-only server.

    Raises:
        ValueError: If credentials are incomplete or a non-loopback server has
            no authentication configured.
    """
    has_username = bool(username)
    has_password = bool(password)
    if has_username != has_password:
        raise ValueError(
            "Set both GRADIO_AUTH_USERNAME and GRADIO_AUTH_PASSWORD, or neither."
        )
    if has_username and has_password:
        return username, password
    if server_name.strip().lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "A non-loopback GRADIO_SERVER_NAME requires Gradio authentication."
        )
    return None


def build_gradio_allowed_paths(*roots: Path) -> list[str]:
    """Resolve the explicit static and generated roots Gradio may serve.

    Args:
        *roots: Static-resource or generated-artifact directories.

    Returns:
        Sorted, de-duplicated absolute path strings.
    """
    return sorted({str(path.expanduser().resolve()) for path in roots})


def build_gradio_blocked_paths(env_path: Path | None) -> list[str]:
    """Resolve repository metadata and dotenv paths Gradio must never serve.

    Args:
        env_path: Active shared dotenv path, if one exists.

    Returns:
        Sorted, de-duplicated absolute path strings.
    """
    blocked = {
        EMBODICHAIN_ROOT / ".env",
        EMBODICHAIN_ROOT / ".git",
        EMBODICHAIN_ROOT / "embodichain" / "gen_sim" / ".env",
    }
    if env_path is not None:
        blocked.add(env_path)
    return build_gradio_allowed_paths(*blocked)


def validate_gradio_artifact_root(root: Path) -> Path:
    """Reject an artifact setting broad enough to expose the repository.

    Args:
        root: Configured directory containing generated artifacts.

    Returns:
        The normalized artifact directory.

    Raises:
        ValueError: If the directory is the repository or one of its ancestors.
    """
    resolved_root = root.expanduser().resolve()
    repository = EMBODICHAIN_ROOT.resolve()
    if resolved_root == repository or repository.is_relative_to(resolved_root):
        raise ValueError(
            "ARTICRAFT_OUTPUT_ROOT must be a dedicated artifact directory, not "
            "the EmbodiChain repository or one of its parents."
        )
    return resolved_root


# Gradio imports its HTTP client during application module loading. Remove the
# same unsupported or credential-bearing proxies that child pipelines exclude
# before importing any view module.
configure_direct_network_env()
