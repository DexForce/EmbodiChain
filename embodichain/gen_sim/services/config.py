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

"""Opt-in JSON profiles; legacy resolution remains a separate path."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any
from .errors import ServiceConfigurationError
from ..env import _iter_env_file

__all__: list[str] = []
DEFAULT_JSON = (
    Path(__file__).resolve().parents[1] / "asset_engine/configs/gen_config.json"
)
DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def read_env(path: Path) -> dict[str, str]:
    """Read an optional dotenv without mutating process environment."""
    if not path.is_file():
        return {}
    try:
        return dict(_iter_env_file(path))
    except ValueError as error:
        raise ServiceConfigurationError(str(error)) from error


def _load_services_config(
    path: str | Path | None = None,
) -> tuple[GenSimServicesCfg | None, str]:
    """Discover one configuration file; selected files never fall through."""
    if path is not None:
        selected, source, required = Path(path).expanduser(), "file", True
    elif "EMBODICHAIN_MODEL_SERVICES_CONFIG" in os.environ:
        value = os.environ["EMBODICHAIN_MODEL_SERVICES_CONFIG"].strip()
        if not value:
            raise ServiceConfigurationError(
                "EMBODICHAIN_MODEL_SERVICES_CONFIG must not be empty"
            )
        selected, source, required = Path(value).expanduser(), "environment", True
    else:
        config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
        user_file = (
            Path(config_home).expanduser() if config_home else Path.home() / ".config"
        ) / "embodichain/model_services.json"
        # lexists semantics: a broken user-file symlink is an invalid selected
        # configuration, not permission to silently switch providers.
        if user_file.exists() or user_file.is_symlink():
            selected, source, required = user_file, "user", True
        else:
            selected, source, required = DEFAULT_JSON, "json", False
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if not required and not selected.is_symlink():
            return None, "legacy"
        raise ServiceConfigurationError(
            f"Model service configuration not found: {selected}"
        ) from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ServiceConfigurationError(
            f"Cannot read valid model service JSON: {selected}"
        ) from None
    if not isinstance(raw, dict):
        raise ServiceConfigurationError(
            f"Model service configuration must be an object: {selected}"
        )
    if "model_services" not in raw:
        if required:
            raise ServiceConfigurationError(
                f"Selected configuration requires model_services: {selected}"
            )
        return None, "legacy"
    section = raw["model_services"]
    if not isinstance(section, dict) or set(section) != {"profiles", "roles"}:
        raise ServiceConfigurationError("model_services requires profiles and roles")
    from ._cfg import GenSimServicesCfg

    return GenSimServicesCfg(**deepcopy(section)), source


def load_services_config(path: str | Path | None = None) -> GenSimServicesCfg | None:
    """Load explicit JSON or discover environment, user, then bundled settings.

    Args:
        path: Explicit JSON file. Otherwise consult
            ``EMBODICHAIN_MODEL_SERVICES_CONFIG``, then
            ``$XDG_CONFIG_HOME/embodichain/model_services.json`` (defaulting to
            ``~/.config``), then the bundled GenSim JSON.

    Returns:
        Validated profiles, or ``None`` when bundled settings select legacy mode.

    Raises:
        ServiceConfigurationError: A selected external file is missing, unreadable
            or invalid. External files must contain ``model_services``.
    """
    return _load_services_config(path)[0]


def resolve_profile(
    profile: Mapping[str, Any], env_path: Path | None = None
) -> dict[str, Any]:
    """Resolve only explicit references, using process values before dotenv."""
    env = {
        **read_env(
            (
                env_path or Path(os.getenv("EMBODICHAIN_ENV_FILE", str(DEFAULT_ENV)))
            ).expanduser()
        ),
        **os.environ,
    }

    def resolve(value: Any) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"env"}:
                name = value["env"]
                if name not in env or not env[name]:
                    raise ServiceConfigurationError(
                        f"Missing service environment variable: {name}"
                    )
                return env[name]
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    result = resolve(deepcopy(dict(profile)))
    if "api_key_env" in result:
        name = result.pop("api_key_env")
        if not env.get(name):
            raise ServiceConfigurationError(
                f"Missing service environment variable: {name}"
            )
        result["api_key"] = env[name]
    for key, default in (("timeout_s", 120), ("max_attempts", 1)):
        value = result.get(key, default)
        if isinstance(value, str):
            try:
                value = int(value) if key == "max_attempts" else float(value)
            except ValueError:
                raise ServiceConfigurationError(f"{key} must be numeric") from None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or (key == "max_attempts" and not isinstance(value, int))
        ):
            raise ServiceConfigurationError(f"{key} must be positive")
        result[key] = value
    return result


def __getattr__(name: str) -> Any:
    if name == "GenSimServicesCfg":
        from ._cfg import GenSimServicesCfg

        globals()[name] = GenSimServicesCfg
        return GenSimServicesCfg
    raise AttributeError(name)
