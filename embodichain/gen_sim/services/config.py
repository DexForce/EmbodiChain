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

__all__: list[str] = []
DEFAULT_JSON = (
    Path(__file__).resolve().parents[1] / "asset_engine/configs/gen_config.json"
)
DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def read_env(path: Path) -> dict[str, str]:
    """Read an optional dotenv without mutating process environment."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not key.isidentifier():
            raise ServiceConfigurationError(f"Invalid dotenv key at line {number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_services_config(path: str | Path = DEFAULT_JSON) -> GenSimServicesCfg | None:
    """Read the optional model_services section; absence selects legacy mode."""
    selected = Path(path)
    if not selected.is_file():
        return None
    raw = json.loads(selected.read_text())
    if "model_services" not in raw:
        return None
    section = raw["model_services"]
    if not isinstance(section, dict) or set(section) != {"profiles", "roles"}:
        raise ServiceConfigurationError("model_services requires profiles and roles")
    from ._cfg import GenSimServicesCfg

    return GenSimServicesCfg(**deepcopy(section))


def resolve_profile(
    profile: Mapping[str, Any], env_path: Path | None = None
) -> dict[str, Any]:
    """Resolve only explicit references, using process values before dotenv."""
    env = {
        **read_env(
            env_path or Path(os.getenv("EMBODICHAIN_ENV_FILE", str(DEFAULT_ENV)))
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
