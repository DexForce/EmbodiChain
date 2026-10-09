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

"""Shared parser for the GenSim ``.env`` file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

__all__: list[str] = []

_DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _read_gen_sim_env_values(
    *keys: str,
    env_path: str | Path = _DEFAULT_ENV_PATH,
    aliases: Mapping[str, str] | None = None,
    owner: str = "GenSim",
) -> dict[str, str]:
    """Read requested values and resolve optional legacy key aliases."""
    path = Path(env_path)
    if not path.is_file():
        raise FileNotFoundError(f"{owner} .env file not found: {path}")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value

    aliases = {} if aliases is None else dict(aliases)
    resolved: dict[str, str] = {}
    for key in keys:
        if key in values:
            resolved[key] = values[key]
            continue
        legacy_key = aliases.get(key)
        if legacy_key is not None and legacy_key in values:
            resolved[key] = values[legacy_key]

    missing_keys = [key for key in keys if key not in resolved]
    if missing_keys:
        raise ValueError(f"Missing required {owner} .env keys: {missing_keys}")
    return resolved
