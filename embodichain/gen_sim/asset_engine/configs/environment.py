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

"""Environment settings owned by the Asset Engine."""

from __future__ import annotations

from pathlib import Path

__all__ = ["read_asset_engine_env_values"]

_ASSET_ENGINE_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def read_asset_engine_env_values(*keys: str) -> dict[str, str]:
    """Read requested Asset Engine settings from ``gen_sim/.env``.

    ``SCENE_ENGINE_ARTICULATED_GENERATION_*`` remains accepted as a legacy
    alias for the articulation service during the ownership migration.

    Args:
        keys: Environment keys to load.

    Returns:
        Mapping of requested keys to their configured values.

    Raises:
        FileNotFoundError: If the shared ``gen_sim/.env`` file is absent.
        ValueError: If any requested key is missing.
    """
    if not _ASSET_ENGINE_ENV_PATH.is_file():
        raise FileNotFoundError(
            f"Asset Engine .env file not found: {_ASSET_ENGINE_ENV_PATH}"
        )

    values: dict[str, str] = {}
    for raw_line in _ASSET_ENGINE_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value

    resolved: dict[str, str] = {}
    for key in keys:
        legacy_key = key.replace(
            "ASSET_ENGINE_ARTICULATED_GENERATION_",
            "SCENE_ENGINE_ARTICULATED_GENERATION_",
        )
        if key in values:
            resolved[key] = values[key]
        elif legacy_key in values:
            resolved[key] = values[legacy_key]

    missing_keys = [key for key in keys if key not in resolved]
    if missing_keys:
        raise ValueError(f"Missing required Asset Engine .env keys: {missing_keys}")
    return resolved
