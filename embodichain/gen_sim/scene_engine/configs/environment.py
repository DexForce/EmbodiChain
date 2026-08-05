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

from pathlib import Path

_SCENE_ENGINE_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def read_scene_engine_env_values(*keys: str) -> dict[str, str]:
    """Read only the requested Scene Engine settings from ``gen_sim/.env``."""
    if not _SCENE_ENGINE_ENV_PATH.is_file():
        raise FileNotFoundError(
            f"Scene Engine .env file not found: {_SCENE_ENGINE_ENV_PATH}"
        )

    requested_keys = set(keys)
    values: dict[str, str] = {}
    for raw_line in _SCENE_ENGINE_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        if key not in requested_keys:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value

    missing_keys = [key for key in keys if key not in values]
    if missing_keys:
        raise ValueError(f"Missing required Scene Engine .env keys: {missing_keys}")
    return values
