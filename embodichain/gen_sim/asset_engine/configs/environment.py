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

from embodichain.gen_sim.environment import _read_gen_sim_env_values

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
    aliases = {
        key: key.replace(
            "ASSET_ENGINE_ARTICULATED_GENERATION_",
            "SCENE_ENGINE_ARTICULATED_GENERATION_",
        )
        for key in keys
    }
    return _read_gen_sim_env_values(
        *keys,
        env_path=_ASSET_ENGINE_ENV_PATH,
        aliases=aliases,
        owner="Asset Engine",
    )
