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

"""Compatibility import for the Asset Engine articulation client.

New code imports the client from ``simready_pipeline.clients``.  This module
keeps the former Scene Engine path working while preserving the existing
``SCENE_ENGINE_ARTICULATED_GENERATION_*`` environment variable names for one
migration cycle.
"""

from __future__ import annotations

from embodichain.gen_sim.scene_engine.configs.environment import (
    read_scene_engine_env_values,
)
from embodichain.gen_sim.simready_pipeline.clients import (
    ArticulatedGenerationClient as _AssetArticulatedGenerationClient,
)
from embodichain.gen_sim.simready_pipeline.clients.articulated_generation import (
    _validate_articulated_usdc,  # noqa: F401
)

__all__ = ["ArticulatedGenerationClient"]


class ArticulatedGenerationClient(_AssetArticulatedGenerationClient):
    """Backward-compatible Scene Engine alias for the Asset Engine client."""

    @classmethod
    def from_dotenv(cls) -> ArticulatedGenerationClient:
        """Load legacy Scene Engine keys and construct the migrated client."""
        values = read_scene_engine_env_values(
            "SCENE_ENGINE_ARTICULATED_GENERATION_BASE_URL",
            "SCENE_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S",
            "SCENE_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS",
            "SCENE_ENGINE_ARTICULATED_GENERATION_HEALTH_PATH",
            "SCENE_ENGINE_ARTICULATED_GENERATION_GENERATE_PATH",
        )
        return cls(
            base_url=values["SCENE_ENGINE_ARTICULATED_GENERATION_BASE_URL"].strip(),
            timeout_s=_positive_int(
                values["SCENE_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S"],
                "SCENE_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S",
            ),
            max_attempts=_positive_int(
                values["SCENE_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS"],
                "SCENE_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS",
            ),
            health_path=values[
                "SCENE_ENGINE_ARTICULATED_GENERATION_HEALTH_PATH"
            ].strip(),
            generate_path=values[
                "SCENE_ENGINE_ARTICULATED_GENERATION_GENERATE_PATH"
            ].strip(),
        )


def _positive_int(value: str, key: str) -> int:
    """Validate a positive legacy environment integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if parsed < 1:
        raise ValueError(f"{key} must be at least 1.")
    return parsed
