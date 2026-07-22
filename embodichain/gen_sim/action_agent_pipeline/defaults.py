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
from typing import Any

from embodichain.utils.utility import load_config

__all__ = [
    "ACTION_AGENT_CONFIG_DEFAULTS",
    "DEFAULT_MAX_EPISODES",
    "DEFAULT_MAX_EPISODE_STEPS",
    "DEFAULT_SURFACE_RELEASE_CLEARANCE",
    "DEFAULT_TARGET_BODY_SCALE",
    "DEFAULT_TASK_NAME",
    "generation_defaults_section",
]

_DEFAULTS_PATH = (
    Path(__file__).resolve().parent / "generation" / "action_agent_config_defaults.yaml"
)
ACTION_AGENT_CONFIG_DEFAULTS: dict[str, Any] = load_config(_DEFAULTS_PATH)


def generation_defaults_section(name: str) -> dict[str, Any]:
    """Return one required section from the generation defaults.

    Args:
        name: Top-level YAML section name.

    Returns:
        The requested configuration section.

    Raises:
        ValueError: If the section is missing or is not a mapping.
    """
    section = ACTION_AGENT_CONFIG_DEFAULTS.get(name)
    if not isinstance(section, dict):
        raise ValueError(
            f"Generation defaults section {name!r} must be a mapping in "
            f"{_DEFAULTS_PATH}."
        )
    return section


_TASK_DEFAULTS = generation_defaults_section("task")
_GEOMETRY_DEFAULTS = generation_defaults_section("geometry")

DEFAULT_TASK_NAME = str(_TASK_DEFAULTS["default_name"])
DEFAULT_MAX_EPISODES = int(_TASK_DEFAULTS["max_episodes"])
DEFAULT_MAX_EPISODE_STEPS = int(_TASK_DEFAULTS["max_episode_steps"])
DEFAULT_TARGET_BODY_SCALE = float(_GEOMETRY_DEFAULTS["target_body_scale"])
DEFAULT_SURFACE_RELEASE_CLEARANCE = float(
    _GEOMETRY_DEFAULTS["surface_release_clearance"]
)
