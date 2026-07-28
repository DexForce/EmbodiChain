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

"""Load defaults shared by config generation, runtime, and environment adapters.

The backing YAML is package-owned rather than generation-owned because several
sections are runtime contracts. Keeping the loader beside the resource makes
that ownership explicit and prevents lower layers from depending on a
generation-stage file path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodichain.utils.utility import load_config

__all__ = [
    "ACTION_AGENT_CONFIG_DEFAULTS",
    "ACTION_AGENT_DEFAULTS",
    "DEFAULT_GENERATED_CONFIG_TASK_NAME",
    "DEFAULT_MAX_EPISODES",
    "DEFAULT_MAX_EPISODE_STEPS",
    "ROBOTIQ_ARG2F_140_CLOSE_QPOS",
    "ROBOTIQ_ARG2F_140_OPEN_QPOS",
    "DEFAULT_SURFACE_RELEASE_CLEARANCE",
    "DEFAULT_TARGET_BODY_SCALE",
    "DEFAULT_TASK_NAME",
    "defaults_section",
    "generation_defaults_section",
]

_DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")
ACTION_AGENT_DEFAULTS: dict[str, Any] = load_config(_DEFAULTS_PATH)

# Historical callers imported this generation-oriented name. It remains an
# identity alias while new pipeline code uses the package-owned terminology.
ACTION_AGENT_CONFIG_DEFAULTS = ACTION_AGENT_DEFAULTS


def defaults_section(name: str) -> dict[str, Any]:
    """Return one required section from the package-wide defaults.

    Args:
        name: Top-level YAML section name.

    Returns:
        The requested configuration section.

    Raises:
        ValueError: If the section is missing or is not a mapping.
    """
    section = ACTION_AGENT_DEFAULTS.get(name)
    if not isinstance(section, dict):
        raise ValueError(
            f"Action-agent defaults section {name!r} must be a mapping in "
            f"{_DEFAULTS_PATH}."
        )
    return section


# Preserve both call behavior and object identity for legacy imports.
generation_defaults_section = defaults_section

_TASK_DEFAULTS = defaults_section("task")
_GEOMETRY_DEFAULTS = defaults_section("geometry")
_ROBOT_FALLBACKS = defaults_section("robot_fallbacks")
_ROBOTIQ_ARG2F_140_FALLBACKS = _ROBOT_FALLBACKS.get("robotiq_arg2f_140")
if not isinstance(_ROBOTIQ_ARG2F_140_FALLBACKS, dict):
    raise ValueError(
        "Action-agent robot_fallbacks.robotiq_arg2f_140 must be a mapping in "
        f"{_DEFAULTS_PATH}."
    )


def _six_element_float_tuple(key: str) -> tuple[float, ...]:
    """Load one Robotiq state while preserving the historical tuple contract."""
    values = _ROBOTIQ_ARG2F_140_FALLBACKS.get(key)
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(
            "Action-agent robot_fallbacks.robotiq_arg2f_140."
            f"{key} must contain exactly six values in {_DEFAULTS_PATH}."
        )
    return tuple(float(value) for value in values)


DEFAULT_GENERATED_CONFIG_TASK_NAME = str(_TASK_DEFAULTS["default_name"])
DEFAULT_MAX_EPISODES = int(_TASK_DEFAULTS["max_episodes"])
DEFAULT_MAX_EPISODE_STEPS = int(_TASK_DEFAULTS["max_episode_steps"])
DEFAULT_TARGET_BODY_SCALE = float(_GEOMETRY_DEFAULTS["target_body_scale"])
DEFAULT_SURFACE_RELEASE_CLEARANCE = float(
    _GEOMETRY_DEFAULTS["surface_release_clearance"]
)
ROBOTIQ_ARG2F_140_OPEN_QPOS = _six_element_float_tuple("open_qpos")
ROBOTIQ_ARG2F_140_CLOSE_QPOS = _six_element_float_tuple("close_qpos")

# The short name is a public compatibility contract. Internally, use the
# explicit generated-config name so it cannot be confused with the pipeline
# CLI's independently configurable task-name default.
DEFAULT_TASK_NAME = DEFAULT_GENERATED_CONFIG_TASK_NAME
