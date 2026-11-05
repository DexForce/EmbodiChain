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
"""Task Engine profile selectors; concrete calibration belongs to robot presets."""

from __future__ import annotations
from typing import Any

__all__ = [
    "canonical_robot_profile",
    "profile_lateral_positions",
    "require_robot_assets",
]


def canonical_robot_profile(profile: str) -> str:
    """Compare the existing CLI alias without rewriting serialized robot IDs."""
    return "tianji_marvin" if profile == "dual_tianji" else profile


_MODEL_IDS = {
    "dual_franka": "dual_franka_robotiq_arg2f_140",
    "dual_franka_robotiq_arg2f_140": "dual_franka_robotiq_arg2f_140",
    "dual_franka_pgi": "dual_franka_robotiq_arg2f_140",
    "dual_ur10": "dual_ur10_robotiq_arg2f_140",
    "tianji_marvin": "tianji_marvin",
}


def _model(profile: str) -> str:
    selected = _MODEL_IDS.get(canonical_robot_profile(profile))
    if selected is None:
        raise ValueError(f"Unsupported source robot deployment profile: {profile!r}")
    return selected


def profile_lateral_positions(
    profile: str, embodiment: dict[str, Any] | None = None
) -> dict[str, float] | None:
    """Delegate declared semantic-resource positions to the concrete model owner.

    Args:
        profile: Explicit source deployment selection, including its legacy alias.
        embodiment: Optional already-selected complete component declaration.

    Returns:
        Source-equivalent lateral resource metadata, or None for an unselected
        model route. No Robot, sensor or simulator is initialized.
    """
    from embodichain_robots.task_program import task_robot_lateral_positions

    return task_robot_lateral_positions(_model(profile), embodiment)


def require_robot_assets(profile: str) -> None:
    """Preflight the selected model's source assets without creating a runtime.

    Args:
        profile: Explicit deployment selection; unrelated CLI aliases reject.
    """
    from importlib import import_module

    # Resolve the actual package even when a model submodule was previously
    # cached; a disabled/missing installation cannot pass CLI preflight.
    import_module("embodichain_robots")
    from embodichain_robots.task_program import require_task_robot_assets

    require_task_robot_assets(_model(profile))
