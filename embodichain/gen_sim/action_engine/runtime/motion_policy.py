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

from collections.abc import Mapping
import math
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.runtime.motion_policy import (
    MOTION_POLICY_REGISTRY as _PIPELINE_MOTION_POLICIES,
)

__all__ = ["resolve_motion_policy"]

# fmt: off
_COMMON_POLICY_EXTENSIONS: dict[str, dict[str, Any]] = {
    "default_transport": {"line_axis_tolerance": 0.06, "line_perpendicular_tolerance": 0.06},
    "upright_in_place_transport": {"upright_xy_tolerance": 0.05, "upright_max_tilt": math.pi / 12.0},
    "default_release": {"cartesian_waypoint_count": 4},
    "default_retreat": {"postcondition_tolerance": 0.05},
    "upright_in_place_retreat": {"postcondition_tolerance": 0.05},
    "default_home": {"postcondition_tolerance": 0.05},
    "default_hold": {"sample_interval": 10, "postcondition_tolerance": 0.05},
    "default_press": {"sample_interval": 80, "press_depth": 0.004, "postcondition_tolerance": 0.03},
    "default_coordinated_transport": {"sample_interval": 120, "object_motion_keyframes": 6, "pre_grasp_distance": 0.10, "lift_height": 0.08, "postcondition_tolerance": 0.06},
    "default_coordinated_place": {"sample_interval": 100, "hand_interp_steps": 10, "hold_steps": 4, "retreat_steps": 16, "postcondition_tolerance": 0.06},
}

_PROFILE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "dual_franka": {},
    "dual_ur3": {},
    "dual_ur5": {},
    "dual_ur10": {},
}
# fmt: on

_PROFILE_ALIASES = dict(
    franka="dual_franka", ur3="dual_ur3", ur5="dual_ur5", ur10="dual_ur10"
)


def resolve_motion_policy(
    robot_profile: str,
    policy_name: str,
    *,
    program_overrides: Mapping[str, Any] | None = None,
    inline_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a detached policy resolved for one robot profile."""
    profile = _PROFILE_ALIASES.get(str(robot_profile), str(robot_profile))
    if profile not in _PROFILE_OVERRIDES:
        raise ValueError(f"Unsupported Action Engine robot profile {robot_profile!r}.")
    profile_policies = _PIPELINE_MOTION_POLICIES.get(profile, {})
    if (
        policy_name not in profile_policies
        and policy_name not in _COMMON_POLICY_EXTENSIONS
    ):
        raise ValueError(f"Unknown Action Engine motion policy {policy_name!r}.")

    policy = deepcopy(dict(profile_policies.get(policy_name, {})))
    policy.update(deepcopy(_COMMON_POLICY_EXTENSIONS.get(policy_name, {})))
    policy.update(deepcopy(_PROFILE_OVERRIDES[profile].get(policy_name, {})))
    if program_overrides is not None:
        policy.update(deepcopy(dict(program_overrides)))
    if inline_overrides is not None:
        policy.update(deepcopy(dict(inline_overrides)))
    return policy
