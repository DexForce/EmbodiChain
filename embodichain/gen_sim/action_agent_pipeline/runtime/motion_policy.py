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

"""Resolve named Seed motion policies for one runtime robot profile."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    MOTION_POLICY_VERSION,
)

__all__ = [
    "MOTION_POLICY_REGISTRY",
    "resolve_motion_policy",
]

_COMMON_POLICIES: dict[str, dict[str, Any]] = {
    "default_pickup": {
        "pre_grasp_distance": 0.08,
        "lift_height": 0.30,
        "sample_interval": 45,
    },
    "default_transport": {
        "sample_interval": 45,
        "relation_distance": 0.18,
        "hover_height": 0.10,
        "line_spacing": 0.14,
        "surface_clearance": 0.005,
        "postcondition_tolerance": 0.08,
    },
    "default_release": {
        "sample_interval": 15,
        "lift_height": 0.0,
        "post_hold_steps": 0,
    },
    "default_retreat": {
        "sample_interval": 20,
        "retreat_height": 0.30,
    },
    "default_home": {"sample_interval": 30},
}

MOTION_POLICY_REGISTRY: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    profile: deepcopy(_COMMON_POLICIES)
    for profile in ("dual_franka", "dual_ur3", "dual_ur5", "dual_ur10")
}


def resolve_motion_policy(
    robot_profile: str,
    policy_id: str,
    *,
    policy_version: str = MOTION_POLICY_VERSION,
) -> dict[str, Any]:
    """Return one validated policy snapshot for runtime recording."""
    if policy_version != MOTION_POLICY_VERSION:
        raise ValueError(
            f"Unsupported motion policy version {policy_version!r}; expected "
            f"{MOTION_POLICY_VERSION!r}."
        )
    profile_policies = MOTION_POLICY_REGISTRY.get(str(robot_profile))
    if profile_policies is None:
        raise ValueError(
            f"No action-agent motion policies are registered for robot profile "
            f"{robot_profile!r}."
        )
    policy = profile_policies.get(str(policy_id))
    if policy is None:
        raise ValueError(
            f"Motion policy {policy_id!r} is not registered for robot profile "
            f"{robot_profile!r}."
        )
    return deepcopy(dict(policy))
