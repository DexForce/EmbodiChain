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
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_engine.config import default_runtime_policy
from embodichain.gen_sim.action_engine.domain.motion import validate_motion_policy

__all__ = ["resolve_motion_policy", "with_motion_modifiers"]

_PROFILE_ALIASES = dict(
    franka="dual_franka", ur3="dual_ur3", ur5="dual_ur5", ur10="dual_ur10"
)


def resolve_motion_policy(
    robot_profile: str,
    atomic_action: str,
    policy_spec: Mapping[str, Any],
    *,
    motion_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    motion_modifiers: (
        Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]] | None
    ) = None,
    inline_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an action base policy plus its composable typed modifiers."""
    profile = _PROFILE_ALIASES.get(str(robot_profile), str(robot_profile))
    runtime_policy = (
        default_runtime_policy(profile)
        if motion_defaults is None or motion_modifiers is None
        else None
    )
    defaults = (
        runtime_policy.motion_defaults
        if motion_defaults is None and runtime_policy is not None
        else motion_defaults
    )
    modifiers = (
        runtime_policy.motion_modifiers
        if motion_modifiers is None and runtime_policy is not None
        else motion_modifiers
    )
    if defaults is None or modifiers is None:
        raise ValueError("Motion defaults and modifiers must be provided together.")
    action = str(atomic_action)
    if action not in defaults:
        raise ValueError(f"Unknown Action Engine motion base {action!r}.")

    spec = validate_motion_policy(policy_spec)
    policy = deepcopy(dict(defaults[action]))
    modifier_values: dict[str, Any] = {}
    modifier_sources: dict[str, tuple[str, str]] = {}
    for modifier in spec["modifiers"]:
        modifier_type = modifier["type"]
        mode = modifier["mode"]
        patch = modifiers.get(modifier_type, {}).get(mode, {}).get(action)
        if not isinstance(patch, Mapping):
            raise ValueError(
                f"Motion modifier {(modifier_type, mode)!r} is not supported "
                f"by AtomicAction {action!r}."
            )
        for key, value in patch.items():
            if key in modifier_values and modifier_values[key] != value:
                raise ValueError(
                    f"Motion modifiers {modifier_sources[key]!r} and "
                    f"{(modifier_type, mode)!r} conflict on parameter {key!r}."
                )
            modifier_values[key] = deepcopy(value)
            modifier_sources[key] = (modifier_type, mode)
    policy.update(modifier_values)
    if inline_overrides is not None:
        policy.update(deepcopy(dict(inline_overrides)))
    return policy


def with_motion_modifiers(
    policy_spec: Mapping[str, Any],
    *modifiers: tuple[str, str],
) -> dict[str, Any]:
    """Return a validated policy reference with missing modifiers appended."""
    policy = validate_motion_policy(policy_spec)
    existing = {
        (modifier["type"], modifier["mode"]) for modifier in policy["modifiers"]
    }
    for modifier_type, mode in modifiers:
        if (modifier_type, mode) not in existing:
            policy["modifiers"].append({"type": modifier_type, "mode": mode})
    return validate_motion_policy(policy)
