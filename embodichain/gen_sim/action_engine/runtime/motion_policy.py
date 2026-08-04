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

__all__ = ["resolve_motion_policy"]

_PROFILE_ALIASES = dict(
    franka="dual_franka", ur3="dual_ur3", ur5="dual_ur5", ur10="dual_ur10"
)


def resolve_motion_policy(
    robot_profile: str,
    policy_name: str,
    *,
    policies: Mapping[str, Mapping[str, Any]] | None = None,
    program_overrides: Mapping[str, Any] | None = None,
    inline_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a detached policy resolved for one robot profile."""
    profile = _PROFILE_ALIASES.get(str(robot_profile), str(robot_profile))
    resolved_policies = (
        default_runtime_policy(profile).motion_policies
        if policies is None
        else policies
    )
    if policy_name not in resolved_policies:
        raise ValueError(f"Unknown Action Engine motion policy {policy_name!r}.")

    policy = deepcopy(dict(resolved_policies[policy_name]))
    if program_overrides is not None:
        policy.update(deepcopy(dict(program_overrides)))
    if inline_overrides is not None:
        policy.update(deepcopy(dict(inline_overrides)))
    return policy
