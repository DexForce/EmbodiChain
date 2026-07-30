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

"""Fail-closed validation for synchronized whole-robot joint paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

__all__ = ["validate_dual_arm_joint_path"]


def validate_dual_arm_joint_path(
    env: Any,
    actions: Sequence[torch.Tensor],
    *,
    payloads: Mapping[str, str | None] | None = None,
) -> tuple[bool, str]:
    """Validate a complete synchronized path before the first simulator step.

    The generic layer checks path integrity and joint limits. Collision
    acceptance is deliberately delegated to the concrete environment because
    only it owns robot link geometry, payload attachment, allowed contacts, and
    scene obstacles. Missing collision capability rejects parallel execution.
    """
    if not actions:
        return False, "empty_joint_path"
    path = torch.stack(
        [torch.as_tensor(action, dtype=torch.float32) for action in actions],
        dim=1,
    )
    if path.ndim != 3 or not bool(torch.isfinite(path).all()):
        return False, "invalid_joint_path"

    robot = getattr(env, "robot", None)
    get_limits = getattr(robot, "get_qpos_limits", None)
    if get_limits is None:
        return False, "joint_limits_unavailable"
    limits = torch.as_tensor(
        get_limits(),
        dtype=path.dtype,
        device=path.device,
    )
    if limits.ndim == 2:
        limits = limits.unsqueeze(0)
    if limits.shape[0] == 1 and path.shape[0] > 1:
        limits = limits.expand(path.shape[0], -1, -1)
    if limits.shape != (path.shape[0], path.shape[2], 2):
        return False, "joint_limit_shape_mismatch"
    if bool(((path < limits[:, None, :, 0]) | (path > limits[:, None, :, 1])).any()):
        return False, "joint_limit_violation"

    checker = _wrapped_attr(env, "validate_dual_arm_joint_path")
    if checker is None:
        return False, "collision_checker_unavailable"
    collision_free = torch.as_tensor(
        checker(path, payloads=dict(payloads or {})),
        dtype=torch.bool,
    ).flatten()
    if collision_free.shape != (path.shape[0],):
        return False, "collision_result_shape_mismatch"
    if not bool(collision_free.all()):
        return False, "joint_path_collision"
    return True, "accepted"


def _wrapped_attr(env: Any, name: str) -> Any:
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr(name)
        except AttributeError:
            pass
    return getattr(env, name, None)
