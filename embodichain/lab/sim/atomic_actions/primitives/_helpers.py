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

"""Shared internal helpers for concrete atomic actions."""

from __future__ import annotations

import torch

from embodichain.utils import logger

from ..bindings import EndpointBinding
from ..state import PlanningContext


def require_shared_task_state_key(
    motion: EndpointBinding,
    grasp: EndpointBinding,
    *,
    participant: str,
) -> str:
    """Return the stable task-state key shared by one participant's endpoints.

    Args:
        motion: Participant endpoint used for motion control.
        grasp: Participant endpoint used for grasp control.
        participant: Human-readable participant name used in validation errors.

    Returns:
        Stable logical key used to address held-object task state.

    Raises:
        ValueError: If the participant endpoints use different task-state keys.
    """
    motion_key = motion.task_state_key
    grasp_key = grasp.task_state_key
    if motion_key != grasp_key:
        raise ValueError(
            f"{participant} motion and grasp endpoints must share one "
            f"task_state_key, but got {motion_key!r} and {grasp_key!r}."
        )
    if not isinstance(motion_key, str) or not motion_key:
        raise ValueError(f"{participant} task_state_key must be a non-empty string.")
    return motion_key


def resolve_object_target(
    target: torch.Tensor,
    *,
    n_envs: int,
    device: torch.device,
    name: str = "object_target_pose",
) -> torch.Tensor:
    """Broadcast an object target pose to ``(n_envs, 4, 4)`` or validate it."""
    target = target.to(device=device, dtype=torch.float32).clone()
    if target.shape == (4, 4):
        target = target.unsqueeze(0).repeat(n_envs, 1, 1)
    if target.shape != (n_envs, 4, 4):
        logger.log_error(
            f"{name} must be (4, 4) or ({n_envs}, 4, 4), but got {target.shape}",
            ValueError,
        )
    return target


def arm_qpos_from_state(
    context: PlanningContext,
    arm_joint_ids: list[int],
) -> torch.Tensor:
    """Extract the arm slice of the measured planning-start joint positions."""
    return context.robot.qpos[:, arm_joint_ids]


__all__ = [
    "arm_qpos_from_state",
    "require_shared_task_state_key",
    "resolve_object_target",
]
