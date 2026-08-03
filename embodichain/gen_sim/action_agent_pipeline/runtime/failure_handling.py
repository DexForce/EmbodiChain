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

"""Per-environment failure propagation for vectorized graph execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import WorldState

__all__ = [
    "_current_robot_qpos",
    "_failed_parallel_hold_result",
    "_hold_failed_action_steps",
    "_hold_failed_world_state_qpos",
    "_merge_failed_env_masks",
    "_normalize_failed_env_mask",
]


def _normalize_failed_env_mask(
    failed_env_mask: torch.Tensor | np.ndarray | None,
    num_envs: int,
    *,
    name: str,
) -> torch.Tensor:
    """Return a CPU boolean failure mask with one entry per environment."""
    if failed_env_mask is None:
        return torch.zeros(num_envs, dtype=torch.bool)
    mask = torch.as_tensor(failed_env_mask, dtype=torch.bool).detach().cpu()
    if mask.ndim != 1 or mask.shape[0] != num_envs:
        raise ValueError(
            f"{name} must have shape ({num_envs},), got {tuple(mask.shape)}."
        )
    return mask


def _merge_failed_env_masks(
    num_envs: int,
    *masks: torch.Tensor | np.ndarray | None,
) -> torch.Tensor:
    """Combine per-environment failures from a graph edge and its ancestors."""
    merged = torch.zeros(num_envs, dtype=torch.bool)
    for index, mask in enumerate(masks):
        merged |= _normalize_failed_env_mask(
            mask,
            num_envs,
            name=f"failed environment mask {index}",
        )
    return merged


def _current_robot_qpos(env: Any, num_envs: int) -> np.ndarray:
    """Read the current batched robot configuration as an action array."""
    qpos = env.robot.get_qpos()
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or qpos.shape[0] != num_envs:
        raise ValueError(
            "Robot qpos must have shape (num_envs, robot_dof), got "
            f"{qpos.shape} for num_envs={num_envs}."
        )
    return qpos


def _hold_failed_action_steps(
    env: Any,
    actions: list[torch.Tensor],
    failed_env_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Replace failed environments' commands with their current robot qpos."""
    if not bool(failed_env_mask.any()):
        return actions
    current_qpos = torch.as_tensor(
        _current_robot_qpos(env, len(failed_env_mask)),
        dtype=torch.float32,
    )
    held_actions = []
    for action in actions:
        held_action = action.clone()
        failed_on_device = failed_env_mask.to(device=held_action.device)
        held_action[failed_on_device] = current_qpos.to(
            device=held_action.device,
            dtype=held_action.dtype,
        )[failed_on_device]
        held_actions.append(held_action)
    return held_actions


def _hold_failed_world_state_qpos(
    state: WorldState | None,
    current_qpos: np.ndarray,
    failed_env_mask: torch.Tensor,
) -> WorldState | None:
    """Keep failed environments' next state at the qpos actually commanded."""
    if state is None or not bool(failed_env_mask.any()):
        return state
    last_qpos = state.last_qpos.clone()
    failed_on_device = failed_env_mask.to(device=last_qpos.device)
    current_qpos_t = torch.as_tensor(
        current_qpos,
        dtype=last_qpos.dtype,
        device=last_qpos.device,
    )
    last_qpos[failed_on_device] = current_qpos_t[failed_on_device]
    return state.with_updates(last_qpos=last_qpos)


def _failed_parallel_hold_result(
    env: Any,
    world_states: Mapping[str, WorldState],
    failed_env_mask: torch.Tensor,
) -> dict[str, Any]:
    """Build one full-robot hold step when every graph environment has failed."""
    current_qpos = _current_robot_qpos(env, len(failed_env_mask))
    hold_step = torch.as_tensor(current_qpos, dtype=torch.float32)
    return {
        "actions": [hold_step],
        "world_states": {
            side: _hold_failed_world_state_qpos(
                state,
                current_qpos,
                failed_env_mask,
            )
            for side, state in world_states.items()
        },
        "arm_actions": {"left": None, "right": None},
        "failed_env_mask": failed_env_mask,
    }
