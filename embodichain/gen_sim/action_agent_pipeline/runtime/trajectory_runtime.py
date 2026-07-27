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

"""Adapt trajectories, mask failures, and synchronize cached agent state."""

from __future__ import annotations

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _select_arm_parts,
    _state_to_hand_qpos,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_action_utils import (
    get_arm_states,
)
from embodichain.lab.sim.atomic_actions import WorldState
from embodichain.utils.logger import log_info

__all__ = [
    "_failed_env_mask",
    "_ensure_failure_hold_step",
    "_pad_failed_trajectory_with_init_qpos",
    "_trajectory_to_agent_action",
    "_sync_agent_state_from_atomic_action",
    "_sync_agent_states_from_coordinated_action",
    "_as_2d_action",
    "_append_hold_steps",
]


def _failed_env_mask(
    success: bool | torch.Tensor | np.ndarray, n_envs: int
) -> torch.Tensor | None:
    """Return a boolean mask of failed environments, or None if all succeeded.

    Args:
        success: Per-action success flag. May be a scalar bool or a per-environment
            boolean tensor/array of shape ``(n_envs,)``.
        n_envs: Number of environments in the batched action.

    Returns:
        ``None`` when every environment succeeded. Otherwise a boolean tensor of
        shape ``(n_envs,)`` with ``True`` entries for failed environments.
    """
    if isinstance(success, torch.Tensor):
        if success.ndim == 0:
            success = bool(success.item())
        else:
            if success.shape[0] != n_envs:
                raise ValueError(
                    f"success tensor has {success.shape[0]} entries but "
                    f"trajectory has {n_envs} environments."
                )
            return ~success.bool()
    if isinstance(success, np.ndarray):
        if success.ndim == 0:
            success = bool(success.item())
        else:
            if success.shape[0] != n_envs:
                raise ValueError(
                    f"success array has {success.shape[0]} entries but "
                    f"trajectory has {n_envs} environments."
                )
            return torch.from_numpy(~success.astype(bool))
    return None if bool(success) else torch.ones(n_envs, dtype=torch.bool)


def _ensure_failure_hold_step(
    trajectory: torch.Tensor,
    state: WorldState,
) -> torch.Tensor:
    """Give a failed empty trajectory one no-op waypoint at its start qpos."""
    if trajectory.ndim != 3:
        raise ValueError(
            "Atomic action trajectory must have shape (n_envs, T, robot_dof), "
            f"got {trajectory.shape}."
        )
    if trajectory.shape[1] > 0:
        return trajectory
    if trajectory.shape[0] != state.last_qpos.shape[0]:
        raise ValueError(
            "Failed action trajectory environment count does not match WorldState: "
            f"{trajectory.shape[0]} != {state.last_qpos.shape[0]}."
        )
    return state.last_qpos.to(
        device=trajectory.device,
        dtype=trajectory.dtype,
    ).unsqueeze(1)


def _pad_failed_trajectory_with_init_qpos(
    trajectory: torch.Tensor,
    state: WorldState,
    joint_ids: list[int],
    failed_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace failed-environment trajectories with their initial joint positions.

    Args:
        trajectory: Batched trajectory tensor of shape ``(n_envs, T, D)``.
        state: World state whose ``last_qpos`` field supplies the initial positions.
        joint_ids: Indices into ``state.last_qpos`` that correspond to the ``D``
            trajectory columns.
        failed_mask: Boolean mask of shape ``(n_envs,)`` with ``True`` for failed
            environments.

    Returns:
        A cloned trajectory where failed environments are replaced by a constant
        sequence of their initial joint positions.
    """
    if not failed_mask.any():
        return trajectory
    device = trajectory.device
    joint_ids_t = torch.as_tensor(joint_ids, dtype=torch.long, device=device)
    init_qpos = state.last_qpos[:, joint_ids_t].to(
        device=device, dtype=trajectory.dtype
    )
    n_failed = int(failed_mask.sum().item())
    padded = trajectory.clone()
    failed_on_device = failed_mask.to(device=device)
    padded[failed_on_device] = (
        init_qpos[failed_on_device].unsqueeze(1).repeat(1, trajectory.shape[1], 1)
    )
    log_info(
        f"Padded {n_failed} failed environment(s) with initial joint positions.",
        color="yellow",
    )
    return padded


def _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids):
    _, _, current_arm_qpos, _, current_gripper_state = get_arm_states(env, robot_name)
    _, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach()
    else:
        trajectory = torch.as_tensor(trajectory)

    if trajectory.dim() == 3:
        n_envs, T, dof = trajectory.shape
    elif trajectory.dim() == 2:
        n_envs, T, dof = 1, trajectory.shape[0], trajectory.shape[1]
    else:
        raise ValueError(
            "Atomic action trajectory must have shape (T, D) or (N, T, D), "
            f"got {trajectory.shape}."
        )
    if T == 0:
        raise ValueError("Atomic action trajectory must have at least one step.")

    joint_ids = [int(joint_id) for joint_id in joint_ids]
    if len(joint_ids) != dof:
        raise ValueError(
            f"Atomic action joint_ids length {len(joint_ids)} does not match "
            f"trajectory width {dof}."
        )

    device = trajectory.device
    current_arm_qpos = torch.as_tensor(
        current_arm_qpos, dtype=torch.float32, device=device
    )
    current_gripper_state = torch.as_tensor(
        current_gripper_state, dtype=torch.float32, device=device
    )

    eef_dof = len(eef_joints)

    if current_arm_qpos.ndim == 1:
        current_arm_qpos = current_arm_qpos.unsqueeze(0).repeat(n_envs, 1)
    if current_gripper_state.ndim == 1:
        hand_qpos = _state_to_hand_qpos(current_gripper_state, eef_dof, device)
        hand_qpos = hand_qpos.unsqueeze(0).repeat(n_envs, 1)
    else:
        if current_gripper_state.shape[-1] == eef_dof:
            hand_qpos = current_gripper_state
        else:
            hand_qpos = torch.stack(
                [
                    _state_to_hand_qpos(state, eef_dof, device)
                    for state in current_gripper_state
                ]
            )

    agent_action = torch.cat([current_arm_qpos, hand_qpos], dim=-1)
    agent_action = agent_action.unsqueeze(1).repeat(1, T, 1)

    joint_id_to_col = {joint_id: col for col, joint_id in enumerate(joint_ids)}
    for out_col, joint_id in enumerate(arm_joints + eef_joints):
        if joint_id in joint_id_to_col:
            traj_col = joint_id_to_col[joint_id]
            if n_envs == 1:
                agent_action[0, :, out_col] = trajectory[0, :, traj_col]
            else:
                agent_action[:, :, out_col] = trajectory[:, :, traj_col]

    if n_envs == 1:
        agent_action = agent_action.squeeze(0)

    return agent_action.detach().cpu().numpy().astype(np.float32)


def _sync_agent_state_from_atomic_action(env, robot_name, action_np, control):
    if action_np is None or len(action_np) == 0:
        raise ValueError("Atomic action is empty; cannot sync agent state.")

    action_np = np.asarray(action_np, dtype=np.float32)
    if action_np.ndim == 2:
        final_action = action_np[-1]
    elif action_np.ndim == 3:
        final_action = action_np[:, -1, :]
    else:
        raise ValueError(
            "Atomic action must have shape (T, D) or (N, T, D), "
            f"got {action_np.shape}."
        )

    is_left, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
    arm_dof = len(arm_joints)

    if control == "arm" and arm_dof > 0:
        arm_qpos = torch.as_tensor(
            final_action[..., :arm_dof],
            dtype=torch.float32,
            device=env.robot.device,
        )
        env.set_current_qpos_agent(arm_qpos, is_left=is_left)
        env.set_current_xpos_agent(
            env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
            is_left=is_left,
        )

    if len(eef_joints) == 0:
        return

    _, _, _, _, current_gripper_state = get_arm_states(env, robot_name)
    eef_qpos = final_action[..., arm_dof : arm_dof + len(eef_joints)]

    current_gripper_state = torch.as_tensor(
        current_gripper_state, dtype=torch.float32, device=env.robot.device
    )
    if current_gripper_state.ndim == 1:
        state_dof = max(int(current_gripper_state.numel()), 1)
    else:
        state_dof = max(int(current_gripper_state.shape[-1]), 1)

    if eef_qpos.shape[-1] >= state_dof:
        gripper_qpos = eef_qpos[..., :state_dof]
    else:
        repeats = int(np.ceil(state_dof / eef_qpos.shape[-1]))
        gripper_qpos = np.tile(eef_qpos, repeats)[..., :state_dof]

    env.set_current_gripper_state_agent(
        torch.as_tensor(
            gripper_qpos,
            dtype=current_gripper_state.dtype,
            device=current_gripper_state.device,
        ),
        is_left=is_left,
    )


def _sync_agent_states_from_coordinated_action(env, action_np) -> None:
    if action_np is None or len(action_np) == 0:
        raise ValueError("Coordinated atomic action is empty; cannot sync state.")
    action_np = np.asarray(action_np, dtype=np.float32)
    if action_np.ndim == 2:
        final_qpos = action_np[-1]
    elif action_np.ndim == 3:
        final_qpos = action_np[:, -1, :]
    else:
        raise ValueError(
            "Coordinated atomic action must have shape (T, D) or (N, T, D), "
            f"got {action_np.shape}."
        )
    for side, is_left in (("left", True), ("right", False)):
        arm_joints = list(getattr(env, f"{side}_arm_joints", []) or [])
        eef_joints = list(getattr(env, f"{side}_eef_joints", []) or [])
        if arm_joints:
            arm_qpos = torch.as_tensor(
                final_qpos[..., arm_joints],
                dtype=torch.float32,
                device=env.robot.device,
            )
            env.set_current_qpos_agent(arm_qpos, is_left=is_left)
            env.set_current_xpos_agent(
                env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
                is_left=is_left,
            )
        if eef_joints:
            env.set_current_gripper_state_agent(
                torch.as_tensor(
                    final_qpos[..., eef_joints],
                    dtype=torch.float32,
                    device=env.robot.device,
                ),
                is_left=is_left,
            )


def _as_2d_action(action, action_name: str):
    """Normalize an action array to shape (n_envs, T, D)."""
    if action is None:
        return None
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, None, :]
    elif action.ndim == 2:
        action = action[None, :, :]
    if action.ndim != 3 or action.shape[1] == 0:
        raise ValueError(
            f"{action_name} must have shape (T, D) or (N, T, D) with T > 0, "
            f"got {action.shape}."
        )
    return action


def _append_hold_steps(action_np, hold_steps: int, log_name: str):
    hold_steps = int(hold_steps)
    if hold_steps <= 0:
        return action_np
    if action_np is None or len(action_np) == 0:
        raise ValueError(f"{log_name} action is empty; cannot append hold steps.")

    action_np = np.asarray(action_np, dtype=np.float32)
    if action_np.ndim == 2:
        hold_actions = np.repeat(action_np[-1:], hold_steps, axis=0)
        action_np = np.concatenate([action_np, hold_actions], axis=0)
    elif action_np.ndim == 3:
        hold_actions = np.repeat(action_np[:, -1:, :], hold_steps, axis=1)
        action_np = np.concatenate([action_np, hold_actions], axis=1)
    else:
        raise ValueError(
            f"{log_name} action must have shape (T, D) or (N, T, D), "
            f"got {action_np.shape}."
        )
    log_info(
        f"Append {hold_steps} hold steps after {log_name}; "
        f"total trajectory length is {action_np.shape[-2]}.",
        color="green",
    )
    return action_np
