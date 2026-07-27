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

"""Execute one normalized atomic action and adapt its trajectory.

The module coordinates focused services but does not implement pose solving,
grasp generation, or trajectory shape conversion itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _build_action_cfg,
    _build_typed_target,
    _get_atomic_action_class,
    _motion_generator_for_env,
    _select_arm_parts,
    _state_with_current_agent_qpos,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
    ExecutedAtomicAction,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_targets import (
    _resolve_pickup_downstream_object_targets,
    _resolve_target,
    _target_summary,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.trajectory_runtime import (
    _append_hold_steps,
    _ensure_failure_hold_step,
    _failed_env_mask,
    _pad_failed_trajectory_with_init_qpos,
    _sync_agent_state_from_atomic_action,
    _sync_agent_states_from_coordinated_action,
    _trajectory_to_agent_action,
)
from embodichain.lab.sim.atomic_actions import WorldState
from embodichain.utils.logger import log_info, log_warning

__all__ = ["execute_atomic_action"]


def execute_atomic_action(
    action_spec: Mapping[str, Any] | AtomicActionSpec,
    *,
    env: Any,
    state: WorldState | None = None,
    **runtime_kwargs: Any,
) -> np.ndarray:
    """Execute one atomic action spec and return local arm+eef qpos actions."""
    executed = _execute_atomic_action_result(
        action_spec,
        env=env,
        state=state,
        **runtime_kwargs,
    )
    if executed.control == "coordinated":
        _sync_agent_states_from_coordinated_action(env, executed.action)
    else:
        _sync_agent_state_from_atomic_action(
            env,
            executed.robot_name,
            executed.action,
            executed.control,
        )
    return executed.action


def _execute_atomic_action_result(
    action_spec: Mapping[str, Any] | AtomicActionSpec,
    *,
    env,
    state: WorldState | None = None,
    **runtime_kwargs,
) -> ExecutedAtomicAction:
    """Execute one atomic action spec and keep the typed WorldState result."""
    spec = (
        action_spec
        if isinstance(action_spec, AtomicActionSpec)
        else AtomicActionSpec.from_mapping(action_spec)
    )

    target = _resolve_target(env, spec, runtime_kwargs, state=state)
    _, arm_part, hand_part, arm_joints, eef_joints = _select_arm_parts(
        env, spec.robot_name
    )
    cfg = _build_action_cfg(env, spec, arm_part, hand_part, len(eef_joints))
    if spec.atomic_action_class == "PickUp":
        cfg.downstream_object_target_poses = _resolve_pickup_downstream_object_targets(
            env, spec, target, runtime_kwargs
        )
    target = _build_typed_target(spec, target)
    if state is None:
        state = WorldState(last_qpos=env.robot.get_qpos().clone())
    state = _state_with_current_agent_qpos(env, spec, state)
    action_cls = _get_atomic_action_class(spec.atomic_action_class)
    action = action_cls(
        motion_generator=_motion_generator_for_env(env, runtime_kwargs),
        cfg=cfg,
    )
    result = action.execute(
        target=target,
        state=state,
    )
    failed_env_mask = _failed_env_mask(result.success, result.trajectory.shape[0])
    if failed_env_mask is not None and bool(failed_env_mask.any()):
        n_failed = int(failed_env_mask.sum().item())
        n_total = result.trajectory.shape[0]
        log_warning(
            f"Atomic action failed in {n_failed}/{n_total} environment(s): "
            f"atomic_action_class={spec.atomic_action_class}, "
            f"robot_name={spec.robot_name}, target={_target_summary(spec)}. "
            "Holding failed environments at their current joint positions."
        )
        result.trajectory = _ensure_failure_hold_step(result.trajectory, state)
        full_joint_ids = list(range(state.last_qpos.shape[-1]))
        result.trajectory = _pad_failed_trajectory_with_init_qpos(
            result.trajectory, state, full_joint_ids, failed_env_mask
        )
        result.next_state.last_qpos = result.next_state.last_qpos.clone()
        device = result.next_state.last_qpos.device
        failed_on_device = failed_env_mask.to(device=device)
        result.next_state.last_qpos[failed_on_device] = state.last_qpos[
            failed_on_device
        ]

    if spec.atomic_action_class == "CoordinatedPickment":
        return _executed_coordinated_atomic_action(
            env,
            spec,
            result,
            failed_env_mask=failed_env_mask,
        )
    if spec.atomic_action_class == "MoveJoints":
        joint_ids = arm_joints if spec.control == "arm" else eef_joints
    else:
        joint_ids = arm_joints + eef_joints
    trajectory = result.trajectory[:, :, joint_ids]

    action_np = _trajectory_to_agent_action(
        env,
        spec.robot_name,
        trajectory,
        joint_ids,
    )
    action_np = _append_hold_steps(
        action_np,
        int(spec.cfg.get("post_hold_steps", 0)),
        "atomic action",
    )
    log_info(
        "Using atomic action: "
        f"atomic_action_class={spec.atomic_action_class}, cfg={cfg.__class__.__name__}, "
        f"control={spec.control}, target={_target_summary(spec)}, "
        f"steps={len(action_np)}.",
        color="green",
    )
    next_state = result.next_state
    if int(spec.cfg.get("post_hold_steps", 0)) > 0:
        next_state = WorldState(
            last_qpos=next_state.last_qpos.clone(),
            held_object=next_state.held_object,
            coordinated_held_object=next_state.coordinated_held_object,
        )
    return ExecutedAtomicAction(
        action=action_np,
        next_state=next_state,
        robot_name=spec.robot_name,
        control=spec.control,
        failed_env_mask=failed_env_mask,
        atomic_action_class=spec.atomic_action_class,
    )


def _executed_coordinated_atomic_action(
    env,
    spec: AtomicActionSpec,
    result,
    *,
    failed_env_mask: torch.Tensor | None = None,
) -> ExecutedAtomicAction:
    trajectory = result.trajectory
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach()
    else:
        trajectory = torch.as_tensor(trajectory)
    if trajectory.dim() == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory.squeeze(0)
    if trajectory.dim() not in (2, 3) or trajectory.shape[-2] == 0:
        raise ValueError(
            "Coordinated atomic action trajectory must have shape (T, D) or "
            f"(N, T, D), got {trajectory.shape}."
        )
    action_np = trajectory.detach().cpu().numpy().astype(np.float32)
    action_np = _append_hold_steps(
        action_np,
        int(spec.cfg.get("post_hold_steps", 0)),
        "coordinated atomic action",
    )
    log_info(
        "Using coordinated atomic action: "
        f"atomic_action_class={spec.atomic_action_class}, "
        f"target={_target_summary(spec)}, steps={action_np.shape[-2]}.",
        color="green",
    )
    return ExecutedAtomicAction(
        action=action_np,
        next_state=result.next_state,
        robot_name=None,
        control="coordinated",
        failed_env_mask=failed_env_mask,
        atomic_action_class=spec.atomic_action_class,
    )
