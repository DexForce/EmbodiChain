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

"""Build and execute synchronized left/right atomic-action streams.

Parallel orchestration owns graph-slot semantics and failure propagation. The
single-action executor remains independent of this higher-level scheduling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_execution import (
    _execute_atomic_action_result,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _select_arm_parts,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
    ExecutedAtomicAction,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_action_utils import (
    resolve_arm_side,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coordinated_payload import (
    _coordinated_transport_failure_mask,
    _has_coordinated_held_object,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.failure_handling import (
    _current_robot_qpos,
    _failed_parallel_hold_result,
    _hold_failed_action_steps,
    _hold_failed_world_state_qpos,
    _merge_failed_env_masks,
    _normalize_failed_env_mask,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.joint_path_safety import (
    validate_dual_arm_joint_path,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.trajectory_runtime import (
    _as_2d_action,
    _sync_agent_state_from_atomic_action,
    _sync_agent_states_from_coordinated_action,
)
from embodichain.lab.sim.atomic_actions import WorldState

__all__ = [
    "build_parallel_action_stream",
    "execute_parallel_atomic_actions",
    "init_parallel_world_states",
    "step_env_with_actions",
]


def execute_parallel_atomic_actions(
    left_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    right_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    *,
    env: Any,
    world_states: dict[str, WorldState] | None = None,
    failed_env_mask: torch.Tensor | np.ndarray | None = None,
    left_active_env_mask: torch.Tensor | np.ndarray | None = None,
    right_active_env_mask: torch.Tensor | np.ndarray | None = None,
    return_result: bool = False,
    **runtime_kwargs: Any,
) -> list[torch.Tensor] | dict[str, Any]:
    """Execute left/right atomic action specs as one synchronized stream.

    ``failed_env_mask`` carries failures from earlier graph edges. Failed
    environments receive a full-robot hold command and remain failed in the
    returned result.
    """
    require_joint_safety = bool(runtime_kwargs.pop("require_joint_safety", False))
    result = build_parallel_action_stream(
        left_arm_action=left_arm_action,
        right_arm_action=right_arm_action,
        env=env,
        world_states=world_states,
        failed_env_mask=failed_env_mask,
        left_active_env_mask=left_active_env_mask,
        right_active_env_mask=right_active_env_mask,
        return_result=True,
        **runtime_kwargs,
    )
    actions = result["actions"]
    if require_joint_safety:
        accepted, reason = validate_dual_arm_joint_path(
            env,
            actions,
            payloads={
                "left": _action_payload_uid(left_arm_action),
                "right": _action_payload_uid(right_arm_action),
            },
        )
        result["parallel_safety"] = {
            "accepted": accepted,
            "reason": reason,
        }
        if not accepted:
            result["parallel_rejected"] = True
            if return_result:
                return result
            return []
    step_env_with_actions(env, actions)
    _sync_agent_states_from_parallel_actions(
        env,
        result["arm_actions"],
        failed_env_mask=result["failed_env_mask"],
        active_env_masks=result.get("active_env_masks"),
    )
    guard_failed = _coordinated_transport_failure_mask(
        env,
        result["world_states"],
        result["arm_actions"],
    )
    result["failed_env_mask"] = _merge_failed_env_masks(
        int(getattr(env, "num_envs", 1)),
        result["failed_env_mask"],
        guard_failed,
    )
    if bool(result["failed_env_mask"].any()):
        current_qpos = _current_robot_qpos(env, int(getattr(env, "num_envs", 1)))
        result["world_states"] = {
            side: _hold_failed_world_state_qpos(
                state,
                current_qpos,
                result["failed_env_mask"],
            )
            for side, state in result["world_states"].items()
        }
    if return_result:
        return result
    return actions


def _action_payload_uid(action: Any) -> str | None:
    if action is None:
        return None
    target = (
        action.target_object
        if isinstance(action, AtomicActionSpec)
        else action.get("target_object", {})
    )
    if isinstance(target, Mapping):
        uid = target.get("obj_name")
        return str(uid) if uid is not None else None
    return None


def build_parallel_action_stream(
    left_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    right_arm_action: Mapping[str, Any] | AtomicActionSpec | np.ndarray | None = None,
    *,
    env: Any,
    world_states: dict[str, WorldState] | None = None,
    failed_env_mask: torch.Tensor | np.ndarray | None = None,
    left_active_env_mask: torch.Tensor | np.ndarray | None = None,
    right_active_env_mask: torch.Tensor | np.ndarray | None = None,
    return_result: bool = False,
    **runtime_kwargs: Any,
) -> list[torch.Tensor] | dict[str, Any]:
    """Build a synchronized left/right atomic action stream without stepping env.

    ``failed_env_mask`` carries failures from earlier graph edges. Failed
    environments receive a full-robot hold command and remain failed in the
    returned result.
    """
    if env is None:
        raise ValueError("env is required to build parallel atomic actions.")
    if world_states is None:
        world_states = init_parallel_world_states(env)
    num_envs = int(getattr(env, "num_envs", 1))
    upstream_failed_env_mask = _normalize_failed_env_mask(
        failed_env_mask,
        num_envs,
        name="failed_env_mask",
    )
    active_env_masks = {
        "left": _normalized_active_mask(
            left_active_env_mask,
            num_envs,
            default=left_arm_action is not None,
        ).to(device=env.device),
        "right": _normalized_active_mask(
            right_active_env_mask,
            num_envs,
            default=right_arm_action is not None,
        ).to(device=env.device),
    }
    if bool(upstream_failed_env_mask.all()):
        result = _failed_parallel_hold_result(
            env,
            world_states,
            upstream_failed_env_mask,
        )
        if return_result:
            return result
        return result["actions"]
    raw_left_arm_action = left_arm_action
    raw_right_arm_action = right_arm_action
    coordinated_action = _pop_coordinated_edge_action(left_arm_action, right_arm_action)
    if coordinated_action is not None:
        executed = _resolve_action_spec(
            coordinated_action,
            env,
            runtime_kwargs,
            state=world_states.get("coordinated"),
        )
        if not isinstance(executed, ExecutedAtomicAction):
            raise TypeError("Coordinated action must resolve to an atomic action.")
        action_np = _as_2d_action(
            _executed_action_array(executed),
            "coordinated_action",
        )
        actions = _full_robot_action_array_to_steps(action_np)
        node_failed_env_mask = _merge_failed_env_masks(
            num_envs,
            upstream_failed_env_mask,
            executed.failed_env_mask,
        )
        actions = _hold_failed_action_steps(env, actions, node_failed_env_mask)
        next_state = _hold_failed_world_state_qpos(
            executed.next_state,
            _current_robot_qpos(env, num_envs),
            node_failed_env_mask,
        )
        result = {
            "actions": actions,
            "world_states": {
                **world_states,
                "coordinated": next_state,
                "left": next_state,
                "right": next_state,
            },
            "arm_actions": {
                "left": executed,
                "right": None,
            },
            "failed_env_mask": node_failed_env_mask,
        }
        if return_result:
            return result
        return actions

    _validate_arm_action_slot(env, "left", left_arm_action)
    _validate_arm_action_slot(env, "right", right_arm_action)
    left_arm_action = _resolve_action_spec(
        left_arm_action,
        env,
        runtime_kwargs,
        state=world_states.get("left"),
    )
    right_arm_action = _resolve_action_spec(
        right_arm_action,
        env,
        runtime_kwargs,
        state=world_states.get("right"),
    )
    _validate_arm_action_slot(env, "left", left_arm_action)
    _validate_arm_action_slot(env, "right", right_arm_action)

    left_action_np = _as_2d_action(
        _executed_action_array(left_arm_action),
        LEFT_ARM_ACTION_KEY,
    )
    right_action_np = _as_2d_action(
        _executed_action_array(right_arm_action),
        RIGHT_ARM_ACTION_KEY,
    )
    arm_actions = {"left": left_action_np, "right": right_action_np}

    if all(action is None for action in arm_actions.values()):
        raise ValueError("At least one atomic arm action must be provided.")

    action_len = max(
        action.shape[1] for action in arm_actions.values() if action is not None
    )
    for side, action in arm_actions.items():
        if action is not None and action.shape[1] < action_len:
            diff = action_len - action.shape[1]
            padding = np.repeat(action[:, -1:, :], diff, axis=1)
            arm_actions[side] = np.concatenate([action, padding], axis=1)

    current_qpos = _current_robot_qpos(env, num_envs)
    actions = np.repeat(current_qpos[:, None, :], action_len, axis=1)

    for side, action in arm_actions.items():
        if action is None:
            continue

        arm_index = list(getattr(env, f"{side}_arm_joints", [])) + list(
            getattr(env, f"{side}_eef_joints", [])
        )
        if not arm_index:
            raise ValueError(
                f"{side}_arm_action was provided, but {side}_arm is not configured "
                f"on robot control parts {getattr(env.robot, 'control_parts', None)}."
            )
        if action.shape[-1] != len(arm_index):
            raise ValueError(
                f"{side}_arm_action width {action.shape[-1]} does not match "
                f"{side}_arm joints plus eef joints ({len(arm_index)})."
            )
        if action.shape[0] != num_envs:
            raise ValueError(
                f"{side}_arm_action has {action.shape[0]} environments but "
                f"env.num_envs={num_envs}."
            )
        inactive_indices = (~active_env_masks[side]).detach().cpu().numpy()
        action = action.copy()
        action[inactive_indices] = np.repeat(
            current_qpos[inactive_indices][:, None, arm_index],
            action_len,
            axis=1,
        )
        actions[:, :, arm_index] = action

    node_failed_env_mask = _merge_failed_env_masks(
        num_envs,
        upstream_failed_env_mask,
        _masked_action_failure(left_arm_action, active_env_masks["left"]),
        _masked_action_failure(right_arm_action, active_env_masks["right"]),
    )
    if bool(node_failed_env_mask.any()):
        # Replace only failed batches with their current qpos. Successful
        # environments keep executing the same synchronized graph edge.
        failed_indices = node_failed_env_mask.detach().cpu().numpy()
        actions[failed_indices] = current_qpos[failed_indices, None, :]

    actions = torch.from_numpy(actions).to(dtype=torch.float32)
    actions = list(actions.unbind(dim=1))
    if not return_result:
        return actions
    next_world_states = dict(world_states)
    for side, executed in {
        "left": left_arm_action,
        "right": right_arm_action,
    }.items():
        if (
            isinstance(executed, ExecutedAtomicAction)
            and executed.next_state is not None
        ):
            next_world_states[side] = _merge_inactive_world_state_qpos(
                executed.next_state,
                previous_state=world_states.get(side),
                current_qpos=current_qpos,
                active_env_mask=active_env_masks[side],
            )
    if bool(node_failed_env_mask.any()):
        next_world_states = {
            side: _hold_failed_world_state_qpos(
                state,
                current_qpos,
                node_failed_env_mask,
            )
            for side, state in next_world_states.items()
        }
    if _is_dual_coordinated_release_edge(
        raw_left_arm_action,
        raw_right_arm_action,
    ) and _has_coordinated_held_object(world_states):
        release_state = _released_coordinated_world_state(
            actions[-1],
            next_world_states,
        )
        next_world_states["coordinated"] = release_state
        next_world_states["left"] = release_state
        next_world_states["right"] = release_state
    return {
        "actions": actions,
        "world_states": next_world_states,
        "arm_actions": {
            "left": left_arm_action,
            "right": right_arm_action,
        },
        "failed_env_mask": node_failed_env_mask,
        "active_env_masks": active_env_masks,
    }


def init_parallel_world_states(env: Any) -> dict[str, WorldState]:
    """Seed independent per-arm WorldState slots from the current robot qpos."""
    qpos = env.robot.get_qpos().clone()
    return {
        "coordinated": WorldState(last_qpos=qpos.clone()),
        "left": WorldState(last_qpos=qpos.clone()),
        "right": WorldState(last_qpos=qpos.clone()),
    }


def _action_failed_env_mask(action: Any) -> torch.Tensor | None:
    """Read the per-environment failure mask from an executed atomic action."""
    if isinstance(action, ExecutedAtomicAction):
        return action.failed_env_mask
    return None


def _masked_action_failure(
    action: Any,
    active_env_mask: torch.Tensor,
) -> torch.Tensor | None:
    failed = _action_failed_env_mask(action)
    if failed is None:
        return None
    return failed.to(device=active_env_mask.device, dtype=torch.bool) & active_env_mask


def _normalized_active_mask(
    value: torch.Tensor | np.ndarray | None,
    num_envs: int,
    *,
    default: bool,
) -> torch.Tensor:
    if value is None:
        return torch.full((num_envs,), default, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool).flatten()
    if mask.shape != (num_envs,):
        raise ValueError(
            f"active_env_mask must have shape ({num_envs},), got {tuple(mask.shape)}."
        )
    return mask


def _merge_inactive_world_state_qpos(
    candidate: WorldState,
    *,
    previous_state: WorldState | None,
    current_qpos: np.ndarray,
    active_env_mask: torch.Tensor,
) -> WorldState:
    """Keep inactive batches at the joint state actually sent to the robot.

    Atomic actions plan the full vectorized batch even when an arm is assigned
    to only some environments. The action stream masks inactive rows, so the
    cached WorldState must apply the same mask before a later step reuses it.
    """
    inactive = ~active_env_mask.to(
        device=candidate.last_qpos.device,
        dtype=torch.bool,
    )
    if not bool(inactive.any()):
        return candidate
    last_qpos = candidate.last_qpos.clone()
    if previous_state is not None:
        inactive_qpos = previous_state.last_qpos.to(
            device=last_qpos.device,
            dtype=last_qpos.dtype,
        )
    else:
        inactive_qpos = torch.as_tensor(
            current_qpos,
            device=last_qpos.device,
            dtype=last_qpos.dtype,
        )
    last_qpos[inactive] = inactive_qpos[inactive]
    return candidate.with_updates(last_qpos=last_qpos)


def step_env_with_actions(
    env: Any,
    actions: list[torch.Tensor],
    *,
    update_obj_info: bool = True,
) -> None:
    """Step an environment through a prebuilt action stream."""
    if env is None:
        raise ValueError("env is required to step action stream.")
    for action in tqdm(actions):
        env.step(action)
        if update_obj_info:
            env.update_obj_info()


def _resolve_action_spec(
    action_spec,
    env,
    runtime_kwargs: dict[str, Any],
    *,
    state: WorldState | None,
):
    if action_spec is None:
        return None
    if isinstance(action_spec, np.ndarray):
        return action_spec
    if isinstance(action_spec, torch.Tensor):
        return action_spec
    return _execute_atomic_action_result(
        action_spec,
        env=env,
        state=state,
        **runtime_kwargs,
    )


def _executed_action_array(action):
    if isinstance(action, ExecutedAtomicAction):
        return action.action
    return action


def _validate_arm_action_slot(env, side: str, action) -> None:
    robot_name = _arm_action_robot_name(action)
    if _arm_action_control(action) == "coordinated" or robot_name is None:
        return
    action_side = resolve_arm_side(env, robot_name)
    if action_side != side:
        raise ValueError(
            f"{side}_arm_action contains robot_name={robot_name!r}, "
            f"which resolves to {action_side}_arm. Keep the outer graph slot "
            "consistent with the semantic arm name."
        )


def _arm_action_robot_name(action) -> str | None:
    if isinstance(action, ExecutedAtomicAction):
        return action.robot_name
    if isinstance(action, AtomicActionSpec):
        return action.robot_name
    if isinstance(action, Mapping):
        value = action.get("robot_name")
        if value is not None:
            return str(value)
    return None


def _arm_action_control(action) -> str | None:
    if isinstance(action, ExecutedAtomicAction):
        return action.control
    if isinstance(action, AtomicActionSpec):
        return action.control
    if isinstance(action, Mapping):
        value = action.get("control")
        if value is not None:
            return str(value)
    return None


def _pop_coordinated_edge_action(left_arm_action, right_arm_action):
    left_is_coordinated = _is_coordinated_action(left_arm_action)
    right_is_coordinated = _is_coordinated_action(right_arm_action)
    if left_is_coordinated and right_is_coordinated:
        raise ValueError(
            "A graph edge may contain only one CoordinatedPickment action."
        )
    if left_is_coordinated:
        if right_arm_action is not None:
            raise ValueError(
                "CoordinatedPickment controls both arms; right_arm_action must be null."
            )
        return left_arm_action
    if right_is_coordinated:
        if left_arm_action is not None:
            raise ValueError(
                "CoordinatedPickment controls both arms; left_arm_action must be null."
            )
        return right_arm_action
    return None


def _is_coordinated_action(action_spec) -> bool:
    if isinstance(action_spec, AtomicActionSpec):
        return action_spec.atomic_action_class == "CoordinatedPickment"
    if isinstance(action_spec, Mapping):
        return action_spec.get("atomic_action_class") == "CoordinatedPickment"
    return False


def _is_dual_coordinated_release_edge(left_arm_action, right_arm_action) -> bool:
    return {
        _gripper_open_release_side(left_arm_action),
        _gripper_open_release_side(right_arm_action),
    } == {"left", "right"}


def _gripper_open_release_side(action_spec) -> str | None:
    if isinstance(action_spec, AtomicActionSpec):
        atomic_action_class = action_spec.atomic_action_class
        robot_name = action_spec.robot_name
        control = action_spec.control
        target_qpos = action_spec.target_qpos
    elif isinstance(action_spec, Mapping):
        atomic_action_class = action_spec.get("atomic_action_class")
        robot_name = action_spec.get("robot_name")
        control = action_spec.get("control")
        target_qpos = action_spec.get("target_qpos") or {}
    else:
        return None

    if (
        atomic_action_class != "MoveJoints"
        or control != "hand"
        or not isinstance(target_qpos, Mapping)
        or target_qpos.get("source") != "gripper_state"
        or target_qpos.get("state") != "open"
    ):
        return None
    if robot_name == "left_arm":
        return "left"
    if robot_name == "right_arm":
        return "right"
    return None


def _released_coordinated_world_state(
    final_action: torch.Tensor,
    world_states: Mapping[str, WorldState],
) -> WorldState:
    held_objects = {}
    for state in world_states.values():
        if isinstance(state, WorldState):
            held_objects.update(state.held_objects)
    final_qpos = torch.as_tensor(final_action, dtype=torch.float32)
    if final_qpos.dim() == 1:
        final_qpos = final_qpos.unsqueeze(0)
    elif final_qpos.dim() == 2:
        # (n_envs, robot_dof) already batched
        pass
    else:
        raise ValueError(
            "Final coordinated action must have shape (robot_dof,) or "
            f"(n_envs, robot_dof), got {final_qpos.shape}."
        )
    return WorldState(
        last_qpos=final_qpos.clone(),
        held_objects=held_objects,
    )


def _full_robot_action_array_to_steps(action_np: np.ndarray) -> list[torch.Tensor]:
    action_np = np.asarray(action_np, dtype=np.float32)
    if action_np.ndim == 2:
        action_np = action_np[None, :, :]
    if action_np.ndim != 3 or action_np.shape[1] == 0:
        raise ValueError(
            "Coordinated action stream must have shape (T, robot_dof) or "
            f"(N, T, robot_dof), got {action_np.shape}."
        )
    actions = torch.from_numpy(action_np).to(dtype=torch.float32)
    return list(actions.unbind(dim=1))


def _sync_agent_states_from_parallel_actions(
    env,
    arm_actions: Mapping[str, Any],
    *,
    failed_env_mask: torch.Tensor | None = None,
    active_env_masks: Mapping[str, torch.Tensor] | None = None,
) -> None:
    for side, executed in arm_actions.items():
        if not isinstance(executed, ExecutedAtomicAction):
            continue
        state_sync_mask = failed_env_mask
        if active_env_masks is not None and side in active_env_masks:
            inactive = ~active_env_masks[side].to(dtype=torch.bool)
            state_sync_mask = (
                inactive
                if state_sync_mask is None
                else state_sync_mask.to(inactive.device, dtype=torch.bool) | inactive
            )
        action_np = _hold_failed_atomic_action_for_state_sync(
            env,
            executed,
            state_sync_mask,
        )
        if executed.control == "coordinated":
            _sync_agent_states_from_coordinated_action(env, action_np)
            continue
        _sync_agent_state_from_atomic_action(
            env,
            executed.robot_name,
            action_np,
            executed.control,
        )


def _hold_failed_atomic_action_for_state_sync(
    env: Any,
    executed: ExecutedAtomicAction,
    failed_env_mask: torch.Tensor | None,
) -> np.ndarray:
    """Mask cached arm state updates to the same qpos sent to failed envs."""
    action_np = np.asarray(executed.action, dtype=np.float32)
    if failed_env_mask is None or not bool(failed_env_mask.any()):
        return action_np

    num_envs = len(failed_env_mask)
    action_is_unbatched = action_np.ndim == 2
    action_batched = _as_2d_action(action_np, "atomic action")
    if action_batched.shape[0] != num_envs:
        raise ValueError(
            "Atomic action state-sync batch size does not match failed_env_mask: "
            f"{action_batched.shape[0]} != {num_envs}."
        )
    current_qpos = _current_robot_qpos(env, num_envs)
    if executed.control == "coordinated":
        hold_qpos = current_qpos
    else:
        _, _, _, arm_joints, eef_joints = _select_arm_parts(env, executed.robot_name)
        hold_qpos = current_qpos[:, arm_joints + eef_joints]
    action_batched = action_batched.copy()
    failed_indices = failed_env_mask.detach().cpu().numpy()
    action_batched[failed_indices] = hold_qpos[failed_indices, None, :]
    if action_is_unbatched:
        return action_batched.squeeze(0)
    return action_batched
