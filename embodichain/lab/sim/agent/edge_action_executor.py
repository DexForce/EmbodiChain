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

from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from embodichain.lab.sim.agent.atom_action_utils import sync_agent_state_from_robot
from embodichain.lab.sim.agent.atomic_action_adapter import (
    validate_pending_public_grasp_after_action,
    validate_pending_public_place_after_action,
)
from embodichain.lab.sim.agent.error_functions import (
    fallen_object,
    inject_forced_recovery_error,
    inject_interactive_error,
    interactive_error_requested,
    misplaced_object,
    restore_interactive_error_input,
    setup_interactive_error_input,
)
from embodichain.utils.logger import log_info, log_warning

__all__ = ["ActionPlan", "EdgeExecutionResult", "EdgeActionExecutor"]


@dataclass
class ActionPlan:
    """Structured plan returned by graph-backed atomic actions."""

    is_success: bool
    trajectory: torch.Tensor
    joint_ids: list[int]
    action_name: str


@dataclass
class EdgeExecutionResult:
    """Execution result for one AgentTaskGraph edge."""

    actions: list[torch.Tensor]
    monitor_index: int | None
    monitor_name: str | None
    step_index: int | None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class EdgeActionExecutor:
    """Execute one graph edge from structured atomic action plans."""

    def execute(
        self,
        *,
        edge,
        env,
        interactive_error_injection: bool = False,
        forced_recovery_injection: Any = None,
        **kwargs,
    ) -> EdgeExecutionResult:
        plan_kwargs = dict(kwargs)
        plan_kwargs["_edge_is_recovery"] = bool(getattr(edge, "is_recovery", False))
        left_plan = self._resolve_plan(
            edge.left_arm_action, env=env, kwargs=plan_kwargs
        )
        right_plan = self._resolve_plan(
            edge.right_arm_action, env=env, kwargs=plan_kwargs
        )
        actions = self._compose_full_actions(env, [left_plan, right_plan])
        return self._execute_actions(
            actions,
            monitor_sequences=edge.monitor_sequences,
            env=env,
            interactive_error_injection=interactive_error_injection,
            forced_recovery_injection=forced_recovery_injection,
            kwargs=kwargs,
        )

    def _resolve_plan(
        self,
        action: Any,
        *,
        env,
        kwargs: dict[str, Any],
    ) -> ActionPlan | None:
        if action is None:
            return None
        if isinstance(action, ActionPlan):
            return action
        plan_func = getattr(action, "plan", None)
        if callable(plan_func):
            plan = plan_func(env=env, **kwargs)
            if not isinstance(plan, ActionPlan):
                raise TypeError(
                    f"Structured action plan() must return ActionPlan, got {type(plan)!r}."
                )
            return plan
        raise TypeError(
            "AgentTaskGraph edges require structured atomic graph actions with "
            f"plan(env, **kwargs); got {type(action)!r}."
        )

    def _compose_full_actions(
        self,
        env,
        plans: list[ActionPlan | None],
    ) -> list[torch.Tensor]:
        active_plans = [plan for plan in plans if plan is not None]
        if not active_plans:
            raise RuntimeError(
                "AgentTaskGraph edge must define at least one structured action plan."
            )

        current_qpos = env.robot.get_qpos()
        if current_qpos.ndim != 2 or current_qpos.shape[0] != 1:
            raise RuntimeError(
                "Structured AgentTaskGraph execution currently supports robot qpos "
                f"shape (1, dof), got {tuple(current_qpos.shape)}."
            )
        device = current_qpos.device
        dtype = current_qpos.dtype

        plan_trajs = [
            self._plan_trajectory(plan, device=device, dtype=dtype)
            for plan in active_plans
        ]
        n_steps = max(traj.shape[0] for traj in plan_trajs)
        actions = current_qpos[0].repeat(n_steps, 1)
        controlled_joint_ids: set[int] = set()

        for plan, traj in zip(active_plans, plan_trajs):
            if not plan.is_success:
                raise RuntimeError(
                    f"Atomic action plan '{plan.action_name}' was not successful."
                )
            if traj.shape[0] < n_steps:
                pad = traj[-1:].repeat(n_steps - traj.shape[0], 1)
                traj = torch.cat([traj, pad], dim=0)
            joint_ids = list(plan.joint_ids)
            actions[:, joint_ids] = traj
            controlled_joint_ids.update(joint_ids)

        self._preserve_cached_gripper_targets(
            actions,
            env=env,
            controlled_joint_ids=controlled_joint_ids,
        )

        return list(actions.to(dtype=torch.float32).unsqueeze(1).unbind(dim=0))

    @staticmethod
    def _plan_trajectory(
        plan: ActionPlan,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        trajectory = torch.as_tensor(plan.trajectory, dtype=dtype, device=device)
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)
        if trajectory.ndim != 3 or trajectory.shape[0] != 1:
            raise RuntimeError(
                "Structured AgentTaskGraph execution currently supports action "
                f"trajectory shape (1, T, dof), got {tuple(trajectory.shape)}."
            )
        if trajectory.shape[-1] != len(plan.joint_ids):
            raise RuntimeError(
                f"ActionPlan '{plan.action_name}' trajectory width "
                f"({trajectory.shape[-1]}) does not match joint_ids "
                f"({len(plan.joint_ids)})."
            )
        return trajectory[0]

    def _execute_actions(
        self,
        actions: list[torch.Tensor],
        *,
        monitor_sequences,
        env,
        interactive_error_injection: bool,
        forced_recovery_injection: Any,
        kwargs: dict[str, Any],
    ) -> EdgeExecutionResult:
        forced_config = (
            forced_recovery_injection
            if isinstance(forced_recovery_injection, dict)
            else None
        )
        force_this_edge = False
        blind_force = bool(forced_config and forced_config.get("blind", False))
        if (
            forced_config is not None
            and forced_config.get("enabled", False)
            and not forced_config.get("_injected", False)
        ):
            if blind_force:
                edge_count = forced_config.get("_seen_edges", 0) + 1
                forced_config["_seen_edges"] = edge_count
                force_this_edge = edge_count == int(forced_config.get("edge_index", 1))
            elif monitor_sequences is not None:
                edge_count = forced_config.get("_seen_monitored_edges", 0) + 1
                forced_config["_seen_monitored_edges"] = edge_count
                force_this_edge = edge_count == int(forced_config.get("edge_index", 1))
            if force_this_edge:
                log_warning(
                    f"Forced error injection armed on "
                    f"{'blind' if blind_force else 'monitored'} edge #{edge_count}."
                )

        interactive_input = setup_interactive_error_input(interactive_error_injection)
        try:
            for step_index in tqdm(range(len(actions))):
                action_tensor = actions[step_index]
                env.step(action_tensor)

                if interactive_error_requested(interactive_input):
                    restore_interactive_error_input(interactive_input)
                    interactive_input = None
                    inject_interactive_error(env)
                    interactive_input = setup_interactive_error_input(
                        interactive_error_injection
                    )

                if (
                    force_this_edge
                    and forced_config is not None
                    and not forced_config.get("_injected", False)
                    and not forced_config.get("_attempted", False)
                ):
                    injected_monitor = self._maybe_inject_forced_error(
                        env=env,
                        monitor_sequences=monitor_sequences,
                        forced_config=forced_config,
                        blind_force=blind_force,
                        step_index=step_index,
                        n_steps=len(actions),
                    )
                    if injected_monitor is not None:
                        forced_config["_injected"] = True
                        forced_config["_injected_at_step"] = step_index
                        forced_config["_injected_monitor"] = injected_monitor

                if monitor_sequences is not None:
                    triggered = self._check_monitors(
                        monitor_sequences,
                        env=env,
                        forced_config=forced_config,
                        step_index=step_index,
                        actions=actions,
                    )
                    if triggered is not None:
                        self._clear_pending_post_action_validators(env)
                        return triggered

                env.update_obj_info()
        finally:
            restore_interactive_error_input(interactive_input)

        sync_agent_state_from_robot(env)
        self._sync_cached_gripper_targets_from_action(env, actions[-1])
        self._run_post_action_validators(env, kwargs)
        if monitor_sequences is not None:
            log_info("No monitor sequences triggered during execution.")
        return EdgeExecutionResult(
            actions=actions,
            monitor_index=None,
            monitor_name=None,
            step_index=None,
        )

    def _maybe_inject_forced_error(
        self,
        *,
        env,
        monitor_sequences,
        forced_config: dict[str, Any],
        blind_force: bool,
        step_index: int,
        n_steps: int,
    ) -> str | None:
        configured_step = int(forced_config.get("step_index", -1))
        should_inject = (
            step_index == n_steps - 1
            if configured_step < 0
            else step_index >= configured_step
        )
        if not should_inject:
            return None

        forced_config["_attempted"] = True
        if blind_force:
            error_obj = forced_config.get("blind_obj_name", "bottle")
            relative_error_xyz = forced_config.get("relative_error_xyz")
            error_type = forced_config.get("error_type", "misplaced_object")
            if error_type == "fallen_object":
                fallen_object(
                    env,
                    error_obj=error_obj,
                    error_pose=None,
                    relative_error_xyz=relative_error_xyz,
                )
            else:
                misplaced_object(
                    env,
                    error_obj=error_obj,
                    error_pose=None,
                    relative_error_xyz=relative_error_xyz,
                )
            log_warning(
                f"Injected blind forced error on {error_obj} "
                f"with error_type={error_type} "
                f"relative_error_xyz={relative_error_xyz}."
            )
            return f"blind:{error_obj}"

        return inject_forced_recovery_error(
            env,
            monitor_sequences,
            relative_error_xyz=forced_config.get("relative_error_xyz"),
            error_type=forced_config.get("error_type", "misplaced_object"),
        )

    def _check_monitors(
        self,
        monitor_sequences,
        *,
        env,
        forced_config: dict[str, Any] | None,
        step_index: int,
        actions: list[torch.Tensor],
    ) -> EdgeExecutionResult | None:
        for monitor_idx, monitor_sequence in enumerate(monitor_sequences):
            for function in monitor_sequence:
                if function() is not True:
                    continue
                env.update_obj_info()
                function_name = _callable_name(function)
                log_warning(
                    f"Monitor function {function_name} triggered at step {step_index}."
                )
                if forced_config is not None and forced_config.get("_injected", False):
                    forced_config["_triggered"] = True
                    forced_config["_triggered_monitor_index"] = monitor_idx
                    forced_config["_triggered_monitor_name"] = function_name
                    forced_config["_triggered_step"] = step_index
                sync_agent_state_from_robot(env)
                return EdgeExecutionResult(
                    actions=actions[: step_index + 1],
                    monitor_index=monitor_idx,
                    monitor_name=function_name,
                    step_index=step_index,
                )
        return None

    @staticmethod
    def _run_post_action_validators(env, kwargs: dict[str, Any]) -> None:
        validate_pending_public_grasp_after_action(env, kwargs)
        validate_pending_public_place_after_action(env, kwargs)

    @staticmethod
    def _preserve_cached_gripper_targets(
        actions: torch.Tensor,
        *,
        env,
        controlled_joint_ids: set[int],
    ) -> None:
        for hand_joints, state_name in (
            (getattr(env, "left_eef_joints", []), "left_arm_current_gripper_state"),
            (getattr(env, "right_eef_joints", []), "right_arm_current_gripper_state"),
        ):
            missing_joints = [
                int(joint_id)
                for joint_id in hand_joints
                if int(joint_id) not in controlled_joint_ids
            ]
            if not missing_joints:
                continue
            hand_target = _expand_joint_state(
                getattr(env, state_name, None),
                n_joints=len(hand_joints),
                device=actions.device,
                dtype=actions.dtype,
            )
            if hand_target is None:
                continue
            joint_values = {
                int(joint_id): hand_target[idx]
                for idx, joint_id in enumerate(hand_joints)
            }
            for joint_id in missing_joints:
                actions[:, joint_id] = joint_values[joint_id]

    @staticmethod
    def _sync_cached_gripper_targets_from_action(env, action: torch.Tensor) -> None:
        action = torch.as_tensor(action).squeeze(0)
        gripper_dtype = getattr(getattr(env, "open_state", None), "dtype", action.dtype)
        gripper_device = getattr(
            getattr(env, "open_state", None),
            "device",
            action.device,
        )
        for hand_joints, state_name in (
            (getattr(env, "left_eef_joints", []), "left_arm_current_gripper_state"),
            (getattr(env, "right_eef_joints", []), "right_arm_current_gripper_state"),
        ):
            if not hand_joints:
                continue
            setattr(
                env,
                state_name,
                action[list(hand_joints)][0]
                .to(dtype=gripper_dtype, device=gripper_device)
                .unsqueeze(0),
            )

    @staticmethod
    def _clear_pending_post_action_validators(env) -> None:
        setattr(env, "_pending_public_grasp_physical_validations", [])
        setattr(env, "_pending_public_grasp_physical_validation", None)
        setattr(env, "_pending_public_place_validations", [])


def _expand_joint_state(
    state,
    *,
    n_joints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if state is None or n_joints == 0:
        return None
    joint_state = torch.as_tensor(state, dtype=dtype, device=device).flatten()
    if joint_state.numel() == 1:
        return joint_state.repeat(n_joints)
    if joint_state.numel() != n_joints:
        raise RuntimeError(
            "Cached gripper target width "
            f"({joint_state.numel()}) does not match hand joint count ({n_joints})."
        )
    return joint_state


def _callable_name(function: Any) -> str:
    func = getattr(function, "func", function)
    return getattr(func, "__name__", function.__class__.__name__)
