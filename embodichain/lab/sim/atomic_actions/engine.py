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

"""Registry and offline compiler for side-effect-free atomic actions."""

from __future__ import annotations

from typing import Iterable, Mapping, TYPE_CHECKING

import torch

from .core import AtomicAction
from .control import ControlPartCommandProfile
from .invocation import ActionInvocation, ResolvedActionRequest
from .plans import ActionPlan, CompiledTrajectory, TimedTrajectory
from .runtime import ActionPlanningServices
from .state import PlanningContext, RobotObservation, SceneSnapshot, TaskState

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator

    from .execution import ExecutionSession


_global_action_registry: dict[str, type[AtomicAction]] = {}


def register_action(action_class: type[AtomicAction]) -> None:
    """Register an atomic action class under its stable skill identifier.

    Args:
        action_class: Concrete :class:`AtomicAction` subclass.

    Raises:
        TypeError: If ``action_class`` is not an AtomicAction subclass.
        ValueError: If another class already owns the same skill identifier.
    """
    if not isinstance(action_class, type) or not issubclass(action_class, AtomicAction):
        raise TypeError("action_class must be an AtomicAction subclass.")
    descriptor = action_class.descriptor()
    existing = _global_action_registry.get(descriptor.skill_id)
    if existing is not None and existing is not action_class:
        raise ValueError(
            f"Skill id {descriptor.skill_id!r} is already registered by "
            f"{existing.__name__}."
        )
    _global_action_registry[descriptor.skill_id] = action_class


def unregister_action(skill_id: str) -> None:
    """Remove a globally registered skill class if present.

    Args:
        skill_id: Stable registered skill identifier.
    """
    _global_action_registry.pop(skill_id, None)


def get_registered_actions() -> dict[str, type[AtomicAction]]:
    """Return a copy of the global skill-class registry."""
    return dict(_global_action_registry)


class AtomicActionEngine:
    """Own planning resources and coordinate side-effect-free atomic actions."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_profiles: Mapping[str, ControlPartCommandProfile] | None = None,
    ) -> None:
        self._planning_services = ActionPlanningServices(
            motion_generator,
            control_profiles=control_profiles,
        )
        self._actions: dict[str, AtomicAction] = {}

    @property
    def motion_generator(self) -> MotionGenerator:
        """Return the single motion generator owned by this engine."""
        return self._planning_services.motion_generator

    @property
    def robot(self) -> Robot:
        """Return the robot controlled by this engine."""
        return self._planning_services.robot

    @property
    def device(self) -> torch.device:
        """Return the concrete planning device used by this engine."""
        return self._planning_services.device

    @property
    def planning_services(self) -> ActionPlanningServices:
        """Engine-owned resources shared by every bound atomic action."""
        return self._planning_services

    @property
    def control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Semantic command profiles registered for robot control parts."""
        return self._planning_services.control_profiles

    @property
    def actions(self) -> dict[str, AtomicAction]:
        """Registered action instances keyed by stable skill identifier."""
        return dict(self._actions)

    def register(self, action: AtomicAction) -> None:
        """Register one action instance using its descriptor.

        Args:
            action: Configured action instance.

        Raises:
            TypeError: If ``action`` is not an AtomicAction.
            ValueError: If it belongs to another engine or its skill identifier
                conflicts with an existing action.
        """
        if not isinstance(action, AtomicAction):
            raise TypeError("action must be an AtomicAction instance.")
        descriptor = action.descriptor()
        existing = self._actions.get(descriptor.skill_id)
        if existing is not None and existing is not action:
            raise ValueError(
                f"Skill id {descriptor.skill_id!r} is already registered in this engine."
            )
        action._bind(self._planning_services)
        self._actions[descriptor.skill_id] = action

    def plan_action(
        self,
        action: AtomicAction,
        invocation: ActionInvocation,
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan with a configured action using this engine's resources.

        Unlike :meth:`plan`, the supplied action does not need to be in the
        skill registry. This supports multiple configured instances with the
        same stable skill identifier while preserving one engine-owned motion
        generator.

        Args:
            action: Configured action implementation to invoke.
            invocation: Grounded request matching the action's skill identifier.
            context: Latest measured planning state.

        Returns:
            Validated side-effect-free action plan.

        Raises:
            TypeError: If ``action`` is not an :class:`AtomicAction`.
            ValueError: If the action, invocation, context, or plan is invalid.
        """
        if not isinstance(action, AtomicAction):
            raise TypeError("action must be an AtomicAction instance.")
        self._validate_context(context)
        action._bind(self._planning_services)
        request = action.resolve_request(invocation)
        plan = action.plan(request, context)
        self._validate_plan(plan, context, request)
        return plan

    def resolve(
        self,
        invocation: ActionInvocation,
    ) -> ResolvedActionRequest:
        """Resolve one registered invocation into an engine-owned snapshot.

        The returned request owns its policy and skill-option values. Closed-loop
        recovery reuses the same request so a replan cannot observe later
        mutations of caller-owned configuration objects.

        Args:
            invocation: Grounded request for a registered skill.

        Returns:
            Validated and embodiment-resolved request snapshot.

        Raises:
            KeyError: If the invocation references an unregistered skill.
        """
        action = self._actions.get(invocation.skill_id)
        if action is None:
            raise KeyError(
                f"No atomic action registered for skill {invocation.skill_id!r}."
            )
        return action.resolve_request(invocation)

    def plan_request(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext | None = None,
    ) -> ActionPlan:
        """Plan an already-resolved request without rebuilding its snapshot.

        This is the planning entry point used by execution recovery. Callers
        normally use :meth:`plan`, while an execution session resolves once and
        calls this method for every replan.

        Args:
            request: Immutable request previously returned by :meth:`resolve`.
            context: Optional latest planning state; captured when omitted.

        Returns:
            Validated side-effect-free action plan.
        """
        if not isinstance(request, ResolvedActionRequest):
            raise TypeError("request must be a ResolvedActionRequest.")
        action = self._actions.get(request.skill_id)
        if action is None:
            raise KeyError(
                f"No atomic action registered for skill {request.skill_id!r}."
            )
        current = self.initial_context() if context is None else context
        self._validate_context(current)
        plan = action.plan(request, current)
        self._validate_plan(plan, current, request)
        return plan

    def plan(
        self,
        invocation: ActionInvocation,
        context: PlanningContext | None = None,
    ) -> ActionPlan:
        """Plan one registered invocation through the engine-owned backend.

        Args:
            invocation: Grounded request for a registered skill.
            context: Optional latest planning state; captured when omitted.

        Returns:
            Validated action plan.

        Raises:
            KeyError: If the invocation references an unregistered skill.
        """
        current = self.initial_context() if context is None else context
        request = self.resolve(invocation)
        return self.plan_request(request, current)

    def initial_context(
        self,
        *,
        task: TaskState | None = None,
        scene: SceneSnapshot | None = None,
        timestamp: float = 0.0,
    ) -> PlanningContext:
        """Capture the robot state needed to start offline compilation.

        Args:
            task: Optional symbolic task state; an empty state is used otherwise.
            scene: Optional scene snapshot; an empty snapshot is used otherwise.
            timestamp: Timestamp assigned to the captured robot observation.

        Returns:
            Planning context containing owned robot tensors.
        """
        qpos = self.robot.get_qpos().to(self.device).clone()
        qvel_value = None
        get_qvel = getattr(self.robot, "get_qvel", None)
        if callable(get_qvel):
            candidate = get_qvel()
            if isinstance(candidate, torch.Tensor):
                qvel_value = candidate.to(self.device)
        qvel = torch.zeros_like(qpos) if qvel_value is None else qvel_value
        batch_size = int(qpos.shape[0])
        if task is None:
            task = TaskState.empty(batch_size=batch_size, device=self.device)
        if scene is None:
            scene = SceneSnapshot.empty()
        return PlanningContext(
            robot=RobotObservation(timestamp=timestamp, qpos=qpos, qvel=qvel),
            task=task,
            scene=scene,
            env_ids=torch.arange(batch_size, dtype=torch.long, device=self.device),
        )

    def compile(
        self,
        invocations: Iterable[ActionInvocation],
        context: PlanningContext | None = None,
    ) -> CompiledTrajectory:
        """Compile a static sequence of grounded invocations.

        Planning is side-effect free. Expected effects are applied only to the
        returned hypothetical ``projected_context`` so following actions can be
        checked against the expected state. No simulator or observed task state
        is mutated.

        Args:
            invocations: Grounded action requests in execution order.
            context: Optional initial planning context captured by the caller.

        Returns:
            Concatenated timed trajectory, individual plans, and projected state.

        Raises:
            KeyError: If an invocation references an unregistered skill.
            ValueError: If context, plan, or trajectory dimensions are incompatible.
        """
        if context is None:
            context = self.initial_context()
        self._validate_context(context)

        alive = torch.ones(context.batch_size, dtype=torch.bool, device=self.device)
        plans: list[ActionPlan] = []
        trajectories: list[TimedTrajectory] = []
        projected = context

        for invocation in invocations:
            if not alive.any():
                break
            previous_qpos = projected.robot.qpos
            plan = self.plan(invocation, projected)
            step_success = alive & plan.plan_success.to(self.device)
            trajectory = plan.trajectory.hold_rows(step_success, previous_qpos)
            plans.append(plan)
            trajectories.append(trajectory)

            candidate_qpos = (
                trajectory.positions[:, -1]
                if trajectory.waypoint_count > 0
                else previous_qpos
            )
            next_qpos = torch.where(
                step_success[:, None], candidate_qpos, previous_qpos
            )
            next_task = plan.expected_effects.apply(projected.task, step_success)
            projected = projected.project(qpos=next_qpos, task=next_task)
            alive = step_success

        compiled = TimedTrajectory.concatenate(trajectories, empty_like=context)
        return CompiledTrajectory(
            plan_success=alive,
            trajectory=compiled,
            action_plans=tuple(plans),
            projected_context=projected,
        )

    def start(
        self,
        invocations: Iterable[ActionInvocation],
        context: PlanningContext | None = None,
    ) -> ExecutionSession:
        """Start closed-loop execution for a grounded invocation sequence.

        Args:
            invocations: Grounded action requests in execution order.
            context: Initial measured state and scene snapshot. The engine
                captures one when omitted.

        Returns:
            Stateful execution session advanced by ``session.tick(...)``.
        """
        from .execution import ExecutionSession

        initial = self.initial_context() if context is None else context
        return ExecutionSession(self, tuple(invocations), initial)

    def _validate_context(self, context: PlanningContext) -> None:
        """Validate an externally supplied planning context."""
        if context.robot.robot_dof != self.robot.dof:
            raise ValueError(
                "PlanningContext robot_dof must match the engine robot, "
                f"got {context.robot.robot_dof} and {self.robot.dof}."
            )
        robot_qpos = self.robot.get_qpos()
        if context.batch_size != int(robot_qpos.shape[0]):
            raise ValueError(
                "PlanningContext batch size must match the engine robot, "
                f"got {context.batch_size} and {robot_qpos.shape[0]}."
            )
        if context.robot.qpos.device != self.device:
            raise ValueError("PlanningContext and engine must share a device.")

    def _validate_plan(
        self,
        plan: ActionPlan,
        context: PlanningContext,
        request: ResolvedActionRequest,
    ) -> None:
        """Validate one action result before it is composed."""
        if plan.skill_id != request.skill_id:
            raise ValueError(
                "ActionPlan.skill_id must match its request, "
                f"got {plan.skill_id!r} and {request.skill_id!r}."
            )
        if plan.invocation_id != request.invocation_id:
            raise ValueError(
                "ActionPlan.invocation_id must preserve the invocation correlation id."
            )
        if plan.invocation_revision != request.revision:
            raise ValueError(
                "ActionPlan.invocation_revision must preserve the request revision."
            )
        trajectory = plan.trajectory
        if trajectory.batch_size != context.batch_size:
            raise ValueError("Action plan batch size does not match the context.")
        if trajectory.robot_dof != self.robot.dof:
            raise ValueError("Action plan robot_dof does not match the engine robot.")
        if trajectory.positions.device != self.device:
            raise ValueError("Action plan and engine must share a device.")
        if not torch.equal(trajectory.env_ids, context.env_ids):
            raise ValueError("Action plan and context must share ordered env_ids.")
        if any(
            phase.planned_scene_version != context.scene.version
            for phase in plan.phases
        ):
            raise ValueError(
                "Every action phase must record the planning scene version."
            )


__all__ = [
    "AtomicActionEngine",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
