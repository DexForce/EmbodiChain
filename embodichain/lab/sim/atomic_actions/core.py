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

"""Core semantic objects and planning contract for atomic actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Generic, TYPE_CHECKING

import torch

from embodichain.lab.sim.common import BatchEntity

from .affordance import Affordance
from .effects import StateDelta
from .goals import collect_scene_dependencies
from .invocation import (
    ActionInvocation,
    ActionOptions,
    GoalT,
    OptionsT,
    ResolvedActionRequest,
)
from .plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
    normalize_success_mask,
)
from .policies import DynamicCollisionMode

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator

    from .runtime import ActionPlanningServices
    from .state import PlanningContext


def resolve_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device to the active concrete GPU index.

    Args:
        device: PyTorch device or device string.

    Returns:
        Concrete runtime device.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return resolved


@dataclass
class ObjectSemantics:
    """Semantic and geometric information about an interaction object."""

    affordance: Affordance
    """Affordance data describing supported interactions."""

    geometry: dict[str, Any]
    """Non-affordance geometric metadata."""

    properties: dict[str, Any] = field(default_factory=dict)
    """Physical properties such as mass and friction."""

    label: str = "none"
    """Semantic object category."""

    entity: BatchEntity | None = None
    """Optional simulation entity used by deterministic grounding."""

    def __post_init__(self) -> None:
        if not isinstance(self.affordance, Affordance):
            raise TypeError("affordance must be an Affordance instance.")
        if not isinstance(self.geometry, dict):
            raise TypeError("geometry must be a dict.")
        if not isinstance(self.properties, dict):
            raise TypeError("properties must be a dict.")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string.")
        self.affordance.object_label = self.label


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Machine-readable metadata for one registered atomic skill."""

    skill_id: str
    goal_type: type[Any] | tuple[type[Any], ...]
    options_type: type[ActionOptions]
    manipulator_roles: tuple[str, ...] = ()
    end_effector_roles: tuple[str, ...] = ()
    agent_visible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id:
            raise ValueError("SkillDescriptor.skill_id must be non-empty.")
        goal_types = (
            self.goal_type if isinstance(self.goal_type, tuple) else (self.goal_type,)
        )
        if not goal_types or not all(isinstance(item, type) for item in goal_types):
            raise TypeError("SkillDescriptor.goal_type must contain concrete types.")
        if not isinstance(self.options_type, type) or not issubclass(
            self.options_type, ActionOptions
        ):
            raise TypeError(
                "SkillDescriptor.options_type must be an ActionOptions subclass."
            )
        for field_name in ("manipulator_roles", "end_effector_roles"):
            roles = tuple(getattr(self, field_name))
            if len(set(roles)) != len(roles) or not all(
                isinstance(role, str) and role for role in roles
            ):
                raise ValueError(f"{field_name} must contain unique non-empty roles.")
            object.__setattr__(self, field_name, roles)


class AtomicAction(Generic[GoalT, OptionsT], ABC):
    """Side-effect-free planner for one semantically meaningful robot skill.

    Actions own only typed default runtime options. An
    :class:`~embodichain.lab.sim.atomic_actions.engine.AtomicActionEngine` binds
    its shared planning services before an action is invoked.
    """

    skill_id: ClassVar[str]
    """Stable registry identifier for this skill."""

    GoalType: ClassVar[type[Any] | tuple[type[Any], ...]]
    """Concrete goal dataclass or dataclasses accepted by this skill."""

    OptionsType: ClassVar[type[ActionOptions]] = ActionOptions
    """Concrete per-invocation runtime options accepted by this skill."""

    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    """Required semantic manipulator roles."""

    end_effector_roles: ClassVar[tuple[str, ...]] = ()
    """Required semantic end-effector roles."""

    agent_visible: ClassVar[bool] = True
    """Whether an Action Agent should expose this skill by default."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject skill classes that bypass framework-owned scene binding."""
        super().__init_subclass__(**kwargs)
        if "plan" in cls.__dict__:
            raise TypeError(
                "AtomicAction subclasses must implement _plan(); the public "
                "plan() method is framework-owned."
            )

    def __init__(
        self,
        default_options: OptionsT | None = None,
    ) -> None:
        selected_options = (
            self.OptionsType() if default_options is None else default_options
        )
        if not isinstance(selected_options, self.OptionsType):
            raise TypeError(
                f"{type(self).__name__} expects default_options of type "
                f"{self.OptionsType.__name__}, got "
                f"{type(selected_options).__name__}."
            )
        self._default_options: OptionsT = deepcopy(selected_options)
        self._planning_services: ActionPlanningServices | None = None

    @property
    def default_options(self) -> OptionsT:
        """Return an owned copy of the action's default runtime options."""
        return deepcopy(self._default_options)

    @property
    def is_bound(self) -> bool:
        """Whether an engine has supplied this action's planning resources."""
        return self._planning_services is not None

    @property
    def planning_services(self) -> ActionPlanningServices:
        """Return the engine-owned services borrowed by this action.

        Raises:
            RuntimeError: If the action has not been registered or planned by
                an
                :class:`~embodichain.lab.sim.atomic_actions.engine.AtomicActionEngine`.
        """
        if self._planning_services is None:
            raise RuntimeError(
                f"Atomic action {self.skill_id!r} is not bound to an "
                "AtomicActionEngine. Register it or call engine.plan_action()."
            )
        return self._planning_services

    @property
    def motion_generator(self) -> MotionGenerator:
        """Return the engine-owned motion generator borrowed by this action."""
        return self.planning_services.motion_generator

    @property
    def robot(self) -> Robot:
        """Return the robot associated with the owning engine."""
        return self.planning_services.robot

    @property
    def device(self) -> torch.device:
        """Return the concrete runtime device associated with the engine."""
        return self.planning_services.device

    def _bind(self, services: ActionPlanningServices) -> None:
        """Bind engine-owned planning services exactly once."""
        if self._planning_services is services:
            return
        if self._planning_services is not None:
            raise ValueError(
                f"Atomic action {self.skill_id!r} is already bound to another "
                "AtomicActionEngine."
            )
        self._planning_services = services
        try:
            self._on_bind()
        except Exception:
            self._planning_services = None
            raise

    def _on_bind(self) -> None:
        """Initialize implementation state that depends on engine resources."""

    @classmethod
    def descriptor(cls) -> SkillDescriptor:
        """Return stable metadata used by registries and Action Agent adapters."""
        return SkillDescriptor(
            skill_id=cls.skill_id,
            goal_type=cls.GoalType,
            options_type=cls.OptionsType,
            manipulator_roles=cls.manipulator_roles,
            end_effector_roles=cls.end_effector_roles,
            agent_visible=cls.agent_visible,
        )

    def resolve_request(
        self,
        invocation: ActionInvocation[GoalT, OptionsT],
    ) -> ResolvedActionRequest[GoalT, OptionsT]:
        """Validate and snapshot an invocation through engine-owned resources.

        Args:
            invocation: Caller-owned invocation to resolve.

        Returns:
            Immutable request reused by planning and recovery replans.

        Raises:
            ValueError: If the stable skill identifier does not match.
            TypeError: If the goal or options type is incompatible.
            KeyError: If a required binding role is missing.
        """
        if invocation.skill_id != self.skill_id:
            raise ValueError(
                f"Invocation skill_id {invocation.skill_id!r} does not match "
                f"{self.skill_id!r}."
            )
        if not isinstance(invocation.goal, self.GoalType):
            expected = (
                " | ".join(item.__name__ for item in self.GoalType)
                if isinstance(self.GoalType, tuple)
                else self.GoalType.__name__
            )
            raise TypeError(
                f"Skill {self.skill_id!r} expects goal {expected}, got "
                f"{type(invocation.goal).__name__}."
            )
        for role in self.manipulator_roles:
            invocation.binding.manipulator(role)
        for role in self.end_effector_roles:
            invocation.binding.end_effector(role)
        options = (
            self._default_options
            if invocation.skill_options is None
            else invocation.skill_options
        )
        if not isinstance(options, self.OptionsType):
            raise TypeError(
                f"Skill {self.skill_id!r} expects options "
                f"{self.OptionsType.__name__}, got {type(options).__name__}."
            )
        required_planner = invocation.motion_policy.planner
        configured_planner_name = self.planning_services.planner_name
        if required_planner is not None and required_planner != configured_planner_name:
            raise ValueError(
                f"Motion policy requires planner {required_planner!r}, but this "
                f"action uses {configured_planner_name!r}."
            )
        return ResolvedActionRequest(
            skill_id=invocation.skill_id,
            goal=invocation.goal,
            binding=self.planning_services.resolve_binding(
                invocation.binding,
                invocation.control_overrides,
            ),
            motion_policy=invocation.motion_policy,
            recovery_policy=invocation.recovery_policy,
            skill_options=options,
            invocation_id=invocation.invocation_id,
            revision=invocation.revision,
        )

    def require_goal(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
    ) -> GoalT:
        """Validate a resolved request and return its concrete goal."""
        if request.skill_id != self.skill_id:
            raise ValueError(
                f"Request skill_id {request.skill_id!r} does not match "
                f"{self.skill_id!r}."
            )
        if not isinstance(request.goal, self.GoalType):
            raise TypeError(
                f"Skill {self.skill_id!r} received incompatible goal "
                f"{type(request.goal).__name__}."
            )
        if not isinstance(request.skill_options, self.OptionsType):
            raise TypeError(
                f"Skill {self.skill_id!r} received incompatible options "
                f"{type(request.skill_options).__name__}."
            )
        for role in self.manipulator_roles:
            request.binding.manipulator(role)
        for role in self.end_effector_roles:
            request.binding.end_effector(role)
        return request.goal

    def plan(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
    ) -> ActionPlan:
        """Bind the current collision scene and invoke the skill planner.

        Args:
            request: Immutable, typed, and embodiment-resolved action request.
            context: Latest observed robot, task, and scene state.

        Returns:
            Scene-bound action plan with expected, uncommitted effects.
        """
        self.require_goal(request)
        prepared = self._prepare_request(request, context)
        return self._plan(prepared, context)

    def _prepare_request(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
    ) -> ResolvedActionRequest[GoalT, OptionsT]:
        """Bind snapshot obstacle poses without mutating the resolved request."""
        if not self._uses_collision_world(request, context):
            return request
        poses = context.scene.collision_obstacle_poses(
            batch_size=context.batch_size,
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        policy = replace(
            request.motion_policy,
            plan_opts=self.motion_generator.bind_collision_world(
                request.motion_policy.plan_opts,
                obstacle_poses=poses,
            ),
        )
        return replace(request, motion_policy=policy)

    def _uses_collision_world(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
    ) -> bool:
        """Return whether this planning attempt consumes collision revisions."""
        mode = request.motion_policy.dynamic_collision_mode
        if mode is DynamicCollisionMode.OFF:
            return False

        uses_motion_generator = request.motion_policy.strategy == "motion_gen"
        has_collision_entities = bool(context.scene.collision_entity_ids)
        supports_updates = (
            getattr(
                self.motion_generator,
                "supports_dynamic_collision_world",
                False,
            )
            is True
        )
        available = (
            uses_motion_generator and has_collision_entities and supports_updates
        )
        if mode is DynamicCollisionMode.REQUIRED and not available:
            missing: list[str] = []
            if not uses_motion_generator:
                missing.append("strategy='motion_gen'")
            if not has_collision_entities:
                missing.append("scene collision entities")
            if not supports_updates:
                missing.append("a planner with dynamic collision-world support")
            raise ValueError(
                "dynamic_collision_mode='required' cannot be satisfied; missing "
                + ", ".join(missing)
                + "."
            )
        return available

    def build_plan(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
        *,
        success: bool | torch.Tensor,
        trajectory: TimedTrajectory | torch.Tensor,
        expected_effects: StateDelta | None = None,
        replannable: bool = True,
        diagnostics: PlannerDiagnostics | None = None,
        segment_lengths: Mapping[str, int] | None = None,
    ) -> ActionPlan:
        """Build a validated action plan for a primitive implementation.

        Args:
            request: Resolved invocation snapshot being planned.
            context: Planning input used for the plan.
            success: Per-environment planning success or scalar planner result.
            trajectory: Full-robot timed trajectory or position tensor.
            expected_effects: Symbolic effects to verify after execution.
            replannable: Whether the execution runtime may replan this action.
            diagnostics: Optional retained planner diagnostics.
            segment_lengths: Optional ordered mapping from semantic segment
                names to waypoint counts. Zero-length entries are omitted.

        Returns:
            Side-effect-free action plan.
        """
        self.require_goal(request)
        success_mask = normalize_success_mask(
            success,
            n_envs=context.batch_size,
            device=self.device,
            name="Planning success",
        )

        if isinstance(trajectory, torch.Tensor):
            timed = TimedTrajectory.from_positions(
                trajectory,
                env_ids=context.env_ids,
                control_dt=request.motion_policy.control_dt,
            )
        elif isinstance(trajectory, TimedTrajectory):
            timed = trajectory
        else:
            raise TypeError("trajectory must be TimedTrajectory or torch.Tensor.")
        if timed.batch_size != context.batch_size:
            raise ValueError("Trajectory and planning context batch sizes must match.")
        if timed.robot_dof != context.robot.robot_dof:
            raise ValueError("Trajectory robot_dof must match the planning context.")
        timed = timed.hold_rows(success_mask, context.robot.qpos)

        segments: list[TrajectorySegment] = []
        if segment_lengths is not None:
            offset = 0
            for name, length in segment_lengths.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Trajectory segment names must be non-empty.")
                if isinstance(length, bool) or not isinstance(length, int):
                    raise TypeError("Trajectory segment lengths must be integers.")
                if length < 0:
                    raise ValueError("Trajectory segment lengths must be non-negative.")
                if length == 0:
                    continue
                segments.append(
                    TrajectorySegment(
                        name=name,
                        start=offset,
                        stop=offset + length,
                    )
                )
                offset += length
            if offset != timed.waypoint_count:
                raise ValueError(
                    "Trajectory segment lengths must sum to the trajectory "
                    f"waypoint count ({timed.waypoint_count}), got {offset}."
                )

        if diagnostics is None:
            diagnostics = PlannerDiagnostics(
                backend=self.planning_services.planner_name
            )
        return ActionPlan(
            skill_id=self.skill_id,
            plan_success=success_mask,
            trajectory=timed,
            recovery_policy=request.recovery_policy,
            planned_scene_version=context.scene.version,
            planned_collision_world_revision=(
                context.scene.collision_world_revisions(context.batch_size)
            ),
            diagnostics=diagnostics,
            segments=tuple(segments),
            scene_dependencies=collect_scene_dependencies(request.goal),
            collision_world_sensitive=self._uses_collision_world(
                request,
                context,
            ),
            replannable=replannable,
            expected_effects=expected_effects or StateDelta(),
            invocation_id=request.invocation_id,
            invocation_revision=request.revision,
        )

    def failed_plan(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
        *,
        message: str | None = None,
    ) -> ActionPlan:
        """Build a failed empty plan without changing task state.

        Args:
            request: Resolved invocation that failed to plan.
            context: Planning input used for the attempt.
            message: Optional diagnostic message.

        Returns:
            Failed action plan with an empty trajectory.
        """
        return self.build_plan(
            request,
            context,
            success=torch.zeros(
                context.batch_size, dtype=torch.bool, device=self.device
            ),
            trajectory=TimedTrajectory.empty(
                batch_size=context.batch_size,
                robot_dof=context.robot.robot_dof,
                device=self.device,
                env_ids=context.env_ids,
            ),
            replannable=True,
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                messages=(() if message is None else (message,)),
            ),
        )

    @abstractmethod
    def _plan(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan one invocation without stepping simulation or committing state.

        Args:
            request: Immutable, typed, and embodiment-resolved action request.
            context: Latest observed robot, task, and scene state.

        Returns:
            Scene-bound action plan with expected, uncommitted effects.
        """


__all__ = [
    "AtomicAction",
    "ObjectSemantics",
    "SkillDescriptor",
    "resolve_runtime_device",
]
