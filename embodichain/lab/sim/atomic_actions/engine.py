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

from types import MappingProxyType
from typing import Iterable, Mapping, TYPE_CHECKING

import torch

from .bindings import ActionBinding
from .core import AtomicAction, SkillDescriptor
from .control import ActionControlOverrides, ControlPartCommandProfile
from .invocation import ActionInvocation, GoalT, OptionsT, ResolvedActionRequest
from .plans import ActionPlan, CompiledTrajectory, TimedTrajectory
from .policies import MotionPolicy, RecoveryPolicy
from .runtime import ActionPlanningServices
from .state import PlanningContext, RobotObservation, SceneSnapshot, TaskState

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.skills import (
        BoundRobotSkillProfile,
        ResourceEndpoint,
        ResourceEndpointAdapter,
        RobotSkillProfile,
    )

    from .execution import ExecutionSession


class AtomicActionEngine:
    """Own planning resources and coordinate side-effect-free atomic actions."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_profiles: Mapping[str, ControlPartCommandProfile] | None = None,
        *,
        load_builtins: bool = True,
        skill_profile: RobotSkillProfile | None = None,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
    ) -> None:
        """Initialize one engine and bind its built-in action implementations.

        Args:
            motion_generator: Engine-owned motion-generation backend.
            control_profiles: Semantic commands keyed by robot control-part name.
            load_builtins: Whether to instantiate and register every built-in
                action. Disable this for isolated tests or fully custom engines.
            skill_profile: Optional authoritative robot skill profile. Its
                command profiles are installed automatically and validated
                after built-in actions are loaded. ``control_profiles`` and
                ``skill_profile`` are mutually exclusive.
            endpoint_adapters: Optional exact-type endpoint adapters used when
                binding ``skill_profile``. Invalid without a profile.
        """
        if endpoint_adapters is not None and skill_profile is None:
            raise ValueError("endpoint_adapters requires skill_profile.")
        if skill_profile is not None:
            from embodichain.lab.sim.skills import RobotSkillProfile

            if not isinstance(skill_profile, RobotSkillProfile):
                raise TypeError("skill_profile must be a RobotSkillProfile or None.")
            if control_profiles is not None:
                raise ValueError(
                    "control_profiles and skill_profile are mutually exclusive; "
                    "the profile is the authoritative semantic-command source."
                )
            control_profiles = skill_profile.action_control_profiles()
        self._planning_services = ActionPlanningServices(
            motion_generator,
            control_profiles=control_profiles,
        )
        self._actions: dict[str, AtomicAction] = {}
        self._skill_catalog_revision = 0
        self._skill_profile: BoundRobotSkillProfile | None = None
        if load_builtins:
            self._load_builtin_actions()
        if skill_profile is not None:
            self._skill_profile = skill_profile.bind(
                self,
                endpoint_adapters=endpoint_adapters,
            )

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
    def binding_owner_id(self) -> str:
        """Return the opaque owner identity required by action bindings."""
        return self._planning_services.binding_owner_id

    @property
    def control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Semantic command profiles registered for robot control parts."""
        return self._planning_services.control_profiles

    @property
    def actions(self) -> dict[str, AtomicAction]:
        """Registered action instances keyed by stable skill identifier."""
        return dict(self._actions)

    @property
    def skills(self) -> Mapping[str, SkillDescriptor]:
        """Return explicitly declared, agent-visible installed skill metadata.

        Process-wide type discovery, engine installation, and semantic exposure
        are separate boundaries. Only an action installed in this engine whose
        concrete class explicitly declares a generic binding contract appears
        here. Direct-core callers may continue to use every entry in
        :attr:`actions`.
        """
        return MappingProxyType(
            {
                skill_id: descriptor
                for skill_id, action in self._actions.items()
                if (descriptor := action.descriptor()).agent_visible
                and descriptor.binding_contract is not None
            }
        )

    @property
    def skill_catalog_revision(self) -> int:
        """Return the monotonic installed semantic-skill catalog revision.

        Replacing an agent-visible implementation advances the revision even
        when its public descriptor is equal. Bound profiles and semantic
        compilers can therefore reject stale implementation ownership.
        """
        return self._skill_catalog_revision

    @property
    def skill_profile(self) -> BoundRobotSkillProfile | None:
        """Return the currently bound semantic robot profile, when configured."""
        return self._skill_profile

    def bind_skill_profile(
        self,
        profile: RobotSkillProfile,
        *,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
    ) -> BoundRobotSkillProfile:
        """Validate and bind a profile after custom action installation.

        The engine's immutable control-part profiles must already contain the
        profile commands lowered into the current action core. Generic
        non-core endpoint commands remain on resolved endpoints. Prefer the
        constructor's ``skill_profile`` argument when no custom actions need
        to be installed first.

        Args:
            profile: Authoritative robot resource and policy profile.
            endpoint_adapters: Optional exact-type endpoint adapters used for
                custom controller declarations.

        Returns:
            Validated profile bound to this engine and its installed actions.
        """
        from embodichain.lab.sim.skills import RobotSkillProfile

        if not isinstance(profile, RobotSkillProfile):
            raise TypeError("profile must be a RobotSkillProfile.")
        bound = profile.bind(self, endpoint_adapters=endpoint_adapters)
        self._skill_profile = bound
        return bound

    def bind_control_parts(
        self,
        skill: str | AtomicAction,
        endpoints: Mapping[str, Mapping[str, str]],
    ) -> ActionBinding:
        """Build an advanced direct-core binding from control-part names.

        Args:
            skill: Installed skill ID or an explicit action passed later to
                :meth:`plan_action`.
            endpoints: Nested ``slot_id -> endpoint_id -> control_part`` mapping.

        Returns:
            Engine-owned generic endpoint binding.
        """
        if isinstance(skill, str):
            action = self._actions.get(skill)
            if action is None:
                raise KeyError(f"No atomic action registered for skill {skill!r}.")
        elif isinstance(skill, AtomicAction):
            action = skill
            if (
                action.is_bound
                and action.planning_services is not self._planning_services
            ):
                raise ValueError(
                    f"Atomic action {action.skill_id!r} belongs to another engine."
                )
        else:
            raise TypeError("skill must be an installed skill ID or AtomicAction.")
        contract = type(action).__dict__.get("binding_contract")
        if contract is None:
            raise ValueError(
                f"Skill {action.skill_id!r} has no explicit SkillBindingContract."
            )
        return self._planning_services.bind_control_parts(contract, endpoints)

    def make_invocation(
        self,
        skill_id: str,
        goal: GoalT,
        *,
        control_parts: Mapping[str, Mapping[str, str]] | None = None,
        resources: Mapping[str, str] | None = None,
        motion_policy: MotionPolicy | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        skill_options: OptionsT | None = None,
        control_overrides: ActionControlOverrides | None = None,
        invocation_id: str | None = None,
        revision: int = 0,
    ) -> ActionInvocation[GoalT, OptionsT]:
        """Construct a grounded invocation while naming the skill only once.

        ``control_parts`` uses the advanced direct-core binding path. When it is
        omitted, the engine must own a bound robot skill profile; ``resources``
        then optionally selects logical resource IDs by skill-local slot. An
        omitted resource selection uses the profile's unique or default binding.
        This method resolves bindings only; profile policy presets and runner
        configuration remain responsibilities of the semantic runtime layer.

        Args:
            skill_id: Stable identifier of an installed atomic skill.
            goal: Action-specific typed goal.
            control_parts: Optional direct ``slot -> endpoint -> control_part``
                mapping.
            resources: Optional profile ``slot -> resource_id`` selections.
            motion_policy: Optional invocation motion policy.
            recovery_policy: Optional invocation recovery policy.
            skill_options: Optional action-specific invocation options.
            control_overrides: Optional endpoint-scoped command overrides.
            invocation_id: Optional correlation identifier.
            revision: Monotonic invocation revision.

        Returns:
            A standard :class:`ActionInvocation` accepted by ``plan``,
            ``compile``, and ``start``.

        Raises:
            ValueError: If binding sources conflict or no binding source is
                available.
            KeyError: If the skill or an explicitly selected resource is unknown.
            TypeError: If an invocation field or binding input has an invalid type.
        """
        if control_parts is not None and resources is not None:
            raise ValueError("control_parts and resources are mutually exclusive.")
        if control_parts is not None:
            binding = self.bind_control_parts(skill_id, control_parts)
        else:
            profile = self.skill_profile
            if profile is None:
                if resources is not None:
                    raise ValueError("resources requires a bound RobotSkillProfile.")
                raise ValueError(
                    "control_parts is required when no RobotSkillProfile is bound."
                )
            binding = profile.resolve(skill_id, resources).action_binding

        return ActionInvocation(
            skill_id=skill_id,
            goal=goal,
            binding=binding,
            motion_policy=MotionPolicy() if motion_policy is None else motion_policy,
            recovery_policy=(
                RecoveryPolicy() if recovery_policy is None else recovery_policy
            ),
            skill_options=skill_options,
            control_overrides=(
                ActionControlOverrides()
                if control_overrides is None
                else control_overrides
            ),
            invocation_id=invocation_id,
            revision=revision,
        )

    def register(self, action: AtomicAction, *, replace: bool = False) -> None:
        """Register one action instance using its descriptor.

        Args:
            action: Configured action instance.
            replace: Whether to replace an implementation already registered
                under the same stable skill identifier. Replacement is always
                explicit so extensions cannot silently shadow built-ins.

        Raises:
            TypeError: If ``action`` is not an AtomicAction.
            ValueError: If it belongs to another engine or its skill identifier
                conflicts with an existing action.
        """
        if not isinstance(action, AtomicAction):
            raise TypeError("action must be an AtomicAction instance.")
        descriptor = action.descriptor()
        existing = self._actions.get(descriptor.skill_id)
        if existing is not None and existing is not action and not replace:
            raise ValueError(
                f"Skill id {descriptor.skill_id!r} is already registered in this engine."
            )
        action._bind(self._planning_services)
        self._actions[descriptor.skill_id] = action
        existing_descriptor = None if existing is None else existing.descriptor()
        if (descriptor.agent_visible and descriptor.binding_contract is not None) or (
            existing_descriptor is not None
            and existing_descriptor.agent_visible
            and existing_descriptor.binding_contract is not None
        ):
            self._skill_catalog_revision += 1
        self._skill_profile = None

    def _load_builtin_actions(self) -> None:
        """Create and bind fresh built-in action instances for this engine."""
        # Import lazily to keep the engine/core dependency independent from the
        # concrete primitive modules and to avoid package import cycles.
        from .primitives import BUILTIN_ACTION_TYPES

        for action_type in BUILTIN_ACTION_TYPES:
            self.register(action_type())

    def plan_action(
        self,
        action: AtomicAction,
        invocation: ActionInvocation,
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan with an unregistered action using this engine's resources.

        This is an advanced extension and testing escape hatch. Built-in
        parameter variants should use invocation ``skill_options`` with the
        engine's registered implementation.

        Args:
            action: Configured action implementation to invoke.
            invocation: Grounded request matching the action skill identifier.
            context: Latest measured planning state.

        Returns:
            Validated side-effect-free action plan.
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
        """Resolve a registered invocation into an engine-owned snapshot."""
        return self._resolve(invocation)

    def plan_request(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext | None = None,
    ) -> ActionPlan:
        """Plan an already-resolved request without rebuilding its snapshot."""
        return self._plan_request(request, context)

    def _resolve(
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

    def _plan_request(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext | None = None,
    ) -> ActionPlan:
        """Plan an already-resolved request without rebuilding its snapshot.

        This is the planning entry point used by execution recovery. Callers
        normally use :meth:`plan`, while an execution session resolves once and
        calls this method for every replan.

        Args:
            request: Immutable request previously returned by :meth:`_resolve`.
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
        request = self._resolve(invocation)
        return self._plan_request(request, current)

    def initial_context(
        self,
        *,
        task: TaskState | None = None,
        scene: SceneSnapshot | None = None,
        timestamp: float = 0.0,
        control_dt: float | None = None,
    ) -> PlanningContext:
        """Capture the robot state needed to start offline compilation.

        Args:
            task: Optional symbolic task state; an empty state is used otherwise.
            scene: Optional scene snapshot; an empty snapshot is used otherwise.
            timestamp: Timestamp assigned to the captured robot observation.
            control_dt: Explicit command period for action-owned interpolation.

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
            control_dt=control_dt,
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
            if plan.joint_trajectory is None:
                raise ValueError(
                    f"Skill {plan.skill_id!r} emits non-joint runtime commands and "
                    "cannot be used with offline joint-trajectory compilation."
                )
            trajectory = plan.joint_trajectory.hold_rows(
                step_success,
                previous_qpos,
            )
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
        commands = plan.commands
        if commands.batch_size != context.batch_size:
            raise ValueError("Action plan batch size does not match the context.")
        if commands.device != self.device:
            raise ValueError("Action plan and engine must share a device.")
        if not torch.equal(commands.env_ids, context.env_ids):
            raise ValueError("Action plan and context must share ordered env_ids.")
        if plan.joint_trajectory is not None:
            if plan.joint_trajectory.robot_dof != self.robot.dof:
                raise ValueError(
                    "Action plan joint_trajectory robot_dof does not match the "
                    "engine robot."
                )
            if plan.joint_trajectory.positions.device != self.device:
                raise ValueError(
                    "Action plan joint_trajectory and engine must share a device."
                )
        if plan.planned_scene_version != context.scene.version:
            raise ValueError("Action plan must record the planning scene version.")
        collision_revision = context.scene.collision_world_revisions(context.batch_size)
        if plan.planned_collision_world_revision != collision_revision:
            raise ValueError(
                "Action plan must record the planning collision-world revision."
            )


__all__ = ["AtomicActionEngine"]
