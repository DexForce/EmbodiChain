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
from typing import Callable, Iterable, Mapping, TYPE_CHECKING

import torch

from .affordance_sampling import AffordanceSamplingContext
from .bindings import ActionBinding
from .core import AtomicAction, SkillDescriptor
from .control import ActionControlOverrides, ControlPartCommandProfile
from .invocation import ActionInvocation, GoalT, OptionsT, ResolvedActionRequest
from .nominal import NominalMotionSource
from .plans import ActionPlan, CompiledTrajectory, TimedTrajectory
from .policies import MotionPolicy, RecoveryPolicy
from .runtime import ActionPlanningServices
from .scene import SceneProvider
from .state import PlanningContext, RobotObservation, SceneSnapshot, TaskState
from .tracking import TrackingPolicy, TrackingRuntime

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.toolkits.graspkit import GraspPoseGenerator
    from embodichain.lab.sim.motion.expansion import (
        MappedRetimeResult,
        TrajectoryTemplate,
    )

    from .execution import ExecutionSession


PlanTransform = Callable[
    [ResolvedActionRequest, PlanningContext, ActionPlan],
    ActionPlan,
]


class AtomicActionEngine:
    """Own planning resources and coordinate side-effect-free atomic actions."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_profiles: Mapping[str, ControlPartCommandProfile] | None = None,
        grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
        *,
        load_builtins: bool = True,
        tracking_runtime: TrackingRuntime | None = None,
        scene_provider: SceneProvider | None = None,
    ) -> None:
        """Initialize one engine and bind its built-in action implementations.

        Args:
            motion_generator: Engine-owned motion-generation backend.
            control_profiles: Semantic commands keyed by robot control-part name.
            grasp_pose_generators: Standalone grasp-pose services keyed by the
                runtime target ID of each grasp endpoint, normally its robot
                control-part name.
            load_builtins: Whether to instantiate and register every built-in
                action. Disable this for isolated tests or fully custom engines.
            tracking_runtime: Optional exact-version feedback, projector, and
                metric registries. Built-in joint tracking is installed when
                omitted.
            scene_provider: Optional default scene-observation source used by
                :meth:`initial_context` when the caller does not supply an
                explicit scene snapshot. The provider is borrowed by reference.
        """
        if scene_provider is not None and not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider.")
        self._planning_services = ActionPlanningServices(
            motion_generator,
            control_profiles=control_profiles,
            tracking_runtime=tracking_runtime,
            grasp_pose_generators=grasp_pose_generators,
        )
        self._scene_provider = scene_provider
        self._actions: dict[str, AtomicAction] = {}
        self._skill_catalog_revision = 0
        if load_builtins:
            self._load_builtin_actions()

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
    def tracking_runtime(self) -> TrackingRuntime:
        """Typed endpoint-feedback runtime used by plans and sessions."""
        return self._planning_services.tracking_runtime

    @property
    def binding_owner_id(self) -> str:
        """Return the opaque owner identity required by action bindings."""
        return self._planning_services.binding_owner_id

    @property
    def control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Semantic command profiles registered for robot control parts."""
        return self._planning_services.control_profiles

    @property
    def grasp_pose_generators(self) -> Mapping[str, GraspPoseGenerator]:
        """Standalone grasp-pose services installed for endpoint targets."""
        return self._planning_services.grasp_pose_generators

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
        """Return the monotonic installed Atomic Skill catalog revision.

        Installing or replacing an agent-visible implementation advances the
        revision even when its public descriptor is equal. External binding and
        compilation layers can therefore reject stale implementation snapshots.
        """
        return self._skill_catalog_revision

    def bind_control_parts(
        self,
        skill_id: str,
        endpoints: Mapping[str, Mapping[str, str]],
        *,
        task_state_keys: Mapping[str, str] | None = None,
    ) -> ActionBinding:
        """Build an advanced direct-core binding from control-part names.

        Args:
            skill_id: Installed skill ID.
            endpoints: Nested ``slot_id -> endpoint_id -> control_part`` mapping.
            task_state_keys: Optional explicit stable task-state key for each
                resource slot. See :meth:`ActionPlanningServices.bind_control_parts`
                for inference rules when omitted.

        Returns:
            Engine-owned generic endpoint binding.
        """
        if type(skill_id) is not str:
            raise TypeError("skill_id must be a string.")
        action = self._actions.get(skill_id)
        if action is None:
            raise KeyError(f"No atomic action registered for skill {skill_id!r}.")
        contract = type(action).__dict__.get("binding_contract")
        if contract is None:
            raise ValueError(
                f"Skill {action.skill_id!r} has no explicit SkillBindingContract."
            )
        return self._planning_services.bind_control_parts(
            contract,
            endpoints,
            task_state_keys=task_state_keys,
        )

    def make_invocation(
        self,
        skill_id: str,
        goal: GoalT,
        *,
        control_parts: Mapping[str, Mapping[str, str]] | None = None,
        motion_policy: MotionPolicy | None = None,
        tracking_policy: TrackingPolicy | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        skill_options: OptionsT | None = None,
        control_overrides: ActionControlOverrides | None = None,
        invocation_id: str | None = None,
        revision: int = 0,
    ) -> ActionInvocation[GoalT, OptionsT]:
        """Construct a grounded invocation while naming the skill only once.

        ``control_parts`` uses the advanced direct-core binding path. Profile-
        based integrations resolve an :class:`ActionBinding` in the semantic
        layer and construct :class:`ActionInvocation` directly.

        Args:
            skill_id: Stable identifier of an installed atomic skill.
            goal: Action-specific typed goal.
            control_parts: Direct ``slot -> endpoint -> control_part`` mapping.
            motion_policy: Optional invocation motion policy.
            tracking_policy: Optional typed tracking and terminal-acceptance policy.
            recovery_policy: Optional invocation recovery policy.
            skill_options: Optional action-specific invocation options.
            control_overrides: Optional endpoint-scoped command overrides.
            invocation_id: Optional correlation identifier.
            revision: Monotonic invocation revision.

        Returns:
            A standard :class:`ActionInvocation` accepted by ``plan``,
            ``compile``, and ``start``.

        Raises:
            ValueError: If ``control_parts`` is omitted.
            KeyError: If the skill or control part is unknown.
            TypeError: If an invocation field or binding input has an invalid type.
        """
        if control_parts is None:
            raise ValueError("control_parts is required for direct-core invocation.")
        binding = self.bind_control_parts(skill_id, control_parts)

        return ActionInvocation(
            skill_id=skill_id,
            goal=goal,
            binding=binding,
            motion_policy=MotionPolicy() if motion_policy is None else motion_policy,
            tracking_policy=(
                TrackingPolicy.joint_position()
                if tracking_policy is None
                else tracking_policy
            ),
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

    def _load_builtin_actions(self) -> None:
        """Create and bind fresh built-in action instances for this engine."""
        # Import lazily to keep the engine/core dependency independent from the
        # concrete primitive modules and to avoid package import cycles.
        from .primitives import BUILTIN_ACTION_TYPES

        for action_type in BUILTIN_ACTION_TYPES:
            self.register(action_type())

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
        *,
        plan_transform: PlanTransform | None = None,
        nominal_source: NominalMotionSource | None = None,
    ) -> ActionPlan:
        """Plan an already-resolved request without rebuilding its snapshot.

        Args:
            request: Immutable request previously returned by :meth:`resolve`.
            context: Optional latest planning state; captured when omitted.
            plan_transform: Optional callback scoped to this planning call. The
                callback receives and must return independently owned plan values.

        Returns:
            Validated side-effect-free action plan.
        """
        return self._plan_request(
            request,
            context,
            plan_transform=plan_transform,
            nominal_source=nominal_source,
        )

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
        *,
        plan_transform: PlanTransform | None = None,
        nominal_source: NominalMotionSource | None = None,
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
        if nominal_source is not None and not isinstance(
            nominal_source, NominalMotionSource
        ):
            raise TypeError("nominal_source must implement NominalMotionSource.")
        plan = (
            action.plan(request, current)
            if nominal_source is None
            else action.plan(request, current, nominal_source=nominal_source)
        )
        self._validate_plan(plan, current, request)
        if plan_transform is not None:
            if not callable(plan_transform):
                raise TypeError("plan_transform must be callable or None.")
            transformed = plan_transform(request, current, plan.snapshot())
            if not isinstance(transformed, ActionPlan):
                raise TypeError("plan_transform must return an ActionPlan.")
            owned = transformed.snapshot()
            self._validate_plan(owned, current, request)
            plan = owned
        return plan

    def plan(
        self,
        invocation: ActionInvocation,
        context: PlanningContext | None = None,
        *,
        plan_transform: PlanTransform | None = None,
        nominal_source: NominalMotionSource | None = None,
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
        return self._plan_request(
            request,
            current,
            plan_transform=plan_transform,
            nominal_source=nominal_source,
        )

    def initial_context(
        self,
        *,
        task: TaskState | None = None,
        scene: SceneSnapshot | None = None,
        timestamp: float = 0.0,
        control_dt: float | None = None,
        affordance_sampling: AffordanceSamplingContext | None = None,
    ) -> PlanningContext:
        """Capture the robot state needed to start offline compilation.

        Args:
            task: Optional symbolic task state; an empty state is used otherwise.
            scene: Optional explicit scene snapshot. It overrides the engine's
                configured scene provider; an empty snapshot is used when both
                are absent.
            timestamp: Timestamp assigned to the captured robot observation.
            control_dt: Explicit command period for action-owned interpolation.
            affordance_sampling: Optional reproducible Affordance sampling stream.

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
        env_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)
        if task is None:
            task = TaskState.empty(batch_size=batch_size, device=self.device)
        if scene is None:
            if self._scene_provider is None:
                scene = SceneSnapshot.empty()
            else:
                scene = self._scene_provider.snapshot(
                    timestamp=timestamp,
                    env_ids=env_ids.clone(),
                )
                if not isinstance(scene, SceneSnapshot):
                    raise TypeError("scene_provider must return a SceneSnapshot.")
        return PlanningContext(
            robot=RobotObservation(timestamp=timestamp, qpos=qpos, qvel=qvel),
            task=task,
            scene=scene,
            env_ids=env_ids,
            control_dt=control_dt,
            affordance_sampling=affordance_sampling,
        )

    def compile(
        self,
        invocations: Iterable[ActionInvocation],
        context: PlanningContext | None = None,
        *,
        plan_transform: PlanTransform | None = None,
    ) -> CompiledTrajectory:
        """Compile a static sequence of grounded invocations.

        Planning is side-effect free. Expected effects are applied only to the
        returned hypothetical ``projected_context`` so following actions can be
        checked against the expected state. No simulator or observed task state
        is mutated.

        Args:
            invocations: Grounded action requests in execution order.
            context: Optional initial planning context captured by the caller.
            plan_transform: Optional same-grid transform applied to each plan
                before concatenation and projected-state propagation.

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
            plan = self.plan(invocation, projected, plan_transform=plan_transform)
            step_success = alive & plan.plan_success.to(self.device)
            if plan.joint_trajectory is None:
                raise ValueError(
                    f"Skill {plan.skill_id!r} emits non-joint runtime commands and "
                    "cannot be used with offline joint-trajectory compilation."
                )
            preserve_failed_positions = (
                getattr(
                    self.motion_generator.planner,
                    "preserve_failed_plan_positions",
                    False,
                )
                is True
            )
            trajectory = (
                plan.joint_trajectory
                if preserve_failed_positions
                else plan.joint_trajectory.hold_rows(step_success, previous_qpos)
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
        *,
        eligible_mask: torch.Tensor | None = None,
        plan_transform: PlanTransform | None = None,
        nominal_source: NominalMotionSource | None = None,
    ) -> ExecutionSession:
        """Start incremental execution for a grounded invocation sequence.

        Args:
            invocations: Grounded action requests in execution order.
            context: Initial measured state and scene snapshot. The engine
                captures one when omitted.
            eligible_mask: Optional per-environment cohort allowed to execute.
                Ineligible rows remain excluded for the whole session. All rows
                are eligible when omitted.
            plan_transform: Optional transform scoped to every planning call in
                the returned execution session.

        Returns:
            Stateful execution session advanced by ``session.tick(...)``.
        """
        from .execution import ExecutionSession

        initial = self.initial_context() if context is None else context
        return ExecutionSession(
            self,
            tuple(invocations),
            initial,
            eligible_mask=eligible_mask,
            plan_transform=plan_transform,
            nominal_source=nominal_source,
        )

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
        if plan.tracking_policy != request.tracking_policy:
            raise ValueError(
                "ActionPlan.tracking_policy must preserve the resolved request "
                "tracking policy."
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

    def rebuild_plan_from_trajectory(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        plan: ActionPlan,
        trajectory: TimedTrajectory,
    ) -> ActionPlan:
        """Rebuild a complete plan from a same-grid full-joint trajectory.

        This is the execution-side boundary for trajectory expansion. The
        caller supplies a trajectory with the same frame count as the original
        plan; timing-grid remapping and phase-index changes remain the
        responsibility of a later adapter.

        Args:
            request: Resolved request that produced ``plan``.
            context: Planning context used for the original plan.
            plan: Original validated action plan.
            trajectory: Replacement full-joint trajectory on the same grid.

        Returns:
            A newly lowered and validated action plan with commands, tracking,
            segments, and effect metadata rebuilt together.
        """
        if not isinstance(plan, ActionPlan):
            raise TypeError("plan must be an ActionPlan.")
        if not isinstance(trajectory, TimedTrajectory):
            raise TypeError("trajectory must be a TimedTrajectory.")
        self._validate_context(context)
        owned_plan = plan.snapshot()
        self._validate_plan(owned_plan, context, request)
        original = owned_plan.joint_trajectory
        if original is None:
            raise ValueError("same-grid plan rebuilding requires a joint trajectory.")
        if trajectory.batch_size != context.batch_size:
            raise ValueError("trajectory batch must match the planning context.")
        if trajectory.waypoint_count != owned_plan.commands.frame_count:
            raise ValueError(
                "same-grid plan rebuilding requires the original frame count."
            )
        if not torch.equal(trajectory.env_ids, context.env_ids):
            raise ValueError("trajectory env_ids must match the planning context.")
        if trajectory.dt.shape != original.dt.shape or not torch.equal(
            trajectory.dt,
            original.dt,
        ):
            raise ValueError("same-grid plan rebuilding requires identical dt.")
        action = self._actions.get(request.skill_id)
        if action is None:
            raise KeyError(
                f"No atomic action registered for skill {request.skill_id!r}."
            )
        segment_lengths = {
            segment.name: segment.stop - segment.start
            for segment in owned_plan.segments
        }
        rebuilt = action.build_plan(
            request,
            context,
            success=owned_plan.plan_success,
            trajectory=trajectory,
            expected_effects=owned_plan.expected_effects,
            effect_candidates=owned_plan.effect_candidates,
            effect_verification=owned_plan.effect_verification,
            replannable=owned_plan.replannable,
            diagnostics=owned_plan.diagnostics,
            segment_lengths=segment_lengths,
            scene_dependency_monitor_until=owned_plan.scene_dependency_monitor_until,
            scene_dependency_end_segment=owned_plan.scene_dependency_end_segment,
        )
        self._validate_plan(rebuilt, context, request)
        return rebuilt

    def rebuild_plan_from_mapped_trajectory(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        plan: ActionPlan,
        reference: TrajectoryTemplate,
        result: MappedRetimeResult,
        *,
        joint_limits: torch.Tensor,
        velocity_limits: torch.Tensor,
        acceleration_limits: torch.Tensor,
    ) -> ActionPlan:
        """Rebuild a B=1 plan after explicit, protected control-grid retiming.

        The mapping must retain every protected phase sample and scene
        dependency boundary. Skills' declared free phases remain authoritative;
        a reference can narrow but never broaden those permissions. Commands,
        tracking and effect information are rebuilt together. This does not
        certify collision freedom or physical task success.

        Args:
            request: Resolved request used for the original plan.
            context: Original planning context.
            plan: Original validated action plan.
            reference: Its exact full-joint expansion template.
            result: Retimed trajectory with source brackets and hard events.
            joint_limits: Full-layout joint limits with shape ``(D, 2)``.
            velocity_limits: Positive full-layout joint speed limits ``(D,)``.
            acceleration_limits: Positive full-layout acceleration limits ``(D,)``.

        Returns:
            A validated complete plan on the new control grid.

        Raises:
            ValueError: Mapping, phase permissions or protected samples differ.
        """
        from embodichain.lab.sim.motion.expansion import (
            MappedRetimeResult,
            TrajectoryTemplate,
            validate_motion_limits,
        )
        from ._expansion_phases import FREE_PHASES

        if not isinstance(reference, TrajectoryTemplate) or not isinstance(
            result, MappedRetimeResult
        ):
            raise TypeError("reference and result must be mapped trajectory values")
        self._validate_context(context)
        owned = plan.snapshot()
        self._validate_plan(owned, context, request)
        original = owned.joint_trajectory
        if context.batch_size != 1 or original is None:
            raise ValueError("mapped plan rebuilding requires a B=1 joint trajectory")
        if tuple(self.robot.joint_names) != reference.joint_names:
            raise ValueError("reference joint names do not match the robot")
        if not torch.equal(
            reference.positions.to(original.positions), original.positions[0]
        ) or not torch.equal(reference.dt.to(original.dt), original.dt[0]):
            raise ValueError("reference must exactly match the original plan grid")
        segments = tuple(owned.segments)
        if tuple(
            (phase.phase_id, phase.start_index, phase.stop_index)
            for phase in reference.phases
        ) != tuple((segment.name, segment.start, segment.stop) for segment in segments):
            raise ValueError("reference phases must exactly match the original plan")
        allowed = FREE_PHASES.get(request.skill_id)
        selected = {
            phase.phase_id
            for phase in reference.phases
            if "retime" in phase.allowed_operators
        }
        if allowed is None or not selected or not selected.issubset(allowed):
            raise ValueError(
                "retiming phases exceed the skill's free-phase permissions"
            )
        source_permissions = owned.diagnostics.metadata.get(
            "prototype_phase_permissions"
        )
        if source_permissions is not None and (
            not isinstance(source_permissions, Mapping)
            or any(
                "retime" not in source_permissions.get(name, ()) for name in selected
            )
        ):
            raise ValueError("retiming phases exceed the source prototype permissions")
        motion_ids = {
            index
            for endpoint in request.binding.endpoints
            if endpoint.endpoint_id == "motion"
            for index in endpoint.joint_ids
        }
        if not set(reference.controlled_joint_indices).issubset(motion_ids):
            raise ValueError(
                "retiming cannot control joints outside the motion binding"
            )
        candidate = result.template
        mapping = result.time_map
        if (
            candidate.joint_names != reference.joint_names
            or candidate.source_id != reference.source_id
            or candidate.source_revision != reference.source_revision
            or candidate.template_id != reference.template_id
            or candidate.controlled_joint_indices != reference.controlled_joint_indices
            or candidate.allowed_operators != reference.allowed_operators
            or candidate.positions.shape[0] != len(mapping.left)
        ):
            raise ValueError("mapped trajectory does not match the original layout")
        count = original.waypoint_count
        if (
            max(mapping.right, default=-1) >= count
            or max(mapping.source_event_indices, default=-1) >= count
            or max(mapping.execution_event_indices, default=-1)
            >= candidate.positions.shape[0]
        ):
            raise ValueError("retiming map addresses samples outside its trajectories")
        device, dtype = original.positions.device, original.positions.dtype
        candidate_positions = candidate.positions.to(device=device, dtype=dtype)
        candidate_dt = candidate.dt.to(original.dt)
        if not torch.allclose(
            candidate_dt[1:],
            torch.full_like(candidate_dt[1:], context.require_control_dt()),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("retimed trajectory must use the environment control grid")
        limits = (
            joint_limits.to(device=device, dtype=dtype)
            if isinstance(joint_limits, torch.Tensor)
            else None
        )
        if (
            limits is None
            or limits.shape != (original.robot_dof, 2)
            or not bool(torch.isfinite(limits).all())
            or not bool((limits[:, 0] <= limits[:, 1]).all())
        ):
            raise ValueError(
                "joint_limits must contain finite ordered bounds for every joint"
            )
        controlled = list(reference.controlled_joint_indices)
        if not controlled or bool(
            (
                (candidate_positions[:, controlled] < limits[controlled, 0])
                | (candidate_positions[:, controlled] > limits[controlled, 1])
            ).any()
        ):
            raise ValueError("retimed trajectory violates controlled joint limits")
        if not validate_motion_limits(
            candidate,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
        ).accepted:
            raise ValueError("retimed trajectory violates sampled motion limits")
        left = torch.as_tensor(mapping.left.copy(), device=device, dtype=torch.long)
        right = torch.as_tensor(mapping.right.copy(), device=device, dtype=torch.long)
        weight = torch.as_tensor(mapping.weight.copy(), device=device, dtype=dtype)
        expected = torch.lerp(
            original.positions[0, left], original.positions[0, right], weight[:, None]
        )
        if not torch.allclose(candidate_positions, expected, atol=1e-7, rtol=0):
            raise ValueError("mapped positions do not match their source brackets")
        event_map = dict(
            zip(
                mapping.source_event_indices.tolist(),
                mapping.execution_event_indices.tolist(),
            )
        )
        editable = set()
        for phase in reference.phases:
            if phase.phase_id in selected:
                editable.update(range(phase.start_index + 1, phase.stop_index - 1))
        hard = set(range(count)) - editable
        for phase in reference.phases:
            hard.update((phase.start_index, phase.stop_index - 1))
        hard.update(
            index
            for index in owned.scene_dependency_monitor_until.values()
            if index < count
        )
        protected = {
            near
            for index in hard
            for near in (index - 1, index, index + 1)
            if 0 <= near < count
        }
        if not protected.issubset(event_map):
            raise ValueError("retiming map does not preserve all protected events")
        for index in protected:
            destination = event_map[index]
            if (
                mapping.left[destination] != index
                or mapping.right[destination] != index
                or mapping.weight[destination] != 0
            ):
                raise ValueError("protected events require exact source-index mappings")
            if not torch.equal(
                candidate_positions[event_map[index]], original.positions[0, index]
            ):
                raise ValueError("retiming modified a protected sample")
        new_lengths = {}
        previous_stop = 0
        for segment in segments:
            start = event_map[segment.start]
            stop = event_map[segment.stop - 1] + 1
            if start != previous_stop or stop <= start:
                raise ValueError("retimed segments must remain contiguous and nonempty")
            new_lengths[segment.name] = stop - start
            previous_stop = stop
        if previous_stop != candidate.positions.shape[0]:
            raise ValueError("retimed segments do not cover the complete trajectory")
        monitor_until = {
            entity: candidate.positions.shape[0] if index == count else event_map[index]
            for entity, index in owned.scene_dependency_monitor_until.items()
        }
        trajectory = TimedTrajectory.from_positions(
            candidate_positions.unsqueeze(0),
            env_ids=context.env_ids,
            dt=candidate_dt.unsqueeze(0),
        )
        if (
            float(trajectory.dt.sum()) + float(trajectory.dt[0, -1])
            > owned.recovery_policy.action_timeout
        ):
            raise ValueError("retimed trajectory exceeds the action timeout")
        action = self._actions[request.skill_id]
        rebuilt = action.build_plan(
            request,
            context,
            success=owned.plan_success,
            trajectory=trajectory,
            expected_effects=owned.expected_effects,
            effect_candidates=owned.effect_candidates,
            effect_verification=owned.effect_verification,
            replannable=owned.replannable,
            diagnostics=owned.diagnostics,
            segment_lengths=new_lengths,
            scene_dependency_monitor_until=monitor_until,
            scene_dependency_end_segment=owned.scene_dependency_end_segment,
        )
        self._validate_plan(rebuilt, context, request)
        if (
            original.velocities is None
            or rebuilt.joint_trajectory is None
            or rebuilt.joint_trajectory.velocities is None
        ):
            raise ValueError(
                "retiming requires original and rebuilt velocity references"
            )
        old_ids = torch.tensor(sorted(hard), dtype=torch.long, device=device)
        new_ids = torch.tensor(
            [event_map[index] for index in sorted(hard)],
            dtype=torch.long,
            device=device,
        )
        if not torch.allclose(
            original.velocities[0, old_ids],
            rebuilt.joint_trajectory.velocities[0, new_ids],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("retiming changed a protected derivative boundary")
        return rebuilt


__all__ = ["AtomicActionEngine"]
