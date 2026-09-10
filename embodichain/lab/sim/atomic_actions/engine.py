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

from .bindings import ActionBinding
from .candidates import AtomicCandidateBatch, AtomicCandidateSelection, _active_rows
from .core import AtomicAction, SkillDescriptor
from .control import ActionControlOverrides, ControlPartCommandProfile
from .invocation import ActionInvocation, GoalT, OptionsT, ResolvedActionRequest
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
    from embodichain.toolkits.graspkit.candidates import GraspCandidateBatch

    from .execution import ExecutionSession


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
        *,
        plan_provider: (
            Callable[[ResolvedActionRequest, PlanningContext], ActionPlan] | None
        ) = None,
    ) -> ActionPlan:
        """Plan an already-resolved request without rebuilding its snapshot.

        This is the planning entry point used by execution recovery. Callers
        normally use :meth:`plan`, while an execution session resolves once and
        calls this method for every replan.

        Args:
            request: Immutable request previously returned by :meth:`_resolve`.
            context: Optional latest planning state; captured when omitted.
            plan_provider: Initial-plan materializer supplied by a new session.
                Recovery leaves this unset and invokes the registered planner.

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
        plan = (
            action.plan(request, current)
            if plan_provider is None
            else action._plan_with_provider(request, current, plan_provider)
        )
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

    def enumerate_candidates(
        self,
        invocation: ActionInvocation,
        context: PlanningContext | None = None,
        *,
        grasp_candidates: GraspCandidateBatch | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> AtomicCandidateBatch:
        """Evaluate all skill candidates at the current invocation start.

        Args:
            invocation: Grounded skill request.
            context: Current observed or projected input for this invocation.
            grasp_candidates: Geometric candidates bound to the real robot rows.
            active_mask: Optional eligible physical rows.

        Returns:
            Evaluated choices; the candidate dimension does not change robot batch.
        """
        current = self.initial_context() if context is None else context
        self._validate_context(current)
        request = self._resolve(invocation)
        return self._actions[request.skill_id].enumerate_candidates(
            request, current, grasp_candidates=grasp_candidates, active_mask=active_mask
        )

    def plan_candidate(
        self,
        invocation: ActionInvocation,
        selection: AtomicCandidateSelection,
        context: PlanningContext | None = None,
        *,
        active_mask: torch.Tensor | None = None,
    ) -> ActionPlan:
        """Plan one selected candidate per physical row without reselecting.

        Args:
            invocation: Grounded skill request used during evaluation.
            selection: Selected evaluated candidates.
            context: Exact current invocation input used during evaluation.
            active_mask: Optional eligible physical rows.

        Returns:
            Ordinary authorized and validated atomic action plan.
        """
        current = self.initial_context() if context is None else context
        self._validate_context(current)
        request = self._resolve(invocation)
        plan = self._actions[request.skill_id].plan_candidate(
            request, selection, current, active_mask=active_mask
        )
        self._validate_plan(plan, current, request)
        return plan

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
            scene: Optional explicit scene snapshot. It overrides the engine's
                configured scene provider; an empty snapshot is used when both
                are absent.
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
        )

    def compile(
        self,
        invocations: Iterable[ActionInvocation],
        context: PlanningContext | None = None,
        *,
        candidate_selections: (
            Mapping[
                str,
                AtomicCandidateSelection
                | Callable[
                    [ResolvedActionRequest, PlanningContext, torch.Tensor],
                    AtomicCandidateSelection,
                ],
            ]
            | None
        ) = None,
        eligible_mask: torch.Tensor | None = None,
    ) -> CompiledTrajectory:
        """Compile a static sequence of grounded invocations.

        Planning is side-effect free. Expected effects are applied only to the
        returned hypothetical ``projected_context`` so following actions can be
        checked against the expected state. No simulator or observed task state
        is mutated.

        Args:
            invocations: Grounded action requests in execution order.
            context: Optional initial planning context captured by the caller.
            candidate_selections: Optional choices keyed by invocation ID. A
                callback evaluates choices at that invocation's projected start.
            eligible_mask: Initial physical-row eligibility; idle rows never
                receive projected effects or count as compilation successes.

        Returns:
            Concatenated timed trajectory, individual plans, and projected state.

        Raises:
            KeyError: If an invocation references an unregistered skill.
            ValueError: If context, plan, or trajectory dimensions are incompatible.
        """
        if context is None:
            context = self.initial_context()
        self._validate_context(context)

        alive = _active_rows(eligible_mask, context.env_ids)
        invocation_values = tuple(invocations)
        selections = {} if candidate_selections is None else dict(candidate_selections)
        invocation_ids = tuple(item.invocation_id for item in invocation_values)
        if any(key not in invocation_ids for key in selections):
            raise ValueError("candidate_selections contains an unknown invocation ID.")
        if selections and len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("Candidate compilation requires unique invocation IDs.")
        plans: list[ActionPlan] = []
        trajectories: list[TimedTrajectory] = []
        projected = context

        for invocation in invocation_values:
            if not alive.any():
                break
            previous_qpos = projected.robot.qpos
            choice = selections.get(invocation.invocation_id)
            if choice is None:
                plan = self.plan(invocation, projected)
            else:
                if callable(choice):
                    choice = choice(self._resolve(invocation), projected, alive.clone())
                plan = self.plan_candidate(
                    invocation, choice, projected, active_mask=alive
                )
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
        *,
        eligible_mask: torch.Tensor | None = None,
        initial_plan_provider: (
            Callable[[ResolvedActionRequest, PlanningContext], ActionPlan] | None
        ) = None,
    ) -> ExecutionSession:
        """Start incremental execution for a grounded invocation sequence.

        Args:
            invocations: Grounded action requests in execution order.
            context: Initial measured state and scene snapshot. The engine
                captures one when omitted.
            eligible_mask: Optional per-environment cohort allowed to execute.
                Ineligible rows remain excluded for the whole session. All rows
                are eligible when omitted.
            initial_plan_provider: Optional materializer for the first invocation's
                initial plan. It receives a newly resolved request with current
                collision options and the initial measured context. Its result
                passes ordinary plan, endpoint, tracking, and phase-gate checks.
                Later invocations and recovery use the registered skill planner.
                Rebuild candidate bindings and effects from these inputs; do not
                reuse a plan or execution session across environment resets.

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
            initial_plan_provider=initial_plan_provider,
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


__all__ = ["AtomicActionEngine"]
