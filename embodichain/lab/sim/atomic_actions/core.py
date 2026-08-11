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
from .bindings import EndpointBinding, JointPositionTarget
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
    EffectVerificationRequirement,
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
    normalize_success_mask,
)
from .policies import DynamicCollisionMode
from .requirements import SkillBindingContract
from .runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    TimedCommandSequence,
)
from .tracking import (
    FeedbackTerminalAcceptance,
    TimedTrackingSequence,
    TrackingFrame,
    TrackingSetpoint,
)

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


@dataclass(frozen=True, slots=True, eq=False)
class ObjectSemantics:
    """Shallow-frozen semantic information about an interaction object.

    .. attention::
        Top-level fields cannot be rebound after construction. Nested
        affordance and metadata objects may remain mutable but never establish
        object identity.
    """

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

    entity_id: str | None = None
    """Stable scene identifier used by snapshot grounding and explicit identity."""

    def __post_init__(self) -> None:
        if not isinstance(self.affordance, Affordance):
            raise TypeError("affordance must be an Affordance instance.")
        if not isinstance(self.geometry, dict):
            raise TypeError("geometry must be a dict.")
        if not isinstance(self.properties, dict):
            raise TypeError("properties must be a dict.")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string.")
        if self.entity_id is not None and (
            not isinstance(self.entity_id, str) or not self.entity_id.strip()
        ):
            raise ValueError("entity_id must be a non-empty string when set.")
        self.affordance.object_label = self.label


def _legacy_object_uid(semantics: ObjectSemantics) -> str | None:
    """Return a valid legacy simulation UID without alias normalization."""
    uid = getattr(semantics.entity, "uid", None)
    return uid if isinstance(uid, str) and uid.strip() else None


def _same_object_identity(
    left: ObjectSemantics,
    right: ObjectSemantics,
) -> bool:
    """Return whether two semantic snapshots identify the same object."""
    if left is right:
        return True
    if left.entity_id is not None or right.entity_id is not None:
        return (
            left.entity_id is not None
            and right.entity_id is not None
            and left.entity_id == right.entity_id
        )
    left_uid = _legacy_object_uid(left)
    right_uid = _legacy_object_uid(right)
    if left_uid is not None or right_uid is not None:
        return left_uid is not None and right_uid is not None and left_uid == right_uid
    return left.entity is not None and left.entity is right.entity


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Machine-readable metadata for one registered atomic skill."""

    skill_id: str
    goal_type: type[Any] | tuple[type[Any], ...]
    options_type: type[ActionOptions]
    agent_visible: bool = True
    binding_contract: SkillBindingContract | None = None
    """Explicit generic resource contract used by the semantic skill layer."""

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
        if self.binding_contract is not None:
            if not isinstance(self.binding_contract, SkillBindingContract):
                raise TypeError(
                    "SkillDescriptor.binding_contract must be a "
                    "SkillBindingContract or None."
                )


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

    agent_visible: ClassVar[bool] = True
    """Whether an Action Agent should expose this skill by default."""

    binding_contract: ClassVar[SkillBindingContract | None] = None
    """Explicit robot-independent requirements for semantic discovery.

    Concrete action classes must declare this attribute in their own class
    body to opt into the semantic catalog. Inheriting another action's contract
    does not silently expose a new skill identifier.
    """

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
            agent_visible=cls.agent_visible,
            binding_contract=cls.__dict__.get("binding_contract"),
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
        contract = type(self).__dict__.get("binding_contract")
        if contract is None:
            raise ValueError(
                f"Skill {self.skill_id!r} has no explicit SkillBindingContract."
            )
        self.planning_services.validate_binding(invocation.binding, contract)
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
            binding=self.planning_services.apply_command_overrides(
                invocation.binding,
                invocation.control_overrides,
            ),
            motion_policy=invocation.motion_policy,
            tracking_policy=invocation.tracking_policy,
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
        contract = type(self).__dict__.get("binding_contract")
        if contract is None:
            raise ValueError(
                f"Skill {self.skill_id!r} has no explicit SkillBindingContract."
            )
        self.planning_services.validate_binding(request.binding, contract)
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
        plan = self._plan(prepared, context)
        if not isinstance(plan, ActionPlan):
            raise TypeError("AtomicAction._plan() must return an ActionPlan.")
        return replace(
            plan,
            commands=self._authorize_command_targets(prepared, plan.commands),
        )

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

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
    ) -> tuple[str, ...]:
        """Return scene entities whose poses materially affect this plan."""
        return collect_scene_dependencies(request.goal)

    def build_plan(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
        *,
        success: bool | torch.Tensor,
        trajectory: TimedTrajectory | torch.Tensor,
        expected_effects: StateDelta | None = None,
        effect_verification: EffectVerificationRequirement | None = None,
        replannable: bool = True,
        diagnostics: PlannerDiagnostics | None = None,
        segment_lengths: Mapping[str, int] | None = None,
        scene_dependency_monitor_until: Mapping[str, int] | None = None,
    ) -> ActionPlan:
        """Build a validated action plan for a primitive implementation.

        Args:
            request: Resolved invocation snapshot being planned.
            context: Planning input used for the plan.
            success: Per-environment planning success or scalar planner result.
            trajectory: Full-robot timed trajectory or position tensor.
            expected_effects: Symbolic effects to verify after execution.
            effect_verification: Optional explicit physical-effect boundary.
                Use this when verification is required without a symbolic task-
                state delta.
            replannable: Whether the execution runtime may replan this action.
            diagnostics: Optional retained planner diagnostics.
            segment_lengths: Optional ordered mapping from semantic segment
                names to waypoint counts. Zero-length entries are omitted.
            scene_dependency_monitor_until: Optional per-entity exclusive
                waypoint-index upper bound for scene-motion invalidation. An
                entity is monitored while the current waypoint index is smaller
                than its bound. ``0`` disables monitoring immediately; omitted
                dependencies remain monitored for the full action. Once the bound
                is reached, all pose changes for that entity are ignored.

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

        commands = self._joint_command_sequence(
            request,
            timed,
            active_mask=success_mask,
        )
        return self.build_command_plan(
            request,
            context,
            success=success_mask,
            commands=commands,
            expected_effects=expected_effects,
            effect_verification=effect_verification,
            replannable=replannable,
            diagnostics=diagnostics,
            segment_lengths=segment_lengths,
            scene_dependency_monitor_until=scene_dependency_monitor_until,
            joint_trajectory=timed,
        )

    def build_command_plan(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        context: PlanningContext,
        *,
        success: bool | torch.Tensor,
        commands: TimedCommandSequence,
        expected_effects: StateDelta | None = None,
        effect_verification: EffectVerificationRequirement | None = None,
        replannable: bool = True,
        diagnostics: PlannerDiagnostics | None = None,
        segment_lengths: Mapping[str, int] | None = None,
        scene_dependency_monitor_until: Mapping[str, int] | None = None,
        joint_trajectory: TimedTrajectory | None = None,
    ) -> ActionPlan:
        """Build a plan from transport-neutral runtime command frames.

        Tracking targets are projected from the command payloads through the
        typed channels declared by each bound endpoint. Semantic effects remain
        externally verified through the execution session.

        Args:
            request: Resolved invocation snapshot being planned.
            context: Planning input used for the plan.
            success: Per-environment planning success or scalar planner result.
            commands: Transport-neutral command sequence for the action.
            expected_effects: Symbolic effects to verify after execution.
            effect_verification: Optional explicit physical-effect boundary.
            replannable: Whether the execution runtime may replan this action.
            diagnostics: Optional retained planner diagnostics.
            segment_lengths: Optional ordered mapping from semantic segment names
                to command-frame counts. Zero-length entries are omitted.
            scene_dependency_monitor_until: Optional per-entity exclusive
                command-frame-index upper bound for scene-motion invalidation. An
                entity is monitored while the current frame index is smaller than
                its bound. ``0`` disables monitoring immediately; omitted
                dependencies remain monitored for the full action. Once the bound
                is reached, all pose changes for that entity are ignored.
            joint_trajectory: Optional joint trajectory retained for offline
                compilation and inspection.

        Returns:
            Side-effect-free action plan.
        """
        self.require_goal(request)
        if not isinstance(commands, TimedCommandSequence):
            raise TypeError("commands must be a TimedCommandSequence.")
        if commands.batch_size != context.batch_size:
            raise ValueError(
                "Command sequence and planning context batch sizes must match."
            )
        if not torch.equal(commands.env_ids, context.env_ids):
            raise ValueError("Command sequence env_ids must match the context.")
        commands = self._authorize_command_targets(request, commands)
        success_mask = normalize_success_mask(
            success,
            n_envs=context.batch_size,
            device=self.device,
            name="Planning success",
        )
        masked_commands = TimedCommandSequence(
            frames=tuple(
                frame.with_active_mask(frame.active_mask & success_mask)
                for frame in commands.frames
            ),
            env_ids=commands.env_ids,
        )
        tracking = self._tracking_sequence(request, masked_commands)
        segments = self._build_segments(
            segment_lengths,
            frame_count=masked_commands.frame_count,
        )
        if diagnostics is None:
            diagnostics = PlannerDiagnostics(
                backend=self.planning_services.planner_name
            )
        return ActionPlan(
            skill_id=self.skill_id,
            plan_success=success_mask,
            commands=masked_commands,
            recovery_policy=request.recovery_policy,
            tracking_policy=request.tracking_policy,
            planned_scene_version=context.scene.version,
            planned_collision_world_revision=(
                context.scene.collision_world_revisions(context.batch_size)
            ),
            diagnostics=diagnostics,
            tracking=tracking,
            joint_trajectory=joint_trajectory,
            segments=segments,
            scene_dependencies=self._scene_dependencies(request),
            scene_dependency_monitor_until=(
                {}
                if scene_dependency_monitor_until is None
                else scene_dependency_monitor_until
            ),
            collision_world_sensitive=self._uses_collision_world(request, context),
            replannable=replannable,
            expected_effects=expected_effects or StateDelta(),
            effect_verification=effect_verification,
            invocation_id=request.invocation_id,
            invocation_revision=request.revision,
        )

    def _tracking_sequence(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        commands: TimedCommandSequence,
    ) -> TimedTrackingSequence | None:
        """Project command payloads through binding-owned tracking channels."""
        policy = request.tracking_policy
        metrics = list(() if policy.in_flight is None else policy.in_flight.metrics)
        if isinstance(policy.terminal, FeedbackTerminalAcceptance):
            metrics.extend(policy.terminal.metrics)
        if not metrics:
            return None

        runtime = self.planning_services.tracking_runtime
        for metric in metrics:
            runtime.evaluators.resolve(metric)
        metrics_by_channel = {metric.channel_id: metric for metric in metrics}

        endpoints_by_destination: dict[
            tuple[str, str],
            tuple[EndpointBinding, ...],
        ] = {}
        for endpoint in request.binding.endpoints:
            endpoints_by_destination.setdefault(endpoint.destination_key, ())
            endpoints_by_destination[endpoint.destination_key] += (endpoint,)

        tracking_frames: list[TrackingFrame] = []
        for frame_index, frame in enumerate(commands.frames):
            setpoints: list[TrackingSetpoint] = []
            for command in frame.commands:
                endpoints = endpoints_by_destination[command.destination_key]
                for endpoint in endpoints:
                    for channel_id in metrics_by_channel:
                        channel = endpoint.tracking_channels.get(channel_id)
                        if channel is None:
                            continue
                        runtime.providers.resolve(channel.source)
                        runtime.projectors.resolve(channel.projector)
                        setpoints.append(
                            TrackingSetpoint(
                                endpoint_key=endpoint.key,
                                binding=channel,
                                desired=runtime.project(command, channel),
                            )
                        )
            covered_channels = {setpoint.binding.channel_id for setpoint in setpoints}
            missing_channels = sorted(
                set(metrics_by_channel).difference(covered_channels)
            )
            if missing_channels:
                raise ValueError(
                    f"Command frame {frame_index} cannot project configured "
                    f"tracking channels {missing_channels}; bound endpoints must "
                    "declare a typed feedback source and projector."
                )
            tracking_frames.append(TrackingFrame(tuple(setpoints)))
        return TimedTrackingSequence(
            env_ids=commands.env_ids,
            frames=tuple(tracking_frames),
        )

    @staticmethod
    def _authorize_command_targets(
        request: ResolvedActionRequest[GoalT, OptionsT],
        commands: TimedCommandSequence,
    ) -> TimedCommandSequence:
        """Bind every emitted command to an endpoint authorized by the request.

        Actions may choose a subset of their bound endpoints for any frame, but
        they cannot synthesize a destination outside the resolved resource
        binding. The returned sequence replaces caller-provided target metadata
        with the engine-owned binding snapshot, so transports never receive
        altered joint claims or other target fields.
        """
        authorized: dict[tuple[str, str], list[EndpointBinding]] = {}
        for endpoint in request.binding.endpoints:
            authorized.setdefault(endpoint.destination_key, []).append(endpoint)
        unknown = sorted(
            {
                command.destination_key
                for frame in commands.frames
                for command in frame.commands
                if command.destination_key not in authorized
            }
        )
        if unknown:
            raise ValueError(
                "Runtime commands reference destinations not authorized by the "
                f"action binding: {unknown}."
            )

        frames: list[RuntimeCommandFrame] = []
        for frame in commands.frames:
            endpoint_commands: list[EndpointCommand] = []
            joint_owners: dict[int, tuple[str, str]] = {}
            token_owners: dict[str, tuple[str, str]] = {}
            for command in frame.commands:
                bound_endpoints = authorized[command.destination_key]
                target = bound_endpoints[0].target
                if any(
                    type(endpoint.target) is not type(target)
                    for endpoint in bound_endpoints[1:]
                ):
                    raise ValueError(
                        f"Action binding destination {command.destination_key} has "
                        "incompatible target declarations."
                    )
                if type(command.target) is not type(target):
                    raise TypeError(
                        f"Runtime command destination {command.destination_key} uses "
                        f"target type {type(command.target).__name__}, but its bound "
                        f"endpoint uses {type(target).__name__}."
                    )
                if isinstance(target, JointPositionTarget) and command.target != target:
                    raise ValueError(
                        f"Runtime command destination {command.destination_key} "
                        "does not preserve its bound joint-position target."
                    )
                joint_ids = {
                    joint_id
                    for endpoint in bound_endpoints
                    for joint_id in endpoint.joint_ids
                }
                claim_tokens = {
                    token
                    for endpoint in bound_endpoints
                    for token in endpoint.claim_tokens
                }
                overlapping_joints = sorted(joint_ids & joint_owners.keys())
                overlapping_tokens = sorted(claim_tokens & token_owners.keys())
                if overlapping_joints or overlapping_tokens:
                    conflicting_destinations = sorted(
                        {joint_owners[joint_id] for joint_id in overlapping_joints}
                        | {token_owners[token] for token in overlapping_tokens}
                    )
                    raise ValueError(
                        f"Runtime command destination {command.destination_key} "
                        f"conflicts with {conflicting_destinations} on bound joint "
                        f"IDs {overlapping_joints} or claim tokens "
                        f"{overlapping_tokens}."
                    )
                for joint_id in joint_ids:
                    joint_owners[joint_id] = command.destination_key
                for token in claim_tokens:
                    token_owners[token] = command.destination_key
                endpoint_commands.append(
                    EndpointCommand(target=target, payload=command.payload)
                )
            frames.append(
                RuntimeCommandFrame(
                    commands=tuple(endpoint_commands),
                    active_mask=frame.active_mask,
                    env_ids=frame.env_ids,
                    hold_duration=frame.hold_duration,
                )
            )
        return TimedCommandSequence(frames=tuple(frames), env_ids=commands.env_ids)

    def _joint_command_sequence(
        self,
        request: ResolvedActionRequest[GoalT, OptionsT],
        trajectory: TimedTrajectory,
        *,
        active_mask: torch.Tensor,
    ) -> TimedCommandSequence:
        """Lower one full-robot planner trajectory to endpoint commands."""
        targets = tuple(
            (
                endpoint,
                endpoint.require_target(JointPositionTarget),
            )
            for endpoint in request.binding.endpoints
        )
        if not targets:
            raise ValueError(
                "Joint trajectory plans require at least one bound "
                "JointPositionTarget endpoint."
            )
        frames: list[RuntimeCommandFrame] = []
        for waypoint_index in range(trajectory.waypoint_count):
            endpoint_commands: list[EndpointCommand] = []
            for _, target in targets:
                joint_ids = list(target.joint_ids)
                velocities = (
                    None
                    if trajectory.velocities is None
                    else trajectory.velocities[:, waypoint_index, joint_ids]
                )
                endpoint_commands.append(
                    EndpointCommand(
                        target=target,
                        payload=JointPositionPayload(
                            positions=trajectory.positions[
                                :, waypoint_index, joint_ids
                            ],
                            velocities=velocities,
                        ),
                    )
                )
            next_waypoint_index = min(
                waypoint_index + 1,
                trajectory.waypoint_count - 1,
            )
            # ``dt[:, i]`` is the arrival interval for waypoint ``i``. After
            # dispatching it, wait for the next arrival interval; the terminal
            # frame deliberately reuses its own interval as a settling window,
            # preserving the closed-loop runner's pre-PR2C timing contract.
            frames.append(
                RuntimeCommandFrame(
                    commands=tuple(endpoint_commands),
                    active_mask=active_mask,
                    env_ids=trajectory.env_ids,
                    hold_duration=trajectory.dt[:, next_waypoint_index],
                )
            )
        return TimedCommandSequence(frames=tuple(frames), env_ids=trajectory.env_ids)

    @staticmethod
    def _build_segments(
        segment_lengths: Mapping[str, int] | None,
        *,
        frame_count: int,
    ) -> tuple[TrajectorySegment, ...]:
        """Validate optional named ranges for one command sequence."""
        if segment_lengths is None:
            return ()
        segments: list[TrajectorySegment] = []
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
                TrajectorySegment(name=name, start=offset, stop=offset + length)
            )
            offset += length
        if offset != frame_count:
            raise ValueError(
                "Trajectory segment lengths must sum to the command frame count "
                f"({frame_count}), got {offset}."
            )
        return tuple(segments)

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
        success = torch.zeros(context.batch_size, dtype=torch.bool, device=self.device)
        diagnostics = PlannerDiagnostics(
            backend=self.planning_services.planner_name,
            messages=(() if message is None else (message,)),
        )
        if request.binding.endpoints and all(
            isinstance(endpoint.target, JointPositionTarget)
            for endpoint in request.binding.endpoints
        ):
            return self.build_plan(
                request,
                context,
                success=success,
                trajectory=TimedTrajectory.empty(
                    batch_size=context.batch_size,
                    robot_dof=context.robot.robot_dof,
                    device=self.device,
                    env_ids=context.env_ids,
                ),
                replannable=True,
                diagnostics=diagnostics,
            )
        return self.build_command_plan(
            request,
            context,
            success=success,
            commands=TimedCommandSequence(frames=(), env_ids=context.env_ids),
            replannable=True,
            diagnostics=diagnostics,
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
