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

"""Canonical execution service and convenience facade for semantic skills."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from ..atomic_actions.bindings import EndpointBinding
from ..atomic_actions.engine import AtomicActionEngine
from ..atomic_actions.execution import (
    EffectVerificationRequest,
    EffectVerificationResult,
    ExecutionEvent,
    ExecutionPlanAttempt,
)
from ..atomic_actions.plans import ExecutionFeedbackMode, TrajectorySegment
from ..atomic_actions.policies import MotionPolicy, RecoveryPolicy
from ..atomic_actions.runner import (
    CommandSink,
    ExecutionClock,
    ExecutionRunner,
    ExecutionRunnerCfg,
    MonotonicExecutionClock,
    ObservationProvider,
    RunnerStatus,
    RunnerStep,
)
from ..atomic_actions.state import PlanningContext, TaskState
from .calls import SemanticCallSpec
from .compiler import SemanticSkillCompiler
from .effects import (
    BinaryEffectEvidenceBatch,
    EffectEvidenceBatch,
    EffectMonitor,
    EffectMonitorRef,
    JointStateEvidenceBatch,
    PoseRelationEvidenceBatch,
    ScalarEffectEvidenceBatch,
    SemanticEffectSpec,
)
from .scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRef,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)


def _snapshot_task_state(state: TaskState) -> TaskState:
    """Return a tensor-owning snapshot of verified symbolic state."""
    return TaskState(
        batch_size=state.batch_size,
        device=state.device,
        held_objects=state.held_objects,
        coordinated_held_objects=state.coordinated_held_objects,
        articulation_joints=state.articulation_joints,
    )


def _snapshot_event(event: ExecutionEvent) -> ExecutionEvent:
    """Return an independently owned execution event."""
    return ExecutionEvent(
        kind=event.kind,
        timestamp=event.timestamp,
        skill_id=event.skill_id,
        invocation_id=event.invocation_id,
        invocation_revision=event.invocation_revision,
        invocation_index=event.invocation_index,
        env_mask=event.env_mask,
        message=event.message,
    )


def _metadata_value(value: object, *, depth: int = 0) -> object:
    """Convert supported runtime diagnostics to deterministic JSON-safe data."""
    if depth > 16:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        return _metadata_value(value.detach().cpu().tolist(), depth=depth + 1)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        if all(type(key) is str and key and key == key.strip() for key, _ in items):
            return {
                key: _metadata_value(nested, depth=depth + 1) for key, nested in items
            }
        return {
            "__entries__": [
                {
                    "key": _metadata_value(key, depth=depth + 1),
                    "value": _metadata_value(nested, depth=depth + 1),
                }
                for key, nested in items
            ]
        }
    if isinstance(value, (tuple, list)):
        return [_metadata_value(nested, depth=depth + 1) for nested in value]
    if isinstance(value, (set, frozenset)):
        return [
            _metadata_value(nested, depth=depth + 1)
            for nested in sorted(value, key=str)
        ]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _snapshot_metadata_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Own one JSON-safe string-keyed metadata mapping."""
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping.")
    normalized = _metadata_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("metadata normalization must produce a dict.")
    return MappingProxyType(normalized)


def _event_to_metadata(event: ExecutionEvent) -> dict[str, object]:
    """Serialize one execution/recovery event without exposing tensors."""
    return {
        "kind": event.kind.value,
        "timestamp": _metadata_value(event.timestamp),
        "skill_id": event.skill_id,
        "invocation_id": event.invocation_id,
        "invocation_revision": event.invocation_revision,
        "invocation_index": event.invocation_index,
        "env_mask": _metadata_value(event.env_mask),
        "message": event.message,
    }


def task_state_to_metadata(state: TaskState) -> dict[str, object]:
    """Return verified symbolic task state as deterministic JSON-safe data."""
    if not isinstance(state, TaskState):
        raise TypeError("state must be a TaskState.")
    held = []
    for resource_id, value in sorted(state.held_objects.items()):
        held.append(
            {
                "resource_id": resource_id,
                "object_id": value.semantics.entity_id,
                "object_label": value.semantics.label,
                "object_to_eef": _metadata_value(value.object_to_eef),
                "grasp_xpos": _metadata_value(value.grasp_xpos),
                "active_mask": _metadata_value(value.env_mask),
            }
        )
    coordinated = []
    for resource_ids, value in sorted(state.coordinated_held_objects.items()):
        coordinated.append(
            {
                "resource_ids": list(resource_ids),
                "object_id": value.semantics.entity_id,
                "object_label": value.semantics.label,
                "left_object_to_eef": _metadata_value(value.left_object_to_eef),
                "right_object_to_eef": _metadata_value(value.right_object_to_eef),
                "left_grasp_xpos": _metadata_value(value.left_grasp_xpos),
                "right_grasp_xpos": _metadata_value(value.right_grasp_xpos),
                "active_mask": _metadata_value(value.env_mask),
            }
        )
    articulations = []
    for (articulation_id, joint_id), value in sorted(state.articulation_joints.items()):
        articulations.append(
            {
                "articulation_id": articulation_id,
                "joint_id": joint_id,
                "position": _metadata_value(value.position),
                "active_mask": _metadata_value(value.env_mask),
            }
        )
    return {
        "batch_size": state.batch_size,
        "device": str(state.device),
        "held_objects": held,
        "coordinated_held_objects": coordinated,
        "articulation_joints": articulations,
    }


class SkillStatus(str, Enum):
    """Lifecycle state of one semantic workflow run."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SkillEndpointBindingTrace:
    """JSON-safe typed projection of one resolved execution endpoint."""

    slot_id: str
    endpoint_id: str
    resource_id: str
    adapter_id: str
    transport_id: str
    target_id: str
    target_type: str
    task_state_key: str
    capabilities: tuple[str, ...]
    command_ids: tuple[str, ...]
    claim_tokens: tuple[str, ...]
    joint_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "slot_id",
            "endpoint_id",
            "resource_id",
            "adapter_id",
            "transport_id",
            "target_id",
            "target_type",
            "task_state_key",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        for name in ("capabilities", "command_ids", "claim_tokens"):
            values = tuple(getattr(self, name))
            if tuple(sorted(set(values))) != values or not all(
                type(value) is str and value for value in values
            ):
                raise ValueError(f"{name} must contain sorted unique identifiers.")
            object.__setattr__(self, name, values)
        joint_ids = tuple(self.joint_ids)
        if len(set(joint_ids)) != len(joint_ids) or not all(
            type(value) is int and value >= 0 for value in joint_ids
        ):
            raise ValueError("joint_ids must contain unique non-negative integers.")
        object.__setattr__(self, "joint_ids", joint_ids)

    @classmethod
    def from_binding(cls, binding: EndpointBinding) -> SkillEndpointBindingTrace:
        """Project one owned endpoint binding without retaining its target."""
        if not isinstance(binding, EndpointBinding):
            raise TypeError("binding must be an EndpointBinding.")
        target = binding.target
        return cls(
            slot_id=binding.slot_id,
            endpoint_id=binding.endpoint_id,
            resource_id=binding.resource_id,
            adapter_id=binding.adapter_id,
            transport_id=target.transport_id,
            target_id=target.target_id,
            target_type=f"{type(target).__module__}.{type(target).__qualname__}",
            task_state_key=binding.task_state_key,
            capabilities=tuple(sorted(binding.capabilities)),
            command_ids=tuple(sorted(binding.commands)),
            claim_tokens=tuple(sorted(binding.claim_tokens)),
            joint_ids=binding.joint_ids,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return stable endpoint, resource, adapter, and transport metadata."""
        return {
            "slot_id": self.slot_id,
            "endpoint_id": self.endpoint_id,
            "resource_id": self.resource_id,
            "adapter_id": self.adapter_id,
            "transport_id": self.transport_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "task_state_key": self.task_state_key,
            "capabilities": list(self.capabilities),
            "command_ids": list(self.command_ids),
            "claim_tokens": list(self.claim_tokens),
            "joint_ids": list(self.joint_ids),
        }


def _motion_policy_to_metadata(policy: MotionPolicy) -> dict[str, object]:
    """Serialize one owned core motion policy without retaining planner objects."""
    plan_options = policy.plan_opts
    options_metadata: object = None
    if plan_options is not None:
        values = (
            plan_options.to_dict()
            if callable(getattr(plan_options, "to_dict", None))
            else None
        )
        options_metadata = {
            "type": f"{type(plan_options).__module__}.{type(plan_options).__qualname__}",
            "values": _metadata_value(values),
        }
    return {
        "strategy": policy.strategy,
        "sample_count": policy.sample_count,
        "dynamic_collision_mode": policy.dynamic_collision_mode.value,
        "plan_options": options_metadata,
    }


def _recovery_policy_to_metadata(policy: RecoveryPolicy) -> dict[str, object]:
    """Serialize all bounded-recovery settings."""
    return {
        "max_replans": policy.max_replans,
        "max_action_retries": policy.max_action_retries,
        "tracking_error_threshold": _metadata_value(policy.tracking_error_threshold),
        "goal_translation_threshold": _metadata_value(
            policy.goal_translation_threshold
        ),
        "goal_rotation_threshold": _metadata_value(policy.goal_rotation_threshold),
        "action_timeout": _metadata_value(policy.action_timeout),
    }


@dataclass(frozen=True, slots=True)
class ResolvedCorePolicyTrace:
    """Resolved preset, core policies, and execution binding for one plan."""

    profile_id: str
    preset_id: str
    preset_schema_version: int
    motion_policy: MotionPolicy
    recovery_policy: RecoveryPolicy
    endpoints: tuple[SkillEndpointBindingTrace, ...]

    def __post_init__(self) -> None:
        for name in ("profile_id", "preset_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if (
            type(self.preset_schema_version) is not int
            or self.preset_schema_version < 1
        ):
            raise ValueError("preset_schema_version must be a positive integer.")
        if not isinstance(self.motion_policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(self.recovery_policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        endpoints = tuple(self.endpoints)
        if not all(type(value) is SkillEndpointBindingTrace for value in endpoints):
            raise TypeError(
                "endpoints must contain exact SkillEndpointBindingTrace values."
            )
        keys = tuple((value.slot_id, value.endpoint_id) for value in endpoints)
        if len(set(keys)) != len(keys):
            raise ValueError("endpoints must use unique slot/endpoint keys.")
        object.__setattr__(self, "motion_policy", replace(self.motion_policy))
        object.__setattr__(self, "recovery_policy", replace(self.recovery_policy))
        object.__setattr__(self, "endpoints", endpoints)

    @classmethod
    def from_resolved_binding(
        cls,
        *,
        profile_id: str,
        preset_id: str,
        preset_schema_version: int,
        motion_policy: MotionPolicy,
        recovery_policy: RecoveryPolicy,
        endpoints: Iterable[EndpointBinding],
    ) -> ResolvedCorePolicyTrace:
        """Project one resolved preset and action binding to a trace."""
        return cls(
            profile_id=profile_id,
            preset_id=preset_id,
            preset_schema_version=preset_schema_version,
            motion_policy=motion_policy,
            recovery_policy=recovery_policy,
            endpoints=tuple(
                SkillEndpointBindingTrace.from_binding(endpoint)
                for endpoint in endpoints
            ),
        )

    def snapshot(self) -> ResolvedCorePolicyTrace:
        """Return an independently owned core-policy and binding trace."""
        return ResolvedCorePolicyTrace(
            profile_id=self.profile_id,
            preset_id=self.preset_id,
            preset_schema_version=self.preset_schema_version,
            motion_policy=self.motion_policy,
            recovery_policy=self.recovery_policy,
            endpoints=self.endpoints,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return deterministic policy and endpoint-binding metadata."""
        return {
            "profile_id": self.profile_id,
            "preset": {
                "preset_id": self.preset_id,
                "schema_version": self.preset_schema_version,
            },
            "motion_policy": _motion_policy_to_metadata(self.motion_policy),
            "recovery_policy": _recovery_policy_to_metadata(self.recovery_policy),
            "endpoints": [endpoint.to_metadata() for endpoint in self.endpoints],
        }


@dataclass(frozen=True, slots=True, eq=False)
class SkillPlanAttemptTrace:
    """Compact, typed trace of one installed action-plan generation.

    ``scene_dependency_monitor_until`` preserves the plan's per-entity exclusive
    waypoint cutoff: an entity is monitored only while the current waypoint index
    is smaller than its configured value.
    """

    attempt_generation: int
    trigger: str
    planned_at: float
    invocation_index: int
    planned_mask: torch.Tensor
    action_retry_counts: tuple[int, ...]
    replan_counts: tuple[int, ...]
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    plan_success_mask: torch.Tensor
    command_frame_count: int
    trajectory_segments: tuple[TrajectorySegment, ...]
    planned_scene_version: int
    planned_collision_world_revision: tuple[int, ...]
    scene_dependencies: tuple[str, ...]
    scene_dependency_monitor_until: Mapping[str, int]
    collision_world_sensitive: bool
    replannable: bool
    feedback_mode: ExecutionFeedbackMode
    effect_verification_kind: str | None
    resolved_core_policy: ResolvedCorePolicyTrace
    planner_backend: str
    planner_messages: tuple[str, ...]
    planner_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.attempt_generation) is not int or self.attempt_generation < 0:
            raise ValueError("attempt_generation must be non-negative.")
        if type(self.trigger) is not str or not self.trigger:
            raise ValueError("trigger must be a non-empty string.")
        if not math.isfinite(self.planned_at) or self.planned_at < 0.0:
            raise ValueError("planned_at must be finite and non-negative.")
        if type(self.invocation_index) is not int or self.invocation_index < 0:
            raise ValueError("invocation_index must be non-negative.")
        for name in ("planned_mask", "plan_success_mask"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.bool
                or value.dim() != 1
            ):
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if self.planned_mask.shape != self.plan_success_mask.shape:
            raise ValueError("Plan-attempt masks must have equal shapes.")
        if self.planned_mask.device != self.plan_success_mask.device:
            raise ValueError("Plan-attempt masks must share a device.")
        batch_size = int(self.planned_mask.numel())
        retries = tuple(self.action_retry_counts)
        replans = tuple(self.replan_counts)
        if len(retries) != batch_size or len(replans) != batch_size:
            raise ValueError("Recovery counters must contain one value per row.")
        if any(type(value) is not int or value < 0 for value in (*retries, *replans)):
            raise ValueError("Recovery counters must be non-negative integers.")
        if type(self.skill_id) is not str or not self.skill_id:
            raise ValueError("skill_id must be a non-empty string.")
        if self.invocation_id is not None and (
            type(self.invocation_id) is not str or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if type(self.invocation_revision) is not int or self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if type(self.command_frame_count) is not int or self.command_frame_count < 0:
            raise ValueError("command_frame_count must be non-negative.")
        segments = tuple(self.trajectory_segments)
        if not all(type(value) is TrajectorySegment for value in segments):
            raise TypeError(
                "trajectory_segments must contain TrajectorySegment values."
            )
        if (
            type(self.planned_scene_version) is not int
            or self.planned_scene_version < 0
        ):
            raise ValueError("planned_scene_version must be non-negative.")
        collision_revisions = tuple(self.planned_collision_world_revision)
        if len(collision_revisions) != batch_size or any(
            type(value) is not int or value < 0 for value in collision_revisions
        ):
            raise ValueError(
                "planned_collision_world_revision must contain one non-negative "
                "integer per row."
            )
        dependencies = tuple(self.scene_dependencies)
        if len(set(dependencies)) != len(dependencies) or not all(
            type(value) is str and value for value in dependencies
        ):
            raise ValueError("scene_dependencies must contain unique identifiers.")
        if not isinstance(self.scene_dependency_monitor_until, Mapping):
            raise TypeError("scene_dependency_monitor_until must be a mapping.")
        monitor_until = dict(self.scene_dependency_monitor_until)
        if not set(monitor_until).issubset(dependencies):
            raise ValueError(
                "scene_dependency_monitor_until keys must be scene dependencies."
            )
        for entity_id, waypoint_index in monitor_until.items():
            if (
                type(entity_id) is not str
                or not entity_id
                or type(waypoint_index) is not int
                or not 0 <= waypoint_index <= self.command_frame_count
            ):
                raise ValueError(
                    "scene_dependency_monitor_until must map non-empty entity IDs "
                    "to waypoint indices within the command sequence."
                )
        if type(self.collision_world_sensitive) is not bool:
            raise TypeError("collision_world_sensitive must be a bool.")
        if type(self.replannable) is not bool:
            raise TypeError("replannable must be a bool.")
        if not isinstance(self.feedback_mode, ExecutionFeedbackMode):
            raise TypeError("feedback_mode must be an ExecutionFeedbackMode.")
        if self.effect_verification_kind is not None and (
            type(self.effect_verification_kind) is not str
            or not self.effect_verification_kind
        ):
            raise ValueError("effect_verification_kind must be non-empty or None.")
        if type(self.resolved_core_policy) is not ResolvedCorePolicyTrace:
            raise TypeError(
                "resolved_core_policy must be exactly ResolvedCorePolicyTrace."
            )
        if type(self.planner_backend) is not str or not self.planner_backend:
            raise ValueError("planner_backend must be a non-empty string.")
        messages = tuple(self.planner_messages)
        if not all(type(value) is str for value in messages):
            raise TypeError("planner_messages must contain strings.")
        object.__setattr__(self, "planned_mask", self.planned_mask.clone())
        object.__setattr__(self, "plan_success_mask", self.plan_success_mask.clone())
        object.__setattr__(self, "action_retry_counts", retries)
        object.__setattr__(self, "replan_counts", replans)
        object.__setattr__(self, "trajectory_segments", segments)
        object.__setattr__(
            self,
            "planned_collision_world_revision",
            collision_revisions,
        )
        object.__setattr__(self, "scene_dependencies", dependencies)
        object.__setattr__(
            self,
            "scene_dependency_monitor_until",
            MappingProxyType(monitor_until),
        )
        object.__setattr__(
            self,
            "resolved_core_policy",
            self.resolved_core_policy.snapshot(),
        )
        object.__setattr__(self, "planner_messages", messages)
        object.__setattr__(
            self,
            "planner_metadata",
            _snapshot_metadata_mapping(self.planner_metadata),
        )

    @classmethod
    def from_execution_attempt(
        cls,
        attempt: ExecutionPlanAttempt,
        *,
        profile_id: str,
        preset_id: str,
        preset_schema_version: int,
    ) -> SkillPlanAttemptTrace:
        """Project one session-owned plan attempt to compact trace metadata."""
        if not isinstance(attempt, ExecutionPlanAttempt):
            raise TypeError("attempt must be an ExecutionPlanAttempt.")
        plan = attempt.plan
        request = attempt.request
        return cls(
            attempt_generation=attempt.attempt_generation,
            trigger=attempt.event_kind.value,
            planned_at=attempt.planned_at,
            invocation_index=attempt.invocation_index,
            planned_mask=attempt.planned_mask,
            action_retry_counts=attempt.action_retry_counts,
            replan_counts=attempt.replan_counts,
            skill_id=plan.skill_id,
            invocation_id=plan.invocation_id,
            invocation_revision=plan.invocation_revision,
            plan_success_mask=plan.plan_success,
            command_frame_count=plan.commands.frame_count,
            trajectory_segments=plan.segments,
            planned_scene_version=plan.planned_scene_version,
            planned_collision_world_revision=plan.planned_collision_world_revision,
            scene_dependencies=plan.scene_dependencies,
            scene_dependency_monitor_until=plan.scene_dependency_monitor_until,
            collision_world_sensitive=plan.collision_world_sensitive,
            replannable=plan.replannable,
            feedback_mode=plan.feedback_mode,
            effect_verification_kind=(
                None
                if plan.effect_verification is None
                else plan.effect_verification.kind
            ),
            resolved_core_policy=ResolvedCorePolicyTrace.from_resolved_binding(
                profile_id=profile_id,
                preset_id=preset_id,
                preset_schema_version=preset_schema_version,
                motion_policy=request.motion_policy,
                recovery_policy=request.recovery_policy,
                endpoints=request.binding.endpoints,
            ),
            planner_backend=plan.diagnostics.backend,
            planner_messages=plan.diagnostics.messages,
            planner_metadata=plan.diagnostics.metadata,
        )

    def snapshot(self) -> SkillPlanAttemptTrace:
        """Return an independently owned compact plan-attempt trace."""
        return SkillPlanAttemptTrace(
            attempt_generation=self.attempt_generation,
            trigger=self.trigger,
            planned_at=self.planned_at,
            invocation_index=self.invocation_index,
            planned_mask=self.planned_mask,
            action_retry_counts=self.action_retry_counts,
            replan_counts=self.replan_counts,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            plan_success_mask=self.plan_success_mask,
            command_frame_count=self.command_frame_count,
            trajectory_segments=self.trajectory_segments,
            planned_scene_version=self.planned_scene_version,
            planned_collision_world_revision=self.planned_collision_world_revision,
            scene_dependencies=self.scene_dependencies,
            scene_dependency_monitor_until=self.scene_dependency_monitor_until,
            collision_world_sensitive=self.collision_world_sensitive,
            replannable=self.replannable,
            feedback_mode=self.feedback_mode,
            effect_verification_kind=self.effect_verification_kind,
            resolved_core_policy=self.resolved_core_policy,
            planner_backend=self.planner_backend,
            planner_messages=self.planner_messages,
            planner_metadata=self.planner_metadata,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return one plan generation as deterministic JSON-safe data."""
        return {
            "attempt_generation": self.attempt_generation,
            "trigger": self.trigger,
            "planned_at": self.planned_at,
            "invocation_index": self.invocation_index,
            "planned_mask": _metadata_value(self.planned_mask),
            "recovery_counters": {
                "action_retries": list(self.action_retry_counts),
                "replans": list(self.replan_counts),
            },
            "skill_id": self.skill_id,
            "invocation_id": self.invocation_id,
            "invocation_revision": self.invocation_revision,
            "plan_success_mask": _metadata_value(self.plan_success_mask),
            "command_frame_count": self.command_frame_count,
            "trajectory_segments": [
                {
                    "name": segment.name,
                    "start": segment.start,
                    "stop": segment.stop,
                    "waypoint_count": segment.waypoint_count,
                }
                for segment in self.trajectory_segments
            ],
            "planned_scene_version": self.planned_scene_version,
            "planned_collision_world_revision": list(
                self.planned_collision_world_revision
            ),
            "scene_dependencies": list(self.scene_dependencies),
            "scene_dependency_monitor_until": {
                entity_id: self.scene_dependency_monitor_until[entity_id]
                for entity_id in sorted(self.scene_dependency_monitor_until)
            },
            "collision_world_sensitive": self.collision_world_sensitive,
            "replannable": self.replannable,
            "feedback_mode": self.feedback_mode.value,
            "effect_verification_kind": self.effect_verification_kind,
            "resolved_core_policy": self.resolved_core_policy.to_metadata(),
            "planner_diagnostics": {
                "backend": self.planner_backend,
                "messages": list(self.planner_messages),
                "metadata": _metadata_value(self.planner_metadata),
            },
        }


@dataclass(frozen=True, slots=True, eq=False)
class SkillEffectTrace:
    """One monitor decision correlated with an atomic verification boundary."""

    call_index: int
    verification_id: int
    observation_revision: int
    timestamp: float
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    effect_spec: SemanticEffectSpec
    monitor_id: str
    monitor_revision: str | None
    configured_monitor_params: Mapping[str, object]
    resolved_monitor_params: Mapping[str, object]
    evidence: Mapping[str, EffectEvidenceBatch]

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer.")
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if type(self.observation_revision) is not int or self.observation_revision < 0:
            raise ValueError("observation_revision must be non-negative.")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")
        for name in ("success_mask", "failure_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if self.success_mask.shape != self.failure_mask.shape:
            raise ValueError("Effect trace masks must have equal shapes.")
        if self.success_mask.device != self.failure_mask.device:
            raise ValueError("Effect trace masks must share a device.")
        if (self.success_mask & self.failure_mask).any():
            raise ValueError("Effect trace masks must not overlap.")
        if not isinstance(self.effect_spec, SemanticEffectSpec):
            raise TypeError("effect_spec must be a SemanticEffectSpec.")
        if type(self.monitor_id) is not str or not self.monitor_id:
            raise ValueError("monitor_id must be a non-empty string.")
        if self.monitor_revision is not None and (
            type(self.monitor_revision) is not str or not self.monitor_revision
        ):
            raise ValueError("monitor_revision must be non-empty or None.")
        evidence_types = (
            PoseRelationEvidenceBatch,
            BinaryEffectEvidenceBatch,
            ScalarEffectEvidenceBatch,
            JointStateEvidenceBatch,
        )
        evidence: dict[str, EffectEvidenceBatch] = {}
        for evidence_id, batch in self.evidence.items():
            if type(evidence_id) is not str or not evidence_id:
                raise ValueError("evidence keys must be non-empty strings.")
            if type(batch) not in evidence_types:
                raise TypeError("evidence values must be exact evidence batches.")
            if batch.evidence_id != evidence_id:
                raise ValueError("evidence keys must match batch evidence_id values.")
            evidence[evidence_id] = batch.snapshot()
        object.__setattr__(self, "success_mask", self.success_mask.clone())
        object.__setattr__(self, "failure_mask", self.failure_mask.clone())
        object.__setattr__(self, "effect_spec", self.effect_spec.snapshot())
        object.__setattr__(
            self,
            "configured_monitor_params",
            _snapshot_metadata_mapping(self.configured_monitor_params),
        )
        object.__setattr__(
            self,
            "resolved_monitor_params",
            _snapshot_metadata_mapping(self.resolved_monitor_params),
        )
        object.__setattr__(self, "evidence", MappingProxyType(evidence))

    def snapshot(self) -> SkillEffectTrace:
        """Return an independently owned trace."""
        return SkillEffectTrace(
            call_index=self.call_index,
            verification_id=self.verification_id,
            observation_revision=self.observation_revision,
            timestamp=self.timestamp,
            success_mask=self.success_mask,
            failure_mask=self.failure_mask,
            effect_spec=self.effect_spec,
            monitor_id=self.monitor_id,
            monitor_revision=self.monitor_revision,
            configured_monitor_params=self.configured_monitor_params,
            resolved_monitor_params=self.resolved_monitor_params,
            evidence=self.evidence,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return monitor contract, evidence, thresholds, and decision metadata."""
        return {
            "call_index": self.call_index,
            "verification_id": self.verification_id,
            "observation_revision": self.observation_revision,
            "timestamp": self.timestamp,
            "effect_spec": self.effect_spec.to_metadata(),
            "monitor": {
                "monitor_id": self.monitor_id,
                "revision": self.monitor_revision,
                "configured_params": _metadata_value(self.configured_monitor_params),
                "resolved_params": _metadata_value(self.resolved_monitor_params),
            },
            "evidence": {
                evidence_id: batch.to_metadata()
                for evidence_id, batch in sorted(self.evidence.items())
            },
            "decision": {
                "success_mask": _metadata_value(self.success_mask),
                "failure_mask": _metadata_value(self.failure_mask),
            },
        }


@dataclass(frozen=True, slots=True, eq=False)
class SkillFailure:
    """Per-environment semantic workflow failure."""

    call_index: int
    semantic_id: str
    env_mask: torch.Tensor
    message: str

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer.")
        if type(self.semantic_id) is not str or not self.semantic_id:
            raise ValueError("semantic_id must be a non-empty string.")
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a one-dimensional bool tensor.")
        if type(self.message) is not str or not self.message:
            raise ValueError("message must be a non-empty string.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())

    def snapshot(self) -> SkillFailure:
        """Return an independently owned failure."""
        return SkillFailure(
            call_index=self.call_index,
            semantic_id=self.semantic_id,
            env_mask=self.env_mask,
            message=self.message,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return one row-local failure as JSON-safe data."""
        return {
            "call_index": self.call_index,
            "semantic_id": self.semantic_id,
            "env_mask": _metadata_value(self.env_mask),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True, eq=False)
class SkillCallTrace:
    """Terminal trace for exactly one semantic call and execution session."""

    call_index: int
    semantic_id: str
    call_metadata: Mapping[str, object]
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    status: RunnerStatus
    entered_mask: torch.Tensor
    completed_mask: torch.Tensor
    failed_mask: torch.Tensor
    command_count: int
    resolved_core_policy: ResolvedCorePolicyTrace
    plan_attempts: tuple[SkillPlanAttemptTrace, ...]
    events: tuple[ExecutionEvent, ...] = ()
    effects: tuple[SkillEffectTrace, ...] = ()

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer.")
        for name in ("semantic_id", "skill_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        normalized_call = _snapshot_metadata_mapping(self.call_metadata)
        if normalized_call.get("semantic_id") != self.semantic_id:
            raise ValueError("call_metadata semantic_id must match semantic_id.")
        if self.invocation_id is not None and (
            type(self.invocation_id) is not str or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if not isinstance(self.status, RunnerStatus):
            raise TypeError("status must be a RunnerStatus.")
        if self.status is RunnerStatus.RUNNING:
            raise ValueError("A terminal call trace cannot have running status.")
        for name in ("entered_mask", "completed_mask", "failed_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if not (
            self.entered_mask.shape
            == self.completed_mask.shape
            == self.failed_mask.shape
        ):
            raise ValueError("Call trace masks must have equal shapes.")
        if not (
            self.entered_mask.device
            == self.completed_mask.device
            == self.failed_mask.device
        ):
            raise ValueError("Call trace masks must share a device.")
        if (self.completed_mask & ~self.entered_mask).any():
            raise ValueError("completed_mask must be a subset of entered_mask.")
        if (self.failed_mask & ~self.entered_mask).any():
            raise ValueError("failed_mask must be a subset of entered_mask.")
        if (self.completed_mask & self.failed_mask).any():
            raise ValueError("completed_mask and failed_mask must not overlap.")
        if type(self.command_count) is not int or self.command_count < 0:
            raise ValueError("command_count must be non-negative.")
        if type(self.resolved_core_policy) is not ResolvedCorePolicyTrace:
            raise TypeError(
                "resolved_core_policy must be exactly ResolvedCorePolicyTrace."
            )
        attempts = tuple(self.plan_attempts)
        if not all(type(attempt) is SkillPlanAttemptTrace for attempt in attempts):
            raise TypeError("plan_attempts must contain SkillPlanAttemptTrace values.")
        if attempts:
            generations = tuple(attempt.attempt_generation for attempt in attempts)
            if generations != tuple(
                range(generations[0], generations[0] + len(attempts))
            ):
                raise ValueError(
                    "plan_attempts must use contiguous ordered generations."
                )
            if attempts[-1].skill_id != self.skill_id:
                raise ValueError("The active plan-attempt skill must match skill_id.")
        elif self.status is not RunnerStatus.FAILED or self.command_count != 0:
            raise ValueError(
                "Only a preparation failure with no commands may omit plan_attempts."
            )
        object.__setattr__(self, "entered_mask", self.entered_mask.clone())
        object.__setattr__(self, "completed_mask", self.completed_mask.clone())
        object.__setattr__(self, "failed_mask", self.failed_mask.clone())
        object.__setattr__(self, "call_metadata", normalized_call)
        object.__setattr__(
            self,
            "resolved_core_policy",
            self.resolved_core_policy.snapshot(),
        )
        object.__setattr__(
            self,
            "plan_attempts",
            tuple(attempt.snapshot() for attempt in attempts),
        )
        object.__setattr__(
            self,
            "events",
            tuple(_snapshot_event(event) for event in self.events),
        )
        object.__setattr__(
            self,
            "effects",
            tuple(effect.snapshot() for effect in self.effects),
        )

    def snapshot(self) -> SkillCallTrace:
        """Return an independently owned call trace."""
        return SkillCallTrace(
            call_index=self.call_index,
            semantic_id=self.semantic_id,
            call_metadata=self.call_metadata,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            status=self.status,
            entered_mask=self.entered_mask,
            completed_mask=self.completed_mask,
            failed_mask=self.failed_mask,
            command_count=self.command_count,
            resolved_core_policy=self.resolved_core_policy,
            plan_attempts=self.plan_attempts,
            events=self.events,
            effects=self.effects,
        )

    @property
    def active_plan(self) -> SkillPlanAttemptTrace:
        """Return the final installed plan generation as an owned trace."""
        if not self.plan_attempts:
            raise RuntimeError("This call failed before an action plan was installed.")
        return self.plan_attempts[-1].snapshot()

    def to_metadata(self) -> dict[str, object]:
        """Return one semantic call, recovery history, and effects as JSON-safe data."""
        attempts = [attempt.to_metadata() for attempt in self.plan_attempts]
        return {
            "call_index": self.call_index,
            "semantic_id": self.semantic_id,
            "call": _metadata_value(self.call_metadata),
            "skill_id": self.skill_id,
            "invocation_id": self.invocation_id,
            "invocation_revision": self.invocation_revision,
            "status": self.status.value,
            "masks": {
                "entered": _metadata_value(self.entered_mask),
                "completed": _metadata_value(self.completed_mask),
                "failed": _metadata_value(self.failed_mask),
            },
            "command_count": self.command_count,
            "active_plan_attempt_generation": (
                None
                if not self.plan_attempts
                else self.plan_attempts[-1].attempt_generation
            ),
            "resolved_core_policy": self.resolved_core_policy.to_metadata(),
            "plan_attempts": attempts,
            "events": [_event_to_metadata(event) for event in self.events],
            "effects": [effect.to_metadata() for effect in self.effects],
        }


@dataclass(frozen=True, slots=True, eq=False)
class SkillResult:
    """Immutable workflow snapshot returned by sync and step-wise execution."""

    status: SkillStatus
    workflow_id: str | None
    current_call_index: int | None
    env_ids: torch.Tensor
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    cancelled_mask: torch.Tensor
    eligible_mask: torch.Tensor
    task_state: TaskState
    events: tuple[ExecutionEvent, ...] = ()
    calls: tuple[SkillCallTrace, ...] = ()
    effects: tuple[SkillEffectTrace, ...] = ()
    failures: tuple[SkillFailure, ...] = ()
    wait_duration: float = 0.0
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SkillStatus):
            raise TypeError("status must be a SkillStatus.")
        if self.workflow_id is not None and (
            type(self.workflow_id) is not str or not self.workflow_id
        ):
            raise ValueError("workflow_id must be a non-empty string or None.")
        if self.current_call_index is not None and (
            type(self.current_call_index) is not int or self.current_call_index < 0
        ):
            raise ValueError("current_call_index must be non-negative or None.")
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.dtype != torch.long or self.env_ids.dim() != 1:
            raise ValueError("env_ids must be a one-dimensional torch.long tensor.")
        if self.env_ids.numel() == 0:
            raise ValueError("env_ids must contain at least one environment.")
        if torch.unique(self.env_ids).numel() != self.env_ids.numel():
            raise ValueError("env_ids must be unique.")
        batch_size = int(self.env_ids.numel())
        for name in (
            "success_mask",
            "failure_mask",
            "cancelled_mask",
            "eligible_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.shape != (batch_size,):
                raise ValueError(f"{name} must be bool with shape ({batch_size},).")
            if value.device != self.env_ids.device:
                raise ValueError(f"{name} and env_ids must share a device.")
        if (self.success_mask & (self.failure_mask | self.cancelled_mask)).any():
            raise ValueError("Successful rows cannot also fail or be cancelled.")
        if (self.failure_mask & self.cancelled_mask).any():
            raise ValueError("Failed and cancelled masks must not overlap.")
        if (self.eligible_mask & (self.failure_mask | self.cancelled_mask)).any():
            raise ValueError("Eligible rows cannot also fail or be cancelled.")
        if (self.success_mask & ~self.eligible_mask).any():
            raise ValueError("success_mask must be a subset of eligible_mask.")
        if not isinstance(self.task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        if self.task_state.batch_size != batch_size:
            raise ValueError("task_state batch size must match env_ids.")
        if self.task_state.device != self.env_ids.device:
            raise ValueError("task_state and env_ids must share a device.")
        if not math.isfinite(self.wait_duration) or self.wait_duration < 0.0:
            raise ValueError("wait_duration must be finite and non-negative.")
        if self.message is not None and type(self.message) is not str:
            raise TypeError("message must be a string or None.")
        object.__setattr__(self, "env_ids", self.env_ids.clone())
        for name in (
            "success_mask",
            "failure_mask",
            "cancelled_mask",
            "eligible_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).clone())
        object.__setattr__(self, "task_state", _snapshot_task_state(self.task_state))
        object.__setattr__(
            self,
            "events",
            tuple(_snapshot_event(event) for event in self.events),
        )
        object.__setattr__(
            self,
            "calls",
            tuple(call.snapshot() for call in self.calls),
        )
        object.__setattr__(
            self,
            "effects",
            tuple(effect.snapshot() for effect in self.effects),
        )
        object.__setattr__(
            self,
            "failures",
            tuple(failure.snapshot() for failure in self.failures),
        )

    @property
    def terminal(self) -> bool:
        """Whether the workflow no longer accepts execution steps."""
        return self.status in {
            SkillStatus.COMPLETED,
            SkillStatus.FAILED,
            SkillStatus.CANCELLED,
        }

    def to_metadata(self) -> dict[str, object]:
        """Return a fresh deterministic JSON-safe workflow result.

        Recovery remains represented by the ordered :class:`ExecutionEvent`
        stream and by each call's complete plan-attempt history.  The returned
        object owns only Python scalars, lists, and dictionaries and can be
        serialized with ``json.dumps(..., allow_nan=False)``.
        """
        return {
            "schema_version": 1,
            "kind": "skill_result",
            "status": self.status.value,
            "workflow_id": self.workflow_id,
            "current_call_index": self.current_call_index,
            "env_ids": _metadata_value(self.env_ids),
            "masks": {
                "success": _metadata_value(self.success_mask),
                "failure": _metadata_value(self.failure_mask),
                "cancelled": _metadata_value(self.cancelled_mask),
                "eligible": _metadata_value(self.eligible_mask),
            },
            "task_state": task_state_to_metadata(self.task_state),
            "events": [_event_to_metadata(event) for event in self.events],
            "calls": [call.to_metadata() for call in self.calls],
            "effects": [effect.to_metadata() for effect in self.effects],
            "failures": [failure.to_metadata() for failure in self.failures],
            "wait_duration": self.wait_duration,
            "message": self.message,
        }


@runtime_checkable
class EffectEvidenceCollectorPort(Protocol):
    """Minimal collector surface consumed by :class:`SkillRuntime`."""

    def collect(
        self,
        spec: SemanticEffectSpec,
        *,
        timestamp: float,
        observation_revision: int,
        env_ids: torch.Tensor | None = None,
    ) -> Mapping[str, EffectEvidenceBatch]:
        """Acquire synchronized raw evidence for one grounded effect."""


@runtime_checkable
class SkillRuntimeProvider(Protocol):
    """Explicit environment adapter installed for :meth:`AtomicSkills.from_env`."""

    def create_skill_runtime(self, *, preset: str) -> SkillRuntime:
        """Build a fully connected semantic runtime for this environment."""


class _PrimedObservationProvider:
    """Return a JIT-grounding observation once before delegating fresh reads."""

    def __init__(
        self,
        context: PlanningContext,
        delegate: ObservationProvider,
    ) -> None:
        self._context: PlanningContext | None = context
        self._delegate = delegate

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Reuse the grounding snapshot for the session's first due cycle."""
        context = self._context
        if context is None:
            return self._delegate.observe(task_state)
        self._context = None
        return PlanningContext(
            robot=context.robot,
            task=task_state,
            scene=context.scene,
            env_ids=context.env_ids,
        )


class SkillRuntime:
    """JIT-ground and execute semantic calls through one runner per call.

    Static workflow analysis occurs once in :meth:`start`. Each call then gets
    a fresh observation, one grounded invocation, one execution session, and
    one :class:`ExecutionRunner`. Verified task state and row eligibility cross
    call barriers; execution sessions never do.
    """

    def __init__(
        self,
        compiler: SemanticSkillCompiler,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        evidence_collector: EffectEvidenceCollectorPort,
        *,
        task_state: TaskState | None = None,
        clock: ExecutionClock | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
    ) -> None:
        if not isinstance(compiler, SemanticSkillCompiler):
            raise TypeError("compiler must be a SemanticSkillCompiler.")
        if not isinstance(observation_provider, ObservationProvider):
            raise TypeError("observation_provider must implement ObservationProvider.")
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        if not isinstance(evidence_collector, EffectEvidenceCollectorPort):
            raise TypeError(
                "evidence_collector must implement EffectEvidenceCollectorPort."
            )
        if clock is not None and not isinstance(clock, ExecutionClock):
            raise TypeError("clock must implement ExecutionClock.")
        if runner_cfg is not None and not isinstance(runner_cfg, ExecutionRunnerCfg):
            raise TypeError("runner_cfg must be an ExecutionRunnerCfg or None.")
        integration = compiler.integration
        engine = integration.engine
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError(
                "compiler.integration.engine must be an AtomicActionEngine."
            )
        initial_task = (
            engine.initial_context().task if task_state is None else task_state
        )
        if not isinstance(initial_task, TaskState):
            raise TypeError("task_state must be a TaskState or None.")
        if initial_task.device != engine.device:
            raise ValueError("task_state and compiler engine must share a device.")

        self._compiler = compiler
        self._engine = engine
        self._observation_provider = observation_provider
        self._command_sink = command_sink
        self._evidence_collector = evidence_collector
        self._clock = clock or MonotonicExecutionClock()
        self._runner_cfg = runner_cfg or ExecutionRunnerCfg()
        self._task_state = _snapshot_task_state(initial_task)
        self._env_ids = torch.arange(
            self._task_state.batch_size,
            dtype=torch.long,
            device=self._task_state.device,
        )
        self._has_observed_env_ids = False
        self._status = SkillStatus.IDLE
        self._workflow: object | None = None
        self._workflow_id: str | None = None
        self._calls: tuple[SemanticCallSpec, ...] = ()
        self._execution_prefix_length = 0
        self._current_call_index: int | None = None
        self._runner: ExecutionRunner | None = None
        self._grounded: object | None = None
        self._call_entered_mask = torch.zeros(
            self._task_state.batch_size,
            dtype=torch.bool,
            device=self._task_state.device,
        )
        self._eligible = torch.ones_like(self._call_entered_mask)
        self._success = torch.zeros_like(self._eligible)
        self._failed = torch.zeros_like(self._eligible)
        self._cancelled = torch.zeros_like(self._eligible)
        self._events: list[ExecutionEvent] = []
        self._call_traces: list[SkillCallTrace] = []
        self._effect_traces: list[SkillEffectTrace] = []
        self._failures: list[SkillFailure] = []
        self._call_event_offset = 0
        self._call_effect_offset = 0
        self._observation_revision = 0
        self._wait_duration = 0.0
        self._message: str | None = None

    @classmethod
    def from_components(
        cls,
        compiler: SemanticSkillCompiler,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        evidence_collector: EffectEvidenceCollectorPort,
        *,
        task_state: TaskState | None = None,
        clock: ExecutionClock | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
    ) -> SkillRuntime:
        """Construct the canonical runtime from explicit reusable ports."""
        return cls(
            compiler,
            observation_provider,
            command_sink,
            evidence_collector,
            task_state=task_state,
            clock=clock,
            runner_cfg=runner_cfg,
        )

    @property
    def compiler(self) -> SemanticSkillCompiler:
        """Return the installed semantic compiler."""
        return self._compiler

    @property
    def clock(self) -> ExecutionClock:
        """Return the shared execution clock used by this runtime.

        Parallel coordinators use the same clock for every derived lane so a
        branch cannot advance independently of the environment step grid.
        """
        return self._clock

    @property
    def scene_registry(self) -> SceneRegistry:
        """Return the authoritative semantic scene registry."""
        return self._compiler.integration.scene_registry

    @property
    def task_state(self) -> TaskState:
        """Return an owned snapshot of persistent verified task state."""
        return _snapshot_task_state(self._task_state)

    def fork(
        self,
        command_sink: CommandSink,
        *,
        task_state: TaskState | None = None,
    ) -> SkillRuntime:
        """Create an independent execution lane from the same runtime ports.

        The derived runtime shares the immutable compiler integration,
        observation/evidence providers, clock, and runner policy, but owns its
        workflow, runner, masks, and verified task state.  Its command sink is
        supplied explicitly so a parallel coordinator can buffer commands
        until all lanes have reached the same environment tick.

        Args:
            command_sink: Lane-local command sink.
            task_state: Optional verified barrier state.  The current owned
                task state is used when omitted.

        Returns:
            A new idle semantic runtime for one independent lane.
        """
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        initial_state = self.task_state if task_state is None else task_state
        if not isinstance(initial_state, TaskState):
            raise TypeError("task_state must be a TaskState or None.")
        return SkillRuntime(
            self._compiler,
            self._observation_provider,
            command_sink,
            self._evidence_collector,
            task_state=initial_state,
            clock=self._clock,
            runner_cfg=self._runner_cfg,
        )

    @property
    def status(self) -> SkillStatus:
        """Return the current workflow status."""
        return self._status

    @property
    def result(self) -> SkillResult:
        """Return an immutable snapshot of the current workflow."""
        return SkillResult(
            status=self._status,
            workflow_id=self._workflow_id,
            current_call_index=self._current_call_index,
            env_ids=self._env_ids,
            success_mask=self._success,
            failure_mask=self._failed,
            cancelled_mask=self._cancelled,
            eligible_mask=self._eligible,
            task_state=self._task_state,
            events=tuple(self._events),
            calls=tuple(self._call_traces),
            effects=tuple(self._effect_traces),
            failures=tuple(self._failures),
            wait_duration=self._wait_duration,
            message=self._message,
        )

    def start(
        self,
        *calls: SemanticCallSpec | Iterable[SemanticCallSpec],
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SkillResult:
        """Analyze once and prepare the first call without blocking on motion.

        Args:
            *calls: Complete ordered semantic analysis window.  Calls after the
                execution prefix participate in static look-ahead but are not
                grounded or executed by this run.
            workflow_id: Stable workflow identifier used in diagnostics.
            eligible_mask: Optional row-local execution eligibility.
            execution_prefix_length: Number of leading calls to execute.  When
                omitted, the complete analysis window is executed.

        Returns:
            Immutable initial runtime result.
        """
        if self._status is SkillStatus.RUNNING:
            raise RuntimeError("A semantic workflow is already running.")
        normalized = self._normalize_calls(calls)
        if type(workflow_id) is not str or not workflow_id:
            raise ValueError("workflow_id must be a non-empty string.")
        prefix_length = self._normalize_execution_prefix_length(
            execution_prefix_length,
            call_count=len(normalized),
        )
        workflow = self._compiler.analyze(normalized, workflow_id=workflow_id)
        self._reset_workflow(
            normalized,
            workflow,
            workflow_id=workflow_id,
            eligible_mask=eligible_mask,
            execution_prefix_length=prefix_length,
        )
        try:
            self._prepare_call(0)
        except Exception as exc:  # noqa: BLE001 - return one uniform result
            self._fail_preparation(0, exc)
        return self.result

    def step(self) -> SkillResult:
        """Advance the current call by at most one due runner cycle."""
        if self._status is not SkillStatus.RUNNING:
            return self.result
        runner = self._require_runner()
        grounded = self._require_grounded()
        monitor = getattr(grounded, "effect_monitor", None)
        verifier = self._effect_verifier if monitor is not None else None
        runner_step = runner.step(effect_verifier=verifier)
        self._consume_runner_step(runner_step)
        if (
            runner_step.status is RunnerStatus.RUNNING
            and runner_step.tick is not None
            and runner_step.tick.pending_effect is not None
            and monitor is None
        ):
            self._abort(
                "The atomic plan requested effect verification, but the grounded "
                "semantic call did not install an effect monitor."
            )
            return self.result
        if runner_step.status is RunnerStatus.RUNNING:
            return self.result
        self._finish_current_call(runner_step)
        if runner_step.status is RunnerStatus.COMPLETED:
            if self._eligible.any() and self._has_next_call:
                assert self._current_call_index is not None
                next_index = self._current_call_index + 1
                try:
                    self._prepare_call(next_index)
                except Exception as exc:  # noqa: BLE001 - preserve workflow trace
                    self._fail_preparation(next_index, exc)
            elif self._eligible.any():
                self._success = self._eligible.clone()
                self._status = SkillStatus.COMPLETED
                self._current_call_index = None
                self._wait_duration = 0.0
            else:
                self._status = (
                    SkillStatus.CANCELLED
                    if self._cancelled.any() and not self._failed.any()
                    else SkillStatus.FAILED
                )
                self._current_call_index = None
                self._wait_duration = 0.0
        elif runner_step.status is RunnerStatus.CANCELLED:
            self._status = SkillStatus.CANCELLED
            self._current_call_index = None
            self._wait_duration = 0.0
        else:
            self._status = SkillStatus.FAILED
            self._current_call_index = None
            self._wait_duration = 0.0
        return self.result

    def run(
        self,
        *calls: SemanticCallSpec | Iterable[SemanticCallSpec],
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
        max_steps: int = 100_000,
    ) -> SkillResult:
        """Synchronously execute an analyzed semantic-call prefix."""
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        result = self.start(
            *calls,
            workflow_id=workflow_id,
            eligible_mask=eligible_mask,
            execution_prefix_length=execution_prefix_length,
        )
        for _ in range(max_steps):
            if result.terminal:
                return result
            if result.wait_duration > 0.0:
                self._clock.sleep(result.wait_duration)
            result = self.step()
        self._abort(f"Semantic runtime exceeded max_steps={max_steps}.")
        return self.result

    def cancel(
        self, reason: str = "Semantic workflow cancelled by caller."
    ) -> SkillResult:
        """Cancel the active runner and inherit its cancel-then-hold behavior."""
        if type(reason) is not str or not reason:
            raise ValueError("reason must be a non-empty string.")
        if self._status is not SkillStatus.RUNNING:
            return self.result
        runner_step = self._require_runner().cancel(reason)
        self._consume_runner_step(runner_step)
        active = self._eligible & ~self._failed
        self._message = runner_step.message or reason
        self._finish_current_call(runner_step)
        self._cancelled |= active
        self._eligible &= ~active
        self._status = (
            SkillStatus.CANCELLED
            if runner_step.status is RunnerStatus.CANCELLED
            else SkillStatus.FAILED
        )
        if self._status is SkillStatus.FAILED:
            self._failed |= active
            self._cancelled &= ~active
        self._current_call_index = None
        self._wait_duration = 0.0
        return self.result

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> SkillResult:
        """Cancel selected rows while the remaining shared call keeps running.

        This is the row-local cancellation boundary used by a parallel
        fail-fast coordinator. The active runner remains the sole owner of
        controller neutralization and effect-request correlation.

        Args:
            env_mask: Rows to remove permanently from this workflow.
            reason: Human-readable cancellation reason.

        Returns:
            Updated immutable workflow result.
        """
        if self._status is not SkillStatus.RUNNING:
            return self.result
        if not isinstance(env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if (
            env_mask.dtype != torch.bool
            or env_mask.shape != self._eligible.shape
            or env_mask.device != self._eligible.device
        ):
            raise ValueError(
                "env_mask must be bool and match the runtime batch/device."
            )
        if type(reason) is not str or not reason:
            raise ValueError("reason must be a non-empty string.")
        changed = self._require_runner().deactivate_rows(
            env_mask & self._eligible,
            reason=reason,
        )
        self._cancelled |= changed
        self._eligible &= ~changed
        if not self._eligible.any():
            runner_step = self._require_runner().cancel(reason)
            self._consume_runner_step(runner_step)
            self._finish_current_call(runner_step)
            self._status = (
                SkillStatus.CANCELLED
                if runner_step.status is RunnerStatus.CANCELLED
                else SkillStatus.FAILED
            )
            if self._status is SkillStatus.FAILED:
                failed = self._call_entered_mask & ~self._cancelled
                self._failed |= failed
            self._current_call_index = None
            self._wait_duration = 0.0
        return self.result

    def adopt_verified_task_state(self, task_state: TaskState) -> SkillResult:
        """Install a verified state snapshot between independent workflows.

        Parallel coordinators use this explicit barrier operation after
        deterministically merging branch-local effects. Running workflows
        cannot replace their runner-owned state.
        """
        if self._status is SkillStatus.RUNNING:
            raise RuntimeError("Cannot replace task state while a workflow is running.")
        if not isinstance(task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        if (
            task_state.batch_size != self._task_state.batch_size
            or task_state.device != self._task_state.device
        ):
            raise ValueError("task_state must match the runtime batch and device.")
        self._task_state = _snapshot_task_state(task_state)
        return self.result

    @property
    def _has_next_call(self) -> bool:
        assert self._current_call_index is not None
        return self._current_call_index + 1 < self._execution_prefix_length

    @staticmethod
    def _normalize_execution_prefix_length(
        value: int | None,
        *,
        call_count: int,
    ) -> int:
        """Normalize a non-empty execution prefix inside one analysis window."""
        if value is None:
            return call_count
        if type(value) is not int:
            raise TypeError("execution_prefix_length must be an integer or None.")
        if not 1 <= value <= call_count:
            raise ValueError(
                "execution_prefix_length must be in " f"[1, {call_count}], got {value}."
            )
        return value

    def _normalize_calls(
        self,
        supplied: tuple[SemanticCallSpec | Iterable[SemanticCallSpec], ...],
    ) -> tuple[SemanticCallSpec, ...]:
        """Normalize varargs and one explicit iterable to the same compiler path."""
        if len(supplied) == 1 and not isinstance(supplied[0], SemanticCallSpec):
            candidate = supplied[0]
            if isinstance(candidate, (str, bytes)):
                raise TypeError("calls must contain SemanticCallSpec values.")
            try:
                calls = tuple(candidate)
            except TypeError as exc:
                raise TypeError(
                    "A single run argument must be a SemanticCallSpec or iterable."
                ) from exc
        else:
            calls = tuple(supplied)
        if not calls:
            raise ValueError("A semantic workflow requires at least one call.")
        if not all(isinstance(call, SemanticCallSpec) for call in calls):
            raise TypeError("calls must contain SemanticCallSpec values.")
        return calls

    def _reset_workflow(
        self,
        calls: tuple[SemanticCallSpec, ...],
        workflow: object,
        *,
        workflow_id: str,
        eligible_mask: torch.Tensor | None,
        execution_prefix_length: int,
    ) -> None:
        """Reset per-run state while retaining verified symbolic state."""
        if eligible_mask is None:
            eligible = torch.ones(
                self._task_state.batch_size,
                dtype=torch.bool,
                device=self._task_state.device,
            )
        else:
            if not isinstance(eligible_mask, torch.Tensor):
                raise TypeError("eligible_mask must be a torch.Tensor or None.")
            if eligible_mask.dtype != torch.bool or eligible_mask.shape != (
                self._task_state.batch_size,
            ):
                raise ValueError(
                    "eligible_mask must be bool with shape "
                    f"({self._task_state.batch_size},)."
                )
            eligible = eligible_mask.to(self._task_state.device).clone()
        if not eligible.any():
            raise ValueError("eligible_mask must contain at least one active row.")
        self._workflow = workflow
        self._workflow_id = workflow_id
        self._calls = calls
        self._execution_prefix_length = execution_prefix_length
        self._current_call_index = 0
        self._runner = None
        self._grounded = None
        self._eligible = eligible
        self._success = torch.zeros_like(eligible)
        self._failed = torch.zeros_like(eligible)
        self._cancelled = torch.zeros_like(eligible)
        self._events = []
        self._call_traces = []
        self._effect_traces = []
        self._failures = []
        self._call_event_offset = 0
        self._call_effect_offset = 0
        self._observation_revision = 0
        self._wait_duration = 0.0
        self._message = None
        self._status = SkillStatus.RUNNING

    def _observe_for_grounding(self) -> PlanningContext:
        """Capture and normalize one fresh context for JIT lowering."""
        context = self._observation_provider.observe(self._task_state)
        if not isinstance(context, PlanningContext):
            raise TypeError(
                "ObservationProvider.observe() must return PlanningContext."
            )
        normalized = PlanningContext(
            robot=context.robot,
            task=self._task_state,
            scene=context.scene,
            env_ids=context.env_ids,
        )
        if normalized.batch_size != self._task_state.batch_size:
            raise ValueError(
                "Observation batch size changed during semantic execution."
            )
        if normalized.robot.qpos.device != self._task_state.device:
            raise ValueError("Observation and verified TaskState must share a device.")
        if self._has_observed_env_ids:
            if normalized.env_ids.device != self._env_ids.device or not torch.equal(
                normalized.env_ids,
                self._env_ids,
            ):
                raise ValueError(
                    "Observation env_ids must remain stable across call barriers."
                )
        else:
            self._env_ids = normalized.env_ids.clone()
            self._has_observed_env_ids = True
        return normalized

    def _prepare_call(self, call_index: int) -> None:
        """Freshly ground and create exactly one session and runner."""
        assert self._workflow is not None
        context = self._observe_for_grounding()
        grounded = self._compiler.ground(
            self._workflow,
            call_index,
            context,
            eligible_mask=self._eligible,
        )
        invocation = getattr(grounded, "invocation", None)
        grounded_eligible = getattr(grounded, "eligible_mask", None)
        effect_spec = getattr(grounded, "effect_spec", None)
        effect_monitor = getattr(grounded, "effect_monitor", None)
        if invocation is None:
            raise TypeError("Semantic compiler ground() must return an invocation.")
        if not isinstance(grounded_eligible, torch.Tensor) or not torch.equal(
            grounded_eligible,
            self._eligible,
        ):
            raise ValueError("Grounded call must preserve runtime eligibility.")
        if (effect_spec is None) != (effect_monitor is None):
            raise ValueError(
                "Grounded effect_spec and effect_monitor must be set together."
            )
        if effect_spec is not None:
            if not isinstance(effect_spec, SemanticEffectSpec):
                raise TypeError("Grounded effect_spec must be a SemanticEffectSpec.")
            if not isinstance(effect_monitor, EffectMonitor):
                raise TypeError("Grounded effect_monitor must be an EffectMonitor.")
            if effect_spec.env_ids.device != context.env_ids.device or not torch.equal(
                effect_spec.env_ids,
                context.env_ids,
            ):
                raise ValueError("Grounded effect env_ids must match the call context.")

        self._grounded = grounded
        session = self._engine.start(
            (invocation,),
            context,
            eligible_mask=self._eligible,
        )
        primed = _PrimedObservationProvider(context, self._observation_provider)
        runner = ExecutionRunner(
            session,
            primed,
            self._command_sink,
            clock=self._clock,
            cfg=self._runner_cfg,
        )
        self._current_call_index = call_index
        self._runner = runner
        self._call_entered_mask = self._eligible.clone()
        self._call_event_offset = len(self._events)
        self._call_effect_offset = len(self._effect_traces)
        self._wait_duration = 0.0

    def _effect_verifier(
        self,
        context: PlanningContext,
        request: EffectVerificationRequest,
    ) -> EffectVerificationResult:
        """Collect raw evidence and feed the grounded call's monitor."""
        grounded = self._require_grounded()
        spec = getattr(grounded, "effect_spec", None)
        monitor = getattr(grounded, "effect_monitor", None)
        if not isinstance(spec, SemanticEffectSpec) or not isinstance(
            monitor,
            EffectMonitor,
        ):
            raise RuntimeError(
                "The active atomic plan requested effect verification, but its "
                "semantic call has no grounded effect monitor."
            )
        if request.skill_id != spec.skill_id:
            raise ValueError("Effect request skill_id does not match the effect spec.")
        if request.invocation_id != spec.invocation_id:
            raise ValueError(
                "Effect request invocation_id does not match the effect spec."
            )
        if request.invocation_revision != spec.invocation_revision:
            raise ValueError("Effect request revision does not match the effect spec.")
        observation_revision = self._observation_revision
        self._observation_revision += 1
        selected_env_ids = spec.env_ids[request.env_mask.to(spec.env_ids.device)]
        evidence = self._evidence_collector.collect(
            spec,
            timestamp=context.robot.timestamp,
            observation_revision=observation_revision,
            env_ids=selected_env_ids,
        )
        decision = monitor.observe(request, evidence)
        analyzed = getattr(grounded, "analyzed", None)
        monitor_ref = getattr(analyzed, "effect_monitor_ref", None)
        if monitor_ref is not None and not isinstance(monitor_ref, EffectMonitorRef):
            raise TypeError("Grounded effect monitor reference must be typed.")
        if monitor_ref is None:
            monitor_id = f"{type(monitor).__module__}.{type(monitor).__qualname__}"
            monitor_revision = None
            configured_monitor_params: Mapping[str, object] = {}
        else:
            monitor_id = monitor_ref.monitor_id
            monitor_revision = monitor_ref.revision
            configured_monitor_params = monitor_ref.params
        resolved_monitor_params = monitor.resolved_params
        if not isinstance(resolved_monitor_params, Mapping):
            raise TypeError("EffectMonitor.resolved_params must return a mapping.")
        trace = SkillEffectTrace(
            call_index=self._require_call_index(),
            verification_id=request.verification_id,
            observation_revision=observation_revision,
            timestamp=context.robot.timestamp,
            success_mask=decision.success_mask,
            failure_mask=decision.failure_mask,
            effect_spec=spec,
            monitor_id=monitor_id,
            monitor_revision=monitor_revision,
            configured_monitor_params=configured_monitor_params,
            resolved_monitor_params=resolved_monitor_params,
            evidence=evidence,
        )
        self._effect_traces.append(trace)
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=decision.success_mask,
            failure_mask=decision.failure_mask,
        )

    def _consume_runner_step(self, runner_step: RunnerStep) -> None:
        """Merge one runner update into workflow-level traces."""
        self._wait_duration = runner_step.wait_duration
        if runner_step.tick is not None:
            self._task_state = _snapshot_task_state(runner_step.tick.task_state)
            self._events.extend(
                _snapshot_event(event) for event in runner_step.tick.events
            )
        if runner_step.message:
            self._message = runner_step.message

    def _finish_current_call(self, runner_step: RunnerStep) -> None:
        """Commit terminal row masks and append exactly one call trace."""
        runner = self._require_runner()
        grounded = self._require_grounded()
        call_index = self._require_call_index()
        self._task_state = _snapshot_task_state(runner.session.task_state)
        after = runner.session.eligible_mask
        invocation = getattr(grounded, "invocation")
        if runner_step.status is RunnerStatus.COMPLETED:
            completed = self._call_entered_mask & after
            failed = self._call_entered_mask & ~after & ~self._cancelled
        elif runner_step.status is RunnerStatus.CANCELLED:
            completed = torch.zeros_like(self._call_entered_mask)
            failed = torch.zeros_like(self._call_entered_mask)
        else:
            completed = torch.zeros_like(self._call_entered_mask)
            failed = self._call_entered_mask & ~self._cancelled
            after = self._eligible & ~failed

        self._eligible = after.clone()
        self._failed |= failed
        if failed.any():
            message = runner_step.message or "Semantic call failed for these rows."
            self._failures.append(
                SkillFailure(
                    call_index=call_index,
                    semantic_id=self._calls[call_index].semantic_id,
                    env_mask=failed,
                    message=message,
                )
            )
        plan_attempts = tuple(
            SkillPlanAttemptTrace.from_execution_attempt(
                attempt,
                profile_id=grounded.analyzed.bound.robot_profile.profile_id,
                preset_id=grounded.analyzed.bound.preset.preset_id,
                preset_schema_version=grounded.analyzed.bound.preset.schema_version,
            )
            for attempt in runner.session.plan_attempts
        )
        self._call_traces.append(
            SkillCallTrace(
                call_index=call_index,
                semantic_id=self._calls[call_index].semantic_id,
                call_metadata=self._calls[call_index].to_metadata(),
                skill_id=invocation.skill_id,
                invocation_id=invocation.invocation_id,
                invocation_revision=invocation.revision,
                status=runner_step.status,
                entered_mask=self._call_entered_mask,
                completed_mask=completed,
                failed_mask=failed,
                command_count=runner_step.command_count,
                resolved_core_policy=plan_attempts[-1].resolved_core_policy,
                plan_attempts=plan_attempts,
                events=tuple(self._events[self._call_event_offset :]),
                effects=tuple(self._effect_traces[self._call_effect_offset :]),
            )
        )
        self._runner = None
        self._grounded = None

    def _fail_preparation(self, call_index: int, exc: Exception) -> None:
        """Convert a post-barrier grounding failure to a terminal result."""
        failed = self._eligible.clone()
        self._failed |= failed
        self._eligible &= ~failed
        semantic_id = self._calls[call_index].semantic_id
        message = (
            f"Could not prepare semantic call {call_index} ({semantic_id!r}): "
            f"{type(exc).__name__}: {exc}"
        )
        self._failures.append(SkillFailure(call_index, semantic_id, failed, message))
        self._append_preparation_failure_trace(call_index, failed)
        self._message = message
        self._status = SkillStatus.FAILED
        self._current_call_index = None
        self._runner = None
        self._grounded = None
        self._wait_duration = 0.0

    def _append_preparation_failure_trace(
        self,
        call_index: int,
        failed_mask: torch.Tensor,
    ) -> None:
        """Record statically resolved policy choices when planning never starts."""
        grounded = self._grounded
        analyzed = getattr(grounded, "analyzed", None)
        invocation = getattr(grounded, "invocation", None)
        if analyzed is None:
            workflow_calls = getattr(self._workflow, "calls", ())
            if call_index < len(workflow_calls):
                analyzed = workflow_calls[call_index]
        bound = getattr(analyzed, "bound", None)
        if bound is None:
            return
        try:
            profile = bound.robot_profile
            preset = bound.preset
            action_binding = (
                bound.binding.action_binding
                if invocation is None
                else invocation.binding
            )
            resolved = ResolvedCorePolicyTrace.from_resolved_binding(
                profile_id=profile.profile_id,
                preset_id=preset.preset_id,
                preset_schema_version=preset.schema_version,
                motion_policy=(
                    preset.motion_policy
                    if invocation is None
                    else invocation.motion_policy
                ),
                recovery_policy=(
                    preset.recovery_policy
                    if invocation is None
                    else invocation.recovery_policy
                ),
                endpoints=action_binding.endpoints,
            )
            skill_id = bound.linked.descriptor.skill_id
        except (AttributeError, TypeError, ValueError):
            return
        self._call_traces.append(
            SkillCallTrace(
                call_index=call_index,
                semantic_id=self._calls[call_index].semantic_id,
                call_metadata=self._calls[call_index].to_metadata(),
                skill_id=skill_id,
                invocation_id=(
                    None if invocation is None else invocation.invocation_id
                ),
                invocation_revision=(0 if invocation is None else invocation.revision),
                status=RunnerStatus.FAILED,
                entered_mask=failed_mask,
                completed_mask=torch.zeros_like(failed_mask),
                failed_mask=failed_mask,
                command_count=0,
                resolved_core_policy=resolved,
                plan_attempts=(),
            )
        )

    def _abort(self, reason: str) -> None:
        """Safe-stop the active runner and mark remaining rows failed."""
        if self._runner is not None:
            safe_stop_step = self._runner.cancel(reason)
            runner_step = replace(
                safe_stop_step,
                status=RunnerStatus.FAILED,
                message=reason,
            )
            self._consume_runner_step(runner_step)
            self._finish_current_call(runner_step)
        failed = self._eligible.clone()
        self._failed |= failed
        self._eligible &= ~failed
        if failed.any() and self._calls:
            call_index = min(
                self._current_call_index or 0,
                len(self._calls) - 1,
            )
            self._failures.append(
                SkillFailure(
                    call_index,
                    self._calls[call_index].semantic_id,
                    failed,
                    reason,
                )
            )
        self._message = reason
        self._status = SkillStatus.FAILED
        self._current_call_index = None
        self._wait_duration = 0.0

    def _require_runner(self) -> ExecutionRunner:
        if self._runner is None:
            raise RuntimeError("No semantic call runner is active.")
        return self._runner

    def _require_grounded(self) -> object:
        if self._grounded is None:
            raise RuntimeError("No grounded semantic call is active.")
        return self._grounded

    def _require_call_index(self) -> int:
        if self._current_call_index is None:
            raise RuntimeError("No semantic call is active.")
        return self._current_call_index


class SkillScene:
    """Typed convenience lookup surface backed by one immutable registry."""

    def __init__(self, registry: SceneRegistry) -> None:
        if not isinstance(registry, SceneRegistry):
            raise TypeError("registry must be a SceneRegistry.")
        self._registry = registry

    @property
    def registry(self) -> SceneRegistry:
        """Return the authoritative scene registry."""
        return self._registry

    def entity(self, identifier: str | SceneEntityRef) -> SceneEntityRef:
        """Resolve any registered semantic entity."""
        return self._registry.resolve(identifier)

    def object(self, identifier: str | SceneObjectRef) -> SceneObjectRef:
        """Resolve a registered semantic object."""
        return self._registry.resolve(identifier, expected_type=SceneObjectRef)

    def articulation(
        self,
        identifier: str | SceneArticulationRef,
    ) -> SceneArticulationRef:
        """Resolve a registered articulation."""
        return self._registry.resolve(identifier, expected_type=SceneArticulationRef)

    def link(self, identifier: str | SceneLinkRef) -> SceneLinkRef:
        """Resolve a registered articulation link."""
        return self._registry.resolve(identifier, expected_type=SceneLinkRef)

    def affordance(
        self,
        identifier: str | SceneAffordanceRef,
    ) -> SceneAffordanceRef:
        """Resolve a registered semantic affordance."""
        return self._registry.resolve(identifier, expected_type=SceneAffordanceRef)


class AtomicSkills:
    """Small application-facing facade over :class:`SkillRuntime`."""

    def __init__(self, runtime: SkillRuntime) -> None:
        if not isinstance(runtime, SkillRuntime):
            raise TypeError("runtime must be a SkillRuntime.")
        self._runtime = runtime
        self._scene = SkillScene(runtime.scene_registry)

    @classmethod
    def from_components(
        cls,
        compiler: SemanticSkillCompiler,
        observation_provider: ObservationProvider,
        command_sink: CommandSink,
        evidence_collector: EffectEvidenceCollectorPort,
        *,
        task_state: TaskState | None = None,
        clock: ExecutionClock | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
    ) -> AtomicSkills:
        """Build a facade from explicit compiler and runtime ports."""
        return cls(
            SkillRuntime.from_components(
                compiler,
                observation_provider,
                command_sink,
                evidence_collector,
                task_state=task_state,
                clock=clock,
                runner_cfg=runner_cfg,
            )
        )

    @classmethod
    def from_env(cls, env: object, *, preset: str = "safe") -> AtomicSkills:
        """Build through an explicitly installed environment integration adapter.

        The method deliberately does not inspect generic environment attributes
        for robots, scenes, controllers, or managers. An environment integration
        must implement :class:`SkillRuntimeProvider` and own those decisions.
        """
        if type(preset) is not str or not preset:
            raise ValueError("preset must be a non-empty string.")
        if not isinstance(env, SkillRuntimeProvider):
            raise TypeError(
                "Environment has no semantic-skill integration adapter. Install "
                "SkillRuntimeProvider.create_skill_runtime(*, preset=...) or use "
                "AtomicSkills.from_components(...) with explicit ports."
            )
        runtime = env.create_skill_runtime(preset=preset)
        if not isinstance(runtime, SkillRuntime):
            raise TypeError(
                "SkillRuntimeProvider.create_skill_runtime() must return "
                "SkillRuntime."
            )
        return cls(runtime)

    @property
    def runtime(self) -> SkillRuntime:
        """Return the canonical runtime for advanced step-wise use."""
        return self._runtime

    @property
    def scene(self) -> SkillScene:
        """Return typed semantic scene lookup helpers."""
        return self._scene

    @property
    def result(self) -> SkillResult:
        """Return the current immutable runtime result."""
        return self._runtime.result

    def start(
        self,
        *calls: SemanticCallSpec | Iterable[SemanticCallSpec],
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SkillResult:
        """Start non-blocking semantic execution without exposing sessions."""
        return self._runtime.start(
            *calls,
            workflow_id=workflow_id,
            eligible_mask=eligible_mask,
            execution_prefix_length=execution_prefix_length,
        )

    def step(self) -> SkillResult:
        """Advance non-blocking execution by one due runner cycle."""
        return self._runtime.step()

    def run(
        self,
        *calls: SemanticCallSpec | Iterable[SemanticCallSpec],
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
        max_steps: int = 100_000,
    ) -> SkillResult:
        """Synchronously execute calls without exposing core runtime objects."""
        return self._runtime.run(
            *calls,
            workflow_id=workflow_id,
            eligible_mask=eligible_mask,
            execution_prefix_length=execution_prefix_length,
            max_steps=max_steps,
        )

    def cancel(
        self, reason: str = "Semantic workflow cancelled by caller."
    ) -> SkillResult:
        """Cancel and safe-stop the active semantic workflow."""
        return self._runtime.cancel(reason)


__all__ = [
    "AtomicSkills",
    "EffectEvidenceCollectorPort",
    "ResolvedCorePolicyTrace",
    "SkillCallTrace",
    "SkillEndpointBindingTrace",
    "SkillEffectTrace",
    "SkillFailure",
    "SkillPlanAttemptTrace",
    "SkillResult",
    "SkillRuntime",
    "SkillRuntimeProvider",
    "SkillScene",
    "SkillStatus",
    "task_state_to_metadata",
]
