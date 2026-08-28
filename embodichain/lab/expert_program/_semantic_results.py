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

"""Immutable results and trace metadata produced by Expert Program execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import math
from types import MappingProxyType

import torch

from embodichain.lab.sim.atomic_actions.bindings import EndpointBinding
from embodichain.lab.sim.atomic_actions.execution import (
    ExecutionEvent,
    ExecutionPlanAttempt,
)
from embodichain.lab.sim.atomic_actions.plans import PlanningFailure, TrajectorySegment
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy, RecoveryPolicy
from embodichain.lab.sim.atomic_actions.runner import RunnerStatus
from embodichain.lab.sim.atomic_actions.state import TaskState
from embodichain.lab.sim.atomic_actions.tracking import (
    FeedbackTerminalAcceptance,
    TimedTrackingSequence,
    TrackingMetricCfg,
    TrackingPolicy,
)
from embodichain.lab.semantic_skills.effects import (
    BinaryEffectEvidenceBatch,
    EffectEvidenceBatch,
    EffectExpectationDecision,
    JointStateEvidenceBatch,
    PoseRelationEvidenceBatch,
    ScalarEffectEvidenceBatch,
    SemanticEffectSpec,
)
from embodichain.lab.semantic_skills.integration import SemanticDiagnostic


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
        segment_name=event.segment_name,
        failure_code=event.failure_code,
        retryable=event.retryable,
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
    if isinstance(value, type):
        return {"__type__": f"{value.__module__}.{value.__qualname__}"}
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


def _freeze_metadata_value(value: object) -> object:
    """Recursively freeze already JSON-safe metadata for immutable traces."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_metadata_value(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(nested) for nested in value)
    return value


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
    metadata: dict[str, object] = {
        "kind": event.kind.value,
        "timestamp": _metadata_value(event.timestamp),
        "skill_id": event.skill_id,
        "invocation_id": event.invocation_id,
        "invocation_revision": event.invocation_revision,
        "invocation_index": event.invocation_index,
        "env_mask": _metadata_value(event.env_mask),
        "message": event.message,
    }
    if event.segment_name is not None:
        metadata["segment_name"] = event.segment_name
    if event.failure_code is not None:
        metadata["failure_code"] = event.failure_code
        metadata["retryable"] = event.retryable
    return metadata


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


class SemanticExecutionStatus(str, Enum):
    """Lifecycle state of one semantic workflow run."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SkillWorkflowRecoveryRole(str, Enum):
    """Role of one real semantic call inside workflow recovery."""

    RETRY_RETAINED = "retry_retained"
    REACQUIRE = "reacquire"
    RETRY_REACQUIRED = "retry_reacquired"


@dataclass(frozen=True, slots=True)
class SkillEndpointTrackingChannelTrace:
    """Stable provider and projector route for one endpoint feedback channel."""

    channel_id: str
    provider_id: str
    provider_revision: str
    projector_id: str
    projector_revision: str
    feedback_address_type: str
    address_fingerprint: object
    route_fingerprint: object

    def __post_init__(self) -> None:
        for name in (
            "channel_id",
            "provider_id",
            "provider_revision",
            "projector_id",
            "projector_revision",
            "feedback_address_type",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        object.__setattr__(
            self,
            "address_fingerprint",
            _freeze_metadata_value(_metadata_value(self.address_fingerprint)),
        )
        object.__setattr__(
            self,
            "route_fingerprint",
            _freeze_metadata_value(_metadata_value(self.route_fingerprint)),
        )

    def to_metadata(self) -> dict[str, object]:
        """Return the exact immutable tracking route without live objects."""
        return {
            "channel_id": self.channel_id,
            "feedback_source": {
                "provider_id": self.provider_id,
                "revision": self.provider_revision,
                "address_type": self.feedback_address_type,
                "address_fingerprint": _metadata_value(self.address_fingerprint),
            },
            "projector": {
                "projector_id": self.projector_id,
                "revision": self.projector_revision,
            },
            "route_fingerprint": _metadata_value(self.route_fingerprint),
        }


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
    tracking_channels: tuple[SkillEndpointTrackingChannelTrace, ...]
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
        tracking_channels = tuple(self.tracking_channels)
        if not all(
            type(value) is SkillEndpointTrackingChannelTrace
            for value in tracking_channels
        ):
            raise TypeError(
                "tracking_channels must contain exact "
                "SkillEndpointTrackingChannelTrace values."
            )
        channel_ids = tuple(value.channel_id for value in tracking_channels)
        if tuple(sorted(set(channel_ids))) != channel_ids:
            raise ValueError(
                "tracking_channels must use sorted unique channel identifiers."
            )
        object.__setattr__(self, "tracking_channels", tracking_channels)
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
            tracking_channels=tuple(
                SkillEndpointTrackingChannelTrace(
                    channel_id=channel_id,
                    provider_id=channel.source.provider_id,
                    provider_revision=channel.source.revision,
                    projector_id=channel.projector.projector_id,
                    projector_revision=channel.projector.revision,
                    feedback_address_type=(
                        f"{type(channel.source.address).__module__}."
                        f"{type(channel.source.address).__qualname__}"
                    ),
                    address_fingerprint=(channel.source.address.address_fingerprint),
                    route_fingerprint=channel.route_fingerprint,
                )
                for channel_id, channel in sorted(binding.tracking_channels.items())
            ),
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
            "tracking_channels": [
                channel.to_metadata() for channel in self.tracking_channels
            ],
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
        "goal_translation_threshold": _metadata_value(
            policy.goal_translation_threshold
        ),
        "goal_rotation_threshold": _metadata_value(policy.goal_rotation_threshold),
        "action_timeout": _metadata_value(policy.action_timeout),
    }


def _tracking_metric_to_metadata(metric: TrackingMetricCfg) -> dict[str, object]:
    """Serialize one exact typed metric and its unit-preserving tolerances."""
    parameters = (
        {
            value.name: _metadata_value(getattr(metric, value.name))
            for value in fields(metric)
        }
        if is_dataclass(metric)
        else {}
    )
    return {
        "metric_id": metric.metric_id,
        "revision": metric.revision,
        "channel_id": metric.channel_id,
        "type": f"{type(metric).__module__}.{type(metric).__qualname__}",
        "parameters": parameters,
    }


def _tracking_policy_to_metadata(policy: TrackingPolicy) -> dict[str, object]:
    """Serialize independent in-flight and terminal tracking contracts."""
    in_flight = policy.in_flight
    terminal = policy.terminal
    return {
        "in_flight": (
            None
            if in_flight is None
            else {
                "metrics": [
                    _tracking_metric_to_metadata(metric) for metric in in_flight.metrics
                ],
                "consecutive_violations": in_flight.consecutive_violations,
                "grace_period": _metadata_value(in_flight.grace_period),
            }
        ),
        "terminal": (
            {
                "mode": "feedback",
                "metrics": [
                    _tracking_metric_to_metadata(metric) for metric in terminal.metrics
                ],
                "settle_timeout": _metadata_value(terminal.settle_timeout),
                "consecutive_acceptances": terminal.consecutive_acceptances,
            }
            if isinstance(terminal, FeedbackTerminalAcceptance)
            else {
                "mode": "timed",
                "settle_duration": _metadata_value(terminal.settle_duration),
            }
        ),
    }


def _tracking_sequence_to_metadata(
    sequence: TimedTrackingSequence | None,
) -> dict[str, object] | None:
    """Serialize the provider/projector shape of one plan-owned contract."""
    if sequence is None:
        return None
    first_frame = None if not sequence.frames else sequence.frames[0]
    return {
        "env_ids": _metadata_value(sequence.env_ids),
        "frame_count": sequence.frame_count,
        "setpoints": [
            {
                "endpoint": list(setpoint.endpoint_key),
                "channel_id": setpoint.binding.channel_id,
                "state_type": (
                    f"{type(setpoint.desired).__module__}."
                    f"{type(setpoint.desired).__qualname__}"
                ),
                "feedback_source": {
                    "provider_id": setpoint.binding.source.provider_id,
                    "revision": setpoint.binding.source.revision,
                    "address_fingerprint": _metadata_value(
                        setpoint.binding.source.address.address_fingerprint
                    ),
                },
                "projector": {
                    "projector_id": setpoint.binding.projector.projector_id,
                    "revision": setpoint.binding.projector.revision,
                },
                "route_fingerprint": _metadata_value(
                    setpoint.binding.route_fingerprint
                ),
            }
            for setpoint in (() if first_frame is None else first_frame.setpoints)
        ],
    }


@dataclass(frozen=True, slots=True)
class ResolvedCorePolicyTrace:
    """Resolved preset, core policies, and execution binding for one plan."""

    profile_id: str
    preset_id: str
    motion_policy: MotionPolicy
    tracking_policy: TrackingPolicy
    recovery_policy: RecoveryPolicy
    endpoints: tuple[SkillEndpointBindingTrace, ...]

    def __post_init__(self) -> None:
        for name in ("profile_id", "preset_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.motion_policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(self.tracking_policy, TrackingPolicy):
            raise TypeError("tracking_policy must be a TrackingPolicy.")
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
        object.__setattr__(self, "tracking_policy", self.tracking_policy.snapshot())
        object.__setattr__(self, "recovery_policy", replace(self.recovery_policy))
        object.__setattr__(self, "endpoints", endpoints)

    @classmethod
    def from_resolved_binding(
        cls,
        *,
        profile_id: str,
        preset_id: str,
        motion_policy: MotionPolicy,
        tracking_policy: TrackingPolicy,
        recovery_policy: RecoveryPolicy,
        endpoints: Iterable[EndpointBinding],
    ) -> ResolvedCorePolicyTrace:
        """Project one resolved preset and action binding to a trace."""
        return cls(
            profile_id=profile_id,
            preset_id=preset_id,
            motion_policy=motion_policy,
            tracking_policy=tracking_policy,
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
            motion_policy=self.motion_policy,
            tracking_policy=self.tracking_policy,
            recovery_policy=self.recovery_policy,
            endpoints=self.endpoints,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return deterministic policy and endpoint-binding metadata."""
        return {
            "profile_id": self.profile_id,
            "preset": {"preset_id": self.preset_id},
            "motion_policy": _motion_policy_to_metadata(self.motion_policy),
            "tracking_policy": _tracking_policy_to_metadata(self.tracking_policy),
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
    tracking_policy: TrackingPolicy
    tracking: TimedTrackingSequence | None
    effect_verification_kind: str | None
    resolved_core_policy: ResolvedCorePolicyTrace
    planner_backend: str
    planner_messages: tuple[str, ...]
    planner_metadata: Mapping[str, object]
    planner_failure: PlanningFailure | None

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
        if not isinstance(self.tracking_policy, TrackingPolicy):
            raise TypeError("tracking_policy must be a TrackingPolicy.")
        if self.tracking is not None and not isinstance(
            self.tracking, TimedTrackingSequence
        ):
            raise TypeError("tracking must be a TimedTrackingSequence or None.")
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
        if (
            self.planner_failure is not None
            and type(self.planner_failure) is not PlanningFailure
        ):
            raise TypeError("planner_failure must be PlanningFailure or None.")
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
        object.__setattr__(self, "tracking_policy", self.tracking_policy.snapshot())
        if self.tracking is not None:
            object.__setattr__(self, "tracking", self.tracking.snapshot())
        object.__setattr__(
            self,
            "resolved_core_policy",
            self.resolved_core_policy.snapshot(),
        )
        object.__setattr__(self, "planner_messages", messages)
        if self.planner_failure is not None:
            object.__setattr__(
                self,
                "planner_failure",
                PlanningFailure(
                    self.planner_failure.code,
                    self.planner_failure.retryable,
                ),
            )
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
            tracking_policy=plan.tracking_policy,
            tracking=plan.tracking,
            effect_verification_kind=(
                None
                if plan.effect_verification is None
                else plan.effect_verification.kind
            ),
            resolved_core_policy=ResolvedCorePolicyTrace.from_resolved_binding(
                profile_id=profile_id,
                preset_id=preset_id,
                motion_policy=request.motion_policy,
                tracking_policy=request.tracking_policy,
                recovery_policy=request.recovery_policy,
                endpoints=request.binding.endpoints,
            ),
            planner_backend=plan.diagnostics.backend,
            planner_messages=plan.diagnostics.messages,
            planner_metadata=plan.diagnostics.metadata,
            planner_failure=plan.diagnostics.failure,
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
            tracking_policy=self.tracking_policy,
            tracking=self.tracking,
            effect_verification_kind=self.effect_verification_kind,
            resolved_core_policy=self.resolved_core_policy,
            planner_backend=self.planner_backend,
            planner_messages=self.planner_messages,
            planner_metadata=self.planner_metadata,
            planner_failure=self.planner_failure,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return one plan generation as deterministic JSON-safe data."""
        planner_diagnostics: dict[str, object] = {
            "backend": self.planner_backend,
            "messages": list(self.planner_messages),
            "metadata": _metadata_value(self.planner_metadata),
        }
        if self.planner_failure is not None:
            planner_diagnostics["failure"] = {
                "code": self.planner_failure.code,
                "retryable": self.planner_failure.retryable,
            }
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
            "tracking_policy": _tracking_policy_to_metadata(self.tracking_policy),
            "tracking_contract": _tracking_sequence_to_metadata(self.tracking),
            "effect_verification_kind": self.effect_verification_kind,
            "resolved_core_policy": self.resolved_core_policy.to_metadata(),
            "planner_diagnostics": planner_diagnostics,
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
    expectation_decisions: tuple[EffectExpectationDecision, ...]
    effect_spec: SemanticEffectSpec
    monitor_id: str
    monitor_revision: str | None
    configured_monitor_params: Mapping[str, object]
    resolved_monitor_params: Mapping[str, object]
    evidence: Mapping[str, EffectEvidenceBatch]
    boundary_kind: str = "terminal"
    guard_id: str | None = None
    gate_id: str | None = None
    segment_name: str | None = None

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer.")
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if type(self.observation_revision) is not int or self.observation_revision < 0:
            raise ValueError("observation_revision must be non-negative.")
        if self.boundary_kind not in {
            "terminal",
            "in_flight_guard",
            "phase_effect_gate",
        }:
            raise ValueError(
                "boundary_kind must be 'terminal', 'in_flight_guard', or "
                "'phase_effect_gate'."
            )
        for name in ("guard_id", "gate_id", "segment_name"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"{name} must be a non-empty string or None.")
        if self.boundary_kind == "terminal":
            if (
                self.guard_id is not None
                or self.gate_id is not None
                or self.segment_name is not None
            ):
                raise ValueError(
                    "Terminal effect traces cannot declare segment-boundary metadata."
                )
        elif self.boundary_kind == "in_flight_guard" and (
            self.guard_id is None
            or self.gate_id is not None
            or self.segment_name is None
        ):
            raise ValueError(
                "In-flight guard traces require only guard_id and segment_name."
            )
        elif self.boundary_kind == "phase_effect_gate" and (
            self.gate_id is None
            or self.guard_id is not None
            or self.segment_name is None
        ):
            raise ValueError(
                "Phase-effect gate traces require only gate_id and segment_name."
            )
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
        expectation_decisions = tuple(self.expectation_decisions)
        if not all(
            type(value) is EffectExpectationDecision for value in expectation_decisions
        ):
            raise TypeError(
                "expectation_decisions must contain exact "
                "EffectExpectationDecision values."
            )
        for value in expectation_decisions:
            if value.satisfied_mask.shape != self.success_mask.shape:
                raise ValueError(
                    "Expectation and aggregate trace masks must have equal shapes."
                )
            if value.satisfied_mask.device != self.success_mask.device:
                raise ValueError(
                    "Expectation and aggregate trace masks must share a device."
                )
        physical_ids = tuple(
            expectation.expectation_id
            for expectation in self.effect_spec.state_expectations
            if any(
                clause.expectation_id == expectation.expectation_id
                for clause in self.effect_spec.clauses
            )
        )
        outcome_ids = tuple(value.expectation_id for value in expectation_decisions)
        if outcome_ids != physical_ids:
            raise ValueError(
                "Effect trace must contain one ordered outcome for every "
                f"physical expectation; expected={physical_ids}, "
                f"got={outcome_ids}."
            )
        if expectation_decisions:
            expected_success = torch.ones_like(self.success_mask)
            expected_failure = torch.zeros_like(self.failure_mask)
            for value in expectation_decisions:
                expected_success &= value.satisfied_mask
                expected_failure |= value.contradicted_mask
            if not torch.equal(self.success_mask, expected_success):
                raise ValueError(
                    "success_mask must equal the conjunction of expectation "
                    "trace outcomes."
                )
            if not torch.equal(self.failure_mask, expected_failure):
                raise ValueError(
                    "failure_mask must equal the union of expectation trace "
                    "outcomes."
                )
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
        object.__setattr__(
            self,
            "expectation_decisions",
            tuple(value.snapshot() for value in expectation_decisions),
        )
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
            expectation_decisions=self.expectation_decisions,
            effect_spec=self.effect_spec,
            monitor_id=self.monitor_id,
            monitor_revision=self.monitor_revision,
            configured_monitor_params=self.configured_monitor_params,
            resolved_monitor_params=self.resolved_monitor_params,
            evidence=self.evidence,
            boundary_kind=self.boundary_kind,
            guard_id=self.guard_id,
            gate_id=self.gate_id,
            segment_name=self.segment_name,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return monitor contract, evidence, thresholds, and decision metadata."""
        metadata = {
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
                "expectations": [
                    {
                        "expectation_id": value.expectation_id,
                        "satisfied_mask": _metadata_value(value.satisfied_mask),
                        "contradicted_mask": _metadata_value(value.contradicted_mask),
                        "inverse_satisfied_mask": _metadata_value(
                            value.inverse_satisfied_mask
                        ),
                    }
                    for value in self.expectation_decisions
                ],
            },
        }
        metadata["boundary"] = {"kind": self.boundary_kind}
        if self.boundary_kind == "in_flight_guard":
            metadata["boundary"].update(
                {
                    "guard_id": self.guard_id,
                    "segment_name": self.segment_name,
                }
            )
        elif self.boundary_kind == "phase_effect_gate":
            metadata["boundary"].update(
                {
                    "gate_id": self.gate_id,
                    "segment_name": self.segment_name,
                }
            )
        return metadata


@dataclass(frozen=True, slots=True, eq=False)
class SkillFailure:
    """Per-environment semantic workflow failure."""

    call_index: int
    semantic_id: str
    env_mask: torch.Tensor
    message: str
    code: str = "semantic_call_failed"
    phase: str = "execution"
    diagnostic: SemanticDiagnostic | None = None

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
        if (
            type(self.code) is not str
            or not self.code
            or self.code != self.code.strip()
        ):
            raise ValueError(
                "code must be a non-empty string without outer whitespace."
            )
        if type(self.phase) is not str:
            raise TypeError("phase must be a string.")
        if self.phase not in {"preparation", "execution", "runtime"}:
            raise ValueError(
                "phase must be one of 'preparation', 'execution', or 'runtime'."
            )
        if self.diagnostic is not None and not isinstance(
            self.diagnostic,
            SemanticDiagnostic,
        ):
            raise TypeError("diagnostic must be a SemanticDiagnostic or None.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())

    def snapshot(self) -> SkillFailure:
        """Return an independently owned failure."""
        return SkillFailure(
            call_index=self.call_index,
            semantic_id=self.semantic_id,
            env_mask=self.env_mask,
            message=self.message,
            code=self.code,
            phase=self.phase,
            diagnostic=self.diagnostic,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return one row-local failure as JSON-safe data."""
        return {
            "call_index": self.call_index,
            "semantic_id": self.semantic_id,
            "env_mask": _metadata_value(self.env_mask),
            "message": self.message,
            "code": self.code,
            "phase": self.phase,
            "diagnostic": (
                None
                if self.diagnostic is None
                else {
                    "code": self.diagnostic.code,
                    "path": _metadata_value(self.diagnostic.path),
                    "rendered_path": self.diagnostic.rendered_path,
                    "message": self.diagnostic.message,
                    "candidates": _metadata_value(self.diagnostic.candidates),
                }
            ),
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
class SkillWorkflowRecoveryTrace:
    """One auditable real semantic call within bounded workflow recovery.

    Args:
        recovery_id: Monotonic runtime-local trace identifier.
        trigger_call_index: Original workflow call held at the shared barrier.
        trigger_semantic_id: Semantic ID of the original failed call.
        attempt_index: One-based per-row recovery-cycle index.
        max_recovery_attempts: Configured per-row recovery-cycle budget.
        role: Whether this call retries, re-acquires, or retries after pickup.
        source_resource_id: Resolved robot resource that owns the source object.
        source_task_state_key: Verified held-object state key for that resource.
        entered_mask: Rows that entered this real recovery call.
        completed_mask: Entered rows that completed this call.
        failed_mask: Entered rows that failed this call.
        call: Nested semantic-call trace, or ``None`` when preparation failed.
        message: Optional terminal or preparation diagnostic.
    """

    recovery_id: int
    trigger_call_index: int
    trigger_semantic_id: str
    attempt_index: int
    max_recovery_attempts: int
    role: SkillWorkflowRecoveryRole
    source_resource_id: str
    source_task_state_key: str
    entered_mask: torch.Tensor
    completed_mask: torch.Tensor
    failed_mask: torch.Tensor
    call: SkillCallTrace | None
    message: str | None = None

    def __post_init__(self) -> None:
        if type(self.recovery_id) is not int or self.recovery_id < 0:
            raise ValueError("recovery_id must be a non-negative integer.")
        if type(self.trigger_call_index) is not int or self.trigger_call_index < 0:
            raise ValueError("trigger_call_index must be non-negative.")
        for name in (
            "trigger_semantic_id",
            "source_resource_id",
            "source_task_state_key",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if type(self.attempt_index) is not int or self.attempt_index <= 0:
            raise ValueError("attempt_index must be a positive integer.")
        if (
            type(self.max_recovery_attempts) is not int
            or self.max_recovery_attempts <= 0
            or self.attempt_index > self.max_recovery_attempts
        ):
            raise ValueError(
                "max_recovery_attempts must cover the positive attempt_index."
            )
        if not isinstance(self.role, SkillWorkflowRecoveryRole):
            raise TypeError("role must be a SkillWorkflowRecoveryRole.")
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
        ) or not (
            self.entered_mask.device
            == self.completed_mask.device
            == self.failed_mask.device
        ):
            raise ValueError("Workflow-recovery masks must share shape and device.")
        if (self.completed_mask & self.failed_mask).any():
            raise ValueError("Recovery completion and failure masks cannot overlap.")
        if ((self.completed_mask | self.failed_mask) & ~self.entered_mask).any():
            raise ValueError("Recovery outcomes must be subsets of entered_mask.")
        if self.call is not None:
            if type(self.call) is not SkillCallTrace:
                raise TypeError("call must be a SkillCallTrace or None.")
            if not torch.equal(self.call.entered_mask, self.entered_mask):
                raise ValueError("Recovery call entered_mask must match the trace.")
            if not torch.equal(self.call.completed_mask, self.completed_mask):
                raise ValueError("Recovery call completed_mask must match the trace.")
            if not torch.equal(self.call.failed_mask, self.failed_mask):
                raise ValueError("Recovery call failed_mask must match the trace.")
        elif not torch.equal(self.failed_mask, self.entered_mask):
            raise ValueError(
                "A recovery preparation failure must fail every entered row."
            )
        if self.message is not None and (
            type(self.message) is not str or not self.message
        ):
            raise ValueError("message must be a non-empty string or None.")
        for name in ("entered_mask", "completed_mask", "failed_mask"):
            object.__setattr__(self, name, getattr(self, name).clone())
        if self.call is not None:
            object.__setattr__(self, "call", self.call.snapshot())

    def snapshot(self) -> SkillWorkflowRecoveryTrace:
        """Return an independently owned workflow-recovery trace."""
        return SkillWorkflowRecoveryTrace(
            recovery_id=self.recovery_id,
            trigger_call_index=self.trigger_call_index,
            trigger_semantic_id=self.trigger_semantic_id,
            attempt_index=self.attempt_index,
            max_recovery_attempts=self.max_recovery_attempts,
            role=self.role,
            source_resource_id=self.source_resource_id,
            source_task_state_key=self.source_task_state_key,
            entered_mask=self.entered_mask,
            completed_mask=self.completed_mask,
            failed_mask=self.failed_mask,
            call=self.call,
            message=self.message,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return deterministic JSON-safe workflow-recovery metadata."""
        return {
            "recovery_id": self.recovery_id,
            "trigger_call_index": self.trigger_call_index,
            "trigger_semantic_id": self.trigger_semantic_id,
            "attempt_index": self.attempt_index,
            "max_recovery_attempts": self.max_recovery_attempts,
            "role": self.role.value,
            "source_resource_id": self.source_resource_id,
            "source_task_state_key": self.source_task_state_key,
            "masks": {
                "entered": _metadata_value(self.entered_mask),
                "completed": _metadata_value(self.completed_mask),
                "failed": _metadata_value(self.failed_mask),
            },
            "call": None if self.call is None else self.call.to_metadata(),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True, eq=False)
class SemanticExecutionResult:
    """Immutable workflow snapshot returned by sync and step-wise execution."""

    status: SemanticExecutionStatus
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
    workflow_recoveries: tuple[SkillWorkflowRecoveryTrace, ...] = ()
    failures: tuple[SkillFailure, ...] = ()
    wait_duration: float = 0.0
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SemanticExecutionStatus):
            raise TypeError("status must be a SemanticExecutionStatus.")
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
        if not all(
            type(recovery) is SkillWorkflowRecoveryTrace
            for recovery in self.workflow_recoveries
        ):
            raise TypeError(
                "workflow_recoveries must contain SkillWorkflowRecoveryTrace values."
            )
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
            "workflow_recoveries",
            tuple(recovery.snapshot() for recovery in self.workflow_recoveries),
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
            SemanticExecutionStatus.COMPLETED,
            SemanticExecutionStatus.FAILED,
            SemanticExecutionStatus.CANCELLED,
        }

    def to_metadata(self) -> dict[str, object]:
        """Return a fresh deterministic JSON-safe workflow result.

        Core recovery remains represented by the ordered
        :class:`ExecutionEvent` stream and each call's plan-attempt history.
        Workflow re-acquisition additionally appears in
        ``workflow_recoveries``. The returned object owns only Python scalars,
        lists, and dictionaries and can be serialized with
        ``json.dumps(..., allow_nan=False)``.
        """
        return {
            "schema_version": 2,
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
            "workflow_recoveries": [
                recovery.to_metadata() for recovery in self.workflow_recoveries
            ],
            "failures": [failure.to_metadata() for failure in self.failures],
            "wait_duration": self.wait_duration,
            "message": self.message,
        }

    def require_all_succeeded(self) -> None:
        """Raise unless every environment completed the workflow successfully.

        This convenience assertion keeps application entry points concise while
        the canonical result continues to expose row-local masks.
        """
        if self.status is not SemanticExecutionStatus.COMPLETED or not bool(
            self.success_mask.all()
        ):
            raise RuntimeError(
                "Semantic workflow did not succeed for every environment: "
                f"status={self.status.value}, "
                f"failed={self.failure_mask.detach().cpu().tolist()}, "
                f"cancelled={self.cancelled_mask.detach().cpu().tolist()}."
            )


__all__: list[str] = []
