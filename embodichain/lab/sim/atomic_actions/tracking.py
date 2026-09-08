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

"""Typed, transport-neutral tracking contracts for atomic-action execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Hashable, Iterable, Mapping, Protocol

import torch

if TYPE_CHECKING:
    from .bindings import RuntimeEndpointTarget
    from .runtime_commands import EndpointCommand
    from .state import PlanningContext


TrackingChannelId = str
"""Open string identifier for one typed endpoint-feedback channel."""

JOINT_POSITION_CHANNEL: TrackingChannelId = "joint.position"
BASE_POSE_CHANNEL: TrackingChannelId = "base.pose"
WHOLE_BODY_POSE_CHANNEL: TrackingChannelId = "whole_body.pose"


def _identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _positive_float(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not torch.isfinite(torch.tensor(normalized)).item() or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


def _non_negative_float(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not torch.isfinite(torch.tensor(normalized)).item() or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _tensor(value: torch.Tensor, *, field_name: str, dimensions: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if value.dim() != dimensions or any(size < 1 for size in value.shape):
        raise ValueError(f"{field_name} must be a non-empty {dimensions}-D tensor.")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
        raise ValueError(f"{field_name} must contain finite floating-point values.")
    return value.clone()


class TrackingFeedbackAddress(ABC):
    """Immutable address understood by one tracking-feedback provider."""

    @property
    @abstractmethod
    def address_fingerprint(self) -> Hashable:
        """Return a stable, hashable address identity."""

    def snapshot(self) -> TrackingFeedbackAddress:
        """Return an independently owned address snapshot."""
        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class EndpointTrackingFeedbackAddress(TrackingFeedbackAddress):
    """Feedback address for one runtime endpoint and open tracking channel."""

    target: RuntimeEndpointTarget
    channel_id: TrackingChannelId

    def __post_init__(self) -> None:
        from .bindings import RuntimeEndpointTarget

        if not isinstance(self.target, RuntimeEndpointTarget):
            raise TypeError("target must be a RuntimeEndpointTarget.")
        snapshot = self.target.snapshot()
        if type(snapshot) is not type(self.target) or snapshot is self.target:
            raise TypeError("RuntimeEndpointTarget.snapshot() must own a new value.")
        if snapshot.address_fingerprint != self.target.address_fingerprint:
            raise ValueError("Target snapshot must preserve its address fingerprint.")
        _identifier(self.channel_id, field_name="channel_id")
        object.__setattr__(self, "target", snapshot)

    @property
    def address_fingerprint(self) -> Hashable:
        """Return the endpoint- and channel-scoped address identity."""
        return self.target.address_fingerprint, self.channel_id


@dataclass(frozen=True, slots=True)
class TrackingFeedbackSourceRef:
    """Versioned provider route plus one immutable feedback address."""

    provider_id: str
    revision: str
    address: TrackingFeedbackAddress

    def __post_init__(self) -> None:
        _identifier(self.provider_id, field_name="provider_id")
        _identifier(self.revision, field_name="revision")
        if not isinstance(self.address, TrackingFeedbackAddress):
            raise TypeError("address must be a TrackingFeedbackAddress.")
        snapshot = self.address.snapshot()
        if type(snapshot) is not type(self.address) or snapshot is self.address:
            raise TypeError("TrackingFeedbackAddress.snapshot() must own a new value.")
        if snapshot.address_fingerprint != self.address.address_fingerprint:
            raise ValueError("Address snapshot must preserve its fingerprint.")
        hash(snapshot.address_fingerprint)
        object.__setattr__(self, "address", snapshot)

    @property
    def source_fingerprint(self) -> Hashable:
        """Return the exact versioned source identity."""
        return self.provider_id, self.revision, self.address.address_fingerprint

    def snapshot(self) -> TrackingFeedbackSourceRef:
        """Return an independently owned source reference."""
        return TrackingFeedbackSourceRef(self.provider_id, self.revision, self.address)


@dataclass(frozen=True, slots=True)
class TrackingProjectorRef:
    """Exact version of a command-to-tracking-state projector."""

    projector_id: str
    revision: str

    def __post_init__(self) -> None:
        _identifier(self.projector_id, field_name="projector_id")
        _identifier(self.revision, field_name="revision")

    def snapshot(self) -> TrackingProjectorRef:
        """Return an independently owned projector route."""
        return TrackingProjectorRef(self.projector_id, self.revision)


@dataclass(frozen=True, slots=True)
class EndpointTrackingChannelBinding:
    """Resolved source and projector for one endpoint tracking channel."""

    channel_id: TrackingChannelId
    source: TrackingFeedbackSourceRef
    projector: TrackingProjectorRef

    def __post_init__(self) -> None:
        _identifier(self.channel_id, field_name="channel_id")
        if not isinstance(self.source, TrackingFeedbackSourceRef):
            raise TypeError("source must be a TrackingFeedbackSourceRef.")
        if not isinstance(self.projector, TrackingProjectorRef):
            raise TypeError("projector must be a TrackingProjectorRef.")
        address = self.source.address
        if isinstance(address, EndpointTrackingFeedbackAddress):
            if address.channel_id != self.channel_id:
                raise ValueError("Binding and feedback-address channels must match.")
        object.__setattr__(self, "source", self.source.snapshot())
        object.__setattr__(self, "projector", self.projector.snapshot())

    def snapshot(self) -> EndpointTrackingChannelBinding:
        """Return an independently owned channel binding."""
        return EndpointTrackingChannelBinding(
            self.channel_id, self.source, self.projector
        )

    @property
    def route_fingerprint(self) -> tuple[str, Hashable, str, str]:
        """Return the exact channel, source, and projector route identity."""
        return (
            self.channel_id,
            self.source.source_fingerprint,
            self.projector.projector_id,
            self.projector.revision,
        )


class TrackingState(ABC):
    """Immutable-by-ownership typed desired or observed tracking state."""

    channel_id: ClassVar[TrackingChannelId]

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return the represented environment count."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the tensor device."""

    @abstractmethod
    def snapshot(self) -> TrackingState:
        """Return an independently owned state snapshot."""


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionTrackingState(TrackingState):
    """Batched joint positions with shape ``(B, D)``."""

    channel_id: ClassVar[str] = JOINT_POSITION_CHANNEL
    positions: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "positions",
            _tensor(self.positions, field_name="positions", dimensions=2),
        )

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def snapshot(self) -> JointPositionTrackingState:
        return JointPositionTrackingState(self.positions)


@dataclass(frozen=True, slots=True, eq=False)
class PoseTrackingState(TrackingState):
    """Batched homogeneous poses with shape ``(B, 4, 4)``."""

    channel_id: ClassVar[str] = BASE_POSE_CHANNEL
    poses: torch.Tensor

    def __post_init__(self) -> None:
        poses = _tensor(self.poses, field_name="poses", dimensions=3)
        if poses.shape[1:] != (4, 4):
            raise ValueError("poses must have shape (batch_size, 4, 4).")
        object.__setattr__(self, "poses", poses)

    @property
    def batch_size(self) -> int:
        return int(self.poses.shape[0])

    @property
    def device(self) -> torch.device:
        return self.poses.device

    def snapshot(self) -> PoseTrackingState:
        return PoseTrackingState(self.poses)


@dataclass(frozen=True, slots=True, eq=False)
class WholeBodyPoseTrackingState(TrackingState):
    """Batched base poses and joint positions for whole-body tracking."""

    channel_id: ClassVar[str] = WHOLE_BODY_POSE_CHANNEL
    root_poses: torch.Tensor
    joint_positions: torch.Tensor

    def __post_init__(self) -> None:
        root_poses = _tensor(self.root_poses, field_name="root_poses", dimensions=3)
        joints = _tensor(
            self.joint_positions,
            field_name="joint_positions",
            dimensions=2,
        )
        if root_poses.shape[1:] != (4, 4):
            raise ValueError("root_poses must have shape (batch_size, 4, 4).")
        if root_poses.shape[0] != joints.shape[0]:
            raise ValueError("root_poses and joint_positions batches must match.")
        if root_poses.device != joints.device:
            raise ValueError("root_poses and joint_positions must share a device.")
        object.__setattr__(self, "root_poses", root_poses)
        object.__setattr__(self, "joint_positions", joints)

    @property
    def batch_size(self) -> int:
        return int(self.root_poses.shape[0])

    @property
    def device(self) -> torch.device:
        return self.root_poses.device

    def snapshot(self) -> WholeBodyPoseTrackingState:
        return WholeBodyPoseTrackingState(self.root_poses, self.joint_positions)


class TrackingMetricCfg(ABC):
    """Immutable tolerance configuration dispatched by exact metric ID/revision."""

    metric_id: ClassVar[str]
    revision: ClassVar[str] = "1"
    channel_id: ClassVar[TrackingChannelId]

    def snapshot(self) -> TrackingMetricCfg:
        """Return an independently owned metric configuration."""
        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class JointPositionTrackingMetric(TrackingMetricCfg):
    """Maximum absolute joint-error tolerance."""

    metric_id: ClassVar[str] = "joint.max_abs"
    channel_id: ClassVar[str] = JOINT_POSITION_CHANNEL
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tolerance", _positive_float(self.tolerance, field_name="tolerance")
        )


@dataclass(frozen=True, slots=True)
class PoseTrackingMetric(TrackingMetricCfg):
    """Independent translation and rotation tolerances for base pose."""

    metric_id: ClassVar[str] = "pose.se3"
    channel_id: ClassVar[str] = BASE_POSE_CHANNEL
    translation_tolerance: float = 0.02
    rotation_tolerance: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_tolerance",
            _positive_float(
                self.translation_tolerance, field_name="translation_tolerance"
            ),
        )
        object.__setattr__(
            self,
            "rotation_tolerance",
            _positive_float(self.rotation_tolerance, field_name="rotation_tolerance"),
        )


@dataclass(frozen=True, slots=True)
class WholeBodyPoseTrackingMetric(TrackingMetricCfg):
    """Independent base-pose and joint-position tolerances."""

    metric_id: ClassVar[str] = "whole_body.pose"
    channel_id: ClassVar[str] = WHOLE_BODY_POSE_CHANNEL
    translation_tolerance: float = 0.02
    rotation_tolerance: float = 0.05
    joint_position_tolerance: float = 0.05

    def __post_init__(self) -> None:
        for field_name in (
            "translation_tolerance",
            "rotation_tolerance",
            "joint_position_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_float(getattr(self, field_name), field_name=field_name),
            )


def _own_metrics(
    metrics: Iterable[TrackingMetricCfg], *, field_name: str
) -> tuple[TrackingMetricCfg, ...]:
    snapshots: list[TrackingMetricCfg] = []
    channels: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, TrackingMetricCfg):
            raise TypeError(f"{field_name} must contain TrackingMetricCfg values.")
        _identifier(metric.metric_id, field_name=f"{field_name}.metric_id")
        _identifier(metric.revision, field_name=f"{field_name}.revision")
        _identifier(metric.channel_id, field_name=f"{field_name}.channel_id")
        if metric.channel_id in channels:
            raise ValueError(
                f"{field_name} contains duplicate channel {metric.channel_id!r}."
            )
        snapshot = metric.snapshot()
        if type(snapshot) is not type(metric) or snapshot is metric:
            raise TypeError("TrackingMetricCfg.snapshot() must own a same-type value.")
        channels.add(metric.channel_id)
        snapshots.append(snapshot)
    if not snapshots:
        raise ValueError(f"{field_name} must contain at least one metric.")
    return tuple(snapshots)


@dataclass(frozen=True, slots=True)
class InFlightTrackingPolicy:
    """Feedback checks used while a command sequence is still in flight."""

    metrics: tuple[TrackingMetricCfg, ...]
    consecutive_violations: int = 1
    grace_period: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metrics", _own_metrics(self.metrics, field_name="metrics")
        )
        if (
            not isinstance(self.consecutive_violations, int)
            or isinstance(self.consecutive_violations, bool)
            or self.consecutive_violations < 1
        ):
            raise ValueError("consecutive_violations must be a positive integer.")
        object.__setattr__(
            self,
            "grace_period",
            _non_negative_float(self.grace_period, field_name="grace_period"),
        )

    def snapshot(self) -> InFlightTrackingPolicy:
        return InFlightTrackingPolicy(
            self.metrics, self.consecutive_violations, self.grace_period
        )


@dataclass(frozen=True, slots=True)
class FeedbackTerminalAcceptance:
    """Terminal acceptance proven by typed endpoint feedback."""

    metrics: tuple[TrackingMetricCfg, ...]
    settle_timeout: float = 0.0
    consecutive_acceptances: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metrics", _own_metrics(self.metrics, field_name="metrics")
        )
        object.__setattr__(
            self,
            "settle_timeout",
            _non_negative_float(self.settle_timeout, field_name="settle_timeout"),
        )
        if (
            not isinstance(self.consecutive_acceptances, int)
            or isinstance(self.consecutive_acceptances, bool)
            or self.consecutive_acceptances < 1
        ):
            raise ValueError("consecutive_acceptances must be a positive integer.")

    def snapshot(self) -> FeedbackTerminalAcceptance:
        return FeedbackTerminalAcceptance(
            self.metrics, self.settle_timeout, self.consecutive_acceptances
        )


@dataclass(frozen=True, slots=True)
class TimedTerminalAcceptance:
    """Explicit terminal acceptance without endpoint feedback."""

    settle_duration: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "settle_duration",
            _non_negative_float(self.settle_duration, field_name="settle_duration"),
        )

    def snapshot(self) -> TimedTerminalAcceptance:
        return TimedTerminalAcceptance(self.settle_duration)


TerminalAcceptance = FeedbackTerminalAcceptance | TimedTerminalAcceptance


@dataclass(frozen=True, slots=True)
class TrackingPolicy:
    """Independent in-flight recovery signal and terminal acceptance contract."""

    in_flight: InFlightTrackingPolicy | None
    terminal: TerminalAcceptance

    def __post_init__(self) -> None:
        if self.in_flight is not None and not isinstance(
            self.in_flight, InFlightTrackingPolicy
        ):
            raise TypeError("in_flight must be InFlightTrackingPolicy or None.")
        if not isinstance(
            self.terminal, (FeedbackTerminalAcceptance, TimedTerminalAcceptance)
        ):
            raise TypeError("terminal must be a terminal-acceptance contract.")
        if self.in_flight is not None:
            object.__setattr__(self, "in_flight", self.in_flight.snapshot())
        object.__setattr__(self, "terminal", self.terminal.snapshot())
        in_flight = self.in_flight
        terminal = self.terminal
        if in_flight is not None and isinstance(terminal, FeedbackTerminalAcceptance):
            in_flight_by_channel = {
                metric.channel_id: metric for metric in in_flight.metrics
            }
            for terminal_metric in terminal.metrics:
                in_flight_metric = in_flight_by_channel.get(terminal_metric.channel_id)
                if in_flight_metric is None:
                    continue
                if (
                    in_flight_metric.metric_id != terminal_metric.metric_id
                    or in_flight_metric.revision != terminal_metric.revision
                    or type(in_flight_metric) is not type(terminal_metric)
                ):
                    raise ValueError(
                        "In-flight and terminal metrics sharing a channel must "
                        "use the same exact metric ID, revision, and type."
                    )

    def snapshot(self) -> TrackingPolicy:
        return TrackingPolicy(self.in_flight, self.terminal)

    @classmethod
    def timed(cls, *, settle_duration: float = 0.0) -> TrackingPolicy:
        """Create an explicit time-only terminal contract with no tracking."""
        return cls(
            in_flight=None,
            terminal=TimedTerminalAcceptance(settle_duration=settle_duration),
        )

    @classmethod
    def joint_position(
        cls,
        *,
        in_flight_max_abs_error: float = 0.05,
        terminal_max_abs_error: float = 0.05,
        terminal_settle_timeout: float = 0.5,
        consecutive_violations: int = 1,
        consecutive_acceptances: int = 1,
        grace_period: float = 0.0,
    ) -> TrackingPolicy:
        """Create the built-in joint-position tracking and acceptance contract."""
        return cls(
            in_flight=InFlightTrackingPolicy(
                metrics=(JointPositionTrackingMetric(in_flight_max_abs_error),),
                consecutive_violations=consecutive_violations,
                grace_period=grace_period,
            ),
            terminal=FeedbackTerminalAcceptance(
                metrics=(JointPositionTrackingMetric(terminal_max_abs_error),),
                settle_timeout=terminal_settle_timeout,
                consecutive_acceptances=consecutive_acceptances,
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class TrackingSetpoint:
    """One endpoint-local desired state and its typed feedback route."""

    endpoint_key: tuple[str, str]
    binding: EndpointTrackingChannelBinding
    desired: TrackingState

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_key, tuple) or len(self.endpoint_key) != 2:
            raise TypeError("endpoint_key must be a (slot_id, endpoint_id) tuple.")
        _identifier(self.endpoint_key[0], field_name="endpoint_key.slot_id")
        _identifier(self.endpoint_key[1], field_name="endpoint_key.endpoint_id")
        if not isinstance(self.binding, EndpointTrackingChannelBinding):
            raise TypeError("binding must be an EndpointTrackingChannelBinding.")
        if not isinstance(self.desired, TrackingState):
            raise TypeError("desired must be a TrackingState.")
        if self.binding.channel_id != self.desired.channel_id:
            raise ValueError("Binding and desired-state channels must match.")
        desired = self.desired.snapshot()
        if type(desired) is not type(self.desired) or desired is self.desired:
            raise TypeError("TrackingState.snapshot() must own a same-type value.")
        object.__setattr__(self, "binding", self.binding.snapshot())
        object.__setattr__(self, "desired", desired)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.endpoint_key[0], self.endpoint_key[1], self.binding.channel_id

    def snapshot(self) -> TrackingSetpoint:
        return TrackingSetpoint(self.endpoint_key, self.binding, self.desired)


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    """Desired endpoint states associated with one command frame."""

    setpoints: tuple[TrackingSetpoint, ...] = ()

    def __post_init__(self) -> None:
        snapshots: list[TrackingSetpoint] = []
        keys: set[tuple[str, str, str]] = set()
        for setpoint in self.setpoints:
            if not isinstance(setpoint, TrackingSetpoint):
                raise TypeError("setpoints must contain TrackingSetpoint values.")
            if setpoint.key in keys:
                raise ValueError(f"Duplicate tracking setpoint {setpoint.key!r}.")
            keys.add(setpoint.key)
            snapshots.append(setpoint.snapshot())
        object.__setattr__(self, "setpoints", tuple(snapshots))

    def snapshot(self) -> TrackingFrame:
        return TrackingFrame(self.setpoints)


@dataclass(frozen=True, slots=True)
class TimedTrackingSequence:
    """Tracking frames aligned by index with an authoritative command sequence."""

    env_ids: torch.Tensor
    frames: tuple[TrackingFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if (
            self.env_ids.dtype != torch.long
            or self.env_ids.dim() != 1
            or self.env_ids.numel() < 1
        ):
            raise ValueError("env_ids must be a non-empty one-dimensional long tensor.")
        if torch.unique(self.env_ids).numel() != self.env_ids.numel():
            raise ValueError("env_ids must be unique.")
        frames: list[TrackingFrame] = []
        for frame in self.frames:
            if not isinstance(frame, TrackingFrame):
                raise TypeError("frames must contain TrackingFrame values.")
            snapshot = frame.snapshot()
            for setpoint in snapshot.setpoints:
                if setpoint.desired.batch_size != self.env_ids.numel():
                    raise ValueError("Every setpoint batch must match env_ids.")
                if setpoint.desired.device != self.env_ids.device:
                    raise ValueError("Every setpoint and env_ids must share a device.")
            frames.append(snapshot)
        object.__setattr__(self, "env_ids", self.env_ids.clone())
        object.__setattr__(self, "frames", tuple(frames))

    @property
    def batch_size(self) -> int:
        """Return the represented environment count."""
        return int(self.env_ids.numel())

    @property
    def device(self) -> torch.device:
        """Return the sequence tensor device."""
        return self.env_ids.device

    @property
    def frame_count(self) -> int:
        """Return the number of command-aligned tracking frames."""
        return len(self.frames)

    def snapshot(self) -> TimedTrackingSequence:
        return TimedTrackingSequence(self.env_ids, self.frames)


@dataclass(frozen=True, slots=True, eq=False)
class TrackingFeedbackBatch:
    """One synchronized typed observation from an exact feedback source."""

    source: TrackingFeedbackSourceRef
    state: TrackingState
    valid_mask: torch.Tensor
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.source, TrackingFeedbackSourceRef):
            raise TypeError("source must be a TrackingFeedbackSourceRef.")
        if not isinstance(self.state, TrackingState):
            raise TypeError("state must be a TrackingState.")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != (
            self.state.batch_size,
        ):
            raise ValueError("valid_mask must have shape (batch_size,) and bool dtype.")
        if self.valid_mask.device != self.state.device:
            raise ValueError("valid_mask and state must share a device.")
        object.__setattr__(self, "source", self.source.snapshot())
        object.__setattr__(self, "state", self.state.snapshot())
        object.__setattr__(self, "valid_mask", self.valid_mask.clone())
        object.__setattr__(
            self,
            "timestamp",
            _non_negative_float(self.timestamp, field_name="timestamp"),
        )

    def snapshot(self) -> TrackingFeedbackBatch:
        return TrackingFeedbackBatch(
            self.source, self.state, self.valid_mask, self.timestamp
        )


@dataclass(frozen=True, slots=True, eq=False)
class TrackingEvaluation:
    """Per-row metric result with unit-preserving component errors."""

    channel_id: TrackingChannelId
    accepted_mask: torch.Tensor
    valid_mask: torch.Tensor
    normalized_error: torch.Tensor
    component_errors: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.channel_id, field_name="channel_id")
        expected = self.accepted_mask.shape
        if self.accepted_mask.dtype != torch.bool or self.accepted_mask.dim() != 1:
            raise ValueError("accepted_mask must be a one-dimensional bool tensor.")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != expected:
            raise ValueError("valid_mask must match accepted_mask with bool dtype.")
        if self.normalized_error.shape != expected or not torch.is_floating_point(
            self.normalized_error
        ):
            raise ValueError("normalized_error must be a floating tensor per row.")
        if not (
            self.accepted_mask.device
            == self.valid_mask.device
            == self.normalized_error.device
        ):
            raise ValueError("Evaluation tensors must share a device.")
        components: dict[str, torch.Tensor] = {}
        for name, value in self.component_errors.items():
            _identifier(name, field_name="component_errors key")
            if value.shape != expected or value.device != self.normalized_error.device:
                raise ValueError("Every component error must be a per-row tensor.")
            components[name] = value.clone()
        object.__setattr__(self, "accepted_mask", self.accepted_mask.clone())
        object.__setattr__(self, "valid_mask", self.valid_mask.clone())
        object.__setattr__(self, "normalized_error", self.normalized_error.clone())
        object.__setattr__(self, "component_errors", MappingProxyType(components))

    def snapshot(self) -> TrackingEvaluation:
        return TrackingEvaluation(
            self.channel_id,
            self.accepted_mask,
            self.valid_mask,
            self.normalized_error,
            self.component_errors,
        )


class TrackingFeedbackProvider(Protocol):
    """Versioned live port that reads one exact tracking source."""

    provider_id: str
    revision: str

    def observe(
        self, source: TrackingFeedbackSourceRef, context: PlanningContext
    ) -> TrackingFeedbackBatch:
        """Read one synchronized typed feedback batch."""


class TrackingCommandProjector(Protocol):
    """Versioned pure projector from an endpoint command to desired state."""

    projector_id: str
    revision: str

    def project(
        self, command: EndpointCommand, binding: EndpointTrackingChannelBinding
    ) -> TrackingState:
        """Project one command into the binding's desired tracking channel."""


class TrackingMetricEvaluator(Protocol):
    """Versioned evaluator for one exact metric configuration type."""

    metric_id: str
    revision: str
    metric_type: type[TrackingMetricCfg]

    def evaluate(
        self,
        desired: TrackingState,
        observed: TrackingState,
        valid_mask: torch.Tensor,
        metric: TrackingMetricCfg,
    ) -> TrackingEvaluation:
        """Evaluate a desired and observed batch row by row."""


class _ExactRegistry:
    __slots__ = ("_values", "_kind")

    def __init__(self, values: Iterable[object], *, kind: str) -> None:
        normalized: dict[tuple[str, str], object] = {}
        for value in values:
            identifier = _identifier(
                getattr(value, f"{kind}_id"), field_name=f"{kind}_id"
            )
            revision = _identifier(getattr(value, "revision"), field_name="revision")
            key = identifier, revision
            if key in normalized:
                raise ValueError(f"Duplicate {kind} registration {key!r}.")
            normalized[key] = value
        self._values = MappingProxyType(normalized)
        self._kind = kind

    @property
    def values(self) -> Mapping[tuple[str, str], object]:
        return self._values

    def _resolve(self, identifier: str, revision: str) -> object:
        key = identifier, revision
        try:
            return self._values[key]
        except KeyError as exc:
            raise KeyError(f"Unknown {self._kind} registration {key!r}.") from exc


class TrackingFeedbackProviderRegistry(_ExactRegistry):
    """Immutable exact-version feedback-provider registry."""

    def __init__(self, providers: Iterable[TrackingFeedbackProvider] = ()) -> None:
        super().__init__(providers, kind="provider")

    def resolve(self, source: TrackingFeedbackSourceRef) -> TrackingFeedbackProvider:
        return self._resolve(source.provider_id, source.revision)  # type: ignore[return-value]


class TrackingProjectorRegistry(_ExactRegistry):
    """Immutable exact-version command-projector registry."""

    def __init__(self, projectors: Iterable[TrackingCommandProjector] = ()) -> None:
        super().__init__(projectors, kind="projector")

    def resolve(self, route: TrackingProjectorRef) -> TrackingCommandProjector:
        return self._resolve(route.projector_id, route.revision)  # type: ignore[return-value]


class TrackingEvaluatorRegistry(_ExactRegistry):
    """Immutable exact-version metric-evaluator registry."""

    def __init__(self, evaluators: Iterable[TrackingMetricEvaluator] = ()) -> None:
        super().__init__(evaluators, kind="metric")

    def resolve(self, metric: TrackingMetricCfg) -> TrackingMetricEvaluator:
        evaluator = self._resolve(metric.metric_id, metric.revision)
        if type(metric) is not evaluator.metric_type:  # type: ignore[attr-defined]
            raise TypeError(
                f"Metric {metric.metric_id!r} requires "
                f"{evaluator.metric_type.__name__}."  # type: ignore[attr-defined]
            )
        return evaluator  # type: ignore[return-value]


class PlanningContextTrackingFeedbackProvider:
    """Built-in provider backed by :class:`PlanningContext.robot`."""

    provider_id = "planning_context.robot"
    revision = "1"

    def observe(
        self, source: TrackingFeedbackSourceRef, context: PlanningContext
    ) -> TrackingFeedbackBatch:
        from .bindings import JointPositionTarget
        from .state import PlanningContext

        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext.")
        address = source.address
        if not isinstance(address, EndpointTrackingFeedbackAddress):
            raise TypeError(
                "Built-in provider requires EndpointTrackingFeedbackAddress."
            )
        target = address.target
        if address.channel_id == JOINT_POSITION_CHANNEL:
            if not isinstance(target, JointPositionTarget):
                raise TypeError("joint.position requires a JointPositionTarget.")
            state: TrackingState = JointPositionTrackingState(
                context.robot.qpos[:, target.joint_ids]
            )
        elif address.channel_id == BASE_POSE_CHANNEL:
            if context.robot.root_pose is None:
                raise RuntimeError("RobotObservation.root_pose is unavailable.")
            state = PoseTrackingState(context.robot.root_pose)
        elif address.channel_id == WHOLE_BODY_POSE_CHANNEL:
            if context.robot.root_pose is None:
                raise RuntimeError("RobotObservation.root_pose is unavailable.")
            joints = (
                context.robot.qpos[:, target.joint_ids]
                if isinstance(target, JointPositionTarget)
                else context.robot.qpos
            )
            state = WholeBodyPoseTrackingState(context.robot.root_pose, joints)
        else:
            raise KeyError(
                f"Unsupported built-in tracking channel {address.channel_id!r}."
            )
        return TrackingFeedbackBatch(
            source=source,
            state=state,
            valid_mask=torch.ones(
                context.batch_size, dtype=torch.bool, device=state.device
            ),
            timestamp=context.robot.timestamp,
        )


class JointPositionTrackingProjector:
    """Built-in projector for joint-position endpoint commands."""

    projector_id = "joint_position_payload"
    revision = "1"

    def project(
        self, command: EndpointCommand, binding: EndpointTrackingChannelBinding
    ) -> JointPositionTrackingState:
        from .runtime_commands import EndpointCommand, JointPositionPayload

        if not isinstance(command, EndpointCommand):
            raise TypeError("command must be an EndpointCommand.")
        if binding.channel_id != JOINT_POSITION_CHANNEL:
            raise ValueError("Joint projector requires the joint.position channel.")
        if not isinstance(command.payload, JointPositionPayload):
            raise TypeError("Joint projector requires JointPositionPayload.")
        address = binding.source.address
        if isinstance(address, EndpointTrackingFeedbackAddress):
            if address.target.address_fingerprint != command.target.address_fingerprint:
                raise ValueError(
                    "Command and feedback binding target different endpoints."
                )
        return JointPositionTrackingState(command.payload.positions)


def _compatible(
    desired: TrackingState,
    observed: TrackingState,
    valid_mask: torch.Tensor,
    expected_type: type[TrackingState],
) -> None:
    if type(desired) is not expected_type or type(observed) is not expected_type:
        raise TypeError(f"Metric requires {expected_type.__name__} values.")
    if desired.batch_size != observed.batch_size or desired.device != observed.device:
        raise ValueError("Desired and observed batches must match.")
    if valid_mask.dtype != torch.bool or valid_mask.shape != (desired.batch_size,):
        raise ValueError("valid_mask must be a bool tensor with one value per row.")
    if valid_mask.device != desired.device:
        raise ValueError("valid_mask and states must share a device.")


def _pose_errors(
    desired: torch.Tensor, observed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    translation = torch.linalg.vector_norm(
        desired[:, :3, 3] - observed[:, :3, 3], dim=1
    )
    relative = desired[:, :3, :3].transpose(1, 2) @ observed[:, :3, :3]
    cosine = ((relative.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return translation, torch.acos(cosine)


class JointPositionTrackingEvaluator:
    """Evaluator for :class:`JointPositionTrackingMetric`."""

    metric_id = JointPositionTrackingMetric.metric_id
    revision = JointPositionTrackingMetric.revision
    metric_type = JointPositionTrackingMetric

    def evaluate(self, desired, observed, valid_mask, metric) -> TrackingEvaluation:
        _compatible(desired, observed, valid_mask, JointPositionTrackingState)
        if type(metric) is not JointPositionTrackingMetric:
            raise TypeError("metric must be JointPositionTrackingMetric.")
        if desired.positions.shape != observed.positions.shape:
            raise ValueError("Joint-position state shapes must match.")
        error = (desired.positions - observed.positions).abs().amax(dim=1)
        normalized = error / metric.tolerance
        return TrackingEvaluation(
            JOINT_POSITION_CHANNEL,
            valid_mask & (error <= metric.tolerance),
            valid_mask,
            normalized,
            {"joint_max_abs": error},
        )


class PoseTrackingEvaluator:
    """Evaluator for :class:`PoseTrackingMetric`."""

    metric_id = PoseTrackingMetric.metric_id
    revision = PoseTrackingMetric.revision
    metric_type = PoseTrackingMetric

    def evaluate(self, desired, observed, valid_mask, metric) -> TrackingEvaluation:
        _compatible(desired, observed, valid_mask, PoseTrackingState)
        if type(metric) is not PoseTrackingMetric:
            raise TypeError("metric must be PoseTrackingMetric.")
        translation, rotation = _pose_errors(desired.poses, observed.poses)
        normalized = torch.maximum(
            translation / metric.translation_tolerance,
            rotation / metric.rotation_tolerance,
        )
        return TrackingEvaluation(
            BASE_POSE_CHANNEL,
            valid_mask & (normalized <= 1.0),
            valid_mask,
            normalized,
            {"translation": translation, "rotation": rotation},
        )


class WholeBodyPoseTrackingEvaluator:
    """Evaluator for :class:`WholeBodyPoseTrackingMetric`."""

    metric_id = WholeBodyPoseTrackingMetric.metric_id
    revision = WholeBodyPoseTrackingMetric.revision
    metric_type = WholeBodyPoseTrackingMetric

    def evaluate(self, desired, observed, valid_mask, metric) -> TrackingEvaluation:
        _compatible(desired, observed, valid_mask, WholeBodyPoseTrackingState)
        if type(metric) is not WholeBodyPoseTrackingMetric:
            raise TypeError("metric must be WholeBodyPoseTrackingMetric.")
        if desired.joint_positions.shape != observed.joint_positions.shape:
            raise ValueError("Whole-body joint-position shapes must match.")
        translation, rotation = _pose_errors(desired.root_poses, observed.root_poses)
        joint = (desired.joint_positions - observed.joint_positions).abs().amax(dim=1)
        normalized = torch.maximum(
            torch.maximum(
                translation / metric.translation_tolerance,
                rotation / metric.rotation_tolerance,
            ),
            joint / metric.joint_position_tolerance,
        )
        return TrackingEvaluation(
            WHOLE_BODY_POSE_CHANNEL,
            valid_mask & (normalized <= 1.0),
            valid_mask,
            normalized,
            {"translation": translation, "rotation": rotation, "joint_max_abs": joint},
        )


class TrackingRuntime:
    """Runtime facade for projecting commands and evaluating typed feedback."""

    __slots__ = ("_providers", "_projectors", "_evaluators")

    def __init__(
        self,
        providers: TrackingFeedbackProviderRegistry,
        projectors: TrackingProjectorRegistry,
        evaluators: TrackingEvaluatorRegistry,
    ) -> None:
        if type(providers) is not TrackingFeedbackProviderRegistry:
            raise TypeError(
                "providers must be exactly TrackingFeedbackProviderRegistry."
            )
        if type(projectors) is not TrackingProjectorRegistry:
            raise TypeError("projectors must be exactly TrackingProjectorRegistry.")
        if type(evaluators) is not TrackingEvaluatorRegistry:
            raise TypeError("evaluators must be exactly TrackingEvaluatorRegistry.")
        self._providers = providers
        self._projectors = projectors
        self._evaluators = evaluators

    @property
    def providers(self) -> TrackingFeedbackProviderRegistry:
        """Return the immutable exact-version provider registry."""
        return self._providers

    @property
    def projectors(self) -> TrackingProjectorRegistry:
        """Return the immutable exact-version projector registry."""
        return self._projectors

    @property
    def evaluators(self) -> TrackingEvaluatorRegistry:
        """Return the immutable exact-version evaluator registry."""
        return self._evaluators

    @classmethod
    def with_builtins(cls) -> TrackingRuntime:
        """Create a runtime with context feedback and built-in typed metrics."""
        return cls(
            TrackingFeedbackProviderRegistry(
                [PlanningContextTrackingFeedbackProvider()]
            ),
            TrackingProjectorRegistry([JointPositionTrackingProjector()]),
            TrackingEvaluatorRegistry(
                [
                    JointPositionTrackingEvaluator(),
                    PoseTrackingEvaluator(),
                    WholeBodyPoseTrackingEvaluator(),
                ]
            ),
        )

    def project(
        self, command: EndpointCommand, binding: EndpointTrackingChannelBinding
    ) -> TrackingState:
        """Project one command through the exact binding-owned projector."""
        return self.projectors.resolve(binding.projector).project(command, binding)

    def observe(
        self, setpoint: TrackingSetpoint, context: PlanningContext
    ) -> TrackingFeedbackBatch:
        """Read the exact feedback source for one setpoint."""
        feedback = self.providers.resolve(setpoint.binding.source).observe(
            setpoint.binding.source, context
        )
        if (
            feedback.source.source_fingerprint
            != setpoint.binding.source.source_fingerprint
        ):
            raise ValueError("Feedback provider returned a different source.")
        if feedback.state.channel_id != setpoint.binding.channel_id:
            raise TypeError("Feedback state does not match the bound channel.")
        if feedback.timestamp != context.robot.timestamp:
            raise ValueError(
                "Tracking feedback must use the current planning-context timestamp."
            )
        if feedback.state.batch_size != context.batch_size:
            raise ValueError("Tracking feedback batch must match the context batch.")
        if feedback.state.device != context.robot.qpos.device:
            raise ValueError("Tracking feedback and context must share a device.")
        return feedback

    def evaluate(
        self,
        setpoint: TrackingSetpoint,
        feedback: TrackingFeedbackBatch,
        metric: TrackingMetricCfg,
    ) -> TrackingEvaluation:
        """Evaluate one observed setpoint with an exact metric implementation."""
        if metric.channel_id != setpoint.binding.channel_id:
            raise ValueError("Metric and setpoint channels must match.")
        if (
            feedback.source.source_fingerprint
            != setpoint.binding.source.source_fingerprint
        ):
            raise ValueError("Feedback source does not match the setpoint binding.")
        return self.evaluators.resolve(metric).evaluate(
            setpoint.desired, feedback.state, feedback.valid_mask, metric
        )

    def evaluate_frame(
        self,
        frame: TrackingFrame,
        metrics: Iterable[TrackingMetricCfg],
        context: PlanningContext,
    ) -> Mapping[tuple[str, str, str], TrackingEvaluation]:
        """Observe and evaluate every setpoint required by one frame."""
        by_channel = {metric.channel_id: metric for metric in metrics}
        results: dict[tuple[str, str, str], TrackingEvaluation] = {}
        for setpoint in frame.setpoints:
            try:
                metric = by_channel[setpoint.binding.channel_id]
            except KeyError as exc:
                raise KeyError(
                    f"No metric configured for channel {setpoint.binding.channel_id!r}."
                ) from exc
            results[setpoint.key] = self.evaluate(
                setpoint, self.observe(setpoint, context), metric
            )
        return MappingProxyType(results)


__all__ = [
    "BASE_POSE_CHANNEL",
    "FeedbackTerminalAcceptance",
    "InFlightTrackingPolicy",
    "JOINT_POSITION_CHANNEL",
    "JointPositionTrackingEvaluator",
    "JointPositionTrackingMetric",
    "JointPositionTrackingProjector",
    "JointPositionTrackingState",
    "EndpointTrackingChannelBinding",
    "EndpointTrackingFeedbackAddress",
    "PlanningContextTrackingFeedbackProvider",
    "PoseTrackingEvaluator",
    "PoseTrackingMetric",
    "PoseTrackingState",
    "TerminalAcceptance",
    "TimedTerminalAcceptance",
    "TimedTrackingSequence",
    "TrackingChannelId",
    "TrackingCommandProjector",
    "TrackingEvaluation",
    "TrackingEvaluatorRegistry",
    "TrackingFeedbackAddress",
    "TrackingFeedbackBatch",
    "TrackingFeedbackProvider",
    "TrackingFeedbackProviderRegistry",
    "TrackingFeedbackSourceRef",
    "TrackingFrame",
    "TrackingMetricCfg",
    "TrackingMetricEvaluator",
    "TrackingPolicy",
    "TrackingProjectorRef",
    "TrackingProjectorRegistry",
    "TrackingRuntime",
    "TrackingSetpoint",
    "TrackingState",
    "WHOLE_BODY_POSE_CHANNEL",
    "WholeBodyPoseTrackingEvaluator",
    "WholeBodyPoseTrackingMetric",
    "WholeBodyPoseTrackingState",
]
