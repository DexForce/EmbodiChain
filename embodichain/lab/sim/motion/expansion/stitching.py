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

"""Phase-aligned trajectory segments, source selection, and safe stitching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal, Protocol, runtime_checkable

import torch

from embodichain.compute.trajectory import interpolate_minimum_jerk

from .contracts import (
    TrajectoryPhase,
    TrajectoryTemplate,
    _digest,
    _names,
    _tensor,
    _text,
)

__all__ = [
    "TrajectorySegment",
    "SegmentCompatibility",
    "TrajectorySegmentBank",
    "TrajectoryBridge",
    "TrajectoryBridgePlanner",
    "MinimumJerkBridgePlanner",
    "StitchedSegmentPlacement",
    "StitchedTrajectory",
    "extract_trajectory_segments",
    "compare_trajectory_segments",
    "stitch_trajectory_segments",
]


def _indices(values: Sequence[int], width: int) -> tuple[int, ...]:
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < width
        for value in result
    ):
        raise ValueError("controlled_joint_indices must index joint_names")
    if len(set(result)) != len(result):
        raise ValueError("controlled_joint_indices must be unique")
    return result


@dataclass(frozen=True)
class TrajectorySegment:
    """One owned source phase normalized onto a segment-local time axis.

    Args:
        segment_id: Stable identity of this source segment.
        source_id: Source trajectory provider identity.
        source_revision: Source provider revision.
        source_template_id: Template containing the original phase.
        source_start_index: Inclusive phase start in the source template.
        source_stop_index: Exclusive phase stop in the source template.
        phase_id: Semantic phase used for replacement compatibility.
        kind: Motion mode: free, contact, or hold.
        joint_names: Complete ordered joint layout.
        positions: Segment positions with shape ``(N >= 2, D)``.
        dt: Local arrival intervals with ``dt[0] == 0``.
        allowed_operators: Operators permitted on this phase.
        controlled_joint_indices: Joints controlled by the source trajectory.
        validator_id: Downstream physical validation contract.
    """

    segment_id: str
    source_id: str
    source_revision: str
    source_template_id: str
    source_start_index: int
    source_stop_index: int
    phase_id: str
    kind: Literal["free", "contact", "hold"]
    joint_names: tuple[str, ...]
    positions: torch.Tensor
    dt: torch.Tensor
    allowed_operators: tuple[str, ...] = ()
    controlled_joint_indices: tuple[int, ...] = ()
    validator_id: str = "default"

    def __post_init__(self) -> None:
        for name in (
            "segment_id",
            "source_id",
            "source_revision",
            "source_template_id",
            "phase_id",
            "validator_id",
        ):
            _text(getattr(self, name), name)
        if (
            isinstance(self.source_start_index, bool)
            or not isinstance(self.source_start_index, int)
            or self.source_start_index < 0
            or isinstance(self.source_stop_index, bool)
            or not isinstance(self.source_stop_index, int)
            or self.source_stop_index <= self.source_start_index
        ):
            raise ValueError("source indices must define a nonempty half-open range")
        if self.kind not in ("free", "contact", "hold"):
            raise ValueError("kind must be free, contact, or hold")
        names = _names(self.joint_names, "joint_names")
        positions = _tensor(self.positions, "positions", 2)
        dt = _tensor(self.dt, "dt", 1)
        if positions.shape[0] < 2 or positions.shape[1] != len(names):
            raise ValueError("positions must have shape (N >= 2, len(joint_names))")
        if dt.shape != positions.shape[:1] or dt.device != positions.device:
            raise ValueError("dt must match positions length and device")
        if dt[0] != 0 or bool((dt[1:] <= 0).any()):
            raise ValueError("dt[0] must be zero and later arrival intervals positive")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(
            self,
            "allowed_operators",
            _names(self.allowed_operators, "allowed_operators"),
        )
        object.__setattr__(
            self,
            "controlled_joint_indices",
            _indices(self.controlled_joint_indices, len(names)),
        )


@dataclass(frozen=True)
class SegmentCompatibility:
    """Replacement compatibility and boundary distances for two segments.

    Args:
        compatible: Whether the candidate may replace the reference segment.
        reasons: Field names that make an incompatible pair unsafe to combine.
        start_distance: Joint-space distance between the starting samples.
        end_distance: Joint-space distance between the ending samples.
    """

    compatible: bool
    reasons: tuple[str, ...]
    start_distance: float
    end_distance: float

    def __post_init__(self) -> None:
        if type(self.compatible) is not bool:
            raise TypeError("compatible must be a boolean")
        reasons = tuple(self.reasons)
        if any(type(reason) is not str or not reason for reason in reasons):
            raise ValueError("reasons must contain nonempty strings")
        if self.compatible == bool(reasons):
            raise ValueError(
                "compatible segments have no reasons; incompatible ones do"
            )
        for name in ("start_distance", "end_distance"):
            value = getattr(self, name)
            if math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and not NaN")
        object.__setattr__(self, "reasons", reasons)


def extract_trajectory_segments(
    template: TrajectoryTemplate,
) -> tuple[TrajectorySegment, ...]:
    """Extract every annotated phase as an independently owned segment.

    Args:
        template: Phase-annotated full-joint source trajectory.

    Returns:
        Source-order segments with local timing and retained source lineage.
    """
    if not isinstance(template, TrajectoryTemplate):
        raise TypeError("template must be a TrajectoryTemplate")
    if not template.phases:
        raise ValueError("segment extraction requires annotated phases")
    result = []
    for ordinal, phase in enumerate(template.phases):
        positions = template.positions[phase.start_index : phase.stop_index]
        if positions.shape[0] < 2:
            raise ValueError(f"phase {phase.phase_id!r} needs at least two samples")
        dt = template.dt[phase.start_index : phase.stop_index].clone()
        dt[0] = 0
        result.append(
            TrajectorySegment(
                segment_id="segment_"
                + _digest(
                    template.source_id,
                    template.source_revision,
                    template.template_id,
                    ordinal,
                    phase.phase_id,
                ),
                source_id=template.source_id,
                source_revision=template.source_revision,
                source_template_id=template.template_id,
                source_start_index=phase.start_index,
                source_stop_index=phase.stop_index,
                phase_id=phase.phase_id,
                kind=phase.kind,
                joint_names=template.joint_names,
                positions=positions,
                dt=dt,
                allowed_operators=phase.allowed_operators,
                controlled_joint_indices=template.controlled_joint_indices,
                validator_id=template.validator_id,
            )
        )
    return tuple(result)


def compare_trajectory_segments(
    reference: TrajectorySegment, candidate: TrajectorySegment
) -> SegmentCompatibility:
    """Compare whether a candidate may replace one semantic source phase.

    Args:
        reference: Segment slot whose semantics and joint layout must be retained.
        candidate: Source segment proposed for that slot.

    Returns:
        Compatibility reasons plus unnormalized start/end joint distances.
    """
    if not isinstance(reference, TrajectorySegment) or not isinstance(
        candidate, TrajectorySegment
    ):
        raise TypeError("reference and candidate must be TrajectorySegment values")
    reasons = []
    if candidate.phase_id != reference.phase_id:
        reasons.append("phase_id")
    if candidate.kind != reference.kind:
        reasons.append("kind")
    if candidate.joint_names != reference.joint_names:
        reasons.append("joint_names")
    if candidate.controlled_joint_indices != reference.controlled_joint_indices:
        reasons.append("controlled_joint_indices")
    if (
        candidate.positions.dtype != reference.positions.dtype
        or candidate.positions.device != reference.positions.device
    ):
        reasons.append("dtype_or_device")
    if candidate.dt.dtype != reference.dt.dtype:
        reasons.append("timing_dtype")
    if (
        candidate.positions.shape[1] == reference.positions.shape[1]
        and candidate.positions.dtype == reference.positions.dtype
        and candidate.positions.device == reference.positions.device
    ):
        start = float(
            torch.linalg.vector_norm(
                candidate.positions[0] - reference.positions[0]
            ).item()
        )
        end = float(
            torch.linalg.vector_norm(
                candidate.positions[-1] - reference.positions[-1]
            ).item()
        )
    else:
        start = end = math.inf
    return SegmentCompatibility(not reasons, tuple(reasons), start, end)


def _compatibility_key(segment: TrajectorySegment) -> tuple[object, ...]:
    return (
        segment.phase_id,
        segment.kind,
        segment.joint_names,
        segment.controlled_joint_indices,
        segment.positions.dtype,
        segment.positions.device,
        segment.dt.dtype,
    )


class TrajectorySegmentBank:
    """Insertion-ordered source segments with deterministic selection policies.

    Args:
        segments: Initial immutable segments in registration order.
    """

    def __init__(self, segments: Sequence[TrajectorySegment] = ()) -> None:
        self._segments: list[TrajectorySegment] = []
        self._ids: set[str] = set()
        self._by_contract: dict[tuple[object, ...], list[TrajectorySegment]] = {}
        for segment in segments:
            self.add(segment)

    def __len__(self) -> int:
        """Return the number of registered source segments.

        Returns:
            Number of segments currently retained by the bank.
        """
        return len(self._segments)

    def add(self, segment: TrajectorySegment) -> None:
        """Register one uniquely identified immutable segment.

        Args:
            segment: Segment to retain in insertion order.
        """
        if not isinstance(segment, TrajectorySegment):
            raise TypeError("segment must be a TrajectorySegment")
        if segment.segment_id in self._ids:
            raise ValueError(f"duplicate segment_id: {segment.segment_id!r}")
        self._segments.append(segment)
        self._ids.add(segment.segment_id)
        self._by_contract.setdefault(_compatibility_key(segment), []).append(segment)

    def candidates(self, reference: TrajectorySegment) -> tuple[TrajectorySegment, ...]:
        """Return insertion-ordered segments compatible with a reference slot.

        Args:
            reference: Semantic phase and layout contract to match.

        Returns:
            Compatible source segments in stable registration order.
        """
        if not isinstance(reference, TrajectorySegment):
            raise TypeError("reference must be a TrajectorySegment")
        return tuple(self._by_contract.get(_compatibility_key(reference), ()))

    def select(
        self,
        reference: TrajectorySegment,
        *,
        strategy: Literal["fifo", "nearest_start"] = "fifo",
        current_position: torch.Tensor | None = None,
        ordinal: int = 0,
    ) -> TrajectorySegment:
        """Select one compatible source segment deterministically.

        Args:
            reference: Semantic phase and layout contract to match.
            strategy: Registration order or nearest starting joint state.
            current_position: Full joint state required by ``nearest_start``.
            ordinal: Stable selection rank after applying the strategy.

        Returns:
            The selected compatible source segment.
        """
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        values = list(self.candidates(reference))
        if not values:
            raise LookupError(f"no compatible segment for phase {reference.phase_id!r}")
        if strategy == "nearest_start":
            if not isinstance(current_position, torch.Tensor):
                raise ValueError("nearest_start requires current_position")
            if (
                current_position.shape != values[0].positions.shape[1:]
                or current_position.dtype != values[0].positions.dtype
                or current_position.device != values[0].positions.device
                or not bool(torch.isfinite(current_position).all())
            ):
                raise ValueError("current_position must match the segment joint layout")
            distances = torch.stack(
                [
                    torch.linalg.vector_norm(segment.positions[0] - current_position)
                    for segment in values
                ]
            )
            order = torch.argsort(distances, stable=True).tolist()
            values = [values[index] for index in order]
        elif strategy != "fifo":
            raise ValueError("strategy must be fifo or nearest_start")
        return values[ordinal % len(values)]


@dataclass(frozen=True)
class TrajectoryBridge:
    """One planner-produced joint transition including both boundary samples.

    Args:
        positions: Full joint positions with shape ``(N >= 2, D)``.
        dt: Local arrival intervals with ``dt[0] == 0``.
    """

    positions: torch.Tensor
    dt: torch.Tensor

    def __post_init__(self) -> None:
        positions = _tensor(self.positions, "positions", 2)
        dt = _tensor(self.dt, "dt", 1)
        if positions.shape[0] < 2 or positions.shape[1] < 1:
            raise ValueError("bridge positions must have shape (N >= 2, D >= 1)")
        if dt.shape != positions.shape[:1] or dt.device != positions.device:
            raise ValueError("bridge dt must match positions length and device")
        if dt[0] != 0 or bool((dt[1:] <= 0).any()):
            raise ValueError("bridge dt[0] must be zero and later intervals positive")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "dt", dt)


@runtime_checkable
class TrajectoryBridgePlanner(Protocol):
    """Plan a joint transition between two segment boundary states."""

    def plan(
        self, start_position: torch.Tensor, goal_position: torch.Tensor
    ) -> TrajectoryBridge:
        """Plan a transition containing the exact start and goal samples.

        Args:
            start_position: Full starting joint state with shape ``(D,)``.
            goal_position: Full goal joint state with shape ``(D,)``.

        Returns:
            A transition whose first and last positions equal the inputs.
        """


@dataclass(frozen=True)
class MinimumJerkBridgePlanner:
    """Generate rest-to-rest bridges under sampled velocity/acceleration limits.

    This geometric fallback does not check collisions. A host should inject a
    collision-aware planner for cluttered transitions and validate every final
    stitched trajectory before execution.

    Args:
        control_dt: Positive host control period in seconds.
        velocity_limits: Positive per-joint velocity limits.
        acceleration_limits: Positive per-joint acceleration limits.
        minimum_sample_count: Lower bound for interpolation samples.
    """

    control_dt: float
    velocity_limits: torch.Tensor
    acceleration_limits: torch.Tensor
    minimum_sample_count: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.control_dt, bool)
            or not isinstance(self.control_dt, (int, float))
            or not math.isfinite(self.control_dt)
            or self.control_dt <= 0
        ):
            raise ValueError("control_dt must be finite and positive")
        if (
            isinstance(self.minimum_sample_count, bool)
            or not isinstance(self.minimum_sample_count, int)
            or self.minimum_sample_count < 2
        ):
            raise ValueError("minimum_sample_count must be an integer of at least two")
        velocity = _tensor(self.velocity_limits, "velocity_limits", 1)
        acceleration = _tensor(self.acceleration_limits, "acceleration_limits", 1)
        if velocity.shape != acceleration.shape or velocity.numel() < 1:
            raise ValueError("bridge limits must have one aligned value per joint")
        if bool((velocity <= 0).any()) or bool((acceleration <= 0).any()):
            raise ValueError("bridge limits must be positive")
        object.__setattr__(self, "velocity_limits", velocity)
        object.__setattr__(self, "acceleration_limits", acceleration)

    def plan(
        self, start_position: torch.Tensor, goal_position: torch.Tensor
    ) -> TrajectoryBridge:
        """Interpolate one exact endpoint pair on the configured control clock.

        Args:
            start_position: Full starting joint state, shape ``(D,)``.
            goal_position: Full goal joint state, shape ``(D,)``.

        Returns:
            A rest-to-rest minimum-jerk transition requiring collision checks.
        """
        if (
            not isinstance(start_position, torch.Tensor)
            or not isinstance(goal_position, torch.Tensor)
            or start_position.ndim != 1
            or goal_position.shape != start_position.shape
        ):
            raise ValueError("bridge endpoints must be aligned rank-one tensors")
        if (
            start_position.shape != self.velocity_limits.shape
            or start_position.dtype != self.velocity_limits.dtype
            or start_position.device != self.velocity_limits.device
            or goal_position.dtype != start_position.dtype
            or goal_position.device != start_position.device
        ):
            raise ValueError(
                "bridge endpoints and limits must share shape/dtype/device"
            )
        keyframes = torch.stack((start_position, goal_position)).unsqueeze(0)
        positions = interpolate_minimum_jerk(
            keyframes,
            self.minimum_sample_count,
            self.control_dt,
            self.velocity_limits,
            self.acceleration_limits,
        )[0]
        dt = positions.new_full((positions.shape[0],), float(self.control_dt))
        dt[0] = 0
        return TrajectoryBridge(positions, dt)


@dataclass(frozen=True)
class StitchedSegmentPlacement:
    """Source-segment lineage aligned with its half-open output sample range.

    Args:
        segment_id: Stable identity of the source segment.
        source_id: Source trajectory provider identity.
        source_revision: Source provider revision.
        source_template_id: Template containing the source phase.
        source_start_index: Inclusive source phase start.
        source_stop_index: Exclusive source phase stop.
        output_phase_id: Unique phase identity in the stitched template.
        output_start_index: Inclusive output phase start.
        output_stop_index: Exclusive output phase stop.
    """

    segment_id: str
    source_id: str
    source_revision: str
    source_template_id: str
    source_start_index: int
    source_stop_index: int
    output_phase_id: str
    output_start_index: int
    output_stop_index: int

    def __post_init__(self) -> None:
        for name in (
            "segment_id",
            "source_id",
            "source_revision",
            "source_template_id",
            "output_phase_id",
        ):
            _text(getattr(self, name), name)
        for prefix in ("source", "output"):
            start = getattr(self, f"{prefix}_start_index")
            stop = getattr(self, f"{prefix}_stop_index")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or isinstance(stop, bool)
                or not isinstance(stop, int)
                or stop <= start
            ):
                raise ValueError(f"{prefix} indices must define a nonempty range")


@dataclass(frozen=True)
class StitchedTrajectory:
    """A stitched qpos template with explicit multi-source segment lineage.

    Args:
        template: Composite full-joint trajectory accepted by expansion sources.
        placements: Source lineage for every non-bridge output phase.
        bridge_phase_ids: Output phases produced by boundary planners.
    """

    template: TrajectoryTemplate
    placements: tuple[StitchedSegmentPlacement, ...]
    bridge_phase_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.template, TrajectoryTemplate):
            raise TypeError("template must be a TrajectoryTemplate")
        placements = tuple(self.placements)
        if not placements or any(
            not isinstance(value, StitchedSegmentPlacement) for value in placements
        ):
            raise ValueError("placements must contain source segment lineage")
        bridge_ids = _names(self.bridge_phase_ids, "bridge_phase_ids")
        phase_ids = {phase.phase_id for phase in self.template.phases}
        phases = {phase.phase_id: phase for phase in self.template.phases}
        if any(value.output_phase_id not in phase_ids for value in placements):
            raise ValueError("placements must reference output trajectory phases")
        if any(
            (
                value.output_start_index,
                value.output_stop_index,
            )
            != (
                phases[value.output_phase_id].start_index,
                phases[value.output_phase_id].stop_index,
            )
            for value in placements
        ):
            raise ValueError("placement ranges must match their output phases")
        if not set(bridge_ids) <= phase_ids:
            raise ValueError("bridge_phase_ids must reference output trajectory phases")
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "bridge_phase_ids", bridge_ids)


def _validate_stitch_layout(segments: tuple[TrajectorySegment, ...]) -> None:
    reference = segments[0]
    for segment in segments[1:]:
        if segment.joint_names != reference.joint_names:
            raise ValueError("stitched segments must share joint_names")
        if segment.controlled_joint_indices != reference.controlled_joint_indices:
            raise ValueError("stitched segments must share controlled joints")
        if (
            segment.positions.dtype != reference.positions.dtype
            or segment.positions.device != reference.positions.device
        ):
            raise ValueError("stitched segments must share dtype and device")
        if segment.dt.dtype != reference.dt.dtype:
            raise ValueError("stitched segments must share timing dtype")


def stitch_trajectory_segments(
    segments: Sequence[TrajectorySegment],
    *,
    source_id: str,
    source_revision: str,
    template_id: str,
    bridge_planner: TrajectoryBridgePlanner | None = None,
    boundary_tolerance: float = 1e-6,
    validator_id: str | None = None,
) -> StitchedTrajectory:
    """Concatenate phase segments and explicitly plan discontinuous boundaries.

    Segment first samples are shared boundaries and are emitted only once. The
    adjacent output phases both reference that one endpoint, which keeps each
    phase independently extractable without duplicating trajectory samples. A
    planned bridge follows the same boundary convention. The result is
    structurally continuous but still requires motion-limit, collision, and
    task validation.

    Args:
        segments: Ordered source phases to compose.
        source_id: Identity of the new composite source.
        source_revision: Revision of the composition policy.
        template_id: Identity of this stitched source unit.
        bridge_planner: Required when adjacent boundary positions differ.
        boundary_tolerance: Maximum absolute joint difference treated as shared.
        validator_id: Override for the final validator; omitted requires all
            source segments to agree.

    Returns:
        A full qpos template plus source and bridge lineage.
    """
    values = tuple(segments)
    if not values or any(not isinstance(value, TrajectorySegment) for value in values):
        raise ValueError("segments must contain at least one TrajectorySegment")
    for value, name in (
        (source_id, "source_id"),
        (source_revision, "source_revision"),
        (template_id, "template_id"),
    ):
        _text(value, name)
    if (
        isinstance(boundary_tolerance, bool)
        or not isinstance(boundary_tolerance, (int, float))
        or not math.isfinite(boundary_tolerance)
        or boundary_tolerance < 0
    ):
        raise ValueError("boundary_tolerance must be finite and non-negative")
    if bridge_planner is not None and not isinstance(
        bridge_planner, TrajectoryBridgePlanner
    ):
        raise TypeError("bridge_planner must implement TrajectoryBridgePlanner")
    _validate_stitch_layout(values)
    validators = {segment.validator_id for segment in values}
    if validator_id is None:
        if len(validators) != 1:
            raise ValueError("source segments require an explicit common validator_id")
        resolved_validator = next(iter(validators))
    else:
        _text(validator_id, "validator_id")
        resolved_validator = validator_id

    position_parts = [values[0].positions]
    dt_parts = [values[0].dt]
    phases = []
    placements = []
    bridge_ids = []
    cursor = values[0].positions.shape[0]
    first_phase_id = f"segment_0:{values[0].phase_id}"
    phases.append(
        TrajectoryPhase(
            first_phase_id,
            0,
            cursor,
            values[0].kind,
            values[0].allowed_operators,
        )
    )
    placements.append(
        StitchedSegmentPlacement(
            values[0].segment_id,
            values[0].source_id,
            values[0].source_revision,
            values[0].source_template_id,
            values[0].source_start_index,
            values[0].source_stop_index,
            first_phase_id,
            0,
            cursor,
        )
    )

    for index, (left, right) in enumerate(zip(values, values[1:]), 1):
        gap = float((left.positions[-1] - right.positions[0]).abs().max().item())
        if gap > boundary_tolerance:
            if bridge_planner is None:
                raise ValueError(
                    f"segments {left.segment_id!r} and {right.segment_id!r} "
                    "need an explicit bridge planner"
                )
            bridge = bridge_planner.plan(left.positions[-1], right.positions[0])
            if not isinstance(bridge, TrajectoryBridge):
                raise TypeError("bridge planner must return a TrajectoryBridge")
            if bridge.positions.shape[1:] != left.positions.shape[1:]:
                raise ValueError("bridge joint width does not match source segments")
            if (
                bridge.positions.dtype != left.positions.dtype
                or bridge.positions.device != left.positions.device
            ):
                raise ValueError("bridge and source segments must share dtype/device")
            if not torch.allclose(
                bridge.positions[0],
                left.positions[-1],
                atol=boundary_tolerance,
                rtol=0,
            ) or not torch.allclose(
                bridge.positions[-1],
                right.positions[0],
                atol=boundary_tolerance,
                rtol=0,
            ):
                raise ValueError("bridge must preserve the exact segment boundaries")
            appended = bridge.positions.shape[0] - 1
            position_parts.append(bridge.positions[1:])
            dt_parts.append(bridge.dt[1:].to(dtype=values[0].dt.dtype))
            bridge_phase_id = f"bridge_{index - 1}_{index}"
            phases.append(
                TrajectoryPhase(
                    bridge_phase_id,
                    cursor - 1,
                    cursor + appended,
                    "free",
                )
            )
            bridge_ids.append(bridge_phase_id)
            cursor += appended

        appended = right.positions.shape[0] - 1
        position_parts.append(right.positions[1:])
        dt_parts.append(right.dt[1:])
        phase_id = f"segment_{index}:{right.phase_id}"
        phases.append(
            TrajectoryPhase(
                phase_id,
                cursor - 1,
                cursor + appended,
                right.kind,
                right.allowed_operators,
            )
        )
        placements.append(
            StitchedSegmentPlacement(
                right.segment_id,
                right.source_id,
                right.source_revision,
                right.source_template_id,
                right.source_start_index,
                right.source_stop_index,
                phase_id,
                cursor - 1,
                cursor + appended,
            )
        )
        cursor += appended

    operators = tuple(
        dict.fromkeys(
            operator for phase in phases for operator in phase.allowed_operators
        )
    )
    template = TrajectoryTemplate(
        source_id=source_id,
        source_revision=source_revision,
        template_id=template_id,
        joint_names=values[0].joint_names,
        positions=torch.cat(position_parts),
        dt=torch.cat(dt_parts),
        phases=tuple(phases),
        allowed_operators=operators,
        validator_id=resolved_validator,
        controlled_joint_indices=values[0].controlled_joint_indices,
    )
    return StitchedTrajectory(template, tuple(placements), tuple(bridge_ids))
