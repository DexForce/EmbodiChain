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

"""Typed source and normalization evidence for annotated motion units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "JointSpec",
    "NormalizationReport",
    "NormalizedReal2SimUnit",
    "PhaseAnnotation",
    "Real2SimTrajectory",
    "TrajectoryTimeMap",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _array(value: object, name: str, ndim: int) -> FloatArray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _indices(value: object, name: str) -> IntArray:
    result = np.array(value, copy=True)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = result.astype(np.int64, copy=False)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class JointSpec:
    """One target joint and its optional linear mimic relationship.

    Input joint values use the units declared by the source; mimic parameters
    use canonical radians or metres. A non-mimic joint missing from the input
    needs an explicit canonical ``initial_hold``.
    """

    name: str
    kind: Literal["revolute", "prismatic"]
    mimic_driver: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0
    initial_hold: float | None = None

    def __post_init__(self) -> None:
        _text(self.name, "joint name")
        if self.kind not in {"revolute", "prismatic"}:
            raise ValueError("joint kind must be revolute or prismatic")
        if self.mimic_driver is not None:
            _text(self.mimic_driver, "mimic driver")
            if self.mimic_driver == self.name or self.initial_hold is not None:
                raise ValueError(
                    "mimic driver must differ and cannot have initial_hold"
                )
        for value in (self.mimic_multiplier, self.mimic_offset, self.initial_hold):
            if value is not None and not np.isfinite(value):
                raise ValueError("joint parameters must be finite")


@dataclass(frozen=True)
class PhaseAnnotation:
    """A named half-open sample range and its authorized motion edits."""

    name: str
    start: int
    stop: int
    kind: Literal["free", "contact", "hold"]
    reference: Literal["object", "target", "gravity"]
    allowed_operators: tuple[str, ...] = ()
    authority: Literal["qpos", "eef_pose"] | None = None

    def __post_init__(self) -> None:
        _text(self.name, "phase name")
        if type(self.start) is not int or type(self.stop) is not int:
            raise ValueError("phase bounds must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("phase bounds must define a nonempty half-open range")
        if self.kind not in {"free", "contact", "hold"}:
            raise ValueError("phase kind must be free, contact, or hold")
        if self.reference not in {"object", "target", "gravity"}:
            raise ValueError("unknown phase reference")
        if self.authority not in {None, "qpos", "eef_pose"}:
            raise ValueError("phase authority must be qpos, eef_pose, or None")
        operators = tuple(self.allowed_operators)
        if any(not isinstance(item, str) or not item.strip() for item in operators):
            raise ValueError("allowed operators must be nonempty strings")
        if len(set(operators)) != len(operators):
            raise ValueError("allowed operators must be unique")
        if self.kind != "free" and operators:
            raise ValueError("contact and hold phases cannot authorize path edits")
        object.__setattr__(self, "allowed_operators", operators)


@dataclass(frozen=True)
class Real2SimTrajectory:
    """One explicitly annotated source unit before conversion.

    ``eef_poses``, when provided, is ``(N, 7)`` position and quaternion in
    ``pose_frame``. The transforms convert from that frame to world and from
    the observed link to TCP. Object and target anchors are world-frame rigid
    transforms in metres. ``origin=measured`` identifies qpos and optional qvel
    as observed state. ``origin=synthetic_reference`` identifies them as a
    design reference; synthetic measured rollout state belongs in a separate
    typed artifact. Neither is evidence of the commands actually submitted.
    Command alignment independently identifies a submitted stream or a
    sample-aligned synthetic design reference.
    """

    source_id: str
    source_revision: str
    unit_id: str
    skill_id: Literal["pick_up", "move_held_object", "place"]
    origin: Literal["measured", "synthetic_reference"]
    joint_names: tuple[str, ...]
    joint_units: tuple[Literal["rad", "deg", "m", "mm"], ...]
    qpos: FloatArray
    timestamps: FloatArray
    time_unit: Literal["s", "ms"]
    phases: tuple[PhaseAnnotation, ...]
    eef_poses: FloatArray | None = None
    pose_length_unit: Literal["m", "mm"] | None = None
    quaternion_order: Literal["xyzw", "wxyz"] | None = None
    world_from_pose_frame: FloatArray | None = None
    observed_link_to_tcp: FloatArray | None = None
    object_pose: FloatArray | None = None
    target_pose: FloatArray | None = None
    qvel: FloatArray | None = None
    initial_observed_qpos: FloatArray | None = None
    commands: FloatArray | None = None
    command_representation: str | None = None
    command_alignment: (
        Literal["between_states", "post_step", "sample_aligned_reference"] | None
    ) = None
    calibration_revision: str = "unverified"
    annotation_revision: str = "unverified"
    pose_origin: Literal["measured", "fk_derived", "transformed"] | None = None
    robot_geometry_sha256: str | None = None
    object_geometry_sha256: str | None = None
    target_geometry_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "robot_geometry_sha256",
            "object_geometry_sha256",
            "target_geometry_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest or None")
        for name in (
            "source_id",
            "source_revision",
            "unit_id",
            "calibration_revision",
            "annotation_revision",
        ):
            _text(getattr(self, name), name)
        if self.skill_id not in {"pick_up", "move_held_object", "place"}:
            raise ValueError("unsupported skill_id")
        if self.origin not in {"measured", "synthetic_reference"}:
            raise ValueError("unsupported trajectory origin")
        names, units = tuple(self.joint_names), tuple(self.joint_units)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(x, str) or not x.strip() for x in names)
        ):
            raise ValueError("joint_names must be unique nonempty strings")
        if len(units) != len(names) or any(
            unit not in {"rad", "deg", "m", "mm"} for unit in units
        ):
            raise ValueError("each joint requires an explicit supported unit")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_units", units)
        qpos = _array(self.qpos, "qpos", 2)
        times = _array(self.timestamps, "timestamps", 1)
        if qpos.shape != (len(times), len(names)) or len(times) < 2:
            raise ValueError("qpos must have shape (N >= 2, len(joint_names))")
        if self.time_unit not in {"s", "ms"}:
            raise ValueError("time_unit must be s or ms")
        if self.pose_length_unit not in {
            None,
            "m",
            "mm",
        } or self.quaternion_order not in {None, "xyzw", "wxyz"}:
            raise ValueError("pose units and quaternion order must be explicit")
        if self.pose_origin not in {None, "measured", "fk_derived", "transformed"}:
            raise ValueError("unknown pose origin")
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "timestamps", times)
        if self.initial_observed_qpos is not None:
            initial = _array(self.initial_observed_qpos, "initial_observed_qpos", 1)
            if initial.shape != (len(names),):
                raise ValueError(
                    "initial_observed_qpos must match the source joint names"
                )
            object.__setattr__(self, "initial_observed_qpos", initial)
        phases = tuple(self.phases)
        if not phases or any(not isinstance(p, PhaseAnnotation) for p in phases):
            raise ValueError("at least one typed phase is required")
        if len({p.name for p in phases}) != len(phases):
            raise ValueError("phase names must be unique within a unit")
        if phases[0].start != 0 or phases[-1].stop != len(times):
            raise ValueError("phases must cover the complete source unit")
        if any(left.stop != right.start for left, right in zip(phases, phases[1:])):
            raise ValueError("phase ranges must be contiguous and non-overlapping")
        object.__setattr__(self, "phases", phases)
        for name in ("qvel", "commands"):
            value = getattr(self, name)
            if value is not None:
                array = _array(value, name, 2)
                if (name == "qvel" and array.shape != qpos.shape) or (
                    name == "commands"
                    and array.shape[0] not in {len(times), len(times) - 1}
                ):
                    raise ValueError(f"{name} must align with source samples")
                object.__setattr__(self, name, array)

        if (self.commands is None) != (self.command_representation is None) or (
            self.commands is None
        ) != (self.command_alignment is None):
            raise ValueError(
                "commands, their representation and alignment must be declared together"
            )
        if self.command_representation is not None:
            _text(self.command_representation, "command representation")
            if self.command_alignment not in {
                "between_states",
                "post_step",
                "sample_aligned_reference",
            }:
                raise ValueError("unknown command alignment")
            expected = (
                len(times) - 1
                if self.command_alignment == "between_states"
                else len(times)
            )
            if self.commands.shape[0] != expected:
                raise ValueError("command count does not match its declared alignment")
            if (
                self.command_alignment == "sample_aligned_reference"
                and self.origin != "synthetic_reference"
            ):
                raise ValueError(
                    "sample-aligned commands are only supported for synthetic references"
                )
        if self.eef_poses is not None:
            if self.pose_origin is None:
                raise ValueError("EEF data requires an explicit pose origin")
            if self.pose_length_unit is None or self.quaternion_order is None:
                raise ValueError(
                    "EEF data requires explicit pose units and quaternion order"
                )
            poses = _array(self.eef_poses, "eef_poses", 2)
            if poses.shape != (len(times), 7):
                raise ValueError("eef_poses must have shape (N, 7)")
            if self.world_from_pose_frame is None or self.observed_link_to_tcp is None:
                raise ValueError("EEF data requires both explicit frame transforms")
            object.__setattr__(self, "eef_poses", poses)
        for name in (
            "world_from_pose_frame",
            "observed_link_to_tcp",
            "object_pose",
            "target_pose",
        ):
            value = getattr(self, name)
            if value is not None:
                array = _array(value, name, 2)
                if array.shape != (4, 4):
                    raise ValueError(f"{name} must have shape (4, 4)")
                object.__setattr__(self, name, array)

    @property
    def qpos_semantics(self) -> Literal["measured_state", "design_reference"]:
        """Describe the qpos stream from the explicitly declared source origin."""
        return "measured_state" if self.origin == "measured" else "design_reference"

    @property
    def qvel_semantics(self) -> Literal["measured_state", "design_reference"] | None:
        """Describe the preserved raw velocity stream, when one was supplied."""
        return self.qpos_semantics if self.qvel is not None else None

    @property
    def command_semantics(
        self,
    ) -> Literal["actual_submission", "design_reference"] | None:
        """Distinguish recorded submissions from synthetic sample references."""
        if self.commands is None:
            return None
        return (
            "design_reference"
            if self.command_alignment == "sample_aligned_reference"
            else "actual_submission"
        )


@dataclass(frozen=True)
class TrajectoryTimeMap:
    """Source brackets and weights for each normalized execution sample."""

    left: IntArray
    right: IntArray
    weight: FloatArray
    source_event_indices: IntArray
    execution_event_indices: IntArray
    max_event_error_s: float

    def __post_init__(self) -> None:
        left, right = _indices(self.left, "left"), _indices(self.right, "right")
        weight = _array(self.weight, "weight", 1)
        source = _indices(self.source_event_indices, "source_event_indices")
        dest = _indices(self.execution_event_indices, "execution_event_indices")
        if (
            left.shape != right.shape
            or weight.shape != left.shape
            or source.shape != dest.shape
        ):
            raise ValueError("time-map arrays must have matching shapes")
        if (
            np.any(left < 0)
            or np.any(right < left)
            or np.any((weight < 0) | (weight > 1))
        ):
            raise ValueError("time-map brackets or weights are invalid")
        if (
            np.any(np.diff(left) < 0)
            or np.any(np.diff(right) < 0)
            or np.any(np.diff(source) <= 0)
            or np.any(np.diff(dest) <= 0)
        ):
            raise ValueError("time map and event indices must be monotone")
        if not np.isfinite(self.max_event_error_s) or self.max_event_error_s < 0:
            raise ValueError("max_event_error_s must be finite and nonnegative")
        for name, value in (
            ("left", left),
            ("right", right),
            ("weight", weight),
            ("source_event_indices", source),
            ("execution_event_indices", dest),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class NormalizationReport:
    """Auditable normalization decisions and corrections."""

    joint_source_indices: tuple[int | None, ...]
    corrections: tuple[str, ...]
    time_map: TrajectoryTimeMap
    initial_state_origin: Literal[
        "observed", "first_measured_sample", "reference_assumption"
    ]


@dataclass(frozen=True)
class NormalizedReal2SimUnit:
    """Canonical full-joint motion and source evidence on one control grid."""

    source: Real2SimTrajectory
    joint_names: tuple[str, ...]
    qpos: FloatArray
    initial_qpos: FloatArray
    timestamps: FloatArray
    phases: tuple[PhaseAnnotation, ...]
    world_tcp: FloatArray | None
    raw_qvel: FloatArray | None
    report: NormalizationReport

    def __post_init__(self) -> None:
        if not isinstance(self.source, Real2SimTrajectory):
            raise TypeError("source must be a Real2SimTrajectory")
        if not isinstance(self.report, NormalizationReport):
            raise TypeError("report must be a NormalizationReport")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be unique and nonempty")
        qpos = _array(self.qpos, "normalized qpos", 2)
        initial = _array(self.initial_qpos, "normalized initial_qpos", 1)
        times = _array(self.timestamps, "normalized timestamps", 1)
        if (
            qpos.shape != (len(times), len(names))
            or len(times) < 2
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("normalized qpos and timestamps do not align")
        if initial.shape != (len(names),):
            raise ValueError("normalized initial_qpos must match the joint layout")
        phases = tuple(self.phases)
        if (
            not phases
            or phases[0].start != 0
            or phases[-1].stop != len(times)
            or any(a.stop != b.start for a, b in zip(phases, phases[1:]))
        ):
            raise ValueError("normalized phases must cover all samples")
        if self.world_tcp is not None:
            poses = _array(self.world_tcp, "world_tcp", 3)
            if poses.shape != (len(times), 4, 4):
                raise ValueError("world_tcp must have shape (N, 4, 4)")
            object.__setattr__(self, "world_tcp", poses)
        if self.raw_qvel is not None:
            velocity = _array(self.raw_qvel, "raw_qvel", 2)
            if velocity.shape != self.source.qpos.shape:
                raise ValueError("raw_qvel must retain its original sample layout")
            object.__setattr__(self, "raw_qvel", velocity)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "initial_qpos", initial)
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "phases", phases)
