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

"""Dataset-independent real-trajectory values; no simulator or dataset imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json
from typing import Literal

import numpy as np

__all__ = [
    "SourcePhase",
    "MimicJoint",
    "RobotJointLayout",
    "Real2SimTrajectory",
    "EefPoseSamples",
]


def _text(value: str, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty stripped string")
    return value


def _names(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    names = tuple(values)
    for item in names:
        _text(item, name)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique names")
    return names


def _array(value: np.ndarray, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != len(shape)
        or any(n is not None and array.shape[i] != n for i, n in enumerate(shape))
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    # Bytes-backed storage cannot be made writable again by a caller.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def _json(value: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({k: freeze(v) for k, v in item.items()})
        if isinstance(item, list):
            return tuple(freeze(v) for v in item)
        return item

    return freeze(json.loads(json.dumps(_plain(value), allow_nan=False)))


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = _array(value, (4, 4), name)
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a rigid transform")
    return matrix


@dataclass(frozen=True)
class SourcePhase:
    """An explicit half-open source-sample interval and augmentation permission.

    Args:
        name: Unique phase identifier within this source unit.
        start: Inclusive first sample index.
        stop: Exclusive final sample index.
        kind: Free motion, contact or hold annotation.
        allowed_operators: Explicit permissions, available only in free phases.
    """

    name: str
    start: int
    stop: int
    kind: Literal["free", "contact", "hold"]
    allowed_operators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "phase name")
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or not 0 <= self.start < self.stop
        ):
            raise ValueError("Phase requires integer 0 <= start < stop")
        if self.kind not in ("free", "contact", "hold"):
            raise ValueError("Unknown phase kind")
        operators = _names(self.allowed_operators, "allowed operators")
        if self.kind != "free" and operators:
            raise ValueError("Contact and hold phases cannot authorize operators")
        object.__setattr__(self, "allowed_operators", operators)


@dataclass(frozen=True)
class MimicJoint:
    """Canonical-unit relation ``joint = multiplier * parent + offset``.

    Args:
        joint: Dependent target joint name.
        parent: Driver target joint name.
        multiplier: Signed coupling ratio.
        offset: Offset in the target joint's canonical unit.
    """

    joint: str
    parent: str
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        _text(self.joint, "mimic joint")
        _text(self.parent, "mimic parent")
        if (
            self.joint == self.parent
            or not np.isfinite([self.multiplier, self.offset]).all()
        ):
            raise ValueError("Invalid mimic relation")


@dataclass(frozen=True)
class RobotJointLayout:
    """Full target joint order and explicit initial-state/augmentation ownership.

    Args:
        robot_id: Target robot identity, matching the recorded embodiment.
        robot_revision: Exact model/calibration revision required for qpos replay.
        joint_names: Full simulator joint order.
        joint_units: Canonical rad/m units in that order.
        initial_positions: Restored initial state used for compatibility checks.
        expandable_joint_names: Independent observed joints authorized for expansion.
        mimics: Acyclic dependent-joint relations in canonical units.
    """

    robot_id: str
    robot_revision: str
    joint_names: tuple[str, ...]
    joint_units: tuple[str, ...]
    initial_positions: np.ndarray
    expandable_joint_names: tuple[str, ...] = ()
    mimics: tuple[MimicJoint, ...] = ()

    def __post_init__(self) -> None:
        _text(self.robot_id, "robot_id")
        _text(self.robot_revision, "robot_revision")
        names = _names(self.joint_names, "joint_names")
        units = tuple(self.joint_units)
        if (
            not names
            or len(units) != len(names)
            or any(u not in ("rad", "m") for u in units)
        ):
            raise ValueError("Target layout requires one rad/m unit per named joint")
        expandable = _names(self.expandable_joint_names, "expandable_joint_names")
        mimics = tuple(self.mimics)
        if not all(isinstance(m, MimicJoint) for m in mimics):
            raise TypeError("mimics must contain MimicJoint values")
        mimic_names = _names(tuple(m.joint for m in mimics), "mimic joints")
        if not set(expandable) <= set(names) - set(mimic_names):
            raise ValueError("Only independent target joints may be expandable")
        available = set(names) - set(mimic_names)
        remaining = list(mimics)
        while remaining:
            ready = [m for m in remaining if m.parent in available and m.joint in names]
            if not ready:
                raise ValueError("Mimic graph has an unknown joint/parent or a cycle")
            for m in ready:
                if units[names.index(m.joint)] != units[names.index(m.parent)]:
                    raise ValueError("Mimic joints must share their parent's unit")
                available.add(m.joint)
                remaining.remove(m)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_units", units)
        object.__setattr__(self, "expandable_joint_names", expandable)
        object.__setattr__(self, "mimics", mimics)
        object.__setattr__(
            self,
            "initial_positions",
            _array(self.initial_positions, (len(names),), "initial_positions"),
        )


@dataclass(frozen=True)
class EefPoseSamples:
    """Optional EEF evidence aligned to the source joint timestamps in v1.

    ``reference_to_base`` maps reference-frame points into the robot base;
    ``link_to_tcp`` is the TCP pose in the observed link frame. Pose positions
    use ``position_unit``; transform translations always use metres. Independent
    EEF clock alignment must be performed explicitly before creating this value.

    Args:
        poses: Source-aligned ``(N,7)`` positions and quaternions.
        reference_frame: Frame in which source poses are expressed.
        observed_link: Link whose pose was recorded.
        tcp_id: Target tool frame after applying the declared link transform.
        reference_to_base: Rigid reference-to-robot-base transform.
        link_to_tcp: Rigid TCP pose expressed in the observed link frame.
        origin: Measurement lineage, kept separate from numerical consistency.
        quaternion_order: Explicit quaternion component convention.
        position_unit: Source pose translation unit.
    """

    poses: np.ndarray
    reference_frame: str
    observed_link: str
    tcp_id: str
    reference_to_base: np.ndarray
    link_to_tcp: np.ndarray
    origin: Literal["measured", "fk_derived", "transformed"]
    quaternion_order: Literal["xyzw", "wxyz"] = "xyzw"
    position_unit: Literal["m", "mm"] = "m"

    def __post_init__(self) -> None:
        for name in ("reference_frame", "observed_link", "tcp_id"):
            _text(getattr(self, name), name)
        if self.origin not in ("measured", "fk_derived", "transformed"):
            raise ValueError("Unknown EEF evidence origin")
        if self.quaternion_order not in ("xyzw", "wxyz") or self.position_unit not in (
            "m",
            "mm",
        ):
            raise ValueError("EEF units and quaternion order must be explicit")
        object.__setattr__(self, "poses", _array(self.poses, (None, 7), "EEF poses"))
        object.__setattr__(
            self,
            "reference_to_base",
            _transform(self.reference_to_base, "reference_to_base"),
        )
        object.__setattr__(
            self, "link_to_tcp", _transform(self.link_to_tcp, "link_to_tcp")
        )


@dataclass(frozen=True)
class Real2SimTrajectory:
    """Owned real-robot samples, explicit units and reviewed semantic annotations.

    Raw arrays retain their dtype; optional velocities use joint units per second.
    Neither calibration nor annotation status
    defaults to verified. This value owns no simulator, task runner or dataset.

    Args:
        source_id: Dataset or source identity.
        source_revision: Immutable source version or file digest.
        unit_id: Selected recording or semantic slice identifier.
        robot_id: Recorded embodiment identity.
        robot_revision: Recorded robot model/calibration revision.
        joint_names: Source column names, before target-order mapping.
        joint_units: Source units for each column; decode gripper encodings first.
        positions: Recorded ``(N,D)`` joint positions.
        timestamps: Strictly increasing source clock values.
        time_unit: Explicit source clock unit.
        calibration_revision: Version of the calibration evidence.
        annotation_revision: Version of the phase annotations.
        calibration_status: Reviewed input calibration state.
        annotation_status: Reviewed annotation state.
        phases: Ordered, nonoverlapping semantic intervals; gaps remain locked.
        velocities: Optional raw joint velocities in joint units per second.
        eef: Optional source-aligned EEF evidence.
        provenance: JSON-compatible preprocessing, correction and source metadata.
    """

    source_id: str
    source_revision: str
    unit_id: str
    robot_id: str
    robot_revision: str
    joint_names: tuple[str, ...]
    joint_units: tuple[str, ...]
    positions: np.ndarray
    timestamps: np.ndarray
    time_unit: Literal["s", "ms", "us", "ns"]
    calibration_revision: str
    annotation_revision: str
    calibration_status: Literal["verified", "unverified", "rejected"] = "unverified"
    annotation_status: Literal["verified", "unverified", "rejected"] = "unverified"
    phases: tuple[SourcePhase, ...] = ()
    velocities: np.ndarray | None = None
    eef: EefPoseSamples | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "source_revision",
            "unit_id",
            "robot_id",
            "robot_revision",
            "calibration_revision",
            "annotation_revision",
        ):
            _text(getattr(self, name), name)
        for status in (self.calibration_status, self.annotation_status):
            if status not in ("verified", "unverified", "rejected"):
                raise ValueError("Unknown verification status")
        names = _names(self.joint_names, "joint_names")
        units = tuple(self.joint_units)
        if (
            not names
            or len(units) != len(names)
            or any(u not in ("rad", "deg", "m", "mm") for u in units)
        ):
            raise ValueError(
                "Every source joint requires an explicit supported unit; decode grippers upstream"
            )
        positions = _array(self.positions, (None, len(names)), "positions")
        n = len(positions)
        if n < 2:
            raise ValueError("A source unit requires at least two samples")
        timestamps = _array(self.timestamps, (n,), "timestamps")
        if self.time_unit not in ("s", "ms", "us", "ns"):
            raise ValueError("Unknown timestamp unit")
        if not (timestamps[1:] > timestamps[:-1]).all():
            raise ValueError(
                "Source timestamps must strictly increase; coalesce duplicates explicitly upstream"
            )
        phases = tuple(self.phases)
        if not all(isinstance(p, SourcePhase) for p in phases):
            raise TypeError("phases must contain SourcePhase values")
        _names(tuple(p.name for p in phases), "phase names")
        previous = 0
        for phase in phases:
            if phase.start < previous or phase.stop > n:
                raise ValueError(
                    "Phases must be ordered, nonoverlapping and within the source"
                )
            previous = phase.stop
        if self.eef is not None and (
            not isinstance(self.eef, EefPoseSamples) or len(self.eef.poses) != n
        ):
            raise ValueError(
                "v1 EEF evidence must align with all source joint timestamps"
            )
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_units", units)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(
            self,
            "velocities",
            (
                None
                if self.velocities is None
                else _array(self.velocities, positions.shape, "velocities")
            ),
        )
        object.__setattr__(self, "provenance", _json(self.provenance))

    @property
    def seconds(self) -> np.ndarray:
        """Elapsed source time in seconds, subtracting integer epochs first."""
        scale = {"s": 1.0, "ms": 0.001, "us": 1e-6, "ns": 1e-9}[self.time_unit]
        return (self.timestamps - self.timestamps[0]).astype(np.float64) * scale

    def fingerprint(self) -> str:
        """Return a content digest including data, units, annotations and calibration.

        Returns:
            Stable SHA-256 hex digest over arrays and explicitly supplied provenance.
        """
        digest = hashlib.sha256()
        metadata = {
            k: getattr(self, k)
            for k in (
                "source_id",
                "source_revision",
                "unit_id",
                "robot_id",
                "robot_revision",
                "joint_names",
                "joint_units",
                "time_unit",
                "calibration_revision",
                "annotation_revision",
                "calibration_status",
                "annotation_status",
            )
        }
        metadata["phases"] = [vars(p) for p in self.phases]
        metadata["provenance"] = _plain(self.provenance)
        arrays = {"positions": self.positions, "timestamps": self.timestamps}
        if self.velocities is not None:
            arrays["velocities"] = self.velocities
        if self.eef is not None:
            metadata["eef"] = {
                k: getattr(self.eef, k)
                for k in (
                    "reference_frame",
                    "observed_link",
                    "tcp_id",
                    "origin",
                    "quaternion_order",
                    "position_unit",
                )
            }
            for k in ("poses", "reference_to_base", "link_to_tcp"):
                arrays["eef_" + k] = getattr(self.eef, k)
        digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        for name, array in sorted(arrays.items()):
            digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
