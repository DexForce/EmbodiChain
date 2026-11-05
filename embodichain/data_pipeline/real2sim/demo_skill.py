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

"""Task-specific demonstration assets independent of standard Atomic Skills."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, asdict, replace
import hashlib
import json
import math
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.utils import configclass
from .contracts import (
    FloatArray,
    JointSpec,
    NormalizationReport,
    TrajectoryTimeMap,
    _array,
    _text,
    _validate_recording,
)
from .normalize import (
    _normalize_recording,
    _rigid,
    _DEFAULT_MAX_SAMPLES,
    _DEFAULT_MAX_OUTPUT_BYTES,
)

__all__ = [
    "RobotDemonstration",
    "DemoSkillRole",
    "DemoSkillPhase",
    "DemoSkillEvent",
    "DemoSkillConstraint",
    "DemoSkillSpec",
    "DemoSkillArtifact",
    "DemoSkillReference",
    "build_demo_skill",
    "decode_demo_skill_spec",
    "retarget_demo_skill",
]


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _positive(value: float, name: str, *, zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )


@dataclass(frozen=True, eq=False)
class RobotDemonstration:
    """Observed robot motion without a standard skill label.

    State and actual submitted commands retain separate clocks/semantics. TCP
    poses use explicitly declared units, frames and calibration. Joint values
    include the recorded gripper, including its explicit mimic relationships.
    """

    source_id: str
    source_revision: str
    unit_id: str
    joint_names: tuple[str, ...]
    joint_units: tuple[Literal["rad", "deg", "m", "mm"], ...]
    qpos: FloatArray
    timestamps: FloatArray
    time_unit: Literal["s", "ms"]
    robot_geometry_sha256: str
    recording_sha256: str
    origin: Literal["measured", "synthetic_reference"] = "measured"
    qvel: FloatArray | None = None
    initial_observed_qpos: FloatArray | None = None
    commands: FloatArray | None = None
    command_representation: str | None = None
    command_alignment: (
        Literal["between_states", "post_step", "sample_aligned_reference"] | None
    ) = None
    eef_poses: FloatArray | None = None
    pose_length_unit: Literal["m", "mm"] | None = None
    quaternion_order: Literal["xyzw", "wxyz"] | None = None
    world_from_pose_frame: FloatArray | None = None
    observed_link_to_tcp: FloatArray | None = None
    calibration_revision: str = "unverified"
    annotation_revision: str = "unverified"
    pose_origin: Literal["measured", "fk_derived", "transformed"] | None = None
    source_sample_offset: int = 0

    def __post_init__(self) -> None:
        _validate_recording(self)
        _sha256(self.robot_geometry_sha256, "robot_geometry_sha256")
        _sha256(self.recording_sha256, "recording_sha256")
        if type(self.source_sample_offset) is not int or self.source_sample_offset < 0:
            raise ValueError("source_sample_offset must be nonnegative")


@dataclass(frozen=True)
class DemoSkillRole:
    """A logical rigid scene participant with frozen geometry and source pose."""

    role_id: str
    geometry_sha256: str
    source_pose: tuple[float, ...]
    reference_kind: Literal["static", "held"] = "static"

    def __post_init__(self) -> None:
        _text(self.role_id, "role_id")
        if self.role_id == "world":
            raise ValueError("world is a reserved reference")
        _sha256(self.geometry_sha256, "role geometry")
        pose = np.asarray(self.source_pose, dtype=np.float64)
        if pose.shape != (16,):
            raise ValueError("source_pose must be a flat rigid transform")
        _rigid(pose.reshape(4, 4), "source_pose")
        object.__setattr__(self, "source_pose", tuple(float(x) for x in pose))
        if self.reference_kind not in {"static", "held"}:
            raise ValueError("reference_kind must be static or held")


@dataclass(frozen=True)
class DemoSkillPhase:
    """A half-open source interval, reference frame, and explicit edit permissions."""

    name: str
    start: int
    stop: int
    kind: Literal["free", "contact", "hold"] = "free"
    reference: str = "world"
    allowed_operators: tuple[str, ...] = ()
    authority: Literal["qpos", "eef_pose"] = "qpos"

    def __post_init__(self) -> None:
        _text(self.name, "phase name")
        _text(self.reference, "phase reference")
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or self.start < 0
            or self.stop <= self.start
        ):
            raise ValueError("phase bounds must define a nonempty half-open range")
        if self.kind not in {"free", "contact", "hold"} or self.authority not in {
            "qpos",
            "eef_pose",
        }:
            raise ValueError("invalid phase kind or authority")
        operators = tuple(self.allowed_operators)
        if len(set(operators)) != len(operators) or set(operators) - {
            "joint_residual",
            "task_space_residual",
            "retime",
        }:
            raise ValueError("unknown or duplicate DemoSkill operators")
        if self.kind != "free" and operators:
            raise ValueError("contact and hold phases cannot authorize edits")
        object.__setattr__(self, "allowed_operators", operators)


@dataclass(frozen=True)
class DemoSkillEvent:
    """A hard source-sample event; its name does not imply physical evidence."""

    event_id: str
    sample_index: int
    kind: Literal["contact", "attach", "release", "reference_change", "checkpoint"]
    role_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("event sample_index must be nonnegative")
        if self.kind not in {
            "contact",
            "attach",
            "release",
            "reference_change",
            "checkpoint",
        }:
            raise ValueError("unknown event kind")
        if self.role_id is not None:
            _text(self.role_id, "event role")


@dataclass(frozen=True)
class DemoSkillConstraint:
    """A measured predicate required at entry, an event, throughout, or at exit.

    Contacts name exact roles (or the reserved robot TCP role ``tcp``). A
    relative-pose predicate uses ``reference_pose`` for subject in other frame.
    Metrics are supplied by a registered task monitor, never by planned motion.
    """

    check_id: str
    kind: Literal["contact", "no_contact", "relative_pose", "metric"]
    scope: Literal["entry", "throughout", "event", "terminal"]
    subject: str = "tcp"
    other: str = "world"
    phase_names: tuple[str, ...] = ()
    event_id: str | None = None
    reference_pose: tuple[float, ...] | None = None
    position_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = 0.05
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        _text(self.check_id, "check_id")
        if self.kind not in {
            "contact",
            "no_contact",
            "relative_pose",
            "metric",
        } or self.scope not in {"entry", "throughout", "event", "terminal"}:
            raise ValueError("invalid constraint kind or scope")
        if (self.scope == "event") != (self.event_id is not None):
            raise ValueError("event constraints require exactly one event_id")
        _text(self.subject, "constraint subject")
        _text(self.other, "constraint other")
        object.__setattr__(self, "phase_names", tuple(self.phase_names))
        _positive(self.position_tolerance_m, "position_tolerance_m")
        _positive(self.rotation_tolerance_rad, "rotation_tolerance_rad")
        if (
            not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum > self.maximum
        ):
            raise ValueError("metric bounds must be finite and ordered")
        if self.kind == "relative_pose":
            if self.reference_pose is None or len(self.reference_pose) != 16:
                raise ValueError("relative_pose requires a flat reference_pose")
            pose = _rigid(
                np.asarray(self.reference_pose).reshape(4, 4), "reference_pose"
            )
            object.__setattr__(
                self, "reference_pose", tuple(float(x) for x in pose.flat)
            )
        elif self.reference_pose is not None:
            raise ValueError("reference_pose is only supported for relative_pose")


@configclass
class DemoSkillSpec:
    """Task-owned skill contract and conservative adaptation limits."""

    asset_id: str = ""
    revision: str = "1"
    instruction: str = ""
    phases: tuple[DemoSkillPhase, ...] = ()
    roles: tuple[DemoSkillRole, ...] = ()
    events: tuple[DemoSkillEvent, ...] = ()
    constraints: tuple[DemoSkillConstraint, ...] = ()
    entry_tolerance: float = 0.001
    allow_entry_connection: bool = False
    scene_position_tolerance_m: float = 0.001
    scene_rotation_tolerance_rad: float = 0.01
    max_translation_m: float = 0.0
    max_yaw_rad: float = 0.0
    max_duration_scale: float = 1.0
    initial_held_role: str | None = None
    terminal_held_role: str | None = None
    release_on_exit: bool = False

    def validate_contract(self) -> None:
        """Reject incomplete contracts before normalization or live assembly."""
        for name in ("asset_id", "revision", "instruction"):
            _text(getattr(self, name), name)
        for name, cls in (
            ("phases", DemoSkillPhase),
            ("roles", DemoSkillRole),
            ("events", DemoSkillEvent),
            ("constraints", DemoSkillConstraint),
        ):
            items = tuple(getattr(self, name))
            if any(type(item) is not cls for item in items):
                raise TypeError(f"{name} must contain {cls.__name__} values")
            identifiers = [
                getattr(
                    item,
                    {
                        "phases": "name",
                        "roles": "role_id",
                        "events": "event_id",
                        "constraints": "check_id",
                    }[name],
                )
                for item in items
            ]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"duplicate {name} identifiers")
            setattr(self, name, items)
        if (
            not self.phases
            or self.phases[0].start != 0
            or any(a.stop != b.start for a, b in zip(self.phases, self.phases[1:]))
        ):
            raise ValueError(
                "phases must cover a contiguous trajectory starting at zero"
            )
        roles = {role.role_id: role for role in self.roles}
        names = {phase.name for phase in self.phases}
        events = {event.event_id for event in self.events}
        for phase in self.phases:
            if phase.reference != "world" and phase.reference not in roles:
                raise ValueError("phase references an undeclared role")
        for event in self.events:
            if event.sample_index >= self.phases[-1].stop or (
                event.role_id is not None and event.role_id not in roles
            ):
                raise ValueError(
                    "event is outside the trajectory or has an unknown role"
                )
        for item in self.constraints:
            if set(item.phase_names) - names or (
                item.event_id is not None and item.event_id not in events
            ):
                raise ValueError("constraint references an unknown phase or event")
            if {item.subject, item.other} - {"tcp", "world"} - set(roles):
                raise ValueError("constraint references an unknown role")
            if item.phase_names and item.scope != "throughout":
                raise ValueError(
                    "phase_names are supported only for throughout constraints"
                )
        if not any(item.scope == "terminal" for item in self.constraints):
            raise ValueError("a measured terminal success constraint is required")
        for event in self.events:
            if event.kind in {"attach", "release", "contact"} and not any(
                item.scope == "event" and item.event_id == event.event_id
                for item in self.constraints
            ):
                raise ValueError(
                    "physical events require an explicit measured event constraint"
                )
        for event in self.events:
            if event.kind in {"attach", "release", "contact"}:
                if event.role_id is None:
                    raise ValueError("physical events require an explicit role")
                predicates = [
                    item
                    for item in self.constraints
                    if item.scope == "event" and item.event_id == event.event_id
                ]
                kind = "no_contact" if event.kind == "release" else "contact"
                if not any(
                    item.kind == kind and event.role_id in {item.subject, item.other}
                    for item in predicates
                ):
                    raise ValueError(
                        "physical events require measured contact or separation evidence"
                    )
                if event.kind == "attach" and not any(
                    item.kind == "relative_pose"
                    and item.subject == "tcp"
                    and item.other == event.role_id
                    for item in predicates
                ):
                    raise ValueError(
                        "attachment events require a measured object-to-TCP relation"
                    )
        for phase in self.phases:
            if phase.kind == "contact" and not any(
                item.kind == "contact"
                and item.scope == "throughout"
                and (not item.phase_names or phase.name in item.phase_names)
                for item in self.constraints
            ):
                raise ValueError(
                    "contact phases require a throughout contact constraint"
                )
        _positive(self.entry_tolerance, "entry_tolerance")
        if type(self.allow_entry_connection) is not bool or (
            self.allow_entry_connection and self.phases[0].kind != "free"
        ):
            raise ValueError(
                "entry connections require explicit permission and a free initial phase"
            )
        _positive(self.scene_position_tolerance_m, "scene_position_tolerance_m")
        _positive(self.scene_rotation_tolerance_rad, "scene_rotation_tolerance_rad")
        _positive(self.max_translation_m, "max_translation_m", zero=True)
        _positive(self.max_yaw_rad, "max_yaw_rad", zero=True)
        _positive(self.max_duration_scale, "max_duration_scale")
        if self.max_duration_scale < 1:
            raise ValueError("max_duration_scale must be at least one")
        if type(self.release_on_exit) is not bool or (
            self.release_on_exit and self.terminal_held_role is not None
        ):
            raise ValueError("terminal attach and release are mutually exclusive")
        if self.initial_held_role is not None and self.initial_held_role not in roles:
            raise ValueError("initial_held_role must name a declared role")
        if (
            self.initial_held_role is not None
            and self.terminal_held_role is None
            and not self.release_on_exit
        ):
            role = self.initial_held_role
            if not any(
                c.scope == "terminal"
                and c.kind == "relative_pose"
                and c.subject == "tcp"
                and c.other == role
                for c in self.constraints
            ):
                raise ValueError(
                    "retained input attachment requires measured terminal relation"
                )
        if self.terminal_held_role is not None and self.terminal_held_role not in roles:
            raise ValueError("terminal_held_role must name a declared role")
        if self.terminal_held_role is not None:
            role = self.terminal_held_role
            if not any(
                c.scope == "terminal"
                and c.kind == "relative_pose"
                and c.subject == "tcp"
                and c.other == role
                for c in self.constraints
            ):
                raise ValueError(
                    "terminal attachment requires measured object-to-TCP relation"
                )
            if not any(
                c.scope == "terminal"
                and c.kind == "contact"
                and {c.subject, c.other} == {"tcp", role}
                for c in self.constraints
            ):
                raise ValueError("terminal attachment requires measured contact")
        if self.release_on_exit:
            held_roles = [
                role.role_id for role in self.roles if role.reference_kind == "held"
            ]
            if len(held_roles) != 1 or not any(
                c.scope == "terminal"
                and c.kind == "no_contact"
                and {c.subject, c.other} == {"tcp", held_roles[0]}
                for c in self.constraints
            ):
                raise ValueError(
                    "terminal release requires one explicit held role and measured no-contact"
                )


def decode_demo_skill_spec(value: object) -> DemoSkillSpec:
    """Decode a closed JSON/YAML skill contract.

    Args:
        value: Mapping with only DemoSkillSpec fields and typed declarations.

    Returns:
        Validated task configuration.
    """
    if type(value) is not dict or set(value) - {
        field.name for field in fields(DemoSkillSpec)
    }:
        raise ValueError("unknown DemoSkillSpec fields")
    values = dict(value)
    for name, cls in (
        ("phases", DemoSkillPhase),
        ("roles", DemoSkillRole),
        ("events", DemoSkillEvent),
        ("constraints", DemoSkillConstraint),
    ):
        values[name] = tuple(cls(**item) for item in values.get(name, ()))
    result = DemoSkillSpec(**values)
    result.validate()
    result.validate_contract()
    return result


@dataclass(frozen=True, eq=False)
class DemoSkillReference:
    """Explicitly adapted nominal motion, separate from measured source state.

    ``source_mask`` distinguishes generated connections from source-backed
    samples. The mapping addresses raw source samples; generated samples retain
    bracketing ancestry without claiming to have been observed in the recording.
    Preparation evidence is immutable JSON and has its own content identity.
    """

    qpos: FloatArray
    timestamps: FloatArray
    phases: tuple[DemoSkillPhase, ...]
    events: tuple[DemoSkillEvent, ...]
    time_map: TrajectoryTimeMap
    source_mask: np.ndarray
    preparation_json: str
    world_tcp: FloatArray | None = None
    initial_qpos: FloatArray | None = None
    role_poses: tuple[tuple[str, tuple[float, ...]], ...] = ()
    input_qpos: FloatArray | None = None

    def __post_init__(self) -> None:
        poses = []
        for role, values in self.role_poses:
            _text(role, "reference role")
            pose = _rigid(np.asarray(values).reshape(4, 4), "reference role pose")
            poses.append((role, tuple(float(x) for x in pose.flat)))
        if len({role for role, _ in poses}) != len(poses):
            raise ValueError("duplicate nominal reference roles")
        object.__setattr__(self, "role_poses", tuple(poses))
        qpos = _array(self.qpos, "reference qpos", 2)
        if self.input_qpos is not None:
            original = _array(self.input_qpos, "input nominal qpos", 2)
            if original.shape != qpos.shape:
                raise ValueError("input and corrected nominal positions must align")
            object.__setattr__(self, "input_qpos", original)
        times = _array(self.timestamps, "reference timestamps", 1)
        if (
            len(times) < 2
            or qpos.shape[0] != len(times)
            or len(self.time_map.left) != len(times)
        ):
            raise ValueError(
                "nominal reference arrays and raw source mapping must align"
            )
        if (
            times[0] != 0
            or np.any(np.diff(times) <= 0)
            or not np.allclose(np.diff(times), times[1], atol=1e-9, rtol=0)
        ):
            raise ValueError(
                "nominal reference requires a uniform zero-based control clock"
            )
        mask = np.array(self.source_mask, copy=True)
        if mask.dtype != np.bool_ or mask.shape != (len(times),):
            raise ValueError("source_mask must identify every nominal reference sample")
        mask.setflags(write=False)
        if self.initial_qpos is not None:
            initial = _array(self.initial_qpos, "reference initial_qpos", 1)
            if initial.shape != (qpos.shape[1],):
                raise ValueError("reference entry must match the robot layout")
            object.__setattr__(self, "initial_qpos", initial)
        evidence = json.loads(self.preparation_json)
        if (
            type(evidence) is not dict
            or not evidence
            or evidence.get("origin") != "adapted_nominal"
        ):
            raise ValueError(
                "reference preparation must explicitly declare adapted_nominal origin"
            )
        for name, value in (
            ("qpos", qpos),
            ("timestamps", times),
            ("source_mask", mask),
            ("phases", tuple(self.phases)),
            ("events", tuple(self.events)),
        ):
            object.__setattr__(self, name, value)
        if self.world_tcp is not None:
            tcp = _array(self.world_tcp, "reference world_tcp", 3)
            if tcp.shape != (len(times), 4, 4):
                raise ValueError("nominal TCP reference must align with qpos")
            for pose in tcp:
                _rigid(pose, "reference TCP")
            object.__setattr__(self, "world_tcp", tcp)


@dataclass(frozen=True, eq=False)
class DemoSkillArtifact:
    """Immutable normalized exemplar, raw source, and copied declarative contract.

    Preparation never grants physical qualification. ``spec`` returns an owned
    configuration so mutations cannot alter the content-addressed asset.
    """

    demonstration: RobotDemonstration
    spec_json: str
    joint_names: tuple[str, ...]
    qpos: FloatArray
    initial_qpos: FloatArray
    timestamps: FloatArray
    world_tcp: FloatArray | None
    report: NormalizationReport
    joint_specs: tuple[JointSpec, ...]
    event_tolerance_s: float = 1e-6
    reference: DemoSkillReference | None = None

    def __post_init__(self) -> None:
        spec = self.spec
        if self.reference is not None:
            ref = self.reference
            if type(ref) is not DemoSkillReference or ref.qpos.shape[1] != len(
                self.joint_names
            ):
                raise ValueError("nominal reference must match the robot joint layout")
            if max(ref.time_map.right, default=-1) >= len(
                self.demonstration.timestamps
            ):
                raise ValueError("nominal reference mapping escapes the raw recording")
            if ref.role_poses and {name for name, _ in ref.role_poses} != {
                role.role_id for role in spec.roles
            }:
                raise ValueError(
                    "nominal scene references must bind exactly the skill roles"
                )
            source_phases = {phase.name: phase for phase in spec.phases}
            # Extra generated connection phases have no source edit authority.
            for phase in ref.phases:
                original = source_phases.get(phase.name)
                if original is None:
                    if phase.kind != "free" or phase.allowed_operators:
                        raise ValueError(
                            "generated connections cannot authorize source edits"
                        )
                elif (phase.kind, phase.reference, phase.authority) != (
                    original.kind,
                    original.reference,
                    original.authority,
                ) or not set(phase.allowed_operators) <= set(
                    original.allowed_operators
                ):
                    raise ValueError(
                        "prepared reference cannot widen source phase authority"
                    )
            if not set(source_phases) <= {phase.name for phase in ref.phases}:
                raise ValueError("prepared reference dropped a source phase")
            if {
                (event.event_id, event.kind, event.role_id) for event in ref.events
            } != {(event.event_id, event.kind, event.role_id) for event in spec.events}:
                raise ValueError(
                    "prepared reference must preserve declared source events"
                )
            if [phase.name for phase in ref.phases if phase.name in source_phases] != [
                phase.name for phase in spec.phases
            ]:
                raise ValueError("prepared reference must preserve source phase order")
            original_events = dict(
                zip(
                    self.report.time_map.execution_event_indices,
                    self.report.time_map.source_event_indices,
                )
            )
            mapped_events = dict(
                zip(
                    ref.time_map.execution_event_indices,
                    ref.time_map.source_event_indices,
                )
            )
            expected_events = {
                event.event_id: original_events[event.sample_index]
                for event in spec.events
            }
            for event in ref.events:
                if (
                    mapped_events.get(event.sample_index)
                    != expected_events[event.event_id]
                    or not ref.source_mask[event.sample_index]
                ):
                    raise ValueError(
                        "prepared reference event does not identify its original source frame"
                    )
            execution_spec = self.execution_spec
            execution_spec.validate_contract()
            if execution_spec.phases[-1].stop != len(ref.timestamps):
                raise ValueError(
                    "reference phases must cover its complete control grid"
                )
            if not np.isclose(
                ref.timestamps[1],
                self.timestamps[1] - self.timestamps[0],
                atol=1e-9,
                rtol=0,
            ):
                raise ValueError("reference and source control clocks must agree")
        names = tuple(self.joint_names)
        qpos = _array(self.qpos, "qpos", 2)
        times = _array(self.timestamps, "timestamps", 1)
        initial = _array(self.initial_qpos, "initial_qpos", 1)
        if (
            len(times) < 2
            or qpos.shape != (len(times), len(names))
            or initial.shape != (len(names),)
            or len(set(names)) != len(names)
        ):
            raise ValueError("artifact joint and time arrays do not align")
        if np.any(np.diff(times) <= 0) or not np.allclose(
            np.diff(times), times[1] - times[0], atol=1e-9, rtol=0
        ):
            raise ValueError("artifact requires a uniform control clock")
        if spec.phases[-1].stop != len(times):
            raise ValueError("artifact phases must cover all normalized samples")
        for key, value in (
            ("qpos", qpos),
            ("timestamps", times),
            ("initial_qpos", initial),
            ("joint_names", names),
        ):
            object.__setattr__(self, key, value)
        if self.world_tcp is not None:
            tcp = _array(self.world_tcp, "world_tcp", 3)
            if tcp.shape != (len(times), 4, 4):
                raise ValueError("world_tcp must align with artifact samples")
            for pose in tcp:
                _rigid(pose, "world_tcp")
            object.__setattr__(self, "world_tcp", tcp)

    @property
    def execution_spec(self) -> DemoSkillSpec:
        """Return the execution timeline while retaining the original source spec."""
        spec = self.spec
        if self.reference is not None:
            spec.phases = self.reference.phases
            spec.events = self.reference.events
        return spec

    @property
    def execution_qpos(self) -> FloatArray:
        """Return the adapted reference or the normalized source positions."""
        return self.qpos if self.reference is None else self.reference.qpos

    @property
    def execution_timestamps(self) -> FloatArray:
        """Return the exact control clock of the selected nominal reference."""
        return self.timestamps if self.reference is None else self.reference.timestamps

    @property
    def execution_tcp(self) -> FloatArray | None:
        """Return the selected nominal TCP path, retaining raw normalized poses."""
        return self.world_tcp if self.reference is None else self.reference.world_tcp

    @property
    def spec(self) -> DemoSkillSpec:
        """Return an independent validated configuration snapshot."""
        return decode_demo_skill_spec(json.loads(self.spec_json))

    @property
    def content_sha256(self) -> str:
        """Hash the full raw source, normalized evidence and contract."""
        digest = hashlib.sha256(self.spec_json.encode())
        for value in (self.demonstration, self.report, self):
            for field in fields(value):
                if field.name in {"demonstration", "spec_json", "report", "reference"}:
                    continue
                item = getattr(value, field.name)
                digest.update(field.name.encode())
                if isinstance(item, np.ndarray):
                    digest.update(str((item.dtype.str, item.shape)).encode())
                    digest.update(item.tobytes())
                else:
                    digest.update(repr(item).encode())
        time_map = self.report.time_map
        for name in (
            "left",
            "right",
            "weight",
            "source_event_indices",
            "execution_event_indices",
        ):
            digest.update(getattr(time_map, name).tobytes())
        if self.reference is not None:
            ref = self.reference
            for field in fields(ref):
                value = getattr(ref, field.name)
                if (field.name == "role_poses" and not value) or (
                    field.name == "input_qpos" and value is None
                ):
                    continue
                digest.update(field.name.encode())
                if isinstance(value, np.ndarray):
                    digest.update(str((value.dtype.str, value.shape)).encode())
                    digest.update(value.tobytes())
                elif field.name != "time_map":
                    digest.update(repr(value).encode())
            for name in (
                "left",
                "right",
                "weight",
                "source_event_indices",
                "execution_event_indices",
            ):
                digest.update(getattr(ref.time_map, name).tobytes())
            digest.update(repr(ref.time_map.max_event_error_s).encode())
        return digest.hexdigest()


def build_demo_skill(
    demonstration: RobotDemonstration,
    spec: DemoSkillSpec,
    *,
    joints: Sequence[JointSpec],
    control_dt: float,
    event_tolerance_s: float = 1e-6,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
) -> DemoSkillArtifact:
    """Normalize one exemplar while preserving hard events and source evidence.

    Args:
        demonstration: Raw observed state and independently declared commands.
        spec: Manually confirmed phases, roles, constraints and permissions.
        joints: Complete ordered joint and mimic declarations.
        control_dt: Destination command period in seconds.
        event_tolerance_s: Maximum hard-event displacement on the control grid.
        max_samples: Caller-owned normalized sample limit.
        max_output_bytes: Caller-owned numeric working-array budget.

    Returns:
        Prepared asset; physical qualification still requires measured execution.
    """
    if type(demonstration) is not RobotDemonstration or type(spec) is not DemoSkillSpec:
        raise TypeError("expected RobotDemonstration and DemoSkillSpec")
    joints = tuple(joints)
    # Copy through the declarative format, including all nested declarations.
    spec = decode_demo_skill_spec(asdict(spec))
    if spec.phases[-1].stop != len(demonstration.timestamps):
        raise ValueError("source phases must cover the complete demonstration")
    values = _normalize_recording(
        demonstration,
        spec.phases,
        joints=joints,
        control_dt=control_dt,
        event_indices=tuple(event.sample_index for event in spec.events),
        event_tolerance_s=event_tolerance_s,
        max_samples=max_samples,
        max_output_bytes=max_output_bytes,
    )
    time_map = values["report"].time_map
    event_grid = dict(
        zip(time_map.source_event_indices, time_map.execution_event_indices)
    )
    spec.phases = values["phases"]
    spec.events = tuple(
        replace(event, sample_index=int(event_grid[event.sample_index]))
        for event in spec.events
    )
    if values["world_tcp"] is None and any(
        phase.authority == "eef_pose" for phase in spec.phases
    ):
        raise ValueError("eef_pose authority requires calibrated source TCP poses")
    if (spec.max_translation_m or spec.max_yaw_rad) and values["world_tcp"] is None:
        raise ValueError("scene transfer requires calibrated source TCP poses")
    return DemoSkillArtifact(
        demonstration,
        json.dumps(
            asdict(spec), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        values["joint_names"],
        values["qpos"],
        values["initial_qpos"],
        values["timestamps"],
        values["world_tcp"],
        values["report"],
        tuple(joints),
        event_tolerance_s,
    )


def retarget_demo_skill(
    artifact: DemoSkillArtifact,
    role_poses: dict[str, np.ndarray],
    *,
    geometry_sha256: dict[str, str],
) -> FloatArray | None:
    """Ground frozen local TCP references in a same-geometry scene snapshot.

    Args:
        artifact: Immutable source asset.
        role_poses: One world transform per logical role, frozen at call entry.
        geometry_sha256: Independently captured live geometry identity per role.

    Returns:
        Owned world TCP targets, or None for an unchanged qpos-only asset.
    """
    spec = artifact.execution_spec
    roles = {role.role_id: role for role in spec.roles}
    if set(role_poses) != set(roles) or set(geometry_sha256) != set(roles):
        raise ValueError("all scene roles require exact pose and geometry bindings")
    transforms = {"world": np.eye(4)}
    for name, role in roles.items():
        if geometry_sha256[name] != role.geometry_sha256:
            raise ValueError(f"geometry mismatch for role {name!r}")
        reference_poses = (
            {} if artifact.reference is None else dict(artifact.reference.role_poses)
        )
        source = np.asarray(reference_poses.get(name, role.source_pose)).reshape(4, 4)
        target = _rigid(role_poses[name], name)
        delta = target @ np.linalg.inv(source)
        angle = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
        if (
            np.linalg.norm(target[:3, 3] - source[:3, 3])
            <= spec.scene_position_tolerance_m
            and np.linalg.norm(angle) <= spec.scene_rotation_tolerance_rad
        ):
            transforms[name] = np.eye(4)
            continue
        # Compare position changes at the role origin, not the world origin.
        if (
            np.linalg.norm(target[:3, 3] - source[:3, 3])
            > spec.max_translation_m + 1e-7
            or np.linalg.norm(angle[:2]) > 1e-6
            or abs(angle[2]) > spec.max_yaw_rad + 1e-7
        ):
            raise ValueError(f"role {name!r} exceeds declared XY/yaw transfer bounds")
        if abs(target[2, 3] - source[2, 3]) > 1e-6:
            raise ValueError("DemoSkill scene transfer preserves world height")
        transforms[name] = delta
    moved = any(
        not np.allclose(delta, np.eye(4), atol=1e-7, rtol=0)
        for delta in transforms.values()
    )
    if artifact.execution_tcp is None:
        if moved:
            raise ValueError("qpos-only demonstrations require the recorded scene")
        return None
    targets = np.array(artifact.execution_tcp, copy=True)
    for phase in spec.phases:
        delta = transforms[phase.reference]
        if phase.authority == "qpos" and not np.allclose(
            delta, np.eye(4), atol=1e-7, rtol=0
        ):
            raise ValueError("a moved reference requires eef_pose authority")
        targets[phase.start : phase.stop] = delta @ targets[phase.start : phase.stop]
    return targets
