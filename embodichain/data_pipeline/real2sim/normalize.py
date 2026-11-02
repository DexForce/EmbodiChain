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

"""Normalize annotated source units without constructing a simulator."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contracts import (
    JointSpec,
    NormalizationReport,
    NormalizedReal2SimUnit,
    PhaseAnnotation,
    Real2SimTrajectory,
    TrajectoryTimeMap,
)

__all__ = ["normalize_trajectory"]

_DEFAULT_MAX_SAMPLES = 1_000_000
_DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def _rigid(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a proper rigid transform in metres")
    return matrix


def _joint_columns(
    source: Real2SimTrajectory,
    joints: Sequence[JointSpec],
    tolerance: float,
    samples: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int | None, ...], tuple[str, ...]]:
    target = tuple(joints)
    samples = source.qpos if samples is None else samples
    if not target or any(not isinstance(joint, JointSpec) for joint in target):
        raise ValueError("joints must contain typed target joints")
    target_names = tuple(joint.name for joint in target)
    if len(set(target_names)) != len(target_names):
        raise ValueError("target joint names must be unique")
    if set(source.joint_names) - set(target_names):
        raise ValueError("source contains joints absent from the target layout")
    source_indices = {name: index for index, name in enumerate(source.joint_names)}
    target_specs = {joint.name: joint for joint in target}
    values: dict[str, np.ndarray] = {}
    corrections: list[str] = []
    for name, index in source_indices.items():
        joint = target_specs[name]
        unit = source.joint_units[index]
        if joint.kind == "revolute" and unit not in {"rad", "deg"}:
            raise ValueError(f"joint {name!r} requires angular units")
        if joint.kind == "prismatic" and unit not in {"m", "mm"}:
            raise ValueError(f"joint {name!r} requires linear units")
        scale = {"rad": 1.0, "deg": math.pi / 180, "m": 1.0, "mm": 0.001}[unit]
        values[name] = samples[:, index] * scale
        if scale != 1:
            corrections.append(f"joint {name}: {unit} converted to canonical units")

    resolved: dict[str, np.ndarray] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> np.ndarray:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError("mimic joint cycle")
        if name not in target_specs:
            raise ValueError(f"unknown mimic driver {name!r}")
        visiting.add(name)
        joint = target_specs[name]
        if joint.mimic_driver is not None:
            if joint.mimic_driver not in target_specs:
                raise ValueError(f"unknown mimic driver {joint.mimic_driver!r}")
            if target_specs[joint.mimic_driver].kind != joint.kind:
                raise ValueError("mimic joint and driver must have matching dimensions")
            expected = (
                resolve(joint.mimic_driver) * joint.mimic_multiplier
                + joint.mimic_offset
            )
            if name in values and not np.allclose(
                values[name], expected, atol=tolerance, rtol=0
            ):
                raise ValueError(
                    f"observed mimic joint {name!r} conflicts with its driver"
                )
            value = expected
            corrections.append(
                f"joint {name}: derived from mimic driver {joint.mimic_driver}"
            )
        elif name in values:
            value = values[name]
        elif joint.initial_hold is not None:
            value = np.full(len(samples), joint.initial_hold, dtype=np.float64)
            corrections.append(f"joint {name}: explicit initial hold")
        else:
            raise ValueError(f"missing independent joint {name!r}")
        visiting.remove(name)
        resolved[name] = value
        return value

    columns = np.stack([resolve(name) for name in target_names], axis=1)
    return (
        columns,
        tuple(source_indices.get(name) for name in target_names),
        tuple(corrections),
    )


def normalize_trajectory(
    source: Real2SimTrajectory,
    *,
    joints: Sequence[JointSpec],
    control_dt: float,
    event_tolerance_s: float = 1e-6,
    mimic_tolerance: float = 1e-6,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
) -> NormalizedReal2SimUnit:
    """Convert a labeled motion to canonical joints, frames and control time.

    Phase starts and the final sample are hard event anchors. They must lie on
    the destination control grid within ``event_tolerance_s``; short phases are
    rejected instead of erased. Duplicate stationary samples are coalesced only
    when neither sample is an event and all recorded payloads agree.

    Args:
        source: Explicitly labeled source trajectory and frame declarations.
        joints: Complete ordered simulator joint layout and mimic contracts.
        control_dt: Destination command period in seconds.
        event_tolerance_s: Maximum time displacement of a hard event.
        mimic_tolerance: Canonical-unit tolerance for redundant mimic evidence.
        max_samples: Maximum generated control-grid sample count.
        max_output_bytes: Budget for a conservative estimate of generated numeric
            arrays and working buffers, excluding the already loaded source.
            This is not a process-RSS or third-party allocator peak bound.

    Returns:
        Canonical unit with an auditable source-to-grid mapping.

    Raises:
        ValueError: The source, calibration, timing or phase contract is invalid.
    """
    if not isinstance(source, Real2SimTrajectory):
        raise TypeError("source must be a Real2SimTrajectory")
    for value, name, positive in (
        (control_dt, "control_dt", True),
        (event_tolerance_s, "event_tolerance_s", False),
        (mimic_tolerance, "mimic_tolerance", False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
            )
    for value, name, minimum in (
        (max_samples, "max_samples", 2),
        (max_output_bytes, "max_output_bytes", 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    joints = tuple(joints)
    times = np.asarray(source.timestamps) * (0.001 if source.time_unit == "ms" else 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        times = times - times[0]
    if not np.isfinite(times).all():
        raise ValueError("normalized source duration must be finite")
    if np.any(np.diff(times) < 0):
        raise ValueError("source timestamps must not move backwards")
    # Reject output expansion before any integer cast or control-grid allocation.
    # A finite timestamp divided by a tiny positive dt may overflow to infinity.
    with np.errstate(over="ignore", invalid="ignore"):
        last_event = np.rint(times[-1] / control_dt)
    if (
        not np.isfinite(last_event)
        or last_event >= max_samples
        or last_event >= np.iinfo(np.int64).max
    ):
        raise ValueError("normalized control grid exceeds max_samples budget")
    count = int(last_event) + 1
    # Joint interpolation can retain indexed inputs, weighted products and an
    # owned normalized copy. Pose interpolation additionally retains rotations,
    # transforms and source/event maps. Account for source-joint conversion too.
    joint_count = len(joints)
    per_sample_values = 8 * joint_count + (96 if source.eef_poses is not None else 16)
    source_values = len(times) * (
        3 * joint_count + (16 if source.eef_poses is not None else 0)
    )
    estimated_bytes = 8 * (count * per_sample_values + source_values)
    if estimated_bytes > max_output_bytes:
        raise ValueError(
            f"normalized arrays require an estimated {estimated_bytes} bytes, "
            f"exceeding max_output_bytes budget {max_output_bytes}"
        )
    qpos, indices, correction_tuple = _joint_columns(source, joints, mimic_tolerance)
    corrections = list(correction_tuple)
    if source.initial_observed_qpos is not None:
        initial, _, _ = _joint_columns(
            source, joints, mimic_tolerance, source.initial_observed_qpos[None, :]
        )
        initial_state_origin = "observed"
    else:
        initial = qpos[:1]
        initial_state_origin = (
            "first_measured_sample"
            if source.origin == "measured"
            else "reference_assumption"
        )
    corrections.append(f"initial state source: {initial_state_origin}")
    events = np.array(
        [phase.start for phase in source.phases] + [len(times) - 1], dtype=np.int64
    )
    if len(np.unique(events)) != len(events):
        raise ValueError("each phase and its final event must be representable")
    retained = np.ones(len(times), dtype=bool)
    for index in np.flatnonzero(np.diff(times) == 0) + 1:
        if index in events or index - 1 in events:
            raise ValueError("duplicate timestamps at an event cannot be coalesced")
        if not np.array_equal(qpos[index], qpos[index - 1]):
            raise ValueError("zero-duration position change")
        if source.commands is not None:
            raise ValueError(
                "duplicate timestamps with command evidence cannot be coalesced"
            )
        for name in ("eef_poses",):
            values = getattr(source, name)
            if values is not None and not np.array_equal(
                values[index], values[index - 1]
            ):
                raise ValueError(f"duplicate timestamps contain a {name} change")
        retained[index] = False
        corrections.append(f"coalesced stationary duplicate sample {index}")
    source_indices = np.flatnonzero(retained)
    source_times = times[retained]
    source_qpos = qpos[retained]
    event_grid = np.rint(times[events] / control_dt).astype(np.int64)
    error = np.abs(event_grid * control_dt - times[events])
    if np.any(error > event_tolerance_s):
        raise ValueError("hard event does not lie on the control grid")
    if np.any(np.diff(event_grid) <= 0):
        raise ValueError("control grid erases a phase or hard event")
    grid = np.arange(count, dtype=np.float64) * control_dt
    if grid[-1] > source_times[-1] + event_tolerance_s:
        raise ValueError("control grid extends past the source trajectory")
    # The exact final event may differ by a declared sub-tolerance amount.
    query = np.minimum(grid, source_times[-1])
    right = np.searchsorted(source_times, query, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.isclose(source_times[right], query, rtol=0, atol=1e-12)
    left[exact] = right[exact]
    denominator = source_times[right] - source_times[left]
    weight = np.divide(
        query - source_times[left],
        denominator,
        out=np.zeros_like(query),
        where=denominator != 0,
    )
    result_qpos = (
        source_qpos[left] * (1 - weight[:, None]) + source_qpos[right] * weight[:, None]
    )
    # Preserve event positions even when the accepted clock mismatch is nonzero.
    result_qpos[event_grid] = qpos[events]
    left[event_grid] = np.searchsorted(source_indices, events)
    right[event_grid] = left[event_grid]
    weight[event_grid] = 0
    result_phases = []
    for idx, phase in enumerate(source.phases):
        stop = int(event_grid[idx + 1]) if idx + 1 < len(source.phases) else count
        result_phases.append(
            PhaseAnnotation(
                phase.name,
                int(event_grid[idx]),
                stop,
                phase.kind,
                phase.reference,
                phase.allowed_operators,
                phase.authority,
            )
        )

    world_tcp = None
    if source.eef_poses is not None:
        frame = _rigid(source.world_from_pose_frame, "world_from_pose_frame")
        tool = _rigid(source.observed_link_to_tcp, "observed_link_to_tcp")
        raw = np.asarray(source.eef_poses)[retained]
        quat = raw[:, 3:]
        if source.quaternion_order == "wxyz":
            quat = quat[:, [1, 2, 3, 0]]
        norms = np.linalg.norm(quat, axis=1)
        if np.any(np.abs(norms - 1) > 1e-5):
            raise ValueError("EEF quaternions must be normalized")
        orientations = Rotation.from_quat(quat)
        out_rot = Slerp(source_times, orientations)(query).as_matrix()
        translation = raw[:, :3] * (0.001 if source.pose_length_unit == "mm" else 1.0)
        out_pos = (
            translation[left] * (1 - weight[:, None])
            + translation[right] * weight[:, None]
        )
        transforms = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        transforms[:, :3, :3] = out_rot
        transforms[:, :3, 3] = out_pos
        # An event with a small clock error must retain its exact source pose.
        transforms[event_grid, :3, :3] = orientations.as_matrix()[left[event_grid]]
        transforms[event_grid, :3, 3] = translation[left[event_grid]]
        world_tcp = frame @ transforms @ tool
    for name in ("object_pose", "target_pose"):
        value = getattr(source, name)
        if value is not None:
            _rigid(value, name)
    time_map = TrajectoryTimeMap(
        source_indices[left],
        source_indices[right],
        weight,
        events,
        event_grid,
        float(error.max()),
    )
    report = NormalizationReport(
        indices, tuple(corrections), time_map, initial_state_origin
    )
    return NormalizedReal2SimUnit(
        source=source,
        joint_names=tuple(joint.name for joint in joints),
        qpos=result_qpos,
        initial_qpos=initial[0],
        timestamps=grid,
        phases=tuple(result_phases),
        world_tcp=world_tcp,
        raw_qvel=source.qvel,
        report=report,
    )
