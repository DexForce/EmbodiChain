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
"""Compile small semantic residuals; model output never supplies joint trajectories."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .inputs import RobotModel
from .trajectory_tools import (
    TrajectoryTools,
    validate_segment_revision,
)

__all__ = ["CORRECTION_LIMITS", "apply_trajectory_corrections"]

# Edit-authority limits, not task-success thresholds. The model cannot enlarge them.
CORRECTION_LIMITS = {
    "max_edits": 4,
    "translation_m": 0.10,
    "rotation_rad": float(np.deg2rad(25)),
    "gripper_width_delta_m": 0.03,
    "gripper_shift_s": 0.5,
}


def apply_trajectory_corrections(
    model: RobotModel,
    current: np.ndarray,
    original: np.ndarray,
    timestamps: np.ndarray,
    phases: list[dict],
    corrections: list[dict],
) -> tuple[np.ndarray, dict]:
    """Convert at most four local residual anchors into source-shaped commands.

    Each edit names contiguous phase_ids, a side and one interior anchor_sample.
    translation_m and rotation_vector_rad offset that sample's current world TCP.
    The residual is interpolated in world TCP space on the recorded time grid.
    Real2Sim's approximate IK fits each interior sample, keeping both boundaries.
    Optional gripper_width_delta_m and gripper_shift_s adjust the measured aperture
    profile with the same envelope. Positive time shift delays the original profile.

    Args:
        model: Recorded robot with the registered world base pose.
        current: Last executed joint commands, shaped (samples, joints).
        original: Immutable source commands for reporting cumulative deviation.
        timestamps: Original increasing sample times in seconds.
        phases: Validated inclusive semantic intervals.
        corrections: Small JSON edit records; executable code, joint values and
            alternative trajectories are not accepted.

    Returns:
        Separate full commands and numerical evidence. IK residuals and edit
        limits do not establish task success or collision freedom.
    """
    current, original = np.asarray(current, dtype=float), np.asarray(
        original, dtype=float
    )
    times = np.asarray(timestamps, dtype=float)
    if (
        current.shape != original.shape
        or current.ndim != 2
        or times.ndim != 1
        or len(times) != len(current)
        or not np.isfinite(current).all()
        or not np.isfinite(original).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Correction requires matching finite source arrays and clock")
    if (
        not isinstance(corrections, list)
        or not 1 <= len(corrections) <= CORRECTION_LIMITS["max_edits"]
    ):
        raise ValueError("Expected one to four structured corrections")
    validate_segment_revision(current, current, phases, [])
    tools = TrajectoryTools(model)
    robot = tools.robot
    result = current.copy()
    by_id = {p["id"]: p for p in phases}
    allowed = {
        "phase_ids",
        "side",
        "anchor_sample",
        "translation_m",
        "rotation_vector_rad",
        "gripper_width_delta_m",
        "gripper_shift_s",
    }
    occupied: dict[int, set[int]] = {}
    changed, reports = set(), []
    for edit in corrections:
        if not isinstance(edit, dict) or set(edit) - allowed:
            raise ValueError("Unknown correction fields; code and qpos are forbidden")
        ids = edit.get("phase_ids", [])
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
            raise ValueError("Correction requires known unique phase_ids")
        selected = sorted((by_id[i] for i in ids), key=lambda p: p["start_frame"])
        if any(
            a["end_frame"] + 1 != b["start_frame"]
            for a, b in zip(selected, selected[1:])
        ):
            raise ValueError("Correction phases must be contiguous")
        start, end = selected[0]["start_frame"], selected[-1]["end_frame"]
        anchor = edit.get("anchor_sample")
        if type(anchor) is not int or not start < anchor < end:
            raise ValueError("Correction anchor must be inside its semantic interval")
        if edit.get("side") not in model.side_names:
            raise ValueError("Correction side must name a recorded arm")
        side = model.side_names.index(edit["side"])
        samples = set(range(start, end + 1))
        if samples & occupied.setdefault(side, set()):
            raise ValueError("Corrections for the same arm must not overlap")
        occupied[side].update(samples)
        translation = np.asarray(edit.get("translation_m", [0, 0, 0]), dtype=float)
        rotation = np.asarray(edit.get("rotation_vector_rad", [0, 0, 0]), dtype=float)
        width = float(edit.get("gripper_width_delta_m", 0))
        shift = float(edit.get("gripper_shift_s", 0))
        if any(
            x.shape != (3,) or not np.isfinite(x).all() for x in (translation, rotation)
        ):
            raise ValueError("Pose residuals require three finite coordinates")
        for value, key in (
            (np.linalg.norm(translation), "translation_m"),
            (np.linalg.norm(rotation), "rotation_rad"),
            (abs(width), "gripper_width_delta_m"),
            (abs(shift), "gripper_shift_s"),
        ):
            if not np.isfinite(value) or value > CORRECTION_LIMITS[key] + 1e-9:
                raise ValueError("Correction exceeds edit authority: " + key)
        columns = robot.arm_columns[side]
        span = np.arange(start, end + 1)
        envelope = np.interp(times[span], times[[start, anchor, end]], [0, 1, 0])
        residual = {}
        if np.any(translation) or np.any(rotation):
            targets = tools.fk(current[span], side)
            targets[:, :3, 3] += envelope[:, None] * translation
            targets[:, :3, :3] = (
                Rotation.from_rotvec(envelope[:, None] * rotation).as_matrix()
                @ targets[:, :3, :3]
            )
            errors = []
            for sample, target in zip(span[1:-1], targets[1:-1]):
                try:
                    solved, error = tools.ik(target, side, current[sample])
                except ValueError as error:
                    raise ValueError(
                        f"Correction {ids} at sample {sample}: {error}"
                    ) from error
                result[sample, columns] = solved[columns]
                errors.append(error)
                if sample == anchor:
                    residual.update(error)
            residual.update(
                samples_solved=len(errors),
                max_tcp_error_m=max(e["tcp_error_m"] for e in errors),
                max_orientation_error_deg=max(
                    e["orientation_error_deg"] for e in errors
                ),
                max_joint_limit_excess_rad=max(
                    e["max_joint_limit_excess_rad"] for e in errors
                ),
                optimizer_unconverged_samples=sum(
                    not e["optimizer_converged"] for e in errors
                ),
            )
        if width or shift:
            aperture = 2 * robot.half_open(current, side)
            warped_times = times[span] - shift * envelope
            if np.any(np.diff(warped_times) <= 0):
                raise ValueError("Gripper time shift must preserve temporal order")
            corrected = np.interp(warped_times, times, aperture) + width * envelope
            low, high = robot.opening_limits[side]
            corrected = np.clip(corrected, 2 * low, 2 * high)
            result[span] = robot.set_half_open(result[span], side, corrected / 2)
        # Exact boundaries and all other arms are preserved, including calibration rounding.
        own_columns = np.r_[columns, robot.jaw_columns[side]]
        result[np.ix_([start, end], own_columns)] = current[
            np.ix_([start, end], own_columns)
        ]
        source_poses, poses = tools.fk(original[span], side), tools.fk(
            result[span], side
        )
        displacement = float(
            np.linalg.norm(poses[:, :3, 3] - source_poses[:, :3, 3], axis=1).max()
        )
        angle = float(
            Rotation.from_matrix(
                poses[:, :3, :3] @ source_poses[:, :3, :3].transpose(0, 2, 1)
            )
            .magnitude()
            .max()
        )
        # Report cumulative source deviation. A fixed world-space radius around
        # a recording is not a task criterion in a reconstructed scene. Each
        # requested residual remains bounded and restricted to its semantic span.
        changed.update(ids)
        reports.append(
            dict(
                **edit,
                ik=residual,
                max_source_translation_m=displacement,
                max_source_rotation_rad=angle,
            )
        )
    scope = validate_segment_revision(current, result, phases, sorted(changed))
    return result, dict(
        method="source-shaped Cartesian residuals with Real2Sim-only approximate IK",
        limits=CORRECTION_LIMITS.copy(),
        edits=reports,
        **scope,
    )
