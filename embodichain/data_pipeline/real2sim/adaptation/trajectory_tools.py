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
"""Numerical trajectory operations for caller-directed revisions, without policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import ArrayLike

from .inputs import RobotModel
from ._kinematics import Robot

__all__: list[str] = [
    "TrajectoryTools",
    "interpolate_joint_knots",
    "validate_segment_revision",
    "read_replay_segment",
]


def read_replay_segment(
    rollout: Path, start_sample: int, end_sample: int
) -> dict[str, Any]:
    """Read native evidence for an inclusive source-trajectory sample interval.

    Args:
        rollout: Native replay directory with diagnostics/ and videos/ subfolders,
            or a legacy flat recording. Clocks are simulation-relative seconds.
        start_sample: First sample of the source trajectory interval.
        end_sample: Last sample of the source trajectory interval.

    Returns:
        Observed joints/object poses and commanded joints selected by their own
        acquisition clocks, contact identities, video paths and original timing
        metadata. Interval contacts include every overlapping control interval
        with its original bounds; last-substep contacts remain a separate field.
        Missing interval evidence is None, not evidence of absent contact.
        The reader does not classify success or prescribe a correction.
    """
    rollout = Path(rollout).resolve()
    diagnostics = (
        rollout / "diagnostics" if (rollout / "diagnostics").is_dir() else rollout
    )
    videos = rollout / "videos" if (rollout / "videos").is_dir() else rollout
    with np.load(diagnostics / "physics_rollout.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for key in ("source_timestamps", "timestamps", "command_timestamps"):
        clock = arrays[key]
        if (
            clock.ndim != 1
            or not len(clock)
            or not np.isfinite(clock).all()
            or np.any(np.diff(clock) <= 0)
        ):
            raise ValueError("Invalid replay clock: " + key)
    source = arrays["source_timestamps"]
    if (
        not isinstance(start_sample, (int, np.integer))
        or not isinstance(end_sample, (int, np.integer))
        or not 0 <= start_sample <= end_sample < len(source)
    ):
        raise ValueError("Segment requires valid inclusive source sample indices")
    start, end = float(source[start_sample]), float(source[end_sample])
    observed = np.flatnonzero(
        (arrays["timestamps"] >= start - 1e-8) & (arrays["timestamps"] <= end + 1e-8)
    )
    commanded = np.flatnonzero(
        (arrays["command_timestamps"] >= start - 1e-8)
        & (arrays["command_timestamps"] <= end + 1e-8)
    )
    for key, clock in (
        ("qpos_observed", "timestamps"),
        ("object_poses", "timestamps"),
        ("qpos_command", "command_timestamps"),
    ):
        if len(arrays[key]) != len(arrays[clock]):
            raise ValueError("Replay data and clock lengths differ: " + key)
    contacts = json.loads((diagnostics / "contacts.json").read_text())
    raw = [
        row
        for row in contacts["samples"]
        if start - 1e-8 <= row["timestamp"] <= end + 1e-8
    ]
    intervals = contacts.get("interval_samples")
    if intervals is not None:
        intervals = [
            row
            for row in intervals
            if row["timestamp"] >= start - 1e-8 and row["start_timestamp"] <= end + 1e-8
        ]
    return dict(
        source_sample_range=[int(start_sample), int(end_sample)],
        time_range_s=[start, end],
        observed_sample_indices=observed,
        timestamps=arrays["timestamps"][observed],
        joint_names=arrays["joint_names"],
        qpos_observed=arrays["qpos_observed"][observed],
        command_sample_indices=commanded,
        command_timestamps=arrays["command_timestamps"][commanded],
        qpos_command=arrays["qpos_command"][commanded],
        object_names=arrays["object_names"],
        object_poses=arrays["object_poses"][observed],
        contact_actors=contacts["actors"],
        contact_sampling=contacts.get("sampling"),
        last_substep_contacts=raw,
        interval_contacts=intervals,
        videos={
            name: str(videos / (name + ".mp4"))
            for name in ("simulation", "overview", "comparison_full")
            if (videos / (name + ".mp4")).is_file()
        },
    )


def _trajectory(value: ArrayLike) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if (
        value.ndim != 2
        or not value.shape[0]
        or not value.shape[1]
        or not np.isfinite(value).all()
    ):
        raise ValueError("Trajectory must be a nonempty finite [frames, joints] array")
    return value


class TrajectoryTools:
    """World-frame FK and numerical IK; no grasp, clearance or success rules."""

    def __init__(self, model: RobotModel) -> None:
        self.robot: Robot = Robot(model)
        self._hand_surfaces: dict[int, dict[str, np.ndarray]] = {}
        self._arm_surfaces: dict[int, dict[str, np.ndarray]] = {}

    def fk(self, qpos: np.ndarray, side: int) -> np.ndarray:
        """Return world TCP pose(s) for one full state or a batch of states."""
        qpos = np.asarray(qpos, dtype=float)
        single = qpos.ndim == 1
        batch = _trajectory(qpos[None] if single else qpos)
        if batch.shape[1] != len(self.robot.names) or not 0 <= side < len(
            self.robot.arm_columns
        ):
            raise ValueError("Joint dimensions or arm index do not match the robot")
        poses = self.robot.tcp(batch, side)
        return poses[0] if single else poses

    def joint_limit_report(self, qpos: np.ndarray) -> dict[str, Any]:
        """Report recorded arm-limit violations without modifying source values.

        Ranges use half-open source sample indices. This diagnoses URDF/source
        mismatch; it does not decide which semantic phases need correction.
        """
        qpos = _trajectory(qpos)
        if qpos.shape[1] != len(self.robot.names):
            raise ValueError("Joint dimensions do not match the robot")
        report = {}
        for side, name in enumerate(self.robot.prefixes):
            columns = self.robot.arm_columns[side]
            lower, upper = self.robot.arm_limits[side]
            excess = np.maximum(
                np.maximum(lower - qpos[:, columns], qpos[:, columns] - upper), 0
            )
            invalid = np.any(excess > 0, axis=1)
            edges = np.flatnonzero(np.diff(np.r_[False, invalid, False]))
            report[name] = dict(
                joint_names=[self.robot.names[i] for i in columns],
                max_excess_rad=excess.max(axis=0).tolist(),
                out_of_limits_sample_ranges=edges.reshape(-1, 2).tolist(),
            )
        return report

    def hand_surfaces(self, qpos: np.ndarray, side: int) -> dict[str, np.ndarray]:
        """Return sampled palm and finger collision surfaces in world coordinates.

        Args:
            qpos: One full joint state, commanded or observed in a replay.
            side: Index of the hand in the source robot model.

        Returns:
            World point arrays keyed by URDF link name. The existing URDF
            collision sampler supplies the link-local geometry; visuals and
            approximate TCP pad offsets are not substituted. Sampling is
            diagnostic geometry, not a proof of continuous collision freedom.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.ndim != 1:
            raise ValueError("Hand geometry requires one full joint state")
        self.fk(qpos, side)
        if side not in self._hand_surfaces:
            from ._hand_geometry import hand_collision_surfaces

            self._hand_surfaces[side] = hand_collision_surfaces(self.robot, side)
        poses = self.robot.links(qpos[None])
        return {
            name: points @ poses[name][0, :3, :3].T + poses[name][0, :3, 3]
            for name, points in self._hand_surfaces[side].items()
        }

    def arm_surfaces(self, qpos: np.ndarray, side: int) -> dict[str, np.ndarray]:
        """Return authored arm collision samples in world coordinates.

        Args:
            qpos: One full commanded or observed joint state.
            side: Recorded arm index; joint topology selects its child links.

        Returns:
            Link-keyed world point arrays from the existing URDF collision sampler.
            These diagnose arm/environment obstruction without certifying a path.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.ndim != 1:
            raise ValueError("Arm geometry requires one full joint state")
        self.fk(qpos, side)
        if side not in self._arm_surfaces:
            from embodichain.toolkits.graspkit.pg_grasp.urdf_surfaces import (
                sample_urdf_collision_surfaces,
            )

            joints = {
                joint.get("name"): joint.find("child").get("link")
                for joint in ET.parse(self.robot.path).getroot().findall("joint")
            }
            links = [joints[self.robot.names[i]] for i in self.robot.arm_columns[side]]
            clouds = sample_urdf_collision_surfaces(
                self.robot.path,
                link_names=links,
                max_edge_by_link={name: 0.02 for name in links},
                max_subdivisions=8,
            )
            self._arm_surfaces[side] = {
                name: points.numpy() for name, points in clouds.items()
            }
        poses = self.robot.links(qpos[None])
        return {
            name: points @ poses[name][0, :3, :3].T + poses[name][0, :3, 3]
            for name, points in self._arm_surfaces[side].items()
        }

    def ik(
        self, target: np.ndarray, side: int, seed: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit near a caller seed using the experimental Real2Sim-only IK.

        Other arm and jaw coordinates are preserved. Target error and URDF limit
        excess are reported without rejecting a finite approximate solution.
        This does not change simulator limits or establish task completion.
        """
        target, seed = np.asarray(target, dtype=float), np.asarray(seed, dtype=float)
        if seed.ndim != 1:
            raise ValueError("IK seed must be one full joint state")
        self.fk(seed, side)
        if (
            target.shape != (4, 4)
            or not np.isfinite(target).all()
            or not np.allclose(target[3], [0, 0, 0, 1])
            or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(target[:3, :3]), 1, atol=1e-6)
        ):
            raise ValueError("IK target must be a finite rigid world 4x4 pose")
        return self.robot.ik(seed.copy(), side, target)


def interpolate_joint_knots(
    qpos: ArrayLike,
    frames: Sequence[int] | np.ndarray,
    values: ArrayLike,
    joint_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Linearly interpolate selected joints between inclusive first/last knots."""
    source = _trajectory(qpos)
    frames, indices = np.asarray(frames), np.asarray(joint_indices)
    values = np.asarray(values, dtype=float)
    if (
        frames.ndim != 1
        or len(frames) < 2
        or frames.dtype.kind not in "iu"
        or np.any(np.diff(frames.astype(np.int64)) <= 0)
        or frames[0] < 0
        or frames[-1] >= len(source)
    ):
        raise ValueError("Knot frames must be increasing valid integer frame indices")
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= source.shape[1])
    ):
        raise ValueError("Joint indices must be unique valid integer columns")
    if values.shape != (len(frames), len(indices)) or not np.isfinite(values).all():
        raise ValueError("Knot values must be finite [knots, selected joints]")
    result = source.copy()
    rows = np.arange(frames[0], frames[-1] + 1)
    for column, joint in enumerate(indices):
        result[rows, joint] = np.interp(rows, frames, values[:, column])
    return result


def validate_segment_revision(
    previous: ArrayLike,
    candidate: ArrayLike,
    phases: Sequence[Mapping[str, object]],
    changed_phase_ids: Sequence[str],
) -> dict[str, list[str] | list[int]]:
    """Check only finite shape and edits outside declared inclusive phase ranges.

    Phases contain unique ``id``, ``start_frame`` and ``end_frame`` fields. This preserves
    the current sample grid but imposes no motion, deviation or clearance rule.
    It does not modify either trajectory or evaluate task success.
    """
    before, after = _trajectory(previous), _trajectory(candidate)
    if before.shape != after.shape:
        raise ValueError("Revision must preserve the current frame/joint grid")
    ranges = {}
    for phase in phases:
        uid, start, end = phase["id"], phase["start_frame"], phase["end_frame"]
        if (
            not isinstance(uid, str)
            or not uid
            or uid in ranges
            or not isinstance(start, (int, np.integer))
            or not isinstance(end, (int, np.integer))
            or not 0 <= start <= end < len(before)
        ):
            raise ValueError(
                "Phases require unique IDs and valid inclusive frame ranges"
            )
        ranges[uid] = (start, end)
    requested = set(changed_phase_ids)
    if not requested <= ranges.keys():
        raise ValueError("Revision names an unknown semantic phase")
    allowed = np.zeros(len(before), dtype=bool)
    for uid in requested:
        start, end = ranges[uid]
        allowed[start : end + 1] = True
    changed = np.any(before != after, axis=1)
    if np.any(changed & ~allowed):
        raise ValueError("Revision changes frames outside its declared semantic phases")
    return {
        "changed_phase_ids": sorted(requested),
        "changed_frames": np.flatnonzero(changed).tolist(),
    }
