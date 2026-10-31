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

"""Bounded ingestion of original DROID HDF5 observations and submissions.

DROID Cartesian observations are robot-reported forward kinematics, rather
than independent optical TCP measurements. The adapter retains the reported
link and requires an explicit link-to-TCP calibration when making a source unit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Literal, Mapping, Sequence

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import PhaseAnnotation, Real2SimTrajectory

__all__ = [
    "DroidEpisode",
    "load_droid_episode",
    "build_droid_trajectory",
    "save_droid_unit_evidence",
    "verify_droid_unit_evidence",
]


def _owned(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DroidEpisode:
    """Owned original robot streams, including commands that were skipped.

    Gripper values are normalized *closed* fractions, not joint positions or
    metres. A calibrated embodiment must convert the observed fraction to its
    joint coordinates independently of the submitted command. Raw arm velocity
    remains available even when a full-layout source cannot express missing
    gripper velocity. Times are original millisecond clock offsets converted
    to seconds; they are not replaced by the nominal DROID frame rate.

    Args:
        sha256: Digest of the original HDF5 bytes.
        arm_qpos: Observed seven-joint arm positions in radians.
        arm_qvel: Observed arm velocities, or None when absent.
        reported_poses: Robot-reported link position and XYZW quaternion.
        gripper_closed_fraction: Observed normalized gripper position.
        arm_commands: Submitted/computed arm position targets in radians.
        gripper_commands: Submitted/computed normalized gripper targets.
        submitted: Whether the record actually submitted its action.
        timestamps: State-read clock offsets in seconds.
        command_timestamps: Control-start clock offsets on the same clock.
    """

    sha256: str
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray | None
    reported_poses: np.ndarray
    gripper_closed_fraction: np.ndarray
    arm_commands: np.ndarray
    gripper_commands: np.ndarray
    submitted: np.ndarray
    timestamps: np.ndarray
    command_timestamps: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("DROID source digest must be a lowercase SHA-256")
        n = len(self.timestamps)
        if n < 2:
            raise ValueError("DROID episode needs at least two observations")
        shapes = {
            "arm_qpos": (n, 7),
            "arm_qvel": (n, 7),
            "reported_poses": (n, 7),
            "gripper_closed_fraction": (n,),
            "arm_commands": (n, 7),
            "gripper_commands": (n,),
            "timestamps": (n,),
            "command_timestamps": (n,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if name == "arm_qvel" and value is None:
                continue
            object.__setattr__(self, name, _owned(value, shape, name))
        for name in ("gripper_closed_fraction", "gripper_commands"):
            values = getattr(self, name)
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("timestamps", "command_timestamps"):
            if np.any(np.diff(getattr(self, name)) <= 0):
                raise ValueError(f"DROID {name} must increase strictly")
        if not np.allclose(
            np.linalg.norm(self.reported_poses[:, 3:], axis=1), 1, atol=1e-7
        ):
            raise ValueError("DROID pose quaternions must be normalized")
        flags = np.array(self.submitted, copy=True)
        if flags.shape != (n,) or flags.dtype.kind != "b":
            raise ValueError("DROID submitted flags must be an aligned boolean stream")
        flags.setflags(write=False)
        object.__setattr__(self, "submitted", flags)


def load_droid_episode(
    trajectory: str | Path,
    *,
    max_samples: int = 200_000,
    max_array_bytes: int = 128 * 1024 * 1024,
) -> DroidEpisode:
    """Read selected original streams without loading embedded images.

    Args:
        trajectory: Original ``trajectory.h5`` file, not an RLDS conversion.
        max_samples: Maximum input observations, checked before array reads.
        max_array_bytes: Limit on selected decoded source-array bytes.

    Returns:
        Immutable state and command evidence on the original clock.

    Raises:
        ValueError: Required streams, shapes, clocks or budgets are invalid.
    """
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("max_samples must be an integer of at least two")
    if type(max_array_bytes) is not int or max_array_bytes < 1:
        raise ValueError("max_array_bytes must be a positive integer")
    paths = {
        "arm_qpos": "observation/robot_state/joint_positions",
        "reported_rpy": "observation/robot_state/cartesian_position",
        "gripper_closed_fraction": "observation/robot_state/gripper_position",
        "arm_commands": "action/joint_position",
        "gripper_commands": "action/gripper_position",
        "timestamps": "observation/timestamp/robot_state/read_start",
        "command_timestamps": "observation/timestamp/control/control_start",
        "skipped": "observation/timestamp/skip_action",
    }
    path = Path(trajectory)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    with h5py.File(path, "r") as archive:
        velocity = "observation/robot_state/joint_velocities"
        if velocity in archive:
            paths["arm_qvel"] = velocity
        datasets = {}
        for name, key in paths.items():
            # External/soft links must not redirect selected evidence to another
            # file or an unbounded, differently named dataset.
            parts = key.split("/")
            group = archive
            for part in parts:
                if not isinstance(group, h5py.Group) or not isinstance(
                    group.get(part, getlink=True), h5py.HardLink
                ):
                    raise ValueError(f"missing or linked DROID stream: {key}")
                group = group[part]
            if not isinstance(group, h5py.Dataset) or group.dtype.kind not in "fiub":
                raise ValueError(f"DROID stream must be numeric: {key}")
            if group.is_virtual or group.external:
                raise ValueError(
                    f"DROID stream must have self-contained storage: {key}"
                )
            datasets[name] = group
        shape = datasets["arm_qpos"].shape
        if len(shape) != 2 or shape[1] != 7 or not 2 <= shape[0] <= max_samples:
            raise ValueError(
                "DROID joint stream exceeds max_samples or has invalid shape"
            )
        n = shape[0]
        vector_fields = {
            "arm_qpos": 7,
            "arm_qvel": 7,
            "reported_rpy": 6,
            "arm_commands": 7,
        }
        for name, dataset in datasets.items():
            expected = (n, vector_fields[name]) if name in vector_fields else (n,)
            if dataset.shape != expected:
                raise ValueError(f"DROID {name} has an incompatible sample layout")
        if (
            sum(
                dataset.size * max(8, dataset.dtype.itemsize)
                for dataset in datasets.values()
            )
            > max_array_bytes
        ):
            raise ValueError("DROID selected arrays exceed max_array_bytes")
        values = {name: dataset[:] for name, dataset in datasets.items()}
    rpy = values.pop("reported_rpy")
    if not np.isfinite(rpy).all():
        raise ValueError("DROID Cartesian observations must be finite")
    poses = np.column_stack(
        (rpy[:, :3], Rotation.from_euler("xyz", rpy[:, 3:]).as_quat())
    )
    epoch = int(values["timestamps"][0])
    for name in ("timestamps", "command_timestamps"):
        raw = values[name]
        if raw.dtype.kind not in "iu":
            raise ValueError("DROID clock streams must contain integer milliseconds")
        values[name] = (raw.astype(np.float64) - epoch) * 0.001
    skipped = values.pop("skipped")
    if skipped.dtype.kind != "b":
        raise ValueError("DROID skip_action must be boolean")
    values.setdefault("arm_qvel", None)
    return DroidEpisode(
        sha256=digest, reported_poses=poses, submitted=~skipped, **values
    )


def build_droid_trajectory(
    episode: DroidEpisode,
    *,
    source_id: str,
    unit_id: str,
    skill_id: Literal["pick_up", "move_held_object", "place"],
    start: int,
    stop: int,
    arm_joint_names: Sequence[str],
    gripper_joint_names: Sequence[str],
    gripper_joint_units: Sequence[str],
    observed_gripper_qpos: np.ndarray,
    phases: Sequence[PhaseAnnotation],
    world_from_base: np.ndarray,
    reported_link_to_tcp: np.ndarray,
    object_pose: np.ndarray,
    target_pose: np.ndarray | None,
    calibration_revision: str,
    annotation_revision: str,
    robot_geometry_sha256: str | None = None,
    object_geometry_sha256: str | None = None,
    target_geometry_sha256: str | None = None,
) -> Real2SimTrajectory:
    """Build one manually annotated unit from original observed states.

    Args:
        episode: Original bounded DROID evidence.
        source_id: Stable episode identity from published metadata.
        unit_id: Unique annotated unit within this episode.
        skill_id: Existing Pick, held-motion or Place Atomic Skill identity.
        start: First included original observation index.
        stop: First excluded original observation index.
        arm_joint_names: Seven names in original DROID arm order.
        gripper_joint_names: Explicit names for calibrated gripper observations.
        gripper_joint_units: Units of those calibrated joint coordinates.
        observed_gripper_qpos: Converted observed gripper stream for the full
            episode, not converted command targets. Conversion provenance belongs
            in the annotation/calibration sidecar.
        phases: Contiguous, unit-local annotated phase ranges.
        world_from_base: Explicit source-base to world calibration.
        reported_link_to_tcp: Explicit reported-link to physical TCP transform.
        object_pose: Annotated frozen world object pose in metres.
        target_pose: Annotated frozen target pose, if used by this skill.
        calibration_revision: Pinned coordinate and gripper calibration identity.
        annotation_revision: Pinned phase/object annotation identity.
        robot_geometry_sha256: Qualified robot model digest, if available.
        object_geometry_sha256: Qualified physical proxy digest, if available.
        target_geometry_sha256: Qualified support proxy digest, if available.

    Returns:
        A measured source unit preserving independent original commands.

    Raises:
        ValueError: The range includes skipped submissions or invalid mappings.
    """
    if not isinstance(episode, DroidEpisode):
        raise TypeError("episode must be DroidEpisode")
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop - 1 <= len(episode.timestamps) - 1
    ):
        raise ValueError("DROID unit range must contain at least two original states")
    arm_names = tuple(arm_joint_names)
    gripper_names = tuple(gripper_joint_names)
    units = tuple(gripper_joint_units)
    if len(arm_names) != 7 or not gripper_names or len(units) != len(gripper_names):
        raise ValueError(
            "explicit arm order and calibrated gripper joints are required"
        )
    gripper = _owned(
        observed_gripper_qpos,
        (len(episode.timestamps), len(gripper_names)),
        "observed gripper qpos",
    )
    if not episode.submitted[start : stop - 1].all():
        raise ValueError("DROID unit includes an action that was not submitted")
    qpos = np.column_stack((episode.arm_qpos[start:stop], gripper[start:stop]))
    # The final record's command has no next observed state in this unit. Keep
    # it in the original episode, not in this between-observations command stream.
    commands = np.column_stack(
        (
            episode.arm_commands[start : stop - 1],
            episode.gripper_commands[start : stop - 1],
        )
    )
    return Real2SimTrajectory(
        source_id=source_id,
        source_revision=episode.sha256,
        unit_id=unit_id,
        skill_id=skill_id,
        origin="measured",
        joint_names=arm_names + gripper_names,
        joint_units=("rad",) * 7 + units,
        qpos=qpos,
        timestamps=episode.timestamps[start:stop] - episode.timestamps[start],
        time_unit="s",
        phases=tuple(phases),
        eef_poses=episode.reported_poses[start:stop],
        pose_length_unit="m",
        quaternion_order="xyzw",
        world_from_pose_frame=world_from_base,
        observed_link_to_tcp=reported_link_to_tcp,
        object_pose=object_pose,
        target_pose=target_pose,
        # Full-layout gripper velocity is absent. Original measured arm velocity
        # stays in DroidEpisode and the pinned raw HDF5, never zero-padded here.
        qvel=None,
        initial_observed_qpos=qpos[0],
        commands=commands,
        command_representation="droid_arm_position_rad_and_gripper_closed_fraction",
        command_alignment="between_states",
        calibration_revision=calibration_revision,
        annotation_revision=annotation_revision,
        pose_origin="fk_derived",
        robot_geometry_sha256=robot_geometry_sha256,
        object_geometry_sha256=object_geometry_sha256,
        target_geometry_sha256=target_geometry_sha256,
    )


def save_droid_unit_evidence(
    episode: DroidEpisode,
    destination: str | Path,
    *,
    source_id: str,
    unit_id: str,
    start: int,
    stop: int,
    arm_joint_names: Sequence[str],
    calibrated_gripper_qpos: np.ndarray,
    annotation: Mapping[str, object],
    calibration: Mapping[str, object],
) -> Path:
    """Save original typed streams and annotation/calibration for one unit.

    Raw arm velocity and skipped submissions remain source evidence; they are
    never padded into simulation observations or a full-layout measured velocity.
    The original HDF5 is pinned separately by ``episode.sha256``.

    Args:
        episode: Original observed/command streams.
        destination: New NPZ file, published without replacing prior evidence.
        source_id: Published original episode identity.
        unit_id: Unit identity used by the prepared prototype.
        start: Original first observation index.
        stop: Original first excluded observation index.
        arm_joint_names: Seven canonical names in original DROID order.
        calibrated_gripper_qpos: Whole-episode conversion of observed aperture.
        annotation: JSON-compatible human annotations and conversion provenance.
        calibration: JSON-compatible calibration including ``world_from_base``
            and ``reported_link_to_tcp`` matrices.

    Returns:
        Published typed evidence path.
    """
    if not isinstance(episode, DroidEpisode):
        raise TypeError("DROID evidence requires an original episode")
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop - 1 < len(episode.timestamps)
    ):
        raise ValueError("DROID evidence has an invalid unit range")
    if (
        any(not isinstance(value, str) or not value for value in (source_id, unit_id))
        or len(tuple(arm_joint_names)) != 7
    ):
        raise ValueError("DROID evidence needs source/unit identities and arm order")
    hand = np.asarray(calibrated_gripper_qpos, dtype=np.float64)
    if (
        hand.ndim != 2
        or hand.shape[0] != len(episode.timestamps)
        or not np.isfinite(hand).all()
    ):
        raise ValueError("DROID evidence needs calibrated observed gripper coordinates")
    metadata = {
        "schema": "droid-unit-evidence/v1",
        "source_id": source_id,
        "unit_id": unit_id,
        "source_revision": episode.sha256,
        "start": start,
        "stop": stop,
        "arm_joint_names": list(arm_joint_names),
        "annotation": dict(annotation),
        "calibration": dict(calibration),
    }
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > 1024 * 1024:
        raise ValueError("DROID evidence metadata exceeds its size budget")
    arrays = {
        name: getattr(episode, name)
        for name in (
            "arm_qpos",
            "reported_poses",
            "gripper_closed_fraction",
            "arm_commands",
            "gripper_commands",
            "submitted",
            "timestamps",
            "command_timestamps",
        )
    }
    if episode.arm_qvel is not None:
        arrays["arm_qvel"] = episode.arm_qvel
    arrays["calibrated_gripper_qpos"] = hand
    arrays["metadata"] = np.frombuffer(encoded, dtype=np.uint8)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def verify_droid_unit_evidence(
    evidence: str | Path,
    original_recording: str | Path,
    source: Real2SimTrajectory,
) -> None:
    """Verify original HDF5, typed streams and the exact imported source unit.

    Args:
        evidence: NPZ produced by :func:`save_droid_unit_evidence`.
        original_recording: Original immutable DROID HDF5 recording.
        source: Measured source unit reconstructed from the prototype bundle.

    Raises:
        ValueError: Source identity, original streams, crop, commands or explicit
            calibration differ, or required raw velocity evidence is missing.
    """
    if not isinstance(source, Real2SimTrajectory) or source.origin != "measured":
        raise ValueError("DROID original evidence requires a measured source")
    path = Path(evidence)
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("DROID evidence exceeds its decoded size budget")
        # ZIP byte counts alone do not bound the allocation requested by an NPY
        # shape header. Validate every member before NumPy materializes arrays.
        names = set()
        for item in archive.infolist():
            if item.filename in names or not item.filename.endswith(".npy"):
                raise ValueError("DROID evidence requires unique NPY members")
            names.add(item.filename)
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported DROID evidence NPY version")
                import math

                size = math.prod(shape) * dtype.itemsize
                budget = (
                    1024 * 1024
                    if item.filename == "metadata.npy"
                    else 128 * 1024 * 1024
                )
                if (
                    dtype.kind not in "fiub"
                    or size > budget
                    or stream.tell() + size != item.file_size
                ):
                    raise ValueError(
                        "DROID evidence NPY shape exceeds its payload or byte budget"
                    )
    episode = load_droid_episode(original_recording)
    base = {
        name
        for name in (
            "arm_qpos",
            "reported_poses",
            "gripper_closed_fraction",
            "arm_commands",
            "gripper_commands",
            "submitted",
            "timestamps",
            "command_timestamps",
        )
    }
    expected = base | {"calibrated_gripper_qpos", "metadata"}
    if episode.arm_qvel is not None:
        expected.add("arm_qvel")
    with np.load(path, allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)) or set(arrays.files) != expected:
            raise ValueError(
                "DROID evidence has missing or unexpected original streams"
            )
        payload = arrays["metadata"]
        if (
            payload.dtype != np.uint8
            or payload.ndim != 1
            or payload.nbytes > 1024 * 1024
        ):
            raise ValueError("DROID evidence metadata must be bounded UTF-8 bytes")
        metadata = json.loads(payload.tobytes().decode())
        if (
            not isinstance(metadata, dict)
            or set(metadata)
            != {
                "schema",
                "source_id",
                "unit_id",
                "source_revision",
                "start",
                "stop",
                "arm_joint_names",
                "annotation",
                "calibration",
            }
            or metadata["schema"] != "droid-unit-evidence/v1"
        ):
            raise ValueError("DROID evidence schema is invalid")
        if (
            metadata["source_id"],
            metadata["unit_id"],
            metadata["source_revision"],
        ) != (
            source.source_id,
            source.unit_id,
            source.source_revision,
        ) or episode.sha256 != source.source_revision:
            raise ValueError("DROID original recording or source identity differs")
        for name in base | ({"arm_qvel"} if episode.arm_qvel is not None else set()):
            if not np.array_equal(arrays[name], getattr(episode, name)):
                raise ValueError(f"DROID evidence changed original {name}")
        start, stop = metadata["start"], metadata["stop"]
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop - 1 < len(episode.timestamps)
        ):
            raise ValueError("DROID evidence unit range is invalid")
        if tuple(metadata["arm_joint_names"]) != source.joint_names[:7]:
            raise ValueError("DROID evidence arm joint order differs")
        if not episode.submitted[start : stop - 1].all():
            raise ValueError("DROID source includes an unsubmitted command interval")
        hand = arrays["calibrated_gripper_qpos"]
        if (
            hand.shape != (len(episode.timestamps), len(source.joint_names) - 7)
            or not np.isfinite(hand).all()
        ):
            raise ValueError("DROID calibrated gripper evidence is invalid")
        comparisons = (
            (source.qpos[:, :7], episode.arm_qpos[start:stop]),
            (source.qpos[:, 7:], hand[start:stop]),
            (
                source.timestamps,
                episode.timestamps[start:stop] - episode.timestamps[start],
            ),
            (source.eef_poses, episode.reported_poses[start:stop]),
            (
                source.commands,
                np.column_stack(
                    (
                        episode.arm_commands[start : stop - 1],
                        episode.gripper_commands[start : stop - 1],
                    )
                ),
            ),
        )
        if (
            source.time_unit != "s"
            or source.pose_origin != "fk_derived"
            or source.command_alignment != "between_states"
            or source.qvel is not None
            or any(not np.array_equal(first, second) for first, second in comparisons)
        ):
            raise ValueError("DROID imported unit differs from original evidence")
        calibration = metadata["calibration"]
        if not isinstance(calibration, dict) or not isinstance(
            metadata["annotation"], dict
        ):
            raise ValueError(
                "DROID annotation/calibration must be explicit JSON mappings"
            )
        for name, actual in (
            ("world_from_base", source.world_from_pose_frame),
            ("reported_link_to_tcp", source.observed_link_to_tcp),
        ):
            if name not in calibration or not np.array_equal(
                np.asarray(calibration[name]), actual
            ):
                raise ValueError("DROID explicit calibration differs from evidence")
