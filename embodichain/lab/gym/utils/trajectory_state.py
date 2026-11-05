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

"""Shared simulation-state capture and restore helpers for trajectories.

All seven-element root and rigid-object poses use EmbodiChain's
``(x, y, z, qx, qy, qz, qw)`` convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TYPE_CHECKING

import torch
from tensordict import TensorDictBase

from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.episode_context import EpisodeContext

__all__ = [
    "capture_trajectory_state",
    "restore_trajectory_state",
    "RecordedRigidPoseSnapshot",
    "RecordedTrajectoryReader",
    "read_recorded_rigid_pose_snapshot",
    "RecordedRigidPoseWindow",
    "read_recorded_rigid_pose_window",
]


class _TrajectoryEnv(Protocol):
    """Environment surface required by trajectory state helpers."""

    robot: Any
    sim: Any


_TrajectoryState: TypeAlias = Mapping[str, Any] | TensorDictBase


def _assign_state(
    target: torch.Tensor,
    source: torch.Tensor,
    env_ids: torch.Tensor,
    step_ids: torch.Tensor | None,
) -> None:
    """Assign selected environment rows, optionally at per-row time indices."""
    if step_ids is None:
        target[env_ids] = source[env_ids]
    else:
        target[env_ids, step_ids] = source[env_ids]


def capture_trajectory_state(
    env: _TrajectoryEnv,
    states: TensorDictBase,
    env_ids: torch.Tensor,
    step_ids: torch.Tensor | None = None,
) -> None:
    """Capture restorable simulation state into a trajectory state view.

    Args:
        env: Environment containing the robot and simulation scene entities.
        states: Destination trajectory state buffer.
        env_ids: Environment rows to capture.
        step_ids: Optional per-environment destination time indices.
    """
    robot_states = states["robot"]
    _assign_state(
        robot_states["root_pose"], env.robot.get_local_pose(), env_ids, step_ids
    )
    _assign_state(robot_states["qpos"], env.robot.get_qpos(), env_ids, step_ids)
    if "qvel" in robot_states.keys():
        _assign_state(robot_states["qvel"], env.robot.get_qvel(), env_ids, step_ids)

    if "articulations" in states.keys():
        articulation_states = states["articulations"]
        for uid, articulation in env.sim._articulations.items():
            if uid not in articulation_states.keys():
                continue
            entity_states = articulation_states[uid]
            _assign_state(
                entity_states["root_pose"],
                articulation.get_local_pose(),
                env_ids,
                step_ids,
            )
            _assign_state(
                entity_states["qpos"], articulation.get_qpos(), env_ids, step_ids
            )
            if "qvel" in entity_states.keys():
                _assign_state(
                    entity_states["qvel"],
                    articulation.get_qvel(),
                    env_ids,
                    step_ids,
                )

    if "rigid_objects" in states.keys():
        rigid_states = states["rigid_objects"]
        for uid, rigid_object in env.sim._rigid_objects.items():
            if uid not in rigid_states.keys():
                continue
            entity_states = rigid_states[uid]
            _assign_state(
                entity_states["pose"],
                rigid_object.get_local_pose(),
                env_ids,
                step_ids,
            )
            if "lin_vel" in entity_states.keys():
                body_state = rigid_object.body_state
                _assign_state(
                    entity_states["lin_vel"],
                    body_state[:, 7:10],
                    env_ids,
                    step_ids,
                )
                _assign_state(
                    entity_states["ang_vel"],
                    body_state[:, 10:13],
                    env_ids,
                    step_ids,
                )


def restore_trajectory_state(
    env: _TrajectoryEnv,
    states: _TrajectoryState,
) -> None:
    """Restore one trajectory timestep into an environment scene.

    Args:
        env: Environment containing the robot and simulation scene entities.
        states: One timestep of recorded robot and scene-object states.
    """
    robot_states = states["robot"]
    env.robot.set_local_pose(robot_states["root_pose"])
    # Restore complete qpos, including mimic joints. Kinematic replay disables
    # physics, so active-joint writes alone cannot propagate to mimic children.
    env.robot.set_qpos(robot_states["qpos"], target=False)
    if "qvel" in robot_states.keys():
        env.robot.set_qvel(robot_states["qvel"], target=False)

    if "articulations" in states.keys():
        articulation_states = states["articulations"]
        trajectory_uids = set(articulation_states.keys())
        scene_uids = set(env.sim._articulations.keys())
        for uid in trajectory_uids - scene_uids:
            logger.log_warning(
                f"Trajectory articulation '{uid}' is not present in the scene; skipping."
            )
        for uid in trajectory_uids & scene_uids:
            articulation = env.sim._articulations[uid]
            entity_states = articulation_states[uid]
            articulation.set_local_pose(entity_states["root_pose"])
            articulation.set_qpos(entity_states["qpos"], target=False)
            if "qvel" in entity_states.keys():
                articulation.set_qvel(entity_states["qvel"], target=False)
        for uid in scene_uids - trajectory_uids:
            logger.log_warning(
                f"Scene articulation '{uid}' is not in the trajectory; leaving initial state."
            )

    if "rigid_objects" in states.keys():
        rigid_states = states["rigid_objects"]
        trajectory_uids = set(rigid_states.keys())
        scene_uids = set(env.sim._rigid_objects.keys())
        for uid in trajectory_uids - scene_uids:
            logger.log_warning(
                f"Trajectory rigid object '{uid}' is not present in the scene; skipping."
            )
        for uid in trajectory_uids & scene_uids:
            rigid_object = env.sim._rigid_objects[uid]
            entity_states = rigid_states[uid]
            rigid_object.set_local_pose(entity_states["pose"])
            if "lin_vel" in entity_states.keys() and not rigid_object.is_non_dynamic:
                rigid_object.set_velocity(
                    lin_vel=entity_states["lin_vel"],
                    ang_vel=entity_states["ang_vel"],
                )
        for uid in scene_uids - trajectory_uids:
            logger.log_warning(
                f"Scene rigid object '{uid}' is not in the trajectory; leaving initial state."
            )


@dataclass(frozen=True, slots=True, eq=False)
class RecordedRigidPoseSnapshot:
    """Owned rigid poses from exact completed trajectory boundary slots.

    Frame indices are row-local trajectory slots, not simulation timestamps.
    They must not be mixed with another clock domain without a proven mapping.
    The snapshot carries the host episode IDs; it decides no task outcome.
    """

    object_uid: str
    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    poses: torch.Tensor

    def __post_init__(self) -> None:
        from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

        if (
            type(self.object_uid) is not str
            or not self.object_uid
            or self.object_uid != self.object_uid.strip()
        ):
            raise ValueError(
                "Recorded object UID must be an exact nonempty identifier."
            )
        size = len(self.env_ids)
        if not size or len(set(self.env_ids)) != size:
            raise ValueError("Recorded environment IDs must be nonempty and unique.")
        for name, lower in (("env_ids", 0), ("episode_ids", 0), ("frame_indices", 1)):
            values = tuple(getattr(self, name))
            if len(values) != size or any(
                type(value) is not int or value < lower for value in values
            ):
                raise ValueError(f"{name} must contain one qualified integer per row.")
            object.__setattr__(self, name, values)
        validate_rigid_pose_batch(self.poses, size)
        object.__setattr__(self, "poses", self.poses.detach().clone())


class RecordedTrajectoryReader:
    """Read live recording buffers through an episode-qualified, read-only port.

    Args:
        source: Owner callback returning recorded states and completed-step
            counts, or None when recording is unavailable. Buffers stay with
            the recorder and are never exposed through this reader.
        episode_context: Recording owner's row-local episode lifecycle. Its
            identity prevents borrowing a reader from another environment.
        is_closed: Owner callback indicating whether recording access has ended.

    The recording host supplies these dependencies during assembly. This reader
    imports no environment or simulator and performs no physical acquisition.
    Each result is an independently owned snapshot of exact recorded slots.
    """

    def __init__(
        self,
        *,
        source: Callable[[], tuple[_TrajectoryState, torch.Tensor] | None],
        episode_context: EpisodeContext,
        is_closed: Callable[[], bool],
    ) -> None:
        if not all(callable(value) for value in (source, is_closed)):
            raise TypeError("Recording sources and lifetime checks must be callable.")
        self._source = source
        self._episode_context = episode_context
        self._is_closed = is_closed

    def _check_episode(self, expected: tuple[int, ...]) -> None:
        current = self._episode_context.snapshot()
        if (
            not current
            or len(expected) != len(current)
            or any(type(value) is not int or value < 0 for value in expected)
            or tuple(row["episode"] for row in current) != expected
            or not all(row["initialized"] for row in current)
        ):
            raise ValueError(
                "Recorded poses belong to another or uninitialized episode."
            )

    def _check_open(self) -> None:
        if self._is_closed():
            raise RuntimeError("Recorded observations are unavailable after close.")

    def validate_episode_source(self, episode_context: EpisodeContext) -> None:
        """Require this recording's actual episode owner before binding a consumer.

        Args:
            episode_context: Consumer's expected recording lifecycle owner.

        Raises:
            ValueError: The reader belongs to another recording episode owner.
        """
        if self._episode_context is not episode_context:
            raise ValueError("Recording reader belongs to another episode owner.")

    def read_rigid_poses(
        self,
        object_uid: str,
        frame_indices: Sequence[int],
        *,
        episode_ids: Sequence[int],
    ) -> RecordedRigidPoseSnapshot:
        """Copy one rigid object's exact recorded poses for the expected episode.

        Args:
            object_uid: Native UID selected for recording.
            frame_indices: Positive completed boundary slot for each row.
            episode_ids: Expected episode IDs captured before execution.

        Returns:
            Owned CPU matrices with environment, episode and frame identities.
            Missing slots never fall back to earlier frames or live state.

        Raises:
            RuntimeError: Recording is unavailable or its owner is closed.
            ValueError: An episode changed, a row is uninitialized, or the
                requested recording boundary is incomplete or invalid.
            KeyError: The selected native object was not recorded.
        """
        self._check_open()
        recording = self._source()
        if recording is None:
            raise RuntimeError("Rigid trajectory recording is unavailable.")
        expected = tuple(episode_ids)
        self._check_episode(expected)
        states, written_steps = recording
        result = read_recorded_rigid_pose_snapshot(
            states,
            written_steps,
            object_uid=object_uid,
            frame_indices=frame_indices,
            episode_ids=expected,
        )
        self._check_open()
        self._check_episode(expected)
        return result

    def read_rigid_pose_window(
        self,
        object_uid: str,
        start_frames: Sequence[int],
        end_frames: Sequence[int],
        *,
        episode_ids: Sequence[int],
    ) -> RecordedRigidPoseWindow:
        """Copy exact inclusive pose windows from this recording episode.

        Args:
            object_uid: Exact native rigid-object UID selected for recording.
            start_frames: First seeded slot for every environment row.
            end_frames: Last exact written slot for every row.
            episode_ids: Expected episode identities captured before execution.

        Returns:
            Owned CPU windows with each row's original length, dtype, and grid.

        Raises:
            RuntimeError: Recording is unavailable or its owner is closed.
            ValueError: An episode changed or a requested interval is invalid.
            KeyError: The selected native object was not recorded.
        """
        self._check_open()
        recording = self._source()
        if recording is None:
            raise RuntimeError("Rigid trajectory recording is unavailable.")
        expected = tuple(episode_ids)
        self._check_episode(expected)
        states, written_steps = recording
        result = read_recorded_rigid_pose_window(
            states,
            written_steps,
            object_uid=object_uid,
            start_frames=start_frames,
            end_frames=end_frames,
            episode_ids=expected,
        )
        self._check_open()
        self._check_episode(expected)
        return result


def read_recorded_rigid_pose_snapshot(
    states: _TrajectoryState,
    written_steps: torch.Tensor,
    *,
    object_uid: str,
    frame_indices: Sequence[int],
    episode_ids: Sequence[int],
) -> RecordedRigidPoseSnapshot:
    """Read seeded post-transition poses without live-state fallback.

    Args:
        states: Existing trajectory state buffers, including ``rigid_objects``.
        written_steps: Completed transition count for each recorded row.
        object_uid: Explicit native rigid-object UID.
        frame_indices: Positive exact boundary slot for each environment row.
        episode_ids: Host-validated episode identity for those buffers.

    Returns:
        Independently owned CPU poses at the requested end-step slots. The
        source dtype and xyzw-to-matrix conversion are preserved.

    Raises:
        ValueError: Recording is incomplete, a row/slot is missing, or poses
            are not qualified. This never substitutes ``end_step - 1`` or a
            current simulation pose.
        KeyError: The selected rigid object was not recorded.
    """
    from embodichain.utils.math import xyz_quat_to_4x4_matrix

    if (
        not isinstance(written_steps, torch.Tensor)
        or written_steps.ndim != 1
        or written_steps.dtype != torch.long
    ):
        raise ValueError("Recording requires an integer completed-step vector.")
    frames, episodes = tuple(frame_indices), tuple(episode_ids)
    size = written_steps.numel()
    if (
        len(frames) != size
        or len(episodes) != size
        or any(type(frame) is not int or frame <= 0 for frame in frames)
    ):
        raise ValueError("Recorded boundaries require one positive frame per row.")
    values = states["rigid_objects"][object_uid]["pose"]
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 3
        or values.shape[0] != size
        or values.shape[-1] != 7
    ):
        raise ValueError("Recorded rigid poses must contain every requested row.")
    completed = written_steps.detach().cpu().tolist()
    if any(frame > completed[row] for row, frame in enumerate(frames)):
        raise ValueError("The requested boundary is not fully recorded.")
    if any(frame >= values.shape[1] for frame in frames):
        raise ValueError("The seeded terminal state slot is unavailable.")
    pose = xyz_quat_to_4x4_matrix(
        torch.stack([values[row, frame] for row, frame in enumerate(frames)])
        .detach()
        .cpu()
    )
    return RecordedRigidPoseSnapshot(
        object_uid,
        tuple(range(size)),
        episodes,
        frames,
        pose,
    )


@dataclass(frozen=True, slots=True, eq=False)
class RecordedRigidPoseWindow:
    """Owned inclusive recorded pose windows, retaining each native row's length.

    No row is padded or resampled. Initial slot zero is available only when it
    was seeded, and the exact end slot must already have been written.
    """

    object_uid: str
    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    start_frames: tuple[int, ...]
    end_frames: tuple[int, ...]
    poses: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

        if (
            type(self.object_uid) is not str
            or not self.object_uid
            or self.object_uid != self.object_uid.strip()
        ):
            raise ValueError(
                "Recorded object UID must be an exact nonempty identifier."
            )
        size = len(self.env_ids)
        if not size or len(set(self.env_ids)) != size:
            raise ValueError("Recorded environment IDs must be nonempty and unique.")
        for name in ("env_ids", "episode_ids", "start_frames", "end_frames"):
            values = tuple(getattr(self, name))
            if len(values) != size or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(
                    f"{name} must contain one qualified nonnegative integer per row."
                )
            object.__setattr__(self, name, values)
        rows = tuple(self.poses)
        if len(rows) != size:
            raise ValueError("Recorded windows require every declared environment row.")
        owned = []
        for start, end, poses in zip(
            self.start_frames, self.end_frames, rows, strict=True
        ):
            if end < start:
                raise ValueError("Recorded window end cannot precede its start.")
            validate_rigid_pose_batch(poses, end - start + 1)
            owned.append(poses.detach().to(device="cpu", copy=True))
        object.__setattr__(self, "poses", tuple(owned))


def read_recorded_rigid_pose_window(
    states: _TrajectoryState,
    written_steps: torch.Tensor,
    *,
    object_uid: str,
    start_frames: Sequence[int],
    end_frames: Sequence[int],
    episode_ids: Sequence[int],
) -> RecordedRigidPoseWindow:
    """Read exact inclusive recorded pose windows without live-state fallback.

    Args:
        states: Existing trajectory state buffers including rigid_objects.
        written_steps: Completed transition count in every native row.
        object_uid: Exact native rigid-object UID in the recording selection.
        start_frames: First requested seeded slot in each row, including zero.
        end_frames: Last exact slot in each row, already written by execution.
        episode_ids: Host-validated episode identities for the stored buffers.

    Returns:
        Owned CPU matrices for each row's own inclusive interval. One bulk
        conversion per row preserves source dtype, time grid and xyzw ordering.

    Raises:
        ValueError: A row/window/seeded slot is absent, incomplete or invalid.
        KeyError: The selected rigid object was not recorded.
    """
    from embodichain.utils.math import xyz_quat_to_4x4_matrix

    if (
        not isinstance(written_steps, torch.Tensor)
        or written_steps.ndim != 1
        or written_steps.dtype != torch.long
    ):
        raise ValueError("Recording requires an integer completed-step vector.")
    starts, ends, episodes = tuple(start_frames), tuple(end_frames), tuple(episode_ids)
    size = written_steps.numel()
    if any(len(values) != size for values in (starts, ends, episodes)) or any(
        type(value) is not int or value < 0
        for values in (starts, ends, episodes)
        for value in values
    ):
        raise ValueError(
            "Recorded windows require one nonnegative boundary and episode per row."
        )
    if any(start > end for start, end in zip(starts, ends, strict=True)):
        raise ValueError("Recorded window end cannot precede its start.")
    values = states["rigid_objects"][object_uid]["pose"]
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 3
        or values.shape[0] != size
        or values.shape[-1] != 7
    ):
        raise ValueError("Recorded rigid poses must contain every requested row.")
    completed = written_steps.detach().cpu().tolist()
    if any(end > completed[row] for row, end in enumerate(ends)):
        raise ValueError("The requested window is not fully recorded.")
    if any(end >= values.shape[1] for end in ends):
        raise ValueError("The seeded terminal state slot is unavailable.")
    poses = tuple(
        xyz_quat_to_4x4_matrix(values[row, start : end + 1].detach().cpu())
        for row, (start, end) in enumerate(zip(starts, ends, strict=True))
    )
    return RecordedRigidPoseWindow(
        object_uid, tuple(range(size)), episodes, starts, ends, poses
    )
