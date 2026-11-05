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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import torch
from tensordict import TensorDictBase

from embodichain.utils import logger

__all__ = [
    "capture_trajectory_state",
    "restore_trajectory_state",
    "RecordedRigidPoseSnapshot",
    "read_recorded_rigid_pose_snapshot",
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
