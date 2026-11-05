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
from typing import Any, Protocol, TypeAlias, TYPE_CHECKING
from dataclasses import asdict, dataclass

import torch
from tensordict import TensorDictBase

from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.cfg import ObservationCfg
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager

__all__ = [
    "RecordedRobotJointSnapshot",
    "RecordedTrajectoryReader",
    "read_recorded_robot_joint_snapshot",
    "capture_trajectory_state",
    "restore_trajectory_state",
    "trajectory_observation_cfgs",
    "export_observation_history",
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


def trajectory_observation_cfgs(
    env: _TrajectoryEnv, uids: list[str] | None = None
) -> dict[str, ObservationCfg]:
    """Declare replay fields through the same raw observation functors as policy.

    Args:
        env: Host exposing robot and scene registries for schema assembly.
        uids: Optional allow-list for non-robot entities.

    Returns:
        Recording-only add functors. No physical state is acquired here; legacy
        trajectory field paths are export names, not another acquisition path.
    """
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.cfg import ObservationCfg, SceneEntityCfg
    from embodichain.lab.gym.envs.managers.observations import get_entity_state

    result = {}

    def add(prefix: str, uid: str, quantities: tuple[str, ...]) -> None:
        for quantity in quantities:
            field = (
                "root_pose"
                if quantity == "pose" and prefix != "rigid_objects"
                else quantity
            )
            name = f"{prefix}/{uid}/{field}" if prefix != "robot" else f"robot/{field}"
            result[f"recording:{name}"] = ObservationCfg(
                func=get_entity_state,
                mode="add",
                name=name,
                params={"entity_cfg": SceneEntityCfg(uid=uid), "quantity": quantity},
                consumers=("recording",),
            )

    add("robot", env.robot.uid, ("pose", "qpos", "qvel"))
    for uid in env.sim._articulations:
        if uids is None or uid in uids:
            add("articulations", uid, ("pose", "qpos", "qvel"))
    for uid in env.sim._rigid_objects:
        if uids is None or uid in uids:
            add("rigid_objects", uid, ("pose", "lin_vel", "ang_vel"))
    return result


def export_observation_history(
    observations: ObservationManager,
    env_ids: list[int] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Project numeric history through its single observation schema for export.

    Args:
        observations: Manager owning the field meanings and canonical IDs.
        env_ids: Selected original environment rows, or all rows.
        device: Snapshot destination device.

    Returns:
        Owned history with source descriptors and alias views. Existing
        trajectory state projections can share the same snapshot tensors.
    """
    snapshot = observations.snapshot_history(env_ids, device=device)
    fields = snapshot["fields"]
    for name, values in fields.items():
        metadata = asdict(observations.schema[name])
        metadata["dtype"] = str(observations.schema[name].dtype)
        values["descriptor"] = metadata
    snapshot["fields"] = {
        name: fields[field_id] for name, field_id in observations.field_ids.items()
    }
    return snapshot


@dataclass(frozen=True, slots=True, eq=False)
class RecordedRobotJointSnapshot:
    """Owned exact recorded robot positions on the original FK input device."""

    robot_uid: str
    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    qpos: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.robot_uid) is not str
            or not self.robot_uid
            or self.robot_uid != self.robot_uid.strip()
        ):
            raise ValueError("Recorded robot UID must be an exact identifier.")
        size = len(self.env_ids)
        if not size or len(set(self.env_ids)) != size:
            raise ValueError("Recorded robot rows must be nonempty and unique.")
        for name, lower in (("env_ids", 0), ("episode_ids", 0), ("frame_indices", 1)):
            values = tuple(getattr(self, name))
            if len(values) != size or any(
                type(value) is not int or value < lower for value in values
            ):
                raise ValueError(
                    "Recorded robot row, episode and slot identities are invalid."
                )
            object.__setattr__(self, name, values)
        if (
            not isinstance(self.qpos, torch.Tensor)
            or self.qpos.ndim != 2
            or self.qpos.shape[0] != size
            or not self.qpos.shape[1]
            or not self.qpos.is_floating_point()
            or not torch.isfinite(self.qpos).all()
        ):
            raise ValueError(
                "Recorded robot positions require finite floating (B, J) values."
            )
        object.__setattr__(self, "qpos", self.qpos.detach().clone())


def read_recorded_robot_joint_snapshot(
    states: _TrajectoryState,
    written_steps: torch.Tensor,
    *,
    robot_uid: str,
    frame_indices: Sequence[int],
    episode_ids: Sequence[int],
) -> RecordedRobotJointSnapshot:
    """Read exact seeded robot qpos slots without changing their device or dtype.

    Args:
        states: Existing recording state buffers containing the selected robot.
        written_steps: Completed transition counts for all native rows.
        robot_uid: Exact robot identity supplied by the recording owner.
        frame_indices: Positive completed end-step slots for every row.
        episode_ids: Host-validated episode identities for these buffers.

    Returns:
        Owned positions suitable for the original Robot FK path. No live qpos,
        previous slot or CPU reinterpretation substitutes for the requested data.
    """
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
        or any(type(value) is not int or value <= 0 for value in frames)
    ):
        raise ValueError("Recorded robot positions require one positive slot per row.")
    values = states["robot"]["qpos"]
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 3
        or values.shape[0] != size
        or not values.shape[-1]
    ):
        raise ValueError("Recorded robot positions must contain every native row.")
    completed = written_steps.detach().cpu().tolist()
    if any(frame > completed[row] for row, frame in enumerate(frames)):
        raise ValueError("The requested robot boundary is not fully recorded.")
    if any(frame >= values.shape[1] for frame in frames):
        raise ValueError("The seeded terminal robot state slot is unavailable.")
    return RecordedRobotJointSnapshot(
        robot_uid,
        tuple(range(size)),
        episodes,
        frames,
        torch.stack([values[row, frame] for row, frame in enumerate(frames)]),
    )


class RecordedTrajectoryReader:
    """Project canonical raw history into owned robot-joint snapshots.

    The reader owns no buffers, cursors or physical acquisition. The observation
    manager remains the only history schema and data source; a missing or invalid
    field never falls back to live state or an earlier recording slot.

    Args:
        observations: Host observation manager, or None when history is disabled.
        episode_context: Exact host episode lifecycle owner.
        is_closed: Read-only host lifetime callback.
        robot_identity: Optional native robot UID and opaque owner callback.
        step_counts: Optional host callback returning its canonical raw cursor.
    """

    def __init__(
        self,
        *,
        observations: ObservationManager | None,
        episode_context: EpisodeContext,
        is_closed: Callable[[], bool],
        robot_identity: Callable[[], tuple[str, object]] | None = None,
        step_counts: Callable[[], torch.Tensor | None] | None = None,
    ) -> None:
        if not callable(is_closed) or any(
            value is not None and not callable(value)
            for value in (robot_identity, step_counts)
        ):
            raise TypeError(
                "Recording identity, cursor and lifetime sources must be callable."
            )
        self._observations = observations
        self._episode_context = episode_context
        self._is_closed = is_closed
        self._robot_identity = robot_identity
        self._step_counts = step_counts

    def validate_episode_source(self, episode_context: EpisodeContext) -> None:
        """Require the consumer's actual recording lifecycle owner.

        Args:
            episode_context: Expected host lifecycle, compared by identity.
        """
        if episode_context is not self._episode_context:
            raise ValueError("Recording reader belongs to another episode owner.")

    def _guard(self, expected: tuple[int, ...]) -> None:
        if self._is_closed():
            raise RuntimeError("Recorded observations are unavailable after close.")
        current = self._episode_context.snapshot()
        if (
            not current
            or len(expected) != len(current)
            or any(type(value) is not int or value < 0 for value in expected)
            or tuple(row["episode"] for row in current) != expected
            or not all(row["initialized"] for row in current)
        ):
            raise ValueError(
                "Recorded positions belong to another or uninitialized episode."
            )
        if self._observations is None or not self._observations.has_history:
            raise RuntimeError("Observation history is unavailable.")
        self._observations.validate_history_owner(self._episode_context)

    def read_robot_qpos(
        self, frame_indices: Sequence[int], *, episode_ids: Sequence[int]
    ) -> RecordedRobotJointSnapshot:
        """Read exact positive slots without changing their dtype or device.

        Args:
            frame_indices: Completed seeded state slot for each native row.
            episode_ids: Expected initialized host episode identities.

        Returns:
            Owned positions and native row/robot identity. Unavailable raw
            acquisition raises with its original reasons, never a valid snapshot.
        """
        expected = tuple(episode_ids)
        self._guard(expected)
        if self._robot_identity is None:
            raise RuntimeError("Recorded robot identity is unavailable.")
        uid, owner = self._robot_identity()
        frames = tuple(frame_indices)
        if len(frames) != len(expected) or any(
            type(frame) is not int or frame <= 0 for frame in frames
        ):
            raise ValueError(
                "Recorded robot positions require one positive slot per row."
            )
        fields = [
            name
            for name, descriptor in self._observations.schema.items()
            if descriptor.entity_uid == uid
            and descriptor.quantity == "qpos"
            and descriptor.coordinate_frame == "joint"
            and descriptor.representation == "joint_order"
            and descriptor.units == ("joint_units",)
            and len(descriptor.shape) == 1
            and descriptor.shape[0] > 0
        ]
        if not fields:
            raise RuntimeError(
                f"Recorded raw joint positions for {uid!r} are unavailable."
            )
        field = fields[0]
        snapshot = self._observations.read_history(
            frames, fields=(field,), episode_ids=expected
        )
        if not bool(snapshot.valid[field].all()):
            raise RuntimeError(
                "Recorded robot acquisition is unavailable: "
                f"{snapshot.acquisition_errors[field]}."
            )
        result = RecordedRobotJointSnapshot(
            uid,
            snapshot.env_ids,
            snapshot.episode_ids,
            snapshot.frame_indices,
            snapshot.values[field],
        )
        self._guard(expected)
        next_uid, next_owner = self._robot_identity()
        if next_uid != uid or next_owner is not owner:
            raise ValueError(
                "The robot or episode changed during recorded qpos capture."
            )
        return result

    def read_step_counts(self, *, episode_ids: Sequence[int]) -> tuple[int, ...]:
        """Copy the host's completed raw boundaries without another history scan.

        Args:
            episode_ids: Expected initialized recording episode identities.

        Returns:
            Independently owned completed counts in original environment order.
        """
        expected = tuple(episode_ids)
        self._guard(expected)
        values = None if self._step_counts is None else self._step_counts()
        if (
            not isinstance(values, torch.Tensor)
            or values.dtype != torch.long
            or values.ndim != 1
            or values.numel() != len(expected)
            or bool((values < 0).any())
        ):
            raise ValueError("No valid recorded environment boundaries are available.")
        result = tuple(values.detach().cpu().tolist())
        self._guard(expected)
        return result
