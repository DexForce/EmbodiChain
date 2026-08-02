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

"""Simulation ports for :class:`~.runner.ExecutionRunner`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING

import torch

from embodichain.utils import configclass

from .execution import JointCommand
from .runner import (
    CommandAcknowledgement,
    CommandAckStatus,
)
from .scene import SceneProvider
from .state import (
    EntityState,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject, Robot
    from embodichain.lab.sim.sim_manager import SimulationManager


@configclass
class RigidObjectSceneProviderCfg:
    """Material-pose thresholds used to advance scene revisions."""

    translation_threshold: float = 1.0e-4
    """Minimum translation in metres considered a scene change."""

    rotation_threshold: float = 1.0e-3
    """Minimum rotation in radians considered a scene change."""

    def __post_init__(self) -> None:
        if self.translation_threshold < 0.0:
            raise ValueError("translation_threshold must be non-negative.")
        if self.rotation_threshold < 0.0:
            raise ValueError("rotation_threshold must be non-negative.")


class RigidObjectSceneProvider:
    """Observe simulation rigid objects and maintain scene revisions.

    The provider increments the general scene version when any tracked entity
    moves materially. For IDs declared as collision entities it additionally
    increments a per-environment collision-world revision, allowing one batch
    row to invalidate its trajectory without failing unrelated rows.

    Args:
        entities: Stable entity IDs mapped to live simulation rigid objects.
        collision_entity_ids: Tracked IDs consumed as dynamic planner obstacles.
        cfg: Optional material-change thresholds.
    """

    def __init__(
        self,
        entities: Mapping[str, RigidObject],
        *,
        collision_entity_ids: Sequence[str] = (),
        cfg: RigidObjectSceneProviderCfg | None = None,
    ) -> None:
        normalized = dict(entities)
        if not normalized:
            raise ValueError("entities must contain at least one rigid object.")
        if not all(
            isinstance(entity_id, str) and entity_id for entity_id in normalized
        ):
            raise ValueError("Scene entity IDs must be non-empty strings.")
        collision_ids = tuple(collision_entity_ids)
        if len(set(collision_ids)) != len(collision_ids):
            raise ValueError("collision_entity_ids must be unique.")
        missing = set(collision_ids).difference(normalized)
        if missing:
            raise ValueError(
                "collision_entity_ids reference untracked objects: "
                f"{sorted(missing)}."
            )
        self.entities = normalized
        self.collision_entity_ids = collision_ids
        self.cfg = cfg if cfg is not None else RigidObjectSceneProviderCfg()
        self._last_timestamp: float | None = None
        self._env_ids: torch.Tensor | None = None
        self._last_poses: dict[str, torch.Tensor] = {}
        self._scene_version = 0
        self._collision_revisions: list[int] = []

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        """Capture object poses and advance material-change revisions.

        Args:
            timestamp: Current simulation observation time.
            env_ids: Stable correlation IDs whose order matches object rows.

        Returns:
            Versioned scene snapshot with per-environment collision revisions.
        """
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Scene provider timestamps must be monotonic.")
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.dim() != 1
            or env_ids.numel() == 0
        ):
            raise ValueError("env_ids must be a non-empty 1D int64 tensor.")
        stable_ids = env_ids.detach().to("cpu")
        if self._env_ids is None:
            self._env_ids = stable_ids.clone()
            self._collision_revisions = [0] * int(env_ids.numel())
        elif not torch.equal(stable_ids, self._env_ids):
            raise ValueError("Scene provider env_ids must remain stable and ordered.")

        poses = {
            entity_id: self._read_pose(entity_id, entity, int(env_ids.numel()))
            for entity_id, entity in self.entities.items()
        }
        if self._last_poses:
            changed_by_entity = {
                entity_id: self._pose_change_mask(
                    self._last_poses[entity_id], current_pose
                )
                for entity_id, current_pose in poses.items()
            }
            if any(mask.any().item() for mask in changed_by_entity.values()):
                self._scene_version += 1
            collision_changed = torch.zeros(env_ids.numel(), dtype=torch.bool)
            for entity_id in self.collision_entity_ids:
                collision_changed |= changed_by_entity[entity_id]
            for row in collision_changed.nonzero(as_tuple=False).flatten().tolist():
                self._collision_revisions[row] += 1

        self._last_timestamp = timestamp
        self._last_poses = {
            entity_id: pose.clone() for entity_id, pose in poses.items()
        }
        return SceneSnapshot(
            timestamp=timestamp,
            version=self._scene_version,
            entities={
                entity_id: EntityState(pose) for entity_id, pose in poses.items()
            },
            collision_world_revision=tuple(self._collision_revisions),
            collision_entity_ids=self.collision_entity_ids,
        )

    @staticmethod
    def _read_pose(
        entity_id: str,
        entity: RigidObject,
        batch_size: int,
    ) -> torch.Tensor:
        """Read and validate one rigid-object pose batch."""
        pose = entity.get_local_pose(to_matrix=True)
        if not isinstance(pose, torch.Tensor):
            raise TypeError(
                f"Scene entity {entity_id!r} get_local_pose() must return a tensor."
            )
        if pose.shape == (4, 4):
            return pose.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if pose.shape != (batch_size, 4, 4):
            raise ValueError(
                f"Scene entity {entity_id!r} pose must have shape "
                f"({batch_size}, 4, 4)."
            )
        return pose.clone()

    def _pose_change_mask(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """Return a CPU mask of rows with material pose changes."""
        current = current.to(device=previous.device, dtype=previous.dtype)
        translation = torch.linalg.vector_norm(
            current[:, :3, 3] - previous[:, :3, 3], dim=1
        )
        relative_rotation = torch.bmm(
            previous[:, :3, :3].transpose(1, 2),
            current[:, :3, :3],
        )
        cosine = (
            (relative_rotation.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0
        ).clamp(-1.0, 1.0)
        rotation = torch.acos(cosine)
        return (
            (
                (translation > self.cfg.translation_threshold)
                | (rotation > self.cfg.rotation_threshold)
            )
            .detach()
            .to("cpu")
        )


class SimulationExecutionAdapter:
    """Adapt a simulation robot to observation, command, and clock protocols.

    The adapter writes joint targets synchronously. Time advances only through
    :meth:`sleep`, which converts the requested runner interval to an integral
    number of physics updates. This makes :meth:`ExecutionRunner.run_until_blocked`
    deterministic and avoids wall-clock sleeps in headless simulation.

    Args:
        simulation: Simulation manager advanced by the execution clock.
        robot: Robot observed and commanded by the adapter.
        physics_dt: Optional physics period. Defaults to the simulation config.
        env_ids: Optional stable correlation IDs matching every robot row. They
            are not used as simulator indices; row order maps to robot instances.
        scene_provider: Optional provider for versioned scene observations.
        initial_time: Initial elapsed simulation time in seconds.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        *,
        physics_dt: float | None = None,
        env_ids: torch.Tensor | None = None,
        scene_provider: SceneProvider | None = None,
        initial_time: float = 0.0,
    ) -> None:
        if not math.isfinite(initial_time) or initial_time < 0.0:
            raise ValueError("initial_time must be finite and non-negative.")
        resolved_physics_dt = (
            float(simulation.sim_config.physics_dt)
            if physics_dt is None
            else float(physics_dt)
        )
        if not math.isfinite(resolved_physics_dt) or resolved_physics_dt <= 0.0:
            raise ValueError("physics_dt must be finite and greater than zero.")
        qpos = robot.get_qpos()
        if not isinstance(qpos, torch.Tensor) or qpos.dim() != 2:
            raise ValueError("robot.get_qpos() must return shape (B, robot_dof).")
        if env_ids is None:
            env_ids = torch.arange(qpos.shape[0], dtype=torch.long, device=qpos.device)
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.shape != (qpos.shape[0],)
        ):
            raise ValueError("env_ids must be int64 with one ID per robot row.")
        if env_ids.device != qpos.device:
            raise ValueError("env_ids and robot state must share a device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")

        self.simulation = simulation
        self.robot = robot
        self.physics_dt = resolved_physics_dt
        self.env_ids = env_ids.clone()
        self._robot_env_indices = list(range(qpos.shape[0]))
        if scene_provider is not None and not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider.")
        self.scene_provider = scene_provider
        self._elapsed_time = float(initial_time)

    def now(self) -> float:
        """Return elapsed simulation time in seconds.

        Returns:
            Elapsed simulation time in seconds.
        """
        return self._elapsed_time

    def sleep(self, duration: float) -> None:
        """Advance physics by at least the requested duration.

        Args:
            duration: Requested simulated duration in seconds.
        """
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be finite and non-negative.")
        if duration == 0.0:
            return
        step_count = max(1, math.ceil(duration / self.physics_dt))
        self.simulation.update(physics_dt=self.physics_dt, step=step_count)
        self._elapsed_time += step_count * self.physics_dt

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Capture full-robot state and the latest supplied scene snapshot.

        Args:
            task_state: Verified symbolic state owned by the execution session.

        Returns:
            Planning context timestamped with elapsed simulation time.
        """
        qpos = self.robot.get_qpos()
        qvel = self._read_optional_tensor("get_qvel")
        if qvel is None:
            qvel = torch.zeros_like(qpos)
        qeffort = self._read_optional_tensor("get_qf")
        scene = (
            SceneSnapshot(timestamp=self._elapsed_time, version=0)
            if self.scene_provider is None
            else self.scene_provider.snapshot(
                timestamp=self._elapsed_time,
                env_ids=self.env_ids,
            )
        )
        if not isinstance(scene, SceneSnapshot):
            raise TypeError("scene_provider must return a SceneSnapshot.")
        return PlanningContext(
            robot=RobotObservation(
                timestamp=self._elapsed_time,
                qpos=qpos,
                qvel=qvel,
                qeffort=qeffort,
            ),
            task=task_state,
            scene=scene,
            env_ids=self.env_ids,
        )

    def send(
        self,
        command: JointCommand,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Write active targets and observed-position holds as one batch.

        Args:
            command: Full-robot batched command. Inactive rows already contain
                observed-position holds and are written with active rows so no
                environment continues tracking a stale target.
            timeout: Positive acknowledgement deadline. Simulation writes are
                synchronous, so this is validated but otherwise unused.

        Returns:
            Accepted acknowledgement or a rejected diagnostic.
        """
        self._validate_timeout(timeout)
        try:
            self._validate_command(command)
            if not command.active_mask.any():
                return CommandAcknowledgement.accepted_ack("No active rows.")
            self.robot.set_qpos(
                command.positions,
                env_ids=self._robot_env_indices,
            )
            if command.velocities is not None:
                self.robot.set_qvel(
                    command.velocities,
                    env_ids=self._robot_env_indices,
                )
            return CommandAcknowledgement.accepted_ack()
        except Exception as exc:
            return CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                f"{type(exc).__name__}: {exc}",
            )

    def hold(
        self,
        command: JointCommand,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Set every represented environment to an observed-position hold.

        Args:
            command: Full-robot hold positions. ``active_mask`` is intentionally
                ignored because safety hold applies to every environment row.
            timeout: Positive acknowledgement deadline.

        Returns:
            Accepted acknowledgement or a rejected diagnostic.
        """
        self._validate_timeout(timeout)
        try:
            self._validate_command(command)
            self.robot.set_qpos(
                command.positions,
                env_ids=self._robot_env_indices,
            )
            if command.velocities is not None:
                self.robot.set_qvel(
                    command.velocities,
                    env_ids=self._robot_env_indices,
                )
            return CommandAcknowledgement.accepted_ack()
        except Exception as exc:
            return CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                f"{type(exc).__name__}: {exc}",
            )

    def cancel(self, *, timeout: float) -> CommandAcknowledgement:
        """Acknowledge cancellation of synchronous simulation target writes.

        Args:
            timeout: Positive acknowledgement deadline.

        Returns:
            Accepted acknowledgement. The following ``hold`` call installs the
            actual safe target.
        """
        self._validate_timeout(timeout)
        return CommandAcknowledgement.accepted_ack(
            "Simulation commands are synchronous; no queued command remained."
        )

    def _read_optional_tensor(self, method_name: str) -> torch.Tensor | None:
        """Read an optional full-robot tensor from the robot API."""
        method = getattr(self.robot, method_name, None)
        if not callable(method):
            return None
        try:
            value = method()
        except (AttributeError, NotImplementedError):
            return None
        return value if isinstance(value, torch.Tensor) else None

    def _validate_command(self, command: JointCommand) -> None:
        """Validate command identity and shape against the attached robot."""
        if not isinstance(command, JointCommand):
            raise TypeError("command must be a JointCommand.")
        qpos = self.robot.get_qpos()
        if command.positions.shape != qpos.shape:
            raise ValueError(
                "Command shape must match full robot qpos, "
                f"got {tuple(command.positions.shape)} and {tuple(qpos.shape)}."
            )
        if not torch.equal(command.env_ids, self.env_ids):
            raise ValueError("Command env_ids must match the simulation adapter.")

    @staticmethod
    def _validate_timeout(timeout: float) -> None:
        """Validate an acknowledgement timeout."""
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be finite and greater than zero.")


__all__ = [
    "RigidObjectSceneProvider",
    "RigidObjectSceneProviderCfg",
    "SimulationExecutionAdapter",
]
