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

from collections.abc import Callable
import math
from typing import TYPE_CHECKING

import torch

from .execution import JointCommand
from .runner import (
    CommandAcknowledgement,
    CommandAckStatus,
)
from .state import PlanningContext, RobotObservation, SceneSnapshot, TaskState

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager


SceneSnapshotSupplier = Callable[[float], SceneSnapshot]
"""Callback that returns the latest scene snapshot for a simulation timestamp."""


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
        scene_supplier: Optional callback for versioned scene observations.
        initial_time: Initial elapsed simulation time in seconds.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        *,
        physics_dt: float | None = None,
        env_ids: torch.Tensor | None = None,
        scene_supplier: SceneSnapshotSupplier | None = None,
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
        self.scene_supplier = scene_supplier
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
        if qeffort is None:
            qeffort = self._read_optional_proprioception_tensor("qf")
        scene = (
            SceneSnapshot(timestamp=self._elapsed_time, version=0)
            if self.scene_supplier is None
            else self.scene_supplier(self._elapsed_time)
        )
        if not isinstance(scene, SceneSnapshot):
            raise TypeError("scene_supplier must return a SceneSnapshot.")
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

    def _read_optional_proprioception_tensor(
        self,
        field_name: str,
    ) -> torch.Tensor | None:
        """Read an optional tensor from the robot proprioception mapping."""
        method = getattr(self.robot, "get_proprioception", None)
        if not callable(method):
            return None
        try:
            value = method()[field_name]
        except (AttributeError, KeyError, NotImplementedError, TypeError):
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


__all__ = ["SceneSnapshotSupplier", "SimulationExecutionAdapter"]
