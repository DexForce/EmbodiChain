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

"""Fixed-cadence execution of timed joint trajectories in simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import torch

from embodichain.compute.trajectory import retime_to_control_grid

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

JointCommandMode = Literal["position", "position_velocity"]

__all__ = [
    "JointTrajectoryPlaybackCfg",
    "play_joint_trajectory",
]


@dataclass(frozen=True, slots=True)
class JointTrajectoryPlaybackCfg:
    """Fixed-cadence standalone simulation playback settings."""

    control_dt: float | None = None
    """Command period, or ``None`` to use the simulation physics period."""

    joint_command_mode: JointCommandMode = "position"
    """Whether playback writes qpos alone or qpos and qvel targets."""

    def __post_init__(self) -> None:
        if self.control_dt is not None and (
            isinstance(self.control_dt, bool)
            or not isinstance(self.control_dt, (int, float))
            or not math.isfinite(self.control_dt)
            or self.control_dt <= 0
        ):
            raise ValueError("control_dt must be a positive finite number or None.")
        if self.joint_command_mode not in ("position", "position_velocity"):
            raise ValueError(
                "joint_command_mode must be 'position' or 'position_velocity'."
            )


def _resolve_playback_timing(
    sim: SimulationManager,
    control_dt: float | None,
) -> tuple[float, float, int]:
    """Resolve one fixed command period as exact simulator substeps."""
    physics_dt = getattr(getattr(sim, "sim_config", None), "physics_dt", None)
    if (
        isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or not math.isfinite(physics_dt)
        or physics_dt <= 0
    ):
        raise ValueError("sim.sim_config.physics_dt must be positive and finite.")
    destination_dt = float(physics_dt if control_dt is None else control_dt)
    ratio = destination_dt / float(physics_dt)
    steps = round(ratio)
    tolerance = max(1.0e-9, abs(ratio) * 1.0e-7)
    if steps <= 0 or not math.isclose(
        ratio,
        steps,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("control_dt must be an integer multiple of physics_dt.")
    return float(physics_dt), destination_dt, steps


def play_joint_trajectory(
    sim: SimulationManager,
    robot: Robot,
    *,
    positions: torch.Tensor,
    dt: torch.Tensor,
    joint_ids: Sequence[int],
    cfg: JointTrajectoryPlaybackCfg | None = None,
) -> None:
    """Play a timed joint path without changing the simulation physics period.

    Planner timing is first quantized to the configured fixed command period.
    Every command interval advances an integer number of unchanged physics
    substeps. Velocity targets are recomputed on that executed grid.

    Args:
        sim: Simulation manager in explicit-update mode.
        robot: Robot receiving joint targets.
        positions: Source positions with shape ``(B, N, D)``.
        dt: Source arrival intervals with shape ``(B, N)``.
        joint_ids: Full-robot joint columns addressed by ``positions``.
        cfg: Playback settings. Defaults to position-only at ``physics_dt``.
    """
    if cfg is None:
        cfg = JointTrajectoryPlaybackCfg()
    if not isinstance(cfg, JointTrajectoryPlaybackCfg):
        raise TypeError("cfg must be a JointTrajectoryPlaybackCfg or None.")
    resolved_joint_ids = list(joint_ids)
    if (
        not resolved_joint_ids
        or any(
            type(joint_id) is not int or joint_id < 0 for joint_id in resolved_joint_ids
        )
        or len(set(resolved_joint_ids)) != len(resolved_joint_ids)
    ):
        raise ValueError("joint_ids must contain unique non-negative integers.")
    if not isinstance(positions, torch.Tensor) or positions.dim() != 3:
        raise ValueError("positions must have shape (B, N, D).")
    if positions.shape[-1] != len(resolved_joint_ids):
        raise ValueError("positions DoF must match joint_ids.")

    physics_dt, control_dt, physics_steps = _resolve_playback_timing(
        sim,
        cfg.control_dt,
    )
    positions, velocities, _, _ = retime_to_control_grid(
        positions,
        dt,
        control_dt,
    )
    for waypoint in range(positions.shape[1] - 1):
        robot.set_qpos(
            positions[:, waypoint],
            joint_ids=resolved_joint_ids,
        )
        if cfg.joint_command_mode == "position_velocity":
            robot.set_qvel(
                velocities[:, waypoint],
                joint_ids=resolved_joint_ids,
            )
        sim.update(physics_dt, physics_steps)

    robot.set_qpos(positions[:, -1], joint_ids=resolved_joint_ids)
    if cfg.joint_command_mode == "position_velocity":
        robot.set_qvel(velocities[:, -1], joint_ids=resolved_joint_ids)
