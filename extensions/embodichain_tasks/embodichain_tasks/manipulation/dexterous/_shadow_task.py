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
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import torch

from embodichain.utils import configclass

__all__ = ["ShadowState", "ShadowTask", "ShadowTaskCfg", "shadow_config"]


class ShadowState(NamedTuple):
    """Batched simulation state in the source MJCF coordinate convention.

    Args:
        qpos: Shape ``(N, 31)``; 24 hand angles, world object position,
            then the object's world orientation as a wxyz quaternion.
        qvel: Shape ``(N, 30)``; 24 hand velocities, world object linear
            velocity, then object-frame angular velocity.
        palm_pos: World position of the ``robot0:palm`` body, shape ``(N, 3)``.
    """

    qpos: torch.Tensor
    qvel: torch.Tensor
    palm_pos: torch.Tensor


@configclass
class ShadowTaskCfg:
    """Task parameters retained from the two selected reference policies."""

    variant: str = "M1"
    ctrl_dt: float = 0.04
    sim_dt: float = 0.002
    episode_length: int = 500
    random_phase_starts: bool = False
    action_smoothing_beta: float = 1.0
    goal_mode: str = "hold"
    goal_step_rad: float = math.pi / 12
    quarter_hold_steps: int = 2
    static_goal_reward_scale: float = 2.0
    rotation_reward_scale: float = 3.0
    segment_progress_scale: float = 5.0
    segment_bonus: float = 5.0
    drop_penalty: float = 50.0
    spin_direction: float = -1.0
    maximum_tilt_deg: float = 45.0


def shadow_config(variant: str) -> ShadowTaskCfg:
    """Return the selected fixed-grasp or continuous-spin configuration."""
    if variant == "M1":
        return ShadowTaskCfg()
    if variant == "M2":
        return ShadowTaskCfg(
            variant="M2",
            random_phase_starts=True,
            action_smoothing_beta=0.7,
            goal_mode="milestone",
            static_goal_reward_scale=0.0,
        )
    raise ValueError(f"Unknown Shadow task variant: {variant}")


class ShadowTask:
    """Compute Shadow pen MDP terms without advancing a physics engine.

    Call ``reset`` and write its selected rows into the simulation. After a
    forward-kinematics update, call ``observe``. For each control interval,
    compute ``action_targets``, advance the simulation for 20 substeps, and call
    ``step`` with the resulting state and the same raw actions. The bridge owns
    the 500-step time limit and resets; ``step`` returns physical termination.
    """

    action_size = 18
    observation_size = 87
    qpos_size = 31
    qvel_size = 30
    axis_length = 0.2

    def __init__(
        self,
        cfg: ShadowTaskCfg,
        num_envs: int,
        q0: torch.Tensor,
        ctrl0: torch.Tensor,
        phase_starts: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> None:
        if num_envs < 1 or q0.shape != (31,) or ctrl0.shape != (20,):
            raise ValueError("Expected N >= 1, q0[31], and ctrl0[20]")
        if phase_starts.ndim != 2 or phase_starts.shape[1] != 31:
            raise ValueError("Expected phase_starts[num_phases, 31]")
        if lower.shape != (20,) or upper.shape != (20,):
            raise ValueError("Expected 20 actuator control bounds")
        if cfg.goal_mode not in ("hold", "milestone"):
            raise ValueError("goal_mode must be hold or milestone")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = q0.device
        self.dtype = q0.dtype
        self.q0 = q0.clone()
        self.ctrl0 = ctrl0.to(q0).clone()
        self.phase_starts = phase_starts.to(q0).clone()
        self.lower = lower.to(q0).clone()
        self.upper = upper.to(q0).clone()
        self.initial_pos = q0.new_zeros((num_envs, 3))
        self.goal = q0.new_full((num_envs,), cfg.goal_step_rad)
        self.held = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.turns = self.held.clone()
        self.rotation = q0.new_zeros(num_envs)
        self.last_heading = q0.new_zeros(num_envs)
        self.last_action = q0.new_zeros((num_envs, self.action_size))
        self.motor_targets = self.ctrl0.expand(num_envs, -1).clone()
        self.phase_index = torch.full(
            (num_envs,), -1, dtype=torch.int64, device=self.device
        )

    @classmethod
    def from_assets(
        cls,
        asset_root: str | Path,
        num_envs: int,
        variant: str = "M1",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ShadowTask:
        """Load retained reset arrays and actuator limits without loading physics."""
        import numpy as np

        assets = Path(asset_root)
        with np.load(assets / "ShadowPenSpin-initial.npz") as initial:
            q0 = torch.as_tensor(initial["qpos"].copy(), device=device, dtype=dtype)
            ctrl0 = torch.as_tensor(initial["ctrl"].copy(), device=device, dtype=dtype)
        with np.load(assets / "ShadowPenSpinPhases-initial-states.npz") as phases:
            starts = torch.as_tensor(phases["qpos"].copy(), device=device, dtype=dtype)
        actuator = ET.parse(assets / "shadow_pen.xml").getroot().find("actuator")
        bounds = torch.tensor(
            [[float(v) for v in item.attrib["ctrlrange"].split()] for item in actuator],
            device=device,
            dtype=dtype,
        )
        return cls(
            shadow_config(variant),
            num_envs,
            q0,
            ctrl0,
            starts,
            bounds[:, 0],
            bounds[:, 1],
        )

    @staticmethod
    def pen_axis(qpos: torch.Tensor) -> torch.Tensor:
        """Return the world unit vector along the object's local positive Z axis."""
        quat = qpos[:, 27:31]
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        w, x, y, z = quat.unbind(-1)
        return torch.stack(
            (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),
            dim=-1,
        )

    @classmethod
    def heading(cls, qpos: torch.Tensor) -> torch.Tensor:
        """Return the world XY heading of the pen's longitudinal axis."""
        axis = cls.pen_axis(qpos)
        return torch.atan2(axis[:, 1], axis[:, 0])

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        qpos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample selected reset states and clear only their task memory.

        Args:
            env_ids: Rows to reset; omission resets all environments.
            generator: Generator on the same device as the task tensors.
            qpos: Optional exact source-coordinate reset rows, used for fixed
                state comparisons. When supplied, no reset noise is added.

        Returns:
            Selected ``qpos[K,31]``, zero ``qvel[K,30]``, and ``ctrl[K,20]``.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        count = env_ids.numel()
        phases = torch.full_like(env_ids, -1)
        if qpos is None:
            if self.cfg.random_phase_starts:
                phases = torch.randint(
                    self.phase_starts.shape[0],
                    (count,),
                    device=self.device,
                    generator=generator,
                )
                qpos = self.phase_starts[phases].clone()
                sign = (
                    2
                    * torch.randint(
                        2, (count, 1), device=self.device, generator=generator
                    )
                    - 1
                )
                qpos[:, 27:31] *= sign
            else:
                qpos = self.q0.expand(count, -1).clone()
            qpos[:, 24:27] += (
                0.006
                * torch.rand(
                    (count, 3),
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )
                - 0.003
            )
            qpos[:, :24] += 0.02 * torch.randn(
                (count, 24),
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
        else:
            qpos = qpos.to(self.q0).clone()
            if qpos.shape != (count, self.qpos_size):
                raise ValueError("Exact reset qpos must have one row per env_id")
        self.initial_pos[env_ids] = qpos[:, 24:27]
        self.goal[env_ids] = self.cfg.goal_step_rad
        self.held[env_ids] = 0
        self.turns[env_ids] = 0
        self.rotation[env_ids] = 0
        self.last_heading[env_ids] = self.heading(qpos)
        self.last_action[env_ids] = 0
        self.motor_targets[env_ids] = self.ctrl0
        self.phase_index[env_ids] = phases
        return (
            qpos,
            qpos.new_zeros((count, self.qvel_size)),
            self.ctrl0.expand(count, -1).clone(),
        )

    def action_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Map raw policy actions to all 20 position-actuator targets."""
        beta = self.cfg.action_smoothing_beta
        smoothed = beta * actions + (1 - beta) * self.last_action
        motor = torch.cat((actions.new_zeros((self.num_envs, 2)), smoothed), dim=-1)
        return (self.upper + self.lower) / 2 + motor.clamp(-1, 1) * (
            self.upper - self.lower
        ) / 2

    def observe(self, state: ShadowState) -> torch.Tensor:
        """Return the reference policy's 87-dimensional observation."""
        delta = self.goal - self.rotation
        return torch.cat(
            (
                state.qpos[:, :24],
                0.05 * state.qvel[:, :24],
                self.last_action,
                10 * (state.qpos[:, 24:27] - self.initial_pos),
                state.qpos[:, 27:31],
                0.1 * state.qvel[:, 24:30],
                torch.stack((torch.sin(delta), torch.cos(delta)), dim=-1),
                10 * (state.qpos[:, 24:27] - state.palm_pos),
                torch.stack(
                    (
                        delta / math.pi,
                        self.held / self.cfg.quarter_hold_steps,
                        self.rotation / (2 * math.pi) - self.turns,
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        )

    def step(
        self, state: ShadowState, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Update task memory after physics and return observation and MDP terms.

        Args:
            state: State after applying ``action_targets(actions)`` for one
                control interval.
            actions: The raw actions passed to ``action_targets`` before physics.

        Returns:
            Observation, reward, physical termination, and reference task metrics.
            Time-limit truncation is handled by the simulation bridge.
        """
        cfg = self.cfg
        beta = cfg.action_smoothing_beta
        smoothed = beta * actions + (1 - beta) * self.last_action
        ctrl = self.action_targets(actions)
        pos = state.qpos[:, 24:27]
        drift = torch.linalg.vector_norm(pos - self.initial_pos, dim=-1)
        axis = self.pen_axis(state.qpos)
        gap = self.axis_length * axis[:, 2].abs()
        done = (
            (pos[:, 2] < 0.04)
            | (drift > 0.20)
            | ~torch.isfinite(state.qpos).all(-1)
            | (
                gap
                > self.axis_length * math.sin(math.radians(cfg.maximum_tilt_deg)) + 1e-6
            )
        )
        heading = torch.atan2(axis[:, 1], axis[:, 0])
        difference = heading - self.last_heading
        increment = torch.atan2(torch.sin(difference), torch.cos(difference))
        increment = increment * cfg.spin_direction
        old_rotation = self.rotation
        rotation = old_rotation + increment
        spin = increment / cfg.ctrl_dt
        cost = 0.02 * (smoothed - self.last_action).square().mean(-1)
        horizontal = (1 - (gap / 0.2).square()).clamp(0, 1)
        reward = (
            cfg.rotation_reward_scale * spin * horizontal - 15 * gap - 5 * drift - cost
        ) * cfg.ctrl_dt - cfg.drop_penalty * done.to(self.dtype)
        turns = torch.floor(rotation.clamp_min(0) / (2 * math.pi))
        success = (turns > self.turns) & (gap < 0.05) & ~done
        reward = reward + 5 * success.to(self.dtype)
        error = (self.goal - rotation).abs()
        previous_error = (self.goal - old_rotation).abs()
        good = (error < 0.1) & (spin.abs() < 1.5) & (gap < 0.05) & ~done
        held = torch.where(good, self.held + 1, 0)
        segment = held >= cfg.quarter_hold_steps
        if cfg.goal_mode == "milestone":
            segment = (rotation >= self.goal) & (gap < 0.05) & ~done
        reward = (
            reward
            + cfg.segment_progress_scale * (previous_error - error)
            + cfg.static_goal_reward_scale * torch.exp(-2 * error) * cfg.ctrl_dt
            + cfg.segment_bonus * segment.to(self.dtype)
        )
        self.goal = self.goal + segment.to(self.dtype) * cfg.goal_step_rad
        self.held = torch.where(segment, 0, held)
        self.turns = self.turns + success.to(torch.int64)
        self.rotation = rotation
        self.last_heading = heading
        self.last_action = smoothed
        self.motor_targets = ctrl
        metrics = {
            "success": success.to(self.dtype),
            "error": error,
            "rotation": rotation,
            "horizontal_gap": gap,
            "retained": (~done).to(self.dtype),
            "reward_spin": spin,
            "segment_success": segment.to(self.dtype),
        }
        return self.observe(state), reward, done, metrics
