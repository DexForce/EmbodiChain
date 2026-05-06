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

"""Differentiable reaching environment for APG training.

This module implements :class:`ReachDiffEnv`, a robot-agnostic differentiable
reaching task that inherits :class:`DiffEnv`.  The environment supports
two execution modes:

* **Differentiable (Newton/Warp)**: When Newton physics is available
  (``_newton_ready=True``), :meth:`ReachDiffEnv.step` dispatches to
  :meth:`_step_differentiable` which uses a Warp-tape bridge
  (``_NewtonStepFunc``) to flow gradients from the reward through FK back to
  the action tensor.  This is the mode used for APG training in production.

* **PyTorch fallback**: When Newton is unavailable (e.g. in tests), the
  default :class:`DiffEnv` step is used with a pure-PyTorch DH-based FK
  mock injected on the instance.  Both paths yield identical reward and
  observation shapes, so all unit tests pass without Newton.

A Franka FR3 preset :class:`FrankaReachDiffEnvCfg` is provided.

Reference:
    APG algorithm and Franka env from the ``analytic_policy_gradients``
    repository.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, field
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from embodichain.lab.gym.envs.diff_env import DiffEnv, DiffEnvCfg
from embodichain.utils import configclass

__all__ = [
    "ReachDiffEnv",
    "ReachDiffEnvCfg",
    "FrankaReachDiffEnvCfg",
    "compute_reach_reward",
    "compute_reach_obs",
    "quat_distance",
    "sample_target_pose",
    "check_reach_success",
]

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DEFAULT_ACTION_SCALE = 0.2  # radians
DEFAULT_MAX_EPISODE_STEPS = 30
DEFAULT_SUCCESS_POS_THRESHOLD = 0.01  # metres
DEFAULT_SUCCESS_ROT_THRESHOLD = 0.3  # quaternion distance

# Franka FR3 constants
FRANKA_NUM_ARM_JOINTS = 7
DEFAULT_ARM_JOINT_Q = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
DEFAULT_JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]

# Target sampling workspace (Franka FR3 default)
TARGET_POS_RANGE = {
    "x": (0.05, 0.70),
    "y": (-0.45, 0.45),
    "z": (0.2, 0.95),
}
TARGET_MAX_TILT = math.pi / 3  # ±60°


# ---------------------------------------------------------------------------
# Warp kernels for differentiable FK step
# ---------------------------------------------------------------------------


@wp.kernel
def _set_joint_targets_kernel(
    action: wp.array(dtype=wp.float32),  # [num_envs * num_arm_joints] flat
    current_q: wp.array(dtype=wp.float32),  # joint_q from Newton state
    target_pos: wp.array(dtype=wp.float32),  # output: new joint_q
    joint_limit_lower: wp.array(dtype=wp.float32),  # [num_arm_joints]
    joint_limit_upper: wp.array(dtype=wp.float32),  # [num_arm_joints]
    action_scale: wp.float32,
    num_joints_per_env: wp.int32,
    num_arm_joints: wp.int32,
    total_dims: wp.int32,  # num_envs * num_arm_joints
):
    """Compute new joint targets: target = clamp(current + action * scale, lo, hi)."""
    tid = wp.tid()
    if tid < total_dims:
        env_idx = tid / num_arm_joints
        j = tid % num_arm_joints
        q_offset = env_idx * num_joints_per_env + j
        new_q = current_q[q_offset] + action[tid] * action_scale
        target_pos[q_offset] = wp.clamp(
            new_q, joint_limit_lower[j], joint_limit_upper[j]
        )


@wp.kernel
def _compute_full_reward_kernel(
    body_q: wp.array(dtype=wp.transformf),
    ee_body_indices: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3f),
    target_quat: wp.array(dtype=wp.quatf),
    action: wp.array(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    num_arm_joints: wp.int32,
    reward_out: wp.array(dtype=wp.float32),
):
    """Full reward matching compute_reach_reward (position + orientation + action_rate)."""
    env_idx = wp.tid()
    ee_global = ee_body_indices[env_idx]

    ee_transform = body_q[ee_global]
    eef_pos = wp.transform_get_translation(ee_transform)
    eef_quat = wp.transform_get_rotation(ee_transform)

    # Position distance
    diff = eef_pos - target_pos[env_idx]
    pos_dist = wp.sqrt(wp.dot(diff, diff) + wp.float32(1e-8))

    # Quaternion distance (double cover)
    tq = target_quat[env_idx]
    dq_x = eef_quat.x - tq.x
    dq_y = eef_quat.y - tq.y
    dq_z = eef_quat.z - tq.z
    dq_w = eef_quat.w - tq.w
    d1 = dq_x * dq_x + dq_y * dq_y + dq_z * dq_z + dq_w * dq_w
    sq_x = eef_quat.x + tq.x
    sq_y = eef_quat.y + tq.y
    sq_z = eef_quat.z + tq.z
    sq_w = eef_quat.w + tq.w
    d2 = sq_x * sq_x + sq_y * sq_y + sq_z * sq_z + sq_w * sq_w
    rot_dist = wp.min(d1, d2)

    # Action rate
    action_rate = wp.float32(0.0)
    for j in range(num_arm_joints):
        idx = env_idx * num_arm_joints + j
        da = action[idx] - last_action[idx]
        action_rate = action_rate + da * da

    reward_out[env_idx] = (
        wp.float32(-0.2) * pos_dist
        + wp.float32(0.1) * wp.exp(-pos_dist * pos_dist / wp.float32(0.02))
        - wp.float32(0.1) * rot_dist
        - wp.float32(0.0001) * action_rate
    )


# ---------------------------------------------------------------------------
# Warp ↔ PyTorch autograd bridge
# ---------------------------------------------------------------------------


class _NewtonStepFunc(torch.autograd.Function):
    """Bridge Warp tape autodiff to PyTorch autograd for APG.

    For APG, we bypass the dynamics solver and compute reward directly from FK.
    The differentiable path is::

        action → joint_q → FK → body_q → EEF pose → reward

    **Forward**: runs FK inside a Warp tape, returns reward + EEF poses.
    **Backward**: uses the stored Warp tape to compute ``d(reward)/d(action)``.

    EEF pose gradients are intentionally left detached; the joint-position
    component of the observation is computed in PyTorch (see
    :meth:`ReachDiffEnv._step_differentiable`) so that multi-step credit
    assignment flows through ``obs[:, :num_joints]``.
    """

    @staticmethod
    def forward(ctx, action_torch, sim_state):
        import newton

        model = sim_state["model"]
        state_joint_q = sim_state["state_joint_q"]
        ee_indices_wp = sim_state["ee_body_indices_wp"]
        target_pos_t = sim_state["target_pos"]
        target_quat_t = sim_state["target_quat"]
        num_envs = sim_state["num_envs"]
        last_action_t = sim_state["last_action"]
        action_scale = sim_state["action_scale"]
        joint_limit_lower_wp = sim_state["joint_limit_lower_wp"]
        joint_limit_upper_wp = sim_state["joint_limit_upper_wp"]
        num_arm_joints = sim_state["num_arm_joints"]
        num_joints_per_env = sim_state["num_joints_per_env"]

        # Convert action to Warp with gradient tracking.
        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        # Last action as constant Warp array (no grad needed).
        last_action_flat = last_action_t.detach().clone().reshape(-1).contiguous()
        last_action_wp = wp.from_torch(last_action_flat, dtype=wp.float32)

        new_joint_q = wp.zeros(
            num_envs * num_joints_per_env,
            dtype=wp.float32,
            device=model.device,
            requires_grad=True,
        )

        # Build Warp target arrays (CPU list → Warp).
        target_pos_list = (
            target_pos_t.cpu().tolist()
            if isinstance(target_pos_t, torch.Tensor)
            else target_pos_t.tolist()
        )
        target_quat_list = (
            target_quat_t.cpu().tolist()
            if isinstance(target_quat_t, torch.Tensor)
            else target_quat_t.tolist()
        )
        target_pos_wp = wp.array(target_pos_list, dtype=wp.vec3f, device=model.device)
        target_quat_wp = wp.array(
            [wp.quatf(q[0], q[1], q[2], q[3]) for q in target_quat_list],
            dtype=wp.quatf,
            device=model.device,
        )
        reward_wp = wp.zeros(
            num_envs, dtype=wp.float32, device=model.device, requires_grad=True
        )

        fk_state = model.state()

        tape = wp.Tape()
        with tape:
            wp.launch(
                _set_joint_targets_kernel,
                dim=num_envs * num_arm_joints,
                inputs=[
                    action_wp,
                    state_joint_q,
                    new_joint_q,
                    joint_limit_lower_wp,
                    joint_limit_upper_wp,
                    wp.float32(action_scale),
                    wp.int32(num_joints_per_env),
                    wp.int32(num_arm_joints),
                    wp.int32(num_envs * num_arm_joints),
                ],
                device=model.device,
            )
            wp.copy(fk_state.joint_qd, model.joint_qd)
            newton.eval_fk(model, new_joint_q, fk_state.joint_qd, fk_state)
            wp.launch(
                _compute_full_reward_kernel,
                dim=num_envs,
                inputs=[
                    fk_state.body_q,
                    ee_indices_wp,
                    target_pos_wp,
                    target_quat_wp,
                    action_wp,
                    last_action_wp,
                    wp.int32(num_arm_joints),
                ],
                outputs=[reward_wp],
                device=model.device,
            )

        reward_t = wp.to_torch(reward_wp).detach().clone()

        # EEF poses: detached from the Warp tape (gradient via new_jpos in PyTorch).
        body_q_torch = wp.to_torch(fk_state.body_q)  # [total_bodies, 7]
        ee_indices_t = wp.to_torch(ee_indices_wp).long()
        eef_poses = body_q_torch[ee_indices_t].detach()  # [num_envs, 7]

        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.reward_wp = reward_wp

        return reward_t, eef_poses

    @staticmethod
    def backward(ctx, grad_reward, grad_eef_poses):
        # grad_eef_poses is ignored (eef_poses was detached in forward).
        grad_reward_wp = wp.from_torch(
            grad_reward.detach().clone().contiguous(), dtype=wp.float32
        )
        wp.copy(ctx.reward_wp.grad, grad_reward_wp)
        ctx.tape.backward()
        action_grad = (
            wp.to_torch(ctx.action_wp.grad).clone().reshape(grad_reward.shape[0], -1)
        )
        ctx.tape.zero()
        return action_grad, None


# ---------------------------------------------------------------------------
# Functor-style helpers
# ---------------------------------------------------------------------------


def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion distance handling double cover (q and -q represent same rotation).

    Args:
        q1: Quaternion tensor ``[batch, 4]``.
        q2: Quaternion tensor ``[batch, 4]``.

    Returns:
        Distance tensor ``[batch]``.
    """
    d1 = ((q1 - q2) ** 2).sum(dim=-1)
    d2 = ((q1 + q2) ** 2).sum(dim=-1)
    return torch.minimum(d1, d2)


def compute_reach_reward(
    eef_pos: torch.Tensor,
    eef_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    action: torch.Tensor | None = None,
    last_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute pose-tracking reward matching the reference APG implementation.

    Reward terms:

    * ``position_tracking``:             ``-0.2 * ||pos_ee - pos_cmd||``
    * ``position_tracking_fine_grained``: ``+0.1 * exp(-dist^2 / 0.02)``
    * ``orientation_tracking``:          ``-0.1 * quat_dist(q_ee, q_cmd)``
    * ``action_rate``:                   ``-1e-4 * ||action - prev_action||^2``

    Args:
        eef_pos: End-effector position ``[batch, 3]``.
        eef_quat: End-effector quaternion ``[batch, 4]``.
        target_pos: Target position ``[batch, 3]``.
        target_quat: Target quaternion ``[batch, 4]``.
        action: Current action ``[batch, num_joints]`` (optional).
        last_action: Previous action ``[batch, num_joints]`` (optional).

    Returns:
        Reward tensor ``[batch]``.
    """
    pos_dist = (eef_pos - target_pos).norm(dim=-1)
    rot_dist = quat_distance(eef_quat, target_quat)
    reward = (
        -0.2 * pos_dist
        + 0.1 * torch.exp(-(pos_dist**2) / (2 * 0.1**2))
        - 0.1 * rot_dist
    )
    if action is not None and last_action is not None:
        action_rate = ((action - last_action) ** 2).sum(dim=-1)
        reward = reward - 0.0001 * action_rate
    return reward


def compute_reach_obs(
    joint_q: torch.Tensor,
    eef_pos: torch.Tensor,
    eef_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    """Compose the observation vector for the reach task.

    Observation layout: ``[joint_q, eef_pos(3), eef_quat(4), target_pos(3),
    target_quat(4), last_action]``.  Total dimension = ``2 * num_joints + 14``.

    Args:
        joint_q: Joint positions ``[batch, num_joints]``.
        eef_pos: End-effector position ``[batch, 3]``.
        eef_quat: End-effector quaternion ``[batch, 4]``.
        target_pos: Target position ``[batch, 3]``.
        target_quat: Target quaternion ``[batch, 4]``.
        last_action: Previous action ``[batch, num_joints]``.

    Returns:
        Observation tensor ``[batch, obs_dim]``.
    """
    return torch.cat(
        [joint_q, eef_pos, eef_quat, target_pos, target_quat, last_action],
        dim=-1,
    )


def check_reach_success(
    eef_pos: torch.Tensor,
    eef_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    pos_threshold: float = DEFAULT_SUCCESS_POS_THRESHOLD,
    rot_threshold: float = DEFAULT_SUCCESS_ROT_THRESHOLD,
) -> torch.Tensor:
    """Check whether each environment has reached the target pose.

    Args:
        eef_pos: End-effector position ``[batch, 3]``.
        eef_quat: End-effector quaternion ``[batch, 4]``.
        target_pos: Target position ``[batch, 3]``.
        target_quat: Target quaternion ``[batch, 4]``.
        pos_threshold: Position threshold in metres.
        rot_threshold: Rotation threshold (quaternion distance).

    Returns:
        Boolean success tensor ``[batch]``.
    """
    pos_dist = (eef_pos - target_pos).norm(dim=-1)
    rot_dist = quat_distance(eef_quat, target_quat)
    return (pos_dist < pos_threshold) & (rot_dist < rot_threshold)


def sample_target_pose(
    num_envs: int,
    device: torch.device | str = "cpu",
    pos_range: dict[str, tuple[float, float]] | None = None,
    max_tilt: float = TARGET_MAX_TILT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random target poses within a reachable workspace.

    Args:
        num_envs: Number of target poses to sample.
        device: Target device.
        pos_range: Workspace bounds ``{"x": (lo, hi), "y": ..., "z": ...}``.
            Defaults to the Franka FR3 workspace if ``None``.
        max_tilt: Maximum tilt from the default downward orientation (radians).

    Returns:
        ``(target_pos, target_quat)`` — shapes ``[num_envs, 3]`` and
        ``[num_envs, 4]`` (x, y, z, w).
    """
    if pos_range is None:
        pos_range = TARGET_POS_RANGE

    target_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    target_pos[:, 0] = pos_range["x"][0] + torch.rand(num_envs, device=device) * (
        pos_range["x"][1] - pos_range["x"][0]
    )
    target_pos[:, 1] = pos_range["y"][0] + torch.rand(num_envs, device=device) * (
        pos_range["y"][1] - pos_range["y"][0]
    )
    target_pos[:, 2] = pos_range["z"][0] + torch.rand(num_envs, device=device) * (
        pos_range["z"][1] - pos_range["z"][0]
    )

    # Default orientation: EEF pointing down → (x=1, y=0, z=0, w=0).
    phi = torch.rand(num_envs, device=device) * 2 * math.pi
    tilt = torch.acos(
        1.0 - torch.rand(num_envs, device=device) * (1.0 - math.cos(max_tilt))
    )
    half_tilt = tilt / 2.0
    sin_ht = torch.sin(half_tilt)
    cos_ht = torch.cos(half_tilt)

    q_delta = torch.stack(
        [
            sin_ht * torch.cos(phi),
            sin_ht * torch.sin(phi),
            torch.zeros(num_envs, device=device),
            cos_ht,
        ],
        dim=-1,
    )
    q_base = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )

    dx, dy, dz, dw = q_delta[:, 0], q_delta[:, 1], q_delta[:, 2], q_delta[:, 3]
    bx, by, bz, bw = q_base[:, 0], q_base[:, 1], q_base[:, 2], q_base[:, 3]
    target_quat = torch.stack(
        [
            dw * bx + dx * bw + dy * bz - dz * by,
            dw * by - dx * bz + dy * bw + dz * bx,
            dw * bz + dx * by - dy * bx + dz * bw,
            dw * bw - dx * bx - dy * by - dz * bz,
        ],
        dim=-1,
    )
    target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)
    return target_pos, target_quat


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------


@configclass
class ReachDiffEnvCfg(DiffEnvCfg):
    """Configuration for the differentiable reach environment.

    Extends :class:`DiffEnvCfg` with reach-task-specific fields.
    Robot-specific parameters (URDF, joints, limits) must be provided by a
    subclass (e.g. :class:`FrankaReachDiffEnvCfg`).

    Args:
        success_pos_threshold: Position threshold for success (metres).
        success_rot_threshold: Rotation threshold for success (quat distance).
        target_pos_range: Workspace bounds for target sampling.
        target_max_tilt: Maximum tilt from downward for target orientation.
    """

    success_pos_threshold: float = DEFAULT_SUCCESS_POS_THRESHOLD
    success_rot_threshold: float = DEFAULT_SUCCESS_ROT_THRESHOLD
    target_pos_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(TARGET_POS_RANGE)
    )
    target_max_tilt: float = TARGET_MAX_TILT


@configclass
class FrankaReachDiffEnvCfg(ReachDiffEnvCfg):
    """Pre-configured :class:`ReachDiffEnvCfg` for the Franka FR3 robot.

    All Franka-specific kinematic parameters (URDF path, joint limits,
    default pose) are pre-filled.  Callers only need to set ``num_envs``.
    """

    urdf_path: str = "Franka/FR3/fr3.urdf"
    end_link_name: str = "fr3_hand_tcp"
    num_joints: int = FRANKA_NUM_ARM_JOINTS
    default_joint_q: list[float] = field(
        default_factory=lambda: list(DEFAULT_ARM_JOINT_Q)
    )
    joint_limits: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_JOINT_LIMITS)
    )
    action_scale: float = DEFAULT_ACTION_SCALE
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS


# ---------------------------------------------------------------------------
# Differentiable Reach Environment
# ---------------------------------------------------------------------------


class ReachDiffEnv(DiffEnv):
    """Batched differentiable reach environment for APG training.

    In production (Newton available), :meth:`step` dispatches to
    :meth:`_step_differentiable` which flows gradients through the Warp tape
    bridge ``_NewtonStepFunc``.  In test mode (``_newton_ready=False``), the
    pure-PyTorch fallback in :class:`DiffEnv` is used instead.

    The interface matches the APG loop::

        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    Attributes:
        cfg: Reach-task environment configuration.
        target_pos: Sampled target positions ``[num_envs, 3]``.
        target_quat: Sampled target quaternions ``[num_envs, 4]``.
        obs_dim: Observation dimension (``2 * num_joints + 14``).
    """

    cfg: ReachDiffEnvCfg

    def __init__(self, cfg: ReachDiffEnvCfg | None = None):
        if cfg is None:
            cfg = FrankaReachDiffEnvCfg()

        super().__init__(cfg)

        # Obs dim: joint_q + eef_pos(3) + eef_quat(4) + target_pos(3) + target_quat(4) + last_action
        self.obs_dim = 2 * cfg.num_joints + 14
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Target pose buffers (device-allocated by base __init__ already set device).
        self.target_pos = torch.zeros(
            cfg.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.target_quat = torch.zeros(
            cfg.num_envs, 4, dtype=torch.float32, device=self.device
        )

        # Warp joint-limit arrays for _NewtonStepFunc (only when Newton is available).
        if self._newton_ready:
            self._joint_limit_lower_wp = wp.array(
                self.joint_limit_lower.cpu().numpy().astype(np.float32),
                dtype=wp.float32,
                device=self._device_str,
            )
            self._joint_limit_upper_wp = wp.array(
                self.joint_limit_upper.cpu().numpy().astype(np.float32),
                dtype=wp.float32,
                device=self._device_str,
            )
        else:
            self._joint_limit_lower_wp = None
            self._joint_limit_upper_wp = None

    # ------------------------------------------------------------------
    # Backward-compatible property aliases
    # ------------------------------------------------------------------

    @property
    def arm_joint_limit_lower(self) -> torch.Tensor:
        """Alias for :attr:`joint_limit_lower`."""
        return self.joint_limit_lower

    @property
    def arm_joint_limit_upper(self) -> torch.Tensor:
        """Alias for :attr:`joint_limit_upper`."""
        return self.joint_limit_upper

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments.

        Dispatches to :meth:`_step_differentiable` (Newton/Warp) when Newton
        is available, otherwise falls back to the pure-PyTorch implementation
        in :class:`DiffEnv`.

        Args:
            action: Normalised actions ``[num_envs, num_joints]`` in ``[-1, 1]``.

        Returns:
            ``(obs, reward, terminated, truncated, info)``
        """
        if self._newton_ready and self.state_0 is not None:
            return self._step_differentiable(action)
        return super().step(action)

    def _step_differentiable(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step with differentiable Newton FK and Warp-tape gradient bridge.

        Implements the APG differentiable path::

            action → joint_q → Newton FK → EEF pose → reward

        The joint-position component of the observation is re-computed in
        PyTorch (without Warp) so that multi-step credit assignment flows
        through ``obs[:, :num_joints] → action``.

        Args:
            action: Normalised actions ``[num_envs, num_joints]``.

        Returns:
            ``(obs, reward, terminated, truncated, info)``
        """
        self.step_count += 1

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action.to(self._device), -1.0, 1.0)

        num_joints = self.cfg.num_joints

        sim_state = {
            "model": self.model,
            "state_joint_q": self.state_0.joint_q,
            "ee_body_indices_wp": self._ee_global,
            "target_pos": self.target_pos,
            "target_quat": self.target_quat,
            "num_envs": self._num_envs,
            "last_action": self.last_action,
            "action_scale": self.cfg.action_scale,
            "joint_limit_lower_wp": self._joint_limit_lower_wp,
            "joint_limit_upper_wp": self._joint_limit_upper_wp,
            "num_arm_joints": num_joints,
            "num_joints_per_env": self._model_joints_per_env,
        }

        reward, eef_poses = _NewtonStepFunc.apply(action, sim_state)

        # Differentiable joint positions in PyTorch — connected to action via autograd.
        # We read the *current* (pre-update) joint_q from the Warp state.
        current_q = (
            wp.to_torch(self.state_0.joint_q)
            .view(self._num_envs, self._model_joints_per_env)[:, :num_joints]
            .detach()
        )
        new_jpos = (current_q + action * self.cfg.action_scale).clamp(
            self.joint_limit_lower, self.joint_limit_upper
        )

        # Update Newton state and PyTorch joint_q (detached).
        with torch.no_grad():
            joint_q_t = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
            joint_q_t[:, :num_joints] += action.detach() * self.cfg.action_scale
            joint_q_t[:, :num_joints] = torch.clamp(
                joint_q_t[:, :num_joints],
                self.joint_limit_lower,
                self.joint_limit_upper,
            )
            self.joint_q = joint_q_t[:, :num_joints].clone()

        self.last_action = action.detach().clone()

        eef_pos = eef_poses[:, :3]
        eef_quat = eef_poses[:, 3:]
        self.eef_pos = eef_pos.detach().clone()
        self.eef_quat = eef_quat.detach().clone()

        # Observation: jpos differentiable through action; eef/target detached.
        obs = compute_reach_obs(
            new_jpos,
            eef_pos.detach(),
            eef_quat.detach(),
            self.target_pos,
            self.target_quat,
            self.last_action,
        )

        truncated = self.step_count >= self.max_episode_steps
        terminated = self._compute_terminated(eef_pos.detach(), eef_quat.detach())
        done_mask = truncated | terminated

        info = {
            "final_distance": (eef_pos.detach()[:, :3] - self.target_pos)
            .norm(dim=-1)
            .detach(),
            "success": terminated.detach(),
        }

        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self.reset(reset_ids)
            fresh_obs = self._compute_obs()
            obs = torch.where(done_mask.unsqueeze(-1).expand_as(obs), fresh_obs, obs)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # DiffEnv abstract method implementations
    # ------------------------------------------------------------------

    def _apply_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply delta-joint-position action (differentiable PyTorch path).

        Args:
            action: Normalised action ``[num_envs, num_joints]`` in ``[-1, 1]``.

        Returns:
            New joint positions ``[num_envs, num_joints]`` (differentiable).
        """
        current_q = self.joint_q.detach()
        return (current_q + action * self.cfg.action_scale).clamp(
            self.joint_limit_lower, self.joint_limit_upper
        )

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset joint positions and sample new target poses.

        Args:
            env_ids: Indices of environments to reset.
        """
        n = len(env_ids)
        default_q = torch.tensor(
            self.cfg.default_joint_q, dtype=torch.float32, device=self.device
        )
        scales = torch.empty(n, self.cfg.num_joints, device=self.device).uniform_(
            0.5, 1.5
        )
        arm_q = torch.clamp(
            default_q * scales,
            self.joint_limit_lower,
            self.joint_limit_upper,
        )
        self.joint_q[env_ids] = arm_q

        # If Newton is active, sync joint_q back into the Warp state.
        if self._newton_ready and self.state_0 is not None:
            with torch.no_grad():
                joint_q_t = wp.to_torch(self.state_0.joint_q).view(
                    self._num_envs, self._model_joints_per_env
                )
                joint_q_t[env_ids, : self.cfg.num_joints] = arm_q

        # Sample random target poses.
        target_pos, target_quat = sample_target_pose(
            n,
            device=self.device,
            pos_range=self.cfg.target_pos_range,
            max_tilt=self.cfg.target_max_tilt,
        )
        self.target_pos[env_ids] = target_pos
        self.target_quat[env_ids] = target_quat

    def _compute_obs(self) -> torch.Tensor:
        """Build observation from current (detached) state."""
        return compute_reach_obs(
            self.joint_q,
            self.eef_pos.detach(),
            self.eef_quat.detach(),
            self.target_pos,
            self.target_quat,
            self.last_action,
        )

    def _compute_obs_from_state(
        self,
        joint_q: torch.Tensor,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Build observation with possibly-differentiable joint positions.

        During the PyTorch fallback step, ``joint_q`` flows from the action
        and keeps gradients alive for APG backpropagation.
        """
        return compute_reach_obs(
            joint_q,
            eef_pos.detach(),
            eef_quat.detach(),
            self.target_pos,
            self.target_quat,
            self.last_action,
        )

    def _compute_reward(
        self,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
        info: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute reach-task reward (differentiable)."""
        return compute_reach_reward(
            eef_pos,
            eef_quat,
            self.target_pos,
            self.target_quat,
            action=action,
            last_action=self.last_action,
        )

    def _compute_terminated(
        self,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Check if end-effector has reached the target."""
        return check_reach_success(
            eef_pos,
            eef_quat,
            self.target_pos,
            self.target_quat,
            self.cfg.success_pos_threshold,
            self.cfg.success_rot_threshold,
        )

    def _compute_info(
        self,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> Dict[str, Any]:
        """Return step info with final distance."""
        return {
            "final_distance": (eef_pos - self.target_pos).norm(dim=-1),
        }
