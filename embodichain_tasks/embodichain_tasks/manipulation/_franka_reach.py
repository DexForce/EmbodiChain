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
"""Franka FR3 6D end-effector reaching task using Newton Physics.

Supports two modes:
  - PPO: Standard gymnasium.Env (black-box, no gradients)
  - APG: Differentiable env with Warp tape -> PyTorch gradient bridge
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import newton
import newton.utils
import newton.viewer as newton_viewer

from embodichain.utils import set_seed

# Franka FR3 arm has 7 actuated arm joints.
FRANKA_NUM_ARM_JOINTS = 7
FRANKA_EE_BODY_NAME = "fr3_hand_tcp"  # last link (flange) in fr3.urdf
DEFAULT_ACTION_SCALE = 0.2  # radians - very small for physics stability
DEFAULT_MAX_EPISODE_STEPS = 30
DEFAULT_W_ROT = 1.0
DEFAULT_SUCCESS_POS_THRESHOLD = 0.01  # meters
DEFAULT_SUCCESS_ROT_THRESHOLD = 0.1  # rotation angle in radians
WORLD_OFFSET = 1.5

# Default initial joint configuration for Franka (home position):
# 7 arm joints + 2 finger joints.
DEFAULT_ARM_JOINT_Q = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
DEFAULT_FINGER_Q = [0.0, 0.0]
DEFAULT_JOINT_Q = DEFAULT_ARM_JOINT_Q + DEFAULT_FINGER_Q
FRANKA_NUM_JOINTS = len(DEFAULT_JOINT_Q)

# Target sampling workspace — full feasible reach of Franka FR3 (~855mm reach).
# x: forward from base; y: lateral; z: vertical (base joint at z≈0.333).
TARGET_POS_RANGE = {
    "x": (0.05, 0.70),
    "y": (-0.45, 0.45),
    "z": (0.2, 0.95),
}
# Maximum tilt angle (radians) from the default downward orientation.
# 0 = EE straight down; pi = any orientation.
TARGET_MAX_TILT = math.pi / 3  # ±60°
TARGET_AXIS_LENGTH = 0.1  # meters, length of each axis line for target visualization


def _build_viewer(model, headless: bool, num_envs: int = 1):
    """Create Newton viewer with explicit headless control.

    When num_envs > 1, applies world offsets so every robot is visible in a
    single scene, matching the pattern used in Newton example_robot_h1.py.
    """
    if headless:
        viewer = newton_viewer.ViewerNull()
    else:
        viewer = newton_viewer.ViewerGL(headless=False)
    viewer.set_model(model)
    # if num_envs > 1:
    #     viewer.set_world_offsets((WORLD_OFFSET, WORLD_OFFSET, 0.0))
    return viewer


def _compute_world_offsets(num_envs, spacing=(WORLD_OFFSET, WORLD_OFFSET, 0.0)):
    """Compute per-environment grid offsets matching Newton's ``compute_world_offsets``."""
    if num_envs <= 1:
        return np.zeros((1, 3), dtype=np.float32)
    spacing = np.array(spacing, dtype=np.float32)
    nonzeros = np.nonzero(spacing)[0]
    num_dim = nonzeros.shape[0]
    if num_dim == 0:
        return np.zeros((num_envs, 3), dtype=np.float32)
    side = int(np.ceil(num_envs ** (1.0 / num_dim)))
    offsets = np.zeros((num_envs, 3), dtype=np.float32)
    if num_dim == 2:
        for i in range(num_envs):
            offsets[i, nonzeros[0]] = (i // side) * spacing[nonzeros[0]]
            offsets[i, nonzeros[1]] = (i % side) * spacing[nonzeros[1]]
    elif num_dim == 1:
        for i in range(num_envs):
            offsets[i, nonzeros[0]] = i * spacing[nonzeros[0]]
    # Center the grid (match Newton: keep up-axis correction at zero)
    mn = offsets.min(axis=0)
    mx = offsets.max(axis=0)
    corr = mn + (mx - mn) / 2.0
    corr[2] = 0.0
    offsets -= corr
    return offsets


def _quat_to_rotmat(q):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )


def _resolve_urdf_path():
    """Resolve Franka URDF path, trying multiple locations."""
    # 1. Try Newton asset cache (requires network for first download)
    try:
        import os

        if os.environ.get("_NEWTON_SKIP_DOWNLOAD"):
            raise RuntimeError("skip")
        urdf = (
            newton.utils.download_asset("franka_emika_panda")
            / "urdf/fr3_franka_hand.urdf"
        )
        if urdf.exists():
            return str(urdf)
    except Exception:
        pass

    # 2. Try local fallback paths
    candidates = [
        Path.home() / "Downloads" / "robot_challenge_model" / "franka_fr3" / "fr3.urdf",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Franka URDF not found. Either set up network access for Newton asset download, "
        "or place the FR3 URDF at one of: " + ", ".join(str(c) for c in candidates)
    )


def _build_franka_model(num_envs=1, requires_grad=False, device="cpu", urdf_path=None):
    """Build a Newton model with Franka FR3 robot(s).

    Returns (model, ee_body_index).
    """
    cache_dir = os.environ.get("WARP_CACHE_DIR")
    if cache_dir:
        wp.config.kernel_cache_dir = cache_dir

    wp.init()

    if urdf_path is None:
        urdf_path = _resolve_urdf_path()

    if num_envs == 1:
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_urdf(
            str(urdf_path),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        for i, q in enumerate(DEFAULT_JOINT_Q):
            if i < len(builder.joint_q):
                builder.joint_q[i] = q
                builder.joint_target_pos[i] = q
        model = builder.finalize(device=device, requires_grad=requires_grad)

        ee_index = None
        for i, key in enumerate(builder.body_label):
            if FRANKA_EE_BODY_NAME in str(key):
                ee_index = i
                break
        if ee_index is None:
            ee_index = 10
        return model, ee_index
    else:
        robot_builder = newton.ModelBuilder()
        robot_builder.add_urdf(
            str(urdf_path),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        for i, q in enumerate(DEFAULT_JOINT_Q):
            if i < len(robot_builder.joint_q):
                robot_builder.joint_q[i] = q
                robot_builder.joint_target_pos[i] = q

        ee_index = None
        for i, key in enumerate(robot_builder.body_label):
            if FRANKA_EE_BODY_NAME in str(key):
                ee_index = i
                break
        if ee_index is None:
            ee_index = 10

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        # Use spacing=(0,0,0) so all worlds are at the physical origin; the
        # viewer applies visual offsets independently via world_offsets.
        # This keeps body_q in local (per-env) frame and fixes reward maths.
        builder.replicate(
            robot_builder,
            world_count=num_envs,
            spacing=(0.0, 0.0, 0.0),
        )
        model = builder.finalize(device=device, requires_grad=requires_grad)

        return model, ee_index


def _sample_target(num_envs, device="cpu"):
    """Sample random target poses within reachable workspace. Returns torch tensors."""
    target_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    target_pos[:, 0] = TARGET_POS_RANGE["x"][0] + torch.rand(
        num_envs, device=device
    ) * (TARGET_POS_RANGE["x"][1] - TARGET_POS_RANGE["x"][0])
    target_pos[:, 1] = TARGET_POS_RANGE["y"][0] + torch.rand(
        num_envs, device=device
    ) * (TARGET_POS_RANGE["y"][1] - TARGET_POS_RANGE["y"][0])
    target_pos[:, 2] = TARGET_POS_RANGE["z"][0] + torch.rand(
        num_envs, device=device
    ) * (TARGET_POS_RANGE["z"][1] - TARGET_POS_RANGE["z"][0])

    # Default orientation: EE pointing down (180° around X) → quat (x,y,z,w) = (1,0,0,0).
    # Perturb with random tilt (bounded cone) + full rotation around the approach axis.
    # Random rotation axis on XY plane (perpendicular to approach direction Z):
    phi = torch.rand(num_envs, device=device) * 2 * math.pi
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    # Tilt angle sampled uniformly in cos(theta) for uniform cone sampling
    tilt = torch.acos(
        1.0 - torch.rand(num_envs, device=device) * (1.0 - math.cos(TARGET_MAX_TILT))
    )
    half_tilt = tilt / 2.0
    sin_ht = torch.sin(half_tilt)
    cos_ht = torch.cos(half_tilt)
    # Delta quaternion from identity: q_delta = [sin_ht*cos_phi, sin_ht*sin_phi, 0, cos_ht]
    q_delta = torch.stack(
        [
            sin_ht * cos_phi,
            sin_ht * sin_phi,
            torch.zeros(num_envs, device=device),
            cos_ht,
        ],
        dim=-1,
    )
    # Base orientation (EE down): [1, 0, 0, 0]
    q_base = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )
    # Quaternion multiply: q_result = q_delta * q_base
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
    # Normalize
    target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)
    return target_pos, target_quat


def _quat_distance(q1, q2):
    """Squared chordal quaternion distance used by the dense reward."""
    d1 = ((q1 - q2) ** 2).sum(dim=-1)
    d2 = ((q1 + q2) ** 2).sum(dim=-1)
    return torch.minimum(d1, d2)


def _quat_angle_distance(q1, q2):
    """Shortest rotation angle in radians, invariant to quaternion sign."""
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _canonicalize_quat(q):
    """Canonicalize quaternion(s) to a single hemisphere.

    q: (..., 4) laid out as [x, y, z, w]. Flips sign so the first non-zero
    component of (w, x, y, z) is positive, resolving the q / -q double cover.
    The sign mask is detached, so the op is differentiable with a piecewise +/-1
    gradient.
    """
    w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    sign = torch.sign(w)
    m = w == 0
    sign = torch.where(m, torch.sign(x), sign)
    m = m & (x == 0)
    sign = torch.where(m, torch.sign(y), sign)
    m = m & (y == 0)
    sign = torch.where(m, torch.sign(z), sign)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(-1)


def _quat_inverse(q):
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def _quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([x, y, z, w], dim=-1)


def _compute_reward(
    eef_pos, eef_quat, target_pos, target_quat, action=None, last_action=None
):
    """Pose-tracking reward adapted from IsaacLab ReachEnvCfg.

    Terms (weights match IsaacLab RewardsCfg, joint_vel term omitted):
      - position_tracking:             -0.2  * ||pos_ee - pos_cmd||
      - position_tracking_fine_grained: +0.1 * exp(-dist^2 / (2*0.1^2))
      - orientation_tracking:          -0.1  * quat_dist(q_ee, q_cmd)
      - orientation_tracking_fine:     +0.1 * exp(-rot_dist^2 / (2*0.3^2))
      - action_rate:                   -1e-4 * ||action - prev_action||^2
    """
    pos_dist = (eef_pos - target_pos).norm(dim=-1)
    rot_dist = _quat_distance(eef_quat, target_quat)
    reward = (
        -0.2 * pos_dist
        + 0.1 * torch.exp(-(pos_dist**2) / (2 * 0.1**2))
        - 0.1 * rot_dist
        + 0.1 * torch.exp(-(rot_dist**2) / (2 * 0.3**2))
    )
    if action is not None and last_action is not None:
        action_rate = ((action - last_action) ** 2).sum(dim=-1)
        reward = reward - 0.0001 * action_rate
    return reward


def _check_success(
    eef_pos,
    eef_quat,
    target_pos,
    target_quat,
    pos_threshold=DEFAULT_SUCCESS_POS_THRESHOLD,
    rot_threshold=DEFAULT_SUCCESS_ROT_THRESHOLD,
):
    """Check whether each environment has reached the target pose.

    Returns a boolean tensor of shape ``[num_envs]`` — ``True`` where the
    end-effector is within threshold of the target.
    """
    pos_dist = (eef_pos - target_pos).norm(dim=-1)
    rot_dist = _quat_angle_distance(eef_quat, target_quat)
    return (pos_dist <= pos_threshold) & (rot_dist <= rot_threshold)


class FrankaReachVecEnv:
    """Batched Franka reach environment for vectorized PPO.

    All robots live in a single Newton model (via ``ModelBuilder.replicate``)
    and are rendered in one shared viewer window, matching the pattern used in
    Newton's ``example_robot_h1.py``.

    Returns torch tensors directly for use in the PPO training loop:
      - ``reset()`` → ``(obs [num_envs, 28] Tensor, info)``
      - ``step(action [num_envs, 7] Tensor)`` → ``(obs Tensor, reward Tensor, terminated Tensor, truncated Tensor, info)``

    Attributes:
        single_observation_space: Observation space for one environment.
        single_action_space: Action space for one environment.
        num_envs: Number of parallel environments.
    """

    def __init__(
        self,
        num_envs: int = 4,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cpu",
        headless: bool = True,
        requires_grad: bool = False,
        canonicalize_quat_obs: bool = False,
    ):
        self._num_envs = num_envs
        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.headless = headless
        self.canonicalize_quat_obs = bool(canonicalize_quat_obs)
        self.frame_dt = 1.0 / 60

        self.model, _ = _build_franka_model(
            num_envs=num_envs,
            requires_grad=requires_grad,
            device=device,
        )

        self.state_0 = self.model.state()

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

        # Global EEF body index per replicated environment.
        ee_indices = [
            i
            for i, key in enumerate(self.model.body_label)
            if FRANKA_EE_BODY_NAME in str(key)
        ]

        if len(ee_indices) != num_envs:
            raise RuntimeError(
                f"Expected {num_envs} '{FRANKA_EE_BODY_NAME}' bodies, "
                f"found {len(ee_indices)}."
            )
        # Assign index 0 to all EEF bodies for use in kernels.
        self.ee_body_indices = torch.zeros(num_envs, dtype=torch.int32) + ee_indices[0]
        # Global EE body indices (one per env, for Warp kernels used by APG).
        self._ee_global = np.asarray(ee_indices, dtype=np.int32)
        self._ee_global_wp = wp.array(
            self._ee_global, dtype=wp.int32, device=self.model.device
        )

        # Single shared viewer — set_world_offsets spreads robots in one scene.
        self.viewer = _build_viewer(self.model, headless=headless, num_envs=num_envs)
        self._sim_time = 0.0

        # joint_pos(7) + ee_pose(7) + target_pose(7) + last_action(7) = 28
        self.obs_dim = (
            FRANKA_NUM_ARM_JOINTS + 3 + 4 + 3 + 4 + FRANKA_NUM_ARM_JOINTS
        )  # 28
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FRANKA_NUM_ARM_JOINTS,),
            dtype=np.float32,
        )

        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.target_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.target_quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)

        self.arm_joint_limit_lower = torch.tensor(
            self.model.joint_limit_lower[:FRANKA_NUM_ARM_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )
        self.arm_joint_limit_upper = torch.tensor(
            self.model.joint_limit_upper[:FRANKA_NUM_ARM_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )

        # Tracks the last applied action for the action-rate reward term.
        self.last_action = torch.zeros(
            num_envs, FRANKA_NUM_ARM_JOINTS, dtype=torch.float32, device=device
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, env_ids=None, seed=None):
        if seed is not None:
            set_seed(seed)

        local_env_ids = (
            env_ids
            if env_ids is not None
            else torch.arange(self._num_envs, device=self.device)
        )

        self.step_count[local_env_ids] = 0

        default_q = torch.tensor(
            DEFAULT_JOINT_Q, dtype=torch.float32, device=self.device
        )
        joint_q_t = default_q.unsqueeze(0).expand(len(local_env_ids), -1).clone()
        joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] = self.arm_joint_limit_lower + torch.rand(
            len(local_env_ids), FRANKA_NUM_ARM_JOINTS, device=self.device
        ) * (self.arm_joint_limit_upper - self.arm_joint_limit_lower)

        joint_q = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
        with torch.no_grad():
            joint_q[local_env_ids] = joint_q_t

        self.last_action[local_env_ids] = 0.0

        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        goal_joint_q_t = default_q.unsqueeze(0).expand(len(local_env_ids), -1).clone()
        goal_joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] = (
            self.arm_joint_limit_lower
            + torch.rand(len(local_env_ids), FRANKA_NUM_ARM_JOINTS, device=self.device)
            * (self.arm_joint_limit_upper - self.arm_joint_limit_lower)
        )

        # IK targets are sampled from FK, so every target pose is feasible.
        with torch.no_grad():
            joint_q[local_env_ids] = goal_joint_q_t
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        body_q_t = wp.to_torch(self.state_0.body_q).view(self._num_envs, -1, 7)
        ee_indices = self.ee_body_indices.to(device=self.device, dtype=torch.long)
        goal_eef_pose = body_q_t[local_env_ids, ee_indices[local_env_ids]].detach()
        self.target_pos[local_env_ids] = goal_eef_pose[:, :3]
        self.target_quat[local_env_ids] = goal_eef_pose[:, 3:7]

        with torch.no_grad():
            joint_q[local_env_ids] = joint_q_t
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        self._render_current_state()
        return self._get_obs(), {}

    def step(self, actions):
        """Step all environments.

        Args:
            actions: Delta joint positions, shape ``[num_envs, 7]``, torch or numpy.

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, infos)`` as torch
            tensors.  Done environments are auto-reset; their pre-reset
            observation is stored in ``infos["final_observation"]``.
        """
        self.step_count += 1
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)

        # Compute new joint targets: target = current + action * scale, clamped.
        joint_q_t = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
        joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] += actions * self.action_scale
        joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
            joint_q_t[:, :FRANKA_NUM_ARM_JOINTS],
            self.arm_joint_limit_lower,
            self.arm_joint_limit_upper,
        )

        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        obs = self._get_obs()

        rewards = self._get_rewards(actions)

        self.last_action = actions.clone()
        self._sim_time += self.frame_dt
        self._render_current_state()

        truncated = self.step_count >= self.max_episode_steps
        terminated = self._check_success()
        done_mask = truncated | terminated

        # Compute EEF distance for infos before auto-reset
        body_q_t = wp.to_torch(self.state_0.body_q).view(self._num_envs, -1, 7)
        eef_pose = body_q_t[
            torch.arange(self._num_envs, device=self.device),
            self.ee_body_indices,
        ]
        final_pos_distance = (eef_pose[:, :3] - self.target_pos).norm(dim=-1).detach()
        final_rot_distance = _quat_distance(eef_pose[:, 3:7], self.target_quat).detach()
        infos = {
            "final_distance": final_pos_distance,
            "final_rot_distance": final_rot_distance,
            "success": terminated.detach(),
        }

        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs, _ = self.reset(reset_ids)

        return (
            obs,
            rewards,
            terminated,
            truncated,
            infos,
        )

    def _get_obs(self):
        joint_q_t = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
        body_q_t = wp.to_torch(self.state_0.body_q).view(
            self._num_envs, -1, 7
        )  # [num_envs, num_bodies, 7]
        env_idx = torch.arange(self._num_envs, device=self.device)
        eef_pose = body_q_t[env_idx, self.ee_body_indices, :]
        target_pose = torch.cat([self.target_pos, self.target_quat], dim=-1)
        if self.canonicalize_quat_obs:
            eef_pose = torch.cat(
                [eef_pose[:, :3], _canonicalize_quat(eef_pose[:, 3:7])], dim=-1
            )
            target_pose = torch.cat(
                [target_pose[:, :3], _canonicalize_quat(target_pose[:, 3:7])],
                dim=-1,
            )
        obs = torch.empty(
            self._num_envs, self.obs_dim, dtype=torch.float32, device=self.device
        )
        obs[:, :FRANKA_NUM_ARM_JOINTS] = joint_q_t[:, :FRANKA_NUM_ARM_JOINTS]
        obs[:, FRANKA_NUM_ARM_JOINTS : FRANKA_NUM_ARM_JOINTS + 7] = eef_pose
        obs[:, FRANKA_NUM_ARM_JOINTS + 7 : FRANKA_NUM_ARM_JOINTS + 14] = target_pose
        obs[:, FRANKA_NUM_ARM_JOINTS + 14 :] = self.last_action
        return obs

    def _get_rewards(self, action: torch.Tensor | None = None):
        body_q_t = wp.to_torch(self.state_0.body_q).view(self._num_envs, -1, 7)
        eef_pose = body_q_t[
            torch.arange(self._num_envs, device=self.device),
            self.ee_body_indices,
        ]
        return _compute_reward(
            eef_pose[:, :3],
            eef_pose[:, 3:7],
            self.target_pos,
            self.target_quat,
            action=action,
            last_action=self.last_action,
        )

    def _check_success(self):
        body_q_t = wp.to_torch(self.state_0.body_q).view(self._num_envs, -1, 7)
        eef_pose = body_q_t[
            torch.arange(self._num_envs, device=self.device),
            self.ee_body_indices,
        ]
        return _check_success(
            eef_pose[:, :3], eef_pose[:, 3:7], self.target_pos, self.target_quat
        )

    def _render_current_state(self) -> None:
        if self.viewer is None:
            return

        self.viewer.begin_frame(self._sim_time)
        self.viewer.log_state(self.state_0)
        self._draw_axes()
        self.viewer.end_frame()

    def _draw_axes(self) -> None:
        """Draw coordinate-frame axes for target pose and current EE pose.

        Target axes are drawn in solid RGB; EE axes use darker/distinct tones
        and are rendered thinner so the two frames are easy to tell apart.
        ``log_lines`` renders in absolute world coordinates (no automatic
        per-world offset), so we add the full visual offset manually.
        """
        if self.headless:
            return

        num_envs = self._num_envs
        L = TARGET_AXIS_LENGTH

        # The viewer applies world_offsets when rendering shapes via log_state,
        # but log_lines renders in absolute world coordinates without any offset.
        # body_q is stored in local (per-env) frame (spacing=0 in replicate),
        # so we must add the same viewer.world_offsets here to stay aligned.
        if self.viewer.world_offsets is not None:
            viewer_offsets = self.viewer.world_offsets.numpy()  # [num_worlds, 3]
        else:
            viewer_offsets = np.zeros((num_envs, 3), dtype=np.float32)

        # --- target axes (solid, bright) ---
        target_pos_np = self.target_pos.cpu().numpy()
        target_quat_np = self.target_quat.cpu().numpy()

        t_begins = np.empty((num_envs * 3, 3), dtype=np.float32)
        t_ends = np.empty((num_envs * 3, 3), dtype=np.float32)
        t_colors = np.empty((num_envs * 3, 3), dtype=np.float32)

        target_axis_colors = np.array(
            [[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]], dtype=np.float32
        )

        for i in range(num_envs):
            origin = target_pos_np[i] + viewer_offsets[i]
            R = _quat_to_rotmat(target_quat_np[i])
            base = i * 3
            for ax in range(3):
                t_begins[base + ax] = origin
                t_ends[base + ax] = origin + R[:, ax] * L
                t_colors[base + ax] = target_axis_colors[ax]

        self.viewer.log_lines(
            "/target_axes",
            wp.array(t_begins, dtype=wp.vec3, device="cpu"),
            wp.array(t_ends, dtype=wp.vec3, device="cpu"),
            wp.array(t_colors, dtype=wp.vec3, device="cpu"),
            width=0.02,
        )

        # --- EE axes (darker, thinner) ---
        body_q_t = wp.to_torch(self.state_0.body_q).view(num_envs, -1, 7).detach()
        ee_pos = (
            body_q_t[
                torch.arange(num_envs, device=self.device), self.ee_body_indices, :3
            ]
            .cpu()
            .numpy()
        )
        ee_quat = (
            body_q_t[
                torch.arange(num_envs, device=self.device), self.ee_body_indices, 3:7
            ]
            .cpu()
            .numpy()
        )

        e_begins = np.empty((num_envs * 3, 3), dtype=np.float32)
        e_ends = np.empty((num_envs * 3, 3), dtype=np.float32)
        e_colors = np.empty((num_envs * 3, 3), dtype=np.float32)

        ee_axis_colors = np.array(
            [[0.7, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.7]], dtype=np.float32
        )

        for i in range(num_envs):
            origin = ee_pos[i] + viewer_offsets[i]
            R = _quat_to_rotmat(ee_quat[i])
            base = i * 3
            for ax in range(3):
                e_begins[base + ax] = origin
                e_ends[base + ax] = origin + R[:, ax] * L
                e_colors[base + ax] = ee_axis_colors[ax]

        self.viewer.log_lines(
            "/ee_axes",
            wp.array(e_begins, dtype=wp.vec3, device="cpu"),
            wp.array(e_ends, dtype=wp.vec3, device="cpu"),
            wp.array(e_colors, dtype=wp.vec3, device="cpu"),
            width=0.01,
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ---------------------------------------------------------------------------
# APG Environment (batched, differentiable)
# ---------------------------------------------------------------------------

# Warp kernels for differentiable action-to-control and reward computation


@wp.kernel
def _set_joint_targets_kernel(
    action: wp.array(dtype=wp.float32),  # [num_envs * 7] flat action
    current_q: wp.array(dtype=wp.float32),  # joint_q from state
    target_pos: wp.array(dtype=wp.float32),  # new joint_q output
    joint_limit_lower: wp.array(dtype=wp.float32),  # [num_arm_joints]
    joint_limit_upper: wp.array(dtype=wp.float32),  # [num_arm_joints]
    action_scale: wp.float32,
    num_joints_per_env: wp.int32,
    num_arm_joints: wp.int32,
    total_dims: wp.int32,  # num_envs * num_arm_joints
):
    """Compute joint targets: target = clamp(current + action * scale, lo, hi)."""
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
    """Full reward matching _compute_reward (position + orientation + action_rate)."""
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
        + wp.float32(0.1) * wp.exp(-rot_dist * rot_dist / wp.float32(0.18))
        - wp.float32(0.0001) * action_rate
    )


class _NewtonStepFunc(torch.autograd.Function):
    """Bridges Warp tape autodiff to PyTorch autograd.

    For APG, we bypass the dynamics solver and compute reward directly from FK.
    This is because the Featherstone solver doesn't propagate gradients through
    the control input. The differentiable path is:
        action → joint_q → FK → body_q → EEF pose → reward

    Forward: runs FK inside Warp tape, returns reward + EEF poses (detached).
    Backward: uses stored Warp tape to compute d(reward)/d(action).

    Obs construction is intentionally left to the caller (FrankaReachAPGEnv.step)
    so that the joint-position component can be computed in PyTorch, giving a
    differentiable obs[:, :7] and enabling multi-step credit assignment through
    the jpos state.
    """

    @staticmethod
    def forward(ctx, action_torch, sim_state):
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

        # Convert action to Warp with gradient tracking
        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        # Last action as constant Warp array (no grad needed)
        last_action_flat = last_action_t.detach().clone().reshape(-1).contiguous()
        last_action_wp = wp.from_torch(last_action_flat, dtype=wp.float32)

        num_joints_per_env = FRANKA_NUM_JOINTS
        new_joint_q = wp.zeros(
            num_envs * num_joints_per_env,
            dtype=wp.float32,
            device=model.device,
            requires_grad=True,
        )

        # Target arrays
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

        # Create temporary state for FK
        fk_state = model.state()

        tape = wp.Tape()
        with tape:
            # Compute new joint_q = current + action * scale
            wp.launch(
                _set_joint_targets_kernel,
                dim=num_envs * FRANKA_NUM_ARM_JOINTS,
                inputs=[
                    action_wp,
                    state_joint_q,
                    new_joint_q,
                    joint_limit_lower_wp,
                    joint_limit_upper_wp,
                    wp.float32(action_scale),
                    wp.int32(num_joints_per_env),
                    wp.int32(FRANKA_NUM_ARM_JOINTS),
                    wp.int32(num_envs * FRANKA_NUM_ARM_JOINTS),
                ],
                device=model.device,
            )

            # Set zero velocities
            wp.copy(fk_state.joint_qd, model.joint_qd)

            # Compute FK to get EEF pose from new joint positions
            newton.eval_fk(model, new_joint_q, fk_state.joint_qd, fk_state)

            # Compute full reward (position + orientation + action_rate)
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
                    wp.int32(FRANKA_NUM_ARM_JOINTS),
                ],
                outputs=[reward_wp],
                device=model.device,
            )

        # Extract results
        reward_t = wp.to_torch(reward_wp).detach().clone()

        # EEF poses from FK (detached — gradient through eef requires a second
        # Warp tape pass and is left for future work; jpos gradient is handled
        # in PyTorch by the caller)
        body_q_torch = wp.to_torch(fk_state.body_q)  # [total_bodies, 7]
        ee_indices_t = wp.to_torch(ee_indices_wp).long()  # [num_envs]
        eef_poses = body_q_torch[ee_indices_t].detach()  # [num_envs, 7]

        # Save for backward
        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.reward_wp = reward_wp

        return reward_t, eef_poses

    @staticmethod
    def backward(ctx, grad_reward, grad_eef_poses):
        # grad_eef_poses is ignored (eef_poses was detached in forward)
        grad_reward_wp = wp.from_torch(
            grad_reward.detach().clone().contiguous(), dtype=wp.float32
        )
        wp.copy(ctx.reward_wp.grad, grad_reward_wp)

        # Backward through Warp tape
        ctx.tape.backward()

        # Extract action gradient and reshape
        action_grad = (
            wp.to_torch(ctx.action_wp.grad).clone().reshape(grad_reward.shape[0], -1)
        )
        ctx.tape.zero()

        return action_grad, None


class FrankaReachAPGEnv(FrankaReachVecEnv):
    """Batched Franka reach environment for APG with differentiable physics.

    Inherits all shared logic (reset, obs, reward, rendering, auto-reset)
    from FrankaReachVecEnv.  The only override is ``step()``, which uses the
    Warp tape bridge (_NewtonStepFunc) so that gradients flow through the
    forward-kinematics and reward computation.

    Interface matches the NMG APG loop:
      - reset() -> (obs [num_envs, 28], info)
      - step(action [num_envs, 7]) -> (obs, reward, terminated, truncated, info)
      - attributes: single_observation_space, single_action_space, num_envs
    """

    def __init__(
        self,
        num_envs: int = 4,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cpu",
        headless: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            device=device,
            headless=headless,
            requires_grad=True,
            **kwargs,
        )
        self._joint_limit_lower_wp = wp.array(
            self.arm_joint_limit_lower.cpu().numpy().astype(np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self._joint_limit_upper_wp = wp.array(
            self.arm_joint_limit_upper.cpu().numpy().astype(np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )

    def step(self, action):
        """Step all envs with differentiable reward. action: [num_envs, 7]."""
        self.step_count += 1
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = torch.clamp(action.to(self.device), -1.0, 1.0)

        sim_state = {
            "model": self.model,
            "state_joint_q": self.state_0.joint_q,
            "ee_body_indices_wp": self._ee_global_wp,
            "target_pos": self.target_pos,
            "target_quat": self.target_quat,
            "num_envs": self._num_envs,
            "last_action": self.last_action,
            "action_scale": self.action_scale,
            "joint_limit_lower_wp": self._joint_limit_lower_wp,
            "joint_limit_upper_wp": self._joint_limit_upper_wp,
        }

        reward, eef_poses = _NewtonStepFunc.apply(action, sim_state)

        # Differentiable joint positions in PyTorch — same values as Warp's new_joint_q
        # but connected to `action` via PyTorch autograd.  This enables multi-step
        # credit assignment through the jpos component of obs.
        # Read current joint positions (before the physics update below).
        current_q = (
            wp.to_torch(self.state_0.joint_q)
            .view(self._num_envs, FRANKA_NUM_JOINTS)[:, :FRANKA_NUM_ARM_JOINTS]
            .detach()
        )
        new_jpos = (current_q + action * self.action_scale).clamp(
            self.arm_joint_limit_lower, self.arm_joint_limit_upper
        )

        # Update env state for next step (detached from computation graph)
        with torch.no_grad():
            joint_q_t = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
            joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] += action.detach() * self.action_scale
            joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
                joint_q_t[:, :FRANKA_NUM_ARM_JOINTS],
                self.arm_joint_limit_lower,
                self.arm_joint_limit_upper,
            )

        self.last_action = action.detach().clone()
        self._sim_time += self.frame_dt
        self._render_current_state()

        # Build obs: jpos is differentiable through action; eef/target/last_action detached.
        if self.canonicalize_quat_obs:
            obs_eef = torch.cat(
                [eef_poses[:, :3], _canonicalize_quat(eef_poses[:, 3:7])], dim=-1
            )
            obs_target_quat = _canonicalize_quat(self.target_quat)
        else:
            obs_eef = eef_poses
            obs_target_quat = self.target_quat
        obs = torch.cat(
            [
                new_jpos,
                obs_eef,
                self.target_pos,
                obs_target_quat,
                self.last_action,
            ],
            dim=-1,
        )

        truncated = self.step_count >= self.max_episode_steps
        terminated = _check_success(
            eef_poses[:, :3],
            eef_poses[:, 3:7],
            self.target_pos,
            self.target_quat,
        )
        done_mask = truncated | terminated

        final_pos_distance = (eef_poses[:, :3] - self.target_pos).norm(dim=-1).detach()
        final_rot_distance = _quat_distance(
            eef_poses[:, 3:7], self.target_quat
        ).detach()
        infos = {
            "final_distance": final_pos_distance,
            "final_rot_distance": final_rot_distance,
            "success": terminated.detach(),
        }

        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self.reset(reset_ids)
            # Replace done envs with fresh detached obs; preserve gradient for live envs.
            fresh_obs = self._get_obs()
            obs = torch.where(done_mask.unsqueeze(-1).expand_as(obs), fresh_obs, obs)

        return obs, reward, terminated, truncated, infos
