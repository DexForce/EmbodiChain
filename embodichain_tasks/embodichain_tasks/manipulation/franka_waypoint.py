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
"""Franka ordered pose-waypoint reaching task.

The end-effector must visit a fixed number of 6D pose waypoints in order.
Actions remain delta joint positions.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

import newton

from embodichain.learning.rl import (
    DifferentiableRolloutSpec,
    register_learning_env,
    stratified_rollout_value,
)
from embodichain.utils import set_seed

from ._franka_reach import (
    DEFAULT_ACTION_SCALE,
    DEFAULT_MAX_EPISODE_STEPS,
    FRANKA_NUM_ARM_JOINTS,
    FRANKA_NUM_JOINTS,
    FrankaReachVecEnv,
    _canonicalize_quat,
    _quat_angle_distance,
    _quat_inverse,
    _quat_mul,
    _set_joint_targets_kernel,
)
from ._franka_se3_sampling import (
    NUM_SE3_PRIMITIVES,
    FrankaStratifiedSE3Sampler,
)
from ._waypoint_types import (
    NUM_WP_TYPES,
    WP_TYPE_CARTESIAN,
    WP_TYPE_JOINT,
    WP_TYPE_POSITION_ONLY,
    WAYPOINT_TASK_FIELDS,
    WAYPOINT_TASK_OPTIONAL_FIELDS,
)
from ._waypoint_sampling import (
    ACTIVE_GROUP_DENSE,
    ACTIVE_GROUP_SINGLE,
    ACTIVE_GROUP_SPARSE,
    NUM_ACTIVE_GROUPS,
    NUM_DIRECTION_RELATIONS,
    MultiscaleWaypointSampler,
    WaypointJointSamples,
)

__all__ = ["FrankaWaypointNMGEnv"]

WAYPOINT_JOINT_EXP_SCALE = 0.7
WAYPOINT_JOINT_DENSE_PEAK = 0.2
WAYPOINT_JOINT_PRECISION_EXP_SCALE = 0.01
WAYPOINT_JOINT_PRECISION_DENSE_PEAK = 0.1
WAYPOINT_POSE_DENSE_PEAK = 0.1
WAYPOINT_JOINT_WEIGHT = 0.2 * 0.01 / 0.02


def waypoint_obs_dim(num_waypoints: int, use_relative_obs: bool) -> int:
    """Flat obs size for the unified waypoint layout (all sampling modes)."""
    n = int(num_waypoints)
    dim = (
        FRANKA_NUM_ARM_JOINTS
        + 7
        + n * 3
        + n * 4
        + n * FRANKA_NUM_ARM_JOINTS
        + n  # active one-hot
        + n  # valid
        + n  # pos_mask
        + n  # rot_mask
        + n  # joint_mask
        + FRANKA_NUM_ARM_JOINTS
        + n  # wp_type
    )
    if use_relative_obs:
        dim += 7 + n * (3 + 4 + FRANKA_NUM_ARM_JOINTS)
    return dim


def waypoint_obs_normalize_mask(
    num_waypoints: int, use_relative_obs: bool, device=None
) -> torch.Tensor:
    """True = normalize continuous dims; False = keep semantic masks/indices raw."""
    n = int(num_waypoints)
    dim = waypoint_obs_dim(n, use_relative_obs)
    mask = torch.ones(dim, dtype=torch.bool, device=device)
    cursor = FRANKA_NUM_ARM_JOINTS + 7 + n * (3 + 4 + FRANKA_NUM_ARM_JOINTS)
    mask[cursor : cursor + 5 * n] = False
    cursor += 5 * n
    cursor += FRANKA_NUM_ARM_JOINTS
    if use_relative_obs:
        cursor += 7 + n * (3 + 4)
        mask[cursor : cursor + n * FRANKA_NUM_ARM_JOINTS] = False
        cursor += n * FRANKA_NUM_ARM_JOINTS
    mask[cursor : cursor + n] = False
    return mask


@wp.kernel
def _compute_waypoint_reward_kernel(
    body_q: wp.array(dtype=wp.transformf),
    ee_body_indices: wp.array(dtype=wp.int32),
    waypoint_pos: wp.array(dtype=wp.vec3f),
    waypoint_quat: wp.array(dtype=wp.quatf),
    pos_weight: wp.float32,
    rot_weight: wp.array(dtype=wp.float32),
    rot_precision_weight: wp.array(dtype=wp.float32),
    rot_required: wp.array(dtype=wp.float32),
    pos_threshold: wp.float32,
    rot_threshold: wp.float32,
    pose_constraint_weight: wp.float32,
    pose_constraint_aggregation: wp.int32,
    pose_feasibility_weight: wp.float32,
    pose_feasibility_beta: wp.float32,
    pose_violation_weight: wp.float32,
    pose_violation_beta: wp.float32,
    reward_out: wp.array(dtype=wp.float32),
):
    """Pose reward for the currently active waypoint."""
    env_idx = wp.tid()
    ee_global = ee_body_indices[env_idx]
    ee_transform = body_q[ee_global]
    eef_pos = wp.transform_get_translation(ee_transform)
    eef_quat = wp.transform_get_rotation(ee_transform)

    diff = eef_pos - waypoint_pos[env_idx]
    pos_dist = wp.sqrt(wp.dot(diff, diff) + wp.float32(1e-8))

    target_quat = waypoint_quat[env_idx]
    dq_x = eef_quat.x - target_quat.x
    dq_y = eef_quat.y - target_quat.y
    dq_z = eef_quat.z - target_quat.z
    dq_w = eef_quat.w - target_quat.w
    d1 = dq_x * dq_x + dq_y * dq_y + dq_z * dq_z + dq_w * dq_w
    sq_x = eef_quat.x + target_quat.x
    sq_y = eef_quat.y + target_quat.y
    sq_z = eef_quat.z + target_quat.z
    sq_w = eef_quat.w + target_quat.w
    d2 = sq_x * sq_x + sq_y * sq_y + sq_z * sq_z + sq_w * sq_w
    rot_dist = wp.min(d1, d2)
    # ``rot_dist`` is squared chordal distance and therefore has a vanishing
    # angular gradient near the goal.  The optional chordal-norm term is
    # approximately linear in the geodesic angle (2*sqrt(rot_dist) ~= theta),
    # providing stationary precision gradients without an eval-driven
    # curriculum.  Subtracting sqrt(eps) keeps the reward value unchanged at
    # exact alignment while avoiding a singular derivative.
    rot_precision_dist = wp.float32(2.0) * (
        wp.sqrt(rot_dist + wp.float32(1e-8)) - wp.float32(1e-4)
    )
    normalized_pos_error = pos_dist / pos_threshold
    normalized_constraint_error = normalized_pos_error
    normalized_rot_error = rot_precision_dist / rot_threshold
    if rot_required[env_idx] > wp.float32(0.5):
        if pose_constraint_aggregation == wp.int32(1):
            normalized_constraint_error = (
                normalized_constraint_error + normalized_rot_error
            )
        elif pose_constraint_aggregation == wp.int32(2):
            normalized_constraint_error = wp.sqrt(
                normalized_constraint_error * normalized_constraint_error
                + normalized_rot_error * normalized_rot_error
                + wp.float32(1.0e-8)
            )
        elif pose_constraint_aggregation == wp.int32(3):
            constraint_max = wp.max(normalized_constraint_error, normalized_rot_error)
            normalized_constraint_error = constraint_max + wp.log(
                wp.exp(normalized_constraint_error - constraint_max)
                + wp.exp(normalized_rot_error - constraint_max)
            )
        else:
            normalized_constraint_error = wp.max(
                normalized_constraint_error,
                normalized_rot_error,
            )

    pos_feasibility_logit = pose_feasibility_beta * (
        wp.float32(1.0) - pos_dist / pos_threshold
    )
    if pos_feasibility_logit >= wp.float32(0.0):
        pos_feasibility = wp.float32(1.0) / (
            wp.float32(1.0) + wp.exp(-pos_feasibility_logit)
        )
    else:
        pos_feasibility_exp = wp.exp(pos_feasibility_logit)
        pos_feasibility = pos_feasibility_exp / (wp.float32(1.0) + pos_feasibility_exp)
    max_single_feasibility = wp.float32(1.0) / (
        wp.float32(1.0) + wp.exp(-pose_feasibility_beta)
    )
    pose_feasibility = pos_feasibility
    max_pose_feasibility = max_single_feasibility
    if rot_required[env_idx] > wp.float32(0.5):
        rot_feasibility_logit = pose_feasibility_beta * (
            wp.float32(1.0) - normalized_rot_error
        )
        if rot_feasibility_logit >= wp.float32(0.0):
            rot_feasibility = wp.float32(1.0) / (
                wp.float32(1.0) + wp.exp(-rot_feasibility_logit)
            )
        else:
            rot_feasibility_exp = wp.exp(rot_feasibility_logit)
            rot_feasibility = rot_feasibility_exp / (
                wp.float32(1.0) + rot_feasibility_exp
            )
        pose_feasibility = pose_feasibility * rot_feasibility
        max_pose_feasibility = max_single_feasibility * max_single_feasibility

    # Independent smooth hinges retain gradient for one violated constraint
    # even when the other pose constraint is much farther from its threshold.
    pos_violation_logit = pose_violation_beta * (normalized_pos_error - wp.float32(1.0))
    if pos_violation_logit > wp.float32(0.0):
        pos_violation = pos_violation_logit + wp.log(
            wp.float32(1.0) + wp.exp(-pos_violation_logit)
        )
    else:
        pos_violation = wp.log(wp.float32(1.0) + wp.exp(pos_violation_logit))
    zero_violation = wp.log(wp.float32(1.0) + wp.exp(-pose_violation_beta))
    pose_violation = pos_violation - zero_violation
    if rot_required[env_idx] > wp.float32(0.5):
        rot_violation_logit = pose_violation_beta * (
            normalized_rot_error - wp.float32(1.0)
        )
        if rot_violation_logit > wp.float32(0.0):
            rot_violation = rot_violation_logit + wp.log(
                wp.float32(1.0) + wp.exp(-rot_violation_logit)
            )
        else:
            rot_violation = wp.log(wp.float32(1.0) + wp.exp(rot_violation_logit))
        pose_violation = pose_violation + rot_violation - zero_violation
    pose_violation = pose_violation / pose_violation_beta

    reward_out[env_idx] = (
        -pos_weight * pos_dist
        # Zero-baseline dense peaks: exact satisfaction yields zero step reward.
        # A positive goal baseline would reward delaying waypoint advancement in
        # finite, independently masked rollouts.
        + wp.float32(0.1)
        * (wp.exp(-pos_dist * pos_dist / wp.float32(0.02)) - wp.float32(1.0))
        - rot_weight[env_idx] * rot_dist
        - rot_precision_weight[env_idx] * rot_precision_dist
        - pose_constraint_weight * normalized_constraint_error
        + pose_feasibility_weight * (pose_feasibility - max_pose_feasibility)
        - pose_violation_weight * pose_violation
        + rot_weight[env_idx]
        * (wp.exp(-rot_dist * rot_dist / wp.float32(0.18)) - wp.float32(1.0))
    )


class _NewtonWaypointStepFunc(torch.autograd.Function):
    """Differentiable APG step for active pose-waypoint reward.

    Joint and mixed joint rewards are computed in PyTorch so the action
    gradient stays exact; this autograd Function only handles FK + pose reward.
    """

    @staticmethod
    def forward(ctx, current_arm_q_torch, action_torch, sim_state):
        model = sim_state["model"]
        state_joint_q = sim_state["state_joint_q"]
        num_envs = sim_state["num_envs"]
        action_scale = sim_state["action_scale"]
        joint_limit_lower_wp = sim_state["joint_limit_lower_wp"]
        joint_limit_upper_wp = sim_state["joint_limit_upper_wp"]

        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        current_full_q = (
            wp.to_torch(state_joint_q)
            .view(num_envs, FRANKA_NUM_JOINTS)
            .detach()
            .clone()
        )
        current_full_q[:, :FRANKA_NUM_ARM_JOINTS] = current_arm_q_torch.detach()
        current_q_wp = wp.from_torch(
            current_full_q.reshape(-1).contiguous(),
            dtype=wp.float32,
            requires_grad=True,
        )

        num_joints_per_env = FRANKA_NUM_JOINTS
        new_joint_q = wp.zeros(
            num_envs * num_joints_per_env,
            dtype=wp.float32,
            device=model.device,
            requires_grad=True,
        )
        reward_wp = wp.zeros(
            num_envs, dtype=wp.float32, device=model.device, requires_grad=True
        )

        ee_indices_wp = sim_state["ee_body_indices_wp"]
        waypoint_pos_wp = wp.array(
            sim_state["active_waypoint_pos"].detach().cpu().tolist(),
            dtype=wp.vec3f,
            device=model.device,
        )
        waypoint_quat_wp = wp.array(
            [
                wp.quatf(q[0], q[1], q[2], q[3])
                for q in sim_state["active_waypoint_quat"].detach().cpu().tolist()
            ],
            dtype=wp.quatf,
            device=model.device,
        )
        rot_weight_wp = wp.array(
            sim_state["active_rot_weight"].detach().cpu().tolist(),
            dtype=wp.float32,
            device=model.device,
        )
        pos_weight = sim_state["waypoint_pos_weight"]
        active_rot_precision_weight = sim_state.get("active_rot_precision_weight")
        if active_rot_precision_weight is None:
            active_rot_precision_weight = torch.zeros_like(
                sim_state["active_rot_weight"]
            )
        rot_precision_weight_wp = wp.array(
            active_rot_precision_weight.detach().cpu().tolist(),
            dtype=wp.float32,
            device=model.device,
        )
        active_rot_required = sim_state.get("active_rot_required")
        if active_rot_required is None:
            active_rot_required = sim_state["active_rot_weight"] > 0.0
        rot_required_wp = wp.array(
            active_rot_required.detach().to(torch.float32).cpu().tolist(),
            dtype=wp.float32,
            device=model.device,
        )
        pos_threshold = sim_state.get("waypoint_pos_threshold", 0.01)
        rot_threshold = sim_state.get("waypoint_rot_threshold", 0.1)
        pose_constraint_weight = sim_state.get("waypoint_pose_constraint_weight", 0.0)
        pose_constraint_aggregation_name = sim_state.get(
            "waypoint_pose_constraint_aggregation", "max"
        )
        pose_constraint_aggregation = {
            "max": 0,
            "sum": 1,
            "l2": 2,
            "smoothmax": 3,
        }[pose_constraint_aggregation_name]
        pose_feasibility_weight = sim_state.get("waypoint_pose_feasibility_weight", 0.0)
        pose_feasibility_beta = sim_state.get("waypoint_pose_feasibility_beta", 4.0)
        pose_violation_weight = sim_state.get("waypoint_pose_violation_weight", 0.0)
        pose_violation_beta = sim_state.get("waypoint_pose_violation_beta", 4.0)

        tape = wp.Tape()
        with tape:
            wp.launch(
                _set_joint_targets_kernel,
                dim=num_envs * FRANKA_NUM_ARM_JOINTS,
                inputs=[
                    action_wp,
                    current_q_wp,
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
            fk_state = model.state()
            wp.copy(fk_state.joint_qd, model.joint_qd)
            newton.eval_fk(model, new_joint_q, fk_state.joint_qd, fk_state)
            wp.launch(
                _compute_waypoint_reward_kernel,
                dim=num_envs,
                inputs=[
                    fk_state.body_q,
                    ee_indices_wp,
                    waypoint_pos_wp,
                    waypoint_quat_wp,
                    wp.float32(pos_weight),
                    rot_weight_wp,
                    rot_precision_weight_wp,
                    rot_required_wp,
                    wp.float32(pos_threshold),
                    wp.float32(rot_threshold),
                    wp.float32(pose_constraint_weight),
                    wp.int32(pose_constraint_aggregation),
                    wp.float32(pose_feasibility_weight),
                    wp.float32(pose_feasibility_beta),
                    wp.float32(pose_violation_weight),
                    wp.float32(pose_violation_beta),
                ],
                outputs=[reward_wp],
                device=model.device,
            )

        reward_t = wp.to_torch(reward_wp).detach().clone()
        body_q_torch = wp.to_torch(fk_state.body_q)
        ee_indices_t = wp.to_torch(ee_indices_wp).long()
        eef_pose = body_q_torch[ee_indices_t].detach()

        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.current_q_wp = current_q_wp
        ctx.reward_wp = reward_wp
        ctx.body_q_wp = fk_state.body_q
        ctx.ee_indices_torch = ee_indices_t
        ctx.num_envs = num_envs
        return reward_t, eef_pose

    @staticmethod
    def backward(ctx, grad_reward, grad_eef_pose):
        grad_reward_wp = wp.from_torch(
            grad_reward.detach().clone().contiguous(), dtype=wp.float32
        )
        wp.copy(ctx.reward_wp.grad, grad_reward_wp)
        if grad_eef_pose is not None:
            body_grad = torch.zeros_like(wp.to_torch(ctx.body_q_wp.grad))
            body_grad[ctx.ee_indices_torch] = grad_eef_pose.detach()
            body_grad_wp = wp.from_torch(body_grad.contiguous(), dtype=wp.transformf)
            wp.copy(ctx.body_q_wp.grad, body_grad_wp)
        ctx.tape.backward()
        current_q_grad = (
            wp.to_torch(ctx.current_q_wp.grad)
            .clone()
            .reshape(ctx.num_envs, FRANKA_NUM_JOINTS)[:, :FRANKA_NUM_ARM_JOINTS]
        )
        action_grad = (
            wp.to_torch(ctx.action_wp.grad).clone().reshape(grad_reward.shape[0], -1)
        )
        ctx.tape.zero()
        return current_q_grad, action_grad, None


class FrankaWaypointReachVecEnv(FrankaReachVecEnv):
    """Batched ordered pose-waypoint reaching environment for vectorized PPO."""

    def __init__(
        self,
        num_envs: int = 4,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        num_waypoints: int = 3,
        waypoint_min_num_waypoints: int = 0,
        waypoint_fixed_num_waypoints: int = 0,
        waypoint_pos_threshold: float = 0.01,
        waypoint_pos_weight: float = 0.2,
        waypoint_rot_threshold: float = 0.1,
        waypoint_rot_weight: float = 0.1,
        waypoint_rot_precision_weight: float = 0.0,
        waypoint_pose_constraint_weight: float = 0.0,
        waypoint_pose_constraint_aggregation: str = "max",
        waypoint_pose_feasibility_weight: float = 0.0,
        waypoint_pose_feasibility_beta: float = 4.0,
        waypoint_pose_violation_weight: float = 0.0,
        waypoint_pose_violation_beta: float = 4.0,
        waypoint_space: str = "cartesian",
        waypoint_joint_weight: float = WAYPOINT_JOINT_WEIGHT,
        waypoint_joint_threshold: float = 0.02,
        waypoint_joint_exp_scale: float = WAYPOINT_JOINT_EXP_SCALE,
        waypoint_joint_dense_peak: float = WAYPOINT_JOINT_DENSE_PEAK,
        waypoint_joint_precision_exp_scale: float = WAYPOINT_JOINT_PRECISION_EXP_SCALE,
        waypoint_joint_precision_dense_peak: float = WAYPOINT_JOINT_PRECISION_DENSE_PEAK,
        waypoint_joint_fraction: float = 0.5,
        waypoint_distance_bucket_lowers: tuple[float, ...] = (
            0.25,
            1.0,
            2.0,
            4.0,
            8.0,
        ),
        waypoint_joint_limit_margin: float = 0.05,
        waypoint_sampling_max_retries: int = 64,
        waypoint_se3_translation_range: tuple[float, float] = (0.03, 0.20),
        waypoint_se3_rotation_range: tuple[float, float] = (0.15, 1.50),
        waypoint_se3_ik_iterations: int = 24,
        waypoint_se3_ik_max_retries: int = 10,
        waypoint_intermediate_orientation: bool = True,
        waypoint_bonus: float = 1.0,
        waypoint_use_relative_obs: bool = True,
        waypoint_steps_per_waypoint: int = 30,
        device: str = "cpu",
        headless: bool = True,
        requires_grad: bool = False,
        canonicalize_quat_obs: bool = False,
        **kwargs,
    ):
        waypoint_steps_per_waypoint = int(waypoint_steps_per_waypoint)
        if waypoint_steps_per_waypoint < 1:
            raise ValueError("waypoint_steps_per_waypoint must be positive")
        super().__init__(
            num_envs=num_envs,
            action_scale=action_scale,
            max_episode_steps=max_episode_steps,
            device=device,
            headless=headless,
            requires_grad=requires_grad,
            canonicalize_quat_obs=canonicalize_quat_obs,
            **kwargs,
        )
        self.num_waypoints = int(num_waypoints)
        if self.num_waypoints < 1:
            raise ValueError("num_waypoints must be >= 1")
        self.waypoint_min_num_waypoints = int(waypoint_min_num_waypoints)
        if self.waypoint_min_num_waypoints <= 0:
            self.waypoint_min_num_waypoints = self.num_waypoints
        if not 1 <= self.waypoint_min_num_waypoints <= self.num_waypoints:
            raise ValueError(
                "waypoint_min_num_waypoints must be in [1, num_waypoints] "
                f"or <=0 for fixed length, got {waypoint_min_num_waypoints}"
            )
        self.waypoint_fixed_num_waypoints = int(waypoint_fixed_num_waypoints)
        if self.waypoint_fixed_num_waypoints < 0:
            raise ValueError("waypoint_fixed_num_waypoints must be >= 0")
        if self.waypoint_fixed_num_waypoints > self.num_waypoints:
            raise ValueError("waypoint_fixed_num_waypoints must be <= num_waypoints")
        self.waypoint_pos_threshold = float(waypoint_pos_threshold)
        if self.waypoint_pos_threshold <= 0.0:
            raise ValueError("waypoint_pos_threshold must be > 0")
        self.waypoint_pos_weight = float(waypoint_pos_weight)
        self.waypoint_rot_threshold = float(waypoint_rot_threshold)
        if self.waypoint_rot_threshold <= 0.0:
            raise ValueError("waypoint_rot_threshold must be > 0")
        self.waypoint_rot_weight = float(waypoint_rot_weight)
        self.waypoint_rot_precision_weight = float(waypoint_rot_precision_weight)
        if self.waypoint_rot_precision_weight < 0.0:
            raise ValueError("waypoint_rot_precision_weight must be >= 0")
        self.waypoint_pose_constraint_weight = float(waypoint_pose_constraint_weight)
        if self.waypoint_pose_constraint_weight < 0.0:
            raise ValueError("waypoint_pose_constraint_weight must be >= 0")
        self.waypoint_pose_constraint_aggregation = str(
            waypoint_pose_constraint_aggregation
        )
        if self.waypoint_pose_constraint_aggregation not in (
            "max",
            "smoothmax",
            "l2",
            "sum",
        ):
            raise ValueError(
                "waypoint_pose_constraint_aggregation must be 'max', "
                "'smoothmax', 'l2', or 'sum', got "
                f"{waypoint_pose_constraint_aggregation!r}"
            )
        self.waypoint_pose_feasibility_weight = float(waypoint_pose_feasibility_weight)
        if self.waypoint_pose_feasibility_weight < 0.0:
            raise ValueError("waypoint_pose_feasibility_weight must be >= 0")
        self.waypoint_pose_feasibility_beta = float(waypoint_pose_feasibility_beta)
        if self.waypoint_pose_feasibility_beta <= 0.0:
            raise ValueError("waypoint_pose_feasibility_beta must be > 0")
        self.waypoint_pose_violation_weight = float(waypoint_pose_violation_weight)
        if self.waypoint_pose_violation_weight < 0.0:
            raise ValueError("waypoint_pose_violation_weight must be >= 0")
        self.waypoint_pose_violation_beta = float(waypoint_pose_violation_beta)
        if self.waypoint_pose_violation_beta <= 0.0:
            raise ValueError("waypoint_pose_violation_beta must be > 0")
        self.waypoint_space = str(waypoint_space)
        if self.waypoint_space not in ("cartesian", "joint", "mixed"):
            raise ValueError(
                f"waypoint_space must be 'cartesian', 'joint', or 'mixed', "
                f"got {waypoint_space!r}"
            )
        self.waypoint_joint_weight = float(waypoint_joint_weight)
        self.waypoint_joint_threshold = float(waypoint_joint_threshold)
        self.waypoint_joint_exp_scale = float(waypoint_joint_exp_scale)
        self.waypoint_joint_dense_peak = float(waypoint_joint_dense_peak)
        self.waypoint_joint_precision_exp_scale = float(
            waypoint_joint_precision_exp_scale
        )
        self.waypoint_joint_precision_dense_peak = float(
            waypoint_joint_precision_dense_peak
        )
        self.waypoint_joint_fraction = float(waypoint_joint_fraction)
        if not 0.0 <= self.waypoint_joint_fraction <= 1.0:
            raise ValueError(
                "waypoint_joint_fraction must be in [0, 1], "
                f"got {waypoint_joint_fraction}"
            )
        self.waypoint_distance_bucket_lowers = tuple(
            float(value) for value in waypoint_distance_bucket_lowers
        )
        self.waypoint_joint_limit_margin = float(waypoint_joint_limit_margin)
        self.waypoint_sampling_max_retries = int(waypoint_sampling_max_retries)
        self.waypoint_se3_translation_range = tuple(
            float(value) for value in waypoint_se3_translation_range
        )
        self.waypoint_se3_rotation_range = tuple(
            float(value) for value in waypoint_se3_rotation_range
        )
        self.waypoint_se3_ik_iterations = int(waypoint_se3_ik_iterations)
        self.waypoint_se3_ik_max_retries = int(waypoint_se3_ik_max_retries)
        self.waypoint_intermediate_orientation = bool(waypoint_intermediate_orientation)
        self.waypoint_bonus = float(waypoint_bonus)
        self.waypoint_use_relative_obs = bool(waypoint_use_relative_obs)
        self.waypoint_steps_per_waypoint = waypoint_steps_per_waypoint
        self.waypoint_sampler = MultiscaleWaypointSampler(
            self.arm_joint_limit_lower,
            self.arm_joint_limit_upper,
            self.action_scale,
            distance_bucket_lowers=self.waypoint_distance_bucket_lowers,
            max_h=float(self.waypoint_steps_per_waypoint),
            joint_limit_margin=self.waypoint_joint_limit_margin,
            max_retries=self.waypoint_sampling_max_retries,
            sobol_seed=int(torch.initial_seed()),
        )
        self.se3_waypoint_sampler = None
        # The IK solver is comparatively large.  Construct it on the first
        # sampled training reset, not in __init__, so a fixed-task evaluation
        # environment never allocates sampler-only GPU buffers.

        self.obs_dim = waypoint_obs_dim(
            self.num_waypoints, self.waypoint_use_relative_obs
        )
        self.single_observation_space = self.single_observation_space.__class__(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.obs_normalize_mask = waypoint_obs_normalize_mask(
            self.num_waypoints, self.waypoint_use_relative_obs, device=device
        )

        self.waypoints = torch.zeros(
            num_envs, self.num_waypoints, 3, dtype=torch.float32, device=device
        )
        self.waypoint_quats = torch.zeros(
            num_envs, self.num_waypoints, 4, dtype=torch.float32, device=device
        )
        self.waypoint_joint_qs = torch.zeros(
            num_envs,
            self.num_waypoints,
            FRANKA_NUM_ARM_JOINTS,
            dtype=torch.float32,
            device=device,
        )
        self.waypoint_type = torch.zeros(
            num_envs, self.num_waypoints, dtype=torch.long, device=device
        )
        self.waypoint_pos_mask = torch.ones(
            num_envs, self.num_waypoints, dtype=torch.float32, device=device
        )
        self.waypoint_rot_mask = torch.ones(
            num_envs, self.num_waypoints, dtype=torch.float32, device=device
        )
        self.waypoint_joint_mask = torch.zeros(
            num_envs, self.num_waypoints, dtype=torch.float32, device=device
        )
        self.waypoint_motion_scale_h = torch.zeros(
            num_envs, self.num_waypoints, dtype=torch.float32, device=device
        )
        self.waypoint_distance_bucket = torch.full(
            (num_envs, self.num_waypoints), -1, dtype=torch.long, device=device
        )
        self.waypoint_active_joint_count = torch.zeros(
            num_envs, self.num_waypoints, dtype=torch.long, device=device
        )
        self.waypoint_direction_relation = torch.full(
            (num_envs, self.num_waypoints), -1, dtype=torch.long, device=device
        )
        self.waypoint_se3_primitive = torch.full(
            (num_envs, self.num_waypoints), -1, dtype=torch.long, device=device
        )
        self.active_waypoint_idx = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.episode_num_waypoints = torch.full(
            (num_envs,), self.num_waypoints, dtype=torch.long, device=device
        )
        self.waypoint_valid_mask = torch.ones(
            num_envs, self.num_waypoints, dtype=torch.float32, device=device
        )
        self.prev_eef_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.episode_path_length = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.episode_min_waypoint_pos_distance = torch.full(
            (num_envs, self.num_waypoints),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self.episode_min_waypoint_rot_distance = torch.full(
            (num_envs, self.num_waypoints),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self.episode_min_waypoint_joint_distance = torch.full(
            (num_envs, self.num_waypoints),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self.episode_min_active_constraint_error = torch.full(
            (num_envs, self.num_waypoints),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self.episode_reached_by_type = torch.zeros(
            num_envs, NUM_WP_TYPES, dtype=torch.float32, device=device
        )
        self.episode_waypoint_reached_step = torch.full(
            (num_envs, self.num_waypoints), -1, dtype=torch.long, device=device
        )
        self.episode_initial_joint_q = torch.zeros(
            num_envs, FRANKA_NUM_JOINTS, dtype=torch.float32, device=device
        )
        self.current_fixed_eval_task_id = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        self._fixed_eval_tasks = None
        self._fixed_eval_indices = None
        self._fixed_eval_cursor = 0

    def _ensure_se3_waypoint_sampler(self) -> None:
        """Lazily allocate IK sampling state only for generated tasks."""
        if self.se3_waypoint_sampler is not None:
            return
        self.se3_waypoint_sampler = FrankaStratifiedSE3Sampler(
            self.waypoint_sampler,
            generation_batch_size=self._num_envs,
            num_waypoints=self.num_waypoints,
            translation_range=self.waypoint_se3_translation_range,
            rotation_range=self.waypoint_se3_rotation_range,
            ik_iterations=self.waypoint_se3_ik_iterations,
            max_retries=self.waypoint_se3_ik_max_retries,
        )

    def reset(self, env_ids=None, seed=None):
        if seed is not None:
            set_seed(seed)
            self.waypoint_sampler.reset_sobol(seed)
            if self.se3_waypoint_sampler is not None:
                self.se3_waypoint_sampler.clear_pool()

        local_env_ids = (
            env_ids
            if env_ids is not None
            else torch.arange(self._num_envs, device=self.device)
        )
        local_env_ids = torch.as_tensor(
            local_env_ids, dtype=torch.long, device=self.device
        )
        if local_env_ids.ndim == 0:
            local_env_ids = local_env_ids.unsqueeze(0)

        self.step_count[local_env_ids] = 0
        self.active_waypoint_idx[local_env_ids] = 0
        self.last_action[local_env_ids] = 0.0
        self.episode_path_length[local_env_ids] = 0.0
        self.episode_min_waypoint_pos_distance[local_env_ids] = float("inf")
        self.episode_min_waypoint_rot_distance[local_env_ids] = float("inf")
        self.episode_min_waypoint_joint_distance[local_env_ids] = float("inf")
        self.episode_min_active_constraint_error[local_env_ids] = float("inf")
        self.episode_reached_by_type[local_env_ids] = 0.0
        self.episode_waypoint_reached_step[local_env_ids] = -1
        self.waypoint_motion_scale_h[local_env_ids] = 0.0
        self.waypoint_distance_bucket[local_env_ids] = -1
        self.waypoint_active_joint_count[local_env_ids] = 0
        self.waypoint_direction_relation[local_env_ids] = -1
        self.waypoint_se3_primitive[local_env_ids] = -1
        self.current_fixed_eval_task_id[local_env_ids] = -1

        if self._fixed_eval_tasks is None:
            self._ensure_se3_waypoint_sampler()
            self._sample_episode_num_waypoints(local_env_ids)
            self._sample_waypoint_modality(local_env_ids)
            sampled_waypoint_count = self.num_waypoints
            if self.waypoint_fixed_num_waypoints > 0:
                sampled_waypoint_count = self.waypoint_fixed_num_waypoints
            se3_samples = self.se3_waypoint_sampler.sample(
                len(local_env_ids), num_waypoints=sampled_waypoint_count
            )
            initial_joint_q = se3_samples.initial_joint_q
            sampling_metadata = se3_samples.waypoint_samples
            self.waypoint_se3_primitive[local_env_ids, :sampled_waypoint_count] = (
                se3_samples.primitive_type
            )
            waypoint_joint_qs = (
                initial_joint_q.unsqueeze(1).expand(-1, self.num_waypoints, -1).clone()
            )
            waypoint_joint_qs[:, :sampled_waypoint_count, :FRANKA_NUM_ARM_JOINTS] = (
                sampling_metadata.joint_qs
            )
            self._write_sampling_metadata(local_env_ids, sampling_metadata)
            for waypoint_idx in range(self.num_waypoints):
                self._write_waypoint_from_joint_q(
                    local_env_ids, waypoint_idx, waypoint_joint_qs[:, waypoint_idx]
                )
        else:
            initial_joint_q = self._load_fixed_eval_tasks(local_env_ids)

        joint_q = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
        with torch.no_grad():
            joint_q[local_env_ids] = initial_joint_q
            self.episode_initial_joint_q[local_env_ids] = initial_joint_q
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        self.prev_eef_pos[local_env_ids] = self._current_eef_pose()[local_env_ids, :3]
        self._sync_active_target_pose(local_env_ids)
        self._render_current_state()
        return self._get_obs(), {}

    def export_task_batch(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        return {
            "initial_joint_q": self.episode_initial_joint_q[env_ids].detach().cpu(),
            "initial_eef_pose": self._current_eef_pose()[env_ids].detach().cpu(),
            "episode_num_waypoints": self.episode_num_waypoints[env_ids].detach().cpu(),
            "waypoint_valid_mask": self.waypoint_valid_mask[env_ids].detach().cpu(),
            "waypoint_type": self.waypoint_type[env_ids].detach().cpu(),
            "waypoint_pos_mask": self.waypoint_pos_mask[env_ids].detach().cpu(),
            "waypoint_rot_mask": self.waypoint_rot_mask[env_ids].detach().cpu(),
            "waypoint_joint_mask": self.waypoint_joint_mask[env_ids].detach().cpu(),
            "waypoints": self.waypoints[env_ids].detach().cpu(),
            "waypoint_quats": self.waypoint_quats[env_ids].detach().cpu(),
            "waypoint_joint_qs": self.waypoint_joint_qs[env_ids].detach().cpu(),
            "waypoint_motion_scale_h": self.waypoint_motion_scale_h[env_ids]
            .detach()
            .cpu(),
            "waypoint_distance_bucket": self.waypoint_distance_bucket[env_ids]
            .detach()
            .cpu(),
            "waypoint_active_joint_count": self.waypoint_active_joint_count[env_ids]
            .detach()
            .cpu(),
            "waypoint_direction_relation": self.waypoint_direction_relation[env_ids]
            .detach()
            .cpu(),
            "waypoint_se3_primitive": self.waypoint_se3_primitive[env_ids]
            .detach()
            .cpu(),
        }

    def set_fixed_eval_tasks(self, payload, waypoint_count=None):
        tasks = payload.get("tasks", payload)
        missing = [field for field in WAYPOINT_TASK_FIELDS if field not in tasks]
        if missing:
            raise ValueError(f"Fixed eval tasks missing fields: {missing}")
        stored_fields = WAYPOINT_TASK_FIELDS + tuple(
            field for field in WAYPOINT_TASK_OPTIONAL_FIELDS if field in tasks
        )
        self._fixed_eval_tasks = {
            field: tasks[field].detach().to(self.device) for field in stored_fields
        }
        if self._fixed_eval_tasks["waypoints"].shape[1] != self.num_waypoints:
            raise ValueError(
                "Eval set waypoint_max does not match env num_waypoints: "
                f"{self._fixed_eval_tasks['waypoints'].shape[1]} vs {self.num_waypoints}"
            )
        self.set_fixed_eval_waypoint_count(waypoint_count)

    def clear_fixed_eval_tasks(self):
        self._fixed_eval_tasks = None
        self._fixed_eval_indices = None
        self._fixed_eval_cursor = 0

    def set_fixed_eval_waypoint_count(self, waypoint_count=None):
        if self._fixed_eval_tasks is None:
            return
        counts = self._fixed_eval_tasks["episode_num_waypoints"].long()
        if waypoint_count is None:
            indices = torch.arange(len(counts), dtype=torch.long, device=self.device)
        else:
            indices = (
                (counts == int(waypoint_count)).nonzero(as_tuple=False).squeeze(-1)
            )
            if indices.numel() == 0:
                raise ValueError(f"Eval set has no tasks with k={waypoint_count}")
        self._fixed_eval_indices = indices
        self._fixed_eval_cursor = 0

    @property
    def fixed_eval_task_count(self):
        if self._fixed_eval_indices is None:
            return 0
        return int(self._fixed_eval_indices.numel())

    def _next_fixed_eval_indices(self, count):
        if self._fixed_eval_indices is None or self._fixed_eval_indices.numel() == 0:
            raise ValueError("Fixed eval task index set is empty")
        cursor = torch.arange(count, dtype=torch.long, device=self.device)
        cursor = (
            cursor + int(self._fixed_eval_cursor)
        ) % self._fixed_eval_indices.numel()
        self._fixed_eval_cursor = (
            int(self._fixed_eval_cursor) + int(count)
        ) % self._fixed_eval_indices.numel()
        return self._fixed_eval_indices[cursor]

    def _load_fixed_eval_tasks(self, local_env_ids):
        task_ids = self._next_fixed_eval_indices(len(local_env_ids))
        tasks = self._fixed_eval_tasks
        self.current_fixed_eval_task_id[local_env_ids] = task_ids
        self.episode_num_waypoints[local_env_ids] = tasks["episode_num_waypoints"][
            task_ids
        ].long()
        self.waypoint_valid_mask[local_env_ids] = tasks["waypoint_valid_mask"][task_ids]
        self.waypoint_type[local_env_ids] = tasks["waypoint_type"][task_ids].long()
        self.waypoint_pos_mask[local_env_ids] = tasks["waypoint_pos_mask"][task_ids]
        self.waypoint_rot_mask[local_env_ids] = tasks["waypoint_rot_mask"][task_ids]
        self.waypoint_joint_mask[local_env_ids] = tasks["waypoint_joint_mask"][task_ids]
        self.waypoints[local_env_ids] = tasks["waypoints"][task_ids]
        self.waypoint_quats[local_env_ids] = tasks["waypoint_quats"][task_ids]
        self.waypoint_joint_qs[local_env_ids] = tasks["waypoint_joint_qs"][task_ids]
        if "waypoint_se3_primitive" in tasks:
            self.waypoint_se3_primitive[local_env_ids] = tasks[
                "waypoint_se3_primitive"
            ][task_ids].long()
        initial_joint_q = tasks["initial_joint_q"][task_ids]
        sampling_fields = (
            "waypoint_motion_scale_h",
            "waypoint_distance_bucket",
            "waypoint_active_joint_count",
            "waypoint_direction_relation",
        )
        if all(field in tasks for field in sampling_fields):
            metadata = WaypointJointSamples(
                joint_qs=self.waypoint_joint_qs[local_env_ids],
                scale_h=tasks["waypoint_motion_scale_h"][task_ids],
                distance_bucket=tasks["waypoint_distance_bucket"][task_ids].long(),
                active_joint_count=tasks["waypoint_active_joint_count"][
                    task_ids
                ].long(),
                direction_relation=tasks["waypoint_direction_relation"][
                    task_ids
                ].long(),
            )
        else:
            metadata = self.waypoint_sampler.describe_waypoints(
                initial_joint_q[:, :FRANKA_NUM_ARM_JOINTS],
                self.waypoint_joint_qs[local_env_ids],
            )
        self._write_sampling_metadata(local_env_ids, metadata)
        return initial_joint_q

    def _sample_balanced_episode_num_waypoints(self, count: int):
        k_values = torch.arange(
            self.waypoint_min_num_waypoints,
            self.num_waypoints + 1,
            dtype=torch.long,
            device=self.device,
        )
        num_k = len(k_values)
        base = count // num_k
        remainder = count % num_k
        counts_per_k = torch.full((num_k,), base, dtype=torch.long, device=self.device)
        if remainder > 0:
            counts_per_k[:remainder] += 1
        episode_counts = torch.repeat_interleave(k_values, counts_per_k)
        perm = torch.randperm(count, device=self.device)
        return episode_counts[perm]

    def _sample_episode_num_waypoints(self, local_env_ids):
        count = len(local_env_ids)
        if self.waypoint_fixed_num_waypoints > 0:
            episode_counts = torch.full(
                (count,),
                self.waypoint_fixed_num_waypoints,
                dtype=torch.long,
                device=self.device,
            )
        elif self.waypoint_min_num_waypoints == self.num_waypoints:
            episode_counts = torch.full(
                (count,), self.num_waypoints, dtype=torch.long, device=self.device
            )
        else:
            episode_counts = self._sample_balanced_episode_num_waypoints(count)
        self.episode_num_waypoints[local_env_ids] = episode_counts
        waypoint_ids = torch.arange(self.num_waypoints, device=self.device).unsqueeze(0)
        self.waypoint_valid_mask[local_env_ids] = (
            waypoint_ids < episode_counts.unsqueeze(1)
        ).to(torch.float32)

    def _sample_waypoint_modality(self, local_env_ids):
        """Assign per-waypoint type and pos/rot/joint masks.

        Sampling mode (``waypoint_space``) only chooses which types appear:
          cartesian — CARTESIAN, or POSITION_ONLY for intermediate when
                      ``waypoint_intermediate_orientation`` is False
          joint     — JOINT
          mixed     — each WP independently JOINT with probability
                      ``waypoint_joint_fraction``, else CARTESIAN / POSITION_ONLY
        Invalid (padded) slots get type 0 and zero masks.
        """
        count = len(local_env_ids)
        valid = self.waypoint_valid_mask[local_env_ids]
        if self.waypoint_space == "joint":
            wp_type = torch.full(
                (count, self.num_waypoints),
                WP_TYPE_JOINT,
                dtype=torch.long,
                device=self.device,
            )
        else:
            wp_type = torch.full(
                (count, self.num_waypoints),
                WP_TYPE_CARTESIAN,
                dtype=torch.long,
                device=self.device,
            )
            if not self.waypoint_intermediate_orientation:
                episode_counts = self.episode_num_waypoints[local_env_ids]
                wp_ids = torch.arange(self.num_waypoints, device=self.device).unsqueeze(
                    0
                )
                intermediate = wp_ids < (episode_counts.unsqueeze(1) - 1).clamp(min=0)
                wp_type = torch.where(
                    intermediate,
                    torch.full_like(wp_type, WP_TYPE_POSITION_ONLY),
                    wp_type,
                )
            if self.waypoint_space == "mixed":
                is_joint = (
                    torch.rand(count, self.num_waypoints, device=self.device)
                    < self.waypoint_joint_fraction
                )
                wp_type = torch.where(
                    is_joint, torch.full_like(wp_type, WP_TYPE_JOINT), wp_type
                )

        wp_type = torch.where(valid > 0.5, wp_type, torch.zeros_like(wp_type))
        pos_mask = (
            (wp_type == WP_TYPE_CARTESIAN) | (wp_type == WP_TYPE_POSITION_ONLY)
        ).to(torch.float32)
        rot_mask = (wp_type == WP_TYPE_CARTESIAN).to(torch.float32)
        joint_mask = (wp_type == WP_TYPE_JOINT).to(torch.float32)
        pos_mask = pos_mask * valid
        rot_mask = rot_mask * valid
        joint_mask = joint_mask * valid

        self.waypoint_type[local_env_ids] = wp_type
        self.waypoint_pos_mask[local_env_ids] = pos_mask
        self.waypoint_rot_mask[local_env_ids] = rot_mask
        self.waypoint_joint_mask[local_env_ids] = joint_mask

    def _write_sampling_metadata(
        self, local_env_ids: torch.Tensor, metadata: WaypointJointSamples
    ) -> None:
        sampled_waypoint_count = int(metadata.scale_h.shape[1])
        target = (local_env_ids, slice(0, sampled_waypoint_count))
        self.waypoint_motion_scale_h[target] = metadata.scale_h
        self.waypoint_distance_bucket[target] = metadata.distance_bucket
        self.waypoint_active_joint_count[target] = metadata.active_joint_count
        self.waypoint_direction_relation[target] = metadata.direction_relation

    def _write_waypoint_from_joint_q(
        self, local_env_ids, waypoint_idx, waypoint_joint_q
    ):
        joint_q = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
        with torch.no_grad():
            joint_q[local_env_ids] = waypoint_joint_q
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        body_q_t = wp.to_torch(self.state_0.body_q).view(self._num_envs, -1, 7)
        ee_indices = self.ee_body_indices.to(device=self.device, dtype=torch.long)
        waypoint_poses = body_q_t[local_env_ids, ee_indices[local_env_ids], :3].detach()
        waypoint_quats = body_q_t[
            local_env_ids, ee_indices[local_env_ids], 3:7
        ].detach()
        self.waypoints[local_env_ids, waypoint_idx] = waypoint_poses
        self.waypoint_quats[local_env_ids, waypoint_idx] = waypoint_quats
        self.waypoint_joint_qs[local_env_ids, waypoint_idx] = waypoint_joint_q[
            :, :FRANKA_NUM_ARM_JOINTS
        ]

    def step(self, actions):
        self.step_count += 1
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        prev_eef_pos = self.prev_eef_pos.clone()

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

        eef_pose = self._current_eef_pose()
        metric_active_waypoint_idx = self.active_waypoint_idx.clone()
        (
            reward,
            reached,
            final_pos_distance,
            final_rot_distance,
            final_joint_distance,
        ) = self._advance_waypoints(joint_q_t[:, :FRANKA_NUM_ARM_JOINTS], eef_pose)
        self._update_episode_metrics(
            prev_eef_pos,
            eef_pose,
            joint_q_t[:, :FRANKA_NUM_ARM_JOINTS],
            active_waypoint_idx=metric_active_waypoint_idx,
        )

        self.last_action = actions.clone()
        self._sim_time += self.frame_dt
        self._sync_active_target_pose()
        self._render_current_state()

        terminated = self.active_waypoint_idx >= self.episode_num_waypoints
        truncated = self._check_truncated()
        done_mask = terminated | truncated
        infos = self._build_infos(
            terminated, final_pos_distance, final_rot_distance, final_joint_distance
        )

        obs = self._get_obs()
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs, _ = self.reset(reset_ids)

        return obs, reward, terminated, truncated, infos

    def _get_obs(self):
        joint_q_t = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
        return self._build_obs(
            joint_q_t[:, :FRANKA_NUM_ARM_JOINTS],
            self._current_eef_pose(),
        )

    def _build_obs(self, joint_pos, eef_pose):
        active_onehot = torch.nn.functional.one_hot(
            self.active_waypoint_idx.clamp(max=self.num_waypoints - 1),
            num_classes=self.num_waypoints,
        ).to(dtype=torch.float32, device=self.device)
        if self.canonicalize_quat_obs:
            eef_pose = torch.cat(
                [eef_pose[:, :3], _canonicalize_quat(eef_pose[:, 3:7])], dim=-1
            )
        waypoint_quats = (
            _canonicalize_quat(self.waypoint_quats)
            if self.canonicalize_quat_obs
            else self.waypoint_quats
        )
        # Clone buffers: APG reset mutates masks/targets in-place.
        pos_sel = self.waypoint_pos_mask.unsqueeze(-1).clone()
        rot_sel = self.waypoint_rot_mask.unsqueeze(-1).clone()
        joint_sel = self.waypoint_joint_mask.unsqueeze(-1).clone()
        waypoints = self.waypoints.clone()
        waypoint_joint_qs = self.waypoint_joint_qs.clone()
        waypoint_quats = waypoint_quats.clone()
        pos_block = waypoints * pos_sel
        quat_block = waypoint_quats * rot_sel
        identity = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=waypoint_quats.dtype, device=self.device
        )
        quat_block = torch.where(
            rot_sel < 0.5,
            identity.view(1, 1, 4).expand_as(quat_block),
            quat_block,
        )
        joint_block = waypoint_joint_qs * joint_sel

        obs_parts = [
            joint_pos,
            eef_pose,
            pos_block.reshape(self._num_envs, self.num_waypoints * 3),
            quat_block.reshape(self._num_envs, self.num_waypoints * 4),
            joint_block.reshape(
                self._num_envs, self.num_waypoints * FRANKA_NUM_ARM_JOINTS
            ),
            active_onehot.clone(),
            self.waypoint_valid_mask.clone(),
            self.waypoint_pos_mask.clone(),
            self.waypoint_rot_mask.clone(),
            self.waypoint_joint_mask.clone(),
            self.last_action.clone(),
        ]
        if self.waypoint_use_relative_obs:
            active_pos, active_quat = self._active_waypoint_pose()
            active_pos = active_pos.clone()
            active_quat = active_quat.clone()
            delta_quat = _quat_mul(active_quat, _quat_inverse(eef_pose[:, 3:7]))
            if self.canonicalize_quat_obs:
                delta_quat = _canonicalize_quat(delta_quat)
            cart_rel = torch.cat([active_pos - eef_pose[:, :3], delta_quat], dim=-1)
            joint_rel = (
                self._active_waypoint_joints().clone()
                - joint_pos[:, :FRANKA_NUM_ARM_JOINTS]
            )
            active_joint = self._active_joint_mask().unsqueeze(-1).clone()
            obs_parts.append(torch.where(active_joint > 0.5, joint_rel, cart_rel))

            eef_pos = eef_pose[:, :3]
            eef_quat = eef_pose[:, 3:7]
            rel_pos = (pos_block - eef_pos.unsqueeze(1)) * pos_sel
            inv_eef = _quat_inverse(eef_quat).unsqueeze(1).expand_as(quat_block)
            rel_quat = _quat_mul(quat_block, inv_eef)
            if self.canonicalize_quat_obs:
                rel_quat = _canonicalize_quat(rel_quat)
            rel_quat = torch.where(
                rot_sel < 0.5,
                identity.view(1, 1, 4).expand_as(rel_quat),
                rel_quat,
            )
            joint_err = (
                joint_block - joint_pos[:, :FRANKA_NUM_ARM_JOINTS].unsqueeze(1)
            ) * joint_sel
            obs_parts.extend(
                [
                    rel_pos.reshape(self._num_envs, self.num_waypoints * 3),
                    rel_quat.reshape(self._num_envs, self.num_waypoints * 4),
                    joint_err.reshape(
                        self._num_envs, self.num_waypoints * FRANKA_NUM_ARM_JOINTS
                    ),
                ]
            )
        obs_parts.append(self.waypoint_type.to(torch.float32).clone())
        return torch.cat(obs_parts, dim=-1)

    def _current_eef_pose(self):
        body_q_t = wp.to_torch(self.state_0.body_q).view(self._num_envs, -1, 7)
        ee_indices = self.ee_body_indices.to(device=self.device, dtype=torch.long)
        return body_q_t[torch.arange(self._num_envs, device=self.device), ee_indices, :]

    def _active_waypoint_pose(self):
        env_idx = torch.arange(self._num_envs, device=self.device)
        waypoint_idx = torch.minimum(
            self.active_waypoint_idx,
            (self.episode_num_waypoints - 1).clamp(min=0),
        )
        return (
            self.waypoints[env_idx, waypoint_idx],
            self.waypoint_quats[env_idx, waypoint_idx],
        )

    def _active_waypoint_joints(self):
        env_idx = torch.arange(self._num_envs, device=self.device)
        waypoint_idx = torch.minimum(
            self.active_waypoint_idx,
            (self.episode_num_waypoints - 1).clamp(min=0),
        )
        return self.waypoint_joint_qs[env_idx, waypoint_idx]

    def _active_masks(self):
        env_idx = torch.arange(self._num_envs, device=self.device)
        waypoint_idx = torch.minimum(
            self.active_waypoint_idx,
            (self.episode_num_waypoints - 1).clamp(min=0),
        )
        return (
            self.waypoint_pos_mask[env_idx, waypoint_idx],
            self.waypoint_rot_mask[env_idx, waypoint_idx],
            self.waypoint_joint_mask[env_idx, waypoint_idx],
        )

    def _active_joint_mask(self):
        return self._active_masks()[2]

    def _active_rot_mask(self):
        return self._active_masks()[1]

    def _active_orientation_required(self):
        return self._active_rot_mask() > 0.5

    def _active_rot_weight(self):
        return self._active_orientation_required().to(torch.float32) * float(
            self.waypoint_rot_weight
        )

    def _active_pose_distances(self, eef_pose):
        active_pos, active_quat = self._active_waypoint_pose()
        pos_dist = (eef_pose[:, :3] - active_pos).norm(dim=-1)
        rot_dist = _quat_angle_distance(eef_pose[:, 3:7], active_quat)
        return pos_dist, rot_dist

    def _check_active_reached(self, pos_dist, rot_dist):
        rot_required = self._active_orientation_required()
        rot_ok = (~rot_required) | (rot_dist <= self.waypoint_rot_threshold)
        return (pos_dist <= self.waypoint_pos_threshold) & rot_ok

    def _check_truncated(self):
        step_budget = self.episode_num_waypoints * self.waypoint_steps_per_waypoint
        return self.step_count >= step_budget

    def _compute_pose_reward(self, pos_dist, rot_dist):
        # Keep the established dense reward in squared chordal distance while
        # success and evaluation use the benchmark's rotation angle in radians.
        rot_chordal_sq = 2.0 - 2.0 * torch.cos(0.5 * rot_dist)
        rot_precision_dist = 2.0 * (torch.sqrt(rot_chordal_sq + 1.0e-8) - 1.0e-4)
        rot_weight = self._active_rot_weight()
        normalized_pos_error = pos_dist / self.waypoint_pos_threshold
        normalized_constraint_error = normalized_pos_error
        normalized_rot_error = rot_dist / self.waypoint_rot_threshold
        if self.waypoint_pose_constraint_aggregation == "sum":
            combined_constraint_error = (
                normalized_constraint_error + normalized_rot_error
            )
        elif self.waypoint_pose_constraint_aggregation == "l2":
            combined_constraint_error = torch.sqrt(
                normalized_constraint_error.square()
                + normalized_rot_error.square()
                + 1.0e-8
            )
        elif self.waypoint_pose_constraint_aggregation == "smoothmax":
            combined_constraint_error = torch.logaddexp(
                normalized_constraint_error, normalized_rot_error
            )
        else:
            combined_constraint_error = torch.maximum(
                normalized_constraint_error, normalized_rot_error
            )
        normalized_constraint_error = torch.where(
            self._active_orientation_required(),
            combined_constraint_error,
            normalized_constraint_error,
        )
        pos_feasibility = torch.sigmoid(
            self.waypoint_pose_feasibility_beta
            * (1.0 - pos_dist / self.waypoint_pos_threshold)
        )
        rot_feasibility = torch.sigmoid(
            self.waypoint_pose_feasibility_beta
            * (1.0 - rot_dist / self.waypoint_rot_threshold)
        )
        max_single_feasibility = torch.sigmoid(
            torch.as_tensor(
                self.waypoint_pose_feasibility_beta,
                dtype=pos_dist.dtype,
                device=pos_dist.device,
            )
        )
        pose_feasibility = torch.where(
            self._active_orientation_required(),
            pos_feasibility * rot_feasibility,
            pos_feasibility,
        )
        max_pose_feasibility = torch.where(
            self._active_orientation_required(),
            max_single_feasibility.square(),
            max_single_feasibility,
        )
        zero_violation = torch.nn.functional.softplus(
            torch.as_tensor(
                -self.waypoint_pose_violation_beta,
                dtype=pos_dist.dtype,
                device=pos_dist.device,
            )
        )
        pos_violation = (
            torch.nn.functional.softplus(
                self.waypoint_pose_violation_beta * (normalized_pos_error - 1.0)
            )
            - zero_violation
        )
        rot_violation = (
            torch.nn.functional.softplus(
                self.waypoint_pose_violation_beta * (normalized_rot_error - 1.0)
            )
            - zero_violation
        )
        pose_violation = (
            torch.where(
                self._active_orientation_required(),
                pos_violation + rot_violation,
                pos_violation,
            )
            / self.waypoint_pose_violation_beta
        )
        peak = WAYPOINT_POSE_DENSE_PEAK
        return (
            -self.waypoint_pos_weight * pos_dist
            + peak * (torch.exp(-(pos_dist**2) / 0.02) - 1.0)
            - rot_weight * rot_chordal_sq
            - self._active_orientation_required().to(torch.float32)
            * self.waypoint_rot_precision_weight
            * rot_precision_dist
            - self.waypoint_pose_constraint_weight * normalized_constraint_error
            + self.waypoint_pose_feasibility_weight
            * (pose_feasibility - max_pose_feasibility)
            - self.waypoint_pose_violation_weight * pose_violation
            + rot_weight * (torch.exp(-(rot_chordal_sq**2) / 0.18) - 1.0)
        )

    def _active_joint_distance(self, joint_pos):
        active_joints = self._active_waypoint_joints()
        return (joint_pos[:, :FRANKA_NUM_ARM_JOINTS] - active_joints).norm(dim=-1)

    def _compute_joint_reward(self, joint_dist):
        return (
            -self.waypoint_joint_weight * joint_dist
            + self.waypoint_joint_dense_peak
            * (torch.exp(-(joint_dist**2) / self.waypoint_joint_exp_scale) - 1.0)
            + self.waypoint_joint_precision_dense_peak
            * (
                torch.exp(-(joint_dist**2) / self.waypoint_joint_precision_exp_scale)
                - 1.0
            )
        )

    def _record_reached_modalities(self, reached):
        if not reached.any():
            return
        env_idx = torch.arange(self._num_envs, device=self.device)
        active_idx = self.active_waypoint_idx.clamp(max=self.num_waypoints - 1)
        active_type = self.waypoint_type[env_idx, active_idx]
        reached_f = reached.to(torch.float32)
        for t in range(NUM_WP_TYPES):
            self.episode_reached_by_type[:, t] += (active_type == t).to(
                torch.float32
            ) * reached_f

    def _select_active_reached(self, joint_dist, pos_dist, rot_dist):
        """Reached criterion from active modality masks.

        Exclusive: joint_mask selects joint vs pose (no MIXED_HARD sum).
        Within pose, rot_mask gates orientation via ``_active_rot_weight``;
        pos_mask is obs-only and does not affect reward/reached.
        """
        if self.waypoint_space == "joint":
            return joint_dist < self.waypoint_joint_threshold
        if self.waypoint_space == "cartesian":
            return self._check_active_reached(pos_dist, rot_dist)
        joint_mask = self._active_joint_mask()
        reached_joint = joint_dist < self.waypoint_joint_threshold
        reached_cart = self._check_active_reached(pos_dist, rot_dist)
        return torch.where(joint_mask > 0.5, reached_joint, reached_cart)

    def _advance_waypoints(self, joint_pos, eef_pose):
        pos_dist, rot_dist = self._active_pose_distances(eef_pose)
        joint_dist = self._active_joint_distance(joint_pos)
        reached = self._select_active_reached(joint_dist, pos_dist, rot_dist)
        if self.waypoint_space == "joint":
            reward = self._compute_joint_reward(joint_dist)
        elif self.waypoint_space == "cartesian":
            reward = self._compute_pose_reward(pos_dist, rot_dist)
        else:
            joint_mask = self._active_joint_mask()
            reward = torch.where(
                joint_mask > 0.5,
                self._compute_joint_reward(joint_dist),
                self._compute_pose_reward(pos_dist, rot_dist),
            )
        reward = reward + reached.to(torch.float32) * self.waypoint_bonus
        self._record_reached_steps(reached)
        self._record_reached_modalities(reached)
        self.active_waypoint_idx = self.active_waypoint_idx + reached.to(torch.long)
        return (
            reward,
            reached,
            pos_dist.detach(),
            rot_dist.detach(),
            joint_dist.detach(),
        )

    def _record_reached_steps(self, reached):
        reached_env_ids = reached.nonzero(as_tuple=False).squeeze(-1)
        if reached_env_ids.numel() > 0:
            reached_waypoint_ids = self.active_waypoint_idx[reached_env_ids]
            self.episode_waypoint_reached_step[
                reached_env_ids, reached_waypoint_ids
            ] = self.step_count[reached_env_ids].to(torch.long)

    def _update_episode_metrics(
        self, prev_eef_pos, eef_pose, joint_pos, active_waypoint_idx=None
    ):
        step_path_length = (eef_pose[:, :3] - prev_eef_pos).norm(dim=-1)
        self.episode_path_length += step_path_length.detach()
        self._update_min_waypoint_distances(
            eef_pose, joint_pos, active_waypoint_idx=active_waypoint_idx
        )
        self.prev_eef_pos = eef_pose[:, :3].detach().clone()

    def _update_min_waypoint_distances(
        self, eef_pose, joint_pos, active_waypoint_idx=None
    ):
        all_pos_distances = (self.waypoints - eef_pose[:, :3].unsqueeze(1)).norm(dim=-1)
        all_rot_distances = _quat_angle_distance(
            eef_pose[:, 3:7].unsqueeze(1), self.waypoint_quats
        )
        all_joint_distances = (
            self.waypoint_joint_qs - joint_pos[:, :FRANKA_NUM_ARM_JOINTS].unsqueeze(1)
        ).norm(dim=-1)
        valid_mask = self.waypoint_valid_mask.to(torch.bool)
        all_pos_distances = torch.where(
            valid_mask, all_pos_distances, self.episode_min_waypoint_pos_distance
        )
        all_rot_distances = torch.where(
            valid_mask, all_rot_distances, self.episode_min_waypoint_rot_distance
        )
        all_joint_distances = torch.where(
            valid_mask, all_joint_distances, self.episode_min_waypoint_joint_distance
        )
        self.episode_min_waypoint_pos_distance = torch.minimum(
            self.episode_min_waypoint_pos_distance, all_pos_distances.detach()
        )
        self.episode_min_waypoint_rot_distance = torch.minimum(
            self.episode_min_waypoint_rot_distance, all_rot_distances.detach()
        )
        self.episode_min_waypoint_joint_distance = torch.minimum(
            self.episode_min_waypoint_joint_distance, all_joint_distances.detach()
        )

        if active_waypoint_idx is None:
            active_waypoint_idx = self.active_waypoint_idx
        active_waypoint_idx = active_waypoint_idx.detach().long()
        active_valid = active_waypoint_idx < self.episode_num_waypoints
        active_idx = torch.minimum(
            active_waypoint_idx,
            (self.episode_num_waypoints - 1).clamp(min=0),
        )
        env_idx = torch.arange(self._num_envs, device=self.device)
        pos_error = all_pos_distances[env_idx, active_idx] / self.waypoint_pos_threshold
        rot_error = all_rot_distances[env_idx, active_idx] / self.waypoint_rot_threshold
        pose_error = torch.where(
            self.waypoint_rot_mask[env_idx, active_idx] > 0.5,
            torch.maximum(pos_error, rot_error),
            pos_error,
        )
        joint_error = (
            all_joint_distances[env_idx, active_idx] / self.waypoint_joint_threshold
        )
        constraint_error = torch.where(
            self.waypoint_joint_mask[env_idx, active_idx] > 0.5,
            joint_error,
            pose_error,
        ).detach()
        previous_error = self.episode_min_active_constraint_error[env_idx, active_idx]
        updated_error = torch.where(
            active_valid,
            torch.minimum(previous_error, constraint_error),
            previous_error,
        )
        self.episode_min_active_constraint_error[env_idx, active_idx] = updated_error

    def _build_infos(
        self, success, final_pos_distance, final_rot_distance, final_joint_distance
    ):
        valid_counts = self.episode_num_waypoints.to(torch.float32).clamp(min=1.0)
        valid_mask = self.waypoint_valid_mask.to(torch.bool)
        min_pos = torch.where(
            valid_mask,
            self.episode_min_waypoint_pos_distance,
            torch.zeros_like(self.episode_min_waypoint_pos_distance),
        )
        min_rot = torch.where(
            valid_mask,
            self.episode_min_waypoint_rot_distance,
            torch.zeros_like(self.episode_min_waypoint_rot_distance),
        )
        max_pos = torch.where(
            valid_mask,
            self.episode_min_waypoint_pos_distance,
            torch.full_like(self.episode_min_waypoint_pos_distance, -float("inf")),
        )
        max_rot = torch.where(
            valid_mask,
            self.episode_min_waypoint_rot_distance,
            torch.full_like(self.episode_min_waypoint_rot_distance, -float("inf")),
        )
        min_joint = torch.where(
            valid_mask,
            self.episode_min_waypoint_joint_distance,
            torch.zeros_like(self.episode_min_waypoint_joint_distance),
        )
        max_joint = torch.where(
            valid_mask,
            self.episode_min_waypoint_joint_distance,
            torch.full_like(self.episode_min_waypoint_joint_distance, -float("inf")),
        )
        active_constraint_visited = valid_mask & torch.isfinite(
            self.episode_min_active_constraint_error
        )
        active_constraint_count = active_constraint_visited.to(torch.float32).sum(
            dim=-1
        )
        active_constraint_values = torch.where(
            active_constraint_visited,
            self.episode_min_active_constraint_error,
            torch.zeros_like(self.episode_min_active_constraint_error),
        )
        active_constraint_max_values = torch.where(
            active_constraint_visited,
            self.episode_min_active_constraint_error,
            torch.full_like(self.episode_min_active_constraint_error, -float("inf")),
        )
        joint_fraction = (
            (self.waypoint_joint_mask * self.waypoint_valid_mask).sum(dim=-1)
            / valid_counts
        ).detach()
        type_counts = torch.zeros(
            self._num_envs, NUM_WP_TYPES, dtype=torch.float32, device=self.device
        )
        for t in range(NUM_WP_TYPES):
            type_counts[:, t] = (
                ((self.waypoint_type == t) & valid_mask).to(torch.float32).sum(dim=-1)
            )
        reached_by_type = self.episode_reached_by_type.detach().clone()
        completion_by_type = reached_by_type / type_counts.clamp(min=1.0)
        waypoint_ids = torch.arange(self.num_waypoints, device=self.device).unsqueeze(0)
        reached_mask = valid_mask & (
            waypoint_ids < self.active_waypoint_idx.unsqueeze(1)
        )

        def _stratified_completion(category_ids, num_categories):
            counts = torch.zeros(
                self._num_envs,
                num_categories,
                dtype=torch.float32,
                device=self.device,
            )
            reached_counts = torch.zeros_like(counts)
            for category in range(num_categories):
                category_mask = valid_mask & (category_ids == category)
                counts[:, category] = category_mask.to(torch.float32).sum(dim=-1)
                reached_counts[:, category] = (
                    (category_mask & reached_mask).to(torch.float32).sum(dim=-1)
                )
            return counts, reached_counts / counts.clamp(min=1.0)

        active_group = torch.full_like(self.waypoint_active_joint_count, -1)
        active_group = torch.where(
            self.waypoint_active_joint_count == 1,
            torch.full_like(active_group, ACTIVE_GROUP_SINGLE),
            active_group,
        )
        active_group = torch.where(
            (self.waypoint_active_joint_count >= 2)
            & (self.waypoint_active_joint_count <= 3),
            torch.full_like(active_group, ACTIVE_GROUP_SPARSE),
            active_group,
        )
        active_group = torch.where(
            self.waypoint_active_joint_count >= 4,
            torch.full_like(active_group, ACTIVE_GROUP_DENSE),
            active_group,
        )
        distance_bucket_counts, completion_by_distance_bucket = _stratified_completion(
            self.waypoint_distance_bucket,
            self.waypoint_sampler.num_distance_buckets,
        )
        active_group_counts, completion_by_active_group = _stratified_completion(
            active_group, NUM_ACTIVE_GROUPS
        )
        direction_relation_counts, completion_by_direction_relation = (
            _stratified_completion(
                self.waypoint_direction_relation, NUM_DIRECTION_RELATIONS
            )
        )
        se3_primitive_counts, completion_by_se3_primitive = _stratified_completion(
            self.waypoint_se3_primitive, NUM_SE3_PRIMITIVES
        )

        def _modality_mean(distances, type_id):
            mask = valid_mask & (self.waypoint_type == type_id)
            denom = mask.to(torch.float32).sum(dim=-1).clamp(min=1.0)
            mean = (distances * mask.to(torch.float32)).sum(dim=-1) / denom
            return torch.where(
                mask.any(dim=-1),
                mean,
                torch.full_like(mean, float("nan")),
            ).detach()

        return {
            "success": success.detach().clone(),
            "waypoints_reached": self.active_waypoint_idx.detach().clone(),
            "episode_num_waypoints": self.episode_num_waypoints.detach().clone(),
            "fixed_eval_task_id": self.current_fixed_eval_task_id.detach().clone(),
            "waypoint_reached_steps": self.episode_waypoint_reached_step.detach().clone(),
            "final_distance": final_pos_distance.detach(),
            "final_rot_distance": final_rot_distance.detach(),
            "final_joint_distance": final_joint_distance.detach(),
            "waypoint_joint_fraction": joint_fraction,
            "waypoint_type_counts": type_counts.detach().clone(),
            "waypoints_reached_by_type": reached_by_type,
            "waypoint_completion_by_type": completion_by_type.detach().clone(),
            "waypoint_motion_scale_h": self.waypoint_motion_scale_h.detach().clone(),
            "waypoint_distance_bucket": self.waypoint_distance_bucket.detach().clone(),
            "waypoint_active_joint_count": self.waypoint_active_joint_count.detach().clone(),
            "waypoint_direction_relation": self.waypoint_direction_relation.detach().clone(),
            "waypoint_se3_primitive": self.waypoint_se3_primitive.detach().clone(),
            "waypoint_distance_bucket_counts": distance_bucket_counts.detach().clone(),
            "waypoint_completion_by_distance_bucket": completion_by_distance_bucket.detach().clone(),
            "waypoint_active_group_counts": active_group_counts.detach().clone(),
            "waypoint_completion_by_active_group": completion_by_active_group.detach().clone(),
            "waypoint_direction_relation_counts": direction_relation_counts.detach().clone(),
            "waypoint_completion_by_direction_relation": completion_by_direction_relation.detach().clone(),
            "waypoint_se3_primitive_counts": se3_primitive_counts.detach().clone(),
            "waypoint_completion_by_se3_primitive": completion_by_se3_primitive.detach().clone(),
            "mean_waypoint_motion_scale_h": (
                (self.waypoint_motion_scale_h * self.waypoint_valid_mask).sum(dim=-1)
                / valid_counts
            )
            .detach()
            .clone(),
            "min_waypoint_pos_distances": self.episode_min_waypoint_pos_distance.detach().clone(),
            "min_waypoint_rot_distances": self.episode_min_waypoint_rot_distance.detach().clone(),
            "min_waypoint_joint_distances": self.episode_min_waypoint_joint_distance.detach().clone(),
            "min_active_constraint_errors": self.episode_min_active_constraint_error.detach().clone(),
            "waypoint_valid_mask": self.waypoint_valid_mask.detach().clone(),
            "waypoint_pos_mask": self.waypoint_pos_mask.detach().clone(),
            "waypoint_rot_mask": self.waypoint_rot_mask.detach().clone(),
            "waypoint_joint_mask": self.waypoint_joint_mask.detach().clone(),
            "mean_min_pos_distance_cartesian": _modality_mean(
                self.episode_min_waypoint_pos_distance, WP_TYPE_CARTESIAN
            ),
            "mean_min_rot_distance_cartesian": _modality_mean(
                self.episode_min_waypoint_rot_distance, WP_TYPE_CARTESIAN
            ),
            "mean_min_pos_distance_position_only": _modality_mean(
                self.episode_min_waypoint_pos_distance, WP_TYPE_POSITION_ONLY
            ),
            "mean_min_joint_distance_joint": _modality_mean(
                self.episode_min_waypoint_joint_distance, WP_TYPE_JOINT
            ),
            "mean_min_waypoint_pos_distance": (min_pos.sum(dim=-1) / valid_counts)
            .detach()
            .clone(),
            "max_min_waypoint_pos_distance": max_pos.max(dim=-1)
            .values.detach()
            .clone(),
            "mean_min_waypoint_rot_distance": (min_rot.sum(dim=-1) / valid_counts)
            .detach()
            .clone(),
            "max_min_waypoint_rot_distance": max_rot.max(dim=-1)
            .values.detach()
            .clone(),
            "mean_min_waypoint_joint_distance": (min_joint.sum(dim=-1) / valid_counts)
            .detach()
            .clone(),
            "max_min_waypoint_joint_distance": max_joint.max(dim=-1)
            .values.detach()
            .clone(),
            "mean_min_active_constraint_error": (
                active_constraint_values.sum(dim=-1)
                / active_constraint_count.clamp(min=1.0)
            )
            .detach()
            .clone(),
            "max_min_active_constraint_error": active_constraint_max_values.max(dim=-1)
            .values.detach()
            .clone(),
            "active_constraint_coverage": (active_constraint_count / valid_counts)
            .detach()
            .clone(),
            "path_length": self.episode_path_length.detach().clone(),
        }

    def _sync_active_target_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        active_pos, active_quat = self._active_waypoint_pose()
        self.target_pos[env_ids] = active_pos[env_ids]
        self.target_quat[env_ids] = active_quat[env_ids]


class FrankaWaypointReachAPGEnv(FrankaWaypointReachVecEnv):
    """Batched ordered-waypoint reaching environment for APG."""

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
        self._differentiable_joint_q = self._current_arm_joint_q()

    def _current_arm_joint_q(self):
        return (
            wp.to_torch(self.state_0.joint_q)
            .view(self._num_envs, FRANKA_NUM_JOINTS)[:, :FRANKA_NUM_ARM_JOINTS]
            .detach()
            .clone()
        )

    def reset(self, env_ids=None, seed=None):
        previous_q = getattr(self, "_differentiable_joint_q", None)
        obs, info = super().reset(env_ids=env_ids, seed=seed)
        fresh_q = self._current_arm_joint_q()
        if previous_q is None or env_ids is None:
            self._differentiable_joint_q = fresh_q
        else:
            reset_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            ).reshape(-1)
            reset_mask = torch.zeros(
                self._num_envs, dtype=torch.bool, device=self.device
            )
            reset_mask[reset_ids] = True
            self._differentiable_joint_q = torch.where(
                reset_mask.unsqueeze(-1), fresh_q, previous_q
            )
        return obs, info

    def detach_state(self):
        """Release the completed full-rollout graph before the next episode."""
        self._differentiable_joint_q = self._differentiable_joint_q.detach()
        self.last_action = self.last_action.detach()

    def step(self, action):
        self.step_count += 1
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = torch.clamp(action.to(self.device), -1.0, 1.0)
        prev_eef_pos = self.prev_eef_pos.clone()

        is_joint = self.waypoint_space == "joint"
        is_mixed = self.waypoint_space == "mixed"
        current_q = self._differentiable_joint_q
        new_jpos = (current_q + action * self.action_scale).clamp(
            self.arm_joint_limit_lower, self.arm_joint_limit_upper
        )

        fn_eef_pose = None
        if is_joint:
            dense_reward = self._compute_joint_reward(
                self._active_joint_distance(new_jpos)
            )
        else:
            active_waypoint_pos, active_waypoint_quat = self._active_waypoint_pose()
            sim_state = {
                "model": self.model,
                "state_joint_q": self.state_0.joint_q,
                "num_envs": self._num_envs,
                "action_scale": self.action_scale,
                "joint_limit_lower_wp": self._joint_limit_lower_wp,
                "joint_limit_upper_wp": self._joint_limit_upper_wp,
                "ee_body_indices_wp": self._ee_global_wp,
                "active_waypoint_pos": active_waypoint_pos,
                "active_waypoint_quat": active_waypoint_quat,
                "active_rot_weight": self._active_rot_weight(),
                "active_rot_required": self._active_orientation_required(),
                "active_rot_precision_weight": self._active_orientation_required().to(
                    torch.float32
                )
                * self.waypoint_rot_precision_weight,
                "waypoint_pos_weight": self.waypoint_pos_weight,
                "waypoint_pos_threshold": self.waypoint_pos_threshold,
                "waypoint_rot_threshold": self.waypoint_rot_threshold,
                "waypoint_pose_constraint_weight": self.waypoint_pose_constraint_weight,
                "waypoint_pose_constraint_aggregation": self.waypoint_pose_constraint_aggregation,
                "waypoint_pose_feasibility_weight": self.waypoint_pose_feasibility_weight,
                "waypoint_pose_feasibility_beta": self.waypoint_pose_feasibility_beta,
                "waypoint_pose_violation_weight": self.waypoint_pose_violation_weight,
                "waypoint_pose_violation_beta": self.waypoint_pose_violation_beta,
            }
            pose_dense_reward, fn_eef_pose = _NewtonWaypointStepFunc.apply(
                current_q, action, sim_state
            )
            if is_mixed:
                joint_dense_reward = self._compute_joint_reward(
                    self._active_joint_distance(new_jpos)
                )
                joint_mask = self._active_joint_mask().detach()
                dense_reward = torch.where(
                    joint_mask > 0.5, joint_dense_reward, pose_dense_reward
                )
            else:
                dense_reward = pose_dense_reward

        with torch.no_grad():
            joint_q_t = wp.to_torch(self.state_0.joint_q).view(self._num_envs, -1)
            joint_q_t[:, :FRANKA_NUM_ARM_JOINTS] = new_jpos.detach()
            newton.eval_fk(
                self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
            )
        self._differentiable_joint_q = new_jpos

        eef_pose = fn_eef_pose if fn_eef_pose is not None else self._current_eef_pose()
        pos_dist, rot_dist = self._active_pose_distances(eef_pose)
        joint_dist = self._active_joint_distance(new_jpos.detach())
        reached = self._select_active_reached(joint_dist, pos_dist, rot_dist)
        self._update_episode_metrics(prev_eef_pos, eef_pose, new_jpos.detach())
        reward = dense_reward + reached.to(torch.float32) * self.waypoint_bonus
        self._record_reached_steps(reached)
        self._record_reached_modalities(reached)
        self.active_waypoint_idx = self.active_waypoint_idx + reached.to(torch.long)

        self.last_action = action.detach().clone()
        self._sim_time += self.frame_dt
        self._sync_active_target_pose()
        self._render_current_state()

        obs = self._build_obs(new_jpos, eef_pose)

        terminated = self.active_waypoint_idx >= self.episode_num_waypoints
        truncated = self._check_truncated()
        done_mask = terminated | truncated
        infos = self._build_infos(
            terminated,
            pos_dist.detach(),
            rot_dist.detach(),
            joint_dist.detach(),
        )

        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self.reset(reset_ids)
            fresh_obs = self._get_obs()
            obs = torch.where(done_mask.unsqueeze(-1).expand_as(obs), fresh_obs, obs)

        return obs, reward, terminated, truncated, infos


@register_learning_env("FrankaWaypointNMG-v0")
class FrankaWaypointNMGEnv(FrankaWaypointReachAPGEnv):
    """EmbodiChain adapter for the reference Franka waypoint APG environment.

    The numerical environment, sampler, reward, and Warp autograd bridge remain
    aligned with ``neural_motion_generator`` PR #6. This adapter contributes the
    variable complete-rollout contract and the canonical observation-normalizer
    mask required by :class:`~embodichain.learning.rl.DifferentiableTrainer`.

    Args:
        num_envs: Number of independent Franka environments.
        device: Torch/Newton device.
        num_waypoints: Maximum number of waypoint tokens and K stratum.
        waypoint_min_num_waypoints: Minimum scheduled K stratum.
        waypoint_fixed_num_waypoints: Fixed K, or zero for stratified scheduling.
        waypoint_steps_per_waypoint: Closed-loop action budget for each waypoint.
        max_episode_steps: Truncation limit; defaults to the maximum K horizon.
        headless: Whether to use Newton's null viewer.
        canonicalize_quat_obs: Whether to canonicalize quaternion observations.
        **kwargs: Reference waypoint environment reward and sampling options.
    """

    def __init__(
        self,
        num_envs: int = 4,
        device: torch.device | str = "cpu",
        num_waypoints: int = 8,
        waypoint_min_num_waypoints: int = 1,
        waypoint_fixed_num_waypoints: int = 0,
        waypoint_steps_per_waypoint: int = 30,
        max_episode_steps: int | None = None,
        headless: bool = True,
        canonicalize_quat_obs: bool = True,
        **kwargs: object,
    ) -> None:
        if max_episode_steps is None:
            max_episode_steps = num_waypoints * waypoint_steps_per_waypoint
        torch_device = torch.device(device)
        self._rollout_fixed_num_waypoints = int(waypoint_fixed_num_waypoints)
        super().__init__(
            num_envs=num_envs,
            device=str(torch_device),
            max_episode_steps=max_episode_steps,
            num_waypoints=num_waypoints,
            waypoint_min_num_waypoints=waypoint_min_num_waypoints,
            waypoint_fixed_num_waypoints=waypoint_fixed_num_waypoints,
            waypoint_steps_per_waypoint=waypoint_steps_per_waypoint,
            headless=headless,
            canonicalize_quat_obs=canonicalize_quat_obs,
            **kwargs,
        )
        self.device = torch_device
        self.observation_normalize_mask = self.obs_normalize_mask

    def prepare_differentiable_rollout(
        self,
        rollout_index: int,
    ) -> DifferentiableRolloutSpec:
        """Select K and return the complete ``K * steps_per_waypoint`` horizon.

        Args:
            rollout_index: Zero-based independent-rollout index.

        Returns:
            Complete horizon with reciprocal-K objective normalization.
        """
        waypoint_count = self._rollout_fixed_num_waypoints
        if waypoint_count == 0:
            waypoint_count = stratified_rollout_value(
                rollout_index,
                self.waypoint_min_num_waypoints,
                self.num_waypoints,
            )
        self.waypoint_fixed_num_waypoints = waypoint_count
        return DifferentiableRolloutSpec(
            num_steps=waypoint_count * self.waypoint_steps_per_waypoint,
            objective_scale=1.0 / waypoint_count,
            metadata={"waypoint_count": float(waypoint_count)},
        )

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Reset all rows or the subset requested by the reference step path.

        Args:
            env_ids: Optional row indices used by automatic reset.
            seed: Optional deterministic reset seed.
            options: Learning-env options; ``reset_ids`` aliases ``env_ids``.

        Returns:
            Fresh unified waypoint observation and reset info.
        """
        if options is not None and "reset_ids" in options:
            if env_ids is not None:
                raise ValueError("Specify env_ids or options['reset_ids'], not both.")
            env_ids = torch.as_tensor(
                options["reset_ids"],
                dtype=torch.long,
                device=self.device,
            )
        observation, info = super().reset(env_ids=env_ids, seed=seed)
        return observation, info

    def detach_state(self) -> torch.Tensor:
        """Release the full-rollout graph and return the current observation.

        Returns:
            Detached observation matching the environment's current state.
        """
        super().detach_state()
        return self._get_obs().detach()
