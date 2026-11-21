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

"""Torch MDP definitions for the three retained LEAP manipulation tasks.

Cube reorientation and rotation equations are adapted from MuJoCo Playground,
Copyright 2025 DeepMind Technologies Limited, under Apache 2.0.
The physics bridge supplies named tensors; this module does not step physics.
Quaternion tensors use wxyz throughout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

__all__ = ["LeapTask", "quat_multiply", "quat_matrix", "quat_integrate"]

TASK_IDS = ("LeapCubeFaceTurn", "LeapCubeReorient", "LeapCubeRotateZAxis")


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Multiply batched wxyz quaternions."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1),
        ),
        dim=-1,
    )


def _normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _inverse(q: Tensor) -> Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_matrix(q: Tensor) -> Tensor:
    """Return batched rotation matrices for normalized wxyz quaternions."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_integrate(q: Tensor, velocity: Tensor, dt: float) -> Tensor:
    """Integrate local angular velocity by right-multiplying the increment."""
    angle = velocity.norm(dim=-1, keepdim=True) * dt
    factor = 0.5 * dt * torch.sinc(angle / (2 * math.pi))
    delta = torch.cat((torch.cos(angle / 2), velocity * factor), dim=-1)
    return _normalize(quat_multiply(q, delta))


def _orientation_error(q: Tensor, goal: Tensor) -> Tensor:
    diff = _normalize(quat_multiply(q, _inverse(goal)))
    return 2 * torch.asin(diff[..., 1:].norm(dim=-1).clamp(max=1))


def _linear_tolerance(error: Tensor, bound: float, margin: float) -> Tensor:
    # Playground tolerance uses value_at_margin=0.1 for the linear sigmoid.
    return (1 - 0.9 * (error - bound).clamp_min(0) / margin).clamp_min(0)


class LeapTask:
    """Maintain task state and evaluate batched source-compatible LEAP MDPs.

    Args:
        task_id: One of LeapCubeFaceTurn, LeapCubeReorient, LeapCubeRotateZAxis.
        num_envs: Number of parallel environments.
        qpos0: Source MJCF home qpos; face turn uses the saved settled qpos.
        ctrl0: Source actuator home targets in the 16-joint policy order.
        lower: Actuator control lower bounds.
        upper: Actuator control upper bounds.
        device: Device shared with physics state tensors.
        seed: Seed for task reset and observation noise.
        noise_level: Multiplier for the reference observation noise.
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int,
        qpos0: Tensor,
        ctrl0: Tensor,
        lower: Tensor,
        upper: Tensor,
        device: str | torch.device = "cpu",
        seed: int = 42,
        noise_level: float = 1.0,
    ) -> None:
        task_id = {"M3": TASK_IDS[0], "M4": TASK_IDS[1], "M5": TASK_IDS[2]}.get(
            task_id, task_id
        )
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown LEAP task: {task_id}")
        self.task_id = task_id
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.qpos0, self.ctrl0, self.lower, self.upper = (
            torch.as_tensor(x, device=self.device, dtype=self.dtype).clone()
            for x in (qpos0, ctrl0, lower, upper)
        )
        self.face_turn = task_id == "LeapCubeFaceTurn"
        self.reorient = task_id == "LeapCubeReorient"
        expected_q = 24 if self.face_turn else 23
        if self.qpos0.shape != (expected_q,) or self.ctrl0.shape != (16,):
            raise ValueError(f"Expected qpos ({expected_q},) and ctrl (16,)")
        if self.lower.shape != (16,) or self.upper.shape != (16,):
            raise ValueError("Expected 16 actuator control bounds")
        self.default_pose = self.qpos0[:16].clone()
        self.action_dim = 16
        self.actor_dim = 65 if self.face_turn else (57 if self.reorient else 32)
        self.critic_dim = 65 if self.face_turn else (128 if self.reorient else 105)
        self.control_dt = 0.05
        self.sim_dt = 0.005 if self.face_turn else 0.01
        self.substeps = round(self.control_dt / self.sim_dt)
        self.episode_length = 1000 if self.reorient else 500
        self.noise_level = noise_level
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_action = self._zeros(16)
        self.last_last_action = self._zeros(16)
        self.motor_targets = self._zeros(16)
        self.initial_pos = self._zeros(3)
        self.initial_quat = self._zeros(4)
        self.goal_quat = self._zeros(4)
        self.goal_velocity = self._zeros(3)
        self.goal_angle = self._zeros()
        self.previous_face_angle = self._zeros()
        self.rotation = self._zeros()
        self.held = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.success_count = torch.zeros_like(self.held)
        self.steps = torch.zeros_like(self.held)
        self.steps_since_success = torch.zeros_like(self.held)
        self.goal_quat[:, 0] = 1

    def _zeros(self, *shape: int) -> Tensor:
        return torch.zeros(
            (self.num_envs, *shape), device=self.device, dtype=self.dtype
        )

    def _rand(self, shape: tuple[int, ...], normal: bool = False) -> Tensor:
        function = torch.randn if normal else torch.rand
        return function(
            shape, device=self.device, dtype=self.dtype, generator=self.generator
        )

    def _ids(self, env_ids: Tensor | None) -> Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _uniform_quat(self, count: int) -> Tensor:
        u, v, w = self._rand((count, 3)).unbind(-1)
        return torch.stack(
            (
                (1 - u).sqrt() * torch.sin(2 * math.pi * v),
                (1 - u).sqrt() * torch.cos(2 * math.pi * v),
                u.sqrt() * torch.sin(2 * math.pi * w),
                u.sqrt() * torch.cos(2 * math.pi * w),
            ),
            dim=-1,
        )

    def reset(self, env_ids: Tensor | None = None) -> dict[str, Tensor]:
        """Reset selected task slots and return their source-ordered states.

        The bridge writes qpos/qvel/ctrl, refreshes kinematics, and calls observe
        on only these slots. All returned arrays have len(env_ids) rows.
        """
        ids = self._ids(env_ids)
        count = ids.numel()
        qpos = self.qpos0.expand(count, -1).clone()
        qvel = torch.zeros(
            (count, self.qpos0.numel() - 1), device=self.device, dtype=self.dtype
        )
        if self.face_turn:
            qpos[:, :16] += 0.02 * self._rand((count, 16), normal=True)
            qpos[:, 16:19] += 0.006 * (self._rand((count, 3)) - 0.5)
            ctrl = self.ctrl0.expand(count, -1).clone()
        else:
            qpos[:, :16] = (
                self.default_pose + 0.1 * self._rand((count, 16), normal=True)
            ).clamp(self.lower, self.upper)
            qpos[:, 16:19] = qpos.new_tensor([0.1, 0, 0.05]) + 0.02 * (
                self._rand((count, 3)) - 0.5
            )
            qpos[:, 19:23] = self._uniform_quat(count)
            ctrl = qpos[:, :16].clone()
        self.initial_pos[ids] = qpos[:, 16:19]
        self.initial_quat[ids] = qpos[:, 19:23]
        self.motor_targets[ids] = ctrl
        self.last_action[ids] = 0
        self.last_last_action[ids] = 0
        self.goal_velocity[ids] = 0
        self.goal_angle[ids] = math.pi / 2
        self.previous_face_angle[ids] = qpos[:, 23] if self.face_turn else 0
        self.rotation[ids] = 0
        self.held[ids] = 0
        self.success_count[ids] = 0
        self.steps[ids] = 0
        self.steps_since_success[ids] = 0
        if self.reorient:
            self.goal_quat[ids] = self._uniform_quat(count)
        else:
            self.goal_quat[ids] = qpos.new_tensor([1, 0, 0, 0])
        return {
            "env_ids": ids,
            "qpos": qpos,
            "qvel": qvel,
            "ctrl": ctrl,
            "goal_quat": self.goal_quat[ids].clone(),
        }

    def control_targets(self, action: Tensor) -> Tensor:
        """Map normalized policy actions to source actuator targets."""
        if action.shape != (self.num_envs, 16):
            raise ValueError(f"Expected actions ({self.num_envs}, 16)")
        if self.face_turn:
            target = (self.ctrl0 + 0.8 * action).clamp(self.lower, self.upper)
        elif self.reorient:
            target = (self.motor_targets + 0.5 * action).clamp(self.lower, self.upper)
        else:
            target = self.default_pose + 0.6 * action
        self.motor_targets.copy_(target)
        return target

    def face_detent(self, face_angle: Tensor) -> Tensor:
        """Return passive face torque, recomputed before each physics substep."""
        offset = face_angle - torch.round(face_angle / (math.pi / 2)) * (math.pi / 2)
        return -0.01 * offset

    def observe(
        self, state: Mapping[str, Tensor], env_ids: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Build observations from full-batch physics tensors for selected slots.

        M3 requires source-ordered qpos/qvel. M4/M5 require hand_qpos,
        hand_qvel, actuator_force, cube_pos, cube_quat, cube_linvel,
        cube_angvel, palm_pos, and fingertip_pos. Fingertips are flattened
        thumb/index/middle/ring positions expressed in the grasp-site frame.
        """
        ids = self._ids(env_ids)
        selected = {key: value[ids] for key, value in state.items()}
        if self.face_turn:
            qpos, qvel = selected["qpos"], selected["qvel"]
            delta = self.goal_angle[ids] - qpos[:, 23]
            obs = torch.cat(
                (
                    qpos[:, :16],
                    0.05 * qvel[:, :16],
                    self.last_action[ids],
                    10 * (qpos[:, 16:19] - self.initial_pos[ids]),
                    qpos[:, 19:23],
                    0.1 * qvel[:, 16:22],
                    torch.stack(
                        (delta.sin(), delta.cos(), delta / math.pi, qvel[:, 22] * 0.1),
                        dim=-1,
                    ),
                ),
                dim=-1,
            )
            return {"state": obs, "privileged_state": obs}
        q, dq = selected["hand_qpos"], selected["hand_qvel"]
        noisy_q = q + self.noise_level * 0.05 * (2 * self._rand(tuple(q.shape)) - 1)
        pos, quat = selected["cube_pos"], selected["cube_quat"]
        pos_error = selected["palm_pos"] - pos
        if self.reorient:
            noisy_pos = pos + self.noise_level * 0.02 * (
                2 * self._rand(tuple(pos.shape)) - 1
            )
            noisy_quat = _normalize(
                quat
                + self.noise_level * 0.1 * self._rand(tuple(quat.shape), normal=True)
            )
            diff = quat_multiply(noisy_quat, _inverse(self.goal_quat[ids]))
            ori_features = quat_matrix(diff).flatten(1)[:, 3:]
            clean_diff = quat_multiply(quat, _inverse(self.goal_quat[ids]))
            clean_ori = quat_matrix(clean_diff).flatten(1)[:, 3:]
            obs = torch.cat(
                (
                    noisy_q,
                    noisy_q - self.motor_targets[ids],
                    selected["palm_pos"] - noisy_pos,
                    ori_features,
                    self.last_action[ids],
                ),
                dim=-1,
            )
            perturbation = selected.get(
                "perturbation_direction", q.new_zeros((len(ids), 6))
            )
            applied = selected.get("cube_external_force", q.new_zeros((len(ids), 6)))
            critic = torch.cat(
                (
                    obs,
                    q,
                    dq,
                    selected["fingertip_pos"],
                    pos_error,
                    clean_ori,
                    selected["cube_linvel"],
                    selected["cube_angvel"],
                    perturbation,
                    applied,
                ),
                dim=-1,
            )
        else:
            obs = torch.cat((noisy_q, self.last_action[ids]), dim=-1)
            critic = torch.cat(
                (
                    obs,
                    q,
                    dq,
                    selected["actuator_force"],
                    selected["fingertip_pos"],
                    pos_error,
                    quat,
                    selected["cube_angvel"],
                    selected["cube_linvel"],
                ),
                dim=-1,
            )
        return {"state": obs, "privileged_state": critic}

    def after_step(
        self, state: Mapping[str, Tensor], action: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Evaluate one control interval after physics and retain task counters.

        No slots reset automatically. M4 updates goal_quat after constructing
        the returned observation, preserving the source implementation order.
        The bridge may copy goal_quat to the visual goal after this call.
        """
        if self.face_turn:
            qpos, qvel = state["qpos"], state["qvel"]
            pos, quat, angle = qpos[:, 16:19], qpos[:, 19:23], qpos[:, 23]
            drift = (pos - self.initial_pos).norm(dim=-1)
            terminated = (
                (pos[:, 2] < -0.03) | (drift > 0.15) | ~torch.isfinite(qpos).all(-1)
            )
            error = (self.goal_angle - angle).abs()
            previous_error = (self.goal_angle - self.previous_face_angle).abs()
            base_angle = 2 * torch.acos(
                (quat * self.initial_quat).sum(-1).abs().clamp(0, 1)
            )
            good = (
                (error < 0.1)
                & (qvel[:, 22].abs() < 1.5)
                & (base_angle < 0.5)
                & ~terminated
            )
            self.held.copy_(torch.where(good, self.held + 1, 0))
            success = self.held >= 20
            cost = 0.02 * (action - self.last_action).square().mean(-1)
            reward = (
                5 * (previous_error - error)
                + (2 * torch.exp(-3 * error) - 0.2 * base_angle - 5 * drift - cost)
                * self.control_dt
                + 10 * success.float()
                - 5 * terminated.float()
            )
            increment = angle - self.previous_face_angle
            self.rotation += torch.atan2(increment.sin(), increment.cos())
            self.previous_face_angle.copy_(angle)
            self.goal_angle += success.float() * (math.pi / 2)
            self.held.masked_fill_(success, 0)
            self.success_count += success.long()
            self.last_last_action.copy_(self.last_action)
            self.last_action.copy_(action)
            obs = self.observe(state)
            metrics = {
                "success": success.float(),
                "error": error,
                "base_angle": base_angle,
                "drift": drift,
                "rotation": self.rotation.clone(),
                "retained": (~terminated).float(),
                "face_angle": angle.clone(),
                "face_velocity": qvel[:, 22].clone(),
            }
        else:
            pos, quat = state["cube_pos"], state["cube_quat"]
            terminated = pos[:, 2] < -0.05
            if self.reorient:
                for name in (
                    "hand_qpos",
                    "hand_qvel",
                    "cube_pos",
                    "cube_quat",
                    "cube_linvel",
                    "cube_angvel",
                ):
                    terminated = terminated | ~torch.isfinite(state[name]).all(-1)
                error = _orientation_error(quat, self.goal_quat)
                success = error < 0.1
                self.success_count += success.long()
                self.steps_since_success.copy_(
                    torch.where(success, 0, self.steps_since_success + 1)
                )
                pos_error = (state["palm_pos"] - pos).norm(dim=-1)
                pose_cost = (state["hand_qpos"] - self.default_pose).square().sum(-1)
                rate_cost = (action - self.last_action).square().sum(-1) + (
                    action - 2 * self.last_action + self.last_last_action
                ).square().sum(-1)
                energy = (state["hand_qvel"].abs() * state["actuator_force"].abs()).sum(
                    -1
                )
                reward = (
                    5 * _linear_tolerance(error, 0.2, math.pi)
                    + 0.5 * _linear_tolerance(pos_error, 0.02, 0.05)
                    - 100 * terminated.float()
                    - 0.5 * pose_cost
                    - 0.001 * rate_cost
                    - 0.001 * energy
                ) * self.control_dt + 100 * success.float()
                obs = self.observe(state)
                self.goal_velocity.copy_(
                    torch.where(
                        success[:, None],
                        1 + 4 * self._rand((self.num_envs, 3)),
                        0.8 * self.goal_velocity,
                    )
                )
                self.goal_quat.copy_(
                    quat_integrate(
                        self.goal_quat, self.goal_velocity, 2 * self.control_dt
                    )
                )
                metrics = {
                    "success": success.float(),
                    "error": error,
                    "position_error": pos_error,
                    "retained": (~terminated).float(),
                    "steps_since_success": self.steps_since_success.clone(),
                }
            else:
                angvel = state["cube_angvel"][:, 2]
                reward = (angvel - 100 * terminated.float()) * self.control_dt
                self.rotation += angvel * self.control_dt
                obs = self.observe(state)
                metrics = {
                    "angular_velocity_z": angvel.clone(),
                    "rotation": self.rotation.clone(),
                    "retained": (~terminated).float(),
                }
            self.last_last_action.copy_(self.last_action)
            self.last_action.copy_(action)
        self.steps += 1
        truncated = (self.steps >= self.episode_length) & ~terminated
        metrics["success_count"] = self.success_count.clone()
        metrics["steps"] = self.steps.clone()
        return obs, reward, terminated, truncated, metrics
