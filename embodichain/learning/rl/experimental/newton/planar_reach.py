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

"""FK-only Newton environment for differentiable RL validation."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import newton
import numpy as np
import torch
import warp as wp

from embodichain.utils import configclass

__all__ = ["NewtonPlanarReachEnv", "NewtonPlanarReachEnvCfg"]

_NUM_JOINTS = 2
_OBSERVATION_DIM = 8


@wp.kernel
def _apply_joint_delta(
    action: wp.array(dtype=wp.float32),
    current_q: wp.array(dtype=wp.float32),
    next_q: wp.array(dtype=wp.float32),
    action_scale: wp.float32,
    joint_limit: wp.float32,
):
    index = wp.tid()
    next_q[index] = wp.clamp(
        current_q[index] + action[index] * action_scale,
        -joint_limit,
        joint_limit,
    )


@wp.kernel
def _compute_reach_reward(
    body_q: wp.array(dtype=wp.transformf),
    end_body_indices: wp.array(dtype=wp.int32),
    target_xy: wp.array(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    position = wp.transform_get_translation(body_q[end_body_indices[env_index]])
    dx = position.x - target_xy[env_index * 2]
    dy = position.y - target_xy[env_index * 2 + 1]
    reward[env_index] = -(dx * dx + dy * dy)


class _NewtonPlanarReachStep(torch.autograd.Function):
    """Bridge a Newton/Warp FK reward tape into PyTorch autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        action: torch.Tensor,
        current_q: torch.Tensor,
        sim_state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = sim_state["model"]
        num_envs = sim_state["num_envs"]
        action_wp = wp.from_torch(
            action.detach().reshape(-1).contiguous(),
            dtype=wp.float32,
            requires_grad=True,
        )
        current_q_wp = wp.from_torch(
            current_q.detach().reshape(-1).contiguous(),
            dtype=wp.float32,
            requires_grad=True,
        )
        next_q_wp = wp.zeros(
            num_envs * _NUM_JOINTS,
            dtype=wp.float32,
            device=model.device,
            requires_grad=True,
        )
        target_xy_wp = wp.from_torch(
            sim_state["target_xy"].detach().reshape(-1).contiguous(),
            dtype=wp.float32,
        )
        reward_wp = wp.zeros(
            num_envs,
            dtype=wp.float32,
            device=model.device,
            requires_grad=True,
        )
        fk_state = model.state()

        tape = wp.Tape()
        with tape:
            wp.launch(
                _apply_joint_delta,
                dim=num_envs * _NUM_JOINTS,
                inputs=[
                    action_wp,
                    current_q_wp,
                    next_q_wp,
                    wp.float32(sim_state["action_scale"]),
                    wp.float32(sim_state["joint_limit"]),
                ],
                device=model.device,
            )
            wp.copy(fk_state.joint_qd, model.joint_qd)
            newton.eval_fk(model, next_q_wp, fk_state.joint_qd, fk_state)
            wp.launch(
                _compute_reach_reward,
                dim=num_envs,
                inputs=[
                    fk_state.body_q,
                    sim_state["end_body_indices"],
                    target_xy_wp,
                    reward_wp,
                ],
                device=model.device,
            )

        body_q = wp.to_torch(fk_state.body_q)
        end_indices = wp.to_torch(sim_state["end_body_indices"]).long()
        end_xy = body_q[end_indices, :2].detach().clone()

        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.current_q_wp = current_q_wp
        ctx.reward_wp = reward_wp
        ctx.action_shape = action.shape
        ctx.current_q_shape = current_q.shape
        ctx.mark_non_differentiable(end_xy)
        return wp.to_torch(reward_wp).detach().clone(), end_xy

    @staticmethod
    def backward(
        ctx: Any,
        reward_gradient: torch.Tensor | None,
        end_xy_gradient: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del end_xy_gradient
        if reward_gradient is None:
            action_gradient = torch.zeros_like(wp.to_torch(ctx.action_wp)).reshape(
                ctx.action_shape
            )
            current_q_gradient = torch.zeros_like(
                wp.to_torch(ctx.current_q_wp)
            ).reshape(ctx.current_q_shape)
            return action_gradient, current_q_gradient, None

        reward_gradient_wp = wp.from_torch(
            reward_gradient.detach().contiguous(),
            dtype=wp.float32,
        )
        wp.copy(ctx.reward_wp.grad, reward_gradient_wp)
        ctx.tape.backward()
        action_gradient = (
            wp.to_torch(ctx.action_wp.grad).clone().reshape(ctx.action_shape)
        )
        current_q_gradient = (
            wp.to_torch(ctx.current_q_wp.grad).clone().reshape(ctx.current_q_shape)
        )
        ctx.tape.zero()
        return action_gradient, current_q_gradient, None


@configclass
class NewtonPlanarReachEnvCfg:
    """Configuration for the temporary two-link Newton reach environment."""

    num_envs: int = 4
    device: str = "cpu"
    action_scale: float = 0.2
    joint_limit: float = float(np.pi)
    max_episode_steps: int = 32
    success_threshold: float = 0.05
    first_link_length: float = 1.0
    second_link_length: float = 0.8
    initial_joint_scale: float = 0.5
    target_joint_scale: float = 1.0


class NewtonPlanarReachEnv:
    """Batched differentiable two-link reach environment using Newton FK."""

    def __init__(self, cfg: NewtonPlanarReachEnvCfg | None = None) -> None:
        self.cfg = cfg or NewtonPlanarReachEnvCfg()
        if self.cfg.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.cfg.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")

        self.num_envs = self.cfg.num_envs
        self.device = torch.device(self.cfg.device)
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_NUM_JOINTS,),
            dtype=np.float32,
        )

        self._model, self._end_body_indices = self._build_model()
        self._state = self._model.state()
        self._joint_q = torch.zeros(
            (self.num_envs, _NUM_JOINTS),
            dtype=torch.float32,
            device=self.device,
        )
        self._target_xy = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self._end_xy = torch.zeros_like(self._target_xy)
        self._last_action = torch.zeros(
            (self.num_envs, _NUM_JOINTS),
            dtype=torch.float32,
            device=self.device,
        )
        self._step_count = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )
        self._generator = torch.Generator(device=self.device)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset all environments with FK-reachable random targets."""
        del options
        if seed is not None:
            self._generator.manual_seed(seed)
        self._joint_q = self._joint_q.detach()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(env_ids)
        return self._get_observation(), {}

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Apply a differentiable joint delta and evaluate the reach reward."""
        if action.shape != (self.num_envs, _NUM_JOINTS):
            raise ValueError(
                f"Expected action shape {(self.num_envs, _NUM_JOINTS)}, "
                f"got {tuple(action.shape)}."
            )
        action = action.to(device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        current_q = self._joint_q
        reward, end_xy = _NewtonPlanarReachStep.apply(
            action,
            current_q,
            {
                "model": self._model,
                "end_body_indices": self._end_body_indices,
                "target_xy": self._target_xy,
                "num_envs": self.num_envs,
                "action_scale": self.cfg.action_scale,
                "joint_limit": self.cfg.joint_limit,
            },
        )
        next_q = (current_q + action * self.cfg.action_scale).clamp(
            -self.cfg.joint_limit,
            self.cfg.joint_limit,
        )
        self._joint_q = next_q
        self._end_xy = end_xy

        self._step_count += 1
        self._last_action = action.detach().clone()
        distance = torch.linalg.vector_norm(end_xy - self._target_xy, dim=-1)
        terminated = distance < self.cfg.success_threshold
        truncated = self._step_count >= self.cfg.max_episode_steps
        done = terminated | truncated
        observation = torch.cat(
            [next_q, end_xy, self._target_xy, self._last_action],
            dim=-1,
        )
        info = {
            "distance": distance.detach(),
            "success": terminated.detach(),
        }

        if done.any():
            self._reset_envs(torch.nonzero(done, as_tuple=False).squeeze(-1))
            fresh_observation = self._get_observation()
            observation = torch.where(
                done.unsqueeze(-1),
                fresh_observation,
                observation,
            )
        return observation, reward, terminated, truncated, info

    def detach_state(self) -> torch.Tensor:
        """Return the current observation at a truncated-gradient boundary."""
        self._joint_q = self._joint_q.detach()
        return self._get_observation().detach()

    def _build_model(self) -> tuple[Any, wp.array]:
        template = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
        first_link = template.add_link(label="planar_first_link")
        template.add_shape_sphere(first_link, radius=0.01)
        second_link = template.add_link(label="planar_end_link")
        template.add_shape_sphere(second_link, radius=0.01)
        first_joint = template.add_joint_revolute(
            parent=-1,
            child=first_link,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat_identity(),
            ),
            child_xform=wp.transform(
                wp.vec3(0.0, self.cfg.first_link_length, 0.0),
                wp.quat_identity(),
            ),
            limit_lower=-self.cfg.joint_limit,
            limit_upper=self.cfg.joint_limit,
        )
        second_joint = template.add_joint_revolute(
            parent=first_link,
            child=second_link,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat_identity(),
            ),
            child_xform=wp.transform(
                wp.vec3(0.0, self.cfg.second_link_length, 0.0),
                wp.quat_identity(),
            ),
            limit_lower=-self.cfg.joint_limit,
            limit_upper=self.cfg.joint_limit,
        )
        template.add_articulation([first_joint, second_joint])

        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
        builder.replicate(
            template,
            world_count=self.num_envs,
            spacing=(0.0, 0.0, 0.0),
        )
        model = builder.finalize(device=str(self.device), requires_grad=True)
        end_indices = [
            index
            for index, label in enumerate(model.body_label)
            if "planar_end_link" in str(label)
        ]
        if len(end_indices) != self.num_envs:
            raise RuntimeError(
                f"Expected {self.num_envs} end links, found {len(end_indices)}."
            )
        return model, wp.array(
            end_indices,
            dtype=wp.int32,
            device=model.device,
        )

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        initial_q = self._sample_joint_positions(
            count,
            scale=self.cfg.initial_joint_scale,
        )
        target_q = self._sample_joint_positions(
            count,
            scale=self.cfg.target_joint_scale,
        )
        reset_q = torch.zeros_like(self._joint_q)
        reset_q[env_ids] = initial_q
        reset_mask = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.bool,
            device=self.device,
        )
        reset_mask[env_ids] = True
        self._joint_q = torch.where(reset_mask, reset_q, self._joint_q)
        with torch.no_grad():
            wp.to_torch(self._state.joint_q).copy_(self._joint_q.detach().reshape(-1))
        self._target_xy[env_ids] = self._analytical_end_xy(target_q)
        self._last_action[env_ids] = 0.0
        self._step_count[env_ids] = 0
        newton.eval_fk(
            self._model,
            self._state.joint_q,
            self._state.joint_qd,
            self._state,
        )
        body_q = wp.to_torch(self._state.body_q)
        end_indices = wp.to_torch(self._end_body_indices).long()
        self._end_xy[env_ids] = body_q[end_indices[env_ids], :2]

    def _sample_joint_positions(self, count: int, scale: float) -> torch.Tensor:
        return (
            torch.rand(
                (count, _NUM_JOINTS),
                device=self.device,
                generator=self._generator,
            )
            * 2.0
            - 1.0
        ) * (self.cfg.joint_limit * scale)

    def _analytical_end_xy(self, joint_q: torch.Tensor) -> torch.Tensor:
        first_angle = joint_q[:, 0]
        combined_angle = first_angle + joint_q[:, 1]
        x = self.cfg.first_link_length * torch.sin(
            first_angle
        ) + self.cfg.second_link_length * torch.sin(combined_angle)
        y = -self.cfg.first_link_length * torch.cos(
            first_angle
        ) - self.cfg.second_link_length * torch.cos(combined_angle)
        return torch.stack([x, y], dim=-1)

    def _get_observation(self) -> torch.Tensor:
        return torch.cat(
            [self._joint_q, self._end_xy, self._target_xy, self._last_action],
            dim=-1,
        )
