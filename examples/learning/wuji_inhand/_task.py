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
"""Wuji HORA20 observations and value-bootstrapped APG physical task."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytorch_kinematics as pk
import torch
from _physics import JOINTS, _WujiPhysics
from sharpa_rl_unilab.algos.supermimic.state import unflatten_state
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.geometry import (
    torch_quat_angular_velocity_from_pair,
)
from sharpa_rl_unilab.training.supermimic_mdp import SharpaSuperMimicMDP
from unilab.base.torch_env import TorchEnvState

from embodichain.lab.sim import SimulationManager
from embodichain.utils.math import quat_mul

__all__: list[str] = []


class _WujiMDP(SharpaSuperMimicMDP):
    """Use real 20-DOF state and cap estimated torques at authored limits."""

    def __init__(
        self, env: Any, device: torch.device, physical: _WujiPhysics, assets: Path
    ) -> None:
        super().__init__(env, device)
        self.chain = pk.build_chain_from_urdf(
            (assets / "compiled/hand.urdf").read_bytes()
        ).to(dtype=torch.float32, device=self.device)
        self.fk_joint_ids = torch.tensor(
            [JOINTS.index(name) for name in self.chain.get_joint_parameter_names()],
            device=self.device,
            dtype=torch.long,
        )
        self.tip_frame_ids = self.chain.get_frame_indices(
            *(f"right_finger{i}_link4" for i in range(1, 6))
        ).to(device=self.device, dtype=torch.long)
        self.tip_centers = physical.tip_centers.clone()
        self.root_rotation = physical.root_rotation.clone()
        self.root_position = torch.tensor(
            physical.profile["root_position"], device=self.device
        )

    def tip_positions(self, q: torch.Tensor) -> torch.Tensor:
        """Differentiate source hand FK without per-frame CPU synchronization."""
        shape = q.shape[:-1]
        with torch.autocast(device_type=self.device.type, enabled=False):
            flat = q.float().reshape(-1, 20).index_select(-1, self.fk_joint_ids)
            frames = self.chain.forward_kinematics_tensor(flat).index_select(
                0, self.tip_frame_ids
            )
            frames = frames.permute(1, 0, 2, 3)
            local = (frames[..., :3, :3] @ self.tip_centers[None, ..., None])[
                ..., 0
            ] + frames[..., :3, 3]
            world = (self.root_rotation @ local[..., None])[..., 0] + self.root_position
        return world.reshape(*shape, 5, 3)

    def grasp_quality(self, state: Any, context: torch.Tensor) -> torch.Tensor:
        """Soft counterpart of the measured opposing-side-pad success rule."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            force = state[6][..., :5].float().clamp_min(0)
            delta = self.tip_positions(state[4]) - state[0].float().unsqueeze(-2)
            radial = torch.nn.functional.normalize(delta[..., :2], dim=-1, eps=1e-6)
            dot = (radial[..., 1:, :] * radial[..., :1, :]).sum(-1)
            opposite = torch.sigmoid((-0.15 - dot) / 0.08)
            halfheight = 0.016 * context[..., 45:46].float()
            side = torch.sigmoid((0.8 * halfheight - delta[..., 2].abs()) / 0.002)
            touch = torch.sigmoid((force - 0.02) / 0.015) * side
            partners = (touch[..., 1:] * opposite).topk(2, dim=-1).values.prod(-1)
            return touch[..., 0] * partners

    def reward(self, state: Any, context: torch.Tensor) -> torch.Tensor:
        position, _, linear, angular, q, qvel, _, target = state
        kp, kd, _, _, _, _, anchor, _, _ = self.context_parts(context)
        torque = (kp * (target - q) - kd * qvel).clamp(-self.effort, self.effort)
        w = self.reward_weights
        grasp = self.grasp_quality(state, context)
        reward = (
            w["rotate"] * angular[..., 2].clamp(-0.5, 0.5) * grasp
            + w["obj_linvel"] * linear.abs().sum(-1)
            + w["pose_diff"] * (q - self.default).square().sum(-1)
            + w["torque"] * torque.square().sum(-1)
            + w["work"] * (torque * qvel).sum(-1).square()
            + w["object_pos"] * grasp / ((position - anchor).norm(dim=-1) + 0.001)
            + 0.5 * grasp
        )
        return reward * self.dt

    def soft_termination(self, state: Any, context: torch.Tensor) -> torch.Tensor:
        """Penalize lost pad participation as well as the inherited height band."""
        height_failure = super().soft_termination(state, context)
        return 1 - (1 - height_failure) * (
            1 - 0.1 * (1 - self.grasp_quality(state, context))
        )


class _WujiTask:
    """Collect physical transitions and retain terminal state before reset."""

    def __init__(
        self,
        cfg: Any,
        assets: Path,
        output: Path,
        *,
        num_envs: int,
        seed: int,
        gain: float = 1.0,
    ) -> None:
        self.physical = _WujiPhysics(
            assets, output / "scene", num_envs=num_envs, seed=seed, gain=gain
        )
        p = self.physical
        self.sim, self.robot, self.object = p.sim, p.robot, p.object
        self.device, self.num_envs, self.width = p.device, num_envs, p.width
        self.step_dt = 0.05
        self.lower, self.upper = p.lower, p.upper
        self.target = torch.zeros(num_envs, 20, device=self.device)
        self.s = torch.zeros(num_envs, 83, device=self.device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.grasp_gap = torch.zeros_like(self.step_count)
        self.grasp_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.signed_angle_delta = torch.zeros(num_envs, device=self.device)
        self.clean_contact = torch.zeros(num_envs, 5, device=self.device)
        self.tactile = torch.zeros_like(self.clean_contact)
        self.previous_contact = torch.zeros_like(self.clean_contact)
        self.last_contact = torch.zeros_like(self.clean_contact)
        self.actor_history = torch.zeros(num_envs, 3, 45, device=self.device)
        self.critic_history = torch.zeros(num_envs, 3, 54, device=self.device)
        self.proprio_history = torch.zeros(num_envs, 30, 45, device=self.device)
        self.xi = torch.cat(
            (
                p.kp.expand(num_envs, -1),
                p.kd.expand(num_envs, -1),
                torch.ones(num_envs, 1, device=self.device),
                torch.full((num_envs, 1), 0.05, device=self.device),
                torch.zeros(num_envs, 3, device=self.device),
                p.scales[:, None],
                torch.zeros(num_envs, 5, device=self.device),
            ),
            -1,
        )
        lo, hi = p.lower.cpu().numpy(), p.upper.cpu().numpy()
        action = SimpleNamespace(
            ctrl_lower=lo,
            ctrl_upper=hi,
            _target_lower=self.lower,
            _target_upper=self.upper,
            _action_scale=1 / 24,
        )
        env = SimpleNamespace(
            step_dt=self.step_dt,
            cfg=cfg,
            action_manager=SimpleNamespace(get_term=lambda _: action),
            event_manager=SimpleNamespace(
                get_term_cfg=lambda _: SimpleNamespace(
                    func=SimpleNamespace(_rotation_axis=(0, 0, 1))
                )
            ),
        )
        try:
            self.mdp = _WujiMDP(env, self.device, p, assets)
        except BaseException:
            p.close()
            SimulationManager.flush_cleanup_queue()
            raise
        self.mdp.default = torch.cat([x[:, :20] for x in p.cache]).median(dim=0).values
        self.mdp.effort = torch.tensor(list(p.effort.values()), device=self.device)
        self.capture = self._capture
        self.state = None
        self.identity = dict(
            p.identity,
            state_dim=83,
            context_dim=51,
            frame_dim=45,
            actor_dim=135,
            critic_dim=162,
            policy_stage="teacher",
            reset="Native screened grasp plus holding-target preload; random axial body phase",
            parameter_randomization=False,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "native_identity.json").write_text(
            json.dumps(self.identity, indent=2) + "\n"
        )
        self.reset(seed=seed)

    def _capture(self) -> dict[str, torch.Tensor]:
        return {"s": self.s.clone(), "xi": self.xi.clone()}

    def _observe(self, ids: torch.Tensor, *, reset: bool) -> dict[str, torch.Tensor]:
        state = unflatten_state(self.s)
        clean = self.mdp.frame(state, noisy=False)
        noisy = self.mdp.frame(state, noisy=True)
        priv = self.mdp.priv(state, self.xi)
        critic = torch.cat((clean, priv), -1)
        if reset:
            self.actor_history[ids] = noisy[ids, None]
            self.critic_history[ids] = critic[ids, None]
            self.proprio_history[ids] = noisy[ids, None]
        else:
            self.actor_history = torch.cat(
                (self.actor_history[:, 1:], noisy[:, None]), 1
            )
            self.critic_history = torch.cat(
                (self.critic_history[:, 1:], critic[:, None]), 1
            )
            self.proprio_history = torch.cat(
                (self.proprio_history[:, 1:], noisy[:, None]), 1
            )
        return {
            "obs": self.actor_history.flatten(1).clone(),
            "critic": self.critic_history.flatten(1).clone(),
            "priv_info": priv.clone(),
            "proprio_hist": self.proprio_history.clone(),
        }

    @torch.no_grad()
    def reset(
        self, env_indices: torch.Tensor | None = None, *, seed: int | None = None
    ) -> tuple[dict[str, torch.Tensor], dict]:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_indices is None
            else env_indices
        )
        self.physical.reset(ids, seed=seed)
        # Axial phase changes visual orientation without changing cylinder geometry.
        pose = self.object.body_state[ids, :7].clone()
        phase = (
            torch.rand(len(ids), device=self.device, generator=self.physical.generator)
            * 2
            * torch.pi
        )
        twist = torch.zeros(len(ids), 4, device=self.device)
        twist[:, 2], twist[:, 3] = (phase / 2).sin(), (phase / 2).cos()
        pose[:, 3:7] = quat_mul(pose[:, 3:7], twist)
        self.object.set_local_pose(pose, env_ids=ids)
        pos, quat, q = self.physical.state()
        self.target[ids] = self.physical.target[ids]
        self.step_count[ids] = 0
        self.grasp_gap[ids] = 0
        self.grasp_valid[ids] = False
        self.signed_angle_delta[ids] = 0
        for t in [
            self.clean_contact,
            self.tactile,
            self.previous_contact,
            self.last_contact,
        ]:
            t[ids] = 0
        self.xi[ids, 46:49] = pos[ids]
        self.xi[ids, 49] = pos[ids, 2] - 0.02
        self.xi[ids, 50] = pos[ids, 2] + 0.02
        zero = torch.zeros(self.num_envs, 3, device=self.device)
        assembled = torch.cat(
            (
                pos,
                quat,
                zero,
                zero,
                q,
                torch.zeros_like(q),
                self.clean_contact,
                self.tactile,
                self.target,
            ),
            -1,
        )
        self.s[ids] = assembled[ids]
        obs = self._observe(ids, reset=True)
        self.state = TorchEnvState(
            obs,
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            {},
        )
        return obs, {}

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> TorchEnvState:
        previous = self.s.clone()
        self.target.add_(action.clamp(-1, 1), alpha=1 / 24).clamp_(
            self.lower, self.upper
        )
        self.physical.step(self.target)
        force, self.grasp_valid = self.physical.grasp_contact()
        force = force.clamp(0, 4)
        self.grasp_gap = torch.where(self.grasp_valid, 0, self.grasp_gap + 1)
        self.clean_contact.copy_(force)
        smooth = 0.5 * (force + self.previous_contact)
        self.previous_contact.copy_(force)
        self.last_contact.copy_(
            torch.where(torch.rand_like(force) < 0.005, self.last_contact, smooth)
        )
        self.tactile.copy_(self.last_contact)
        pos, quat, q = self.physical.state()
        linear = (pos - previous[:, :3]) / 0.05
        angular = torch_quat_angular_velocity_from_pair(quat, previous[:, 3:7], 0.05)
        velocity = (q - previous[:, 13:33]) / 0.05
        self.s = torch.cat(
            (
                pos,
                quat,
                linear,
                angular,
                q,
                velocity,
                self.clean_contact,
                self.tactile,
                self.target,
            ),
            -1,
        )
        self.signed_angle_delta.copy_(angular[:, 2] * 0.05)
        self.step_count += 1
        terminated = (
            (pos[:, 2] < self.xi[:, 49])
            | (pos[:, 2] > self.xi[:, 50])
            | (self.grasp_gap >= 20)
        )
        truncated = (self.step_count >= 400) & ~terminated
        obs = self._observe(
            torch.arange(self.num_envs, device=self.device), reset=False
        )
        reward = self.mdp.reward(unflatten_state(self.s), self.xi)
        self.state = TorchEnvState(obs, reward, terminated, truncated, {})
        return self.state

    def close(self) -> None:
        self.sim = self.robot = self.object = None
        self.physical.close()
        self.physical = None
        SimulationManager.flush_cleanup_queue()
