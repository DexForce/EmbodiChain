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

import torch
import torch.nn as nn

from embodichain.utils import configclass
from embodichain.utils.math import (
    convert_quat,
    quat_error_magnitude,
    quat_from_matrix,
)
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver
from embodichain.lab.sim.solvers.qpos_seed_sampler import QposSeedSampler

__all__ = ["NeuralIKSolverCfg", "NeuralIKSolver"]


@configclass
class NeuralIKSolverCfg(SolverCfg):
    """Configuration for the neural network IK solver."""

    class_type: str = "NeuralIKSolver"

    checkpoint_path: str = ""
    """Path to the trained policy checkpoint (.pt file)."""

    max_steps: int = 30
    """Number of policy inference iterations per IK solve."""

    action_scale: float = 0.2
    """Action scaling factor (radians)."""

    obs_dim: int | None = None
    """Observation dimension. If None, auto-computed as ``2 * num_arm_joints + 14``."""

    num_arm_joints: int = 7
    """Number of arm joints (policy only controls arm, not fingers)."""

    hidden_dims: list[int] = [256, 256]
    """Hidden layer dimensions for the MLP policy network."""

    pos_eps: float = 0.01
    """Position convergence tolerance (meters) for success check."""

    rot_eps: float = 0.1
    """Rotation convergence tolerance (radians) for success check."""

    num_samples: int = 1
    """Number of random initial qpos seeds to sample per target pose."""

    def init_solver(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> NeuralIKSolver:
        if self.obs_dim is None:
            self.obs_dim = 2 * self.num_arm_joints + 14
        solver = NeuralIKSolver(cfg=self, device=device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


def _build_mlp(obs_dim: int, hidden_dims: list[int], action_dim: int) -> nn.Sequential:
    """Build an MLP with Tanh activations between hidden layers."""
    layers = []
    in_dim = obs_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.Tanh())
        in_dim = h
    layers.append(nn.Linear(in_dim, action_dim))
    return nn.Sequential(*layers)


class NeuralIKSolver(BaseSolver):
    """IK solver using a trained neural network policy.

    Loads a checkpoint containing actor_mean weights and obs_normalizer stats,
    then runs iterative inference to solve IK queries.
    """

    def __init__(self, cfg: NeuralIKSolverCfg, device=None, **kwargs):
        super().__init__(cfg=cfg, device=device, **kwargs)

        self._max_steps = cfg.max_steps
        self._action_scale = cfg.action_scale
        self._num_arm_joints = cfg.num_arm_joints
        self._pos_eps = cfg.pos_eps
        self._rot_eps = cfg.rot_eps
        self._num_samples = cfg.num_samples

        ckpt = torch.load(
            cfg.checkpoint_path, map_location=self.device, weights_only=False
        )

        if "agent" not in ckpt:
            raise KeyError(
                f"Checkpoint at '{cfg.checkpoint_path}' is missing 'agent' key. "
                f"Available keys: {list(ckpt.keys())}. "
                f"Expected a checkpoint from the analytic_policy_gradients training pipeline."
            )
        actor_keys = [k for k in ckpt["agent"] if k.startswith("actor_mean.")]
        if not actor_keys:
            raise KeyError(
                f"Checkpoint 'agent' has no 'actor_mean.*' keys. "
                f"Available: {list(ckpt['agent'].keys())}."
            )
        if "obs_normalizer" not in ckpt:
            raise KeyError(
                f"Checkpoint at '{cfg.checkpoint_path}' is missing 'obs_normalizer'. "
                f"Available keys: {list(ckpt.keys())}."
            )
        for subkey in ("mean", "var"):
            if subkey not in ckpt["obs_normalizer"]:
                raise KeyError(
                    f"Checkpoint 'obs_normalizer' is missing '{subkey}'. "
                    f"Available: {list(ckpt['obs_normalizer'].keys())}."
                )

        self.mlp = _build_mlp(cfg.obs_dim, cfg.hidden_dims, cfg.num_arm_joints)

        state_dict = {
            k.replace("actor_mean.", ""): v
            for k, v in ckpt["agent"].items()
            if k.startswith("actor_mean.")
        }
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device).eval()

        self._obs_mean = ckpt["obs_normalizer"]["mean"].to(self.device)
        self._obs_var = ckpt["obs_normalizer"]["var"].to(self.device)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using stored running mean/var."""
        return (obs - self._obs_mean) / (self._obs_var.sqrt() + 1e-8)

    def _build_obs(
        self,
        qpos: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        """Build observation vector: [joint_pos(N), ee_pose(7), target_pose(7), last_action(N)]."""
        return torch.cat(
            [
                qpos[:, : self._num_arm_joints],
                ee_pos,
                ee_quat,
                target_pos,
                target_quat,
                last_action,
            ],
            dim=-1,
        )

    def _run_policy(
        self,
        qpos: torch.Tensor,
        target_xpos: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the iterative neural policy loop and check convergence.

        Args:
            qpos: Joint positions, shape (B, dof). Modified in-place.
            target_xpos: Target poses, shape (B, 4, 4).
            target_pos: Target positions, shape (B, 3).
            target_quat: Target quaternions (xyzw), shape (B, 4).

        Returns:
            Tuple of (success [B], ik_qpos [B, dof]).
        """
        B = qpos.shape[0]
        last_action = torch.zeros(B, self._num_arm_joints, device=self.device)

        with torch.no_grad():
            for _ in range(self._max_steps):
                ee_xpos = self.get_fk(qpos)
                ee_pos = ee_xpos[:, :3, 3]
                ee_quat = convert_quat(quat_from_matrix(ee_xpos[:, :3, :3]), to="xyzw")

                obs = self._build_obs(
                    qpos, ee_pos, ee_quat, target_pos, target_quat, last_action
                )
                action = self.mlp(self._normalize_obs(obs)).clamp(-1.0, 1.0)

                qpos[:, : self._num_arm_joints] += action * self._action_scale
                qpos[:, : self._num_arm_joints] = torch.clamp(
                    qpos[:, : self._num_arm_joints],
                    self.lower_qpos_limits[: self._num_arm_joints],
                    self.upper_qpos_limits[: self._num_arm_joints],
                )
                last_action = action

        # Convergence check
        ik_xpos = self.get_fk(qpos)
        pos_err = (ik_xpos[:, :3, 3] - target_pos).norm(dim=-1)
        ik_quat_wxyz = quat_from_matrix(ik_xpos[:, :3, :3])
        target_quat_wxyz = quat_from_matrix(target_xpos[:, :3, :3])
        rot_err = quat_error_magnitude(target_quat_wxyz, ik_quat_wxyz)
        success = (pos_err < self._pos_eps) & (rot_err < self._rot_eps)

        return success, qpos

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve IK using the trained neural policy.

        Args:
            target_xpos: Target pose as 4x4 matrix, shape (4,4) or (B,4,4).
            qpos_seed: Initial joint positions, shape (dof,) or (B,dof).
            num_samples: Number of random initial seeds per target pose.
                Defaults to ``cfg.num_samples`` (1). When > 1, generates
                multiple random seeds within joint limits and returns the
                solution closest to ``qpos_seed``.
            return_all_solutions: If True, return all sampled solutions
                with shape (B, num_samples, dof) instead of the closest.

        Returns:
            Tuple of (success [B], target_joints [B,1,dof] or [B,num_samples,dof]).
        """
        return_all_solutions = kwargs.get("return_all_solutions", False)

        n = num_samples if num_samples is not None else self._num_samples

        target_xpos = torch.as_tensor(
            target_xpos, device=self.device, dtype=torch.float32
        )
        if target_xpos.dim() == 2:
            target_xpos = target_xpos.unsqueeze(0)
        B = target_xpos.shape[0]

        target_pos = target_xpos[:, :3, 3]
        target_quat = convert_quat(quat_from_matrix(target_xpos[:, :3, :3]), to="xyzw")

        if qpos_seed is None:
            qpos_seed = torch.zeros(B, self.dof, device=self.device)
        else:
            qpos_seed = torch.as_tensor(
                qpos_seed, device=self.device, dtype=torch.float32
            )
            if qpos_seed.dim() == 1:
                qpos_seed = qpos_seed.unsqueeze(0).expand(B, -1)
            qpos_seed = qpos_seed.clone()

        # Single sample: run directly without QposSeedSampler overhead.
        if n <= 1:
            success, ik_qpos = self._run_policy(
                qpos_seed, target_xpos, target_pos, target_quat
            )
            return success, ik_qpos.unsqueeze(1)

        # Multiple samples: use QposSeedSampler for random seeds.
        sampler = QposSeedSampler(num_samples=n, dof=self.dof, device=self.device)
        all_seeds = sampler.sample(
            qpos_seed, self.lower_qpos_limits, self.upper_qpos_limits, B
        )
        target_xpos_repeated = sampler.repeat_target_xpos(target_xpos, n)
        target_pos_rep = target_xpos_repeated[:, :3, 3]
        target_quat_rep = convert_quat(
            quat_from_matrix(target_xpos_repeated[:, :3, :3]), to="xyzw"
        )

        success_flat, ik_qpos_flat = self._run_policy(
            all_seeds, target_xpos_repeated, target_pos_rep, target_quat_rep
        )

        all_success = success_flat.reshape(B, n)
        all_results = ik_qpos_flat.reshape(B, n, self.dof)

        if return_all_solutions:
            return all_success.any(dim=1), all_results

        # Pick solution closest to seed.
        seed_repeat = qpos_seed.unsqueeze(1).repeat(1, n, 1)
        dist = (all_results - seed_repeat).norm(dim=-1)
        dist[~all_success] = float("inf")
        closest_idx = torch.argmin(dist, dim=1)
        closest_qpos = all_results[torch.arange(B, device=self.device), closest_idx]
        return all_success.any(dim=1), closest_qpos[:, None, :]
