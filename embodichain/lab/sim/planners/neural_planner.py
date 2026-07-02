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

from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from embodichain.lab.sim.planners.base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    _infer_batch_size,
    validate_plan_options,
)
from embodichain.lab.sim.planners.utils import MoveType, PlanResult, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import convert_quat, quat_error_magnitude, quat_from_matrix

__all__ = [
    "NeuralPlanner",
    "NeuralPlannerCfg",
    "NeuralPlanOptions",
]


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class _RunningObsNormalizer:
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        self.mean = mean
        self.var = var

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


class _WaypointTransformerActor(nn.Module):
    """APG waypoint actor runtime copied in lightweight form for inference."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_waypoints: int,
        use_relative_obs: bool = True,
        hidden_dim: int = 256,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_ff_dim: int | None = None,
    ):
        super().__init__()
        self.num_waypoints = int(num_waypoints)
        self.use_relative_obs = bool(use_relative_obs)
        if int(action_dim) != 7:
            raise ValueError(
                "Waypoint transformer checkpoints currently assume a 7-DoF arm. "
                f"Got action_dim={action_dim}."
            )
        self.state_dim = 7 + 7 + 7 + (7 if self.use_relative_obs else 0)
        self.waypoint_token_dim = 3 + 4 + 1 + 1

        expected_obs_dim = 7 + 7 + self.num_waypoints * 3
        expected_obs_dim += self.num_waypoints * 4 + self.num_waypoints * 2 + 7
        if self.use_relative_obs:
            expected_obs_dim += 7
        if int(obs_dim) != expected_obs_dim:
            raise ValueError(
                "Waypoint transformer expected obs_dim "
                f"{expected_obs_dim}, got {obs_dim}."
            )

        if hidden_dim % transformer_nhead != 0:
            raise ValueError("hidden_dim must be divisible by transformer_nhead")

        ff_dim = transformer_ff_dim or hidden_dim * 4
        self.state_proj = _layer_init(nn.Linear(self.state_dim, hidden_dim))
        self.waypoint_proj = _layer_init(nn.Linear(self.waypoint_token_dim, hidden_dim))
        self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.waypoint_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.waypoint_index_embedding = nn.Parameter(
            torch.zeros(1, self.num_waypoints, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers
        )
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def _parse_obs(self, x: torch.Tensor):
        n = self.num_waypoints
        cursor = 0
        joint = x[:, cursor : cursor + 7]
        cursor += 7
        eef_pose = x[:, cursor : cursor + 7]
        cursor += 7
        waypoint_pos = x[:, cursor : cursor + 3 * n].reshape(-1, n, 3)
        cursor += 3 * n
        waypoint_quat = x[:, cursor : cursor + 4 * n].reshape(-1, n, 4)
        cursor += 4 * n
        active_onehot = x[:, cursor : cursor + n].reshape(-1, n, 1)
        cursor += n
        valid_mask = x[:, cursor : cursor + n].reshape(-1, n, 1)
        cursor += n
        last_action = x[:, cursor : cursor + 7]
        cursor += 7

        state_parts = [joint, eef_pose, last_action]
        if self.use_relative_obs:
            state_parts.append(x[:, cursor : cursor + 7])

        state = torch.cat(state_parts, dim=-1)
        waypoint_tokens = torch.cat(
            [waypoint_pos, waypoint_quat, active_onehot, valid_mask], dim=-1
        )
        return state, waypoint_tokens, valid_mask.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state, waypoint_tokens, valid_mask = self._parse_obs(x)
        state_token = self.state_proj(state).unsqueeze(1) + self.state_type_embedding
        waypoint_tokens = (
            self.waypoint_proj(waypoint_tokens)
            + self.waypoint_type_embedding
            + self.waypoint_index_embedding
        )
        tokens = torch.cat([state_token, waypoint_tokens], dim=1)
        state_padding = torch.zeros(
            valid_mask.shape[0], 1, dtype=torch.bool, device=valid_mask.device
        )
        waypoint_padding = valid_mask < 0.5
        padding_mask = torch.cat([state_padding, waypoint_padding], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.actor_head(encoded[:, 0])


@configclass
class NeuralPlannerCfg(BasePlannerCfg):
    planner_type: str = "neural"

    checkpoint_path: str = MISSING
    """Path to an APG waypoint checkpoint (.pt), e.g. from ``download_neural_planner_checkpoint()``."""

    control_part: str | None = None
    """Robot control part used for FK and qpos, e.g. 'left_arm'."""

    max_steps: int | None = None
    """Maximum rollout steps. If None, uses checkpoint max_episode_steps."""

    action_scale: float = 0.2
    """Delta joint scaling factor in radians."""

    num_arm_joints: int = 7
    """Number of arm joints controlled by the APG policy."""

    pos_eps: float | None = None
    """Waypoint position threshold. If None, uses checkpoint waypoint_pos_threshold."""

    rot_eps: float | None = None
    """Waypoint rotation threshold. If None, uses checkpoint waypoint_rot_threshold."""

    dt: float = 0.01
    """Nominal timestep reported in PlanResult."""


@configclass
class NeuralPlanOptions(PlanOptions):
    control_part: str | None = None
    start_qpos: torch.Tensor | None = None
    max_steps: int | None = None


class NeuralPlanner(BasePlanner):
    def __init__(self, cfg: NeuralPlannerCfg):
        super().__init__(cfg)

        self.cfg: NeuralPlannerCfg = cfg
        if cfg.checkpoint_path is MISSING or not str(cfg.checkpoint_path):
            logger.log_error("checkpoint_path is required", ValueError)
        self._load_checkpoint(Path(cfg.checkpoint_path))

    def default_plan_options(self) -> NeuralPlanOptions:
        return NeuralPlanOptions()

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            logger.log_error(
                f"Checkpoint not found: {checkpoint_path}", FileNotFoundError
            )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "agent" not in ckpt:
            raise KeyError(
                f"Checkpoint at '{checkpoint_path}' is missing 'agent'. "
                f"Available keys: {list(ckpt.keys())}."
            )
        if "obs_normalizer" not in ckpt:
            raise KeyError(
                f"Checkpoint at '{checkpoint_path}' is missing 'obs_normalizer'. "
                f"Available keys: {list(ckpt.keys())}."
            )
        for subkey in ("mean", "var"):
            if subkey not in ckpt["obs_normalizer"]:
                raise KeyError(
                    f"Checkpoint obs_normalizer is missing '{subkey}'. "
                    f"Available: {list(ckpt['obs_normalizer'].keys())}."
                )

        self._ckpt_args = ckpt.get("args", {})
        self._num_waypoints = int(self._ckpt_args.get("waypoint_max", 1))
        self._use_relative_obs = bool(
            self._ckpt_args.get("waypoint_use_relative_obs", True)
        )
        self._policy_arch = self._ckpt_args.get("policy_arch")
        if self._policy_arch != "transformer":
            raise ValueError(
                "NeuralPlanner only supports transformer waypoint checkpoints. "
                f"Got policy_arch={self._policy_arch!r}."
            )
        self._hidden_dim = int(self._ckpt_args.get("hidden_dim", 256))
        self._action_dim = int(self.cfg.num_arm_joints)
        self._obs_dim = int(ckpt["obs_normalizer"]["mean"].numel())
        self._max_steps = int(
            self.cfg.max_steps or self._ckpt_args.get("max_episode_steps", 30)
        )
        self._pos_eps = float(
            self.cfg.pos_eps
            if self.cfg.pos_eps is not None
            else self._ckpt_args.get("waypoint_pos_threshold", 0.05)
        )
        self._rot_eps = float(
            self.cfg.rot_eps
            if self.cfg.rot_eps is not None
            else self._ckpt_args.get("waypoint_rot_threshold", 0.3)
        )
        self._intermediate_orientation = bool(
            self._ckpt_args.get("waypoint_intermediate_orientation", True)
        )

        self._normalizer = _RunningObsNormalizer(
            ckpt["obs_normalizer"]["mean"].to(self.device),
            ckpt["obs_normalizer"]["var"].to(self.device),
        )
        self._actor = self._build_actor().to(self.device)

        state_dict = {
            k.replace("actor_mean.", ""): v.to(self.device)
            for k, v in ckpt["agent"].items()
            if k.startswith("actor_mean.")
        }
        if not state_dict:
            raise KeyError("Checkpoint agent has no actor_mean.* weights.")
        self._actor.load_state_dict(state_dict)
        self._actor.eval()

    def _build_actor(self) -> nn.Module:
        return _WaypointTransformerActor(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            num_waypoints=self._num_waypoints,
            use_relative_obs=self._use_relative_obs,
            hidden_dim=self._hidden_dim,
            transformer_nhead=int(self._ckpt_args.get("transformer_nhead", 4)),
            transformer_num_layers=int(
                self._ckpt_args.get("transformer_num_layers", 2)
            ),
            transformer_ff_dim=(
                int(self._ckpt_args.get("transformer_ff_dim", 0)) or None
            ),
        )

    @validate_plan_options(options_cls=NeuralPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: NeuralPlanOptions = NeuralPlanOptions(),
    ) -> PlanResult:
        if not target_states:
            return PlanResult(
                success=torch.zeros(0, dtype=torch.bool, device=self.device),
                positions=None,
            )

        control_part = options.control_part or self.cfg.control_part
        if control_part is None:
            logger.log_error(
                "control_part is required for NeuralPlanner",
                ValueError,
            )

        waypoints_pos, waypoints_quat, valid_mask, episode_k = self._parse_waypoints(
            target_states
        )
        qpos = self._initial_qpos(control_part, options.start_qpos)
        b = qpos.shape[0]
        limits = self.robot.get_qpos_limits(name=control_part)[0].to(self.device)
        lower = limits[: self._action_dim, 0]
        upper = limits[: self._action_dim, 1]

        last_action = torch.zeros(b, self._action_dim, device=self.device)
        active_idx = torch.zeros(b, dtype=torch.long, device=self.device)
        positions = [qpos.clone()]
        xpos_list = [self._fk_matrix(qpos, control_part)]
        converged = torch.zeros(b, dtype=torch.bool, device=self.device)
        max_steps = int(options.max_steps or self._max_steps)

        with torch.no_grad():
            for _ in range(max_steps):
                ee_pose = self._fk_pose_xyzw(qpos, control_part)
                obs = self._build_obs(
                    qpos[:, : self._action_dim],
                    ee_pose,
                    waypoints_pos,
                    waypoints_quat,
                    valid_mask,
                    active_idx,
                    last_action,
                )
                action = self._actor(self._normalizer.normalize(obs)).clamp(-1.0, 1.0)
                qpos[:, : self._action_dim] += action * float(self.cfg.action_scale)
                qpos[:, : self._action_dim] = torch.clamp(
                    qpos[:, : self._action_dim], lower, upper
                )
                last_action = action
                positions.append(qpos.clone())
                xpos_list.append(self._fk_matrix(qpos, control_part))

                ee_pose = self._fk_pose_xyzw(qpos, control_part)
                reached = self._is_active_reached(
                    ee_pose, waypoints_pos, waypoints_quat, active_idx, episode_k
                )
                active_idx = torch.where(reached, active_idx + 1, active_idx)
                converged = converged | (active_idx >= episode_k)
                if converged.all():
                    break

        positions_t = torch.stack(positions)
        xpos_t = torch.stack(xpos_list)
        dt = torch.full(
            (positions_t.shape[0],),
            float(self.cfg.dt),
            dtype=torch.float32,
            device=self.device,
        )
        dt = dt.unsqueeze(0).expand(b, -1)
        positions_t = positions_t.permute(1, 0, 2)
        xpos_t = xpos_t.permute(1, 0, 2, 3)
        success = active_idx >= episode_k
        return PlanResult(
            success=success,
            positions=positions_t,
            xpos_list=xpos_t,
            dt=dt,
            duration=torch.full(
                (b,),
                float(max(positions_t.shape[1] - 1, 0) * self.cfg.dt),
                device=self.device,
            ),
        )

    def _parse_waypoints(
        self, target_states: list[PlanState]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(target_states) > self._num_waypoints:
            logger.log_error(
                f"Received {len(target_states)} waypoints, but checkpoint supports "
                f"at most {self._num_waypoints}.",
                ValueError,
            )
        b = _infer_batch_size(target_states) or 1
        waypoint_pos = torch.zeros(b, self._num_waypoints, 3, device=self.device)
        waypoint_quat = torch.zeros(b, self._num_waypoints, 4, device=self.device)
        valid_mask = torch.zeros(b, self._num_waypoints, device=self.device)
        for idx, target in enumerate(target_states):
            if target.move_type != MoveType.EEF_MOVE or target.xpos is None:
                logger.log_error(
                    "NeuralPlanner expects EEF_MOVE PlanState entries with xpos.",
                    ValueError,
                )
            xpos = torch.as_tensor(target.xpos, dtype=torch.float32, device=self.device)
            if xpos.dim() == 2:
                xpos = xpos.unsqueeze(0)
            waypoint_pos[:, idx] = xpos[:, :3, 3]
            waypoint_quat[:, idx] = convert_quat(
                quat_from_matrix(xpos[:, :3, :3]), to="xyzw"
            )
            valid_mask[:, idx] = 1.0
        return waypoint_pos, waypoint_quat, valid_mask, len(target_states)

    def _initial_qpos(
        self, control_part: str, start_qpos: torch.Tensor | None
    ) -> torch.Tensor:
        if start_qpos is None:
            qpos = self.robot.get_qpos(name=control_part)
        else:
            qpos = torch.as_tensor(start_qpos, dtype=torch.float32, device=self.device)
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[-1] < self._action_dim:
            logger.log_error(
                f"start_qpos has {qpos.shape[-1]} joints, but policy expects "
                f"{self._action_dim}.",
                ValueError,
            )
        return qpos.to(self.device).clone()

    def _fk_matrix(self, qpos: torch.Tensor, control_part: str) -> torch.Tensor:
        return self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=True)

    def _fk_pose_xyzw(self, qpos: torch.Tensor, control_part: str) -> torch.Tensor:
        fk = self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=False)
        pos = fk[:, :3]
        quat_xyzw = convert_quat(fk[:, 3:7], to="xyzw")
        return torch.cat([pos, quat_xyzw], dim=-1)

    def _build_obs(
        self,
        joint_pos: torch.Tensor,
        ee_pose: torch.Tensor,
        waypoint_pos: torch.Tensor,
        waypoint_quat: torch.Tensor,
        valid_mask: torch.Tensor,
        active_idx: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        b = joint_pos.shape[0]
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_onehot = torch.zeros(b, self._num_waypoints, device=self.device)
        active_onehot.scatter_(1, active_idx_clamped.unsqueeze(1), 1.0)
        obs_parts = [
            joint_pos,
            ee_pose,
            waypoint_pos.reshape(b, self._num_waypoints * 3),
            waypoint_quat.reshape(b, self._num_waypoints * 4),
            active_onehot,
            valid_mask,
            last_action,
        ]
        if self._use_relative_obs:
            idx = torch.arange(b, device=self.device)
            active_pos = waypoint_pos[idx, active_idx_clamped]
            active_quat = waypoint_quat[idx, active_idx_clamped]
            obs_parts.append(
                torch.cat([active_pos - ee_pose[:, :3], active_quat], dim=-1)
            )
        obs = torch.cat(obs_parts, dim=-1)
        if obs.shape[-1] != self._obs_dim:
            raise ValueError(
                f"Built obs dim {obs.shape[-1]}, expected {self._obs_dim}."
            )
        return obs

    def _is_active_reached(
        self,
        ee_pose: torch.Tensor,
        waypoint_pos: torch.Tensor,
        waypoint_quat: torch.Tensor,
        active_idx: torch.Tensor,
        episode_k: int,
    ) -> torch.Tensor:
        b = ee_pose.shape[0]
        idx = torch.arange(b, device=self.device)
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_pos = waypoint_pos[idx, active_idx_clamped]
        active_quat_xyzw = waypoint_quat[idx, active_idx_clamped]
        pos_dist = (ee_pose[:, :3] - active_pos).norm(dim=-1)
        ee_quat_wxyz = convert_quat(ee_pose[:, 3:7], to="wxyz")
        active_quat_wxyz = convert_quat(active_quat_xyzw, to="wxyz")
        rot_dist = quat_error_magnitude(ee_quat_wxyz, active_quat_wxyz)
        orientation_required = self._intermediate_orientation | (
            active_idx >= episode_k - 1
        )
        rot_ok = torch.where(
            orientation_required,
            rot_dist < self._rot_eps,
            torch.ones_like(rot_dist, dtype=torch.bool),
        )
        reached = (pos_dist < self._pos_eps) & rot_ok
        return reached
