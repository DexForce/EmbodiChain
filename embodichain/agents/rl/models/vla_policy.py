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

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from embodichain.agents.rl.vla_registry import create_vla_backend
from .policy import Policy

__all__ = ["VLAPolicy"]


class VLAPolicy(Policy):
    """Wraps DexForceVLA as Policy for GRPO fine-tuning."""

    def __init__(
        self,
        device: torch.device,
        policy_cfg: dict[str, Any],
        obs_space=None,
        action_space=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.policy_cfg = dict(policy_cfg)
        self.vla_cfg = dict(self.policy_cfg.get("vla", {}))
        self.model_path = str(self.vla_cfg.get("model_path", ""))
        self.action_horizon = int(self.vla_cfg.get("action_horizon", 32))
        self.gaussian_sigma = float(self.vla_cfg.get("gaussian_sigma", 0.1))

        if not self.model_path:
            raise ValueError("VLAPolicy requires 'policy.vla.model_path'.")

        self._vla_model: nn.Module | None = None
        self._action_indices: list[int] | None = None

        if action_space is None:
            self.action_dim = 14
        elif isinstance(action_space, int):
            self.action_dim = action_space
        elif hasattr(action_space, "shape") and len(action_space.shape) > 0:
            self.action_dim = int(action_space.shape[-1])
        else:
            self.action_dim = 14
        self.obs_dim = 0  # VLA uses raw ob
        self.use_raw_obs = True  # Tell collector to pass raw ob

        self.use_action_chunk = True
        self.action_chunk_size = self.action_horizon
        self.execute_full_chunk = bool(self.vla_cfg.get("execute_full_chunk", True))
        self._env = None

    def set_env(self, env) -> None:
        """Set env reference in forward."""
        self._env = env

    def _load_vla(self) -> None:
        if self._vla_model is not None:
            return
        backend = create_vla_backend(
            "dexforce_vla",
            model_path=self.model_path,
            device=self.device,
            action_horizon=self.action_horizon,
            **{
                k: v
                for k, v in self.vla_cfg.items()
                if k not in ("backend", "model_path", "action_horizon")
            },
        )
        self._vla_model, self._action_indices, self._prepare_batch_fn = backend

    def _vla_chunk_to_env_chunk(
        self, action_chunk: torch.Tensor, env=None
    ) -> torch.Tensor:
        """Convert VLA output (N, T, va_dim) chunk to env format (N, T, env_dim)."""
        if self._action_indices is not None:
            step = action_chunk[:, :, self._action_indices]
        else:
            step = action_chunk

        if env is not None:
            env_dim = getattr(env.action_space, "shape", (None,))
            if len(env_dim) > 0 and env_dim[-1] is not None:
                env_dim = int(env_dim[-1])
                if step.shape[-1] > env_dim:
                    step = step[..., :env_dim]
                elif step.shape[-1] < env_dim:
                    pad = torch.zeros(
                        step.shape[0],
                        step.shape[1],
                        env_dim - step.shape[-1],
                        device=step.device,
                        dtype=step.dtype,
                    )
                    step = torch.cat([step, pad], dim=-1)
        return step

    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        obs = tensordict["obs"]
        env = getattr(tensordict, "env", None)
        if env is None:
            env = getattr(self, "_env", None)
        if env is None:
            raise ValueError(
                "VLAPolicy needs env. Set policy._env or pass env in tensordict."
            )

        self._load_vla()
        self._vla_model.eval()
        if hasattr(obs, "batch_size") and len(obs.batch_size) > 0:
            batch_size = int(obs.batch_size[0])
        elif isinstance(obs, dict) and "robot" in obs and "qpos" in obs["robot"]:
            q = obs["robot"]["qpos"]
            batch_size = q.shape[0] if hasattr(q, "shape") and len(q.shape) > 0 else 1
        else:
            batch_size = 1
        if batch_size == 1:
            batch = self._prepare_batch_fn(obs, env)
            vla_chunk = self._vla_model.predict_action(
                batch,
                action_only=True,
                inference_horizon=self.action_horizon,
                allow_grad=False,
                use_fix_aug=False,
            )
            action_chunk_env = self._vla_chunk_to_env_chunk(vla_chunk, env=env)
        else:
            chunks_env = []
            for i in range(batch_size):
                obs_i = obs[i] if hasattr(obs, "__getitem__") else obs
                batch_i = self._prepare_batch_fn(obs_i, env)
                vla_chunk = self._vla_model.predict_action(
                    batch_i,
                    action_only=True,
                    inference_horizon=self.action_horizon,
                    allow_grad=False,
                    use_fix_aug=False,
                )
                chunk_i = self._vla_chunk_to_env_chunk(vla_chunk, env=env)
                chunks_env.append(chunk_i)
            action_chunk_env = torch.cat(chunks_env, dim=0)

        action_chunk_env = action_chunk_env.to(self.device, dtype=torch.float32)
        action = action_chunk_env[:, 0]

        tensordict["action"] = action
        tensordict["sample_log_prob"] = torch.zeros(
            action.shape[0], device=self.device, dtype=torch.float32
        )
        tensordict["value"] = torch.zeros(
            action.shape[0], device=self.device, dtype=torch.float32
        )
        if self.use_action_chunk:
            tensordict["action_chunk"] = action_chunk_env
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        b = tensordict.batch_size[0]
        tensordict["value"] = torch.zeros(b, device=self.device, dtype=torch.float32)
        return tensordict

    def evaluate_actions(
        self, tensordict: TensorDict, rollout=None, **kwargs
    ) -> TensorDict:
        """Compute log_prob via Gaussian proxy"""
        b = tensordict.batch_size[0]
        env = getattr(self, "_env", None)
        if env is None:
            raise ValueError(
                "VLAPolicy.evaluate_actions requires env. Call policy.set_env(env)."
            )

        raw_obs = getattr(rollout, "raw_obs", None)
        chunk_step = tensordict.get("chunk_step", None)
        indices = tensordict.get("_indices", None)
        if raw_obs is None or chunk_step is None or indices is None:
            raise ValueError(
                "VLAPolicy.evaluate_actions requires rollout.raw_obs, chunk_step, and _indices. "
                "Ensure collector uses use_raw_obs and use_action_chunk, and GRPO passes rollout."
            )

        time_dim = len(raw_obs) - 1
        sigma = self.gaussian_sigma
        log_probs = []
        self._load_vla()
        self._vla_model.eval()

        for i in range(b):
            idx = int(indices[i].item())
            env_idx = idx // time_dim
            step_idx = idx % time_dim
            step_in_chunk = int(chunk_step[i].item())
            # Action came from chunk predicted at chunk start
            chunk_start_idx = max(0, step_idx - step_in_chunk)
            obs_i = raw_obs[chunk_start_idx][env_idx]
            action_gt = tensordict["action"][i]

            batch_i = self._prepare_batch_fn(obs_i, env)
            vla_chunk = self._vla_model.predict_action(
                batch_i,
                action_only=True,
                inference_horizon=self.action_horizon,
                allow_grad=True,
                use_fix_aug=False,
            )
            pred_chunk_env = self._vla_chunk_to_env_chunk(vla_chunk, env=env)
            pred = pred_chunk_env[0, step_in_chunk]
            if pred.shape[-1] != action_gt.shape[-1]:
                pred = pred[: action_gt.shape[-1]]
            mse = ((action_gt - pred).pow(2)).sum(-1)
            log_prob = -0.5 * mse / (sigma * sigma + 1e-8)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs)
        entropy = (
            0.5 * self.action_dim * (1 + np.log(2 * np.pi) + 2 * np.log(sigma + 1e-8))
        )
        entropy = torch.full((b,), entropy, device=self.device, dtype=torch.float32)

        return TensorDict(
            {
                "sample_log_prob": log_probs,
                "entropy": entropy,
                "value": torch.zeros(b, device=self.device, dtype=torch.float32),
            },
            batch_size=tensordict.batch_size,
            device=self.device,
        )
