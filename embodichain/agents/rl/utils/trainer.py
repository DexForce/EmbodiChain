# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from typing import Dict, Any, Tuple, Callable
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import wandb

from embodichain.lab.gym.envs.managers.event_manager import EventManager
from .helper import flatten_dict_observation


class Trainer:
    """Algorithm-agnostic trainer that coordinates training loop, logging, and evaluation."""

    def __init__(
        self,
        policy,
        env,
        algorithm,
        num_steps: int,
        batch_size: int,
        writer: SummaryWriter | None,
        eval_freq: int,
        save_freq: int,
        checkpoint_dir: str,
        exp_name: str,
        use_wandb: bool = True,
        eval_env=None,
        event_cfg=None,
        eval_event_cfg=None,
    ):
        self.policy = policy
        self.env = env
        self.eval_env = eval_env
        self.algorithm = algorithm
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.writer = writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.exp_name = exp_name
        self.use_wandb = use_wandb

        if event_cfg is not None:
            self.event_manager = EventManager(event_cfg, env=self.env)
        if eval_event_cfg is not None:
            self.eval_event_manager = EventManager(eval_event_cfg, env=self.eval_env)

        # Get device from algorithm
        self.device = self.algorithm.device
        self.global_step = 0
        self.start_time = time.time()
        self.ret_window = deque(maxlen=100)
        self.len_window = deque(maxlen=100)

        # initial obs (assume env returns torch tensors already on target device)
        obs, _ = self.env.reset()

        # Initialize algorithm's buffer
        # Flatten dict observations from ObservationManager to tensor for RL algorithms
        if isinstance(obs, dict):
            obs_tensor = flatten_dict_observation(obs)
            obs_dim = obs_tensor.shape[-1]
            num_envs = obs_tensor.shape[0]
            # Store flattened observation for RL training
            self.obs = obs_tensor

        action_space = getattr(self.env, "action_space", None)
        action_dim = action_space.shape[-1] if action_space else None
        if action_dim is None:
            raise RuntimeError(
                "Env must expose action_space with shape for buffer initialization."
            )

        # Algorithm manages its own buffer
        self.algorithm.initialize_buffer(num_steps, num_envs, obs_dim, action_dim)

        # episode stats tracked on device to avoid repeated CPU round-trips
        self.curr_ret = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.curr_len = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

    # ---- lightweight helpers for dense logging ----
    @staticmethod
    def _mean_scalar(x) -> float:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return float(np.mean(x))

    def _log_scalar_dict(self, prefix: str, data: dict):
        if not self.writer or not isinstance(data, dict):
            return
        for k, v in data.items():
            try:
                self.writer.add_scalar(
                    f"{prefix}/{k}", self._mean_scalar(v), self.global_step
                )
            except Exception:
                continue

    def _pack_log_dict(self, prefix: str, data: dict) -> dict:
        if not isinstance(data, dict):
            return {}
        out = {}
        for k, v in data.items():
            try:
                out[f"{prefix}/{k}"] = self._mean_scalar(v)
            except Exception:
                continue
        return out

    def train(self, total_timesteps: int):
        print(f"Start training, total steps: {total_timesteps}")
        while self.global_step < total_timesteps:
            self._collect_rollout()
            losses = self.algorithm.update()
            self._log_train(losses)
            if self.global_step % self.eval_freq == 0:
                self._eval_once()
            if self.global_step % self.save_freq == 0:
                self.save_checkpoint()

    @torch.no_grad()
    def _collect_rollout(self):
        """Collect a rollout. Algorithm controls the data collection process."""

        # Callback function for statistics and logging
        def on_step(obs, actions, reward, done, info, next_obs):
            """Callback called at each step during rollout collection."""
            # Episode stats (stay on device; convert only when episode ends)
            self.curr_ret += reward
            self.curr_len += 1
            done_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                finished_ret = self.curr_ret[done_idx].detach().cpu().tolist()
                finished_len = self.curr_len[done_idx].detach().cpu().tolist()
                self.ret_window.extend(finished_ret)
                self.len_window.extend(finished_len)
                self.curr_ret[done_idx] = 0
                self.curr_len[done_idx] = 0

            # Update global step and observation
            # next_obs is already flattened in algorithm's collect_rollout
            self.obs = next_obs
            self.global_step += next_obs.shape[0]

            if isinstance(info, dict):
                rewards_dict = info.get("rewards")
                metrics_dict = info.get("metrics")
                self._log_scalar_dict("rewards", rewards_dict)
                self._log_scalar_dict("metrics", metrics_dict)
                log_dict = {}
                log_dict.update(self._pack_log_dict("rewards", rewards_dict))
                log_dict.update(self._pack_log_dict("metrics", metrics_dict))
                if log_dict and self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)

        # Algorithm controls data collection
        result = self.algorithm.collect_rollout(
            env=self.env,
            policy=self.policy,
            obs=self.obs,
            num_steps=self.num_steps,
            on_step_callback=on_step,
        )

    def _log_train(self, losses: Dict[str, float]):
        if self.writer:
            for k, v in losses.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)
            elapsed = max(1e-6, time.time() - self.start_time)
            sps = self.global_step / elapsed
            self.writer.add_scalar("charts/SPS", sps, self.global_step)
            if len(self.ret_window) > 0:
                self.writer.add_scalar(
                    "charts/episode_reward_avg_100",
                    float(np.mean(self.ret_window)),
                    self.global_step,
                )
            if len(self.len_window) > 0:
                self.writer.add_scalar(
                    "charts/episode_length_avg_100",
                    float(np.mean(self.len_window)),
                    self.global_step,
                )
        # console
        sps = self.global_step / max(1e-6, time.time() - self.start_time)
        avgR = np.mean(self.ret_window) if len(self.ret_window) > 0 else float("nan")
        avgL = np.mean(self.len_window) if len(self.len_window) > 0 else float("nan")
        print(
            f"[train] step={self.global_step} sps={sps:.0f} avgReward(100)={avgR:.3f} avgLength(100)={avgL:.1f}"
        )

        # wandb (mirror TB logs)
        if self.use_wandb:
            log_dict = {f"train/{k}": v for k, v in losses.items()}
            log_dict["charts/SPS"] = sps
            if not np.isnan(avgR):
                log_dict["charts/episode_reward_avg_100"] = float(avgR)
            if not np.isnan(avgL):
                log_dict["charts/episode_length_avg_100"] = float(avgL)
            wandb.log(log_dict, step=self.global_step)

    @torch.no_grad()
    def _eval_once(self, num_episodes: int = 5):
        self.policy.eval()
        returns = []
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            obs = flatten_dict_observation(obs)

            done_any = torch.zeros(
                obs.shape[0] if obs.ndim == 2 else 1,
                dtype=torch.bool,
                device=self.device,
            )
            num_envs_eval = obs.shape[0] if obs.ndim == 2 else 1
            ep_ret = torch.zeros(num_envs_eval, dtype=torch.float32, device=self.device)
            while not done_any.any():
                actions, _, _ = self.policy.get_action(obs, deterministic=True)

                # Wrap action as dict for env processing
                action_type = getattr(self.eval_env, "action_type", "delta_qpos")
                action_dict = {action_type: actions}

                obs, reward, terminated, truncated, info = self.eval_env.step(
                    action_dict
                )

                # Flatten dict observation from step
                if isinstance(obs, dict):
                    obs = flatten_dict_observation(obs)

                done = terminated | truncated
                reward = reward.float()
                done_any = done
                ep_ret += reward

                if hasattr(self, "eval_event_manager"):
                    if "interval" in self.eval_event_manager.available_modes:
                        self.eval_event_manager.apply(mode="interval")

            returns.extend(ep_ret.detach().cpu().tolist())
        if self.writer and len(returns) > 0:
            self.writer.add_scalar(
                "eval/avg_reward", float(np.mean(returns)), self.global_step
            )

    def save_checkpoint(self):
        # minimal model-only checkpoint; trainer/algorithm states can be added
        path = f"{self.checkpoint_dir}/{self.exp_name}_step_{self.global_step}.pt"
        torch.save(
            {
                "global_step": self.global_step,
                "policy": self.policy.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
