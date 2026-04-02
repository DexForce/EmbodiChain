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

from typing import Any, Dict
import time
import numpy as np
import torch
import wandb
from embodichain.utils import logger
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tensordict import TensorDict
from typing import Dict

from embodichain.lab.gym.envs.managers.action_manager import ActionManager
from embodichain.agents.rl.buffer import RolloutBuffer
from embodichain.agents.rl.collector import SyncCollector
from embodichain.lab.gym.envs.managers.event_manager import EventManager
from .helper import flatten_dict_observation


class Trainer:
    """Algorithm-agnostic trainer that coordinates training loop, logging, and evaluation."""

    def __init__(
        self,
        policy,
        env,
        algorithm,
        buffer_size: int,
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
        num_eval_episodes: int = 5,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.policy = policy
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.env = env
        self.eval_env = eval_env
        self.algorithm = algorithm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.writer = writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.exp_name = exp_name
        self.use_wandb = use_wandb
        self.num_eval_episodes = num_eval_episodes

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
        self.train_history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, float]] = []
        self.last_eval_metrics: dict[str, float] = {}
        self.last_train_metrics: dict[str, float] = {}
        self.latest_checkpoint_path: str | None = None
        num_envs = getattr(self.env, "num_envs", None)
        if num_envs is None:
            raise RuntimeError("Env must expose num_envs for trainer statistics.")
        obs_dim = getattr(self.policy, "obs_dim", None)
        action_dim = getattr(self.policy, "action_dim", None)
        if obs_dim is None or action_dim is None:
            raise RuntimeError("Policy must expose obs_dim and action_dim.")

        self.buffer = RolloutBuffer(
            num_envs=num_envs,
            rollout_len=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device,
        )
        self.collector = SyncCollector(
            env=self.env,
            policy=self.policy,
            device=self.device,
            reset_every_rollout=bool(
                getattr(
                    getattr(self.algorithm, "cfg", None), "reset_every_rollout", False
                )
            ),
        )

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

    def train(self, total_timesteps: int) -> Dict[str, Any]:
        if self.rank == 0:
            print(f"Start training, total steps: {total_timesteps}")
        while self.global_step < total_timesteps:
            self._collect_rollout()
            losses = self.algorithm.update(self.buffer.get(flatten=False))
            self._log_train(losses)
            if (
                self.eval_freq > 0
                and self.eval_env is not None
                and self.global_step % self.eval_freq == 0
            ):
                self._eval_once(num_episodes=self.num_eval_episodes)
            if self.global_step % self.save_freq == 0:
                self.save_checkpoint()
        return self.get_summary()

    @torch.no_grad()
    def _collect_rollout(self):
        """Collect a rollout with the synchronous collector."""

        # Callback function for statistics and logging
        def on_step(tensordict: TensorDict, info: dict):
            """Callback called at each step during rollout collection."""
            reward = tensordict["reward"]
            done = tensordict["done"]
            # Episode stats
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

            if not self.distributed:
                self.global_step += tensordict.batch_size[0]

            if self.rank == 0 and isinstance(info, dict):
                rewards_dict = info.get("rewards")
                metrics_dict = info.get("metrics")
                self._log_scalar_dict("rewards", rewards_dict)
                self._log_scalar_dict("metrics", metrics_dict)
                log_dict = {}
                log_dict.update(self._pack_log_dict("rewards", rewards_dict))
                log_dict.update(self._pack_log_dict("metrics", metrics_dict))
                if log_dict and self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)

        rollout = self.buffer.start_rollout()
        rollout = self.collector.collect(
            num_steps=self.buffer_size,
            rollout=rollout,
            on_step_callback=on_step,
        )
        self.buffer.add(rollout)

        # Sync global_step and episode stats across ranks in distributed mode
        if self.distributed:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Distributed training was requested (distributed=True), "
                    "but torch.distributed is not available. "
                    "Please ensure PyTorch was built with distributed support or "
                    "set distributed=False."
                )
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "Distributed training was requested (distributed=True), "
                    "but the torch.distributed process group is not initialized. "
                    "Call torch.distributed.init_process_group(...) before creating "
                    "or using Trainer(distributed=True, ...)."
                )
            local_delta = self.env.num_envs * self.buffer_size
            delta_tensor = torch.tensor(
                [local_delta], dtype=torch.int64, device=self.device
            )
            torch.distributed.all_reduce(
                delta_tensor, op=torch.distributed.ReduceOp.SUM
            )
            self.global_step += int(delta_tensor.item())
            self._sync_episode_stats()

    def _sync_episode_stats(self) -> None:
        """Sync ret_window and len_window across ranks; rank 0 gets merged stats."""
        if not self.distributed or not torch.distributed.is_initialized():
            return
        maxlen = 100
        ret_list = list(self.ret_window)
        len_list = list(self.len_window)
        n = min(len(ret_list), maxlen)
        if n > 0:
            ret_list = ret_list[-n:]
            len_list = len_list[-n:]

        ret_tensor = torch.zeros(maxlen, dtype=torch.float32, device=self.device)
        len_tensor = torch.zeros(maxlen, dtype=torch.float32, device=self.device)
        if n > 0:
            ret_tensor[:n] = torch.tensor(
                ret_list, dtype=torch.float32, device=self.device
            )
            len_tensor[:n] = torch.tensor(
                len_list, dtype=torch.float32, device=self.device
            )
        count_tensor = torch.tensor([n], dtype=torch.int64, device=self.device)

        ret_list_all = [torch.zeros_like(ret_tensor) for _ in range(self.world_size)]
        len_list_all = [torch.zeros_like(len_tensor) for _ in range(self.world_size)]
        count_list_all = [
            torch.zeros_like(count_tensor) for _ in range(self.world_size)
        ]
        torch.distributed.all_gather(ret_list_all, ret_tensor)
        torch.distributed.all_gather(len_list_all, len_tensor)
        torch.distributed.all_gather(count_list_all, count_tensor)

        if self.rank == 0:
            all_ret = []
            all_len = []
            for r in range(self.world_size):
                c = int(count_list_all[r].item())
                if c > 0:
                    all_ret.extend(ret_list_all[r][:c].cpu().tolist())
                    all_len.extend(len_list_all[r][:c].cpu().tolist())
            self.ret_window.clear()
            self.len_window.clear()
            n_total = len(all_ret)
            start = max(0, n_total - maxlen)
            self.ret_window.extend(all_ret[start:])
            self.len_window.extend(all_len[start:])

    def _log_train(self, losses: Dict[str, float]):
        elapsed = max(1e-6, time.time() - self.start_time)
        sps = self.global_step / elapsed
        avgR = np.mean(self.ret_window) if len(self.ret_window) > 0 else float("nan")
        avgL = np.mean(self.len_window) if len(self.len_window) > 0 else float("nan")
        history_entry = {
            "global_step": float(self.global_step),
            "charts/SPS": float(sps),
            "charts/episode_reward_avg_100": float(avgR),
            "charts/episode_length_avg_100": float(avgL),
        }
        history_entry.update({f"train/{k}": float(v) for k, v in losses.items()})
        self.train_history.append(history_entry)
        self.last_train_metrics = history_entry

        if self.writer:
            for k, v in losses.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)
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
        # console and external logging are rank-0 only in distributed mode.
        if self.rank == 0:
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
    def _eval_once(self, num_episodes: int = 5) -> Dict[str, float]:
        """Run evaluation for specified number of episodes.

        Each episode runs all parallel environments until completion, allowing
        environments to finish at different times.

        Args:
            num_episodes: Number of episodes to evaluate
        """
        self.policy.eval()
        episode_returns = []
        episode_lengths = []
        episode_successes = []
        metric_values: dict[str, list[float]] = {}

        # Evaluation does not consume the training rollout buffer; binding it here can
        # overflow the shared RL buffer when eval episodes are longer than buffer_size.
        for _ in range(num_episodes):
            # Reset and initialize episode tracking
            obs, _ = self.eval_env.reset()
            obs = flatten_dict_observation(obs)
            num_envs = obs.shape[0] if obs.ndim == 2 else 1

            done_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            cumulative_reward = torch.zeros(
                num_envs, dtype=torch.float32, device=self.device
            )
            step_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

            # Run episode until all environments complete
            while not done_mask.all():
                # Get deterministic actions from policy
                action_td = TensorDict(
                    {"obs": obs},
                    batch_size=[num_envs],
                    device=self.device,
                )
                action_td = self.policy.get_action(action_td, deterministic=True)
                actions = action_td["action"]
                am: ActionManager = getattr(self.eval_env, "action_manager", None)
                if am is None:
                    action_in = actions
                else:
                    action_in = am.convert_policy_action_to_env_action(actions)

                # Environment step
                obs, reward, terminated, truncated, info = self.eval_env.step(action_in)
                obs = (
                    flatten_dict_observation(obs)
                    if isinstance(obs, TensorDict)
                    else obs
                )

                # Update statistics only for still-running environments
                done = terminated | truncated
                still_running = ~done_mask
                cumulative_reward[still_running] += reward[still_running].float()
                step_count[still_running] += 1
                newly_done = done & (~done_mask)
                if newly_done.any():
                    if isinstance(info, dict) and "success" in info:
                        successes = info["success"][newly_done].detach().cpu().tolist()
                        episode_successes.extend([float(v) for v in successes])
                    if isinstance(info, dict) and "metrics" in info:
                        for key, value in info["metrics"].items():
                            values = value[newly_done].detach().cpu().tolist()
                            metric_values.setdefault(key, []).extend(
                                [float(v) for v in values]
                            )
                done_mask |= done

                # Trigger evaluation events (e.g., video recording)
                if hasattr(self, "eval_event_manager"):
                    if "interval" in self.eval_event_manager.available_modes:
                        self.eval_event_manager.apply(mode="interval")

            # Collect episode statistics
            episode_returns.extend(cumulative_reward.cpu().tolist())
            episode_lengths.extend(step_count.cpu().tolist())

        # Finalize evaluation functors (e.g., video recording)
        if hasattr(self, "eval_event_manager"):
            for functor_cfg in self.eval_event_manager._mode_functor_cfgs.get(
                "interval", []
            ):
                functor = functor_cfg.func
                save_path = functor_cfg.params.get("save_path", "./outputs/videos/eval")

                if hasattr(functor, "flush"):
                    functor.flush(save_path)
                if hasattr(functor, "finalize"):
                    functor.finalize(save_path)

        # Log evaluation metrics
        if self.writer and episode_returns:
            self.writer.add_scalar(
                "eval/avg_reward", float(np.mean(episode_returns)), self.global_step
            )
            self.writer.add_scalar(
                "eval/avg_length", float(np.mean(episode_lengths)), self.global_step
            )
            if episode_successes:
                self.writer.add_scalar(
                    "eval/success_rate",
                    float(np.mean(episode_successes)),
                    self.global_step,
                )

        summary = {
            "global_step": float(self.global_step),
            "eval/avg_reward": float(np.mean(episode_returns))
            if episode_returns
            else float("nan"),
            "eval/avg_length": float(np.mean(episode_lengths))
            if episode_lengths
            else float("nan"),
            "eval/success_rate": float(np.mean(episode_successes))
            if episode_successes
            else float("nan"),
        }
        for key, values in metric_values.items():
            if values:
                summary[f"eval/metrics/{key}"] = float(np.mean(values))
        self.eval_history.append(summary)
        self.last_eval_metrics = summary
        if self.rank == 0 and self.use_wandb:
            log_dict = {
                key: value
                for key, value in summary.items()
                if key != "global_step" and not np.isnan(value)
            }
            if log_dict:
                wandb.log(log_dict, step=self.global_step)
        return summary

    def save_checkpoint(self) -> str | None:
        # minimal model-only checkpoint; trainer/algorithm states can be added
        if self.rank != 0:
            return None
        path = f"{self.checkpoint_dir}/{self.exp_name}_step_{self.global_step}.pt"
        policy_state = (
            self.policy.module.state_dict()
            if hasattr(self.policy, "module")
            else self.policy.state_dict()
        )
        torch.save(
            {
                "global_step": self.global_step,
                "policy": policy_state,
            },
            path,
        )
        self.latest_checkpoint_path = path
        print(f"Checkpoint saved: {path}")
        return path

    def get_summary(self) -> Dict[str, Any]:
        elapsed = max(1e-6, time.time() - self.start_time)
        return {
            "global_step": int(self.global_step),
            "elapsed_time_sec": float(elapsed),
            "training_fps": float(self.global_step / elapsed),
            "last_train_metrics": dict(self.last_train_metrics),
            "last_eval_metrics": dict(self.last_eval_metrics),
            "train_history": list(self.train_history),
            "eval_history": list(self.eval_history),
            "latest_checkpoint_path": self.latest_checkpoint_path,
        }
