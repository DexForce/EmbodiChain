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

"""Training orchestration for truncated differentiable rollouts."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import wandb

from embodichain.learning.rl.algo import APG
from embodichain.learning.rl.collector import DifferentiableCollector
from embodichain.learning.rl.env import DifferentiableVecEnv
from embodichain.learning.rl.evaluation import evaluate_episodes
from embodichain.learning.rl.models import Policy
from embodichain.learning.rl.utils import LRSchedulerCfg, build_lr_scheduler
from embodichain.utils import configclass

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__all__ = ["DifferentiableTrainer", "DifferentiableTrainerCfg"]

_CHECKPOINT_SCHEMA_VERSION = 1


@configclass
class DifferentiableTrainerCfg:
    """Configuration for graph-preserving segmented training."""

    segment_length: int = 16
    update_horizon: int | None = None
    deterministic_actions: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    experiment_name: str = "apg"
    save_frequency_updates: int = 0
    eval_frequency_steps: int = 0
    num_eval_episodes: int = 5
    eval_seed: int | None = None
    use_wandb: bool = False
    best_eval_metric: str = "eval/avg_reward"
    best_eval_mode: str = "max"


class DifferentiableTrainer:
    """Coordinate APG updates and truncated-backpropagation boundaries."""

    def __init__(
        self,
        cfg: DifferentiableTrainerCfg,
        env: DifferentiableVecEnv,
        policy: Policy,
        algorithm: APG,
        writer: SummaryWriter | None = None,
        eval_env: DifferentiableVecEnv | None = None,
    ) -> None:
        if cfg.segment_length <= 0:
            raise ValueError("segment_length must be positive.")
        update_horizon = (
            cfg.segment_length if cfg.update_horizon is None else cfg.update_horizon
        )
        if update_horizon < cfg.segment_length:
            raise ValueError("update_horizon must be at least segment_length.")
        if update_horizon % cfg.segment_length != 0:
            raise ValueError("update_horizon must be divisible by segment_length.")
        if cfg.save_frequency_updates < 0:
            raise ValueError("save_frequency_updates cannot be negative.")
        if cfg.eval_frequency_steps < 0:
            raise ValueError("eval_frequency_steps cannot be negative.")
        if cfg.best_eval_mode not in {"min", "max"}:
            raise ValueError("best_eval_mode must be 'min' or 'max'.")
        if algorithm.policy is not policy:
            raise ValueError("Trainer and APG must reference the same policy instance.")
        if torch.device(env.device) != algorithm.device:
            raise ValueError("Environment and APG must use the same device.")

        self.cfg = cfg
        self.update_horizon = update_horizon
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.writer = writer
        self.eval_env = eval_env
        self.collector = DifferentiableCollector(
            env=env,
            policy=policy,
            device=algorithm.device,
        )
        self.global_step = 0
        self.num_updates = 0
        self.train_history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, float]] = []
        self.last_eval_metrics: dict[str, float] = {}
        self.latest_checkpoint_path: str | None = None
        self.best_checkpoint_path: str | None = None
        self.best_eval_value: float | None = None
        self.start_time = time.time()
        self.ret_window: deque[float] = deque(maxlen=100)
        self.len_window: deque[float] = deque(maxlen=100)
        self._episode_return = torch.zeros(
            env.num_envs, dtype=torch.float32, device=algorithm.device
        )
        self._episode_length = torch.zeros(
            env.num_envs, dtype=torch.long, device=algorithm.device
        )
        self._next_eval_step = (
            cfg.eval_frequency_steps if cfg.eval_frequency_steps > 0 else None
        )

    def train(self, total_timesteps: int) -> dict[str, Any]:
        """Train until at least ``total_timesteps`` vector transitions exist."""
        if total_timesteps < 0:
            raise ValueError("total_timesteps cannot be negative.")

        steps_per_update = self.update_horizon * self.env.num_envs
        if total_timesteps > 0 and steps_per_update > 0:
            total_updates = math.ceil(total_timesteps / steps_per_update)
            self.algorithm.bind_schedule(total_updates=total_updates)

        self.policy.train()
        while self.global_step < total_timesteps:
            remaining_vector_steps = math.ceil(
                (total_timesteps - self.global_step) / self.env.num_envs
            )
            update_steps = min(self.update_horizon, remaining_vector_steps)
            collected_steps = 0
            self.algorithm.begin_update()
            try:
                while collected_steps < update_steps:
                    segment_steps = min(
                        self.cfg.segment_length,
                        update_steps - collected_steps,
                    )
                    rollout = self.collector.collect(
                        segment_steps,
                        deterministic=self.cfg.deterministic_actions,
                        on_step_callback=self._on_step,
                    )
                    self.algorithm.accumulate_segment(rollout)
                    self.collector.detach_state()
                    collected_steps += rollout.num_steps
                metrics = self.algorithm.finish_update()
            except Exception:
                self.algorithm.cancel_update()
                raise

            self.global_step += collected_steps * self.env.num_envs
            self.num_updates += 1
            elapsed = max(time.time() - self.start_time, 1e-6)
            entry = {
                "global_step": float(self.global_step),
                "num_updates": float(self.num_updates),
                "charts/SPS": float(self.global_step / elapsed),
                "charts/episode_reward_avg_100": (
                    float(sum(self.ret_window) / len(self.ret_window))
                    if self.ret_window
                    else float("nan")
                ),
                "charts/episode_length_avg_100": (
                    float(sum(self.len_window) / len(self.len_window))
                    if self.len_window
                    else float("nan")
                ),
                **{f"train/{key}": value for key, value in metrics.items()},
            }
            self.train_history.append(entry)
            self._log(metrics)

            if (
                self._next_eval_step is not None
                and self.eval_env is not None
                and self.global_step >= self._next_eval_step
            ):
                self._evaluate()
                while self._next_eval_step <= self.global_step:
                    self._next_eval_step += self.cfg.eval_frequency_steps

            if (
                self.cfg.save_frequency_updates > 0
                and self.num_updates % self.cfg.save_frequency_updates == 0
            ):
                self.save_checkpoint()

        return self.get_summary()

    def save_checkpoint(self, path: str | Path | None = None) -> str:
        """Save policy, optimizer, and trainer counters."""
        if path is None:
            path = (
                Path(self.cfg.checkpoint_dir)
                / f"{self.cfg.experiment_name}_step_{self.global_step}.pt"
            )
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "policy": self.policy.state_dict(),
            "optimizer": self.algorithm.optimizer.state_dict(),
            "best_eval_value": self.best_eval_value,
        }
        if self.algorithm.lr_scheduler is not None:
            payload["lr_scheduler"] = self.algorithm.lr_scheduler.state_dict()
            payload["lr_scheduler_cfg"] = {
                "name": self.algorithm._lr_scheduler_cfg.name,
                "kwargs": dict(self.algorithm._lr_scheduler_cfg.kwargs),
            }
        torch.save(payload, checkpoint_path)
        self.latest_checkpoint_path = str(checkpoint_path)
        return self.latest_checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore policy, optimizer, and trainer counters."""
        try:
            checkpoint = torch.load(
                path,
                map_location=self.algorithm.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.algorithm.device)
        version = checkpoint.get("schema_version")
        if version != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema version {version!r}; "
                f"expected {_CHECKPOINT_SCHEMA_VERSION}."
            )
        self.policy.load_state_dict(checkpoint["policy"])
        self.algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        sched_cfg_data = checkpoint.get("lr_scheduler_cfg")
        if sched_cfg_data is not None and checkpoint.get("lr_scheduler") is not None:
            bound_cfg = LRSchedulerCfg(**sched_cfg_data)
            self.algorithm._lr_scheduler_cfg = bound_cfg
            self.algorithm.lr_scheduler = build_lr_scheduler(
                self.algorithm.optimizer,
                bound_cfg,
            )
            self.algorithm.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.global_step = int(checkpoint["global_step"])
        self.num_updates = int(checkpoint["num_updates"])
        self.best_eval_value = checkpoint.get("best_eval_value")
        self.latest_checkpoint_path = str(path)

    def get_summary(self) -> dict[str, Any]:
        """Return the current in-memory training summary."""
        elapsed = max(1e-6, time.time() - self.start_time)
        return {
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "elapsed_time_sec": float(elapsed),
            "training_fps": float(self.global_step / elapsed),
            "last_train_metrics": (
                dict(self.train_history[-1]) if self.train_history else {}
            ),
            "last_eval_metrics": dict(self.last_eval_metrics),
            "train_history": list(self.train_history),
            "eval_history": list(self.eval_history),
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
        }

    def _on_step(self, transition: Any) -> None:
        reward = transition.reward.detach()
        done = transition.done.detach()
        self._episode_return += reward
        self._episode_length += 1
        done_indices = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_indices.numel() == 0:
            return
        self.ret_window.extend(
            float(value) for value in self._episode_return[done_indices].cpu().tolist()
        )
        self.len_window.extend(
            float(value) for value in self._episode_length[done_indices].cpu().tolist()
        )
        self._episode_return[done_indices] = 0.0
        self._episode_length[done_indices] = 0

    def _evaluate(self) -> dict[str, float]:
        if self.eval_env is None:
            return {}
        metrics = evaluate_episodes(
            policy=self.policy,
            env=self.eval_env,
            num_episodes=self.cfg.num_eval_episodes,
            device=self.algorithm.device,
            seed=self.cfg.eval_seed,
        )
        entry = {"global_step": float(self.global_step), **metrics}
        self.eval_history.append(entry)
        self.last_eval_metrics = entry
        if self.writer is not None:
            for key, value in metrics.items():
                if math.isfinite(value):
                    self.writer.add_scalar(key, value, self.global_step)
        if self.cfg.use_wandb:
            wandb.log(
                {key: value for key, value in metrics.items() if math.isfinite(value)},
                step=self.global_step,
            )
        candidate = metrics.get(self.cfg.best_eval_metric)
        if candidate is not None and math.isfinite(candidate):
            improved = self.best_eval_value is None or (
                candidate > self.best_eval_value
                if self.cfg.best_eval_mode == "max"
                else candidate < self.best_eval_value
            )
            if improved:
                self.best_eval_value = candidate
                path = (
                    Path(self.cfg.checkpoint_dir)
                    / f"{self.cfg.experiment_name}_best.pt"
                )
                self.best_checkpoint_path = self.save_checkpoint(path)
        return metrics

    def _log(self, metrics: dict[str, float]) -> None:
        elapsed = max(time.time() - self.start_time, 1e-6)
        values = {
            **{f"train/{key}": value for key, value in metrics.items()},
            "charts/SPS": self.global_step / elapsed,
        }
        if self.ret_window:
            values["charts/episode_reward_avg_100"] = sum(self.ret_window) / len(
                self.ret_window
            )
            values["charts/episode_length_avg_100"] = sum(self.len_window) / len(
                self.len_window
            )
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(key, value, self.global_step)
        if self.cfg.use_wandb:
            wandb.log(values, step=self.global_step)
