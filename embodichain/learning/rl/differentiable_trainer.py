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
from embodichain.learning.rl.collector import (
    DifferentiableCollector,
    DifferentiableRollout,
)
from embodichain.learning.rl.env import (
    DifferentiableRolloutSpec,
    DifferentiableVecEnv,
    ScheduledDifferentiableVecEnv,
)
from embodichain.learning.rl.evaluation import evaluate_episodes
from embodichain.learning.rl.policies import Policy
from embodichain.learning.rl.normalization import RunningObservationNormalizer
from embodichain.learning.rl.checkpoint import _load, _save
from embodichain.learning.rl.utils import scheduler_needs_horizon
from embodichain.utils import configclass

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__all__ = ["DifferentiableTrainer", "DifferentiableTrainerCfg"]


@configclass
class DifferentiableTrainerCfg:
    """Configuration for segmented or complete graph-preserving training."""

    segment_length: int = 16
    update_horizon: int | None = None
    rollout_mode: str = "segmented"
    gradient_accumulation_steps: int = 1
    deterministic_actions: bool = False
    clip_actions_to_space: bool = False
    action_adjoint_max_norm: float = 0.0
    normalize_observations: bool = False
    rollout_seed: int | None = None
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
    """Coordinate APG updates, full rollouts, and optional TBPTT boundaries."""

    def __init__(
        self,
        cfg: DifferentiableTrainerCfg,
        env: DifferentiableVecEnv,
        policy: Policy,
        algorithm: APG,
        writer: SummaryWriter | None = None,
        eval_env: DifferentiableVecEnv | None = None,
        observation_normalizer: RunningObservationNormalizer | None = None,
    ) -> None:
        if cfg.segment_length <= 0:
            raise ValueError("segment_length must be positive.")
        update_horizon = (
            cfg.segment_length if cfg.update_horizon is None else cfg.update_horizon
        )
        if cfg.rollout_mode not in {"segmented", "complete"}:
            raise ValueError("rollout_mode must be 'segmented' or 'complete'.")
        if cfg.rollout_mode == "segmented" and update_horizon < cfg.segment_length:
            raise ValueError("update_horizon must be at least segment_length.")
        if cfg.rollout_mode == "segmented" and update_horizon % cfg.segment_length != 0:
            raise ValueError("update_horizon must be divisible by segment_length.")
        if update_horizon <= 0:
            raise ValueError("update_horizon must be positive.")
        if cfg.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if cfg.action_adjoint_max_norm < 0.0:
            raise ValueError("action_adjoint_max_norm cannot be negative.")
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
        if observation_normalizer is None and cfg.normalize_observations:
            observation_dim = int(env.single_observation_space.shape[-1])
            normalize_mask = getattr(env, "observation_normalize_mask", None)
            observation_normalizer = RunningObservationNormalizer(
                observation_dim,
                algorithm.device,
                normalize_mask=normalize_mask,
            )
        self.observation_normalizer = observation_normalizer
        self.collector = DifferentiableCollector(
            env=env,
            policy=policy,
            device=algorithm.device,
            observation_normalizer=observation_normalizer,
            clip_actions_to_space=cfg.clip_actions_to_space,
            action_adjoint_max_norm=cfg.action_adjoint_max_norm,
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
        self._start_step = 0
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
        self._rollout_index = 0

    def train(
        self,
        total_timesteps: int | None = None,
        *,
        total_updates: int | None = None,
    ) -> dict[str, Any]:
        """Train to a vector-transition or optimizer-update budget.

        ``total_updates`` is the stable budget for scheduled variable-horizon
        training. Exactly one budget may be supplied.

        Args:
            total_timesteps: Optional absolute vector-transition budget.
            total_updates: Optional absolute optimizer-update budget.

        Returns:
            Current training counters, metrics, histories, and checkpoint paths.

        Raises:
            ValueError: If neither/both budgets are supplied or one is negative.
        """
        if total_timesteps is None and total_updates is None:
            raise ValueError("Provide total_timesteps or total_updates.")
        if total_timesteps is not None and total_updates is not None:
            raise ValueError("Provide only one training budget.")
        if total_timesteps is not None and total_timesteps < 0:
            raise ValueError("total_timesteps cannot be negative.")
        if total_updates is not None and total_updates < 0:
            raise ValueError("total_updates cannot be negative.")

        if total_updates is not None:
            estimated_updates = total_updates
            remaining_updates = max(0, total_updates - self.num_updates)
        else:
            assert total_timesteps is not None
            if isinstance(
                self.env, ScheduledDifferentiableVecEnv
            ) and scheduler_needs_horizon(self.algorithm._lr_scheduler_cfg):
                raise ValueError(
                    "Variable-horizon APG requires total_updates or an explicit scheduler horizon."
                )
            steps_per_update = (
                self.update_horizon
                * self.env.num_envs
                * (
                    self.cfg.gradient_accumulation_steps
                    if self.cfg.rollout_mode == "complete"
                    else 1
                )
            )
            estimated_updates = math.ceil(total_timesteps / steps_per_update)
            remaining_updates = math.ceil(
                max(0, total_timesteps - self.global_step) / steps_per_update
            )
        self.algorithm._prepare_schedule(estimated_updates, remaining_updates)
        self.start_time = time.time()
        self._start_step = self.global_step

        self.policy.train()
        while (
            self.num_updates < total_updates
            if total_updates is not None
            else self.global_step < total_timesteps
        ):
            if self.cfg.rollout_mode == "complete":
                collected_steps, metrics = self._complete_rollout_update()
            else:
                update_steps = self.update_horizon
                if total_timesteps is not None:
                    remaining_vector_steps = math.ceil(
                        (total_timesteps - self.global_step) / self.env.num_envs
                    )
                    update_steps = min(self.update_horizon, remaining_vector_steps)
                collected_steps, metrics = self._segmented_update(update_steps)

            self.global_step += collected_steps * self.env.num_envs
            self.num_updates += 1
            elapsed = max(time.time() - self.start_time, 1e-6)
            entry = {
                "global_step": float(self.global_step),
                "num_updates": float(self.num_updates),
                "charts/SPS": float((self.global_step - self._start_step) / elapsed),
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

            saved_best = False
            if (
                self._next_eval_step is not None
                and self.eval_env is not None
                and self.global_step >= self._next_eval_step
            ):
                previous_best = self.best_checkpoint_path
                self._evaluate()
                saved_best = self.best_checkpoint_path != previous_best
                while self._next_eval_step <= self.global_step:
                    self._next_eval_step += self.cfg.eval_frequency_steps

            if (
                not saved_best
                and self.cfg.save_frequency_updates > 0
                and self.num_updates % self.cfg.save_frequency_updates == 0
            ):
                self.save_checkpoint()

        return self.get_summary()

    def _segmented_update(self, update_steps: int) -> tuple[int, dict[str, float]]:
        """Run one backward-compatible TBPTT optimizer update."""
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
                self._update_observation_normalizer(rollout)
                self.collector.detach_state()
                collected_steps += rollout.num_steps
            return collected_steps, self.algorithm.finish_update()
        except Exception:
            self.algorithm.cancel_update()
            raise

    def _complete_rollout_update(self) -> tuple[int, dict[str, float]]:
        """Accumulate independent full trajectories and apply one APG step."""
        accumulation_steps = self.cfg.gradient_accumulation_steps
        collected_steps = 0
        metadata: dict[str, list[float]] = {}
        self.algorithm.begin_update()
        try:
            for _ in range(accumulation_steps):
                spec = self._prepare_complete_rollout()
                seed = self.cfg.rollout_seed if self._rollout_index == 0 else None
                self.collector.reset(seed=seed)
                rollout = self.collector.collect(
                    spec.num_steps,
                    deterministic=self.cfg.deterministic_actions,
                    on_step_callback=self._on_step,
                )
                self.algorithm.accumulate_complete_rollout(
                    rollout,
                    objective_scale=spec.objective_scale,
                    accumulation_scale=1.0 / accumulation_steps,
                )
                self._update_observation_normalizer(rollout)
                self.collector.detach_state()
                collected_steps += rollout.num_steps
                for key, value in spec.metadata.items():
                    metadata.setdefault(str(key), []).append(float(value))
                self._rollout_index += 1
            metrics = self.algorithm.finish_update()
        except Exception:
            self.algorithm.cancel_update()
            raise
        for key, values in metadata.items():
            metrics[f"rollout_{key}_mean"] = sum(values) / len(values)
        return collected_steps, metrics

    def _prepare_complete_rollout(self) -> DifferentiableRolloutSpec:
        if isinstance(self.env, ScheduledDifferentiableVecEnv):
            return self.env.prepare_differentiable_rollout(self._rollout_index)
        return DifferentiableRolloutSpec(num_steps=self.update_horizon)

    def _update_observation_normalizer(
        self,
        rollout: DifferentiableRollout,
    ) -> None:
        if self.observation_normalizer is None:
            return
        observations = rollout.observations
        alive_observations = observations[rollout.alive_mask]
        self.observation_normalizer.update(alive_observations.detach())

    def save_checkpoint(self, path: str | Path | None = None) -> str:
        """Atomically save learning state and update the run index.

        Args:
            path: Destination, or the current step's default checkpoint path.

        Returns:
            Saved checkpoint path.
        """
        if path is None:
            path = (
                Path(self.cfg.checkpoint_dir)
                / f"{self.cfg.experiment_name}_step_{self.global_step}.pt"
            )
        extra = {"rollout_index": self._rollout_index}
        if self.observation_normalizer is not None:
            extra["observation_normalizer"] = self.observation_normalizer.state_dict()
        return _save(self, path, extra)

    def load_checkpoint(
        self, path: str | Path, *, resume_scheduler: str = "restore"
    ) -> None:
        """Validate and restore learning state, starting with a fresh rollout.

        Args:
            path: Checkpoint file, including existing APG schema-v1 files.
            resume_scheduler: Restore the saved LR curve or restart the
                configured curve while preserving optimizer momentum.
        """
        checkpoint = _load(self, path, resume_scheduler)
        self._rollout_index = int(checkpoint.get("rollout_index", self.num_updates))
        frequency = self.cfg.eval_frequency_steps
        self._next_eval_step = (
            (self.global_step // frequency + 1) * frequency if frequency > 0 else None
        )
        self.start_time = time.time()
        self._start_step = self.global_step

    def get_summary(self) -> dict[str, Any]:
        """Return the current in-memory training summary."""
        elapsed = max(1e-6, time.time() - self.start_time)
        return {
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "elapsed_time_sec": float(elapsed),
            "training_fps": float((self.global_step - self._start_step) / elapsed),
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
            observation_transform=(
                self.observation_normalizer.normalize
                if self.observation_normalizer is not None
                else None
            ),
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
                    / f"{self.cfg.experiment_name}_best_step_{self.global_step}.pt"
                )
                previous_best = self.best_checkpoint_path
                self.best_checkpoint_path = str(path)
                try:
                    self.save_checkpoint(path)
                except Exception:
                    self.best_checkpoint_path = previous_best
                    raise
        return metrics

    def _log(self, metrics: dict[str, float]) -> None:
        elapsed = max(time.time() - self.start_time, 1e-6)
        values = {
            **{f"train/{key}": value for key, value in metrics.items()},
            "charts/SPS": (self.global_step - self._start_step) / elapsed,
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
