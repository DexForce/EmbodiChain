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

import math
import time
from collections.abc import Mapping
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from embodichain.learning.rl._trainer_reporting import (
    _evaluation_improved,
    _write_scalars,
    _write_tensorboard_scalars,
)
from embodichain.learning.rl.checkpoint import _load, _save, _sync_errors
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.evaluation import evaluate_episodes

__all__ = ["Trainer"]


class Trainer:
    """Coordinate algorithm training, logging and evaluation.

    Enabled ``event_cfg`` and ``eval_event_cfg`` declarations require the
    corresponding environment's ``add_events()`` lifecycle. Environments own
    event dispatch, reset and cleanup; empty or disabled events require no event API.
    """

    def __init__(
        self,
        policy,
        env,
        algorithm,
        buffer_size: int,
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
        eval_seed: int | None = None,
        best_eval_metric: str = "eval/avg_reward",
        best_eval_mode: str = "max",
        save_frequency_updates: int = 0,
    ):
        if best_eval_mode not in {"min", "max"}:
            raise ValueError("best_eval_mode must be 'min' or 'max'.")
        self.policy = policy
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.env = env
        self.eval_env = eval_env
        self.algorithm = algorithm
        self.buffer_size = buffer_size
        self.writer = writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_frequency_updates = save_frequency_updates
        self.checkpoint_dir = checkpoint_dir
        self.exp_name = exp_name
        self.use_wandb = use_wandb
        self.num_eval_episodes = num_eval_episodes
        self.eval_seed = eval_seed
        self.best_eval_metric = best_eval_metric
        self.best_eval_mode = best_eval_mode
        self.best_eval_value: float | None = None
        self.best_checkpoint_path: str | None = None

        self._bind_events(self.env, event_cfg)
        if self.eval_env is not None:
            self._bind_events(self.eval_env, eval_event_cfg)

        self.device = self.algorithm.device
        self.global_step = 0
        self._training_start_step = 0
        self.num_updates = 0
        self.start_time = time.time()
        self._pending_best = False
        self.ret_window = deque(maxlen=100)
        self.len_window = deque(maxlen=100)
        self.train_history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, float]] = []
        self.last_eval_metrics: dict[str, float] = {}
        self.last_train_metrics: dict[str, float] = {}
        self.latest_checkpoint_path: str | None = None
        self._next_eval_step = eval_freq if eval_freq > 0 else None
        self._next_save_step = save_freq if save_freq > 0 else None
        num_envs = getattr(self.env, "num_envs", None)
        if num_envs is None:
            raise RuntimeError("Env must expose num_envs for trainer statistics.")
        policy_module = getattr(self.policy, "module", self.policy)
        obs_dim = getattr(policy_module, "obs_dim", None)
        action_dim = getattr(policy_module, "action_dim", None)
        if obs_dim is None or action_dim is None:
            raise RuntimeError("Policy must expose obs_dim and action_dim.")
        critic_obs_dim = (
            getattr(policy_module, "critic_obs_dim", None)
            if getattr(policy_module, "uses_separate_critic_obs", False)
            else None
        )

        self.buffer = RolloutBuffer(
            num_envs=num_envs,
            rollout_len=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device,
            critic_obs_dim=critic_obs_dim,
            distribution_param_dim=getattr(
                policy_module,
                "distribution_param_dim",
                None,
            ),
            training_field_specs=getattr(policy_module, "training_field_specs", None),
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
            bootstrap_timeouts=getattr(self.algorithm, "uses_value_bootstrap", True),
        )

        self.curr_ret = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.curr_len = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        # Keep completion snapshots on device until the rollout boundary. The
        # time-major layout preserves completion order when several rows finish.
        self._episode_returns = torch.empty(
            (self.buffer_size, num_envs), dtype=self.curr_ret.dtype, device=self.device
        )
        self._episode_lengths = torch.empty(
            (self.buffer_size, num_envs), dtype=self.curr_len.dtype, device=self.device
        )

    @staticmethod
    def _bind_events(env: Any, event_cfg: object) -> None:
        """Attach event declarations to their environment's lifecycle."""
        if event_cfg is None:
            return
        declarations = event_cfg if isinstance(event_cfg, Mapping) else vars(event_cfg)
        if not declarations:
            return
        add_events = getattr(env, "add_events", None)
        if not callable(add_events):
            if not any(cfg is not None for cfg in declarations.values()):
                return
            raise ValueError(
                "Trainer events require an environment with add_events() "
                "that owns event execution, reset and cleanup."
            )
        add_events(event_cfg)

    @staticmethod
    def _reduce_log_dicts(
        groups: dict[str, dict | None],
    ) -> dict[str, float | torch.Tensor]:
        """Snapshot means before the environment can reuse its metric buffers."""
        out: dict[str, float | torch.Tensor] = {}
        for prefix, data in groups.items():
            if not isinstance(data, dict):
                continue
            for key, value in data.items():
                tag = f"{prefix}/{key}"
                try:
                    if isinstance(value, torch.Tensor) and not value.is_complex():
                        value = value.detach()
                        out[tag] = (
                            value.mean()
                            if value.is_floating_point()
                            else value.mean(dtype=torch.float64)
                        )
                    else:
                        array = (
                            value.detach().cpu().numpy()
                            if hasattr(value, "detach")
                            else np.asarray(value)
                        )
                        out[tag] = float(np.mean(array))
                except Exception:
                    continue
        return out

    @staticmethod
    def _materialize_log_dicts(
        snapshots: list[dict[str, float | torch.Tensor]],
    ) -> list[dict[str, float]]:
        """Copy rollout metric scalars together, retaining every step and tag.

        Both sinks consume identical per-step means without a host read between
        environment steps. Scalar reductions own their storage, so a later
        environment step cannot change an earlier log record.
        """
        result: list[dict[str, float]] = [{} for _ in snapshots]
        pending: dict[torch.device, list[tuple[int, str, torch.Tensor]]] = {}
        for index, snapshot in enumerate(snapshots):
            for tag, value in snapshot.items():
                if isinstance(value, torch.Tensor):
                    pending.setdefault(value.device, []).append((index, tag, value))
                else:
                    result[index][tag] = value
        for values in pending.values():
            try:
                means = torch.stack([mean for _, _, mean in values]).cpu().tolist()
            except Exception:
                continue
            for (index, tag, _), mean in zip(values, means):
                result[index][tag] = float(mean)
        return [
            {tag: values[tag] for tag in snapshot if tag in values}
            for snapshot, values in zip(snapshots, result)
        ]

    def train(self, total_timesteps: int) -> dict[str, Any]:
        self._training_start_step = self.global_step
        self.start_time = time.time()
        if self.rank == 0:
            print(f"Start training, total steps: {total_timesteps}")
        num_envs = int(self.env.num_envs)
        steps_per_update = self.buffer_size * num_envs * self.world_size
        total_updates = math.ceil(total_timesteps / steps_per_update)
        remaining = math.ceil(
            max(0, total_timesteps - self.global_step) / steps_per_update
        )
        self.algorithm._prepare_schedule(total_updates, remaining)
        while self.global_step < total_timesteps:
            self._collect_rollout()
            losses = self.algorithm.update(self.buffer.get(flatten=False))
            self.num_updates += 1
            self._log_train(losses)
            self._pending_best = False
            if (
                self._next_eval_step is not None
                and self.global_step >= self._next_eval_step
            ):
                error = None
                if self.rank == 0 and self.eval_env is not None:
                    try:
                        self._eval_once(num_episodes=self.num_eval_episodes)
                    except Exception as exc:
                        error = exc
                _sync_errors(error, self.distributed)
                if self.distributed:
                    decision = [self._pending_best, self.best_eval_value]
                    torch.distributed.broadcast_object_list(decision, src=0)
                    self._pending_best, self.best_eval_value = decision
                while self._next_eval_step <= self.global_step:
                    self._next_eval_step += self.eval_freq
            periodic = (
                self._next_save_step is not None
                and self.global_step >= self._next_save_step
            )
            by_update = (
                self.save_frequency_updates > 0
                and self.num_updates % self.save_frequency_updates == 0
            )
            if self._pending_best:
                previous_best = self.best_checkpoint_path
                best_path = str(
                    Path(self.checkpoint_dir)
                    / f"{self.exp_name}_best_step_{self.global_step}.pt"
                )
                self.best_checkpoint_path = best_path
                try:
                    self.save_checkpoint(best_path)
                except Exception:
                    self.best_checkpoint_path = previous_best
                    raise
            elif periodic or by_update:
                self.save_checkpoint()
            if periodic:
                while self._next_save_step <= self.global_step:
                    self._next_save_step += self.save_freq
        return self.get_summary()

    @torch.no_grad()
    def _collect_rollout(self):
        """Collect a rollout with the synchronous collector."""
        start_step = self.global_step
        step_index = 0
        metric_steps: list[int] = []
        metric_snapshots: list[dict[str, float | torch.Tensor]] = []

        def on_step(tensordict: TensorDict, info: dict):
            nonlocal step_index
            reward = tensordict["reward"]
            done = tensordict["done"]
            self.curr_ret += reward
            self.curr_len += 1
            self._episode_returns[step_index].copy_(self.curr_ret)
            self._episode_lengths[step_index].copy_(self.curr_len)
            self.curr_ret.masked_fill_(done, 0)
            self.curr_len.masked_fill_(done, 0)
            step_index += 1

            if not self.distributed:
                self.global_step += tensordict.batch_size[0]

            if (
                self.rank == 0
                and isinstance(info, dict)
                and (self.writer or self.use_wandb)
            ):
                metric_steps.append(step_index)
                metric_snapshots.append(
                    self._reduce_log_dicts(
                        {"rewards": info.get("rewards"), "metrics": info.get("metrics")}
                    )
                )

        rollout = self.buffer.start_rollout()
        rollout = self.collector.collect(
            num_steps=self.buffer_size,
            rollout=rollout,
            on_step_callback=on_step,
        )
        self.buffer.add(rollout)
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
        self._record_episode_stats(rollout)
        steps_per_frame = (self.global_step - start_step) // self.buffer_size
        for frame, log_dict in zip(
            metric_steps, self._materialize_log_dicts(metric_snapshots)
        ):
            step = start_step + frame * steps_per_frame
            if self.writer:
                _write_tensorboard_scalars(
                    self.writer, log_dict, step, ignore_errors=True
                )
            if self.use_wandb and log_dict:
                wandb.log(log_dict, step=step)

    def _record_episode_stats(self, rollout: TensorDict) -> None:
        """Append new completions in step, rank, then row order.

        Rank zero owns the distributed history. Only this rollout's completions
        cross ranks; a previously merged window must never become local input.
        """
        done = rollout["done"][:, :-1].transpose(0, 1).reshape(-1)
        completed = done.nonzero(as_tuple=False).squeeze(-1)
        # Earlier completions cannot survive either bounded history window.
        completed = completed[-self.ret_window.maxlen :]
        if self.distributed:
            # Fixed-size messages also let ranks without completions participate.
            # FP64 preserves FP32 returns and integer episode lengths/indices.
            stats = torch.zeros(
                (self.ret_window.maxlen, 4), dtype=torch.float64, device=self.device
            )
            stats[:, 0] = -1
            n = completed.numel()
            stats[:n, 0] = completed // self.env.num_envs
            stats[:n, 1] = completed % self.env.num_envs
            stats[:n, 2] = self._episode_returns.reshape(-1)[completed]
            stats[:n, 3] = self._episode_lengths.reshape(-1)[completed]
            gathered = [torch.empty_like(stats) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, stats)
            if self.rank == 0:
                episodes = [
                    (step, rank, row, reward, length)
                    for rank, values in enumerate(torch.stack(gathered).cpu().tolist())
                    for step, row, reward, length in values
                    if step >= 0
                ]
                episodes.sort(key=lambda episode: episode[:3])
                self.ret_window.extend(episode[3] for episode in episodes)
                self.len_window.extend(int(episode[4]) for episode in episodes)
                return
        if completed.numel() == 0:
            return
        self.ret_window.extend(
            self._episode_returns.reshape(-1)[completed].cpu().tolist()
        )
        self.len_window.extend(
            self._episode_lengths.reshape(-1)[completed].cpu().tolist()
        )

    def _log_train(self, losses: dict[str, float]):
        elapsed = max(1e-6, time.time() - self.start_time)
        sps = (self.global_step - self._training_start_step) / elapsed
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

        values = {f"train/{k}": v for k, v in losses.items()}
        values["charts/SPS"] = sps
        if self.ret_window:
            values["charts/episode_reward_avg_100"] = float(avgR)
        if self.len_window:
            values["charts/episode_length_avg_100"] = float(avgL)
        # TensorBoard retains NaN episode means; W&B historically omits them.
        wandb_values = {
            key: value
            for key, value in values.items()
            if not (key.startswith("charts/episode_") and np.isnan(value))
        }
        _write_scalars(
            values,
            step=self.global_step,
            writer=self.writer,
            use_wandb=self.rank == 0 and self.use_wandb,
            wandb_metrics=wandb_values,
        )
        if self.rank == 0:
            print(
                f"[train] step={self.global_step} sps={sps:.0f} avgReward(100)={avgR:.3f} avgLength(100)={avgL:.1f}"
            )

    @torch.no_grad()
    def _eval_once(self, num_episodes: int = 5) -> dict[str, float]:
        """Evaluate ``num_episodes`` completed asynchronous episodes."""

        # The first reset must not commit an unfinished preceding evaluation.
        # Completed episodes were already submitted before the host auto-reset.
        self._flush_eval_recordings()
        try:
            metrics = evaluate_episodes(
                policy=self.policy,
                env=self.eval_env,
                num_episodes=num_episodes,
                device=self.device,
                seed=self.eval_seed,
            )
        except BaseException as error:
            try:
                self._flush_eval_recordings()
            except Exception as cleanup_error:
                error.add_note(
                    f"Evaluation event cleanup also failed: {cleanup_error!r}"
                )
            raise
        else:
            self._flush_eval_recordings()
        summary = {"global_step": float(self.global_step), **metrics}
        self.eval_history.append(summary)
        self.last_eval_metrics = summary
        _write_scalars(
            {key: value for key, value in metrics.items() if np.isfinite(value)},
            step=self.global_step,
            writer=self.writer,
            use_wandb=self.rank == 0 and self.use_wandb,
        )
        candidate = metrics.get(self.best_eval_metric)
        if _evaluation_improved(candidate, self.best_eval_value, self.best_eval_mode):
            self.best_eval_value = candidate
            self._pending_best = True
        return summary

    def _flush_eval_recordings(self) -> None:
        """Drain host recordings without closing reusable event functors."""
        flush_recordings = getattr(self.eval_env, "flush_recordings", None)
        if callable(flush_recordings):
            flush_recordings()

    def save_checkpoint(self, path: str | Path | None = None) -> str:
        """Save complete learning state; every distributed rank must participate.

        Args:
            path: Destination, or the current step's default checkpoint path.

        Returns:
            Saved checkpoint path. Rank zero owns file and manifest writes.
        """
        if path is None:
            path = (
                Path(self.checkpoint_dir)
                / f"{self.exp_name}_step_{self.global_step}.pt"
            )
        return _save(self, path, {})

    def load_checkpoint(
        self, path: str | Path, *, resume_scheduler: str = "restore"
    ) -> None:
        """Validate and restore learning state, then start a new episode.

        Args:
            path: Checkpoint to load on every participating rank.
            resume_scheduler: Restore the saved LR curve or restart the
                configured curve while preserving optimizer momentum.
        """
        _load(self, path, resume_scheduler)
        self._next_eval_step = (
            (self.global_step // self.eval_freq + 1) * self.eval_freq
            if self.eval_freq > 0
            else None
        )
        self._next_save_step = (
            (self.global_step // self.save_freq + 1) * self.save_freq
            if self.save_freq > 0
            else None
        )
        self.start_time = time.time()
        self._training_start_step = self.global_step

    def get_summary(self) -> dict[str, Any]:
        """Return persistent counters and throughput for the current train call."""
        elapsed = max(1e-6, time.time() - self.start_time)
        return {
            "global_step": int(self.global_step),
            "num_updates": int(self.num_updates),
            "elapsed_time_sec": float(elapsed),
            "training_fps": float(
                (self.global_step - self._training_start_step) / elapsed
            ),
            "last_train_metrics": dict(self.last_train_metrics),
            "last_eval_metrics": dict(self.last_eval_metrics),
            "train_history": list(self.train_history),
            "eval_history": list(self.eval_history),
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
        }
