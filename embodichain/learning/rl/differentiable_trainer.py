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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from embodichain.learning.rl.algo import APG
from embodichain.learning.rl.collector import DifferentiableCollector
from embodichain.learning.rl.env import DifferentiableVecEnv
from embodichain.learning.rl.models import Policy
from embodichain.utils import configclass

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__all__ = ["DifferentiableTrainer", "DifferentiableTrainerCfg"]

_CHECKPOINT_SCHEMA_VERSION = 1


@configclass
class DifferentiableTrainerCfg:
    """Configuration for graph-preserving segmented training."""

    segment_length: int = 16
    deterministic_actions: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    experiment_name: str = "apg"
    save_frequency_updates: int = 0


class DifferentiableTrainer:
    """Coordinate APG updates and truncated-backpropagation boundaries."""

    def __init__(
        self,
        cfg: DifferentiableTrainerCfg,
        env: DifferentiableVecEnv,
        policy: Policy,
        algorithm: APG,
        writer: SummaryWriter | None = None,
    ) -> None:
        if cfg.segment_length <= 0:
            raise ValueError("segment_length must be positive.")
        if cfg.save_frequency_updates < 0:
            raise ValueError("save_frequency_updates cannot be negative.")
        if algorithm.policy is not policy:
            raise ValueError("Trainer and APG must reference the same policy instance.")
        if torch.device(env.device) != algorithm.device:
            raise ValueError("Environment and APG must use the same device.")

        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.writer = writer
        self.collector = DifferentiableCollector(
            env=env,
            policy=policy,
            device=algorithm.device,
        )
        self.global_step = 0
        self.num_updates = 0
        self.train_history: list[dict[str, float]] = []
        self.latest_checkpoint_path: str | None = None

    def train(self, total_timesteps: int) -> dict[str, Any]:
        """Train until at least ``total_timesteps`` vector transitions exist."""
        if total_timesteps < 0:
            raise ValueError("total_timesteps cannot be negative.")

        while self.global_step < total_timesteps:
            remaining_vector_steps = math.ceil(
                (total_timesteps - self.global_step) / self.env.num_envs
            )
            segment_steps = min(self.cfg.segment_length, remaining_vector_steps)
            rollout = self.collector.collect(
                segment_steps,
                deterministic=self.cfg.deterministic_actions,
            )
            metrics = self.algorithm.update(rollout)
            self.collector.detach_state()

            self.global_step += rollout.num_steps * self.env.num_envs
            self.num_updates += 1
            entry = {
                "global_step": float(self.global_step),
                "num_updates": float(self.num_updates),
                **{f"train/{key}": value for key, value in metrics.items()},
            }
            self.train_history.append(entry)
            self._log(metrics)

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
        torch.save(
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "global_step": self.global_step,
                "num_updates": self.num_updates,
                "policy": self.policy.state_dict(),
                "optimizer": self.algorithm.optimizer.state_dict(),
            },
            checkpoint_path,
        )
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
        self.global_step = int(checkpoint["global_step"])
        self.num_updates = int(checkpoint["num_updates"])
        self.latest_checkpoint_path = str(path)

    def get_summary(self) -> dict[str, Any]:
        """Return the current in-memory training summary."""
        return {
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "last_train_metrics": (
                dict(self.train_history[-1]) if self.train_history else {}
            ),
            "train_history": list(self.train_history),
            "latest_checkpoint_path": self.latest_checkpoint_path,
        }

    def _log(self, metrics: dict[str, float]) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)
