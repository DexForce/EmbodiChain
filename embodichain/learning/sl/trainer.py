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

"""Offline training orchestration and reproducible committed-update recovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
from embodichain.utils import configclass
from .learner import SupervisedLearner
from .sources import SupervisedSource

__all__ = ["SupervisedTrainerCfg", "SupervisedTrainer"]


@configclass
class SupervisedTrainerCfg:
    """Configure an offline, single-device supervised run.

    Args:
        total_updates: Absolute optimizer-update budget, including resumed work.
        batch_size: Source examples per update.
        device: Tensor destination; runtime adapters remain caller-owned.
        seed: Independent source-sampling seed. Initialize models before building
            the trainer using the recipe's explicitly chosen model seed.
        output_dir: Local metrics and checkpoint directory.
        checkpoint_interval: Periodic save frequency; zero saves only at the end.
        validation_interval: Optional validation callback frequency.
    """

    total_updates: int = 1000
    batch_size: int = 64
    device: str = "cpu"
    seed: int = 0
    output_dir: str = "outputs/sl"
    checkpoint_interval: int = 1000
    validation_interval: int = 0


class SupervisedTrainer:
    """Compose sampling and a learner without owning acquisition or simulation.

    Args:
        learner: Shared single-update learner.
        source: Fixed data source with explicit identity and RNG-based sampling.
        cfg: Update budget, device and observation settings.
        signature: Recipe semantics, including feature/action units, frames,
            model configuration and numerical policy where applicable.
        scheduler: Optional scheduler stepped after each successful update.
        evaluate: Optional callback receiving the committed learner. It owns
            validation metrics and must not select a model on a test split.
    """

    def __init__(
        self,
        learner: SupervisedLearner,
        source: SupervisedSource,
        cfg: SupervisedTrainerCfg | None = None,
        *,
        signature: Mapping[str, Any] | None = None,
        scheduler: Any = None,
        evaluate: Callable[[SupervisedLearner], Mapping[str, float]] | None = None,
    ) -> None:
        self.cfg = cfg or SupervisedTrainerCfg()
        self.cfg.validate()
        for name in (
            "total_updates",
            "batch_size",
            "checkpoint_interval",
            "validation_interval",
            "seed",
        ):
            value = getattr(self.cfg, name)
            if type(value) is not int or value < (1 if name == "batch_size" else 0):
                raise ValueError(f"Invalid supervised configuration: {name}")
        self.learner, self.source = learner, source
        self.signature = json.loads(
            json.dumps(dict(signature or {}), sort_keys=True, allow_nan=False)
        )
        self.scheduler, self.evaluate = scheduler, evaluate
        self.sampler = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        self.output = Path(self.cfg.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.learner.model.to(self.cfg.device)

    def fit(self, *, stop_after: int | None = None) -> dict[str, Any]:
        """Train up to the configured or an earlier committed update boundary.

        Args:
            stop_after: Optional absolute early-stop update for resumable runs.

        Returns:
            Completed update count, final metrics and last-checkpoint path.
        """
        end = (
            self.cfg.total_updates
            if stop_after is None
            else min(stop_after, self.cfg.total_updates)
        )
        if end < self.learner.num_updates:
            raise ValueError("stop_after precedes the restored update cursor")
        metrics = {}
        while self.learner.num_updates < end:
            candidate = torch.Generator(device="cpu")
            candidate.set_state(self.sampler.get_state())
            batch = self.source.sample(self.cfg.batch_size, generator=candidate).to(
                self.cfg.device
            )
            metrics = self.learner.update(batch)
            self.sampler.set_state(candidate.get_state())
            if self.scheduler is not None:
                self.scheduler.step()
            step = self.learner.num_updates
            if (
                self.evaluate is not None
                and self.cfg.validation_interval
                and step % self.cfg.validation_interval == 0
            ):
                rng = self._rng_state()
                was_training = self.learner.model.training
                try:
                    self.learner.model.eval()
                    with torch.no_grad():
                        metrics.update(
                            {
                                f"validation/{key}": float(value)
                                for key, value in self.evaluate(self.learner).items()
                            }
                        )
                finally:
                    self._restore_rng(rng)
                    self.learner.model.train(was_training)
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            if (
                self.cfg.checkpoint_interval
                and step % self.cfg.checkpoint_interval == 0
            ):
                self.save(self.output / f"step_{step}.pt")
        path = self.output / "last.pt"
        self.save(path)
        return {
            "num_updates": self.learner.num_updates,
            "metrics": metrics,
            "checkpoint": str(path),
        }

    def _rng_state(self) -> dict[str, Any]:
        device = torch.device(self.cfg.device)
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }

    def _restore_rng(self, state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"], self.cfg.device)

    def save(self, path: str | Path) -> None:
        """Atomically save a complete committed learning and sampling boundary.

        Args:
            path: Local checkpoint destination, replaced atomically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "learner": self.learner.state_dict(),
            "sampler": self.sampler.get_state(),
            "rng": self._rng_state(),
            "signature": self.signature,
            "source_id": self.source.source_id,
            "config": self.cfg.to_dict(),
            "scheduler_type": (
                None
                if self.scheduler is None
                else f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
            ),
            "scheduler": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path: str | Path) -> None:
        """Restore the same run after validating data, recipe and configuration.

        Args:
            path: Trusted training checkpoint from this trainer.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (
            state.get("version") != 1
            or state["signature"] != self.signature
            or state["source_id"] != self.source.source_id
        ):
            raise ValueError("Supervised checkpoint recipe or dataset identity differs")
        current, saved = self.cfg.to_dict(), dict(state["config"])
        for key in ("output_dir",):
            current.pop(key, None)
            saved.pop(key, None)
        if current != saved or (state["scheduler"] is None) != (self.scheduler is None):
            raise ValueError("Supervised checkpoint training configuration differs")
        scheduler_type = (
            None
            if self.scheduler is None
            else f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
        )
        if state.get("scheduler_type") != scheduler_type:
            raise ValueError("Supervised checkpoint scheduler type differs")
        self.learner.load_state_dict(state["learner"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.sampler.set_state(state["sampler"])
        self._restore_rng(state["rng"])
