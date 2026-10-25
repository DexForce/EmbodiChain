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
from .distributed import _DistributedContext
from .learner import SupervisedLearner
from .sources import SupervisedSource

__all__ = ["SupervisedTrainerCfg", "SupervisedTrainer"]


@configclass
class SupervisedTrainerCfg:
    """Configure an offline run, using one device per process under DDP.

    Args:
        total_updates: Absolute optimizer-update budget, including resumed work.
        batch_size: Source examples per rank and update. Global batch size under
            DDP is this value times the process-group world size.
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
        self._context = _DistributedContext(self.cfg.device)
        self.sampler = torch.Generator(device="cpu").manual_seed(
            self.cfg.seed + self._context.rank
        )
        self.output = Path(self.cfg.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.learner.model.to(self.cfg.device)
        self._owned_artifacts: dict[Path, tuple[int, int, int, int]] = {}
        self._fit_started = False
        self._restored = False
        self._distributed_prepared = False
        self._evaluation_snapshot: list[Any] | None = None

    def _prepare_distributed(self) -> None:
        if not self._context.enabled or self._distributed_prepared:
            return
        configuration = self.cfg.to_dict()
        configuration["device"] = torch.device(self.cfg.device).type
        configuration["output_dir"] = str(self.output.resolve())
        self._context.agree(
            {
                "config": configuration,
                "signature": self.signature,
                "source_id": self.source.source_id,
                "objective": f"{type(self.learner.objective).__module__}.{type(self.learner.objective).__qualname__}",
                "optimizer_type": f"{type(self.learner.optimizer).__module__}.{type(self.learner.optimizer).__qualname__}",
                "optimizer_layout": self.learner._optimizer_layout(),
                "optimizer_options": [
                    {
                        key: (
                            value.detach().cpu().tolist()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in group.items()
                        if key != "params"
                    }
                    for group in self.learner.optimizer.param_groups
                ],
                "max_grad_norm": self.learner.max_grad_norm,
                "amp_dtype": str(self.learner.amp_dtype),
                "find_unused_parameters": self.learner.ddp_find_unused_parameters,
                "scheduler_type": (
                    None
                    if self.scheduler is None
                    else f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
                ),
                "validation": self.evaluate is not None,
            },
            "recipe or dataset identity",
        )
        self.learner._configure_distributed()
        # Factories use a common seed for construction. After DDP synchronizes
        # replicas, stochastic training and sampling use independent rank seeds.
        seed = self.cfg.seed + self._context.rank
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        self._distributed_prepared = True

    def fit(self, *, stop_after: int | None = None) -> dict[str, Any]:
        """Train up to the configured or an earlier committed update boundary.

        Args:
            stop_after: Optional absolute early-stop update for resumable runs.

        Returns:
            Completed update count, final metrics and last-checkpoint path.
        """
        self._prepare_distributed()
        end = (
            self.cfg.total_updates
            if stop_after is None
            else min(stop_after, self.cfg.total_updates)
        )
        self._context.agree((end, self.learner.num_updates), "update boundary")

        def admit_output() -> None:
            if end < self.learner.num_updates:
                raise ValueError("stop_after precedes the restored update cursor")
            if self._fit_started or self._restored:
                return
            artifacts = [
                self.output / "metrics.jsonl",
                self.output / "last.pt",
                *self.output.glob("step_*.pt"),
            ]
            for path in artifacts:
                if path.exists() and self._owned_artifacts.get(
                    path.resolve()
                ) != self._artifact_stamp(path):
                    raise FileExistsError(
                        "SL output contains another run; restore a checkpoint or use a fresh output directory"
                    )

        self._context.root(admit_output, "output admission")
        self._fit_started = True
        metrics = {}
        while self.learner.num_updates < end:
            candidate = torch.Generator(device="cpu")
            candidate.set_state(self.sampler.get_state())
            batch = self._context.run(
                lambda: self.source.sample(self.cfg.batch_size, generator=candidate).to(
                    self.cfg.device
                ),
                "sampling",
            )
            metrics = self.learner.update(batch)
            self.sampler.set_state(candidate.get_state())
            if self.scheduler is not None:
                self._context.run(self.scheduler.step, "scheduler")
            step = self.learner.num_updates
            if (
                self.evaluate is not None
                and self.cfg.validation_interval
                and step % self.cfg.validation_interval == 0
            ):
                metrics.update(self._evaluate_step())

            def append_metrics() -> None:
                with (self.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")

            self._context.root(append_metrics, "metric logging")
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
            "world_size": self._context.world_size,
            "global_batch_size": self.cfg.batch_size * self._context.world_size,
        }

    def _local_rank_state(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler.get_state(),
            "rng": self._rng_state(),
            "buffers": {
                name: buffer.detach().cpu().clone()
                for name, buffer in self.learner.model.named_buffers()
            },
        }

    def _gather_rank_states(self) -> list[Any] | None:
        state = self._context.run(self._local_rank_state, "checkpoint rank state")
        return self._context.gather(state)

    def _evaluate_step(self) -> dict[str, float]:
        # Validation may publish a checkpoint from rank zero. Capture every
        # rank's committed boundary first, so callback save() has no collectives.
        snapshot = self._gather_rank_states()

        def evaluate() -> dict[str, float]:
            rng = self._rng_state()
            assert snapshot is not None
            buffers = snapshot[0]["buffers"]
            was_training = self.learner.model.training
            self._evaluation_snapshot = snapshot
            try:
                self.learner.model.eval()
                assert self.evaluate is not None
                with torch.no_grad():
                    metrics = {
                        f"validation/{key}": float(value)
                        for key, value in self.evaluate(self.learner).items()
                    }
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Non-finite validation metric")
                return metrics
            finally:
                self._evaluation_snapshot = None
                with torch.no_grad():
                    for name, value in self.learner.model.named_buffers():
                        value.copy_(buffers[name])
                self._restore_rng(rng)
                self.learner.model.train(was_training)

        return self._context.root(evaluate, "validation")

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

    @staticmethod
    def _artifact_stamp(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def save(self, path: str | Path) -> None:
        """Atomically save a complete committed learning and sampling boundary.

        Args:
            path: Local checkpoint destination, replaced atomically.

        All ranks call this method together, except inside the trainer's
        rank-zero validation callback, where a captured boundary is available.
        """
        self._prepare_distributed()
        if self._evaluation_snapshot is not None:
            self._write_checkpoint(path, self._evaluation_snapshot)
            return
        self._context.agree(str(Path(path).resolve()), "checkpoint destination")
        rank_states = self._gather_rank_states() if self._context.enabled else None
        self._context.root(
            lambda: self._write_checkpoint(path, rank_states), "checkpoint write"
        )

    def _write_checkpoint(
        self, path: str | Path, rank_states: list[Any] | None
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 2 if self._context.enabled else 1,
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
        if self._context.enabled:
            assert rank_states is not None
            state["distributed"] = {
                "world_size": self._context.world_size,
                "rank_states": rank_states,
            }
        if rank_states is not None:
            state["sampler"] = rank_states[0]["sampler"]
            state["rng"] = rank_states[0]["rng"]
            # An eval forward may maintain caches/counters. A callback save is
            # still the pre-validation training boundary, including buffers.
            for name, value in rank_states[0]["buffers"].items():
                if name in state["learner"]["model"]:
                    state["learner"]["model"][name] = value
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        self._owned_artifacts[path.resolve()] = self._artifact_stamp(path)

    def resume(self, path: str | Path) -> None:
        """Restore the same run after validating data, recipe and configuration.

        Args:
            path: Trusted training checkpoint from this trainer.

        DDP recovery is collective and requires the original world size.
        """
        self._context.agree(str(Path(path).resolve()), "resume checkpoint")
        state = self._context.run(
            lambda: torch.load(path, map_location="cpu", weights_only=False),
            "checkpoint read",
        )
        self._context.run(
            lambda: self._validate_checkpoint(state), "checkpoint compatibility"
        )
        # Initial DDP broadcasts must happen before restoring rank-local buffers.
        self._prepare_distributed()

        def restore() -> None:
            self.learner.load_state_dict(state["learner"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            if self._context.enabled:
                local = state["distributed"]["rank_states"][self._context.rank]
                with torch.no_grad():
                    for name, buffer in self.learner.model.named_buffers():
                        buffer.copy_(local["buffers"][name])
            else:
                local = state
            self.sampler.set_state(local["sampler"])
            self._restore_rng(local["rng"])

        self._context.run(restore, "checkpoint restore")
        self._restored = True

    def _validate_checkpoint(self, state: Mapping[str, Any]) -> None:
        distributed = state.get("distributed", {})
        if distributed.get("world_size", 1) != self._context.world_size:
            raise ValueError(
                "Supervised checkpoint world size differs; exact recovery requires the same world size"
            )
        if (
            state.get("version") != (2 if self._context.enabled else 1)
            or state["signature"] != self.signature
            or state["source_id"] != self.source.source_id
        ):
            raise ValueError("Supervised checkpoint recipe or dataset identity differs")
        current, saved = self.cfg.to_dict(), dict(state["config"])
        for key in ("output_dir",):
            current.pop(key, None)
            saved.pop(key, None)
        if self._context.enabled:
            current["device"] = torch.device(current["device"]).type
            saved["device"] = torch.device(saved["device"]).type
        if current != saved or (state["scheduler"] is None) != (self.scheduler is None):
            raise ValueError("Supervised checkpoint training configuration differs")
        scheduler_type = (
            None
            if self.scheduler is None
            else f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
        )
        if state.get("scheduler_type") != scheduler_type:
            raise ValueError("Supervised checkpoint scheduler type differs")
        metrics_path = self.output / "metrics.jsonl"
        if metrics_path.exists():
            last_line = ""
            with metrics_path.open() as handle:
                for line in handle:
                    if line.strip():
                        last_line = line
            if (
                last_line
                and json.loads(last_line)["num_updates"]
                > state["learner"]["num_updates"]
            ):
                raise ValueError(
                    "SL output history is ahead of this checkpoint; use a fresh output directory"
                )
        self.learner._validate_state_dict(state["learner"])
        if self._context.enabled:
            rank_states = distributed["rank_states"]
            if len(rank_states) != self._context.world_size:
                raise ValueError("Supervised checkpoint rank states differ")
            buffers = dict(self.learner.model.named_buffers())
            local_buffers = rank_states[self._context.rank]["buffers"]
            if buffers.keys() != local_buffers.keys() or any(
                value.shape != local_buffers[name].shape
                or value.dtype != local_buffers[name].dtype
                for name, value in buffers.items()
            ):
                raise ValueError("Supervised checkpoint rank-local buffers differ")
