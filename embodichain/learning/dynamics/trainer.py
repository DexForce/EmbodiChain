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

"""Single-device supervised NeRD training with committed-step resume."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from embodichain.utils import configclass
from embodichain.data_pipeline.dynamics.shards import DynamicsWindowDataset

__all__ = [
    "DynamicsTrainCfg",
    "DynamicsTrainer",
    "encode_batch",
    "normalized_mse",
    "compute_statistics",
    "load_statistics",
    "save_statistics",
    "evaluate_model",
]


@configclass
class DynamicsTrainCfg:
    """Optimizer and deterministic sampling configuration.

    ``lr_update_interval`` holds the rate for a fixed number of updates, allowing
    an official epoch schedule. ``lr_offset`` is measured in schedule intervals;
    the official first trained epoch uses offset one because epoch zero evaluates.
    The durable cursor is the count of completed optimizer updates.
    """

    total_steps: int = 10000
    batch_size: int = 64
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"
    validation_interval: int = 100
    validation_batches: int | None = None
    checkpoint_interval: int = 100
    lr_update_interval: int = 1
    lr_offset: int = 0


def encode_batch(
    batch: dict[str, Any], codec: Any, device: str | torch.device
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Encode a collated raw trajectory window through the shared codec."""
    states = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in batch["states"].items()
    }
    transitions = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in batch["transitions"].items()
    }
    effort = transitions.get("joint_f", transitions.get("effort"))
    if effort is None:
        raise ValueError("Dataset requires transitions/joint_f")
    contacts = {
        key: value for key, value in transitions.items() if key.startswith("contact_")
    }
    q, qd = states["q"], states["qd"]
    inputs = codec.encode(
        q[:, :-1], qd[:, :-1], effort, contacts or None, transitions.get("gravity_dir")
    )
    target = codec.targets(q[:, :-1], qd[:, :-1], q[:, 1:], qd[:, 1:])
    return inputs, target, batch["valid_mask"].to(device)


def normalized_mse(
    prediction: Tensor,
    target: Tensor,
    variance: Tensor | None,
    mask: Tensor | None = None,
) -> Tensor:
    """Official MSE over normalized physical targets, excluding padded steps."""
    if variance is None:
        error = prediction - target
    else:
        weights = torch.rsqrt(variance + 1e-5)
        # Preserve official arithmetic ordering for numerical parity.
        error = prediction * weights - target * weights
    squared = error.square()
    if mask is None:
        return squared.mean()
    valid = mask.to(squared.dtype).unsqueeze(-1)
    return (squared * valid).sum() / (valid.sum() * squared.shape[-1]).clamp_min(1)


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.mean: Tensor | None = None
        self.m2: Tensor | None = None

    def add(self, values: Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        count = values.shape[0]
        if not count:
            return
        mean = values.mean(0)
        m2 = (values - mean).square().sum(0)
        if self.mean is None:
            self.count, self.mean, self.m2 = count, mean, m2
        else:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += m2 + delta.square() * self.count * count / total
            self.mean += delta * count / total
            self.count = total

    def result(self) -> tuple[Tensor, Tensor]:
        if self.mean is None:
            raise ValueError("Cannot normalize an empty dataset")
        return self.mean.float(), (self.m2 / self.count).float()


def compute_statistics(
    dataset: DynamicsWindowDataset, codec: Any, batch_size: int = 64
) -> dict[str, Any]:
    """Compute bounded training-only, overlapping-window-weighted moments.

    Statistical weighting is explicit. Official comparison runs can instead load
    the exact shared reference artifact; no claim of matching an upstream running
    estimator's epsilon initialization is implied by these population moments.
    """
    if dataset.split != "train":
        raise ValueError("Normalization statistics may only use the training split")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    moments: dict[str, _Moments] = {}
    targets = _Moments()
    for start in range(0, len(dataset), batch_size):
        batch = dataset.get_batch(range(start, min(len(dataset), start + batch_size)))
        inputs, target, mask = encode_batch(batch, codec, "cpu")
        for key, value in inputs.items():
            moments.setdefault(key, _Moments()).add(value[mask])
        targets.add(target[mask])
    mean, variance = targets.result()
    return {
        "schema_version": 1,
        "weighting": "valid_window_occurrences",
        "dataset_fingerprint": dataset.fingerprint,
        "inputs": {
            key: {"mean": item.result()[0], "var": item.result()[1]}
            for key, item in moments.items()
        },
        "output": {"mean": mean, "var": variance},
    }


def save_statistics(path: str | Path, statistics: dict[str, Any]) -> None:
    """Write a tensor-only statistics artifact usable by both trainers."""
    _atomic_save(Path(path), statistics)


def load_statistics(path: str | Path) -> dict[str, Any]:
    """Load a tensor-only shared reference statistics artifact."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported statistics schema")
    for item in [*value["inputs"].values(), value["output"]]:
        if (
            not torch.isfinite(item["mean"]).all()
            or not torch.isfinite(item["var"]).all()
            or (item["var"] < 0).any()
        ):
            raise ValueError("Invalid normalizer moments")
    return value


def _atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _model_variance(model: torch.nn.Module) -> Tensor | None:
    return model.output_rms.var if model.cfg.normalize_output else None


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    codec: Any,
    dataset: DynamicsWindowDataset,
    batch_size: int = 64,
    device: str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate all valid held-out positions without model selection on test."""
    if not len(dataset):
        raise ValueError(f"No eligible windows in {dataset.split} split")
    if batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("Evaluation batch limits must be positive")
    window_count = (
        len(dataset)
        if max_batches is None
        else min(len(dataset), max_batches * batch_size)
    )
    was_training = model.training
    model.eval()
    accumulated: dict[str, float] = {
        "normalized_mse": 0.0,
        "target_mse": 0.0,
        "q_error_norm": 0.0,
        "qd_error_norm": 0.0,
    }
    count = 0
    try:
        for start in range(0, window_count, batch_size):
            batch = dataset.get_batch(
                range(start, min(window_count, start + batch_size))
            )
            inputs, target, mask = encode_batch(batch, codec, device)
            prediction = model(inputs)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite evaluation prediction")
            n = int(mask.sum())
            accumulated["normalized_mse"] += (
                float(normalized_mse(prediction, target, _model_variance(model), mask))
                * n
            )
            accumulated["target_mse"] += (
                float((prediction - target).square()[mask].mean()) * n
            )
            q = batch["states"]["q"].to(device=device, dtype=torch.float32)
            qd = batch["states"]["qd"].to(device=device, dtype=torch.float32)
            next_q, next_qd = codec.decode(q[:, :-1], qd[:, :-1], prediction)
            q_error = next_q - q[:, 1:]
            periodic = codec.continuous_q_indices
            if periodic:
                q_error[..., list(periodic)] = torch.atan2(
                    torch.sin(q_error[..., list(periodic)]),
                    torch.cos(q_error[..., list(periodic)]),
                )
            if codec.robot == "ant":
                position = q_error[..., :3].norm(dim=-1)[mask].sum()
                dot = (
                    (
                        torch.nn.functional.normalize(next_q[..., 3:7], dim=-1)
                        * torch.nn.functional.normalize(q[:, 1:, 3:7], dim=-1)
                    )
                    .sum(-1)
                    .abs()
                    .clamp(max=1)
                )
                angle = (2 * torch.acos(dot))[mask].sum()
                accumulated["base_position_error"] = accumulated.get(
                    "base_position_error", 0.0
                ) + float(position)
                accumulated["base_orientation_error_rad"] = accumulated.get(
                    "base_orientation_error_rad", 0.0
                ) + float(angle)
                q_error = q_error[..., 7:]
            accumulated["q_error_norm"] += float(q_error.norm(dim=-1)[mask].sum())
            accumulated["qd_error_norm"] += float(
                (next_qd - qd[:, 1:]).norm(dim=-1)[mask].sum()
            )
            count += n
    finally:
        model.train(was_training)
    return {
        **{key: value / count for key, value in accumulated.items()},
        "valid_positions": count,
        "windows": window_count,
        "split": dataset.split,
    }


class DynamicsTrainer:
    """Train from immutable windows using an explicit committed sample cursor.

    Batches are sampled with replacement by a dedicated CPU generator. Only a
    completed optimizer step advances its durable state. External worker prefetch
    therefore cannot silently skip examples on resume. ``sample_order`` supplies
    a shared explicit index matrix for official-reference comparisons.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        codec: Any,
        train_dataset: DynamicsWindowDataset,
        validation_dataset: DynamicsWindowDataset,
        cfg: DynamicsTrainCfg,
        output_dir: str | Path,
        *,
        manifest: dict[str, Any],
        statistics: dict[str, Any] | None = None,
        initialize_normalization: bool = True,
        sample_order: Tensor | str | Path | None = None,
    ) -> None:
        cfg.validate()
        if (
            min(
                cfg.total_steps,
                cfg.batch_size,
                cfg.validation_interval,
                cfg.checkpoint_interval,
                cfg.lr_update_interval,
            )
            < 1
        ):
            raise ValueError("Training sizes and intervals must be positive")
        if cfg.validation_batches is not None and cfg.validation_batches < 1:
            raise ValueError("validation_batches must be positive")
        if cfg.lr_offset < 0:
            raise ValueError("lr_offset must be nonnegative")
        if (
            cfg.max_grad_norm <= 0
            or min(cfg.learning_rate, cfg.final_learning_rate) < 0
        ):
            raise ValueError("Invalid learning rate or gradient clipping norm")
        if train_dataset.split != "train" or validation_dataset.split != "validation":
            raise ValueError("Trainer requires train and validation splits")
        if train_dataset.fingerprint != validation_dataset.fingerprint:
            raise ValueError("Training and validation must share an immutable dataset")
        if not len(train_dataset) or not len(validation_dataset):
            raise ValueError("Training and validation need eligible windows")
        self.cfg, self.codec = cfg, codec
        self.model = model.to(cfg.device)
        self.train_dataset, self.validation_dataset = train_dataset, validation_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.global_step = 0
        self.best_validation = float("inf")
        self.sampler = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self._sample_order_identity: dict[str, Any] | None = None
        self.sample_order = self._prepare_order(sample_order)
        if train_dataset.history_length != model.cfg.history_length:
            raise ValueError("Dataset/model history length mismatch")
        if initialize_normalization:
            self.statistics = statistics or compute_statistics(
                train_dataset, codec, cfg.batch_size
            )
            self.model.set_normalization(
                {
                    key: (value["mean"], value["var"])
                    for key, value in self.statistics["inputs"].items()
                },
                self.statistics["output"]["mean"],
                self.statistics["output"]["var"],
            )
        else:
            self.statistics = statistics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

    def _prepare_order(
        self, source: Tensor | str | Path | None
    ) -> Tensor | np.ndarray | None:
        if source is None:
            return None
        if isinstance(source, (str, Path)):
            path = Path(source).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            order = np.load(path, mmap_mode="r", allow_pickle=False)
            if order.dtype.kind not in "iu":
                raise ValueError("Memory-mapped sample order requires integer indices")
            self._sample_order_identity = {
                "path": str(path),
                "sha256": digest.hexdigest(),
            }
        else:
            order = source.to(device="cpu", dtype=torch.long)
        if tuple(order.shape) != (self.cfg.total_steps, self.cfg.batch_size):
            raise ValueError("sample_order must have total_steps x batch_size indices")
        return order

    def _order_snapshot(self) -> Any:
        return (
            self._sample_order_identity
            if self._sample_order_identity is not None
            else self.sample_order
        )

    def _lr(self) -> float:
        interval = self.cfg.lr_update_interval
        ratio = (self.global_step // interval + self.cfg.lr_offset) / (
            self.cfg.total_steps / interval + self.cfg.lr_offset
        )
        return (
            self.cfg.learning_rate * (1 - ratio) + self.cfg.final_learning_rate * ratio
        )

    def step(self) -> dict[str, float | int]:
        """Commit one Adam update, returning loss, rate and unclipped norm."""
        if self.global_step >= self.cfg.total_steps:
            raise RuntimeError("Configured training budget exhausted")
        candidate_sampler = torch.Generator(device="cpu")
        candidate_sampler.set_state(self.sampler.get_state())
        indices = (
            self.sample_order[self.global_step]
            if self.sample_order is not None
            else torch.randint(
                len(self.train_dataset),
                (self.cfg.batch_size,),
                generator=candidate_sampler,
            )
        )
        if any(not 0 <= int(index) < len(self.train_dataset) for index in indices):
            raise ValueError("sample_order contains out-of-range dataset indices")
        batch = self.train_dataset.get_batch(indices)
        inputs, target, mask = encode_batch(batch, self.codec, self.cfg.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self._lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        prediction = self.model(inputs)
        loss = normalized_mse(prediction, target, _model_variance(self.model), mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite dynamics training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.sampler.set_state(candidate_sampler.get_state())
        self.global_step += 1
        return {
            "step": self.global_step,
            "loss": float(loss.detach()),
            "learning_rate": lr,
            "grad_norm": float(norm),
        }

    def save(self, path: str | Path) -> None:
        """Atomically save weights and exact committed training/RNG progress."""
        numpy_state = np.random.get_state()
        state = {
            "format": "embodichain.nerd.training.v1",
            "model_cfg": self.model.cfg.to_dict(),
            "model": self.model.state_dict(),
            "manifest": self.manifest,
            "train_cfg": self.cfg.to_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "scheduler": {"next_step": self.global_step},
            "best_validation": self.best_validation,
            "sampler_state": self.sampler.get_state(),
            "sample_order": self._order_snapshot(),
            "dataset_fingerprint": self.train_dataset.fingerprint,
            "window_contract": {
                "history_length": self.train_dataset.history_length,
                "include_prefix": self.train_dataset.include_prefix,
            },
            "statistics": self.statistics,
            "rng": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": (
                    numpy_state[0],
                    torch.from_numpy(numpy_state[1].astype(np.int64)),
                    numpy_state[2],
                    numpy_state[3],
                    numpy_state[4],
                ),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.device(self.cfg.device).type == "cuda"
                    else []
                ),
            },
        }
        _atomic_save(Path(path), state)

    def resume(self, path: str | Path) -> None:
        """Restore an identical run; changing its budget or data requires fine-tune."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "embodichain.nerd.training.v1":
            raise ValueError("Unsupported dynamics training checkpoint")
        for key, expected in (
            ("dataset_fingerprint", self.train_dataset.fingerprint),
            ("train_cfg", self.cfg.to_dict()),
            ("model_cfg", self.model.cfg.to_dict()),
            ("manifest", self.manifest),
            (
                "window_contract",
                {
                    "history_length": self.train_dataset.history_length,
                    "include_prefix": self.train_dataset.include_prefix,
                },
            ),
        ):
            if state[key] != expected:
                raise ValueError(
                    f"Resume {key} mismatch; use fine-tuning for a new run"
                )
        saved_order = state["sample_order"]
        if self.sample_order is not None:
            same = (
                self._sample_order_identity == saved_order
                if isinstance(saved_order, dict)
                else isinstance(saved_order, Tensor)
                and isinstance(self.sample_order, Tensor)
                and torch.equal(self.sample_order, saved_order)
            )
            if not same:
                raise ValueError("Resume sample order mismatch")
        if isinstance(saved_order, dict):
            self.sample_order = self._prepare_order(saved_order["path"])
            if self._sample_order_identity != saved_order:
                raise ValueError("Resume sample order checksum mismatch")
        else:
            self.sample_order = saved_order
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step, self.best_validation = (
            state["global_step"],
            state["best_validation"],
        )
        self.statistics = state["statistics"]
        self.sampler.set_state(state["sampler_state"])
        torch.set_rng_state(state["rng"]["torch"])
        random.setstate(state["rng"]["python"])
        name, keys, pos, gaussian, cached = state["rng"]["numpy"]
        np.random.set_state(
            (name, keys.numpy().astype(np.uint32), pos, gaussian, cached)
        )
        if state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])

    def train(self, *, stop_after: int | None = None) -> dict[str, Any]:
        """Run to the configured budget or an explicit interruption checkpoint."""
        end = (
            self.cfg.total_steps
            if stop_after is None
            else min(stop_after, self.cfg.total_steps)
        )
        if end < self.global_step:
            raise ValueError("stop_after precedes the committed cursor")
        last: dict[str, Any] = {}
        log_path = self.output_dir / "metrics.jsonl"
        # Drop records newer than a restored checkpoint before appending.
        if log_path.exists():
            temporary_log = log_path.with_suffix(".jsonl.pending")
            with log_path.open() as source, temporary_log.open("w") as target:
                for line in source:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break  # Interrupted trailing write is not a committed record.
                    if record["step"] <= self.global_step:
                        target.write(line)
            os.replace(temporary_log, log_path)
        with log_path.open("a") as log:
            while self.global_step < end:
                last = self.step()
                if (
                    self.global_step % self.cfg.validation_interval == 0
                    or self.global_step == self.cfg.total_steps
                ):
                    validation = evaluate_model(
                        self.model,
                        self.codec,
                        self.validation_dataset,
                        self.cfg.batch_size,
                        self.cfg.device,
                        max_batches=self.cfg.validation_batches,
                    )
                    last["validation"] = validation
                    score = float(validation["normalized_mse"])
                    if score < self.best_validation:
                        self.best_validation = score
                        self.save(self.output_dir / "best.pt")
                log.write(json.dumps(last, allow_nan=False) + "\n")
                log.flush()
                if self.global_step % self.cfg.checkpoint_interval == 0:
                    self.save(self.output_dir / "last.pt")
        self.save(self.output_dir / "last.pt")
        return {
            "global_step": self.global_step,
            "budget": self.cfg.total_steps,
            "status": (
                "completed"
                if self.global_step == self.cfg.total_steps
                else "interrupted"
            ),
            "best_validation": (
                self.best_validation if math.isfinite(self.best_validation) else None
            ),
            "last": last,
        }
