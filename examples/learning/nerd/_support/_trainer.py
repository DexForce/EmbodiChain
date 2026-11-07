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
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from embodichain.utils import configclass
from embodichain_models.dynamics.nerd.data import DynamicsWindowDataset
from embodichain.learning.sl import SupervisedLearner
from embodichain.learning.sl._artifacts import _RunArtifacts
from embodichain_models.dynamics.nerd.supervised import NeRDObjective, make_nerd_batch
from ._sample_order import _CheckpointedOrder, _OfficialShuffleOrder, _open_order
from embodichain_models.dynamics.nerd._numerics import (
    _attention_context,
    _checkpoint_numerical_profile,
    _clip_grad_norm_,
    _validate_numerical_profile,
)

from embodichain_models.dynamics.nerd.integration import (
    _PREPROCESSING_EXECUTION,
    encode_batch,
    normalized_mse,
    compute_statistics,
    save_statistics,
    load_statistics,
    _atomic_save,
    _model_variance,
    evaluate_model,
)

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

    ``numerical_profile="reference"`` selects math attention and an ordered
    FP64 reduction of FP32 per-parameter gradient norms. The default retains
    Torch's gradient clipping and the model's configured attention backend.
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
    log_interval: int = 1
    resident_cache_bytes: int = 0
    numerical_profile: str = "default"


def _capture_rng_state(device: str) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
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
            if torch.device(device).type == "cuda"
            else []
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, gaussian, cached))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_cfg_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare legal head-width sequences without changing feature order."""
    identity = dict(cfg)
    if isinstance(identity.get("head_hidden_dims"), (tuple, list)):
        identity["head_hidden_dims"] = list(identity["head_hidden_dims"])
    return identity


class DynamicsTrainer:
    """Train from immutable windows using an explicit committed sample cursor.

    Batches are sampled with replacement by a dedicated CPU generator. Only a
    completed optimizer step advances its durable state. External worker prefetch
    therefore cannot silently skip examples on resume. ``sample_order`` supplies
    a shared explicit index matrix for official-reference comparisons.

    Explicit statistics must identify the training dataset even when
    ``initialize_normalization=False``. That flag leaves existing model
    normalizers untouched; it does not waive statistics provenance.
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
        _validate_numerical_profile(cfg.numerical_profile)
        if (
            min(
                cfg.total_steps,
                cfg.batch_size,
                cfg.validation_interval,
                cfg.checkpoint_interval,
                cfg.lr_update_interval,
                cfg.log_interval,
            )
            < 1
        ):
            raise ValueError("Training sizes and intervals must be positive")
        if cfg.validation_batches is not None and cfg.validation_batches < 1:
            raise ValueError("validation_batches must be positive")
        if cfg.lr_offset < 0:
            raise ValueError("lr_offset must be nonnegative")
        if type(cfg.resident_cache_bytes) is not int or cfg.resident_cache_bytes < 0:
            raise ValueError("resident_cache_bytes must be nonnegative integer bytes")
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
        if statistics is not None and (
            statistics.get("dataset_fingerprint") != train_dataset.fingerprint
        ):
            raise ValueError("NeRD statistics/dataset fingerprint mismatch")
        self.cfg, self.codec = cfg, codec
        self.model = model.to(cfg.device)
        if cfg.numerical_profile == "reference":
            self.model.cfg.attention_backend = "math"
        self.train_dataset, self.validation_dataset = train_dataset, validation_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = _RunArtifacts(self.output_dir)
        self.manifest = manifest
        self.global_step = 0
        self.best_validation = float("inf")
        self._best_model: dict[str, Tensor] | None = None
        self._best_step: int | None = None
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
        # The original trainer preprocesses one real window while constructing
        # its integrator/model metadata. Match that JIT warmup before updates;
        # cold and optimized CUDA coordinate graphs can round differently.
        encode_batch(train_dataset.get_batch([0]), self.codec, cfg.device)
        if cfg.resident_cache_bytes:
            used = train_dataset.enable_device_cache(
                cfg.device, cfg.resident_cache_bytes
            )
            validation_dataset.enable_device_cache(
                cfg.device, cfg.resident_cache_bytes - used
            )
        if torch.device(cfg.device).type == "cuda":
            # Specialize the real batch layout before committing updates. This
            # makes the first step after resume use the same CUDA graph as an
            # uninterrupted run, without consuming sampling/RNG or model state.
            warm_batch = train_dataset.get_batch([0] * cfg.batch_size)
            with torch.enable_grad():
                for _ in range(3):
                    encode_batch(warm_batch, self.codec, cfg.device)
            del warm_batch
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.learner = SupervisedLearner(
            self.model,
            NeRDObjective(include_prefix=train_dataset.include_prefix),
            self.optimizer,
            max_grad_norm=cfg.max_grad_norm,
            clip_grad=lambda parameters, max_norm: _clip_grad_norm_(
                parameters,
                max_norm,
                profile=cfg.numerical_profile,
                error_if_nonfinite=cfg.numerical_profile == "default",
            ),
            numerical_context=lambda: _attention_context(cfg.numerical_profile),
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

    def _prepare_order(
        self, source: Tensor | str | Path | None
    ) -> Tensor | np.ndarray | _CheckpointedOrder | _OfficialShuffleOrder | None:
        order, identity = self._load_order(source)
        self._sample_order_identity = identity
        return order

    def _load_order(self, source: Tensor | str | Path | None) -> tuple[
        Tensor | np.ndarray | _CheckpointedOrder | _OfficialShuffleOrder | None,
        dict[str, Any] | None,
    ]:
        if source is None:
            return None, None
        identity = None
        if isinstance(source, (str, Path)):
            path = Path(source).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            order = _open_order(path)
            if isinstance(
                order, (_CheckpointedOrder, _OfficialShuffleOrder)
            ) and order.dataset_size != len(self.train_dataset):
                raise ValueError("Sample-order dataset size mismatch")
            identity = {
                "path": str(path),
                "sha256": digest.hexdigest(),
            }
        else:
            order = source.to(device="cpu", dtype=torch.long)
        if tuple(order.shape) != (self.cfg.total_steps, self.cfg.batch_size):
            raise ValueError("sample_order must have total_steps x batch_size indices")
        return order, identity

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
        self._artifacts.claim()
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
        supervised_batch = make_nerd_batch(batch, self.codec, self.cfg.device)
        lr = self._lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        metrics = self.learner.update(supervised_batch)
        self.sampler.set_state(candidate_sampler.get_state())
        self.global_step += 1
        return {
            "step": self.global_step,
            "loss": metrics["loss"],
            "learning_rate": lr,
            "grad_norm": metrics["grad_norm"],
        }

    def save(self, path: str | Path) -> None:
        """Atomically save weights and exact committed training/RNG progress."""
        self._artifacts.claim()
        self._ensure_best_snapshot()
        state = {
            "format": "embodichain.nerd.training.v1",
            "run_id": self._artifacts.run_id,
            "preprocessing_execution": _PREPROCESSING_EXECUTION,
            "model_cfg": self.model.cfg.to_dict(),
            "model": self.model.state_dict(),
            "manifest": self.manifest,
            "train_cfg": self.cfg.to_dict(),
            "numerical_profile": self.cfg.numerical_profile,
            "attention_backend": self.model.cfg.attention_backend,
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "scheduler": {"next_step": self.global_step},
            "best_validation": self.best_validation,
            "best_model": self._best_model,
            "best_step": self._best_step,
            "sampler_state": self.sampler.get_state(),
            "sample_order": self._order_snapshot(),
            "dataset_fingerprint": self.train_dataset.fingerprint,
            "window_contract": {
                "history_length": self.train_dataset.history_length,
                "include_prefix": self.train_dataset.include_prefix,
            },
            "statistics": self.statistics,
            "rng": _capture_rng_state(self.cfg.device),
        }
        _atomic_save(Path(path), state)

    def resume(self, path: str | Path) -> None:
        """Restore an identical run; changing its budget or data requires fine-tune."""
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "embodichain.nerd.training.v1":
            raise ValueError("Unsupported dynamics training checkpoint")
        if state.get("resumable") is False:
            raise ValueError(
                "Best-model snapshots support export only; resume from last.pt"
            )
        if state.get("preprocessing_execution") != _PREPROCESSING_EXECUTION:
            raise ValueError(
                "Resume preprocessing execution mismatch; use matching code or fine-tuning"
            )
        profile = _checkpoint_numerical_profile(state)
        if profile != self.cfg.numerical_profile:
            raise ValueError("Resume numerical_profile mismatch; use fine-tuning")
        saved_train_cfg = {**state["train_cfg"], "numerical_profile": profile}
        saved_model_cfg = {"attention_backend": "default", **state["model_cfg"]}
        for key, expected in (
            ("dataset_fingerprint", self.train_dataset.fingerprint),
            ("train_cfg", self.cfg.to_dict()),
            ("model_cfg", _model_cfg_identity(self.model.cfg.to_dict())),
            ("manifest", self.manifest),
            (
                "window_contract",
                {
                    "history_length": self.train_dataset.history_length,
                    "include_prefix": self.train_dataset.include_prefix,
                },
            ),
        ):
            actual = (
                saved_train_cfg
                if key == "train_cfg"
                else (
                    _model_cfg_identity(saved_model_cfg)
                    if key == "model_cfg"
                    else state[key]
                )
            )
            if actual != expected:
                raise ValueError(
                    f"Resume {key} mismatch; use fine-tuning for a new run"
                )
        if list(saved_model_cfg["input_dims"].items()) != list(
            self.model.cfg.input_dims.items()
        ):
            raise ValueError("Resume model_cfg input_dims order mismatch")
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
            order, order_identity = self._load_order(saved_order["path"])
            if order_identity != saved_order:
                raise ValueError("Resume sample order checksum mismatch")
        else:
            order, order_identity = self._load_order(saved_order)
        step = state["global_step"]
        if (
            type(step) is not int
            or not 0 <= step <= self.cfg.total_steps
            or state["scheduler"] != {"next_step": step}
        ):
            raise ValueError("Resume committed step or scheduler mismatch")
        self._validate_model_state(state["model"], "model")
        groups = state["optimizer"].get("param_groups", [])
        if len(groups) != len(self.optimizer.param_groups) or any(
            len(saved.get("params", [])) != len(current["params"])
            for saved, current in zip(groups, self.optimizer.param_groups)
        ):
            raise ValueError("Resume optimizer parameter groups mismatch")
        sampler = torch.Generator(device="cpu")
        sampler.set_state(state["sampler_state"])
        best_model, best_step = self._checkpoint_best(state, path)
        run_id = self._artifacts.validate_resume(state.get("run_id"))
        self._artifacts.validate_history(step, key="step")
        self._validate_destination_best(state, best_model, best_step, run_id)
        # Directory identity, order checksums and every semantic contract are
        # admitted before changing model, optimizer, sampling or global RNG.
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step, self.best_validation = (
            state["global_step"],
            state["best_validation"],
        )
        self.statistics = state["statistics"]
        self.sample_order, self._sample_order_identity = order, order_identity
        self._best_model, self._best_step = best_model, best_step
        self.learner.num_updates = self.global_step
        self.sampler.set_state(state["sampler_state"])
        _restore_rng_state(state["rng"])
        self._artifacts.adopt_resume(run_id)

    def _validate_model_state(self, saved: dict[str, Tensor], label: str) -> None:
        current = self.model.state_dict()
        if current.keys() != saved.keys() or any(
            not isinstance(saved[key], Tensor)
            or saved[key].shape != value.shape
            or saved[key].dtype != value.dtype
            for key, value in current.items()
        ):
            raise ValueError(f"Resume {label} tensor layout mismatch")

    def _checkpoint_best(
        self, state: dict[str, Any], path: Path
    ) -> tuple[dict[str, Tensor] | None, int | None]:
        best_model, best_step = state.get("best_model"), state.get("best_step")
        score = state["best_validation"]
        if not math.isfinite(score):
            if score != float("inf") or best_model is not None or best_step is not None:
                raise ValueError("Resume best-model metadata mismatch")
            return None, None
        if best_model is None:
            # Legacy checkpoints stored the score but kept best weights only in
            # the sibling file. Read and validate that source without touching
            # the destination; an absent best model cannot be invented.
            best_path = path.with_name("best.pt")
            if not best_path.is_file():
                raise ValueError(
                    "Resume needs matching best.pt for legacy best weights"
                )
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            best_cfg = {"attention_backend": "default", **best["model_cfg"]}
            if (
                best.get("format") != "embodichain.nerd.training.v1"
                or _model_cfg_identity(best_cfg)
                != _model_cfg_identity(self.model.cfg.to_dict())
                or list(best_cfg["input_dims"].items())
                != list(self.model.cfg.input_dims.items())
                or best["manifest"] != state["manifest"]
                or best["dataset_fingerprint"] != state["dataset_fingerprint"]
                or best["best_validation"] != score
                or best.get("run_id") != state.get("run_id")
            ):
                raise ValueError("Resume legacy best.pt identity mismatch")
            best_model, best_step = best["model"], best["global_step"]
        if type(best_step) is not int or not 0 <= best_step <= state["global_step"]:
            raise ValueError("Resume best-model step mismatch")
        self._validate_model_state(best_model, "best model")
        return {
            key: value.detach().to(device="cpu", copy=True)
            for key, value in best_model.items()
        }, best_step

    def _write_best_snapshot(self) -> None:
        assert self._best_model is not None and self._best_step is not None
        _atomic_save(
            self.output_dir / "best.pt",
            {
                "format": "embodichain.nerd.training.v1",
                "resumable": False,
                "run_id": self._artifacts.run_id,
                "model_cfg": self.model.cfg.to_dict(),
                "model": self._best_model,
                "manifest": self.manifest,
                "global_step": self._best_step,
                "dataset_fingerprint": self.train_dataset.fingerprint,
                "numerical_profile": self.cfg.numerical_profile,
                "attention_backend": self.model.cfg.attention_backend,
                "best_validation": self.best_validation,
            },
        )

    def _validate_destination_best(
        self,
        state: dict[str, Any],
        best_model: dict[str, Tensor] | None,
        best_step: int | None,
        run_id: str,
    ) -> None:
        path = self.output_dir / "best.pt"
        if not path.exists():
            return
        try:
            best = torch.load(path, map_location="cpu", weights_only=True)
            best_cfg = {"attention_backend": "default", **best["model_cfg"]}
            if (
                best_model is None
                or best.get("format") != "embodichain.nerd.training.v1"
                or best.get("run_id") != run_id
                or best["global_step"] != best_step
                or best["best_validation"] != state["best_validation"]
                or _model_cfg_identity(best_cfg)
                != _model_cfg_identity(self.model.cfg.to_dict())
                or list(best_cfg["input_dims"].items())
                != list(self.model.cfg.input_dims.items())
                or best["manifest"] != state["manifest"]
                or best["dataset_fingerprint"] != state["dataset_fingerprint"]
            ):
                raise ValueError("Best selection boundary differs")
            self._validate_model_state(best["model"], "destination best model")
            if any(
                not torch.equal(best["model"][key], value)
                for key, value in best_model.items()
            ):
                raise ValueError("Best selection weights differ")
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Resume destination best.pt differs from this checkpoint; "
                "use a fresh output directory to preserve existing best weights"
            ) from error

    def _ensure_best_snapshot(self) -> None:
        if self._best_model is not None and not (self.output_dir / "best.pt").exists():
            self._write_best_snapshot()

    def train(self, *, stop_after: int | None = None) -> dict[str, Any]:
        """Run to the configured budget or an explicit interruption checkpoint."""
        end = (
            self.cfg.total_steps
            if stop_after is None
            else min(stop_after, self.cfg.total_steps)
        )
        if end < self.global_step:
            raise ValueError("stop_after precedes the committed cursor")
        self._artifacts.claim()
        self._artifacts.validate_history(self.global_step, key="step")
        self._ensure_best_snapshot()
        last: dict[str, Any] = {}
        log_path = self.output_dir / "metrics.jsonl"
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
                        self._best_model = {
                            key: value.detach().to(device="cpu", copy=True)
                            for key, value in self.model.state_dict().items()
                        }
                        self._best_step = self.global_step
                        self._write_best_snapshot()
                if (
                    self.global_step % self.cfg.log_interval == 0
                    or self.global_step == end
                    or "validation" in last
                ):
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
