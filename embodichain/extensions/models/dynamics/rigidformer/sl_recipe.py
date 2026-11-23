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

"""Installable author MOVi recipe using the shared supervised trainer."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TYPE_CHECKING
import torch
from .model import RigidFormerCfg, RigidFormerModel, RIGIDFORMER_REVISION
from .data import MOViSequenceSource
from .ema import RigidFormerEMA
from .supervised import RigidFormerObjective

if TYPE_CHECKING:
    from embodichain.learning.sl import SupervisedTrainer
__all__ = ["build_rigidformer_trainer"]


class _EpochSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        steps: int,
        config: Mapping[str, Any],
        max_epochs: int,
    ) -> None:
        self.steps, self.num_updates = steps, 0
        self.policy = {
            "scheduler": dict(config),
            "steps_per_epoch": steps,
            "max_epochs": max_epochs,
        }
        warmup = int(config.get("warmup_epochs", 0))
        if config.get("type", "cosine") == "constant":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        elif config.get("type", "cosine") == "cosine":
            # Match the author's construction order as well as the LR curve:
            # Cosine records its initial rates after LinearLR's initialization.
            linear = None
            if warmup:
                linear = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=float(config.get("start_factor", 0.1)),
                    total_iters=warmup,
                )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs - warmup),
                eta_min=float(config.get("eta_min", config.get("min_lr", 1e-6))),
            )
            if linear is not None:
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [linear, cosine], milestones=[warmup]
                )
            else:
                self.scheduler = cosine
        else:
            raise ValueError(
                "Only the released constant/cosine epoch schedules are supported"
            )

    def on_update(self, learner: Any) -> None:
        self.num_updates = learner.num_updates
        if self.num_updates % self.steps == 0:
            self.scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": deepcopy(self.policy),
            "num_updates": self.num_updates,
            "scheduler": deepcopy(self.scheduler.state_dict()),
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("policy") != self.policy
            or type(state.get("num_updates")) is not int
            or state["num_updates"] < 0
        ):
            raise ValueError("Author epoch schedule differs")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.num_updates = state["num_updates"]
        self.scheduler.load_state_dict(state["scheduler"])


def _optimizer(model: RigidFormerModel, cfg: Mapping[str, Any]) -> torch.optim.AdamW:
    if cfg.get("type", "AdamW").lower() != "adamw":
        raise ValueError("The author recipe requires AdamW")
    lr = float(cfg.get("lr", 1e-4))
    buckets: dict[float, list[torch.nn.Parameter]] = {1.0: []}
    for name, param in model.network.named_parameters():
        if not param.requires_grad:
            continue
        scale = 1.0
        if any(
            k in name
            for k in (
                "predictor.fusion_norm1",
                "predictor.fusion_w1",
                "predictor.fusion_g1",
            )
        ):
            scale = float(cfg.get("fusion_lr_mult", 10.0))
        elif "shared_obj_encoder" in name:
            scale = float(cfg.get("encoder_lr_mult", 1.0))
        elif name.startswith(("decoder.", "predictor.")):
            scale = float(cfg.get("decoder_lr_mult", 1.0))
        buckets.setdefault(scale, []).append(param)
    embeddings = {
        id(p) for n, p in model.network.named_parameters() if "register_tokens" in n
    }
    groups = []
    for scale, params in buckets.items():
        if not params:
            continue
        if cfg.get("decay_matrices_only", True):
            matrices = [p for p in params if p.ndim >= 2 and id(p) not in embeddings]
            vectors = [p for p in params if p.ndim < 2 or id(p) in embeddings]
            if matrices:
                groups.append({"params": matrices, "lr": lr * scale})
            if vectors:
                groups.append(
                    {"params": vectors, "lr": lr * scale, "weight_decay": 0.0}
                )
        else:
            groups.append({"params": params, "lr": lr * scale})
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=tuple(cfg.get("betas", (0.9, 0.999))),
        eps=float(cfg.get("eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )


def build_rigidformer_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Build an author-based MOVi recipe without importing example helpers.

    Args:
        config: Model settings; prepared data root, split_file and scale_file;
            optional author training overrides; common trainer configuration.
            Relative paths resolve from the invoking working directory.

    Returns:
        Shared trainer with committed sampling, epoch schedule and FP32 EMA.
    """
    from embodichain.learning.sl import (
        SupervisedLearner,
        SupervisedTrainer,
        SupervisedTrainerCfg,
    )
    from ._native.data.gpu_augmentation import GPUAugmentation

    model_cfg = RigidFormerCfg(**config.get("model", {}))
    reference = model_cfg.reference_config()
    training = {**reference["training"], **config.get("training", {})}
    for key in ("optimizer", "scheduler", "loss_weights", "ema"):
        training[key] = {
            **reference["training"].get(key, {}),
            **config.get("training", {}).get(key, {}),
        }
    if training.get("scheduled_sampling", {}).get("enabled", False):
        raise ValueError(
            "The released MOVi recipe uses teacher forcing; scheduled sampling is not part of this adapter"
        )
    raw_data = dict(config["data"])
    sampling = {**reference["data"], **raw_data.pop("sampling", {})}
    if sampling.get("anchor_k", 4) != reference["data"].get("anchor_k", 4):
        raise ValueError("data.sampling.anchor_k must match the model architecture")
    seed = int(config.get("trainer", {}).get("seed", 42))
    source = MOViSequenceSource(
        raw_data.pop("root"),
        raw_data.pop("split_file"),
        raw_data.pop("scale_file"),
        split=raw_data.pop("split", "train"),
        data_config=sampling,
        seed=seed,
        **raw_data,
    )
    kwargs = {
        "device": "cuda",
        "batch_size": int(training["train_batch_size"]),
        **config.get("trainer", {}),
    }
    steps = source.updates_per_epoch(kwargs["batch_size"])
    kwargs.setdefault("total_updates", int(training["max_epochs"]) * steps)
    trainer_cfg = SupervisedTrainerCfg(**kwargs)
    torch.manual_seed(seed)
    model = RigidFormerModel(model_cfg).to(trainer_cfg.device)
    if source.world_size > 1:
        model.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.network)
    optimizer = _optimizer(model, training["optimizer"])
    augmentation_cfg = deepcopy(reference)
    augmentation_cfg["data"] = sampling
    objective = RigidFormerObjective(
        {
            name: training["loss_weights"][name]
            for name in (
                "position_before_icp",
                "position_after_icp",
                "acceleration_before_icp",
                "acceleration_after_icp",
            )
        },
        time_step_scale=training["loss_weights"].get("if_time_step_scale", True),
        augmentation=GPUAugmentation(
            augmentation_cfg, torch.device(trainer_cfg.device)
        ),
    )
    precision = config.get("precision", "bf16")
    if precision not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    learner = SupervisedLearner(
        model,
        objective,
        optimizer,
        max_grad_norm=float(training.get("gradient_clip", 1.0)),
        amp_dtype=torch.bfloat16 if precision == "bf16" else None,
        ddp_find_unused_parameters=True,
        ddp_broadcast_buffers=True,
    )
    schedule = _EpochSchedule(
        optimizer, steps, training["scheduler"], int(training["max_epochs"])
    )
    states: dict[str, Any] = {"geometry_source": source, "epoch_schedule": schedule}
    if training["ema"].get("enable", True):
        states["ema"] = RigidFormerEMA(
            model, float(training["ema"].get("decay", 0.9999))
        )
    return SupervisedTrainer(
        learner,
        source,
        trainer_cfg,
        signature={
            "family": "rigidformer-author-v1",
            "author_revision": RIGIDFORMER_REVISION,
            "architecture": reference["model"],
            "reference_config": reference,
            "sampling": sampling,
            "training": training,
            "precision": precision,
            "attention_backend": model_cfg.attention_backend,
            "steps_per_epoch": steps,
            "ddp_broadcast_buffers": True,
        },
        training_states=states,
    )
