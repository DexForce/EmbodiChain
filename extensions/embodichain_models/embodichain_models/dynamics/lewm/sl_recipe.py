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

"""Installable single-device LeWM recipe for the common SL trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from embodichain.learning.sl import (
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)

from .data import LeWMSequenceSource
from .model import LeWM, LeWMCfg
from .supervised import LeWMObjective

__all__ = ["build_lewm_trainer"]


def build_lewm_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Construct a simulation-independent LeWM training run.

    Args:
        config: Mapping containing model, data, objective, optimizer and
            trainer sections. ``data.train_path`` and ``data.validation_path``
            must reference disjoint LeWM HDF5 splits with matching contracts.

    Returns:
        Shared supervised trainer with train-only action normalization.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
        raise ValueError("LeWM SIGReg currently supports only one training process")
    trainer_cfg = SupervisedTrainerCfg(**dict(config.get("trainer", {})))
    trainer_cfg.validate()
    if trainer_cfg.total_updates < 1 or trainer_cfg.batch_size < 2:
        raise ValueError(
            "LeWM needs positive updates and at least two examples per batch"
        )
    torch.manual_seed(trainer_cfg.seed)
    model_cfg = LeWMCfg(**dict(config.get("model", {})))
    model = LeWM(model_cfg)
    data = config["data"]
    train = LeWMSequenceSource(data["train_path"], "train", model_cfg.history_size)
    validation = LeWMSequenceSource(
        data["validation_path"], "validation", model_cfg.history_size
    )
    for key in (
        "action_dim",
        "action_semantics",
        "step_dt",
        "action_scale",
        "action_clip",
        "camera",
        "environment",
        "physics",
    ):
        if train.metadata.get(key) != validation.metadata.get(key):
            raise ValueError(f"training and validation {key} differ")
    if train.seeds & validation.seeds:
        raise ValueError("training and validation episode seeds overlap")
    if train.metadata["action_dim"] != model_cfg.action_dim:
        raise ValueError("corpus action width differs from the model")
    model.set_action_statistics(*train.action_statistics())
    model.to(trainer_cfg.device)
    objective_cfg = dict(config.get("objective", {}))
    objective = LeWMObjective(**objective_cfg)
    optim_cfg = dict(config.get("optimizer", {}))
    lr = optim_cfg.pop("learning_rate", 5e-5)
    wd = optim_cfg.pop("weight_decay", 1e-3)
    clip = optim_cfg.pop("max_grad_norm", 1.0)
    use_bf16 = optim_cfg.pop("amp_bfloat16", False)
    if optim_cfg:
        raise ValueError(f"unknown optimizer options: {sorted(optim_cfg)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    learner = SupervisedLearner(
        model,
        objective,
        optimizer,
        max_grad_norm=clip,
        amp_dtype=torch.bfloat16 if use_bf16 else None,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=trainer_cfg.total_updates
    )

    def evaluate(current: SupervisedLearner) -> Mapping[str, float]:
        totals: dict[str, float] = {}
        count = 0.0
        current.model.eval()
        with torch.no_grad():
            for batch in validation.iter_batches(trainer_cfg.batch_size):
                result = objective.compute_loss(
                    current.model, batch.to(trainer_cfg.device)
                )
                for key, value in {"loss": result.loss, **result.metrics}.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * result.weight
                count += result.weight
        return {key: value / count for key, value in totals.items()}

    return SupervisedTrainer(
        learner,
        train,
        trainer_cfg,
        scheduler=scheduler,
        evaluate=evaluate,
        signature={
            "recipe": "lewm-v1",
            "model": model_cfg.to_dict(),
            "data_contract": train.metadata,
            "validation_source": validation.source_id,
            "train_seeds": sorted(train.seeds),
            "validation_seeds": sorted(validation.seeds),
            "objective": objective_cfg,
            "optimizer": dict(config.get("optimizer", {})),
        },
    )
