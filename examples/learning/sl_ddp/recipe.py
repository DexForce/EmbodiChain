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

"""A fixed masked-regression recipe for single-device or distributed SL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from embodichain.learning.sl import (
    LossResult,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
    TensorBatchSource,
)

__all__ = ["build_trainer"]


class _MaskedMSE:
    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Return a local mean weighted by the number of valid scalar targets."""
        predictions = model(batch.inputs)
        mask = batch.masks["targets"].to(dtype=predictions.dtype)
        weight = mask.sum()
        loss = ((predictions - batch.targets).square() * mask).sum() / weight
        return LossResult(loss, {"mse": loss.detach()}, weight=float(weight.detach()))


def _corpus(size: int, *, seed: int) -> SupervisedBatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(size, 4, generator=generator)
    targets = torch.stack(
        (
            inputs[:, 0] + 0.5 * inputs[:, 1] - 0.25 * inputs[:, 2],
            inputs[:, 0].sin() + inputs[:, 3],
        ),
        dim=-1,
    )
    # Every row retains one supervised scalar. The second varies by sample,
    # so equally sized rank-local batches may have different valid weights.
    mask = torch.stack((torch.ones(size, dtype=torch.bool), inputs[:, 0] > 0), dim=-1)
    return SupervisedBatch(
        inputs,
        targets,
        masks={"targets": mask},
        metadata={"features": 4, "targets": 2, "units": "dimensionless"},
    )


def build_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Build a tiny recipe using the same factory for one or multiple workers.

    Args:
        config: Mapping containing data counts/seeds, hidden width/model seed,
            AdamW options and the shared trainer configuration. The SL workflow
            resolves the trainer device before calling this factory.

    Returns:
        A trainer with a fixed CPU corpus and an independent validation corpus.
    """
    data = {
        "train_count": 1024,
        "validation_count": 128,
        "seed": 17,
        **config.get("data", {}),
    }
    model_config = {"hidden_dim": 32, "seed": 29, **config.get("model", {})}
    optimizer_config = {"lr": 1e-3, "weight_decay": 0.0, **config.get("optimizer", {})}
    cfg = SupervisedTrainerCfg(**config.get("trainer", {}))
    training = _corpus(data["train_count"], seed=data["seed"])
    validation = _corpus(data["validation_count"], seed=data["seed"] + 1)
    source = TensorBatchSource(training, source_id="masked-regression-v1")
    torch.manual_seed(model_config["seed"])
    model = nn.Sequential(
        nn.Linear(4, model_config["hidden_dim"]),
        nn.Tanh(),
        nn.Linear(model_config["hidden_dim"], 2),
    ).to(cfg.device)
    objective = _MaskedMSE()
    learner = SupervisedLearner(
        model,
        objective,
        torch.optim.AdamW(model.parameters(), **optimizer_config),
    )

    def evaluate(current: SupervisedLearner) -> Mapping[str, float]:
        result = objective.compute_loss(current.model, validation.to(cfg.device))
        return {"mse": float(result.loss)}

    return SupervisedTrainer(
        learner,
        source,
        cfg,
        signature={
            "recipe": "masked-regression-v1",
            "model": model_config,
            "data": data,
            "optimizer": optimizer_config,
            "objective": "masked-mse-per-valid-scalar-v1",
        },
        evaluate=evaluate,
    )
