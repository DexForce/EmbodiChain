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

"""Private CPU recipe for testing real torchrun lifecycle and resume."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from embodichain.learning.sl import (
    LossResult,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
    TensorBatchSource,
)


class _Regression:
    def compute_loss(
        self, model: torch.nn.Module, batch: SupervisedBatch
    ) -> LossResult:
        error = (model(batch.inputs) - batch.targets).square()
        return LossResult(error.mean(), weight=error.numel())


class _BoundedTrainer(SupervisedTrainer):
    boundary: int | None = None

    def fit(self, *, stop_after: int | None = None) -> dict[str, Any]:
        return super().fit(
            stop_after=self.boundary if stop_after is None else stop_after
        )


def build(config: dict[str, Any]) -> SupervisedTrainer:
    """Construct a tiny learner only after the production workflow initialized DDP."""
    initialized = dist.is_initialized()
    assert initialized, "torchrun workflow must initialize the process group"
    rank = dist.get_rank()
    target = Path(config["test_audit_dir"]) / f"{config['test_phase']}-rank{rank}.json"

    def audit_exit() -> None:
        target.write_text(
            json.dumps(
                {
                    "initialized_in_factory": initialized,
                    "initialized_at_exit": dist.is_initialized(),
                }
            )
        )

    atexit.register(audit_exit)
    torch.manual_seed(31)
    model = torch.nn.Linear(2, 1)
    values = torch.arange(16, dtype=torch.float32).view(8, 2) / 16
    source = TensorBatchSource(SupervisedBatch(values, values.sum(-1, keepdim=True)))
    learner = SupervisedLearner(
        model, _Regression(), torch.optim.Adam(model.parameters(), lr=0.02)
    )
    trainer = _BoundedTrainer(
        learner, source, SupervisedTrainerCfg(**config["trainer"])
    )
    trainer.boundary = config["test_stop_after"]
    return trainer
