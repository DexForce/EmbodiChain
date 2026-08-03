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

"""Tests for shared optimizer and LR-scheduler builders."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from embodichain.learning.rl.utils import (
    LRSchedulerCfg,
    OptimizerCfg,
    build_lr_scheduler,
    build_optimizer,
    scheduler_needs_horizon,
)


def test_build_optimizer_adamw_kwargs() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(
        model.parameters(),
        OptimizerCfg(
            name="adamw",
            learning_rate=1e-3,
            kwargs={"weight_decay": 0.01},
        ),
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(1e-3)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.01)


def test_build_optimizer_rejects_lr_in_kwargs() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="OptimizerCfg.learning_rate"):
        build_optimizer(
            model.parameters(),
            OptimizerCfg(kwargs={"lr": 1e-2}),
        )


def test_build_optimizer_accepts_mapping_cfg() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(
        model.parameters(),
        {"name": "sgd", "learning_rate": 0.05, "kwargs": {"momentum": 0.9}},
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == pytest.approx(0.9)


def test_linear_and_cosine_need_horizon_until_bound() -> None:
    assert scheduler_needs_horizon(LRSchedulerCfg(name="linear"))
    assert scheduler_needs_horizon(LRSchedulerCfg(name="cosine"))
    assert not scheduler_needs_horizon(
        LRSchedulerCfg(name="linear", kwargs={"total_iters": 10})
    )
    assert not scheduler_needs_horizon(LRSchedulerCfg(name=None))


def test_build_lr_scheduler_linear_decays() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(
        model.parameters(),
        OptimizerCfg(learning_rate=1.0),
    )
    scheduler = build_lr_scheduler(
        optimizer,
        LRSchedulerCfg(
            name="linear",
            kwargs={"total_iters": 4, "start_factor": 1.0, "end_factor": 0.0},
        ),
    )
    assert scheduler is not None
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.0)
    assert rates[1] < rates[0]


def test_build_lr_scheduler_cosine_requires_t_max() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model.parameters(), OptimizerCfg())
    with pytest.raises(ValueError, match="T_max"):
        build_lr_scheduler(optimizer, LRSchedulerCfg(name="cosine"))
