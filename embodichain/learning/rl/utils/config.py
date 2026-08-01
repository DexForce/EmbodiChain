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

from __future__ import annotations

from typing import Any

from embodichain.utils import configclass

__all__ = ["AlgorithmCfg", "LRSchedulerCfg", "OptimizerCfg"]


@configclass
class OptimizerCfg:
    """Policy optimizer configuration."""

    name: str = "adam"
    learning_rate: float = 3e-4
    kwargs: dict[str, Any] = dict()


@configclass
class LRSchedulerCfg:
    """Optional LR scheduler. ``name=None`` disables scheduling.

    Horizon keys (``total_iters`` / ``T_max``) may be omitted and bound later by
    ``BaseAlgorithm.bind_schedule``.
    """

    name: str | None = None
    kwargs: dict[str, Any] = dict()


@configclass
class AlgorithmCfg:
    """Shared fields for RL algorithm configs."""

    device: str = "cuda"
    optimizer: OptimizerCfg = OptimizerCfg()
    lr_scheduler: LRSchedulerCfg = LRSchedulerCfg()
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
