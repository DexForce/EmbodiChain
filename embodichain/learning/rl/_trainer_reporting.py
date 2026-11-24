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

"""Scalar reporting shared by standard and differentiable training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import wandb

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__all__: list[str] = []


def _write_scalars(
    metrics: Mapping[str, float],
    *,
    step: int,
    writer: SummaryWriter | None,
    use_wandb: bool,
    wandb_metrics: Mapping[str, float] | None = None,
) -> None:
    """Write prepared scalars; the caller owns rank and filtering decisions."""
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
    if use_wandb:
        wandb.log(dict(metrics if wandb_metrics is None else wandb_metrics), step=step)


def _evaluation_improved(
    candidate: float | None,
    best_value: float | None,
    mode: str,
) -> bool:
    """Compare a finite evaluation score without changing trainer state."""
    if candidate is None or not math.isfinite(candidate):
        return False
    return best_value is None or (
        candidate > best_value if mode == "max" else candidate < best_value
    )
