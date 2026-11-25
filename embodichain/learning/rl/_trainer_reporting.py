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
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import wandb

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__all__: list[str] = []


def _write_tensorboard_scalars(
    writer: SummaryWriter,
    metrics: Mapping[str, float],
    step: int,
    *,
    ignore_errors: bool = False,
) -> None:
    """Use native batching for plain payloads, retaining scalar-call fallback."""
    if type(metrics) is dict and not metrics:
        return
    writer_module = sys.modules.get("torch.utils.tensorboard.writer")
    if (
        writer_module is not None
        and type(writer) is writer_module.SummaryWriter
        and "add_scalar" not in writer.__dict__
        and type(metrics) is dict
        and all(
            type(tag) is str and type(value) is float for tag, value in metrics.items()
        )
    ):
        from tensorboard.compat.proto.summary_pb2 import Summary

        try:
            summary = Summary(
                value=[
                    Summary.Value(tag=tag, simple_value=value)
                    for tag, value in metrics.items()
                ]
            )
        except Exception:
            # No writer call occurred: the old loop retains prefix/error behavior.
            pass
        else:
            # Submission can enqueue before raising; never retry through the loop.
            try:
                writer._get_file_writer().add_summary(summary, step)
            except Exception:
                if not ignore_errors:
                    raise
            return
    for tag, value in metrics.items():
        try:
            writer.add_scalar(tag, value, step)
        except Exception:
            if not ignore_errors:
                raise


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
        _write_tensorboard_scalars(writer, metrics, step)
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
