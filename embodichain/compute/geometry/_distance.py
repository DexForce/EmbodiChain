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

"""Batched geometry shared by near-target condition adapters."""

from __future__ import annotations

from collections.abc import Sequence
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


def _positions(value: torch.Tensor) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("Positions must have shape (num_envs, 3).")
    if not value.is_floating_point():
        raise TypeError("Positions must be floating tensors.")


def _validate_near_target(
    tolerance: float, axes: Sequence[int], inclusive: bool
) -> tuple[int, ...]:
    axes = tuple(axes)
    if (
        not axes
        or any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes)
        or len(set(axes)) != len(axes)
    ):
        raise ValueError("axes must select distinct indices from 0, 1, 2.")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be positive and finite.")
    if type(inclusive) is not bool:
        raise TypeError("inclusive must be a bool.")
    return axes


def _measure_near_target(
    position: torch.Tensor,
    target: torch.Tensor,
    tolerance: float,
    axes: tuple[int, ...],
    inclusive: bool,
    *,
    portable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch

    target = target.to(device=position.device, dtype=position.dtype)
    if target.shape == (3,):
        target = target.unsqueeze(0).expand_as(position)
    _positions(target)
    if target.shape != position.shape:
        raise ValueError("Target batch must match object positions.")
    valid = torch.isfinite(position).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    delta = (position - target)[:, axes]
    if portable:
        # Fix the accumulation order and round once to the observation dtype.
        squares = delta.to(torch.float64).square().unbind(-1)
        total = squares[0]
        for square in squares[1:]:
            total = total + square
        distance = total.sqrt().to(position.dtype)
    else:
        distance = torch.linalg.vector_norm(delta, dim=-1)
    satisfied = distance <= tolerance if inclusive else distance < tolerance
    return satisfied, valid, distance
