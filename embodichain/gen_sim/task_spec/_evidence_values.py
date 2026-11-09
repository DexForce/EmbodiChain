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

"""Shared numeric values for normalized normative evidence."""

from __future__ import annotations
import math
from typing import Any

__all__: list[str] = []


def one(values: Any) -> Any:
    values = list(values)
    if len(values) != 1:
        raise ValueError("Held evidence requires one exact occurrence.")
    return values[0]


def flags(values: Any, count: int, expected: bool = True) -> None:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) is not bool or value is not expected for value in values)
    ):
        raise ValueError("Held evidence masks are invalid or not accepted.")


def number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Held evidence requires finite nonnegative clocks/metrics.")
    return float(value)


def matrix(value: Any, count: int) -> torch.Tensor:
    import torch
    from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

    result = torch.as_tensor(value, dtype=torch.float64).cpu()
    validate_rigid_pose_batch(result, count)
    return result


def delta(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from embodichain.compute.geometry.rigid import rigid_pose_displacement

    return rigid_pose_displacement(second, first)
