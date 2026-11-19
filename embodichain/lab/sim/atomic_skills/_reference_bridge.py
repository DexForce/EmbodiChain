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

"""Plan a rest-to-rest command-reference handoff without changing observations."""

from __future__ import annotations
import torch
from embodichain.compute.trajectory import interpolate_minimum_jerk

__all__: list[str] = []


def _reference_prefix(
    reference: torch.Tensor,
    initial: torch.Tensor,
    ids: tuple[int, ...],
    dt: float,
    velocity: torch.Tensor,
    acceleration: float,
) -> torch.Tensor:
    """Return an owned prefix excluding the original initial sample."""
    start = initial.clone()
    start[:, list(ids)] = reference[:, list(ids)]
    if torch.allclose(start, initial, atol=1e-7, rtol=0):
        return initial[:, None, :][:, :0].clone()
    controlled = interpolate_minimum_jerk(
        torch.stack((start[:, list(ids)], initial[:, list(ids)]), dim=1),
        3,
        dt,
        velocity[list(ids)],
        acceleration,
    )
    path = initial[:, None, :].expand(-1, controlled.shape[1], -1).clone()
    path[:, :, list(ids)] = controlled
    return path[:, :-1].clone()
