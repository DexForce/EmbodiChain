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
"""Device-resident constrained-probe history and overflow accumulation."""

from __future__ import annotations

import warp as wp

__all__: list[str] = []


@wp.kernel
def record_probe_state(
    qpos: wp.array2d(dtype=float),
    overflow: wp.array(dtype=int),
    history: wp.array3d(dtype=float),
    health: wp.array(dtype=int),
    step_index: int,
):
    world, coordinate = wp.tid()
    history[world, step_index, coordinate] = qpos[world, coordinate]
    if coordinate == 0:
        wp.atomic_or(health, world, overflow[world])
