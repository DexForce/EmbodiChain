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

"""Task-owned adaptation of cuRobo's kinematics-state collision contract."""

from __future__ import annotations

from typing import Any

import torch

__all__: list[str] = []


def validate_configurations(
    checker: Any, q: torch.Tensor, env_query_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """Check bounds, self-collision and scene collision without changing samples."""
    if q.ndim != 3 or 0 in q.shape or not torch.isfinite(q).all():
        raise ValueError("Collision samples must be finite non-empty (B, T, D).")
    batch, horizon, _ = q.shape
    if checker.self_collision_cost is None or checker.collision_constraint is None:
        raise ValueError("GenSim requires both self and scene collision constraints.")
    state = checker.get_kinematics(q)
    spheres = state.robot_spheres
    if (
        spheres.ndim != 4
        or spheres.shape[:2] != (batch, horizon)
        or spheres.shape[-1] != 4
        or spheres.shape[-2] == 0
        or not torch.isfinite(spheres).all()
    ):
        raise ValueError("Collision kinematics must provide finite robot spheres.")
    # The installed checker leaves scene sphere counts uninitialized and passes
    # a Tensor to a cost that now requires the complete KinematicsState.
    checker.collision_constraint.update_num_spheres(
        spheres.shape[-2], batch_size=batch, horizon=horizon
    )
    checker.setup_batch_tensors(batch, horizon)
    costs = (
        checker.get_bound(q),
        checker.get_self_collision(spheres),
        checker.collision_constraint.forward(state, idxs_env_query=env_query_idx),
    )
    accepted = torch.ones((batch, horizon), dtype=torch.bool, device=q.device)
    for cost in costs:
        if cost.shape[:2] != (batch, horizon):
            raise ValueError("Collision costs must retain batch and horizon axes.")
        accepted &= (
            (torch.isfinite(cost) & (cost == 0)).reshape(batch, horizon, -1).all(-1)
        )
    return accepted
