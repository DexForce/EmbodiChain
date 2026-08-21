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

"""Sequential IK plus joint interpolation diagnostic adapter."""

from __future__ import annotations

import math

import torch

from embodichain.lab.sim.planners import PlanResult
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance

from ..models import BenchmarkCase
from ..registry import register_planner_adapter
from .base import PlannerAdapter

__all__ = ["IkInterpolateAdapter"]


class IkInterpolateAdapter(PlannerAdapter):
    """Use the robot IK solver followed by fixed-count joint interpolation."""

    capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})

    def build(self) -> None:
        """No planner object is required beyond the benchmark robot."""

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Solve each waypoint sequentially while retaining per-env failures."""
        robot = self.context.robot
        interpolation_dt = self.spec.config.get("interpolation_dt")
        if isinstance(interpolation_dt, bool) or not isinstance(
            interpolation_dt, (int, float)
        ):
            raise ValueError(
                "ik_interpolate requires an explicit numeric interpolation_dt."
            )
        interpolation_dt = float(interpolation_dt)
        if not math.isfinite(interpolation_dt) or interpolation_dt <= 0.0:
            raise ValueError("interpolation_dt must be finite and greater than zero.")
        seed = case.start_qpos
        alive = torch.ones(case.batch_size, dtype=torch.bool, device=robot.device)
        targets = [seed]
        for waypoint_index in range(case.num_waypoints):
            success, solved = robot.compute_ik(
                pose=case.target_waypoints[:, waypoint_index],
                name=self.context.control_part,
                joint_seed=seed,
            )
            success_tensor = torch.as_tensor(
                success, dtype=torch.bool, device=robot.device
            ).flatten()
            if success_tensor.numel() == 1 and case.batch_size > 1:
                success_tensor = success_tensor.expand(case.batch_size)
            alive &= success_tensor
            solved = torch.as_tensor(solved, device=robot.device, dtype=seed.dtype)
            seed = torch.where(success_tensor[:, None], solved, seed)
            targets.append(seed)

        sparse_path = torch.stack(targets, dim=1)
        positions = interpolate_with_distance(
            trajectory=sparse_path,
            interp_num=self.context.sample_interval,
            device=robot.device,
        )
        dt = torch.zeros(positions.shape[:2], dtype=torch.float32, device=robot.device)
        if positions.shape[1] > 1:
            dt[:, 1:] = interpolation_dt
        return PlanResult(
            success=alive,
            positions=positions,
            dt=dt,
        )


register_planner_adapter("ik_interpolate", IkInterpolateAdapter)
