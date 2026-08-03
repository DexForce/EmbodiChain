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

"""IK plus TOPPRA time-parameterization diagnostic adapter."""

from __future__ import annotations

import importlib.util

from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    PlanResult,
    PlanState,
    ToppraPlannerCfg,
    ToppraPlanOptions,
)
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod

from ..config import PlannerSpecCfg
from ..models import BenchmarkCase
from ..registry import register_planner_adapter
from .base import PlannerAdapter, PlannerContext

__all__ = ["ToppraAdapter"]


class ToppraAdapter(PlannerAdapter):
    """Pre-interpolate EEF targets through IK, then run TOPPRA."""

    capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None

    def availability(self) -> tuple[bool, str | None]:
        """Report whether the optional TOPPRA package is installed."""
        if importlib.util.find_spec("toppra") is None:
            return False, "TOPPRA is not installed."
        return True, None

    def build(self) -> None:
        """Construct the TOPPRA MotionGenerator."""
        self.motion_generator = MotionGenerator(
            MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.context.robot.uid))
        )

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan EEF waypoints through the existing IK-to-TOPPRA pipeline."""
        if self.motion_generator is None:
            raise RuntimeError("TOPPRA adapter must be built before plan().")
        config = self.spec.config
        targets = [
            PlanState.from_xpos(case.target_waypoints[:, index])
            for index in range(case.num_waypoints)
        ]
        return self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=self.context.control_part,
                is_interpolate=True,
                is_linear=True,
                plan_opts=ToppraPlanOptions(
                    constraints={
                        "velocity": float(config.get("velocity", 0.2)),
                        "acceleration": float(config.get("acceleration", 0.5)),
                    },
                    sample_method=TrajectorySampleMethod.QUANTITY,
                    sample_interval=self.context.sample_interval,
                ),
            ),
        )


register_planner_adapter("toppra", ToppraAdapter)
