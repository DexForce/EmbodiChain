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
"""Shared two-joint timing scenario for numeric and visual diagnostics."""

from __future__ import annotations

from typing import Literal

import torch

from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
    _plan_linear_profiles,
)
from embodichain.lab.sim.motion.planners.utils import PlanResult


def plan_example(
    waypoints: torch.Tensor,
    profile: Literal["trapezoidal", "double_s"],
    *,
    samples: int = 201,
    blend_tolerance: float = 0.0,
    minimum_duration: float | None = None,
    backend: Literal["auto", "torch", "warp"] = "torch",
) -> PlanResult:
    """Plan the shared two-joint scenario using explicit diagnostic settings."""
    return _plan_linear_profiles(
        waypoints,
        TrapezoidalPlanOptions(
            profile=profile,
            constraints={
                "velocity": [0.8, 0.6],
                "acceleration": [1.5, 1.2],
                "jerk": [4.0, 3.0],
            },
            sample_interval=samples,
            stop_at_waypoints=blend_tolerance == 0.0,
            blend_tolerance=blend_tolerance,
            minimum_duration=minimum_duration,
            backend=backend,
        ),
    )
