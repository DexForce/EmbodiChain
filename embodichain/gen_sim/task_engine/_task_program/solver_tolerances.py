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

"""GenSim pose validation budgets for iterative and analytic IK backends."""

from __future__ import annotations

from typing import Any

__all__: list[str] = []


def _pose_validation_parameters(solver: Any) -> dict[str, Any]:
    getter = getattr(solver, "get_iteration_params", None)
    if callable(getter):
        return getter()
    from embodichain.lab.sim.motion.solvers import URSolver

    if isinstance(solver, URSolver):
        # Analytic UR IK has no iteration controls. Check its emitted FK against
        # the baseline 0.5 mm / 0.5 mrad pose budget, not an invented iteration API.
        return {"pos_eps": 5e-4, "rot_eps": 5e-4}
    raise ValueError("GenSim solver pose tolerances are unavailable.")
