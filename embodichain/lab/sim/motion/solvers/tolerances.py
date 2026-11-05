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

"""Explicit emitted-pose tolerance contracts for stateful IK solvers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_solver import BaseSolver
    from embodichain.lab.sim.objects import Robot

__all__ = ["pose_validation_tolerances", "configured_pose_uncertainty"]


def pose_validation_tolerances(solver: BaseSolver) -> tuple[float, float]:
    """Resolve positive position/rotation budgets without inventing solver controls.

    Args:
        solver: Iterative solver exposing get_iteration_params, or analytic UR IK.

    Returns:
        Position metres and rotation radians. UR retains the source 0.5 mm /
        0.5 mrad validation budget; other unsupported solvers fail explicitly.
    """
    getter = getattr(solver, "get_iteration_params", None)
    if callable(getter):
        values = getter()
        result = (values["pos_eps"], values["rot_eps"])
    else:
        from .ur_solver import URSolver

        if not isinstance(solver, URSolver):
            raise ValueError("Solver pose validation tolerances are unavailable.")
        result = (5e-4, 5e-4)
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(
            "Solver pose validation tolerances must be finite and positive."
        )
    return result


def configured_pose_uncertainty(robot: Robot, control_part: str) -> tuple[float, float]:
    """Return original full-pose PytorchSolver configuration tolerances.

    These parameters bound geometric endpoint uncertainty and are not physical
    acceptance thresholds. Position-only configurations and missing/invalid
    budgets remain unsupported; no runtime tolerance is silently substituted.

    Args:
        robot: Robot whose assembled solver configuration owns the uncertainty.
        control_part: Explicit group using a full-pose PytorchSolver.

    Returns:
        Configured position uncertainty in metres and angular uncertainty in
        radians, without an added execution or task-acceptance margin.
    """
    try:
        solver = robot.cfg.solver_cfg[control_part]
        if isinstance(solver.pos_eps, bool) or isinstance(solver.rot_eps, bool):
            raise ValueError("Boolean IK tolerances are unavailable.")
        position, rotation = float(solver.pos_eps), float(solver.rot_eps)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("handover_exchange_ik_uncertainty_unavailable") from error
    if (
        getattr(solver, "class_type", None) != "PytorchSolver"
        or getattr(solver, "is_only_position_constraint", False)
        or not math.isfinite(position)
        or not math.isfinite(rotation)
        or position <= 0
        or rotation <= 0
    ):
        raise ValueError("handover_exchange_ik_uncertainty_unavailable")
    return position, rotation
