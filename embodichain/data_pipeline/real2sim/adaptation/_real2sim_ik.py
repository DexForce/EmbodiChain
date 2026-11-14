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
"""Experimental Real2Sim-only IK, separate from EmbodiChain's native solvers.

Retains the original source-near least-squares objective and +/-0.65 rad local
search bounds. URDF limit excess and target errors are diagnostics, not rejection
gates. This returns approximate joint references; simulator limits and physical
execution remain unchanged. It is not registered as a native motion solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from ._kinematics import Robot

__all__: list[str] = []


def solve_real2sim_ik(
    robot: Robot, source: np.ndarray, side: int, target: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a world TCP target near the supplied full joint state.

    Args:
        robot: Recorded robot kinematics and world/TCP calibration.
        source: Full joint seed; other arms and gripper joints are preserved.
        side: Arm index in the recorded robot model.
        target: Desired world TCP transform.

    Returns:
        Separate approximate joints and numerical diagnostics. Optimizer
        convergence does not certify target accuracy or physical task success.
    """
    source = np.asarray(source, dtype=float)
    columns = robot.arm_columns[side]
    initial = source[columns]

    def residual(x: np.ndarray) -> np.ndarray:
        qpos = source.copy()
        qpos[columns] = x
        actual = robot.tcp(qpos[None], side)[0]
        return np.r_[
            (actual[:3, 3] - target[:3, 3]) / 0.003,
            Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec() / 0.035,
            (x - initial) / 0.5,
        ]

    fit = least_squares(
        residual, initial, bounds=(initial - 0.65, initial + 0.65), max_nfev=80
    )
    result = source.copy()
    result[columns] = fit.x
    if not np.isfinite(result).all():
        raise ValueError("Real2Sim approximate IK returned non-finite joints")
    actual = robot.tcp(result[None], side)[0]
    lower, upper = robot.arm_limits[side]
    excess = np.maximum(np.maximum(lower - fit.x, fit.x - upper), 0)
    return result, dict(
        solver="real2sim_source_near_least_squares",
        optimizer_converged=bool(fit.success),
        tcp_error_m=float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        orientation_error_deg=float(
            np.rad2deg(
                Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
            )
        ),
        max_joint_residual_rad=float(np.max(abs(fit.x - initial))),
        max_joint_limit_excess_rad=float(excess.max()),
    )
