# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import numpy as np
from embodichain.utils import logger
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.planners.base_planner import BasePlanner

from typing import TYPE_CHECKING, Union, Optional, Tuple

try:
    import toppra as ta
    import toppra.constraint as constraint
except ImportError:
    logger.log_error(
        "toppra not installed. Install with `pip install toppra==0.6.3`", ImportError
    )

ta.setup_logging(level="WARN")


class ToppraPlanner(BasePlanner):
    def __init__(self, dofs, max_constraints):
        r"""Initialize the TOPPRA trajectory planner.

        Args:
            dofs: Number of degrees of freedom
            max_constraints: Dictionary containing 'velocity' and 'acceleration' constraints
        """
        super().__init__(dofs, max_constraints)

        # Create TOPPRA-specific constraint arrays (symmetric format)
        # This format is required by TOPPRA library
        self.vlims = np.array([[-v, v] for v in max_constraints["velocity"]])
        self.alims = np.array([[-a, a] for a in max_constraints["acceleration"]])

    def plan(
        self,
        current_state: dict,
        target_states: list[dict],
        **kwargs,
    ):
        r"""Execute trajectory planning.

        Args:
            current_state: Dictionary containing 'position', 'velocity', 'acceleration' for current state
            target_states: List of dictionaries containing target states

        Returns:
            Tuple of (success, positions, velocities, accelerations, times, duration)
        """
        sample_method = kwargs.get("sample_method", TrajectorySampleMethod.TIME)
        sample_interval = kwargs.get("sample_interval", 0.01)
        if not isinstance(sample_interval, (float, int)):
            logger.log_error(
                f"sample_interval must be float/int, got {type(sample_interval)}",
                TypeError,
            )
        if sample_method == TrajectorySampleMethod.TIME and sample_interval <= 0:
            logger.log_error("Time interval must be positive", ValueError)
        elif sample_method == TrajectorySampleMethod.QUANTITY and sample_interval < 2:
            logger.log_error("At least 2 sample points required", ValueError)

        # Check waypoints
        if len(current_state["position"]) != self.dofs:
            logger.log_info("Current wayponit does not align")
            return False, None, None, None, None, None
        for target in target_states:
            if len(target["position"]) != self.dofs:
                logger.log_info("Target Wayponits does not align")
                return False, None, None, None, None, None

        if (
            len(target_states) == 1
            and np.sum(
                np.abs(
                    np.array(target_states[0]["position"])
                    - np.array(current_state["position"])
                )
            )
            < 1e-3
        ):
            logger.log_info("Only two same waypoints, do not plan")
            return (
                True,
                np.array([current_state["position"], target_states[0]["position"]]),
                np.array([[0.0] * self.dofs, [0.0] * self.dofs]),
                np.array([[0.0] * self.dofs, [0.0] * self.dofs]),
                0,
                0,
            )

        # Build waypoints
        waypoints = [np.array(current_state["position"])]
        for target in target_states:
            waypoints.append(np.array(target["position"]))
        waypoints = np.array(waypoints)

        # Create spline interpolation
        # NOTE: Suitable for dense waypoints
        ss = np.linspace(0, 1, len(waypoints))

        # NOTE: Suitable for sparse waypoints; for dense waypoints, CubicSpline may fail strict monotonicity requirement
        # len_total = 0
        # len_from_start = [0]
        # for i in range(len(waypoints)-1):
        #     len_total += np.sum(np.abs(waypoints[i+1] - waypoints[i]))
        #     len_from_start.append(len_total)
        # ss = np.array([cur/len_total for cur in len_from_start])

        path = ta.SplineInterpolator(ss, waypoints)

        # Set constraints
        pc_vel = constraint.JointVelocityConstraint(self.vlims)
        pc_acc = constraint.JointAccelerationConstraint(self.alims)

        # Create TOPPRA instance
        instance = ta.algorithm.TOPPRA(
            [pc_vel, pc_acc],
            path,
            parametrizer="ParametrizeConstAccel",
            gridpt_min_nb_points=max(100, 10 * len(waypoints)),
        )
        # NOTES:合理设置gridpt_min_nb_points对加速度约束很重要

        # Compute parameterized trajectory
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            # raise RuntimeError("Unable to find feasible trajectory")
            logger.log_info("Unable to find feasible trajectory")
            return False, None, None, None, None, None

        duration = jnt_traj.duration
        # Sample trajectory points
        if duration <= 0:
            logger.log_error(f"Duration must be positive, got {duration}", ValueError)
        if sample_method == TrajectorySampleMethod.TIME:
            n_points = max(2, int(np.ceil(duration / sample_interval)) + 1)
            ts = np.linspace(0, duration, n_points)
        else:
            ts = np.linspace(0, duration, num=int(sample_interval))

        positions = []
        velocities = []
        accelerations = []

        for t in ts:
            positions.append(jnt_traj.eval(t))
            velocities.append(jnt_traj.evald(t))
            accelerations.append(jnt_traj.evaldd(t))

        return (
            True,
            np.array(positions),
            np.array(velocities),
            np.array(accelerations),
            ts,
            duration,
        )
