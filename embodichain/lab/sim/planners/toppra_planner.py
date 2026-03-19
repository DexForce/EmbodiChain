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

import torch
import numpy as np

from embodichain.utils import logger, configclass
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.planners.base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
)
from .utils import PlanState, PlanResult

try:
    import toppra as ta
    import toppra.constraint as constraint
except ImportError:
    logger.log_error(
        "toppra not installed. Install with `pip install toppra==0.6.3`", ImportError
    )

ta.setup_logging(level="WARN")


__all__ = ["ToppraPlanner", "ToppraPlannerCfg", "ToppraOptions"]


@configclass
class ToppraPlannerCfg(BasePlannerCfg):

    planner_type: str = "toppra"


@configclass
class ToppraOptions(PlanOptions):
    constraints: dict = {
        "velocity": 0.2,
        "acceleration": 0.5,
    }
    """Constraints for the planner, including velocity and acceleration limits. Should be a 
    dictionary with keys 'velocity' and 'acceleration', each containing a value or a list of limits for each joint.
    """
    sample_method: TrajectorySampleMethod = TrajectorySampleMethod.TIME
    """Method for sampling the trajectory. Options are 'time' for uniform time intervals or 'quantity' for a fixed number of samples."""
    sample_interval: float | int = 0.01
    """Interval for sampling the trajectory. If sample_method is 'time', this is the time interval in seconds. If sample_method is 'quantity', this is the total number of samples."""


class ToppraPlanner(BasePlanner):
    def __init__(self, cfg: ToppraPlannerCfg):
        r"""Initialize the TOPPRA trajectory planner.

        Args:
            cfg: Configuration object containing ToppraPlanner settings
        """
        super().__init__(cfg)

    def plan(
        self,
        target_states: list[PlanState],
        plan_option: ToppraOptions = ToppraOptions(),
    ) -> PlanResult:
        r"""Execute trajectory planning.

        Args:
            target_states: List of dictionaries containing target states
            cfg: ToppraOptions

        Returns:
            PlanResult containing the planned trajectory details.
        """
        joint_ids = self.robot.get_joint_ids(
            plan_option.control_part, remove_mimic=True
        )
        dofs = len(joint_ids)

        # set constraints
        if isinstance(plan_option.constraints["velocity"], float):
            self.vlims = np.array(
                [
                    [
                        -plan_option.constraints["velocity"],
                        plan_option.constraints["velocity"],
                    ]
                    for _ in range(dofs)
                ]
            )
        else:
            self.vlims = np.array(plan_option.constraints["velocity"])

        if isinstance(plan_option.constraints["acceleration"], float):
            self.alims = np.array(
                [
                    [
                        -plan_option.constraints["acceleration"],
                        plan_option.constraints["acceleration"],
                    ]
                    for _ in range(dofs)
                ]
            )
        else:
            self.alims = np.array(plan_option.constraints["acceleration"])

        sample_method = plan_option.sample_method
        sample_interval = plan_option.sample_interval
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
        start_qpos = (
            plan_option.start_qpos
            if plan_option.start_qpos is not None
            else target_states[0].qpos
        )
        if len(start_qpos) != dofs:
            logger.log_error("Current waypoint does not align")
        for target in target_states:
            if len(target.qpos) != dofs:
                logger.log_error("Target waypoints do not align")

        if (
            len(target_states) == 1
            and np.sum(np.abs(np.array(target_states[0].qpos) - np.array(start_qpos)))
            < 1e-3
        ):
            logger.log_warning("Only two same waypoints, returning trivial trajectory.")
            return PlanResult(
                success=True,
                positions=torch.as_tensor(
                    np.array([start_qpos, target_states[0].qpos]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                velocities=torch.as_tensor(
                    np.array([[0.0] * dofs, [0.0] * dofs]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                accelerations=torch.as_tensor(
                    np.array([[0.0] * dofs, [0.0] * dofs]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                dt=torch.as_tensor([0.0, 0.0], dtype=torch.float32, device=self.device),
                duration=0.0,
            )

        # Build waypoints
        waypoints = [np.array(start_qpos)]
        for target in target_states:
            waypoints.append(np.array(target.qpos))
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
        # NOTES: Important to set a large number of grid points for better performance in dense waypoint scenarios.

        # Compute parameterized trajectory
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            # raise RuntimeError("Unable to find feasible trajectory")
            logger.log_warning("Unable to find feasible trajectory")
            return PlanResult(success=False)

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

        dt = torch.as_tensor(ts, dtype=torch.float32, device=self.device)
        dt = torch.diff(dt, prepend=torch.tensor([0.0], device=self.device))

        return PlanResult(
            success=True,
            positions=torch.as_tensor(
                np.array(positions), dtype=torch.float32, device=self.device
            ),
            velocities=torch.as_tensor(
                np.array(velocities), dtype=torch.float32, device=self.device
            ),
            accelerations=torch.as_tensor(
                np.array(accelerations), dtype=torch.float32, device=self.device
            ),
            dt=dt,
            duration=duration,
        )
