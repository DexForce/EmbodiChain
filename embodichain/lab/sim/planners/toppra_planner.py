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
    validate_plan_options,
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


__all__ = ["ToppraPlanner", "ToppraPlannerCfg", "ToppraPlanOptions"]


@configclass
class ToppraPlannerCfg(BasePlannerCfg):

    planner_type: str = "toppra"


@configclass
class ToppraPlanOptions(PlanOptions):

    constraints: dict = {
        "velocity": 0.2,
        "acceleration": 0.5,
    }
    """Constraints for the planner, including velocity and acceleration limits. 

    Should be a dictionary with keys 'velocity' and 'acceleration', each containing a value or a list of limits for each joint.
    """

    sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY
    """Method for sampling the trajectory. 
    
    Options are 'time' for uniform time intervals or 'quantity' for a fixed number of samples.
    """

    sample_interval: float | int = 0.01
    """Interval for sampling the trajectory. 
    
    If sample_method is 'time', this is the time interval in seconds. 
    If sample_method is 'quantity', this is the total number of samples.
    """


class ToppraPlanner(BasePlanner):
    def __init__(self, cfg: ToppraPlannerCfg):
        r"""Initialize the TOPPRA trajectory planner.

        References:
            - TOPPRA: Time-Optimal Path Parameterization for Robotic Systems (https://github.com/hungpham2511/toppra)

        Args:
            cfg: Configuration object containing ToppraPlanner settings
        """
        super().__init__(cfg)

        if self.robot.num_instances > 1:
            logger.log_error(
                "ToppraPlanner does not support multiple robot instances",
                NotImplementedError,
            )

    @validate_plan_options(options_cls=ToppraPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: ToppraPlanOptions = ToppraPlanOptions(),
    ) -> PlanResult:
        r"""Execute trajectory planning.

        Args:
            target_states: List of dictionaries containing target states
            cfg: ToppraPlanOptions

        Returns:
            PlanResult containing the planned trajectory details.
        """

        for i, target in enumerate(target_states):
            if target.qpos is None:
                logger.log_error(f"Target state at index {i} missing qpos")

        dofs = len(target_states[0].qpos)

        # set constraints
        if isinstance(options.constraints["velocity"], float):
            vlims = np.array(
                [
                    [
                        -options.constraints["velocity"],
                        options.constraints["velocity"],
                    ]
                    for _ in range(dofs)
                ]
            )
        else:
            vlims = np.array(options.constraints["velocity"])

        if isinstance(options.constraints["acceleration"], float):
            alims = np.array(
                [
                    [
                        -options.constraints["acceleration"],
                        options.constraints["acceleration"],
                    ]
                    for _ in range(dofs)
                ]
            )
        else:
            alims = np.array(options.constraints["acceleration"])

        sample_method = options.sample_method
        sample_interval = options.sample_interval
        if sample_method == TrajectorySampleMethod.TIME and sample_interval <= 0:
            logger.log_error("Time interval must be positive", ValueError)
        if sample_method == TrajectorySampleMethod.TIME and isinstance(
            sample_interval, int
        ):
            logger.log_error(
                "Time interval must be a float when sample_method is TIME", TypeError
            )
        if sample_method == TrajectorySampleMethod.QUANTITY and sample_interval < 2:
            logger.log_error("At least 2 sample points required", ValueError)
        if sample_method == TrajectorySampleMethod.QUANTITY and isinstance(
            sample_interval, float
        ):
            logger.log_error(
                "Number of samples must be an integer when sample_method is QUANTITY",
                TypeError,
            )

        # Check waypoints
        for i, target in enumerate(target_states):
            if target.qpos is None:
                logger.log_error(f"Target state at index {i} missing qpos")
            if len(target.qpos) != dofs:
                logger.log_error(f"Target waypoints do not align at index {i}")

        if (
            len(target_states) == 2
            and torch.sum(torch.abs(target_states[1].qpos - target_states[0].qpos))
            < 1e-3
        ):
            logger.log_warning("Only two same waypoints, returning trivial trajectory.")
            return PlanResult(
                success=True,
                positions=torch.as_tensor(
                    np.stack([target_states[0].qpos, target_states[1].qpos]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                velocities=torch.zeros(
                    (2, dofs), dtype=torch.float32, device=self.device
                ),
                accelerations=torch.zeros(
                    (2, dofs), dtype=torch.float32, device=self.device
                ),
                dt=torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device),
                duration=0.0,
            )

        # Build waypoints
        waypoints = []
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
        pc_vel = constraint.JointVelocityConstraint(vlims)
        pc_acc = constraint.JointAccelerationConstraint(alims)

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
