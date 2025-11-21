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
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt

from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.utils import logger


class BasePlanner(ABC):
    r"""Base class for trajectory planners.

    This class provides common functionality that can be shared across different
    planner implementations, such as constraint checking and trajectory visualization.

    Args:
        dofs: Number of degrees of freedom
        max_constraints: Dictionary containing 'velocity' and 'acceleration' constraints
    """

    def __init__(self, dofs: int, max_constraints: Dict[str, List[float]]):
        self.dofs = dofs
        self.max_constraints = max_constraints

    @abstractmethod
    def plan(
        self,
        current_state: Dict,
        target_states: List[Dict],
        **kwargs,
    ) -> Tuple[
        bool,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
    ]:
        r"""Execute trajectory planning.

        This method must be implemented by subclasses to provide the specific
        planning algorithm.

        Args:
            current_state: Dictionary containing 'position', 'velocity', 'acceleration' for current state
            target_states: List of dictionaries containing target states

        Returns:
            Tuple of (success, positions, velocities, accelerations, times, duration):
                - success: bool, whether planning succeeded
                - positions: np.ndarray (N, DOF), joint positions along trajectory
                - velocities: np.ndarray (N, DOF), joint velocities along trajectory
                - accelerations: np.ndarray (N, DOF), joint accelerations along trajectory
                - times: np.ndarray (N,), time stamps for each point
                - duration: float, total trajectory duration
        """
        logger.log_error("Subclasses must implement plan() method", NotImplementedError)

    def is_satisfied_constraint(
        self, velocities: np.ndarray, accelerations: np.ndarray
    ) -> bool:
        r"""Check if the trajectory satisfies velocity and acceleration constraints.

        This method checks whether the given velocities and accelerations satisfy
        the constraints defined in max_constraints. It allows for some tolerance
        to account for numerical errors in dense waypoint scenarios.

        Args:
            velocities: Velocity array (N, DOF) where N is the number of trajectory points
            accelerations: Acceleration array (N, DOF) where N is the number of trajectory points

        Returns:
            bool: True if all constraints are satisfied, False otherwise

        Note:
            - Allows 10% tolerance for velocity constraints
            - Allows 25% tolerance for acceleration constraints
            - Prints exceed information if constraints are violated
            - Assumes symmetric constraints (velocities and accelerations can be positive or negative)
        """
        # Convert max_constraints to symmetric format for constraint checking
        # This assumes symmetric constraints (common for most planners)
        vlims = np.array([[-v, v] for v in self.max_constraints["velocity"]])
        alims = np.array([[-a, a] for a in self.max_constraints["acceleration"]])

        vel_check = np.all((velocities >= vlims[:, 0]) & (velocities <= vlims[:, 1]))
        acc_check = np.all(
            (accelerations >= alims[:, 0]) & (accelerations <= alims[:, 1])
        )

        # 超限情况
        if not vel_check:
            vel_exceed_info = []
            min_vel = np.min(velocities, axis=0)
            max_vel = np.max(velocities, axis=0)
            for i in range(self.dofs):
                exceed_percentage = 0
                max_vel_limit = self.max_constraints["velocity"][i]
                if min_vel[i] < -max_vel_limit:
                    exceed_percentage = (min_vel[i] + max_vel_limit) / max_vel_limit
                if max_vel[i] > max_vel_limit:
                    temp = (max_vel[i] - max_vel_limit) / max_vel_limit
                    if temp > exceed_percentage:
                        exceed_percentage = temp
                vel_exceed_info.append(exceed_percentage * 100)
            logger.log_info(f"Velocity exceed info: {vel_exceed_info} percentage")

        if not acc_check:
            acc_exceed_info = []
            min_acc = np.min(accelerations, axis=0)
            max_acc = np.max(accelerations, axis=0)
            for i in range(self.dofs):
                exceed_percentage = 0
                max_acc_limit = self.max_constraints["acceleration"][i]
                if min_acc[i] < -max_acc_limit:
                    exceed_percentage = (min_acc[i] + max_acc_limit) / max_acc_limit
                if max_acc[i] > max_acc_limit:
                    temp = (max_acc[i] - max_acc_limit) / max_acc_limit
                    if temp > exceed_percentage:
                        exceed_percentage = temp
                acc_exceed_info.append(exceed_percentage * 100)
            logger.log_info(f"Acceleration exceed info: {acc_exceed_info} percentage")

        return vel_check and acc_check

    def plot_trajectory(
        self, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray
    ) -> None:
        r"""Plot trajectory data.

        This method visualizes the trajectory by plotting position, velocity, and
        acceleration curves for each joint over time. It also displays the constraint
        limits for reference.

        Args:
            positions: Position array (N, DOF) where N is the number of trajectory points
            velocities: Velocity array (N, DOF) where N is the number of trajectory points
            accelerations: Acceleration array (N, DOF) where N is the number of trajectory points

        Note:
            - Creates a 3-subplot figure (position, velocity, acceleration)
            - Shows constraint limits as dashed lines
            - Requires matplotlib to be installed
        """
        time_step = 0.01
        time_steps = np.arange(positions.shape[0]) * time_step
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        for i in range(self.dofs):
            axs[0].plot(time_steps, positions[:, i], label=f"Joint {i+1}")
            axs[1].plot(time_steps, velocities[:, i], label=f"Joint {i+1}")
            axs[2].plot(time_steps, accelerations[:, i], label=f"Joint {i+1}")

        # Plot velocity constraints (only for first joint to avoid clutter)
        # Convert max_constraints to symmetric format for visualization
        if self.dofs > 0:
            max_vel = self.max_constraints["velocity"][0]
            max_acc = self.max_constraints["acceleration"][0]
            axs[1].plot(
                time_steps,
                [-max_vel] * len(time_steps),
                "k--",
                label="Max Velocity",
            )
            axs[1].plot(time_steps, [max_vel] * len(time_steps), "k--")
            # Plot acceleration constraints (only for first joint to avoid clutter)
            axs[2].plot(
                time_steps,
                [-max_acc] * len(time_steps),
                "k--",
                label="Max Accleration",
            )
            axs[2].plot(time_steps, [max_acc] * len(time_steps), "k--")

        axs[0].set_title("Position")
        axs[1].set_title("Velocity")
        axs[2].set_title("Acceleration")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()
