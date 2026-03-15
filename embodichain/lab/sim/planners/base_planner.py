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

from abc import ABC, abstractmethod
from dataclasses import MISSING
import matplotlib.pyplot as plt

from embodichain.utils import logger
from embodichain.utils import configclass
from embodichain.lab.sim.sim_manager import SimulationManager
from .utils import PlanState, PlanResult


__all__ = ["BasePlannerCfg", "BasePlanner"]


@configclass
class BasePlannerCfg:

    robot_uid: str = MISSING
    """UID of the robot to control. Must correspond to a robot added to the simulation with this UID."""

    control_part: str | None = None
    """Name of the robot part to control, e.g. 'left_arm'. Must correspond to a valid control part defined in the robot's configuration."""

    planner_type: str = "Base"


class BasePlanner(ABC):
    r"""Base class for trajectory planners.

    This class provides common functionality that can be shared across different
    planner implementations.

    Args:
        cfg: Configuration object for the planner.
    """

    def __init__(self, cfg: BasePlannerCfg):
        self.cfg = cfg

        if cfg.robot_uid is MISSING:
            logger.log_error("robot_uid is required in planner config", ValueError)

        self.robot = SimulationManager.get_instance().get_robot(cfg.robot_uid)
        if self.robot is None:
            logger.log_error(f"Robot with uid {cfg.robot_uid} not found", ValueError)

        joint_ids = self.robot.get_joint_ids(cfg.control_part, remove_mimic=True)
        self.dofs = len(joint_ids)
        self.device = self.robot.device

    @abstractmethod
    def plan(
        self,
        current_state: PlanState,
        target_states: list[PlanState],
        **kwargs,
    ) -> PlanResult:
        r"""Execute trajectory planning.

        This method must be implemented by subclasses to provide the specific
        planning algorithm.

        Args:
            current_state: Dictionary containing 'position', 'velocity', 'acceleration' for current state
            target_states: List of dictionaries containing target states

        Returns:
            PlanResult: An object containing:
                - success: bool, whether planning succeeded
                - positions: torch.Tensor (N, DOF), joint positions along trajectory
                - velocities: torch.Tensor (N, DOF), joint velocities along trajectory
                - accelerations: torch.Tensor (N, DOF), joint accelerations along trajectory
                - times: torch.Tensor (N,), time stamps for each point
                - duration: float, total trajectory duration
                - error_msg: Optional error message if planning failed
        """
        logger.log_error("Subclasses must implement plan() method", NotImplementedError)

    def is_satisfied_constraint(self, vels: torch.Tensor, accs: torch.Tensor) -> bool:
        r"""Check if the trajectory satisfies velocity and acceleration constraints.

        This method checks whether the given velocities and accelerations satisfy
        the constraints defined in constraints. It allows for some tolerance
        to account for numerical errors in dense waypoint scenarios.

        Args:
            vels: Velocity tensor (..., DOF) where the last dimension is DOF
            accs: Acceleration tensor (..., DOF) where the last dimension is DOF

        Returns:
            bool: True if all constraints are satisfied, False otherwise

        Note:
            - Allows 10% tolerance for velocity constraints
            - Allows 25% tolerance for acceleration constraints
            - Prints exceed information if constraints are violated
            - Assumes symmetric constraints (velocities and accelerations can be positive or negative)
            - Supports batch dimension computation, e.g. (B, N, DOF) or (N, DOF)
        """
        device = vels.device

        # Convert constraints to tensors for vectorized constraint checking
        if not hasattr(self.cfg, "constraints") or self.cfg.constraints is None:
            logger.log_error("constraints not found in planner config")
            return True

        max_vel = torch.tensor(
            self.cfg.constraints["velocity"], dtype=vels.dtype, device=device
        )
        max_acc = torch.tensor(
            self.cfg.constraints["acceleration"], dtype=accs.dtype, device=device
        )

        # To support batching, we compute along all dimensions except the last one (DOF)
        reduce_dims = tuple(range(vels.ndim - 1))

        # Check bounds
        vel_check = torch.all(torch.abs(vels) <= max_vel).item()
        acc_check = torch.all(torch.abs(accs) <= max_acc).item()

        if not vel_check:
            # max absolute value over all trajectory points and batches
            max_abs_vel = torch.amax(torch.abs(vels), dim=reduce_dims)
            exceed_percentage = torch.clamp((max_abs_vel - max_vel) / max_vel, min=0.0)
            vel_exceed_info = (exceed_percentage * 100).tolist()
            logger.log_info(f"Velocity exceed info: {vel_exceed_info} percentage")

        if not acc_check:
            max_abs_acc = torch.amax(torch.abs(accs), dim=reduce_dims)
            exceed_percentage = torch.clamp((max_abs_acc - max_acc) / max_acc, min=0.0)
            acc_exceed_info = (exceed_percentage * 100).tolist()
            logger.log_info(f"Acceleration exceed info: {acc_exceed_info} percentage")

        return vel_check and acc_check

    def plot_trajectory(
        self,
        positions: torch.Tensor,
        vels: torch.Tensor | None = None,
        accs: torch.Tensor | None = None,
    ) -> None:
        r"""Plot trajectory data.

        This method visualizes the trajectory by plotting position, velocity, and
        acceleration curves for each joint over time. It also displays the constraint
        limits for reference. Supports plotting batched trajectories.

        Args:
            positions: Position tensor (N, DOF) or (B, N, DOF)
            vels: Velocity tensor (N, DOF) or (B, N, DOF), optional
            accs: Acceleration tensor (N, DOF) or (B, N, DOF), optional

        Note:
            - Creates a multi-subplot figure (position, and optional velocity/acceleration)
            - Shows constraint limits as dashed lines
            - If input is (B, N, DOF), plots elements separately per batch sequence.
            - Requires matplotlib to be installed
        """
        # Ensure we're dealing with CPU tensors for plotting
        positions = positions.detach().cpu()
        if vels is not None:
            vels = vels.detach().cpu()
        if accs is not None:
            accs = accs.detach().cpu()

        time_step = 0.01

        # Helper to unsqueeze unbatched (N, DOF) -> (1, N, DOF)
        def ensure_batch_dim(tensor):
            if tensor is None:
                return None
            return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

        positions = ensure_batch_dim(positions)
        vels = ensure_batch_dim(vels)
        accs = ensure_batch_dim(accs)

        batch_size, num_steps, _ = positions.shape
        time_steps = np.arange(num_steps) * time_step

        num_plots = 1 + (1 if vels is not None else 0) + (1 if accs is not None else 0)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))

        # Ensure axs is iterable even if there relies only 1 subplot
        if num_plots == 1:
            axs = [axs]

        for b in range(batch_size):
            line_style = "-" if batch_size == 1 else f"C{b}-"
            alpha = 1.0 if batch_size == 1 else max(0.2, 1.0 / np.sqrt(batch_size))

            for i in range(self.dofs):
                label = f"Joint {i+1}" if b == 0 else ""
                axs[0].plot(
                    time_steps,
                    positions[b, :, i].numpy(),
                    line_style,
                    alpha=alpha,
                    label=label,
                )

                plot_idx = 1
                if vels is not None:
                    axs[plot_idx].plot(
                        time_steps,
                        vels[b, :, i].numpy(),
                        line_style,
                        alpha=alpha,
                        label=label,
                    )
                    plot_idx += 1
                if accs is not None:
                    axs[plot_idx].plot(
                        time_steps,
                        accs[b, :, i].numpy(),
                        line_style,
                        alpha=alpha,
                        label=label,
                    )

        # Plot constraints (only for first joint to avoid clutter)
        has_constraints = (
            hasattr(self.cfg, "constraints") and self.cfg.constraints is not None
        )
        if self.dofs > 0 and has_constraints:
            plot_idx = 1
            if vels is not None:
                max_vel = self.cfg.constraints["velocity"][0]
                axs[plot_idx].plot(
                    time_steps,
                    [-max_vel] * len(time_steps),
                    "k--",
                    label="Max Velocity",
                )
                axs[plot_idx].plot(time_steps, [max_vel] * len(time_steps), "k--")
                plot_idx += 1

            if accs is not None:
                max_acc = self.cfg.constraints["acceleration"][0]
                axs[plot_idx].plot(
                    time_steps,
                    [-max_acc] * len(time_steps),
                    "k--",
                    label="Max Acceleration",
                )
                axs[plot_idx].plot(time_steps, [max_acc] * len(time_steps), "k--")

        axs[0].set_title("Position")
        plot_idx = 1
        if vels is not None:
            axs[plot_idx].set_title("Velocity")
            plot_idx += 1
        if accs is not None:
            axs[plot_idx].set_title("Acceleration")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
            ax.grid()

        plt.tight_layout()
        plt.show()
