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
from .utils import PlanState, PlanResult, calculate_point_allocations, interpolate_xpos
from .utils import MovePart, MoveType

__all__ = ["BasePlannerCfg", "PlanOptions", "BasePlanner"]


@configclass
class BasePlannerCfg:

    robot_uid: str = MISSING
    """UID of the robot to control. Must correspond to a robot added to the simulation with this UID."""

    planner_type: str = "Base"


@configclass
class PlanOptions:
    start_qpos: torch.Tensor | None = None
    """Optional starting joint configuration for the trajectory. If provided, the planner will ensure that the trajectory starts from this configuration. If not provided, the planner will use the current joint configuration of the robot as the starting point."""

    control_part: str | None = None
    """Name of the robot part to control, e.g. 'left_arm'. Must correspond to a valid control part defined in the robot's configuration."""

    is_pre_interpolate: bool = False
    """Whether to perform interpolation before planning. If True, the planner will first interpolate the trajectory based on the provided waypoints and then plan a trajectory through the interpolated points. If False, the planner will directly plan through the provided waypoints without interpolation."""

    is_linear: bool = False
    """If True, use cartesian linear interpolation, else joint space"""

    interpolate_position_step: float = 0.002
    """Step size for interpolation. If is_linear is True, this is the step size in Cartesian space (meters). If is_linear is False, this is the step size in joint space (radians)."""

    interpolate_angle_step: float = np.pi / 90
    """Angular step size for interpolation in joint space (radians). Only used if is_linear is False."""


class BasePlanner(ABC):
    r"""Base class for trajectory planners.

    This class provides common functionality that can be shared across different
    planner implementations.

    Args:
        cfg: Configuration object for the planner.
    """

    def __init__(self, cfg: BasePlannerCfg):
        self.cfg: BasePlannerCfg = cfg

        if cfg.robot_uid is MISSING:
            logger.log_error("robot_uid is required in planner config", ValueError)

        self.robot = SimulationManager.get_instance().get_robot(cfg.robot_uid)
        if self.robot is None:
            logger.log_error(f"Robot with uid {cfg.robot_uid} not found", ValueError)

        self.device = self.robot.device

    @abstractmethod
    def plan(
        self,
        target_states: list[PlanState],
        plan_option: PlanOptions = PlanOptions(),
    ) -> PlanResult:
        r"""Execute trajectory planning.

        This method must be implemented by subclasses to provide the specific
        planning algorithm.

        Args:
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

    def interpolate_trajectory(
        self,
        control_part: str | None = None,
        xpos_list: torch.Tensor | None = None,
        qpos_list: torch.Tensor | None = None,
        cfg: PlanOptions = PlanOptions(),
    ) -> tuple[PlanState, list[PlanState]]:
        if qpos_list is not None:
            if not isinstance(qpos_list, torch.Tensor):
                qpos_list = np.asarray(qpos_list)
            qpos_tensor = torch.as_tensor(
                qpos_list, dtype=torch.float32, device=self.robot.device
            )
            if qpos_tensor.dim() == 1:
                qpos_tensor = qpos_tensor.unsqueeze(0)

            qpos_batch = qpos_tensor.unsqueeze(0)  # [n_env=1, n_batch=N, dof]
            xpos_batch = self.robot.compute_batch_fk(
                qpos=qpos_batch,
                name=control_part,
                to_matrix=True,
            )
            xpos_list = xpos_batch.squeeze(0)
            qpos_list = qpos_tensor

        if xpos_list is None:
            logger.log_warning("Either xpos_list or qpos_list must be provided")
            empty_qpos = torch.empty((0, self.dof), dtype=torch.float32)
            empty_xpos = torch.empty((0, 4, 4), dtype=torch.float32)
            return empty_qpos, empty_xpos

        if not isinstance(xpos_list, torch.Tensor):
            xpos_list = torch.as_tensor(
                np.asarray(xpos_list),
                dtype=torch.float32,
                device=self.robot.device,
            )
        else:
            xpos_list = xpos_list.to(dtype=torch.float32, device=self.robot.device)

        if cfg.start_qpos is not None:
            start_xpos = self.robot.compute_fk(
                qpos=cfg.start_qpos.unsqueeze(0), name=control_part, to_matrix=True
            )
            qpos_list = (
                torch.cat([cfg.start_qpos.unsqueeze(0), qpos_list], dim=0)
                if qpos_list is not None
                else None
            )
            xpos_list = torch.cat([start_xpos, xpos_list], dim=0)
        # Input validation
        if len(xpos_list) < 2:
            logger.log_warning("xpos_list must contain at least 2 points")
            return None, None

        # Calculate point allocations for interpolation
        interpolated_point_allocations = calculate_point_allocations(
            xpos_list, step_size=0.002, angle_step=np.pi / 90, device=self.device
        )
        # Input validation
        if len(xpos_list) < 2:
            logger.log_warning("xpos_list must contain at least 2 points")
            empty_qpos = torch.empty((0, self.dof), dtype=torch.float32)
            empty_xpos = torch.empty((0, 4, 4), dtype=torch.float32)
            return empty_qpos, empty_xpos

        # Calculate point allocations for interpolation
        interpolated_point_allocations = calculate_point_allocations(
            xpos_list, step_size=0.002, angle_step=np.pi / 90
        )

        # currently we use
        qpos_seed = cfg.start_qpos
        if qpos_seed is None and qpos_list is not None:
            qpos_seed = qpos_list[0]
        # Generate trajectory
        interpolate_qpos_list = []
        if cfg.is_linear or qpos_list is None:
            # Linear cartesian interpolation
            feasible_pose_targets = []
            for i in range(len(xpos_list) - 1):
                interpolated_poses = interpolate_xpos(
                    (
                        xpos_list[i].detach().cpu().numpy()
                        if isinstance(xpos_list, torch.Tensor)
                        else xpos_list[i]
                    ),
                    (
                        xpos_list[i + 1].detach().cpu().numpy()
                        if isinstance(xpos_list, torch.Tensor)
                        else xpos_list[i + 1]
                    ),
                    interpolated_point_allocations[i],
                )
                for xpos in interpolated_poses:
                    success, qpos = self.robot.compute_ik(
                        pose=xpos, joint_seed=qpos_seed, name=control_part
                    )

                    if isinstance(success, torch.Tensor):
                        is_success = bool(success.all())
                    elif isinstance(success, np.ndarray):
                        is_success = bool(np.all(success))
                    elif isinstance(success, (list, tuple)):
                        is_success = all(success)
                    else:
                        is_success = bool(success)

                    if isinstance(qpos, torch.Tensor):
                        has_nan = torch.isnan(qpos).any().item()
                    else:
                        has_nan = np.isnan(qpos).any()

                    if not is_success or qpos is None or has_nan:
                        logger.log_debug(
                            f"IK failed or returned nan at pose, skipping this point."
                        )
                        continue

                    q_entry = qpos[0] if isinstance(qpos, (np.ndarray, list)) else qpos
                    if isinstance(q_entry, torch.Tensor) and q_entry.dim() > 1:
                        q_entry = q_entry.squeeze(0)
                    interpolate_qpos_list.append(q_entry)
                    feasible_pose_targets.append(xpos)
                    qpos_seed = q_entry

            # Vectorized FK feasibility check to keep only physically consistent IK outputs.
            if len(interpolate_qpos_list) > 0:
                qpos_tensor = torch.stack(
                    [
                        (
                            q.to(dtype=torch.float32, device=self.robot.device)
                            if isinstance(q, torch.Tensor)
                            else torch.as_tensor(
                                q, dtype=torch.float32, device=self.robot.device
                            )
                        )
                        for q in interpolate_qpos_list
                    ]
                )
                fk_batch = self.robot.compute_batch_fk(
                    qpos=qpos_tensor.unsqueeze(0),
                    name=control_part,
                    to_matrix=True,
                ).squeeze(0)
                target_pose_tensor = torch.as_tensor(
                    np.asarray(feasible_pose_targets),
                    dtype=torch.float32,
                    device=self.robot.device,
                )
                pos_err = torch.norm(
                    fk_batch[:, :3, 3] - target_pose_tensor[:, :3, 3], dim=-1
                )
                rot_err = torch.norm(
                    fk_batch[:, :3, :3] - target_pose_tensor[:, :3, :3],
                    dim=(-2, -1),
                )
                valid_mask = (pos_err < 0.02) & (rot_err < 0.2)
                interpolate_qpos_list = [
                    q
                    for q, is_valid in zip(interpolate_qpos_list, valid_mask)
                    if bool(is_valid.item())
                ]
        else:
            # Joint space interpolation
            interpolate_qpos_list = [q for q in qpos_list]

        if len(interpolate_qpos_list) < 2:
            logger.log_error("Need at least 2 waypoints for trajectory planning")
            return None, None

        init_state = PlanState(
            move_type=MoveType.JOINT_MOVE, qpos=interpolate_qpos_list[0]
        )
        target_states = []
        for qpos in interpolate_qpos_list:
            target_states.append(PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos))

        return init_state, target_states
