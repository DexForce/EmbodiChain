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

from dataclasses import MISSING
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any

from embodichain.lab.sim.planners import (
    BasePlannerCfg,
    PlanOptions,
    BasePlanner,
    ToppraPlanner,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.utils import logger, configclass
from .utils import MovePart, MoveType, PlanState, PlanResult
from .utils import calculate_point_allocations, interpolate_xpos


__all__ = ["MotionGenerator", "MotionGenCfg", "MotionGenOptions"]


@configclass
class MotionGenCfg:

    planner_cfg: BasePlannerCfg = MISSING
    """Configuration for the underlying planner. Must include 'planner_type' attribute to specify 
    which planner to use, and any additional parameters required by that planner.
    """

    # TODO: More configuration options can be added here in the future.


@configclass
class MotionGenOptions:

    start_qpos: torch.Tensor | None = None
    """Optional starting joint configuration for the trajectory. If provided, the planner will ensure that the trajectory starts from this configuration. If not provided, the planner will use the current joint configuration of the robot as the starting point."""

    control_part: str | None = None
    """Name of the robot part to control, e.g. 'left_arm'. Must correspond to a valid control part defined in the robot's configuration."""

    plan_opts: PlanOptions | None = None
    """Options to pass to the underlying planner during the planning phase."""

    is_interpolate: bool = False
    """Whether to perform interpolation before planning. 
    
    Note:
        - The pre-interpolation only works for PlanState with MoveType.EEF_MOVE or MoveType.JOINT_MOVE.
    """

    interpolate_nums: int | list[int] = 10
    """Number of interpolation points to generate between each pair of waypoints. 
    
    Can be an integer (same for all segments) or a list of integers with len(PlanState) specifying the number of points for each segment."""

    is_linear: bool = False
    """If True, use cartesian linear interpolation, else joint space"""

    interpolate_position_step: float = 0.002
    """Step size for interpolation. If is_linear is True, this is the step size in Cartesian space (meters). If is_linear is False, this is the step size in joint space (radians)."""

    interpolate_angle_step: float = np.pi / 90
    """Angular step size for interpolation in joint space (radians). Only used if is_linear is False."""


class MotionGenerator:
    r"""Unified motion generator for robot trajectory planning.

    This class provides a unified interface for trajectory planning with and without
    collision checking.

    Args:
        cfg: Configuration object for motion generation, must include 'planner_cfg' attribute
    """

    _support_planner_dict = {
        "toppra": (ToppraPlanner, ToppraPlannerCfg),
    }

    def __init__(self, cfg: MotionGenCfg) -> None:

        # Create planner based on planner_type
        self.planner: BasePlanner = self._create_planner(cfg.planner_cfg)

        self.robot = self.planner.robot
        self.device = self.robot.device

    @classmethod
    def register_planner_type(cls, name: str, planner_class, planner_cfg_class) -> None:
        """
        Register a new planner type.
        """
        cls._support_planner_dict[name] = (planner_class, planner_cfg_class)

    def _create_planner(
        self,
        planner_cfg: BasePlannerCfg,
    ) -> BasePlanner:
        r"""Create planner instance based on planner type.

        Args:
            planner_cfg: Configuration object for the planner, must include 'planner_type' attribute

        Returns:
            Planner instance
        """
        planner_type = planner_cfg.planner_type
        if planner_type not in self._support_planner_dict.keys():
            logger.log_error(
                f"Unsupported planner type: {planner_type}. "
                f"Supported types: {list(self._support_planner_dict.keys())}"
            )
        cls = self._support_planner_dict[planner_type][0](cfg=planner_cfg)
        return cls

    def generate(
        self,
        target_states: List[PlanState],
        options: MotionGenOptions = MotionGenOptions(),
    ) -> PlanResult:
        r"""Generate motion with given options.

        This method generates a smooth trajectory using the selected planner that satisfies
        constraints and perform pre-interpolation if specified in the options.

        Args:
            target_states: List[PlanState].
            options: MotionGenOptions.

        Returns:
            PlanResult containing the planned trajectory details.
        """
        if options.is_interpolate:
            # interpolate trajectory to generate more waypoints for smoother motion and better constraint handling
            if target_states[0].move_type == MoveType.EEF_MOVE:
                xpos_list = []
                for state in target_states:
                    if state.move_type != MoveType.EEF_MOVE:
                        logger.log_error(
                            f"All states must be the same. First state is {target_states[0].move_type}, but got {state.move_type}"
                        )
                    xpos_list.append(state.xpos)
                    qpos_list = None
            elif target_states[0].move_type == MoveType.JOINT_MOVE:
                qpos_list = []
                for state in target_states:
                    if state.move_type != MoveType.JOINT_MOVE:
                        logger.log_error(
                            f"All states must be the same. First state is {target_states[0].move_type}, but got {state.move_type}"
                        )
                    qpos_list.append(state.qpos)
                    xpos_list = None
            else:
                logger.log_error(
                    f"Unsupported move type for pre-interpolation: {target_states[0].move_type}"
                )

            if qpos_list is not None:
                qpos_list = torch.stack(qpos_list)
            if xpos_list is not None:
                xpos_list = torch.stack(xpos_list)

            if options.start_qpos is not None:
                if qpos_list is not None:
                    qpos_list = torch.cat(
                        [options.start_qpos.unsqueeze(0), qpos_list], dim=0
                    )
                if xpos_list is not None:
                    start_xpos = self.robot.compute_fk(
                        qpos=options.start_qpos.unsqueeze(0),
                        name=options.control_part,
                        to_matrix=True,
                    )
                    xpos_list = torch.cat([start_xpos, xpos_list], dim=0)

            qpos_interpolated, xpos_interpolated = self.interpolate_trajectory(
                control_part=options.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                options=options,
            )

            if not options.plan_opts:
                # Directly return the interpolated trajectory if no further planning is needed
                return PlanResult(
                    success=True,
                    positions=qpos_interpolated,
                    xpos_list=xpos_interpolated,
                )

            target_plan_states = []
            for qpos in qpos_interpolated:
                target_plan_states.append(
                    PlanState(
                        move_type=MoveType.JOINT_MOVE,
                        qpos=qpos,
                    )
                )
        else:
            target_plan_states = target_states

        options.plan_opts.control_part = options.control_part
        result = self.planner.plan(
            target_states=target_plan_states, options=options.plan_opts
        )
        return result

    def estimate_trajectory_sample_count(
        self,
        xpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        qpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        step_size: float | torch.Tensor = 0.01,
        angle_step: float | torch.Tensor = np.pi / 90,
        control_part: str | None = None,
    ) -> torch.Tensor:
        """Estimate the number of trajectory sampling points required.

        This function estimates the total number of sampling points needed to generate
        a trajectory based on the given waypoints and sampling parameters. Supports
        parallel computation for batched input trajectories.

        Args:
            xpos_list: Tensor of 4x4 transformation matrices, shape [B, N, 4, 4] or [N, 4, 4]
            qpos_list: Tensor of joint positions, shape [B, N, D] or [N, D] (optional)
            step_size: Maximum allowed distance between points (meters). Float or Tensor [B]
            angle_step: Maximum allowed angular difference between points (radians). Float or Tensor [B]

        Returns:
            torch.Tensor: Estimated number of sampling points per trajectory, shape [B]
                          (or scalar tensor if single trajectory)
        """
        # Input validation
        if xpos_list is None and qpos_list is None:
            return torch.tensor(0)

        # Handle lists gracefully if passed by legacy code
        if isinstance(xpos_list, list):
            xpos_list = torch.stack(
                [
                    x if isinstance(x, torch.Tensor) else torch.tensor(x)
                    for x in xpos_list
                ]
            ).float()
        elif isinstance(xpos_list, np.ndarray):
            xpos_list = torch.as_tensor(xpos_list, dtype=torch.float32)

        if isinstance(qpos_list, list):
            qpos_list = torch.stack(
                [
                    q if isinstance(q, torch.Tensor) else torch.tensor(q)
                    for q in qpos_list
                ]
            ).float()
        elif isinstance(qpos_list, np.ndarray):
            qpos_list = torch.as_tensor(qpos_list, dtype=torch.float32)

        device = qpos_list.device if qpos_list is not None else xpos_list.device

        original_dim = qpos_list.dim() if qpos_list is not None else xpos_list.dim()

        # If joint position list is provided but end effector position list is not,
        # convert through forward kinematics
        if qpos_list is not None and xpos_list is None:
            if original_dim == 2:  # [N, D]
                qpos_list = qpos_list.unsqueeze(0)  # [1, N, D]

            B, N, D = qpos_list.shape

            if N < 2:
                return torch.ones((B,), dtype=torch.int32, device=device)

            xpos_list = self.robot.compute_batch_fk(
                qpos=qpos_list,
                name=control_part,
                to_matrix=True,
            )
        else:
            if original_dim == 3:  # [N, 4, 4]
                xpos_list = xpos_list.unsqueeze(0)
            B, N, _, _ = xpos_list.shape

            if N < 2:
                return torch.ones((B,), dtype=torch.int32, device=device)

        # Convert step metrics to tensors
        if not isinstance(step_size, torch.Tensor):
            step_size = torch.full((B,), step_size, device=device, dtype=torch.float32)
        else:
            step_size = step_size.to(device)

        if not isinstance(angle_step, torch.Tensor):
            angle_step = torch.full(
                (B,), angle_step, device=device, dtype=torch.float32
            )
        else:
            angle_step = angle_step.to(device)

        # Calculate position distances
        start_poses = xpos_list[:, :-1]  # [B, N-1, 4, 4]
        end_poses = xpos_list[:, 1:]  # [B, N-1, 4, 4]

        pos_diffs = end_poses[:, :, :3, 3] - start_poses[:, :, :3, 3]
        pos_dists = torch.norm(pos_diffs, dim=-1)  # [B, N-1]
        total_pos_dist = pos_dists.sum(dim=-1)  # [B]

        # Calculate rotation angles
        start_rot = start_poses[:, :, :3, :3]  # [B, N-1, 3, 3]
        end_rot = end_poses[:, :, :3, :3]  # [B, N-1, 3, 3]

        start_rot_T = start_rot.transpose(-1, -2)
        rel_rot = torch.matmul(start_rot_T, end_rot)

        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        # Add epsilon to prevent NaN in acos at boundaries
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)

        angles = torch.acos(cos_angle)  # [B, N-1]
        total_angle = angles.sum(dim=-1)  # [B]

        # Compute sampling points
        pos_samples = torch.clamp((total_pos_dist / step_size).int(), min=1)
        rot_samples = torch.clamp((total_angle / angle_step).int(), min=1)

        total_samples = torch.max(pos_samples, rot_samples)

        out_samples = torch.clamp(total_samples, min=2)

        if original_dim in (2, 3):  # Reshape back to scalar tensor if not batched
            return out_samples[0]

        return out_samples

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
        options: MotionGenOptions = MotionGenOptions(),
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Interpolate trajectory based on provided waypoints.
            This method performs interpolation on the provided waypoints to generate a smoother trajectory.
            It supports both Cartesian (end-effector) and joint space interpolation based on the control part and options specified.

        Args:
            control_part: Name of the robot part to control, e.g. 'left_arm'. Must correspond to a valid control part defined in the robot's configuration.
            xpos_list: List of end-effector poses (torch.Tensor of shape [N, 4, 4]) to interpolate through. Required if control_part is an end-effector control part.
            qpos_list: List of joint positions (torch.Tensor of shape [N, DOF]) to interpolate through. Required if control_part is a joint control part.
            options: MotionGenOptions containing interpolation settings such as step size and whether to use linear interpolation.

        Returns:
            Tuple containing:
                - interpolate_qpos_list: Tensor of interpolated joint positions along the trajectory, shape [M, DOF]
                - feasible_pose_targets: Tensor of corresponding end-effector poses for the interpolated joint positions, shape [M, 4, 4]. This is useful for verifying the interpolation results and can be None if not applicable.
        """

        if qpos_list is not None and xpos_list is None and options.is_linear:
            qpos_batch = qpos_list.unsqueeze(0)  # [n_env=1, n_batch=N, dof]
            xpos_batch = self.robot.compute_batch_fk(
                qpos=qpos_batch,
                name=control_part,
                to_matrix=True,
            )
            xpos_list = xpos_batch.squeeze_(0)

        if xpos_list is None and qpos_list is None:
            logger.log_error("Either xpos_list or qpos_list must be provided")

        # if options.start_qpos is not None:
        #     start_xpos = self.robot.compute_fk(
        #         qpos=options.start_qpos.unsqueeze(0), name=control_part, to_matrix=True
        #     )
        #     qpos_list = (
        #         torch.cat([options.start_qpos.unsqueeze(0), qpos_list], dim=0)
        #         if qpos_list is not None
        #         else None
        #     )
        #     if xpos_list is not None:
        #         xpos_list = torch.cat([start_xpos, xpos_list], dim=0)

        # Input validation
        if (xpos_list is not None and len(xpos_list) < 2) or (
            qpos_list is not None and len(qpos_list) < 2
        ):
            logger.log_error(
                "xpos_list and qpos_list must contain at least 2 way points"
            )

        qpos_seed = options.start_qpos
        if qpos_seed is None and qpos_list is not None:
            qpos_seed = qpos_list[0]

        # Generate trajectory
        interpolate_qpos_list = []
        if options.is_linear or qpos_list is None:
            # Calculate point allocations for interpolation
            interpolated_point_allocations = calculate_point_allocations(
                xpos_list,
                step_size=options.interpolate_position_step,
                angle_step=options.interpolate_angle_step,
                device=self.device,
            )

            # Linear cartesian interpolation
            total_interpolated_poses = []

            # TODO: We may improve the computation efficiency using warp for parallel interpolation of all segments if necessary.
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
                total_interpolated_poses.extend(interpolated_poses)

            total_interpolated_poses = torch.as_tensor(
                np.asarray(total_interpolated_poses),
                dtype=torch.float32,
                device=self.device,
            )

            # Use batch IK for performance
            # compute_batch_ik expects (n_envs, n_batch, 7) or (n_envs, n_batch, 4, 4)
            # Here we assume n_envs = 1 or we want to apply this to all envs if available.
            # Since MotionGenerator usually works with self.robot.device, we use its batching capabilities.
            success_batch, qpos_batch = self.robot.compute_batch_ik(
                pose=total_interpolated_poses.unsqueeze(0),
                joint_seed=None,  # Or use qpos_seed if properly shaped
                name=control_part,
            )

            # success_batch: (n_envs, n_batch), qpos_batch: (n_envs, n_batch, dof)
            success_mask = success_batch[0]  # Take first env
            qpos_results = qpos_batch[0]
            has_nan_mask = torch.isnan(qpos_results).any(dim=-1)

            valid_mask = success_mask & (~has_nan_mask)
            valid_indices = torch.where(valid_mask)[0]

            interpolate_qpos_list = qpos_results[valid_indices]
            feasible_pose_targets = total_interpolated_poses[valid_indices]

            # Vectorized FK feasibility check to keep only physically consistent IK outputs.
            if len(interpolate_qpos_list) > 0:
                fk_batch = self.robot.compute_batch_fk(
                    qpos=interpolate_qpos_list.unsqueeze(0),
                    name=control_part,
                    to_matrix=True,
                ).squeeze_(0)
                pos_err = torch.norm(
                    fk_batch[:, :3, 3] - feasible_pose_targets[:, :3, 3], dim=-1
                )
                rot_err = torch.norm(
                    fk_batch[:, :3, :3] - feasible_pose_targets[:, :3, :3],
                    dim=(-2, -1),
                )
                valid_mask = (pos_err < 0.02) & (rot_err < 0.2)
                interpolate_qpos_list = interpolate_qpos_list[valid_mask]
                feasible_pose_targets = feasible_pose_targets[valid_mask]
        else:
            # Perform joint space interpolation directly if not linear or if end-effector poses are not provided
            qpos_interpolated = qpos_list.unsqueeze_(0).permute(1, 0, 2)  # [N, 1, DOF]

            if isinstance(options.interpolate_nums, int):
                interp_nums = [options.interpolate_nums] * (len(qpos_list) - 1)
            else:
                if len(options.interpolate_nums) != len(qpos_list) - 1:
                    logger.log_error(
                        "Length of interpolate_nums list must be equal to number of segments (len(qpos_list) - 1)"
                    )
                interp_nums = options.interpolate_nums

            interpolate_qpos_list = (
                interpolate_with_nums(
                    qpos_interpolated,
                    interp_nums=interp_nums,
                    device=self.device,
                )
                .permute(1, 0, 2)
                .squeeze_(0)
            )  # [M, DOF]

            feasible_pose_targets = None

        return interpolate_qpos_list, feasible_pose_targets
