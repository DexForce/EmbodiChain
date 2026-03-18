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
from embodichain.utils import logger, configclass
from .utils import MovePart, MoveType, PlanState, PlanResult
from .utils import calculate_point_allocations, interpolate_xpos


__all__ = ["MotionGenerator", "MotionGenCfg"]


@configclass
class MotionGenCfg:

    planner_cfg: BasePlannerCfg = MISSING
    """Configuration for the underlying planner. Must include 'planner_type' attribute to specify 
    which planner to use, and any additional parameters required by that planner.
    """

    # TODO: More configuration options can be added here in the future.


@configclass
class MotionGenOptions:

    plan_opts: PlanOptions = PlanOptions()

    is_pre_interpolate: bool = False
    """Whether to perform interpolation before planning. 
    
    If True, the planner will first interpolate the trajectory based on the provided waypoints and then plan a trajectory through the interpolated points. 
    If False, the planner will directly plan through the provided waypoints without interpolation.
    
    Note:
        - The pre-interpolation only works for PlanState with MoveType.EEF_MOVE or MoveType.JOINT_MOVE.
    """

    is_linear: bool = True
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

    def plan(
        self,
        target_states: List[PlanState],
        opts: MotionGenOptions = MotionGenOptions(),
    ) -> PlanResult:
        r"""Plan trajectory with given options.

        This method generates a smooth trajectory using the selected planner that satisfies
        constraints and perform pre-interpolation if specified in the options.

        Args:
            target_states: List[PlanState]
            opts: MotionGenOptions

        Returns:
            PlanResult containing the planned trajectory details.
        """
        if opts.is_pre_interpolate:
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
            target_plan_states = self.interpolate_trajectory(
                control_part=opts.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                cfg=opts,
            )
        else:
            target_plan_states = target_states

        result = self.planner.plan(
            target_states=target_plan_states, plan_opts=opts.plan_opts
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
        cfg: PlanOptions = PlanOptions(),
    ) -> tuple[PlanState, list[PlanState]]:
        r"""Interpolate trajectory based on provided waypoints.
            This method performs interpolation on the provided waypoints to generate a smoother trajectory.
            It supports both Cartesian (end-effector) and joint space interpolation based on the control part and options specified.

        Args:
            control_part: Name of the robot part to control, e.g. 'left_arm'. Must correspond to a valid control part defined in the robot's configuration.
            xpos_list: List of end-effector poses (torch.Tensor of shape [N, 4, 4]) to interpolate through. Required if control_part is an end-effector control part.
            qpos_list: List of joint positions (torch.Tensor of shape [N, DOF]) to interpolate through. Required if control_part is a joint control part.
            cfg: PlanOptions containing interpolation settings such as step size and whether to use linear interpolation.

        Returns:
            init_state: PlanState representing the initial state of the trajectory (first waypoint).
            target_states: List of PlanState representing the interpolated waypoints for the trajectory.
        """

        if qpos_list is not None:
            if qpos_list.dim() == 1:
                qpos_list = qpos_list.unsqueeze(0)

            qpos_batch = qpos_list.unsqueeze(0)  # [n_env=1, n_batch=N, dof]
            xpos_batch = self.robot.compute_batch_fk(
                qpos=qpos_batch,
                name=control_part,
                to_matrix=True,
            )
            xpos_list = xpos_batch.squeeze(0)
            qpos_list = qpos_tensor

        if xpos_list is None:
            logger.log_error("Either xpos_list or qpos_list must be provided")

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

        # currently we use
        qpos_seed = cfg.start_qpos
        if qpos_seed is None and qpos_list is not None:
            qpos_seed = qpos_list[0]

        # Generate trajectory
        interpolate_qpos_list = []
        if cfg.is_linear or qpos_list is None:
            # Calculate point allocations for interpolation
            interpolated_point_allocations = calculate_point_allocations(
                xpos_list,
                step_size=cfg.interpolate_position_step,
                angle_step=cfg.interpolate_angle_step,
                device=self.device,
            )

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

        target_states = []
        for qpos in interpolate_qpos_list:
            target_states.append(PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos))

        return target_states
