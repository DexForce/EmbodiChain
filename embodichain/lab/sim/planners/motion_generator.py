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
    NeuralPlanner,
    NeuralPlannerCfg,
)
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.utils import logger, configclass
from .utils import MovePart, MoveType, PlanState, PlanResult
from .utils import (
    calculate_point_allocations,
    interpolate_xpos,
    interpolate_xpos_batched,
)

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
    """Optional starting joint configuration for the trajectory, shape (B, DOF). If provided, the planner will ensure that the trajectory starts from this configuration. If not provided, the planner will use the current joint configuration of the robot as the starting point."""

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
        "neural": (NeuralPlanner, NeuralPlannerCfg),
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
            move_type = target_states[0].move_type
            if move_type == MoveType.EEF_MOVE:
                for s in target_states:
                    if s.move_type != move_type:
                        logger.log_error(
                            f"All states must share move_type; got {s.move_type}",
                            ValueError,
                        )
                xpos_list = torch.stack([s.xpos for s in target_states]).transpose(
                    0, 1
                )  # (B, N, 4, 4)
                qpos_list = None
            elif move_type == MoveType.JOINT_MOVE:
                for s in target_states:
                    if s.move_type != move_type:
                        logger.log_error(
                            f"All states must share move_type; got {s.move_type}",
                            ValueError,
                        )
                qpos_list = torch.stack([s.qpos for s in target_states]).transpose(
                    0, 1
                )  # (B, N, DOF)
                xpos_list = None
            else:
                logger.log_error(
                    f"Unsupported move type for pre-interpolation: {move_type}"
                )

            if options.start_qpos is not None:
                start = options.start_qpos
                if start.dim() == 1:
                    start = start.unsqueeze(0)
                if qpos_list is not None:
                    qpos_list = torch.cat([start.unsqueeze(1), qpos_list], dim=1)
                if xpos_list is not None:
                    start_xpos = self.robot.compute_fk(
                        qpos=start, name=options.control_part, to_matrix=True
                    )
                    if start_xpos.dim() == 3:
                        start_xpos = start_xpos.unsqueeze(1)
                    xpos_list = torch.cat([start_xpos, xpos_list], dim=1)

            qpos_interpolated, xpos_interpolated = self.interpolate_trajectory(
                control_part=options.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                options=options,
            )
            if not options.plan_opts:
                return PlanResult(
                    success=True,
                    positions=qpos_interpolated,
                    xpos_list=xpos_interpolated,
                )

            target_plan_states = [
                PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos_interpolated[:, j])
                for j in range(qpos_interpolated.shape[1])
            ]
        else:
            target_plan_states = target_states

        if options.plan_opts is None:
            if hasattr(self.planner, "default_plan_options"):
                options.plan_opts = self.planner.default_plan_options()
            else:
                options.plan_opts = PlanOptions()
        return self.planner.plan(
            target_states=target_plan_states, options=options.plan_opts
        )

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

        This method performs interpolation on the provided waypoints to generate a
        smoother trajectory. It supports both Cartesian (end-effector) and joint
        space interpolation based on the control part and options specified.

        Args:
            control_part: Name of the robot part to control, e.g. 'left_arm'. Must
                correspond to a valid control part defined in the robot's configuration.
            xpos_list: End-effector poses, shape ``(B, N, 4, 4)`` or ``(N, 4, 4)``.
                Required if control_part is an end-effector control part.
            qpos_list: Joint positions, shape ``(B, N, DOF)`` or ``(N, DOF)``.
                Required if control_part is a joint control part.
            options: MotionGenOptions containing interpolation settings such as step
                size and whether to use linear interpolation.

        Returns:
            Tuple containing:
                - interpolate_qpos_list: Interpolated joint positions, shape
                  ``(B, M, DOF)``.
                - feasible_pose_targets: Corresponding end-effector poses, shape
                  ``(B, M, 4, 4)``, or ``None`` if not applicable.
        """

        # Normalize single-env inputs to batched form.
        if qpos_list is not None and qpos_list.dim() == 2:
            qpos_list = qpos_list.unsqueeze(0)
        if xpos_list is not None and xpos_list.dim() == 3:
            xpos_list = xpos_list.unsqueeze(0)

        if qpos_list is not None and xpos_list is None and options.is_linear:
            # qpos_list is (B, N, DOF); compute_batch_fk handles batched qpos directly.
            xpos_list = self.robot.compute_batch_fk(
                qpos=qpos_list,
                name=control_part,
                to_matrix=True,
            )  # (B, N, 4, 4)

        if xpos_list is None and qpos_list is None:
            logger.log_error("Either xpos_list or qpos_list must be provided")

        # Input validation: the waypoint count is the second-to-last or last batch dim.
        if (xpos_list is not None and xpos_list.shape[-3] < 2) or (
            qpos_list is not None and qpos_list.shape[-2] < 2
        ):
            logger.log_error(
                "xpos_list and qpos_list must contain at least 2 way points"
            )

        qpos_seed = options.start_qpos
        if qpos_seed is not None and qpos_seed.dim() == 1:
            qpos_seed = qpos_seed.unsqueeze(0)
        if qpos_seed is None and qpos_list is not None:
            # First waypoint per env as seed.
            qpos_seed = qpos_list[:, 0]  # (B, DOF)
        if qpos_seed is None:
            # Fallback to current robot state as seed.
            qpos_seed = self.robot.get_qpos(name=control_part)  # (B, DOF)

        # Generate trajectory
        if options.is_linear or qpos_list is None:
            # ``calculate_point_allocations`` only handles single-env (N, 4, 4),
            # so compute allocations per env and use the per-segment maximum so
            # all envs can share the same interpolated pose count.
            per_env_allocations = [
                calculate_point_allocations(
                    xpos_list[b],
                    step_size=options.interpolate_position_step,
                    angle_step=options.interpolate_angle_step,
                    device=self.device,
                )
                for b in range(xpos_list.shape[0])
            ]
            n_segments = xpos_list.shape[1] - 1
            interpolated_point_allocations = [
                max(alloc[i] for alloc in per_env_allocations)
                for i in range(n_segments)
            ]

            # Linear cartesian interpolation, batched across B envs.
            total_interpolated_poses = []
            for i in range(n_segments):
                seg = interpolate_xpos_batched(
                    xpos_list[:, i],
                    xpos_list[:, i + 1],
                    interpolated_point_allocations[i],
                )  # (B, seg, 4, 4)
                total_interpolated_poses.append(seg)
            total_interpolated_poses = torch.cat(
                total_interpolated_poses, dim=1
            )  # (B, M, 4, 4)

            qpos_seed_b = qpos_seed
            if qpos_seed_b.dim() == 1:
                qpos_seed_b = qpos_seed_b.unsqueeze(0).repeat(xpos_list.shape[0], 1)
            joint_seed = qpos_seed_b.unsqueeze(1).repeat(
                1, total_interpolated_poses.shape[1], 1
            )  # (B, M, D)
            success_batch, qpos_batch = self.robot.compute_batch_ik(
                pose=total_interpolated_poses,
                joint_seed=joint_seed,
                name=control_part,
            )  # (B, M), (B, M, D)

            has_nan = torch.isnan(qpos_batch).any(dim=-1)
            valid = success_batch.bool() & (~has_nan)  # (B, M)

            # Vectorized FK feasibility check to keep only physically consistent IK outputs.
            if valid.any():
                fk_batch = self.robot.compute_batch_fk(
                    qpos=qpos_batch,
                    name=control_part,
                    to_matrix=True,
                )  # (B, M, 4, 4)
                pos_err = torch.norm(
                    fk_batch[:, :, :3, 3] - total_interpolated_poses[:, :, :3, 3],
                    dim=-1,
                )
                rot_err = torch.norm(
                    fk_batch[:, :, :3, :3] - total_interpolated_poses[:, :, :3, :3],
                    dim=(-2, -1),
                )
                fk_valid = (pos_err < 0.02) & (rot_err < 0.2)
                valid = valid & fk_valid

            # Per-env filter: keep only valid rows; pad short envs by repeating last valid.
            B, M, D = qpos_batch.shape
            max_valid = int(valid.sum(dim=1).max().item())
            max_valid = max(max_valid, 1)
            interp_q = torch.zeros(
                B, max_valid, D, device=self.device, dtype=torch.float32
            )
            feasible = torch.zeros(
                B, max_valid, 4, 4, device=self.device, dtype=torch.float32
            )
            for b in range(B):
                v = qpos_batch[b][valid[b]]
                f = total_interpolated_poses[b][valid[b]]
                if v.shape[0] == 0:
                    v = qpos_batch[b : b + 1, 0]
                    f = total_interpolated_poses[b : b + 1, 0]
                interp_q[b, : v.shape[0]] = v
                interp_q[b, v.shape[0] :] = v[-1]
                feasible[b, : f.shape[0]] = f
                feasible[b, f.shape[0] :] = f[-1]
            interpolate_qpos_list = interp_q
            feasible_pose_targets = feasible
        else:
            # Joint-space interpolation. qpos_list is (B, N, DOF).
            if isinstance(options.interpolate_nums, int):
                interp_nums = [options.interpolate_nums] * (qpos_list.shape[1] - 1)
            else:
                if len(options.interpolate_nums) != qpos_list.shape[1] - 1:
                    logger.log_error(
                        "Length of interpolate_nums list must equal number of segments",
                        ValueError,
                    )
                interp_nums = options.interpolate_nums

            interpolate_qpos_list = interpolate_with_nums(
                qpos_list, interp_nums=interp_nums, device=self.device
            )  # (B, M, DOF)
            feasible_pose_targets = None

        return interpolate_qpos_list, feasible_pose_targets
