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

import os
import torch
import numpy as np
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from embodichain.utils import logger

from embodichain.lab.sim.utility.import_utils import (
    lazy_import_pinocchio,
    lazy_import_pink,
)
from embodichain.lab.sim.utility.solver_utils import (
    build_reduced_pinocchio_robot,
    compute_pinocchio_fk,
)

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver

from embodichain.utils.string import (
    is_regular_expression,
    resolve_matching_names_values,
)

if TYPE_CHECKING:
    from typing import Self


@configclass
class PinkSolverCfg(SolverCfg):
    """Configuration for Pink IK Solver."""

    class_type: str = "PinkSolver"

    # Solver iteration parameters
    pos_eps: float = 5e-4  # Tolerance for convergence for position
    rot_eps: float = 5e-4  # Tolerance for convergence for rotation
    max_iterations: int = 1000  # Maximum number of iterations for the solver
    dt: float = 0.1  # Time step for numerical integration
    damp: float = 1e-6  # Damping factor to prevent numerical instability

    # Constraint configuration
    is_only_position_constraint: bool = (
        False  # Whether to only consider position constraints
    )

    # Path to the mesh files associated with the robot. These files are also loaded by Pinocchio's `robot_wrapper.BuildFromURDF`.
    mesh_path: Optional[str] = None

    # A list of tasks for the Pink IK controller. These tasks are controllable by the env action.
    # These tasks can be used to control the pose of a frame or the angles of joints.
    # For more details, visit: https://github.com/stephane-caron/pink
    variable_input_tasks: List["pink.tasks.FrameTask"] = None

    # A list of tasks for the Pink IK controller. These tasks are fixed and not controllable by the env action.
    # These tasks can be used to fix the pose of a frame or the angles of joints to a desired configuration.
    # For more details, visit: https://github.com/stephane-caron/pink
    fixed_input_tasks: List["pink.tasks.FrameTask"] = None

    # Show warning if IK solver fails to find a solution.
    show_ik_warnings: bool = True

    # If True, the Pink IK solver will fail and raise an error if any joint limit is violated during optimization.
    # PinkSolver will handle the error by setting the last joint positions.
    # If False, the solver will ignore joint limit violations and return the closest solution found.
    fail_on_joint_limit_violation: bool = True

    # Solver options:
    # "clarabel": High-performance SOCP solver written in Rust.
    #   - Suitable for large-scale problems.
    #   - Fast and supports sparse matrices.
    #
    # "ecos": Efficient SOCP solver for small to medium-scale problems.
    #   - Fast and memory-efficient.
    #
    # "osqp": Quadratic programming solver based on ADMM.
    #   - Ideal for sparse and large-scale QP problems.
    #   - Numerically stable and widely used in robotics/control.
    #
    # "proxqp": C++ solver for dense and sparse QP problems.
    #   - Optimized for real-time applications.
    #
    # "scs": Solver for linear cone programming and SOCP.
    #   - Suitable for large-scale problems with low precision requirements.
    #
    # "daqp": Specialized QP solver for real-time and embedded systems.
    #   - Designed for fast and reliable quadratic programming.
    solver_type = "osqp"

    def init_solver(self, **kwargs) -> "PinkSolver":
        """Initialize the solver with the configuration.

        Args:
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            PinkSolver: An initialized solver instance.
        """

        solver = PinkSolver(cfg=self, **kwargs)

        # Set the Tool Center Point (TCP) for the solver
        solver.set_tcp(self._get_tcp_as_numpy())

        return solver


class PinkSolver(BaseSolver):
    """Standalone implementation of Pink IK Solver."""

    def __init__(self, cfg: PinkSolverCfg, **kwargs):
        """Initialize the solver with the configuration.

        Args:
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            PinkSolver: An initialized solver instance.
        """
        super().__init__(cfg=cfg, **kwargs)

        self.pin = lazy_import_pinocchio()
        self.pink = lazy_import_pink()

        from embodichain.lab.sim.solvers.null_space_posture_task import (
            NullSpacePostureTask,
        )

        self.tcp = cfg.tcp

        if cfg.mesh_path is None:
            urdf_dir = os.path.dirname(cfg.urdf_path)
            cfg.mesh_path = urdf_dir

        # Initialize robot model
        self.entire_robot = self.pin.RobotWrapper.BuildFromURDF(
            self.cfg.urdf_path, self.cfg.mesh_path, root_joint=None
        )

        self.pink_joint_names = self.entire_robot.model.names.tolist()[
            1:
        ]  # Exclude 'universe' joint

        self.pink_dof = (
            self.entire_robot.model.njoints - 1
        )  # Degrees of freedom of robot joints

        # Get reduced robot model
        self.robot = build_reduced_pinocchio_robot(self.entire_robot, self.joint_names)

        # Initialize Pink configuration
        self.pink_cfg = self.pink.configuration.Configuration(
            self.robot.model, self.robot.data, self.robot.q0
        )

        if self.cfg.variable_input_tasks is None:
            self.cfg.variable_input_tasks = [
                self.pink.tasks.FrameTask(
                    frame=self.cfg.end_link_name,  # Frame name (use actual frame name from URDF)
                    position_cost=1.0,  # Position cost weight
                    orientation_cost=1.0,  # Orientation cost weight
                )
            ]

        if self.cfg.fixed_input_tasks is None:
            self.cfg.fixed_input_tasks = []

        # Set default targets for tasks
        for task in self.cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_qpos)
                continue
            task.set_target_from_configuration(self.pink_cfg)
        for task in self.cfg.fixed_input_tasks:
            task.set_target_from_configuration(self.pink_cfg)

        # Create joint name mappings if provided
        if self.cfg.joint_names:
            pink_joint_names = self.robot.model.names.tolist()[
                1:
            ]  # Exclude 'universe' joint
            self.dexsim_to_pink_ordering = [
                self.cfg.joint_names.index(pink_joint)
                for pink_joint in pink_joint_names
            ]
            self.pink_to_dexsim_ordering = [
                pink_joint_names.index(isaac_joint)
                for isaac_joint in self.cfg.joint_names
            ]
        else:
            self.dexsim_to_pink_ordering = None
            self.pink_to_dexsim_ordering = None

    def reorder_array(
        self, input_array: List[float], reordering_array: List[int]
    ) -> List[float]:
        """Reorder array elements based on provided indices.

        Args:
            input_array: Array to reorder
            reordering_array: Indices for reordering

        Returns:
            Reordered array
        """
        return [input_array[i] for i in reordering_array]

    def update_null_space_joint_targets(self, current_qpos: np.ndarray):
        """Update the null space joint targets.

        This method updates the target joint positions for null space posture tasks based on the current
        joint configuration. This is useful for maintaining desired joint configurations when the primary
        task allows redundancy.

        Args:
            current_qpos: The current joint positions of shape (num_joints,).
        """
        from embodichain.lab.sim.solvers.null_space_posture_task import (
            NullSpacePostureTask,
        )

        for task in self.cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(current_qpos)

    def get_ik(
        self,
        target_xpos: Optional[Union[torch.Tensor, np.ndarray]],
        qpos_seed: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute target joint positions using inverse kinematics.

        Args:
            target_pose (Optional[Union[torch.Tensor, np.ndarray]]): Target end-effector pose
            qpos_seed (Optional[Union[torch.Tensor, np.ndarray]]): Seed joint positions
            return_all_solutions (bool, optional): Whether to return all IK solutions or just the best one. Defaults to False.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            Target joint positions. (n_sample, 1, dof) of float.
        """
        if qpos_seed is None:
            qpos_seed = np.zeros(self.dof)

        if isinstance(qpos_seed, torch.Tensor):
            qpos_seed = qpos_seed.detach().cpu().numpy()
        if qpos_seed.ndim > 1:
            qpos_seed = qpos_seed.flatten()

        if target_xpos.ndim == 2:
            target_xpos = target_xpos.unsqueeze(0)
        if isinstance(target_xpos, torch.Tensor):
            target_xpos = target_xpos.detach().cpu().numpy()

        if target_xpos.shape == (1, 4, 4):
            target_xpos = target_xpos[0]

        if target_xpos.shape == (4, 4):
            xpos = self.pin.SE3(target_xpos)
        else:
            raise ValueError(
                f"target_xpos shape {target_xpos.shape} not supported for SE3 construction."
            )

        self.cfg.variable_input_tasks[0].set_target(xpos)

        # Handle joint ordering if mapping provided
        if self.dexsim_to_pink_ordering:
            qpos_pink = np.array(
                self.reorder_array(qpos_seed, self.dexsim_to_pink_ordering)
            )
        else:
            qpos_pink = np.array(qpos_seed)

        # Update configuration with current joint positions
        self.pink_cfg.update(qpos_pink)

        tasks = self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks

        try:
            num_iter = 1 if self.cfg.max_iterations == 1 else self.cfg.max_iterations

            for i in range(num_iter):
                # Solve IK to get joint velocities
                velocity = self.pink.solve_ik(
                    configuration=self.pink_cfg,
                    tasks=tasks,
                    damping=self.cfg.damp,
                    dt=self.cfg.dt,
                    solver=self.cfg.solver_type,
                    safety_break=self.cfg.fail_on_joint_limit_violation,
                )
                self.pink_cfg.integrate_inplace(velocity, self.cfg.dt)
                err = self.cfg.variable_input_tasks[0].compute_error(self.pink_cfg)

                # Compute joint position changes
                # Update joint positions
                # delta_q = velocity * self.cfg.dt
                # self.pink_cfg.update(delta_q)
                # logger.log_warning(f"Iteration {i}, error: {err}, delta_q: {delta_q}")
                pos_achieved = np.linalg.norm(err[:3]) <= self.cfg.pos_eps

                if self.cfg.is_only_position_constraint:
                    if pos_achieved:
                        break
                else:
                    ori_achieved = np.linalg.norm(err[3:]) <= self.cfg.rot_eps
                    if pos_achieved and ori_achieved:
                        break

        # except NoSolutionFound as e:
        except (AssertionError, Exception) as e:
            # Print warning and return the current joint positions as the target
            # Not using omni.log since its not available in CI during docs build
            if self.cfg.show_ik_warnings:
                logger.log_warning(
                    "Warning: IK quadratic solver could not find a solution! Did not update the target joint"
                    f" positions.\nError: {e}"
                )
            return torch.tensor(False, dtype=torch.bool), torch.tensor(
                qpos_seed, device=self.device, dtype=torch.float32
            )

        qpos = torch.tensor(
            self.pink_cfg.q[self.pink_to_dexsim_ordering],
            device=self.device,
            dtype=torch.float32,
        )

        if return_all_solutions:
            logger.log_warning(
                "return_all_solutions=True is not supported in DifferentialSolver. Returning the best solution only."
            )

        # Add the velocity changes to the current joint positions to get the target joint positions
        # target_qpos = torch.add(
        #     qvel_dexsim,
        #     torch.tensor(joint_seed, device=self.device, dtype=torch.float32),
        # )
        dof = qpos.shape[-1]
        qpos = qpos.reshape(-1, 1, dof)
        return torch.tensor(True, dtype=torch.bool), qpos

    def _get_fk(
        self,
        qpos: Optional[Union[torch.Tensor, np.ndarray]],
        **kwargs,
    ) -> torch.tensor:
        """Compute the forward kinematics for the robot given joint positions.

        Args:
            qpos (torch.Tensor or np.ndarray): Joint positions, shape should be (nq,).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            torch.Tensor: The homogeneous transformation matrix (4x4) of the end-effector (after applying TCP).
        """
        result = compute_pinocchio_fk(
            self.pin, self.robot, qpos, self.end_link_name, self.tcp_xpos
        )
        return torch.from_numpy(result)
