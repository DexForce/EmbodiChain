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
from typing import Optional, Union, Tuple, Any, List, TYPE_CHECKING
from itertools import product
from copy import deepcopy

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver

from embodichain.lab.sim.utility.import_utils import (
    lazy_import_pinocchio,
    lazy_import_casadi,
    # lazy_import_pinocchio_casadi,
)


if TYPE_CHECKING:
    from typing import Self


@configclass
class PinocchioSolverCfg(SolverCfg):

    class_type: str = "PinocchioSolver"

    mesh_path: str = None

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

    # Sampling configuration
    num_samples: int = (
        30  # Number of samples to generate different joint seeds for IK iterations
    )

    def init_solver(self, **kwargs) -> "PinocchioSolver":
        """Initialize the solver with the configuration.

        Args:
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            PinocchioSolver: An initialized solver instance.
        """

        solver = PinocchioSolver(cfg=self, **kwargs)

        # Set the Tool Center Point (TCP) for the solver
        if isinstance(self.tcp, torch.Tensor):
            tcp = self.tcp.cpu().numpy()
        else:
            tcp = self.tcp

        solver.set_tcp(tcp)

        return solver


class PinocchioSolver(BaseSolver):
    def __init__(self, cfg: PinocchioSolverCfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)

        self.pin = lazy_import_pinocchio()
        self.casadi = lazy_import_casadi()
        # self.cpin = lazy_import_pinocchio_casadi()

        # Set Tool Center Point (TCP)
        self.tcp = cfg.tcp

        # Set IK solver parameters
        self.pos_eps = cfg.pos_eps
        self.rot_eps = cfg.rot_eps
        self.max_iterations = cfg.max_iterations
        self.dt = cfg.dt
        self.damp = cfg.damp
        self.is_only_position_constraint = cfg.is_only_position_constraint
        self.num_samples = cfg.num_samples

        # Set mesh path if not provided
        if cfg.mesh_path is None:
            urdf_dir = os.path.dirname(cfg.urdf_path)
            cfg.mesh_path = urdf_dir

        # Load full robot model from URDF
        self.entire_robot = self.pin.RobotWrapper.BuildFromURDF(
            cfg.urdf_path, cfg.mesh_path, root_joint=None
        )

        # Get all joint names and degrees of freedom (excluding 'universe')
        self.all_joint_names = self.entire_robot.model.names.tolist()[
            1:
        ]  # Exclude 'universe' joint
        self.all_dof = (
            self.entire_robot.model.njoints - 1
        )  # Degrees of freedom of robot joints

        # Build reduced robot model (only relevant joints unlocked)
        self.robot = self._get_reduce_robot()
        self.joint_names = self.robot.model.names.tolist()[
            1:
        ]  # Exclude 'universe' joint
        self.dof = (
            self.robot.model.njoints - 1
        )  # Degrees of freedom of reduced robot joints

        self.upper_position_limits = self.robot.model.upperPositionLimit
        self.lower_position_limits = self.robot.model.lowerPositionLimit

        self.ik_nearest_weight = np.ones(self.dof)

        # TODO: The Casadi-based solver is currently disabled due to stability issues.
        # Note: Casadi-based optimization is currently prone to divergence and requires further debugging and optimization.
        if __debug__ and False:
            # Creating Casadi models and data for symbolic computing
            self.cmodel = self.cpin.Model(self.robot.model)
            self.cdata = self.cmodel.createData()
            self.cq = self.casadi.SX.sym("q", self.robot.model.nq, 1)
            self.cTf = self.casadi.SX.sym("Tf", 4, 4)
            self.cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
            self.ee_id = self.robot.model.getFrameId(self.end_link_name)

            # Define error functions for position and orientation
            self.translational_error = self.casadi.Function(
                "translational_error",
                [self.cq, self.cTf],
                [self.cdata.oMf[self.ee_id].translation - self.cTf[:3, 3]],
            )
            self.rotational_error = self.casadi.Function(
                "rotational_error",
                [self.cq, self.cTf],
                [
                    self.cpin.log3(
                        self.cdata.oMf[self.ee_id].rotation @ self.cTf[:3, :3].T
                    )
                ],
            )

            # Set up CasADi optimization problem
            self.opti = self.casadi.Opti()
            self.var_q = self.opti.variable(self.robot.model.nq)
            self.var_q_last = self.opti.parameter(self.robot.model.nq)
            self.param_tf = self.opti.parameter(4, 4)
            self.translational_cost = self.casadi.sumsqr(
                self.translational_error(self.var_q, self.param_tf)
            )
            self.rotation_cost = self.casadi.sumsqr(
                self.rotational_error(self.var_q, self.param_tf)
            )
            self.regularization_cost = self.casadi.sumsqr(self.var_q)
            self.smooth_cost = self.casadi.sumsqr(self.var_q - self.var_q_last)

            # Add joint position constraints to ensure the solution stays within physical joint limits.
            self.opti.subject_to(
                self.opti.bounded(
                    self.robot.model.lowerPositionLimit,
                    self.var_q,
                    self.robot.model.upperPositionLimit,
                )
            )

            # Define the objective function for IK optimization:
            # - Prioritize end-effector position accuracy (high weight)
            # - Include orientation accuracy
            # - Add regularization to avoid extreme joint values
            # - Encourage smoothness between consecutive solutions
            self.opti.minimize(
                100 * self.translational_cost
                + 50 * self.rotation_cost
                + 0.02 * self.regularization_cost
                + 0.1 * self.smooth_cost
            )

            # Set solver options for IPOPT
            opts = {
                "ipopt": {
                    "print_level": 0,
                    "max_iter": self.max_iterations,
                    "tol": self.pos_eps,
                },
                "print_time": False,
                "calc_lam_p": True,
            }
            self.opti.solver("ipopt", opts)

        # Initialize joint positions to zero
        self.init_qpos = np.zeros(self.robot.model.nq)

        # Perform forward kinematics with zero configuration
        self.pin.forwardKinematics(self.robot.model, self.robot.data, self.init_qpos)

        # Retrieve the pose of the specified root link
        frame_index = self.robot.model.getFrameId(self.root_link_name)
        root_base_pose = self.robot.model.frames[frame_index].placement
        self.root_base_xpos = np.eye(4)
        self.root_base_xpos[:3, :3] = root_base_pose.rotation
        self.root_base_xpos[:3, 3] = root_base_pose.translation.T

    def _get_reduce_robot(self) -> "pin.RobotWrapper":
        """Build a reduced robot model by locking all joints except those in self.joint_names.

        Returns:
            pin.RobotWrapper: The reduced robot model with specified joints unlocked.
        """
        all_joint_names = self.entire_robot.model.names.tolist()

        # Lock all joints except those in self.joint_names and 'universe'
        fixed_joint_names = [
            name
            for name in all_joint_names
            if name not in self.joint_names and name != "universe"
        ]

        reduced_robot = self.entire_robot.buildReducedRobot(
            list_of_joints_to_lock=fixed_joint_names
        )
        return reduced_robot

    def set_tcp(self, tcp: np.ndarray):
        self.tcp = tcp

    def get_iteration_params(self) -> dict:
        r"""Returns the current iteration parameters.

        Returns:
            dict: A dictionary containing the current values of:
                - pos_eps (float): Pos convergence threshold
                - rot_eps (float): Rot convergence threshold
                - max_iterations (int): Maximum number of iterations.
                - dt (float): Time step size.
                - damp (float): Damping factor.
                - num_samples (int): Number of samples.
                - is_only_position_constraint (bool): Flag to indicate whether the solver should only consider position constraints.
        """
        return {
            "pos_eps": self._pos_eps,
            "rot_eps": self._rot_eps,
            "max_iterations": self._max_iterations,
            "dt": self._dt,
            "damp": self._damp,
            "num_samples": self._num_samples,
        }

    def set_iteration_params(
        self,
        pos_eps: float = 5e-4,
        rot_eps: float = 5e-4,
        max_iterations: int = 1000,
        dt: float = 0.1,
        damp: float = 1e-6,
        num_samples: int = 30,
        is_only_position_constraint: bool = False,
    ) -> bool:
        r"""Sets the iteration parameters for the kinematics solver.

        Args:
            pos_eps (float): Pos convergence threshold, must be positive.
            rot_eps (float): Rot convergence threshold, must be positive.
            max_iterations (int): Maximum number of iterations, must be positive.
            dt (float): Time step size, must be positive.
            damp (float): Damping factor, must be non-negative.
            num_samples (int): Number of samples, must be positive.
            is_only_position_constraint (bool): Flag to indicate whether the solver should only consider position constraints.

        Returns:
            bool: True if all parameters are valid and set, False otherwise.
        """
        # TODO: Check which parameters are no longer needed.
        # Validate parameters
        if pos_eps <= 0:
            logger.log_warning("Pos epsilon must be positive.")
            return False
        if rot_eps <= 0:
            logger.log_warning("Rot epsilon must be positive.")
            return False
        if max_iterations <= 0:
            logger.log_warning("Max iterations must be positive.")
            return False
        if dt <= 0:
            logger.log_warning("Time step must be positive.")
            return False
        if damp < 0:
            logger.log_warning("Damping factor must be non-negative.")
            return False
        if num_samples <= 0:
            logger.log_warning("Number of samples must be positive.")
            return False

        # Set parameters if all are valid
        self.pos_eps = pos_eps
        self.rot_eps = rot_eps
        self.max_iterations = max_iterations
        self.dt = dt
        self.damp = damp
        self.num_samples = num_samples
        self.is_only_position_constraint = is_only_position_constraint

        if False:
            opts = {
                "ipopt": {
                    "print_level": 0,
                    "max_iter": self.max_iterations,
                    "tol": self.pos_eps,
                },
                "print_time": False,
                "calc_lam_p": False,
            }
            self.opti.solver("ipopt", opts)

        return True

    def qpos_to_limits(
        self,
        q: np.ndarray,
        joint_seed: np.ndarray,
    ):
        """Adjusts the joint positions (q) to be within specified limits and as close as possible to the joint seed,
        while minimizing the total weighted difference.

        Args:
            q (np.ndarray): The original joint positions.
            joint_seed (np.ndarray): The desired (seed) joint positions.

        Returns:
            np.ndarray: The adjusted joint positions within the specified limits.
        """
        best_qpos_limit = np.copy(q)
        best_total_q_diff = float("inf")

        # Initialize a list for possible values for each joint
        possible_arrays = []

        if self.ik_nearest_weight is None:
            self.ik_nearest_weight = np.ones_like(best_qpos_limit)

        # Generate possible values for each joint
        dof_num = len(q)
        for i in range(dof_num):
            current_possible_values = []

            # Calculate how many 2π fits into the adjustment to the limits
            lower_adjustment = (q[i] - self.lower_position_limits[i]) // (2 * np.pi)
            upper_adjustment = (self.upper_position_limits[i] - q[i]) // (2 * np.pi)

            # Consider the current value and its periodic adjustments
            for offset in range(
                int(lower_adjustment) - 1, int(upper_adjustment) + 2
            ):  # Adjust by calculated limits
                adjusted_value = q[i] + offset * (2 * np.pi)

                # Check if the adjusted value is within limits
                if (
                    self.lower_position_limits[i]
                    <= adjusted_value
                    <= self.upper_position_limits[i]
                ):
                    current_possible_values.append(adjusted_value)

            # Also check the original value
            if self.lower_position_limits[i] <= q[i] <= self.upper_position_limits[i]:
                current_possible_values.append(q[i])

            if not current_possible_values:
                return []  # If no possible values for an active joint
            possible_arrays.append(current_possible_values)

        # Generate all possible combinations
        all_possible_combinations = product(*possible_arrays)

        # Check each combination and calculate the absolute difference sum
        for combination in all_possible_combinations:
            total_q_diff = np.sum(
                np.abs(np.array(combination) - joint_seed) * self.ik_nearest_weight
            )

            # If a smaller difference sum is found, update the best solution
            if total_q_diff < best_total_q_diff:
                best_total_q_diff = total_q_diff
                best_qpos_limit = np.array(combination)

        return best_qpos_limit

    def get_ik(
        self,
        target_xpos: Optional[Union[torch.Tensor, np.ndarray]],
        qpos_seed: np.ndarray = None,
        qvel_seed: np.ndarray = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[bool, np.ndarray]:
        """Solve inverse kinematics (IK) for the robot to achieve the specified end-effector pose.

        Args:
            target_xpos (torch.Tensor or np.ndarray): Desired end-effector pose as a (4, 4) homogeneous transformation matrix.
            qpos_seed (np.ndarray, optional): Initial joint positions used as the seed for optimization. If None, uses zero configuration.
            qvel_seed (np.ndarray, optional): Initial joint velocities (not used in current implementation).
            return_all_solutions (bool, optional): If True, return all valid IK solutions found; otherwise, return only the best solution. Default is False.
            **kwargs: Additional keyword arguments for future extensions.

        Returns:
            Tuple[bool, np.ndarray]:
                - success (bool or torch.BoolTensor): True if a valid solution is found, False otherwise.
                - qpos (np.ndarray or torch.Tensor): Joint positions that achieve the target pose. If no solution, returns the seed joint positions.
        """
        if qpos_seed is not None:
            if isinstance(qpos_seed, torch.Tensor):
                self.init_qpos = qpos_seed.detach().cpu().numpy()
            else:
                self.init_qpos = np.array(qpos_seed)

        if isinstance(target_xpos, torch.Tensor):
            target_xpos = target_xpos.detach().cpu().numpy()

        if target_xpos.ndim == 3:
            target_xpos = target_xpos[0]

        target_xpos = self.root_base_xpos @ target_xpos
        compute_xpos = target_xpos @ np.linalg.inv(self.tcp_xpos)

        frame_index = self.robot.model.getFrameId(self.end_link_name)
        joint_index = self.robot.model.frames[frame_index].parent

        l2w = self.pin.SE3()
        l2w.translation[:] = compute_xpos[:3, 3]
        l2w.rotation[:] = compute_xpos[:3, :3]
        l2j = self.robot.model.frames[frame_index].placement
        oMdes = l2w * l2j.inverse()

        # Deep copy joint seed to avoid modifying the original seed
        q = deepcopy(self.init_qpos).astype(np.float64).flatten()

        for i in range(self.max_iterations):
            # Perform forward kinematics to compute the current pose
            self.pin.forwardKinematics(self.robot.model, self.robot.data, q)
            current_pose_se3 = self.robot.data.oMi[joint_index]

            if self.is_only_position_constraint:
                # Fix the rotation part of the pose
                fixed_pose = np.eye(4)
                fixed_pose[:3, :3] = compute_xpos[:3, :3]  # Use target rotation
                fixed_pose[:3, 3] = (
                    current_pose_se3.translation.T
                )  # Use current position
                fixed_pose_SE3 = self.pin.SE3(fixed_pose)
                current_pose_se3 = self.pin.SE3(fixed_pose_SE3)

            iMd = current_pose_se3.actInv(oMdes)  # Calculate the pose error
            err = self.pin.log6(iMd).vector  # Get the error vector

            # Check position convergence
            pos_converged = np.linalg.norm(err[:3]) < self.pos_eps

            if self.is_only_position_constraint:
                if pos_converged:
                    # Convergence achieved, apply joint limits
                    q = self.qpos_to_limits(q, self.init_qpos)
                    if 0 == len(q):
                        continue
                    return torch.tensor([True], dtype=torch.bool), torch.from_numpy(
                        q
                    ).to(dtype=torch.float32)
            else:
                # Check rotation convergence
                rot_converged = np.linalg.norm(err[3:]) < self.rot_eps

                # Check for overall convergence
                if pos_converged and rot_converged:
                    # Convergence achieved, apply joint limits
                    q = self.qpos_to_limits(q, self.init_qpos)
                    if 0 == len(q):
                        continue
                    return torch.tensor([True], dtype=torch.bool), torch.from_numpy(
                        q
                    ).to(dtype=torch.float32)

            # Compute the Jacobian
            J = self.pin.computeJointJacobian(
                self.robot.model, self.robot.data, q, joint_index
            )
            Jlog = self.pin.Jlog6(iMd.inverse())
            J = -Jlog @ J

            # Damped least squares
            JJt = J @ J.T
            JJt[np.diag_indices_from(JJt)] += self.damp
            # Compute the velocity update
            v = -(J.T @ np.linalg.solve(JJt, err))

            # Update joint positions
            new_q = self.pin.integrate(self.robot.model, q, v * self.dt)
            q = new_q

        # Return failure and the last computed joint positions
        return torch.tensor([False], dtype=torch.bool), torch.from_numpy(
            np.array(q)
        ).to(dtype=torch.float32)

        # TODO: The Casadi-based solver is currently disabled due to stability issues.
        # Note: Casadi-based optimization is currently prone to divergence and requires further debugging and optimization.
        if __debug__ and False:
            self.opti.set_initial(self.var_q, self.init_qpos)

            self.opti.set_value(self.param_tf, compute_xpos)

            try:
                num_iter = 1 if self.max_iterations == 1 else self.max_iterations

                for i in range(num_iter):
                    self.opti.set_value(self.var_q_last, self.init_qpos)
                    sol = self.opti.solve()
                    sol_q = self.opti.value(self.var_q)
                    # self.smooth_filter.add_data(sol_q)
                    # sol_q = self.smooth_filter.filtered_data
                    self.init_qpos = sol_q

                    # if qvel_seed is not None:
                    #     v = qvel_seed * 0.0
                    # else:
                    #     v = (sol_q - self.init_qpos) * 0.0
                    # sol_tauff = pin.rnea(
                    #     self.robot.model,
                    #     self.robot.data,
                    #     sol_q,
                    #     v,
                    #     np.zeros(self.robot.model.nv),
                    # )

                    temp_xpos = self._get_fk(sol_q)
                    err = temp_xpos - target_xpos
                    pos_converged = np.linalg.norm(err[:3]) < self.pos_eps
                    print(f"Iter {i}: pos_err={np.linalg.norm(err[:3])}")

                    if self.is_only_position_constraint:
                        if pos_converged:
                            break
                    else:
                        rot_converged = np.linalg.norm(err[:3, :3]) < self.rot_eps
                        if pos_converged and rot_converged:
                            break

                if return_all_solutions:
                    logger.log_warning(
                        "return_all_solutions=True is not supported in DifferentialSolver. Returning the best solution only."
                    )

                return torch.tensor(True, dtype=torch.bool), torch.from_numpy(sol_q).to(
                    dtype=torch.float32
                )

            except Exception as e:
                logger.log_warning(f"IK solver failed to converge. Debug info: {e}")

                sol_q = self.opti.debug.value(self.var_q)
                # self.smooth_filter.add_data(sol_q)
                # sol_q = self.smooth_filter.filtered_data
                self.init_qpos = sol_q

                # if qvel_seed is not None:
                #     v = qvel_seed * 0.0
                # else:
                #     v = (sol_q - self.init_qpos) * 0.0

                # sol_tauff = pin.rnea(
                #     self.robot.model,
                #     self.robot.data,
                #     sol_q,
                #     v,
                #     np.zeros(self.robot.model.nv),
                # )

                logger.log_debug(
                    f"sol_q:{sol_q} \nmotorstate: \n{qpos_seed} \nwrist_pose: \n{target_xpos}"
                )

                if return_all_solutions:
                    logger.log_warning(
                        "return_all_solutions=True is not supported in DifferentialSolver. Returning the best solution only."
                    )

                return torch.tensor(False, dtype=torch.bool), torch.from_numpy(
                    np.array(qpos_seed)
                ).to(dtype=torch.float32)

    def _get_fk(
        self,
        qpos: Optional[Union[torch.Tensor, np.ndarray]],
        **kwargs,
    ) -> np.ndarray:
        """Compute the forward kinematics for the robot given joint positions.

        Args:
            qpos (torch.Tensor or np.ndarray): Joint positions, shape should be (nq,).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            np.ndarray: The resulting end-effector pose as a (4, 4) homogeneous transformation matrix.
        """
        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.array(qpos)

        qpos_np = np.squeeze(qpos_np)
        if qpos_np.ndim != 1:
            raise ValueError(f"qpos shape must be (nq,), but got {qpos_np.shape}")

        self.pin.forwardKinematics(self.robot.model, self.robot.data, qpos_np)

        # Retrieve the pose of the specified link
        frame_index = self.robot.model.getFrameId(self.end_link_name)
        joint_index = self.robot.model.frames[frame_index].parent
        xpos_se3 = self.robot.data.oMi.tolist()[joint_index]

        xpos = np.eye(4)
        xpos[:3, :3] = xpos_se3.rotation
        xpos[:3, 3] = xpos_se3.translation.T

        result = np.dot(xpos, self.tcp_xpos)
        return result
