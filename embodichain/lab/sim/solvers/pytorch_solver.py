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

import torch

from typing import Optional, Union, Tuple, List, TYPE_CHECKING
from dataclasses import MISSING
from copy import deepcopy

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver
from embodichain.lab.sim.solvers.qpos_seed_sampler import QposSeedSampler
from embodichain.lab.sim.utility.solver_utils import validate_iteration_params

if TYPE_CHECKING:
    from typing import Self

from embodichain.lab.sim.utility.import_utils import (
    lazy_import_pytorch_kinematics,
)


@configclass
class PytorchSolverCfg(SolverCfg):
    """Configuration for the pytorch kinematics solver used in the robot simulation.

    This configuration includes properties related to the solver setup, such as the URDF path,
    the end link name, and the root link name, along with the Tool Center Point (TCP).
    """

    class_type: str = "PytorchSolver"

    # Solver iteration parameters
    pos_eps: float = 5e-4
    """Tolerance for convergence for position"""

    rot_eps: float = 5e-4
    """Tolerance for convergence for rotation"""

    max_iterations: int = 500
    """Maximum number of iterations for the solver"""

    dt: float = 0.1
    """Time step for numerical integration"""

    damp: float = 1e-6
    """Damping factor to prevent numerical instability"""

    is_only_position_constraint: bool = False
    """Flag to indicate whether the solver should only consider position constraints."""

    num_samples: int = 5
    """Number of samples to generate different joint seeds for IK iterations.

    A higher number of samples increases the chances of finding a valid solution
    """

    ik_nearest_weight: Optional[List[float]] = None
    """Weights for the inverse kinematics nearest calculation.
    
    The weights influence how the solver prioritizes closeness to the seed position
    when multiple solutions are available.
    """

    def init_solver(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> "PytorchSolver":
        """Initialize the solver with the configuration.

        Args:
            device (torch.device): The device to use for the solver. Defaults to CPU.
            **kwargs: Additional keyword arguments that may be used for solver initialization.

        Returns:
            PytorchSolver: An initialized solver instance.
        """

        solver = PytorchSolver(cfg=self, device=device, **kwargs)

        # Set the Tool Center Point (TCP) for the solver
        solver.set_tcp(self._get_tcp_as_numpy())

        return solver


def ensure_pose_shape(func):
    """
    Decorator to ensure the input target_pose is of shape (n, 4, 4).
    If input is (4, 4), it will be converted to (1, 4, 4).
    Raises ValueError if shape is invalid.
    """

    def wrapper(self, target_xpos, *args, **kwargs):
        target_xpos = torch.as_tensor(
            target_xpos, device=self.device, dtype=torch.float32
        )
        if target_xpos.dim() == 2:
            if target_xpos.shape != (4, 4):
                raise ValueError("target_xpos must be of shape (4, 4) or (n, 4, 4).")
            target_xpos = target_xpos.unsqueeze(0)
        elif target_xpos.dim() == 3:
            if target_xpos.shape[1:] != (4, 4):
                raise ValueError("target_xpos must be of shape (4, 4) or (n, 4, 4).")
        else:
            raise ValueError(
                "target_xpos must be a tensor of shape (4, 4) or (n, 4, 4)."
            )
        return func(self, target_xpos, *args, **kwargs)

    return wrapper


class PytorchSolver(BaseSolver):
    def __init__(
        self,
        cfg: PytorchSolverCfg,
        device: str = None,
        **kwargs,
    ):
        r"""Initializes the PyTorch kinematics solver.

            This constructor sets up the kinematics solver using PyTorch,
            allowing for efficient computation of robot kinematics based on
            the specified URDF model.

        Args:
            cfg: The configuration for the solver.
            device (str, optional): The device to use for the solver (e.g., "cpu" or "cuda").
            **kwargs: Additional keyword arguments passed to the base solver.

        """
        super().__init__(cfg=cfg, device=device, **kwargs)

        self.pk = lazy_import_pytorch_kinematics()

        # Initialize solver parameters from configuration
        self._pos_eps = cfg.pos_eps
        self._rot_eps = cfg.rot_eps
        self._max_iterations = cfg.max_iterations
        self._dt = cfg.dt
        self._damp = cfg.damp
        self._is_only_position_constraint = cfg.is_only_position_constraint
        self._num_samples = cfg.num_samples

        # Get agent joint limits.
        self.lim = torch.tensor(
            self.pk_serial_chain.get_joint_limits(), device=self.device
        )

        # Inverse kinematics is available via damped least squares (iterative steps with Jacobian pseudo-inverse damped to avoid oscillation near singularlities).
        self.pik = self.pk.PseudoInverseIK(
            self.pk_serial_chain,
            pos_tolerance=self._pos_eps,
            rot_tolerance=self._rot_eps,
            joint_limits=self.lim.T,
            early_stopping_any_converged=True,
            max_iterations=self._max_iterations,
            lr=self._dt,
            num_retries=1,
        )

        self.dof = self.pk_serial_chain.n_joints

        self.upper_position_limits = self.pk_serial_chain.high
        self.lower_position_limits = self.pk_serial_chain.low

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
        # Validate parameters
        if not validate_iteration_params(
            pos_eps, rot_eps, max_iterations, dt, damp, num_samples
        ):
            return False

        # Set parameters if all are valid
        self._pos_eps = pos_eps
        self._rot_eps = rot_eps
        self._max_iterations = max_iterations
        self._dt = dt
        self._damp = damp
        self._num_samples = num_samples
        self._is_only_position_constraint = is_only_position_constraint

        self.pik = self.pk.PseudoInverseIK(
            self.pk_serial_chain,
            pos_tolerance=self._pos_eps,
            rot_tolerance=self._rot_eps,
            joint_limits=self.lim.T,
            early_stopping_any_converged=True,
            max_iterations=self._max_iterations,
            lr=self._dt,
            num_retries=1,
        )

        return True

    def _compute_inverse_kinematics(
        self, target_pose: torch.Tensor, joint_seed: torch.Tensor
    ) -> Tuple[Union[bool, torch.Tensor], torch.Tensor]:
        r"""Computes the inverse kinematics solutions for the given target poses and joint seeds.

        Args:
            target_pose (torch.Tensor): The target poses represented as a (batch_size, 4, 4) tensor.
            joint_seed (torch.Tensor): The initial joint positions used as a seed. It can be either a 1D tensor of shape (dof,) or a 2D tensor of shape (batch_size, dof).

        Returns:
            Tuple[Union[bool, torch.Tensor], torch.Tensor]:
                - First element:
                    - If solutions exist: torch.BoolTensor of shape (batch_size,) indicating convergence per pose
                    - If no solutions: Python False
                - Second element:
                    - If solutions exist: torch.Tensor of shape (batch_size, dof) containing joint solutions
                    - If no solutions: Empty torch.Tensor
        """
        target_pose = target_pose.to(self.device).float()
        joint_seed = joint_seed.to(self.device).float()

        # Extract translation and rotation parts
        pos = target_pose[:, :3, 3]
        rot = target_pose[:, :3, :3]

        tf = self.pk.Transform3d(
            pos=pos,
            rot=rot,
            device=self.device,
        )
        self.pik.initial_config = joint_seed

        result = self.pik.solve(tf)

        if result.converged_any.any().item():
            return result.converged_any, result.solutions[:, 0, :].squeeze(0)

        return False, torch.empty(0)

    @staticmethod
    def _qpos_to_limits_single(
        q: torch.Tensor,
        joint_seed: torch.Tensor,
        lower_position_limits: torch.Tensor,
        upper_position_limits: torch.Tensor,
        ik_nearest_weight: torch.Tensor,
        periodic_mask: torch.Tensor = None,  # Optional mask for periodic joints
    ) -> torch.Tensor:
        """
        Adjusts the given joint positions (q) to fit within the specified limits while minimizing the difference to the seed position.

        Args:
            q (torch.Tensor): The initial joint positions.
            joint_seed (torch.Tensor): The seed joint positions for comparison.
            lower_position_limits (torch.Tensor): The lower bounds for the joint positions.
            upper_position_limits (torch.Tensor): The upper bounds for the joint positions.
            ik_nearest_weight (torch.Tensor): The weights for the inverse kinematics nearest calculation.
            periodic_mask (torch.Tensor, optional): Boolean mask indicating which joints are periodic.

        Returns:
            torch.Tensor: The adjusted joint positions that fit within the limits.
        """
        device = q.device
        joint_seed = joint_seed.to(device)
        lower = lower_position_limits.to(device)
        upper = upper_position_limits.to(device)
        weight = ik_nearest_weight.to(device)

        # If periodic_mask is not provided, assume all joints are periodic
        if periodic_mask is None:
            periodic_mask = torch.ones_like(q, dtype=torch.bool, device=device)

        # Only enumerate [-2π, 0, 2π] for periodic joints, single value for non-periodic
        offsets = torch.tensor([-2 * torch.pi, 0, 2 * torch.pi], device=device)
        candidate_list = []
        for i in range(q.size(0)):
            if periodic_mask[i]:
                candidate_list.append(q[i] + offsets)
            else:
                candidate_list.append(q[i].unsqueeze(0))
        # Generate all possible combinations
        mesh = torch.meshgrid(*candidate_list, indexing="ij")
        candidates = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        # Filter candidates that are out of limits
        mask = (candidates >= lower) & (candidates <= upper)
        valid_mask = mask.all(dim=1)
        valid_candidates = candidates[valid_mask]
        if valid_candidates.shape[0] == 0:
            return torch.tensor([]).to(device)
        # Compute weighted distance to seed and select the closest
        diffs = torch.abs(valid_candidates - joint_seed) * weight
        distances = torch.sum(diffs, dim=1)
        min_idx = torch.argmin(distances)
        return valid_candidates[min_idx]

    def _qpos_to_limits(
        self, qpos_list_split: torch.Tensor, joint_seed: torch.Tensor
    ) -> torch.Tensor:
        r"""Adjusts a batch of joint positions to fit within joint limits and minimize the weighted distance to the seed position.

        Args:
            qpos_list_split (torch.Tensor): Batch of candidate joint positions, shape (N, dof).
            joint_seed (torch.Tensor): The reference joint positions for comparison, shape (dof,).

        Returns:
            torch.Tensor: Batch of adjusted joint positions that fit within the limits, shape (M, dof),
                        where M <= N (invalid candidates are filtered out).
        """

        periodic_mask = torch.ones_like(
            qpos_list_split[0], dtype=torch.bool, device=self.device
        )

        adjusted_qpos_list = [
            self._qpos_to_limits_single(
                q,
                joint_seed,
                self.lower_position_limits,
                self.upper_position_limits,
                self.ik_nearest_weight,
                periodic_mask,
            )
            for q in qpos_list_split
        ]

        # Filter out empty results
        adjusted_qpos_list = [q for q in adjusted_qpos_list if q.numel() > 0]

        return (
            torch.stack(adjusted_qpos_list).to(qpos_list_split.device)
            if adjusted_qpos_list
            else torch.tensor([], device=self.device)
        )

    @ensure_pose_shape
    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor = None,
        num_samples: int = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the inverse kinematics for given target poses.

        This function generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint positions that are closest to the joint_seed.

        Args:
            target_xpos (torch.Tensor): A tensor representing the target positions. It can be of shape
                                         (batch_size, 3) for multiple positions or (3,) for a single position.
            qpos_seed (torch.Tensor, optional): Initial joint positions used as seed for IK solving.
                                                Can be:
                                                - 1D tensor of shape (dof,): Single seed for all target positions
                                                - 2D tensor of shape (batch_size, dof): Individual seed per position
                                                If None, defaults to zero configuration. Defaults to None.
            num_samples (int, optional): The number of random samples to generate. Must be positive.
                                         Defaults to None.
            return_all_solutions (bool, optional): If True, returns all valid solutions found.
            **kwargs: Additional arguments for future extensions.

        Returns:
            Tuple[List[bool], torch.Tensor]: A tuple containing:
                - A tensor of booleans indicating whether valid solutions were found for each target pose. (Shape: (batch_size,))
                - A tensor of shape (batch_size, 1, dof) containing joint positions for
                  each target pose, or an empty tensor if no valid solutions were found.
        """
        # Convert target_pose to tensor and ensure correct device and dtype
        target_xpos = torch.as_tensor(
            target_xpos, device=self.device, dtype=torch.float32
        )
        if num_samples is not None:
            self._num_samples = num_samples

        # Prepare qpos_seed
        if qpos_seed is None:
            qpos_seed = torch.zeros(self.dof, device=self.device)
        else:
            qpos_seed = torch.as_tensor(qpos_seed, device=self.device)

        # Check qpos_seed dimensions
        if qpos_seed.dim() == 1:
            qpos_seed = qpos_seed.unsqueeze(0)
            qpos_seed_ndim = 1
        elif qpos_seed.dim() == 2:
            qpos_seed_ndim = 2
            if qpos_seed.shape[0] != target_xpos.shape[0]:
                raise ValueError(
                    "Batch size of qpos_seed must match batch size of target_xpos when qpos_seed is a 2D tensor."
                )
        else:
            raise ValueError("`qpos_seed` must be a tensor of shape (n,) or (n, n).")

        # Transform target_xpos by TCP
        tcp_xpos = torch.as_tensor(
            deepcopy(self.tcp_xpos), device=self.device, dtype=torch.float32
        )
        target_xpos = target_xpos @ torch.inverse(tcp_xpos)

        # Get joint limits and ensure shape matches dof
        upper_limits = self.upper_position_limits.float()
        lower_limits = self.lower_position_limits.float()

        batch_size = target_xpos.shape[0]

        sampler = QposSeedSampler(
            num_samples=self._num_samples, dof=self.dof, device=self.device
        )
        random_qpos_seeds = sampler.sample(
            qpos_seed, lower_limits, upper_limits, batch_size
        )
        target_xpos_repeated = sampler.repeat_target_xpos(
            target_xpos, self._num_samples
        )

        # Compute IK solutions for all samples
        res_list, qpos_list = self._compute_inverse_kinematics(
            target_xpos_repeated, random_qpos_seeds
        )

        if not isinstance(res_list, torch.Tensor) or not res_list.any():
            logger.log_warning(
                "Pk: No valid solutions found for the given target poses and joint seeds."
            )
            return torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            ), torch.zeros((batch_size, self.dof), device=self.device)

        # Split res_list and qpos_list according to self._num_samples
        res_list_split = torch.split(res_list, self._num_samples)
        qpos_list_split = torch.split(qpos_list, self._num_samples)

        # Initialize the final results and the closest joint positions
        final_results = []
        final_qpos = []

        # For each batch, select the closest valid solution to qpos_seed
        for i in range(batch_size):
            target_qpos_seed = qpos_seed[i] if qpos_seed_ndim == 2 else qpos_seed

            if not res_list_split[i].any():
                final_results.append(False)
                final_qpos.append(torch.zeros((1, self.dof), device=self.device))
                continue

            result_qpos_limit = self._qpos_to_limits(
                qpos_list_split[i], target_qpos_seed
            )

            if result_qpos_limit.shape[0] == 0:
                final_results.append(False)
                final_qpos.append(torch.zeros((self.dof), device=self.device))
                continue

            distances = torch.norm(result_qpos_limit - target_qpos_seed, dim=1)
            sorted_indices = torch.argsort(distances)
            # shape: (N, dof)
            sorted_qpos_array = result_qpos_limit[sorted_indices]
            final_qpos.append(sorted_qpos_array)
            final_results.append(True)

        # Pad all batches to the same number of solutions for stacking
        max_solutions = max([q.shape[0] for q in final_qpos]) if final_qpos else 1
        final_qpos_tensor = torch.zeros(
            (batch_size, max_solutions, self.dof), device=self.device
        )
        for i, q in enumerate(final_qpos):
            n = q.shape[0]
            final_qpos_tensor[i, :n, :] = q

        final_results = torch.tensor(
            final_results, dtype=torch.bool, device=self.device
        )

        if return_all_solutions:
            # Return all sorted solutions for each batch (shape: batch_size, max_solutions, dof)
            return final_results, final_qpos_tensor

        # Only return the closest solution for each batch (shape: batch_size, 1, dof)
        # If multiple solutions, take the first (closest)
        final_qpos_tensor = final_qpos_tensor[:, :1, :]
        return final_results, final_qpos_tensor

    def get_all_fk(self, qpos: torch.tensor) -> torch.tensor:
        r"""Get the forward kinematics for all links from root to end link.

        Args:
            qpos (torch.Tensor): The joint positions.

        Returns:
            list: A list of 4x4 homogeneous transformation matrices representing the poses of all links from root to end link.
        """
        qpos = torch.as_tensor(qpos)
        qpos = qpos.to(self.device)

        ret = self.pk_serial_chain.forward_kinematics(qpos, end_only=False)
        link_names = list(ret.keys())

        if self.root_link_name is not None:
            try:
                start_index = link_names.index(self.root_link_name)
            except ValueError:
                raise KeyError(
                    f"Root link name '{self.root_link_name}' not found in the kinematic chain"
                )
        else:
            start_index = 0

        if self.end_link_name is not None:
            try:
                end_index = link_names.index(self.end_link_name) + 1
            except ValueError:
                raise KeyError(
                    f"End link name '{self.end_link_name}' not found in the kinematic chain"
                )
        else:
            end_index = len(link_names)

        poses = []
        for link_name in link_names[start_index:end_index]:
            xpos = ret[link_name]
            if not hasattr(xpos, "get_matrix"):
                raise AttributeError(
                    f"The result for link '{link_name}' must have 'get_matrix' attributes."
                )
            xpos_t = torch.eye(4, device=xpos.get_matrix().device)
            m = xpos.get_matrix()
            xpos_t[:3, 3] = m[:, :3, 3]
            xpos_t[:3, :3] = m[:, :3, :3]
            poses.append(xpos_t)

        return poses
