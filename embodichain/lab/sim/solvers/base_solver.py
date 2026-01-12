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
import numpy as np
from typing import List, Dict, Any, Union, TYPE_CHECKING, Tuple
from abc import abstractmethod, ABCMeta

from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from typing import Self

from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain


@configclass
class SolverCfg:
    """Configuration for the kinematic solver used in the robot simulation."""

    class_type: str = "BaseSolver"
    """The class type of the solver to be used."""

    urdf_path: str | None = None
    """The file path to the URDF model of the robot."""

    joint_names: list[str] | None = None
    """List of joint names for the solver.
    
    If None, all joints in the URDF will be used.
    If specified, only these named joints will be included in the kinematic chain.
    """

    end_link_name: str = None
    """The name of the end-effector link for the solver.

    This defines the target link for forward/inverse kinematics calculations.
    Must match a link name in the URDF file.
    """

    root_link_name: str = None
    """The name of the root/base link for the solver.

    This defines the starting point of the kinematic chain.
    Must match a link name in the URDF file.
    """

    # TODO: may be support pos and rot separately for easier manipulation.
    tcp: torch.Tensor | np.ndarray = np.eye(4)
    """The tool center point (TCP) position as a 4x4 homogeneous matrix.

    This represents the position and orientation of the tool in the robot's end-effector frame.
    """

    ik_nearest_weight: List[float] | None = None
    """Weights for the inverse kinematics nearest calculation.
    
    The weights influence how the solver prioritizes closeness to the seed position
    when multiple solutions are available.
    """

    @abstractmethod
    def init_solver(self, device: torch.device, **kwargs) -> "BaseSolver":
        pass

    def _get_tcp_as_numpy(self) -> np.ndarray:
        """Convert TCP to numpy array.

        This helper method handles the conversion of TCP from torch.Tensor to numpy
        if needed. Used by subclass init_solver methods to set TCP on the solver.

        Returns:
            np.ndarray: The TCP as a numpy array.
        """
        if isinstance(self.tcp, torch.Tensor):
            return self.tcp.cpu().numpy()
        return self.tcp

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> "SolverCfg":
        """Initialize the configuration from a dictionary."""
        from embodichain.utils.utility import get_class_instance

        if "class_type" not in init_dict:
            logger.log_error("class type must be specified in the configuration.")

        cfg = get_class_instance(
            "embodichain.lab.sim.solvers", init_dict["class_type"] + "Cfg"
        )()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


class BaseSolver(metaclass=ABCMeta):
    def __init__(self, cfg: SolverCfg = None, device: str = None, **kwargs):
        r"""Initializes the kinematics solver with a robot model.

        Args:
            cfg (SolverCfg): The configuration for the solver.
            device (str or torch.device, optional): The device to run the solver on. Defaults to "cuda" if available, otherwise "cpu".
            **kwargs: Additional keyword arguments for customization.
        """
        self.cfg = cfg

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.urdf_path = cfg.urdf_path

        self.joint_names = cfg.joint_names

        self.end_link_name = cfg.end_link_name

        self.root_link_name = cfg.root_link_name

        # TODO: Check whether the joint name is revolute or prismatic
        # Degrees of freedom of robot joints
        self.dof = len(self.joint_names) if self.joint_names else 0

        # Weight for nearest neighbor search in IK (Inverse Kinematics) algorithms
        if cfg.ik_nearest_weight is not None:
            if len(cfg.ik_nearest_weight) != self.dof:
                logger.log_error(
                    f"Length of ik_nearest_weight ({len(cfg.ik_nearest_weight)}) does not match the number of DOF ({self.dof})."
                )
            self.ik_nearest_weight = torch.tensor(
                cfg.ik_nearest_weight, dtype=torch.float32, device=self.device
            )
        else:
            self.ik_nearest_weight = torch.ones(
                self.dof, dtype=torch.float32, device=self.device
            )

        self.tcp_xpos = np.eye(4)

        self.pk_serial_chain = kwargs.get("pk_serial_chain", None)
        if self.pk_serial_chain is None:
            self.pk_serial_chain = create_pk_serial_chain(
                urdf_path=self.urdf_path,
                end_link_name=self.end_link_name,
                root_link_name=self.root_link_name,
                device=self.device,
            )

    def set_ik_nearest_weight(
        self, ik_weight: np.ndarray, joint_ids: np.ndarray | None = None
    ) -> bool:
        r"""Sets the inverse kinematics nearest weight.

        Args:
            ik_weight (np.ndarray): A numpy array representing the nearest weights for inverse kinematics.
            joint_ids (np.ndarray, optional): A numpy array representing the indices of the joints to which the weights apply.
                                            If None, defaults to all joint indices.

        Returns:
            bool: True if the weights are set successfully, False otherwise.
        """
        ik_weight = np.array(ik_weight)

        # Set joint_ids to all joint indices if it is None
        if joint_ids is None:
            joint_ids = np.arange(self.dof)

        joint_ids = np.array(joint_ids)

        # Check if joint_ids has valid indices
        if np.any(joint_ids >= self.dof) or np.any(joint_ids < 0):
            logger.log_warning(
                "joint_ids must contain valid indices between 0 and {}.".format(
                    self.dof - 1
                )
            )
            return False

        # Check if ik_weight and joint_ids have the same length
        if ik_weight.shape[0] != joint_ids.shape[0]:
            logger.log_warning("ik_weight and joint_ids must have the same length.")
            return False

        # Initialize the weights
        if self.ik_nearest_weight is None:
            # If ik_nearest_weight is None, set all weights to 1
            self.ik_nearest_weight = np.ones(self.dof)

            # Set specific weights for joint_ids to the provided ik_weight
            for i, joint_id in enumerate(joint_ids):
                self.ik_nearest_weight[joint_id] = ik_weight[i]
        else:
            # If ik_nearest_weight is not None, only fill joint_ids
            for i, joint_id in enumerate(joint_ids):
                self.ik_nearest_weight[joint_id] = ik_weight[i]

        return True

    def get_ik_nearest_weight(self):
        r"""Gets the inverse kinematics nearest weight.

        Returns:
            np.ndarray: A numpy array representing the nearest weights for inverse kinematics.
        """
        return self.ik_nearest_weight

    def set_position_limits(
        self,
        lower_position_limits: List[float],
        upper_position_limits: List[float],
    ) -> bool:
        r"""Sets the upper and lower joint position limits.

        Parameters:
            lower_position_limits (List[float]): A list of lower limits for each joint.
            upper_position_limits (List[float]): A list of upper limits for each joint.

        Returns:
            bool: True if limits are successfully set, False if the input is invalid.
        """
        if (
            len(lower_position_limits) != self.model.nq
            or len(upper_position_limits) != self.model.nq
        ):
            logger.log_warning("Length of limits must match the number of joints.")
            return False

        if any(
            lower > upper
            for lower, upper in zip(lower_position_limits, upper_position_limits)
        ):
            logger.log_warning(
                "Each lower limit must be less than or equal to the corresponding upper limit."
            )
            return False

        self.lower_position_limits = np.array(lower_position_limits)
        self.upper_position_limits = np.array(upper_position_limits)
        return True

    def get_position_limits(self) -> dict:
        r"""Returns the current joint position limits.

        Returns:
            dict: A dictionary containing:
                - lower_position_limits (List[float]): The current lower limits for each joint.
                - upper_position_limits (List[float]): The current upper limits for each joint.
        """
        return {
            "lower_position_limits": self.lower_position_limits.tolist(),
            "upper_position_limits": self.upper_position_limits.tolist(),
        }

    def set_tcp(self, xpos: np.ndarray):
        r"""Sets the TCP position with the given 4x4 homogeneous matrix.

        Args:
            xpos (np.ndarray): The 4x4 homogeneous matrix to be set as the TCP position.

        Raises:
            ValueError: If the input is not a 4x4 numpy array.
        """
        xpos = np.array(xpos)
        if xpos.shape != (4, 4):
            raise ValueError("Input must be a 4x4 homogeneous matrix")
        self.tcp_xpos = xpos

    def get_tcp(self) -> np.ndarray:
        r"""Returns the current TCP position.

        Returns:
            np.ndarray: The current TCP position.

        Raises:
            ValueError: If the TCP position has not been set.
        """
        return self.tcp_xpos

    @abstractmethod
    def get_ik(
        self,
        target_pose: torch.Tensor,
        joint_seed: torch.Tensor | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the inverse kinematics for a given target pose.

        This method generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint position that is closest to the joint_seed.

        Args:
            target_pose (torch.Tensor): The target pose represented as a 4x4 transformation matrix.
            joint_seed (torch.Tensor | None): The initial joint positions used as a seed.
            num_samples (int | None): The number of random joint seeds to generate.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - success (torch.Tensor): Boolean tensor indicating IK solution validity for each environment, shape (num_envs,).
                - target_joints (torch.Tensor): Computed target joint positions, shape (num_envs, num_joints).
        """
        pass

    def get_fk(self, qpos: torch.tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes the forward kinematics for the end-effector link.

        Args:
            qpos (torch.Tensor): Joint positions. Can be a single configuration (dof,) or a batch (batch_size, dof).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            torch.Tensor: The homogeneous transformation matrix of the end link with TCP applied.
                        Shape is (4, 4) for single input, or (batch_size, 4, 4) for batch input.
        """
        tcp_xpos = torch.as_tensor(
            self.tcp_xpos, device=self.device, dtype=torch.float32
        )
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)

        # Compute forward kinematics
        result = self.pk_serial_chain.forward_kinematics(
            qpos, end_only=(self.end_link_name is None)
        )

        # Extract transformation matrices
        if isinstance(result, dict):
            matrices = result[self.end_link_name].get_matrix()
        elif isinstance(result, list):
            matrices = torch.stack([xpos.get_matrix().squeeze() for xpos in result])
        else:
            matrices = result.get_matrix()

        # Ensure batch format
        if matrices.dim() == 2:
            matrices = matrices.unsqueeze(0)

        # Create result tensor with proper homogeneous coordinates
        result = (
            torch.eye(4, device=self.device).expand(matrices.shape[0], 4, 4).clone()
        )
        result[:, :3, :] = matrices[:, :3, :]

        # Ensure batch format for TCP
        batch_size = result.shape[0]
        tcp_xpos_batch = tcp_xpos.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply TCP transformation
        return torch.bmm(result, tcp_xpos_batch)

    def get_jacobian(
        self,
        qpos: torch.Tensor,
        locations: torch.Tensor | np.ndarray | None = None,
        jac_type: str = "full",
    ) -> torch.Tensor:
        r"""Compute the Jacobian matrix for the given joint positions.

        Args:
            qpos (torch.Tensor): The joint positions. Shape: (dof,) or (batch_size, dof).
            locations (torch.Tensor | np.ndarray | None): The offset points (relative to the end-effector coordinate system). Shape: (batch_size, 3) or (3,) for a single offset.
            jac_type (str): 'full', 'trans', or 'rot' for full, translational, or rotational Jacobian. Defaults to 'full'.

        Returns:
            torch.Tensor: The Jacobian matrix. Shape:
                        - (batch_size, 6, dof) for 'full'
                        - (batch_size, 3, dof) for 'trans' or 'rot'
        """
        if qpos is None:
            qpos = torch.zeros(self.dof, device=self.device)

        # Ensure qpos is a tensor
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)

        # Ensure locations is a tensor if provided
        if locations is not None:
            locations = torch.as_tensor(
                locations, dtype=torch.float32, device=self.device
            )

        # Compute the Jacobian using the kinematics chain
        J = self.pk_serial_chain.jacobian(th=qpos, locations=locations)

        # Handle jac_type to return the desired part of the Jacobian
        if jac_type == "trans":
            return J[:, :3, :] if J.dim() == 3 else J[:3, :]
        elif jac_type == "rot":
            return J[:, 3:, :] if J.dim() == 3 else J[3:, :]
        elif jac_type == "full":
            return J
        else:
            raise ValueError(
                f"Invalid jac_type '{jac_type}'. Must be 'full', 'trans', or 'rot'."
            )
