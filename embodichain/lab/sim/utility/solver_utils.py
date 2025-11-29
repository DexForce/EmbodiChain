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
from embodichain.lab.sim.utility.io_utils import suppress_stdout_stderr

from typing import Optional, Union, Tuple, Any, List, TYPE_CHECKING
from copy import deepcopy

from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from typing import Self

from embodichain.lab.sim.utility.import_utils import (
    lazy_import_pytorch_kinematics,
)


def create_pk_chain(
    urdf_path: str,
    device: torch.device,
    **kwargs,
) -> "pk.SerialChain":
    """
    Factory method to create a pk.SerialChain object from a URDF file.

    Args:
        urdf_path (str): Path to the URDF file.
        end_link_name (str): Name of the end-effector link.
        root_link_name (Optional[str]): Name of the root link. If None, the chain starts from the base.
        device (torch.device): The device to which the chain will be moved.
        is_serial (bool): Whether the chain is serial or not.

    Returns:
        pk.SerialChain: The created serial chain object.
    """
    pk = lazy_import_pytorch_kinematics()
    with open(urdf_path, "rb") as f:
        urdf_str = f.read()

    with suppress_stdout_stderr():
        return pk.build_chain_from_urdf(urdf_str).to(device=device)


def create_pk_serial_chain(
    urdf_path: str = None,
    device: torch.device = None,
    end_link_name: str = None,
    root_link_name: Optional[Union[str, None]] = None,
    chain: Optional["pk.SerialChain"] = None,
    **kwargs,
) -> "pk.SerialChain":
    """
    Factory method to create a pk.SerialChain object from a URDF file.

    Args:
        urdf_path (str): Path to the URDF file.
        end_link_name (str): Name of the end-effector link.
        root_link_name (Optional[str]): Name of the root link. If None, the chain starts from the base.
        device (torch.device): The device to which the chain will be moved.
        is_serial (bool): Whether the chain is serial or not.

    Returns:
        pk.SerialChain: The created serial chain object.
    """
    if urdf_path is None and chain is None:
        raise ValueError("Either `urdf_path` or `chain` must be provided.")
    if urdf_path and chain:
        raise ValueError("`urdf_path` and `chain` cannot be provided at the same time.")

    pk = lazy_import_pytorch_kinematics()

    if chain is None:
        try:
            with open(urdf_path, "rb") as f:
                urdf_str = f.read()
        except FileNotFoundError:
            raise ValueError(f"URDF file not found at path: {urdf_path}")
        except IOError as e:
            raise ValueError(f"Failed to read URDF file: {e}")

        with suppress_stdout_stderr():
            if root_link_name is None:
                return pk.build_serial_chain_from_urdf(
                    urdf_str,
                    end_link_name=end_link_name,
                ).to(device=device)
            else:
                return pk.build_serial_chain_from_urdf(
                    urdf_str,
                    end_link_name=end_link_name,
                    root_link_name=root_link_name,
                ).to(device=device)
    else:
        return pk.SerialChain(
            chain=chain, end_frame_name=end_link_name, root_frame_name=root_link_name
        )


def build_reduced_pinocchio_robot(
    entire_robot: "pin.RobotWrapper",
    joint_names: List[str],
) -> "pin.RobotWrapper":
    """Build a reduced robot model by locking all joints except those specified.

    This utility function creates a reduced Pinocchio robot model by locking all joints
    except those in the provided joint_names list and the 'universe' joint.

    Args:
        entire_robot: The full Pinocchio RobotWrapper model.
        joint_names: List of joint names to keep unlocked in the reduced model.

    Returns:
        pin.RobotWrapper: The reduced robot model with specified joints unlocked.
    """
    all_joint_names = entire_robot.model.names.tolist()

    # Lock all joints except those in joint_names and 'universe'
    fixed_joint_names = [
        name
        for name in all_joint_names
        if name not in joint_names and name != "universe"
    ]

    reduced_robot = entire_robot.buildReducedRobot(
        list_of_joints_to_lock=fixed_joint_names
    )
    return reduced_robot


def validate_iteration_params(
    pos_eps: float,
    rot_eps: float,
    max_iterations: int,
    dt: float,
    damp: float,
    num_samples: int,
) -> bool:
    """Validate iteration parameters for IK solvers.

    This helper validates common iteration parameters used by IK solvers. Returns
    True if all parameters are valid, False otherwise, and logs warnings for invalid
    parameters.

    Args:
        pos_eps: Position convergence threshold, must be positive.
        rot_eps: Rotation convergence threshold, must be positive.
        max_iterations: Maximum number of iterations, must be positive.
        dt: Time step size, must be positive.
        damp: Damping factor, must be non-negative.
        num_samples: Number of samples, must be positive.

    Returns:
        bool: True if all parameters are valid, False otherwise.
    """
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
    return True


def compute_pinocchio_fk(
    pin_module: Any,
    robot: "pin.RobotWrapper",
    qpos: Union[torch.Tensor, "np.ndarray"],
    end_link_name: str,
    tcp_xpos: "np.ndarray",
) -> "np.ndarray":
    """Compute forward kinematics using Pinocchio for the specified end-effector.

    This utility function computes FK using the Pinocchio library and applies
    the TCP transformation.

    Args:
        pin_module: The imported pinocchio module.
        robot: The Pinocchio RobotWrapper model.
        qpos: Joint positions, shape should be (nq,).
        end_link_name: The name of the end-effector link.
        tcp_xpos: The TCP transformation matrix (4x4).

    Returns:
        np.ndarray: The resulting end-effector pose as a (4, 4) homogeneous
            transformation matrix.

    Raises:
        ValueError: If qpos shape is not (nq,) or if the end_link_name is not found.
    """
    if isinstance(qpos, torch.Tensor):
        qpos_np = qpos.detach().cpu().numpy()
    else:
        qpos_np = np.array(qpos)

    qpos_np = np.squeeze(qpos_np)
    if qpos_np.ndim != 1:
        raise ValueError(f"qpos shape must be (nq,), but got {qpos_np.shape}")

    pin_module.forwardKinematics(robot.model, robot.data, qpos_np)

    # Retrieve the pose of the specified link
    frame_index = robot.model.getFrameId(end_link_name)
    if frame_index >= robot.model.nframes:
        raise ValueError(f"End link name '{end_link_name}' not found in robot model.")
    joint_index = robot.model.frames[frame_index].parent
    xpos_se3 = robot.data.oMi.tolist()[joint_index]

    xpos = np.eye(4)
    xpos[:3, :3] = xpos_se3.rotation
    xpos[:3, 3] = xpos_se3.translation.T

    result = np.dot(xpos, tcp_xpos)
    return result
