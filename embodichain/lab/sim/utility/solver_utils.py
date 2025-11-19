# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import torch
from embodichain.lab.sim.utility.io_utils import suppress_stdout_stderr

from typing import Optional, Union, Tuple, Any, TYPE_CHECKING
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
