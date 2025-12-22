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

from embodichain.utils import logger


def lazy_import_pytorch_kinematics():
    """
    Lazily import pytorch_kinematics and return the module.

    Returns:
        module: The pytorch_kinematics module if available.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        import pytorch_kinematics as pk

        return pk
    except ImportError as e:
        logger.log_warning(
            "pytorch_kinematics not installed. Install with `pip install pytorch_kinematics==0.7.6`"
        )
        raise e


def lazy_import_pinocchio():
    """
    Lazily import pinocchio and return the module.

    Returns:
        module: The pinocchio module if available.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        import pinocchio as pin

        return pin
    except ImportError as e:
        logger.log_warning(
            "pinocchio not installed. Install with `conda install pinocchio==3.1.0 -c conda-forge`"
        )
        raise e


def lazy_import_casadi():
    """
    Lazily import casadi and return the module.

    Returns:
        module: The casadi module if available.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        import casadi

        return casadi
    except ImportError as e:
        logger.log_warning(
            "casadi not installed. Install with `pip install casadi==3.6.7`"
        )
        raise e


def lazy_import_pinocchio_casadi():
    """
    Lazily import pinocchio.casadi and return the module.

    Returns:
        module: The pinocchio.casadi module if available.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        from pinocchio import casadi as cpin

        return cpin
    except ImportError as e:
        logger.log_warning(
            f"Failed to import pinocchio.casadi: {e}. Install with `conda install pinocchio-casadi -c conda-forge` first."
        )
        raise e


def lazy_import_pink():
    """
    Lazily import pin-pink and return its components.

    Returns:
        tuple: The solve_ik, Configuration, and FrameTask components.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        from pink import solve_ik
        from pink.configuration import Configuration
        from pink.tasks import FrameTask
        import pink

        return pink
    except ImportError as e:
        logger.log_warning(
            "Failed to import 'pin-pink'. Please install it using `pip install pin-pink==3.4.0`."
        )
        raise ImportError("pin-pink is required but not installed.") from e
