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

import abc  # for abstract base class definitions
from typing import TYPE_CHECKING, Optional, Dict
import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot


class Device(metaclass=abc.ABCMeta):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    """

    @abc.abstractmethod
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop_control(self):
        """
        Method that should be called externally to stop the controller.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_controller_state(self):
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""
        raise NotImplementedError

    def map_to_robot(
        self, robot: "Robot", device_data: Dict[str, float]
    ) -> Optional[torch.Tensor]:
        """Map device input to robot action (optional, device-specific).

        Args:
            robot: Robot instance to control.
            device_data: Device input data.

        Returns:
            Robot action tensor [num_envs, num_joints], or None if not implemented.
        """
        return None  # Default: no custom mapping
