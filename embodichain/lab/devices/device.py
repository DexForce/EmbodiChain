# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import abc  # for abstract base class definitions


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
