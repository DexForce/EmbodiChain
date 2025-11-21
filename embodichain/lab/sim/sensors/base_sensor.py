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

from __future__ import annotations

import torch

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Sequence, Tuple, Union
from embodichain.lab.sim.cfg import ObjectBaseCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils.math import matrix_from_quat
from embodichain.lab.sim.utility import get_dexsim_arena_num
from embodichain.utils import configclass, is_configclass, logger


@configclass
class SensorCfg(ObjectBaseCfg):
    """Configuration class for sensors.

    This class can be extended to include specific sensor configurations.
    """

    @configclass
    class OffsetCfg:
        """Configuration of the sensor offset relative to the parent frame."""

        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the sensor in the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Orientation of the sensor in the parent frame as a quaternion (w, x, y, z). Defaults to (1.0, 0.0, 0.0, 0.0)."""

        parent: Optional[str] = None
        """Name of the parent frame. If not specified, the sensor will be placed in the arena frame.

        This is usually the case when the sensor is not attached to any specific object, eg, link of a robot arm.
        """

        @property
        def transformation(self) -> torch.Tensor:
            pos = torch.tensor(self.pos, dtype=torch.float32)
            quat = torch.tensor(self.quat, dtype=torch.float32)
            rot = matrix_from_quat(quat.unsqueeze(0)).squeeze(0)
            T = torch.eye(4, dtype=torch.float32)
            T[:3, :3] = rot
            T[:3, 3] = pos
            return T

        @classmethod
        def from_dict(cls, init_dict: dict) -> SensorCfg.OffsetCfg:
            cfg = cls()
            for key, value in init_dict.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                else:
                    logger.log_warning(f"Key '{key}' not found in {cls.__name__}.")
            return cfg

    @abstractmethod
    def get_data_types(self) -> List[str]:
        """Get the data types supported by this sensor configuration.

        Returns:
            A list of data types that this sensor configuration supports.
        """
        return []

    sensor_type: str = "BaseSensor"

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> "SensorCfg":
        """Initialize the configuration from a dictionary."""
        from embodichain.utils.utility import get_class_instance

        cfg = get_class_instance(
            "embodichain.lab.sim.sensors", init_dict["sensor_type"] + "Cfg"
        )()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


class BaseSensor(BatchEntity):
    """Base class for sensor abstraction in the simulation engine.

    Sensors should inherit from this class and implement the `update` and `get_data` methods.
    """

    SUPPORTED_DATA_TYPES = []

    def __init__(
        self, config: SensorCfg, device: torch.device = torch.device("cpu")
    ) -> None:

        self._data_buffer: Dict[str, torch.Tensor] = {}

        self._entities = [None for _ in range(get_dexsim_arena_num())]
        self._build_sensor_from_config(config, device=device)

        super().__init__(config, self._entities, device)

    @abstractmethod
    def _build_sensor_from_config(
        self, config: SensorCfg, device: torch.device
    ) -> None:
        """Build the sensor from the provided configuration.

        Args:
            config: The configuration for the sensor.
            device: The device of the sensor
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update the sensor state based on the current simulation state.

        This method is called periodically to ensure the sensor data is up-to-date.

        Args:
            **kwargs: Additional keyword arguments for sensor update.
        """
        pass

    @abstractmethod
    def get_arena_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get the pose of the sensor in the arena frame.

        Args:
            to_matrix: If True, return the pose as a 4x4 transformation matrix.

        Returns:
            A tensor representing the pose of the sensor in the arena frame.
        """
        logger.log_error("Not implemented yet.")

    def get_data(self, copy: bool = True) -> Dict[str, torch.Tensor]:
        """Retrieve data from the sensor.

        Args:
            copy: If True, return a copy of the data buffer. Defaults to True.

        Returns:
            The data collected by the sensor.
        """
        if copy:
            return {key: value.clone() for key, value in self._data_buffer.items()}
        return self._data_buffer

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        return super().reset(env_ids)
