# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

import sys
import torch

from abc import abstractmethod
from collections.abc import Mapping
from typing import (
    Dict,
    List,
    Any,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
    get_origin,
    get_args,
    get_type_hints,
)
from functools import cached_property
from dataclasses import MISSING
from types import SimpleNamespace, UnionType
from tensordict import TensorDict

from embodichain.lab.sim.cfg import ObjectBaseCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils.math import matrix_from_quat
from embodichain.lab.sim.utility import get_dexsim_arena_num
from embodichain.utils import configclass, is_configclass, logger

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["BaseSensor", "SensorCfg"]


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
        quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Orientation in the parent frame as ``(x, y, z, w)``. Defaults to identity."""

        parent: str | None = None
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
        from .registry import get_sensor_registration

        sensor_type = init_dict["sensor_type"]
        try:
            registration = get_sensor_registration(sensor_type)
        except KeyError as error:
            raise ValueError(str(error)) from error
        return _populate_sensor_config(registration.config_cls(), init_dict)


def _populate_sensor_config(cfg: Any, init_dict: dict[str, Any]) -> Any:
    """Decode typed nested configclasses without comparing values to MISSING."""
    type_hints = {}
    for base in reversed(type(cfg).__mro__):
        annotations = base.__dict__.get(
            "__configclass_annotations__", base.__dict__.get("__annotations__", {})
        )
        if not annotations:
            continue
        localns = {
            name: value
            for ancestor in reversed(base.__mro__)
            for name, value in vars(ancestor).items()
        }
        # Evaluate only this class's declarations, not flattened inherited fields.
        type_hints.update(
            get_type_hints(
                SimpleNamespace(__annotations__=annotations),
                globalns=vars(sys.modules[base.__module__]),
                localns=localns,
            )
        )

    for key, value in init_dict.items():
        if hasattr(cfg, key):
            setattr(
                cfg,
                key,
                _decode_config_value(
                    value, type_hints.get(key), getattr(cfg, key), key
                ),
            )
        else:
            logger.log_warning(f"Key '{key}' not found in {type(cfg).__name__}.")
    return cfg


def _decode_config_value(
    value: Any, annotation: Any, default: Any = None, field_name: str = ""
) -> Any:
    """Preserve custom decoders and resolve nullable/required nested types."""
    nullable = get_origin(annotation) in (Union, UnionType) and type(None) in get_args(
        annotation
    )
    if value is None:
        if not nullable and (is_configclass(annotation) or is_configclass(default)):
            raise ValueError(
                f"Sensor configuration field {field_name!r} does not accept null."
            )
        return None
    if get_origin(annotation) in (Union, UnionType):
        candidates = [item for item in get_args(annotation) if item is not type(None)]
        if len(candidates) == 1:
            annotation = candidates[0]
    config_type = (
        type(default)
        if is_configclass(default)
        else annotation if is_configclass(annotation) else None
    )
    if config_type is not None:
        if isinstance(value, config_type) or (
            is_configclass(annotation) and isinstance(value, annotation)
        ):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Sensor configuration field {field_name!r} expects "
                f"{config_type.__name__} or a mapping, received {type(value).__name__}."
            )
        decoder = getattr(config_type, "from_dict", None)
        if callable(decoder):
            if is_configclass(default):
                defaults = _resolved_config_defaults(default, default.to_dict())
                value = _merge_config_overrides(defaults, value)
            return decoder(value)
        cfg = default if is_configclass(default) else config_type()
        return _populate_sensor_config(cfg, value)
    if isinstance(value, list) and get_origin(annotation) is list:
        args = get_args(annotation)
        if args:
            return [
                _decode_config_value(item, args[0], field_name=f"{field_name}[{index}]")
                for index, item in enumerate(value)
            ]
    return value


def _resolved_config_defaults(source: Any, serialized: Any) -> Any:
    """Exclude unresolved defaults that generic serialization represents as {}."""
    # Nested config copies may contain a distinct _MISSING_TYPE instance;
    # match configclass.validate(), which recognizes missing values by type.
    if type(source) is type(MISSING):
        return MISSING
    if isinstance(serialized, dict):
        resolved = {}
        for key, value in serialized.items():
            original = (
                source.get(key)
                if isinstance(source, dict)
                else getattr(source, key, None)
            )
            if type(original) is not type(MISSING):
                resolved[key] = _resolved_config_defaults(original, value)
        return resolved
    if isinstance(serialized, (list, tuple)) and isinstance(source, (list, tuple)):
        return type(serialized)(
            _resolved_config_defaults(original, value)
            for original, value in zip(source, serialized)
        )
    return serialized


def _merge_config_overrides(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Keep calibrated defaults while letting custom decoders normalize overrides."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            value = _merge_config_overrides(merged[key], value)
        merged[key] = value
    return merged


class BaseSensor(BatchEntity):
    """Base class for sensor abstraction in the simulation engine.

    Sensors should inherit from this class and implement the `update` and `get_data` methods.

    Args:
        config: Sensor configuration.
        device: Output device. Defaults to the context device, or CPU without
            a context. An explicit device must agree with the context.
        num_instances: Explicit environment count for legacy constructors.
            Defaults to the context count when provided; only the legacy path
            without an explicit count queries the native arena count.
        context: Runtime dependencies supplied by the registered-factory path.
            This bypasses singleton/native arena discovery. Legacy direct
            constructors remain supported without a context.
    """

    SUPPORTED_DATA_TYPES = []

    def __init__(
        self,
        config: SensorCfg,
        device: torch.device | None = None,
        *,
        num_instances: int | None = None,
        context: SensorContext | None = None,
    ) -> None:
        if context is not None:
            if device is not None and torch.device(device) != context.device:
                raise ValueError("Sensor device must match its SensorContext.")
            if num_instances is not None and num_instances != context.num_instances:
                raise ValueError("Sensor instance count must match its SensorContext.")
            device = context.device
            num_instances = context.num_instances
        elif device is None:
            device = torch.device("cpu")
        self.context = context
        num_envs = (
            get_dexsim_arena_num() if num_instances is None else int(num_instances)
        )
        if num_envs <= 0:
            raise ValueError("A sensor requires at least one simulation instance.")
        self._num_instances = num_envs
        self._data_buffer: TensorDict[str, torch.Tensor] = TensorDict(
            {}, batch_size=[num_envs], device=device
        )

        self._entities = [None for _ in range(num_envs)]
        self._build_sensor_from_config(config, device=device)

        super().__init__(config, self._entities, device)

    @cached_property
    def num_instances(self) -> int:
        return self._num_instances

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
            to_matrix: If True, return the pose as a 4x4 transformation matrix;
                otherwise return ``(x, y, z, qx, qy, qz, qw)``.

        Returns:
            A tensor representing the pose of the sensor in the arena frame.
        """
        logger.log_error("Not implemented yet.")

    def get_data(self) -> TensorDict:
        """Retrieve data from the sensor.

        Args:
            copy: If True, return a copy of the data buffer. Defaults to True.

        Returns:
            The data collected by the sensor.
        """
        return self._data_buffer

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        return super().reset(env_ids)

    @property
    def requires_substep_update(self) -> bool:
        """Whether the environment must sample this sensor each physics substep."""
        return False

    def begin_control_step(self) -> None:
        """Begin interval accumulation for a sensor that samples physics substeps."""

    def update_physics_step(self, dt: float) -> None:
        """Sample a sensor that opts into physics substeps.

        Args:
            dt: Elapsed physics time in seconds.
        """
        self.update()
