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
"""Deprecated flat physics configs for the Default physics backend only.

The public transition import remains ``embodichain.lab.sim.cfg``. New code
must use the grouped ``RigidBodyPhysicsCfg`` hierarchy from that module. This
private module exists only to keep old Default-backend configurations working
while they are migrated and can be removed as one unit later.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from dexsim.types import PhysicalAttr

from embodichain.utils import configclass, logger
from embodichain.utils.math import convert_quat

__all__ = ["RigidBodyAttributesCfg", "RigidBodyAttributesOverrideCfg"]


@configclass
class RigidBodyAttributesCfg:
    """Deprecated flat rigid-body attributes for the Default backend.

    .. deprecated::
        Use ``RigidBodyPhysicsCfg`` and its grouped property configs. This
        compatibility class is not accepted by the Newton backend.
    """

    mass: float = 1.0
    """Mass of the rigid body in kilograms; zero selects density-based mass."""

    density: float = 1000.0
    """Density of the rigid body in kilograms per cubic meter."""

    inertia: Sequence[float] | np.ndarray | None = None
    """Optional principal moments or body-frame inertia tensor."""

    com_position: Sequence[float] | np.ndarray | None = None
    """Optional center-of-mass position in the body frame."""

    com_quaternion: Sequence[float] | np.ndarray | None = None
    """Optional center-of-mass orientation quaternion in ``xyzw`` order."""

    angular_damping: float = 0.7
    linear_damping: float = 0.7
    max_depenetration_velocity: float = 10.0
    sleep_threshold: float = 0.001
    min_position_iters: int = 4
    min_velocity_iters: int = 1
    max_linear_velocity: float = 1e2
    max_angular_velocity: float = 1e2
    enable_ccd: bool = False
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    enable_collision: bool = True
    restitution: float = 0.0
    dynamic_friction: float = 0.5
    static_friction: float = 0.5

    def attr(self) -> PhysicalAttr:
        """Convert the compatibility config to a Default-backend attribute."""
        attr = PhysicalAttr()
        for field_name in (
            "mass",
            "density",
            "contact_offset",
            "rest_offset",
            "dynamic_friction",
            "static_friction",
            "angular_damping",
            "linear_damping",
            "sleep_threshold",
            "restitution",
            "enable_ccd",
            "max_linear_velocity",
            "max_angular_velocity",
            "max_depenetration_velocity",
            "min_position_iters",
            "min_velocity_iters",
        ):
            setattr(attr, field_name, getattr(self, field_name))
        for field_name in ("inertia", "com_position", "com_quaternion"):
            value = getattr(self, field_name)
            if value is not None:
                array = np.asarray(value, dtype=np.float32)
                if field_name == "com_quaternion":
                    array = convert_quat(array, to="wxyz")
                setattr(attr, field_name, array)
        return attr

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> RigidBodyAttributesCfg:
        """Parse the deprecated flat Default-backend schema."""
        if "newton" in init_dict:
            raise ValueError(
                "Legacy RigidBodyAttributesCfg no longer accepts 'newton'. "
                "Use grouped NewtonCollisionPropertiesCfg and "
                "NewtonRigidBodyMaterialCfg instead."
            )
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(f"Key '{key}' not found in {cls.__name__}.")
        return cfg

    @classmethod
    def from_grouped(cls, grouped: Any) -> RigidBodyAttributesCfg:
        """Project a grouped config into this Default-only compatibility type."""
        cfg = cls()
        for field_name in cfg.__dataclass_fields__:
            setattr(cfg, field_name, getattr(grouped, field_name))
        return cfg


@configclass
class RigidBodyAttributesOverrideCfg:
    """Deprecated partial per-link override for the Default backend only."""

    mass: float | None = None
    density: float | None = None
    inertia: Sequence[float] | np.ndarray | None = None
    com_position: Sequence[float] | np.ndarray | None = None
    com_quaternion: Sequence[float] | np.ndarray | None = None
    angular_damping: float | None = None
    linear_damping: float | None = None
    max_depenetration_velocity: float | None = None
    sleep_threshold: float | None = None
    min_position_iters: int | None = None
    min_velocity_iters: int | None = None
    max_linear_velocity: float | None = None
    max_angular_velocity: float | None = None
    enable_ccd: bool | None = None
    contact_offset: float | None = None
    rest_offset: float | None = None
    enable_collision: bool | None = None
    restitution: float | None = None
    dynamic_friction: float | None = None
    static_friction: float | None = None

    def merge_with(self, base: RigidBodyAttributesCfg) -> PhysicalAttr:
        """Merge this override onto a flat config and return ``PhysicalAttr``."""
        return self.merged_cfg(base).attr()

    def merged_cfg(self, base: RigidBodyAttributesCfg) -> RigidBodyAttributesCfg:
        """Merge this override onto a full legacy Default-backend config."""
        merged = base.copy()
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                setattr(merged, field_name, value)
        return merged

    @classmethod
    def from_dict(
        cls,
        init_dict: dict[str, Any],
    ) -> RigidBodyAttributesOverrideCfg:
        """Parse a deprecated flat per-link override."""
        if "newton" in init_dict:
            raise ValueError(
                "Legacy RigidBodyAttributesOverrideCfg no longer accepts "
                "'newton'. Use grouped per-link physics instead."
            )
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(f"Key '{key}' not found in {cls.__name__}.")
        return cfg
