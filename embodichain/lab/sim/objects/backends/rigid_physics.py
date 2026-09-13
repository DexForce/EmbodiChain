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
"""Backend-specific physical-property helpers for rigid objects."""

from __future__ import annotations

from typing import Any

from embodichain.utils import logger

__all__ = [
    "apply_legacy_physical_attr",
    "get_legacy_damping",
    "get_legacy_friction",
    "get_legacy_inertia",
    "get_legacy_mass",
    "can_use_newton_entity_dynamics_fallback",
    "get_newton_physical_attr",
    "get_newton_physical_attr_or_none",
    "mirror_newton_physical_attr",
    "newton_lifecycle_state",
    "set_legacy_collision_filter",
    "set_legacy_damping",
    "set_legacy_friction",
    "set_legacy_inertia",
    "set_legacy_mass",
]

_UINT64_MAX = (1 << 64) - 1


def _newton_metadata_attr(scene: object, entity: object) -> Any:
    """Return the legacy Newton metadata attribute for one entity."""
    entity_handle = int(entity.get_native_handle())
    if entity_handle < 0:
        entity_handle &= _UINT64_MAX
    manager = getattr(scene, "manager", None)
    if manager is None:
        return None
    return getattr(manager, "dexsim_meta", {}).get(entity_handle, {}).get("attr")


def get_newton_physical_attr(
    scene: object,
    entity: object,
    env_idx: int,
    object_uid: str,
) -> Any:
    """Return required Newton metadata physical attributes for an entity."""
    attr = _newton_metadata_attr(scene, entity)
    if attr is None:
        logger.log_error(
            f"Newton physical attributes for rigid object {object_uid!r} "
            f"env {env_idx} are unavailable."
        )
    return attr


def get_newton_physical_attr_or_none(scene: object, entity: object) -> Any:
    """Return Newton metadata physical attributes when available."""
    return _newton_metadata_attr(scene, entity)


def mirror_newton_physical_attr(
    scene: object,
    entity: object,
    env_idx: int,
    object_uid: str,
    physical_attr: Any,
) -> None:
    """Copy public physical attributes onto the Newton rebuild metadata."""
    attr = get_newton_physical_attr(scene, entity, env_idx, object_uid)
    for name in (
        "mass",
        "density",
        "dynamic_friction",
        "static_friction",
        "restitution",
        "contact_offset",
        "rest_offset",
        "linear_damping",
        "angular_damping",
        "sleep_threshold",
        "enable_ccd",
        "max_depenetration_velocity",
        "min_position_iters",
        "min_velocity_iters",
        "max_linear_velocity",
        "max_angular_velocity",
    ):
        setattr(attr, name, getattr(physical_attr, name))


def apply_legacy_physical_attr(entity: object, physical_attr: Any) -> None:
    """Apply a DexSim ``PhysicalAttr`` through the legacy entity API."""
    entity.set_physical_attr(physical_attr)


def set_legacy_collision_filter(entity: object, filter_data: Any) -> None:
    """Set collision filters through the legacy physical-body API."""
    entity.get_physical_body().set_collision_filter_data(filter_data)


def set_legacy_mass(entity: object, value: float) -> None:
    """Set mass through the legacy physical-body API."""
    entity.get_physical_body().set_mass(value)


def get_legacy_mass(entity: object) -> Any:
    """Read mass through the legacy physical-body API."""
    return entity.get_physical_body().get_mass()


def set_legacy_friction(entity: object, value: float) -> None:
    """Set both legacy static and dynamic friction coefficients."""
    body = entity.get_physical_body()
    body.set_dynamic_friction(value)
    body.set_static_friction(value)


def get_legacy_friction(entity: object) -> Any:
    """Read dynamic friction through the legacy physical-body API."""
    return entity.get_physical_body().get_dynamic_friction()


def set_legacy_damping(entity: object, linear: float, angular: float) -> None:
    """Set legacy linear and angular damping."""
    body = entity.get_physical_body()
    body.set_linear_damping(linear)
    body.set_angular_damping(angular)


def get_legacy_damping(entity: object) -> tuple[Any, Any]:
    """Read legacy linear and angular damping."""
    body = entity.get_physical_body()
    return body.get_linear_damping(), body.get_angular_damping()


def set_legacy_inertia(entity: object, value: Any) -> None:
    """Set legacy mass-space inertia diagonal."""
    entity.get_physical_body().set_mass_space_inertia_tensor(value)


def get_legacy_inertia(entity: object) -> Any:
    """Read legacy mass-space inertia diagonal."""
    return entity.get_physical_body().get_mass_space_inertia_tensor()


def newton_lifecycle_state(scene: object) -> str:
    """Return the current Newton lifecycle state name."""
    manager = getattr(scene, "manager", None)
    return getattr(getattr(manager, "lifecycle_state", None), "name", "")


def can_use_newton_entity_dynamics_fallback(scene: object) -> bool:
    """Whether legacy Newton entity dynamics helpers are safe to call."""
    return newton_lifecycle_state(scene) == "BUILDER"
