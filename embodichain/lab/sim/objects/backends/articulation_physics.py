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
"""Backend-specific articulation link property accessors."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "apply_link_com_pose",
    "apply_link_inertia",
    "apply_link_mass",
    "get_link_properties",
    "set_link_physical_attr",
]


def get_link_properties(entity: object, link_name: str, *, is_newton: bool) -> Any:
    """Read a link's physical properties through the active backend API."""
    if is_newton:
        return entity.get_newton_link_properties(link_name)
    return entity.get_physical_attr(link_name)


def set_link_physical_attr(
    entity: object,
    physical_attr: object,
    link_name: str,
    *,
    replace_inertial: bool,
) -> None:
    """Apply a DexSim ``PhysicalAttr`` to one articulation link."""
    entity.set_physical_attr(
        physical_attr,
        link_name,
        is_replace_inertial=replace_inertial,
    )


def apply_link_mass(
    entity: object,
    link_name: str,
    value: float,
    *,
    is_spawn_bound: bool,
    is_newton: bool,
) -> None:
    """Apply one link mass using Spawn, Newton, or legacy APIs."""
    if is_spawn_bound or is_newton:
        entity.set_link_mass(link_name, value)
    else:
        entity.set_mass(link_name, value)


def apply_link_inertia(
    entity: object,
    link_name: str,
    value: np.ndarray,
    *,
    is_spawn_bound: bool,
    is_newton: bool,
) -> int | None:
    """Apply one link inertia and return a Spawn status when available."""
    if is_spawn_bound:
        return int(entity.set_link_inertia(link_name, value))
    if not is_newton:
        entity.get_physical_body(link_name).set_mass_space_inertia_tensor(value)
        return None
    attr = entity.get_physical_attr(link_name)
    attr.inertia = value
    entity.set_physical_attr(attr, link_name, is_replace_inertial=False)
    return None


def apply_link_com_pose(
    entity: object,
    link_name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    *,
    is_spawn_bound: bool,
    is_newton: bool,
) -> int | None:
    """Apply one link local COM pose and return a Spawn status when available."""
    if is_spawn_bound:
        return int(entity.set_link_com_pose(link_name, position, quaternion))
    if not is_newton:
        entity.get_physical_body(link_name).set_cmass_local_pose(position, quaternion)
        return None
    attr = entity.get_physical_attr(link_name)
    attr.com_position = position
    attr.com_quaternion = quaternion
    entity.set_physical_attr(attr, link_name, is_replace_inertial=False)
    return None
