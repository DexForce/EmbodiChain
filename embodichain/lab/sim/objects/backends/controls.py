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
"""Native entity control helpers shared by simulation object facades."""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_body_scale",
    "set_articulation_flag",
    "set_body_scale",
    "set_collision_enabled",
    "set_physical_visible",
    "set_gravity_enabled",
    "create_physical_visible_node",
    "set_visible",
]


def get_body_scale(entity: object) -> Any:
    """Read one native rigid-body visual scale."""
    return entity.get_body_scale()


def set_body_scale(entity: object, scale: Any) -> None:
    """Set one native rigid-body visual scale."""
    entity.set_body_scale(*scale)


def set_collision_enabled(entity: object, enabled: bool) -> None:
    """Enable or disable collision on one native rigid entity."""
    entity.enable_collision(bool(enabled))


def create_physical_visible_node(
    entity: object,
    rgba: Any,
    link_name: str | None = None,
) -> None:
    """Create a native collision-visual node for an entity or link."""
    if link_name is None:
        entity.create_physical_visible_node(rgba)
    else:
        entity.create_physical_visible_node(rgba, link_name)


def set_physical_visible(
    entity: object,
    visible: bool,
    link_name: str | None = None,
) -> None:
    """Set native collision-visual visibility for an entity or link."""
    if link_name is None:
        entity.set_physical_visible(bool(visible))
    else:
        entity.set_physical_visible(bool(visible), link_name)


def set_visible(entity: object, visible: bool) -> None:
    """Set one native entity's render visibility."""
    entity.set_visible(bool(visible))


def set_articulation_flag(entity: object, flag: object, enabled: bool) -> None:
    """Set one native articulation flag."""
    entity.set_articulation_flag(flag, bool(enabled))


def set_gravity_enabled(entity: object, enabled: bool) -> None:
    """Enable or disable gravity on one native articulation."""
    entity.enable_gravity(bool(enabled))
