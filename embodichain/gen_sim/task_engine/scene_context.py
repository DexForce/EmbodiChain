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

"""Resolve caller-owned scene object declarations for hybrid authoring."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

__all__: list[str] = []

_SCENE_ROLE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class ArticulationDeclarationError(ValueError):
    """An authored operation requires a joint the caller has not enabled."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _effective_asset_kind(
    object_id: str, scene_objects: Mapping[str, Mapping[str, str]]
) -> str:
    """Resolve the caller's declaration, defaulting to a rigid asset."""
    return scene_objects.get(object_id, {}).get("asset_kind", "rigid")


def require_articulation(
    object_id: str,
    scene_objects: Mapping[str, Mapping[str, str]],
    part_owners: Mapping[str, str],
) -> None:
    """Require an articulated owner without overriding explicit rigid declarations."""
    visited = set()
    while (
        object_id in part_owners
        and scene_objects.get(object_id, {}).get("asset_kind") != "rigid"
    ):
        if object_id in visited:
            raise ValueError(f"Cyclic scene object ownership for {object_id!r}")
        visited.add(object_id)
        object_id = part_owners[object_id]
    if _effective_asset_kind(object_id, scene_objects) == "articulated":
        return
    if scene_objects.get(object_id, {}).get("asset_kind") == "rigid":
        raise ArticulationDeclarationError(
            f"Scene object {object_id!r} is declared rigid but requires articulation"
        )
    if scene_objects.get(object_id, {}).get("asset_kind") is None and any(
        row.get("asset_kind") == "articulated" for row in scene_objects.values()
    ):
        raise ArticulationDeclarationError(
            f"Unresolved articulated owner for {object_id!r}; provide the source-supported "
            "part_of link to an articulated object declared in --scene-context",
            retryable=True,
        )
    raise ArticulationDeclarationError(
        f"Scene object {object_id!r} requires articulation; declare it with "
        "asset_kind: articulated in --scene-context"
    )


def scene_objects_requirement(
    scene_objects: Mapping[str, Mapping[str, str]],
    role_descriptions: Mapping[str, str],
) -> dict[str, Any]:
    """Publish scene objects without introducing any additional task roles."""
    descriptions = {**role_descriptions}
    descriptions.update(
        {name: row["description"] for name, row in scene_objects.items()}
    )
    objects = []
    for name, description in sorted(descriptions.items()):
        kind = _effective_asset_kind(name, scene_objects)
        objects.append(
            dict(
                id=name,
                description=description,
                asset_kind=kind,
                is_articulated=kind == "articulated",
            )
        )
    return dict(consumer="scene", key="scene_objects", value=objects)
