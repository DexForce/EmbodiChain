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

"""Stable source-path mappings for a declared USD import."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import (
        Articulation,
        DeformableObject,
        RigidObject,
        Robot,
        VisualObject,
    )

__all__ = ["UsdImportResult"]


class UsdImportResult(
    Mapping[str, "VisualObject | RigidObject | Articulation | Robot | DeformableObject"]
):
    """Map source prim paths to stable, declared EmbodiChain facades.

    Facades bind in place at ``SimulationManager.prepare()``. Resource maps
    include native-only scene resources and constraints; their values are
    logical Spawn UIDs, not native handles or builder indices.
    """

    def __init__(
        self,
        assets: Mapping[
            str, VisualObject | RigidObject | Articulation | Robot | DeformableObject
        ],
        *,
        resource_uids: Mapping[str, str],
        resource_types: Mapping[str, str],
        constraint_uids: Mapping[str, str],
    ) -> None:
        """Capture mappings for one successful declaration.

        Args:
            assets: Source prim paths mapped to stable EmbodiChain facades.
            resource_uids: Source paths mapped to logical Spawn identifiers.
            resource_types: Logical scene resource kinds for those source paths.
            constraint_uids: Source constraint paths mapped to logical identifiers.
        """
        self._assets = MappingProxyType(dict(assets))
        self.resource_uids: Mapping[str, str] = MappingProxyType(dict(resource_uids))
        self.resource_types: Mapping[str, str] = MappingProxyType(dict(resource_types))
        self.constraint_uids: Mapping[str, str] = MappingProxyType(
            dict(constraint_uids)
        )

    def __getitem__(
        self, source_path: str
    ) -> VisualObject | RigidObject | Articulation | Robot | DeformableObject:
        return self._assets[source_path]

    def __iter__(self) -> Iterator[str]:
        return iter(self._assets)

    def __len__(self) -> int:
        return len(self._assets)
