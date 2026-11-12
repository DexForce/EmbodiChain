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
    """Map source resource keys to stable, declared EmbodiChain facades.

    Facades bind in place at ``SimulationManager.prepare()``. Resource maps
    include native-only scene resources and constraints; their values are
    logical Spawn UIDs, not native handles or builder indices. A unique source
    keeps its prim path as the resource key. When a source prim produces several
    resources, each uses ``<prim_path>#<ordinal>`` in parser order, starting at
    zero. Keys are independent of namespace and runtime UID changes;
    :attr:`resource_source_paths` retains the exact prim path for every key.
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
        resource_source_paths: Mapping[str, str] | None = None,
    ) -> None:
        """Capture mappings for one successful declaration.

        Args:
            assets: Source resource keys mapped to stable EmbodiChain facades.
            resource_uids: Source resource keys mapped to logical Spawn identifiers.
            resource_types: Logical scene resource kinds for the same resource keys.
            constraint_uids: Source constraint resource keys mapped to identifiers.
            resource_source_paths: Resource keys mapped to their exact source prim
                paths. Defaults to identity mapping for singleton resources.
        """
        self._assets = MappingProxyType(dict(assets))
        self.resource_source_paths: Mapping[str, str] = MappingProxyType(
            dict(resource_source_paths)
            if resource_source_paths is not None
            else {key: key for key in resource_uids}
        )
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
