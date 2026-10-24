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

"""Bind physical collision links independently of task recipe membership."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .drawer_geometry import collision_meshes

__all__: list[str] = []


def additional_collision_links(
    configs: Iterable[Mapping[str, Any]], existing: Iterable[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Reuse semantic link identities and bind every remaining collision owner."""
    declared = tuple(existing)
    native_ids: dict[tuple[str, str], str] = {}
    used_ids = {str(link["entity_id"]) for link in declared}
    for link in declared:
        key = str(link["articulation_id"]), str(link["native_link_name"])
        identity = str(link["entity_id"])
        if key in native_ids and native_ids[key] != identity:
            raise ValueError(
                "One physical collision link has ambiguous semantic identities."
            )
        native_ids[key] = identity
    result = []
    for config in sorted(configs, key=lambda value: str(value["uid"])):
        uid = str(config["uid"])
        for native in sorted({owner for owner, _ in collision_meshes(dict(config))}):
            key = uid, native
            if key in native_ids:
                continue
            identity = f"gen_sim.collision_link.{uid}.{native}"
            if identity in used_ids:
                raise ValueError(
                    "Collision-link identity conflicts with an existing link."
                )
            used_ids.add(identity)
            native_ids[key] = identity
            result.append(
                {
                    "entity_id": identity,
                    "articulation_id": uid,
                    "native_link_name": native,
                }
            )
    return result
