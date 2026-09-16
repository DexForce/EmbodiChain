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
"""Backend-specific articulation lifecycle hooks."""

from __future__ import annotations

from typing import Sequence

from embodichain.lab.sim.utility.sim_utils import (
    _apply_default_articulation_root_properties,
)

__all__ = ["prepare_default_spawn_runtime_config"]


def prepare_default_spawn_runtime_config(
    result: object | None,
    entities: Sequence[object],
    root_props: object | None,
    prepared_topology_revision: int,
) -> int:
    """Apply Default root properties before its GPU runtime is initialized."""
    if result is None or getattr(result, "backend", None) != "dexsim":
        return prepared_topology_revision

    topology_revision = int(result.topology_revision)
    if prepared_topology_revision == topology_revision:
        return prepared_topology_revision

    configured = root_props is not None and (
        root_props.sleep_threshold is not None
        or root_props.min_position_iters is not None
        or root_props.min_velocity_iters is not None
    )
    if configured:
        for entity in entities:
            native_articulation = getattr(entity, "_physics_binding", None)
            if native_articulation is None:
                raise RuntimeError(
                    "Default Spawn articulation has no native physics binding."
                )
            _apply_default_articulation_root_properties(
                native_articulation,
                root_props,
            )
    return topology_revision
