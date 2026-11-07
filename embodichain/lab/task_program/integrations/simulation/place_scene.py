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

"""Occurrence-bound ordinary Place scene input capture."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping, Callable
from typing import TYPE_CHECKING
from embodichain.lab.sim.atomic_actions.planning_probes import PlaceSceneInputs
from embodichain.lab.sim.scene.mesh_snapshot import capture_scene_meshes

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ResolvedActionRequest,
        PlanningContext,
    )

__all__ = ["PlaceSceneBinding", "SimulationPlaceSceneProvider"]


@dataclass(frozen=True, slots=True)
class PlaceSceneBinding:
    """Exact ordinary Place occurrence and declared contact roles.

    Args:
        invocation_id: Exact execution occurrence identity.
        floor_entity_id: Declared rigid floor geometry owner.
        support_owner_id: Optional intended support; never inferred from a name.
    """

    invocation_id: str
    floor_entity_id: str
    support_owner_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("invocation_id", "floor_entity_id", "support_owner_id"):
            value = getattr(self, name)
            if name == "support_owner_id" and value is None:
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")


class SimulationPlaceSceneProvider:
    """Capture owned input geometry for declared ordinary Place occurrences.

    Args:
        simulation: Borrowed B1 physical owner.
        rigid_uids: Complete ordered logical-to-native rigid mappings.
        bindings: Exact occurrence floor/support declarations.
        require_open: Optional owning factory lifetime validator.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        *,
        rigid_uids: Mapping[str, str],
        bindings: tuple[PlaceSceneBinding, ...],
        require_open: Callable[[], None] | None = None,
    ) -> None:
        if simulation.num_envs != 1:
            raise ValueError("Place scene input capture supports only B=1.")
        if (
            type(bindings) is not tuple
            or not bindings
            or any(type(x) is not PlaceSceneBinding for x in bindings)
        ):
            raise TypeError(
                "Place scene bindings require a nonempty immutable typed tuple."
            )
        if len({x.invocation_id for x in bindings}) != len(bindings):
            raise ValueError("Duplicate Place scene invocation identity.")
        self._simulation = simulation
        self._rigid = dict(rigid_uids)
        self._bindings = {x.invocation_id: x for x in bindings}
        self._require_open = require_open
        owners = set(self._rigid)
        if any(
            x.floor_entity_id not in owners
            or (x.support_owner_id is not None and x.support_owner_id not in owners)
            for x in bindings
        ):
            raise ValueError(
                "Place floor/support roles must select declared rigid geometry owners."
            )

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceSceneInputs:
        """Capture independent geometry for the exact declared B1/env0 Place.

        Args:
            request: Resolved ordinary Place invocation.
            context: Current planning observation.
        Returns:
            Owned meshes and explicit floor/support roles.
        """
        if self._require_open is not None:
            self._require_open()
        if (
            request.skill_id != "place"
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
        ):
            raise ValueError("Place scene inputs require a Place request in B1/env0.")
        binding = self._bindings.get(request.invocation_id)
        if binding is None:
            raise ValueError("Place occurrence has no declared scene-input binding.")
        geometry = capture_scene_meshes(self._simulation, rigid_uids=self._rigid)
        if self._require_open is not None:
            self._require_open()
        return PlaceSceneInputs(
            geometry=geometry,
            floor_entity_id=binding.floor_entity_id,
            support_owner_id=binding.support_owner_id,
        )
