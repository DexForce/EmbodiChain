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

"""Occurrence-bound physical scene inputs for shared Place recovery."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from embodichain.lab.sim.atomic_actions.planning_probes import (
    PlaceSceneInputs,
    PlaceInsideLanding,
)
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement
from .bindings import ContainerAffordanceBinding
from embodichain.lab.sim.scene.mesh_snapshot import capture_scene_meshes

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ResolvedActionRequest,
        PlanningContext,
    )

__all__ = ["PlaceSceneBinding", "SimulationPlaceSceneProvider"]


@dataclass(frozen=True, slots=True)
class PlaceSceneBinding:
    """Bind floor and intended-contact roles to an exact execution occurrence.

    Args:
        invocation_id: Complete execution occurrence identity.
        floor_entity_id: Explicit floor geometry owner.
        support_owner_id: Optional intended-contact owner; never inferred from a name.
        inside_subject_id: Optional exact rigid payload for late inside recovery.
        inside_container_id: Optional canonical container affordance. Both inside
            identities must be supplied together; its parent owns release clearance.
    """

    invocation_id: str
    floor_entity_id: str
    support_owner_id: str | None = None
    inside_subject_id: str | None = None
    inside_container_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "invocation_id",
            "floor_entity_id",
            "support_owner_id",
            "inside_subject_id",
            "inside_container_id",
        ):
            value = getattr(self, name)
            if (
                name in {"support_owner_id", "inside_subject_id", "inside_container_id"}
                and value is None
            ):
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")
        if (self.inside_subject_id is None) != (self.inside_container_id is None):
            raise ValueError(
                "Inside landing requires both subject and container identities."
            )


class SimulationPlaceSceneProvider:
    """Capture planning geometry using explicit native mappings and occurrence roles.

    Args:
        simulation: Borrowed physical scene owner.
        rigid_uids: Ordered logical rigid IDs mapped to native UIDs.
        articulation_uids: Ordered logical articulation IDs mapped to native UIDs.
        link_bindings: Ordered articulation/native-link/pose-ID triples.
        bindings: Exact invocation-specific floor and contact roles.
        containers: Existing authoritative container-frame declarations; optional
            inside recovery reads their rigid parent and release offset.

    No poses, policy values or task labels are inferred here. Geometry remains
    source input; physical release/support evidence is still independently required.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        *,
        rigid_uids: Mapping[str, str],
        articulation_uids: Mapping[str, str] | None = None,
        link_bindings: Sequence[tuple[str, str, str]] = (),
        bindings: tuple[PlaceSceneBinding, ...],
        containers: tuple[ContainerAffordanceBinding, ...] = (),
    ) -> None:
        if simulation.num_envs != 1:
            raise ValueError("Source Place scene inputs support only B=1.")
        if (
            type(bindings) is not tuple
            or not bindings
            or any(type(value) is not PlaceSceneBinding for value in bindings)
        ):
            raise TypeError("Place scene bindings require a nonempty immutable tuple.")
        selected = {value.invocation_id: value for value in bindings}
        if len(selected) != len(bindings):
            raise ValueError("Duplicate Place scene invocation identity.")
        self._simulation = simulation
        self._rigid = dict(rigid_uids)
        self._articulations = dict(articulation_uids or {})
        self._links = tuple(tuple(value) for value in link_bindings)
        self._bindings = selected
        if type(containers) is not tuple or any(
            type(value) is not ContainerAffordanceBinding for value in containers
        ):
            raise TypeError("Place containers require immutable typed declarations.")
        self._containers = {value.entity_id: value for value in containers}
        if len(self._containers) != len(containers):
            raise ValueError("Inside landing container identities must be unique.")
        self._inside_geometry: dict[tuple[str, str], NativePlanarPlacement] = {}
        owners = set(self._rigid) | set(self._articulations)
        for binding in bindings:
            if binding.floor_entity_id not in self._rigid or (
                binding.support_owner_id is not None
                and binding.support_owner_id not in owners
            ):
                raise ValueError(
                    "Place floor/support roles must select declared geometry owners."
                )
            if binding.inside_subject_id is not None:
                _validate_inside_binding(binding, self._rigid, self._containers)

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceSceneInputs:
        """Capture owned geometry for this exact Place occurrence.

        Args:
            request: The engine's resolved action request.
            context: The current B1/env0 planning observation.

        Returns:
            Owned input meshes and the explicitly declared contact roles.
        """
        if (
            request.skill_id != "place"
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
        ):
            raise ValueError("Place scene inputs require a Place request in B1/env0.")
        binding = self._bindings.get(request.invocation_id)
        if binding is None:
            raise ValueError("Place occurrence has no declared scene-input binding.")
        geometry = self._capture_geometry()
        return PlaceSceneInputs(
            geometry=geometry,
            floor_entity_id=binding.floor_entity_id,
            support_owner_id=binding.support_owner_id,
        )

    def _capture_geometry(self):
        return capture_scene_meshes(
            self._simulation,
            rigid_uids=self._rigid,
            articulation_uids=self._articulations,
            link_bindings=self._links,
        )

    def prepare_inside_landing(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceSceneInputs:
        """Acquire geometry on the failure branch, retaining exact native owners.

        Args:
            request: Resolved Place request for one declared occurrence.
            context: Same B1/env0 planning observation used by the action.

        Returns:
            Native geometry and existing container-frame declaration. No live
            poses or IK are read, and replacing native owners invalidates the
            cached geometry instead of silently refreshing it.
        """
        if (
            request.skill_id != "place"
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
        ):
            raise ValueError("Inside landing requires a Place request in B1/env0.")
        binding = self._bindings.get(request.invocation_id)
        if binding is None or binding.inside_subject_id is None:
            raise ValueError("Place occurrence has no declared inside landing binding.")
        scene_geometry = self._capture_geometry()
        container = self._containers[binding.inside_container_id]
        key = (binding.inside_subject_id, container.parent_id)
        geometry = self._inside_geometry.get(key)
        if geometry is None:
            geometry = NativePlanarPlacement(
                self._simulation, self._rigid[key[0]], self._rigid[key[1]], batch_size=1
            )
            self._inside_geometry[key] = geometry
        landing = PlaceInsideLanding(
            geometry=geometry,
            subject_id=key[0],
            container_id=container.entity_id,
            parent_id=key[1],
            release_clearance=container.release_clearance,
        )
        return PlaceSceneInputs(
            geometry=scene_geometry,
            floor_entity_id=binding.floor_entity_id,
            support_owner_id=binding.support_owner_id,
            inside_landing=landing,
        )


def _validate_inside_binding(binding, rigid_uids, containers):
    container = containers.get(binding.inside_container_id)
    if (
        container is None
        or binding.inside_subject_id not in rigid_uids
        or container.parent_id not in rigid_uids
    ):
        raise ValueError(
            "Inside landing requires declared rigid subject/parent and an exact container frame."
        )
    if (
        binding.inside_subject_id == container.parent_id
        or binding.support_owner_id != container.parent_id
    ):
        raise ValueError(
            "Inside landing requires a distinct subject and matching explicit support owner."
        )
    if container.release_clearance != 0.12:
        raise ValueError(
            "Inside yaw recovery requires the declared 0.12 m release clearance."
        )
