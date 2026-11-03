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

"""Owned rigid visual envelopes and articulated collision-input components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "SceneMeshComponent",
    "SceneMeshSnapshot",
    "capture_scene_meshes",
    "iter_scene_components",
]

_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
    OSError,
    NotImplementedError,
)


def _owned_vertices(value: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or not value.shape[0]
        or value.shape[-1] != 3
        or not value.shape[1]
        or not torch.isfinite(value).all()
    ):
        raise ValueError("scene_collision_vertices_unavailable")
    return value.detach().clone()


@dataclass(frozen=True, slots=True)
class SceneMeshComponent:
    """One owned component expressed in its owning rigid-body/link frame.

    Args:
        owner_id: Caller-selected logical object identity.
        pose_entity_id: Identity used to obtain its separately observed pose.
        name: Stable source/part name; component order is preserved.
        vertices: Already-scaled local points, shape ``(B, V, 3)``.
        faces: Optional triangle topology, shape ``(F, 3)``.
        geometry_kind: Distinguishes visual envelopes from collision inputs.
        provenance: Input route, not a native-cooked or contact certificate.
    """

    owner_id: str
    pose_entity_id: str
    name: str
    vertices: torch.Tensor
    faces: torch.Tensor | None = None
    geometry_kind: Literal["rigid_visual_envelope", "articulation_collision_input"] = (
        "articulation_collision_input"
    )
    provenance: str = "provided_link_local_input"

    def __post_init__(self) -> None:
        if self.geometry_kind not in {
            "rigid_visual_envelope",
            "articulation_collision_input",
        }:
            raise ValueError("Unknown scene mesh geometry kind.")
        object.__setattr__(self, "vertices", _owned_vertices(self.vertices))
        if self.faces is not None:
            if (
                self.faces.ndim != 2
                or self.faces.shape[1] != 3
                or self.faces.dtype not in (torch.int32, torch.int64)
                or (self.faces < 0).any()
                or (self.faces >= self.vertices.shape[1]).any()
            ):
                raise ValueError("scene_collision_descriptor_invalid_topology")
            object.__setattr__(self, "faces", self.faces.detach().clone())


class SceneMeshSnapshot(dict[str, torch.Tensor]):
    """Rigid-only mapping compatibility plus ordered articulation components.

    Args:
        rigid: Already-scaled visual-envelope vertices keyed by logical identity.
        components: Ordered link-local collision-input components.
        collision_roles: Neutral planner metadata. ``none`` never removes a collider.
        unavailable: Explicit failure for incomplete articulation input capture.

    Geometry capture owns its arrays and reads no live poses. This is neither a
    cooked-shape oracle nor continuous collision detection. An empty rigid-only
    result retains the existing unavailable/empty mapping behavior.
    """

    components: tuple[SceneMeshComponent, ...]
    collision_roles: dict[str, str | None]
    pose_entity_ids: tuple[str, ...]
    unavailable: str | None
    rigid_geometry_kind: str
    rigid_provenance: str

    def __init__(
        self,
        rigid: Mapping[str, torch.Tensor],
        *,
        components: tuple[SceneMeshComponent, ...] = (),
        collision_roles: Mapping[str, str | None] | None = None,
        unavailable: str | None = None,
    ) -> None:
        self._initialize(
            {name: _owned_vertices(value) for name, value in rigid.items()},
            components,
            collision_roles,
            unavailable,
        )

    def _initialize(
        self,
        rigid: Mapping[str, torch.Tensor],
        components: tuple[SceneMeshComponent, ...],
        roles: Mapping[str, str | None] | None,
        unavailable: str | None,
    ) -> None:
        super().__init__(rigid)
        self.components = tuple(components)
        self.collision_roles = dict(roles or {})
        if any(
            value is not None and type(value) is not str
            for value in self.collision_roles.values()
        ):
            raise TypeError("Collision roles must be neutral strings or None.")
        self.pose_entity_ids = tuple(
            dict.fromkeys(c.pose_entity_id for c in components)
        )
        self.unavailable = unavailable
        self.rigid_geometry_kind = "rigid_visual_envelope"
        self.rigid_provenance = "scaled_visual_vertices"

    @classmethod
    def _from_owned(
        cls,
        rigid: Mapping[str, torch.Tensor],
        *,
        components: tuple[SceneMeshComponent, ...] = (),
        collision_roles: Mapping[str, str | None] | None = None,
        unavailable: str | None = None,
    ) -> SceneMeshSnapshot:
        # Capture already cloned each native array at its read boundary. Transfer
        # those owned values without a second full-size geometry allocation.
        result = dict.__new__(cls)
        result._initialize(rigid, components, collision_roles, unavailable)
        return result


def capture_scene_meshes(
    simulation: SimulationManager,
    *,
    rigid_uids: Mapping[str, str],
    articulation_uids: Mapping[str, str] | None = None,
    link_bindings: Sequence[tuple[str, str, str]] = (),
    collision_roles: Mapping[str, str | None] | None = None,
    rigidized_uids: Sequence[str] = (),
) -> SceneMeshSnapshot:
    """Capture geometry through explicit logical/native identities, without poses.

    Args:
        simulation: Owner of the selected native objects.
        rigid_uids: Logical rigid identity to exact simulation UID.
        articulation_uids: Logical articulation identity to simulation UID.
        link_bindings: Ordered ``(articulation_id, native_link, pose_entity_id)``
            triples. Each collision link requires exactly one pose binding.
        collision_roles: Plain string/None planner metadata, never collider filters.
        rigidized_uids: Explicit unsupported rigidized-articulation selection.

    Returns:
        Owned scaled rigid visual vertices and convex connected collision-input
        components. Articulation failures preserve captured rigid data and mark
        the result unavailable; rigid-only failures return an empty mapping.
        Default/USD materialization descriptors and the legacy USD-file route
        preserve source topology/order and do not claim cooked-shape equivalence.
    """
    articulation_uids = articulation_uids or {}
    selected_roles = dict(collision_roles or {})
    if any(
        value is not None and type(value) is not str
        for value in selected_roles.values()
    ):
        raise TypeError("Collision roles must be neutral strings or None.")
    rigid: dict[str, torch.Tensor] = {}
    components: list[SceneMeshComponent] = []
    roles: dict[str, str | None] = {}
    articulated = bool(articulation_uids or rigidized_uids)
    try:
        for entity_id, simulation_uid in rigid_uids.items():
            native = simulation.get_rigid_object(simulation_uid)
            if native is None or not callable(getattr(native, "get_vertices", None)):
                raise ValueError(f"scene_rigid_geometry_unavailable:{entity_id}")
            rigid[entity_id] = _owned_vertices(native.get_vertices(scale=True))
            roles[entity_id] = selected_roles.get(entity_id)
        if not articulated:
            return SceneMeshSnapshot._from_owned(rigid, collision_roles=roles)
        if rigidized_uids:
            raise ValueError("rigidized_articulation_collision_components_unavailable")
        from embodichain.lab.sim.objects.backends._collision_inputs import (
            _descriptor_collision_meshes,
            _usd_collision_meshes,
        )
        from pxr import Tf

        links: dict[tuple[str, str], list[str]] = {}
        for articulation_id, native_link, pose_entity_id in link_bindings:
            links.setdefault((articulation_id, native_link), []).append(pose_entity_id)
        for entity_id, simulation_uid in articulation_uids.items():
            native = simulation.get_articulation(simulation_uid)
            if native is None:
                raise ValueError(f"scene_articulation_unavailable:{entity_id}")
            roles[entity_id] = selected_roles.get(entity_id)
            try:
                meshes = _descriptor_collision_meshes(
                    simulation, native, simulation_uid
                )
                provenance = "retained_default_usd_collision_inputs"
                if meshes is None:
                    meshes = _usd_collision_meshes(
                        {"fpath": native.cfg.fpath, "body_scale": native.cfg.body_scale}
                    )
                    provenance = "usd_collision_inputs"
            except Tf.ErrorException as error:
                raise ValueError(
                    f"scene_collision_meshes_unavailable:{entity_id}:{error}"
                ) from error
            if not meshes:
                raise ValueError(f"scene_collision_meshes_unavailable:{entity_id}")
            for (link, path), mesh in meshes.items():
                ids = links.get((entity_id, link), ())
                if len(ids) != 1:
                    raise ValueError(
                        f"scene_collision_link_binding_unavailable:{entity_id}:{link}"
                    )
                panels = mesh.split(only_watertight=False)
                if not len(panels):
                    raise ValueError(f"scene_collision_component_unavailable:{path}")
                for index, panel in enumerate(panels):
                    if not panel.is_convex:
                        raise ValueError(
                            f"scene_collision_component_nonconvex_unavailable:{path}"
                        )
                    components.append(
                        SceneMeshComponent(
                            entity_id,
                            ids[0],
                            f"{path}:{index}",
                            torch.as_tensor(panel.vertices, dtype=torch.float32)[None],
                            torch.as_tensor(panel.faces, dtype=torch.long),
                            provenance=provenance,
                        )
                    )
        return SceneMeshSnapshot._from_owned(
            rigid, components=tuple(components), collision_roles=roles
        )
    except _UNAVAILABLE as error:
        if not articulated:
            return SceneMeshSnapshot._from_owned({})
        return SceneMeshSnapshot._from_owned(
            rigid, collision_roles=roles, unavailable=str(error)
        )


def iter_scene_components(
    meshes: SceneMeshSnapshot,
    observed_poses: Mapping[str, torch.Tensor],
    *,
    excluded_owner: str,
) -> Iterator[tuple[str, str, torch.Tensor, torch.Tensor]]:
    """Pair owning-frame geometry with explicit same-observation poses.

    Args:
        meshes: Owned geometry snapshot, including complete articulation inputs.
        observed_poses: Positive-confidence poses keyed by pose entity identity.
            Callers omit missing/unqualified observations; this function never
            reads simulation or Task Program state.
        excluded_owner: One explicitly omitted owner, such as a carried object.

    Yields:
        Owner, component name, local vertices and observed pose in source order.

    Raises:
        ValueError: If geometry or a required B=1 observation is unavailable.
    """
    if meshes.unavailable:
        raise ValueError(meshes.unavailable)
    components = (
        *((uid, uid, uid, value) for uid, value in meshes.items()),
        *(
            (c.owner_id, c.pose_entity_id, c.name, c.vertices)
            for c in meshes.components
        ),
    )
    for owner, pose_id, name, vertices in components:
        if owner == excluded_owner:
            continue
        pose = observed_poses.get(pose_id)
        if (
            not isinstance(pose, torch.Tensor)
            or pose.shape != (1, 4, 4)
            or not torch.isfinite(pose).all()
        ):
            raise ValueError(f"scene_pose_unavailable:{pose_id}")
        yield owner, name, vertices, pose
