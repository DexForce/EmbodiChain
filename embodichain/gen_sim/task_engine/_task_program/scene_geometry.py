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

"""Bound rigid visual proxies and articulated collision components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

__all__: list[str] = []

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


@dataclass(frozen=True, slots=True)
class SceneComponent:
    """One connected collision component expressed in a canonical link frame."""

    owner_id: str
    pose_entity_id: str
    name: str
    vertices: torch.Tensor
    faces: torch.Tensor | None = None


class SceneVertices(dict[str, torch.Tensor]):
    """Keep the rigid-only mapping contract while retaining articulated panels."""

    def __init__(
        self,
        rigid: Mapping[str, torch.Tensor],
        *,
        components: tuple[SceneComponent, ...] = (),
        collision_roles: Mapping[str, Any] | None = None,
        unavailable: str | None = None,
    ) -> None:
        super().__init__(rigid)
        self.components = components
        self.collision_roles = dict(collision_roles or {})
        self.pose_entity_ids = tuple(
            dict.fromkeys(c.pose_entity_id for c in components)
        )
        self.unavailable = unavailable


def _vertices(value: Any) -> torch.Tensor:
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


def capture_scene_vertices(simulation: Any, bindings: Any) -> dict[str, torch.Tensor]:
    """Capture geometry only; never freeze or independently read live link poses.

    Rigid vertices retain the existing visual-envelope contract. They are not
    physical collision geometry and must not stand in for a native shape oracle.
    Collision roles classify the planner, not whether a native collider exists.
    In particular ``NONE`` cannot omit a physical collider from this safety gate.
    Retained Default/USD descriptors preserve the native convex approximation
    and SDK part preprocessing; the legacy USD adapter remains for other routes.
    Parts use owning-link frames, separate from freshly observed link poses.
    This is materialization-input geometry, not a cooked-shape/contact oracle.
    """
    rigid: dict[str, torch.Tensor] = {}
    components: list[SceneComponent] = []
    roles: dict[str, Any] = {}
    articulated = bool(
        getattr(bindings, "articulations", ())
        or getattr(bindings, "rigidized_articulations", ())
    )
    try:
        for binding in getattr(bindings, "rigid_objects", ()):
            native = simulation.get_rigid_object(binding.simulation_uid)
            if native is None or not callable(getattr(native, "get_vertices", None)):
                raise ValueError(
                    f"scene_rigid_geometry_unavailable:{binding.entity_id}"
                )
            rigid[binding.entity_id] = _vertices(native.get_vertices(scale=True))
            roles[binding.entity_id] = getattr(binding, "collision_role", None)
        if not articulated:
            return rigid
        if getattr(bindings, "rigidized_articulations", ()):
            raise ValueError("rigidized_articulation_collision_components_unavailable")
        from .collision_descriptors import descriptor_collision_meshes
        from .drawer_geometry import collision_meshes
        from pxr import Tf

        links: dict[tuple[str, str], list[str]] = {}
        for binding in getattr(bindings, "links", ()):
            links.setdefault(
                (binding.articulation_id, binding.native_link_name), []
            ).append(binding.entity_id)
        for binding in bindings.articulations:
            native = simulation.get_articulation(binding.simulation_uid)
            if native is None:
                raise ValueError(f"scene_articulation_unavailable:{binding.entity_id}")
            roles[binding.entity_id] = getattr(binding, "collision_role", None)
            try:
                meshes = descriptor_collision_meshes(
                    simulation, native, binding.simulation_uid
                )
                if meshes is None:
                    meshes = collision_meshes(
                        {"fpath": native.cfg.fpath, "body_scale": native.cfg.body_scale}
                    )
            except Tf.ErrorException as error:
                raise ValueError(
                    f"scene_collision_meshes_unavailable:{binding.entity_id}:{error}"
                ) from error
            if not meshes:
                raise ValueError(
                    f"scene_collision_meshes_unavailable:{binding.entity_id}"
                )
            for (link, path), mesh in meshes.items():
                ids = links.get((binding.entity_id, link), ())
                if len(ids) != 1:
                    raise ValueError(
                        f"scene_collision_link_binding_unavailable:{binding.entity_id}:{link}"
                    )
                panels = mesh.split(only_watertight=False)
                if not len(panels):
                    raise ValueError(f"scene_collision_component_unavailable:{path}")
                for index, panel in enumerate(panels):
                    # A connected shell's AABB would fill its interior cavity.
                    # Only convex components have a conservative solid envelope.
                    if not panel.is_convex:
                        raise ValueError(
                            f"scene_collision_component_nonconvex_unavailable:{path}"
                        )
                    components.append(
                        SceneComponent(
                            binding.entity_id,
                            ids[0],
                            f"{path}:{index}",
                            _vertices(
                                torch.as_tensor(panel.vertices, dtype=torch.float32)[
                                    None
                                ]
                            ),
                            torch.as_tensor(panel.faces, dtype=torch.long).clone(),
                        )
                    )
        return SceneVertices(rigid, components=tuple(components), collision_roles=roles)
    except _UNAVAILABLE as error:
        if not articulated:
            return {}
        return SceneVertices(rigid, collision_roles=roles, unavailable=str(error))


def scene_components(
    meshes: Mapping[str, torch.Tensor], context: Any, *, excluded_owner: str
) -> Iterator[tuple[str, str, torch.Tensor, torch.Tensor]]:
    """Yield complete owner/component/local-vertices/observed-pose snapshots."""
    unavailable = getattr(meshes, "unavailable", None)
    if unavailable:
        raise ValueError(unavailable)
    components = (
        *(SceneComponent(uid, uid, uid, value) for uid, value in meshes.items()),
        *getattr(meshes, "components", ()),
    )
    for component in components:
        if component.owner_id == excluded_owner:
            continue
        entity = context.scene.entities.get(component.pose_entity_id)
        if entity is None or getattr(entity, "confidence", 1.0) <= 0:
            raise ValueError(f"scene_pose_unavailable:{component.pose_entity_id}")
        pose = entity.pose
        if (
            not isinstance(pose, torch.Tensor)
            or pose.shape != (1, 4, 4)
            or not torch.isfinite(pose).all()
        ):
            raise ValueError(f"scene_pose_unavailable:{component.pose_entity_id}")
        yield component.owner_id, component.name, component.vertices, pose


def world_bounds(
    mesh: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        mesh.ndim != 3
        or mesh.shape[0] != 1
        or mesh.shape[-1] != 3
        or not mesh.shape[1]
        or not torch.isfinite(mesh).all()
        or pose.shape != (1, 4, 4)
        or not torch.isfinite(pose).all()
    ):
        raise ValueError("scene_geometry_or_pose_unavailable")
    world = mesh.to(pose) @ pose[:, :3, :3].transpose(-1, -2) + pose[:, None, :3, 3]
    return world.amin(1), world.amax(1)
