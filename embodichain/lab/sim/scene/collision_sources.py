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

"""Explicit collision-shape providers for one observed planning world."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape
from ._validation import _owned_b1_pose, _owned_batch_pose

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "CollisionObjectSnapshot",
    "read_articulation_link_boxes",
    "snapshot_collision_sources",
    "SimulationCollisionShapeReader",
]


def _copy_shape(shape: CollisionShapeDesc) -> CollisionShapeDesc:
    return replace(
        shape,
        **{
            name: (
                None
                if (value := getattr(shape, name)) is None
                else value.detach().clone()
            )
            for name in ("local_pose", "half_extents", "vertices", "triangles")
        },
    )


@dataclass(frozen=True, slots=True)
class CollisionObjectSnapshot:
    """Owned shape descriptors and a B=1 pose, without native object handles.

    Args:
        shapes: Already-scaled owner-local shape descriptors. The provider's
            provenance determines whether these are native shapes or input proxies.
        pose: Observed owner pose in the consumer's frame, shape ``(1, 4, 4)``.
        provenance: Neutral input-description label, not a collision certificate.
    """

    shapes: tuple[CollisionShapeDesc, ...]
    pose: torch.Tensor
    provenance: str = "provided_collision_descriptors"

    def __post_init__(self) -> None:
        pose = _owned_b1_pose(self.pose, "Collision snapshot pose")
        if not all(isinstance(shape, CollisionShapeDesc) for shape in self.shapes):
            raise TypeError("Collision snapshots require CollisionShapeDesc values.")
        object.__setattr__(self, "pose", pose)
        object.__setattr__(
            self, "shapes", tuple(_copy_shape(shape) for shape in self.shapes)
        )

    def get_collision_shapes(self, env_id: int = 0) -> list[CollisionShapeDesc]:
        """Return the captured owner-local descriptors for the supported row.

        Args:
            env_id: Must be zero for the B=1 snapshot.

        Returns:
            A new list of borrowed, read-only snapshot descriptors. Their tensors
            must not be mutated; construct a new snapshot to change geometry.
        """
        if env_id != 0:
            raise IndexError("Collision snapshots support only environment zero.")
        return list(self.shapes)

    def get_local_pose(self, *, to_matrix: bool = True) -> torch.Tensor:
        """Return an owned copy of the captured matrix without querying simulation.

        Args:
            to_matrix: Must be true; this provider stores matrix poses only.

        Returns:
            Pose matrix with shape ``(1, 4, 4)``.
        """
        if not to_matrix:
            raise ValueError("Collision snapshots expose matrix poses only.")
        return self.pose.clone()


def read_articulation_link_boxes(
    source_path: str | Path,
    body_scale: Sequence[float],
) -> dict[str, tuple[CollisionShapeDesc, ...]]:
    """Read enabled USD input components as separate link-local bounding boxes.

    Args:
        source_path: Self-contained, metre-authored articulation USD file.
        body_scale: Positive uniform scale applied exactly once to input vertices.

    Returns:
        Ordered native-link mappings of component boxes. Connected components
        remain separate, preserving openings between panels. This input envelope
        does not establish equivalence to native cooked geometry or CCD.

    Raises:
        ValueError: If source geometry, topology, frames or scales are unsupported.
    """
    from embodichain.lab.sim.objects.backends._collision_inputs import (
        _usd_collision_meshes,
    )

    meshes = _usd_collision_meshes({"fpath": source_path, "body_scale": body_scale})
    by_link: dict[str, list[CollisionShapeDesc]] = {}
    for (link, _), mesh in meshes.items():
        shapes = by_link.setdefault(link, [])
        for component in mesh.split(only_watertight=False):
            low, high = component.bounds
            pose = torch.eye(4)
            pose[:3, 3] = torch.as_tensor((low + high) / 2, dtype=torch.float32)
            shapes.append(
                CollisionShapeDesc(
                    name=f"component_{len(shapes)}",
                    shape_type=RigidBodyShape.BOX,
                    local_pose=pose,
                    half_extents=torch.as_tensor(
                        (high - low) / 2, dtype=torch.float32
                    ).clamp_min(1e-5),
                )
            )
    return {link: tuple(shapes) for link, shapes in by_link.items()}


def snapshot_collision_sources(
    shapes: Mapping[str, Sequence[CollisionShapeDesc]],
    observed_poses: Mapping[str, torch.Tensor],
    world_to_frame: torch.Tensor,
    *,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, CollisionObjectSnapshot]:
    """Project explicit same-observation collision inputs into a planning frame.

    Args:
        shapes: Canonical obstacle identity to owner-local, already-scaled shapes.
            A batched capture declares identical geometry in all environments.
        observed_poses: Same-observation world poses for exactly those obstacles.
        world_to_frame: Explicit world-to-consumer transforms, shape ``(B, 4, 4)``.
            Each obstacle pose must have the same batch. Projected poses must
            agree across rows (absolute tolerance 1e-5, zero relative tolerance).
            The returned B=1 providers describe that shared world; no row is
            silently selected from different layouts. For cuRobo these transforms
            must match the query snapshot.
        provenance: Optional neutral description of each input geometry route.

    Returns:
        Owned providers accepted by ``CuroboWorldCfg.rigid_objects``. No live pose
        getter is called, no scale is reapplied, and mapping order is preserved.

    Raises:
        ValueError: If poses are invalid, batches differ, or projected worlds differ.
    """
    if set(shapes) != set(observed_poses):
        raise ValueError(
            "Collision source shapes and observed poses must have equal IDs."
        )
    world_to_frame = _owned_batch_pose(world_to_frame, "world_to_frame")
    result = {}
    for uid, values in shapes.items():
        pose = _owned_batch_pose(observed_poses[uid], f"scene_pose:{uid}")
        if pose.shape != world_to_frame.shape:
            raise ValueError("Collision pose batch must match world_to_frame.")
        projected = world_to_frame @ pose.to(world_to_frame)
        if not torch.allclose(
            projected, projected[:1].expand_as(projected), atol=1e-5, rtol=0
        ):
            raise ValueError(
                "Observed collision worlds must be identical in the planning frame."
            )
        result[uid] = CollisionObjectSnapshot(
            tuple(values),
            projected[:1],
            provenance=(provenance or {}).get(uid, "provided_collision_descriptors"),
        )
    return result


class SimulationCollisionShapeReader:
    """Read geometry for an explicit ordered roster without reading live poses.

    Args:
        simulation: Exact owner of the borrowed rigid/articulation objects.
        rigid_objects: Ordered logical obstacle ID to native rigid UID mapping.
        articulation_links: Ordered logical ID to ``(native UID, link name)``.
            Every collision link needed by the world must be declared, including
            links that no semantic call directly references.

    Rigid descriptors remain native-owner inputs; articulated links retain the
    source component-box envelope. Capture preserves declaration order and
    excludes only explicitly named IDs. It never infers a role from a UID.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        *,
        rigid_objects: Mapping[str, str],
        articulation_links: Mapping[str, tuple[str, str]],
    ) -> None:
        def identifier(value: object) -> str:
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Collision source IDs must be exact nonempty strings.")
            return value

        if not isinstance(rigid_objects, Mapping) or not isinstance(
            articulation_links, Mapping
        ):
            raise TypeError("Collision source rosters must be mappings.")
        if set(rigid_objects) & set(articulation_links):
            raise ValueError("Collision source logical IDs must be unique.")
        self._simulation = simulation
        self._rigid = {}
        self._links = {}
        self._rigid_owners = {}
        self._articulation_owners = {}
        for key, uid in rigid_objects.items():
            key, uid = identifier(key), identifier(uid)
            body = simulation.get_rigid_object(uid)
            if body is None or body.num_instances != 1:
                raise ValueError("Collision sources require bound B1 rigid objects.")
            self._rigid[key] = uid
            self._rigid_owners[uid] = body
        if len(set(self._rigid.values())) != len(self._rigid):
            raise ValueError("A native rigid object must occur only once in the world.")
        for key, selection in articulation_links.items():
            key = identifier(key)
            if type(selection) not in (tuple, list) or len(selection) != 2:
                raise ValueError(
                    "Articulation collision sources require UID/link pairs."
                )
            uid, link = map(identifier, selection)
            body = simulation.get_articulation(uid)
            if body is None or body.num_instances != 1:
                raise ValueError("Collision sources require bound B1 articulations.")
            self._links[key] = (uid, link)
            self._articulation_owners[uid] = body
        if len(set(self._links.values())) != len(self._links):
            raise ValueError(
                "A native articulation link must occur only once in the world."
            )

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Return the complete declared obstacle order."""
        return (*self._rigid, *self._links)

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return neutral geometry routes, not native-collision certificates."""
        return MappingProxyType(
            {
                **{key: "rigid_collision_descriptors" for key in self._rigid},
                **{key: "articulation_link_component_boxes" for key in self._links},
            }
        )

    def capture(
        self, *, excluded_entity_ids: tuple[str, ...] = ()
    ) -> dict[str, tuple[CollisionShapeDesc, ...]]:
        """Read fresh shapes for the selected roster, preserving source order.

        Args:
            excluded_entity_ids: Explicit carried or intentionally absent IDs.

        Returns:
            Borrowed read-only rigid descriptors and fresh owned link envelopes.
            Pass these immediately to ``snapshot_collision_sources`` to own a
            planning world; never retain mutable simulator buffers as snapshots.
        """
        if any(
            type(key) is not str or not key or key != key.strip()
            for key in excluded_entity_ids
        ):
            raise ValueError("Excluded collision IDs must be exact identifiers.")
        result = {}
        for key, uid in self._rigid.items():
            if key in excluded_entity_ids:
                continue
            body = self._simulation.get_rigid_object(uid)
            if body is not self._rigid_owners[uid] or body.num_instances != 1:
                raise ValueError(
                    "Rigid collision source ownership changed; rebind the reader."
                )
            result[key] = tuple(body.get_collision_shapes())
        by_articulation = {}
        for key, (uid, link) in self._links.items():
            if key in excluded_entity_ids:
                continue
            body = self._simulation.get_articulation(uid)
            if body is not self._articulation_owners[uid] or body.num_instances != 1:
                raise ValueError(
                    "Articulation collision source ownership changed; rebind the reader."
                )
            if uid not in by_articulation:
                by_articulation[uid] = read_articulation_link_boxes(
                    body.cfg.fpath, body.cfg.body_scale
                )
            if link not in by_articulation[uid]:
                raise ValueError(
                    f"Declared collision link {uid!r}/{link!r} has no source envelope."
                )
            result[key] = by_articulation[uid][link]
        return result
