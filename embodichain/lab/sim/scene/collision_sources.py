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

import torch

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape
from ._validation import _owned_b1_pose

__all__ = [
    "CollisionObjectSnapshot",
    "read_articulation_link_boxes",
    "snapshot_collision_sources",
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
        observed_poses: Same-observation world poses for exactly those obstacles.
        world_to_frame: Explicit world-to-consumer transform, shape ``(1, 4, 4)``.
            For cuRobo this must match the transform used for goals and payload.
        provenance: Optional neutral description of each input geometry route.

    Returns:
        Owned providers accepted by ``CuroboWorldCfg.rigid_objects``. No live pose
        getter is called, no scale is reapplied, and mapping order is preserved.

    Raises:
        ValueError: If pose coverage or the B=1 transform is invalid.
    """
    if set(shapes) != set(observed_poses):
        raise ValueError(
            "Collision source shapes and observed poses must have equal IDs."
        )
    world_to_frame = _owned_b1_pose(world_to_frame, "world_to_frame")
    result = {}
    for uid, values in shapes.items():
        pose = _owned_b1_pose(observed_poses[uid], f"scene_pose:{uid}")
        result[uid] = CollisionObjectSnapshot(
            tuple(values),
            world_to_frame @ pose.to(world_to_frame),
            provenance=(provenance or {}).get(uid, "provided_collision_descriptors"),
        )
    return result
