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

"""Owned native geometry for planar placement observations."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import trimesh

from .rigid_mesh import capture_rigid_mesh

from embodichain.compute.geometry.landing import (
    planar_floor_region,
    planar_placement_metrics,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["NativePlanarPlacement"]


def _native_mesh(
    simulation: SimulationManager, uid: str, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = capture_rigid_mesh(simulation, uid, batch_size=batch_size)
    return mesh.vertices, mesh.triangles


class NativePlanarPlacement:
    """Retain model geometry for two exact native objects, never live poses.

    Recreate after replacing either object or changing its model or scale.
    Missing or unsupported geometry raises an acquisition error; it never
    produces an ordinary negative measurement or a successful task result.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        object_uid: str,
        reference_uid: str,
        *,
        batch_size: int,
    ) -> None:
        """Capture scaled native geometry without advancing the simulator.

        Args:
            simulation: Runtime owner of both exact native rigid objects.
            object_uid: Explicit payload UID; no semantic binding is inspected.
            reference_uid: Explicit reference UID.
            batch_size: Number of native environment rows to preserve.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if any(type(uid) is not str or not uid for uid in (object_uid, reference_uid)):
            raise ValueError("Object UIDs must be explicit nonempty strings.")
        vertices, _ = _native_mesh(simulation, object_uid, batch_size)
        self._vertices = torch.unique(vertices, dim=1).clone()
        reference_vertices, faces = _native_mesh(simulation, reference_uid, batch_size)
        floors = []
        for row in range(batch_size):
            floor = planar_floor_region(
                trimesh.Trimesh(reference_vertices[row].numpy(), faces[row].numpy())
            )
            if floor is None:
                raise ValueError(
                    "Placement reference has no measured planar landing region."
                )
            floors.append(floor)
        self._floors = tuple(floors)
        self._simulation = simulation
        self._object_uid, self._reference_uid = object_uid, reference_uid
        self._object = simulation.get_rigid_object(object_uid)
        self._reference = simulation.get_rigid_object(reference_uid)
        self._batch_size = batch_size

    def measure(
        self,
        object_pose: torch.Tensor,
        reference_pose: torch.Tensor,
        *,
        exact_silhouette: bool = False,
    ) -> dict[str, Any]:
        """Measure supplied observation poses against the owned native geometry.

        Args:
            object_pose: Observed CPU rigid payload poses, shape ``(B, 4, 4)``.
            reference_pose: Corresponding observed reference poses.
            exact_silhouette: Read native triangles for the exact projected union;
                False retains the original convex-hull measurement.

        Returns:
            Geometry measurements only. The caller owns tolerances and evidence
            timestamps, invocation association, availability and task acceptance.
        """
        if type(exact_silhouette) is not bool:
            raise TypeError("exact_silhouette must be bool.")
        if (
            self._object is not self._simulation.get_rigid_object(self._object_uid)
            or self._reference
            is not self._simulation.get_rigid_object(self._reference_uid)
            or object_pose.shape[0] != self._batch_size
        ):
            raise ValueError(
                "Placement geometry cache differs from its native bindings or batch."
            )
        vertices, faces, reference_vertices = self._vertices, None, None
        if exact_silhouette:
            vertices, faces = _native_mesh(
                self._simulation, self._object_uid, self._batch_size
            )
            reference_vertices, _ = _native_mesh(
                self._simulation, self._reference_uid, self._batch_size
            )
        return planar_placement_metrics(
            vertices,
            self._floors,
            object_pose,
            reference_pose,
            faces=faces,
            reference_vertices=reference_vertices,
        )
