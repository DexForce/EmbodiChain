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

"""Owned batched rigid mesh inputs for geometric planning and observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["RigidMeshSnapshot", "capture_rigid_mesh"]


def _validate_mesh_metadata(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    *,
    expected_batch_size: int | None = None,
) -> None:
    """Reject invalid layouts before transferring or copying mesh buffers."""
    if not isinstance(vertices, torch.Tensor) or not isinstance(
        triangles, torch.Tensor
    ):
        raise TypeError("Rigid mesh inputs must be tensors.")
    if (
        vertices.ndim != 3
        or vertices.shape[-1] != 3
        or not vertices.shape[0]
        or not vertices.shape[1]
        or not vertices.is_floating_point()
    ):
        raise ValueError(
            "Rigid mesh vertices require finite floating (B, V, 3) coordinates."
        )
    if (
        triangles.ndim != 3
        or triangles.shape[0] != vertices.shape[0]
        or triangles.shape[-1] != 3
        or not triangles.shape[1]
        or triangles.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Rigid mesh triangles must index their own environment's vertices."
        )
    if expected_batch_size is not None and vertices.shape[0] != expected_batch_size:
        raise ValueError("Rigid mesh capture differs from the expected observed batch.")


@dataclass(frozen=True, slots=True, eq=False)
class RigidMeshSnapshot:
    """Own scaled local vertices and their exact indexed triangle topology.

    Args:
        source_uid: Exact source identity supplied by the capture owner.
        vertices: Scaled object-local coordinates, shape ``(B, V, 3)``.
        triangles: Matching integer triangle indices, shape ``(B, F, 3)``.

    Arrays are detached CPU snapshots. Construction preserves order, duplicate
    vertices, winding and environment rows; it performs no mesh processing or
    pose read. These are input meshes, not cooked collision/contact evidence.
    Manual construction cannot establish native provenance. Treat the exposed
    tensors as read-only; snapshot again before retaining caller-owned values.
    """

    source_uid: str
    vertices: torch.Tensor
    triangles: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.source_uid) is not str
            or not self.source_uid
            or self.source_uid != self.source_uid.strip()
        ):
            raise ValueError("Rigid mesh source_uid must be an exact identifier.")
        _validate_mesh_metadata(self.vertices, self.triangles)
        vertices = self.vertices.detach().cpu().clone()
        triangles = self.triangles.detach().cpu().clone()
        # Retained buffers still validate independently of mutable input tensors.
        _validate_mesh_metadata(vertices, triangles)
        if not torch.isfinite(vertices).all():
            raise ValueError(
                "Rigid mesh vertices require finite floating (B, V, 3) coordinates."
            )
        if (triangles < 0).any() or (triangles >= vertices.shape[1]).any():
            raise ValueError(
                "Rigid mesh triangles must index their own environment's vertices."
            )
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)

    @property
    def batch_size(self) -> int:
        """Return the preserved native environment-axis length."""
        return self.vertices.shape[0]

    def snapshot(self) -> RigidMeshSnapshot:
        """Revalidate and independently own a snapshot, including mutable tensors."""
        return RigidMeshSnapshot(self.source_uid, self.vertices, self.triangles)


def capture_rigid_mesh(
    simulation: SimulationManager, uid: str, *, batch_size: int
) -> RigidMeshSnapshot:
    """Capture semantic vertices and matching triangles without reading poses.

    Independent semantic geometry preserves placement inputs across appearance
    replacement. Objects without that API retain their source visual-mesh route.

    Args:
        simulation: Owner of the exact native rigid object.
        uid: Explicit runtime UID; no name/role inference is performed.
        batch_size: Expected full native environment count. Partial-row capture
            is not inferred and cannot merge vertices across environments.

    Returns:
        An owned CPU input mesh whose vertices already include native body scale.
        Callers supply separately observed poses and recapture after model/scale
        changes. No inferred hull or AABB substitutes for missing topology.

    Raises:
        ValueError: The source, batch or complete finite topology is unavailable.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Rigid mesh batch_size must be a positive integer.")
    if type(uid) is not str or not uid or uid != uid.strip():
        raise ValueError("Rigid mesh UID must be an exact identifier.")
    native = simulation.get_rigid_object(uid)
    if native is None:
        raise ValueError("Rigid mesh capture requires its exact bound native object.")
    semantic_mesh = getattr(native, "get_semantic_mesh", None)
    if callable(semantic_mesh):
        vertices, triangles = semantic_mesh()
    else:
        vertices, triangles = native.get_vertices(scale=True), native.get_triangles()
    _validate_mesh_metadata(vertices, triangles, expected_batch_size=batch_size)
    return RigidMeshSnapshot(uid, vertices, triangles)
