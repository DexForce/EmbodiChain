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
"""Authored USD geometry shared by object adapters and static Scene readers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["LinkMesh", "UsdSourceSnapshot", "read_usd_source"]


@dataclass(frozen=True, slots=True)
class LinkMesh:
    """Owned link-local input geometry, preserving source vertex and face order.

    Args:
        path: Authored collider identity, within its source asset.
        owner_path: Owning link identity; vertices use this link's frame.
        vertices: Finite real ``(V, 3)`` coordinates with source dtype preserved.
        faces: Integer ``(F, 3)`` vertex indices. Empty faces explicitly mean
            vertex-only input and cannot establish a triangle surface.
        approximation: Authored collision approximation declaration.

    Arrays are copied once into immutable bytes and exposed read-only. This
    value contains no native cooked shapes, runtime transforms, support evidence
    or collision results.
    """

    path: str
    owner_path: str
    vertices: np.ndarray
    faces: np.ndarray
    approximation: str

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.dtype.kind not in "fiu"
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Link mesh vertices must be finite (V, 3) inputs.")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or len(faces)
            and (faces.min() < 0 or faces.max() >= len(vertices))
        ):
            raise ValueError("Link mesh faces must index the captured input vertices.")
        for name, array in (("vertices", vertices), ("faces", faces)):
            owned = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
                array.shape
            )
            object.__setattr__(self, name, owned)

    def __deepcopy__(self, memo: dict[int, Any]) -> LinkMesh:
        copied = LinkMesh(
            self.path, self.owner_path, self.vertices, self.faces, self.approximation
        )
        memo[id(self)] = copied
        return copied


@dataclass(frozen=True, slots=True)
class UsdSourceSnapshot:
    """One fresh static USD read and its source-byte identity.

    Args:
        path: Resolved source file address.
        raw: Captured source bytes.
        stage: Source USD stage; consumers must not edit this inspection input.

    A snapshot may be explicitly shared within one assembly. It owns no native
    object or simulator. Authored inputs do not certify cooked collision shapes.
    Consumers retain their own axes, units and physical qualification rules.
    """

    path: Path
    raw: bytes = field(repr=False)
    stage: Any = field(repr=False, compare=False)
    source_sha256: str = field(init=False)
    _xforms: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from pxr import UsdGeom

        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "source_sha256", hashlib.sha256(self.raw).hexdigest())
        object.__setattr__(self, "_xforms", UsdGeom.XformCache())

    def assert_current(self, path: str | Path | None = None) -> None:
        """Reject changed source selection or bytes without reparsing USD.

        Args:
            path: Optional current declaration address to compare.
        """
        if path is not None and Path(path).resolve() != self.path:
            raise ValueError("USD source selection changed during inspection.")
        if self.path.read_bytes() != self.raw:
            raise ValueError("USD source changed during inspection.")

    def require_geometry(self) -> None:
        """Require the existing metre-authored, self-contained geometry subset.

        Metadata inspection can first distinguish missing control annotations
        from invalid declared controls. Geometry consumers call this before
        interpreting source coordinates as metres.
        """
        from pxr import UsdGeom

        if UsdGeom.GetStageMetersPerUnit(self.stage) != 1.0:
            raise ValueError("USD source requires metre-authored geometry.")
        layer = self.stage.GetRootLayer()
        if (
            layer.subLayerPaths
            or layer.GetExternalReferences()
            or any(
                item not in (layer, self.stage.GetSessionLayer())
                for item in self.stage.GetUsedLayers()
            )
        ):
            raise ValueError("USD source requires a self-contained layer.")

    def mesh_arrays(self, prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read finite polygon inputs without changing source order.

        Args:
            prim_path: Exact source mesh path.

        Returns:
            Float64 points, polygon sizes and flattened integer vertex indices.
            Strict surface qualification remains with the consumer.
        """
        from pxr import UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError("USD source requires an exact mesh path.")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
            or counts.ndim != 1
            or not len(counts)
            or (counts < 3).any()
            or indices.ndim != 1
            or not len(indices)
            or counts.sum() != len(indices)
            or indices.min() < 0
            or indices.max() >= len(points)
        ):
            raise ValueError("USD source mesh has invalid topology.")
        return points, counts, indices

    def rigid_owner(self, prim_path: str) -> str:
        """Resolve the nearest authored rigid body, including the prim itself.

        Args:
            prim_path: Exact source prim path.

        Returns:
            The owning body's absolute source path.
        """
        from pxr import UsdPhysics

        owner = self.stage.GetPrimAtPath(prim_path)
        while owner and not owner.IsPseudoRoot():
            if owner.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(owner.GetPath())
            owner = owner.GetParent()
        raise ValueError("USD source mesh has no rigid-body owner.")

    def link_mesh_arrays(
        self,
        prim_path: str,
        owner_path: str,
        *,
        scale: float,
        rigid_tolerance: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bake fan-triangulated inputs into their owning rigid-link frame.

        Args:
            prim_path: Exact source collider path.
            owner_path: Its owning rigid-body path.
            scale: Positive uniform deployment scale.
            rigid_tolerance: Consumer tolerance for unscaled link frames.

        Returns:
            Newly allocated scaled vertices and faces without backend cooking.
        """
        from pxr import UsdGeom, UsdPhysics

        self.require_geometry()
        vertices, counts, indices = self.mesh_arrays(prim_path)
        prim = self.stage.GetPrimAtPath(prim_path)
        body = self.stage.GetPrimAtPath(owner_path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("USD source mesh requires a rigid-body owner.")
        if self.rigid_owner(prim_path) != str(body.GetPath()):
            raise ValueError(
                "USD source collider does not belong to its selected link."
            )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("USD source mesh scale must be finite and positive.")
        cache = self._xforms
        pose = np.asarray(cache.GetLocalToWorldTransform(body)).T
        if not np.allclose(
            pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=rigid_tolerance
        ):
            raise ValueError(
                "USD source link frames must be unscaled rigid transforms."
            )
        transform = (
            np.linalg.inv(pose) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        )
        vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) * scale
        faces, offset = [], 0
        for count in counts:
            polygon = indices[offset : offset + count]
            faces.extend(
                (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1)
            )
            offset += count
        return vertices, np.asarray(faces)

    def link_mesh(self, prim_path: str, owner_path: str, *, scale: float) -> LinkMesh:
        """Capture owned immutable link-local authored geometry.

        Args:
            prim_path: Exact source collider path.
            owner_path: Owning rigid-body path.
            scale: Uniform deployment scale, baked once into the vertices.

        Returns:
            An owned source mesh, distinct from native cooked shapes.
        """
        from pxr import UsdPhysics

        vertices, faces = self.link_mesh_arrays(prim_path, owner_path, scale=scale)
        prim = self.stage.GetPrimAtPath(prim_path)
        return LinkMesh(
            prim_path,
            owner_path,
            vertices,
            faces,
            str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()),
        )


def read_usd_source(path: str | Path | UsdSourceSnapshot) -> UsdSourceSnapshot:
    """Read fresh USD metadata and source bytes, or reuse a checked read.

    Args:
        path: USD/USDA/USDC file or an explicit snapshot from this assembly.

    Returns:
        A source snapshot loaded anonymously to avoid stale global USD layers.
        Source bytes are rechecked before returning. Call ``require_geometry``
        before interpreting geometry as self-contained metre inputs. No simulator
        is created.
    """
    from pxr import Sdf, Usd

    if isinstance(path, UsdSourceSnapshot):
        path.assert_current()
        return path
    source = Path(path).resolve()
    if source.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("USD source requires a self-contained USD asset.")
    raw = source.read_bytes()
    layer = Sdf.Layer.OpenAsAnonymous(str(source))
    stage = Usd.Stage.Open(layer) if layer is not None else None
    if stage is None:
        raise ValueError("USD source cannot be read.")
    result = UsdSourceSnapshot(source, raw, stage)
    result.assert_current()
    return result
