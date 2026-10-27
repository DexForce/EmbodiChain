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
"""File-backed native visuals for assembly bodies; never used for collisions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.toolkits.assembly._surface import (
    _checked_surface_arrays,
    _mesh_surface_arrays,
)

if TYPE_CHECKING:
    from embodichain.toolkits.assembly.model import AssemblyBody, AssemblyGeometry

__all__: list[str] = []

_VISUAL_FORMAT_VERSION = 2


def write_visual_mesh(body: AssemblyBody, path: str | Path) -> Path:
    """Write one body-local OBJ and relative MTL from its visible surfaces.

    This preparation-only representation concatenates source triangle surfaces
    without welding, taking a convex hull, or changing the body's colliders or
    inertia. Mesh face normals preserve hard edges; primitive visuals retain
    the native SDK's authored normals and tessellation.

    Args:
        body: Source body whose geometry and physical properties stay unchanged.
        path: Destination OBJ path. Its MTL is written alongside the OBJ.

    Returns:
        Absolute OBJ path, ready for the native mesh-file loader.
    """
    target = Path(path).resolve()
    if target.suffix.lower() != ".obj":
        raise ValueError("Assembly visual mesh output must end in .obj.")
    surfaces = []
    materials: dict[tuple[float, ...], int] = {}
    for geometry in body.geometries:
        if not geometry.visible:
            continue
        points, faces, normals, normal_indices = _surface_arrays(geometry)
        rotation = Rotation.from_quat(geometry.quaternion).as_matrix()
        points = points @ rotation.T + np.asarray(geometry.position, dtype=np.float64)
        normals = normals @ rotation.T
        color = tuple(float(value) for value in geometry.rgba)
        material_id = materials.setdefault(color, len(materials))
        surfaces.append((points, faces, normals, normal_indices, material_id))
    if not surfaces:
        raise ValueError(f"Assembly body {body.name!r} has no visible surfaces.")

    target.parent.mkdir(parents=True, exist_ok=True)
    material_file = target.with_suffix(".mtl")
    with material_file.open("w", encoding="utf-8") as stream:
        for rgba, index in materials.items():
            stream.write(f"newmtl rgba_{index}\n")
            stream.write(
                "Kd " + " ".join(format(value, ".17g") for value in rgba[:3]) + "\n"
            )
            stream.write(f"d {rgba[3]:.17g}\nNs 32\nillum 2\n\n")

    with target.open("w", encoding="utf-8") as stream:
        stream.write(f"mtllib {material_file.name}\no assembly_visual\ns off\n")
        stream.write(f"# assembly_visual_version {_VISUAL_FORMAT_VERSION}\n")
        for points, _, _, _, _ in surfaces:
            for point in points:
                stream.write(
                    "v " + " ".join(format(value, ".17g") for value in point) + "\n"
                )
        for _, _, normals, _, _ in surfaces:
            for normal in normals:
                stream.write(
                    "vn " + " ".join(format(value, ".17g") for value in normal) + "\n"
                )
        # One OBJ group per RGBA gives the file importer a small number of
        # visual meshes. Disconnected convex pieces retain their own faces.
        for material_id in materials.values():
            stream.write(f"g rgba_{material_id}\nusemtl rgba_{material_id}\n")
            offset = 1
            normal_offset = 1
            for points, faces, normals, normal_indices, surface_material in surfaces:
                if surface_material == material_id:
                    for face, face_normals in zip(
                        faces + offset, normal_indices + normal_offset, strict=True
                    ):
                        stream.write(
                            "f "
                            + " ".join(
                                f"{int(vertex)}//{int(normal)}"
                                for vertex, normal in zip(
                                    face, face_normals, strict=True
                                )
                            )
                            + "\n"
                        )
                offset += len(points)
                normal_offset += len(normals)
    return target


def _surface_arrays(
    geometry: AssemblyGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read exact source meshes or the existing SDK's primitive visual surface."""
    if geometry.kind == "mesh":
        return _mesh_surface_arrays(geometry)
    else:
        from dexsim.spawn.descs import GeometryDesc
        from dexsim.spawn.adapters.common import generated_mesh

        if geometry.kind == "cylinder":
            descriptor = GeometryDesc.cylinder(*geometry.size[:2])
        elif geometry.kind == "box":
            descriptor = GeometryDesc.cube(
                tuple(2 * value for value in geometry.size[:3])
            )
        elif geometry.kind == "sphere":
            descriptor = GeometryDesc.sphere(geometry.size[0])
        else:
            raise ValueError(f"Unsupported assembly visual geometry {geometry.kind!r}.")
        points, faces, normals, _ = generated_mesh(descriptor)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return _checked_surface_arrays(points, faces, normals, name=geometry.name)
