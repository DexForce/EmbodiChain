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

"""Shared URDF collision decoding, independent of surface sampling and FK."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

__all__: list[str] = []


def _vector(value: str, *, field: str) -> np.ndarray:
    result = np.fromstring(value, sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must contain three finite values.")
    return result


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4)
    if element is not None:
        result[:3, :3] = Rotation.from_euler(
            "xyz", _vector(element.get("rpy", "0 0 0"), field="origin rpy")
        ).as_matrix()
        result[:3, 3] = _vector(element.get("xyz", "0 0 0"), field="origin xyz")
    return result


def _load_collision_geometry(
    collision: ET.Element,
    directory: Path,
    *,
    mesh_paths: Mapping[str, str | Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """Decode scaled mesh vertices, faces, collision origin and source path.

    An authored box is triangulated exactly. The caller owns sampling, FK and
    provenance hashing; the origin is returned without transforming vertices.
    """
    mesh_node = collision.find("geometry/mesh")
    box_node = collision.find("geometry/box")
    if mesh_node is not None:
        filename = mesh_node.attrib["filename"]
        mesh_path = Path(
            (mesh_paths or {}).get(filename, directory / filename)
        ).resolve(strict=True)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        scale = _vector(mesh_node.get("scale", "1 1 1"), field="mesh scale")
    elif box_node is not None:
        size = _vector(box_node.get("size", ""), field="collision box size")
        if (size <= 0).any():
            raise ValueError("Collision box dimensions must be positive.")
        mesh = trimesh.creation.box(extents=size)
        mesh_path = None
        scale = np.ones(3)
    else:
        raise ValueError("Selected collision surfaces require URDF meshes or boxes.")
    return (
        np.asarray(mesh.vertices) * scale,
        np.asarray(mesh.faces),
        _origin_transform(collision.find("origin")),
        mesh_path,
    )
