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

"""Explicit authored URDF collision inputs without robot or simulator state."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.lab.sim.spawn.source_geometry import LinkMesh

__all__ = ["load_urdf_collision_inputs"]


def _vector(value: str, field: str, *, positive: bool = False) -> NDArray[np.float64]:
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if (
        result.shape != (3,)
        or not np.isfinite(result).all()
        or (positive and (result <= 0).any())
    ):
        raise ValueError(
            f"{field} requires three finite{' positive' if positive else ''} values."
        )
    return result


def load_urdf_collision_inputs(
    urdf_path: str | Path,
    link_names: Sequence[str],
    *,
    body_scale: Sequence[float] = (1.0, 1.0, 1.0),
    vertices_only: bool = False,
) -> dict[str, tuple[LinkMesh, ...]]:
    """Load complete selected mesh/box collision inputs in native link frames.

    Args:
        urdf_path: Existing local URDF; relative meshes resolve beside it.
        link_names: Nonempty unique exact link names, preserving requested order.
            Every selected link must own at least one explicit collider.
        body_scale: Positive XYZ robot-body scale, applied after collision scale
            and origin so both vertices and origin translation scale once.
        vertices_only: Preserve all authored vertices but return empty faces and
            an explicit vertex-envelope approximation. Surface mode validates
            nondegenerate triangle topology without changing vertex/face order.

    Returns:
        Independently owned read-only ``LinkMesh`` arrays grouped by link. Paths
        retain URDF, link, collision index and original mesh/box provenance.

    Raises:
        ValueError: Link selection, geometry, scale or local source is unsupported.
        OSError: The URDF or a referenced local mesh is unavailable.

    This captures authored inputs, not backend cooked colliders. It never uses
    visuals, expands a URI, samples/subdivides a surface, infers link names or
    accesses robot state. Runtime frame and identity qualification belong to its
    Lab consumer. Vertex-envelope mode carries no triangle-surface certificate.
    """
    path = Path(urdf_path).expanduser().resolve(strict=True)
    if isinstance(link_names, str):
        raise ValueError("link_names must be a sequence of exact identifiers.")
    names = tuple(link_names)
    scale = np.asarray(body_scale, dtype=np.float64)
    if (
        not names
        or any(
            type(name) is not str or not name or name != name.strip() for name in names
        )
        or len(set(names)) != len(names)
        or scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "URDF collision inputs require exact links and positive XYZ scale."
        )
    if type(vertices_only) is not bool:
        raise TypeError("vertices_only must be a bool.")
    document = ET.parse(path).getroot()
    result = {}
    for name in names:
        matches = [
            link for link in document.findall("link") if link.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(f"URDF collision link {name!r} must be unambiguous.")
        values = []
        for index, collision in enumerate(matches[0].findall("collision")):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("Each URDF collider requires one explicit geometry.")
            element = geometry[0]
            if element.tag == "mesh":
                source = element.get("filename", "")
                if not source or "://" in source:
                    raise ValueError(
                        "URDF collision meshes require resolved local paths."
                    )
                mesh_path = path.parent / source
                shape = trimesh.load(mesh_path, force="mesh", process=False)
                vertices = np.asarray(shape.vertices, dtype=np.float64) * _vector(
                    element.get("scale", "1 1 1"), "mesh scale", positive=True
                )
                label, approximation = str(mesh_path.resolve()), "urdfMeshInput"
            elif element.tag == "box":
                size = _vector(element.get("size", ""), "box size", positive=True)
                shape = trimesh.creation.box(extents=size)
                vertices = np.asarray(shape.vertices, dtype=np.float64)
                label = f"box:{' '.join(map(str, size))}"
                approximation = "urdfBoxInput"
            else:
                raise ValueError("URDF collision inputs support only meshes and boxes.")
            faces = np.asarray(shape.faces, dtype=np.int64)
            if (
                vertices.ndim != 2
                or vertices.shape[1:] != (3,)
                or not len(vertices)
                or not np.isfinite(vertices).all()
                or faces.ndim != 2
                or faces.shape[1:] != (3,)
                or not len(faces)
                or faces.min() < 0
                or faces.max() >= len(vertices)
            ):
                raise ValueError("URDF collision mesh has invalid source geometry.")
            if vertices_only:
                faces = np.empty((0, 3), dtype=np.int64)
                approximation = approximation.removesuffix("Input") + "VertexEnvelope"
            else:
                triangle = vertices[faces]
                area = np.linalg.norm(
                    np.cross(
                        triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
                    ),
                    axis=1,
                )
                if (area == 0).any():
                    raise ValueError("URDF collision triangle surface is degenerate.")
            origin = collision.find("origin")
            translation, rotation = np.zeros(3), np.eye(3)
            if origin is not None:
                translation = _vector(origin.get("xyz", "0 0 0"), "origin xyz")
                rotation = Rotation.from_euler(
                    "xyz", _vector(origin.get("rpy", "0 0 0"), "origin rpy")
                ).as_matrix()
            vertices = (vertices @ rotation.T + translation) * scale
            if not np.isfinite(vertices).all():
                raise ValueError("URDF scale produced nonfinite collision vertices.")
            values.append(
                LinkMesh(
                    f"{path}#{name}/collision[{index}]:{label}",
                    name,
                    vertices,
                    faces,
                    approximation,
                )
            )
        if not values:
            raise ValueError(f"URDF link {name!r} has no collision geometry.")
        result[name] = tuple(values)
    return result
