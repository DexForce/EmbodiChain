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
"""Scene import surface for shared authored collider inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from trimesh import Trimesh

from embodichain.lab.sim.spawn.source_geometry import LinkMesh

__all__ = [
    "LinkMesh",
    "read_gltf_z_up_mesh",
    "estimate_source_mesh_top_z",
    "read_source_vertices",
    "source_rotation",
    "source_position",
    "read_source_world_mesh",
    "source_local_bounds",
    "source_longest_local_axis",
]


def read_gltf_z_up_mesh(
    path: str | Path, *, body_scale: Sequence[float] = (1.0, 1.0, 1.0)
) -> Trimesh:
    """Read authored scene geometry in the source-compatible scaled Z-up basis.

    Scene-node transforms and face order are retained through Trimesh scene
    concatenation. Conversion is explicitly glTF Y-up to object-local Z-up,
    followed by one body scale. Finite signed scales retain the source numerical
    contract; this reader does not qualify a physical configuration or cooked
    collision geometry. No Robot, simulator or planner is constructed.
    """
    import trimesh

    try:
        loaded = trimesh.load(str(path), force="scene")
        geometry = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    except Exception as error:
        raise ValueError(
            f"Could not inspect source mesh {str(path)!r}: {error}"
        ) from error
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not vertices.size:
        raise ValueError("Source mesh requires nonempty (V, 3) vertices.")
    scale = np.asarray(body_scale, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all():
        raise ValueError("Source mesh body_scale must be a finite three-vector.")
    geometry.vertices = (
        np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1])) * scale
    )
    return geometry


def estimate_source_mesh_top_z(config: Mapping[str, Any]) -> float | None:
    """Estimate the world top of an authored mesh; unavailable geometry stays None."""
    shape = config.get("shape", {})
    if not isinstance(shape, Mapping) or not shape.get("fpath"):
        return None
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation

        sim_vertices = np.asarray(
            read_gltf_z_up_mesh(
                shape["fpath"], body_scale=config.get("body_scale", [1.0, 1.0, 1.0])
            ).vertices,
            dtype=np.float64,
        )
        rotated = Rotation.from_euler(
            "XYZ", config.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).apply(sim_vertices)
        rotated += np.asarray(config.get("init_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        return float(rotated[:, 2].max())
    except Exception:
        # Mesh bounds improve robot placement but are not needed to preserve the
        # exported scene. The robot builder has a conservative tabletop fallback.
        return None


def read_source_vertices(
    source: Mapping[str, Any],
    *,
    articulation: bool = False,
    local_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Read scaled source vertices; an explicit local override already owns scale.

    Args:
        source: Authored physical pose, scale and source shape/asset declaration.
        articulation: Select the fixed articulation source reader.
        local_vertices: Already-scaled selected-link source vertices, when supplied.

    Returns:
        Local Z-up source vertices without reading native state.
    """
    if local_vertices is not None:
        return np.asarray(local_vertices)
    if articulation:
        from .articulation_geometry import read_articulation_geometry

        path = source.get("fpath")
        if not path:
            raise ValueError("Articulated reference has no native USD asset.")
        scale = np.asarray(source.get("body_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(
                "Articulated reference requires a finite positive body_scale."
            )
        return read_articulation_geometry(path).vertices * scale
    shape = source.get("shape")
    if not isinstance(shape, Mapping) or not shape.get("fpath"):
        raise ValueError("Source object has no mesh geometry.")
    return np.asarray(
        read_gltf_z_up_mesh(
            shape["fpath"], body_scale=source.get("body_scale", [1.0] * 3)
        ).vertices,
        dtype=np.float64,
    )


def source_rotation(source: Mapping[str, Any]) -> np.ndarray:
    """Read matrix-authoritative or intrinsic-XYZ declared orientation.

    Args:
        source: Physical declaration with an optional pose matrix or Euler angles.

    Returns:
        Local-to-world 3x3 orientation, without observing native state.
    """
    from scipy.spatial.transform import Rotation

    if source.get("init_local_pose") is not None:
        rotation = np.asarray(source["init_local_pose"], dtype=float)[:3, :3]
    else:
        rotation = Rotation.from_euler(
            "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix()
    return rotation


def source_position(source: Mapping[str, Any]) -> tuple[float, float, float]:
    """Read one explicit finite source position.

    Args:
        source: Physical declaration with a three-value ``init_pos`` list.

    Returns:
        Three world-frame coordinates; no missing position is inferred.
    """
    value = source.get("init_pos")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"Scene object {source.get('uid')!r} has no three-value init_pos."
        )
    result = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"Scene object {source.get('uid')!r} has invalid init_pos.")
    return result


def source_local_bounds(
    source: Mapping[str, Any],
) -> tuple[list[float], list[float]] | None:
    """Read authored local bounds before body scale or Z-up conversion.

    Args:
        source: Physical Cube or mesh shape declaration.

    Returns:
        Original local minimum/maximum bounds, or None when unavailable.
        These raw dimensions retain source recipe calibration semantics.
    """
    shape = source.get("shape", {})
    if not isinstance(shape, dict):
        return None
    if shape.get("shape_type") == "Cube":
        size = shape.get("size")
        if isinstance(size, list) and len(size) == 3:
            extents = [float(value) for value in size]
            if min(extents) <= 0.0:
                return None
            return (
                [-value / 2.0 for value in extents],
                [value / 2.0 for value in extents],
            )
    path = shape.get("fpath")
    if not path:
        return None
    try:
        import trimesh

        geometry = trimesh.load(str(path), force="scene").to_geometry()
        bounds = [[float(value) for value in row] for row in geometry.bounds]
    except (OSError, TypeError, ValueError):
        return None
    if (
        len(bounds) != 2
        or any(len(row) != 3 for row in bounds)
        or any(maximum <= minimum for minimum, maximum in zip(*bounds, strict=True))
    ):
        return None
    return bounds[0], bounds[1]


def read_source_world_mesh(source: Mapping[str, Any]) -> Trimesh:
    """Read authored table triangles with their original topology and world pose.

    Args:
        source: Physical Cube or mesh declaration with explicit pose and scale.

    Returns:
        Owned world-frame triangles, not a native cooked collision query.
    """
    import trimesh
    from .source_envelopes import source_world_vertices

    points = source_world_vertices(dict(source))
    if source["shape"].get("shape_type") == "Cube":
        mesh = trimesh.creation.box(extents=source["shape"]["size"])
    else:
        mesh = trimesh.load(str(source["shape"]["fpath"]), force="scene").to_geometry()
    mesh.vertices = points
    return mesh


def source_longest_local_axis(source: Mapping[str, Any]) -> list[float]:
    """Map the longest authored source axis into the imported object's basis.

    Args:
        source: Physical shape declaration; raw local bounds own axis selection.

    Returns:
        Signed local basis vector, preserving glTF Y-up conversion and the
        original fallback when source dimensions are unavailable.
    """
    bounds = source_local_bounds(source)
    if bounds is None:
        axis = [0.0, 0.0, 1.0]
    else:
        extents = [maximum - minimum for minimum, maximum in zip(*bounds, strict=True)]
        axis = [0.0, 0.0, 0.0]
        axis[max(range(3), key=extents.__getitem__)] = 1.0
    shape = source.get("shape", {})
    path = shape.get("fpath") if isinstance(shape, dict) else None
    if path and Path(str(path)).suffix.lower() in {".glb", ".gltf"}:
        x_axis, y_axis, z_axis = axis
        return [x_axis, -z_axis, y_axis]
    return axis
