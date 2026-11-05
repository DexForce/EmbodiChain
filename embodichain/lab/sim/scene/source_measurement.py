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

"""Source envelope measurements independent of Task Engine and live simulation."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

__all__ = ["measure_source_geometry"]
_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def measure_source_geometry(
    entry: Mapping[str, Any],
    *,
    convert_y_up: bool,
) -> dict[str, Any] | None:
    """Measure original authored scene bounds and local extents without live state.

    Args:
        entry: Physical mesh/primitive/articulation source declaration.
        convert_y_up: Apply the source mesh basis conversion. GLTF inputs always
            retain their declared Y-up convention; primitive sizes are runtime XYZ.

    Returns:
        Owned bounds/extents/axis metadata, or None for unavailable source shape
        kinds. This is an authored envelope, not a cooked collision certificate.
    """
    shape = entry.get("shape")
    if not isinstance(shape, Mapping) and entry.get("fpath"):
        from .articulation_geometry import read_articulation_geometry

        path = Path(str(entry["fpath"]))
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            return None
        geometry = read_articulation_geometry(path)
        vertices = geometry.world_vertices(entry)
        scale = np.asarray(_vector(entry.get("body_scale"), default=(1, 1, 1)))
        return {
            "bounds": np.stack((vertices.min(0), vertices.max(0))),
            "local_extents": np.ptp(geometry.vertices * scale, axis=0),
            "axis_transform": Rotation.from_euler(
                "XYZ", _vector(entry.get("init_rot"), default=(0, 0, 0)), degrees=True
            ).as_matrix(),
            "shape_type": "Articulation",
        }
    if not isinstance(shape, Mapping):
        return None
    shape_type = str(shape.get("shape_type", ""))
    if shape_type == "Mesh":
        path = Path(str(shape.get("fpath", ""))).expanduser().resolve()
        if not path.is_file():
            return None
        loaded = trimesh.load(path, force="scene")
        mesh = loaded.to_geometry()
        convert_y_up = convert_y_up or path.suffix.lower() in {".glb", ".gltf"}
    elif shape_type == "Cube":
        convert_y_up = False
        mesh = trimesh.creation.box(
            extents=_vector(shape.get("size"), default=(1, 1, 1))
        )
    elif shape_type == "Sphere":
        convert_y_up = False
        radius = float(shape.get("radius", 1.0))
        mesh = trimesh.creation.icosphere(radius=radius)
    else:
        return None
    scale = np.asarray(_vector(entry.get("body_scale"), default=(1, 1, 1)))
    conversion = _Y_UP_TO_Z_UP if convert_y_up else np.eye(3)
    basis = np.eye(4)
    basis[:3, :3] = conversion
    mesh.apply_transform(basis)
    mesh.apply_scale(scale)
    local_extents = np.asarray(mesh.extents, dtype=float)
    rotation = Rotation.from_euler(
        "XYZ",
        _vector(entry.get("init_rot"), default=(0, 0, 0)),
        degrees=True,
    ).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = _vector(entry.get("init_pos"), default=(0, 0, 0))
    mesh.apply_transform(transform)
    return {
        "bounds": np.asarray(mesh.bounds, dtype=float),
        "local_extents": local_extents,
        "axis_transform": transform[:3, :3],
        "shape_type": shape_type,
    }


def _vector(value: Any, *, default: tuple[float, float, float]) -> list[float]:
    raw = default if value is None else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("Scene geometry vectors must be sequences.")
    result = [float(item) for item in raw]
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("Scene geometry vectors must contain three finite values.")
    return result
