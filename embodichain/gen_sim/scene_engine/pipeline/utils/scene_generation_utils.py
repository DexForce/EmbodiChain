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


from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def quaternion_wxyz_to_euler_xyz_degrees(
    quaternion_wxyz: Sequence[float],
) -> list[float]:
    """Convert a ``[w, x, y, z]`` quaternion to [roll_x, pitch_y, yaw_z] degrees."""
    if len(quaternion_wxyz) != 4:
        raise ValueError("Rotation quaternion must contain exactly four values.")

    w, x, y, z = quaternion_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True).tolist()


def layout_object_to_transform_matrix(
    layout_object: dict[str, object],
) -> np.ndarray:
    """Return the matrix that maps an object's local coordinates to world coordinates."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = Rotation.from_euler(
        "xyz",
        _three_floats(layout_object.get("rot"), field_name="rot"),
        degrees=True,
    ).as_matrix() @ np.diag(
        _three_floats(layout_object.get("scale"), field_name="scale")
    )
    transform_matrix[:3, 3] = _three_floats(layout_object.get("pos"), field_name="pos")
    return transform_matrix


def transform_matrix_to_layout_object(
    object_id: str,
    transform_matrix: np.ndarray,
) -> dict[str, object]:
    """Convert a non-sheared 4x4 transform matrix into one layout object."""
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Layout object id must be a non-empty string.")
    matrix = np.asarray(transform_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Transform matrix must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Transform matrix must be affine.")

    linear_matrix = matrix[:3, :3]
    scale = np.linalg.norm(linear_matrix, axis=0)
    if np.any(scale <= 1e-8):
        raise ValueError("Transform matrix has a zero scale axis.")
    rotation_matrix = linear_matrix / scale
    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-6):
        raise ValueError("Transform matrix contains shear and cannot be decomposed.")
    if np.linalg.det(rotation_matrix) <= 0:
        raise ValueError(
            "Transform matrix contains a reflection and cannot be decomposed."
        )

    return {
        "id": object_id,
        "rot": Rotation.from_matrix(rotation_matrix)
        .as_euler("xyz", degrees=True)
        .tolist(),
        "pos": matrix[:3, 3].tolist(),
        "scale": scale.tolist(),
    }


def load_glb_mesh(glb_path: str | Path) -> trimesh.Trimesh:
    """Load one GLB as a single trimesh mesh."""
    resolved_glb_path = Path(glb_path).expanduser().resolve()
    if not resolved_glb_path.is_file():
        raise FileNotFoundError(f"GLB geometry not found: {resolved_glb_path}")
    loaded_mesh = trimesh.load(resolved_glb_path, process=False)
    if isinstance(loaded_mesh, trimesh.Scene):
        return loaded_mesh.dump(concatenate=True)
    if isinstance(loaded_mesh, trimesh.Trimesh):
        return loaded_mesh
    raise ValueError(f"GLB geometry is not a mesh: {resolved_glb_path}")


def _three_floats(value: object, *, field_name: str) -> list[float]:

    # Validate whether the value is a list of three numeric values.
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Coarse layout field {field_name} must contain three values.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Coarse layout field {field_name} must contain numeric values."
        ) from exc
