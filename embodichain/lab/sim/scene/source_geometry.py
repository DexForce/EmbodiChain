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

__all__ = ["LinkMesh", "read_gltf_z_up_mesh", "estimate_source_mesh_top_z"]


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
