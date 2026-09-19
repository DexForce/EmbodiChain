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

"""Load assembly assets as closed solids without filling intentional cavities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

__all__ = ["load_mesh"]


def load_mesh(path: Path, scale: float = 1.0) -> trimesh.Trimesh:
    """Load and weld an asset, requiring a finite, watertight solid boundary.

    Args:
        path: Mesh asset, normally an OBJ exported by the Blender worker.
        scale: Positive conversion factor from asset units to meters.

    Returns:
        A solid mesh with consistent normals and welded texture/normal seams.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mesh scale must be finite and positive")
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"{path} contains no triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{path} contains nonfinite vertices")
    mesh.apply_scale(scale)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{path} contains nonfinite scaled vertices")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or not mesh.is_watertight:
        raise ValueError(
            f"{path} is not watertight after seam welding; repair it first"
        )
    mesh.fix_normals(multibody=True)
    if not mesh.is_volume:
        raise ValueError(f"{path} does not bound a consistently oriented solid")
    return mesh
