# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import os
import dexsim.engine
import numpy as np
import open3d as o3d
import trimesh
import dexsim

from typing import Tuple, List, Dict, Any, Optional, Union

from embodichain.utils import logger


def export_articulation_mesh(
    articulation: Union[dexsim.engine.Articulation, list],
    output_path: str = "./articulation.obj",
    link_names: Optional[Union[List[str], Dict[Any, List[str]]]] = None,
    base_xpos: Optional[np.ndarray] = None,
    base_link_name: Optional[str] = None,
    **kwargs: Any,
) -> o3d.geometry.TriangleMesh:
    r"""Export a combined mesh from all links of one or more articulations to a mesh file format.

    This function retrieves the link geometries and poses from the given articulation(s),
    transforms each link mesh to its world pose, merges them into a single mesh, and
    exports the result to the specified file path. The export format is inferred from
    the file extension (e.g., .obj, .ply, .stl, .glb, .gltf).

    Args:
        articulation (dexsim.engine.Articulation or list): The articulation object or list of articulations.
        output_path (str): The output file path including the file name and extension.
                           Supported extensions: .obj, .ply, .stl, .glb, .gltf.
        link_names (list[str] or dict[Any, list[str]], optional):
            Specify which links to export. If None, export all links.
        base_xpos (np.ndarray, optional): 4x4 homogeneous transformation matrix.
            All meshes will be transformed into this base pose coordinate system.
        base_link_name (str, optional): If specified, use the pose of this link as the base pose.
            The link will be searched from all link_names of all articulations.

    Returns:
        o3d.geometry.TriangleMesh: The combined Open3D mesh object of all articulations.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    combined_mesh = o3d.geometry.TriangleMesh()
    articulations = (
        articulation if isinstance(articulation, (list, tuple)) else [articulation]
    )

    # Determine base transform: priority base_xpos > base_link_name > identity
    base_inv = None
    if base_xpos is not None:
        base_inv = np.linalg.inv(base_xpos)
    elif base_link_name is not None:
        # Search base_link_name from all link_names of all articulations
        found = False
        for art in articulations:
            # Get all possible link names for this articulation
            if link_names is None:
                cur_link_names = art.get_link_names()
            elif isinstance(link_names, dict):
                cur_link_names = link_names.get(art, art.get_link_names())
            else:
                cur_link_names = link_names
            if base_link_name in cur_link_names:
                base_pose = art.get_link_pose(base_link_name)
                base_inv = np.linalg.inv(base_pose)
                found = True
                break
        if not found:
            logger.log_warning(
                f"base_link_name '{base_link_name}' not found in any articulation, using identity."
            )
            base_inv = np.eye(4)
    else:
        base_inv = np.eye(4)

    for art in articulations:
        if link_names is None:
            cur_link_names = art.get_link_names()
        elif isinstance(link_names, dict):
            cur_link_names = link_names.get(art, art.get_link_names())
        else:
            cur_link_names = link_names

        link_poses = [art.get_link_pose(name) for name in cur_link_names]

        for i, link_name in enumerate(cur_link_names):
            verts, faces = art.get_link_vert_face(link_name)
            logger.log_debug(
                f"Link '{link_name}' has {verts.shape[0]} vertices, {verts.shape[1]} faces."
            )
            if verts.shape[0] == 0:
                continue

            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
            )
            mesh.compute_vertex_normals()
            mesh.transform(link_poses[i])
            mesh.transform(base_inv)
            combined_mesh += mesh

    combined_mesh.compute_vertex_normals()

    ext = os.path.splitext(output_path)[1].lower()

    if ext in [".obj", ".ply", ".stl"]:
        o3d.io.write_triangle_mesh(output_path, combined_mesh)
        logger.log_info(f"Mesh exported using Open3D to: {output_path}")

    elif ext in [".glb", ".gltf"]:
        mesh_trimesh = trimesh.Trimesh(
            vertices=np.asarray(combined_mesh.vertices),
            faces=np.asarray(combined_mesh.triangles),
            vertex_normals=np.asarray(combined_mesh.vertex_normals),
        )
        mesh_trimesh.export(output_path)
        logger.log_info(f"Mesh exported using trimesh to: {output_path}")

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Supported: obj, ply, stl, glb, gltf"
        )

    return combined_mesh
