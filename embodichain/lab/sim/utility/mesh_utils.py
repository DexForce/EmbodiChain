import os
import dexsim.engine
import numpy as np
import open3d as o3d
import trimesh
from typing import Tuple, List, Dict, Any, Optional, Union

import dexsim
from embodichain.utils import logger
from embodichain.data import get_data_path


def process_meshes(
    mesh_config: Union[List[Dict], Dict], processor_config: Dict = None
) -> List[str]:
    r"""Process a list of mesh files using the specified processor configuration.

    Args:
        mesh_config (list): A list of dictionaries containing mesh file paths.
        processor_config (dict): A dictionary containing the processor configuration.

    Returns:
        list: A list of processed mesh file paths.
    """
    from embodichain.toolkits.processor.function.mesh_processor import (
        build_mesh_processors,
    )
    from embodichain.toolkits.processor.component import TriangleComponent
    from embodichain.toolkits.processor.entity import MeshEntity

    processors, replace = None, False
    if processor_config is not None:
        if "replace" in processor_config:
            replace = processor_config.pop("replace")
        processors = build_mesh_processors(processor_config)

    if isinstance(mesh_config, dict):
        mesh_config_list = list(mesh_config.values())
    else:
        mesh_config_list = mesh_config
    batch_meshes, batch_index = [], []
    for idx, config in enumerate(mesh_config_list):
        if "mesh_file" not in config and "mesh_path" not in config:
            logger.log_error("Config must contain 'mesh_file' and 'mesh_path' keys.")
        key = "mesh_file" if "mesh_file" in config else "mesh_path"
        mesh_fpath = config[key]
        mesh_fpath = get_data_path(mesh_fpath)
        if not os.path.exists(mesh_fpath):
            logger.log_error(f"Mesh file not found at path: {mesh_fpath}")
        config[key] = mesh_fpath
        save_fpath = (
            os.path.dirname(config[key])
            + "/mesh_processed_"
            + os.path.basename(config[key])
        )

        if processors is None and "mesh_processor" not in config:
            # No processors specified, so just return
            continue
        elif os.path.exists(save_fpath) and not replace:
            config[key] = save_fpath
            continue
        elif "mesh_processor" in config:
            # Process the mesh file with the specified processor
            mesh_processor = build_mesh_processors(config["mesh_processor"])
            tri_component = TriangleComponent.from_fpath(mesh_fpath)
            mesh_entity = MeshEntity("mesh", tri_component)
            mesh = mesh_processor.apply([mesh_entity])[0]
            mesh.save_mesh(save_fpath)
            # Update the mesh file path in the config
            config[key] = save_fpath
        else:
            tri_component = TriangleComponent.from_fpath(mesh_fpath)
            mesh_entity = MeshEntity("mesh", tri_component)
            batch_meshes.append(mesh_entity)
            batch_index.append(idx)

    # Process the batch of meshes with the default processors
    if batch_meshes and processors is not None:
        meshes = processors.apply(batch_meshes)
        for idx, config in enumerate(mesh_config_list):
            if idx in batch_index:
                save_fpath = (
                    os.path.dirname(config[key])
                    + "/mesh_processed_"
                    + os.path.basename(config[key])
                )
                meshes[batch_index.index(idx)].save_mesh(save_fpath)
                config[key] = save_fpath
    if isinstance(mesh_config, dict):
        mesh_config = {k: v for k, v in zip(mesh_config.keys(), mesh_config_list)}
    return mesh_config


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
