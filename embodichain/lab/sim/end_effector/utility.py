# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os
import typing
import pathlib
import hashlib
import numpy as np
import open3d as o3d
from dexsim.kit.meshproc import convex_decomposition_coacd
from dexsim.kit.meshproc.utility import mesh_list_to_file
from embodichain.utils import logger


def load_model_from_file(**kwargs) -> typing.Optional[str]:
    """Loads a model from the specified file path.

    This function checks the provided file path to determine if it is a URDF file
    or a mesh file (STL, OBJ, PLY). If it is a URDF file, it is loaded directly.
    If it is a mesh file, a URDF file is generated from the mesh.

    Args:
        file_path (str): The path to the input file (URDF or mesh file).

    Returns:
        Optional[str]: The path to the loaded URDF file, or None if the file path is not provided or unsupported.
    """
    file_path = kwargs.get("file_path", None)

    if file_path is None:
        logger.log_warning("No file path provided for the model.")
        return None

    file_suffix = pathlib.Path(file_path).suffix
    mesh_suffix_list = [".stl", ".obj", ".ply"]

    if file_suffix == ".urdf":
        # Load the URDF file directly
        urdf_path = file_path
    elif file_suffix in mesh_suffix_list:
        # Generate URDF from the mesh file
        urdf_path = generate_gripper_urdf_from_meshpath(file_path)
    else:
        logger.log_warning(
            f"Unsupported file extension {file_suffix} for model file {file_path}."
        )
        return None  # Return None for unsupported file types

    return urdf_path


def generate_gripper_urdf_from_meshpath(
    mesh_file: str, cache_dir: str = None, max_convex_hull_num: int = 16
) -> str:
    r"""Generate URDF for a gripper given a mesh file path.

    Args:
        mesh_file (str): The path of mesh file.
        cache_dir (str, optional): Cache directory. Defaults to None.
        max_convex_hull_num (int, optional): The maximum convex hull number. Defaults to 16.

    Returns:
        str: Urdf file path.
    """
    mesh_md5_key = hashlib.md5(open(mesh_file, "rb").read()).hexdigest()

    # Set cache directory
    save_dir = (
        pathlib.Path(cache_dir)
        if cache_dir
        else pathlib.Path.home() / "urdf_generate_cache"
    )
    # Create the directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define cache file names
    acd_file = f"{mesh_md5_key}_acd_{max_convex_hull_num}.obj"
    visual_file = f"{mesh_md5_key}_visual.obj"
    acd_cache_path = save_dir / acd_file
    visual_cache_path = save_dir / visual_file

    # Generate convex decomposition cache if not exists
    if not acd_cache_path.is_file() or not visual_cache_path.is_file():
        try:
            in_mesh = o3d.t.io.read_triangle_mesh(mesh_file)
            _, out_mesh_list = convex_decomposition_coacd(
                in_mesh, max_convex_hull_num=max_convex_hull_num
            )

            # Write approximate convex decomposition result
            mesh_list_to_file(str(acd_cache_path), out_mesh_list)
            # Write visual mesh
            o3d.t.io.write_triangle_mesh(str(visual_cache_path), in_mesh)
        except Exception as e:
            raise RuntimeError(f"Error during mesh processing: {e}")

    # Create URDF string
    urdf_str = f"""<?xml version="1.0" ?>
<robot name="{mesh_md5_key}">
    <link name="{mesh_md5_key}_base">
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{visual_file}"/>
            </geometry>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{acd_file}"/>
            </geometry>
        </collision>
    </link>
</robot>"""

    urdf_cache_path = save_dir / f"{mesh_md5_key}.urdf"

    try:
        with open(urdf_cache_path, "w") as writer:
            writer.write(urdf_str)
    except IOError as e:
        raise RuntimeError(f"Failed to write URDF file: {e}")

    return str(urdf_cache_path)


def inv_transform(transform: np.ndarray) -> np.ndarray:
    r"""Compute the inverse transformation.

    Args:
        transform (np.ndarray): A [4 x 4] transformation matrix.

    Returns:
        np.ndarray: The inverse transformation matrix.
    """
    r = transform[:3, :3]
    t = transform[:3, 3].T
    inv_r = r.T
    inv_t = -inv_r @ t

    inv_pose = np.eye(4, dtype=np.float32)
    inv_pose[:3, :3] = inv_r
    inv_pose[:3, 3] = inv_t

    return inv_pose
