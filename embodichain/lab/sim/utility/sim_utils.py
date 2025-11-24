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
import dexsim
import open3d as o3d

from typing import List, Union, Optional

from dexsim.types import DriveType, ArticulationFlag, LoadOption, RigidBodyShape
from dexsim.engine import Articulation
from dexsim.environment import Env, Arena
from dexsim.models import MeshObject

from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg, SoftObjectCfg
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg, SphereCfg
from embodichain.utils import logger
from dexsim.kit.meshproc import get_mesh_auto_uv
import numpy as np


def get_dexsim_arenas() -> List[dexsim.environment.Arena]:
    """Get all arenas in the default dexsim world.

    Returns:
        List[dexsim.environment.Arena]: A list of arenas in the default world, or an empty list if no world is found.
    """
    world = dexsim.default_world()
    if world is None:
        logger.log_warning(f"No default world found. Returning empty arena list.")
        return []

    env = world.get_env()
    arenas = env.get_all_arenas()
    if len(arenas) == 0:
        return [env]
    return arenas


def get_dexsim_arena_num() -> int:
    """Get the number of arenas in the default dexsim world.

    Returns:
        int: The number of arenas in the default world, or 0 if no world is found.
    """
    arenas = get_dexsim_arenas()
    return len(arenas)


def get_dexsim_drive_type(drive_type: str) -> DriveType:
    """Get the dexsim drive type from a string.

    Args:
        drive_type (str): The drive type as a string.

    Returns:
        DriveType: The corresponding DriveType enum.
    """
    if drive_type == "force":
        return DriveType.FORCE
    elif drive_type == "acceleration":
        return DriveType.ACCELERATION
    else:
        logger.error(f"Invalid dexsim drive type: {drive_type}")


def set_dexsim_articulation_cfg(arts: List[Articulation], cfg: ArticulationCfg) -> None:
    """Set articulation configuration for a list of dexsim articulations.

    Args:
        arts (List[Articulation]): List of dexsim articulations to configure.
        cfg (ArticulationCfg): Configuration object containing articulation settings.
    """

    def get_drive_type(drive_pros):
        if isinstance(drive_pros, dict):
            return drive_pros.get("drive_type", None)
        return getattr(drive_pros, "drive_type", None)

    drive_pros = getattr(cfg, "drive_pros", None)
    drive_type = get_drive_type(drive_pros) if drive_pros is not None else None

    if drive_type == "force":
        drive_type = DriveType.FORCE
    elif drive_type == "acceleration":
        drive_type = DriveType.ACCELERATION
    else:
        logger.log_error(f"Unknow drive type {drive_type}")

    for art in arts:
        art.set_physical_attr(cfg.attrs.attr())
        art.set_articulation_flag(ArticulationFlag.FIX_BASE, cfg.fix_base)
        art.set_articulation_flag(
            ArticulationFlag.DISABLE_SELF_COLLISION, cfg.disable_self_collision
        )
        art.set_solver_iteration_counts(
            min_position_iters=cfg.min_position_iters,
            min_velocity_iters=cfg.min_velocity_iters,
        )
        link_names = art.get_link_names()
        for name in link_names:
            physical_body = art.get_physical_body(name)
            inertia = physical_body.get_mass_space_inertia_tensor()
            inertia = np.maximum(inertia, 1e-4)
            physical_body.set_mass_space_inertia_tensor(inertia)


def is_rt_enabled() -> bool:
    """Check if Ray Tracing rendering backend is enabled in the default dexsim world.

    Returns:
        bool: True if Ray Tracing rendering is enabled, False otherwise.
    """
    config = dexsim.get_world_config()

    return config.renderer == dexsim.types.Renderer.FASTRT


def create_cube(
    envs: List[Union[Env, Arena]], size: List[float], uid: str = "cube"
) -> List[MeshObject]:
    """Create cube objects in the specified environments or arenas.

    Args:
        envs (List[Union[Env, Arena]]): List of environments or arenas to create cubes in.
        size (List[float]): Size of the cube as [length, width, height] in meters.
        uid (str, optional): Unique identifier for the cube objects. Defaults to "cube".

    Returns:
        List[MeshObject]: List of created cube mesh objects.
    """
    cubes = []
    for i, env in enumerate(envs):
        cube = env.create_cube(size[0], size[1], size[2])
        cube.set_name(f"{uid}_{i}")
        cubes.append(cube)
    return cubes


def create_sphere(
    envs: List[Union[Env, Arena]],
    radius: float,
    resolution: int = 20,
    uid: str = "sphere",
) -> List[MeshObject]:
    """Create sphere objects in the specified environments or arenas.

    Args:
        envs (List[Union[Env, Arena]]): List of environments or arenas to create spheres in.
        radius (float): Radius of the sphere in meters.
        resolution (int, optional): Resolution of the sphere mesh. Defaults to 20.
        uid (str, optional): Unique identifier for the sphere objects. Defaults to "sphere".

    Returns:
        List[MeshObject]: List of created sphere mesh objects.
    """
    spheres = []
    for i, env in enumerate(envs):
        sphere = env.create_sphere(radius, resolution)
        sphere.set_name(f"{uid}_{i}")
        spheres.append(sphere)
    return spheres


def load_mesh_objects_from_cfg(
    cfg: RigidObjectCfg, env_list: List[Arena], cache_dir: Optional[str] = None
) -> List[MeshObject]:
    """Load mesh objects from configuration.

    Args:
        cfg (RigidObjectCfg): Configuration for the rigid object.
        env_list (List[Arena]): List of arenas to load the objects into.

        cache_dir (Optional[str], optional): Directory for caching convex decomposition files. Defaults to None
    Returns:
        List[MeshObject]: List of loaded mesh objects.
    """
    obj_list = []
    body_type = cfg.to_dexsim_body_type()
    if isinstance(cfg.shape, MeshCfg):

        option = LoadOption()
        option.rebuild_normals = cfg.shape.load_option.rebuild_normals
        option.rebuild_tangent = cfg.shape.load_option.rebuild_tangent
        option.rebuild_3rdnormal = cfg.shape.load_option.rebuild_3rdnormal
        option.rebuild_3rdtangent = cfg.shape.load_option.rebuild_3rdtangent
        option.smooth = cfg.shape.load_option.smooth

        cfg: RigidObjectCfg
        max_convex_hull_num = cfg.max_convex_hull_num
        fpath = cfg.shape.fpath

        compute_uv = cfg.shape.compute_uv

        for i, env in enumerate(env_list):
            if max_convex_hull_num > 1:
                obj = env.load_actor_with_coacd(
                    fpath,
                    duplicate=True,
                    attach_scene=True,
                    option=option,
                    cache_path=cache_dir,
                    actor_type=body_type,
                    max_convex_hull_num=max_convex_hull_num,
                )
            else:
                obj = env.load_actor(
                    fpath, duplicate=True, attach_scene=True, option=option
                )
                obj.add_rigidbody(body_type, RigidBodyShape.CONVEX)
            obj.set_name(f"{cfg.uid}_{i}")
            obj_list.append(obj)

            if compute_uv:
                vertices = obj.get_vertices()
                triangles = obj.get_triangles()

                o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
                _, uvs = get_mesh_auto_uv(
                    o3d_mesh, np.array(cfg.shape.project_direction)
                )
                obj.set_uv_mapping(uvs)

    elif isinstance(cfg.shape, CubeCfg):
        from embodichain.lab.sim.utility.sim_utils import create_cube

        obj_list = create_cube(env_list, cfg.shape.size, uid=cfg.uid)
        for obj in obj_list:
            obj.add_rigidbody(body_type, RigidBodyShape.BOX)

    elif isinstance(cfg.shape, SphereCfg):
        from embodichain.lab.sim.utility.sim_utils import create_sphere

        obj_list = create_sphere(
            env_list, cfg.shape.radius, cfg.shape.resolution, uid=cfg.uid
        )
        for obj in obj_list:
            obj.add_rigidbody(body_type, RigidBodyShape.SPHERE)
    else:
        logger.log_error(
            f"Unsupported rigid object shape type: {type(cfg.shape)}. Supported types: MeshCfg, CubeCfg, SphereCfg."
        )
    return obj_list


def load_soft_object_from_cfg(
    cfg: SoftObjectCfg, env_list: List[Arena]
) -> List[MeshObject]:
    obj_list = []

    option = LoadOption()
    option.rebuild_normals = cfg.shape.load_option.rebuild_normals
    option.rebuild_tangent = cfg.shape.load_option.rebuild_tangent
    option.rebuild_3rdnormal = cfg.shape.load_option.rebuild_3rdnormal
    option.rebuild_3rdtangent = cfg.shape.load_option.rebuild_3rdtangent
    option.smooth = cfg.shape.load_option.smooth
    option.share_mesh = False

    for i, env in enumerate(env_list):
        obj = env.load_actor(
            fpath=cfg.shape.fpath, duplicate=True, attach_scene=True, option=option
        )
        obj.add_softbody(cfg.voxel_attr.attr(), cfg.physical_attr.attr())
        if cfg.shape.compute_uv:
            vertices = obj.get_vertices()
            triangles = obj.get_triangles()

            o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
            _, uvs = get_mesh_auto_uv(o3d_mesh, cfg.shape.project_direction)
            obj.set_uv_mapping(uvs)
        obj.set_name(f"{cfg.uid}_{i}")
        obj_list.append(obj)
    return obj_list
