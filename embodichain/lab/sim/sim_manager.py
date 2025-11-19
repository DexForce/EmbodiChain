# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os
import sys
import dexsim
import torch
import numpy as np
import open3d as o3d
import warp as wp

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from functools import cached_property
from typing import List, Union, Optional, Dict, Tuple, Union, Sequence
from dataclasses import dataclass, asdict, field, MISSING

# Global cache directories
SIM_CACHE_DIR = Path.home() / ".cache" / "embodichain_cache"
MATERIAL_CACHE_DIR = SIM_CACHE_DIR / "mat_cache"
CONVEX_DECOMP_DIR = SIM_CACHE_DIR / "convex_decomposition"
REACHABLE_XPOS_DIR = SIM_CACHE_DIR / "robot_reachable_xpos"

from dexsim.types import (
    Backend,
    ThreadMode,
    PhysicalAttr,
    ActorType,
    RigidBodyShape,
    RigidBodyGPUAPIReadType,
    ArticulationGPUAPIReadType,
)
from dexsim.engine import CudaArray, Material
from dexsim.models import MeshObject
from dexsim.render import Light as _Light, LightType
from dexsim.render import GizmoController
from dexsim.sensor import Sensor, MonocularCam, BinocularCam
from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectGroup,
    SoftObject,
    Articulation,
    Light,
)

# TODO: temporarily named `RobotV2` to avoid conflict with the old `Robot` class.
from embodichain.lab.sim.objects.robot import Robot as RobotV2
from embodichain.lab.sim.objects.gizmo import Gizmo
from embodichain.lab.sim.robots import Robot, Manipulator
from embodichain.lab.sim.end_effector import EndEffector, Suctor, Gripper
from embodichain.lab.sim.sensors import (
    SensorCfg,
    BaseSensor,
    Camera,
    StereoCamera,
)
from embodichain.lab.sim.cfg import (
    PhysicsCfg,
    MarkerCfg,
    GPUMemoryCfg,
    LightCfg,
    RigidObjectCfg,
    SoftObjectCfg,
    RigidObjectGroupCfg,
    ArticulationCfg,
    RobotCfg,
)
from embodichain.lab.sim import VisualMaterial, VisualMaterialCfg
from embodichain.data import SimResources
from embodichain.utils import configclass
from embodichain.utils import logger

__all__ = [
    "SimulationManager",
    "SimulationManagerCfg",
    "SIM_CACHE_DIR",
    "MATERIAL_CACHE_DIR",
    "CONVEX_DECOMP_DIR",
    "REACHABLE_XPOS_DIR",
]


@configclass
class SimulationManagerCfg:
    """Global robot simulation configuration."""

    width: int = 1920
    """The width of the simulation window."""

    height: int = 1080
    """The height of the simulation window."""

    headless: bool = False
    """Whether to run the simulation in headless mode (no Window)."""

    # TODO: We will add a more efficient hybrid rendering backend in the near future.
    # Then the rendering cofnig should be refactored.
    rendering_backend: Backend = Backend.VULKAN
    """The rendering backend to use (to be deprecated) """

    enable_rt: bool = False
    """Whether to enable ray tracing rendering."""

    enable_denoiser: bool = True
    """Whether to enable denoising for ray tracing rendering."""

    spp: int = 64
    """Samples per pixel for ray tracing rendering. This parameter is only valid when ray tracing is enabled and enable_denoiser is False."""

    gpu_id: int = 0
    """The gpu index that the simulation engine will be used. 
    
    Note: it will affect the gpu physics device if using gpu physics.
    """

    thread_mode: ThreadMode = ThreadMode.RENDER_SHARE_ENGINE
    """The threading mode for the simulation engine.
    
    - RENDER_SHARE_ENGINE: The rendering thread shares the same thread with the simulation engine.
    - RENDER_SCENE_SHARE_ENGINE: The rendering thread and scene update thread share the same thread with the simulation engine.
    """

    arena_space: float = 5.0
    """The distance between each arena when building multiple arenas."""

    physics_dt: float = 1.0 / 100.0
    """The time step for the physics simulation."""

    sim_device: Union[str, torch.device] = "cpu"
    """The device for the simulation engine. Can be 'cpu', 'cuda', or a torch.device object."""

    physics_config: PhysicsCfg = field(default_factory=PhysicsCfg)
    """The physics configuration parameters."""
    gpu_memory_config: GPUMemoryCfg = field(default_factory=GPUMemoryCfg)
    """The GPU memory configuration parameters."""

    # TODO: To be removed after refactoring.
    # default time step is for robot.
    time_step: float = 0.05


class SimulationManager:
    r"""Global Embodied AI simulation manager.

    This class is used to manage the global simulation environment and simulated assets.
        - assets loading, creation, modification and deletion.
            - assets include robots, fixed actors, dynamic actors and background.
        - manager the scenes and the simulation environment.
            - parallel scenes simulation on both CPU and GPU.
            - sensors arrangement
            - lighting and indirect lighting
            - physics simulation parameters control
        - ...

    Note:
        1. The arena is used as a standalone space for robots to simulate in. When :meth:`build_multiple_arenas` is called,
             it will create multiple arenas in a grid pattern. Meanwhile, each simulation assets adding interface will
             take an additional parameter `arena_index` to specify which arena to place the asset. The name of the asset to
             be added will be appended with the arena index to avoid name conflict.
        2. In GUI mode, the physics will be set to a fps (or a wait time for manual mode) for better visualization.


    Args:
        sim_config (SimulationManagerCfg, optional): simulation configuration. Defaults to SimulationManagerCfg().
    """

    # TODO: remove specific robot type like InspireHand, UnitreeH1, etc.
    supported_robot_types = {
        "Manipulator": Manipulator,
    }

    # TODO: deprecate sensor types.
    supported_sensor_types = {
        "MonocularCam": MonocularCam,
        "BinocularCam": BinocularCam,
    }

    SUPPORTED_SENSOR_TYPES = {"Camera": Camera, "StereoCamera": StereoCamera}

    supported_end_effector_types = {
        "Suctor": Suctor,
        "Gripper": Gripper,
    }

    def __init__(
        self, sim_config: SimulationManagerCfg = SimulationManagerCfg()
    ) -> None:
        # Cache paths
        self._sim_cache_dir = SIM_CACHE_DIR
        self._material_cache_dir = MATERIAL_CACHE_DIR
        self._convex_decomp_dir = CONVEX_DECOMP_DIR
        self._reachable_xpos_dir = REACHABLE_XPOS_DIR

        # Setup cache file path.
        for path in [
            self._sim_cache_dir,
            self._material_cache_dir,
            self._convex_decomp_dir,
            self._reachable_xpos_dir,
        ]:
            os.makedirs(path, exist_ok=True)

        self.sim_config = sim_config
        self.device = torch.device("cpu")

        world_config = self._convert_sim_config(sim_config)

        # Initialize warp runtime context before creating the world.
        wp.init()
        self._world = dexsim.World(world_config)

        fps = int(1.0 / sim_config.physics_dt)
        self._world.set_physics_fps(fps)

        self._world.set_time_scale(1.0)
        self._world.set_delta_time(sim_config.physics_dt)
        self._world.show_coordinate_axis(False)

        if sys.platform == "linux":
            dexsim.set_physics_config(**sim_config.physics_config.to_dexsim_args())
            dexsim.set_physics_gpu_memory_config(
                **sim_config.gpu_memory_config.to_dict()
            )

        self._is_initialized_gpu_physics = False
        self._ps = self._world.get_physics_scene()

        # activate physics
        self.enable_physics(True)

        self._env = self._world.get_env()
        self._env.clean()

        self._default_resources = SimResources()

        # set unique material path to accelerate material creation.
        if self.sim_config.enable_rt is False:
            self._env.set_unique_mat_path(
                os.path.join(self._material_cache_dir, "dexsim_mat")
            )

        # arena is used as a standalone space for robots to simulate in.
        self._arenas: List[dexsim.environment.Arena] = []

        # manager critical simulated assets in the world
        self._robots: Dict[str, Robot] = dict()
        self._end_effectors: Dict[str, EndEffector] = dict()
        # gizmo management
        self._gizmos: Dict[str, object] = dict()  # Store active gizmos
        self._sensors: Dict[str, Sensor] = dict()

        # TODO: fixed and dynamic actor will be deprecated.
        self._fixed_actors: Dict[str, Tuple[MeshObject, int]] = dict()
        self._dynamic_actors: Dict[str, Tuple[MeshObject, int]] = dict()

        self._rigid_objects: Dict[str, RigidObject] = dict()
        self._rigid_object_groups: Dict[str, RigidObjectGroup] = dict()
        self._soft_objects: Dict[str, SoftObject] = dict()
        self._articulations: Dict[str, Articulation] = dict()
        self._robots_v2: Dict[str, RobotV2] = dict()

        self._sensors_v2: Dict[str, BaseSensor] = dict()
        self._lights: Dict[str, _Light] = dict()

        # material placeholder.
        self._materials: Dict[str, Material] = dict()
        self._visual_materials: Dict[str, VisualMaterial] = dict()

        # Global texture cache for material creation or randomization.
        # The structure is keys to the loaded texture data. The keys represent the texture group.
        self._texture_cache: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = dict()

        # TODO: maybe need to add some interface to interact with background and layouts.
        # background and layouts are 3d assets that can has only render body for visualization.

        # TODO: add lighting setter. (point light, etc.)

        self._create_default_plane()
        self.set_default_background()

    def _convert_sim_config(
        self, sim_config: SimulationManagerCfg
    ) -> dexsim.WorldConfig:
        world_config = dexsim.WorldConfig()
        win_config = dexsim.WindowsConfig()
        win_config.width = sim_config.width
        win_config.height = sim_config.height
        world_config.win_config = win_config
        world_config.open_windows = not sim_config.headless
        self.is_window_opened = not sim_config.headless
        world_config.backend = sim_config.rendering_backend
        world_config.thread_mode = sim_config.thread_mode
        world_config.cache_path = str(self._material_cache_dir)
        world_config.length_tolerance = sim_config.physics_config.length_tolerance
        world_config.speed_tolerance = sim_config.physics_config.speed_tolerance

        if sim_config.enable_rt:
            world_config.renderer = dexsim.types.Renderer.FASTRT
            if sim_config.enable_denoiser is False:
                world_config.raytrace_config.spp = sim_config.spp
                world_config.raytrace_config.open_denoise = False

        if type(sim_config.sim_device) is str:
            self.device = torch.device(sim_config.sim_device)
        else:
            self.device = sim_config.sim_device

        if self.device.type == "cuda":
            world_config.enable_gpu_sim = True
            world_config.direct_gpu_api = True

            if self.device.index is not None and sim_config.gpu_id != self.device.index:
                logger.log_warning(
                    f"Conflict gpu_id {sim_config.gpu_id} and device index {self.device.index}. Using device index."
                )
                sim_config.gpu_id = self.device.index

                self.device = torch.device(f"cuda:{sim_config.gpu_id}")

        world_config.gpu_id = sim_config.gpu_id

        return world_config

    def get_default_resources(self) -> SimResources:
        """Get the default resources instance.

        Returns:
            SimResources: The default resources path.
        """
        return self._default_resources

    @property
    def num_envs(self) -> int:
        """Get the number of arenas in the simulation.

        Returns:
            int: number of arenas.
        """
        return len(self._arenas) if len(self._arenas) > 0 else 1

    @cached_property
    def is_use_gpu_physics(self) -> bool:
        """Check if the physics simulation is using GPU."""
        world_config = dexsim.get_world_config()
        return self.device.type == "cuda" and world_config.enable_gpu_sim

    @property
    def is_rt_enabled(self) -> bool:
        """Check if Ray Tracing rendering backend is enabled."""
        return self.sim_config.enable_rt

    @property
    def is_physics_manually_update(self) -> bool:
        return self._world.is_physics_manually_update()

    @property
    def asset_uids(self) -> List[str]:
        """Get all assets uid in the simulation.

        The assets include lights, sensors, robots, rigid objects and articulations.

        Returns:
            List[str]: list of all assets uid.
        """
        uid_list = ["default_plane"]
        uid_list.extend(list(self._lights.keys()))
        uid_list.extend(list(self._sensors_v2.keys()))
        uid_list.extend(list(self._robots_v2.keys()))
        uid_list.extend(list(self._rigid_objects.keys()))
        uid_list.extend(list(self._rigid_object_groups.keys()))
        uid_list.extend(list(self._articulations.keys()))
        return uid_list

    def enable_physics(self, enable: bool) -> None:
        """Enable or disable physics simulation.

        Args:
            enable (bool): whether to enable physics simulation.
        """
        self._world.enable_physics(enable)

    def set_manual_update(self, enable: bool) -> None:
        """Set manual update for physics simulation.

        If enable is True, the physics simulation will be updated manually by calling :meth:`update`.
        If enable is False, the physics simulation will be updated automatically by the engine thread loop.

        Args:
            enable (bool): whether to enable manual update.
        """
        self._world.set_manual_update(enable)

    def init_gpu_physics(self) -> None:
        """Initialize the GPU physics simulation."""
        if self.device.type != "cuda":
            logger.log_warning(
                "The simulation device is not cuda, cannot initialize GPU physics."
            )
            return

        if self._is_initialized_gpu_physics:
            return

        # init rigid body.
        rigid_body_num = (
            0
            if self._get_non_static_rigid_obj_num() == 0
            else len(self._ps.gpu_rigid_indices)
        )
        self._rigid_body_pose = torch.zeros(
            (rigid_body_num, 7), dtype=torch.float32, device=self.device
        )

        # init articulation.
        articulation_num = (
            0
            if len(self._articulations) == 0 and len(self._robots_v2) == 0
            else len(self._ps.gpu_articulation_indices)
        )
        max_link_count = self._ps.gpu_get_articulation_max_link_count()
        self._link_pose = torch.zeros(
            (articulation_num, max_link_count, 7),
            dtype=torch.float32,
            device=self.device,
        )
        for art in self._articulations.values():
            art.reallocate_body_data()
        for robot in self._robots_v2.values():
            robot.reallocate_body_data()

        # We do not perform reallocate body data for robot.

        self._is_initialized_gpu_physics = True

    def render_camera_group(self) -> None:
        """Render all camera group in the simulation.

        Note: This interface is only valid when Ray Tracing rendering backend is enabled.
        """

        if self.is_rt_enabled:
            self._world.render_camera_group()
        else:
            logger.log_warning(
                "This interface is only valid when Ray Tracing rendering backend is enabled."
            )

    def update(self, physics_dt: Optional[float] = None, step: int = 10) -> None:
        """Update the physics.

        Args:
            physics_dt (Optional[float], optional): the time step for physics simulation. Defaults to None.
            step (int, optional): the number of steps to update physics. Defaults to 10.
        """
        if self.is_use_gpu_physics and not self._is_initialized_gpu_physics:
            logger.log_warning(
                f"Using GPU physics, but not initialized yet. Forcing initialization."
            )
            self.init_gpu_physics()

        if self.is_physics_manually_update:
            if physics_dt is None:
                physics_dt = self.sim_config.physics_dt
            for i in range(step):
                self._world.update(physics_dt)

            if self.sim_config.enable_rt is False:
                self._sync_gpu_data()

        else:
            logger.log_warning("Physics simulation is not manually updated.")

    def _sync_gpu_data(self) -> None:
        if not self.is_use_gpu_physics:
            return

        if not self._is_initialized_gpu_physics:
            logger.log_warning(
                "GPU physics is not initialized. Skipping GPU data synchronization."
            )
            return

        if self.is_window_opened or self._sensors_v2:
            if len(self._rigid_body_pose) > 0:
                self._ps.gpu_fetch_rigid_body_data(
                    data=CudaArray(self._rigid_body_pose),
                    gpu_indices=self._ps.gpu_rigid_indices,
                    data_type=RigidBodyGPUAPIReadType.POSE,
                )

            if len(self._link_pose) > 0:
                self._ps.gpu_fetch_link_data(
                    data=CudaArray(self._link_pose),
                    gpu_indices=self._ps.gpu_articulation_indices,
                    data_type=ArticulationGPUAPIReadType.LINK_GLOBAL_POSE,
                )

            # TODO: might be optimized.
            self._world.sync_poses_gpu_to_cpu(
                rigid_pose=CudaArray(self._rigid_body_pose),
                link_pose=CudaArray(self._link_pose),
            )

    def get_env(self, arena_index: int = -1) -> dexsim.environment.Arena:
        """Get the arena or env by index.

        If arena_index is -1, return the global env.
        If arena_index is valid, return the corresponding arena.

        Args:
            arena_index (int, optional): the index of arena to get, -1 for global env. Defaults to -1.

        Returns:
            dexsim.environment.Arena: The arena or global env.
        """
        if arena_index >= 0:
            if arena_index > len(self._arenas) - 1:
                logger.log_error(
                    f"Invalid arena index: {arena_index}. Current number of arenas: {len(self._arenas)}"
                )
            return self._arenas[arena_index]
        else:
            return self._env

    def get_world(self) -> dexsim.World:
        return self._world

    def open_window(self) -> None:
        """Open the simulation window."""
        self._world.open_window()
        self.is_window_opened = True

    def close_window(self) -> None:
        """Close the simulation window."""
        self._world.close_window()
        self.is_window_opened = False

    def build_multiple_arenas(self, num: int, space: Optional[float] = None) -> None:
        """Build multiple arenas in a grid pattern.

        This interface is used for vectorized simulation.

        Args:
            num (int): number of arenas to build.
            space (float, optional): The distance between each arena. Defaults to the arena_space in sim_config.
        """

        if space is None:
            space = self.sim_config.arena_space

        if num <= 0:
            logger.log_warning("Number of arenas must be greater than 0.")
            return

        scene_grid_length = int(np.ceil(np.sqrt(num)))

        for i in range(num):
            arena = self._env.add_arena(f"arena_{i}")

            id_x, id_y = i % scene_grid_length, i // scene_grid_length
            arena.set_root_node_position([id_x * space, id_y * space, 0])
            self._arenas.append(arena)

    def set_indirect_lighting(self, name: str) -> None:
        """Set indirect lighting.

        Args:
            name (str): name of path of the indirect lighting.
        """
        if name.startswith("/") is False:
            ibl_path = self._default_resources.get_ibl_path(name)
            logger.log_info(f"Set IBL {name} from sim default resources.")
        else:
            ibl_path = name
            logger.log_info(f"Set IBL {name} from custom path.")

        self._env.set_IBL(ibl_path)

    def set_emission_light(
        self, color: Optional[Sequence[float]] = None, intensity: Optional[float] = None
    ) -> None:
        """Set environment emission light.

        Args:
            color (Sequence[float]): color of the light.
            intensity (float): intensity of the light.
        """
        if color is None:
            self._env.set_env_light_emission(color)
        if intensity is None:
            self._env.set_env_light_intensity(intensity)

    def _create_default_plane(self):
        default_length = 1000
        repeat_uv_size = int(default_length / 2)
        self._default_plane = self._env.create_plane(
            0, default_length, repeat_uv_size, repeat_uv_size
        )
        self._default_plane.set_name("default_plane")
        plane_collision = self._env.create_cube(
            default_length, default_length, default_length / 10
        )
        plane_collision_pose = np.eye(4, dtype=float)
        plane_collision_pose[2, 3] = -default_length / 20 - 0.001
        plane_collision.set_local_pose(plane_collision_pose)
        plane_collision.add_rigidbody(ActorType.KINEMATIC, RigidBodyShape.CONVEX)

        # TODO: add default physics attributes for the plane.

    def set_default_background(self) -> None:
        """Set default background."""

        mat_name = "plane_mat"
        mat = None
        mat_path = self._default_resources.get_material_path("PlaneDark")
        color_texture = os.path.join(mat_path, "PlaneDark_2K_Color.jpg")
        roughness_texture = os.path.join(mat_path, "PlaneDark_2K_Roughness.jpg")
        mat = self.create_visual_material(
            cfg=VisualMaterialCfg(
                uid=mat_name,
                base_color_texture=color_texture,
                roughness_texture=roughness_texture,
            )
        )

        if self.sim_config.enable_rt:
            self.set_emission_light([0.1, 0.1, 0.1], 10.0)
        else:
            self.set_indirect_lighting("lab_day")

        self._default_plane.set_material(mat.get_instance("plane_mat").mat)
        self._visual_materials[mat_name] = mat

    def set_texture_cache(
        self, key: str, texture: Union[torch.Tensor, List[torch.Tensor]]
    ) -> None:
        """Set the texture to the global texture cache.

        Args:
            key (str): The key of the texture.
            texture (Union[torch.Tensor, List[torch.Tensor]]): The texture data.
        """
        self._texture_cache[key] = texture

    def get_texture_cache(
        self, key: Optional[str] = None
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """Get the texture from the global texture cache.

        Args:
            key (str, optional): The key of the texture. If None, return None. Defaults to None.

        Returns:
            Optional[Union[torch.Tensor, List[torch.Tensor]]]: The texture if found, otherwise None.
        """
        if key is None:
            return self._texture_cache

        if key not in self._texture_cache:
            logger.log_warning(f"Texture {key} not found in global texture cache.")
            return None
        return self._texture_cache[key]

    def get_asset(
        self, uid: str
    ) -> Optional[Union[Light, BaseSensor, RobotV2, RigidObject, Articulation]]:
        """Get an asset by its UID.

        The asset can be a light, sensor, robot, rigid object or articulation.

        Args:
            uid (str): The UID of the asset.

        Returns:
            Light | BaseSensor | RobotV2 | RigidObject | Articulation | None: The asset instance if found, otherwise None.
        """
        if uid in self._lights:
            return self._lights[uid]
        if uid in self._sensors_v2:
            return self._sensors_v2[uid]
        if uid in self._robots_v2:
            return self._robots_v2[uid]
        if uid in self._rigid_objects:
            return self._rigid_objects[uid]
        if uid in self._rigid_object_groups:
            return self._rigid_object_groups[uid]
        if uid in self._articulations:
            return self._articulations[uid]

        logger.log_warning(f"Asset {uid} not found.")
        return None

    def add_light(self, cfg: LightCfg) -> Light:
        """Create a light in the scene.

        Args:
            cfg (LightCfg): Configuration for the light, including type, color, intensity, and radius.

        Returns:
            Light: The created light instance.
        """
        if cfg.uid is None:
            uid = "light"
            cfg.uid = uid
        else:
            uid = cfg.uid

        if uid in self._lights:
            logger.log_error(f"Light {uid} already exists.")

        light_type = cfg.light_type
        if light_type == "point":
            light_type = LightType.POINT
        else:
            logger.log_error(
                f"Unsupported light type: {light_type}. Supported types: point."
            )

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        light_list = []
        for i, env in enumerate(env_list):
            light_name = f"{uid}_{i}"
            light = env.create_light(light_name, light_type)
            light_list.append(light)

        batch_lights = Light(cfg=cfg, entities=light_list)

        self._lights[uid] = batch_lights

        return batch_lights

    def get_light(self, uid: str) -> Optional[Light]:
        """Get a light by its UID.

        Args:
            uid (str): The UID of the light.

        Returns:
            Light | None: The light instance if found, otherwise None.
        """
        if uid not in self._lights:
            logger.log_warning(f"Light {uid} not found.")
            return None
        return self._lights[uid]

    def add_rigid_object(
        self,
        cfg: RigidObjectCfg,
    ) -> RigidObject:
        """Add a rigid object to the scene.

        Args:
            cfg (RigidObjectCfg): Configuration for the rigid object.

        Returns:
            RigidObject: The added rigid object instance handle.
        """
        from embodichain.lab.sim.utility.sim_utils import (
            load_mesh_objects_from_cfg,
        )

        uid = cfg.uid
        if uid is None:
            logger.log_error("Rigid object uid must be specified.")
        if uid in self._rigid_objects:
            logger.log_error(f"Rigid object {uid} already exists.")

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = load_mesh_objects_from_cfg(
            cfg=cfg,
            env_list=env_list,
            cache_dir=self._convex_decomp_dir,
        )

        rigid_obj = RigidObject(cfg=cfg, entities=obj_list, device=self.device)

        if cfg.shape.visual_material:
            mat = self.create_visual_material(cfg.shape.visual_material)
            rigid_obj.set_visual_material(mat)

        self._rigid_objects[uid] = rigid_obj

        return rigid_obj

    def add_soft_object(self, cfg: SoftObjectCfg) -> SoftObject:
        """Add a soft object to the scene.

        Args:
            cfg (SoftObjectCfg): Configuration for the soft object.

        Returns:
            SoftObject: The added soft object instance handle.
        """
        if not self.is_use_gpu_physics:
            logger.log_error("Soft object requires GPU physics to be enabled.")

        from embodichain.lab.sim.utility import (
            load_soft_object_from_cfg,
        )

        uid = cfg.uid
        if uid is None:
            logger.log_error("Soft object uid must be specified.")

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = load_soft_object_from_cfg(
            cfg=cfg,
            env_list=env_list,
        )

        soft_obj = SoftObject(cfg=cfg, entities=obj_list, device=self.device)
        self._soft_objects[uid] = soft_obj
        return soft_obj

    def get_rigid_object(self, uid: str) -> Optional[RigidObject]:
        """Get a rigid object by its unique ID.

        Args:
            uid (str): The unique ID of the rigid object.

        Returns:
            Optional[RigidObject]: The rigid object instance if found, otherwise None.
        """
        if uid not in self._rigid_objects:
            logger.log_warning(f"Rigid object {uid} not found.")
            return None
        return self._rigid_objects[uid]

    def get_rigid_object_uid_list(self) -> List[str]:
        """Get current rigid body uid list

        Returns:
            List[str]: list of rigid body uid.
        """
        return list(self._rigid_objects.keys())

    def add_rigid_object_group(self, cfg: RigidObjectGroupCfg) -> RigidObjectGroup:
        """Add a rigid object group to the scene.

        Args:
            cfg (RigidObjectGroupCfg): Configuration for the rigid object group.
        """
        from embodichain.lab.sim.utility.sim_utils import (
            load_mesh_objects_from_cfg,
        )

        uid = cfg.uid
        if uid is None:
            logger.log_error("Rigid object group uid must be specified.")
        if uid in self._rigid_object_groups:
            logger.log_error(f"Rigid object group {uid} already exists.")

        if cfg.body_type == "static":
            logger.log_error("Rigid object group cannot be static.")

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas

        obj_group_list = []
        for key, rigid_cfg in tqdm(
            cfg.rigid_objects.items(), desc="Loading rigid objects"
        ):
            obj_list = load_mesh_objects_from_cfg(
                cfg=rigid_cfg,
                env_list=env_list,
                cache_dir=self._convex_decomp_dir,
            )
            obj_group_list.append(obj_list)

        # Convert [a1, a2, ...], [b1, b2, ...] to [(a1, b1, ...), (a2, b2, ...), ...]
        obj_group_list = list(zip(*obj_group_list))
        rigid_obj_group = RigidObjectGroup(
            cfg=cfg, entities=obj_group_list, device=self.device
        )

        self._rigid_object_groups[uid] = rigid_obj_group

        return rigid_obj_group

    def get_rigid_object_group(self, uid: str) -> Optional[RigidObjectGroup]:
        """Get a rigid object group by its unique ID.

        Args:
            uid (str): The unique ID of the rigid object group.

        Returns:
            Optional[RigidObjectGroup]: The rigid object group instance if found, otherwise None.
        """
        if uid not in self._rigid_object_groups:
            logger.log_warning(f"Rigid object group {uid} not found.")
            return None
        return self._rigid_object_groups[uid]

    def _get_non_static_rigid_obj_num(self) -> int:
        """Get the number of non-static rigid objects in the scene.

        Returns:
            int: The number of non-static rigid objects.
        """
        count = 0
        for obj in self._rigid_objects.values():
            if obj.cfg.body_type != "static":
                count += 1
        return count

    def add_articulation(
        self,
        cfg: ArticulationCfg,
    ) -> Articulation:
        """Add an articulation to the scene.

        Args:
            cfg (ArticulationCfg): Configuration for the articulation.

        Returns:
            Articulation: The added articulation instance handle.
        """

        uid = cfg.uid
        if uid is None:
            uid = os.path.splitext(os.path.basename(cfg.fpath))[0]
            cfg.uid = uid
        if uid in self._articulations:
            logger.log_error(f"Articulation {uid} already exists.")

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = []

        for env in env_list:
            art = env.load_urdf(cfg.fpath)
            obj_list.append(art)

        articulation = Articulation(cfg=cfg, entities=obj_list, device=self.device)

        self._articulations[uid] = articulation

        return articulation

    def get_articulation(self, uid: str) -> Optional[Articulation]:
        """Get an articulation by its unique ID.

        Args:
            uid (str): The unique ID of the articulation.

        Returns:
            Optional[Articulation]: The articulation instance if found, otherwise None.
        """
        if uid not in self._articulations:
            logger.log_warning(f"Articulation {uid} not found.")
            return None
        return self._articulations[uid]

    def get_articulation_uid_list(self) -> List[str]:
        """Get current articulation uid list

        Returns:
            List[str]: list of articulation uid.
        """
        return list(self._articulations.keys())

    def add_robot_v2(self, cfg: RobotCfg) -> Optional[RobotV2]:
        """Add a Robot to the scene.

        Args:
            cfg (RobotCfg): Configuration for the robot.

        Returns:
            Optional[RobotV2]: The added robot instance handle, or None if failed.
        """

        uid = cfg.uid
        if cfg.fpath is None:
            if cfg.urdf_cfg is None:
                logger.log_error(
                    "Robot configuration must have a valid fpath or urdf_cfg."
                )
                return None

            cfg.fpath = cfg.urdf_cfg.assemble_urdf()

        if uid is None:
            uid = os.path.splitext(os.path.basename(cfg.fpath))[0]
            cfg.uid = uid
        if uid in self._robots_v2:
            logger.log_error(f"Robot {uid} already exists.")
            return self._robots_v2[uid]

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = []

        for env in env_list:
            art = env.load_urdf(cfg.fpath)
            obj_list.append(art)

        robot = RobotV2(cfg=cfg, entities=obj_list, device=self.device)

        self._robots_v2[uid] = robot

        return robot

    def get_robot_v2(self, uid: str) -> Optional[RobotV2]:
        """Get a Robot by its unique ID.

        Args:
            uid (str): The unique ID of the robot.

        Returns:
            Optional[RobotV2]: The robot instance if found, otherwise None.
        """
        if uid not in self._robots_v2:
            logger.log_warning(f"Robot {uid} not found.")
            return None
        return self._robots_v2[uid]

    def get_robot_v2_uid_list(self) -> List[str]:
        """
        Retrieves a list of unique identifiers (UIDs) for all robots in the V2 system.

        Returns:
            list: A list containing the UIDs of the robots.
        """
        return list(self._robots_v2.keys())

    def add_robot(self, robot_type: str, robot_uid: str = None, **kwargs) -> Robot:
        """General interface to add a robot to the scene and returns a handle.

        Args:
            robot_type (str): type of robot
            robot_uid (str, optional): unique id of robot, if None, use default id. Defaults to None.
            **kwargs: other parameters for robot creation.
                - arena_index: the index of arena to place the robot.

        Returns:
            Robot: The robot instance handle.
        """
        if robot_type not in self.supported_robot_types:
            logger.log_warning(f"Unsupported robot type: {robot_type}")
            return None

        if robot_uid is None:
            robot_uid = robot_type

        if robot_uid in self._robots:
            logger.log_warning(f"Robot {robot_uid} already exists.")
            return None

        arena_index = kwargs.get("arena_index", -1)
        if arena_index >= 0:
            robot_uid = f"{robot_uid}_{arena_index}"

        # time_step from sim_config as default, can be overrided by kwargs.
        init_params = {
            "robot_uid": robot_uid,
            "env": self.get_env(arena_index),
            "time_step": self.sim_config.time_step,
        }
        init_params.update(kwargs)
        robot = self.supported_robot_types[robot_type](**init_params)
        robot.uid = robot_uid

        self._robots[robot_uid] = robot
        return robot

    def get_robot(self, robot_name: str, arena_index: int = -1) -> Robot:
        """Get robot by name.

        Args:
            robot_name (str): robot name.
            arena_index (int, optional): the index of arena to get, -1 for global env. Defaults to -1.
        """
        if arena_index >= 0:
            robot_name = f"{robot_name}_{arena_index}"

        if robot_name not in self._robots:
            logger.log_warning(f"Robot {robot_name} not found.")
            return None

        return self._robots[robot_name]

    def enable_gizmo(
        self, uid: str, control_part: Optional[str] = None, gizmo_cfg: object = None
    ) -> None:
        """Enable gizmo control for any simulation object (Robot, RigidObject, Camera, etc.).

        Args:
            uid (str): UID of the object to attach gizmo to (searches in robots_v2, rigid_objects, sensors_v2, etc.)
            control_part (Optional[str], optional): Control part name for robots. Defaults to "arm".
            gizmo_cfg (object, optional): Gizmo configuration object. Defaults to None.
        """
        # Create gizmo key combining uid and control_part
        gizmo_key = f"{uid}:{control_part}" if control_part else uid

        # Check if gizmo already exists
        if gizmo_key in self._gizmos:
            logger.log_warning(
                f"Gizmo for '{uid}' with control_part '{control_part}' already exists."
            )
            return

        # Search for target object in different collections
        target = None
        object_type = None

        if uid in self._robots_v2:
            target = self._robots_v2[uid]
            object_type = "robot"
        elif uid in self._rigid_objects:
            target = self._rigid_objects[uid]
            object_type = "rigid_object"
        elif uid in self._sensors_v2:
            target = self._sensors_v2[uid]
            object_type = "sensor"

        else:
            logger.log_error(
                f"Object with uid '{uid}' not found in any collection (robots_v2, rigid_objects, sensors_v2, articulations)."
            )
            return

        try:
            gizmo = Gizmo(target, gizmo_cfg, control_part)
            self._gizmos[gizmo_key] = gizmo
            logger.log_info(
                f"Gizmo enabled for {object_type} '{uid}' with control_part '{control_part}'"
            )

            # Initialize GizmoController if not already done.
            if not hasattr(self, "_gizmo_controller") or self._gizmo_controller is None:
                windows = (
                    self._world.get_windows()
                    if hasattr(self._world, "get_windows")
                    else None
                )
                self._gizmo_controller = GizmoController(windows)
                print("GizmoController attributes and methods:")
                print(dir(self._gizmo_controller))

        except Exception as e:
            logger.log_error(
                f"Failed to create gizmo for {object_type} '{uid}' with control_part '{control_part}': {e}"
            )

    def disable_gizmo(self, uid: str, control_part: Optional[str] = None) -> None:
        """Disable and remove gizmo for a robot.

        Args:
            uid (str): Object UID to disable gizmo for
            control_part (Optional[str], optional): Control part name for robots. Defaults to None.
        """
        # Create gizmo key combining uid and control_part
        gizmo_key = f"{uid}:{control_part}" if control_part else uid

        if gizmo_key not in self._gizmos:
            from embodichain.utils import logger

            logger.log_warning(
                f"No gizmo found for '{uid}' with control_part '{control_part}'."
            )
            return

        try:
            gizmo = self._gizmos[gizmo_key]
            if gizmo is not None:
                gizmo.destroy()
            del self._gizmos[gizmo_key]

            from embodichain.utils import logger

            logger.log_info(
                f"Gizmo disabled for '{uid}' with control_part '{control_part}'"
            )

        except Exception as e:
            from embodichain.utils import logger

            logger.log_error(
                f"Failed to disable gizmo for '{uid}' with control_part '{control_part}': {e}"
            )

    def get_gizmo(self, uid: str, control_part: Optional[str] = None) -> object:
        """Get gizmo instance for a robot.

        Args:
            uid (str): Object UID
            control_part (Optional[str], optional): Control part name for robots. Defaults to None.

        Returns:
            object: Gizmo instance if found, None otherwise.
        """
        # Create gizmo key combining uid and control_part
        gizmo_key = f"{uid}:{control_part}" if control_part else uid
        return self._gizmos.get(gizmo_key, None)

    def has_gizmo(self, uid: str, control_part: Optional[str] = None) -> bool:
        """Check if a gizmo exists for the given UID and control part.

        Args:
            uid (str): Object UID to check
            control_part (Optional[str], optional): Control part name for robots. Defaults to None.

        Returns:
            bool: True if gizmo exists, False otherwise.
        """
        # Create gizmo key combining uid and control_part
        gizmo_key = f"{uid}:{control_part}" if control_part else uid
        return gizmo_key in self._gizmos

    def list_gizmos(self) -> Dict[str, bool]:
        """List all active gizmos and their status.

        Returns:
            Dict[str, bool]: Dictionary mapping gizmo keys (uid:control_part) to gizmo active status.
        """
        return {
            gizmo_key: (gizmo is not None) for gizmo_key, gizmo in self._gizmos.items()
        }

    def update_gizmos(self):
        """Update all active gizmos."""
        for gizmo_key, gizmo in list(
            self._gizmos.items()
        ):  # Use list() to avoid modification during iteration
            if gizmo is not None:
                try:
                    gizmo.update()
                except Exception as e:
                    from embodichain.utils import logger

                    logger.log_error(f"Error updating gizmo '{gizmo_key}': {e}")

    def toggle_gizmo_visibility(
        self, uid: str, control_part: Optional[str] = None
    ) -> bool:
        """
        Toggle the visibility of a gizmo by uid and optional control_part.
        Returns the new visibility state (True=visible, False=hidden), or None if not found.
        """
        gizmo = self.get_gizmo(uid, control_part)
        if gizmo is not None:
            return gizmo.toggle_visibility()
        return None

    def set_gizmo_visibility(
        self, uid: str, visible: bool, control_part: Optional[str] = None
    ) -> None:
        """
        Set the visibility of a gizmo by uid and optional control_part.
        """
        gizmo = self.get_gizmo(uid, control_part)
        if gizmo is not None:
            gizmo.set_visibility(visible)

    def remove_robot(self, robot_name: str) -> bool:
        """Remove a robot from the scene.

        Args:
            robot_name (str): robot name
        """
        if robot_name not in self._robots:
            logger.log_warning(f"Robot {robot_name} not found.")
            return False

        self._robots[robot_name].destroy()
        self._robots.pop(robot_name)
        return True

    def get_robot_uid_list(self) -> List[str]:
        """Get current robot uid list

        Returns:
            List[str]: list of robot uid.
        """
        return list(self._robots.keys())

    def add_end_effector(
        self, ef_type: str, ef_uid: str = None, **kwargs
    ) -> EndEffector:
        """General interface to add an end effector to the scene and returns a handle.

        Args:
            ef_type (str): type of end effector.
            ef_uid (str, optional): unique id of end effector. Defaults to None.
            **kwargs: other parameters for end effector creation.
                - arena_index: the index of arena to place the end effector.

        Returns:
            EndEffector: The added end effector instance handle.
        """
        if ef_type not in self.supported_end_effector_types:
            logger.log_warning(f"Unsupported end effector type: {ef_type}")
            return None

        if ef_uid is None:
            ef_uid = ef_type

        if ef_uid in self._end_effectors:
            logger.log_warning(f"End effector {ef_uid} already exists.")
            return None

        arena_index = kwargs.get("arena_index", -1)
        if arena_index >= 0:
            ef_uid = f"{ef_uid}_{arena_index}"

        init_params = {"env": self.get_env(arena_index), "uid": ef_uid}
        init_params.update(kwargs)
        end_effector = self.supported_end_effector_types[ef_type](**init_params)

        self._end_effectors[ef_uid] = end_effector
        return end_effector

    def get_end_effector(self, ef_name: str, arena_index: int = -1) -> EndEffector:
        """Get end effector by name.

        Args:
            ef_name (str): end effector name.
            arena_index (int, optional): the index of arena to get, -1 for global env. Defaults to -1.
        """
        _ = self.get_end_effector_uid_list()

        if arena_index >= 0:
            ef_name = f"{ef_name}_{arena_index}"

        if ef_name not in self._end_effectors:
            logger.log_warning(f"End effector {ef_name} not found.")
            return None

        return self._end_effectors[ef_name]

    def remove_end_effector(self, ef_name: str) -> bool:
        """Remove an end effector from the scene.

        Args:
            ef_name (str): end effector name
        """
        if ef_name not in self._end_effectors:
            logger.log_warning(f"End effector {ef_name} not found.")
            return False
        # robot
        ee = self._end_effectors[ef_name]
        attach_robot_uid = ee.attach_robot_uid
        if attach_robot_uid in self._robots:
            robot = self._robots[attach_robot_uid]
            robot.detach_end_effector(ef_name)
        env = self._end_effectors[ef_name].get_env()
        # TODO: remove articulation might be result in crash when
        # the program exit. We should forbid the articulation destructor
        # to be called by python code.
        env.remove_articulation(self._end_effectors[ef_name].get_articulation())
        self._end_effectors.pop(ef_name)
        return True

    def get_end_effector_uid_list(self) -> List[str]:
        """Get current end effector uid list

        Returns:
            List[str]: list of end effector uid.
        """
        for key, robot in self._robots.items():
            if hasattr(robot, "get_end_effector"):
                end_effector_dict = robot.get_end_effector()
                if isinstance(end_effector_dict, Dict):
                    self._end_effectors.update(end_effector_dict)

        return list(self._end_effectors.keys())

    def add_sensor(self, sensor_type: str, sensor_uid: str = None, **kwargs) -> Sensor:
        """General interface to add a sensor to the scene and returns a handle.

        Args:
            sensor_type (str): type of sensor.
            sensor_uid (str): unique id of sensor.
            **kwargs: other parameters for sensor creation.
                - arena_index: the index of arena to place the sensor.

        Returns:
            Sensor: The added sensor instance handle.
        """
        if sensor_type not in self.supported_sensor_types:
            logger.log_warning(f"Unsupported sensor type: {sensor_type}")
            return None

        if sensor_uid is None:
            sensor_uid = sensor_type

        if sensor_uid in self._sensors:
            logger.log_warning(f"Sensor {sensor_uid} already exists.")
            return None

        arena_index = kwargs.get("arena_index", -1)
        if arena_index >= 0:
            sensor_uid = f"{sensor_uid}_{arena_index}"

        init_params = {"env": self.get_env(arena_index), "name": sensor_uid}
        init_params.update(kwargs)
        sensor = self.supported_sensor_types[sensor_type](**init_params)

        self._sensors[sensor_uid] = sensor
        return sensor

    def add_sensor_v2(self, sensor_cfg: SensorCfg) -> BaseSensor:
        """General interface to add a sensor to the scene and returns a handle.

        Args:
            sensor_cfg (SensorCfg): configuration for the sensor.

        Returns:
            BaseSensor: The added sensor instance handle.
        """
        sensor_type = sensor_cfg.sensor_type
        if sensor_type not in self.SUPPORTED_SENSOR_TYPES:
            logger.log_warning(f"Unsupported sensor type: {sensor_type}")
            return None

        sensor_uid = sensor_cfg.uid
        if sensor_uid is None:
            sensor_uid = f"{sensor_type.lower()}_{len(self._sensors_v2)}"
            sensor_cfg.uid = sensor_uid

        if sensor_uid in self._sensors_v2:
            logger.log_warning(f"Sensor {sensor_uid} already exists.")
            return None

        sensor = self.SUPPORTED_SENSOR_TYPES[sensor_type](sensor_cfg, self.device)

        self._sensors_v2[sensor_uid] = sensor

        # Check if the sensor needs to change the parent frame.

        return sensor

    def get_sensor_v2(self, uid: str) -> Optional[BaseSensor]:
        """Get a sensor by its UID.

        Args:
            uid (str): The UID of the sensor.

        Returns:
            BaseSensor | None: The sensor instance if found, otherwise None.
        """
        if uid not in self._sensors_v2:
            logger.log_warning(f"Sensor {uid} not found.")
            return None
        return self._sensors_v2[uid]

    def get_sensor(self, sensor_name: str, arena_index: int = -1) -> Sensor:
        """Get sensor by name.

        Args:
            sensor_name (str): sensor name
        """
        if arena_index >= 0:
            sensor_name = f"{sensor_name}_{arena_index}"

        if sensor_name not in self._sensors:
            logger.log_warning(f"Sensor {sensor_name} not found.")
            return None

        return self._sensors[sensor_name]

    def get_sensor_v2_uid_list(self) -> List[str]:
        """Get current sensor uid list

        Returns:
            List[str]: list of sensor uid.
        """
        return list(self._sensors_v2.keys())

    def remove_sensor(self, sensor_name: str) -> bool:
        """Remove a sensor from the scene.

        Args:
            sensor_name (str): sensor name
        """
        if sensor_name not in self._sensors:
            logger.log_warning(f"Sensor {sensor_name} not found.")
            return False

        self._sensors.pop(sensor_name)
        return True

    def get_sensor_uid_list(self) -> List[str]:
        """Get current sensor uid list

        Returns:
            List[str]: list of sensor uid.
        """
        return list(self._sensors.keys())

    def add_fixed_actor(
        self,
        fpath: str,
        init_pose: np.ndarray,
        scale: Union[List, np.ndarray] = [1.0, 1.0, 1.0],
        material: Material = None,
        texture_path: str = None,
        is_convex_decomposition: bool = False,
        damping_coeff: float = 0.5,
        friction_coeff: float = 0.7,
        **kwargs,
    ) -> MeshObject:
        """Add a fixed rigid object.

        The fixed object will have a static physical body and will not move during the simulation.

        Note:
            1. Currently, only base color texture is supported to add to the object.

        Args:
            fpath (str): path to mesh file.
            init_pose (np.ndarray): object pose in world frame.
            scale (Union[List, np.ndarray], optional): scale of object. Defaults to [1.0, 1.0, 1.0].
            material (Material, optional): path to material file. Defaults to None.
            texture_path (str, optional): path to texture file. Defaults to None.
            convex_decomposition_path (str, optional): path to decomposite convex file. Defaults to None.
            is_convex_decomposition (bool, optional): whether to use convex decomposition
            damping_coeff (float, optional): damage coefficient. Defaults to 0.5.
            friction_coeff (float, optional): friction coefficient. Defaults to 0.7.
            **kwargs: other parameters for fixed object creation.
                - arena_index (int): the index of arena to place the fixed object.
                - max_convex_hull_num (int): maximum convex hull number. Defaults to 32.
                - contact_offset (float): contact offset. Defaults to 0.015.
                - rest_offset (float): rest offset. Defaults to 0.000.
                - density (float): density of object. Defaults to 1.
                - restitution (float): restitution coefficient. Defaults to 0.05.
                - min_position_iters (int): minimum position iterations. Defaults to 10.
                - min_velocity_iters (int): minimum velocity iterations. Defaults to 10.
                - projection_direction (np.ndarray): projection direction. Defaults to np.array([1.0, 1.0, 1.0]).
                - compute_uv (bool): whether to compute uv mapping. Defaults to False.

        Returns:
            MeshObject: The added fixed object instance handle.
        """
        self._world.enable_physics(False)

        arena_index = kwargs.get("arena_index", -1)
        env = self.get_env(arena_index)

        duplicate = False if arena_index <= 0 else True

        if is_convex_decomposition:
            max_convex_hull_num = kwargs.get("max_convex_hull_num", 32)
            fixed_actor = env.load_actor_with_coacd(
                fpath,
                duplicate=duplicate,
                attach_scene=False,
                cache_path=os.path.join(self._sim_cache_dir, "convex_decomposition"),
                actor_type=ActorType.STATIC,
                max_convex_hull_num=max_convex_hull_num,
            )
        else:
            fixed_actor = env.load_actor(fpath, duplicate, False)
            fixed_actor.add_rigidbody(ActorType.STATIC, RigidBodyShape.CONVEX)

        fixed_actor.set_body_scale(scale[0], scale[1], scale[2])

        compute_uv = kwargs.get("compute_uv", False)
        if material is not None:
            compute_uv = True
        if compute_uv and fixed_actor.has_uv_mapping() is False:
            from dexsim.kit.meshproc import get_mesh_auto_uv

            vertices = fixed_actor.get_vertices()
            triangles = fixed_actor.get_triangles()

            o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
            projection_direction = kwargs.get(
                "projection_direction", np.array([1.0, 1.0, 1.0])
            )
            _, uvs = get_mesh_auto_uv(o3d_mesh, projection_direction)
            fixed_actor.set_uv_mapping(uvs)

        if material is not None:
            fixed_actor.set_material(material)

            if texture_path is not None:
                material.set_base_color_map(texture_path)

        fixed_actor.set_local_pose(init_pose)

        attr = PhysicalAttr()
        attr.mass = kwargs.get("mass", 0.01)
        # attr.density = density    # TODO: using default mass
        attr.contact_offset = kwargs.get("contact_offset", 0.006)
        attr.rest_offset = kwargs.get("rest_offset", 0.001)
        attr.dynamic_friction = damping_coeff
        attr.static_friction = damping_coeff
        attr.angular_damping = friction_coeff
        attr.linear_damping = friction_coeff
        attr.restitution = kwargs.get("restitution", 0.05)
        attr.min_position_iters = kwargs.get("min_position_iters", 4)
        attr.min_velocity_iters = kwargs.get("min_velocity_iters", 1)
        attr.max_depenetration_velocity = kwargs.get("max_depenetration_velocity", 10.0)
        fixed_actor.set_physical_attr(attr)

        body_name = fixed_actor.get_name()
        if arena_index >= 0:
            body_name = f"{body_name}_{arena_index}"

        self._fixed_actors[body_name] = (fixed_actor, arena_index)

        fixed_actor.node.attach_node(env.get_root_node())
        self._world.enable_physics(True)

        return fixed_actor

    def get_fixed_actor(self, actor_name: str, arena_index: int = -1) -> MeshObject:
        """Get fixed actor by name.

        Args:
            actor_name (str): name of fixed actor.
            arena_index (int, optional): the index of arena to get, -1 for global env. Defaults to -1.
        """
        if arena_index >= 0:
            actor_name = f"{actor_name}_{arena_index}"

        if actor_name not in self._fixed_actors:
            logger.log_warning(f"Fixed actor {actor_name} not found.")
            return None

        return self._fixed_actors[actor_name][0]

    def remove_fixed_actor(self, actor_name: str):
        """Remove fixed body with its name

        Args:
            body_name (str): key, md5 of mesh
        """
        if actor_name not in self._fixed_actors:
            logger.log_warning(f"Fixed actor {actor_name} not found.")
            return

        env = self.get_env(self._fixed_actors[actor_name][1])
        env.remove_actor(actor_name)
        self._fixed_actors.pop(actor_name)

    def get_fixed_actor_uid_list(self) -> List[str]:
        """Get current fixed body key list

        Returns:
            List[str]: list of fixed body key
        """
        return list(self._fixed_actors.keys())

    def add_dynamic_actor(
        self,
        fpath: str,
        init_pose: np.ndarray,
        scale: Union[List, np.ndarray] = [1.0, 1.0, 1.0],
        density: float = 1,
        damping_coeff: float = 0.7,
        friction_coeff: float = 0.9,
        max_depenetration_velocity: float = 10.0,
        material: Material = None,
        texture_path: str = None,
        is_convex_decomposition: bool = False,
        max_convex_hull_num: int = 32,
        **kwargs,
    ) -> MeshObject:
        """Add a dynamic rigid object.

        The dynamic object will have a dynamic physical body and will move during the simulation.

        Args:
            fpath (str): path to mesh file.
            init_pose (np.ndarray): object pose in world frame.
            scale (Union[List, np.ndarray], optional): scale of object. Defaults to [1.0, 1.0, 1.0].
            density (float, optional): density of object. Defaults to 1.
            damping_coeff (float, optional): damage coefficient. Defaults to 0.7.
            friction_coeff (float, optional): friction coefficient. Defaults to 0.9.
            max_depenetration_velocity (float, optional): maximum depenetration velocity. Defaults to 10.0.
            material (Material, optional): the material of object. Defaults to None.
            texture_path (str, optional): path to texture file. Defaults to None.
            is_convex_decomposition (bool, optional): whether to use convex decomposition. Defaults to False.
            max_convex_hull_num (int, optional): maximum convex hull number. Defaults to 32.
            **kwargs: other parameters for dynamic object creation.
                - arena_index (int): the index of arena to place the fixed object.
                - contact_offset (float): contact offset. Defaults to 0.015.
                - rest_offset (float): rest offset. Defaults to 0.000.
                - restitution (float): restitution coefficient. Defaults to 0.05.
                - min_position_iters (int): minimum position iterations. Defaults to 10.
                - min_velocity_iters (int): minimum velocity iterations. Defaults to 10.
                - projection_direction (np.ndarray): projection direction. Defaults to np.array([1.0, 1.0, 1.0]).
                - compute_uv (bool): whether to compute uv mapping. Defaults to False.

        Returns:
            MeshObject: The added dynamic object instance handle.
        """

        self._world.enable_physics(False)

        arena_index = kwargs.get("arena_index", -1)
        env = self.get_env(arena_index)

        duplicate = False if arena_index <= 0 else True

        if is_convex_decomposition:
            dynamic_actor = env.load_actor_with_coacd(
                fpath,
                duplicate=duplicate,
                attach_scene=False,
                cache_path=os.path.join(self._sim_cache_dir, "convex_decomposition"),
                actor_type=ActorType.DYNAMIC,
                max_convex_hull_num=max_convex_hull_num,
            )
        else:
            dynamic_actor = env.load_actor(fpath, duplicate, False)
            dynamic_actor.add_rigidbody(ActorType.DYNAMIC, RigidBodyShape.CONVEX)

        dynamic_actor.set_body_scale(scale[0], scale[1], scale[2])
        dynamic_actor.node.attach_node(env.get_root_node())
        compute_uv = kwargs.get("compute_uv", False)
        if material is not None:
            compute_uv = True
        if compute_uv and dynamic_actor.has_uv_mapping() is False:
            from dexsim.kit.meshproc import get_mesh_auto_uv

            vertices = dynamic_actor.get_vertices()
            triangles = dynamic_actor.get_triangles()

            o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
            projection_direction = kwargs.get(
                "projection_direction", np.array([1.0, 1.0, 1.0])
            )
            _, uvs = get_mesh_auto_uv(o3d_mesh, projection_direction)
            dynamic_actor.set_uv_mapping(uvs)

        if material is not None:
            dynamic_actor.set_material(material)

            if texture_path is not None:
                material.set_base_color_map(texture_path)

        dynamic_actor.set_local_pose(init_pose)

        attr = PhysicalAttr()
        attr.mass = kwargs.get("mass", 0.01)
        # attr.density = density    # TODO: using default mass
        attr.contact_offset = kwargs.get("contact_offset", 0.006)
        attr.rest_offset = kwargs.get("rest_offset", 0.001)
        attr.dynamic_friction = friction_coeff
        attr.static_friction = friction_coeff
        attr.angular_damping = damping_coeff
        attr.linear_damping = damping_coeff
        attr.restitution = kwargs.get("restitution", 0.05)
        attr.min_position_iters = kwargs.get("min_position_iters", 4)
        attr.min_velocity_iters = kwargs.get("min_velocity_iters", 1)
        attr.max_depenetration_velocity = max_depenetration_velocity

        dynamic_actor.set_physical_attr(attr)

        body_name = dynamic_actor.get_name()
        if arena_index >= 0:
            body_name = f"{body_name}_{arena_index}"

        self._dynamic_actors[body_name] = (dynamic_actor, arena_index)

        self._world.enable_physics(True)
        return dynamic_actor

    def get_dynamic_actor(self, actor_name: str, arena_index: int = -1) -> MeshObject:
        """Get dynamic actor by name.

        Args:
            actor_name (str): name of dynamic actor.
            arena_index (int, optional): the index of arena to get, -1 for global env. Defaults to -1.
        """
        if arena_index >= 0:
            actor_name = f"{actor_name}_{arena_index}"

        if actor_name not in self._dynamic_actors:
            logger.log_warning(f"Dynamic actor {actor_name} not found.")
            return None

        return self._dynamic_actors[actor_name][0]

    def remove_dynamic_actor(self, actor_name: str):
        """Remove dynamic body with its name.

        Args:
            actor_name (str): name of dynamic actor.
        """
        if actor_name not in self._dynamic_actors:
            logger.log_warning(f"Dynamic actor {actor_name} not found.")
            return

        env = self.get_env(self._dynamic_actors[actor_name][1])
        env.remove_actor(actor_name)
        self._dynamic_actors.pop(actor_name)
        # pop attach actor
        for robot_uid, robot in self._robots.items():
            if actor_name in robot.attached_actors:
                robot.attached_actors.pop(actor_name)
        for ef_uid, ef in self._end_effectors.items():
            if actor_name in ef._attached_nodes:
                ef._attached_nodes.pop(actor_name)

    def get_dynamic_actor_uid_list(self) -> List[str]:
        """Get current fixed body key list

        Returns:
            List[str]: list of fixed body key
        """
        return list(self._dynamic_actors.keys())

    def create_cube(
        self,
        name: str,
        length: float,
        width: float,
        height: float,
        actor_type: Optional[ActorType] = None,
        **kwargs,
    ) -> MeshObject:
        """Create a cube with given dimensions and actor type.

        Args:
            name (str): name of cube.
            length (float): length of cube.
            width (float): width of cube.
            height (float): height of cube.
            actor_type (Optional[ActorType], optional): actor type of cube. Defaults to None.
            **kwargs: other parameters for cube creation.
                - arena_index: the index of arena to place the cube.

        Returns:
            MeshObject: The created cube instance handle.
        """
        arena_index = kwargs.get("arena_index", -1)
        if arena_index >= 0:
            name = f"{name}_{arena_index}"

        env = self.get_env(arena_index)
        cube = env.create_cube(length, width, height)
        cube.set_name(name)

        if actor_type is not None:
            attr = PhysicalAttr()
            cube.add_rigidbody(actor_type, RigidBodyShape.CONVEX, attr)

        if actor_type == ActorType.DYNAMIC or actor_type == ActorType.KINEMATIC:
            self._dynamic_actors[name] = (cube, arena_index)
        else:
            self._fixed_actors[name] = (cube, arena_index)

        return cube

    def remove_asset(self, uid: str) -> bool:
        """Remove an asset by its UID.

        The asset can be a light, sensor, robot, rigid object or articulation.

        Note:
            Currently, lights and sensors are not supported to be removed.

        Args:
            uid (str): The UID of the asset.
        Returns:
            bool: True if the asset is removed successfully, otherwise False.
        """
        if uid in self._rigid_objects:
            obj = self._rigid_objects.pop(uid)
            obj.destroy()
            return True

        if uid in self._soft_objects:
            obj = self._soft_objects.pop(uid)
            obj.destroy()
            return True

        if uid in self._rigid_object_groups:
            group = self._rigid_object_groups.pop(uid)
            group.destroy()
            return True

        if uid in self._articulations:
            art = self._articulations.pop(uid)
            art.destroy()
            return True

        if uid in self._robots_v2:
            robot = self._robots_v2.pop(uid)
            robot.destroy()
            return True

        return False

    def get_asset(
        self, uid: str
    ) -> Optional[Union[RigidObject, Articulation, RobotV2, Light, BaseSensor]]:
        """Get an asset by its UID.

        The asset can be a rigid object, articulation or robot.

        Args:
            uid (str): The UID of the asset.
        """
        if uid in self._rigid_objects:
            return self._rigid_objects[uid]

        if uid in self._articulations:
            return self._articulations[uid]

        if uid in self._robots_v2:
            return self._robots_v2[uid]

        if uid in self._lights:
            return self._lights[uid]

        if uid in self._sensors_v2:
            return self._sensors_v2[uid]

        logger.log_warning(f"Asset {uid} not found.")
        return None

    def draw_marker(
        self,
        cfg: MarkerCfg,
    ) -> MeshObject:
        """Draw visual markers in the simulation scene for debugging and visualization.

        Args:
            cfg (MarkerCfg): Marker configuration with the following key parameters:
                - name (str): Unique identifier for the marker group
                - marker_type (str): Type of marker ("axis" currently supported)
                - axis_xpos (np.ndarray | List[np.ndarray]): 4x4 transformation matrices
                  for marker positions and orientations
                - axis_size (float): Thickness of axis arrows
                - axis_len (float): Length of axis arrows
                - arena_index (int): Arena index for placement (-1 for global)

        Returns:
            List[MeshObject]: List of created marker handles, False if invalid input,
            None if no poses provided.

        Example:
            ```python
            cfg = MarkerCfg(name="test_axis", marker_type="axis", axis_xpos=np.eye(4))
            markers = sim.draw_marker(cfg)
            ```
        """
        # Validate marker type
        if cfg.marker_type != "axis":
            logger.log_error(
                f"Unsupported marker type '{cfg.marker_type}'. Currently only 'axis' is supported."
            )
            return False

        draw_xpos = deepcopy(cfg.axis_xpos)
        draw_xpos = np.array(draw_xpos)
        if draw_xpos.ndim == 2:
            if draw_xpos.shape == (4, 4):
                draw_xpos = np.expand_dims(draw_xpos, axis=0)
            else:
                logger.log_error(
                    f"axis_xpos must be of shape (N, 4, 4), got {draw_xpos.shape}."
                )
                return False
        elif draw_xpos.ndim != 3 or draw_xpos.shape[1:] != (4, 4):
            logger.log_error(
                f"axis_xpos must be of shape (N, 4, 4), got {draw_xpos.shape}."
            )
            return False

        original_name = cfg.name
        name = original_name
        count = 0

        while name in self._fixed_actors:
            count += 1
            name = f"{original_name}_{count}"
        if count > 0:
            logger.log_warning(
                f"Marker name '{original_name}' already exists. Using '{name}'."
            )

        marker_num = len(draw_xpos)
        if marker_num == 0:
            logger.log_warning(f"No marker poses provided.")
            return None

        if cfg.arena_index >= 0:
            name = f"{name}_{cfg.arena_index}"

        env = self.get_env(cfg.arena_index)

        # Create markers based on marker type
        marker_handles = []

        if cfg.marker_type == "axis":
            # Create coordinate axes
            axis_option = dexsim.types.AxisOption(
                lx=cfg.axis_len,
                ly=cfg.axis_len,
                lz=cfg.axis_len,
                size=cfg.axis_size,
                arrow_type=cfg.arrow_type,
                corner_type=cfg.corner_type,
                tag_type=dexsim.types.AxisTagType.NONE,
            )

            for i, pose in enumerate(draw_xpos):
                axis_handle = env.create_axis(axis_option)
                axis_handle.set_local_pose(pose)
                marker_handles.append(axis_handle)

        # TODO: Add support for other marker types in the future
        # elif cfg.marker_type == "line":
        #     # Create line markers
        #     pass
        # elif cfg.marker_type == "point":
        #     # Create point markers
        #     pass

        self._fixed_actors[name] = (marker_handles, cfg.arena_index)
        return marker_handles

    def remove_marker(self, name: str) -> bool:
        """Remove markers (including axis) with the given name.

        Args:
            name (str): The name of the marker to remove.
        Returns:
            bool: True if the marker was removed successfully, False otherwise.
        """
        if name not in self._fixed_actors:
            logger.log_warning(f"Marker {name} not found.")
            return False
        try:
            env = self.get_env(self._fixed_actors[name][1])
            marker_handles, arena_index = self._fixed_actors[name]
            for marker_handle in marker_handles:
                if marker_handle is not None:
                    env.remove_actor(marker_handle.get_name())
            self._fixed_actors.pop(name)
            return True
        except Exception as e:
            logger.log_warning(f"Failed to remove marker {name}: {str(e)}")
            return False

    def create_material(
        self, name: str, type: str = "color", num_inst: int = 1, **kwargs
    ) -> Material:
        """Create a material with given type and parameters.

        Material is dexsim.models.Material, which can have multiple instances with shared pbr attributes.
        For each instance, the properties can be set individually.

        Note:
            The `name` is just the key of :attr:`_materials` dictionary.
            If num_inst == 1, the material instance will have the same name.
            If num_inst > 1, the material instance will have the name with suffix "_i" where i is the index of instance.

        Args:
            name (str): name of material.
            type (str, optional): material type. Defaults to "color".
            num_inst (int, optional): number of instances of material. Defaults to 1.
            **kwargs: other parameters for material creation.
                - rgba: the rgba of color material.
                - rgb: the rgb of color material.
                - has_alpha: whether the pbr material has alpha channel.

        Returns:
            Material: the created material instance handle.
        """

        if name in self._materials:
            logger.log_warning(f"Material {name} already exists. Returning it.")
            return self._materials[name]

        if type == "color":
            color = kwargs.get("rgb", None)
            if color is None:
                color = kwargs.get("rgba", None)
            if color is None:
                color = [0.5, 0.5, 0.5]

            has_alpha = len(color) == 4
            if num_inst == 1:
                mat = self._env.create_color_material(color, name, has_alpha)
            else:
                inst_name = f"{name}_0"
                mat = self._env.create_color_material(color, inst_name, has_alpha)
                for i in range(1, num_inst):
                    inst_name = f"{name}_{i}"
                    mat.get_inst(inst_name)
                    mat.set_inst_color(color, inst_name)

        elif type == "pbr":
            has_alpha = kwargs.get("has_alpha", False)
            if num_inst == 1:
                mat = self._env.create_pbr_material(name, has_alpha)
            else:
                inst_name = f"{name}_0"
                mat = self._env.create_pbr_material(inst_name, has_alpha)
                for i in range(1, num_inst):
                    inst_name = f"{name}_{i}"
                    mat.get_inst(inst_name)

        else:
            logger.log_error(f"Unsupported material type {type}.")

        self._materials[name] = mat
        return mat

    def load_material(self, path: str, name: str, num_inst: int = 1) -> Material:
        """Load a PBR material from file.

        Args:
            path (str): path to material.
            name (str): name of material.
            num_inst (int, optional): number of instances of material. Defaults to 1.

        Returns:
            Material: the loaded material instance handle.
        """
        if name in self._materials:
            logger.log_warning(f"Material {name} already exists. Returning it.")
            return self._materials[name]

        if num_inst == 1:
            mat = self._env.load_material(path, name)
        else:
            inst_name = f"{name}_0"
            mat = self._env.load_material(path, inst_name)
            for i in range(1, num_inst):
                inst_name = f"{name}_{i}"
                mat.get_inst(inst_name)

        mat.set_roughness(0.5)

        self._materials[name] = mat
        return mat

    def get_material(self, name: str) -> Material:
        """Get material by name.

        Args:
            name (str): name of material.
        """

        # TODO: Temporary workaround for plane material
        # Will be removed in the future
        if name == "plane_mat":
            return self.get_visual_material("plane_mat").get_instance("plane_mat").mat

        if name not in self._materials:
            logger.log_warning(f"Material {name} not found.")
            return None

        return self._materials[name]

    def create_visual_material(self, cfg: VisualMaterialCfg) -> VisualMaterial:
        """Create a visual material with given configuration.

        Args:
            cfg (VisualMaterialCfg): configuration for the visual material.

        Returns:
            VisualMaterial: the created visual material instance handle.
        """

        if cfg.uid in self._visual_materials:
            logger.log_warning(
                f"Visual material {cfg.uid} already exists. Returning the existing one."
            )
            return self._visual_materials[cfg.uid]

        mat: Material = self._env.create_pbr_material(cfg.uid, True)
        visual_mat = VisualMaterial(cfg, mat)

        self._visual_materials[cfg.uid] = visual_mat
        return visual_mat

    def get_visual_material(self, uid: str) -> VisualMaterial:
        """Get visual material by UID.

        Args:
            uid (str): uid of visual material.
        """
        if uid not in self._visual_materials:
            logger.log_warning(f"Visual material {uid} not found.")
            return None

        return self._visual_materials[uid]

    def clean_materials(self):
        self._materials = {}
        self._visual_materials = {}
        self._env.clean_materials()

    def reset_objects_state(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Reset the state of all objects in the scene.

        Args:
            env_ids: The environment IDs to reset. If None, reset all environments.
        """
        for robot in self._robots_v2.values():
            robot.reset(env_ids)
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        for rigid_obj in self._rigid_objects.values():
            rigid_obj.reset(env_ids)
        for rigid_obj_group in self._rigid_object_groups.values():
            rigid_obj_group.reset(env_ids)
        for light in self._lights.values():
            light.reset(env_ids)
        for sensor in self._sensors_v2.values():
            sensor.reset(env_ids)

    def destroy(self) -> None:
        """Destroy all simulated assets and release resources."""
        # Clean up all gizmos before destroying the simulation
        for uid in list(self._gizmos.keys()):
            self.disable_gizmo(uid)

        self.clean_materials()

        self._env.clean()
        self._world.quit()
