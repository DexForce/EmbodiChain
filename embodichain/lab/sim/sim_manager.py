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

from __future__ import annotations

import os
import sys
import dexsim
import torch
import numpy as np
import warp as wp

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from functools import cached_property
from typing import List, Union, Dict, Union, Sequence
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
from dexsim.render import Light as _Light, LightType, Windows
from dexsim.engine import GizmoController, ObjectManipulator

from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectGroup,
    SoftObject,
    Articulation,
    Robot,
    Light,
)
from embodichain.lab.sim.objects.gizmo import Gizmo
from embodichain.lab.sim.sensors import (
    SensorCfg,
    BaseSensor,
    Camera,
    StereoCamera,
    ContactSensor,
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
from embodichain.utils import configclass, logger

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

    cpu_num: int = 1
    """The number of CPU threads to use for the simulation engine."""

    num_envs: int = 1
    """The number of parallel environments (arenas) to simulate."""

    arena_space: float = 5.0
    """The distance between each arena when building multiple arenas."""

    physics_dt: float = 1.0 / 100.0
    """The time step for the physics simulation."""

    sim_device: Union[str, torch.device] = "cpu"
    """The device for the physics simulation. Can be 'cpu', 'cuda', or a torch.device object."""

    physics_config: PhysicsCfg = field(default_factory=PhysicsCfg)
    """The physics configuration parameters."""
    gpu_memory_config: GPUMemoryCfg = field(default_factory=GPUMemoryCfg)
    """The GPU memory configuration parameters."""


class SimulationManager:
    r"""Global Embodied AI simulation manager.

    This class is used to manage the global simulation environment and simulated assets.
        - assets loading, creation, modification and deletion.
            - assets include rigid objects, soft objects, articulations, robots, sensors and lights.
        - manager the scenes and the simulation environment.
            - parallel scenes simulation on both CPU and GPU.
            - create and setup the rendering related settings, eg. environment map, lighting, materials, etc.
            - physics simulation management, eg. time step, manual update, etc.
            - interactive control via gizmo and window callbacks events.

    Args:
        sim_config (SimulationManagerCfg, optional): simulation configuration. Defaults to SimulationManagerCfg().
    """

    _instances = {}

    SUPPORTED_SENSOR_TYPES = {
        "Camera": Camera,
        "StereoCamera": StereoCamera,
        "ContactSensor": ContactSensor,
    }

    def __new__(cls, sim_config: SimulationManagerCfg = SimulationManagerCfg()):
        """Create or return the instance based on instance_id."""
        n_instance = len(list(cls._instances.keys()))
        instance = super(SimulationManager, cls).__new__(cls)
        # Store sim_config in the instance for use in __init__ or elsewhere
        instance.sim_config = sim_config
        cls._instances[n_instance] = instance
        return instance

    def __init__(
        self, sim_config: SimulationManagerCfg = SimulationManagerCfg()
    ) -> None:
        instance_id = SimulationManager.get_instance_num() - 1

        # Mark as initialized
        self.instance_id = instance_id

        if sim_config.enable_rt and instance_id > 0:
            logger.log_error(
                f"Ray Tracing rendering backend is only supported for single instance (instance_id=0). "
            )

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
        self._world: dexsim.World = dexsim.World(world_config)

        self._window: Windows | None = None
        self._is_registered_window_control = False

        fps = int(1.0 / sim_config.physics_dt)
        self._world.set_physics_fps(fps)

        self._world.set_time_scale(1.0)
        self._world.set_delta_time(sim_config.physics_dt)
        self._world.show_coordinate_axis(False)

        dexsim.set_physics_config(**sim_config.physics_config.to_dexsim_args())
        dexsim.set_physics_gpu_memory_config(**sim_config.gpu_memory_config.to_dict())

        self._is_initialized_gpu_physics = False
        self._ps = self._world.get_physics_scene()

        # activate physics
        self.enable_physics(True)

        self._env = self._world.get_env()

        # set unique material path to accelerate material creation.
        # TODO: This will be removed.
        if self.sim_config.enable_rt is False:
            self._env.set_unique_mat_path(
                os.path.join(self._material_cache_dir, "default_mat")
            )

        # arena is used as a standalone space for robots to simulate in.
        self._arenas: List[dexsim.environment.Arena] = []

        # gizmo management
        self._gizmos: Dict[str, object] = dict()  # Store active gizmos

        # marker management
        self._markers: Dict[str, MeshObject] = dict()

        self._rigid_objects: Dict[str, RigidObject] = dict()
        self._rigid_object_groups: Dict[str, RigidObjectGroup] = dict()
        self._soft_objects: Dict[str, SoftObject] = dict()
        self._articulations: Dict[str, Articulation] = dict()
        self._robots: Dict[str, Robot] = dict()

        self._sensors: Dict[str, BaseSensor] = dict()
        self._lights: Dict[str, _Light] = dict()

        # material placeholder.
        self._visual_materials: Dict[str, VisualMaterial] = dict()

        # Global texture cache for material creation or randomization.
        # The structure is keys to the loaded texture data. The keys represent the texture group.
        self._texture_cache: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = dict()

        self._init_sim_resources()

        self._create_default_plane()
        self.set_default_background()

        # Set physics to manual update mode by default.
        self.set_manual_update(True)

        self._build_multiple_arenas(sim_config.num_envs)

        if sim_config.headless is False:
            self._window = self._world.get_windows()
            self._register_default_window_control()

    @classmethod
    def get_instance(cls, instance_id: int = 0) -> SimulationManager:
        """Get the instance of SimulationManager by id.

        Args:
            instance_id (int): The instance id. Defaults to 0.

        Returns:
            SimulationManager: The instance.

        Raises:
            RuntimeError: If the instance has not been created yet.
        """
        if instance_id not in cls._instances:
            logger.log_error(
                f"SimulationManager (id={instance_id}) has not been instantiated yet. "
                f"Create an instance first using SimulationManager(sim_config, instance_id={instance_id})."
            )
        return cls._instances[instance_id]

    @classmethod
    def get_instance_num(cls) -> int:
        """Get the number of instantiated SimulationManager instances.

        Returns:
            int: The number of instances.
        """
        return len(cls._instances)

    @classmethod
    def reset(cls, instance_id: int = 0) -> None:
        """Reset the instance.

        This allows creating a new instance with different configuration.
        """
        if instance_id in cls._instances:
            logger.log_info(f"Resetting SimulationManager instance {instance_id}.")
            del cls._instances[instance_id]

    @classmethod
    def is_instantiated(cls, instance_id: int = 0) -> bool:
        """Check if the instance has been created.

        Returns:
            bool: True if the instance exists, False otherwise.
        """
        return instance_id in cls._instances

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
        uid_list.extend(list(self._sensors.keys()))
        uid_list.extend(list(self._robots.keys()))
        uid_list.extend(list(self._rigid_objects.keys()))
        uid_list.extend(list(self._rigid_object_groups.keys()))
        uid_list.extend(list(self._soft_objects.keys()))
        uid_list.extend(list(self._articulations.keys()))
        return uid_list

    def _convert_sim_config(
        self, sim_config: SimulationManagerCfg
    ) -> dexsim.WorldConfig:
        world_config = dexsim.WorldConfig()
        win_config = dexsim.WindowsConfig()
        win_config.width = sim_config.width
        win_config.height = sim_config.height
        world_config.cpu_num = sim_config.cpu_num
        world_config.win_config = win_config
        world_config.open_windows = not sim_config.headless
        self.is_window_opened = not sim_config.headless
        world_config.backend = Backend.VULKAN
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

    def _init_sim_resources(self) -> None:
        """Initialize the default simulation resources."""
        from embodichain.data.assets import SimResources

        self._default_resources = SimResources()

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
            else len(self._ps.get_gpu_rigid_indices())
        )
        self._rigid_body_pose = torch.zeros(
            (rigid_body_num, 7), dtype=torch.float32, device=self.device
        )

        # init articulation.
        articulation_num = (
            0
            if len(self._articulations) == 0 and len(self._robots) == 0
            else len(self._ps.get_gpu_articulation_indices())
        )
        max_link_count = self._ps.gpu_get_articulation_max_link_count()
        self._link_pose = torch.zeros(
            (articulation_num, max_link_count, 7),
            dtype=torch.float32,
            device=self.device,
        )
        for art in self._articulations.values():
            art.reallocate_body_data()
        for robot in self._robots.values():
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

    def update(self, physics_dt: float | None = None, step: int = 10) -> None:
        """Update the physics.

        Args:
            physics_dt (float | None, optional): the time step for physics simulation. Defaults to None.
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

        if self.is_window_opened or self._sensors:
            if len(self._rigid_body_pose) > 0:
                self._ps.gpu_fetch_rigid_body_data(
                    data=CudaArray(self._rigid_body_pose),
                    gpu_indices=self._ps.get_gpu_rigid_indices(),
                    data_type=RigidBodyGPUAPIReadType.POSE,
                )

            if len(self._link_pose) > 0:
                self._ps.gpu_fetch_link_data(
                    data=CudaArray(self._link_pose),
                    gpu_indices=self._ps.get_gpu_articulation_indices(),
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
        self._window = self._world.get_windows()
        self._register_default_window_control()
        self.is_window_opened = True

    def close_window(self) -> None:
        """Close the simulation window."""
        self._world.close_window()
        self.is_window_opened = False

    def _build_multiple_arenas(self, num: int, space: float | None = None) -> None:
        """Build multiple arenas in a grid pattern.

        This interface is used for vectorized simulation.

        Args:
            num (int): number of arenas to build.
            space (float | None, optional): The distance between each arena. Defaults to the arena_space in sim_config.
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
        self, color: Sequence[float] | None = None, intensity: float | None = None
    ) -> None:
        """Set environment emission light.

        Args:
            color (Sequence[float] | None): color of the light.
            intensity (float | None): intensity of the light.
        """
        if color is not None:
            self._env.set_env_light_emission(color)
        if intensity is not None:
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
            self.set_emission_light([1.0, 1.0, 1.0], 80.0)
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
        self, key: str | None = None
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """Get the texture from the global texture cache.

        Args:
            key (str | None, optional): The key of the texture. If None, return None. Defaults to None.

        Returns:
            torch.Tensor | list[torch.Tensor] | None: The texture if found, otherwise None.
        """
        if key is None:
            return self._texture_cache

        if key not in self._texture_cache:
            logger.log_warning(f"Texture {key} not found in global texture cache.")
            return None
        return self._texture_cache[key]

    def get_asset(
        self, uid: str
    ) -> Light | BaseSensor | Robot | RigidObject | Articulation | None:
        """Get an asset by its UID.

        The asset can be a light, sensor, robot, rigid object or articulation.

        Args:
            uid (str): The UID of the asset.

        Returns:
            Light | BaseSensor | Robot | RigidObject | Articulation | None: The asset instance if found, otherwise None.
        """
        if uid in self._lights:
            return self._lights[uid]
        if uid in self._sensors:
            return self._sensors[uid]
        if uid in self._robots:
            return self._robots[uid]
        if uid in self._rigid_objects:
            return self._rigid_objects[uid]
        if uid in self._rigid_object_groups:
            return self._rigid_object_groups[uid]
        if uid in self._soft_objects:
            return self._soft_objects[uid]
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

    def get_light(self, uid: str) -> Light | None:
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

    def get_rigid_object(self, uid: str) -> RigidObject | None:
        """Get a rigid object by its unique ID.

        Args:
            uid (str): The unique ID of the rigid object.

        Returns:
            RigidObject | None: The rigid object instance if found, otherwise None.
        """
        if uid not in self._rigid_objects:
            logger.log_warning(f"Rigid object {uid} not found.")
            return None
        return self._rigid_objects[uid]

    def get_soft_object(self, uid: str) -> SoftObject | None:
        """Get a soft object by its unique ID.

        Args:
            uid (str): The unique ID of the soft object.

        Returns:
            SoftObject | None: The soft object instance if found, otherwise None.
        """
        if uid not in self._soft_objects:
            logger.log_warning(f"Soft object {uid} not found.")
            return None
        return self._soft_objects[uid]

    def get_rigid_object_uid_list(self) -> List[str]:
        """Get current rigid body uid list

        Returns:
            List[str]: list of rigid body uid.
        """
        return list(self._rigid_objects.keys())

    def get_soft_object_uid_list(self) -> List[str]:
        """Get current soft body uid list

        Returns:
            List[str]: list of soft body uid.
        """
        return list(self._soft_objects.keys())

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

    def get_rigid_object_group(self, uid: str) -> RigidObjectGroup | None:
        """Get a rigid object group by its unique ID.

        Args:
            uid (str): The unique ID of the rigid object group.

        Returns:
            RigidObjectGroup | None: The rigid object group instance if found, otherwise None.
        """
        if uid not in self._rigid_object_groups:
            logger.log_warning(f"Rigid object group {uid} not found.")
            return None
        return self._rigid_object_groups[uid]

    def get_rigid_object_group_uid_list(self) -> List[str]:
        """Get current rigid body group uid list

        Returns:
            List[str]: list of rigid body group uid.
        """
        return list(self._rigid_object_groups.keys())

    @cached_property
    def arena_offsets(self) -> torch.Tensor:
        """Get the arena offsets for all arenas.

        Returns:
            torch.Tensor: The arena offsets of shape (num_arenas, 3).
        """
        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        arena_offsets = torch.zeros(
            (len(env_list), 3), dtype=torch.float32, device=self.device
        )
        for i, env in enumerate(env_list):
            arena_position = env.get_root_node().get_world_pose()[:3, 3]
            arena_offsets[i] = torch.tensor(
                arena_position, dtype=torch.float32, device=self.device
            )
        return arena_offsets

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

    def get_articulation(self, uid: str) -> Articulation | None:
        """Get an articulation by its unique ID.

        Args:
            uid (str): The unique ID of the articulation.

        Returns:
            Articulation | None: The articulation instance if found, otherwise None.
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

    def add_robot(self, cfg: RobotCfg) -> Robot | None:
        """Add a Robot to the scene.

        Args:
            cfg (RobotCfg): Configuration for the robot.

        Returns:
            Robot | None: The added robot instance handle, or None if failed.
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
        if uid in self._robots:
            logger.log_error(f"Robot {uid} already exists.")
            return self._robots[uid]

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = []

        for env in env_list:
            art = env.load_urdf(cfg.fpath)
            obj_list.append(art)

        robot = Robot(cfg=cfg, entities=obj_list, device=self.device)

        self._robots[uid] = robot

        return robot

    def get_robot(self, uid: str) -> Robot | None:
        """Get a Robot by its unique ID.

        Args:
            uid (str): The unique ID of the robot.

        Returns:
            Robot | None: The robot instance if found, otherwise None.
        """
        if uid not in self._robots:
            logger.log_warning(f"Robot {uid} not found.")
            return None
        return self._robots[uid]

    def get_robot_uid_list(self) -> List[str]:
        """
        Retrieves a list of unique identifiers (UIDs) for all robots in the V2 system.

        Returns:
            list: A list containing the UIDs of the robots.
        """
        return list(self._robots.keys())

    def enable_gizmo(
        self, uid: str, control_part: str | None = None, gizmo_cfg: object = None
    ) -> None:
        """Enable gizmo control for any simulation object (Robot, RigidObject, Camera, etc.).

        Args:
            uid (str): UID of the object to attach gizmo to (searches in robots, rigid_objects, sensors, etc.)
            control_part (str | None, optional): Control part name for robots. Defaults to "arm".
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

        if uid in self._robots:
            target = self._robots[uid]
            object_type = "robot"
        elif uid in self._rigid_objects:
            target = self._rigid_objects[uid]
            object_type = "rigid_object"
        elif uid in self._sensors:
            target = self._sensors[uid]
            object_type = "sensor"

        else:
            logger.log_error(
                f"Object with uid '{uid}' not found in any collection (robots, rigid_objects, sensors, articulations)."
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
                window = (
                    self._world.get_windows()
                    if hasattr(self._world, "get_windows")
                    else None
                )
                self._gizmo_controller = GizmoController()
                window.add_input_control(self._gizmo_controller)

        except Exception as e:
            logger.log_error(
                f"Failed to create gizmo for {object_type} '{uid}' with control_part '{control_part}': {e}"
            )

    def disable_gizmo(self, uid: str, control_part: str | None = None) -> None:
        """Disable and remove gizmo for a robot.

        Args:
            uid (str): Object UID to disable gizmo for
            control_part (str | None, optional): Control part name for robots. Defaults to None.
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

    def get_gizmo(self, uid: str, control_part: str | None = None) -> object:
        """Get gizmo instance for a robot.

        Args:
            uid (str): Object UID
            control_part (str | None, optional): Control part name for robots. Defaults to None.

        Returns:
            object: Gizmo instance if found, None otherwise.
        """
        # Create gizmo key combining uid and control_part
        gizmo_key = f"{uid}:{control_part}" if control_part else uid
        return self._gizmos.get(gizmo_key, None)

    def has_gizmo(self, uid: str, control_part: str | None = None) -> bool:
        """Check if a gizmo exists for the given UID and control part.

        Args:
            uid (str): Object UID to check
            control_part (str | None, optional): Control part name for robots. Defaults to None.

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
        self, uid: str, control_part: str | None = None
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
        self, uid: str, visible: bool, control_part: str | None = None
    ) -> None:
        """
        Set the visibility of a gizmo by uid and optional control_part.
        """
        gizmo = self.get_gizmo(uid, control_part)
        if gizmo is not None:
            gizmo.set_visible(visible)

    def add_sensor(self, sensor_cfg: SensorCfg) -> BaseSensor:
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
            sensor_uid = f"{sensor_type.lower()}_{len(self._sensors)}"
            sensor_cfg.uid = sensor_uid

        if sensor_uid in self._sensors:
            logger.log_warning(f"Sensor {sensor_uid} already exists.")
            return None

        sensor = self.SUPPORTED_SENSOR_TYPES[sensor_type](sensor_cfg, self.device)

        self._sensors[sensor_uid] = sensor

        # Check if the sensor needs to change the parent frame.

        return sensor

    def get_sensor(self, uid: str) -> BaseSensor | None:
        """Get a sensor by its UID.

        Args:
            uid (str): The UID of the sensor.

        Returns:
            BaseSensor | None: The sensor instance if found, otherwise None.
        """
        if uid not in self._sensors:
            logger.log_warning(f"Sensor {uid} not found.")
            return None
        return self._sensors[uid]

    def get_sensor_uid_list(self) -> List[str]:
        """Get current sensor uid list

        Returns:
            List[str]: list of sensor uid.
        """
        return list(self._sensors.keys())

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

        if uid in self._robots:
            robot = self._robots.pop(uid)
            robot.destroy()
            return True

        return False

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

        while name in self._markers:
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

        self._markers[name] = (marker_handles, cfg.arena_index)

        if self.is_physics_manually_update:
            self.update(step=1)

        return marker_handles

    def remove_marker(self, name: str) -> bool:
        """Remove markers (including axis) with the given name.

        Args:
            name (str): The name of the marker to remove.
        Returns:
            bool: True if the marker was removed successfully, False otherwise.
        """
        if name not in self._markers:
            logger.log_warning(f"Marker {name} not found.")
            return False
        try:
            env = self.get_env(self._markers[name][1])
            marker_handles, arena_index = self._markers[name]
            for marker_handle in marker_handles:
                if marker_handle is not None:
                    env.remove_actor(marker_handle.get_name())
            self._markers.pop(name)
            return True
        except Exception as e:
            logger.log_warning(f"Failed to remove marker {name}: {str(e)}")
            return False

    def _register_default_window_control(self) -> None:
        """Register default window controls for better simulation interaction."""
        from dexsim.types import InputKey

        # TODO: window control has stucking issue with extra sensor under Raster renderer backend.
        # Will be fixed in next dexsim release.
        if self.is_rt_enabled is False:
            return

        if self._is_registered_window_control:
            return

        class WindowDefaultEvent(ObjectManipulator):

            def on_key_down(self, key):
                if key == InputKey.SCANCODE_C.value:
                    print(f"Raycast distance: {self.selected_distance}")
                    print(f"Hit position: {self.selected_position}")

        manipulator = WindowDefaultEvent()
        manipulator.enable_selection_cache(True)
        self._window.add_input_control(manipulator)

        self._is_registered_window_control = True

    def add_custom_window_control(self, controls: list[ObjectManipulator]) -> None:
        """Add one or more custom window input controls.

        This method registers additional :class:`ObjectManipulator` instances
        with the simulation window so they can handle input events alongside
        any default controls.

        Args:
            controls (list[ObjectManipulator]): A list of initialized
                ObjectManipulator instances to add to the current window.
                Each control will be registered via ``window.add_input_control``.
                If no window is available, the controls are not added and a
                warning is logged.
        """
        if self._window is None:
            logger.log_warning("No window available to add custom controls.")
            return

        for control in controls:
            self._window.add_input_control(control)

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
        self._visual_materials = {}
        self._env.clean_materials()

    def reset_objects_state(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the state of all objects in the scene.

        Args:
            env_ids (Sequence[int] | None): The environment IDs to reset. If None, reset all environments.
        """
        for robot in self._robots.values():
            robot.reset(env_ids)
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        for rigid_obj in self._rigid_objects.values():
            rigid_obj.reset(env_ids)
        for rigid_obj_group in self._rigid_object_groups.values():
            rigid_obj_group.reset(env_ids)
        for light in self._lights.values():
            light.reset(env_ids)
        for sensor in self._sensors.values():
            sensor.reset(env_ids)

    def destroy(self) -> None:
        """Destroy all simulated assets and release resources."""
        # Clean up all gizmos before destroying the simulation
        for uid in list(self._gizmos.keys()):
            self.disable_gizmo(uid)

        self.clean_materials()

        self._env.clean()
        self._world.quit()

        SimulationManager.reset(self.instance_id)
