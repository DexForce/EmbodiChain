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

import os
import gc
import sys
import queue
import time
import threading
import dexsim
import torch
import numpy as np
import warp as wp

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from typing import Callable, Dict, List, Sequence, Union
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
from dexsim.core import TASK_RETURN
from dexsim.engine import CudaArray, Material
from dexsim.models import MeshObject
from dexsim.render import Light as _Light, LightType, Windows
from dexsim.engine import GizmoController, ObjectManipulator

from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectGroup,
    SoftObject,
    ClothObject,
    Articulation,
    Robot,
    Light,
    RigidConstraint,
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
    RenderCfg,
    PhysicsCfg,
    MarkerCfg,
    GPUMemoryCfg,
    WindowRecordCfg,
    WindowCameraPoseCfg,
    LightCfg,
    RigidObjectCfg,
    SoftObjectCfg,
    ClothObjectCfg,
    RigidObjectGroupCfg,
    ArticulationCfg,
    RobotCfg,
    RigidConstraintCfg,
)
from embodichain.lab.sim import VisualMaterial, VisualMaterialCfg
from embodichain.utils import configclass, logger
from embodichain.utils.math import look_at_to_pose, pose_inv

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

    render_cfg: RenderCfg = field(default_factory=RenderCfg)
    """The rendering configuration parameters."""

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

    window_record: WindowRecordCfg = field(default_factory=WindowRecordCfg)
    """Viewer window recording settings (hotkey, paths, FPS, memory budget)."""

    window_camera_pose: WindowCameraPoseCfg = field(default_factory=WindowCameraPoseCfg)
    """Interactive viewer camera-pose printing settings."""


@dataclass
class _WindowRecordState:
    """Internal state for simulation recording."""

    time_step: float
    max_memory_bytes: int
    output_dir: str
    video_name: str
    save_kwargs: dict[str, object]
    record_camera: object | None = None
    pose_provider: Callable[[], np.ndarray] | None = None
    fixed_pose: np.ndarray | None = None
    frames: list[np.ndarray] = field(default_factory=list)
    current_memory_bytes: int = 0
    last_capture_time: float = field(default_factory=time.time)
    accumulated_sim_time: float = 0.0
    capture_from_sim_update: bool = False
    task_status: int = TASK_RETURN.TASK_LOOP
    loop_handle: object | None = None


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

    _cleanup_queue: queue.Queue = queue.Queue()

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
        self._window_record_state: _WindowRecordState | None = None
        self._window_record_camera: object | None = None
        wr = sim_config.window_record
        self._window_record_hotkey_cfg: dict[str, object] | None = (
            {
                "save_path": wr.save_path,
                "fps": wr.fps,
                "max_memory": wr.max_memory,
                "video_prefix": wr.video_prefix,
            }
            if wr.enable_hotkey
            else None
        )
        self._window_record_input_control: ObjectManipulator | None = None
        self._window_record_save_threads: list[threading.Thread] = []
        wcp = sim_config.window_camera_pose
        self._window_camera_pose_hotkey_cfg: dict[str, object] | None = (
            {"convert_to_look_at": wcp.convert_to_look_at}
            if wcp.enable_hotkey
            else None
        )
        self._window_camera_pose_input_control: ObjectManipulator | None = None

        self._world.set_delta_time(sim_config.physics_dt)
        self._world.show_coordinate_axis(False)

        dexsim.set_physics_config(**sim_config.physics_config.to_dexsim_args())
        dexsim.set_physics_gpu_memory_config(**sim_config.gpu_memory_config.to_dict())

        self._is_initialized_gpu_physics = False
        self._ps = self._world.get_physics_scene()

        # activate physics
        self.enable_physics(True)

        self._env = self._world.get_env()

        # arena is used as a standalone space for robots to simulate in.
        self._arenas: List[dexsim.environment.Arena] = []

        # gizmo management
        self._gizmos: Dict[str, object] = dict()  # Store active gizmos

        # marker management
        self._markers: Dict[str, MeshObject] = dict()

        self._rigid_objects: Dict[str, RigidObject] = dict()
        self._constraints: Dict[str, RigidConstraint] = dict()
        self._rigid_object_groups: Dict[str, RigidObjectGroup] = dict()
        self._soft_objects: Dict[str, SoftObject] = dict()
        self._cloth_objects: Dict[str, ClothObject] = dict()
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
        self.set_default_global_lighting()

        # Set physics to manual update mode by default.
        self.set_manual_update(True)

        self._build_multiple_arenas(sim_config.num_envs)

        if sim_config.headless is False:
            self._window = self._world.get_windows()

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
            logger.log_debug(f"Resetting SimulationManager instance {instance_id}.")
            del cls._instances[instance_id]

    @classmethod
    def is_instantiated(cls, instance_id: int = 0) -> bool:
        """Check if the instance has been created.

        Returns:
            bool: True if the instance exists, False otherwise.
        """
        return instance_id in cls._instances

    @classmethod
    def set_default_renderer(cls, renderer: str = "auto", gpu_id: int = 0) -> str:
        """Set the global default renderer used by new simulations.

        This updates :data:`embodichain.lab.sim.cfg.DEFAULT_RENDERER`, which is
        consulted by :func:`embodichain.lab.sim.utility.render_utils.select_default_renderer`
        when ``render_cfg.renderer="auto"`` is resolved during :class:`SimulationManager`
        construction.

        Args:
            renderer: The renderer to set. One of ``"auto"``, ``"hybrid"``,
                ``"fast-rt"``, or ``"rt"``. When ``"auto"``, the renderer is
                resolved immediately from the detected GPU via
                :func:`embodichain.lab.sim.utility.render_utils.select_default_renderer`.
            gpu_id: The CUDA device index to query when ``renderer="auto"``.

        Returns:
            The resolved renderer name that was set as the default.
        """
        from embodichain.lab.sim import cfg
        from embodichain.lab.sim.utility.render_utils import select_default_renderer

        valid = {"auto", "hybrid", "fast-rt", "rt"}
        if renderer not in valid:
            logger.log_error(
                f"Invalid renderer '{renderer}'. Must be one of {sorted(valid)}."
            )

        if renderer == "auto":
            # Force auto-detection regardless of any previously forced default.
            cfg.DEFAULT_RENDERER = "auto"
            resolved = select_default_renderer(gpu_id)
        else:
            resolved = renderer

        cfg.DEFAULT_RENDERER = resolved
        logger.log_info(f"Default renderer set to '{resolved}'.")
        return resolved

    @cached_property
    def num_envs(self) -> int:
        """Get the number of arenas in the simulation.

        Returns:
            int: number of arenas.
        """
        return len(self._arenas) if len(self._arenas) > 0 else 1

    @property
    def is_use_gpu_physics(self) -> bool:
        """Check if the physics simulation is using GPU."""
        return self.device.type == "cuda"

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
        uid_list.extend(list(self._cloth_objects.keys()))
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

        if sim_config.render_cfg.renderer == "auto":
            from embodichain.lab.sim.utility.render_utils import (
                select_default_renderer,
            )

            resolved_renderer = select_default_renderer(sim_config.gpu_id)
            logger.log_info(
                f"Auto-selected '{resolved_renderer}' renderer for gpu_id={sim_config.gpu_id}."
            )
            sim_config.render_cfg.renderer = resolved_renderer

        world_config.renderer = sim_config.render_cfg.to_dexsim_flags()
        world_config.raytrace_config.render_iterations_per_frame = (
            sim_config.render_cfg.spp
        )

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

        for art in self._articulations.values():
            art.reallocate_body_data()
        for robot in self._robots.values():
            robot.reallocate_body_data()

        # Re-establish rigid object positions after articulation resets, ensuring
        # no articulation kinematics step has inadvertently corrupted the broadphase
        # state for rigid bodies.
        for rigid_obj in self._rigid_objects.values():
            rigid_obj.reset()

        self._is_initialized_gpu_physics = True

    def render_camera_group(self, group_ids: list[int]) -> None:
        """Render all camera group in the simulation.

        Args:
            group_ids (list[int]): The list of camera group ids to render.

        Note: This interface is only valid when Ray Tracing rendering backend is enabled.
        """

        self._world.render_camera_group(group_ids)

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
                if (
                    self._window_record_state is not None
                    and self._window_record_state.capture_from_sim_update
                ):
                    self._step_window_record_from_sim_update(
                        self._window_record_state, physics_dt
                    )

        else:
            logger.log_warning("Physics simulation is not manually updated.")

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

        if (
            self._window_record_hotkey_cfg is not None
            and self._window_record_input_control is None
        ):
            self.enable_window_record_hotkey(**self._window_record_hotkey_cfg)
        if (
            self._window_camera_pose_hotkey_cfg is not None
            and self._window_camera_pose_input_control is None
        ):
            self.enable_window_camera_pose_hotkey(**self._window_camera_pose_hotkey_cfg)
        self.is_window_opened = True

    def close_window(self) -> None:
        """Close the simulation window."""
        if self.is_window_recording():
            self.stop_window_record()
        self._world.close_window()
        self._window = None
        self._window_record_input_control = None
        self._window_camera_pose_input_control = None
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
        plane_collision.set_visible(False)
        plane_collision_pose = np.eye(4, dtype=float)
        plane_collision_pose[2, 3] = -default_length / 20 - 0.001
        plane_collision.set_local_pose(plane_collision_pose)
        plane_collision.add_rigidbody(ActorType.KINEMATIC, RigidBodyShape.CONVEX)

        # TODO: add default physics attributes for the plane.

    def set_default_global_lighting(self) -> None:
        """Set default global lighting for the scene.

        Configures both the environment emission (ambient) light and a
        directional light to provide default scene illumination. The
        directional light is a global scene light (infinite distance)
        pointing downward along the -Z axis.
        """
        # Environment emission light
        self.set_emission_light([1.0, 1.0, 1.0], 120.0)

        # Directional light as global scene light
        dir_light_cfg = LightCfg(
            uid="default_global_light",
            light_type="sun",
            intensity=8.0,
            direction=(0.0, 0.0, -1.0),
            color=(1.0, 0.95, 0.85),
            enable_shadow=True,
        )
        self.add_light(dir_light_cfg)

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
                roughness=1.0,
            )
        )

        self._default_plane.set_material(mat.get_instance("plane_mat").mat)
        self._visual_materials[mat_name] = mat

    def set_ground_plane_visibility(self, visible: bool) -> None:
        """_summary_

        Args:
            visible (bool): _description_
        """
        if visible:
            self._default_plane.set_visible(True)
        else:
            self._default_plane.set_visible(False)

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
        if uid in self._cloth_objects:
            return self._cloth_objects[uid]
        if uid in self._articulations:
            return self._articulations[uid]

        logger.log_warning(f"Asset {uid} not found.")
        return None

    # Light type string → dexsim LightType enum mapping
    _LIGHT_TYPE_MAP: dict[str, LightType] = {
        "point": LightType.POINT,
        "sun": LightType.SUN,
        "direction": LightType.DIRECTION,
        "spot": LightType.SPOT,
        "rect": LightType.RECT,
        "mesh": LightType.MESH,
    }

    # Light types that are created as a single global scene light (not per-environment).
    _GLOBAL_LIGHT_TYPES: tuple[str, ...] = ("sun", "direction")

    def add_light(self, cfg: LightCfg) -> Light:
        """Create a light in the scene.

        Supports six light types: ``"point"``, ``"sun"``, ``"direction"``,
        ``"spot"``, ``"rect"``, and ``"mesh"``. See :class:`LightCfg` for
        type-specific configuration fields.

        .. attention::
            ``"sun"`` and ``"direction"`` lights are global scene lights
            (infinite-distance directional light sources). They are created
            as a single instance on the root environment, not batched per
            environment. All other types are created as per-environment
            batched lights.

        Args:
            cfg (LightCfg): Configuration for the light, including type, color,
                intensity, and type-specific properties.

        Returns:
            Light: The created light instance.

        Raises:
            RuntimeError: If ``cfg.light_type`` is not one of the supported types.
        """
        if cfg.uid is None:
            uid = "light"
            cfg.uid = uid
        else:
            uid = cfg.uid

        if uid in self._lights:
            logger.log_error(f"Light {uid} already exists.")

        light_type_str = cfg.light_type
        light_type = self._LIGHT_TYPE_MAP.get(light_type_str)
        if light_type is None:
            supported = ", ".join(self._LIGHT_TYPE_MAP.keys())
            logger.log_error(
                f"Unsupported light type: '{light_type_str}'. "
                f"Supported types: {supported}."
            )

        # Validation warnings for type-specific constraints
        if light_type_str == "mesh" and not cfg.mesh_path:
            logger.log_warning(
                f"Mesh light '{uid}' has no mesh_path set. "
                f"Use set_mesh() to assign a MeshObject."
            )
        if light_type_str == "rect" and (cfg.rect_width <= 0 or cfg.rect_height <= 0):
            logger.log_warning(
                f"Rect light '{uid}' has zero or negative dimensions "
                f"(width={cfg.rect_width}, height={cfg.rect_height})."
            )

        if cfg.light_type in self._GLOBAL_LIGHT_TYPES:
            # Global scene light: create a single instance on the root
            # environment. Infinite-distance lights (sun, direction) are
            # physically scene-global and should not be duplicated per arena.
            light = self._env.create_light(uid, light_type)
            batch_lights = Light(cfg=cfg, entities=[light])
        else:
            # Per-environment batched light: one instance per arena.
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

    def get_light_uid_list(self) -> List[str]:
        """Get current light uid list

        Returns:
            List[str]: list of light uid.
        """
        return list(self._lights.keys())

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

    def add_cloth_object(self, cfg: ClothObjectCfg) -> ClothObject:
        """Add a cloth object to the scene.

        Args:
            cfg (ClothObjectCfg): Configuration for the cloth object.

        Returns:
            ClothObject: The added cloth object instance handle.
        """
        if not self.is_use_gpu_physics:
            logger.log_error("Cloth object requires GPU physics to be enabled.")

        from embodichain.lab.sim.utility import (
            load_cloth_object_from_cfg,
        )

        uid = cfg.uid
        if uid is None:
            logger.log_error("Cloth object uid must be specified.")

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = load_cloth_object_from_cfg(
            cfg=cfg,
            env_list=env_list,
        )

        cloth_obj = ClothObject(cfg=cfg, entities=obj_list, device=self.device)
        self._cloth_objects[uid] = cloth_obj
        return cloth_obj

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

    def get_cloth_object(self, uid: str) -> ClothObject | None:
        """Get a cloth object by its unique ID.

        Args:
            uid (str): The unique ID of the cloth object.

        Returns:
            ClothObject | None: The cloth object instance if found, otherwise None.
        """
        if uid not in self._cloth_objects:
            logger.log_warning(f"Cloth object {uid} not found.")
            return None
        return self._cloth_objects[uid]

    def get_rigid_object_uid_list(self) -> List[str]:
        """Get current rigid body uid list

        Returns:
            List[str]: list of rigid body uid.
        """
        return list(self._rigid_objects.keys())

    @staticmethod
    def _broadcast_frame(
        frame: np.ndarray | None,
        num_envs: int,
        env_ids: Sequence[int],
        name: str,
    ) -> list[np.ndarray]:
        """Broadcast a local-frame spec to one matrix per target env.

        Args:
            frame: None -> identity; (4,4) -> repeated; (N,4,4) -> indexed per env.
            num_envs: Total number of arenas (used to validate (N,4,4)).
            env_ids: Target env indices to produce frames for.
            name: Constraint name (for error messages).

        Returns:
            A list of (4,4) numpy arrays, one per env in env_ids.

        Raises:
            RuntimeError: If an (N,4,4) frame's N != num_envs, or shape is invalid.
        """
        if frame is None:
            identity = np.eye(4, dtype=np.float32)
            return [identity for _ in env_ids]
        frame_np = np.asarray(frame, dtype=np.float32)
        if frame_np.shape == (4, 4):
            return [frame_np for _ in env_ids]
        if frame_np.ndim == 3 and frame_np.shape[1:] == (4, 4):
            if frame_np.shape[0] != num_envs:
                logger.log_error(
                    f"Constraint '{name}' local frame has shape {frame_np.shape} "
                    f"but num_envs is {num_envs}. Expected ({num_envs}, 4, 4)."
                )
            return [frame_np[i] for i in env_ids]
        logger.log_error(
            f"Constraint '{name}' local frame has invalid shape {frame_np.shape}. "
            "Expected None, (4, 4), or (N, 4, 4)."
        )

    @staticmethod
    def _normalize_env_ids(
        env_ids: Sequence[int] | torch.Tensor | None,
        num_envs: int,
    ) -> list[int]:
        """Normalize an ``env_ids`` spec to a plain ``list[int]``.

        Accepts ``None`` (-> all envs), a ``torch.Tensor`` (as passed by the
        :class:`EventManager`), or any ``Sequence[int]``, and returns a list of
        Python ints. Normalizing here keeps the per-arena constraint names clean
        (e.g. ``"weld_0"`` rather than relying on a tensor's string form) and
        avoids depending on implicit tensor-to-int conversions downstream.

        Args:
            env_ids: None, a tensor, or a sequence of ints.
            num_envs: Total number of arenas (used when env_ids is None).

        Returns:
            A list of int env indices.
        """
        if env_ids is None:
            return list(range(num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.detach().cpu().tolist()
        return [int(i) for i in env_ids]

    def create_rigid_constraint(
        self,
        cfg: RigidConstraintCfg,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> RigidConstraint:
        """Create a fixed constraint between two RigidObjects.

        Binds ``rigid_object_a``'s entity[i] to ``rigid_object_b``'s entity[i]
        within arena[i], for each env in ``env_ids``. Local frames default to
        welding the objects at their *current* relative pose:
        ``local_frame_a`` defaults to identity (object A's origin) and
        ``local_frame_b`` defaults to ``inv(pose_B) @ pose_A`` (computed per env),
        so the offset is preserved rather than the two origins being pulled
        together. Pass explicit frames to define a specific joint frame.

        Args:
            cfg: The constraint configuration.
            env_ids: Target environment indices. Accepts a tensor (as passed by
                the :class:`EventManager`) or a sequence of ints. None -> all arenas.

        Returns:
            The created :class:`RigidConstraint`.

        Raises:
            RuntimeError: If either object is missing, the name is already in use,
                a frame shape is invalid, or dexsim fails to create a handle.
        """
        # validate constraint type (only fixed supported in v1)
        if cfg.constraint_type != "fixed":
            logger.log_error(
                f"Constraint '{cfg.name}' has unsupported type "
                f"'{cfg.constraint_type}'. Only 'fixed' is supported in v1."
            )

        # resolve objects
        if cfg.rigid_object_a_uid not in self._rigid_objects:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_a_uid}' not found for constraint "
                f"'{cfg.name}'. Available: {list(self._rigid_objects.keys())}."
            )
        if cfg.rigid_object_b_uid not in self._rigid_objects:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_b_uid}' not found for constraint "
                f"'{cfg.name}'. Available: {list(self._rigid_objects.keys())}."
            )
        rigid_object_a = self._rigid_objects[cfg.rigid_object_a_uid]
        rigid_object_b = self._rigid_objects[cfg.rigid_object_b_uid]

        # validate duplicate name
        if cfg.name in self._constraints:
            logger.log_error(
                f"Constraint '{cfg.name}' already exists. Remove it before recreating."
            )

        # validate object entity counts match num_envs
        num_envs = self.num_envs
        if rigid_object_a.num_instances != num_envs:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_a_uid}' has "
                f"{rigid_object_a.num_instances} instances but num_envs is {num_envs}."
            )
        if rigid_object_b.num_instances != num_envs:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_b_uid}' has "
                f"{rigid_object_b.num_instances} instances but num_envs is {num_envs}."
            )

        # resolve target env_ids (accepts None / tensor / sequence)
        target_env_ids = self._normalize_env_ids(env_ids, num_envs)

        # broadcast local frames.
        # local_frame_a defaults to identity (object A's origin).
        # local_frame_b defaults to the current relative pose of A w.r.t. B
        # (inv(pose_B) @ pose_A), so that with both frames left as None the
        # constraint welds the objects at their *current* relative pose instead
        # of pulling their origins together.
        frames_a = self._broadcast_frame(
            cfg.local_frame_a, num_envs, target_env_ids, cfg.name
        )
        if cfg.local_frame_b is None:
            pose_a = rigid_object_a.get_local_pose(to_matrix=True)
            pose_b = rigid_object_b.get_local_pose(to_matrix=True)
            frame_b = torch.bmm(pose_inv(pose_b), pose_a)  # (N, 4, 4)
            frame_b = frame_b.cpu().numpy().astype(np.float32)
            frames_b = [frame_b[i] for i in target_env_ids]
        else:
            frames_b = self._broadcast_frame(
                cfg.local_frame_b, num_envs, target_env_ids, cfg.name
            )

        # pre-size handles list with None, fill target envs
        handles: list = [None] * num_envs
        try:
            for idx, env_id in enumerate(target_env_ids):
                arena = self.get_env(env_id)
                name_i = cfg.name if num_envs <= 1 else f"{cfg.name}_{env_id}"
                handle = arena.create_fixed_constraint(
                    name_i,
                    rigid_object_a._entities[env_id],
                    rigid_object_b._entities[env_id],
                    frames_a[idx],
                    frames_b[idx],
                )
                if handle is None:
                    logger.log_error(
                        f"Failed to create constraint '{name_i}' in arena {env_id}."
                    )
                handles[env_id] = handle
        except Exception:
            # Ensure partially created per-arena constraints are removed if a later
            # arena fails, so create/remove semantics stay consistent.
            RigidConstraint(
                cfg=cfg,
                constraint_handles=handles,
                rigid_object_a=rigid_object_a,
                rigid_object_b=rigid_object_b,
                device=self.device,
            ).destroy(env_ids=target_env_ids, arena_resolver=self.get_env)
            raise

        constraint = RigidConstraint(
            cfg=cfg,
            constraint_handles=handles,
            rigid_object_a=rigid_object_a,
            rigid_object_b=rigid_object_b,
            device=self.device,
        )
        self._constraints[cfg.name] = constraint
        return constraint

    def get_soft_object_uid_list(self) -> List[str]:
        """Get current soft body uid list

        Returns:
            List[str]: list of soft body uid.
        """
        return list(self._soft_objects.keys())

    def get_cloth_object_uid_list(self) -> List[str]:
        """Get current cloth body uid list

        Returns:
            List[str]: list of cloth body uid.
        """
        return list(self._cloth_objects.keys())

    def remove_rigid_constraint(
        self,
        name: str,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> bool:
        """Remove a rigid constraint by name.

        With ``env_ids=None`` the constraint is removed from every arena and
        dropped from the registry. With a subset, only those arenas are cleared;
        the registry entry is kept until all handles become None.

        Args:
            name: The base constraint name.
            env_ids: Subset of arenas to clear. Accepts a tensor (as passed by
                the :class:`EventManager`) or a sequence of ints. None -> all.

        Returns:
            True if the constraint was found (and removed or partially removed),
            False if the name is unknown.
        """
        constraint = self._constraints.get(name, None)
        if constraint is None:
            logger.log_warning(f"Constraint '{name}' not found. Nothing to remove.")
            return False

        target_env_ids = self._normalize_env_ids(env_ids, constraint.num_envs)
        constraint.destroy(env_ids=target_env_ids, arena_resolver=self.get_env)

        # drop from registry if no handles remain active
        if all(h is None for h in constraint.constraint_handles):
            del self._constraints[name]
        return True

    def get_rigid_constraint(self, name: str) -> RigidConstraint | None:
        """Get a rigid constraint by its base name.

        Args:
            name: The base constraint name.

        Returns:
            The constraint, or None if not found.
        """
        if name not in self._constraints:
            logger.log_warning(f"Constraint '{name}' not found.")
            return None
        return self._constraints[name]

    def get_rigid_constraint_uid_list(self) -> List[str]:
        """Get the list of registered constraint base names.

        Returns:
            List[str]: list of constraint names.
        """
        return list(self._constraints.keys())

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

    def has_non_static_rigid_object(self) -> bool:
        """Check if there is any non-static rigid object in the simulation.

        Returns:
            bool: True if there is at least one non-static rigid object, False otherwise.
        """
        for rigid_obj in self._rigid_objects.values():
            if rigid_obj.body_type != "static":
                return True

        if len(self._rigid_object_groups) > 0:
            return True

        return False

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

        is_usd = cfg.fpath.endswith((".usd", ".usda", ".usdc"))
        if is_usd:
            # TODO: Currently add checking for num_envs when file is USD. After we support spawn via cloning, we can remove this.
            if len(env_list) > 1:
                logger.log_error(f"Currently not supporting multiple arenas for USD.")
            env = self._env
            results = env.import_from_usd_file(
                cfg.fpath, return_object=True, cache_dir=self._convex_decomp_dir
            )
            # print("USD import results:", results)

            articulations_found = []
            for key, value in results.items():
                if isinstance(value, dexsim.engine.Articulation):
                    articulations_found.append(value)

            if len(articulations_found) == 0:
                logger.log_error(f"No articulation found in USD file {cfg.fpath}.")
            elif len(articulations_found) > 1:
                logger.log_error(
                    f"Multiple articulations found in USD file {cfg.fpath}. "
                )
            elif len(articulations_found) == 1:
                obj_list.append(articulations_found[0])
        else:
            # non-usd file does not support this option, will be forced set False to avoid potential issues.
            cfg.use_usd_properties = False

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

            if cfg.solver_cfg is not None:
                if isinstance(cfg.solver_cfg, dict):
                    for key, value in cfg.solver_cfg.items():
                        if hasattr(value, "urdf_path") and value.urdf_path is None:
                            value.urdf_path = cfg.fpath

        if uid is None:
            uid = os.path.splitext(os.path.basename(cfg.fpath))[0]
            cfg.uid = uid
        if uid in self._robots:
            logger.log_error(f"Robot {uid} already exists.")
            return self._robots[uid]

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        obj_list = []

        is_usd = cfg.fpath.endswith((".usd", ".usda", ".usdc"))
        if is_usd:
            # TODO: Currently add checking for num_envs when file is USD. After we support spawn via cloning, we can remove this.
            if len(env_list) > 1:
                logger.log_error(f"Currently not supporting multiple arenas for USD.")
            env = self._env
            results = env.import_from_usd_file(cfg.fpath, return_object=True)
            # print("USD import results:", results)

            articulations_found = []
            for key, value in results.items():
                if isinstance(value, dexsim.engine.Articulation):
                    articulations_found.append(value)

            if len(articulations_found) == 0:
                logger.log_error(f"No articulation found in USD file {cfg.fpath}.")
            elif len(articulations_found) > 1:
                logger.log_error(
                    f"Multiple articulations found in USD file {cfg.fpath}. "
                )
            elif len(articulations_found) == 1:
                obj_list.append(articulations_found[0])
        else:
            # non-usd file does not support this option, will be forced set False to avoid potential issues.
            cfg.use_usd_properties = False

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
    ) -> Gizmo:
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

        return gizmo

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

        if uid in self._cloth_objects:
            obj = self._cloth_objects.pop(uid)
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
        if isinstance(draw_xpos, torch.Tensor):
            draw_xpos = draw_xpos.detach().cpu().numpy()
        elif isinstance(draw_xpos, (list, tuple)):
            draw_xpos = [
                item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
                for item in draw_xpos
            ]
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

    def _build_window_record_output(
        self, save_path: str | None, video_prefix: str
    ) -> tuple[str, str]:
        """Resolve the output directory and file name for viewer recording."""
        if save_path is None:
            output_dir = os.path.join(os.getcwd(), "outputs", "videos")
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_name = f"{video_prefix}_{timestamp}"
        else:
            output_dir = os.path.dirname(save_path) or os.getcwd()
            video_name = Path(os.path.basename(save_path)).stem
        return output_dir, video_name

    def is_window_recording(self) -> bool:
        """Check whether the viewer window is currently recording."""
        return self._window_record_state is not None

    def _build_window_record_pose_from_look_at(
        self,
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """Build a camera pose matrix for the recorder from look-at inputs."""
        pose = look_at_to_pose(eye, target, up)[0].cpu().numpy()
        pose[:3, 1] = -pose[:3, 1]
        pose[:3, 2] = -pose[:3, 2]
        return np.asarray(pose, dtype=np.float32)

    def _resolve_window_record_pose(
        self, state: _WindowRecordState
    ) -> np.ndarray | None:
        """Resolve the camera pose used by the recorder for the current frame."""
        if state.pose_provider is not None:
            pose = state.pose_provider()
            return np.asarray(pose, dtype=np.float32)

        if state.fixed_pose is not None:
            return np.asarray(state.fixed_pose, dtype=np.float32)

        if self._window is not None:
            return np.asarray(self._window.get_pose_matrix(), dtype=np.float32)

        return None

    def _step_window_record(self, state: _WindowRecordState) -> int:
        """Capture frames in the render thread without blocking the UI loop."""
        if state.task_status != TASK_RETURN.TASK_LOOP:
            return state.task_status

        now = time.time()
        if now - state.last_capture_time < state.time_step:
            return state.task_status

        state.last_capture_time = now
        return self._capture_window_record_frame(state)

    def _capture_window_record_frame(self, state: _WindowRecordState) -> int:
        """Render one frame for the active recording session."""
        frame: np.ndarray | None = None
        pose = self._resolve_window_record_pose(state)
        if pose is not None and state.record_camera is not None:
            state.record_camera.set_world_pose(pose)
            state.record_camera.render()
            rgb = np.asarray(state.record_camera.get_rgb_map())
            if rgb.size != 0:
                frame = np.ascontiguousarray(rgb[..., :3])

        if frame is None:
            return state.task_status

        state.frames.append(frame)
        state.current_memory_bytes += frame.nbytes
        if state.current_memory_bytes > state.max_memory_bytes:
            logger.log_warning(
                "Viewer recording exceeded the configured memory budget. "
                "Press 'r' again to flush the buffered frames to disk."
            )
            state.task_status = TASK_RETURN.TASK_EXIT

        return state.task_status

    def _step_window_record_from_sim_update(
        self, state: _WindowRecordState, physics_dt: float
    ) -> int:
        """Capture recording frames based on simulation time progression."""
        if state.task_status != TASK_RETURN.TASK_LOOP:
            return state.task_status

        state.accumulated_sim_time += physics_dt
        if state.accumulated_sim_time + 1e-9 < state.time_step:
            return state.task_status

        state.accumulated_sim_time = max(
            0.0, state.accumulated_sim_time - state.time_step
        )
        return self._capture_window_record_frame(state)

    def _save_window_record_worker(
        self,
        frames: list[np.ndarray],
        output_dir: str,
        video_name: str,
        save_kwargs: dict[str, object],
    ) -> None:
        """Encode buffered frames into a video file in a background thread."""
        from dexsim.utility import images_to_video

        try:
            os.makedirs(output_dir, exist_ok=True)
            images_to_video(
                images=frames,
                output_dir=output_dir,
                video_name=video_name,
                **save_kwargs,
            )
            logger.log_info(
                f"Viewer recording saved to {os.path.join(output_dir, video_name + '.mp4')}"
            )
        except Exception as exc:
            logger.log_error(f"Failed to save viewer recording: {exc}")

    def start_window_record(
        self,
        save_path: str | None = None,
        fps: int = 20,
        max_memory: int = 1024,
        video_prefix: str = "viewer_record",
        pose_provider: Callable[[], np.ndarray] | None = None,
        fixed_pose: np.ndarray | None = None,
        look_at: (
            tuple[
                Sequence[float],
                Sequence[float],
                Sequence[float],
            ]
            | None
        ) = None,
        use_sim_time: bool | None = None,
    ) -> bool:
        """Start asynchronously recording the simulation to a video buffer.

        The recorder can either follow the live viewer camera or run without a
        window by using a fixed pose or a pose callback supplied by the caller.

        Args:
            save_path: Optional output path for the recorded video.
            fps: Target output frames per second. Must be positive.
            max_memory: Maximum buffered frame memory in MB. Must be positive.
            video_prefix: File name prefix used when ``save_path`` is not provided.
            pose_provider: Optional callback that returns the current camera pose.
            fixed_pose: Optional fixed 4x4 camera pose matrix.
            look_at: Optional ``(eye, target, up)`` tuple used to derive a fixed pose.
            use_sim_time: Whether to capture frames from simulation time instead of
                wall time. Defaults to headless mode when no viewer window exists.

        Returns:
            bool: True if recording starts successfully, otherwise False.
        """
        if self.is_window_recording():
            logger.log_error(
                "A viewer recording session is already active. Stop it before starting a new recording."
            )
        if fps <= 0:
            logger.log_error(f"Viewer recording FPS must be positive, got {fps}.")
        if max_memory <= 0:
            logger.log_error(
                f"Viewer recording max_memory must be positive, got {max_memory}."
            )
        if pose_provider is not None and fixed_pose is not None:
            logger.log_error(
                "Recorder accepts only one explicit pose source: `pose_provider` or `fixed_pose`."
            )
        if pose_provider is not None and look_at is not None:
            logger.log_error(
                "Recorder accepts only one explicit pose source: `pose_provider` or `look_at`."
            )
        if fixed_pose is not None and look_at is not None:
            logger.log_error(
                "Recorder accepts only one explicit pose source: `fixed_pose` or `look_at`."
            )

        if look_at is not None:
            fixed_pose = self._build_window_record_pose_from_look_at(*look_at)

        if pose_provider is None and fixed_pose is None and self._window is None:
            logger.log_warning(
                "No simulation window available for viewer recording. "
                "Provide `pose_provider`, `fixed_pose`, or `look_at` to record in headless mode."
            )
            return False

        if use_sim_time is None:
            use_sim_time = self._window is None

        width = self.sim_config.width
        height = self.sim_config.height
        if self._window_record_camera is None:
            camera_name = f"viewer_record_camera_{self.instance_id}"
            self._window_record_camera = self._env.create_camera(
                camera_name, width, height
            )
        record_camera = self._window_record_camera
        if hasattr(record_camera, "is_open") and record_camera.is_open() is False:
            record_camera.open_camera()

        time_step = 1.0 / float(fps)
        output_dir, video_name = self._build_window_record_output(
            save_path, video_prefix
        )
        state = _WindowRecordState(
            time_step=time_step,
            max_memory_bytes=max_memory * 1024 * 1024,
            output_dir=output_dir,
            video_name=video_name,
            save_kwargs={"fps": fps},
            record_camera=record_camera,
            pose_provider=pose_provider,
            fixed_pose=(
                None if fixed_pose is None else np.asarray(fixed_pose, dtype=np.float32)
            ),
            capture_from_sim_update=use_sim_time,
            last_capture_time=time.time() - time_step,
        )

        if not state.capture_from_sim_update:

            def _window_record_loop(_: float) -> int:
                return self._step_window_record(state)

            state.loop_handle = self._world.thread_rt().add_loop(
                _window_record_loop, time_step
            )
        self._window_record_state = state

        follow_source = (
            "live viewer pose"
            if pose_provider is None and fixed_pose is None and self._window is not None
            else "custom pose source"
        )
        timing_source = (
            "simulation time" if state.capture_from_sim_update else "wall time"
        )
        save_target = os.path.join(output_dir, video_name + ".mp4")
        if self._window is not None:
            logger.log_info(
                f"Viewer recording started ({follow_source}, {timing_source}). Press 'r' again to stop and save to "
                f"{save_target}"
            )
        else:
            logger.log_info(
                f"Viewer recording started ({follow_source}, {timing_source}). Call `stop_window_record()` to save to "
                f"{save_target}"
            )
        return True

    def stop_window_record(self, save_path: str | None = None) -> bool:
        """Stop the active viewer recording and save frames in the background."""
        if self._window_record_state is None:
            logger.log_warning("No active viewer recording session found.")
            return False

        state = self._window_record_state
        state.task_status = TASK_RETURN.TASK_EXIT
        if save_path is not None:
            output_dir, video_name = self._build_window_record_output(
                save_path, "viewer_record"
            )
        else:
            output_dir, video_name = state.output_dir, state.video_name

        if state.record_camera is not None and hasattr(state.record_camera, "is_open"):
            if state.record_camera.is_open():
                state.record_camera.close_camera()

        frames = list(state.frames)
        self._window_record_state = None
        if len(frames) == 0:
            logger.log_warning(
                "Viewer recording stopped, but no frames were captured. Skipping video export."
            )
            return False

        self._window_record_save_threads = [
            thread for thread in self._window_record_save_threads if thread.is_alive()
        ]
        save_thread = threading.Thread(
            target=self._save_window_record_worker,
            args=(frames, output_dir, video_name, dict(state.save_kwargs)),
            daemon=False,
        )
        save_thread.start()
        self._window_record_save_threads.append(save_thread)
        logger.log_info(
            "Viewer recording stopped. Saving video to "
            f"{os.path.join(output_dir, video_name + '.mp4')} in background."
        )
        return True

    def wait_window_record_saves(self) -> None:
        """Wait for all background video export threads to finish."""
        for thread in self._window_record_save_threads:
            thread.join()
        self._window_record_save_threads = []

    def toggle_window_record(
        self,
        save_path: str | None = None,
        fps: int = 20,
        max_memory: int = 1024,
        video_prefix: str = "viewer_record",
    ) -> bool:
        """Toggle viewer recording on or off."""
        if self.is_window_recording():
            return self.stop_window_record(save_path=save_path)
        return self.start_window_record(
            save_path=save_path,
            fps=fps,
            max_memory=max_memory,
            video_prefix=video_prefix,
        )

    def enable_window_record_hotkey(
        self,
        save_path: str | None = None,
        fps: int = 20,
        max_memory: int = 1024,
        video_prefix: str = "viewer_record",
    ) -> bool:
        """Register the ``r`` key to start/stop viewer recording."""
        self._window_record_hotkey_cfg = {
            "save_path": save_path,
            "fps": fps,
            "max_memory": max_memory,
            "video_prefix": video_prefix,
        }
        if self._window is None:
            logger.log_warning(
                "No simulation window available yet. The viewer record hotkey will be registered after `open_window()`."
            )
            return False
        if self._window_record_input_control is not None:
            return True

        from dexsim.types import InputKey

        sim = self
        hotkey_cfg = dict(self._window_record_hotkey_cfg)

        class WindowRecordEvent(ObjectManipulator):
            def on_key_down(self, key):
                if key == InputKey.SCANCODE_R.value:
                    sim.toggle_window_record(**hotkey_cfg)

        self._window_record_input_control = WindowRecordEvent()
        self._window.add_input_control(self._window_record_input_control)
        logger.log_info(
            "Viewer record hotkey registered. Press 'r' to start/stop recording."
        )
        return True

    @staticmethod
    def _window_camera_pose_to_look_at(
        pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a DexSim window model matrix to look-at vectors.

        DexSim stores the viewer camera model matrix with columns
        ``[right, up, -forward]``. ``Windows.set_look_at`` expects the
        corresponding world-space look-at vectors, so the conversion is done
        from the matrix columns rather than by assuming a fixed world axis.

        Args:
            pose: A 4x4 homogeneous viewer camera pose matrix.

        Returns:
            The ``(eye, look_at, up)`` vectors accepted by
            ``Windows.set_look_at``.

        Raises:
            ValueError: If ``pose`` is not a 4x4 homogeneous matrix.
        """
        matrix = np.asarray(pose, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Window camera pose must have shape (4, 4), got {matrix.shape}."
            )
        eye = matrix[:3, 3]
        look_at = eye - matrix[:3, 2]
        up = matrix[:3, 1]
        return eye, look_at, up

    @staticmethod
    def _format_window_camera_pose(
        pose: np.ndarray, convert_to_look_at: bool = True
    ) -> str:
        """Format a DexSim window pose as an executable Python snippet.

        Args:
            pose: A 4x4 homogeneous viewer camera pose matrix.
            convert_to_look_at: Print a ``set_look_at`` call when true;
                otherwise print the raw pose matrix.

        Returns:
            An executable Python snippet containing the camera pose.

        Raises:
            ValueError: If ``pose`` is not a 4x4 homogeneous matrix.
        """
        matrix = np.asarray(pose, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Window camera pose must have shape (4, 4), got {matrix.shape}."
            )

        def _format_float(value: float) -> str:
            if abs(value) < 1e-12:
                return "0.0"
            formatted = format(value, ".8g")
            if "e" not in formatted and "." not in formatted:
                formatted += ".0"
            return formatted

        def _vector_literal(vector: np.ndarray) -> str:
            values = ", ".join(_format_float(float(value)) for value in vector)
            return f"np.array([{values}], dtype=np.float32)"

        if convert_to_look_at:
            eye, look_at, up = SimulationManager._window_camera_pose_to_look_at(matrix)
            return (
                "window.set_look_at("
                f"eye={_vector_literal(eye)}, "
                f"look_at={_vector_literal(look_at)}, "
                f"up={_vector_literal(up)})"
            )

        rows = ",\n    ".join(
            "[" + ", ".join(_format_float(float(value)) for value in row) + "]"
            for row in matrix
        )
        return f"window_pose = np.array([\n    {rows}\n], dtype=np.float32)"

    def print_window_camera_pose(self, convert_to_look_at: bool = True) -> str | None:
        """Print the current viewer camera pose as reusable Python code.

        Args:
            convert_to_look_at: Print ``window.set_look_at(...)`` by default.
                Set false to print the raw 4x4 pose matrix instead.

        Returns:
            The printed snippet, or ``None`` when no viewer window is open.
        """
        if self._window is None:
            logger.log_warning("No simulation window available to print its pose.")
            return None

        pose = np.asarray(self._window.get_pose_matrix(), dtype=np.float32)
        snippet = self._format_window_camera_pose(pose, convert_to_look_at)
        print(snippet)
        return snippet

    def enable_window_camera_pose_hotkey(self, convert_to_look_at: bool = True) -> bool:
        """Register ``p`` to print the current viewer camera pose.

        Args:
            convert_to_look_at: Print a ``window.set_look_at(...)`` call when
                true, which is the default. Set false to print the raw matrix.

        Returns:
            Whether the control is registered on an available window.
        """
        self._window_camera_pose_hotkey_cfg = {"convert_to_look_at": convert_to_look_at}
        if self._window is None:
            logger.log_warning(
                "No simulation window available yet. The camera pose print "
                "hotkey will be registered after `open_window()`."
            )
            return False
        if self._window_camera_pose_input_control is not None:
            return True

        from dexsim.types import InputKey

        sim = self
        hotkey_cfg = dict(self._window_camera_pose_hotkey_cfg)

        class WindowCameraPoseEvent(ObjectManipulator):
            def on_key_down(self, key):
                if key == InputKey.SCANCODE_P.value:
                    sim.print_window_camera_pose(**hotkey_cfg)

        self._window_camera_pose_input_control = WindowCameraPoseEvent()
        self._window.add_input_control(self._window_camera_pose_input_control)
        logger.log_info(
            "Camera pose print hotkey registered. Press 'p' to print the "
            "current viewer pose."
        )
        return True

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
        if self._env:
            self._env.clean_materials()

    def reset_objects_state(
        self,
        env_ids: Sequence[int] | None = None,
        excluded_uids: Sequence[str] | None = None,
    ) -> None:
        """Reset the state of the simulated assets given the environment IDs and excluded UIDs.

        Args:
            env_ids (Sequence[int] | None): The environment IDs to reset. If None, reset all environments.
            excluded_uids (Sequence[str] | None): List of asset UIDs to exclude from resetting. If None, reset all assets.
        """
        excluded_uids = set(excluded_uids) if excluded_uids is not None else set()
        for uid, robot in self._robots.items():
            if uid not in excluded_uids:
                robot.reset(env_ids)
        for uid, articulation in self._articulations.items():
            if uid not in excluded_uids:
                articulation.reset(env_ids)
        for uid, rigid_obj in self._rigid_objects.items():
            if uid not in excluded_uids:
                rigid_obj.reset(env_ids)
        for uid, rigid_obj_group in self._rigid_object_groups.items():
            if uid not in excluded_uids:
                rigid_obj_group.reset(env_ids)
        for uid, light in self._lights.items():
            if uid not in excluded_uids:
                light.reset(env_ids)
        for uid, sensor in self._sensors.items():
            if uid not in excluded_uids:
                sensor.reset(env_ids)

    def export_usd(self, fpath: str) -> bool:
        """Export the current simulation scene to a USD file.

        Args:
            fpath (str): The file path to save the USD file.

        Returns:
            bool: True if export is successful, False otherwise.
        """
        try:
            self._env.export_to_usd_file(fpath)
            logger.log_info(f"Simulation scene exported to USD file: {fpath}")
            return True
        except Exception as e:
            logger.log_error(f"Failed to export simulation scene to USD: {e}")
            return False

    @staticmethod
    def wait_scene_destruction(timeout_ms: int = 10000) -> None:
        """A public helper to wait for the underlying C++ scenes (dexsim.World) to destruct completely."""
        import dexsim
        import gc

        # Force garbage collection to break cycle references
        gc.collect()

        import time

        wait_times = 0
        scene_count = dexsim.get_world_num()
        max_loops = timeout_ms // 10
        while scene_count > 0 and wait_times < max_loops:
            time.sleep(0.01)
            scene_count = dexsim.get_world_num()
            wait_times += 1
            if wait_times % 50 == 0:
                from embodichain.utils import logger

                logger.log_info(
                    f"Waiting for dexsim.World scenes to destruct. Remaining scenes: {scene_count}"
                )
        if scene_count > 0:
            from embodichain.utils import logger

            logger.log_warning(
                f"Scene destruction wait timeout, {scene_count} C++ scene(s) still alive!"
            )

    def destroy(self, exit_process: bool | None = None) -> None:
        """
        No longer destructs C++ objects in place due to lingering deep local variables;
        instead, packages itself into a destruction task, submits to the cleanup queue,
        and waits for top-level delayed consumption.

        Args:
            exit_process (bool | None): Whether to call os._exit(0) after queuing
                the destruction task. If None, reads EMBODICHAIN_SIM_EXIT_PROCESS.
        """

        if exit_process is None:
            exit_process = (
                os.getenv("EMBODICHAIN_SIM_EXIT_PROCESS", "1").strip().lower()
            )
            exit_process = exit_process not in ("0", "false", "no", "off")

        self._is_pending_kill = True
        # Transfer the actual destruction logic to the cleanup queue
        SimulationManager._cleanup_queue.put(self._deferred_destroy)

        if exit_process:
            os._exit(0)

    def _deferred_destroy(self) -> None:
        """Destroy all simulated assets and release resources."""
        # Clean up all gizmos before destroying the simulation
        for uid in list(self._gizmos.keys()):
            self.disable_gizmo(uid)

        if self.is_window_recording():
            self.stop_window_record()
        self.wait_window_record_saves()

        import sys, gc

        self.clean_materials()

        if self._env:
            self._env.clean()
        if self._world:
            self._world.quit()

        # REMOVE INSTANCE FROM POOL
        instance_id = getattr(self, "instance_id", 0)
        SimulationManager.reset(instance_id)

        # Helper to aggressively decouple C++ wrapped objects
        def _sever_wrapper_refs(obj_registry):
            if not hasattr(self, obj_registry):
                return
            registry = getattr(self, obj_registry)
            if not isinstance(registry, dict):
                return
            for uid, obj in registry.items():
                if hasattr(obj, "_world"):
                    obj._world = None
                if hasattr(obj, "_ps"):
                    obj._ps = None
                if hasattr(obj, "_env"):
                    obj._env = None
                if hasattr(obj, "_entities"):
                    obj._entities = []
            registry.clear()

        _sever_wrapper_refs("_gizmos")
        _sever_wrapper_refs("_markers")
        _sever_wrapper_refs("_rigid_objects")
        _sever_wrapper_refs("_constraints")
        _sever_wrapper_refs("_rigid_object_groups")
        _sever_wrapper_refs("_soft_objects")
        _sever_wrapper_refs("_cloth_objects")
        _sever_wrapper_refs("_articulations")
        _sever_wrapper_refs("_robots")
        _sever_wrapper_refs("_sensors")
        _sever_wrapper_refs("_lights")

        # Explicitly clear Python references to trigger C++ object destructors
        self._ps = None
        self._env = None
        self._world = None
        self._default_plane = None

        # Try to break ANY possible frame cycle
        gc.collect()

        self._visual_materials.clear()
        self._texture_cache.clear()
        self._arenas.clear()
        self._markers.clear()
        self._gizmos.clear()
        self._constraints.clear()

        SimulationManager.reset(self.instance_id)

        # Forcefully drop underlying C++ object wrappers
        self._env = None
        self._world = None

        gc.collect()

    @staticmethod
    def flush_cleanup_queue():
        """Dequeue executor and synchronization barrier provided for top-level main loop / Pytest Fixture calls"""
        import gc

        while not SimulationManager._cleanup_queue.empty():
            task = SimulationManager._cleanup_queue.get_nowait()
            try:
                task()
            except Exception as e:
                from embodichain.utils import logger

                logger.log_error(f"Error during delayed destruction: {e}")
                pass

        # After the queue is emptied, perform a top-level full GC to thoroughly reclaim dead objects that haven't released their RefPtrs yet
        gc.collect()

        # At this point, wait for the C++ Scene to return to zero, since the stack is at the top level, there will definitely be no deadlock
        SimulationManager.wait_scene_destruction()
