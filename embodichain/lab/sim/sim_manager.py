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
from contextlib import contextmanager
import dexsim
import torch
import numpy as np
import warp as wp

from pathlib import Path
from copy import deepcopy
from datetime import datetime
from functools import cached_property, partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Sequence, Union
from dataclasses import dataclass, asdict, field, MISSING

# Global cache directories
SIM_CACHE_DIR = Path.home() / ".cache" / "embodichain_cache"
MATERIAL_CACHE_DIR = SIM_CACHE_DIR / "mat_cache"
CONVEX_DECOMP_DIR = SIM_CACHE_DIR / "convex_decomposition"
REACHABLE_XPOS_DIR = SIM_CACHE_DIR / "robot_reachable_xpos"


def _is_usd_path(path: object | None) -> bool:
    """Return whether a source path is a USD stage."""
    return path is not None and str(path).lower().endswith((".usd", ".usda", ".usdc"))


from dexsim.types import (
    ActorType,
    Backend,
    ThreadMode,
)
from dexsim.core import TASK_RETURN
from dexsim.engine import Material
from dexsim.models import MeshObject
from dexsim.render import LightType, Windows
from dexsim.engine import GizmoController, ObjectManipulator

from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectGroup,
    DeformableObject,
    SurfaceDeformableObject,
    VolumeDeformableObject,
    SoftObject,
    ClothObject,
    Articulation,
    Robot,
    Light,
    RigidConstraint,
)
from embodichain.lab.sim.objects.gizmo import Gizmo, GizmoCfg
from embodichain.lab.sim.sensors import (
    SensorCfg,
    BaseSensor,
    Camera,
    StereoCamera,
    ContactSensor,
)
from embodichain.lab.sim.cfg import (
    RenderCfg,
    PhysicsBackendCfg,
    GPUMemoryCfg,
    DefaultPhysicsCfg,
    NewtonPhysicsCfg,
    validate_physics_cfg,
    MarkerCfg,
    WindowRecordCfg,
    WindowCameraPoseCfg,
    LightCfg,
    RigidObjectCfg,
    DeformableObjectCfg,
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
    SoftObjectCfg,
    ClothObjectCfg,
    RigidObjectGroupCfg,
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    RobotCfg,
    RobotPresetCfg,
    RigidConstraintCfg,
)
from embodichain.lab.sim.physics import NewtonPhysicsBackend, make_physics_backend
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    configure_articulation_desc,
    rigid_desc_from_cfg,
    surface_deformable_desc_from_cfg,
    volume_deformable_desc_from_cfg,
)
from embodichain.lab.sim.spawn.usd import (
    articulation_desc_from_usd,
    rigid_desc_from_usd,
)
from embodichain.lab.sim.spawn.scene import SpawnScene
from embodichain.lab.sim import VisualMaterial, VisualMaterialCfg
from embodichain.lab.sim.profiler import Profiler, ProfilerCfg
from embodichain.lab.visualization.cfg import VisualizationCfg
from embodichain.utils import configclass, logger
from embodichain.utils.math import (
    convert_quat,
    look_at_to_pose,
    matrix_from_quat,
    pose_inv,
)

if TYPE_CHECKING:
    from dexsim.engine import PhysicsScene
    from dexsim.spawn import SpawnResult

    from embodichain.lab.visualization import (
        RuntimeHealth,
        RuntimeStats,
        SceneManifest,
        SceneOverlays,
        VisualizationRuntime,
    )

__all__ = [
    "SimulationManager",
    "SimulationManagerCfg",
    "get_physics_scene",
    "SIM_CACHE_DIR",
    "MATERIAL_CACHE_DIR",
    "CONVEX_DECOMP_DIR",
    "REACHABLE_XPOS_DIR",
]


@contextmanager
def _temporary_warp_kernel_log_suppression(
    physics_cfg: PhysicsBackendCfg,
) -> Iterator[None]:
    """Temporarily suppress informational Warp logs for Newton operations."""
    if not (
        isinstance(physics_cfg, NewtonPhysicsCfg)
        and physics_cfg.suppress_warp_kernel_logs
    ):
        yield
        return

    previous_log_level = wp.config.log_level
    try:
        # Warp emits its startup banner and module-load timers at INFO level.
        # Keep warnings and errors visible.
        wp.config.log_level = wp.LOG_WARNING
        yield
    finally:
        wp.config.log_level = previous_log_level


def _initialize_warp_runtime(physics_cfg: PhysicsBackendCfg) -> None:
    """Initialize Warp while honoring Newton startup-log suppression."""
    with _temporary_warp_kernel_log_suppression(physics_cfg):
        wp.init()


# Deformable implementations remain backend-specific even though their public
# object/data contract is shared. Newton is an explicit empty placeholder until
# its native object adapters are integrated and validated.
_DEFORMABLE_BACKEND_IMPLEMENTATIONS = {
    "default": {
        "volume": (
            VolumeDeformableObjectCfg,
            VolumeDeformableObject,
            volume_deformable_desc_from_cfg,
            "soft_object",
        ),
        "surface": (
            SurfaceDeformableObjectCfg,
            SurfaceDeformableObject,
            surface_deformable_desc_from_cfg,
            "cloth_object",
        ),
    },
    "newton": {},
}


@configclass
class SimulationManagerCfg:
    """Global robot simulation configuration."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        headless: bool = False,
        render_cfg: RenderCfg | None = None,
        gpu_id: int = 0,
        thread_mode: ThreadMode = ThreadMode.RENDER_SHARE_ENGINE,
        cpu_num: int = 1,
        num_envs: int = 1,
        arena_space: float = 5.0,
        physics_dt: float | None = None,
        device: str | torch.device | None = None,
        physics_cfg: PhysicsBackendCfg | None = None,
        sim_device: str | torch.device | None = None,
        physics_config: DefaultPhysicsCfg | None = None,
        gpu_memory_config: GPUMemoryCfg | None = None,
        profiler: ProfilerCfg | None = None,
        visualization: VisualizationCfg | None = None,
        window_record: WindowRecordCfg | None = None,
        window_camera_pose: WindowCameraPoseCfg | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.headless = headless
        self.render_cfg = RenderCfg() if render_cfg is None else render_cfg
        self.gpu_id = gpu_id
        self.thread_mode = thread_mode
        self.cpu_num = cpu_num
        self.num_envs = num_envs
        self.arena_space = arena_space
        if physics_cfg is None:
            physics_cfg = (
                DefaultPhysicsCfg() if physics_config is None else physics_config
            )
        self.physics_cfg = physics_cfg
        if gpu_memory_config is not None:
            if not isinstance(self.physics_cfg, DefaultPhysicsCfg):
                logger.log_error(
                    "gpu_memory_config is only supported by the default physics backend.",
                    ValueError,
                )
            self.physics_cfg.gpu_memory = gpu_memory_config
        self.profiler = profiler
        self.visualization = (
            VisualizationCfg() if visualization is None else visualization
        )
        self.window_record = (
            WindowRecordCfg() if window_record is None else window_record
        )
        self.window_camera_pose = (
            WindowCameraPoseCfg() if window_camera_pose is None else window_camera_pose
        )
        if physics_dt is not None:
            self.physics_cfg.physics_dt = physics_dt
        runtime_device = device if device is not None else sim_device
        if runtime_device is not None:
            # Env tensors may use CPU while Newton/Warp sim stays on CUDA for GPU render sync.
            if isinstance(self.physics_cfg, NewtonPhysicsCfg):
                torch_device = (
                    torch.device(runtime_device)
                    if isinstance(runtime_device, str)
                    else runtime_device
                )
                if torch_device.type != "cpu":
                    self.physics_cfg.device = runtime_device
            else:
                self.physics_cfg.device = runtime_device

        self.__post_init__()

    width: int = 1920
    """The width of the simulation window."""

    height: int = 1080
    """The height of the simulation window."""

    headless: bool = False
    """Whether to run without an automatically opened native window.

    This is forced to ``True`` when the Viser backend is enabled. Viser and
    the native DexSim window are mutually exclusive; browser Gizmos do not
    require a native window.
    """

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

    physics_cfg: PhysicsBackendCfg = field(default_factory=DefaultPhysicsCfg)
    """Physics backend configuration (type selects default vs Newton backend)."""

    profiler: ProfilerCfg | None = None
    """Optional simulation profiler. ``None`` disables profiling.

    Standalone calls to :meth:`SimulationManager.update` are recorded below a
    ``sim_update`` root. When the manager is owned by an environment, the same
    profiler instance composes with the environment's step/reset hierarchy.
    """

    window_record: WindowRecordCfg = field(default_factory=WindowRecordCfg)
    """Viewer window recording settings (hotkey, paths, FPS, memory budget)."""

    window_camera_pose: WindowCameraPoseCfg = field(default_factory=WindowCameraPoseCfg)
    """Interactive viewer camera-pose printing settings."""

    visualization: VisualizationCfg = field(default_factory=VisualizationCfg)
    """Live browser visualization settings."""

    def __post_init__(self) -> None:
        """Validate physics and apply visualization-dependent defaults."""
        validate_physics_cfg(self.physics_cfg)
        if self.visualization.backend == "viser":
            self.headless = True

    @property
    def physics_dt(self) -> float:
        """The time step for the physics simulation."""
        return self.physics_cfg.physics_dt

    @physics_dt.setter
    def physics_dt(self, value: float) -> None:
        self.physics_cfg.physics_dt = value

    @property
    def device(self) -> str | torch.device:
        """The device for the physics simulation."""
        return self.physics_cfg.device

    @device.setter
    def device(self, value: str | torch.device) -> None:
        self.physics_cfg.device = value

    @property
    def sim_device(self) -> str | torch.device:
        """Legacy alias for :attr:`device`."""
        return self.device

    @sim_device.setter
    def sim_device(self, value: str | torch.device) -> None:
        self.device = value

    @property
    def physics_config(self) -> PhysicsBackendCfg:
        """Legacy alias for :attr:`physics_cfg`."""
        return self.physics_cfg

    @physics_config.setter
    def physics_config(self, value: PhysicsBackendCfg) -> None:
        validate_physics_cfg(value)
        self.physics_cfg = value

    @property
    def gpu_memory_config(self) -> GPUMemoryCfg | None:
        """Legacy alias for the default backend GPU-memory configuration."""
        if not isinstance(self.physics_cfg, DefaultPhysicsCfg):
            return None
        return self.physics_cfg.gpu_memory

    @gpu_memory_config.setter
    def gpu_memory_config(self, value: GPUMemoryCfg) -> None:
        if not isinstance(self.physics_cfg, DefaultPhysicsCfg):
            raise AttributeError(
                "gpu_memory_config is unavailable for the Newton physics backend."
            )
        self.physics_cfg.gpu_memory = value


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


@dataclass(frozen=True)
class _AxisMarkerGroup:
    """Native axis handles and their backend-neutral display dimensions."""

    handles: tuple[MeshObject, ...]
    arena_index: int
    axis_length: float
    axis_radius: float


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
        instance._is_constructed = False
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

        # Initialize physics backend (selected by the type of physics_cfg).
        # The backend is held as an instance member; SimulationManager delegates
        # all backend-specific lifecycle/scene/capability logic to it instead of
        # branching on a backend name throughout the manager.
        self.physics = make_physics_backend(sim_config.physics_cfg, self)

        world_config = self._convert_sim_config(sim_config)
        self.profiler = Profiler(sim_config.profiler, self.device)

        # Initialize Warp before creating the world. For Newton, honor the
        # configured startup/kernel-log suppression from the very first init.
        _initialize_warp_runtime(sim_config.physics_cfg)
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

        self._world.set_delta_time(sim_config.physics_cfg.physics_dt)
        self._world.show_coordinate_axis(False)

        # Activate the physics backend now that the dexsim World exists.
        self.physics.activate(sim_config)

        # activate physics
        self.enable_physics(True)

        self._env = self._world.get_env()

        # arena is used as a standalone space for robots to simulate in.
        self._arenas: List[dexsim.environment.Arena] = []

        # gizmo management
        self._gizmos: Dict[str, object] = dict()  # Store active gizmos

        # marker management
        self._markers: dict[str, _AxisMarkerGroup] = {}

        self._rigid_objects: Dict[str, RigidObject] = dict()
        self._constraints: Dict[str, RigidConstraint] = dict()
        self._rigid_object_groups: Dict[str, RigidObjectGroup] = dict()
        self._deformable_objects: Dict[str, DeformableObject] = dict()
        self._articulations: Dict[str, Articulation] = dict()
        self._robots: Dict[str, Robot] = dict()

        self._sensors: Dict[str, BaseSensor] = dict()
        self._pending_sensor_attachments: list[Camera] = []
        self._lights: Dict[str, Light] = dict()

        self._spawn_scene = SpawnScene(
            self._world,
            num_envs=sim_config.num_envs,
            spacing=(sim_config.arena_space, sim_config.arena_space, 0.0),
        )
        self._arenas = list(self._spawn_scene.builder.prepare_arenas())
        self._prepared_spawn_topology_revision = -1
        self._synced_spawn_render_topology_revision = -1

        self._visualization_runtime = None
        self._visualization_overlays: SceneOverlays | None = None
        self._visualization_topology_revision = 0
        self._visualization_manifest_topology_revision = -1
        self._visualization_sim_step = 0
        self._visualization_sim_time = 0.0
        self._visualization_error_reported = False

        # material placeholder.
        self._visual_materials: Dict[str, VisualMaterial] = dict()

        # Global texture cache for material creation or randomization.
        # The structure is keys to the loaded texture data. The keys represent the texture group.
        self._texture_cache: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = dict()

        self._init_sim_resources()

        # The plane material and visibility are authored before declaration so
        # both eager Default loading and deferred Newton loading see them.
        self._spawn_default_plane_visibility = True
        self._default_plane = None
        self.set_default_background()
        self._declare_spawn_default_plane()
        self.set_default_global_lighting()

        # Set physics to manual update mode by default.
        self.set_manual_update(True)
        if sim_config.headless is False:
            self._window = self._world.get_windows()

        self._is_constructed = True

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
        return self.sim_config.num_envs

    @property
    def spawn_result(self) -> "SpawnResult | None":
        """Return the current SpawnResult, or ``None`` before first prepare."""
        spawn_scene = getattr(self, "_spawn_scene", None)
        if spawn_scene is None or not spawn_scene.builder.is_finalized:
            return None
        return spawn_scene.builder.result

    @property
    def is_use_gpu_physics(self) -> bool:
        """Whether the active physics backend is running on GPU."""
        return self.device.type == "cuda"

    @property
    def physics_backend(self) -> str:
        """Return the active physics backend name."""
        return self.physics.name

    @property
    def is_default_backend(self) -> bool:
        """Whether the Default physics backend is active."""
        return self.physics.name == "default"

    @property
    def is_newton_backend(self) -> bool:
        """Whether the Newton physics backend is active."""
        return self.physics.name == "newton"

    @property
    def _active_newton_solver_type(self) -> str | None:
        """Return the resolved Newton solver without widening the base contract."""
        if isinstance(self.physics, NewtonPhysicsBackend):
            return self.physics.solver_type
        return None

    @property
    def newton_manager(self):
        """Compatibility accessor for the removed NewtonManager API.

        A non-Newton backend still returns ``None``. The Newton backend raises
        an actionable error because Spawn owns its World-level runtime and no
        independent NewtonManager exists.
        """
        if not self.is_newton_backend:
            logger.log_warning("Newton backend is not active.")
            return None
        return self.physics.newton_manager

    @property
    def differentiable_runtime(self):
        """Return the differentiable facade over the Spawn-owned Newton runtime."""
        if not self.is_newton_backend:
            raise RuntimeError(
                "differentiable_runtime requires the Newton physics backend."
            )
        return self.physics.differentiable_runtime

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
        uid_list.extend(list(self._deformable_objects.keys()))
        uid_list.extend(list(self._articulations.keys()))
        return uid_list

    @property
    def visualization_runtime(self) -> VisualizationRuntime | None:
        """Return the active visualization runtime, if one has been started."""
        return self._visualization_runtime

    @property
    def visualization_health(self) -> RuntimeHealth:
        """Return current visualization service and client health."""
        from embodichain.lab.visualization import RuntimeHealth

        if self._visualization_runtime is not None:
            return self._visualization_runtime.health
        configured = self.sim_config.visualization.backend == "viser"
        return RuntimeHealth(
            status="stopped" if configured else "disabled",
            running=False,
            endpoint=None,
            client_count=0,
            published_scene_revision=0,
        )

    @property
    def visualization_stats(self) -> RuntimeStats | None:
        """Return visualization telemetry, or ``None`` before startup."""
        if self._visualization_runtime is None:
            return None
        return self._visualization_runtime.stats

    @property
    def visualization_overlays(self) -> SceneOverlays | None:
        """Return the overlays included in every Viser scene frame."""
        return self._visualization_overlays

    def set_visualization_overlays(self, overlays: SceneOverlays | None) -> None:
        """Set persistent overlays for automatic Viser captures.

        The overlays remain active across :meth:`update` calls until replaced
        or cleared with ``None``. When Viser is running, the new overlays are
        published immediately.

        Args:
            overlays: Backend-neutral overlays to publish with every frame, or
                ``None`` to clear all persistent overlays.
        """
        self._visualization_overlays = overlays
        if (
            self.sim_config.visualization.backend == "viser"
            and self._visualization_runtime is not None
        ):
            self.capture_visualization_safely(force=True)

    def notify_visualization_topology_changed(self) -> int:
        """Mark scene topology dirty and return its new local revision."""
        self._visualization_topology_revision += 1
        return self._visualization_topology_revision

    def start_visualization(self) -> VisualizationRuntime | None:
        """Start the configured live visualizer and publish the current scene."""
        if self.sim_config.visualization.backend == "none":
            return None
        if getattr(self, "_spawn_scene", None) is not None:
            self.prepare()
        if getattr(self, "is_window_opened", False):
            raise RuntimeError(
                "Cannot start the Viser backend while the native DexSim window "
                "is open. Close the native window before starting Viser."
            )
        if self._visualization_runtime is not None:
            if self._visualization_runtime.is_running:
                return self._visualization_runtime
            self._visualization_runtime.stop()
            self._visualization_runtime = None

        from embodichain.lab.visualization import SceneExporter, VisualizationRuntime

        visualization_cfg = self.sim_config.visualization
        if (
            visualization_cfg.allow_commands
            and visualization_cfg.viser_server.host
            not in {"127.0.0.1", "localhost", "::1"}
        ):
            logger.log_warning(
                "Viser simulation commands are enabled on a non-loopback interface. "
                "Only expose this endpoint behind a trusted, authenticated boundary."
            )
        runtime = VisualizationRuntime(
            SceneExporter(self, visualization_cfg),
            visualization_cfg,
        )
        runtime.start()
        self._visualization_runtime = runtime
        self._visualization_manifest_topology_revision = (
            self._visualization_topology_revision
        )
        self._visualization_error_reported = False
        logger.log_info(f"Viser visualization ready at {runtime.endpoint}")
        runtime.capture(
            sim_step=self._visualization_sim_step,
            sim_time=self._visualization_sim_time,
            overlays=self._visualization_overlays,
            force=True,
        )
        return runtime

    def refresh_visualization(self) -> SceneManifest | None:
        """Publish current scene topology when Viser is active."""
        runtime = self.start_visualization()
        if runtime is None:
            return None
        for _, gizmo in self.get_gizmo_items():
            cancel = getattr(gizmo, "cancel_interaction", None)
            if cancel is not None:
                cancel("viser:")
        manifest = runtime.refresh_scene()
        self._visualization_manifest_topology_revision = (
            self._visualization_topology_revision
        )
        return manifest

    def capture_visualization(
        self,
        force: bool = False,
        *,
        capture_camera_images: bool = True,
    ) -> bool:
        """Capture current scene data for the configured visualizer.

        Args:
            force: Whether to bypass visualization frame-rate limiting.
            capture_camera_images: Whether camera images may be captured.

        Returns:
            Whether scene or camera data was captured.
        """
        runtime = self.start_visualization()
        if runtime is None:
            return False
        if (
            self._visualization_manifest_topology_revision
            != self._visualization_topology_revision
        ):
            self.refresh_visualization()
        return runtime.capture(
            sim_step=self._visualization_sim_step,
            sim_time=self._visualization_sim_time,
            overlays=self._visualization_overlays,
            force=force,
            capture_camera_images=capture_camera_images,
        )

    def capture_visualization_safely(
        self,
        force: bool = False,
        *,
        capture_camera_images: bool = True,
    ) -> None:
        """Update visualization without allowing failures to stop simulation.

        The first visualization failure is logged and subsequent captures are
        skipped until the runtime is restarted.

        Args:
            force: Whether to bypass visualization frame-rate limiting.
            capture_camera_images: Whether camera images may be captured.
        """
        if self._visualization_error_reported:
            return
        try:
            self.capture_visualization(
                force=force,
                capture_camera_images=capture_camera_images,
            )
        except Exception as error:
            if not self._visualization_error_reported:
                logger.log_warning(f"Viser visualization update failed: {error!r}")
                self._visualization_error_reported = True

    def stop_visualization(self) -> None:
        """Stop the visualization server and release its worker thread."""
        runtime = self._visualization_runtime
        if runtime is None:
            return
        try:
            runtime.stop()
        finally:
            for _, gizmo in self.get_gizmo_items():
                cancel = getattr(gizmo, "cancel_interaction", None)
                if cancel is not None:
                    cancel("viser:")
            self._visualization_runtime = None

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

        if sim_config.render_cfg.renderer == "auto":
            from embodichain.lab.sim.utility.render_utils import (
                select_default_renderer,
            )

            resolved_renderer = select_default_renderer(sim_config.gpu_id)
            logger.log_info(
                f"Auto-selected '{resolved_renderer}' renderer for gpu_id={sim_config.gpu_id}."
            )
            sim_config.render_cfg.renderer = resolved_renderer

        sim_config.render_cfg.apply_to_dexsim_config(world_config)

        if type(sim_config.device) is str:
            self.device = torch.device(sim_config.device)
        else:
            self.device = sim_config.device

        if self.device.type == "cuda":
            if self.device.index is not None and sim_config.gpu_id != self.device.index:
                logger.log_warning(
                    f"Conflict gpu_id {sim_config.gpu_id} and device index {self.device.index}. Using device index."
                )
                sim_config.gpu_id = self.device.index

                self.device = torch.device(f"cuda:{sim_config.gpu_id}")

        world_config.gpu_id = sim_config.gpu_id

        # Apply backend-specific WorldConfig fields (default tolerances/GPU flags
        # or the Newton cfg) via the active backend.
        self.physics.configure_world(world_config, sim_config)

        return world_config

    def _init_sim_resources(self) -> None:
        """Initialize the default simulation resources."""
        from embodichain.data.assets import SimResources

        self._default_resources = SimResources()

    def prepare(self) -> None:
        """Materialize declarations, bind state, and resolve sensor parents."""
        scene = self._spawn_scene
        result = scene.builder.result
        if (
            not scene.builder.is_finalized
            or result is None
            or result.needs_rebuild
            or scene.builder.has_pending_changes
        ):
            result = scene.commit()
            self._env = result.get_arena("default")
            self._arenas = [result.get_arena(name) for name in scene.arena_names]
            self.__dict__.pop("arena_offsets", None)
            if self._default_plane is None:
                self._bind_default_plane(scene.handles("default_plane")[0])

        # Runtime readiness belongs to the SimulationManager. Keep this and
        # facade binding outside the topology-change branch so a failed call
        # remains retryable without rematerializing the scene.
        scene.prepare_runtime_config(result)
        self._prepare_spawn_runtime(result)
        scene.bind()
        self._sync_spawn_render_state(result)

        while self._pending_sensor_attachments:
            sensor = self._pending_sensor_attachments[0]
            self._attach_camera_parent(sensor)
            self._pending_sensor_attachments.pop(0)

    def _prepare_spawn_runtime(self, result: dexsim.spawn.SpawnResult) -> None:
        """Prepare backend runtime buffers for one Spawn topology revision."""
        topology_revision = int(result.topology_revision)
        if getattr(self, "_prepared_spawn_topology_revision", -1) == topology_revision:
            return
        if self.is_default_backend and self.device.type == "cuda":
            self._world.init_gpu_physics()
        self._prepared_spawn_topology_revision = topology_revision

    def _sync_spawn_render_state(self, result: dexsim.spawn.SpawnResult) -> None:
        """Publish newly bound state once for each Spawn topology revision."""
        topology_revision = int(result.topology_revision)
        if (
            getattr(self, "_synced_spawn_render_topology_revision", -1)
            == topology_revision
        ):
            return
        self.physics.sync_render_state(result)
        self._synced_spawn_render_topology_revision = topology_revision

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
        if not self.physics.can_disable_manual_update and enable is False:
            logger.log_warning(
                "The active physics backend does not support switching between "
                "manual and automatic update. Ignoring set_manual_update call."
            )
            return
        self._world.set_manual_update(enable)

    def init_gpu_physics(self) -> None:
        """Prepare the Spawn-owned physics runtime.

        This backwards-compatible alias now has the same backend-neutral
        behavior as :meth:`prepare`.
        """
        self.prepare()

    def finalize_newton_physics(self) -> None:
        """Prepare the Spawn-owned physics runtime.

        This backwards-compatible alias now has the same backend-neutral
        behavior as :meth:`prepare`.
        """
        self.prepare()

    def create_differentiable_stepper(self):
        """Create a single-step differentiable physics primitive (Newton-only).

        Requires the Newton backend with ``requires_grad=True`` and
        ``solver_type="semi_implicit"``. Delegates to
        :meth:`dexsim.engine.newton_physics.NewtonManager.create_differentiable_stepper`.

        Raises:
            RuntimeError: If the active backend is not Newton or if the
                Newton manager is not ready / not in grad mode.
        """
        if not self.is_newton_backend:
            logger.log_error(
                "create_differentiable_stepper requires the Newton backend."
            )
        return self.differentiable_runtime.create_differentiable_stepper()

    def create_gradient_rollout(
        self,
        record_steps: int,
        substeps_per_record: int | None = None,
        record_dt: float | None = None,
    ):
        """Create a gradient rollout buffer (Newton-only).

        Delegates to
        :meth:`dexsim.engine.newton_physics.NewtonManager.create_gradient_rollout`.

        Args:
            record_steps: Number of record points to capture in the rollout
                buffer.
            substeps_per_record: Newton substeps between successive record
                points. Defaults to the Newton manager's configured
                ``num_substeps``.
            record_dt: Time interval between successive record points.
                Defaults to the Newton manager's configured ``dt``.

        Raises:
            RuntimeError: If the active backend is not Newton or if the
                Newton manager is not ready / not in grad mode.
        """
        if not self.is_newton_backend:
            logger.log_error("create_gradient_rollout requires the Newton backend.")
        return self.differentiable_runtime.create_gradient_rollout(
            record_steps=record_steps,
            substeps_per_record=substeps_per_record,
            record_dt=record_dt,
        )

    def render_camera_group(self, group_ids: list[int]) -> None:
        """Render all camera group in the simulation.

        Args:
            group_ids (list[int]): The list of camera group ids to render.

        Note: This interface is only valid when Ray Tracing rendering backend is enabled.
        """

        self._world.render_camera_group(group_ids)

    def update(self, physics_dt: float | None = None, step: int = 1) -> None:
        """Update the physics.

        Args:
            physics_dt (float | None, optional): the time step for physics simulation. Defaults to None.
            step (int, optional): the number of :meth:`World.update` calls per invocation. Defaults to 1.
        """
        with self.profiler.section("sim_update", is_root=True):
            with self.profiler.section("gpu_physics_check"):
                self.prepare()

            if self.is_physics_manually_update:
                with self.profiler.section("manual_update"):
                    if physics_dt is None:
                        with self.profiler.section("resolve_physics_dt"):
                            physics_dt = self.sim_config.physics_dt
                    for i in range(step):
                        with self.profiler.section("gizmo_update"):
                            self.update_gizmos()
                        with self.profiler.section("world_update"):
                            with _temporary_warp_kernel_log_suppression(
                                self.sim_config.physics_cfg
                            ):
                                self._world.update(physics_dt)
                        self._visualization_sim_step += 1
                        self._visualization_sim_time += physics_dt
                        if (
                            self._window_record_state is not None
                            and self._window_record_state.capture_from_sim_update
                        ):
                            with self.profiler.section("window_record_capture"):
                                self._step_window_record_from_sim_update(
                                    self._window_record_state, physics_dt
                                )
                        if self.sim_config.visualization.backend == "viser":
                            with self.profiler.section("visualization_capture"):
                                self.capture_visualization_safely(
                                    capture_camera_images=i == step - 1
                                )

            else:
                with self.profiler.section("manual_update_disabled"):
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

    def visualize_point_cloud(
        self,
        points: torch.Tensor | np.ndarray,
        colors: torch.Tensor | np.ndarray | None = None,
        point_size: float = 2.0,
        name: str = "point_cloud",
    ) -> dexsim.models.PointCloud:
        """Visualize a static point cloud in the native simulation viewer.

        Each invocation creates a separate native point-cloud object. This
        convenience API is intended for static data, not incremental or
        streaming updates.

        Args:
            points: Point positions with shape ``(N, 3)``.
            colors: Optional per-point RGB or RGBA colors with shape ``(N, 3)``
                or ``(N, 4)``. Values in ``[0, 255]`` are normalized to
                ``[0, 1]``. The alpha channel of RGBA input is ignored by the
                native renderer. Defaults to green.
            point_size: Native renderer point size. Defaults to ``2.0``.
            name: Name assigned to the native point-cloud object.

        Returns:
            The native DexSim point-cloud handle.

        Raises:
            RuntimeError: If there is no active simulation environment.
            ValueError: If the points or colors do not have a supported shape.
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
        if len(points) == 0:
            raise ValueError("Points array is empty")

        if colors is None:
            colors = np.tile(
                np.array((0.0, 1.0, 0.0), dtype=np.float32), (len(points), 1)
            )
        else:
            if isinstance(colors, torch.Tensor):
                colors = colors.detach().cpu().numpy()
            colors = np.asarray(colors)
            if colors.ndim != 2 or colors.shape[0] != len(points):
                raise ValueError(
                    f"Colors must have shape ({len(points)}, 3) or ({len(points)}, 4), "
                    f"got {colors.shape}"
                )
            if colors.shape[1] not in (3, 4):
                raise ValueError(
                    "Colors must have 3 (RGB) or 4 (RGBA) channels, "
                    f"got {colors.shape[1]}"
                )
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = np.asarray(colors[:, :3], dtype=np.float32)

        env = self.get_env()
        if env is None:
            raise RuntimeError("Simulation manager has no active simulation")

        point_cloud = env.create_point_cloud(name=name)
        point_cloud.add_points(points)
        point_cloud.set_colors(colors)
        point_cloud.set_point_size(point_size)

        logger.log_info(
            f"Created point cloud '{name}' with {len(points)} points "
            f"(point_size={point_size})"
        )
        return point_cloud

    def get_world(self) -> dexsim.World:
        return self._world

    def get_physics_scene(self) -> "PhysicsScene":
        """Return the Default backend's compatibility scene after Spawn preparation.

        Newton has no ``PhysicsScene`` facade and raises with guidance to use
        :attr:`spawn_result` instead.
        """
        return self.physics.get_scene()

    def can_open_native_window(self) -> bool:
        """Return whether the native DexSim window may be opened.

        The Viser backend owns visualization while it is configured or
        running, so a native window must not be opened for the same simulation.

        Returns:
            ``True`` unless the Viser backend is configured or running.
        """
        return (
            self.sim_config.visualization.backend != "viser"
            and self._visualization_runtime is None
        )

    def open_window(self) -> bool:
        """Open the native DexSim simulation window when allowed.

        Viser owns visualization while it is configured or running. In that
        case this method safely skips the native window so launchers do not
        need a separate Viser condition.

        Returns:
            ``True`` when the native window is open, otherwise ``False``.
        """
        if not self.can_open_native_window():
            logger.log_info(
                "Skipping the native DexSim window because the Viser backend "
                "is configured or running."
            )
            return False
        if self.is_window_opened:
            return True
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
        return True

    def close_window(self) -> None:
        """Close the simulation window."""
        if self.is_window_recording():
            self.stop_window_record()
        self._world.close_window()
        self._window = None
        self._window_record_input_control = None
        self._window_camera_pose_input_control = None
        self.is_window_opened = False

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

    def _declare_spawn_default_plane(self) -> None:
        """Declare the global ground in the World's Spawn scene."""

        from dexsim.spawn import (
            CollisionApproximation,
            CollisionDesc,
            DexsimCollisionDesc,
            GeometryDesc,
            NewtonCollisionDesc,
            ObjectDesc,
            RenderDesc,
            RigidBodyPhysicsDesc,
        )

        default_length = 1000.0
        geometry = GeometryDesc.plane(default_length)
        repeat_uv_size = default_length / 2.0
        render = RenderDesc.from_geometry(
            geometry,
            material=self._spawn_default_plane_material,
        )
        render.uv_coords = np.asarray(
            [
                [0.0, 0.0],
                [repeat_uv_size, 0.0],
                [repeat_uv_size, repeat_uv_size],
                [0.0, repeat_uv_size],
            ],
            dtype=np.float32,
        )
        collision = CollisionDesc.from_geometry(
            geometry,
            approximation=CollisionApproximation.NONE,
        )
        collision.dexsim = DexsimCollisionDesc(
            dynamic_friction=0.5,
            static_friction=0.5,
        )
        collision.newton = NewtonCollisionDesc(mu=0.5)
        collision.render_source_index = 0
        descriptor = ObjectDesc(
            name="default_plane",
            renders=[render],
            collisions=[collision],
            physics=RigidBodyPhysicsDesc.static(),
            per_env=False,
        )

        self._spawn_scene.declare(
            "rigid_object",
            "default_plane",
            descriptor,
        )
        handles = self._spawn_scene.handles("default_plane")
        if handles:
            self._bind_default_plane(handles[0])

    def _bind_default_plane(self, plane: Any) -> None:
        """Retain the spawned ground plane and apply its visibility."""
        self._default_plane = plane
        plane.set_visible(self._spawn_default_plane_visibility)

    def set_default_global_lighting(self) -> None:
        """Set default global lighting for the scene.

        Configures both the environment emission (ambient) light and a
        directional light to provide default scene illumination. The
        directional light is a global scene light (infinite distance)
        pointing downward along the -Z axis.
        """
        # Environment emission light
        self.set_emission_light([1.0, 1.0, 1.0], 100.0)

    def set_default_background(self) -> None:
        """Set default background."""

        mat_name = "plane_mat"
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

        material = mat.get_instance("plane_mat").mat
        # Consumed by _declare_spawn_default_plane(). Keeping the native
        # material in the descriptor preserves the VisualMaterial registry
        # used by visual randomization without forcing finalization.
        self._spawn_default_plane_material = material
        self._visual_materials[mat_name] = mat

    def set_ground_plane_visibility(self, visible: bool) -> None:
        """_summary_

        Args:
            visible (bool): _description_
        """
        self._spawn_default_plane_visibility = bool(visible)
        if self._default_plane is None:
            return
        self._default_plane.set_visible(bool(visible))

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
    ) -> (
        Light
        | BaseSensor
        | Robot
        | RigidObject
        | RigidObjectGroup
        | DeformableObject
        | Articulation
        | None
    ):
        """Get an asset by its UID.

        The asset can be a light, sensor, robot, rigid object, deformable, or
        articulation.

        Args:
            uid (str): The UID of the asset.

        Returns:
            The asset instance if found, otherwise ``None``.
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
        if uid in self._deformable_objects:
            return self._deformable_objects[uid]
        if uid in self._articulations:
            return self._articulations[uid]

        logger.log_warning(f"Asset {uid} not found.")
        return None

    _LIGHT_TYPE_MAP: dict[str, LightType] = {
        "point": LightType.POINT,
        "sun": LightType.SUN,
        "direction": LightType.DIRECTION,
        "spot": LightType.SPOT,
        "rect": LightType.RECT,
        "mesh": LightType.MESH,
    }
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
            ValueError: If ``cfg.light_type`` is not supported.
        """
        if cfg.uid is None:
            uid = "light"
            cfg.uid = uid
        else:
            uid = cfg.uid

        if uid in self._lights:
            logger.log_error(f"Light {uid} already exists.")

        light_type = self._LIGHT_TYPE_MAP.get(cfg.light_type)
        if light_type is None:
            supported = ", ".join(self._LIGHT_TYPE_MAP)
            raise ValueError(
                f"Unsupported light type {cfg.light_type!r}. "
                f"Supported types: {supported}."
            )

        if cfg.light_type == "mesh" and not cfg.mesh_path:
            logger.log_warning(
                f"Mesh light '{uid}' has no mesh_path set. "
                f"Use set_mesh() to assign a MeshObject."
            )
        if cfg.light_type == "rect" and (cfg.rect_width <= 0 or cfg.rect_height <= 0):
            logger.log_warning(
                f"Rect light '{uid}' has zero or negative dimensions "
                f"(width={cfg.rect_width}, height={cfg.rect_height})."
            )

        if cfg.light_type in self._GLOBAL_LIGHT_TYPES:
            batch_lights = Light(
                cfg=cfg,
                entities=[self._env.create_light(uid, light_type)],
            )
        else:
            batch_lights = Light(
                cfg=cfg,
                entities=[
                    arena.create_light(f"{uid}_{index}", light_type)
                    for index, arena in enumerate(self._arenas)
                ],
            )

        self._lights[uid] = batch_lights
        self.notify_visualization_topology_changed()
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

    def add_usd(
        self,
        name: str,
        file_path: str,
        *,
        pose: np.ndarray | None = None,
        robot_cfgs: dict[str, RobotCfg] | None = None,
    ) -> dict[str, RigidObject | Articulation | Robot]:
        """Declare the supported entities in a USD scene.

        The returned facades are keyed by their USD prim paths. They remain in
        declared state until :meth:`prepare` finalizes the shared Spawn scene,
        then bind in place to the resulting DexSim handles.

        USD does not identify which articulations should expose EmbodiChain's
        robot interface. Pass those explicitly through ``robot_cfgs``; all
        other articulation descriptions become :class:`Articulation` objects.

        Args:
            name: Name passed to DexSim's USD scene parser.
            file_path: USD, USDA, or USDC file path.
            pose: Optional scene-root transform.
            robot_cfgs: Robot configurations keyed by USD prim path. These
                provide robot-side metadata while physics remains authored by
                the USD scene.

        Returns:
            Supported EmbodiChain facades keyed by USD prim path.

        Raises:
            RuntimeError: If called after the Spawn scene was finalized.
        """
        if self.spawn_result is not None:
            raise RuntimeError(
                "add_usd() must be called before SimulationManager.prepare()."
            )

        from dexsim.spawn import ArticulationDesc, MeshObjectDesc

        descriptors = self._spawn_scene.builder.add_usd(
            name,
            file_path,
            pose=pose,
            per_env=True,
        )
        assets: dict[str, RigidObject | Articulation | Robot] = {}
        robot_cfgs = robot_cfgs or {}

        for descriptor in descriptors:
            source_path = (
                descriptor.usd.prim_path
                if descriptor.usd is not None and descriptor.usd.prim_path
                else descriptor.name
            )

            if type(descriptor) is MeshObjectDesc:
                body_type = "static"
                if descriptor.physics is not None:
                    body_type = {
                        ActorType.DYNAMIC: "dynamic",
                        ActorType.KINEMATIC: "kinematic",
                        ActorType.STATIC: "static",
                    }[descriptor.physics.actor_type]
                cfg = RigidObjectCfg(
                    uid=descriptor.name,
                    init_local_pose=descriptor.pose.copy(),
                    body_type=body_type,
                    body_scale=tuple(float(value) for value in descriptor.body_scale),
                    asset_physics_mode="preserve",
                )
                facade = RigidObject(
                    cfg=cfg,
                    entities=None,
                    device=self.device,
                    declared_num_instances=self.sim_config.num_envs,
                )

                self._spawn_scene.track(
                    "rigid_object",
                    descriptor.name,
                    descriptor,
                    facade=facade,
                )
                self._rigid_objects[descriptor.name] = facade
                assets[source_path] = facade
                continue

            if isinstance(descriptor, ArticulationDesc):
                robot_cfg = robot_cfgs.get(source_path)
                facade_type: type[Articulation] = (
                    Robot if robot_cfg is not None else Articulation
                )
                cfg = (
                    deepcopy(robot_cfg)
                    if robot_cfg is not None
                    else ArticulationCfg(uid=descriptor.name)
                )
                cfg.uid = descriptor.name
                cfg.fpath = file_path
                cfg.init_local_pose = descriptor.pose.copy()
                cfg.asset_physics_mode = "preserve"
                if robot_cfg is None:
                    cfg.root_props = ArticulationRootPropertiesCfg()
                else:
                    cfg.root_props = cfg.root_props.copy()
                cfg.root_props.fixed_base = bool(descriptor.fixed_base)
                cfg.root_props.self_collision_enabled = descriptor.enable_self_collision
                cfg.body_scale = tuple(float(value) for value in descriptor.body_scale)
                cfg.build_pk_chain = False
                facade = facade_type(
                    cfg=cfg,
                    entities=None,
                    device=self.device,
                    declared_num_instances=self.sim_config.num_envs,
                )

                self._spawn_scene.track(
                    "articulation",
                    descriptor.name,
                    descriptor,
                    facade=facade,
                )
                registry = (
                    self._robots if robot_cfg is not None else self._articulations
                )
                registry[descriptor.name] = facade
                assets[source_path] = facade

        self.notify_visualization_topology_changed()
        return assets

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
        uid = cfg.uid
        if uid is None:
            raise ValueError("Rigid object uid must be specified.")
        if uid in self._rigid_objects:
            raise ValueError(f"Rigid object {uid!r} already exists.")
        source_path = getattr(cfg.shape, "fpath", None)
        if _is_usd_path(source_path):
            descriptor, materials = rigid_desc_from_usd(
                cfg,
                per_env=True,
                newton_solver_type=self._active_newton_solver_type,
            )
        else:
            descriptor, materials = rigid_desc_from_cfg(
                cfg,
                per_env=True,
                newton_solver_type=self._active_newton_solver_type,
            )
        self._spawn_scene.builder.materials.update(materials)

        rigid_obj = RigidObject(
            cfg=cfg,
            entities=None,
            device=self.device,
            declared_num_instances=self.sim_config.num_envs,
        )

        was_materialized = self.spawn_result is not None
        self._spawn_scene.declare(
            "rigid_object",
            uid,
            descriptor,
            facade=rigid_obj,
        )
        self._rigid_objects[uid] = rigid_obj
        self.notify_visualization_topology_changed()

        # Preserve the legacy immediate-availability behavior for runtime
        # additions. Initial environment construction still batches all
        # declarations into one finalize at BaseEnv's prepare boundary.
        if was_materialized:
            self.prepare()
        return rigid_obj

    def add_deformable_object(self, cfg: DeformableObjectCfg) -> DeformableObject:
        """Declare a volume or surface deformable in the scene.

        DexSim is the only deformable implementation currently registered.
        Backend capability flags and the dispatch boundary are intentionally
        explicit so a future Newton adapter can be added without changing this
        public method or its callers.

        Args:
            cfg: Volume- or surface-deformable configuration.

        Returns:
            The declared deformable facade.

        Raises:
            NotImplementedError: If the active backend or device cannot host
                the requested deformable type.
            ValueError: If the discriminator or UID is invalid.
        """
        deformable_type = cfg.deformable_type
        if deformable_type == "volume":
            supported = self.physics.supports_volume_deformables
        elif deformable_type == "surface":
            supported = self.physics.supports_surface_deformables
        else:
            raise ValueError(
                f"Unsupported deformable_type {deformable_type!r}; expected "
                "'volume' or 'surface'."
            )
        if not supported:
            raise NotImplementedError(
                f"The {self.physics.name} backend does not yet provide a "
                f"{deformable_type}-deformable object adapter."
            )
        if self.device.type != "cuda":
            raise NotImplementedError(
                "DexSim deformable objects currently require a CUDA device."
            )
        if self.spawn_result is not None:
            raise NotImplementedError(
                "DexSim Spawn does not yet support adding deformables after "
                "finalization."
            )

        uid = cfg.uid
        if uid is None:
            raise ValueError("Deformable object uid must be specified.")
        if uid in self._deformable_objects:
            raise ValueError(f"Deformable object {uid!r} already exists.")

        backend_implementations = _DEFORMABLE_BACKEND_IMPLEMENTATIONS.get(
            self.physics.name
        )
        if not backend_implementations:
            raise NotImplementedError(
                f"No deformable implementation is registered for the "
                f"{self.physics.name} backend."
            )

        config_cls, object_cls, descriptor_factory, spawn_kind = (
            backend_implementations[deformable_type]
        )
        if not isinstance(cfg, config_cls):
            raise TypeError(
                f"A {deformable_type} deformable requires "
                f"{config_cls.__name__}, got {type(cfg).__name__}."
            )
        descriptor, materials = descriptor_factory(cfg, per_env=True)
        self._spawn_scene.builder.materials.update(materials)
        deformable = object_cls(
            cfg,
            entities=None,
            device=self.device,
            declared_num_instances=self.sim_config.num_envs,
        )
        self._spawn_scene.declare(
            spawn_kind,
            uid,
            descriptor,
            facade=deformable,
        )
        self._deformable_objects[uid] = deformable
        self.notify_visualization_topology_changed()
        return deformable

    def add_soft_object(self, cfg: SoftObjectCfg) -> SoftObject:
        """Compatibility wrapper for adding a volume deformable."""
        deformable = self.add_deformable_object(cfg)
        assert isinstance(deformable, VolumeDeformableObject)
        return deformable

    def add_cloth_object(self, cfg: ClothObjectCfg) -> ClothObject:
        """Compatibility wrapper for adding a surface deformable."""
        deformable = self.add_deformable_object(cfg)
        assert isinstance(deformable, SurfaceDeformableObject)
        return deformable

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

    def get_deformable_object(self, uid: str) -> DeformableObject | None:
        """Get a deformable object by its unique ID."""
        if uid not in self._deformable_objects:
            logger.log_warning(f"Deformable object {uid} not found.")
            return None
        return self._deformable_objects[uid]

    def get_soft_object(self, uid: str) -> SoftObject | None:
        """Get a volume deformable through the legacy soft-object API."""
        deformable = self._deformable_objects.get(uid)
        if not isinstance(deformable, VolumeDeformableObject):
            logger.log_warning(f"Soft object {uid} not found.")
            return None
        return deformable

    def get_cloth_object(self, uid: str) -> ClothObject | None:
        """Get a surface deformable through the legacy cloth-object API."""
        deformable = self._deformable_objects.get(uid)
        if not isinstance(deformable, SurfaceDeformableObject):
            logger.log_warning(f"Cloth object {uid} not found.")
            return None
        return deformable

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
        """Broadcast a local constraint frame to the selected environments."""
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
        """Create a fixed constraint between two rigid objects.

        Constraints are native Default-backend resources owned by each Arena.
        Spawn owns the two actors; this method only borrows their native actor
        handles while creating the constraint.

        Args:
            cfg: The constraint configuration.
            env_ids: Target environment indices. Accepts a tensor (as passed by
                the :class:`EventManager`) or a sequence of ints. None -> all arenas.

        Returns:
            The created constraint batch.
        """
        if hasattr(self, "physics") and not self.is_default_backend:
            raise NotImplementedError(
                "Rigid constraints are currently supported only by the Default "
                "backend."
            )
        if cfg.constraint_type != "fixed":
            logger.log_error(
                f"Constraint '{cfg.name}' has unsupported type "
                f"'{cfg.constraint_type}'. Only 'fixed' is supported."
            )
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
        if cfg.name in self._constraints:
            logger.log_error(
                f"Constraint '{cfg.name}' already exists. Remove it before recreating."
            )

        rigid_object_a = self._rigid_objects[cfg.rigid_object_a_uid]
        rigid_object_b = self._rigid_objects[cfg.rigid_object_b_uid]
        if hasattr(self, "_spawn_scene"):
            self.prepare()

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

        target_env_ids = self._normalize_env_ids(env_ids, num_envs)
        frames_a = self._broadcast_frame(
            cfg.local_frame_a, num_envs, target_env_ids, cfg.name
        )
        if cfg.local_frame_b is None:
            pose_a = rigid_object_a.get_local_pose(to_matrix=True)
            pose_b = rigid_object_b.get_local_pose(to_matrix=True)
            frame_b = (
                torch.bmm(pose_inv(pose_b), pose_a).cpu().numpy().astype(np.float32)
            )
            frames_b = [frame_b[i] for i in target_env_ids]
        else:
            frames_b = self._broadcast_frame(
                cfg.local_frame_b, num_envs, target_env_ids, cfg.name
            )

        handles: list = [None] * num_envs
        try:
            for index, env_id in enumerate(target_env_ids):
                actor_a = rigid_object_a._entities[env_id]
                actor_b = rigid_object_b._entities[env_id]
                if getattr(rigid_object_a, "is_spawn_bound", False) is True:
                    actor_a = actor_a.native
                if getattr(rigid_object_b, "is_spawn_bound", False) is True:
                    actor_b = actor_b.native
                if actor_a is None or actor_b is None:
                    logger.log_error(
                        f"Constraint '{cfg.name}' references a released Spawn actor "
                        f"in environment {env_id}."
                    )

                arena = self.get_env(env_id)
                name = cfg.name if num_envs <= 1 else f"{cfg.name}_{env_id}"
                handle = arena.create_fixed_constraint(
                    name,
                    actor_a,
                    actor_b,
                    frames_a[index],
                    frames_b[index],
                )
                if handle is None:
                    logger.log_error(
                        f"Failed to create constraint '{name}' in arena {env_id}."
                    )
                handles[env_id] = handle
        except Exception:
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

    def get_deformable_object_uid_list(self) -> List[str]:
        """Return all deformable object UIDs in declaration order."""
        return list(self._deformable_objects.keys())

    def get_soft_object_uid_list(self) -> List[str]:
        """Return volume-deformable UIDs through the legacy soft API."""
        return [
            uid
            for uid, asset in self._deformable_objects.items()
            if asset.deformable_type == "volume"
        ]

    def get_cloth_object_uid_list(self) -> List[str]:
        """Return surface-deformable UIDs through the legacy cloth API."""
        return [
            uid
            for uid, asset in self._deformable_objects.items()
            if asset.deformable_type == "surface"
        ]

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

        Returns:
            The stable Group facade. During initial scene construction it is
            bound to Spawn handles by :meth:`prepare`.
        """
        if not self.physics.supports_rigid_object_group:
            raise NotImplementedError(
                f"The {self.physics.name} backend does not support rigid object groups."
            )
        uid = cfg.uid
        if uid is None:
            raise ValueError("Rigid object group uid must be specified.")
        if uid in self._rigid_object_groups:
            raise ValueError(f"Rigid object group {uid!r} already exists.")
        if cfg.body_type == "static":
            raise ValueError("Rigid object group cannot be static.")
        if not cfg.rigid_objects:
            raise ValueError("Rigid object group must contain at least one object.")

        actor_type = {
            "dynamic": ActorType.DYNAMIC,
            "kinematic": ActorType.KINEMATIC,
        }[cfg.body_type]
        descriptors = []
        for index, member in enumerate(cfg.rigid_objects.values()):
            member_cfg = deepcopy(member)
            member_cfg.uid = f"{uid}__member_{index}"
            member_cfg.body_type = cfg.body_type
            source_path = getattr(member_cfg.shape, "fpath", None)
            if _is_usd_path(source_path):
                descriptor, materials = rigid_desc_from_usd(
                    member_cfg,
                    per_env=True,
                    newton_solver_type=self._active_newton_solver_type,
                )
            else:
                descriptor, materials = rigid_desc_from_cfg(
                    member_cfg,
                    per_env=True,
                    newton_solver_type=self._active_newton_solver_type,
                )
            if descriptor.physics is None:
                raise ValueError(
                    f"Rigid object group member {index} has no rigid-body physics."
                )
            descriptor.physics.actor_type = actor_type
            self._spawn_scene.builder.materials.update(materials)
            descriptors.append(descriptor)

        group = RigidObjectGroup(
            cfg,
            entities=None,
            device=self.device,
            declared_num_instances=self.sim_config.num_envs,
        )

        was_materialized = self.spawn_result is not None
        self._spawn_scene.declare(
            "rigid_object_group",
            uid,
            tuple(descriptors),
            facade=group,
        )
        self._rigid_object_groups[uid] = group
        self.notify_visualization_topology_changed()
        if was_materialized:
            self.prepare()
        return group

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
            if cfg.fpath is None:
                raise ValueError(
                    "Articulation configuration must provide fpath when uid "
                    "is not specified."
                )
            uid = os.path.splitext(os.path.basename(cfg.fpath))[0]
            cfg.uid = uid
        if uid in self._articulations:
            raise ValueError(f"Articulation {uid!r} already exists.")

        was_materialized = self.spawn_result is not None
        articulation = self._declare_spawn_articulation(cfg, Articulation)
        self._articulations[uid] = articulation
        if was_materialized:
            self.prepare()
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

    def add_robot(self, cfg: RobotCfg | RobotPresetCfg) -> Robot | None:
        """Add a Robot to the scene.

        Args:
            cfg: A concrete robot configuration or a replace-only backend
                preset. Presets are resolved from ``physics_cfg`` before the
                robot is declared.

        Returns:
            Robot | None: The added robot instance handle, or None if failed.
        """
        if not self.physics.supports_robot:
            logger.log_error(
                f"Robot support is not enabled for the "
                f"{self.physics.name} backend yet.",
                error_type=NotImplementedError,
            )

        if isinstance(cfg, RobotPresetCfg):
            cfg = cfg.resolve(
                self.sim_config.physics_cfg,
                newton_solver_type=self._active_newton_solver_type,
            )

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

        was_materialized = self.spawn_result is not None
        robot = self._declare_spawn_articulation(cfg, Robot)
        self._robots[uid] = robot
        if was_materialized:
            self.prepare()
        return robot

    def _declare_spawn_articulation(
        self,
        cfg: ArticulationCfg,
        facade_type: type[Articulation],
    ) -> Articulation:
        """Declare an articulation facade and bind its Batch after finalize.

        DexSim remains the sole articulation source loader. EmbodiChain applies
        regex/group configuration to the resolved descriptor before either
        backend materializes it. Runtime Batch data is created at the shared
        prepare boundary.
        """
        if _is_usd_path(cfg.fpath):
            descriptor, materials = articulation_desc_from_usd(
                cfg,
                per_env=True,
                newton_solver_type=self._active_newton_solver_type,
            )
            self._spawn_scene.builder.materials.update(materials)
        else:
            descriptor = articulation_desc_from_cfg(
                cfg,
                per_env=True,
                newton_solver_type=self._active_newton_solver_type,
            )
        if cfg.uid is None:
            cfg.uid = descriptor.name

        facade = facade_type(
            cfg=cfg,
            entities=None,
            device=self.device,
            declared_num_instances=self.sim_config.num_envs,
        )

        self._spawn_scene.declare(
            "articulation",
            descriptor.name,
            descriptor,
            facade=facade,
            configure_source=partial(
                configure_articulation_desc,
                cfg=cfg,
                newton_solver_type=self._active_newton_solver_type,
            ),
        )
        self.notify_visualization_topology_changed()
        return facade

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
        self,
        uid: str,
        control_part: str | None = None,
        gizmo_cfg: GizmoCfg | None = None,
        *,
        enable_native: bool | None = None,
    ) -> Gizmo | None:
        """Enable gizmo control for any simulation object (Robot, RigidObject, Camera, etc.).

        Args:
            uid: UID of the robot, rigid object, or camera sensor.
            control_part: Robot control part used for IK/FK.
            gizmo_cfg: Native and Viser Gizmo appearance configuration.
            enable_native: Whether to create a DexSim Gizmo. By default, native
                controls are created only when a native window is active.

        Returns:
            The created Gizmo, or ``None`` if setup failed.
        """
        # Create gizmo key combining uid and control_part
        gizmo_key = f"{uid}:{control_part}" if control_part else uid

        # Check if gizmo already exists
        if gizmo_key in self._gizmos:
            logger.log_warning(
                f"Gizmo for '{uid}' with control_part '{control_part}' already exists."
            )
            return self._gizmos[gizmo_key]

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
            return None

        if enable_native is None:
            enable_native = self.is_window_opened or not self.sim_config.headless
        gizmo: Gizmo | None = None
        try:
            gizmo = Gizmo(
                target,
                gizmo_cfg,
                control_part,
                enable_native=enable_native,
            )
            if enable_native and (
                not hasattr(self, "_gizmo_controller") or self._gizmo_controller is None
            ):
                window = (
                    self._world.get_windows()
                    if hasattr(self._world, "get_windows")
                    else None
                )
                if window is None:
                    raise RuntimeError(
                        "A native window is required for the DexSim Gizmo controller."
                    )
                self._gizmo_controller = GizmoController()
                window.add_input_control(self._gizmo_controller)
            self._gizmos[gizmo_key] = gizmo
            self.notify_visualization_topology_changed()
            logger.log_info(
                f"Gizmo enabled for {object_type} '{uid}' with control_part "
                f"'{control_part}' (native={enable_native}, "
                f"viser={self.sim_config.visualization.allow_commands})"
            )

        except Exception as e:
            if gizmo is not None:
                gizmo.destroy()
            logger.log_error(
                f"Failed to create gizmo for {object_type} '{uid}' with control_part '{control_part}': {e}"
            )
            return None

        return gizmo

    def disable_gizmo(self, uid: str, control_part: str | None = None) -> None:
        """Disable and remove a Gizmo.

        Args:
            uid: Target asset UID.
            control_part: Robot control part, if applicable.
        """
        gizmo_key = f"{uid}:{control_part}" if control_part else uid
        if gizmo_key not in self._gizmos:
            logger.log_warning(
                f"No gizmo found for '{uid}' with control_part '{control_part}'."
            )
            return

        try:
            gizmo = self._gizmos.pop(gizmo_key)
            try:
                if gizmo is not None:
                    gizmo.destroy()
            finally:
                self.notify_visualization_topology_changed()
            logger.log_info(
                f"Gizmo disabled for '{uid}' with control_part '{control_part}'"
            )
        except Exception as error:
            logger.log_error(
                f"Failed to disable gizmo for '{uid}' with control_part "
                f"'{control_part}': {error}"
            )

    def get_gizmo(
        self,
        uid: str,
        control_part: str | None = None,
    ) -> Gizmo | None:
        """Return an active Gizmo.

        Args:
            uid: Target asset UID.
            control_part: Robot control part, if applicable.

        Returns:
            Gizmo instance if found, otherwise ``None``.
        """
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
        gizmo_key = f"{uid}:{control_part}" if control_part else uid
        return gizmo_key in self._gizmos

    def list_gizmos(self) -> dict[str, bool]:
        """List active Gizmo IDs and availability.

        Returns:
            Mapping from ``uid[:control_part]`` to availability.
        """
        return {
            gizmo_key: (gizmo is not None) for gizmo_key, gizmo in self._gizmos.items()
        }

    def get_gizmo_items(self) -> tuple[tuple[str, Gizmo], ...]:
        """Return a stable snapshot of active Gizmo IDs and controllers."""
        return tuple(
            (gizmo_key, gizmo)
            for gizmo_key, gizmo in getattr(self, "_gizmos", {}).items()
            if gizmo is not None
        )

    def process_visualization_commands(self) -> int:
        """Apply queued Viser Gizmo commands on the simulation thread.

        Returns:
            Number of commands accepted for active Gizmos.
        """
        runtime = self._visualization_runtime
        if runtime is None or not getattr(
            self.sim_config.visualization,
            "allow_commands",
            False,
        ):
            return 0
        accepted = 0
        for command in runtime.drain_gizmo_commands():
            if (
                command.run_id != runtime.exporter.run_id
                or command.scene_revision != runtime.exporter.scene_revision
            ):
                continue
            gizmo = self._gizmos.get(command.gizmo_id)
            if gizmo is None:
                continue
            source_id = f"viser:{command.client_id}"
            if command.phase in {"start", "update"} and not gizmo.begin_interaction(
                source_id
            ):
                continue
            position = torch.as_tensor(
                command.position,
                dtype=torch.float32,
                device=self.device,
            )
            position = position - self.arena_offsets[0]
            xyzw = convert_quat(
                torch.as_tensor(
                    command.wxyz,
                    dtype=torch.float32,
                    device=self.device,
                ),
                to="xyzw",
            ).unsqueeze(0)
            pose = torch.eye(
                4,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            pose[0, :3, :3] = matrix_from_quat(xyzw)[0]
            pose[0, :3, 3] = position
            if not gizmo.request_local_pose(pose, source_id=source_id):
                continue
            accepted += 1
            if command.phase == "end":
                gizmo.end_interaction(source_id)
        return accepted

    def update_gizmos(self) -> None:
        """Apply Viser commands and update all active Gizmos."""
        self.process_visualization_commands()
        for gizmo_key, gizmo in list(
            getattr(self, "_gizmos", {}).items()
        ):  # Use list() to avoid modification during iteration
            if gizmo is not None:
                try:
                    gizmo.update()
                except Exception as error:
                    logger.log_error(f"Error updating gizmo '{gizmo_key}': {error}")

    def toggle_gizmo_visibility(
        self, uid: str, control_part: str | None = None
    ) -> bool | None:
        """Toggle Gizmo visibility and return the new state, if it exists."""
        gizmo = self.get_gizmo(uid, control_part)
        if gizmo is not None:
            return gizmo.toggle_visibility()
        return None

    def set_gizmo_visibility(
        self, uid: str, visible: bool, control_part: str | None = None
    ) -> None:
        """Set Gizmo visibility by target UID and optional control part."""
        gizmo = self.get_gizmo(uid, control_part)
        if gizmo is not None:
            gizmo.set_visible(visible)

    def add_sensor(self, sensor_cfg: SensorCfg) -> BaseSensor:
        """Create a sensor on the pre-created simulation Arenas.

        Cameras keep EmbodiChain's native CameraGroup implementation. A camera
        attached to an articulation link is created immediately and attached
        after the physical Spawn scene is prepared. Contact sensors are created
        after preparation and query contacts through the backend-neutral Spawn
        Scene API.

        Args:
            sensor_cfg (SensorCfg): configuration for the sensor.

        Returns:
            BaseSensor: The added sensor instance handle.
        """
        sensor_type = sensor_cfg.sensor_type
        uid = sensor_cfg.uid
        if uid is None:
            uid = f"{sensor_type.lower()}_{len(self._sensors)}"
            sensor_cfg.uid = uid
        if uid in self._sensors:
            raise ValueError(f"Sensor {uid!r} already exists.")

        sensor_factory = self.SUPPORTED_SENSOR_TYPES.get(sensor_type)
        if sensor_factory is None:
            raise ValueError(
                f"Unsupported sensor type {sensor_type!r}. Supported types: "
                f"{sorted(self.SUPPORTED_SENSOR_TYPES)}."
            )
        if isinstance(sensor_factory, type) and issubclass(sensor_factory, Camera):
            if len(self._arenas) != self.num_envs:
                raise RuntimeError(
                    "Camera creation requires all Spawn Arenas to be "
                    f"prepared ({len(self._arenas)} of {self.num_envs} ready)."
                )
            sensor = sensor_factory(
                sensor_cfg,
                self.device,
                owner=self,
            )
            if sensor_cfg.extrinsics.parent is not None:
                scene = self._spawn_scene
                if scene.builder.result is not None:
                    self._attach_camera_parent(sensor)
                else:
                    self._pending_sensor_attachments.append(sensor)
        elif isinstance(sensor_factory, type) and issubclass(
            sensor_factory, ContactSensor
        ):
            self.prepare()
            sensor = sensor_factory(
                sensor_cfg,
                self.device,
                owner=self,
            )
        else:
            # Custom native sensors require a prepared physics scene; cameras
            # only depend on the pre-created Arenas.
            self.prepare()
            # Preserve custom test/plugin factories whose two-argument
            # constructor predates the manager-owned render context.
            sensor = sensor_factory(sensor_cfg, self.device)

        self._sensors[uid] = sensor
        self.notify_visualization_topology_changed()
        return sensor

    def _attach_camera_parent(self, sensor: Camera) -> None:
        """Resolve and attach one camera to its configured parent nodes."""
        parent = sensor.cfg.extrinsics.parent
        if parent is None:
            return
        parent_nodes = self._resolve_spawn_sensor_parent_nodes(parent)
        sensor.attach_to_parent_nodes(parent_nodes)

    def _resolve_spawn_sensor_parent_nodes(self, parent: str) -> list[object]:
        """Resolve one canonical articulation link to a render node per Arena.

        A plain link name remains compatible with existing CameraCfg values.
        When more than one robot/articulation owns that link, callers can use
        ``"<asset_uid>/<link_name>"`` to disambiguate without introducing
        backend clone suffixes.
        """
        assets: dict[str, Articulation] = {
            **self._articulations,
            **self._robots,
        }
        asset_uid: str | None = None
        link_name = parent
        if "/" in parent:
            candidate_uid, candidate_link = parent.split("/", maxsplit=1)
            if candidate_uid in assets:
                asset_uid = candidate_uid
                link_name = candidate_link

        matches: list[tuple[str, list[object]]] = []
        for uid, asset in assets.items():
            if asset_uid is not None and uid != asset_uid:
                continue
            handles = list(getattr(asset, "_entities", ()))
            if len(handles) != self.num_envs:
                continue
            if link_name not in handles[0].get_link_names():
                continue

            nodes: list[object] = []
            for handle in handles:
                if link_name not in handle.get_link_names():
                    raise RuntimeError(
                        f"Articulation {uid!r} has heterogeneous link topology; "
                        f"link {link_name!r} is missing in one Arena."
                    )
                render_body = handle.get_render_body(link_name)
                if render_body is None:
                    raise RuntimeError(
                        f"Articulation {uid!r} link {link_name!r} has no public "
                        "render node for camera attachment."
                    )
                nodes.append(render_body.render_node())
            matches.append((uid, nodes))

        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            owners = ", ".join(uid for uid, _ in matches)
            raise ValueError(
                f"Camera parent link {link_name!r} is ambiguous across assets "
                f"[{owners}]; use '<asset_uid>/{link_name}'."
            )
        scope = f" on asset {asset_uid!r}" if asset_uid is not None else ""
        raise ValueError(
            f"Camera parent link {link_name!r} was not found{scope} in any "
            "Spawn-bound Robot or Articulation. Attachment to arbitrary render "
            "nodes is not yet supported by the Spawn-only bridge."
        )

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

        Native render lights are not removed by this method. Sensors and
        Spawn-owned physical assets are supported.

        Args:
            uid (str): The UID of the asset.
        Returns:
            bool: True if the asset is removed successfully, otherwise False.
        """
        if uid in self._sensors:
            sensor = self._sensors.pop(uid)
            if sensor in self._pending_sensor_attachments:
                self._pending_sensor_attachments.remove(sensor)
            destroy = getattr(sensor, "destroy", None)
            if callable(destroy):
                destroy()
            self.notify_visualization_topology_changed()
            return True

        scene = self._spawn_scene
        if uid not in scene:
            return False
        if uid == "default_plane":
            raise ValueError("The Spawn-owned default plane cannot be removed.")

        was_materialized = scene.builder.is_finalized
        scene.remove(uid)
        if was_materialized:
            self.prepare()

        self._rigid_objects.pop(uid, None)
        self._rigid_object_groups.pop(uid, None)
        self._deformable_objects.pop(uid, None)
        self._articulations.pop(uid, None)
        self._robots.pop(uid, None)
        self.notify_visualization_topology_changed()
        return True

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

        self._markers[name] = _AxisMarkerGroup(
            handles=tuple(marker_handles),
            arena_index=cfg.arena_index,
            axis_length=cfg.axis_len,
            axis_radius=cfg.axis_size,
        )

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
            marker_group = self._markers[name]
            env = self.get_env(marker_group.arena_index)
            for marker_handle in marker_group.handles:
                if marker_handle is not None:
                    env.remove_actor(marker_handle.get_name())
            self._markers.pop(name)
            return True
        except Exception as e:
            logger.log_warning(f"Failed to remove marker {name}: {str(e)}")
            return False

    def get_axis_marker_items(
        self,
    ) -> tuple[tuple[str, tuple[MeshObject, ...], float, float], ...]:
        """Return active axes for backend-neutral visualization.

        Returns:
            Tuples containing the marker name, native handles, axis length, and
            axis radius for each active marker group.
        """
        return tuple(
            (
                name,
                group.handles,
                group.axis_length,
                group.axis_radius,
            )
            for name, group in self._markers.items()
        )

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
        ``[right, up, -forward]``. The local camera up axis changes while the
        viewer orbits, but ``Windows.set_look_at`` uses a world-up reference.
        Always use DexSim's default Z-up vector so a captured snippet retains
        the standard viewer controls.

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
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
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
        for uid, deformable_obj in self._deformable_objects.items():
            if uid not in excluded_uids:
                deformable_obj.reset(env_ids)
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

        try:
            self.stop_visualization()
        except Exception as error:
            logger.log_warning(f"Failed to stop Viser visualization cleanly: {error!r}")

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

        # Stop the render loop before releasing scene resources. Vulkan window
        # presentation may otherwise continue acquiring swapchain images while
        # Env::Clean tears down render objects used by the in-flight frame.
        if getattr(self, "is_window_opened", False):
            self.close_window()

        import sys, gc

        # Release backend-owned views before SpawnResult closes the native
        # resources that back them. Newton also synchronizes its device here.
        self.physics.prepare_for_teardown()
        # Run wrapper destructors while their World is still alive. The later
        # collections continue to break cycles left by the native teardown.
        gc.collect()

        # Render-only cameras may be attached to Spawn articulation link
        # nodes. Remove their Arena views before closing SpawnResult, which
        # releases those parent nodes, and before World.quit releases their
        # CameraGroups.
        for sensor in list(getattr(self, "_sensors", {}).values()):
            try:
                sensor.destroy()
            except Exception as error:
                logger.log_warning(
                    f"Failed to destroy sensor {getattr(sensor, 'uid', None)!r}: "
                    f"{error!r}"
                )

        if self._spawn_scene is not None:
            # Release result-scoped batches/facades before closing the
            # SpawnResult and, finally, the World that owns native resources.
            for registry_name in (
                "_rigid_objects",
                "_rigid_object_groups",
                "_deformable_objects",
                "_articulations",
                "_robots",
            ):
                for asset in getattr(self, registry_name, {}).values():
                    if hasattr(asset, "_data"):
                        asset._data = None
                    if hasattr(asset, "_spawn_result"):
                        asset._spawn_result = None
                    if hasattr(asset, "_entities"):
                        asset._entities = []
            try:
                self._spawn_scene.close()
            finally:
                self._spawn_scene = None

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
        _sever_wrapper_refs("_deformable_objects")
        _sever_wrapper_refs("_articulations")
        _sever_wrapper_refs("_robots")
        _sever_wrapper_refs("_sensors")
        _sever_wrapper_refs("_lights")

        # Explicitly clear Python references to trigger C++ object destructors
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
    def flush_cleanup_queue() -> None:
        """Run pending destruction tasks and wait for their scenes to disappear.

        An empty queue means that no manager requested destruction.  In that
        case, returning immediately is important: other managers may still own
        live worlds, and waiting for the global world count to reach zero would
        block until the timeout even though there is nothing to clean up.
        """
        import gc

        drained_task = False
        while True:
            try:
                task = SimulationManager._cleanup_queue.get_nowait()
            except queue.Empty:
                break

            drained_task = True
            try:
                task()
            except Exception as e:
                from embodichain.utils import logger

                logger.log_error(f"Error during delayed destruction: {e}")

        if not drained_task:
            return

        # After the queue is emptied, perform a top-level full GC to thoroughly reclaim dead objects that haven't released their RefPtrs yet
        gc.collect()

        # At this point, wait for the C++ Scene to return to zero, since the stack is at the top level, there will definitely be no deadlock
        SimulationManager.wait_scene_destruction()


def get_physics_scene(instance_id: int = 0):
    """Return the active physics scene from a SimulationManager instance.

    This is the unified EmbodiChain access point for code that previously
    reached through ``dexsim.default_world().get_physics_scene()``.
    """
    return SimulationManager.get_instance(instance_id).get_physics_scene()
