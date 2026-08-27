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

from collections.abc import Mapping
import enum
import json
import os
import warnings

import dexsim
import numpy as np
import torch

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from dataclasses import field, fields, MISSING

from dexsim.types import (
    DenoiserType,
    Renderer,
    ToneMappingType,
    PhysicalAttr,
    ActorType,
    AxisArrowType,
    AxisCornerType,
    VoxelConfig,
    SoftBodyAttr,
    SoftBodyMaterialModel,
    ClothBodyAttr,
)
from embodichain.utils import configclass, is_configclass
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.utils.utility import key_in_nested_dict

from ._legacy_cfg import RigidBodyAttributesCfg, RigidBodyAttributesOverrideCfg
from .shapes import ShapeCfg, MeshCfg
from .workspace.cfg import RobotWorkspaceCfg

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import NewtonCfg
    from dexsim.engine.newton_physics.solvers_cfg import NewtonSolverCfg

# Global default renderer settings for simulation.
#
# The sentinel value ``"auto"`` defers the choice to GPU-based auto-selection
# performed lazily when a :class:`SimulationManager` is constructed (see
# :func:`embodichain.lab.sim.utility.render_utils.select_default_renderer`). Assigning a
# concrete renderer here (e.g. in test fixtures) forces that renderer and takes
# precedence over auto-selection.
DEFAULT_RENDERER: Literal["auto", "hybrid", "fast-rt", "rt"] = "auto"

AssetPhysicsMode = Literal["preserve", "overlay"]
"""Policy for applying EmbodiChain physics to a file-backed asset."""


def _resolve_asset_physics_mode(
    mode: AssetPhysicsMode | None,
    legacy_use_usd_properties: bool | None,
    *,
    default: AssetPhysicsMode,
) -> AssetPhysicsMode:
    """Resolve the source-agnostic policy and its deprecated USD alias."""
    if mode is not None and mode not in ("preserve", "overlay"):
        raise ValueError(
            f"asset_physics_mode must be 'preserve' or 'overlay', got {mode!r}."
        )
    if legacy_use_usd_properties is not None:
        legacy_mode: AssetPhysicsMode = (
            "preserve" if legacy_use_usd_properties else "overlay"
        )
        if mode is not None and mode != legacy_mode:
            raise ValueError(
                "asset_physics_mode conflicts with deprecated use_usd_properties."
            )
        warnings.warn(
            "use_usd_properties is deprecated; set "
            "asset_physics_mode='preserve' or 'overlay' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy_mode
    return default if mode is None else mode


@configclass
class RenderCfg:
    renderer: Literal["auto", "hybrid", "fast-rt", "rt"] = "auto"
    """Renderer backend to use for the simulation. Options are 'auto', 'hybrid', 'fast-rt', and 'rt'.

    Note:
    - 'auto' selects a default renderer based on the detected GPU: RTX-series cards use
        'hybrid', while datacenter cards (A100/A800, H100/H800/H200/H20) use 'fast-rt'.
        If no CUDA device is available or the GPU is unknown, it falls back to 'hybrid'.
    - 'hybrid' uses ray tracing for shadows and reflections while keeping rasterization for primary rendering,
        providing a balance between performance and visual quality.
    - 'fast-rt' is a fully ray-traced renderer for maximum visual fidelity, but may have higher computational cost.
    - 'rt' is an offline ray-traced renderer for maximum visual fidelity, suitable for high-quality rendering tasks.
    """

    spp: int = 1
    """Samples per pixel for ray tracing rendering. This parameter is only valid when renderer is 'hybrid' or 'fast-rt' and enable_denoiser is False."""

    tone_mapping_enabled: bool = False
    """Whether to map HDR RGB output with the modified Reinhard curve."""

    tone_mapping_exposure: float = 1.0
    """Fixed linear exposure multiplier applied before tone mapping."""

    def __post_init__(self) -> None:
        """Validate rendering parameters."""
        if self.spp < 1:
            logger.log_error("RenderCfg.spp must be at least 1.", ValueError)
        if self.tone_mapping_exposure < 0.0:
            logger.log_error(
                "RenderCfg.tone_mapping_exposure must be non-negative.", ValueError
            )

    def to_dexsim_flags(self) -> Renderer:
        """Convert the renderer name to DexSim's renderer enum."""
        if self.renderer == "hybrid":
            return Renderer.HYBRID
        elif self.renderer == "fast-rt":
            return Renderer.FASTRT
        elif self.renderer == "rt":
            return Renderer.OFFLINERT
        elif self.renderer == "auto":
            # 'auto' is normally resolved by the SimulationManager before this is
            # called. If it reaches here (e.g. used standalone), fall back safely.
            logger.log_warning(
                "Renderer 'auto' was not resolved before converting to dexsim flags. "
                "Falling back to 'hybrid'."
            )
            return Renderer.HYBRID
        else:
            logger.log_error(
                f"Invalid renderer type '{self.renderer}' specified. Must be one of 'auto', 'hybrid', 'fast-rt', or 'rt'."
            )

    def apply_to_dexsim_config(self, world_config: dexsim.WorldConfig) -> None:
        """Apply rendering settings to a DexSim world configuration.

        Args:
            world_config: DexSim world configuration to update in place.
        """
        world_config.renderer = self.to_dexsim_flags()
        world_config.raytrace_config.render_iterations_per_frame = self.spp
        world_config.raytrace_config.open_denoise = True
        world_config.postprocess_config.tone_mapping_enabled = self.tone_mapping_enabled
        world_config.postprocess_config.tone_mapping_type = (
            ToneMappingType.MODIFIED_REINHARD
        )
        world_config.postprocess_config.tone_mapping_exposure = (
            self.tone_mapping_exposure
        )


@configclass
class GPUMemoryCfg:
    """GPU memory configuration for default-backend GPU physics simulation."""

    temp_buffer_capacity: int = 2**24
    """Increase this if you get 'PxgPinnedHostLinearMemoryAllocator: overflowing initial allocation size, increase capacity to at least %.' """

    max_rigid_contact_count: int = 2**19
    """Increase this if you get 'Contact buffer overflow detected'"""

    max_rigid_patch_count: int = (
        2**18
    )  # 81920 is DexSim default but most tasks work with 2**18
    """Increase this if you get 'Patch buffer overflow detected'"""

    heap_capacity: int = 2**26

    found_lost_pairs_capacity: int = (
        2**25
    )  # 262144 is DexSim default but most tasks work with 2**25
    found_lost_aggregate_pairs_capacity: int = 2**10
    total_aggregate_pairs_capacity: int = 2**10


def _gravity_vector(
    gravity: Sequence[float] | np.ndarray,
) -> list[float]:
    """Validate and normalize a backend-neutral gravity vector."""
    values = np.asarray(gravity, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Gravity must contain three finite values.")
    return values.tolist()


@configclass
class PhysicsBackendCfg:
    """Backend-neutral simulation timing, device, and gravity configuration.

    Concrete backend configs inherit this class.  The config type selects the
    backend; no independent backend string can disagree with it.
    """

    physics_dt: float = 1.0 / 100.0
    """Control-level simulation time step in seconds."""

    device: str | torch.device = "cpu"
    """Device used by the selected physics backend."""

    gravity: Sequence[float] | np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )
    """World gravity vector in meters per second squared."""


@configclass
class PhysicsCfg(PhysicsBackendCfg):
    """Configuration for the DexSim default physics backend.

    ``DefaultPhysicsCfg`` is the explicit backend-selecting subclass used by
    new code. This base name remains concrete for compatibility with existing
    configurations that instantiate ``PhysicsCfg`` directly.
    """

    bounce_threshold: float = 2.0
    """The speed threshold below which collisions will not produce bounce effects."""

    enable_ccd: bool = False
    """Enable continuous collision detection (CCD) for fast-moving objects."""

    length_tolerance: float = 0.05
    """The length tolerance for the simulation.

    Note: the larger the tolerance, the faster the simulation will be.
    """
    speed_tolerance: float = 0.25
    """The speed tolerance for the simulation.

    Note: the larger the tolerance, the faster the simulation will be.
    """

    gpu_memory: GPUMemoryCfg = field(default_factory=GPUMemoryCfg)
    """GPU memory configuration for GPU physics simulation."""

    def to_dexsim_args(self) -> Dict[str, Any]:
        """Convert to DexSim physics arguments.

        Solver implementation details that are not exposed by :class:`PhysicsCfg`
        retain their established defaults here.
        """
        args = {
            "gravity": _gravity_vector(self.gravity),
            "bounce_threshold": self.bounce_threshold,
            "enable_ccd": self.enable_ccd,
            "enable_enhanced_determinism": False,
            "enable_friction_every_iteration": True,
        }
        return args


@configclass
class DefaultPhysicsCfg(PhysicsCfg):
    """Explicit configuration selector for the default physics backend."""


@configclass
class NewtonCollisionPipelineCfg:
    """Newton collision-pipeline settings owned at scene scope.

    These values map to DexSim's ``NewtonCollisionPipelineCfg``.  Per-shape
    contact and SDF values belong to :class:`NewtonCollisionPropertiesCfg`
    instead.
    """

    reduce_contacts: bool = True
    """Whether to reduce mesh-mesh contacts."""

    rigid_contact_max: int | None = None
    """Optional rigid-contact capacity; ``None`` lets Newton estimate it."""

    max_triangle_pairs: int = 4_000_000
    """Maximum triangle pairs allocated by the narrow phase."""

    soft_contact_max: int | None = None
    """Optional soft-contact capacity."""

    soft_contact_margin: float = 0.01
    """Soft-contact generation margin in meters."""

    broad_phase: Literal["nxn", "sap", "explicit"] | Any | None = None
    """Built-in broad-phase mode or an expert backend object."""

    shape_pairs_filtered: Any | None = None
    """Optional precomputed shape pairs for explicit broad phase."""

    narrow_phase: Any | None = None
    """Optional expert narrow-phase object."""

    sdf_hydroelastic_config: Any | None = None
    """Optional Newton hydroelastic SDF configuration."""


@configclass
class NewtonPhysicsCfg(PhysicsBackendCfg):
    """Configuration for DexSim Newton physics backend."""

    device: str | torch.device = "cuda:0"
    """The device for Newton physics simulation (e.g. ``cuda:0``)."""

    num_substeps: int = 10
    """Number of Newton solver substeps per EmbodiChain physics step."""

    requires_grad: bool = False
    """Whether to finalize the Newton model for differentiable simulation."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graph capture for Newton stepping when supported."""

    debug_mode: bool = False
    """Whether to enable Newton debug mode."""

    suppress_warp_kernel_logs: bool = True
    """Whether to hide Warp startup and kernel compile/load messages during Newton updates."""

    solver_cfg: Mapping[str, Any] | NewtonSolverCfg | None = None
    """Optional Newton solver configuration.

    A mapping is converted to the matching DexSim Newton solver config. Include
    ``solver_type`` or ``class_type`` to select the solver, then add any
    parameters accepted by that DexSim solver config. If omitted, the Newton
    backend uses DexSim's MuJoCo Warp solver config by default.
    """

    collision_cfg: NewtonCollisionPipelineCfg | Mapping[str, Any] = field(
        default_factory=NewtonCollisionPipelineCfg
    )
    """Scene-level Newton collision-pipeline configuration."""

    enable_collision_pipeline: bool = True
    """Whether Newton runs its rigid-contact collision pipeline."""

    broad_phase: Literal["nxn", "sap", "explicit"] | None = None
    """Deprecated shortcut for ``collision_cfg.broad_phase``.

    If both are set, ``collision_cfg.broad_phase`` wins.
    """

    visualizer_enabled: bool = False
    """Whether to enable the Newton visualizer."""

    def __post_init__(self) -> None:
        """Normalize dictionary collision settings at the config boundary."""
        if isinstance(self.collision_cfg, Mapping):
            self.collision_cfg = NewtonCollisionPipelineCfg(**self.collision_cfg)

    def to_dexsim_cfg(
        self,
        gpu_id: int,
    ) -> NewtonCfg:
        """Convert this config to ``dexsim.engine.newton_physics.NewtonCfg``."""
        from dexsim.engine.newton_physics import (
            FeatherstoneSolverCfg,
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            SemiImplicitSolverCfg,
            VBDSolverCfg,
            XPBDSolverCfg,
        )

        torch_device = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        device = (
            f"cuda:{gpu_id}"
            if torch_device.type == "cuda" and torch_device.index is None
            else str(torch_device)
        )

        solver_cfg_map = {
            "mujoco_warp": MJWarpSolverCfg,
            "xpbd": XPBDSolverCfg,
            "semi_implicit": SemiImplicitSolverCfg,
            "featherstone": FeatherstoneSolverCfg,
            "vbd": VBDSolverCfg,
        }
        solver_cfg = _newton_solver_cfg_to_dexsim(
            solver_cfg=self.solver_cfg,
            solver_cfg_map=solver_cfg_map,
        )

        if self.requires_grad and solver_cfg.solver_type != "semi_implicit":
            logger.log_error(
                "Newton gradient mode requires solver_type='semi_implicit'."
            )

        collision_values = {
            item.name: getattr(self.collision_cfg, item.name)
            for item in fields(self.collision_cfg)
        }
        if collision_values["broad_phase"] is None:
            collision_values["broad_phase"] = self.broad_phase
        collision_values["requires_grad"] = self.requires_grad

        cfg = NewtonCfg(
            dt=self.physics_dt,
            num_substeps=self.num_substeps,
            device=device,
            gravity=_gravity_vector(self.gravity),
            debug_mode=self.debug_mode,
            requires_grad=self.requires_grad,
            suppress_warp_kernel_logs=self.suppress_warp_kernel_logs,
            solver_cfg=solver_cfg,
            collision_pipeline_cfg=NewtonCollisionPipelineCfg(**collision_values),
            enable_collision_pipeline=self.enable_collision_pipeline,
            sync_to_dexsim=True,
        )
        cfg.use_cuda_graph = self.use_cuda_graph and not self.requires_grad
        cfg._visualizer_enabled = self.visualizer_enabled
        return cfg


def _normalize_newton_solver_type(solver_type: str) -> str:
    """Normalize public EmbodiChain and DexSim Newton solver aliases."""
    key = solver_type.replace("-", "_").lower()
    aliases = {
        "mjwarp": "mujoco_warp",
        "mjwarpsolver": "mujoco_warp",
        "mjwarpsolvercfg": "mujoco_warp",
        "mjwarp_solver": "mujoco_warp",
        "mjwarp_solver_cfg": "mujoco_warp",
        "mujoco_warp": "mujoco_warp",
        "mujocowarp": "mujoco_warp",
        "mujocowarpsolver": "mujoco_warp",
        "mujocowarpsolvercfg": "mujoco_warp",
        "xpbdsolver": "xpbd",
        "xpbdsolvercfg": "xpbd",
        "xpbd": "xpbd",
        "semiimplicit": "semi_implicit",
        "semi_implicit": "semi_implicit",
        "semiimplicitsolver": "semi_implicit",
        "semiimplicitsolvercfg": "semi_implicit",
        "featherstone": "featherstone",
        "featherstonesolver": "featherstone",
        "featherstonesolvercfg": "featherstone",
        "vbd": "vbd",
        "vbdsolver": "vbd",
        "vbdsolvercfg": "vbd",
    }
    if key not in aliases:
        logger.log_error(
            f"Unsupported Newton solver type '{solver_type}'. "
            "Expected one of 'mjwarp', 'xpbd', 'semi_implicit', "
            "'featherstone', or 'vbd'."
        )
    return aliases[key]


def _newton_solver_cfg_to_dexsim(
    solver_cfg: Mapping[str, Any] | object | None,
    solver_cfg_map: Mapping[str, type],
) -> object:
    """Convert EmbodiChain Newton solver config input to a DexSim config."""
    if solver_cfg is None:
        return solver_cfg_map["mujoco_warp"]()

    if not isinstance(solver_cfg, Mapping):
        if not hasattr(solver_cfg, "solver_type"):
            logger.log_error(
                "Newton solver_cfg must be a mapping or a DexSim Newton solver "
                "config object with a 'solver_type' attribute."
            )
        return solver_cfg

    solver_cfg_data = dict(solver_cfg)
    configured_solver_type = (
        solver_cfg_data.pop("solver_type", None)
        or solver_cfg_data.pop("class_type", None)
        or "mujoco_warp"
    )
    normalized_solver_type = _normalize_newton_solver_type(str(configured_solver_type))
    return solver_cfg_map[normalized_solver_type](**solver_cfg_data)


@configclass
class MarkerCfg:
    """Configuration for visual markers in the simulation.

    This class defines properties for creating visual markers such as coordinate frames,
    lines, and points that can be used for debugging, visualization, or reference purposes
    in the simulation environment.
    """

    name: str = "empty-mesh"
    """Name of the marker for identification purposes."""

    marker_type: Literal["axis", "line", "point"] = "axis"
    """Type of marker to display. Can be 'axis' (3D coordinate frame), 'line', or 'point'. (only axis supported now)"""

    axis_xpos: torch.Tensor | None = None
    """List of 4x4 transformation matrices defining the position and orientation of each axis marker."""

    axis_size: float = 0.002
    """Thickness/size of the axis lines in meters."""

    axis_len: float = 0.005
    """Length of each axis arm in meters."""

    line_color: List[float] = [1, 1, 0, 1.0]
    """RGBA color values for the marker lines. Values should be between 0.0 and 1.0."""

    arrow_type: AxisArrowType = AxisArrowType.CONE
    """Type of arrow head for axis markers (e.g., CONE, ARROW, etc.)."""

    corner_type: AxisCornerType = AxisCornerType.SPHERE
    """Type of corner/joint visualization for axis markers (e.g., SPHERE, CUBE, etc.)."""

    arena_index: int = -1
    """Index of the arena where the marker should be placed. -1 means all arenas."""


@configclass
class WindowRecordCfg:
    """Configuration for interactive viewer window recording."""

    enable_hotkey: bool = True
    """Whether to register the ``r`` hotkey for viewer recording when the window opens."""

    save_path: str | None = None
    """Optional output path for viewer recordings. If None, use the default outputs directory."""

    fps: int = 20
    """Frames per second for viewer recording."""

    max_memory: int = 1024
    """Maximum buffered recording memory in MB before auto-stopping capture."""

    video_prefix: str = "viewer_record"
    """Video file prefix used when no explicit save path is provided."""


def physics_cfg_for_backend(
    backend: Literal["default", "newton"],
) -> PhysicsBackendCfg:
    """Return a default physics configuration instance for the given backend."""
    if backend == "newton":
        return NewtonPhysicsCfg()
    return DefaultPhysicsCfg()


def physics_backend_from_cfg(
    physics_cfg: PhysicsBackendCfg,
) -> Literal["default", "newton"]:
    """Infer the physics backend name from a physics configuration instance."""
    if isinstance(physics_cfg, NewtonPhysicsCfg):
        return "newton"
    if isinstance(physics_cfg, PhysicsCfg):
        return "default"
    logger.log_error(
        f"Unsupported physics_cfg type '{type(physics_cfg).__name__}'. "
        "Expected PhysicsCfg, DefaultPhysicsCfg, or NewtonPhysicsCfg."
    )


def validate_physics_cfg(physics_cfg: PhysicsBackendCfg) -> None:
    """Validate that ``physics_cfg`` is a supported backend configuration."""
    physics_backend_from_cfg(physics_cfg)


@configclass
class WindowCameraPoseCfg:
    """Configuration for printing the interactive viewer camera pose."""

    enable_hotkey: bool = True
    """Whether to register the ``p`` hotkey when the window opens."""

    convert_to_look_at: bool = True
    """Whether the hotkey prints a ``set_look_at`` call instead of a matrix."""


@configclass
class MassPropertiesCfg:
    """Backend-neutral rigid-body mass properties.

    ``None`` means that the source asset or selected backend keeps ownership of
    that value. Explicit inertia is used together with a positive mass;
    otherwise mass has priority over density during Spawn compilation.
    """

    mass: float | None = None
    """Body mass in kilograms."""

    density: float | None = None
    """Uniform collision-shape density in kilograms per cubic meter."""

    inertia: Sequence[float] | np.ndarray | None = None
    """Three principal moments or a full 3-by-3 body-frame inertia tensor."""

    com_position: Sequence[float] | np.ndarray | None = None
    """Center-of-mass position in the body frame."""

    com_quaternion: Sequence[float] | np.ndarray | None = None
    """Center-of-mass orientation quaternion in ``wxyz`` order."""


@configclass
class RigidBodyPropertiesCfg:
    """Single-root base for backend-specific rigid-body properties.

    Actor type and mass properties are already backend-neutral, so the common
    root intentionally has no fields today.
    """


@configclass
class DexsimRigidBodyPropertiesCfg(RigidBodyPropertiesCfg):
    """DexSim/default-backend rigid-body properties."""

    linear_damping: float | None = None
    angular_damping: float | None = None
    has_gravity: bool | None = None
    max_linear_velocity: float | None = None
    max_angular_velocity: float | None = None
    max_depenetration_velocity: float | None = None
    retain_acceleration: bool | None = None
    enable_ccd: bool | None = None
    min_position_iters: int | None = None
    min_velocity_iters: int | None = None
    sleep_threshold: float | None = None


@configclass
class NewtonRigidBodyPropertiesCfg(RigidBodyPropertiesCfg):
    """Newton rigid-body extension point.

    Newton currently consumes common mass properties and per-shape settings,
    but exposes no additional body-level fields through DexSim Spawn.
    """


@configclass
class CollisionPropertiesCfg:
    """Backend-neutral collision properties."""

    collision_enabled: bool | None = None
    """Whether collision is enabled; ``None`` preserves the source/default."""


@configclass
class DexsimCollisionPropertiesCfg(CollisionPropertiesCfg):
    """DexSim/default-backend collision geometry properties."""

    contact_offset: float | None = None
    """Distance at which contact generation starts."""

    rest_offset: float | None = None
    """Separation distance maintained at rest."""


@configclass
class NewtonCollisionPropertiesCfg(CollisionPropertiesCfg):
    """Newton-native collision geometry, filtering, and SDF properties."""

    margin: float | None = None
    gap: float | None = None
    is_solid: bool | None = None
    collision_group: int | None = None
    collision_filter_parent: bool | None = None
    has_particle_collision: bool | None = None
    is_visible: bool | None = None
    is_site: bool | None = None
    is_hydroelastic: bool | None = None
    sdf_narrow_band_range: tuple[float, float] | None = None
    sdf_target_voxel_size: float | None = None
    sdf_max_resolution: int | None = None
    sdf_texture_format: str | None = None
    force_sdf: bool | None = None
    sdf_padding: float | None = None


@configclass
class RigidBodyMaterialCfg:
    """Backend-neutral rigid contact material properties."""

    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None


@configclass
class DexsimRigidBodyMaterialCfg(RigidBodyMaterialCfg):
    """DexSim/default-backend material extensions."""

    torsional_patch_radius: float | None = None
    min_torsional_patch_radius: float | None = None
    disable_strong_friction: bool | None = None


@configclass
class NewtonRigidBodyMaterialCfg(RigidBodyMaterialCfg):
    """Newton contact-material extensions.

    Solver support differs by field.  The Spawn compiler warns through
    DexSim when the selected Newton solver cannot consume a configured value.
    """

    ke: float | None = None
    kd: float | None = None
    kf: float | None = None
    ka: float | None = None
    kh: float | None = None
    torsional_friction: float | None = None
    rolling_friction: float | None = None


_RIGID_PHYSICS_LEGACY_FIELD_GROUPS = {
    "mass": "mass_props",
    "density": "mass_props",
    "inertia": "mass_props",
    "com_position": "mass_props",
    "com_quaternion": "mass_props",
    "linear_damping": "rigid_props",
    "angular_damping": "rigid_props",
    "max_linear_velocity": "rigid_props",
    "max_angular_velocity": "rigid_props",
    "max_depenetration_velocity": "rigid_props",
    "enable_ccd": "rigid_props",
    "min_position_iters": "rigid_props",
    "min_velocity_iters": "rigid_props",
    "sleep_threshold": "rigid_props",
    "contact_offset": "collision_props",
    "rest_offset": "collision_props",
    "static_friction": "material_props",
    "dynamic_friction": "material_props",
    "restitution": "material_props",
}

_RIGID_PHYSICS_GROUP_FIELDS = frozenset(
    {"mass_props", "rigid_props", "collision_props", "material_props"}
)


def _physics_property_cfg_from_dict(
    value: Mapping[str, Any] | object | None,
    *,
    common_type: type,
    dexsim_type: type,
    newton_type: type,
    field_name: str,
) -> object | None:
    """Parse one polymorphic rigid-physics property slot."""
    if value is None:
        return None
    if isinstance(value, common_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or {common_type.__name__}.")
    data = dict(value)
    configured_backend = data.pop("backend", None)
    if configured_backend is None:
        common_fields = {item.name for item in fields(common_type)}
        dexsim_fields = {item.name for item in fields(dexsim_type)} - common_fields
        newton_fields = {item.name for item in fields(newton_type)} - common_fields
        has_dexsim_fields = bool(dexsim_fields.intersection(data))
        has_newton_fields = bool(newton_fields.intersection(data))
        if has_dexsim_fields and has_newton_fields:
            raise ValueError(
                f"{field_name} mixes DexSim and Newton-only fields; select one "
                "backend-specific property config."
            )
        backend = (
            "dexsim"
            if has_dexsim_fields
            else "newton" if has_newton_fields else "common"
        )
    else:
        backend = str(configured_backend).replace("-", "_").lower()
    config_type = {
        "common": common_type,
        "default": dexsim_type,
        "dexsim": dexsim_type,
        "physx": dexsim_type,
        "newton": newton_type,
    }.get(backend)
    if config_type is None:
        raise ValueError(
            f"{field_name}.backend must be 'common', 'dexsim', or 'newton', "
            f"got {backend!r}."
        )
    try:
        return config_type(**data)
    except TypeError as exc:
        raise TypeError(f"Invalid {field_name} configuration: {exc}") from exc


def _physics_property_cfg_to_dict(
    value: object | None,
    *,
    common_type: type,
    dexsim_type: type,
    newton_type: type,
    field_name: str,
) -> dict[str, Any] | None:
    """Serialize one polymorphic property slot with a stable discriminator."""
    if value is None:
        return None
    if isinstance(value, newton_type):
        backend = "newton"
    elif isinstance(value, dexsim_type):
        backend = "dexsim"
    elif type(value) is common_type:
        backend = None
    else:
        raise TypeError(
            f"Unsupported {field_name} config type {type(value).__name__!r}."
        )
    data = dict(value.to_dict())
    if backend is not None:
        data["backend"] = backend
    return data


@configclass
class RigidBodyPhysicsCfg:
    """Grouped rigid-body physics configuration used by Spawn.

    Each logical property group has one slot.  A common config is portable;
    a DexSim or Newton subclass adds only fields owned by that backend.  Every
    field defaults to ``None`` so partial configs compose with source assets
    without resetting unrelated properties.
    """

    mass_props: MassPropertiesCfg | None = None
    rigid_props: RigidBodyPropertiesCfg | None = None
    collision_props: CollisionPropertiesCfg | None = None
    material_props: RigidBodyMaterialCfg | None = None

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> RigidBodyPhysicsCfg:
        """Parse grouped physics properties from a YAML/JSON-style mapping."""
        unknown = set(init_dict) - _RIGID_PHYSICS_GROUP_FIELDS
        if unknown:
            raise KeyError(f"Unknown RigidBodyPhysicsCfg fields: {sorted(unknown)}")
        cfg = cls()
        if "mass_props" in init_dict:
            value = init_dict["mass_props"]
            if value is not None:
                if not isinstance(value, (MassPropertiesCfg, Mapping)):
                    raise TypeError(
                        "mass_props must be a mapping or MassPropertiesCfg."
                    )
                cfg.mass_props = (
                    value
                    if isinstance(value, MassPropertiesCfg)
                    else MassPropertiesCfg(**value)
                )
        if "rigid_props" in init_dict:
            cfg.rigid_props = _physics_property_cfg_from_dict(
                init_dict["rigid_props"],
                common_type=RigidBodyPropertiesCfg,
                dexsim_type=DexsimRigidBodyPropertiesCfg,
                newton_type=NewtonRigidBodyPropertiesCfg,
                field_name="rigid_props",
            )
        if "collision_props" in init_dict:
            cfg.collision_props = _physics_property_cfg_from_dict(
                init_dict["collision_props"],
                common_type=CollisionPropertiesCfg,
                dexsim_type=DexsimCollisionPropertiesCfg,
                newton_type=NewtonCollisionPropertiesCfg,
                field_name="collision_props",
            )
        if "material_props" in init_dict:
            cfg.material_props = _physics_property_cfg_from_dict(
                init_dict["material_props"],
                common_type=RigidBodyMaterialCfg,
                dexsim_type=DexsimRigidBodyMaterialCfg,
                newton_type=NewtonRigidBodyMaterialCfg,
                field_name="material_props",
            )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize grouped properties without losing backend subclasses."""
        return {
            "mass_props": (
                None if self.mass_props is None else self.mass_props.to_dict()
            ),
            "rigid_props": _physics_property_cfg_to_dict(
                self.rigid_props,
                common_type=RigidBodyPropertiesCfg,
                dexsim_type=DexsimRigidBodyPropertiesCfg,
                newton_type=NewtonRigidBodyPropertiesCfg,
                field_name="rigid_props",
            ),
            "collision_props": _physics_property_cfg_to_dict(
                self.collision_props,
                common_type=CollisionPropertiesCfg,
                dexsim_type=DexsimCollisionPropertiesCfg,
                newton_type=NewtonCollisionPropertiesCfg,
                field_name="collision_props",
            ),
            "material_props": _physics_property_cfg_to_dict(
                self.material_props,
                common_type=RigidBodyMaterialCfg,
                dexsim_type=DexsimRigidBodyMaterialCfg,
                newton_type=NewtonRigidBodyMaterialCfg,
                field_name="material_props",
            ),
        }

    @property
    def enable_collision(self) -> bool:
        """Compatibility view used by legacy object initialization."""
        value = (
            None
            if self.collision_props is None
            else self.collision_props.collision_enabled
        )
        return True if value is None else bool(value)

    def attr(self) -> PhysicalAttr:
        """Project supported values to the legacy DexSim ``PhysicalAttr``."""
        attr = PhysicalAttr()
        for cfg in (
            self.mass_props,
            (
                self.rigid_props
                if isinstance(self.rigid_props, DexsimRigidBodyPropertiesCfg)
                else None
            ),
            (
                self.collision_props
                if isinstance(self.collision_props, DexsimCollisionPropertiesCfg)
                else None
            ),
            self.material_props,
        ):
            if cfg is None:
                continue
            for item in fields(cfg):
                value = getattr(cfg, item.name)
                if value is not None and hasattr(attr, item.name):
                    setattr(attr, item.name, value)
        return attr

    def __getattr__(self, name: str) -> Any:
        """Provide read-only compatibility for legacy flat property access."""
        group_name = _RIGID_PHYSICS_LEGACY_FIELD_GROUPS.get(name)
        if group_name is None:
            raise AttributeError(name)
        group = object.__getattribute__(self, group_name)
        if group is not None and hasattr(group, name):
            value = getattr(group, name)
            if value is not None:
                return value
        legacy_defaults = PhysicalAttr()
        return getattr(legacy_defaults, name, None)


def _rigid_body_attrs_from_dict(
    value: Mapping[str, Any],
    *,
    override: bool = False,
) -> RigidBodyPhysicsCfg | RigidBodyAttributesCfg | RigidBodyAttributesOverrideCfg:
    """Parse grouped physics or the deprecated Default-only flat schema."""
    grouped_fields = _RIGID_PHYSICS_GROUP_FIELDS.intersection(value)
    if grouped_fields:
        flat_fields = set(value) - _RIGID_PHYSICS_GROUP_FIELDS
        if flat_fields:
            raise ValueError(
                "Do not mix deprecated flat rigid-body fields with grouped "
                f"RigidBodyPhysicsCfg fields: {sorted(flat_fields)}"
            )
        return RigidBodyPhysicsCfg.from_dict(value)
    legacy_type = RigidBodyAttributesOverrideCfg if override else RigidBodyAttributesCfg
    return legacy_type.from_dict(dict(value))


@configclass
class ArticulationRootPropertiesCfg:
    """Backend-neutral articulation-root properties."""

    fixed_base: bool | None = None
    """Whether the root is fixed to the world."""

    self_collision_enabled: bool | None = None
    """Whether links in the articulation may collide with each other."""

    @classmethod
    def from_dict(
        cls,
        init_dict: Mapping[str, Any],
    ) -> ArticulationRootPropertiesCfg:
        """Parse a common, DexSim, or Newton articulation-root config."""
        data = dict(init_dict)
        backend = str(data.pop("backend", "common")).replace("-", "_").lower()
        config_type = {
            "common": cls,
            "default": DexsimArticulationRootPropertiesCfg,
            "dexsim": DexsimArticulationRootPropertiesCfg,
            "physx": DexsimArticulationRootPropertiesCfg,
            "newton": NewtonArticulationRootPropertiesCfg,
        }.get(backend)
        if config_type is None:
            raise ValueError(
                "articulation_props.backend must be 'common', 'dexsim', or "
                f"'newton', got {backend!r}."
            )
        return config_type(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize articulation properties with their backend subtype."""
        data: dict[str, Any] = {
            "fixed_base": self.fixed_base,
            "self_collision_enabled": self.self_collision_enabled,
        }
        if isinstance(self, NewtonArticulationRootPropertiesCfg):
            data["backend"] = "newton"
        elif isinstance(self, DexsimArticulationRootPropertiesCfg):
            data["backend"] = "dexsim"
        return data


@configclass
class DexsimArticulationRootPropertiesCfg(ArticulationRootPropertiesCfg):
    """DexSim articulation-root extension point."""


@configclass
class NewtonArticulationRootPropertiesCfg(ArticulationRootPropertiesCfg):
    """Newton articulation-root extension point."""


@configclass
class LinkPhysicsOverrideCfg:
    """Per-link physics override matched by regex on articulation link names."""

    link_names_expr: list[str] = MISSING
    """Regex patterns matched against link names (full match)."""

    attrs: RigidBodyPhysicsCfg | RigidBodyAttributesOverrideCfg = RigidBodyPhysicsCfg()
    """Partial grouped overrides, or a deprecated Default-only flat override."""

    replace_inertial: bool = False
    """Whether to recompute inertia when mass is overridden (DexSim flag)."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> LinkPhysicsOverrideCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if key == "attrs" and isinstance(value, dict):
                setattr(cfg, key, _rigid_body_attrs_from_dict(value, override=True))
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


def link_attrs_from_dict(
    value: dict[str, Any],
) -> dict[str, LinkPhysicsOverrideCfg]:
    """Parse a ``link_attrs`` mapping from YAML/JSON-style dicts."""
    link_attrs: dict[str, LinkPhysicsOverrideCfg] = {}
    for group_name, group_cfg in value.items():
        if isinstance(group_cfg, LinkPhysicsOverrideCfg):
            link_attrs[group_name] = group_cfg
        elif isinstance(group_cfg, dict):
            link_attrs[group_name] = LinkPhysicsOverrideCfg.from_dict(group_cfg)
        else:
            raise TypeError(
                f"link_attrs['{group_name}'] must be a dict or "
                f"LinkPhysicsOverrideCfg, got {type(group_cfg)}."
            )
    return link_attrs


@configclass
class SoftbodyVoxelAttributesCfg:
    # voxel config
    triangle_remesh_resolution: int = 8
    """Resolution to remesh the softbody mesh before building physics collision mesh."""

    triangle_simplify_target: int = 0
    """Simplify mesh faces to target value. Do nothing if this value is zero."""

    # TODO: this value will be automatically computed with simulation_mesh_resolution and mesh scale.
    maximal_edge_length: float = 0
    # """To shorten edges that are too long, additional points get inserted at their center leading to a subdivision of the input mesh. Do nothing if this value is zero."""

    simulation_mesh_resolution: int = 8
    """Resolution to build simulation voxelize textra mesh. This value must be greater than 0."""

    simulation_mesh_output_obj: bool = False
    """Whether to output the simulation mesh as an obj file for debugging."""

    def attr(self) -> VoxelConfig:
        """Convert to dexsim VoxelConfig"""
        attr = VoxelConfig()
        attr.triangle_remesh_resolution = self.triangle_remesh_resolution
        attr.maximal_edge_length = self.maximal_edge_length
        attr.simulation_mesh_resolution = self.simulation_mesh_resolution
        attr.triangle_simplify_target = self.triangle_simplify_target
        return attr


@configclass
class SoftbodyPhysicalAttributesCfg:
    # material properties
    youngs: float = 1e6
    """Young's modulus (higher = stiffer)."""

    poissons: float = 0.45
    """Poisson's ratio (higher = closer to incompressible)."""

    dynamic_friction: float = 0.0
    """Dynamic friction coefficient."""

    elasticity_damping: float = 0.0
    """Elasticity damping factor."""

    # soft body properties
    material_model: SoftBodyMaterialModel = SoftBodyMaterialModel.CO_ROTATIONAL
    """Material constitutive model."""

    # --- Mode / collision switches ---
    enable_kinematic: bool = False
    """If True, (partially) kinematic behavior is enabled."""

    enable_ccd: bool = False
    """Enable continuous collision detection (CCD)."""

    enable_self_collision: bool = False
    """Enable self-collision handling."""

    has_gravity: bool = True
    """Whether the soft body is affected by gravity."""

    # --- Self-collision & simplification parameters ---
    self_collision_stress_tolerance: float = 0.9
    """Stress tolerance threshold for self-collision constraints."""

    collision_mesh_simplification: bool = True
    """Whether to simplify the collision mesh for self-collision."""

    self_collision_filter_distance: float = 0.1
    """Distance threshold below which vertex pairs may be filtered from self-collision checks."""

    # --- Damping, sleep & settling ---
    vertex_velocity_damping: float = 0.005
    """Per-vertex velocity damping."""

    linear_damping: float = 0.0
    """Global linear damping applied to the soft body."""

    sleep_threshold: float = 0.05
    """Velocity/energy threshold below which the soft body can go to sleep."""

    settling_threshold: float = 0.1
    """Threshold used to decide convergence/settling state."""

    settling_damping: float = 10.0
    """Additional damping applied during settling phase."""

    # --- Mass / density & velocity limits ---
    mass: float = -1.0
    """Total mass of the soft body. If set to a negative value, density will be used to compute mass."""

    density: float = 1000.0
    """Material density in kg/m^3."""

    max_depenetration_velocity: float = 1e6
    """Maximum velocity used to resolve penetrations. Must be larger than zero."""

    max_velocity: float = 100
    """Clamp for linear (or vertex) velocity. If set to zero, the limit is ignored."""

    # --- Solver iteration counts ---
    min_position_iters: int = 4
    """Minimum solver iterations for position correction."""

    min_velocity_iters: int = 1
    """Minimum solver iterations for velocity updates."""

    def attr(self) -> SoftBodyAttr:
        attr = SoftBodyAttr()
        attr.youngs = self.youngs
        attr.poissons = self.poissons
        attr.dynamic_friction = self.dynamic_friction
        attr.elasticity_damping = self.elasticity_damping
        attr.material_model = self.material_model
        attr.enable_kinematic = self.enable_kinematic
        attr.enable_ccd = self.enable_ccd
        attr.enable_self_collision = self.enable_self_collision
        attr.has_gravity = self.has_gravity
        attr.self_collision_stress_tolerance = self.self_collision_stress_tolerance
        attr.collision_mesh_simplification = self.collision_mesh_simplification
        attr.vertex_velocity_damping = self.vertex_velocity_damping
        attr.mass = self.mass
        attr.density = self.density
        attr.max_depenetration_velocity = self.max_depenetration_velocity
        attr.max_velocity = self.max_velocity
        attr.self_collision_filter_distance = self.self_collision_filter_distance
        attr.linear_damping = self.linear_damping
        attr.sleep_threshold = self.sleep_threshold
        attr.settling_threshold = self.settling_threshold
        attr.settling_damping = self.settling_damping
        attr.min_position_iters = self.min_position_iters
        attr.min_velocity_iters = self.min_velocity_iters
        return attr


@configclass
class ClothPhysicalAttributesCfg:
    # material properties
    youngs: float = 1e10
    """Young's modulus (higher = stiffer)."""

    poissons: float = 0.3
    """Poisson's ratio."""

    dynamic_friction: float = 0.5
    """Dynamic friction coefficient."""

    elasticity_damping: float = 0.0
    """Elasticity damping factor."""

    thickness: float = 0.001
    """Cloth thickness (m)."""

    bending_stiffness: float = 0.00001
    """Bending stiffness."""

    bending_damping: float = 0.0
    """Bending damping."""

    # cloth body properties
    enable_kinematic: bool = False
    """If True, (partially) kinematic behavior is enabled."""

    enable_ccd: bool = True
    """Enable continuous collision detection (CCD)."""

    enable_self_collision: bool = False
    """Enable self-collision handling."""

    has_gravity: bool = True
    """Whether the cloth is affected by gravity."""

    self_collision_stress_tolerance: float = 0.9
    """Stress tolerance threshold for self-collision constraints."""

    collision_mesh_simplification: bool = True
    """Whether to simplify the collision mesh for self-collision."""

    vertex_velocity_damping: float = 0.005
    """Per-vertex velocity damping."""

    mass: float = -1.0
    """Total mass of the cloth. If negative, density is used to compute mass."""

    density: float = 1.0
    """Material density in kg/m^3."""

    max_depenetration_velocity: float = 1e6
    """Maximum velocity used to resolve penetrations."""

    max_velocity: float = 100.0
    """Clamp for linear (or vertex) velocity."""

    self_collision_filter_distance: float = 0.1
    """Distance threshold for filtering self-collision vertex pairs."""

    linear_damping: float = 0.05
    """Global linear damping applied to the cloth."""

    sleep_threshold: float = 0.05
    """Velocity/energy threshold below which the cloth can go to sleep."""

    settling_threshold: float = 0.1
    """Threshold used to decide convergence/settling state."""

    settling_damping: float = 10.0
    """Additional damping applied during settling phase."""

    min_position_iters: int = 4
    """Minimum solver iterations for position correction."""

    min_velocity_iters: int = 1
    """Minimum solver iterations for velocity updates."""

    def attr(self) -> ClothBodyAttr:
        """Convert to dexsim ClothBodyAttr."""
        attr = ClothBodyAttr()
        attr.youngs = self.youngs
        attr.poissons = self.poissons
        attr.dynamic_friction = self.dynamic_friction
        attr.elasticity_damping = self.elasticity_damping
        attr.thickness = self.thickness
        attr.bending_stiffness = self.bending_stiffness
        attr.bending_damping = self.bending_damping
        attr.enable_kinematic = self.enable_kinematic
        attr.enable_ccd = self.enable_ccd
        attr.enable_self_collision = self.enable_self_collision
        attr.has_gravity = self.has_gravity
        attr.self_collision_stress_tolerance = self.self_collision_stress_tolerance
        attr.collision_mesh_simplification = self.collision_mesh_simplification
        attr.vertex_velocity_damping = self.vertex_velocity_damping
        attr.mass = self.mass
        attr.density = self.density
        attr.max_depenetration_velocity = self.max_depenetration_velocity
        attr.max_velocity = self.max_velocity
        attr.self_collision_filter_distance = self.self_collision_filter_distance
        attr.linear_damping = self.linear_damping
        attr.sleep_threshold = self.sleep_threshold
        attr.settling_threshold = self.settling_threshold
        attr.settling_damping = self.settling_damping
        attr.min_position_iters = self.min_position_iters
        attr.min_velocity_iters = self.min_velocity_iters
        return attr


@configclass
class JointDrivePropertiesCfg:
    """Properties to define the drive mechanism of a joint."""

    drive_type: Literal["force", "acceleration", "none"] | None = None
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force and the acceleration is computed based on the force applied.
    If the drive type is "acceleration", then the joint is driven by an acceleration and the force is computed based on the acceleration applied.
    If the drive type is "none", then no force will be applied to joint.
    """

    stiffness: Dict[str, float] | float | None = None
    """Stiffness of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s^2 (N/m).
    * For angular joints, the unit is kg-m^2/s^2/rad (N-m/rad).
    """

    damping: Dict[str, float] | float | None = None
    """Damping of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s (N-s/m).
    * For angular joints, the unit is kg-m^2/s/rad (N-m-s/rad).
    """

    max_effort: Dict[str, float] | float | None = None
    """Maximum effort that can be applied to the joint (in kg-m^2/s^2)."""

    max_velocity: Dict[str, float] | float | None = None
    """Maximum velocity that the joint can reach (in rad/s or m/s).

    For linear joints, this is the maximum linear velocity with unit m/s.
    For angular joints, this is the maximum angular velocity with unit rad/s.
    """

    friction: Dict[str, float] | float | None = None
    """Friction coefficient of the joint"""

    armature: Dict[str, float] | float | None = None
    """Joint armature added to joint-space spatial inertia.

    Units depend on the joint model:

    * For prismatic (linear) joints, the unit is mass [kg].
    * For revolute (angular) joints, the unit is mass * scene_length^2 [kg-m^2].
    """

    @classmethod
    def from_dict(
        cls,
        init_dict: Dict[str, Any],
        *,
        defaults: JointDrivePropertiesCfg | None = None,
    ) -> JointDrivePropertiesCfg:
        """Initialize the configuration from a dictionary.

        Args:
            init_dict: Joint-drive properties to override.
            defaults: Optional base properties whose unspecified values are
                preserved. If omitted, the class defaults are used.

        Returns:
            Parsed joint-drive properties.
        """
        data = dict(init_dict)
        backend = str(data.pop("backend", "common")).replace("-", "_").lower()
        wants_newton = backend == "newton" or "target_mode" in data
        if backend not in {"common", "default", "dexsim", "physx", "newton"}:
            raise ValueError(
                "drive_pros.backend must be 'common', 'dexsim', or 'newton', "
                f"got {backend!r}."
            )
        if wants_newton and not isinstance(defaults, NewtonJointDrivePropertiesCfg):
            cfg = NewtonJointDrivePropertiesCfg()
            if defaults is not None:
                for item in fields(JointDrivePropertiesCfg):
                    setattr(cfg, item.name, getattr(defaults, item.name))
        else:
            cfg = defaults.copy() if defaults is not None else cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize joint properties with their backend subtype."""
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        if isinstance(self, NewtonJointDrivePropertiesCfg):
            data["backend"] = "newton"
        return data


@configclass
class NewtonJointDrivePropertiesCfg(JointDrivePropertiesCfg):
    """Newton-targeted joint-drive config.

    Common gain, limit, friction, and armature fields are inherited rather
    than repeated under native aliases.  ``target_mode`` is the only Newton
    extension currently exposed by DexSim Spawn.
    """

    target_mode: (
        Literal["none", "position", "velocity", "position_velocity"]
        | Dict[
            str,
            Literal["none", "position", "velocity", "position_velocity"] | int,
        ]
        | int
        | None
    ) = None
    """Newton actuator target mode, as a scalar or regex mapping."""


@configclass
class ObjectBaseCfg:
    """Base configuration for an asset in the simulation.

    This class defines the basic properties of an asset, such as its type, initial state, and collision group.
    It is used as a base class for specific asset configurations.
    """

    uid: str | None = None

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Euler angles (in degree) of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_local_pose: np.ndarray | None = None
    """4x4 transformation matrix of the root in local frame. If specified, it will override init_pos and init_rot."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, str | float | tuple]) -> ObjectBaseCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "attrs" and isinstance(value, Mapping):
                    setattr(cfg, key, _rigid_body_attrs_from_dict(value))
                elif is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )

        # Automatically infer init_local_pose if not provided
        if cfg.init_local_pose is None:
            # If only init_pos or init_rot are provided, generate the 4x4 pose matrix
            from scipy.spatial.transform import Rotation as R

            T = np.eye(4)
            T[:3, 3] = np.array(cfg.init_pos)
            T[:3, :3] = R.from_euler("xyz", np.deg2rad(cfg.init_rot)).as_matrix()
            cfg.init_local_pose = T
        else:
            # If only init_local_pose is provided, extract init_pos and init_rot
            from scipy.spatial.transform import Rotation as R

            T = np.array(cfg.init_local_pose)
            cfg.init_pos = tuple(T[:3, 3])
            cfg.init_rot = tuple(R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True))

        return cfg


@configclass
class LightCfg(ObjectBaseCfg):
    """Configuration for a light asset in the simulation.

    Supports six light types matching the dexsim rendering backend:

    - ``"point"``: Per-environment omnidirectional point light with position
      and falloff radius. Created as a batched light (one per environment).
    - ``"sun"``: Global directional sun light (infinite distance). Created as
      a single scene-level instance. Uses direction only; position is ignored.
      Sun-specific fields (``angular_radius``, ``halo_size``, ``halo_falloff``)
      are reserved for future backend support.
    - ``"direction"``: Global pure directional light at infinite distance.
      Created as a single scene-level instance. Direction only; no position.
    - ``"spot"``: Per-environment spotlight with position, direction, and
      inner/outer cone angles. Created as a batched light.
    - ``"rect"``: Per-environment rectangular area light with position,
      direction, width, and height. Created as a batched light.
    - ``"mesh"``: Per-environment mesh-based emissive light. Requires a
      :class:`~dexsim.models.MeshObject` via
      :meth:`embodichain.lab.sim.objects.light.Light.set_mesh`
      (not tensor-batched). Created as a batched light.

    .. attention::
        The ``angular_radius``, ``halo_size``, and ``halo_falloff`` fields are
        reserved for future use. The dexsim Python bindings do not yet expose
        setters for these sun-specific properties.
    """

    light_type: Literal["point", "sun", "direction", "spot", "rect", "mesh"] = "point"
    """Light type. Supported: ``"point"``, ``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, ``"mesh"``."""

    # ------------------------------------------------------------------
    # Universal properties (apply to all light types)
    # ------------------------------------------------------------------

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """RGB color of the light source. Defaults to white ``(1.0, 1.0, 1.0)``."""

    intensity: float = 30.0
    """Intensity of the light source in watts/m^2. Defaults to ``30.0``."""

    enable_shadow: bool = True
    """Whether the light casts shadows. Defaults to ``True``."""

    # ------------------------------------------------------------------
    # Point light
    # ------------------------------------------------------------------

    radius: float = 10.0
    """Falloff radius for point lights. Only used when ``light_type="point"``. Defaults to ``10.0``."""

    # ------------------------------------------------------------------
    # Directional properties (sun, direction, spot, rect, mesh)
    # ------------------------------------------------------------------

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Direction vector for directional, spot, rect, and mesh lights.
    Defaults to ``(0.0, 0.0, -1.0)`` (pointing down along -Z)."""

    # ------------------------------------------------------------------
    # Sun light (reserved — Python bindings not yet available)
    # ------------------------------------------------------------------

    angular_radius: float = 0.5
    """Angular radius of the sun disc in degrees. Reserved for future use."""

    halo_size: float = 10.0
    """Halo size for sun light. Reserved for future use."""

    halo_falloff: float = 3.0
    """Halo falloff for sun light. Reserved for future use."""

    # ------------------------------------------------------------------
    # Spot light
    # ------------------------------------------------------------------

    spot_angle_inner: float = 30.0
    """Inner cone angle of the spotlight in degrees. Only used when ``light_type="spot"``.
    Defaults to ``30.0``."""

    spot_angle_outer: float = 45.0
    """Outer cone angle of the spotlight in degrees. Only used when ``light_type="spot"``.
    Defaults to ``45.0``."""

    # ------------------------------------------------------------------
    # Rect light
    # ------------------------------------------------------------------

    rect_width: float = 1.0
    """Width of the rectangular area light. Only used when ``light_type="rect"``.
    Defaults to ``1.0``."""

    rect_height: float = 1.0
    """Height of the rectangular area light. Only used when ``light_type="rect"``.
    Defaults to ``1.0``."""

    # ------------------------------------------------------------------
    # Mesh light
    # ------------------------------------------------------------------

    mesh_path: str = ""
    """Asset path for mesh-based emissive lights. Only used when ``light_type="mesh"``.
    The actual mesh assignment is done via
    :meth:`embodichain.lab.sim.objects.light.Light.set_mesh` which accepts a
    :class:`dexsim.models.MeshObject`. This field stores the path for reference."""


@configclass
class RigidObjectCfg(ObjectBaseCfg):
    """Configuration for a rigid body asset in the simulation.

    This class extends the base asset configuration to include specific properties for rigid bodies,
    such as physical attributes and collision group.
    """

    shape: ShapeCfg = ShapeCfg()
    """Shape configuration for the rigid body. """

    # TODO: supoort basic primitive shapes, such as box, sphere, etc cfg and spawn method.

    attrs: RigidBodyPhysicsCfg | RigidBodyAttributesCfg = RigidBodyPhysicsCfg()
    """Rigid-body physics.

    The grouped :class:`RigidBodyPhysicsCfg` is backend-aware. The deprecated
    flat :class:`RigidBodyAttributesCfg` is accepted by the Default backend only.
    """

    body_type: Literal["dynamic", "kinematic", "static"] = "dynamic"

    max_convex_hull_num: int = MISSING
    """The maximum number of convex hulls that will be created for the rigid body.

    .. deprecated::
        Use :attr:`MeshCfg.max_convex_hull_num` instead. This field is kept for
        backward compatibility and overrides the shape-level value when explicitly set.

    If set to larger than 1, the rigid body will be decomposed into multiple convex hulls
    using the approximate convex decomposition method specified by :attr:`acd_method`.
    Reference: https://github.com/SarahWeiii/CoACD
    """

    acd_method: str = MISSING
    """The method used for approximate convex decomposition (ACD) of the mesh.

    .. deprecated::
        Use :attr:`MeshCfg.acd_method` instead. This field is kept for
        backward compatibility and overrides the shape-level value when explicitly set.

    Currently, ``"coacd"`` and ``"vhacd"`` are supported. Only used when
    :attr:`max_convex_hull_num` is set to larger than 1.
    """

    sdf_resolution: int = MISSING
    """Resolution for the signed distance field (SDF) of the rigid body.

    .. deprecated::
        Use :attr:`MeshCfg.sdf_resolution` instead. This field is kept for
        backward compatibility and overrides the shape-level value when explicitly set.

    The spacing of the uniformly sampled SDF is equal to the largest AABB extent
    of the mesh, divided by the resolution. If ``sdf_resolution`` is set to larger
    than 0, an SDF will be generated for collision detection. SDF will increase the
    accuracy of collision, but also takes more time to initialize and simulate.
    """

    body_scale: tuple | list = (1.0, 1.0, 1.0)
    """Scale of the rigid body in the simulation world frame."""

    asset_physics_mode: AssetPhysicsMode | None = None
    """How a file-backed asset's physical properties are handled.

    ``"preserve"`` keeps the USD-authored physics. ``"overlay"`` applies
    configured properties on top of the parsed asset. ``None`` selects the
    rigid-object default, ``"preserve"``. Procedural shapes always use config.
    """

    use_usd_properties: bool | None = None
    """Deprecated alias for :attr:`asset_physics_mode`.

    ``True`` maps to ``"preserve"`` and ``False`` maps to ``"overlay"``.
    """

    def resolve_asset_physics_mode(self) -> AssetPhysicsMode:
        """Return the effective file-backed physics policy."""
        return _resolve_asset_physics_mode(
            self.asset_physics_mode,
            self.use_usd_properties,
            default="preserve",
        )

    def to_dexsim_body_type(self) -> ActorType:
        """Convert the body type to dexsim ActorType."""
        if self.body_type == "dynamic":
            return ActorType.DYNAMIC
        elif self.body_type == "kinematic":
            return ActorType.KINEMATIC
        elif self.body_type == "static":
            return ActorType.STATIC
        else:
            logger.log_error(
                f"Invalid body type '{self.body_type}' specified. Must be one of 'dynamic', 'kinematic', or 'static'."
            )


@configclass
class DeformableObjectCfg(ObjectBaseCfg):
    """Common configuration contract for one deformable asset.

    Concrete volume and surface configurations retain their native DexSim
    properties. The discriminator is explicit so manager and visualization
    code do not need to infer topology from a mesh or material type.
    """

    deformable_type: Literal["volume", "surface"] = MISSING
    """Physical topology represented by the asset."""

    shape: MeshCfg = MeshCfg()
    """Render and source-mesh configuration."""


@configclass
class VolumeDeformableObjectCfg(DeformableObjectCfg):
    """Configuration for a volume deformable backed by DexSim ``SoftBody``."""

    deformable_type: Literal["volume"] = "volume"

    voxel_attr: SoftbodyVoxelAttributesCfg = SoftbodyVoxelAttributesCfg()
    """Tetrahedral simulation-mesh voxelization attributes."""

    physical_attr: SoftbodyPhysicalAttributesCfg = SoftbodyPhysicalAttributesCfg()
    """DexSim volume-deformable physical attributes."""


@configclass
class SoftObjectCfg(VolumeDeformableObjectCfg):
    """Compatibility name for :class:`VolumeDeformableObjectCfg`."""


@configclass
class SurfaceDeformableObjectCfg(DeformableObjectCfg):
    """Configuration for a surface deformable backed by DexSim ``ClothBody``."""

    deformable_type: Literal["surface"] = "surface"

    physical_attr: ClothPhysicalAttributesCfg = ClothPhysicalAttributesCfg()
    """DexSim surface-deformable physical attributes."""


@configclass
class ClothObjectCfg(SurfaceDeformableObjectCfg):
    """Compatibility name for :class:`SurfaceDeformableObjectCfg`."""


@configclass
class RigidObjectGroupCfg:
    """Configuration for a rigid object group asset in the simulation.

    Rigid object groups can be initialized from multiple rigid object configurations specified in a folder.
    If `folder_path` is specified, user should provide a RigidObjectCfg in `rigid_objects` as a template configuration for
    all objects in the group.

    For example:
    ```python
    rigid_object_group: RigidObjectGroupCfg(
        folder_path="path/to/folder",
        max_num=5,
        rigid_objects={
            "template_obj": RigidObjectCfg(
                shape=MeshCfg(
                    fpath="",  # fpath will be ignored when folder_path is specified
                ),
                body_type="dynamic",
            )
        }
    )
    """

    uid: str | None = None

    rigid_objects: Dict[str, RigidObjectCfg] = MISSING
    """Configuration for the rigid objects in the group."""

    body_type: Literal["dynamic", "kinematic"] = "dynamic"
    """Body type for all rigid objects in the group. """

    folder_path: str | None = None
    """Path to the folder containing the rigid object assets.
    
    This is used to initialize multiple rigid object configurations from a folder.
    """

    max_num: int = 1
    """Maximum number of rigid objects to initialize from the folder.
    
    This is only used when `folder_path` is specified.
    """

    ext: str = ".obj"
    """File extension for the rigid object assets.
    
    This is only used when `folder_path` is specified.
    """

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> RigidObjectGroupCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif key == "rigid_objects" and "folder_path" not in init_dict:
                    rigid_objects_cfg = {}
                    for obj_name, obj_cfg in value.items():
                        rigid_objects_cfg[obj_name] = RigidObjectCfg.from_dict(obj_cfg)
                    setattr(cfg, key, rigid_objects_cfg)
                elif key == "rigid_objects" and "folder_path" in init_dict:
                    folder_path = init_dict["folder_path"]
                    max_num = init_dict.get("max_num", 1)
                    rigid_objects_cfg = {}
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        files = os.listdir(folder_path)
                        files = [f for f in files if f.endswith(cfg.ext)]
                        # select files up to max_num
                        n_file = len(files)
                        select_files = []
                        for i in range(max_num):
                            select_files.append(files[i % n_file])

                        for i, file_name in enumerate(select_files):
                            file_path = os.path.join(folder_path, file_name)
                            rigid_obj_cfg: RigidObjectCfg = RigidObjectCfg.from_dict(
                                list(init_dict["rigid_objects"].values())[0]
                            )
                            rigid_obj_cfg.uid = f"{cfg.uid}_obj_{i}"
                            rigid_obj_cfg.shape.fpath = file_path
                            rigid_objects_cfg[rigid_obj_cfg.uid] = rigid_obj_cfg
                        setattr(cfg, "rigid_objects", rigid_objects_cfg)
                    else:
                        logger.log_error(
                            f"Folder '{folder_path}' does not exist or is not a directory."
                        )
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


@configclass
class RigidConstraintCfg:
    """Configuration for a fixed constraint between two RigidObjects.

    The constraint binds rigid_object_a's entity[i] to rigid_object_b's entity[i]
    within arena[i] (one constraint per arena).

    Args:
        name: Base constraint name. Per-arena names are derived as ``f"{name}"``
            (single env) or ``f"{name}_{i}"`` (multi env).
        rigid_object_a_uid: UID of the first RigidObject (must exist in the sim).
        rigid_object_b_uid: UID of the second RigidObject (must exist in the sim).
        local_frame_a: 4x4 joint frame in object A's local coordinates.
            ``None`` -> identity (object A's origin). Accepts a single
            ``(4, 4)`` matrix (shared by all envs) or an ``(N, 4, 4)`` array
            (one frame per env). Defaults to None.
        local_frame_b: 4x4 joint frame in object B's local coordinates.
            ``None`` -> the frame is computed per env as ``inv(pose_B) @ pose_A``
            from the objects' current poses, so the constraint welds the objects
            at their *current* relative pose (rather than pulling their origins
            together). An explicit ``(4, 4)`` or ``(N, 4, 4)`` value is used
            verbatim. Defaults to None.
        constraint_type: Reserved for future typed constraints (prismatic,
            revolute, spherical, d6). Only ``"fixed"`` is supported in v1.

    .. attention::
        Both objects must be :class:`RigidObject` instances and must share the
        same number of arenas.
    """

    name: str = MISSING
    """Base name of the constraint (per-arena names are derived from this)."""

    rigid_object_a_uid: str = MISSING
    """UID of the first RigidObject."""

    rigid_object_b_uid: str = MISSING
    """UID of the second RigidObject."""

    local_frame_a: np.ndarray | None = None
    """Local joint frame on object A. None -> identity (object A's origin)."""

    local_frame_b: np.ndarray | None = None
    """Local joint frame on object B. None -> ``inv(pose_B) @ pose_A`` per env
    (weld at the objects' current relative pose)."""

    constraint_type: Literal["fixed"] = "fixed"
    """Constraint type. Only ``"fixed"`` is supported in v1."""


@configclass
class URDFCfg:
    """Standalone configuration class for URDF assembly."""

    components: Dict[str, Dict[str, str | Dict | np.ndarray]] = field(
        default_factory=dict
    )
    """Dictionary of robot components to be assembled."""

    sensors: Dict[str, Dict[str, str | np.ndarray]] = field(default_factory=dict)
    """Dictionary of sensors to be attached to the robot."""

    use_signature_check: bool = True
    """Whether to use signature check when merging URDFs."""

    base_link_name: str = "base_link"
    """Name of the base link in the assembled robot."""

    fpath: str | None = None
    """Full output file path for the assembled URDF. If specified, overrides fname and fpath_prefix."""

    fname: str | None = None
    """Name used for output file and directory. If not specified, auto-generated from component names."""

    fpath_prefix: str = EMBODICHAIN_DEFAULT_DATA_ROOT + "/assembled"
    """Output directory prefix for the assembled URDF file."""

    component_prefix: List[tuple[str, str | None]] = field(
        default_factory=lambda: [
            ("chassis", None),
            ("legs", None),
            ("torso", None),
            ("head", None),
            ("left_arm", "left_"),
            ("right_arm", "right_"),
            ("left_hand", "left_"),
            ("right_hand", "right_"),
            ("arm", None),
            ("hand", None),
        ]
    )
    """Component name prefixes used during URDF assembly.

    Preferred form is a list of ``(component_name, prefix)`` tuples. For
    convenience, a mapping ``{component_name: prefix}`` is also accepted when
    constructing :class:`URDFCfg` and will be normalized internally.
    """

    name_case: dict[str, str] = field(
        default_factory=lambda: {
            "joint": "original",
            "link": "original",
        }
    )
    """Case normalization policy applied to joint/link names during URDF assembly.

    Supported values per key are ``"upper"``, ``"lower"`` or ``"original"``
    (legacy alias ``"none"``). The default preserves source URDF casing.
    """

    def __init__(
        self,
        components: list[dict[str, str | np.ndarray]] | None = None,
        sensors: dict[str, dict[str, str | np.ndarray]] | None = None,
        fpath: str | None = None,
        fname: str | None = None,
        fpath_prefix: str = EMBODICHAIN_DEFAULT_DATA_ROOT + "/assembled",
        use_signature_check: bool = True,
        base_link_name: str = "base_link",
        component_prefix: list[tuple[str, str | None]] | None = None,
        name_case: dict[str, str] | None = None,
    ):
        """
        Initialize URDFCfg with optional list of components and output path settings.

        Args:
            components (list[dict[str, str | np.ndarray]] | None): List of component configurations. Each dict should contain:
                - 'component_type' (str): The type/name of the component (e.g., 'chassis', 'arm', 'hand').
                - 'urdf_path' (str): Path to the component's URDF file.
                - 'transform' (np.ndarray | None): 4x4 transformation matrix (optional).
                - Additional params can be included as extra keys.
            sensors (dict[str, dict[str, str | np.ndarray]] | None): Sensor configurations for the robot.
            fpath (str | None): Full output file path for the assembled URDF. If specified, overrides fname and fpath_prefix.
            fname (str | None): Name used for output file and directory. If not specified, auto-generated from component names.
            fpath_prefix (str): Output directory prefix for the assembled URDF file.
            use_signature_check (bool): Whether to use signature check when merging URDFs.
            base_link_name (str): Name of the base link in the assembled robot.
            component_prefix (list[tuple[str, str | None]] | None): Optional
                list of (component_type, prefix) pairs to override default
                component name prefixes.
        """
        self.components = {}
        self.sensors = sensors or {}
        self.fpath = fpath
        self.use_signature_check = use_signature_check
        self.base_link_name = base_link_name
        self.fname = fname
        self.fpath_prefix = fpath_prefix

        # Initialize component prefixes (patch-style mapping per component type)
        if component_prefix is None:
            # Use the same default as the dataclass field
            self.component_prefix = [
                ("chassis", None),
                ("legs", None),
                ("torso", None),
                ("head", None),
                ("left_arm", "left_"),
                ("right_arm", "right_"),
                ("left_hand", "left_"),
                ("right_hand", "right_"),
                ("arm", None),
                ("hand", None),
            ]
        elif isinstance(component_prefix, dict):
            # Allow dict-style config: {"left_hand": "l_", ...}
            self.component_prefix = list(component_prefix.items())
        else:
            # Assume caller provided a list of (component_name, prefix) tuples
            self.component_prefix = component_prefix

        if name_case is None:
            self.name_case = {
                "joint": "original",
                "link": "original",
            }
        else:
            self.name_case = name_case

        # Auto-add components if provided
        if components:
            for comp_config in components:
                if not isinstance(comp_config, dict):
                    logger.log_error(
                        f"Component configuration must be a dict, got {type(comp_config)}"
                    )
                    continue

                # Extract required fields
                component_type = comp_config.get("component_type")
                urdf_path = comp_config.get("urdf_path")

                if not component_type or not urdf_path:
                    logger.log_error(
                        f"Component configuration must contain 'component_type' and 'urdf_path', got {comp_config}"
                    )
                    continue

                # Extract optional fields
                transform = comp_config.get("transform", np.eye(4))

                # Extract additional params (exclude known keys)
                params = {
                    k: v
                    for k, v in comp_config.items()
                    if k not in ["component_type", "urdf_path", "transform"]
                }

                # Add the component
                self.add_component(component_type, urdf_path, transform, **params)

        if sensors is not None:
            # Accept both list and dict; serialization round-trips an empty
            # dict when no sensors are configured (the field default).
            if isinstance(sensors, dict) and not sensors:
                self.sensors = []
            elif not isinstance(sensors, (list, dict)):
                logger.log_error(
                    f"sensors must be a list of dicts or a dict, got {type(sensors)}"
                )
                self.sensors = []
            elif isinstance(sensors, dict):
                # dict keyed by sensor_name -> config
                self.sensors = list(sensors.values())
            else:
                # Optionally check each sensor dict
                valid_sensors = []
                for sensor_config in sensors:
                    if not isinstance(sensor_config, dict):
                        logger.log_error(
                            f"Sensor configuration must be a dict, got {type(sensor_config)}"
                        )
                        continue
                    sensor_name = sensor_config.get("sensor_name")
                    if not sensor_name:
                        logger.log_error(
                            f"Sensor configuration must contain 'sensor_name', got {sensor_config}"
                        )
                        continue
                    valid_sensors.append(sensor_config)
                self.sensors = valid_sensors

    def set_urdf(self, urdf_path: str) -> "URDFCfg":
        """Directly specify a single URDF file for the robot, compatible with the single-URDF robot case.

        Args:
            urdf_path (str): Path to the robot's URDF file.

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        self.components.clear()
        urdf_file = os.path.splitext(os.path.basename(urdf_path))[0]
        self.components[urdf_file] = {
            "urdf_path": urdf_path,
            "transform": None,
            "params": {},
        }
        self.fpath = urdf_path
        return self

    def add_component(
        self,
        component_type: str,
        urdf_path: str,
        transform: np.ndarray | None = None,
        **params,
    ) -> URDFCfg:
        """Add a robot component to the assembly configuration.

        Args:
            component_type (str): The type/name of the component. Should be one of SUPPORTED_COMPONENTS
                (e.g., 'chassis', 'torso', 'head', 'left_arm', 'right_hand', 'arm', 'hand', etc.).
            urdf_path (str): Path to the component's URDF file.
            transform (np.ndarray | None): 4x4 transformation matrix for the component in the robot frame (default: None).
            **params: Additional keyword parameters for the component (e.g., color, material, etc.).

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        if urdf_path:
            if not os.path.exists(urdf_path):
                urdf_path_candidate = get_data_path(urdf_path)
                if os.path.exists(urdf_path_candidate):
                    urdf_path = urdf_path_candidate
                else:
                    logger.log_error(f"URDF path '{urdf_path}' does not exist.")
                    raise FileNotFoundError(f"URDF path '{urdf_path}' does not exist.")

        if transform is None:
            transform = np.eye(4)

        self.components[component_type] = {
            "urdf_path": urdf_path,
            "transform": np.array(transform),
            "params": params,
        }

        if self.fname:
            self.fpath = f"{self.fpath_prefix}/{self.fname}/{self.fname}.urdf"
        else:
            # Update output_path to use all component urdf file names joined by underscores as directory
            if len(self.components) == 1:
                # Only one component, use its urdf file name
                urdf_file = os.path.splitext(os.path.basename(urdf_path))[0]
                name = urdf_file
            else:
                # Multiple components, join all urdf file names
                urdf_files = [
                    os.path.splitext(os.path.basename(v["urdf_path"]))[0]
                    for v in self.components.values()
                ]
                name = "_".join(urdf_files)
            self.fpath = f"{self.fpath_prefix}/{name}/{name}.urdf"

        return self

    def add_sensor(self, sensor_name: str, **sensor_config) -> URDFCfg:
        """Add a sensor to the robot configuration.

        Args:
            sensor_name (str): The name of the sensor.
            **sensor_config: Additional configuration parameters for the sensor.

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        self.sensors.append({"sensor_name": sensor_name, **sensor_config})
        return self

    def assemble_urdf(self) -> str:
        """Assemble URDF files for the robot based on the configuration.

        Returns:
            str: The path to the resulting (possibly merged) URDF file.
        """
        components = list(self.components.items())
        # If there is only one component, return its URDF path directly.
        if len(components) == 1:
            _, comp_config = components[0]
            return comp_config["urdf_path"]

        from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

        # If there are multiple components, merge them into a single URDF file.
        manager = URDFAssemblyManager()
        manager.base_link_name = self.base_link_name

        if self.component_prefix is None:
            self.component_prefix = [
                ("left_arm", "left_"),
                ("right_arm", "right_"),
                ("left_hand", "left_"),
                ("right_hand", "right_"),
            ]
        if isinstance(self.component_prefix, dict):
            self.component_prefix = list(self.component_prefix.items())
        # Forward configured component prefixes to the assembly manager
        manager.component_prefix = self.component_prefix

        if self.name_case is not None:
            manager.name_case = self.name_case

        for comp_type, comp_config in components:
            params = comp_config.get("params", {})
            success = manager.add_component(
                comp_type,
                comp_config["urdf_path"],
                comp_config.get("transform"),
                **params,
            )
            if not success:
                logger.log_error(
                    f"Failed to add component '{comp_type}' with config: {comp_config}"
                )

        for sensor in self.sensors:
            manager.attach_sensor(
                sensor_name=sensor.get("sensor_name"),
                sensor_source=sensor.get("sensor_source"),
                parent_component=sensor.get("parent_component"),
                parent_link=sensor.get("parent_link"),
                sensor_type=sensor.get("sensor_type"),
                **{
                    k: v
                    for k, v in sensor.items()
                    if k
                    not in [
                        "sensor_name",
                        "sensor_source",
                        "parent_component",
                        "parent_link",
                        "sensor_type",
                    ]
                },
            )

        try:
            # Merge all added components into a single URDF file at the specified output path.
            merged_urdf_xml = manager.merge_urdfs(self.fpath, self.use_signature_check)
        except Exception as e:
            logger.log_error(f"URDF merge failed: {e}")

        return self.fpath

    @classmethod
    def from_dict(cls, init_dict: Dict) -> "URDFCfg":
        if isinstance(init_dict, cls):
            return init_dict
        components = init_dict.get("components", None)
        if isinstance(components, dict):
            components = [{"component_type": k, **v} for k, v in components.items()]
        sensors = init_dict.get("sensors", None)
        fpath = init_dict.get("fpath", None)
        use_signature_check = init_dict.get("use_signature_check", True)
        base_link_name = init_dict.get("base_link_name", "base_link")
        component_prefix = init_dict.get("component_prefix", None)
        name_case = init_dict.get("name_case", None)
        return cls(
            components=components,
            sensors=sensors,
            fpath=fpath,
            use_signature_check=use_signature_check,
            base_link_name=base_link_name,
            component_prefix=component_prefix,
            name_case=name_case,
        )


@configclass
class ArticulationCfg(ObjectBaseCfg):
    """Configuration for an articulation asset in the simulation.

    This class extends the base asset configuration to include specific properties for articulations,
    such as joint drive properties, physical attributes.
    """

    fpath: str = None
    """Path to the articulation asset file."""

    drive_pros: JointDrivePropertiesCfg | None = None
    """Optional joint-drive overrides.

    ``None`` preserves source drive properties. Individual ``None`` fields in
    a provided config also preserve the corresponding source values.
    """

    body_scale: tuple | list = (1.0, 1.0, 1.0)
    """Scale of the articulation in the simulation world frame."""

    attrs: RigidBodyPhysicsCfg | RigidBodyAttributesCfg = RigidBodyPhysicsCfg()
    """Physical attributes for all links. We use default mass from the USD/URDF file if available.
    The mass and density in attrs will only be used if specified. Deprecated
    flat :class:`RigidBodyAttributesCfg` inputs are Default-backend-only.
    """

    link_attrs: dict[str, LinkPhysicsOverrideCfg] | None = None
    """Named per-link physics override groups keyed by regex on link names.

    Each group applies :attr:`LinkPhysicsOverrideCfg.attrs` on top of :attr:`attrs` for
    matched links only. A link must not match more than one group.
    """

    articulation_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg()
    """Grouped articulation-root properties.

    Non-``None`` values take precedence over the legacy ``fix_base`` and
    ``disable_self_collision`` fields.
    """

    fix_base: bool = True
    """Whether to fix the base of the articulation.

    Set to True for articulations that should not move, such as a fixed base robot arm or a door.
    Set to False for articulations that should move freely, such as a mobile robot or a humanoid robot.
    """

    disable_self_collision: bool = True
    """Whether to enable or disable self-collisions."""

    init_qpos: torch.Tensor | np.ndarray | Sequence[float] = None
    """Initial joint positions of the articulation.

    If None, the joint positions will be set to zero.
    If provided, it should be a array of shape (num_joints,).
    """

    qpos_limits: (
        torch.Tensor | np.ndarray | Sequence[float] | Dict[str, List[float]] | None
    ) = None
    """Override joint position limits of the articulation.

    If None, the joint position limits from the asset file (URDF/USD) are used.
    If provided as a tensor/array of shape (num_joints, 2), it is applied to all
    joints in the order of ``joint_names``.
    If provided as a dictionary, keys are joint names or regular expressions and
    values are ``[min, max]`` limits.

    This field replaces the asset limits for the articulation and can be used to
    either tighten or expand the allowed range.
    """

    sleep_threshold: float = 0.005
    """Energy below which the articulation may go to sleep. Range: [0, max_float32]"""

    min_position_iters: int = 4
    """Number of position iterations the solver should perform for this articulation. Range: [1,255]."""

    min_velocity_iters: int = 1
    """Number of velocity iterations the solver should perform for this articulation. Range: [0,255]."""

    build_pk_chain: bool = True
    """Whether to build pytorch-kinematics chain for forward kinematics and jacobian computation."""

    compute_uv: bool = False
    """Whether to compute the UV mapping for the articulation link.
    
    Currently, the uv mapping is computed for each link with projection uv mapping method.
    """

    asset_physics_mode: AssetPhysicsMode | None = None
    """How source-authored articulation physics is handled.

    ``"preserve"`` keeps link, joint-drive, and joint-limit properties from
    either USD or URDF. ``"overlay"`` applies only explicitly configured
    values after the source has been resolved. ``None`` selects the generic
    articulation default, ``"preserve"``.

    Import policy such as URDF root fixation and body scale remains controlled
    by its dedicated fields because standard URDF does not author those values.
    """

    use_usd_properties: bool | None = None
    """Deprecated alias for :attr:`asset_physics_mode`.

    ``True`` maps to ``"preserve"`` and ``False`` maps to ``"overlay"`` for
    both USD and URDF sources.
    """

    def resolve_asset_physics_mode(self) -> AssetPhysicsMode:
        """Return the effective file-backed physics policy."""
        return _resolve_asset_physics_mode(
            self.asset_physics_mode,
            self.use_usd_properties,
            default=self._default_asset_physics_mode(),
        )

    def _default_asset_physics_mode(self) -> AssetPhysicsMode:
        """Return the policy used when no compatibility field is authored."""
        return "preserve"

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, str | float | tuple | dict]
    ) -> ArticulationCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if key == "link_attrs" and isinstance(value, dict):
                cfg.link_attrs = link_attrs_from_dict(value)
            elif key == "attrs" and isinstance(value, Mapping):
                cfg.attrs = _rigid_body_attrs_from_dict(value)
            elif key == "drive_pros" and isinstance(value, Mapping):
                cfg.drive_pros = JointDrivePropertiesCfg.from_dict(
                    dict(value),
                    defaults=cfg.drive_pros,
                )
            elif hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(cfg, key, attr.from_dict(value))
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )

        if cfg.init_local_pose is None:
            from scipy.spatial.transform import Rotation as R

            T = np.eye(4)
            T[:3, 3] = np.array(cfg.init_pos)
            T[:3, :3] = R.from_euler("xyz", np.deg2rad(cfg.init_rot)).as_matrix()
            cfg.init_local_pose = T
        else:
            from scipy.spatial.transform import Rotation as R

            cfg.init_pos = tuple(cfg.init_local_pose[:3, 3])
            cfg.init_rot = tuple(
                R.from_matrix(cfg.init_local_pose[:3, :3]).as_euler("xyz", degrees=True)
            )

        return cfg


@configclass
class RobotCfg(ArticulationCfg):
    from embodichain.lab.sim.solvers import SolverCfg

    """Configuration for a robot asset in the simulation.
    """

    drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg(
        drive_type="force",
        stiffness=1e4,
        damping=1e3,
        max_effort=1e10,
        max_velocity=1e10,
        friction=0.0,
        armature=0.0,
    )
    """Properties to define the drive mechanism of a joint."""

    def _default_asset_physics_mode(self) -> AssetPhysicsMode:
        """Keep the established Robot behavior of applying drive config."""
        return "overlay"

    control_parts: Dict[str, List[str]] | None = None
    """Control parts is the mapping from part name to joint names.

    For example, {'left_arm': ['joint1', 'joint2'], 'right_arm': ['joint3', 'joint4']}
    If no control part is specified, the robot will use all joints as a single control part.

    Note: 
        - if `control_parts` is specified, `solver_cfg` must be a dict with part names as
            keys corresponding to the control parts name.
        - The joint names in the control parts support regular expressions, e.g., 'joint[1-6]'.
            After initialization of robot, the names will be expanded to a list of full joint names.
        - `Robot` is a derived class of `Articulation`, with control parts support. So the `drive_pros`
            in `ArticulationCfg` can use control part as key to specify the corresponding joint drive properties, 
            which will be overridden if these joint names are already specified.
    """

    urdf_cfg: URDFCfg | None = None
    """URDF assembly configuration which allows for assembling a robot from multiple URDF components.
    """

    # TODO: how to support one solver for multiple parts?
    solver_cfg: SolverCfg | Dict[str, SolverCfg] | None = None
    """Solver is used to compute forward and inverse kinematics for the robot.
    """

    workspace_cfg: Dict[str, RobotWorkspaceCfg] | None = None
    """Runtime workspace cache configuration keyed by control-part name."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, str | float | tuple]) -> RobotCfg:
        """Initialize the configuration from a dictionary."""
        if isinstance(init_dict, cls):
            return init_dict

        import importlib

        solver_module = importlib.import_module("embodichain.lab.sim.solvers")

        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if key == "link_attrs" and isinstance(value, dict):
                cfg.link_attrs = link_attrs_from_dict(value)
            elif key == "attrs" and isinstance(value, Mapping):
                cfg.attrs = _rigid_body_attrs_from_dict(value)
            elif hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "urdf_cfg":
                    from embodichain.lab.sim.cfg import URDFCfg

                    setattr(cfg, key, URDFCfg.from_dict(value))
                elif key == "workspace_cfg" and isinstance(value, dict):
                    setattr(
                        cfg,
                        key,
                        {
                            part: (
                                part_cfg
                                if isinstance(part_cfg, RobotWorkspaceCfg)
                                else RobotWorkspaceCfg(**part_cfg)
                            )
                            for part, part_cfg in value.items()
                        },
                    )
                elif key == "fpath":
                    setattr(cfg, key, get_data_path(value))
                elif isinstance(attr, JointDrivePropertiesCfg) and isinstance(
                    value, dict
                ):
                    setattr(
                        cfg,
                        key,
                        JointDrivePropertiesCfg.from_dict(value, defaults=attr),
                    )
                elif is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif isinstance(value, dict) and "class_type" in value:
                    setattr(
                        cfg,
                        key,
                        getattr(solver_module, f"{value['class_type']}Cfg").from_dict(
                            value
                        ),
                    )
                elif isinstance(value, dict) and key_in_nested_dict(
                    value, "class_type"
                ):
                    setattr(
                        cfg,
                        key,
                        {
                            k: getattr(
                                solver_module, f"{v['class_type']}Cfg"
                            ).from_dict(v)
                            for k, v in value.items()
                        },
                    )

                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default config fields from ``init_dict``.

        Subclasses override this to read variant/version fields from
        ``init_dict``, set them on ``self``, and populate ``urdf_cfg``,
        ``control_parts``, ``solver_cfg``, ``drive_pros`` and ``attrs``.
        The base implementation is a no-op.

        .. attention::
            Do NOT call :func:`merge_robot_cfg` from here -- the subclass
            ``from_dict`` calls this hook first, then ``merge_robot_cfg``.
            Calling ``merge_robot_cfg`` here would recurse, because
            ``merge_robot_cfg`` itself calls ``RobotCfg.from_dict``.

        Args:
            init_dict: The raw override dict passed to ``from_dict``.
        """
        return None

    def to_dict(self):
        """Serialize config to a plain dict (enums, numpy, nested configclass)."""

        def serialize(obj, _visited=None):
            if _visited is None:
                _visited = set()
            if isinstance(obj, enum.Enum):
                return obj.value
            tracked_id = None
            if not isinstance(obj, (str, int, float, bool, type(None))):
                tracked_id = id(obj)
                if tracked_id in _visited:
                    return None
                _visited.add(tracked_id)

            try:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {
                        (k.value if isinstance(k, enum.Enum) else str(k)): serialize(
                            v, _visited
                        )
                        for k, v in obj.items()
                    }
                if isinstance(obj, (list, tuple)):
                    return [serialize(v, _visited) for v in obj]
                if hasattr(obj, "to_dict") and obj is not self:
                    return serialize(obj.to_dict(), _visited)
                if hasattr(obj, "__dict__"):
                    return {
                        k: serialize(v, _visited)
                        for k, v in obj.__dict__.items()
                        if v is not None
                    }
                return obj
            finally:
                if tracked_id is not None:
                    _visited.remove(tracked_id)

        return serialize(self)

    def to_string(self):
        """Return config as a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, filepath):
        """Save config to a local file as JSON."""
        with open(filepath, "w") as f:
            f.write(self.to_string())

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build the serial chain from the URDF file.

        Note:
            This method is usually used in imitation dataset saving (compute eef pose from qpos using FK)
            and model training (provide a differentiable FK layer or loss computation).

        Args:
            device (torch.device): The device to which the chain will be moved. Defaults to CPU.
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            Dict[str, pk.SerialChain]: The serial chain of the robot for specified control part.
        """
        return {}
