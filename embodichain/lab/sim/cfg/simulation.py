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

"""World-level rendering and physics-backend configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field, fields
from typing import Any, Literal, Sequence, TYPE_CHECKING

import dexsim
import numpy as np
import torch
from dexsim.types import Renderer, ToneMappingType

from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import NewtonCfg
    from dexsim.engine.newton_physics.solvers_cfg import NewtonSolverCfg


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
    """GPU buffer capacities for the Default backend's GPU dynamics pipeline.

    Default-backend GPU buffers cannot all grow dynamically. Values that are
    too small may therefore produce overflow warnings, dropped contacts, or an
    invalid simulation. These settings are applied only when the Default
    backend runs on CUDA; they have no effect on Default CPU or Newton.
    """

    temp_buffer_capacity: int = 2**24
    """Temporary pinned-host buffer capacity in bytes.

    Increase this when the Default backend reports a pinned-host linear
    allocator overflow.
    """

    max_rigid_contact_count: int = 2**19
    """Maximum number of rigid-contact records in the GPU contact stream.

    Increase this when the Default backend reports
    ``Contact buffer overflow detected``.
    """

    max_rigid_patch_count: int = (
        2**18
    )  # 81920 is DexSim default but most tasks work with 2**18
    """Maximum number of rigid-contact patches in the GPU patch stream.

    A patch groups nearby contact points that share a contact normal. Increase
    this when the Default backend reports ``Patch buffer overflow detected``.
    """

    heap_capacity: int = 2**26
    """Initial capacity in bytes of the GPU and pinned-host memory heaps."""

    found_lost_pairs_capacity: int = (
        2**25
    )  # 262144 is DexSim default but most tasks work with 2**25
    """Capacity of broad-phase found/lost pair records."""

    found_lost_aggregate_pairs_capacity: int = 2**10
    """Capacity of found/lost pair records generated by aggregates."""

    total_aggregate_pairs_capacity: int = 2**10
    """Capacity of all aggregate-pair records in the GPU pipeline."""


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
    """Duration of one physics step in seconds.

    Environment control steps may contain multiple physics steps.  For Newton,
    this interval is further divided by :attr:`NewtonPhysicsCfg.num_substeps`.
    """

    device: str | torch.device = "cpu"
    """Compute device used to build and step the selected physics backend.

    Concrete backend configurations may redeclare this field when their native
    runtime has a different default.  In particular,
    :class:`NewtonPhysicsCfg` intentionally shadows this CPU default with
    ``"cuda:0"``.  Callers can still provide an explicit device override.
    """

    gravity: Sequence[float] | np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )
    """World-frame gravity vector in meters per second squared."""


@configclass
class DefaultPhysicsCfg(PhysicsBackendCfg):
    """Configuration selector for the Default physics backend."""

    bounce_threshold: float = 2.0
    """Relative normal-speed threshold below which contacts do not bounce [m/s]."""

    enable_ccd: bool = False
    """Whether to enable scene-level continuous collision detection (CCD).

    A rigid body must also set :attr:`DefaultRigidBodyPropertiesCfg.enable_ccd`
    for CCD to be used on that body.
    """

    length_tolerance: float = 0.05
    """Representative scene length used by the Default backend's tolerance scale [m].

    Set this near the characteristic size of simulated objects.  It is a scene
    scale, not an accuracy knob, and must be configured before world creation.
    """

    speed_tolerance: float = 0.25
    """Representative scene speed used by the Default backend's tolerance scale [m/s].

    The backend derives several internal thresholds from this value and
    :attr:`length_tolerance`.
    """

    gpu_memory: GPUMemoryCfg = field(default_factory=GPUMemoryCfg)
    """Fixed-capacity GPU buffers used by Default-backend CUDA simulation."""

    def to_dexsim_args(self) -> dict[str, Any]:
        """Convert to DexSim physics arguments.

        Solver implementation details that are not exposed by
        :class:`DefaultPhysicsCfg` retain their established defaults here.
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
class NewtonCollisionPipelineCfg:
    """Newton collision-pipeline settings owned at scene scope.

    These values map to DexSim's ``NewtonCollisionPipelineCfg``.  Per-shape
    contact and SDF values belong to :class:`NewtonCollisionPropertiesCfg`
    instead.  The pipeline performs broad-phase pair selection, narrow-phase
    contact generation, and optional contact reduction for the complete scene.

    See the `Newton collision guide
    <https://newton-physics.github.io/newton/latest/concepts/collisions.html>`_
    for the native pipeline semantics.
    """

    reduce_contacts: bool = True
    """Whether to reduce dense mesh contacts to a representative subset.

    Reduction lowers contact count and usually improves performance and solver
    stability for mesh-heavy scenes.
    """

    rigid_contact_max: int | None = None
    """Maximum number of allocated rigid contacts.

    ``None`` uses the model-provided capacity when available and otherwise lets
    Newton estimate it from the scene's shapes and candidate pairs.
    """

    max_triangle_pairs: int = 4_000_000
    """Maximum triangle-pair candidates allocated by the narrow phase.

    Increase this only when complex meshes or heightfields report triangle-pair
    overflow.  EmbodiChain intentionally uses a larger default than upstream
    Newton for mesh-heavy robotics scenes.
    """

    soft_contact_max: int | None = None
    """Maximum number of allocated particle/soft contacts.

    ``None`` lets Newton derive the capacity from shape and particle counts.
    """

    soft_contact_margin: float = 0.01
    """Distance margin used to generate particle/soft contacts [m]."""

    broad_phase: Literal["nxn", "sap", "explicit"] | Any | None = None
    """Built-in broad-phase mode or a prebuilt Newton broad-phase object.

    ``"explicit"`` tests precomputed pairs, ``"nxn"`` performs an all-pairs
    test, and ``"sap"`` uses sweep-and-prune.  ``None`` keeps Newton's default.
    A prebuilt object is an expert path and must be compatible with
    :attr:`narrow_phase`.
    """

    shape_pairs_filtered: Any | None = None
    """Optional precomputed pairs for ``"explicit"`` broad phase.

    When provided, this must be a Warp array of shape-index pairs with
    ``dtype=wp.vec2i``.  ``None`` uses the model's contact-pair list.
    """

    narrow_phase: Any | None = None
    """Optional prebuilt Newton narrow-phase object for expert pipelines."""

    sdf_hydroelastic_config: Any | None = None
    """Optional Newton ``HydroelasticSDF.Config``-compatible object.

    ``None`` disables the hydroelastic pipeline.  Individual participating
    procedural meshes must also opt in through
    :attr:`~embodichain.lab.sim.shapes.MeshCollisionCfg.is_hydroelastic`.
    """


@configclass
class NewtonPhysicsCfg(PhysicsBackendCfg):
    """Configuration selector for the Newton physics backend.

    DexSim wraps and extends Newton for EmbodiChain. The selected solver and
    collision pipeline are scene-wide. Shape, contact, material, and joint
    values are configured separately on object and articulation configs and
    compiled into DexSim Spawn descriptors.
    """

    device: str | torch.device = "cuda:0"
    """Warp device used to build and step Newton, for example ``"cuda:0"``.

    This redeclaration intentionally takes precedence over
    :class:`PhysicsBackendCfg.device`, so ``NewtonPhysicsCfg()`` always starts
    on CUDA unless the caller explicitly supplies another device.
    """

    num_substeps: int = 10
    """Number of Newton solver substeps per EmbodiChain physics step.

    The effective solver interval is ``physics_dt / num_substeps``.
    """

    requires_grad: bool = False
    """Whether to finalize the Newton model with differentiable state enabled.

    EmbodiChain currently requires the Semi-implicit solver for this mode and
    disables CUDA graph capture when gradients are enabled.
    """

    use_cuda_graph: bool = True
    """Whether to capture Newton stepping in a CUDA graph when supported.

    This is ignored for gradient mode and is unavailable on a CPU device.
    """

    debug_mode: bool = False
    """Whether to enable additional Newton runtime diagnostics."""

    suppress_warp_kernel_logs: bool = True
    """Whether to hide Warp startup and kernel compile/load messages.

    Genuine Newton/Warp warnings and errors are not suppressed.
    """

    solver_cfg: Mapping[str, Any] | NewtonSolverCfg | None = None
    """Optional Newton solver configuration.

    A mapping is converted to the matching DexSim Newton solver config. Include
    ``solver_type`` or ``class_type`` to select the solver, then add any
    parameters accepted by that DexSim solver config. If omitted, EmbodiChain
    preserves DexSim's scene-aware ``AutoSolverCfg`` default. A DexSim build
    exporting ``AutoSolverCfg`` is required; no concrete-solver fallback is used.
    """

    collision_cfg: NewtonCollisionPipelineCfg | Mapping[str, Any] = field(
        default_factory=NewtonCollisionPipelineCfg
    )
    """Scene-level Newton collision-pipeline configuration."""

    enable_collision_pipeline: bool = True
    """Whether Newton generates rigid contacts before each solver substep.

    Disable this only for a solver/workflow that deliberately obtains contacts
    elsewhere; ordinary rigid-body scenes require it.
    """

    broad_phase: Literal["nxn", "sap", "explicit"] | None = None
    """Deprecated shortcut for ``collision_cfg.broad_phase``.

    If both are set, ``collision_cfg.broad_phase`` wins.
    """

    visualizer_enabled: bool = False
    """Whether to enable DexSim Newton's optional diagnostic visualizer."""

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
            AutoSolverCfg,
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

        solver_cfg_map: dict[str, type] = {
            "auto": AutoSolverCfg,
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

        if self.requires_grad and (
            solver_cfg is None or solver_cfg.solver_type != "semi_implicit"
        ):
            logger.log_error(
                "Newton gradient mode requires an explicit "
                "solver_type='semi_implicit'; AutoSolver does not select a "
                "differentiable solver."
            )

        collision_values = {
            item.name: getattr(self.collision_cfg, item.name)
            for item in fields(self.collision_cfg)
        }
        if collision_values["broad_phase"] is None:
            collision_values["broad_phase"] = self.broad_phase
        collision_values["requires_grad"] = self.requires_grad

        newton_cfg_args: dict[str, Any] = {
            "dt": self.physics_dt,
            "num_substeps": self.num_substeps,
            "device": device,
            "gravity": _gravity_vector(self.gravity),
            "debug_mode": self.debug_mode,
            "requires_grad": self.requires_grad,
            "suppress_warp_kernel_logs": self.suppress_warp_kernel_logs,
            "collision_pipeline_cfg": NewtonCollisionPipelineCfg(**collision_values),
            "enable_collision_pipeline": self.enable_collision_pipeline,
            "sync_to_dexsim": True,
        }
        if solver_cfg is not None:
            newton_cfg_args["solver_cfg"] = solver_cfg

        cfg = NewtonCfg(
            **newton_cfg_args,
        )
        cfg.use_cuda_graph = self.use_cuda_graph and not self.requires_grad
        cfg._visualizer_enabled = self.visualizer_enabled
        return cfg


def _normalize_newton_solver_type(solver_type: str) -> str:
    """Normalize public EmbodiChain and DexSim Newton solver aliases."""
    key = solver_type.replace("-", "_").lower()
    aliases = {
        "auto": "auto",
        "autosolver": "auto",
        "autosolvercfg": "auto",
        "auto_solver": "auto",
        "auto_solver_cfg": "auto",
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
            "Expected one of 'auto', 'mjwarp', 'xpbd', 'semi_implicit', "
            "'featherstone', or 'vbd'."
        )
    return aliases[key]


def _newton_solver_cfg_to_dexsim(
    solver_cfg: Mapping[str, Any] | object | None,
    solver_cfg_map: Mapping[str, type],
) -> object | None:
    """Convert EmbodiChain Newton solver config input to a DexSim config."""
    if solver_cfg is None:
        return None

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
        or "auto"
    )
    normalized_solver_type = _normalize_newton_solver_type(str(configured_solver_type))
    solver_cfg_type = solver_cfg_map[normalized_solver_type]
    return solver_cfg_type(**solver_cfg_data)


def physics_cfg_for_backend(
    backend: Literal["default", "newton"],
) -> DefaultPhysicsCfg | NewtonPhysicsCfg:
    """Return a default physics configuration instance for the given backend."""
    if backend == "newton":
        return NewtonPhysicsCfg()
    if backend == "default":
        return DefaultPhysicsCfg()
    raise ValueError(
        f"Unsupported physics backend {backend!r}; expected 'default' or 'newton'."
    )


def physics_backend_from_cfg(
    physics_cfg: PhysicsBackendCfg,
) -> Literal["default", "newton"]:
    """Infer the physics backend name from a physics configuration instance."""
    if isinstance(physics_cfg, NewtonPhysicsCfg):
        return "newton"
    if isinstance(physics_cfg, DefaultPhysicsCfg):
        return "default"
    logger.log_error(
        f"Unsupported physics_cfg type '{type(physics_cfg).__name__}'. "
        "Expected DefaultPhysicsCfg or NewtonPhysicsCfg."
    )


def validate_physics_cfg(physics_cfg: PhysicsBackendCfg) -> None:
    """Validate that ``physics_cfg`` is a supported backend configuration."""
    physics_backend_from_cfg(physics_cfg)
