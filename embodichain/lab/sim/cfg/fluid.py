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

"""Initial-fill fluid configuration; numerical solvers remain owned by DexSim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
import math
import numpy as np

from embodichain.utils import configclass
from ..material import VisualMaterialCfg
from .asset import ObjectBaseCfg

__all__ = ["FluidSpawnCfg", "FluidRenderCfg", "FluidObjectCfg"]


@configclass
class FluidSpawnCfg:
    """Deterministic local particle source, with no runtime emission.

    Mesh sources describe a closed *fill volume*, not the solid container wall.
    File paths are resolved by the caller; NPZ sources use the ``positions`` key.
    ``reference_uid`` resolves a declared rigid body's initial pose once.
    """

    kind: Literal[
        "points", "box", "sphere", "cylinder", "revolved_volume", "mesh_volume"
    ] = "points"
    positions: Any = None
    file_path: str | None = None
    size: tuple[float, float, float] = (0.05, 0.05, 0.05)
    radius: float = 0.025
    height: float = 0.05
    clearance: float = 0.0
    profile: tuple[tuple[float, float], ...] | None = None
    """Piecewise-linear ``(local Z, radius)`` pairs for a capped revolved fill volume."""
    fill_level: float = 1.0
    """Filled fraction of the complete source Z extent; only for volume sources."""
    reference_uid: str | None = None
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_particles: int = 1_000_000
    """Bound the candidate grid allocation before sampling."""

    def sample(self, spacing: float) -> np.ndarray:
        """Return independent finite local positions of shape ``(N, 3)``.

        Args:
            spacing: Positive lattice spacing in metres.

        Returns:
            Contiguous float32 positions in deterministic lattice order.
        """
        from embodichain.compute.geometry.particles import sample_volume_particles

        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive.")
        if type(self.max_particles) is not int or self.max_particles <= 0:
            raise ValueError("max_particles must be a positive integer.")
        if self.kind == "points":
            if self.profile is not None or self.fill_level != 1.0:
                raise ValueError(
                    "points sources cannot apply profile or fill_level; use a volume source."
                )
            if (self.positions is None) == (self.file_path is None):
                raise ValueError(
                    "points requires exactly one of positions or file_path."
                )
            if self.file_path is not None:
                with np.load(self.file_path, allow_pickle=False) as source:
                    points = np.asarray(source["positions"], dtype=np.float32).copy()
            else:
                points = np.asarray(self.positions, dtype=np.float32).copy()
        else:
            if self.positions is not None:
                raise ValueError("positions is only valid for kind='points'.")
            points = sample_volume_particles(
                self.kind,
                spacing,
                size=self.size,
                radius=self.radius,
                height=self.height,
                clearance=self.clearance,
                profile=self.profile,
                fill_level=self.fill_level,
                mesh_path=self.file_path,
                max_particles=self.max_particles,
            )
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("Fluid positions must be finite, nonempty (N, 3) points.")
        if len(points) > self.max_particles:
            raise ValueError("Fluid source exceeds max_particles.")
        return np.ascontiguousarray(points)


@configclass
class FluidRenderCfg:
    """Native particle/surface display, independent of physical particle state.

    ``mesh_surface`` forwards DexSim FluidMeshSurfaceRenderDesc fields directly.
    Optional world-space clipping affects only a detached render copy.
    Native fluid geometry requires ``fast-rt`` or ``rt``; Hybrid's Vulkan
    primary surface does not contain the OptiX liquid instances.
    ``mode="none"`` currently requires the simulation NoRender backend.
    """

    mode: Literal["none", "particles", "mesh_surface"] = "particles"
    render_radius: float | None = None
    mesh_surface: dict[str, Any] = {}
    material: VisualMaterialCfg | None = None
    clip_below_z: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.material, Mapping):
            self.material = VisualMaterialCfg.from_dict(dict(self.material))
        if self.mode not in {"none", "particles", "mesh_surface"}:
            raise ValueError("Unsupported fluid render mode.")
        for name in ("render_radius", "clip_below_z"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.render_radius is not None and self.render_radius <= 0:
            raise ValueError("render_radius must be positive.")


@configclass
class FluidObjectCfg(ObjectBaseCfg):
    """A fixed-count Newton SPH fluid, declared before scene preparation.

    Density and spacing determine mass and must agree across fluid objects.
    Solver viscosity, surface tension and iteration controls belong exclusively
    to DexSim's SPHSolverCfg supplied to NewtonPhysicsCfg.solver_cfg.

    This config owns density and spacing. Native SPH defaults for those fields
    are placeholders in both mapping and native solver configs; non-default
    solver values must match the fluid declaration.
    """

    density: float = 1000.0
    particle_spacing: float = 0.0015
    particle_radius: float | None = None
    spawn: FluidSpawnCfg = FluidSpawnCfg()
    render: FluidRenderCfg = FluidRenderCfg()

    def __post_init__(self) -> None:
        if isinstance(self.spawn, Mapping):
            self.spawn = FluidSpawnCfg(**self.spawn)
        if isinstance(self.render, Mapping):
            self.render = FluidRenderCfg(**self.render)
        for name in ("density", "particle_spacing", "particle_radius"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        velocity = np.asarray(self.spawn.initial_velocity)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("initial_velocity must contain three finite values.")
        if (
            not isinstance(self.spawn.max_particles, int)
            or self.spawn.max_particles <= 0
        ):
            raise ValueError("max_particles must be a positive integer.")

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> FluidObjectCfg:
        """Decode the same typed configuration used by Python callers."""
        return cls(**init_dict)
