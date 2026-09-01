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

"""Deformable-body physical and object configuration."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal, Sequence

import numpy as np

from embodichain.utils import configclass

from ..shapes import MeshCfg
from .asset import ObjectBaseCfg

__all__: list[str] = []


@configclass
class SoftbodyVoxelAttributesCfg:
    """Newton tetrahedralization and render-volume binding parameters."""

    triangle_remesh_resolution: int = 8
    """Resolution used to remesh the source surface before tetrahedralization."""

    triangle_simplify_target: int = 0
    """Target face count for the proxy surface; zero disables simplification."""

    simulation_mesh_resolution: int = 8
    """Voxel resolution used to build the tetrahedral simulation mesh."""

    voxel_num_relaxation_iters: int = 5
    """Number of tetrahedral-mesh relaxation iterations."""

    voxel_rel_min_tet_volume: float = 0.05
    """Minimum tetrahedron volume relative to the voxel volume."""

    voxel_surface_dist_ratio: float = 0.2
    """Maximum surface distance expressed as a voxel-size ratio."""

    embedding_impl: str = "dexsim_exact_cpu"
    """DexSim implementation used to bind render vertices to tetrahedra."""


@configclass
class SoftbodyPhysicalAttributesCfg:
    """Newton volumetric and optional surface-element material parameters."""

    youngs: float = 1e6
    """Young's modulus (higher = stiffer)."""

    poissons: float = 0.45
    """Poisson's ratio (higher = closer to incompressible)."""

    elasticity_damping: float = 0.0
    """Volumetric damping coefficient forwarded as Newton ``k_damp``."""

    density: float = 1000.0
    """Volume density in kg/m³."""

    surface_tri_ke: float = 0.0
    """Surface triangle elastic stiffness."""

    surface_tri_ka: float = 0.0
    """Surface triangle area stiffness."""

    surface_tri_kd: float = 0.0
    """Surface triangle damping."""

    surface_tri_drag: float = 0.0
    """Surface aerodynamic drag coefficient."""

    surface_tri_lift: float = 0.0
    """Surface aerodynamic lift coefficient."""

    add_surface_edges: bool = True
    """Whether Newton creates surface bending-edge constraints."""

    surface_edge_ke: float = 0.0
    """Surface bending-edge stiffness."""

    surface_edge_kd: float = 0.0
    """Surface bending-edge damping."""


@configclass
class ClothPhysicalAttributesCfg:
    """Newton cloth triangle, bending-edge, and spring parameters."""

    density: float = 1.0
    """Surface density in kg/m²."""

    tri_ke: float | None = None
    """Triangle elastic stiffness; ``None`` uses the Newton default."""

    tri_ka: float | None = None
    """Triangle area stiffness; ``None`` uses the Newton default."""

    tri_kd: float | None = None
    """Triangle damping; ``None`` uses the Newton default."""

    tri_drag: float | None = None
    """Aerodynamic drag; ``None`` uses the Newton default."""

    tri_lift: float | None = None
    """Aerodynamic lift; ``None`` uses the Newton default."""

    edge_ke: float | None = None
    """Bending-edge stiffness; ``None`` uses the Newton default."""

    edge_kd: float | None = None
    """Bending-edge damping; ``None`` uses the Newton default."""

    add_springs: bool = False
    """Whether Newton creates explicit mesh-edge springs."""

    spring_ke: float | None = None
    """Spring stiffness; ``None`` uses the Newton default."""

    spring_kd: float | None = None
    """Spring damping; ``None`` uses the Newton default."""


@configclass
class DeformableObjectCfg(ObjectBaseCfg):
    """Common configuration contract for one deformable asset.

    Concrete volume and surface configurations author Newton particle-set
    properties. The discriminator is explicit so manager and visualization
    code do not need to infer topology from a mesh or material type.
    """

    deformable_type: Literal["volume", "surface"] = MISSING
    """Physical topology represented by the asset."""

    shape: MeshCfg = MeshCfg()
    """Render and source-mesh configuration."""

    particle_radius: float | None = None
    """Newton particle radius; ``None`` uses the active solver default."""

    particle_flags: int | Sequence[int] | np.ndarray | None = None
    """Newton particle flags, provided as one broadcast value or one value per node.

    Clear the Newton ``ACTIVE`` bit for nodes that will be driven kinematically.
    Per-node arrays must follow the resolved simulation-particle order. For a
    surface deformable, an array-backed
    :class:`~embodichain.lab.sim.shapes.MeshCfg` preserves this order. A volume
    deformable is voxelized into a separate tetrahedral simulation mesh, so its
    particle indices do not correspond to source-mesh vertex indices.
    """

    validate_mesh: bool = False
    """Whether Newton reports source-mesh quality validation warnings."""


@configclass
class VolumeDeformableObjectCfg(DeformableObjectCfg):
    """Configuration for a Newton volume-deformable particle set."""

    deformable_type: Literal["volume"] = "volume"

    voxel_attr: SoftbodyVoxelAttributesCfg = SoftbodyVoxelAttributesCfg()
    """Tetrahedral simulation-mesh voxelization attributes."""

    physical_attr: SoftbodyPhysicalAttributesCfg = SoftbodyPhysicalAttributesCfg()
    """Newton volume-deformable physical attributes."""


@configclass
class SoftObjectCfg(VolumeDeformableObjectCfg):
    """Compatibility name for :class:`VolumeDeformableObjectCfg`."""


@configclass
class SurfaceDeformableObjectCfg(DeformableObjectCfg):
    """Configuration for a Newton surface-deformable particle set."""

    deformable_type: Literal["surface"] = "surface"

    visual_shape: MeshCfg | None = None
    """Optional render mesh driven by the simulation surface.

    When omitted, :attr:`shape` supplies both simulation topology and rendering.
    Use a separately indexed mesh here when the visual asset needs authored UVs,
    seam vertices, or other detail that should not change the simulation mesh.
    """

    visual_binding_mode: Literal["auto", "nearest_vertex"] = "auto"
    """Binding used to drive :attr:`visual_shape` from simulation particles.

    ``"auto"`` uses DexSim's surface embedding. ``"nearest_vertex"`` is useful
    when the render mesh duplicates simulation vertices along texture seams.
    """

    physical_attr: ClothPhysicalAttributesCfg = ClothPhysicalAttributesCfg()
    """Newton surface-deformable physical attributes."""


@configclass
class ClothObjectCfg(SurfaceDeformableObjectCfg):
    """Compatibility name for :class:`SurfaceDeformableObjectCfg`."""
