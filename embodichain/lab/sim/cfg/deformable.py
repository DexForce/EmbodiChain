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

from collections.abc import Mapping
from dataclasses import MISSING
from typing import Any
from typing import Literal, Sequence

import numpy as np

from embodichain.utils import configclass

from ..shapes import MeshCfg
from .asset import ObjectBaseCfg

__all__: list[str] = []


@configclass
class VolumeDeformableMeshingCfg:
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
class SurfaceElementPropertiesCfg:
    """Newton surface triangle, bending, and aerodynamic properties.

    ``None`` preserves Newton's cloth defaults; volume objects resolve it to
    zero, matching their disabled-by-default surface forces.
    """

    tri_ke: float | None = None
    """Triangle elastic stiffness."""

    tri_ka: float | None = None
    """Triangle area stiffness."""

    tri_kd: float | None = None
    """Triangle damping."""

    tri_drag: float | None = None
    """Aerodynamic drag coefficient."""

    tri_lift: float | None = None
    """Aerodynamic lift coefficient."""

    edge_ke: float | None = None
    """Bending-edge stiffness."""

    edge_kd: float | None = None
    """Bending-edge damping."""


_SURFACE_FIELDS = (
    "tri_ke",
    "tri_ka",
    "tri_kd",
    "tri_drag",
    "tri_lift",
    "edge_ke",
    "edge_kd",
)


@configclass
class VolumeDeformablePhysicsCfg:
    """Volume density, elasticity, and optional Newton surface constraints."""

    youngs: float = 1e6
    """Young's modulus [Pa]; higher values make the volume stiffer."""

    poissons: float = 0.45
    """Poisson's ratio; values approaching 0.5 resist volume change."""

    elasticity_damping: float = 0.0
    """Volumetric damping coefficient forwarded as Newton ``k_damp``."""

    density: float = 1000.0
    """Volume density [kg/m³]."""

    surface_props: SurfaceElementPropertiesCfg = SurfaceElementPropertiesCfg(
        **dict.fromkeys(_SURFACE_FIELDS, 0.0)
    )
    """Optional surface forces; all coefficients default to zero."""

    add_surface_edges: bool = True
    """Whether Newton creates surface bending-edge constraints."""

    def __post_init__(self) -> None:
        if isinstance(self.surface_props, Mapping):
            self.surface_props = SurfaceElementPropertiesCfg(**self.surface_props)


@configclass
class SurfaceDeformablePhysicsCfg:
    """Surface density, Newton surface elements, and optional mesh springs."""

    density: float = 1.0
    """Surface density [kg/m²]."""

    surface_props: SurfaceElementPropertiesCfg = SurfaceElementPropertiesCfg()
    """Triangle, bending, and aerodynamic overrides; None uses Newton defaults."""

    add_springs: bool = False
    """Whether Newton creates explicit mesh-edge springs."""

    spring_ke: float | None = None
    """Spring stiffness; ``None`` uses the Newton default."""

    spring_kd: float | None = None
    """Spring damping; ``None`` uses the Newton default."""

    def __post_init__(self) -> None:
        if isinstance(self.surface_props, Mapping):
            self.surface_props = SurfaceElementPropertiesCfg(**self.surface_props)


def _mesh_cfg_from_dict(data: Mapping[str, Any]) -> MeshCfg:
    cfg = MeshCfg.from_dict({"shape_type": "Mesh", **data})
    if not isinstance(cfg, MeshCfg):
        raise TypeError("Deformable shape must be a MeshCfg.")
    return cfg


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

    def __post_init__(self) -> None:
        if isinstance(self.shape, Mapping):
            self.shape = _mesh_cfg_from_dict(self.shape)
        visual_shape = getattr(self, "visual_shape", None)
        if isinstance(visual_shape, Mapping):
            self.visual_shape = _mesh_cfg_from_dict(visual_shape)

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> DeformableObjectCfg:
        """Parse nested deformable configs using the current schema.

        Args:
            init_dict: Configuration fields from a dictionary or YAML loader.

        Returns:
            A typed configuration with the base asset pose conventions applied.
        """
        cfg = cls(**dict(init_dict))
        pose = ObjectBaseCfg.from_dict(
            {
                "init_pos": cfg.init_pos,
                "init_rot": cfg.init_rot,
                "init_local_pose": cfg.init_local_pose,
            }
        )
        cfg.init_pos, cfg.init_rot = pose.init_pos, pose.init_rot
        cfg.init_local_pose = pose.init_local_pose
        return cfg


@configclass
class VolumeDeformableObjectCfg(DeformableObjectCfg):
    """Configuration for a Newton volume-deformable particle set."""

    deformable_type: Literal["volume"] = "volume"

    meshing: VolumeDeformableMeshingCfg = VolumeDeformableMeshingCfg()
    """Tetrahedral simulation-mesh voxelization attributes."""

    attrs: VolumeDeformablePhysicsCfg = VolumeDeformablePhysicsCfg()
    """Newton volume-deformable physical attributes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.attrs, Mapping):
            self.attrs = VolumeDeformablePhysicsCfg(**self.attrs)
        if isinstance(self.meshing, Mapping):
            self.meshing = VolumeDeformableMeshingCfg(**self.meshing)


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

    attrs: SurfaceDeformablePhysicsCfg = SurfaceDeformablePhysicsCfg()
    """Newton surface-deformable physical attributes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.attrs, Mapping):
            self.attrs = SurfaceDeformablePhysicsCfg(**self.attrs)
