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
from typing import Literal

from dexsim.types import (
    ClothBodyAttr,
    SoftBodyAttr,
    SoftBodyMaterialModel,
    VoxelConfig,
)

from embodichain.utils import configclass

from ..shapes import MeshCfg
from .asset import ObjectBaseCfg


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
