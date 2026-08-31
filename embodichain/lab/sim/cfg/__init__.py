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

"""Public simulation-configuration facade.

The implementation is split by domain while this package preserves the
historical ``embodichain.lab.sim.cfg`` import surface.
"""

from __future__ import annotations

from typing import Literal

from embodichain.data import get_data_path

from .._legacy_cfg import RigidBodyAttributesCfg, RigidBodyAttributesOverrideCfg
from ..shapes import MeshCfg, ShapeCfg
from ..workspace.cfg import RobotWorkspaceCfg
from .articulation import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    NewtonJointDrivePropertiesCfg,
    _normalize_joint_target_mode,
    _raise_removed_articulation_cfg_fields,
    link_attrs_from_dict,
)
from .asset import AssetPhysicsMode, ObjectBaseCfg, _resolve_asset_physics_mode
from .deformable import (
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    DeformableObjectCfg,
    SoftObjectCfg,
    SoftbodyPhysicalAttributesCfg,
    SoftbodyVoxelAttributesCfg,
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
)
from .rigid import (
    CollisionPropertiesCfg,
    DefaultCollisionPropertiesCfg,
    DefaultRigidBodyPhysicsCfg,
    DefaultRigidBodyMaterialCfg,
    DefaultRigidBodyPropertiesCfg,
    MassPropertiesCfg,
    MeshCollisionPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonMeshCollisionPropertiesCfg,
    NewtonRigidBodyPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    NewtonRigidBodyPropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidBodyPropertiesCfg,
)
from .rigid_object import RigidObjectCfg, RigidObjectGroupCfg
from .scene import LightCfg, RigidConstraintCfg
from .simulation import (
    DefaultPhysicsCfg,
    GPUMemoryCfg,
    NewtonCollisionPipelineCfg,
    NewtonPhysicsCfg,
    PhysicsBackendCfg,
    PhysicsCfg,
    RenderCfg,
    physics_backend_from_cfg,
    physics_cfg_for_backend,
    validate_physics_cfg,
)
from .urdf import URDFCfg
from .viewer import MarkerCfg, WindowCameraPoseCfg, WindowRecordCfg

# The renderer selection code intentionally mutates this package-level value.
DEFAULT_RENDERER: Literal["auto", "hybrid", "fast-rt", "rt"] = "auto"

# Robot imports are kept last because SolverCfg discovery imports simulation
# modules that themselves rely on the public facade above.
from .robot import RobotCfg, RobotPresetCfg  # noqa: E402

__all__ = [
    "DEFAULT_RENDERER",
    "AssetPhysicsMode",
    "RenderCfg",
    "GPUMemoryCfg",
    "PhysicsBackendCfg",
    "PhysicsCfg",
    "DefaultPhysicsCfg",
    "NewtonCollisionPipelineCfg",
    "NewtonPhysicsCfg",
    "physics_cfg_for_backend",
    "physics_backend_from_cfg",
    "validate_physics_cfg",
    "MarkerCfg",
    "WindowRecordCfg",
    "WindowCameraPoseCfg",
    "ShapeCfg",
    "MeshCfg",
    "MassPropertiesCfg",
    "RigidBodyPropertiesCfg",
    "DefaultRigidBodyPropertiesCfg",
    "NewtonRigidBodyPropertiesCfg",
    "CollisionPropertiesCfg",
    "DefaultCollisionPropertiesCfg",
    "NewtonCollisionPropertiesCfg",
    "MeshCollisionPropertiesCfg",
    "NewtonMeshCollisionPropertiesCfg",
    "RigidBodyMaterialCfg",
    "DefaultRigidBodyMaterialCfg",
    "NewtonRigidBodyMaterialCfg",
    "DefaultRigidBodyPhysicsCfg",
    "NewtonRigidBodyPhysicsCfg",
    "RigidBodyPhysicsCfg",
    "RigidBodyAttributesCfg",
    "RigidBodyAttributesOverrideCfg",
    "ObjectBaseCfg",
    "LightCfg",
    "RigidObjectCfg",
    "DeformableObjectCfg",
    "VolumeDeformableObjectCfg",
    "SoftObjectCfg",
    "SurfaceDeformableObjectCfg",
    "ClothObjectCfg",
    "RigidObjectGroupCfg",
    "RigidConstraintCfg",
    "SoftbodyVoxelAttributesCfg",
    "SoftbodyPhysicalAttributesCfg",
    "ClothPhysicalAttributesCfg",
    "ArticulationRootPropertiesCfg",
    "LinkPhysicsOverrideCfg",
    "link_attrs_from_dict",
    "JointDrivePropertiesCfg",
    "NewtonJointDrivePropertiesCfg",
    "ArticulationCfg",
    "URDFCfg",
    "RobotCfg",
    "RobotPresetCfg",
    "RobotWorkspaceCfg",
    "get_data_path",
]
