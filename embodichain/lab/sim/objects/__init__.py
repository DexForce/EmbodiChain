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

"""Scene-object classes spawned into the ``SimulationManager``.

Covers lights, render-only objects, rigid bodies (and groups), articulations, robots, deformables (soft/cloth), gizmos, and rigid constraints; every object derives from ``BatchEntity``.
"""

from __future__ import annotations

from ..common import BatchEntity
from .cable import CableObject, CableObjectData
from .rigid_object import CollisionShapeDesc, RigidObject, RigidBodyData, RigidObjectCfg
from .visual_object import VisualObject, VisualObjectCfg
from .rigid_object_group import (
    RigidObjectGroup,
    RigidBodyGroupData,
    RigidObjectGroupCfg,
)
from .fluid import FluidObject, FluidObjectData
from .deformable import (
    DeformableObject,
    DeformableObjectData,
    SurfaceDeformableObject,
    VolumeDeformableObject,
)

# Compatibility aliases for the former native soft-body and cloth facades.
SoftObject = VolumeDeformableObject
ClothObject = SurfaceDeformableObject
SoftBodyData = DeformableObjectData
ClothBodyData = DeformableObjectData
from ..cfg import (
    DeformableObjectCfg,
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
)
from .articulation import (
    Articulation,
    ArticulationData,
    ArticulationJointKinematics,
    ArticulationCfg,
)
from .robot import Robot, RobotCfg, RobotWorkspaceCfg
from .light import Light, LightCfg
from .gizmo import Gizmo, GizmoCfg, create_robot_ik_gizmo_controller
from .constraint import RigidConstraint


from dexsim.engine import RenderBody
import numpy as np


def set_projective_uv(self: RenderBody, proj_direct: np.ndarray | None = None) -> None:
    """Apply projective UVs through the shared native-binding adapter.

    Args:
        proj_direct: Projection direction; omission selects automatic projection.
    """
    from ..utility.dynamic_pybind import set_projective_uv as apply_projective_uv

    apply_projective_uv(self, proj_direct)
