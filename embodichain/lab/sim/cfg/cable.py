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

"""Cable authoring configuration for the Newton DexUni backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING
from typing import Any, Literal

import numpy as np

from embodichain.utils import configclass

from ..material import VisualMaterialCfg
from ..shapes import MeshCfg
from .asset import ObjectBaseCfg
from .rigid import CollisionPropertiesCfg, RigidBodyMaterialCfg, RigidBodyPhysicsCfg

__all__ = ["CablePhysicsCfg", "CableAttachmentCfg", "CableObjectCfg"]


@configclass
class CablePhysicsCfg:
    """Uniform per-connection stretch, shear, bend and twist coefficients.

    Linear stiffness/damping use N/m and N s/m; angular coefficients use
    N m/rad and N m s/rad. ``None`` preserves DexSim's solver defaults and
    is distinct from an explicit zero. Dahl options remain scene-owned in
    ``NewtonPhysicsCfg.solver_cfg``.
    """

    density: float = 1000.0
    """Volume density [kg/m³]."""
    stretch_stiffness: float | None = None
    stretch_damping: float | None = None
    shear_stiffness: float | None = None
    shear_damping: float | None = None
    bend_stiffness: float | None = None
    bend_damping: float | None = None
    twist_stiffness: float | None = None
    twist_damping: float | None = None
    collision_props: CollisionPropertiesCfg | None = None
    """Existing portable or Newton contact-envelope configuration."""
    material_props: RigidBodyMaterialCfg | None = None
    """Existing contact-material configuration for segment capsules."""

    def __post_init__(self) -> None:
        contact = RigidBodyPhysicsCfg.from_dict(
            {
                "collision_props": self.collision_props,
                "material_props": self.material_props,
            }
        )
        self.collision_props = contact.collision_props
        self.material_props = contact.material_props


@configclass
class CableAttachmentCfg:
    """Permanent endpoint weld to a sibling rigid object or articulation link.

    The endpoint must coincide with ``position`` in the target body frame at
    preparation. This creation-time configuration does not provide runtime
    grasping or detachment.
    """

    target: str = MISSING
    """Target simulation UID in the same environment."""
    position: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0)
    """Anchor position in the target body frame [m]."""
    link_name: str | None = None
    """Articulation link name; omit for a rigid object."""


@configclass
class CableObjectCfg(ObjectBaseCfg):
    """One cable replicated through the existing Spawn scene lifecycle.

    Points are ordered endpoints in object-local metres. Physical topology
    and material changes require creating a new simulation. Runtime segment
    teleports and whole-environment episode reset use the public Scene API.
    """

    points: Sequence[Sequence[float]] | np.ndarray = MISSING
    """Finite ``(N, 3)`` centreline endpoints, with ``N >= 2``."""
    radius: float = MISSING
    """Positive capsule/tube radius [m]."""
    segment_quaternions: Sequence[Sequence[float]] | np.ndarray | None = None
    """Optional local material frames ``(N - 1, 4)`` in **xyzw** order.

    Local +Z follows each segment. Rotation about +Z specifies material twist.
    ``None`` uses DexSim's parallel-transport frames.
    """
    closed: bool = False
    fixed_segments: Sequence[int] = ()
    attrs: CablePhysicsCfg = CablePhysicsCfg()
    collision_filter_span: int = 1
    start_attachment: CableAttachmentCfg | None = None
    end_attachment: CableAttachmentCfg | None = None
    render_mode: Literal["tube", "capsules"] = "tube"
    visual_material: VisualMaterialCfg | None = None
    """Material of generated tube/capsule visuals."""
    visual_shape: MeshCfg | None = None
    """Optional custom visual mesh; its own ``visual_material`` owns styling."""

    def __post_init__(self) -> None:
        if isinstance(self.attrs, Mapping):
            self.attrs = CablePhysicsCfg(**self.attrs)
        for name in ("start_attachment", "end_attachment"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                setattr(self, name, CableAttachmentCfg(**value))
        if isinstance(self.visual_material, Mapping):
            self.visual_material = VisualMaterialCfg(**self.visual_material)
        if isinstance(self.visual_shape, Mapping):
            self.visual_shape = MeshCfg.from_dict(
                {"shape_type": "Mesh", **self.visual_shape}
            )

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> CableObjectCfg:
        """Parse typed nested configuration and the common asset pose.

        Args:
            init_dict: Cable configuration from YAML/JSON or ``to_dict()``.

        Returns:
            A cable config with typed physics, attachments and visuals.
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
