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

"""Standalone parallel-jaw Robot configurations sharing pure model defaults."""

from __future__ import annotations

# Configclass regenerates inherited constructor annotations in this module.
# These names are needed by static tooling, without runtime import cycles.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from typing import Dict, List, Sequence
    from embodichain.lab.sim.cfg import (
        AssetPhysicsMode,
        RigidBodyPhysicsCfg,
        LinkPhysicsOverrideCfg,
        ArticulationRootPropertiesCfg,
        JointDrivePropertiesCfg,
        URDFCfg,
        RobotWorkspaceCfg,
        UsdSourceCfg,
    )
    from embodichain.lab.sim.motion.solvers import SolverCfg
    from embodichain.lab.sim.sensors import SensorCfg


from copy import deepcopy
from typing import Any

from embodichain.utils import configclass
from .robot import RobotCfg

__all__ = ["ParallelJawGripperCfg"]


@configclass
class ParallelJawGripperCfg(RobotCfg):
    """Native physics and supported generators; callers choose mounting and endpoints."""

    model_id: str | None = None
    """Exact physical model identity, independent of the mounting robot."""
    catalog_id: str | None = None
    """Originating catalog identity; never used by GraspKit for lookup."""
    root_link: str | None = None
    """Native hand root before assembly name mapping."""
    collision_links: tuple[str, ...] = ()
    """Native collision links defining the complete hand surfaces."""
    pad_links: tuple[str, str] | None = None
    """Optional ordered native contact pads."""
    root_from_contact_frame: list[list[float]] | None = None
    """Canonical grasp contact frame expressed in the hand root."""
    min_opening_width: float | None = None
    """Qualified minimum usable full contact opening in metres."""
    max_opening_width: float | None = None
    """Qualified maximum usable full contact opening in metres."""
    grasp_generators: tuple[str, ...] = ()
    """Explicit compatible generator kinds; empty grants no grasp service."""
    joint_commands: dict[str, dict[str, float]] = {}
    """Complete commands keyed by native joint name, including followers."""
    default_root_from_tcp: list[list[float]] | None = None
    """Default tool reference in hand-root coordinates."""
    variant: str | None = None
    """Explicit native model variant; unrelated to the mounting arm side."""

    @classmethod
    def get_gripper_defaults(cls, variant: str | None = None) -> dict[str, object]:
        """Return model-owned primitive defaults without accessing an asset."""
        raise NotImplementedError("A concrete gripper must provide defaults.")

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> ParallelJawGripperCfg:
        """Build an independently controlled hand from the shared model data."""
        from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg

        cfg = cls()
        cfg._build_defaults(init_dict)
        result = merge_robot_cfg(cfg, init_dict)
        return result

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        from embodichain.data import get_data_path
        from embodichain.data.urdf import materialize_urdf_subtree
        from .articulation import JointDrivePropertiesCfg, ArticulationRootPropertiesCfg

        self.variant = (init_dict or {}).get("variant", self.variant)
        from embodichain.robot_registry import _validate_gripper_description

        values = self.get_gripper_defaults(self.variant)
        for key in (
            "model_id",
            "catalog_id",
            "root_link",
            "collision_links",
            "pad_links",
            "root_from_contact_frame",
            "min_opening_width",
            "max_opening_width",
            "grasp_generators",
            "joint_commands",
            "default_root_from_tcp",
        ):
            if key in (init_dict or {}):
                values[key] = deepcopy(init_dict[key])
        defaults = _validate_gripper_description(values)
        for key in (
            "model_id",
            "catalog_id",
            "root_link",
            "collision_links",
            "pad_links",
            "root_from_contact_frame",
            "min_opening_width",
            "max_opening_width",
            "grasp_generators",
        ):
            setattr(self, key, deepcopy(defaults.get(key, getattr(self, key))))
        self.joint_commands = deepcopy(defaults["joint_commands"])
        self.default_root_from_tcp = deepcopy(defaults["default_root_from_tcp"])
        self.uid = defaults["uid"]
        self.control_parts = deepcopy(defaults["control_parts"])
        self.solver_cfg = {}
        self.qpos_limits = deepcopy(defaults["qpos_limits"])
        self.init_qpos = list(self.joint_commands["open"].values())
        self.init_pos = [0.0, 0.0, 0.3]
        self.root_props = ArticulationRootPropertiesCfg(fixed_base=True)
        self.joint_drive_props = JointDrivePropertiesCfg(
            **defaults["joint_drive_props"]
        )
        self.fpath = (init_dict or {}).get("fpath") or get_data_path(defaults["fpath"])
        if not (init_dict or {}).get("fpath") and defaults.get("extract_subtree"):
            self.fpath = str(
                materialize_urdf_subtree(
                    self.fpath,
                    self.root_link,
                    mesh_paths={
                        k: get_data_path(v)
                        for k, v in defaults.get("mesh_paths", {}).items()
                    },
                )
            )

        # The normal assembler supplies a fixed mounting root. This preserves
        # geometryless zero-inertia adapter frames without inventing hand mass.
        from .urdf import URDFCfg

        source = self.fpath
        self.urdf_cfg = URDFCfg(
            components=[{"component_type": "hand", "urdf_path": source}],
            component_prefix=[("hand", None)],
            force_assembly=True,
            base_link_name=f"{self.uid}_mount",
            fname=f"{self.uid}_{self.variant or 'default'}_standalone",
        )
        self.fpath = (init_dict or {}).get("fpath")
