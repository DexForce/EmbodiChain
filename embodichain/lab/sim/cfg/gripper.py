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
    from embodichain.toolkits.graspkit.gripper_models import ParallelJawGripperModelCfg


from copy import deepcopy
from typing import Any

from embodichain.utils import configclass
from embodichain.toolkits.graspkit.gripper_models import ParallelJawGripperModelCfg
from .robot import RobotCfg

__all__ = ["ParallelJawGripperCfg"]


@configclass
class ParallelJawGripperCfg(RobotCfg):
    """A native hand; mounting and logical capabilities belong to its caller."""

    grasp_cfg: ParallelJawGripperModelCfg | None = None
    """Model-owned geometry and contact-frame declaration."""
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
        for key in ("grasp_cfg", "joint_commands", "default_root_from_tcp", "variant"):
            if key in init_dict:
                setattr(result, key, deepcopy(init_dict[key]))
        if isinstance(result.grasp_cfg, dict):
            result.grasp_cfg = ParallelJawGripperModelCfg(**result.grasp_cfg)
        return result

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        from embodichain.data import get_data_path
        from embodichain.data.urdf import materialize_urdf_subtree
        from .articulation import JointDrivePropertiesCfg, ArticulationRootPropertiesCfg

        self.variant = (init_dict or {}).get("variant", self.variant)
        defaults = self.get_gripper_defaults(self.variant)
        self.grasp_cfg = ParallelJawGripperModelCfg(**defaults["grasp_cfg"])
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
                    self.grasp_cfg.root_link,
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
