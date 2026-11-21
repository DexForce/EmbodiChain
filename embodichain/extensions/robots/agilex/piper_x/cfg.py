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

"""Six-axis AgileX PiPER-X arm with a flange-frame TCP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    from pytorch_kinematics import SerialChain

__all__ = ["AgilexPiperXCfg"]

_ROOT_LINK = "base_link"
_FLANGE_LINK = "flange_link"
_ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]


@configclass
class AgilexPiperXCfg(RobotCfg):
    """Fixed-base PiPER-X with six arm joints and an optional external tool."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> AgilexPiperXCfg:
        """Build the stock arm and merge caller-supplied overrides."""
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        self.uid = "AgilexPiperX"
        self.urdf_cfg = URDFCfg(
            base_link_name="piper_x_assembly_root",
            components=[dict(component_type="arm", urdf_path=self._pk_urdf_path)],
        )
        self.control_parts = {"arm": list(_ARM_JOINTS)}
        self.init_qpos = [0.0, 0.72, -1.25, 0.0, 0.72, 0.0]
        # J6 is ±170° in the asset and ±165° in the PiPER-X manual.
        self.qpos_limits = {"joint6": (-math.radians(165), math.radians(165))}
        self.root_props = ArticulationRootPropertiesCfg(fixed_base=True)
        self.attrs = RigidBodyPhysicsCfg()
        # Position gains are simulation settings; cap speed at the manual's
        # per-axis ratings and explicitly apply the pinned URDF's effort caps.
        # Default imports undriven joints with mode NONE and zero drive effort.
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"joint[1-3]": 300.0, "joint[4-6]": 100.0},
            damping={"joint[1-3]": 30.0, "joint[4-6]": 10.0},
            max_effort={"arm": 100.0},
            max_velocity={
                name: math.radians(speed)
                for name, speed in zip(_ARM_JOINTS, (180, 195, 195, 225, 225, 225))
            },
        )
        self.solver_cfg = {
            "arm": PytorchSolverCfg(
                root_link_name=_ROOT_LINK, end_link_name=_FLANGE_LINK
            )
        }

    @property
    def _pk_urdf_path(self) -> str:
        if self.urdf_cfg is not None:
            return self.urdf_cfg.components["arm"]["urdf_path"]
        return get_data_path("AgilexPiperX/piper_x.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, SerialChain]:
        """Build the six-joint chain from the simulation arm asset."""
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        return {
            "arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                root_link_name=_ROOT_LINK,
                end_link_name=_FLANGE_LINK,
                device=device,
            )
        }
