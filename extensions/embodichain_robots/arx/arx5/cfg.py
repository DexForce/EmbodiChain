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

"""ARX5 X5A arm with its two independently actuated parallel fingers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
)
from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["ARX5Cfg"]


@configclass
class ARX5Cfg(RobotCfg):
    """Configure the bundled X5A asset without changing its source joint names.

    The arm has six revolute joints; ``eef`` contains the two independent
    prismatic fingers, with no synthetic mimic relation. Increasing either
    finger coordinate opens that finger. Drives and the tool transform are
    simulation defaults, not hardware calibration or motor specifications.
    The solver's end frame is ``ee_link``; its TCP aligns X with jaw opening
    and Z with approach, as required by the parallel-jaw grasp generator.
    The default ready pose uses an outward elbow and unflipped wrist, with
    self-collision enabled. Task-specific IK branch limits remain caller-owned.
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> ARX5Cfg:
        """Build asset defaults, then apply isolated caller overrides.

        Args:
            init_dict: Robot configuration overrides.

        Returns:
            The configured X5A robot.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    @property
    def _pk_urdf_path(self) -> str:
        return self.fpath or get_data_path("ARX5/X5A.urdf")

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        self.uid = "ARX5"
        if init_dict and init_dict.get("fpath"):
            self.fpath = init_dict["fpath"]
        self.fpath = self._pk_urdf_path
        self.control_parts = {
            "arm": [f"joint{i}" for i in range(1, 7)],
            "eef": ["joint7", "joint8"],
        }
        self.init_qpos = [-2.84, 1.825, 1.743, -1.489, 0.0, -1.266, 0.044, 0.044]
        self.solver_cfg = {
            "arm": SSIKSolverCfg(
                geometry_mode="near_parallel",
                root_link_name="base_link",
                end_link_name="ee_link",
                tcp=[
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.085],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        }
        self.joint_drive_props = JointDrivePropertiesCfg(
            stiffness={"arm": 1e4, "eef": 1e3},
            damping={"arm": 1e3, "eef": 1e2},
            max_effort={"arm": 100.0, "eef": 40.0},
            max_velocity={"arm": 2.0, "eef": 0.1},
        )
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True,
            self_collision_enabled=True,
            min_position_iters=16,
            min_velocity_iters=4,
        )
        self.attrs = RigidBodyPhysicsCfg()

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build the six-DOF flange chain from the simulation asset.

        Args:
            device: Device for kinematics tensors.
            **kwargs: Reserved for the RobotCfg protocol.

        Returns:
            The ``arm`` chain, excluding the separate solver TCP transform.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        solver = self.solver_cfg["arm"]
        return {
            "arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                root_link_name=solver.root_link_name,
                end_link_name=solver.end_link_name,
                device=device,
            )
        }
