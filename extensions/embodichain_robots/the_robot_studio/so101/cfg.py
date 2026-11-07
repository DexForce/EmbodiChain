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

"""SO101 follower defaults based on The Robot Studio's official simulation model."""

from __future__ import annotations

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
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["SO101FollowerCfg"]

_ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


@configclass
class SO101FollowerCfg(RobotCfg):
    """Configure the five-axis SO101 follower and its actuated jaw.

    Simulation and serial chains share the official ``so101_new_calib.urdf``.
    Motor gains, friction, armature and effort follow the pinned upstream
    ``joints_properties.xml``; these are simulation references, not identified
    hardware parameters. Link mass, COM and inertia remain asset-owned.

    The default numerical IK constrains position because the five-axis arm
    cannot realize an arbitrary six-dimensional end-effector pose.
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> SO101FollowerCfg:
        """Build defaults and merge standard RobotCfg overrides.

        Args:
            init_dict: Robot fields overriding the official defaults.

        Returns:
            An isolated SO101 follower configuration.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        self.uid = "SO101Follower"
        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": get_data_path(
                        "SO101FollowerAssets/so101_new_calib.urdf"
                    ),
                }
            ]
        )
        self.control_parts = {"arm": list(_ARM_JOINTS), "eef": ["gripper"]}
        self.solver_cfg = {
            "arm": PytorchSolverCfg(
                root_link_name="base_link",
                end_link_name="gripper_frame_link",
                is_only_position_constraint=True,
            )
        }
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True, self_collision_enabled=False
        )
        # Upstream STS3215 simulation reference, not servo stall specifications.
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=17.8,
            damping=0.60,
            friction=0.052,
            armature=0.028,
            max_effort=3.35,
            max_velocity=10.0,
        )
        self.attrs = RigidBodyPhysicsCfg()
        self.init_qpos = [0.0] * 6

    @property
    def _pk_urdf_path(self) -> str:
        """Use the same component source as the simulation assembly."""
        return self.urdf_cfg.components["arm"]["urdf_path"]

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build the arm chain and the jaw chain from their physical roots.

        Args:
            device: Torch device for chain tensors.
            **kwargs: Reserved for the common RobotCfg chain-builder interface.

        Returns:
            Five-DOF arm and one-DOF jaw chains keyed by control part.
        """
        return {
            part: create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                root_link_name=root,
                end_link_name=end,
            )
            for part, root, end in (
                ("arm", "base_link", "gripper_frame_link"),
                ("eef", "gripper_link", "moving_jaw_so101_v1_link"),
            )
        }
