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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["MarvinCfg"]

_MARVIN_ASSET_PATH = "Marvin/robot_with_ee.urdf"

_ARM_JOINTS = {
    "left_arm": [
        "SHOULDER_PITCH_L_J1",
        "SHOULDER_ROLL_L_J2",
        "ELBOW_PITCH_L_J3",
        "ELBOW_YAW_L_J4",
        "WRIST_PITCH_L_J5",
        "WRIST_YAW_L_J6",
        "WRIST_ROLL_L_J7",
    ],
    "right_arm": [
        "SHOULDER_PITCH_R_J1",
        "SHOULDER_ROLL_R_J2",
        "ELBOW_PITCH_R_J3",
        "ELBOW_YAW_R_J4",
        "WRIST_PITCH_R_J5",
        "WRIST_YAW_R_J6",
        "WRIST_ROLL_R_J7",
    ],
}

_EEF_JOINTS = {
    "left_eef": ["LEFT_HAND_FINGER_1", "LEFT_HAND_FINGER_2"],
    "right_eef": ["RIGHT_HAND_FINGER_1", "RIGHT_HAND_FINGER_2"],
}

# Exact actuator limits from the supplied URDF, indexed by joint number.
_MAX_EFFORT = {1: 108.0, 2: 108.0, 3: 66.0, 4: 66.0, 5: 18.0, 6: 18.0, 7: 18.0}
_MAX_VELOCITY = 3.1416

# Provisional controller gains. The URDF contains link inertias and actuator
# limits, but no motor/gearbox response or measured closed-loop gains. These
# conservative values should be tuned against the real controller response.
_STIFFNESS = {1: 600.0, 2: 600.0, 3: 400.0, 4: 400.0, 5: 120.0, 6: 120.0, 7: 120.0}
_DAMPING = {1: 50.0, 2: 50.0, 3: 35.0, 4: 35.0, 5: 8.0, 6: 8.0, 7: 8.0}

# Gripper joint limits and dynamics from robot_with_ee.urdf. Finger 2 mimics
# finger 1 and has no independent actuator effort in the source URDF.
_EEF_STIFFNESS = 1e3
_EEF_DAMPING = 8.0
_EEF_MAX_VELOCITY = 0.1
_EEF_FRICTION = 0.2

# Provisional TCP supplied with the initial Marvin configuration. The URDF
# defines the wrist-to-EE transform, but has no geometry from the EE frame to a
# tool center point, so the additional +0.14 m offset cannot yet be verified.
_PROVISIONAL_TCP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.14],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _joint_values(values_by_index: dict[int, float]) -> dict[str, float]:
    """Expand joint-indexed values to the exact Marvin URDF joint names."""
    return {
        joint_name: values_by_index[joint_index]
        for joint_names in _ARM_JOINTS.values()
        for joint_index, joint_name in enumerate(joint_names, start=1)
    }


@configclass
class MarvinCfg(RobotCfg):
    """Configuration for the Marvin dual-arm robot.

    ``urdf_path`` may be an absolute path or a data-root-relative path. The
    default expects the asset at ``Marvin/robot_with_ee.urdf`` under
    ``EMBODICHAIN_DATA_ROOT``.
    """

    urdf_path: str = _MARVIN_ASSET_PATH

    @classmethod
    def from_dict(cls, init_dict: dict) -> MarvinCfg:
        """Create a Marvin configuration and merge caller overrides."""
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate Marvin URDF, control, solver, and physics defaults."""
        init_dict = init_dict or {}
        requested_urdf_path = init_dict.get("urdf_path", self.urdf_path)
        urdf_path = (
            requested_urdf_path
            if os.path.isabs(requested_urdf_path)
            else get_data_path(requested_urdf_path)
        )

        self.uid = "Marvin"
        self.urdf_path = urdf_path
        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "robot",
                    "urdf_path": urdf_path,
                    "transform": np.eye(4),
                }
            ]
        )
        arm_control_parts = {
            part: list(joint_names) for part, joint_names in _ARM_JOINTS.items()
        }
        eef_control_parts = {
            part: list(joint_names) for part, joint_names in _EEF_JOINTS.items()
        }
        self.control_parts = {**arm_control_parts, **eef_control_parts}
        self.solver_cfg = {
            "left_arm": PytorchSolverCfg(
                root_link_name="left_arm_base",
                end_link_name="left_ee",
                tcp=_PROVISIONAL_TCP.copy(),
            ),
            "right_arm": PytorchSolverCfg(
                root_link_name="right_arm_base",
                end_link_name="right_ee",
                tcp=_PROVISIONAL_TCP.copy(),
            ),
        }

        self.min_position_iters = 8
        self.min_velocity_iters = 2
        self.drive_pros = JointDrivePropertiesCfg(
            stiffness={
                **_joint_values(_STIFFNESS),
                **{
                    joint_name: _EEF_STIFFNESS
                    for joint_names in _EEF_JOINTS.values()
                    for joint_name in joint_names
                },
            },
            damping={
                **_joint_values(_DAMPING),
                **{
                    joint_name: _EEF_DAMPING
                    for joint_names in _EEF_JOINTS.values()
                    for joint_name in joint_names
                },
            },
            max_effort={
                **_joint_values(_MAX_EFFORT),
                "LEFT_HAND_FINGER_1": 30.0,
                "LEFT_HAND_FINGER_2": 0.0,
                "RIGHT_HAND_FINGER_1": 30.0,
                "RIGHT_HAND_FINGER_2": 0.0,
            },
            max_velocity={
                **{
                    joint_name: _MAX_VELOCITY
                    for joint_names in _ARM_JOINTS.values()
                    for joint_name in joint_names
                },
                **{
                    joint_name: _EEF_MAX_VELOCITY
                    for joint_names in _EEF_JOINTS.values()
                    for joint_name in joint_names
                },
            },
            friction={
                joint_name: _EEF_FRICTION
                for joint_names in _EEF_JOINTS.values()
                for joint_name in joint_names
            },
        )
        self.attrs = RigidBodyAttributesCfg(
            static_friction=0.95,
            dynamic_friction=0.9,
            contact_offset=0.001,
        )

    @property
    def _pk_urdf_path(self) -> str:
        """Return the same URDF used by simulation for both PK arm chains."""
        return self.urdf_path

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> dict[str, "pk.SerialChain"]:
        """Build left and right 7-DOF serial chains from the Marvin URDF."""
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        left_arm_chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path,
            device=device,
            root_link_name="left_arm_base",
            end_link_name="left_ee",
        )
        right_arm_chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path,
            device=device,
            root_link_name="right_arm_base",
            end_link_name="right_ee",
        )
        return {"left_arm": left_arm_chain, "right_arm": right_arm_chain}


if __name__ == "__main__":
    config = MarvinCfg.from_dict({})
    chains = config.build_pk_serial_chain()
    for part, chain in chains.items():
        assert len(chain.get_joint_parameter_names()) == len(config.control_parts[part])
    print("Marvin configuration and PK chains are valid.")
