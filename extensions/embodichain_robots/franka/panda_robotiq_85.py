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

"""Original Panda arm and explicitly calibrated Robotiq 2F-85 composition."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    ArticulationRootPropertiesCfg,
)
from embodichain.lab.sim.motion.solvers import PinkSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass
from ._panda_robotiq_physics import make_inertial_overlays

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FrankaPandaRobotiq85Cfg", "robotiq85_joint_positions"]

_ARM = [f"Joint{i}" for i in range(1, 8)]
_GRIPPER = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
]
_SIGNS = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])


def robotiq85_joint_positions(closed_fraction: np.ndarray) -> np.ndarray:
    """Convert measured normalized aperture to the selected 2F-85 joint layout.

    DROID uses ``width = 0.085 * (1 - closed_fraction)``. The standard Robotiq
    linkage has ``width = 0.010 + 0.1143*sin(theta_zero - theta)``; calibration
    uses the open endpoint and clips only to the URDF's 0.8 rad closed stop.
    Returned columns follow the preset's explicit ``eef`` control part. This
    converts observed aperture independently of any closing command.

    Args:
        closed_fraction: Finite normalized closed fractions in [0, 1].

    Returns:
        Float64 joint positions with shape ``(*closed_fraction.shape, 6)``.
    """
    fraction = np.asarray(closed_fraction, dtype=np.float64)
    if not np.isfinite(fraction).all() or np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("Robotiq closed fraction must lie in [0, 1]")
    zero = np.arcsin((0.085 - 0.010) / 0.1143)
    angle = np.minimum(zero - np.arcsin((0.085 * (1 - fraction) - 0.010) / 0.1143), 0.8)
    return angle[..., None] * _SIGNS


@configclass
class FrankaPandaRobotiq85Cfg(RobotCfg):
    """Original Panda with the six-coordinate Robotiq 2F-85 mimic mechanism.

    Mounting and TCP are explicit model parameters. A data adapter must verify
    them against its recorded setup before claiming a qualified real transfer.
    The arm archive is pinned separately from the existing FR3-named assets.
    """

    gripper_mount_z_m: float = 0.0
    gripper_mount_yaw_rad: float = 0.0
    tcp_offset_m: float = 0.15

    @classmethod
    def from_dict(cls, init_dict: dict) -> FrankaPandaRobotiq85Cfg:
        """Decode model parameters and ordinary RobotCfg overrides.

        Args:
            init_dict: Serialized robot configuration.

        Returns:
            Configured Panda/Robotiq embodiment.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        values = init_dict or {}
        for name in ("gripper_mount_z_m", "gripper_mount_yaw_rad", "tcp_offset_m"):
            value = values.get(name, getattr(self, name))
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite explicit model parameter")
            setattr(self, name, float(value))
        if self.tcp_offset_m <= 0 or self.gripper_mount_z_m < 0:
            raise ValueError("gripper mounting/TCP distances are invalid")
        self.uid = "FrankaPandaRobotiq85"
        mount = np.eye(4)
        angle = self.gripper_mount_yaw_rad
        mount[:3, :3] = [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
        mount[2, 3] = self.gripper_mount_z_m
        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": self._pk_urdf_path,
                    "transform": np.eye(4),
                },
                {
                    "component_type": "hand",
                    "urdf_path": get_data_path(
                        "Robotiq/robotiq_arg2f_85/robotiq_arg2f_85.urdf"
                    ),
                    "transform": mount,
                },
            ],
            fname="panda_robotiq85_"
            + hashlib.sha256(
                mount.astype("<f8").tobytes()
                + np.asarray([self.tcp_offset_m], dtype="<f8").tobytes()
            ).hexdigest()[:12],
        )
        self.control_parts = {"arm": list(_ARM), "eef": list(_GRIPPER)}
        # Legacy Robotiq CAD lists positive follower limits even for a -1
        # mimic. The effective limits follow the one independent drive.
        self.qpos_limits = {
            name: ([-0.8, 0.0] if sign < 0 else [0.0, 0.8])
            for name, sign in zip(_GRIPPER, _SIGNS)
        }
        tcp = mount.copy()
        tcp[:3, 3] += tcp[:3, :3] @ np.array([0.0, 0.0, self.tcp_offset_m])
        self.solver_cfg = {
            "arm": PinkSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=tcp.tolist(),
                pos_eps=1e-5,
                rot_eps=1e-4,
                max_iterations=100,
                show_ik_warnings=False,
            )
        }
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True, min_position_iters=64, min_velocity_iters=16
        )
        self.link_attrs = make_inertial_overlays()
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"Joint[1-7]": 1e5, ".*finger_joint|.*knuckle_joint": 200.0},
            damping={"Joint[1-7]": 1e4, ".*finger_joint|.*knuckle_joint": 5.0},
            max_effort={
                "Joint[1-4]": 87.0,
                "Joint[5-7]": 12.0,
                ".*finger_joint|.*knuckle_joint": 0.25,
            },
            max_velocity={
                "Joint[1-4]": 2.175,
                "Joint[5-7]": 2.61,
                ".*finger_joint|.*knuckle_joint": 2.0,
            },
        )
        self.init_qpos = [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741, *([0.0] * 6)]

    @property
    def _pk_urdf_path(self) -> str:
        return get_data_path("FrankaPandaArm/Panda/PandaExactV1.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: object
    ) -> dict[str, pk.SerialChain]:
        """Build the same seven-joint arm chain used by the simulator.

        Args:
            device: Torch device for kinematics.
            **kwargs: Compatibility arguments for the robot configuration port.

        Returns:
            The named arm serial chain; solver-owned TCP is applied separately.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        return {
            "arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                end_link_name="ee_link",
                root_link_name="base_link",
            )
        }
