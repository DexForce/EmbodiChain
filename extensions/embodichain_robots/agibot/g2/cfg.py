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

"""AgiBot G2 with official CRS arms and OmniPicker grippers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import re
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

__all__ = ["AgibotG2Cfg"]


@configclass
class AgibotG2Cfg(RobotCfg):
    """Configure the full wheeled G2 with two seven-axis CRS arms.

    Native joint names, limits, mass and COM come from the pinned official
    model. Two nonphysical source inertia tensors have documented minimal
    repairs in the asset lock. Each OmniPicker exposes eight joints, including
    seven mimic followers, and one independent command in [-pi/4, 0]
    (open to closed).
    The tree uses upstream mimic approximations instead of GenieSim-specific
    closed-loop constraints. The root is fixed by default for manipulation;
    set ``root_props.fixed_base=False`` for a free mobile base. Steering uses
    position targets and wheel rotation uses velocity targets.

    Arm FK/IK runs from ``arm_base_link`` to the corresponding
    ``gripper_l_base_link`` / ``gripper_r_base_link``. These are mounting
    frames, with an identity TCP; configure a tool offset for a grasp target.
    Strict Warp SSIK derives the spherical shoulder/wrist and 18 mm elbow
    offset from the original chain, with seed-local redundancy search on
    CPU/CUDA. It does not replace the chain with a zero-offset approximation.
    Mounted camera geometry is retained without inventing optical calibration.
    The neutral startup pose has all source joint coordinates at zero.
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> AgibotG2Cfg:
        """Build model defaults and apply isolated RobotCfg overrides.

        Args:
            init_dict: Robot fields overriding the preset.

        Returns:
            The configured G2 robot.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        cfg._prepare_effort_overrides(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _prepare_effort_overrides(self, init_dict: dict[str, Any]) -> None:
        """Let explicit caller rules replace the selected model effort caps."""
        overrides = init_dict.get("joint_drive_props")
        if not isinstance(overrides, dict):
            return
        provided = overrides.get("max_effort")
        if not isinstance(provided, dict):
            return
        parts = dict(self.control_parts)
        if isinstance(init_dict.get("control_parts"), dict):
            parts.update(init_dict["control_parts"])
        joint_names = [name for names in self.control_parts.values() for name in names]
        shadowed = {
            name
            for name in joint_names
            if any(
                re.fullmatch(expression, name)
                for pattern in provided
                for expression in parts.get(pattern, [pattern])
            )
        }
        remaining = {}
        for pattern, value in self.joint_drive_props.max_effort.items():
            if pattern in provided:
                continue
            matched = {
                name
                for name in joint_names
                if any(
                    re.fullmatch(expression, name)
                    for expression in parts.get(pattern, [pattern])
                )
            }
            if shadowed.isdisjoint(matched):
                remaining[pattern] = value
            else:
                remaining.update(
                    {name: value for name in matched if name not in shadowed}
                )
        self.joint_drive_props.max_effort = remaining

    @property
    def _pk_urdf_path(self) -> str:
        return self.fpath or get_data_path("AgibotG2/g2.urdf")

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        self.uid = "AgibotG2"
        if init_dict and init_dict.get("fpath"):
            self.fpath = init_dict["fpath"]
        self.fpath = self._pk_urdf_path
        self.control_parts = {
            "torso": [f"idx0{i}_body_joint{i}" for i in range(1, 6)],
            "head": [f"idx{10+i}_head_joint{i}" for i in range(1, 4)],
            "chassis_steering": [
                f"idx{index}1_chassis_{wheel}_joint1"
                for index, wheel in (
                    (11, "lwheel_front"),
                    (13, "rwheel_front"),
                    (14, "rwheel_rear"),
                    (12, "lwheel_rear"),
                )
            ],
            "chassis_wheels": [
                f"idx{index}2_chassis_{wheel}_joint2"
                for index, wheel in (
                    (11, "lwheel_front"),
                    (13, "rwheel_front"),
                    (14, "rwheel_rear"),
                    (12, "lwheel_rear"),
                )
            ],
        }
        for side, label, offset in (("left", "l", 20), ("right", "r", 60)):
            self.control_parts[f"{side}_arm"] = [
                f"idx{offset+i}_arm_{label}_joint{i}" for i in range(1, 8)
            ]
            self.control_parts[f"{side}_eef"] = [
                f"idx{offset+index}_gripper_{label}_{name}"
                for index, name in (
                    (11, "inner_joint1"),
                    (12, "inner_joint3"),
                    (13, "inner_joint4"),
                    (19, "inner_joint0"),
                    (21, "outer_joint1"),
                    (22, "outer_joint3"),
                    (23, "outer_joint4"),
                    (29, "outer_joint0"),
                )
            ]
        self.solver_cfg = {
            f"{side}_arm": SSIKSolverCfg(
                backend="warp",
                srs_search="seed_local",
                root_link_name="arm_base_link",
                end_link_name=f"gripper_{label}_base_link",
            )
            for side, label in (("left", "l"), ("right", "r"))
        }
        # Source-authored torso/head/arm/leader gains. Followers are constrained
        # by mimic lowering and must not receive competing position drives.
        stiffness = {
            "torso": 1e5,
            "head": 1e3,
            "chassis_steering": 1e3,
            "chassis_wheels": 0.0,
            "left_eef": 0.0,
            "right_eef": 0.0,
        }
        damping = {
            "torso": 1e3,
            "head": 20.0,
            "chassis_steering": 20.0,
            "chassis_wheels": 10.0,
            "left_eef": 0.0,
            "right_eef": 0.0,
        }
        modes = {part: "position" for part in self.control_parts}
        modes.update(chassis_wheels="velocity", left_eef="none", right_eef="none")
        # Default imports these joints without drives and reports zero effort.
        # Activating a drive therefore needs the pinned source effort budget;
        # an omitted cap preserves that zero rather than creating an actuator.
        max_effort = {
            "torso": 50.0,
            "head": 50.0,
            "chassis_steering": 15.0,
            "chassis_wheels": 8.0,
            "left_eef": 10.0,
            "right_eef": 10.0,
        }
        for side in ("left", "right"):
            for joint, kp, kd, effort in zip(
                self.control_parts[f"{side}_arm"],
                (40000.0, 40000.0, 15000.0, 15000.0, 15000.0, 8000.0, 8000.0),
                (600.0, 600.0, 220.0, 220.0, 220.0, 100.0, 100.0),
                (108.0, 108.0, 35.0, 35.0, 35.0, 18.0, 18.0),
            ):
                stiffness[joint], damping[joint] = kp, kd
                max_effort[joint] = effort
            leader = self.control_parts[f"{side}_eef"][0]
            stiffness[leader], damping[leader], modes[leader] = 1000.0, 10.0, "position"
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            target_mode=modes,
            stiffness=stiffness,
            damping=damping,
            max_effort=max_effort,
        )
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True,
            self_collision_enabled=False,
            min_position_iters=16,
            min_velocity_iters=4,
        )
        self.attrs = RigidBodyPhysicsCfg()
        # Source wheel centers are at Z=0.03 m with 0.07 m collision radii.
        # Raise the fixed root 4 cm plus 1 mm clearance above the default ground.
        self.init_pos = [0.0, 0.0, 0.041]
        self.init_qpos = [0.0] * 46

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build both arm chains from the same URDF used by simulation.

        Args:
            device: Device for kinematics tensors.
            **kwargs: Reserved for the RobotCfg protocol.

        Returns:
            Seven-joint ``left_arm`` and ``right_arm`` serial chains.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        return {
            part: create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                root_link_name=solver.root_link_name,
                end_link_name=solver.end_link_name,
                device=device,
            )
            for part, solver in self.solver_cfg.items()
        }
