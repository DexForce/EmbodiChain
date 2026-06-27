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

import enum
import json
import numpy as np
import typing
import torch

from typing import TYPE_CHECKING, Dict

from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1HandBrand,
    DexforceW1ArmSide,
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.utils import (
    build_dexforce_w1_cfg,
)
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.lab.sim.cfg import (
    RobotCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.data import get_data_path
from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    import pytorch_kinematics as pk


@configclass
class DexforceW1Cfg(RobotCfg):
    """DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters."""

    version: DexforceW1Version = DexforceW1Version.V021
    arm_kind: DexforceW1ArmKind = DexforceW1ArmKind.INDUSTRIAL
    with_default_eef: bool = True

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, str | float | tuple | dict]
    ) -> DexforceW1Cfg:
        """Initialize DexforceW1Cfg from a dictionary.

        Args:
            init_dict: Dictionary of configuration parameters.

        Returns:
            A DexforceW1Cfg instance. Defaults are built via
            :meth:`_build_defaults`, then ``init_dict`` overrides are merged.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Build default urdf/control/solver/physics from variant fields.

        Reads ``version``/``arm_kind``/``with_default_eef`` from ``init_dict``,
        sets them on ``self``, then populates ``urdf_cfg``, ``control_parts``,
        ``solver_cfg``, ``drive_pros`` and ``attrs``.
        """
        init_dict = init_dict or {}
        version = init_dict.get("version", DexforceW1Version.V021)
        arm_kind = init_dict.get("arm_kind", DexforceW1ArmKind.INDUSTRIAL)
        with_default_eef = init_dict.get("with_default_eef", True)

        self.version = (
            DexforceW1Version(version) if isinstance(version, str) else version
        )
        self.arm_kind = (
            DexforceW1ArmKind(arm_kind) if isinstance(arm_kind, str) else arm_kind
        )
        self.with_default_eef = with_default_eef

        # urdf_cfg + control_parts (build_dexforce_w1_cfg no longer sets solver_cfg)
        if self.arm_kind == DexforceW1ArmKind.INDUSTRIAL:
            hand_types = {
                DexforceW1ArmSide.LEFT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
            }
        else:
            hand_types = {
                DexforceW1ArmSide.LEFT: DexforceW1HandBrand.BRAINCO_HAND,
                DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.BRAINCO_HAND,
            }
        hand_versions = {
            DexforceW1ArmSide.LEFT: self.version,
            DexforceW1ArmSide.RIGHT: self.version,
        }
        base_cfg = build_dexforce_w1_cfg(
            arm_kind=self.arm_kind,
            hand_types=hand_types,
            hand_versions=hand_versions,
            include_hand=with_default_eef,
        )
        self.urdf_cfg = base_cfg.urdf_cfg
        self.control_parts = base_cfg.control_parts

        # physics
        physics = self._build_default_physics_cfgs(
            arm_kind=self.arm_kind, with_default_eef=with_default_eef
        )
        for key, value in physics.items():
            setattr(self, key, value)

        # solver (set exactly once -- was previously double-set)
        self.solver_cfg = self._build_default_solver_cfg(arm_kind=self.arm_kind)

    def _build_default_solver_cfg(self, arm_kind: DexforceW1ArmKind):
        """Build the default SRS solver config for the given arm kind.

        Note: the W1ArmKineParams below intentionally use DexforceW1Version.V021
        (matching the original behavior -- version does not flow into the solver).
        """
        from embodichain.lab.sim.solvers import SRSSolverCfg
        from embodichain.lab.sim.robots.dexforce_w1.params import (
            W1ArmKineParams,
        )

        if arm_kind == DexforceW1ArmKind.INDUSTRIAL:
            w1_left_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.LEFT,
                arm_kind=DexforceW1ArmKind.INDUSTRIAL,
                version=DexforceW1Version.V021,
            )
            w1_right_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.RIGHT,
                arm_kind=DexforceW1ArmKind.INDUSTRIAL,
                version=DexforceW1Version.V021,
            )
            left_arm_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.15],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            right_arm_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.15],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        else:
            w1_left_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.LEFT,
                arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
                version=DexforceW1Version.V021,
            )
            w1_right_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.RIGHT,
                arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
                version=DexforceW1Version.V021,
            )
            left_arm_tcp = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.012],
                    [0.0, 0.0, 1.0, 0.0675],
                    [0.0, 1.0, 0.0, 0.127],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            right_arm_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.012],
                    [0.0, 0.0, -1.0, -0.0675],
                    [0.0, 1.0, 0.0, 0.127],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        return {
            "right_arm": SRSSolverCfg(
                end_link_name="right_ee",
                root_link_name="right_arm_base",
                dh_params=w1_right_arm_params.dh_params,
                user_qpos_limits=w1_right_arm_params.qpos_limits,
                T_e_oe=w1_right_arm_params.T_e_oe,
                T_b_ob=w1_right_arm_params.T_b_ob,
                link_lengths=w1_right_arm_params.link_lengths,
                rotation_directions=w1_right_arm_params.rotation_directions,
                tcp=right_arm_tcp,
            ),
            "left_arm": SRSSolverCfg(
                end_link_name="left_ee",
                root_link_name="left_arm_base",
                dh_params=w1_left_arm_params.dh_params,
                user_qpos_limits=w1_left_arm_params.qpos_limits,
                T_e_oe=w1_left_arm_params.T_e_oe,
                T_b_ob=w1_left_arm_params.T_b_ob,
                link_lengths=w1_left_arm_params.link_lengths,
                rotation_directions=w1_left_arm_params.rotation_directions,
                tcp=left_arm_tcp,
            ),
        }

    def _build_default_physics_cfgs(
        self, arm_kind: DexforceW1ArmKind, with_default_eef: bool = True
    ) -> typing.Dict[str, typing.Any]:
        """Build default physics configurations for DexforceW1.

        Args:
            arm_kind: The arm kind enum.
            with_default_eef: Whether to include default end-effector configurations

        Returns:
            Dictionary containing physics configuration parameters
        """
        DEFAULT_EEF_JOINT_DRIVE_PARAMS = {
            "stiffness": 1e2,
            "damping": 1e1,
            "max_effort": 1e3,
        }

        DEFAULT_EEF_HAND_JOINT_NAMES = (
            "(LEFT|RIGHT)_HAND_(THUMB[12]|INDEX|MIDDLE|RING|PINKY)"
        )
        DEFAULT_EEF_GRIPPER_JOINT_NAMES = "(LEFT|RIGHT)_FINGER[1-2]"
        ARM_JOINTS = "(RIGHT|LEFT)_J[0-9]"
        BODY_JOINTS = "(ANKLE|KNEE|BUTTOCK|WAIST)"

        joint_params = {
            "stiffness": {ARM_JOINTS: 1e4, BODY_JOINTS: 1e7},
            "damping": {ARM_JOINTS: 1e3, BODY_JOINTS: 1e4},
            "max_effort": {ARM_JOINTS: 1e5, BODY_JOINTS: 1e10},
        }
        drive_pros = JointDrivePropertiesCfg(**joint_params)

        if with_default_eef:
            eef_joint_names = (
                DEFAULT_EEF_HAND_JOINT_NAMES
                if arm_kind == DexforceW1ArmKind.ANTHROPOMORPHIC
                else DEFAULT_EEF_GRIPPER_JOINT_NAMES
            )
            drive_pros.stiffness.update(
                {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["stiffness"]}
            )
            drive_pros.damping.update(
                {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["damping"]}
            )
            drive_pros.max_effort.update(
                {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["max_effort"]}
            )

        return {
            "min_position_iters": 32,
            "min_velocity_iters": 8,
            "drive_pros": drive_pros,
            "attrs": RigidBodyAttributesCfg(
                mass=1.0,
                static_friction=0.95,
                dynamic_friction=0.9,
                linear_damping=0.7,
                angular_damping=0.7,
                contact_offset=0.005,
                rest_offset=0.001,
                restitution=0.05,
                max_depenetration_velocity=10.0,
            ),
        }

    # to_dict, to_string, save_to_file inherited from RobotCfg

    def _pk_urdf_path(self) -> str:
        """URDF used for FK/IK serial chains, by arm kind.

        .. attention::
            The root_link->end_link kinematics here must match the arms in the
            simulation (assembled) URDF. A DOF drift guard in the tests checks this.
        """
        if self.arm_kind == DexforceW1ArmKind.INDUSTRIAL:
            return get_data_path("DexforceW1V021/DexforceW1_v02_2.urdf")
        return get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        urdf_path = self._pk_urdf_path()

        left_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path,
            device=device,
            end_link_name="left_ee",
            root_link_name="left_arm_base",
        )
        right_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path,
            device=device,
            end_link_name="right_ee",
            root_link_name="right_arm_base",
        )

        return {
            "left_arm": left_arm_chain,
            "right_arm": right_arm_chain,
        }


if __name__ == "__main__":
    # Example usage
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.robots.dexforce_w1.types import (
        DexforceW1ArmKind,
    )

    config = SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=4)
    sim = SimulationManager(config)

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "anthropomorphic"}
    )

    robot = sim.add_robot(cfg=cfg)
    sim.update(step=1)
    print("DexforceW1 robot added to the simulation.")

    # from IPython import embed

    # embed()
