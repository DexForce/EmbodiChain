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

"""DexforceW1 robot definition using the RobotDef protocol."""

from __future__ import annotations

import numpy as np
import torch
import typing

from typing import Dict

from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import SolverCfg, SRSSolverCfg
from embodichain.lab.sim.robots.protocol import RobotDef
from embodichain.lab.sim.robots.registry import register_robot
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmKind,
    DexforceW1ArmSide,
    DexforceW1Version,
    DexforceW1HandBrand,
)
from embodichain.lab.sim.robots.dexforce_w1.utils import (
    build_dexforce_w1_assembly_urdf_cfg,
    build_dexforce_w1_cfg,
)
from embodichain.data import get_data_path

__all__ = ["DexforceW1Def"]


# ---------------------------------------------------------------------------
# Helper functions extracted from cfg.py
# ---------------------------------------------------------------------------


def _build_solver_cfg(
    arm_kind: str,
) -> Dict[str, SRSSolverCfg]:
    """Build SRS solver configurations for both arms.

    Args:
        arm_kind: Arm type string, either ``"industrial"`` or
            ``"anthropomorphic"``.

    Returns:
        Dictionary mapping ``"left_arm"`` and ``"right_arm"`` to their
        respective :class:`SRSSolverCfg` instances.
    """
    from embodichain.lab.sim.robots.dexforce_w1.params import W1ArmKineParams

    is_industrial = arm_kind == "industrial"
    enum_arm_kind = (
        DexforceW1ArmKind.INDUSTRIAL
        if is_industrial
        else DexforceW1ArmKind.ANTHROPOMORPHIC
    )

    w1_left_arm_params = W1ArmKineParams(
        arm_side=DexforceW1ArmSide.LEFT,
        arm_kind=enum_arm_kind,
        version=DexforceW1Version.V021,
    )
    w1_right_arm_params = W1ArmKineParams(
        arm_side=DexforceW1ArmSide.RIGHT,
        arm_kind=enum_arm_kind,
        version=DexforceW1Version.V021,
    )

    if is_industrial:
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


def _build_drive_pros(
    arm_kind: str,
    include_hand: bool = True,
) -> JointDrivePropertiesCfg:
    """Build joint drive properties for DexforceW1.

    Args:
        arm_kind: Arm type string, either ``"industrial"`` or
            ``"anthropomorphic"``.
        include_hand: Whether to include end-effector drive parameters.

    Returns:
        A :class:`JointDrivePropertiesCfg` with appropriate stiffness,
        damping, and max_effort settings.
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
        "stiffness": {
            ARM_JOINTS: 1e4,
            BODY_JOINTS: 1e7,
        },
        "damping": {
            ARM_JOINTS: 1e3,
            BODY_JOINTS: 1e4,
        },
        "max_effort": {
            ARM_JOINTS: 1e5,
            BODY_JOINTS: 1e10,
        },
    }

    drive_pros = JointDrivePropertiesCfg(**joint_params)

    if include_hand:
        eef_joint_names = (
            DEFAULT_EEF_HAND_JOINT_NAMES
            if arm_kind == "anthropomorphic"
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

    return drive_pros


def _build_attrs() -> RigidBodyAttributesCfg:
    """Build default rigid-body physics attributes for DexforceW1.

    Returns:
        A :class:`RigidBodyAttributesCfg` with default physics parameters.
    """
    return RigidBodyAttributesCfg(
        mass=1.0,
        static_friction=0.95,
        dynamic_friction=0.9,
        linear_damping=0.7,
        angular_damping=0.7,
        contact_offset=0.005,
        rest_offset=0.001,
        restitution=0.05,
        max_depenetration_velocity=10.0,
    )


# ---------------------------------------------------------------------------
# DexforceW1Def
# ---------------------------------------------------------------------------


@register_robot("DexforceW1")
class DexforceW1Def:
    """Robot definition for the DexforceW1 humanoid robot.

    This class satisfies the :class:`~embodichain.lab.sim.robots.protocol.RobotDef`
    protocol and is registered in the global robot registry under the name
    ``"DexforceW1"``.

    The DexforceW1 robot supports multiple variants determined by *version*,
    *arm_kind*, and optional *hand_types*.  All configuration-building logic
    delegates to the existing utility functions in
    :mod:`embodichain.lab.sim.robots.dexforce_w1.utils`.

    Attributes:
        name: Unique identifier string for this robot.
        version: Component version string (e.g. ``"v021"``).
        arm_kind: Arm type, either ``"industrial"`` or ``"anthropomorphic"``.
        hand_types: Optional mapping from arm side to hand brand enum.
        include_chassis: Whether the chassis component is included.
        include_torso: Whether the torso component is included.
        include_head: Whether the head component is included.
        include_hand: Whether the hand end-effectors are included.
        min_position_iters: Minimum position iterations for the solver.
        min_velocity_iters: Minimum velocity iterations for the solver.
    """

    name: str = "DexforceW1"

    def __init__(
        self,
        version: str = "v021",
        arm_kind: str = "anthropomorphic",
        hand_types: dict | None = None,
        include_chassis: bool = True,
        include_torso: bool = True,
        include_head: bool = True,
        include_hand: bool = True,
    ) -> None:
        self.version = version
        self.arm_kind = arm_kind
        self.include_chassis = include_chassis
        self.include_torso = include_torso
        self.include_head = include_head
        self.include_hand = include_hand
        self._hand_types = hand_types
        self.__post_init__()

    def __post_init__(self) -> None:
        """Fill default hand_types based on arm_kind if not explicitly set."""
        if self._hand_types is None:
            if self.arm_kind == "industrial":
                self._hand_types = {
                    DexforceW1ArmSide.LEFT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                    DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                }
            else:
                self._hand_types = {
                    DexforceW1ArmSide.LEFT: DexforceW1HandBrand.BRAINCO_HAND,
                    DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.BRAINCO_HAND,
                }

    # -- URDF configuration ----------------------------------------------------

    @property
    def urdf_cfg(self) -> URDFCfg:
        """URDF assembly configuration for the full robot."""
        arm_kind_enum = DexforceW1ArmKind(self.arm_kind)
        version_enum = DexforceW1Version(self.version)
        hand_versions = {
            DexforceW1ArmSide.LEFT: version_enum,
            DexforceW1ArmSide.RIGHT: version_enum,
        }
        return build_dexforce_w1_assembly_urdf_cfg(
            arm_kind=arm_kind_enum,
            hand_types=self._hand_types,
            hand_versions=hand_versions,
            include_chassis=self.include_chassis,
            include_torso=self.include_torso,
            include_head=self.include_head,
            include_hand=self.include_hand,
        )

    # -- Control parts ---------------------------------------------------------

    @property
    def control_parts(self) -> dict[str, list[str]]:
        """Mapping from part name to joint name lists."""
        arm_kind_enum = DexforceW1ArmKind(self.arm_kind)
        version_enum = DexforceW1Version(self.version)
        hand_versions = {
            DexforceW1ArmSide.LEFT: version_enum,
            DexforceW1ArmSide.RIGHT: version_enum,
        }
        cfg = build_dexforce_w1_cfg(
            arm_kind=arm_kind_enum,
            hand_types=self._hand_types,
            hand_versions=hand_versions,
            include_chassis=self.include_chassis,
            include_torso=self.include_torso,
            include_head=self.include_head,
            include_hand=self.include_hand,
        )
        return cfg.control_parts

    # -- Solver configuration --------------------------------------------------

    @property
    def solver_cfg(self) -> Dict[str, SRSSolverCfg]:
        """SRS IK solver configuration for each arm."""
        return _build_solver_cfg(arm_kind=self.arm_kind)

    # -- Drive properties ------------------------------------------------------

    @property
    def drive_pros(self) -> JointDrivePropertiesCfg:
        """Joint drive properties (stiffness, damping, max_effort)."""
        return _build_drive_pros(
            arm_kind=self.arm_kind,
            include_hand=self.include_hand,
        )

    # -- Rigid-body attributes -------------------------------------------------

    @property
    def attrs(self) -> RigidBodyAttributesCfg:
        """Rigid-body physics attributes."""
        return _build_attrs()

    # -- Extra fields ----------------------------------------------------------

    min_position_iters: int = 32
    min_velocity_iters: int = 8

    # -- Methods ---------------------------------------------------------------

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build pytorch-kinematics serial chains for each arm.

        Args:
            device: Torch device to place chains on.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary mapping ``"left_arm"`` and ``"right_arm"`` to their
            respective serial chain objects.
        """
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_chain,
            create_pk_serial_chain,
        )

        arm_kind_enum = DexforceW1ArmKind(self.arm_kind)

        if arm_kind_enum == DexforceW1ArmKind.INDUSTRIAL:
            urdf_path = get_data_path("DexforceW1V021/DexforceW1_v02_2.urdf")
        else:
            urdf_path = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")

        chain = create_pk_chain(urdf_path, device)

        left_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="left_ee", root_link_name="left_arm_base"
        ).to(device=device)
        right_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="right_ee", root_link_name="right_arm_base"
        ).to(device=device)

        return {
            "left_arm": left_arm_chain,
            "right_arm": right_arm_chain,
        }

    def build_cfg(self, **overrides: object) -> RobotCfg:
        """Build a :class:`RobotCfg` from this robot definition.

        Delegates to :meth:`RobotDef.build_cfg` for the default
        implementation.

        Args:
            **overrides: Optional overrides applied on top of the defaults.

        Returns:
            A fully-populated :class:`RobotCfg`.
        """
        return RobotDef.build_cfg(self, **overrides)
