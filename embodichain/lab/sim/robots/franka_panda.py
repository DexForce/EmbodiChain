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

import math
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.motion.solvers import FEPSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FrankaPandaCfg", "FrankaPandaPgiCfg"]

# ``robot_type`` -> URDF directory / file name.
_FRANKA_URDF_DIR: Dict[str, str] = {
    "panda": "Panda",
}

# Default init_qpos: arm in a neutral ready pose, fingers open.
# Derived from the Isaac Lab Franka Emika Panda reference configuration.
_FRANKA_DEFAULT_INIT_QPOS = [
    0.0,  # Joint1
    -0.569,  # Joint2
    0.0,  # Joint3
    -2.810,  # Joint4
    0.0,  # Joint5
    3.037,  # Joint6
    0.741,  # Joint7
    0.04,  # finger_joint1
    0.04,  # finger_joint2 (mimic of finger_joint1)
]

_FRANKA_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]

# DH Robotics PGI-140-80 parallel gripper mounted on the arm-only Panda flange.
_PGI_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
_PGI_JOINT = "gripper_finger1_joint_1"
_PGI_FINGER_LINKS = ["gripper_finger1_link_1", "gripper_finger2_link_1"]
_PGI_TCP = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.17],
    [0.0, 0.0, 0.0, 1.0],
]


def _newton_gripper_contacts(link_names_expr: list[str]) -> LinkPhysicsOverrideCfg:
    """Return the Newton fingertip contact group for the given finger links."""
    return LinkPhysicsOverrideCfg(
        link_names_expr=list(link_names_expr),
        attrs=RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(condim=4),
            material_props=NewtonRigidBodyMaterialCfg(
                ke=40000.0,
                kd=400.0,
                torsional_friction=0.1,
                rolling_friction=0.01,
            ),
        ),
    )


def _resolve_robot_type(init_dict: dict, default: str) -> str:
    """Return the requested Franka variant, rejecting unknown ones."""
    robot_type = init_dict.get("robot_type", default)
    if robot_type not in _FRANKA_URDF_DIR:
        raise ValueError(
            f"Unknown Franka robot_type: {robot_type!r}. "
            f"Expected one of {sorted(_FRANKA_URDF_DIR)}."
        )
    return robot_type


@configclass
class FrankaPandaCfg(RobotCfg):
    """Configuration for the Franka Emika Panda robot with Panda hand.

    The PandaWithHand URDF includes both the 7-DOF arm and the parallel-jaw
    gripper in a single file. The arm defaults to
    :class:`~embodichain.lab.sim.motion.solvers.FEPSolverCfg` with redundancy
    search enabled.

    Example:

        cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
        robot = sim.add_robot(cfg=cfg)
    """

    robot_type: str = "panda"

    @classmethod
    def from_dict(cls, init_dict):
        """Initialize ``FrankaPandaCfg`` from a dictionary.

        Args:
            init_dict: Dictionary of configuration parameters. ``robot_type``
                selects the Franka variant (currently ``"panda"``). All other
                keys are merged on top of the defaults via
                :func:`merge_robot_cfg`.

        Returns:
            A ``FrankaPandaCfg`` instance.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default urdf/control/solver/physics for the Franka variant.

        Args:
            init_dict: The raw override dict passed to ``from_dict``.
                ``robot_type`` is read from here (falling back to the class
                default).
        """
        init_dict = init_dict or {}
        robot_type = _resolve_robot_type(init_dict, self.robot_type)

        self.robot_type = robot_type
        self.uid = "FrankaPanda"

        urdf_dir = _FRANKA_URDF_DIR[robot_type]
        urdf_path = get_data_path(f"Franka/{urdf_dir}/{urdf_dir}WithHand.urdf")

        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": urdf_path,
                    "transform": np.eye(4),
                }
            ]
        )

        self.control_parts = {
            "arm": list(_FRANKA_ARM_JOINTS),
            "eef": ["fr3_finger_joint1", "fr3_finger_joint2"],
        }

        self.solver_cfg = {
            "arm": FEPSolverCfg(
                end_link_name="fr3_hand_tcp",
                root_link_name="base",
                tcp=[
                    [1, 0, 0.0, 0.0],
                    [0, 1, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                redundancy_search=True,
            ),
        }

        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={
                "fr3_joint[1-7]": 1e4,
                "fr3_finger_joint[1-2]": 1e3,
            },
            damping={
                "fr3_joint[1-7]": 1e3,
                "fr3_finger_joint[1-2]": 1e2,
            },
            max_effort={
                "fr3_joint[1-7]": 1e5,
                "fr3_finger_joint[1-2]": 1e4,
            },
        )
        self.link_attrs = {
            "newton_gripper_contacts": _newton_gripper_contacts(
                ["fr3_leftfinger|fr3_rightfinger"]
            )
        }

        self.init_qpos = list(_FRANKA_DEFAULT_INIT_QPOS)

    @property
    def _pk_urdf_path(self) -> str:
        """URDF used for the FK/IK serial chain."""
        urdf_dir = _FRANKA_URDF_DIR[self.robot_type]
        return get_data_path(f"Franka/{urdf_dir}/{urdf_dir}WithHand.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build the pytorch-kinematics serial chain for the arm.

        Args:
            device: The device to which the chain will be moved. Defaults to CPU.
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            A ``{"arm": pk.SerialChain}`` mapping.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path,
            device=device,
            end_link_name="fr3_hand_tcp",
            root_link_name="base",
        )
        return {"arm": chain}


@configclass
class FrankaPandaPgiCfg(RobotCfg):
    """Configuration for the Franka Emika Panda arm with a DH Robotics PGI gripper.

    The arm-only Panda URDF is assembled with the ``DH_PGI_140_80`` parallel
    gripper on the ``fr3_link8`` flange. The configuration owns the whole
    assembly: the ``arm`` and ``eef`` control parts, the PGI finger drive and
    open initial state, the PGI tool center point, and the Newton fingertip
    contacts on the PGI finger links. The stock Panda hand settings of
    :class:`FrankaPandaCfg` do not apply.

    The arm solver and the FK/IK serial chain both end at ``fr3_link8`` on the
    arm URDF used by the assembly. The solver's ``tcp`` gives the PGI tool
    center point in that frame; :meth:`build_pk_serial_chain` returns the
    flange pose without it.

    Example:

        cfg = FrankaPandaPgiCfg.from_dict({"init_pos": [0.5, 0.0, 0.0]})
        robot = sim.add_robot(cfg=cfg)
    """

    robot_type: str = "panda"

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> FrankaPandaPgiCfg:
        """Initialize ``FrankaPandaPgiCfg`` from a dictionary.

        Args:
            init_dict: Dictionary of configuration parameters. ``robot_type``
                selects the Franka arm variant (currently ``"panda"``). All
                other keys are merged on top of the defaults via
                :func:`merge_robot_cfg`.

        Returns:
            A ``FrankaPandaPgiCfg`` instance.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate the arm-only Panda and PGI assembly defaults.

        Args:
            init_dict: The raw override dict passed to ``from_dict``.
                ``robot_type`` is read from here (falling back to the class
                default).
        """
        init_dict = init_dict or {}
        self.robot_type = _resolve_robot_type(init_dict, self.robot_type)
        self.uid = "FrankaPandaPgi"

        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": self._pk_urdf_path,
                    "transform": np.eye(4),
                },
                {
                    "component_type": "hand",
                    "urdf_path": get_data_path(_PGI_URDF_PATH),
                    "transform": np.eye(4),
                },
            ]
        )

        self.control_parts = {
            "arm": list(_FRANKA_ARM_JOINTS),
            "eef": [_PGI_JOINT],
        }

        self.solver_cfg = {
            "arm": FEPSolverCfg(
                end_link_name="fr3_link8",
                root_link_name="base",
                tcp=[list(row) for row in _PGI_TCP],
                redundancy_search=True,
            ),
        }

        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"fr3_joint[1-7]": 1e4, _PGI_JOINT: 1e3},
            damping={"fr3_joint[1-7]": 1e3, _PGI_JOINT: 1e2},
            max_effort={"fr3_joint[1-7]": 1e5, _PGI_JOINT: 1e4},
        )
        self.link_attrs = {
            "newton_gripper_contacts": _newton_gripper_contacts(_PGI_FINGER_LINKS)
        }

        # Arm in the stock ready pose; both PGI finger joints open.
        self.init_qpos = [*_FRANKA_DEFAULT_INIT_QPOS[:7], 0.0, 0.0]

    @property
    def _pk_urdf_path(self) -> str:
        """Arm-only URDF used by both the assembly and the FK/IK chain."""
        urdf_dir = _FRANKA_URDF_DIR[self.robot_type]
        return get_data_path(f"Franka/{urdf_dir}/{urdf_dir}.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build the pytorch-kinematics serial chain for the arm.

        The chain uses the arm solver's root and end links, so its end pose is
        the ``fr3_link8`` flange; compose it with the solver ``tcp`` to obtain
        the PGI tool center point.

        Args:
            device: The device to which the chain will be moved. Defaults to CPU.
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            A ``{"arm": pk.SerialChain}`` mapping.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        solver = self.solver_cfg["arm"]
        chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path,
            device=device,
            end_link_name=solver.end_link_name,
            root_link_name=solver.root_link_name,
        )
        return {"arm": chain}


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=5, suppress=True)

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend

    parser = argparse.ArgumentParser(description="Launch the Franka Panda robot")
    parser.add_argument(
        "--physics",
        choices=("default", "newton"),
        default="default",
        help="Physics backend to launch (default: default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Runtime device override; otherwise the selected backend default is used.",
    )
    args = parser.parse_args()

    config = SimulationManagerCfg(
        headless=True,
        device=args.device,
        num_envs=1,
        physics_cfg=physics_cfg_for_backend(args.physics),
        render_cfg=RenderCfg(renderer="hybrid"),
    )
    sim = SimulationManager(config)

    cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
    robot = sim.add_robot(cfg=cfg)
    sim.prepare()
    sim.open_window()

    from IPython import embed

    embed()  # noqa: F401
