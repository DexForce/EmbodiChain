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
from typing import TYPE_CHECKING, Dict

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

__all__ = ["FrankaPandaCfg"]

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


def _franka_tcp() -> list[list[float]]:
    """Build the Franka Panda TCP as a 4x4 homogeneous matrix.

    The TCP applies a 45-degree rotation about Z and a 0.1034 m Z-offset,
    matching the standard Franka hand geometry.
    """
    c = math.cos(-math.pi / 4)
    s = math.sin(-math.pi / 4)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]


@configclass
class FrankaPandaCfg(RobotCfg):
    """Configuration for the Franka Emika Panda robot with Panda hand.

    The PandaWithHand URDF includes both the 7-DOF arm and the parallel-jaw
    gripper in a single file. The solver defaults to
    :class:`~embodichain.lab.sim.solvers.PytorchSolverCfg`.

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
        robot_type = init_dict.get("robot_type", self.robot_type)
        if robot_type not in _FRANKA_URDF_DIR:
            raise ValueError(
                f"Unknown Franka robot_type: {robot_type!r}. "
                f"Expected one of {sorted(_FRANKA_URDF_DIR)}."
            )

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
            "arm": [f"Joint{i}" for i in range(1, 8)],
            "hand": ["finger_joint1", "finger_joint2"],
        }

        self.solver_cfg = {
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=_franka_tcp(),
                num_samples=30,
            ),
        }

        self.drive_pros = JointDrivePropertiesCfg(
            stiffness={
                "Joint[1-7]": 1e4,
                "finger_joint[1-2]": 1e3,
            },
            damping={
                "Joint[1-7]": 1e3,
                "finger_joint[1-2]": 1e2,
            },
            max_effort={
                "Joint[1-7]": 1e5,
                "finger_joint[1-2]": 1e4,
            },
        )

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
            end_link_name="ee_link",
            root_link_name="base_link",
        )
        return {"arm": chain}


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    config = SimulationManagerCfg(
        headless=False,
        sim_device="cpu",
        num_envs=1,
        render_cfg=RenderCfg(renderer="hybrid"),
    )
    sim = SimulationManager(config)

    cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
    robot = sim.add_robot(cfg=cfg)
    sim.open_window()

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    from IPython import embed

    embed()  # noqa: F401
