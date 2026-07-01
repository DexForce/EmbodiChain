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

import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.data import get_data_path
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["URRobotCfg"]

# ``robot_type`` -> URDF directory / file name. The base is capitalized
# (UR3/UR5/UR10) and the "-e" suffix keeps a lowercase ``e`` (UR3e, UR5e, UR10e).
_URDF_DIR: Dict[str, str] = {
    "ur3": "UR3",
    "ur3e": "UR3e",
    "ur5": "UR5",
    "ur5e": "UR5e",
    "ur10": "UR10",
    "ur10e": "UR10e",
}

# Approximate per-variant joint torque limit (N·m), scaled by robot size.
# These are sim defaults (safety clamp on the PD drive), not factory motor specs.
_UR_MAX_EFFORT: Dict[str, float] = {
    "ur3": 56.0,
    "ur3e": 56.0,
    "ur5": 150.0,
    "ur5e": 150.0,
    "ur10": 330.0,
    "ur10e": 330.0,
}


@configclass
class URRobotCfg(RobotCfg):
    """Configuration for the UR family of robots.

    One config class covers UR3 / UR3e / UR5 / UR5e / UR10 / UR10e, selected via
    ``robot_type``. The kinematic (DH) parameters are owned by
    :class:`~embodichain.lab.sim.solvers.URSolverCfg`; this config owns the URDF,
    control parts, drive properties and rigid-body attributes.

    Example:

        cfg = URRobotCfg.from_dict({"robot_type": "ur5"})
        robot = sim.add_robot(cfg=cfg)
    """

    robot_type: str = "ur10"

    @classmethod
    def from_dict(cls, init_dict):
        """Initialize ``URRobotCfg`` from a dictionary.

        Args:
            init_dict: Dictionary of configuration parameters. ``robot_type``
                selects the UR variant (``ur3``/``ur3e``/``ur5``/``ur5e``/
                ``ur10``/``ur10e``); all other keys are merged on top of the
                defaults via :func:`merge_robot_cfg`.

        Returns:
            A ``URRobotCfg`` instance.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default urdf/control/solver/physics for the chosen UR variant.

        Args:
            init_dict: The raw override dict passed to ``from_dict``. ``robot_type``
                is read from here (falling back to the class default).
        """
        init_dict = init_dict or {}
        robot_type = init_dict.get("robot_type", self.robot_type)
        if robot_type not in _URDF_DIR:
            raise ValueError(
                f"Unknown UR robot_type: {robot_type!r}. "
                f"Expected one of {sorted(_URDF_DIR)}."
            )

        self.robot_type = robot_type
        self.uid = "URRobot"

        urdf_dir = _URDF_DIR[robot_type]
        urdf_path = get_data_path(f"UniversalRobots/{urdf_dir}/{urdf_dir}.urdf")

        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": urdf_path,
                    "transform": np.eye(4),
                }
            ]
        )

        # The UR5 URDF uses lowercase joint names; every other variant uses
        # ``Joint1``..``Joint6``. Build the explicit list per variant so control
        # parts match the loaded URDF exactly.
        joint_prefix = "joint" if robot_type == "ur5" else "Joint"
        self.control_parts = {
            "arm": [f"{joint_prefix}{i}" for i in range(1, 7)],
        }

        self.solver_cfg = {
            "arm": URSolverCfg(
                ur_type=robot_type,
                end_link_name="ee_link",
                root_link_name="base_link",
            ),
        }

        self.drive_pros = JointDrivePropertiesCfg(
            stiffness={"arm": 1e4},
            damping={"arm": 1e3},
            max_effort={"arm": _UR_MAX_EFFORT[robot_type]},
        )

    @property
    def _pk_urdf_path(self) -> str:
        """URDF used for the FK/IK serial chain (the same arm URDF as the sim).

        .. attention::
            The ``base_link``→``ee_link`` kinematics here must match the arm in
            the simulation URDF. A DOF drift guard in the tests checks this.
        """
        urdf_dir = _URDF_DIR[self.robot_type]
        return get_data_path(f"UniversalRobots/{urdf_dir}/{urdf_dir}.urdf")

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
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    config = SimulationManagerCfg(
        headless=False,
        device="cpu",
        num_envs=1,
        render_cfg=RenderCfg(renderer="fast-rt"),
    )
    sim = SimulationManager(config)

    # Switch the UR variant via robot_type (ur3 / ur3e / ur5 / ur5e / ur10 / ur10e).
    cfg = URRobotCfg.from_dict(
        {"robot_type": "ur10e", "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}
    )
    robot = sim.add_robot(cfg=cfg)
    sim.open_window()

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    from IPython import embed

    embed()  # noqa: F401
