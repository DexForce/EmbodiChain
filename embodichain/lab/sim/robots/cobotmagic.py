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

import torch
import numpy as np

from typing import TYPE_CHECKING, Dict, List, Union

from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import SolverCfg, OPWSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.data import get_data_path
from embodichain.utils import configclass
from embodichain.utils import logger

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["CobotMagicCfg"]


@configclass
class CobotMagicCfg(RobotCfg):
    urdf_cfg: URDFCfg = None
    control_parts: Dict[str, List[str]] | None = None
    solver_cfg: Dict[str, "SolverCfg"] | None = None

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Union[str, float, int]]) -> CobotMagicCfg:
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default urdf/control/solver/physics for CobotMagic."""
        init_dict = init_dict or {}
        arm_urdf = get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf")
        left_arm_xpos = np.array(
            [
                [1.0, 0.0, 0.0, 0.233],
                [0.0, 1.0, 0.0, 0.300],
                [0.0, 0.0, 1.0, 0.000],
                [0.0, 0.0, 0.0, 1.000],
            ]
        )
        right_arm_xpos = np.array(
            [
                [1.0, 0.0, 0.0, 0.233],
                [0.0, 1.0, 0.0, -0.300],
                [0.0, 0.0, 1.0, 0.000],
                [0.0, 0.0, 0.0, 1.000],
            ]
        )
        self.uid = "CobotMagic"
        self.urdf_cfg = URDFCfg(
            components=[
                {
                    "component_type": "left_arm",
                    "urdf_path": arm_urdf,
                    "transform": left_arm_xpos,
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": arm_urdf,
                    "transform": right_arm_xpos,
                },
            ]
        )
        self.control_parts = {
            "left_arm": [
                "LEFT_JOINT1",
                "LEFT_JOINT2",
                "LEFT_JOINT3",
                "LEFT_JOINT4",
                "LEFT_JOINT5",
                "LEFT_JOINT6",
            ],
            "left_eef": ["LEFT_JOINT7", "LEFT_JOINT8"],
            "right_arm": [
                "RIGHT_JOINT1",
                "RIGHT_JOINT2",
                "RIGHT_JOINT3",
                "RIGHT_JOINT4",
                "RIGHT_JOINT5",
                "RIGHT_JOINT6",
            ],
            "right_eef": ["RIGHT_JOINT7", "RIGHT_JOINT8"],
        }
        self.solver_cfg = {
            "left_arm": OPWSolverCfg(
                end_link_name="left_link6",
                root_link_name="left_arm_base",
                tcp=np.array(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]]
                ),
            ),
            "right_arm": OPWSolverCfg(
                end_link_name="right_link6",
                root_link_name="right_arm_base",
                tcp=np.array(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]]
                ),
            ),
        }
        self.min_position_iters = 8
        self.min_velocity_iters = 2
        self.drive_pros = JointDrivePropertiesCfg(
            stiffness={
                "LEFT_JOINT[1-6]": 7e4,
                "RIGHT_JOINT[1-6]": 7e4,
                "LEFT_JOINT[7-8]": 3e2,
                "RIGHT_JOINT[7-8]": 3e2,
            },
            damping={
                "LEFT_JOINT[1-6]": 1e3,
                "RIGHT_JOINT[1-6]": 1e3,
                "LEFT_JOINT[7-8]": 3e1,
                "RIGHT_JOINT[7-8]": 3e1,
            },
            max_effort={
                "LEFT_JOINT[1-6]": 3e6,
                "RIGHT_JOINT[1-6]": 3e6,
                "LEFT_JOINT[7-8]": 3e3,
                "RIGHT_JOINT[7-8]": 3e3,
            },
        )
        self.attrs = RigidBodyAttributesCfg(
            mass=0.1,
            static_friction=0.95,
            dynamic_friction=0.9,
            linear_damping=0.7,
            angular_damping=0.7,
            contact_offset=0.001,
            rest_offset=0.001,
            restitution=0.01,
            max_depenetration_velocity=1e1,
        )

    @property
    def _pk_urdf_path(self) -> str:
        """URDF used for FK/IK serial chains (arm-only, gripper-stripped).

        .. attention::
            The root_link->end_link kinematics here must match the arm in the
            simulation URDF. A DOF drift guard in the tests checks this.
        """
        return get_data_path("CobotMagicArm/CobotMagicNoGripper.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        urdf_path = self._pk_urdf_path

        left_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path,
            device=device,
            end_link_name="link6",
            root_link_name="base_link",
        )
        right_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path,
            device=device,
            end_link_name="link6",
            root_link_name="base_link",
        )
        return {"left_arm": left_arm_chain, "right_arm": right_arm_chain}


if __name__ == "__main__":
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.sim.robots import CobotMagicCfg

    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(
        headless=True,
        sim_device="cuda",
        num_envs=2,
        render_cfg=RenderCfg(renderer="fast-rt"),
    )
    sim = SimulationManager(config)

    config = {"init_pos": [0.0, 0.0, 1.0], "init_qpos": [0.1] * 16}

    cfg = CobotMagicCfg.from_dict(config)
    robot = sim.add_robot(cfg=cfg)
    sim.open_window()

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    print("CobotMagic added to the simulation.")

    from IPython import embed

    embed()
