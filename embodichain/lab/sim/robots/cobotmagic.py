# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from typing import Dict, List, Optional, Any, Union

from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import SolverCfg, OPWSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import configclass
from embodichain.utils import logger


@configclass
class CobotMagicCfg(RobotCfg):
    urdf_cfg: URDFCfg = None
    control_parts: Optional[Dict[str, List[str]]] = None
    solver_cfg: Optional[Dict[str, "SolverCfg"]] = None

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Union[str, float, int]]) -> CobotMagicCfg:
        from embodichain.lab.sim.solvers import merge_solver_cfg

        cfg = cls()
        default_cfgs = cls()._build_default_cfgs()
        for key, value in default_cfgs.items():
            setattr(cfg, key, value)

        robot_cfg = RobotCfg.from_dict(init_dict)

        # set attrs into cfg from the robot_cfg
        for key, value in init_dict.items():
            if key == "solver_cfg":
                # merge provided solver_cfg values into default solver config
                provided_solver_cfg = init_dict.get("solver_cfg")
                if provided_solver_cfg:
                    for part, item in provided_solver_cfg.items():
                        if "class_type" in provided_solver_cfg[part]:
                            cfg.solver_cfg[part] = robot_cfg.solver_cfg[part]
                        else:
                            try:
                                merged = merge_solver_cfg(
                                    cfg.solver_cfg, provided_solver_cfg
                                )
                                cfg.solver_cfg = merged
                            except Exception:
                                logger.log_error(
                                    f"Failed to merge solver_cfg, using provided config outright."
                                )
            else:
                setattr(cfg, key, getattr(robot_cfg, key))

        return cfg

    @staticmethod
    def _build_default_cfgs() -> Dict[str, Any]:
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
        urdf_cfg = URDFCfg(
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
        return {
            "uid": "CobotMagic",
            "urdf_cfg": urdf_cfg,
            "control_parts": {
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
            },
            "solver_cfg": {
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
            },
            "min_position_iters": 8,
            "min_velocity_iters": 2,
            "drive_pros": JointDrivePropertiesCfg(
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
            ),
            "attrs": RigidBodyAttributesCfg(
                mass=0.1,
                static_friction=0.95,
                dynamic_friction=0.9,
                linear_damping=0.7,
                angular_damping=0.7,
                contact_offset=0.005,
                rest_offset=0.001,
                restitution=0.01,
                max_depenetration_velocity=1e1,
            ),
        }

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_chain,
            create_pk_serial_chain,
        )

        urdf_path = get_data_path("CobotMagicArm/CobotMagicNoGripper.urdf")
        chain = create_pk_chain(urdf_path, device)

        left_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="link6", root_link_name="base_link"
        ).to(device=device)
        right_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="link6", root_link_name="base_link"
        ).to(device=device)
        return {"left_arm": left_arm_chain, "right_arm": right_arm_chain}


if __name__ == "__main__":
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.robots import CobotMagicCfg

    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(headless=False, sim_device="cuda")
    sim = SimulationManager(config)
    sim.build_multiple_arenas(2)

    config = {
        "init_pos": [0.0, 0.0, 1.0],
    }

    cfg = CobotMagicCfg.from_dict(config)
    robot = sim.add_robot(cfg=cfg)

    sim.init_gpu_physics()
    print("CobotMagic added to the simulation.")

    from IPython import embed

    embed()
