# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
import os
import time
import numpy as np
import torch
from IPython import embed

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots.dexforce_w1.params import (
    W1ArmKineParams,
)
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)


def main():
    # Set print options for better readability
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Initialize simulation
    sim_device = "cpu"
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=False, sim_device=sim_device, width=2200, height=1200
        )
    )

    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    is_industrial = False

    # Load robot URDF file
    if is_industrial:
        urdf = get_data_path("DexforceW1V020/DexforceW1_v02_2.urdf")
    else:
        urdf = get_data_path("DexforceW1V020/DexforceW1_v02_1.urdf")
    assert os.path.isfile(urdf)

    if is_industrial:
        w1_left_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.LEFT,
            arm_kind=DexforceW1ArmKind.INDUSTRIAL,
            version=DexforceW1Version.V020,
        )
        w1_right_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.RIGHT,
            arm_kind=DexforceW1ArmKind.INDUSTRIAL,
            version=DexforceW1Version.V020,
        )
    else:
        w1_left_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.LEFT,
            arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
            version=DexforceW1Version.V020,
        )
        w1_right_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.RIGHT,
            arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
            version=DexforceW1Version.V020,
        )

    # Robot configuration dictionary
    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
            "right_arm": [f"RIGHT_J{i+1}" for i in range(7)],
            "torso": ["ANKLE", "KNEE", "BUTTOCK", "WAIST"],
            "head": [f"NECK{i+1}" for i in range(2)],
        },
        "drive_pros": {
            "stiffness": {
                "LEFT_J[1-7]": 1e4,
                "RIGHT_J[1-7]": 1e4,
                "ANKLE": 1e7,
                "KNEE": 1e7,
                "BUTTOCK": 1e7,
                "WAIST": 1e7,
            },
            "damping": {
                "LEFT_J[1-7]": 1e3,
                "RIGHT_J[1-7]": 1e3,
                "ANKLE": 1e4,
                "KNEE": 1e4,
                "BUTTOCK": 1e4,
                "WAIST": 1e4,
            },
            "max_effort": {
                "LEFT_J[1-7]": 1e5,
                "RIGHT_J[1-7]": 1e5,
                "ANKLE": 1e10,
                "KNEE": 1e10,
                "BUTTOCK": 1e10,
                "WAIST": 1e10,
            },
        },
        "attrs": {
            "mass": 1e-1,
            "static_friction": 0.95,
            "dynamic_friction": 0.9,
            "linear_damping": 0.7,
            "angular_damping": 0.7,
            "max_depenetration_velocity": 10.0,
        },
        "solver_cfg": {
            "left_arm": {
                "class_type": "SRSSolver",
                "end_link_name": "left_ee",
                "root_link_name": "left_arm_base",
                "dh_params": w1_left_arm_params.dh_params,
                "qpos_limits": w1_left_arm_params.qpos_limits,
                "T_e_oe": w1_left_arm_params.T_e_oe,
                "T_b_ob": w1_left_arm_params.T_b_ob,
                "link_lengths": w1_left_arm_params.link_lengths,
                "rotation_directions": w1_left_arm_params.rotation_directions,
            },
            "right_arm": {
                "class_type": "SRSSolver",
                "end_link_name": "right_ee",
                "root_link_name": "right_arm_base",
                "dh_params": w1_right_arm_params.dh_params,
                "qpos_limits": w1_right_arm_params.qpos_limits,
                "T_e_oe": w1_right_arm_params.T_e_oe,
                "T_b_ob": w1_right_arm_params.T_b_ob,
                "link_lengths": w1_right_arm_params.link_lengths,
                "rotation_directions": w1_right_arm_params.rotation_directions,
            },
        },
    }

    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
    arm_name = "left_arm"
    # Set initial joint positions for left arm
    qpos_fk_list = [
        torch.tensor([[0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]], dtype=torch.float32),
    ]
    robot.set_qpos(qpos_fk_list[0], joint_ids=robot.get_joint_ids(arm_name))

    time.sleep(0.5)

    fk_xpos_batch = torch.cat(qpos_fk_list, dim=0)

    fk_xpos_list = robot.compute_fk(qpos=fk_xpos_batch, name=arm_name, to_matrix=True)

    start_time = time.time()
    res, ik_qpos = robot.compute_ik(
        pose=fk_xpos_list,
        name=arm_name,
        # joint_seed=qpos_fk_list[0],
        return_all_solutions=True,
    )
    end_time = time.time()
    print(
        f"Batch IK computation time for {len(fk_xpos_list)} poses: {end_time - start_time:.6f} seconds"
    )

    if ik_qpos.dim() == 3:
        first_solutions = ik_qpos[:, 0, :]
    else:
        first_solutions = ik_qpos
    robot.set_qpos(first_solutions, joint_ids=robot.get_joint_ids(arm_name))

    ik_xpos_list = robot.compute_fk(qpos=first_solutions, name=arm_name, to_matrix=True)

    print("fk_xpos_list: ", fk_xpos_list)
    print("ik_xpos_list: ", ik_xpos_list)

    embed(header="Test SRSSolver example. Press Ctrl-D to exit.")


if __name__ == "__main__":
    main()
