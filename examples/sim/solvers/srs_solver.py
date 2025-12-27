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

import time
import numpy as np
import torch

from IPython import embed

from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg


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

    sim.set_manual_update(False)

    robot: Robot = sim.add_robot(cfg=DexforceW1Cfg.from_dict({"uid": "dexforce_w1"}))
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
