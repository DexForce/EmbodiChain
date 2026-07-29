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

"""Demonstrate batched FK and IK with the SRS solver."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)


def main() -> None:
    """Run the SRS solver tutorial."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Run the SRS FK/IK tutorial.")
    )
    args = parser.parse_args()
    setup_print_options()

    sim = create_default_sim(
        args,
        width=2200,
        height=1200,
        add_default_light=False,
    )

    try:
        sim.set_manual_update(False)

        robot: Robot = sim.add_robot(
            cfg=DexforceW1Cfg.from_dict({"uid": "dexforce_w1"})
        )
        arm_name = "left_arm"
        qpos = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        robot.set_qpos(qpos, joint_ids=robot.get_joint_ids(arm_name))

        if not args.auto_play:
            time.sleep(0.5)

        fk_pose = robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)

        started_at = time.perf_counter()
        success, ik_qpos = robot.compute_ik(
            pose=fk_pose,
            name=arm_name,
            return_all_solutions=True,
        )
        elapsed = time.perf_counter() - started_at
        print(
            f"Batch IK computation time for {len(fk_pose)} poses: {elapsed:.6f} seconds"
        )
        print("IK success:", success)

        first_solution = ik_qpos[:, 0, :] if ik_qpos.dim() == 3 else ik_qpos
        robot.set_qpos(first_solution, joint_ids=robot.get_joint_ids(arm_name))
        ik_pose = robot.compute_fk(
            qpos=first_solution,
            name=arm_name,
            to_matrix=True,
        )

        print("FK poses:", fk_pose)
        print("IK poses:", ik_pose)
        maybe_open_window(sim, args)
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
