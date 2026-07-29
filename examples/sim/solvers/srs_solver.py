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

"""Demonstrate SRS FK/IK on a DexForce W1 arm."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

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
    """Solve one left-arm pose with all analytic solutions enabled."""
    args = add_demo_args(
        argparse.ArgumentParser(description="Run the SRS solver example.")
    ).parse_args()
    setup_print_options()

    sim = create_default_sim(
        args,
        width=2200,
        height=1200,
        add_default_light=False,
    )
    try:
        robot = sim.add_robot(cfg=DexforceW1Cfg.from_dict({"uid": "dexforce_w1"}))
        maybe_open_window(sim, args)

        arm_name = "left_arm"
        qpos = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=robot.device,
        )
        target_pose = robot.compute_fk(
            qpos=qpos,
            name=arm_name,
            to_matrix=True,
        )

        started_at = time.perf_counter()
        success, solutions = robot.compute_ik(
            pose=target_pose,
            name=arm_name,
            return_all_solutions=True,
        )
        elapsed = time.perf_counter() - started_at
        if not torch.as_tensor(success).all():
            raise RuntimeError("SRS IK failed.")

        solution = solutions[:, 0, :] if solutions.dim() == 3 else solutions
        robot.set_qpos(solution, joint_ids=robot.get_joint_ids(arm_name))
        achieved_pose = robot.compute_fk(
            qpos=solution,
            name=arm_name,
            to_matrix=True,
        )

        print(f"IK computation time: {elapsed:.6f} seconds")
        print("Target FK pose:", target_pose)
        print("IK result pose:", achieved_pose)
        sim.update(step=1)
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
