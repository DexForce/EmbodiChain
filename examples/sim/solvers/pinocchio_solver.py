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

"""Demonstrate Pinocchio FK/IK on a DexForce W1 arm."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import MarkerCfg, RobotCfg
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)


def main() -> None:
    """Solve one left-arm pose and compare FK with the IK result."""
    args = add_demo_args(
        argparse.ArgumentParser(description="Run the Pinocchio solver example.")
    ).parse_args()
    setup_print_options()

    sim = create_default_sim(args, add_default_light=False)
    try:
        robot = sim.add_robot(
            cfg=RobotCfg.from_dict(
                {
                    "fpath": get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf"),
                    "control_parts": {
                        "left_arm": [f"LEFT_J{i + 1}" for i in range(7)],
                    },
                    "solver_cfg": {
                        "left_arm": {
                            "class_type": "PinocchioSolver",
                            "end_link_name": "left_ee",
                            "root_link_name": "left_arm_base",
                        },
                    },
                }
            )
        )
        maybe_open_window(sim, args)

        arm_name = "left_arm"
        target_qpos = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=robot.device,
        )
        seed = target_qpos.clone()
        seed[:, 1] = 0.1
        target_pose = robot.compute_fk(
            qpos=target_qpos,
            name=arm_name,
            to_matrix=True,
        )

        started_at = time.perf_counter()
        success, solution = robot.compute_ik(
            pose=target_pose,
            name=arm_name,
            joint_seed=seed,
        )
        elapsed = time.perf_counter() - started_at
        if not torch.as_tensor(success).all():
            raise RuntimeError("Pinocchio IK failed.")
        if solution.dim() == 3:
            solution = solution[:, 0, :]

        achieved_pose = robot.compute_fk(
            qpos=solution,
            name=arm_name,
            to_matrix=True,
        )
        robot.set_qpos(solution, joint_ids=robot.get_joint_ids(arm_name))
        print(f"IK computation time: {elapsed:.6f} seconds")

        for suffix, pose in (("target", target_pose), ("result", achieved_pose)):
            sim.draw_marker(
                cfg=MarkerCfg(
                    name=f"pinocchio_{suffix}",
                    marker_type="axis",
                    axis_xpos=pose,
                    axis_size=0.002,
                    axis_len=0.005,
                )
            )

        sim.update(step=1)
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
