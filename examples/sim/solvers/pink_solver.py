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

"""Demonstrate iterative Pink IK along a Cartesian path."""

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
    """Solve a short Cartesian path with Pink."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Run the Pink solver example.")
    )
    parser.add_argument(
        "--num_steps",
        "--num-steps",
        type=int,
        default=100,
        help="Number of Cartesian interpolation steps.",
    )
    args = parser.parse_args()
    setup_print_options()

    sim = create_default_sim(args, add_default_light=False)
    try:
        robot = sim.add_robot(
            cfg=RobotCfg.from_dict(
                {
                    "fpath": get_data_path("Rokae/SR5/SR5.urdf"),
                    "control_parts": {
                        "main_arm": [f"joint{i}" for i in range(1, 7)],
                    },
                    "solver_cfg": {
                        "main_arm": {
                            "class_type": "PinkSolver",
                            "end_link_name": "ee_link",
                            "root_link_name": "base_link",
                        },
                    },
                }
            )
        )
        maybe_open_window(sim, args)

        arm_name = "main_arm"
        qpos = torch.tensor(
            [[0.0, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0]],
            dtype=torch.float32,
            device=robot.device,
        )
        robot.set_qpos(qpos, joint_ids=robot.get_joint_ids(arm_name))
        start_pose = robot.compute_fk(
            qpos=qpos,
            name=arm_name,
            to_matrix=True,
        )
        target_pose = start_pose.clone()
        target_pose[:, 1, 3] += 0.4
        sim.draw_marker(
            cfg=MarkerCfg(
                name="pink_target",
                marker_type="axis",
                axis_xpos=target_pose,
                axis_size=0.002,
                axis_len=0.005,
            )
        )

        poses = torch.stack(
            [
                torch.lerp(start_pose, target_pose, t)
                for t in torch.linspace(
                    0.0,
                    1.0,
                    args.num_steps,
                    device=robot.device,
                )
            ],
            dim=1,
        )
        started_at = time.perf_counter()
        for pose in poses.unbind(dim=1):
            success, solution = robot.compute_ik(
                pose=pose,
                joint_seed=qpos,
                name=arm_name,
            )
            if not torch.as_tensor(success).all():
                raise RuntimeError("Pink IK failed along the Cartesian path.")
            qpos = solution[:, 0, :] if solution.dim() == 3 else solution
            robot.set_qpos(qpos, joint_ids=robot.get_joint_ids(arm_name))
            sim.update(step=1)
        print(
            f"Solved {args.num_steps} Pink IK steps in "
            f"{time.perf_counter() - started_at:.6f} seconds"
        )

        achieved_pose = robot.compute_fk(
            qpos=qpos,
            name=arm_name,
            to_matrix=True,
        )
        sim.draw_marker(
            cfg=MarkerCfg(
                name="pink_result",
                marker_type="axis",
                axis_xpos=achieved_pose,
                axis_size=0.002,
                axis_len=0.005,
            )
        )
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
