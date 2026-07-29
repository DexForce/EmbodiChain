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

"""Demonstrate OPW FK/IK on both CobotMagic arms."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)


def _first_solution(qpos: torch.Tensor) -> torch.Tensor:
    """Normalize analytic IK output to one solution per environment."""
    return qpos[:, 0, :] if qpos.dim() == 3 else qpos


def main() -> None:
    """Solve and visualize one pose for each arm."""
    args = add_demo_args(
        argparse.ArgumentParser(description="Run the OPW solver example.")
    ).parse_args()
    setup_print_options()

    sim = create_default_sim(args, add_default_light=False)
    try:
        robot = sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {
                    "uid": "CobotMagic",
                    "init_pos": [0.0, 0.0, 0.7775],
                    "solver_cfg": {
                        arm: {
                            "class_type": "OPWSolver",
                            "end_link_name": f"{side}_link6",
                            "root_link_name": f"{side}_arm_base",
                            "tcp": [
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0.143],
                                [0, 0, 0, 1],
                            ],
                        }
                        for arm, side in (
                            ("left_arm", "left"),
                            ("right_arm", "right"),
                        )
                    },
                }
            )
        )
        maybe_open_window(sim, args)

        target_qpos = torch.tensor(
            [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]],
            dtype=torch.float32,
            device=robot.device,
        )
        seed = torch.zeros_like(target_qpos)
        for arm_name in ("left_arm", "right_arm"):
            target_pose = robot.compute_fk(
                qpos=target_qpos,
                name=arm_name,
                to_matrix=True,
            )
            started_at = time.perf_counter()
            success, solutions = robot.compute_ik(
                pose=target_pose,
                name=arm_name,
                joint_seed=seed,
            )
            elapsed = time.perf_counter() - started_at
            if not torch.as_tensor(success).all():
                raise RuntimeError(f"OPW IK failed for {arm_name}.")

            solution = _first_solution(solutions)
            achieved_pose = robot.compute_fk(
                qpos=solution,
                name=arm_name,
                to_matrix=True,
            )
            robot.set_qpos(solution, joint_ids=robot.get_joint_ids(arm_name))
            print(f"{arm_name} IK computation time: {elapsed:.6f} seconds")

            for suffix, pose in (
                ("target", target_pose),
                ("result", achieved_pose),
            ):
                sim.draw_marker(
                    cfg=MarkerCfg(
                        name=f"{arm_name}_{suffix}",
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
