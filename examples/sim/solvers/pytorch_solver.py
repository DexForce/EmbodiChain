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

"""Demonstrate batched PyTorch IK on a DexForce W1 arm."""

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
    maybe_init_gpu_physics,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)

TARGET_OFFSETS = (
    (0.2, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, -0.2, -0.5),
    (-0.2, 0.0, 0.0),
    (-0.2, 0.0, 0.0),
    (0.0, -0.2, 0.0),
    (0.0, 0.0, -0.5),
    (-0.2, 0.2, 0.0),
    (0.0, 0.2, -0.5),
)


def main() -> None:
    """Solve and replay one batched Cartesian path."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Run batched PyTorch IK.")
    )
    parser.set_defaults(num_envs=9)
    parser.add_argument(
        "--num_steps",
        "--num-steps",
        type=int,
        default=50,
        help="Number of Cartesian interpolation steps.",
    )
    args = parser.parse_args()
    setup_print_options()

    sim = create_default_sim(
        args,
        num_envs=args.num_envs,
        arena_space=2.0,
        add_default_light=False,
    )
    try:
        robot = sim.add_robot(
            cfg=RobotCfg.from_dict(
                {
                    "fpath": get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf"),
                    "control_parts": {
                        "left_arm": [f"LEFT_J{i}" for i in range(1, 8)],
                    },
                    "solver_cfg": {
                        "left_arm": {
                            "class_type": "PytorchSolver",
                            "end_link_name": "left_ee",
                            "root_link_name": "left_arm_base",
                        },
                    },
                }
            )
        )
        maybe_init_gpu_physics(sim)
        maybe_open_window(sim, args)

        arm_name = "left_arm"
        qpos = torch.tensor(
            [0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=robot.device,
        ).repeat(args.num_envs, 1)
        robot.set_qpos(qpos, joint_ids=robot.get_joint_ids(arm_name))
        start_pose = robot.compute_fk(
            qpos=qpos,
            name=arm_name,
            to_matrix=True,
        )
        target_pose = start_pose.clone()
        target_pose[:, :3, 3] += torch.tensor(
            [
                TARGET_OFFSETS[env_id % len(TARGET_OFFSETS)]
                for env_id in range(args.num_envs)
            ],
            dtype=target_pose.dtype,
            device=target_pose.device,
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

        qpos_history = []
        success_history = []
        seed = qpos
        started_at = time.perf_counter()
        for pose in poses.unbind(dim=1):
            success, solution = robot.compute_ik(
                pose=pose,
                joint_seed=seed,
                name=arm_name,
            )
            seed = solution[:, 0, :] if solution.dim() == 3 else solution
            qpos_history.append(seed.clone())
            success_history.append(torch.as_tensor(success, device=robot.device))
        print(
            f"Solved {args.num_steps} x {args.num_envs} PyTorch IK targets in "
            f"{time.perf_counter() - started_at:.4f} seconds"
        )

        final_pose = robot.compute_fk(
            qpos=qpos_history[-1],
            name=arm_name,
            to_matrix=True,
        )
        if not args.no_vis_eef_axis:
            for env_id in range(args.num_envs):
                for suffix, pose in (
                    ("target", target_pose[env_id]),
                    ("result", final_pose[env_id]),
                ):
                    sim.draw_marker(
                        cfg=MarkerCfg(
                            name=f"pytorch_{suffix}_{env_id}",
                            marker_type="axis",
                            axis_xpos=pose,
                            axis_size=0.002,
                            axis_len=0.005,
                            arena_index=env_id,
                        )
                    )

        joint_ids = robot.get_joint_ids(arm_name)
        for solution, success in zip(qpos_history, success_history):
            success_ids = success.nonzero(as_tuple=True)[0]
            if success_ids.numel():
                robot.set_qpos(
                    solution[success_ids],
                    joint_ids=joint_ids,
                    env_ids=success_ids,
                )
            sim.update(step=1)
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
