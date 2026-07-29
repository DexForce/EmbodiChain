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

"""Demonstrate batched neural IK on Franka Panda."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.data.assets.solver_assets import download_neural_ik_checkpoint
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg
from embodichain.lab.sim.solvers import NeuralIKSolverCfg
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
    (0.3, 0.4, -0.2),
    (0.2, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, -0.2, -0.1),
    (-0.2, 0.0, 0.0),
    (0.0, -0.2, 0.0),
    (0.0, 0.0, -0.15),
    (-0.2, 0.2, 0.0),
    (0.0, 0.2, -0.15),
)


def main() -> None:
    """Solve and replay one neural-IK Cartesian path per environment."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Run the NeuralIKSolver example.")
    )
    parser.add_argument(
        "--num_steps",
        "--num-steps",
        type=int,
        default=50,
        help="Number of Cartesian interpolation steps.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    setup_print_options()

    # Download before allocating simulation resources so download errors do
    # not leave a partially initialized renderer behind.
    checkpoint_path = download_neural_ik_checkpoint()
    sim = create_default_sim(
        args,
        num_envs=args.num_envs,
        arena_space=2.0,
        add_default_light=False,
    )
    try:
        cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
        cfg.solver_cfg["arm"] = NeuralIKSolverCfg(
            end_link_name="fr3_hand_tcp",
            root_link_name="base",
            tcp=np.eye(4).tolist(),
            checkpoint_path=checkpoint_path,
            num_arm_joints=7,
            max_steps=30,
            action_scale=0.2,
            hidden_dims=[256, 256],
            pos_eps=0.1,
            rot_eps=0.5,
        )
        robot = sim.add_robot(cfg=cfg)
        maybe_init_gpu_physics(sim)
        maybe_open_window(sim, args)

        arm_name = "arm"
        qpos = torch.tensor(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
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
            f"Solved {args.num_steps} x {args.num_envs} neural IK targets in "
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
                            name=f"neural_ik_{suffix}_{env_id}",
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
            sim.update(step=5)
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
