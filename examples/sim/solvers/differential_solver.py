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
import os
import time
import numpy as np
import torch
from IPython import embed

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg


def main(visualize: bool = True):
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Set up simulation with specified device (CPU or CUDA)
    sim_device = "cpu"
    num_envs = 9  # Number of parallel arenas/environments
    config = SimulationManagerCfg(
        headless=False, sim_device=sim_device, arena_space=1.5, num_envs=num_envs
    )
    sim = SimulationManager(config)
    sim.set_manual_update(False)

    # Load robot URDF file
    urdf = get_data_path("Rokae/SR5/SR5.urdf")
    assert os.path.isfile(urdf)

    # Robot configuration
    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "main_arm": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
        },
        "solver_cfg": {
            "main_arm": {
                "class_type": "DifferentialSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
            },
        },
    }

    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

    # Prepare initial joint positions for all environments
    rad = torch.deg2rad(torch.tensor(45.0))
    arm_name = "main_arm"
    fk_qpos = torch.full((num_envs, 6), rad, dtype=torch.float32, device="cpu")
    # All envs start with the same qpos (can be randomized)
    qpos = torch.from_numpy(np.array([0.0, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0])).to(
        fk_qpos.device
    )
    qpos = qpos.unsqueeze(0).repeat(num_envs, 1)
    robot.set_qpos(qpos=qpos, joint_ids=robot.get_joint_ids(arm_name))

    time.sleep(3.0)
    fk_xpos = robot.compute_fk(
        qpos=qpos, name=arm_name, to_matrix=True
    )  # (num_envs, 4, 4)

    # Prepare batch start and end poses for all envs
    start_pose = fk_xpos.clone()  # (num_envs, 4, 4)
    end_pose = fk_xpos.clone()
    move_vecs = torch.tensor(
        [
            [0.3, 0.0, 0.0],
            [0.2, -0.2, 0.0],
            [0.0, 0.0, 0.2],
            [0.2, 0.0, 0.2],
            [-0.3, 0.0, 0.0],
            [-0.2, 0.2, 0.0],
            [0.0, 0.0, -0.2],
            [-0.2, 0.0, -0.2],
            [0.1, 0.1, -0.1],
        ],
        dtype=end_pose.dtype,
        device=end_pose.device,
    )
    end_pose[
        :, :3, 3
    ] += move_vecs  # Move each env's end-effector in a different direction

    num_steps = 100
    # Interpolate poses for each env
    interpolated_poses = torch.stack(
        [torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)],
        dim=1,
    )  # (num_envs, num_steps, 4, 4)

    # Initial joint positions for all envs
    ik_qpos = qpos.clone()

    ik_qpos_results = []
    ik_success_flags = []

    ik_compute_begin = time.time()
    # Batch IK solving for each step
    for step in range(num_steps):
        poses = interpolated_poses[:, step, :, :]  # (num_envs, 4, 4)
        if poses.shape[0] != num_envs:
            poses = poses.expand(num_envs, *poses.shape[1:])
        if ik_qpos.shape[0] != num_envs:
            ik_qpos = ik_qpos.expand(num_envs, *ik_qpos.shape[1:])
        assert (
            poses.shape[0] == num_envs
        ), f"poses batch mismatch: {poses.shape[0]} vs {num_envs}"
        assert (
            ik_qpos.shape[0] == num_envs
        ), f"ik_qpos batch mismatch: {ik_qpos.shape[0]} vs {num_envs}"

        # Parallel batch IK solving
        res, ik_qpos_new = robot.compute_ik(
            pose=poses, joint_seed=ik_qpos, name=arm_name
        )
        ik_qpos_results.append(ik_qpos_new.clone())
        ik_success_flags.append(res)
        ik_qpos = ik_qpos_new  # Update joint seed
    ik_compute_end = time.time()
    print(
        f"IK compute time for {num_steps} steps and {num_envs} envs: {ik_compute_end - ik_compute_begin:.4f} seconds"
    )

    # Collect visualization data for all steps and environments
    if visualize:
        draw_data = [[] for _ in range(num_envs)]
    for env_id in range(num_envs):
        for step in range(num_steps):
            ik_qpos_new = ik_qpos_results[step]
            ik_xpos = robot.compute_fk(qpos=ik_qpos_new, name=arm_name, to_matrix=True)
            local_pose = robot._entities[env_id].get_world_pose()
            if visualize:
                fk_axis = local_pose @ end_pose[env_id].cpu().numpy()
                ik_axis = local_pose @ ik_xpos[env_id].cpu().numpy()
                local_axis = local_pose @ ik_xpos[env_id].cpu().numpy()

                draw_data[env_id].append(
                    {
                        "step": step,
                        "fk_axis": fk_axis,
                        "ik_axis": ik_axis,
                        "local_axis": local_axis,
                    }
                )

    if visualize:
        # Batch draw all steps' data for each environment
        for env_id in range(num_envs):
            # Only draw fk_axis and ik_axis once per env (first step)
            fk_axis = draw_data[env_id][0]["fk_axis"]
            ik_axis = draw_data[env_id][0]["ik_axis"]

            sim.draw_marker(
                cfg=MarkerCfg(
                    name=f"fk_axis_env{env_id}",
                    marker_type="axis",
                    axis_xpos=fk_axis,
                    axis_size=0.002,
                    axis_len=0.005,
                    arena_index=env_id,
                )
            )

            sim.draw_marker(
                cfg=MarkerCfg(
                    name=f"ik_axis_env{env_id}",
                    marker_type="axis",
                    axis_xpos=ik_axis,
                    axis_size=0.002,
                    axis_len=0.005,
                    arena_index=env_id,
                )
            )

            # Draw the whole local_axis trajectory as a single call
            local_axes = np.stack(
                [item["local_axis"] for item in draw_data[env_id]], axis=0
            )  # (num_steps, 4, 4) or (num_steps, 3)

            sim.draw_marker(
                cfg=MarkerCfg(
                    name=f"local_axis_env{env_id}_trajectory",
                    marker_type="axis",
                    axis_xpos=local_axes,
                    axis_size=0.002,
                    axis_len=0.005,
                    arena_index=env_id,
                )
            )

    # Optionally, set qpos for each step (replay or animate)
    for step in range(num_steps):
        ik_qpos_new = ik_qpos_results[step]
        res = ik_success_flags[step]
        # Only set qpos for successful IK results
        if isinstance(res, (list, np.ndarray, torch.Tensor)):
            for env_id, success in enumerate(res):
                if success:
                    q = (
                        ik_qpos_new[env_id]
                        if ik_qpos_new.dim() == 3
                        else ik_qpos_new[env_id]
                    )
                    robot.set_qpos(
                        qpos=q,
                        joint_ids=robot.get_joint_ids(arm_name),
                        env_ids=[env_id],
                    )
        else:
            # fallback: set all
            if ik_qpos_new.dim() == 3:
                robot.set_qpos(
                    qpos=ik_qpos_new[:, 0, :], joint_ids=robot.get_joint_ids(arm_name)
                )
            else:
                robot.set_qpos(
                    qpos=ik_qpos_new, joint_ids=robot.get_joint_ids(arm_name)
                )
        time.sleep(0.005)

    embed(header="Test DifferentialSolver example. Press Ctrl-D to exit.")


if __name__ == "__main__":
    main(visualize=True)
