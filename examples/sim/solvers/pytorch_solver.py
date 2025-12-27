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


def main():
    # Set numpy and torch print options for better readability
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Initialize simulation environment (CPU or CUDA)
    sim_device = "cpu"
    num_envs = 9  # Number of parallel environments
    config = SimulationManagerCfg(
        headless=False, sim_device=sim_device, arena_space=2.0, num_envs=num_envs
    )
    sim = SimulationManager(config)
    sim.set_manual_update(False)

    # Load robot URDF file
    urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
    assert os.path.isfile(urdf)

    # Robot configuration dictionary
    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "left_arm": [
                "LEFT_J1",
                "LEFT_J2",
                "LEFT_J3",
                "LEFT_J4",
                "LEFT_J5",
                "LEFT_J6",
                "LEFT_J7",
            ],
        },
        "solver_cfg": {
            "left_arm": {
                "class_type": "PytorchSolver",
                "end_link_name": "left_ee",
                "root_link_name": "left_arm_base",
            },
        },
    }

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

    # Prepare initial joint positions for all environments
    arm_name = "left_arm"
    qpos = (
        torch.tensor([0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0], dtype=torch.float32)
        .unsqueeze(0)
        .repeat(num_envs, 1)
    )
    robot.set_qpos(qpos=qpos, joint_ids=robot.get_joint_ids(arm_name))

    time.sleep(2.0)
    fk_xpos = robot.compute_fk(
        qpos=qpos, name=arm_name, to_matrix=True
    )  # (num_envs, 4, 4)

    # Prepare batch start and end poses for all envs
    start_pose = fk_xpos.clone()
    end_pose = fk_xpos.clone()
    move_vecs = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, -0.5],
            [-0.2, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, -0.5],
            [-0.2, 0.2, 0.0],
            [0.0, 0.2, -0.5],
        ],
        dtype=end_pose.dtype,
        device=end_pose.device,
    )
    end_pose[:, :3, 3] += move_vecs

    num_steps = 50
    # Interpolate poses for each env
    interpolated_poses = torch.stack(
        [torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)],
        dim=1,
    )  # (num_envs, num_steps, 4, 4)

    # Initial joint positions for all envs
    ik_qpos = qpos.clone()
    ik_qpos_results = []
    ik_success_flags = []

    # Batch IK solving for each step
    ik_compute_begin = time.time()
    for step in range(num_steps):
        poses = interpolated_poses[:, step, :, :]  # (num_envs, 4, 4)
        if poses.shape[0] != num_envs:
            poses = poses.expand(num_envs, *poses.shape[1:])
        if ik_qpos.shape[0] != num_envs:
            ik_qpos = ik_qpos.expand(num_envs, *ik_qpos.shape[1:])
        assert poses.shape[0] == num_envs
        assert ik_qpos.shape[0] == num_envs

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
    draw_data = [[] for _ in range(num_envs)]
    for env_id in range(num_envs):
        for step in range(num_steps):
            ik_qpos_new = ik_qpos_results[step]
            ik_xpos = robot.compute_fk(qpos=ik_qpos_new, name=arm_name, to_matrix=True)
            local_pose = robot._entities[env_id].get_link_pose("left_arm_base")
            if isinstance(local_pose, np.ndarray):
                local_pose = torch.from_numpy(local_pose).to(
                    ik_xpos.device, dtype=ik_xpos.dtype
                )
            fk_axis = (local_pose @ end_pose[env_id]).cpu().numpy()
            ik_axis = (local_pose @ ik_xpos[env_id]).cpu().numpy()
            local_axis = (local_pose @ ik_xpos[env_id]).cpu().numpy()
            draw_data[env_id].append(
                {
                    "step": step,
                    "fk_axis": fk_axis,
                    "ik_axis": ik_axis,
                    "local_axis": local_axis,
                }
            )

    # Batch draw: only draw fk_axis and ik_axis once per env, draw local_axis trajectory for all steps
    for env_id in range(num_envs):
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

        # Draw the whole local_axis trajectory as a single call (if supported)
        local_axes = np.stack(
            [item["local_axis"] for item in draw_data[env_id]], axis=0
        )

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
            if ik_qpos_new.dim() == 3:
                robot.set_qpos(
                    qpos=ik_qpos_new[:, 0, :], joint_ids=robot.get_joint_ids(arm_name)
                )
            else:
                robot.set_qpos(
                    qpos=ik_qpos_new, joint_ids=robot.get_joint_ids(arm_name)
                )
        time.sleep(0.005)

    embed(header="Test PytorchSolver batch example. Press Ctrl-D to exit.")


if __name__ == "__main__":
    main()
