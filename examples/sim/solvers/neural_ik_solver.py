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
import argparse
import math
import time

import numpy as np
import torch
from IPython import embed

from embodichain.data.assets.solver_assets import download_neural_ik_checkpoint
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg
from embodichain.lab.sim.solvers import NeuralIKSolverCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuralIKSolver example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for tensors and the neural IK solver (default: cpu).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to simulate. IK is solved for all "
        "environments simultaneously at each step (default: 1).",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu or install "
            "a CUDA-enabled PyTorch build."
        )
    return device


def _squeeze_ik_qpos(ik_qpos: torch.Tensor) -> torch.Tensor:
    """Normalize IK output to (num_envs, dof)."""
    if ik_qpos.dim() == 3:
        return ik_qpos[:, 0, :]
    return ik_qpos


def _pose_with_arena_offset(
    pose: torch.Tensor | np.ndarray, arena_offset: torch.Tensor
) -> np.ndarray:
    """Convert arena-local 4x4 pose to world frame by adding arena translation."""
    if isinstance(pose, torch.Tensor):
        xpos = pose.detach().cpu().numpy()
    else:
        xpos = np.asarray(pose)
    xpos = np.array(xpos, copy=True, dtype=np.float64)
    offset = arena_offset.detach().cpu().numpy().reshape(3)
    if xpos.ndim == 2:
        xpos[:3, 3] += offset
    elif xpos.ndim == 3:
        xpos[:, :3, 3] += offset
    return xpos


def main():
    args = parse_args()
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    device = _resolve_device(args.device)
    num_envs = args.num_envs

    config = SimulationManagerCfg(
        headless=True,
        device=device,
        num_envs=num_envs,
        arena_space=2.0,
    )
    sim = SimulationManager(config)

    checkpoint_path = download_neural_ik_checkpoint()

    cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
    cfg.solver_cfg["arm"] = NeuralIKSolverCfg(
        end_link_name="fr3_hand_tcp",
        root_link_name="base",
        tcp=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        checkpoint_path=checkpoint_path,
        num_arm_joints=7,
        max_steps=30,
        action_scale=0.2,
        hidden_dims=[256, 256],
        pos_eps=0.1,
        rot_eps=0.5,
    )

    robot: Robot = sim.add_robot(cfg=cfg)

    sim.open_window()

    arm_name = "arm"
    device = robot.device

    seed_qpos = torch.tensor(
        [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
        dtype=torch.float32,
        device=device,
    )
    qpos = seed_qpos.unsqueeze(0).expand(num_envs, -1).clone()
    robot.set_qpos(qpos=qpos, joint_ids=robot.get_joint_ids(arm_name))
    time.sleep(3.0)

    fk_xpos = robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
    print(f"fk_xpos shape: {tuple(fk_xpos.shape)}")

    start_pose = fk_xpos.clone()
    end_pose = fk_xpos.clone()

    # Per-environment target offsets (cycle if num_envs exceeds preset count)
    move_vecs = torch.tensor(
        [
            [0.3, 0.4, -0.2],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, -0.1],
            [-0.2, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, -0.15],
            [-0.2, 0.2, 0.0],
            [0.0, 0.2, -0.15],
        ],
        dtype=torch.float32,
        device=device,
    )
    for env_id in range(num_envs):
        end_pose[env_id, :3, 3] += move_vecs[env_id % move_vecs.shape[0]]

    num_steps = 50
    interpolated_poses = torch.stack(
        [
            torch.lerp(start_pose, end_pose, t)
            for t in torch.linspace(0.0, 1.0, num_steps, device=device)
        ],
        dim=1,
    )

    ik_qpos = qpos.clone()
    ik_qpos_results: list[torch.Tensor] = []
    ik_success_flags: list[torch.Tensor] = []

    print(
        f"\nRunning {num_steps} batch IK steps: num_envs={num_envs}, device='{device}' ..."
    )
    ik_compute_begin = time.time()
    for step in range(num_steps):
        poses = interpolated_poses[:, step, :, :]
        res, ik_qpos_new = robot.compute_ik(
            pose=poses, joint_seed=ik_qpos, name=arm_name
        )
        ik_qpos = _squeeze_ik_qpos(ik_qpos_new)
        ik_qpos_results.append(ik_qpos.clone())
        ik_success_flags.append(res)
    ik_compute_end = time.time()
    print(
        f"IK compute time for {num_steps} steps and {num_envs} envs: "
        f"{ik_compute_end - ik_compute_begin:.4f}s"
    )

    # Draw target and achieved EE axes for each environment (final step)
    final_step = num_steps - 1
    final_ik_qpos = ik_qpos_results[final_step]
    final_res = ik_success_flags[final_step]
    ik_xpos_all = robot.compute_fk(qpos=final_ik_qpos, name=arm_name, to_matrix=True)
    arena_offsets = sim.arena_offsets

    for env_id in range(num_envs):
        target_axis = _pose_with_arena_offset(end_pose[env_id], arena_offsets[env_id])
        sim.draw_marker(
            cfg=MarkerCfg(
                name=f"fk_target_env{env_id}",
                marker_type="axis",
                axis_xpos=target_axis,
                axis_size=0.002,
                axis_len=0.005,
                arena_index=-1,
            )
        )

        if final_res[env_id]:
            ik_axis = _pose_with_arena_offset(
                ik_xpos_all[env_id], arena_offsets[env_id]
            )
            sim.draw_marker(
                cfg=MarkerCfg(
                    name=f"ik_result_env{env_id}",
                    marker_type="axis",
                    axis_xpos=ik_axis,
                    axis_size=0.002,
                    axis_len=0.005,
                    arena_index=-1,
                )
            )

    # Animate: batch-apply IK qpos for successful envs, then step simulation
    joint_ids = robot.get_joint_ids(arm_name)
    for step in range(num_steps):
        ik_qpos_step = ik_qpos_results[step]
        res = ik_success_flags[step]
        if res.any():
            success_ids = res.nonzero(as_tuple=True)[0]
            robot.set_qpos(
                qpos=ik_qpos_step[success_ids],
                joint_ids=joint_ids,
                env_ids=success_ids,
            )
        sim.update(step=5)

    embed(header="NeuralIKSolver example. Press Ctrl+D to exit.")


if __name__ == "__main__":
    main()
