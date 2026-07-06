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

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.data.assets.planner_assets import download_neural_planner_checkpoint
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg, RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    NeuralPlannerCfg,
    PlanState,
)
from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="NeuralPlanner waypoint example")
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["cpu", "cuda"],
        help="Simulation and planner device.",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=5,
        help="Number of EEF waypoints to send to the neural planner.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the viewer window.",
    )
    parser.add_argument(
        "--step-repeat",
        type=int,
        default=10,
        help="Simulation updates per planned waypoint during playback.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=60,
        help="Simulation updates to hold before and after playback.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drop into IPython after playback.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu instead."
        )
    return device


def _franka_tcp() -> list[list[float]]:
    c = math.cos(-math.pi / 4)
    s = math.sin(-math.pi / 4)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]


def create_franka(sim: SimulationManager) -> Robot:
    urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
    assert os.path.isfile(urdf)

    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "main_arm": [
                "Joint1",
                "Joint2",
                "Joint3",
                "Joint4",
                "Joint5",
                "Joint6",
                "Joint7",
            ],
        },
        "solver_cfg": {
            "main_arm": {
                "class_type": "PytorchSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
                "tcp": _franka_tcp(),
            },
        },
    }
    return sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))


def make_waypoints(start_pose: torch.Tensor, num_waypoints: int) -> torch.Tensor:
    """Create a compact pose path around the start pose."""
    offsets = torch.tensor(
        [
            [0.10, 0.00, 0.00],
            [0.10, 0.10, 0.00],
            [0.00, 0.10, -0.08],
            [-0.10, 0.10, -0.08],
            [-0.10, 0.00, 0.00],
            [0.00, -0.10, 0.00],
            [0.10, -0.10, -0.06],
            [0.00, 0.00, -0.12],
        ],
        dtype=start_pose.dtype,
        device=start_pose.device,
    )
    num_waypoints = max(1, min(int(num_waypoints), offsets.shape[0]))
    waypoints = start_pose.unsqueeze(0).repeat(num_waypoints, 1, 1)
    waypoints[:, :3, 3] += offsets[:num_waypoints]
    return waypoints


def draw_waypoint_markers(
    sim: SimulationManager,
    waypoints: torch.Tensor,
    arena_offset: torch.Tensor,
) -> None:
    marker_poses = waypoints.detach().cpu().numpy().copy()
    marker_poses[:, :3, 3] += arena_offset.detach().cpu().numpy().reshape(1, 3)
    sim.draw_marker(
        cfg=MarkerCfg(
            name="neural_planner_waypoints",
            marker_type="axis",
            axis_xpos=list(marker_poses),
            axis_size=0.003,
            axis_len=0.03,
            arena_index=-1,
        )
    )


def play_trajectory(
    sim: SimulationManager,
    robot: Robot,
    arm_name: str,
    positions: torch.Tensor,
    step_repeat: int = 4,
    delay: float = 0.0,
) -> None:
    joint_ids = robot.get_joint_ids(arm_name)
    # ``positions`` may be env-batched (B, N, DOF); this example is single-env.
    if positions.dim() == 3:
        positions = positions[0]
    for qpos in positions:
        robot.set_qpos(qpos=qpos.unsqueeze(0), joint_ids=joint_ids)
        sim.update(step=step_repeat)
        if delay > 0.0:
            time.sleep(delay)


def main() -> None:
    args = parse_args()
    checkpoint_path = download_neural_planner_checkpoint()

    sim_device = _resolve_device(args.device)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            sim_device=sim_device,
            num_envs=1,
            arena_space=2.0,
        )
    )

    robot = create_franka(sim)
    arm_name = "main_arm"
    device = robot.device

    if not args.headless:
        sim.open_window()

    start_qpos = torch.tensor(
        [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
        dtype=torch.float32,
        device=device,
    )
    robot.set_qpos(
        qpos=start_qpos.unsqueeze(0), joint_ids=robot.get_joint_ids(arm_name)
    )
    sim.update(step=args.hold_steps)

    start_pose = robot.compute_fk(
        qpos=start_qpos.unsqueeze(0), name=arm_name, to_matrix=True
    )[0]
    waypoints = make_waypoints(start_pose, args.num_waypoints)
    draw_waypoint_markers(sim, waypoints, sim.arena_offsets[0])

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid=robot.uid,
                checkpoint_path=checkpoint_path,
                control_part=arm_name,
            )
        )
    )
    target_states = [
        PlanState.single(move_type=MoveType.EEF_MOVE, xpos=waypoint)
        for waypoint in waypoints
    ]
    result = motion_generator.generate(
        target_states=target_states,
        options=MotionGenOptions(
            plan_opts=NeuralPlanOptions(
                control_part=arm_name,
                start_qpos=start_qpos,
            ),
        ),
    )

    print(f"NeuralPlanner success: {result.success}")
    print(f"positions shape: {tuple(result.positions.shape)}")
    print(f"xpos_list shape: {tuple(result.xpos_list.shape)}")
    print(f"duration: {result.duration.item():.3f}s")

    play_trajectory(
        sim,
        robot,
        arm_name,
        result.positions,
        step_repeat=max(args.step_repeat, 1),
    )
    sim.update(step=args.hold_steps)

    if args.interactive:
        from IPython import embed

        embed(header="NeuralPlanner example. Press Ctrl+D to exit.")


if __name__ == "__main__":
    main()
