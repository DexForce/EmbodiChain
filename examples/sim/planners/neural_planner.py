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

"""Run the env-batched NeuralPlanner waypoint example.

From the repository root::

    python examples/sim/planners/neural_planner.py --headless
    python examples/sim/planners/neural_planner.py --headless --device cuda:1
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.data.assets.planner_assets import download_neural_planner_checkpoint
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.cfg import MarkerCfg, RenderCfg, physics_cfg_for_backend
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg
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
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(device=default_device, arena_space=2.0)
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=5,
        help="Number of EEF waypoints to send to the neural planner.",
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


def _resolve_device(device: str, gpu_id: int) -> str:
    """Resolve launcher device syntax to an explicit simulation device."""
    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device value {device!r}.") from exc
    if resolved.type != "cuda":
        return str(resolved)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested, but CUDA is not available."
        )
    index = gpu_id if resolved.index is None else resolved.index
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; torch reports "
            f"{torch.cuda.device_count()} device(s)."
        )
    return f"cuda:{index}"


def create_franka(sim: SimulationManager) -> Robot:
    return sim.add_robot(cfg=FrankaPandaCfg.from_dict({"robot_type": "panda"}))


def make_waypoints(start_pose: torch.Tensor, num_waypoints: int) -> torch.Tensor:
    """Create a compact pose path around the start pose."""
    is_unbatched = start_pose.dim() == 2
    if is_unbatched:
        start_pose = start_pose.unsqueeze(0)
    if start_pose.dim() != 3 or start_pose.shape[-2:] != (4, 4):
        raise ValueError(
            "start_pose must have shape (4, 4) or (B, 4, 4), got "
            f"{tuple(start_pose.shape)}."
        )
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
    waypoints = start_pose.unsqueeze(1).repeat(1, num_waypoints, 1, 1)
    waypoints[:, :, :3, 3] += offsets[None, :num_waypoints]
    return waypoints[0] if is_unbatched else waypoints


def draw_waypoint_markers(
    sim: SimulationManager,
    waypoints: torch.Tensor,
    arena_offsets: torch.Tensor,
) -> None:
    if waypoints.dim() == 3:
        waypoints = waypoints.unsqueeze(0)
    if arena_offsets.dim() == 1:
        arena_offsets = arena_offsets.unsqueeze(0)
    if waypoints.shape[0] != arena_offsets.shape[0]:
        raise ValueError(
            f"Waypoint batch {waypoints.shape[0]} does not match arena-offset "
            f"batch {arena_offsets.shape[0]}."
        )
    marker_poses = waypoints.detach().cpu().numpy().copy()
    marker_poses[:, :, :3, 3] += arena_offsets.detach().cpu().numpy().reshape(-1, 1, 3)
    sim.draw_marker(
        cfg=MarkerCfg(
            name="neural_planner_waypoints",
            marker_type="axis",
            axis_xpos=list(marker_poses.reshape(-1, 4, 4)),
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
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    if positions.dim() != 3 or positions.shape[0] != robot.num_instances:
        raise ValueError(
            "positions must have shape "
            f"({robot.num_instances}, N, controlled_dof), got "
            f"{tuple(positions.shape)}."
        )
    for waypoint_idx in range(positions.shape[1]):
        robot.set_qpos(qpos=positions[:, waypoint_idx], joint_ids=joint_ids)
        sim.update(step=step_repeat)
        if delay > 0.0:
            time.sleep(delay)


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num_envs must be at least 1.")
    if args.num_waypoints < 1:
        raise ValueError("--num-waypoints must be at least 1.")
    if args.step_repeat < 1:
        raise ValueError("--step-repeat must be at least 1.")
    if args.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative.")
    checkpoint_path = download_neural_planner_checkpoint()

    sim_device = _resolve_device(args.device, args.gpu_id)
    resolved_device = torch.device(sim_device)
    effective_gpu_id = (
        resolved_device.index if resolved_device.type == "cuda" else int(args.gpu_id)
    )
    assert effective_gpu_id is not None
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            device=sim_device,
            num_envs=args.num_envs,
            arena_space=args.arena_space,
            gpu_id=effective_gpu_id,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_cfg=physics_cfg_for_backend(args.physics),
            visualization=visualization_cfg_from_args(args),
        )
    )
    try:
        robot = create_franka(sim)
        arm_name = "arm"
        device = robot.device

        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        if not args.headless:
            sim.open_window()

        start_qpos = torch.tensor(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        start_qpos = start_qpos.repeat(sim.num_envs, 1)
        robot.set_qpos(qpos=start_qpos, joint_ids=robot.get_joint_ids(arm_name))
        sim.update(step=args.hold_steps)

        start_pose = robot.compute_fk(
            qpos=start_qpos,
            name=arm_name,
            to_matrix=True,
        )
        waypoints = make_waypoints(start_pose, args.num_waypoints)
        draw_waypoint_markers(sim, waypoints, sim.arena_offsets)

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
            PlanState.from_xpos(
                waypoints[:, waypoint_idx],
                move_type=MoveType.EEF_MOVE,
            )
            for waypoint_idx in range(waypoints.shape[1])
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
        print(f"duration by environment: {result.duration.tolist()}s")
        if not result.is_all_success():
            failed_env_ids = (
                torch.nonzero(~result.success, as_tuple=False).flatten().tolist()
            )
            raise RuntimeError(
                f"NeuralPlanner failed for environment(s) {failed_env_ids}."
            )

        play_trajectory(
            sim,
            robot,
            arm_name,
            result.positions,
            step_repeat=args.step_repeat,
        )
        sim.update(step=args.hold_steps)

        if args.interactive:
            from IPython import embed

            embed(header="NeuralPlanner example. Press Ctrl+D to exit.")
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
