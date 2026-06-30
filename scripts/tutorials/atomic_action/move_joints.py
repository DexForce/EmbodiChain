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

"""Demonstrate MoveJoints with named and explicit joint-space targets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    JointPositionTarget,
    MoveJoints,
    MoveJointsCfg,
    NamedJointPositionTarget,
)
from embodichain.lab.sim.cfg import LightCfg, RenderCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_ur5_gripper_robot_cfg,
    draw_axis_marker,
    get_tutorial_window_size,
    start_auto_play_recording,
    stop_auto_play_recording,
)

MOVE_JOINTS_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveJoints with named and explicit qpos targets."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw the current end-effector/TCP coordinate frame before planning.",
    )
    return parser.parse_args()


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    width, height = get_tutorial_window_size(args)
    cfg = SimulationManagerCfg(
        width=width,
        height=height,
        headless=True,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_dt=1.0 / 100.0,
        arena_space=2.5,
    )
    sim = SimulationManager(cfg)
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0.0, 3.0),
        )
    )
    return sim


def create_robot(sim: SimulationManager, position=(0.0, 0.0, 0.0)) -> Robot:
    cfg = create_ur5_gripper_robot_cfg(init_pos=position)
    return sim.add_robot(cfg=cfg)


def make_arm_qpos(values: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def draw_start_eef_axis(sim: SimulationManager, robot: Robot) -> None:
    eef_pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"),
        name="arm",
        to_matrix=True,
    )
    draw_axis_marker(sim, "move_joints_start_eef_axis", eef_pose)


def main() -> None:
    """Move the robot arm through named and explicit joint targets."""
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Step 1: Set up simulation and robot                                 #
    # ------------------------------------------------------------------ #
    sim = initialize_simulation(args)
    robot = create_robot(sim)

    # ------------------------------------------------------------------ #
    # Step 2: Create a MotionGenerator for the robot                      #
    # ------------------------------------------------------------------ #
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    # ------------------------------------------------------------------ #
    # Step 3: Configure the MoveJoints atomic action                      #
    # ------------------------------------------------------------------ #
    ready_qpos = make_arm_qpos([0.35, -1.20, 1.30, -1.65, -1.57, 0.20], sim.device)
    home_qpos = make_arm_qpos([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], sim.device)
    move_joints_cfg = MoveJointsCfg(
        control_part="arm",
        sample_interval=MOVE_JOINTS_SAMPLE_INTERVAL,
        named_joint_positions={"ready": ready_qpos},
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    # ------------------------------------------------------------------ #
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(MoveJoints(motion_gen, cfg=move_joints_cfg))

    # ------------------------------------------------------------------ #
    # Step 5: Open the viewer and show the starting end-effector frame    #
    # ------------------------------------------------------------------ #
    if not args.headless:
        sim.open_window()
    if not args.no_vis_eef_axis:
        draw_start_eef_axis(sim, robot)
    if not args.auto_play:
        input("Inspect the robot, then press Enter to plan MoveJoints...")

    # ------------------------------------------------------------------ #
    # Step 6: Plan the declared (name, typed_target) sequence             #
    # ------------------------------------------------------------------ #
    logger.log_info(
        "Planning MoveJoints: NamedJointPositionTarget('ready') -> explicit home qpos"
    )
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("move_joints", NamedJointPositionTarget(name="ready")),
            ("move_joints", JointPositionTarget(qpos=home_qpos)),
        ]
    )
    if not is_success:
        logger.log_warning("Failed to plan MoveJoints demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the MoveJoints demo...")

    # ------------------------------------------------------------------ #
    # Step 7: Replay the planned trajectory                               #
    # ------------------------------------------------------------------ #
    recording_started = start_auto_play_recording(
        sim, args, video_prefix="move_joints_auto_play"
    )
    try:
        for i in range(traj.shape[1]):
            robot.set_qpos(traj[:, i, :])
            sim.update(step=4)
            time.sleep(1e-2)

        final_qpos = traj[:, -1, :]
        for _ in range(POST_TRAJECTORY_STEPS):
            robot.set_qpos(final_qpos)
            sim.update(step=2)
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if not args.auto_play:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
