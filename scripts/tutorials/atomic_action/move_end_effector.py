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

"""Demonstrate MoveEndEffector with a multi-waypoint pose trajectory."""

from __future__ import annotations

import argparse

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    format_tensor,
    maybe_open_window,
    maybe_wait_for_user,
    replay_trajectory,
    setup_print_options,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_ur5_gripper_robot_cfg,
    draw_axis_marker,
    get_tutorial_window_size,
)

MOVE_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


class MoveEndEffectorDemo(DemoBase):
    """Demo that moves a UR5 end effector through a multi-waypoint trajectory."""

    def setup(self) -> None:
        """Create simulation, robot, motion generator and atomic action engine."""
        width, height = get_tutorial_window_size(self.args)
        self.sim = create_default_sim(
            self.args,
            width=width,
            height=height,
            physics_dt=1.0 / 100.0,
            arena_space=2.5,
        )
        self.robot = self.sim.add_robot(
            cfg=create_ur5_gripper_robot_cfg(init_pos=(0.0, 0.0, 0.0))
        )
        motion_gen = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )

        move_cfg = MoveEndEffectorCfg(
            control_part="arm",
            sample_interval=MOVE_SAMPLE_INTERVAL,
        )

        self.atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
        self.atomic_engine.register(MoveEndEffector(motion_gen, cfg=move_cfg))

        self.target_pose = self._make_top_down_eef_pose()
        self.side_pose = self._make_side_eef_pose()

        maybe_open_window(self.sim, self.args)
        if not self.args.no_vis_eef_axis:
            draw_axis_marker(
                self.sim, "move_end_effector_target_axis", self.target_pose
            )
            draw_axis_marker(self.sim, "move_end_effector_side_axis", self.side_pose)

    def run(self) -> None:
        """Plan and replay the MoveEndEffector trajectory."""
        maybe_wait_for_user(
            self.args, "Inspect the robot, then press Enter to plan MoveEndEffector..."
        )

        n_envs = self.robot.get_qpos().shape[0]
        multi_waypoint_xpos = (
            torch.stack([self.target_pose, self.side_pose], dim=0)
            .unsqueeze(0)
            .repeat(n_envs, 1, 1, 1)
        )
        logger.log_info(
            "Planning MoveEndEffector through multi-waypoint trajectory: "
            f"xpos0={format_tensor(self.target_pose[:3, 3])} -> "
            f"xpos1={format_tensor(self.side_pose[:3, 3])}"
        )
        is_success, traj, _ = self.atomic_engine.run(
            steps=[
                (
                    "move_end_effector",
                    EndEffectorPoseTarget(xpos=multi_waypoint_xpos),
                )
            ]
        )
        if not is_success:
            logger.log_warning("Failed to plan MoveEndEffector demo trajectory.")
            return

        maybe_wait_for_user(
            self.args, "Press Enter to replay the MoveEndEffector demo..."
        )

        with DemoRecording(self.sim, self.args, prefix="move_end_effector_auto_play"):
            replay_trajectory(
                self.sim,
                self.robot,
                traj,
                post_steps=POST_TRAJECTORY_STEPS,
                step_size=4,
                sleep=1e-2,
            )

        maybe_wait_for_user(self.args, "Press Enter to exit the simulation...")

    def _make_top_down_eef_pose(self) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32, device=self.sim.device)
        pose[:3, :3] = torch.tensor(
            [
                [-0.0539, -0.9985, -0.0022],
                [-0.9977, 0.0540, -0.0401],
                [0.0401, 0.0000, -0.9992],
            ],
            dtype=torch.float32,
            device=self.sim.device,
        )
        pose[:3, 3] = torch.tensor(
            [0.30, -0.20, 0.36], dtype=torch.float32, device=self.sim.device
        )
        return pose

    def _make_side_eef_pose(self) -> torch.Tensor:
        """A second waypoint offset from the top-down pose for the multi-waypoint demo."""
        pose = torch.eye(4, dtype=torch.float32, device=self.sim.device)
        pose[:3, :3] = torch.tensor(
            [
                [-0.0539, -0.9985, -0.0022],
                [-0.9977, 0.0540, -0.0401],
                [0.0401, 0.0000, -0.9992],
            ],
            dtype=torch.float32,
            device=self.sim.device,
        )
        pose[:3, 3] = torch.tensor(
            [0.45, 0.10, 0.30], dtype=torch.float32, device=self.sim.device
        )
        return pose


def main() -> None:
    """Entry point for the MoveEndEffector demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveEndEffector with a multi-waypoint pose trajectory."
    )
    parser = add_demo_args(parser)
    args = parser.parse_args()
    MoveEndEffectorDemo(args).main()


if __name__ == "__main__":
    main()
