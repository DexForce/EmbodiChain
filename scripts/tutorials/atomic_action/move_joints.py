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

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    JointPositionTarget,
    MoveJoints,
    MoveJointsCfg,
    NamedJointPositionTarget,
)
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
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

MOVE_JOINTS_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


class MoveJointsDemo(DemoBase):
    """Demo that moves a UR5 arm through named and explicit joint targets."""

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

        ready_qpos = self._make_arm_qpos([0.35, -1.20, 1.30, -1.65, -1.57, 0.20])
        mid_qpos = self._make_arm_qpos([0.15, -1.40, 1.45, -1.60, -1.57, 0.10])
        home_qpos = self._make_arm_qpos([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        move_joints_cfg = MoveJointsCfg(
            control_part="arm",
            sample_interval=MOVE_JOINTS_SAMPLE_INTERVAL,
            named_joint_positions={"ready": ready_qpos},
        )

        self.atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
        self.atomic_engine.register(MoveJoints(motion_gen, cfg=move_joints_cfg))

        self.mid_qpos = mid_qpos
        self.home_qpos = home_qpos

        maybe_open_window(self.sim, self.args)
        if not self.args.no_vis_eef_axis:
            self._draw_start_eef_axis()

    def run(self) -> None:
        """Plan and replay the MoveJoints trajectory."""
        maybe_wait_for_user(
            self.args, "Inspect the robot, then press Enter to plan MoveJoints..."
        )

        n_envs = self.robot.get_qpos().shape[0]
        multi_waypoint_qpos = (
            torch.stack([self.mid_qpos, self.home_qpos], dim=0)
            .unsqueeze(0)
            .repeat(n_envs, 1, 1)
        )
        logger.log_info(
            "Planning MoveJoints: NamedJointPositionTarget('ready') -> "
            "multi-waypoint trajectory (mid -> home)"
        )
        is_success, traj, _ = self.atomic_engine.run(
            steps=[
                ("move_joints", NamedJointPositionTarget(name="ready")),
                ("move_joints", JointPositionTarget(qpos=multi_waypoint_qpos)),
            ]
        )
        if not is_success:
            logger.log_warning("Failed to plan MoveJoints demo trajectory.")
            return

        maybe_wait_for_user(self.args, "Press Enter to replay the MoveJoints demo...")

        with DemoRecording(self.sim, self.args, prefix="move_joints_auto_play"):
            replay_trajectory(
                self.sim,
                self.robot,
                traj,
                post_steps=POST_TRAJECTORY_STEPS,
                step_size=4,
                sleep=1e-2,
            )

        maybe_wait_for_user(self.args, "Press Enter to exit the simulation...")

    def _make_arm_qpos(self, values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.sim.device)

    def _draw_start_eef_axis(self) -> None:
        eef_pose = self.robot.compute_fk(
            qpos=self.robot.get_qpos(name="arm"),
            name="arm",
            to_matrix=True,
        )
        draw_axis_marker(self.sim, "move_joints_start_eef_axis", eef_pose)


def main() -> None:
    """Entry point for the MoveJoints demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveJoints with named and explicit qpos targets."
    )
    parser = add_demo_args(parser)
    args = parser.parse_args()
    MoveJointsDemo(args).main()


if __name__ == "__main__":
    main()
