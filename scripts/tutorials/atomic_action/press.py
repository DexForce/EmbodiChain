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

"""Demonstrate Press on the center of a regular wooden block."""

from __future__ import annotations

import argparse
import time

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    Press,
    PressCfg,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    format_tensor,
    maybe_init_gpu_physics,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    draw_axis_marker,
    get_tutorial_window_size,
    make_ur5_solver_cfg,
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"
ARM_JOINT_PATTERN = "joint[0-9]"
GRIPPER_TCP_Z = 0.15

MOVE_SAMPLE_INTERVAL = 60
PRESS_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 180
TABLE_SIZE = [1.0, 1.4, 0.05]
TABLE_TOP_Z = -0.045
BLOCK_SIZE = [0.12, 0.12, 0.06]
BLOCK_CENTER = [-0.30, -0.12, TABLE_TOP_Z + 0.5 * BLOCK_SIZE[2]]
PRESS_CLEARANCE = 0.13
PRESS_SURFACE_OFFSET = 0.003
DEFAULT_PRESS_TOLERANCE = 0.01


class PressDemo(DemoBase):
    """Demo that presses the center of a regular wooden block."""

    def setup(self) -> None:
        """Create simulation, custom robot, table, block and action engine."""
        width, height = get_tutorial_window_size(self.args)
        self.sim = create_default_sim(
            self.args,
            width=width,
            height=height,
            physics_dt=1.0 / 100.0,
            arena_space=2.5,
        )
        self.robot = self._create_robot()
        self._create_table()
        block_center = [
            self.args.block_pos[0],
            self.args.block_pos[1],
            TABLE_TOP_Z + 0.5 * BLOCK_SIZE[2],
        ]
        self.block = self._create_wooden_block(block_center)

        self._settle_object(self.block, step=5)
        motion_gen = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        hand_close = self._get_hand_close_qpos()

        self.atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
        self.atomic_engine.register(
            MoveEndEffector(
                motion_gen,
                cfg=MoveEndEffectorCfg(
                    control_part="arm",
                    sample_interval=MOVE_SAMPLE_INTERVAL,
                ),
            )
        )
        self.atomic_engine.register(
            Press(
                motion_gen,
                cfg=PressCfg(
                    control_part="arm",
                    hand_control_part="hand",
                    hand_close_qpos=hand_close,
                    sample_interval=PRESS_SAMPLE_INTERVAL,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            )
        )

        block_pose = self.block.get_local_pose(to_matrix=True)
        block_top_z = block_pose[0, 2, 3] + 0.5 * BLOCK_SIZE[2]
        press_position = block_pose[0, :3, 3].clone()
        press_position[2] = block_top_z + PRESS_SURFACE_OFFSET
        move_position = press_position.clone()
        move_position[2] = block_top_z + PRESS_CLEARANCE

        self.move_target = self._make_top_down_eef_pose(move_position)
        self.press_target = self._make_top_down_eef_pose(press_position)

        maybe_open_window(self.sim, self.args)
        if not self.args.no_vis_eef_axis:
            self._draw_press_target_axis(self.press_target)

    def run(self) -> None:
        """Plan and replay the MoveEndEffector -> Press trajectory."""
        maybe_wait_for_user(
            self.args, "Inspect the wooden block, then press Enter to plan..."
        )

        logger.log_info("Planning MoveEndEffector -> Press")
        start_time = time.time()
        is_success, traj, _ = self.atomic_engine.run(
            steps=[
                ("move_end_effector", EndEffectorPoseTarget(xpos=self.move_target)),
                ("press", EndEffectorPoseTarget(xpos=self.press_target)),
            ]
        )
        cost_time = time.time() - start_time
        logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
        if not is_success:
            logger.log_warning("Failed to plan Press demo trajectory.")
            return

        is_center_hit, center_error, hit_step, hit_pos, expected_pos = (
            self._compute_press_center_check(traj)
        )
        logger.log_info(
            "Press center check: "
            f"success={is_center_hit}, "
            f"xy_error={center_error:.4f} m, "
            f"hit_step={hit_step}, "
            f"hit_pos={format_tensor(hit_pos)}, "
            f"expected={format_tensor(expected_pos)}"
        )
        if not is_center_hit:
            logger.log_warning(
                "Press planned trajectory did not reach the block center within "
                f"{self.args.press_tolerance:.4f} m."
            )
            return

        maybe_wait_for_user(self.args, "Press Enter to replay the Press demo...")

        with DemoRecording(self.sim, self.args, prefix="press_auto_play"):
            self._replay_press_trajectory(traj)

        maybe_wait_for_user(self.args, "Press Enter to exit the simulation...")

    def _create_robot(self) -> Robot:
        ur5_urdf_path = get_data_path("UniversalRobots/UR5/UR5.urdf")
        gripper_urdf_path = get_data_path(GRIPPER_URDF_PATH)
        cfg = RobotCfg(
            uid="UR5",
            urdf_cfg=URDFCfg(
                components=[
                    {"component_type": "arm", "urdf_path": ur5_urdf_path},
                    {"component_type": "hand", "urdf_path": gripper_urdf_path},
                ]
            ),
            drive_pros=JointDrivePropertiesCfg(
                stiffness={ARM_JOINT_PATTERN: 1e4, GRIPPER_HAND_JOINT_PATTERN: 1e3},
                damping={ARM_JOINT_PATTERN: 1e3, GRIPPER_HAND_JOINT_PATTERN: 1e2},
                max_effort={ARM_JOINT_PATTERN: 1e5, GRIPPER_HAND_JOINT_PATTERN: 1e4},
                drive_type="force",
            ),
            control_parts={
                "arm": [ARM_JOINT_PATTERN],
                "hand": [GRIPPER_HAND_JOINT_PATTERN],
            },
            solver_cfg={"arm": make_ur5_solver_cfg(GRIPPER_TCP_Z)},
            init_qpos=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
            init_pos=(0.0, 0.0, 0.0),
        )
        return self.sim.add_robot(cfg=cfg)

    def _create_table(self) -> RigidObject:
        cfg = RigidObjectCfg(
            uid="table",
            shape=CubeCfg(size=TABLE_SIZE),
            body_type="static",
            attrs=RigidBodyAttributesCfg(
                dynamic_friction=0.8,
                static_friction=0.9,
            ),
            init_pos=[-0.30, 0.10, TABLE_TOP_Z - 0.5 * TABLE_SIZE[2]],
        )
        return self.sim.add_rigid_object(cfg=cfg)

    def _create_wooden_block(self, block_center: list[float]) -> RigidObject:
        cfg = RigidObjectCfg(
            uid="wooden_block",
            shape=CubeCfg(
                size=BLOCK_SIZE,
                visual_material=VisualMaterialCfg(
                    uid="wooden_block_mat",
                    base_color=[0.58, 0.32, 0.14, 1.0],
                    roughness=0.85,
                ),
            ),
            body_type="static",
            attrs=RigidBodyAttributesCfg(
                dynamic_friction=0.8,
                static_friction=0.9,
            ),
            init_pos=block_center,
        )
        return self.sim.add_rigid_object(cfg=cfg)

    def _settle_object(self, obj: RigidObject, step: int = 5) -> None:
        maybe_init_gpu_physics(self.sim)
        obj.reset()
        self.sim.update(step=step)
        obj.clear_dynamics()

    def _get_hand_close_qpos(self) -> torch.Tensor:
        hand_limits = self.robot.get_qpos_limits(name="hand")[0].to(
            device=self.sim.device, dtype=torch.float32
        )
        return hand_limits[:, 1]

    def _make_top_down_eef_pose(self, position: torch.Tensor) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32, device=position.device)
        pose[:3, :3] = torch.tensor(
            [
                [-0.0539, -0.9985, -0.0022],
                [-0.9977, 0.0540, -0.0401],
                [0.0401, 0.0000, -0.9992],
            ],
            dtype=torch.float32,
            device=position.device,
        )
        pose[:3, 3] = position
        return pose

    def _draw_press_target_axis(self, press_target: torch.Tensor) -> None:
        if press_target.dim() == 2:
            press_target = press_target.unsqueeze(0)
        draw_axis_marker(self.sim, "press_target_axis", press_target)

    def _compute_press_center_check(
        self, traj: torch.Tensor
    ) -> tuple[bool, float, int, torch.Tensor, torch.Tensor]:
        arm_joint_ids = self.robot.get_joint_ids(name="arm")
        press_segment_start = MOVE_SAMPLE_INTERVAL + HAND_INTERP_STEPS
        press_segment_end = MOVE_SAMPLE_INTERVAL + PRESS_SAMPLE_INTERVAL
        arm_traj = traj[:, press_segment_start:press_segment_end, arm_joint_ids]
        fk_pose = torch.stack(
            [
                self.robot.compute_fk(
                    qpos=waypoint.unsqueeze(0),
                    name="arm",
                    to_matrix=True,
                )[0]
                for waypoint in arm_traj[0]
            ],
            dim=0,
        )

        block_pose = self.block.get_local_pose(to_matrix=True)
        block_center = block_pose[0, :3, 3]
        block_top_z = block_center[2] + 0.5 * BLOCK_SIZE[2]
        target_xy = block_center[:2]
        target_z = block_top_z + PRESS_SURFACE_OFFSET

        xy_error = torch.linalg.norm(fk_pose[:, :2, 3] - target_xy, dim=1)
        z_error = torch.abs(fk_pose[:, 2, 3] - target_z)
        combined_error = xy_error + z_error
        best_idx = int(torch.argmin(combined_error).item())
        best_pos = fk_pose[best_idx, :3, 3]
        center_error = float(torch.linalg.norm(best_pos[:2] - target_xy).item())
        return (
            center_error <= self.args.press_tolerance,
            center_error,
            press_segment_start + best_idx,
            best_pos,
            torch.tensor(
                [target_xy[0], target_xy[1], target_z],
                dtype=torch.float32,
                device=traj.device,
            ),
        )

    def _replay_press_trajectory(self, traj: torch.Tensor) -> None:
        log_stride = max(1, traj.shape[1] // 10)
        for i in range(traj.shape[1]):
            self.robot.set_qpos(traj[:, i, :])
            self.sim.update(step=4)
            if self.args.debug_state and (
                i % log_stride == 0 or i == traj.shape[1] - 1
            ):
                self._log_block_state(f"replay step {i}/{traj.shape[1] - 1}")
            time.sleep(1e-2)

        logger.log_info("Press returned the end-effector to the starting pose.")
        final_qpos = traj[:, -1, :]
        for i in range(POST_TRAJECTORY_STEPS):
            self.robot.set_qpos(final_qpos)
            self.sim.update(step=2)
            if self.args.debug_state and i % max(1, POST_TRAJECTORY_STEPS // 5) == 0:
                self._log_block_state(f"post step {i}/{POST_TRAJECTORY_STEPS - 1}")
            time.sleep(1e-2)

    def _log_block_state(self, label: str) -> None:
        block_pose = self.block.get_local_pose(to_matrix=True)
        logger.log_info(
            f"{label}: pos={format_tensor(block_pose[0, :3, 3])}, "
            f"z_axis={format_tensor(block_pose[0, :3, 2])}"
        )


def main() -> None:
    """Entry point for the Press demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Demonstrate Press on the center of a regular wooden block."
    )
    parser = add_demo_args(parser)
    parser.add_argument(
        "--debug_state",
        action="store_true",
        help="Log the block pose during replay.",
    )
    parser.add_argument(
        "--press_tolerance",
        type=float,
        default=DEFAULT_PRESS_TOLERANCE,
        help="XY tolerance in meters for checking whether press reaches block center.",
    )
    parser.add_argument(
        "--block_pos",
        type=float,
        nargs=2,
        default=BLOCK_CENTER[:2],
        metavar=("X", "Y"),
        help="Initial XY position of the wooden block center.",
    )
    args = parser.parse_args()
    PressDemo(args).main()


if __name__ == "__main__":
    main()
