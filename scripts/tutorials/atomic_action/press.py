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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    Press,
    PressCfg,
)
from embodichain.lab.sim.cfg import (
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    broadcast_pose_batch,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    draw_axis_marker,
    format_tensor,
    get_hand_open_close_qpos,
    make_top_down_eef_pose,
    prepare_tutorial_scene,
    replay_trajectory,
)

MOVE_SAMPLE_INTERVAL = 60
PRESS_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 180
BLOCK_SIZE = (0.12, 0.12, 0.06)
PRESS_CLEARANCE = 0.13
PRESS_SURFACE_OFFSET = 0.003
DEFAULT_PRESS_TOLERANCE = 0.01


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Press tutorial."""
    parser = argparse.ArgumentParser(description="Demonstrate Press on a wooden block.")
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument("--debug_state", action="store_true")
    parser.add_argument(
        "--press_tolerance", type=float, default=DEFAULT_PRESS_TOLERANCE
    )
    parser.add_argument("--block_pos", type=float, nargs=2, default=(-0.30, -0.12))
    parser.add_argument("--no_vis_eef_axis", action="store_true")
    return parser.parse_args()


def create_wooden_block(sim, center: list[float]) -> RigidObject:
    """Create the static block used as a press target."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="wooden_block",
            shape=CubeCfg(
                size=list(BLOCK_SIZE),
                visual_material=VisualMaterialCfg(
                    uid="wooden_block_mat",
                    base_color=[0.58, 0.32, 0.14, 1.0],
                    roughness=0.85,
                ),
            ),
            body_type="static",
            attrs=RigidBodyAttributesCfg(dynamic_friction=0.8, static_friction=0.9),
            init_pos=center,
        )
    )


def compute_press_center_check(
    robot,
    trajectory: torch.Tensor,
    block: RigidObject,
    tolerance: float,
) -> tuple[bool, float, int, torch.Tensor, torch.Tensor]:
    """Return whether the press trajectory reaches the block center tolerance."""
    arm_joint_ids = robot.get_joint_ids(name="arm")
    start = MOVE_SAMPLE_INTERVAL + HAND_INTERP_STEPS
    arm_traj = trajectory[
        :, start : MOVE_SAMPLE_INTERVAL + PRESS_SAMPLE_INTERVAL, arm_joint_ids
    ]
    fk_pose = torch.stack(
        [
            robot.compute_fk(qpos=qpos, name="arm", to_matrix=True)
            for qpos in arm_traj.unbind(dim=1)
        ],
        dim=1,
    )
    block_center = block.get_local_pose(to_matrix=True)[:, :3, 3]
    target_z = block_center[:, 2] + 0.5 * BLOCK_SIZE[2] + PRESS_SURFACE_OFFSET
    xy_error = torch.linalg.norm(
        fk_pose[:, :, :2, 3] - block_center[:, None, :2], dim=2
    )
    z_error = torch.abs(fk_pose[:, :, 2, 3] - target_z[:, None])
    best_idx = (xy_error + z_error).argmin(dim=1)
    env_idx = torch.arange(trajectory.shape[0], device=trajectory.device)
    best_pos = fk_pose[env_idx, best_idx, :3, 3]
    center_error = torch.linalg.norm(best_pos[:, :2] - block_center[:, :2], dim=1)
    worst_env = int(center_error.argmax().item())
    expected = torch.stack(
        [block_center[worst_env, 0], block_center[worst_env, 1], target_z[worst_env]]
    )
    return (
        bool(torch.all(center_error <= tolerance)),
        float(center_error[worst_env].item()),
        start + int(best_idx[worst_env].item()),
        best_pos[worst_env],
        expected,
    )


def main() -> None:
    """Plan, verify, and replay MoveEndEffector followed by Press."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    block = create_wooden_block(sim, [*args.block_pos, 0.5 * BLOCK_SIZE[2]])
    if sim.device.type == "cuda":
        sim.init_gpu_physics()
    block.reset()
    sim.update(step=5)
    block.clear_dynamics()

    motion_gen = create_toppra_motion_generator(robot)
    hand_close = get_hand_open_close_qpos(robot)[1]
    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(
        MoveEndEffector(
            motion_gen, MoveEndEffectorCfg(sample_interval=MOVE_SAMPLE_INTERVAL)
        )
    )
    engine.register(
        Press(
            motion_gen,
            PressCfg(
                hand_close_qpos=hand_close,
                sample_interval=PRESS_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )

    block_center = block.get_local_pose(to_matrix=True)[0, :3, 3]
    press_position = block_center.clone()
    press_position[2] += 0.5 * BLOCK_SIZE[2] + PRESS_SURFACE_OFFSET
    move_position = press_position.clone()
    move_position[2] += PRESS_CLEARANCE - PRESS_SURFACE_OFFSET
    n_envs = robot.get_qpos().shape[0]
    move_target = broadcast_pose_batch(make_top_down_eef_pose(move_position), n_envs)
    press_target = broadcast_pose_batch(make_top_down_eef_pose(press_position), n_envs)
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "press_target_axis", press_target)
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the wooden block, then press Enter to plan..."
    )

    success, trajectory, _ = engine.run(
        [
            ("move_end_effector", EndEffectorPoseTarget(move_target)),
            ("press", EndEffectorPoseTarget(press_target)),
        ]
    )
    if not success.all():
        logger.log_warning("Failed to plan Press demo trajectory.")
        return
    is_center_hit, center_error, hit_step, hit_pos, expected_pos = (
        compute_press_center_check(robot, trajectory, block, args.press_tolerance)
    )
    logger.log_info(
        "Press center check: "
        f"success={is_center_hit}, xy_error={center_error:.4f} m, hit_step={hit_step}, "
        f"hit_pos={format_tensor(hit_pos)}, expected={format_tensor(expected_pos)}"
    )
    if not is_center_hit:
        logger.log_warning(
            "Press trajectory did not reach the block center within tolerance."
        )
        return

    if wait_for_user:
        input("Press Enter to replay the Press demo...")

    def log_state(step_idx: int, total_steps: int) -> None:
        if args.debug_state and (
            step_idx % max(1, total_steps // 10) == 0 or step_idx == total_steps - 1
        ):
            logger.log_info(
                f"replay step {step_idx}/{total_steps - 1}: "
                f"pos={format_tensor(block.get_local_pose(to_matrix=True)[0, :3, 3])}"
            )

    replay_trajectory(
        sim,
        robot,
        trajectory,
        args,
        video_prefix="press_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=log_state,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
