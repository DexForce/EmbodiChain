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

"""Demonstrate moving a held object to an object-centric target pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    PickUp,
    PickUpCfg,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    broadcast_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    make_eef_pose_at,
    prepare_tutorial_scene,
    replay_trajectory,
)

OBJECT_MESH_PATH = "PaperCup/paper_cup.ply"
OBJECT_XY = (-0.42, -0.08)
MOVE_SAMPLE_INTERVAL = 60
PICK_SAMPLE_INTERVAL = 120
MOVE_HELD_OBJECT_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MoveHeldObject tutorial."""
    parser = argparse.ArgumentParser(
        description="Pick up a paper cup and move it by object pose."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--force_reannotate", action="store_true")
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument("--no_vis_eef_axis", action="store_true")
    return parser.parse_args()


def create_pick_object(sim) -> RigidObject:
    """Create and settle the paper cup used by the tutorial."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="paper_cup",
            shape=MeshCfg(fpath=get_data_path(OBJECT_MESH_PATH)),
            attrs=RigidBodyAttributesCfg(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
            ),
            max_convex_hull_num=16,
            init_pos=[*OBJECT_XY, 0.0],
            body_scale=(0.75, 0.75, 1.0),
        )
    )
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def make_object_target_pose(device: torch.device) -> torch.Tensor:
    """Build the desired final paper-cup pose in the object frame."""
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = torch.tensor([-0.3, -0.3, 0.5], device=device)
    return pose


def main() -> None:
    """Plan MoveEndEffector -> PickUp -> MoveHeldObject."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    obj = create_pick_object(sim)
    motion_gen = create_toppra_motion_generator(robot)
    hand_open, hand_close = get_hand_open_close_qpos(robot)

    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(
        MoveEndEffector(
            motion_gen, MoveEndEffectorCfg(sample_interval=MOVE_SAMPLE_INTERVAL)
        )
    )
    engine.register(
        PickUp(
            motion_gen,
            PickUpCfg(
                hand_open_qpos=hand_open,
                hand_close_qpos=hand_close,
                pre_grasp_distance=0.15,
                lift_height=0.16,
                sample_interval=PICK_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )
    engine.register(
        MoveHeldObject(
            motion_gen,
            MoveHeldObjectCfg(
                hand_close_qpos=hand_close,
                sample_interval=MOVE_HELD_OBJECT_SAMPLE_INTERVAL,
            ),
        )
    )

    semantics = create_antipodal_semantics(
        obj,
        label="paper_cup",
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    move_position = obj.get_local_pose(to_matrix=True)[0, :3, 3].clone()
    move_position[2] = 0.36
    n_envs = robot.get_qpos().shape[0]
    move_target = broadcast_pose_batch(make_eef_pose_at(robot, move_position), n_envs)
    object_target = broadcast_pose_batch(make_object_target_pose(sim.device), n_envs)
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "move_held_object_target_axis", object_target)
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the paper cup, then press Enter to plan..."
    )

    success, trajectory, _ = engine.run(
        [
            ("move_end_effector", EndEffectorPoseTarget(move_target)),
            ("pick_up", GraspTarget(semantics)),
            ("move_held_object", HeldObjectPoseTarget(object_target)),
        ]
    )
    if not success.all():
        logger.log_warning("Failed to plan MoveHeldObject demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the MoveHeldObject demo...")
    clear_after_step = (
        MOVE_SAMPLE_INTERVAL
        + round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6)
        + HAND_INTERP_STEPS
    )
    dynamics_cleared = False

    def clear_object_dynamics(step_idx: int, _: int) -> None:
        nonlocal dynamics_cleared
        if not dynamics_cleared and step_idx + 1 >= clear_after_step:
            obj.clear_dynamics()
            dynamics_cleared = True

    replay_trajectory(
        sim,
        robot,
        trajectory,
        args,
        video_prefix="move_held_object_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=clear_object_dynamics,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
