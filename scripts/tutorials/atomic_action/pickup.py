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

"""Demonstrate PickUp on a cube with a configurable approach direction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    PickUpOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_XY = (-0.42, -0.08)
PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "side": (0.0, 1.0, 0.0),
    "side_y": (0.0, -1.0, 0.0),
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the PickUp tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate PickUp on a cube.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--approach", choices=[*APPROACH_DIRECTIONS, "custom"], default="top"
    )
    parser.add_argument("--custom_approach_direction", type=float, nargs=3)
    return parser.parse_args()


def create_pick_object(sim) -> RigidObject:
    """Create a settled cube for antipodal grasp planning."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=list(OBJECT_SIZE)),
            attrs=RigidBodyAttributesCfg(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
            ),
            max_convex_hull_num=16,
            init_pos=[*OBJECT_XY, OBJECT_SIZE[2]],
        )
    )
    sim.prepare()
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def resolve_approach_direction(
    args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    """Resolve and validate a normalized approach direction."""
    direction = (
        args.custom_approach_direction
        if args.approach == "custom"
        else APPROACH_DIRECTIONS[args.approach]
    )
    if direction is None:
        raise ValueError(
            "--custom_approach_direction is required for --approach custom."
        )
    approach = torch.tensor(direction, dtype=torch.float32, device=device)
    if torch.linalg.norm(approach) < 1e-6:
        raise ValueError("approach_direction must be non-zero.")
    return torch.nn.functional.normalize(approach, dim=0)


def main() -> None:
    """Plan and replay a sampled antipodal PickUp trajectory."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    obj = create_pick_object(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    motion_gen = create_curobo_motion_generator(robot)

    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    semantics = create_antipodal_semantics(
        obj,
        label="cube",
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "pickup_object_axis", obj.get_local_pose(to_matrix=True))
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the cube, then press Enter to plan PickUp..."
    )

    compiled = engine.compile(
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICK_SAMPLE_INTERVAL,
                ),
                skill_options=PickUpOptions(
                    approach_direction=resolve_approach_direction(args, sim.device),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan PickUp demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the PickUp demo...")
    clear_after_step = compiled.segment(0, "lift").start
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="pickup_cube_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
