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

"""Demonstrate AxisAlign on the same cube and robot used by PickUp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from typing import Sequence
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
    ControlPartCommandProfile,
    MotionPolicy,
    ObjectSemantics,
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
DEFAULT_INTERNAL_AXIS = (1.0, 0.0, 0.0)
DEFAULT_TARGET_AXIS = (0.0, 0.0, 1.0)
HORIZONTAL_TARGET_AXIS = (0.0, 1.0, 0.0)
ALIGNMENT_AXES = {
    "upright": (DEFAULT_INTERNAL_AXIS, DEFAULT_TARGET_AXIS),
    "horizontal_align": (DEFAULT_INTERNAL_AXIS, HORIZONTAL_TARGET_AXIS),
}
ALIGN_SAMPLE_INTERVAL = 180
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the AxisAlign tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate upright or horizontal AxisAlign on a cube.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--alignment",
        choices=tuple(ALIGNMENT_AXES),
        default="upright",
        help="Choose the object-axis alignment example.",
    )
    return parser.parse_args()


def create_align_object(sim) -> RigidObject:
    """Create the same settled cube used by the PickUp tutorial."""
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
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def create_axis_align_semantics(
    obj: RigidObject, args: argparse.Namespace, obj_internal_axis: Sequence[float]
) -> ObjectSemantics:
    """Extend the tutorial antipodal affordance with a local alignment axis."""
    semantics = create_antipodal_semantics(
        obj,
        label="cube",
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    antipodal = semantics.affordance
    return ObjectSemantics(
        label=semantics.label,
        geometry=semantics.geometry,
        properties=semantics.properties,
        entity=semantics.entity,
        entity_id=semantics.entity_id,
        affordance=AxisAlignAffordance(
            mesh_vertices=antipodal.mesh_vertices,
            mesh_triangles=antipodal.mesh_triangles,
            generator_cfg=antipodal.generator_cfg,
            gripper_collision_cfg=antipodal.gripper_collision_cfg,
            force_reannotate=antipodal.force_reannotate,
            internal_axis=torch.tensor(obj_internal_axis, dtype=torch.float32),
        ),
    )


def main() -> None:
    """Plan and replay a grasp, axis alignment, lowering, and release."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    obj = create_align_object(sim)
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
    # apply object internal axis to the semantics creation
    obj_internal_axis = ALIGNMENT_AXES[args.alignment][0]
    # apply target axis for the alignment skill
    align_target_axis = ALIGNMENT_AXES[args.alignment][1]
    semantics = create_axis_align_semantics(obj, args, obj_internal_axis)
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "axis_align_object_axis",
            obj.get_local_pose(to_matrix=True),
        )
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        f"Inspect the cube, then press Enter to plan {args.alignment} AxisAlign...",
    )

    compiled = engine.compile(
        (
            ActionInvocation(
                skill_id="axis_align",
                goal=AxisAlignGoal(semantics),
                binding=ActionBinding(
                    manipulators={"primary": "arm"},
                    end_effectors={"primary": "hand"},
                ),
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=ALIGN_SAMPLE_INTERVAL,
                ),
                skill_options=AxisAlignOptions(
                    target_axis=torch.tensor(
                        align_target_axis,
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    approach_direction=torch.tensor(
                        [0.0, 0.0, -1.0],
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    lower_distance=0.03,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan AxisAlign demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the AxisAlign demo...")
    clear_after_step = compiled.segment(0, "manipulate").start
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix=f"axis_align_{args.alignment}_cube_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
