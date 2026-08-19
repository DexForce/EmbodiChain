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

"""Demonstrate Slide on a translating drawer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    ControlPartCommandProfile,
    EntityState,
    MotionPolicy,
    ObjectSemantics,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
    SceneEntityPose,
    SceneSnapshot,
)
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.objects import Articulation
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

DRAWER_ASSET = "Drawer/model_split_links_with_inertials.urdf"
HANDLE_LINK_NAME = "large_handle_bar"
DRAWER_POSITION = (-1.1, 0.0, 0.0)
DRAWER_ORIENTATION = (0.0, 0.0, 90.0)  # degrees
TRANSLATION_AXIS = (0.0, 1.0, 0.0)  # handle-link frame, approach/push direction
TRAJECTORY_SAMPLE_COUNT = 140
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
HANDLE_SCENE_ENTITY_ID = "drawer-large-handle"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the drawer pull/push tutorial."""
    parser = create_tutorial_argument_parser(
        "Pull a drawer open, then push it closed with Slide.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument("--translation_distance", type=float, default=0.18)
    parser.add_argument("--approach_distance", type=float, default=0.10)
    return parser.parse_args()


def create_drawer(
    sim: SimulationManager,
) -> Articulation:
    """Create the fixed-base drawer in its closed initial state."""
    drawer = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="drawer",
            fpath=get_data_path(DRAWER_ASSET),
            init_pos=DRAWER_POSITION,
            init_rot=DRAWER_ORIENTATION,
            init_qpos=(0.0,),
            drive_pros=JointDrivePropertiesCfg(drive_type="none"),
            attrs=RigidBodyAttributesCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            fix_base=True,
        )
    )
    sim.update(step=10)
    return drawer


def create_drawer_semantics(
    drawer: Articulation,
    *,
    n_sample: int,
    force_reannotate: bool,
) -> ObjectSemantics:
    """Create sampled-grasp translation semantics for the drawer handle.

    Args:
        drawer: Drawer articulation that owns the target handle link.
        n_sample: Number of antipodal surface samples.
        force_reannotate: Whether to ignore a cached grasp annotation.

    Returns:
        Pure target-local semantics for the handle's pull/push affordance.
    """
    vertices, triangles = drawer.get_link_vert_face(HANDLE_LINK_NAME)
    return ObjectSemantics(
        label="drawer_large_handle",
        geometry={},
        entity_id=HANDLE_SCENE_ENTITY_ID,
        affordance=SlideAffordance(
            mesh_vertices=torch.as_tensor(vertices),
            mesh_triangles=torch.as_tensor(triangles),
            translation_axis=torch.tensor(
                TRANSLATION_AXIS,
                dtype=torch.float32,
                device=drawer.device,
            ),
            generator_cfg=GraspGeneratorCfg(
                antipodal_sampler_cfg=AntipodalSamplerCfg(
                    n_sample=n_sample,
                    max_length=0.1,
                    min_length=0.003,
                ),
                is_partial_annotate=False,
                is_filter_ground_collision=False,
            ),
            gripper_collision_cfg=GripperCollisionCfg(
                max_open_length=0.1,
                finger_length=0.1,
                y_thickness=0.04,
                root_z_width=0.096,
                open_check_margin=0.03,
                point_sample_dense=0.012,
            ),
            force_reannotate=force_reannotate,
        ),
    )


def create_invocation(
    semantics: ObjectSemantics,
    *,
    direction: Literal["pull", "push"],
    approach_distance: float,
    translation_distance: float,
) -> ActionInvocation:
    """Create one pull or push invocation for the shared drawer target.

    Args:
        semantics: Drawer-handle semantics shared by both operations.
        direction: Whether this invocation pulls open or pushes closed.
        approach_distance: Pre-grasp offset opposite the approach axis.
        translation_distance: Drawer travel distance for this operation.

    Returns:
        A grounded pull/push invocation for the tutorial UR5.
    """
    return ActionInvocation(
        skill_id="slide",
        goal=SlideGoal(
            semantics,
            SceneEntityPose(HANDLE_SCENE_ENTITY_ID),
        ),
        binding=ActionBinding(
            manipulators={"primary": "arm"},
            end_effectors={"primary": "hand"},
        ),
        motion_policy=MotionPolicy(sample_count=TRAJECTORY_SAMPLE_COUNT),
        skill_options=SlideOptions(
            direction=direction,
            hand_interp_steps=HAND_INTERP_STEPS,
            approach_distance=approach_distance,
            translation_distance=translation_distance,
        ),
    )


def main() -> None:
    """Plan and replay a drawer pull followed by a push."""
    args = parse_arguments()
    if args.translation_distance <= 0.0:
        raise ValueError("--translation_distance must be positive.")
    if args.translation_distance > 0.285:
        raise ValueError(
            "--translation_distance must not exceed the drawer limit 0.285."
        )

    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0], tcp_z=0.15
    )
    drawer = create_drawer(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    motion_gen = create_toppra_motion_generator(robot)
    semantics = create_drawer_semantics(
        drawer,
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    affordance = semantics.affordance
    assert isinstance(affordance, SlideAffordance)
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "drawer_handle_link_pose",
            drawer.get_link_pose(HANDLE_LINK_NAME, to_matrix=True),
        )

    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the closed drawer, then press Enter to plan the pull...",
    )

    for scene_version, direction in enumerate(("pull", "push")):
        if direction == "push" and wait_for_user:
            input(
                "Pull replay finished. Press Enter to read the moved handle "
                "pose and plan the push..."
            )

        handle_pose = drawer.get_link_pose(HANDLE_LINK_NAME, to_matrix=True)
        compiled = engine.compile(
            (
                create_invocation(
                    semantics,
                    direction=direction,
                    approach_distance=args.approach_distance,
                    translation_distance=args.translation_distance,
                ),
            ),
            context=engine.initial_context(
                scene=SceneSnapshot(
                    timestamp=float(scene_version),
                    version=scene_version,
                    entities={
                        HANDLE_SCENE_ENTITY_ID: EntityState(handle_pose),
                    },
                )
            ),
        )
        if not compiled.plan_success.all():
            logger.log_warning(f"Failed to plan the Slide {direction} trajectory.")
            return

        if wait_for_user:
            input(f"Press Enter to replay the drawer {direction}...")
        replay_trajectory(
            sim,
            robot,
            compiled.trajectory,
            args,
            video_prefix=f"{direction}_drawer_auto_play",
            hold_steps=POST_TRAJECTORY_STEPS,
            look_at=(
                (-1.35, -1.15, 1.0),
                (-0.55, -0.2, 0.45),
                (0.0, 0.0, 1.0),
            ),
        )

    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
