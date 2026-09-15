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

"""Demonstrate PickUp on an articulated Rubik's cube with its turn joint locked.

The ``rubiks_cube_001`` asset is an articulation, not a rigid body: a
``top_turn`` revolute joint couples ``top_layer`` to ``lower_two_layers``. Two
consequences drive this tutorial:

* The USD parser rejects the asset as a rigid object, so it is spawned through
  :class:`~embodichain.lab.sim.cfg.ArticulationCfg`.
* :class:`~embodichain.lab.sim.objects.Articulation` has no whole-body
  ``get_vertices()``; grasp geometry is read per link with
  ``get_link_vert_face()``, the same source
  ``scripts/tutorials/atomic_action/slide.py`` uses for its handle affordance.
  Grasps are then sampled by the shared parallel-jaw generator.

The turn joint is locked with a stiff position drive so the cube behaves as a
single rigid body while it is grasped and lifted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ControlPartCommandProfile,
    create_simulation_atomic_action_engine,
    GraspGoal,
    MotionPolicy,
    ObjectSemantics,
    PickUpOptions,
)
from embodichain.lab.sim.cfg import ArticulationCfg, ArticulationRootPropertiesCfg
from embodichain.lab.sim.objects import Articulation
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    create_curobo_motion_generator,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

CUBE_UID = "rubiks_cube"
CUBE_EDGE = 0.0576
"""Edge length of the standard 3x3 cube in meters, measured from the asset."""

CUBE_BOTTOM_CENTER_OFFSET = (0.0, 0.028780000284314156, 0.0)
"""Root-frame offset of the cube center authored by the Scene Engine.

``_canonicalize_articulated_usdc_bottom_center`` seats articulated assets on
their bounding-box minimum along **Y**, while EmbodiChain stages are Z-up. The
asset therefore carries an ``xformOp:translate:scene_engine_bottom_center`` of
``(0, +edge/2, 0)``: its geometry straddles the root frame in Z instead of
resting on it, and sits half an edge away in Y. This constant compensates both
so the tutorial spawns a cube that rests on the table where it is asked to.
"""

DEFAULT_ASSET_PATH = str(
    Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "RubiksCube" / "rubiks_cube_001.usdc"
)
TURN_JOINT = "top_turn"
GRASP_LINK = "lower_two_layers"
"""Link whose mesh feeds antipodal grasp sampling.

``top_turn`` couples this body to ``top_layer``; it is held at zero, so the two
layers move as one. The lower body carries two of the three layers and is the
joint's parent, which makes its link frame the stable reference for grasping.
"""

LOCK_STIFFNESS = 1.0e4
LOCK_DAMPING = 1.0e3

OBJECT_XY = (-0.42, -0.08)
GRASP_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
APPROACH_DIRECTION = (0.0, 0.0, -1.0)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Rubik's-cube PickUp tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate PickUp on an articulated Rubik's cube.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--asset_path",
        default=DEFAULT_ASSET_PATH,
        help="Path to the rubiks_cube_001 USD asset.",
    )
    return parser.parse_args()


def lock_turn_joint(cube: Articulation) -> None:
    """Hold ``top_turn`` at zero so the cube grasps as one rigid body.

    The asset ships the joint with zero stiffness and zero damping, which lets
    the top layer swing freely under gravity and during the lift. A stiff
    position drive removes that degree of freedom without editing the asset.

    Args:
        cube: Spawned Rubik's-cube articulation.
    """
    joint_ids = [cube.joint_names.index(TURN_JOINT)]
    num_envs = cube.get_local_pose(to_matrix=True).shape[0]
    shape = (num_envs, len(joint_ids))
    cube.set_joint_drive(
        stiffness=torch.full(shape, LOCK_STIFFNESS, device=cube.device),
        damping=torch.full(shape, LOCK_DAMPING, device=cube.device),
        joint_ids=joint_ids,
        target_mode="position",
    )
    cube.set_qpos(torch.zeros(shape, device=cube.device), joint_ids=joint_ids)


def create_pick_object(sim, asset_path: str) -> Articulation:
    """Spawn the Rubik's cube resting on the table with its turn joint locked.

    Args:
        sim: Simulation manager that owns the articulation.
        asset_path: Filesystem path to the cube USD asset.

    Returns:
        Settled articulation ready for grasp planning.

    Raises:
        FileNotFoundError: If the asset is missing from ``asset_path``.
    """
    if not Path(asset_path).is_file():
        raise FileNotFoundError(
            f"Rubik's cube asset not found at {asset_path!r}. Pass --asset_path "
            "to point at rubiks_cube_001.usdc."
        )
    offset_x, offset_y, _ = CUBE_BOTTOM_CENTER_OFFSET
    cube = sim.add_articulation(
        cfg=ArticulationCfg(
            uid=CUBE_UID,
            fpath=asset_path,
            init_pos=[
                OBJECT_XY[0] - offset_x,
                OBJECT_XY[1] - offset_y,
                CUBE_EDGE / 2.0,
            ],
            # ArticulationCfg anchors roots to the world by default, which suits
            # drawers and doors but welds a graspable object to the table: the
            # lift plans and executes while the cube never moves.
            root_props=ArticulationRootPropertiesCfg(fixed_base=False),
        )
    )
    sim.prepare()
    lock_turn_joint(cube)
    sim.update(step=10)
    cube.clear_dynamics()
    return cube


def create_link_antipodal_semantics(
    cube: Articulation,
    link_name: str,
    *,
    label: str,
) -> ObjectSemantics:
    """Describe an articulated target using one link's antipodal geometry.

    ``create_antipodal_semantics`` reads ``get_vertices()`` and
    ``get_triangles()``, which only rigid objects expose. Articulations publish
    geometry per link instead, so grasp sampling names the link it should use.

    Args:
        cube: Spawned articulation that will be grasped.
        link_name: Link whose mesh defines the graspable surface.
        label: Human-readable object category.

    Returns:
        Object semantics carrying the link mesh on its affordance.
    """
    vertices, triangles = cube.get_link_vert_face(link_name)
    return ObjectSemantics(
        label=label,
        geometry={},
        affordance=AntipodalAffordance(
            mesh_vertices=torch.as_tensor(vertices),
            mesh_triangles=torch.as_tensor(triangles),
        ),
        entity_id=cube.uid,
    )


def cube_center_pose(cube: Articulation) -> torch.Tensor:
    """Return the cube's world center pose, compensating the authored offset.

    Args:
        cube: Spawned Rubik's-cube articulation.

    Returns:
        Batched ``(num_envs, 4, 4)`` pose whose translation is the geometric
        center of the cube rather than its root frame.
    """
    pose = cube.get_local_pose(to_matrix=True).clone()
    offset = torch.tensor(
        CUBE_BOTTOM_CENTER_OFFSET, dtype=pose.dtype, device=pose.device
    )
    pose[:, :3, 3] = pose[:, :3, 3] + offset
    return pose


def main() -> None:
    """Plan and replay a top-down PickUp of the articulated Rubik's cube."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot, tcp_z=0.15)
    cube = create_pick_object(sim, args.asset_path)
    sim.prepare()
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, cube, hand_open)
    motion_gen = create_curobo_motion_generator(
        robot,
        use_cuda_graph=args.physics != "newton",
        planner=getattr(args, "planner", "trapezoidal"),
    )
    engine = create_simulation_atomic_action_engine(
        motion_generator=motion_gen,
        scene_entities=(cube,),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
        grasp_pose_generators={
            "hand": create_parallel_jaw_grasp_pose_generator(
                n_sample=args.n_sample,
                force_refresh=args.force_reannotate,
            )
        },
    )
    semantics = create_link_antipodal_semantics(cube, GRASP_LINK, label="rubiks_cube")
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "pickup_cube_axis", cube_center_pose(cube))
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the Rubik's cube, then press Enter to plan PickUp..."
    )

    compiled = engine.compile(
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=GRASP_SAMPLE_INTERVAL,
                ),
                skill_options=PickUpOptions(
                    approach_direction=torch.tensor(
                        APPROACH_DIRECTION, dtype=torch.float32, device=sim.device
                    ),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan the Rubik's-cube PickUp trajectory.")
        return
    logger.log_info(
        "Planned Rubik's-cube PickUp over "
        f"{compiled.trajectory.positions.shape[1]} control steps."
    )

    if wait_for_user:
        input("Press Enter to replay the PickUp demo...")
    # Rigid-object tutorials freeze the target with make_clear_dynamics_callback
    # at the start of the lift. That helper takes a RigidObject, and applying it
    # to this floating-base articulation clears the root velocity exactly as the
    # gripper begins to lift: the grasp breaks and the cube drops back after
    # rising ~3 mm. The articulation needs no such freeze.
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="pickup_rubiks_cube_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
