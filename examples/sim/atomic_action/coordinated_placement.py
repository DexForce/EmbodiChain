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

"""Demonstrate dual-arm coordinated placement with bread and pan meshes.

The left arm picks up bread. The right arm picks up a pan and moves it to the
lower alignment pose. The left arm places the bread above the pan and releases
it while the right hand keeps holding the pan.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as SciRotation

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    create_simulation_atomic_action_engine,
    CoordinatedPlacementOptions,
    CoordinatedPlacementGoal,
    GraspGoal,
    HeldObjectState,
    ObjectSemantics,
    PickUpOptions,
    MotionPolicy,
    TaskState,
)
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import MeshCfg, MeshCollisionCfg
from embodichain.utils import logger

from embodichain_benchmarks.motion_generation.scenarios._placement_geometry import (
    TABLE_TOP_Z,
    build_flat_object_grasp_pose,
    build_pan_handle_grasp_pose,
    build_support_object_target_pose,
    build_placing_object_target_pose,
    get_pan_basin_vertices,
)
from examples.sim.atomic_action.scenario import (
    add_dual_tutorial_robot,
    compute_local_bounds,
    create_manual_object_semantics,
    get_local_vertices,
    invert_pose,
    log_action_plan,
    resolve_cached_data_path,
    settle_object,
)
from examples.sim.atomic_action.support import (
    TutorialRobot,
    broadcast_pose_batch,
    clone_local_pose_from_first_env,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_rigid_body_physics,
    create_tutorial_simulation,
    draw_axis_marker,
    format_tensor,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

# Bake the GLB geometry into the authored Z-up frame before native loading.
# Preserve the gym table/bread orientation; turn the pan's opening upward.
DEFAULT_MESH_FRAME_CORRECTION_EULER_DEG = (0.0, 0.0, 0.0)
PAN_MESH_FRAME_CORRECTION_EULER_DEG = (180.0, 0.0, 0.0)
PAN_WORLD_YAW_CORRECTION_DEG = 270.0


def transform_baseline_pose(
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float],
    *,
    z_offset: float = 0.0,
    mesh_frame_correction_euler_deg: tuple[float, float, float] = (
        DEFAULT_MESH_FRAME_CORRECTION_EULER_DEG
    ),
    world_yaw_correction_deg: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Apply mesh-frame correction while preserving baseline world placement."""
    pos = np.asarray(init_pos, dtype=np.float64)
    pos[2] += z_offset
    rot = (
        SciRotation.from_euler("Z", world_yaw_correction_deg, degrees=True)
        * SciRotation.from_euler("XYZ", init_rot, degrees=True)
        * SciRotation.from_euler("XYZ", mesh_frame_correction_euler_deg, degrees=True)
    ).as_euler("XYZ", degrees=True)
    return tuple(float(value) for value in pos), tuple(float(value) for value in rot)


PLACEMENT_ASSET_ROOT = "CoordinatedPlacementAndPickment"
TABLE_MESH_PATH = f"{PLACEMENT_ASSET_ROOT}/table.glb"
BREAD_MESH_PATH = f"{PLACEMENT_ASSET_ROOT}/bread.glb"
PAN_MESH_PATH = f"{PLACEMENT_ASSET_ROOT}/pan.glb"
BREAD_LABEL = "bread"
PAN_LABEL = "pan"
# Leave opening clearance inside the PGI fingers' roughly 73 mm contact span.
BREAD_MESH_SCALE = 1.4
GRIPPER_TCP_Z = 0.121
PICK_SAMPLE_INTERVAL = 100
COORDINATED_SAMPLE_INTERVAL = 120
BASELINE_ROBOT_INIT_POS = (1.85, 0.0, 0.1)
BASELINE_TABLE_TOP_Z = 0.3621708124799265
SCENE_Z_OFFSET = TABLE_TOP_Z - BASELINE_TABLE_TOP_Z
ROBOT_INIT_POS = (
    BASELINE_ROBOT_INIT_POS[0],
    BASELINE_ROBOT_INIT_POS[1],
    BASELINE_ROBOT_INIT_POS[2] + SCENE_Z_OFFSET,
)
BASELINE_TABLE_INIT_POS = (
    0.00014585733079742588,
    0.00023304896730074557,
    -0.019599792839044783,
)
BASELINE_TABLE_INIT_ROT = (
    0.0001074673904926984,
    0.00865572768366991,
    -90.6562109309317,
)
BASELINE_BREAD_INIT_POS = (
    0.007266042530919159,
    0.17218712515099063,
    0.38805152145807564,
)
BASELINE_BREAD_INIT_ROT = (
    179.93952112929065,
    -0.12776179446053365,
    85.59207565132371,
)
BASELINE_PAN_INIT_POS = (
    0.0009683294205463406,
    -0.14189524793277888,
    0.38900474548025743,
)
BASELINE_PAN_INIT_ROT = (
    -179.23950670370294,
    -0.4795764805552328,
    98.19364391929443,
)
TABLE_INIT_POS, TABLE_INIT_ROT = transform_baseline_pose(
    BASELINE_TABLE_INIT_POS,
    BASELINE_TABLE_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
)
BREAD_INIT_POS, BREAD_INIT_ROT = transform_baseline_pose(
    BASELINE_BREAD_INIT_POS,
    BASELINE_BREAD_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
)
PAN_INIT_POS, PAN_INIT_ROT = transform_baseline_pose(
    BASELINE_PAN_INIT_POS,
    BASELINE_PAN_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
    mesh_frame_correction_euler_deg=PAN_MESH_FRAME_CORRECTION_EULER_DEG,
    world_yaw_correction_deg=PAN_WORLD_YAW_CORRECTION_DEG,
)
# The bread is about 37 mm thick at the selected mesh scale.
# Seat both objects clear of the surface before Newton builds its contacts.
BREAD_INIT_POS = (BREAD_INIT_POS[0], BREAD_INIT_POS[1], TABLE_TOP_Z + 0.045)
PAN_INIT_POS = (PAN_INIT_POS[0], PAN_INIT_POS[1], TABLE_TOP_Z + 0.008)
BREAD_GRASP_Z_CLEARANCE = 0.015
PAN_GRASP_Z_CLEARANCE = 0.0
# Use the straight outer handle rather than the rounded basin neck.
PAN_GRASP_HANDLE_TIP_SHIFT = 0.055
PAN_HANDLE_CLOSE_QPOS = 0.045
DEXUNI_PAN_HANDLE_CLOSE_QPOS = 0.033
PAN_PICK_SAMPLE_INTERVAL = 130
PAN_PICK_HAND_INTERP_STEPS = 32
BREAD_TARGET_HEIGHT_OFFSET = 0.0
SUPPORT_TARGET_HEIGHT_OFFSET = 0.0
PICK_APPROACH_DISTANCE = 0.12
PLACE_LIFT_HEIGHT = 0.05
TRAJECTORY_SIM_STEPS = 8


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = create_tutorial_argument_parser(
        "Dual-arm coordinated placement demo",
        features=(
            "debug_state",
            "diagnose_plan",
            "headless_play",
            "visualize_axes",
        ),
        default_device="cuda",
        default_renderer="hybrid",
    )
    return parser.parse_args()


def create_dual_robot(
    sim: SimulationManager,
    robot_type: TutorialRobot,
) -> Robot:
    """Create the selected dual-arm robot with its matching grippers."""
    return add_dual_tutorial_robot(
        sim,
        robot_type=robot_type,
        uid=f"Dual{robot_type.title()}CoordinatedPlacement",
        urdf_name=f"dual_{robot_type}_coordinated_placement",
        tcp_z=GRIPPER_TCP_Z,
        init_pos=ROBOT_INIT_POS,
        newton_gripper_friction=4.0 if sim.is_newton_backend else None,
        newton_mimic_stabilizer=True if sim.is_newton_backend else None,
    )


def _placement_mesh_path(data_path: str) -> str:
    """Cache a textured OBJ with the same authored Z-up frame in every renderer."""
    source = Path(resolve_cached_data_path(data_path))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    cache_dir = (
        Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
        / "derived"
        / "coordinated_placement_z_up_v1"
        / digest
    )
    target = cache_dir / "mesh.obj"
    if not target.is_file():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        # Publish the OBJ, MTL and textures together so concurrent examples
        # cannot observe a partially exported asset.
        with tempfile.TemporaryDirectory(dir=cache_dir.parent) as staging:
            mesh = trimesh.load(str(source), force="mesh", process=False)
            mesh.export(Path(staging) / target.name)
            try:
                Path(staging).rename(cache_dir)
            except OSError:
                if not target.is_file():
                    raise
    return str(target)


def create_table(sim: SimulationManager) -> RigidObject:
    """Create the table mesh from the bread-pan gym project."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="table",
            shape=MeshCfg(fpath=_placement_mesh_path(TABLE_MESH_PATH)),
            attrs=create_tutorial_rigid_body_physics(
                mass=10.0,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
            ),
            # This fixture never moves; static shapes also enter DexUni's MuJoCo view.
            body_type="static",
            init_pos=list(TABLE_INIT_POS),
            init_rot=list(TABLE_INIT_ROT),
        )
    )


def create_bread(sim: SimulationManager) -> RigidObject:
    """Create the bread mesh to be placed by the left arm."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="bread",
            shape=MeshCfg(
                fpath=_placement_mesh_path(BREAD_MESH_PATH),
                compute_uv=False,
                collision=MeshCollisionCfg(
                    approximation="convex_decomposition",
                    max_hulls=8,
                ),
            ),
            attrs=create_tutorial_rigid_body_physics(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=2.0,
                linear_damping=1.0,
                contact_offset=0.003,
                rest_offset=0.001,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=10.0,
                newton_contact=sim.is_newton_backend,
            ),
            body_scale=(BREAD_MESH_SCALE,) * 3,
            init_pos=list(BREAD_INIT_POS),
            init_rot=list(BREAD_INIT_ROT),
        )
    )


def create_pan(sim: SimulationManager) -> RigidObject:
    """Create the pan mesh held below the bread by the right arm."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="pan",
            shape=MeshCfg(
                fpath=_placement_mesh_path(PAN_MESH_PATH),
                compute_uv=False,
                collision=MeshCollisionCfg(
                    approximation="convex_decomposition",
                    max_hulls=16,
                ),
            ),
            attrs=create_tutorial_rigid_body_physics(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=2.0,
                linear_damping=1.0,
                contact_offset=0.001,
                rest_offset=0.0,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=2.0,
                newton_contact=sim.is_newton_backend,
            ),
            body_scale=(1.75, 1.75, 1.75),
            init_pos=list(PAN_INIT_POS),
            init_rot=list(PAN_INIT_ROT),
        )
    )


def compute_actual_held_state(
    robot: Robot,
    semantics: ObjectSemantics,
    object_pose: torch.Tensor,
    arm_control_part: str,
    device: torch.device,
) -> HeldObjectState:
    """Build held-object state from current object pose and current TCP FK."""
    arm_qpos = robot.get_qpos(name=arm_control_part).to(device=device)
    tcp_pose = robot.compute_fk(
        arm_qpos,
        name=arm_control_part,
        to_matrix=True,
    ).to(device=device, dtype=torch.float32)
    object_pose = broadcast_pose_batch(
        object_pose.to(device=device, dtype=torch.float32),
        num_envs=tcp_pose.shape[0],
    )
    object_to_eef = torch.bmm(invert_pose(object_pose), tcp_pose)
    return HeldObjectState(
        semantics=semantics,
        object_to_eef=object_to_eef,
        grasp_xpos=tcp_pose,
    )


def log_scene_targets(
    bread_pose: torch.Tensor,
    pan_pose: torch.Tensor,
    support_target_pose: torch.Tensor | None = None,
    placing_target_pose: torch.Tensor | None = None,
) -> None:
    """Log compact object and target positions for diagnosis."""
    logger.log_info(
        "scene objects: "
        f"bread_origin={format_tensor(bread_pose[:3, 3])}, "
        f"pan_origin={format_tensor(pan_pose[:3, 3])}"
    )
    if support_target_pose is not None and placing_target_pose is not None:
        logger.log_info(
            "coordinated targets: "
            f"support_pan_origin={format_tensor(support_target_pose[:3, 3])}, "
            f"placing_bread_origin={format_tensor(placing_target_pose[:3, 3])}"
        )


def draw_coordinated_axes(
    sim: SimulationManager,
    support_target_pose: torch.Tensor,
    placing_target_pose: torch.Tensor,
    num_envs: int,
) -> None:
    """Draw coordinate-frame markers for coordinated placement targets."""
    draw_axis_marker(
        sim,
        "support_pan_target_axis",
        broadcast_pose_batch(support_target_pose, num_envs=num_envs),
        axis_len=0.08,
        axis_size=0.004,
    )
    draw_axis_marker(
        sim,
        "placing_bread_target_axis",
        broadcast_pose_batch(placing_target_pose, num_envs=num_envs),
        axis_len=0.08,
        axis_size=0.004,
    )


def log_execution_state(
    robot: Robot,
    bread: RigidObject,
    pan: RigidObject,
    step_idx: int,
    total_steps: int,
) -> None:
    """Log hand and object state during execution."""
    bread_pose = bread.get_local_pose(to_matrix=True)
    pan_pose = pan.get_local_pose(to_matrix=True)
    left_hand = robot.get_qpos(name="left_eef")
    right_hand = robot.get_qpos(name="right_eef")
    logger.log_info(
        f"step={step_idx}/{total_steps - 1}, "
        f"left_hand={format_tensor(left_hand[0])}, "
        f"right_hand={format_tensor(right_hand[0])}, "
        f"bread_pos={format_tensor(bread_pose[0, :3, 3])}, "
        f"pan_pos={format_tensor(pan_pose[0, :3, 3])}"
    )


def run_coordinated_placement_demo(
    args: argparse.Namespace, sim: SimulationManager, robot: Robot
) -> None:
    """Plan and optionally execute pick-up and coordinated placement."""
    create_table(sim)
    bread = create_bread(sim)
    pan = create_pan(sim)
    sim.prepare()
    settle_object(sim, bread, step=0)
    settle_object(sim, pan, step=0)
    sim.update(step=10)
    bread_pose_batch = clone_local_pose_from_first_env(bread)
    pan_pose_batch = clone_local_pose_from_first_env(pan)
    initial_bread_z = bread_pose_batch[:, 2, 3].clone()
    initial_pan_z = pan_pose_batch[:, 2, 3].clone()
    bread.clear_dynamics()
    pan.clear_dynamics()
    bread_pose = bread_pose_batch[0].to(device=sim.device, dtype=torch.float32)
    pan_pose = pan_pose_batch[0].to(device=sim.device, dtype=torch.float32)
    num_envs = bread_pose_batch.shape[0]
    bread_vertices = get_local_vertices(bread)
    pan_vertices = get_local_vertices(pan)
    bread_local_min, bread_local_max = compute_local_bounds(bread_vertices)
    log_scene_targets(bread_pose, pan_pose)
    bread_semantics = create_manual_object_semantics(bread, BREAD_LABEL)
    pan_semantics = create_manual_object_semantics(pan, PAN_LABEL)
    motion_gen = create_toppra_motion_generator(
        robot,
        planner=getattr(args, "planner", "trapezoidal"),
    )

    right_open, right_close = get_hand_open_close_qpos(
        robot,
        hand_control_part="right_eef",
        close_qpos=(
            DEXUNI_PAN_HANDLE_CLOSE_QPOS
            if sim.physics.solver_type == "dexuni"
            else PAN_HANDLE_CLOSE_QPOS
        ),
    )
    left_open, left_close = get_hand_open_close_qpos(
        robot,
        hand_control_part="left_eef",
        close_qpos=0.030,
    )
    left_pick_options = PickUpOptions(
        pre_grasp_distance=PICK_APPROACH_DISTANCE,
        lift_height=0.12,
        hand_interp_steps=10,
    )
    right_pick_options = PickUpOptions(
        pre_grasp_distance=PICK_APPROACH_DISTANCE,
        lift_height=0.10,
        hand_interp_steps=PAN_PICK_HAND_INTERP_STEPS,
    )
    coordinated_options = CoordinatedPlacementOptions(
        release=True,
        placing_height_offset=BREAD_TARGET_HEIGHT_OFFSET,
        support_height_offset=SUPPORT_TARGET_HEIGHT_OFFSET,
        lift_height=PLACE_LIFT_HEIGHT,
        hand_interp_steps=10,
        hold_steps=6,
        retreat_steps=18,
    )
    engine = create_simulation_atomic_action_engine(
        motion_generator=motion_gen,
        scene_entities=(bread, pan),
        control_profiles={
            "left_eef": ControlPartCommandProfile.joint_positions(
                open=left_open,
                grasp=left_close,
            ),
            "right_eef": ControlPartCommandProfile.joint_positions(
                open=right_open,
                grasp=right_close,
            ),
        },
    )
    full_joint_ids = list(range(robot.dof))
    state = engine.initial_context(control_dt=sim.sim_config.physics_dt)

    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the scene, then press Enter to compile both pick-ups..."
    )

    bread_grasp_pose = build_flat_object_grasp_pose(
        bread_pose,
        bread_vertices,
        bread_local_min,
        bread_local_max,
        sim.device,
        z_clearance=BREAD_GRASP_Z_CLEARANCE,
    )
    pan_grasp_pose = build_pan_handle_grasp_pose(
        pan_pose,
        pan_vertices,
        sim.device,
        z_clearance=PAN_GRASP_Z_CLEARANCE,
    )
    pan_grasp_pose[:3, 3] += PAN_GRASP_HANDLE_TIP_SHIFT * pan_grasp_pose[:3, 1]
    pick_invocations = (
        engine.make_invocation(
            "pick_up",
            GraspGoal(
                semantics=bread_semantics,
                grasp_xpos=broadcast_pose_batch(bread_grasp_pose, num_envs=num_envs),
            ),
            control_parts={"primary": {"motion": "left_arm", "grasp": "left_eef"}},
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=PICK_SAMPLE_INTERVAL,
            ),
            skill_options=left_pick_options,
        ),
        engine.make_invocation(
            "pick_up",
            GraspGoal(
                semantics=pan_semantics,
                grasp_xpos=broadcast_pose_batch(pan_grasp_pose, num_envs=num_envs),
            ),
            control_parts={"primary": {"motion": "right_arm", "grasp": "right_eef"}},
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=PAN_PICK_SAMPLE_INTERVAL,
            ),
            skill_options=right_pick_options,
        ),
    )
    start_time = time.time()
    pick_compiled = engine.compile(pick_invocations, state)
    logger.log_info(
        f"Compile both pick-ups cost time: {time.time() - start_time:.2f} seconds"
    )
    if len(pick_compiled.action_plans) != len(pick_invocations):
        raise RuntimeError("Failed to compile both pick-up trajectories.")
    left_pick_result, right_pick_result = pick_compiled.action_plans
    if not left_pick_result.plan_success.all():
        raise RuntimeError("Failed to plan left bread pick-up trajectory.")
    if not pick_compiled.plan_success.all():
        raise RuntimeError("Failed to plan right pan pick-up trajectory.")
    left_pick_trajectory = left_pick_result.joint_trajectory
    right_pick_trajectory = right_pick_result.joint_trajectory
    if left_pick_trajectory is None or right_pick_trajectory is None:
        raise RuntimeError("PickUp did not produce joint trajectories.")
    left_pick_traj = left_pick_trajectory.positions
    right_pick_traj = right_pick_trajectory.positions
    state = pick_compiled.projected_context
    bread_held_state = state.get_held_object("left_arm")
    if bread_held_state is None:
        raise RuntimeError("PickUp did not produce a held state for the bread.")
    pan_held_state = state.get_held_object("right_arm")
    if pan_held_state is None:
        raise RuntimeError("PickUp did not produce a held state for the pan.")
    log_action_plan(robot, "left_pick_up", left_pick_traj, full_joint_ids)
    log_action_plan(robot, "right_pick_up", right_pick_traj, full_joint_ids)

    if args.diagnose_plan:
        robot.set_qpos(state.last_qpos, joint_ids=full_joint_ids)
    else:
        if wait_for_user:
            input("Press Enter to execute both pick-up trajectories...")

        def log_trajectory_execution(step_idx: int, total_steps: int) -> None:
            if args.debug_state and (
                step_idx % max(1, total_steps // 10) == 0 or step_idx == total_steps - 1
            ):
                log_execution_state(robot, bread, pan, step_idx, total_steps)

        replay_trajectory(
            sim,
            robot,
            left_pick_trajectory,
            args,
            video_prefix="",
            hold_steps=0,
            trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
            joint_ids=full_joint_ids,
            on_trajectory_step=log_trajectory_execution,
            record=False,
        )
        bread_lift = bread.get_local_pose(to_matrix=True)[:, 2, 3] - initial_bread_z
        if bool((bread_lift < 0.5 * left_pick_options.lift_height).any()):
            raise RuntimeError(
                f"Bread pickup did not lift the object: bread_lift={bread_lift.tolist()}."
            )
        bread.clear_dynamics()
        replay_trajectory(
            sim,
            robot,
            right_pick_trajectory,
            args,
            video_prefix="",
            hold_steps=0,
            trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
            joint_ids=full_joint_ids,
            on_trajectory_step=log_trajectory_execution,
            record=False,
        )
        pan.clear_dynamics()
        # Reconcile the projected task state with measurements before compiling
        # placement. This keeps the second stage robust to pick-up execution error.
        bread_pose_batch = (
            bread.get_local_pose(to_matrix=True)
            .clone()
            .to(device=sim.device, dtype=torch.float32)
        )
        pan_pose_batch = (
            pan.get_local_pose(to_matrix=True)
            .clone()
            .to(device=sim.device, dtype=torch.float32)
        )
        bread.clear_dynamics()
        pan.clear_dynamics()
        bread_pose = bread_pose_batch[0]
        pan_pose = pan_pose_batch[0]
        bread_lift = bread_pose_batch[:, 2, 3] - initial_bread_z
        pan_lift = pan_pose_batch[:, 2, 3] - initial_pan_z
        if bool((bread_lift < 0.5 * left_pick_options.lift_height).any()) or bool(
            (pan_lift < 0.5 * right_pick_options.lift_height).any()
        ):
            raise RuntimeError(
                "Both objects must be lifted before coordinated placement: "
                f"bread_lift={bread_lift.tolist()}, pan_lift={pan_lift.tolist()}."
            )
        bread_held_state = compute_actual_held_state(
            robot,
            bread_semantics,
            bread_pose_batch,
            "left_arm",
            sim.device,
        )
        pan_held_state = compute_actual_held_state(
            robot,
            pan_semantics,
            pan_pose_batch,
            "right_arm",
            sim.device,
        )
        held_objects = dict(state.task.held_objects)
        held_objects["left_arm"] = bread_held_state
        held_objects["right_arm"] = pan_held_state
        # Use measured arm kinematics, but retain the grasp commands. A blocked
        # jaw's measured position is more open than its commanded preload;
        # replaying that measurement would relax the grasp during placement.
        placement_start_qpos = robot.get_qpos().clone()
        placement_start_qpos[:, robot.get_joint_ids("left_eef")] = left_close
        placement_start_qpos[:, robot.get_joint_ids("right_eef")] = right_close
        state = state.project(
            qpos=placement_start_qpos,
            task=TaskState(
                batch_size=state.batch_size,
                device=state.robot.qpos.device,
                held_objects=held_objects,
            ),
        )

    support_target_pose = build_support_object_target_pose(pan_pose, sim.device)
    placing_target_pose = build_placing_object_target_pose(
        bread_pose,
        bread_vertices,
        pan_vertices,
        support_target_pose,
        sim.device,
    )
    log_scene_targets(
        bread_pose,
        pan_pose,
        support_target_pose,
        placing_target_pose,
    )
    if not args.auto_play and not args.no_vis_eef_axis:
        draw_coordinated_axes(
            sim,
            support_target_pose,
            placing_target_pose,
            num_envs=num_envs,
        )
    coordinated_target = CoordinatedPlacementGoal(
        placing_object_target_pose=broadcast_pose_batch(
            placing_target_pose, num_envs=num_envs
        ),
        support_object_target_pose=broadcast_pose_batch(
            support_target_pose, num_envs=num_envs
        ),
        placing_height_offset=BREAD_TARGET_HEIGHT_OFFSET,
        support_height_offset=SUPPORT_TARGET_HEIGHT_OFFSET,
        release=True,
    )
    start_time = time.time()
    placement_compiled = engine.compile(
        (
            engine.make_invocation(
                "coordinated_placement",
                coordinated_target,
                control_parts={
                    "placing": {"motion": "left_arm", "grasp": "left_eef"},
                    "support": {"motion": "right_arm", "grasp": "right_eef"},
                },
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=COORDINATED_SAMPLE_INTERVAL,
                ),
                skill_options=coordinated_options,
            ),
        ),
        state,
    )
    coordinated_success = placement_compiled.plan_success
    coordinated_traj = placement_compiled.trajectory.positions
    state = placement_compiled.projected_context
    logger.log_info(
        "Plan coordinated placement cost time: "
        f"{time.time() - start_time:.2f} seconds"
    )
    if not coordinated_success.all():
        raise RuntimeError("Failed to plan coordinated placement trajectory.")
    log_action_plan(
        robot,
        "coordinated_placement",
        coordinated_traj,
        full_joint_ids,
        {
            segment.name: segment.waypoint_count
            for segment in placement_compiled.action_plans[0].segments
        },
    )

    if args.diagnose_plan:
        return

    if args.auto_play and not args.no_vis_eef_axis:
        draw_coordinated_axes(
            sim,
            support_target_pose,
            placing_target_pose,
            num_envs=num_envs,
        )
    if wait_for_user:
        input("Press Enter to execute coordinated placement...")

    replay_trajectory(
        sim,
        robot,
        placement_compiled.trajectory,
        args,
        video_prefix="coordinated_placement_auto_play",
        hold_steps=80,
        trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
        joint_ids=full_joint_ids,
        on_trajectory_step=log_trajectory_execution,
        look_at=(
            (-0.25, 0.0, 2.5),
            (-0.05, 0.0, 0.72),
            (0.0, 0.0, 1.0),
        ),
    )
    bread_final_pose = bread.get_local_pose(to_matrix=True)
    pan_final_pose = pan.get_local_pose(to_matrix=True)
    bread_world = (
        bread_vertices.unsqueeze(0) @ bread_final_pose[:, :3, :3].transpose(1, 2)
        + bread_final_pose[:, None, :3, 3]
    )
    basin_world = (
        get_pan_basin_vertices(pan_vertices).unsqueeze(0)
        @ pan_final_pose[:, :3, :3].transpose(1, 2)
        + pan_final_pose[:, None, :3, 3]
    )
    bread_center_xy = 0.5 * (
        bread_world[:, :, :2].amin(dim=1) + bread_world[:, :, :2].amax(dim=1)
    )
    bread_bottom_z = bread_world[:, :, 2].amin(dim=1)
    inside_basin = (
        (bread_center_xy >= basin_world[:, :, :2].amin(dim=1) - 0.015)
        & (bread_center_xy <= basin_world[:, :, :2].amax(dim=1) + 0.015)
    ).all(dim=1)
    raised_pan = pan_final_pose[:, 2, 3] > TABLE_TOP_Z + 0.03
    raised_bread = bread_bottom_z > TABLE_TOP_Z + 0.02
    if not bool((inside_basin & raised_pan & raised_bread).all()):
        raise RuntimeError(
            "Bread must be released into the raised pan: "
            f"inside_basin={inside_basin.tolist()}, "
            f"bread_bottom_z={bread_bottom_z.tolist()}, "
            f"pan_z={pan_final_pose[:, 2, 3].tolist()}."
        )
    logger.log_info(
        "Measured placement: "
        f"bread_bottom_z={bread_bottom_z.tolist()}, "
        f"pan_z={pan_final_pose[:, 2, 3].tolist()}, "
        f"inside_basin={inside_basin.tolist()}"
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the coordinated placement demo."""
    args = parse_arguments()
    sim = create_tutorial_simulation(
        args,
        arena_space=3.0,
        newton_solver_options=(
            {
                "coupling": "two_way",
                # DexSim's native two-way pick/place example uses soft contacts
                # without rigid contact history and increases the proxy mass.
                "coupling_options": {"mass_scale": 4.0, "proxy_relaxation": 0.5},
                "vbd_options": {
                    "iterations": 12,
                    "friction_epsilon": 1e-4,
                    "rigid_contact_hard": False,
                    "rigid_contact_history": False,
                },
            }
            if args.physics == "newton"
            else None
        ),
    )
    robot = create_dual_robot(sim, args.robot)
    run_coordinated_placement_demo(args, sim, robot)


if __name__ == "__main__":
    run_tutorial(main)
