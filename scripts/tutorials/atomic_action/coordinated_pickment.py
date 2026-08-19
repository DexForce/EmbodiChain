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

"""Demonstrate dual-arm coordinated pickment with selectable object meshes.

The two selected arms pinch opposite sides of one object, lift it together, and
move the object to an object-centric target pose while both grippers stay closed.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.data import get_data_path
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    ControlPartCommandProfile,
    CoordinatedPickGoal,
    CoordinatedPickmentOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import (
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_euler
from scripts.tutorials.atomic_action.scenario_utils import (
    add_dual_tutorial_robot,
    add_support_surface,
    compute_world_bounds,
    get_local_vertices,
    log_action_plan,
    resolve_cached_data_path,
    rotate_pose_about_world_z,
    settle_object,
)
from scripts.tutorials.atomic_action.tutorial_utils import (
    TutorialRobot,
    broadcast_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    format_tensor,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

PICKMENT_ASSET_ROOT = "CoordinatedPlacementAndPickment"
GRIPPER_TCP_Z = 0.121
SUPPORT_SURFACE_Z = 0.55
SUPPORT_SURFACE_SIZE = (0.7, 1.20, 0.02)
SUPPORT_SURFACE_CENTER = (
    0.0,
    0.0,
    SUPPORT_SURFACE_Z - 0.5 * SUPPORT_SURFACE_SIZE[2],
)
PICKMENT_RECORD_LOOK_AT = (
    (-0.25, 0.02, 2.5),
    (0.0, 0.02, 0.75),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class PickmentObjectPreset:
    """Configuration for an object used by the coordinated pickment demo."""

    label: str
    mesh_path: str
    init_xy: tuple[float, float]
    init_rot: tuple[float, float, float]
    surface_clearance: float
    body_scale: tuple[float, float, float]
    target_translation: tuple[float, float, float]
    target_world_yaw_deg: float
    hand_close_qpos: float


OBJECT_PRESETS = {
    "pencil": PickmentObjectPreset(
        label="pencil",
        mesh_path=f"{PICKMENT_ASSET_ROOT}/pencil.glb",
        init_xy=(-0.02, 0.02),
        # Rotate the imported pencil from its default upright orientation to a supported pose.
        init_rot=(90.0, 0.0, 0.0),
        surface_clearance=0.008,
        body_scale=(2.0, 2.0, 2.0),
        target_translation=(0.22, -0.04, 0.16),
        target_world_yaw_deg=0.0,
        hand_close_qpos=0.026,
    ),
    "pot": PickmentObjectPreset(
        label="pot",
        mesh_path=f"{PICKMENT_ASSET_ROOT}/pot.glb",
        init_xy=(-0.02, 0.02),
        init_rot=(-90.0, 90.0, 0.0),
        surface_clearance=0.008,
        body_scale=(2.0, 2.0, 2.0),
        target_translation=(-0.12, -0.03, 0.12),
        target_world_yaw_deg=0.0,
        hand_close_qpos=0.026,
    ),
    "water_basin": PickmentObjectPreset(
        label="water_basin",
        mesh_path=get_data_path("WaterBasin/water_basin.glb"),
        init_xy=(0.0, 0.02),
        init_rot=(0.0, 0.0, 0.0),
        surface_clearance=0.008,
        body_scale=(1.0, 1.0, 1.0),
        target_translation=(-0.12, -0.03, 0.12),
        target_world_yaw_deg=0.0,
        hand_close_qpos=0.026,
    ),
    "plastic_tray": PickmentObjectPreset(
        label="plastic_tray",
        mesh_path=get_data_path("PlasticTray/plastic_tray.glb"),
        init_xy=(-0.02, 0.02),
        init_rot=(0.0, 0.0, 90.0),
        surface_clearance=0.008,
        body_scale=(1.0, 1.0, 1.0),
        target_translation=(-0.12, -0.03, 0.12),
        target_world_yaw_deg=0.0,
        hand_close_qpos=0.026,
    ),
}
PICKMENT_SAMPLE_INTERVAL = 96
PICKMENT_OBJECT_MOTION_KEYFRAMES = 6
PICKMENT_PRE_GRASP_DISTANCE = 0.11
PICKMENT_LIFT_HEIGHT = 0.10
PICKMENT_HAND_INTERP_STEPS = 10
PICKMENT_HOLD_STEPS = 4
TRAJECTORY_SIM_STEPS = 4


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = create_tutorial_argument_parser(
        "Dual-arm coordinated pickment demo",
        features=(
            "debug_state",
            "diagnose_plan",
            "grasp_sampling",
            "headless_play",
            "visualize_axes",
        ),
        default_device="cpu",
        default_renderer="hybrid",
    )
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_PRESETS),
        default="plastic_tray",
        help="Object mesh to grasp in the coordinated pickment demo.",
    )
    return parser.parse_args()


def create_dual_robot(
    sim: SimulationManager,
    robot_type: TutorialRobot,
) -> Robot:
    """Create the selected dual-arm robot with one PGI gripper per arm."""
    return add_dual_tutorial_robot(
        sim,
        robot_type=robot_type,
        uid=f"Dual{robot_type.title()}CoordinatedPickment",
        urdf_name=f"dual_{robot_type}_coordinated_pickment",
        tcp_z=GRIPPER_TCP_Z,
        solver="pytorch",
    )


def create_support_surface(sim: SimulationManager) -> RigidObject:
    """Create a compact support slab under the staged object."""
    return add_support_surface(
        sim,
        size=SUPPORT_SURFACE_SIZE,
        center=SUPPORT_SURFACE_CENTER,
    )


def create_pickment_object(
    sim: SimulationManager,
    preset: PickmentObjectPreset,
) -> RigidObject:
    """Create the selected object mesh on the support surface."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=preset.label,
            shape=MeshCfg(
                fpath=resolve_cached_data_path(preset.mesh_path), compute_uv=False
            ),
            attrs=RigidBodyAttributesCfg(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=1.0,
                linear_damping=0.5,
                contact_offset=0.001,
                rest_offset=0.0,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=2.0,
            ),
            max_convex_hull_num=16,
            init_pos=[preset.init_xy[0], preset.init_xy[1], SUPPORT_SURFACE_Z],
            init_rot=list(preset.init_rot),
            body_scale=preset.body_scale,
        )
    )
    obj.cfg.init_pos = compute_supported_init_pos(obj, preset)
    obj.reset()
    return obj


def compute_supported_init_pos(
    obj: RigidObject,
    preset: PickmentObjectPreset,
) -> tuple[float, float, float]:
    """Place an object so its rotated mesh bottom sits on the support surface."""
    vertices = get_local_vertices(obj)
    rot = torch.as_tensor(preset.init_rot, dtype=torch.float32, device=vertices.device)
    rot = rot.unsqueeze(0) * torch.pi / 180.0
    upright_rot = matrix_from_euler(rot, "XYZ")[0]
    rotated_vertices = vertices @ upright_rot.T
    bottom_z = rotated_vertices[:, 2].min().item()
    z = SUPPORT_SURFACE_Z + preset.surface_clearance - bottom_z
    return (preset.init_xy[0], preset.init_xy[1], z)


def compute_left_to_right_arm_direction(
    robot: Robot,
    device: torch.device | str,
) -> torch.Tensor:
    """World-frame unit direction from the left arm base to the right arm base.

    The affordance sampler projects the object mesh onto this direction to split
    it into left/right grasp regions, so it must share the frame of the object
    pose (the local arena frame). Reading the two arm base links keeps the
    direction correct regardless of the arms' joint configuration.

    Args:
        robot: Dual-arm robot whose arms define the left/right sides.
        device: Device on which the returned direction should live.

    Returns:
        A normalized ``(3,)`` direction vector.
    """
    left_root = robot.cfg.solver_cfg["left_arm"].root_link_name
    right_root = robot.cfg.solver_cfg["right_arm"].root_link_name
    left_base = robot.get_link_pose(link_name=left_root, env_ids=[0], to_matrix=True)[
        0, :3, 3
    ]
    right_base = robot.get_link_pose(link_name=right_root, env_ids=[0], to_matrix=True)[
        0, :3, 3
    ]
    direction = (right_base - left_base).to(device=device, dtype=torch.float32)
    return direction / direction.norm().clamp_min(1e-6)


def build_object_target_pose(
    object_pose: torch.Tensor,
    object_vertices: torch.Tensor,
    preset: PickmentObjectPreset,
    device: torch.device,
) -> torch.Tensor:
    """Build the target pose for the whole object."""
    pose = rotate_pose_about_world_z(
        object_pose.clone().to(device=device, dtype=torch.float32),
        preset.target_world_yaw_deg,
    )
    pose[:3, 3] += torch.tensor(
        preset.target_translation, dtype=torch.float32, device=device
    )
    bottom_z = compute_world_bounds(pose, object_vertices)[0][2]
    pose[2, 3] += SUPPORT_SURFACE_Z + preset.surface_clearance + 0.10 - bottom_z
    return pose


def log_scene_targets(
    object_label: str,
    object_pose: torch.Tensor,
    target_pose: torch.Tensor,
) -> None:
    """Log compact object and target positions."""
    logger.log_info(
        "pickment scene: "
        f"object={object_label}, "
        f"object_origin={format_tensor(object_pose[:3, 3])}, "
        f"target_origin={format_tensor(target_pose[:3, 3])}"
    )


def draw_pickment_target_axes(
    sim: SimulationManager,
    object_target_pose: torch.Tensor,
    num_envs: int,
) -> None:
    """Draw the semantic axis for the target object pose.

    Left/right grasp poses are sampled inside the action from the antipodal
    affordance, so only the object-centric target axis is drawn ahead of
    planning.
    """
    draw_axis_marker(
        sim,
        "coordinated_pickment_object_target_axis",
        broadcast_pose_batch(object_target_pose, num_envs=num_envs),
        axis_len=0.12,
        axis_size=0.005,
    )


def log_execution_state(
    robot: Robot,
    obj: RigidObject,
    step_idx: int,
    total_steps: int,
) -> None:
    """Log hand and object state during execution."""
    object_pose = obj.get_local_pose(to_matrix=True)
    left_hand = robot.get_qpos(name="left_hand")
    right_hand = robot.get_qpos(name="right_hand")
    logger.log_info(
        f"step={step_idx}/{total_steps - 1}, "
        f"left_hand={format_tensor(left_hand[0])}, "
        f"right_hand={format_tensor(right_hand[0])}, "
        f"{obj.uid}_pos={format_tensor(object_pose[0, :3, 3])}"
    )


def run_coordinated_pickment_demo(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
) -> None:
    """Plan and optionally execute coordinated object pickment."""
    preset = OBJECT_PRESETS[args.object]
    create_support_surface(sim)
    obj = create_pickment_object(sim, preset)
    settle_object(sim, obj, step=0)
    object_pose_batch = clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    object_pose = object_pose_batch[0].to(device=sim.device, dtype=torch.float32)
    num_envs = object_pose_batch.shape[0]
    object_vertices = get_local_vertices(obj)
    object_semantics = create_antipodal_semantics(
        obj,
        label=preset.label,
        n_sample=args.n_sample,
        # n_sample = 1000,
        force_reannotate=args.force_reannotate,
    )
    left_to_right_arm_direction = compute_left_to_right_arm_direction(robot, sim.device)
    motion_gen = create_toppra_motion_generator(robot)

    left_open, left_close = get_hand_open_close_qpos(
        robot,
        hand_control_part="left_hand",
        close_qpos=preset.hand_close_qpos,
    )
    right_open, right_close = get_hand_open_close_qpos(
        robot,
        hand_control_part="right_hand",
        close_qpos=preset.hand_close_qpos,
    )
    pickment_options = CoordinatedPickmentOptions(
        pre_grasp_distance=PICKMENT_PRE_GRASP_DISTANCE,
        lift_height=PICKMENT_LIFT_HEIGHT,
        hand_interp_steps=PICKMENT_HAND_INTERP_STEPS,
        hold_steps=PICKMENT_HOLD_STEPS,
        object_motion_keyframes=PICKMENT_OBJECT_MOTION_KEYFRAMES,
        left_to_right_arm_direction=left_to_right_arm_direction,
        middle_empty_ratio=0.7,
    )
    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "left_hand": ControlPartCommandProfile.joint_positions(
                open=left_open,
                grasp=left_close,
            ),
            "right_hand": ControlPartCommandProfile.joint_positions(
                open=right_open,
                grasp=right_close,
            ),
        },
    )
    target_pose = build_object_target_pose(
        object_pose,
        object_vertices,
        preset,
        sim.device,
    )
    log_scene_targets(preset.label, object_pose, target_pose)
    if not args.no_vis_eef_axis:
        draw_pickment_target_axes(sim, target_pose, num_envs=num_envs)

    pickment_target = CoordinatedPickGoal(
        semantics=object_semantics,
        object_target_pose=broadcast_pose_batch(target_pose, num_envs=num_envs),
        object_initial_pose=broadcast_pose_batch(object_pose, num_envs=num_envs),
    )

    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the scene, then press Enter to plan pickment..."
    )

    start_time = time.time()
    binding = engine.bind_control_parts(
        "coordinated_pickment",
        {
            "left": {"motion": "left_arm", "grasp": "left_hand"},
            "right": {"motion": "right_arm", "grasp": "right_hand"},
        },
    )
    compiled = engine.compile(
        (
            ActionInvocation(
                "coordinated_pickment",
                pickment_target,
                binding,
                MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICKMENT_SAMPLE_INTERVAL,
                ),
                skill_options=pickment_options,
            ),
        )
    )
    success = compiled.plan_success
    traj = compiled.trajectory.positions
    logger.log_info(
        f"Plan coordinated pickment cost time: {time.time() - start_time:.2f} seconds"
    )
    if not success.all():
        logger.log_warning("Failed to plan coordinated pickment trajectory.")
        return
    joint_ids = list(range(robot.dof))
    log_action_plan(
        robot,
        "coordinated_pickment",
        traj,
        joint_ids,
        {
            segment.name: segment.waypoint_count
            for segment in compiled.action_plans[0].segments
        },
    )

    if args.diagnose_plan:
        return

    if wait_for_user:
        input("Press Enter to execute coordinated pickment...")

    def log_execution(step_idx: int, total_steps: int) -> None:
        if args.debug_state and (
            step_idx % max(1, total_steps // 10) == 0 or step_idx == total_steps - 1
        ):
            log_execution_state(robot, obj, step_idx, total_steps)

    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix=f"coordinated_pickment_{args.object}_auto_play",
        hold_steps=0,
        trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
        on_trajectory_step=log_execution,
        look_at=PICKMENT_RECORD_LOOK_AT,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the coordinated pickment demo."""
    args = parse_arguments()
    sim = create_tutorial_simulation(
        args,
        arena_space=3.0,
        light_pos=(0.0, -0.4, 3.0),
    )
    robot = create_dual_robot(sim, args.robot)
    run_coordinated_pickment_demo(args, sim, robot)


if __name__ == "__main__":
    run_tutorial(main)
