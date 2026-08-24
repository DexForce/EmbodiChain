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

"""Move an obstacle onto an active path and visualize online replanning."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager, VisualMaterialCfg
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseGoal,
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    JointPositionPayload,
    JointPositionTarget,
    MotionPolicy,
    RecoveryPolicy,
    RigidObjectSceneProvider,
    RunnerStatus,
    RunnerStep,
    SimulationExecutionAdapter,
    TaskState,
    TimedCommandSequence,
    TrackingPolicy,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg, Robot
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboAutoGenCfg,
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import SceneOverlays, TrajectoryOverlay
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    prepare_tutorial_scene,
    publish_tutorial_scene,
    run_tutorial,
    serve_tutorial_scene,
    start_auto_play_recording,
    stop_auto_play_recording,
)

OBSTACLE_UID = "dynamic_obstacle"
CONTROL_PART = "arm"
SAMPLE_COUNT = 80
COMMAND_CYCLE_TIME = 0.1
COLLISION_SPHERE_FIT_DENSITY = 0.3
ROBOT_COLLISION_BUFFER = 0.005
MOVE_AFTER_COMMAND = 12
OBSTACLE_SIZE = (0.10, 0.10, 0.12)
OBSTACLE_START_POSITION = (0.59, -0.20, 0.455)
BLOCKING_PATH_FRACTION = 0.50
OBSTACLE_MOVE_DURATION = 0.6
AUTO_PLAY_LEAD_IN_DURATION = 0.75
POST_EXECUTION_HOLD_DURATION = 1.0
TRACKING_ERROR_THRESHOLD = 0.1
MINIMUM_REPLAN_DETOUR = 0.04
MINIMUM_REPLAN_CLEARANCE = 0.01
MAXIMUM_FINAL_EEF_ERROR = 0.04
TRAJECTORY_MARKER_STRIDE = 8


def _animate_obstacle_to_pose(
    obstacle: RigidObject,
    adapter: SimulationExecutionAdapter,
    start_pose: torch.Tensor,
    *,
    target_pose: torch.Tensor,
    duration: float,
    pace_wall_time: bool,
) -> torch.Tensor:
    """Move one obstacle smoothly while holding the latest robot command.

    The runner does not observe the scene while this function is active. The
    completed animation therefore appears as one material scene change at the
    next execution tick instead of consuming one replan per animation frame.

    Args:
        obstacle: Kinematic obstacle to move.
        adapter: Simulation clock used to advance physics and recording frames.
        start_pose: Batched obstacle poses with shape ``(B, 4, 4)``.
        target_pose: Batched destination poses with shape ``(B, 4, 4)``.
        duration: Minimum simulated animation duration in seconds.
        pace_wall_time: Whether to pace the animation for a live viewer.

    Returns:
        Independently owned final batched obstacle pose.

    Raises:
        ValueError: If the poses or duration are invalid.
    """
    if start_pose.dim() != 3 or start_pose.shape[-2:] != (4, 4):
        raise ValueError(
            f"start_pose must have shape (B, 4, 4), got {tuple(start_pose.shape)}."
        )
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and greater than zero.")
    if target_pose.shape != start_pose.shape:
        raise ValueError(
            "target_pose must match start_pose shape, got "
            f"{tuple(target_pose.shape)} and {tuple(start_pose.shape)}."
        )
    target_pose = target_pose.to(device=start_pose.device, dtype=start_pose.dtype)
    if (
        not torch.isfinite(start_pose).all().item()
        or not torch.isfinite(target_pose).all().item()
    ):
        raise ValueError("start_pose and target_pose must contain only finite values.")
    if not torch.allclose(start_pose[:, :3, :3], target_pose[:, :3, :3]):
        raise ValueError("Obstacle animation requires an unchanged orientation.")

    step_count = max(1, math.ceil(duration / adapter.physics_dt))
    for step_index in range(1, step_count + 1):
        alpha = step_index / step_count
        pose = start_pose.clone()
        pose[:, :3, 3] = torch.lerp(
            start_pose[:, :3, 3],
            target_pose[:, :3, 3],
            alpha,
        )
        obstacle.set_local_pose(pose)
        adapter.sleep(adapter.physics_dt)
        if pace_wall_time:
            time.sleep(adapter.physics_dt)
    return target_pose.clone()


def _blocking_obstacle_pose(
    start_pose: torch.Tensor,
    eef_path: torch.Tensor,
    *,
    path_fraction: float,
) -> tuple[torch.Tensor, int]:
    """Place an obstacle at one waypoint of an already planned EEF path.

    Args:
        start_pose: Current obstacle poses with shape ``(B, 4, 4)``.
        eef_path: Planned EEF positions with shape ``(B, N, 3)``.
        path_fraction: Fraction of the path at which to block, in ``[0, 1]``.

    Returns:
        The target obstacle poses and selected waypoint index.

    Raises:
        ValueError: If inputs have incompatible shapes or an invalid fraction.
    """
    if start_pose.dim() != 3 or start_pose.shape[-2:] != (4, 4):
        raise ValueError("start_pose must have shape (B, 4, 4).")
    if (
        eef_path.dim() != 3
        or eef_path.shape[0] != start_pose.shape[0]
        or eef_path.shape[1] < 2
        or eef_path.shape[2] != 3
    ):
        raise ValueError("eef_path must have shape (B, N, 3) with N >= 2.")
    if not math.isfinite(path_fraction) or not 0.0 <= path_fraction <= 1.0:
        raise ValueError("path_fraction must be finite and within [0, 1].")

    waypoint_index = round((eef_path.shape[1] - 1) * path_fraction)
    target_pose = start_pose.clone()
    target_pose[:, :3, 3] = eef_path[:, waypoint_index].to(
        device=start_pose.device,
        dtype=start_pose.dtype,
    )
    return target_pose, waypoint_index


def _maximum_path_deviation(
    path: torch.Tensor,
    reference_path: torch.Tensor,
) -> torch.Tensor:
    """Measure each path's largest distance from a reference polyline.

    Args:
        path: Candidate path positions with shape ``(B, N, 3)``.
        reference_path: Reference positions with shape ``(B, M, 3)``.

    Returns:
        Per-environment maximum point-to-polyline distance in metres.

    Raises:
        ValueError: If either path has an invalid or incompatible shape.
    """
    if path.dim() != 3 or path.shape[2] != 3 or path.shape[1] == 0:
        raise ValueError("path must have shape (B, N, 3) with N >= 1.")
    if (
        reference_path.dim() != 3
        or reference_path.shape[0] != path.shape[0]
        or reference_path.shape[1] < 2
        or reference_path.shape[2] != 3
    ):
        raise ValueError(
            "reference_path must have shape (B, M, 3) with matching B and M >= 2."
        )

    reference_path = reference_path.to(device=path.device, dtype=path.dtype)
    segment_start = reference_path[:, :-1]
    segment = reference_path[:, 1:] - segment_start
    denominator = torch.sum(segment * segment, dim=-1).clamp_min(1.0e-12)
    relative = path[:, :, None, :] - segment_start[:, None, :, :]
    alpha = (
        torch.sum(relative * segment[:, None, :, :], dim=-1) / denominator[:, None, :]
    ).clamp(0.0, 1.0)
    projection = (
        segment_start[:, None, :, :] + alpha[..., None] * segment[:, None, :, :]
    )
    distances = torch.linalg.vector_norm(
        path[:, :, None, :] - projection,
        dim=-1,
    )
    return distances.amin(dim=2).amax(dim=1)


def _minimum_cuboid_clearance(
    path: torch.Tensor,
    cuboid_pose: torch.Tensor,
    *,
    size: tuple[float, float, float],
) -> torch.Tensor:
    """Measure the minimum signed distance from a path to an oriented cuboid.

    Positive values are outside the cuboid, zero lies on its surface, and
    negative values indicate penetration.

    Args:
        path: World-frame points with shape ``(B, N, 3)``.
        cuboid_pose: Cuboid world poses with shape ``(B, 4, 4)``.
        size: Full cuboid extents along its local XYZ axes.

    Returns:
        Per-environment minimum signed clearance in metres.

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values.
    """
    if path.dim() != 3 or path.shape[1] == 0 or path.shape[2] != 3:
        raise ValueError("path must have shape (B, N, 3) with N >= 1.")
    if cuboid_pose.shape != (path.shape[0], 4, 4):
        raise ValueError("cuboid_pose must have shape (B, 4, 4) matching path.")
    half_extent = torch.as_tensor(size, dtype=path.dtype, device=path.device) / 2.0
    if (
        half_extent.shape != (3,)
        or not torch.isfinite(half_extent).all().item()
        or (half_extent <= 0.0).any().item()
    ):
        raise ValueError("size must contain three finite positive extents.")
    cuboid_pose = cuboid_pose.to(device=path.device, dtype=path.dtype)
    if (
        not torch.isfinite(path).all().item()
        or not torch.isfinite(cuboid_pose).all().item()
    ):
        raise ValueError("path and cuboid_pose must contain only finite values.")

    relative = path - cuboid_pose[:, None, :3, 3]
    local_points = torch.matmul(relative, cuboid_pose[:, :3, :3])
    local_offset = torch.abs(local_points) - half_extent
    outside_distance = torch.linalg.vector_norm(
        torch.clamp_min(local_offset, 0.0),
        dim=-1,
    )
    inside_distance = torch.clamp_max(local_offset.amax(dim=-1), 0.0)
    return (outside_distance + inside_distance).amin(dim=1)


def _command_eef_positions(
    robot: Robot,
    commands: TimedCommandSequence,
    *,
    control_part: str,
) -> torch.Tensor:
    """Convert one endpoint command sequence to batched EEF positions."""
    if not commands.frames:
        raise ValueError("commands must contain at least one frame.")
    positions = []
    for frame in commands.frames:
        matching_commands = tuple(
            command
            for command in frame.commands
            if isinstance(command.target, JointPositionTarget)
            and command.target.control_part == control_part
        )
        if len(matching_commands) != 1:
            raise ValueError(
                f"Expected one joint command for control part {control_part!r}, "
                f"got {len(matching_commands)}."
            )
        payload = matching_commands[0].payload
        if not isinstance(payload, JointPositionPayload):
            raise TypeError(
                f"Control part {control_part!r} did not receive joint positions."
            )
        pose = robot.compute_fk(
            qpos=payload.positions,
            name=control_part,
            to_matrix=True,
        )
        positions.append(pose[:, :3, 3])
    return torch.stack(positions, dim=1)


def _draw_eef_path(
    sim: SimulationManager,
    name: str,
    path: torch.Tensor,
    *,
    stride: int,
    axis_len: float,
    axis_size: float,
) -> None:
    """Draw sparse native-viewer breadcrumbs for environment zero."""
    indices = list(range(0, path.shape[1], stride))
    if indices[-1] != path.shape[1] - 1:
        indices.append(path.shape[1] - 1)
    poses = (
        torch.eye(4, dtype=path.dtype, device=path.device)
        .unsqueeze(0)
        .repeat(len(indices), 1, 1)
    )
    poses[:, :3, 3] = path[0, indices]
    draw_axis_marker(
        sim,
        name,
        poses,
        axis_len=axis_len,
        axis_size=axis_size,
    )


def _publish_path_overlays(
    sim: SimulationManager,
    initial_path: torch.Tensor,
    replanned_path: torch.Tensor | None = None,
) -> None:
    """Publish colored initial and replanned paths to the Viser backend."""
    env_offset = sim.arena_offsets[0].detach().cpu().numpy()
    trajectories = [
        TrajectoryOverlay(
            overlay_id="initial_collision_path",
            points=initial_path[0].detach().cpu().numpy() + env_offset,
            color=(255, 70, 70),
            line_width=4.0,
        )
    ]
    if replanned_path is not None:
        trajectories.append(
            TrajectoryOverlay(
                overlay_id="replanned_avoidance_path",
                points=replanned_path[0].detach().cpu().numpy() + env_offset,
                color=(60, 230, 110),
                line_width=5.0,
            )
        )
    sim.set_visualization_overlays(SceneOverlays(trajectories=tuple(trajectories)))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the dynamic-obstacle tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate collision-world revision recovery with cuRobo."
    )
    parser.add_argument(
        "--no_obstacle_motion",
        action="store_true",
        help="Execute without moving the obstacle after planning.",
    )
    return parser.parse_args()


def main() -> None:
    """Move an obstacle during execution and replan from the latest snapshot."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    obstacle = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=OBSTACLE_UID,
            shape=CubeCfg(
                size=list(OBSTACLE_SIZE),
                visual_material=VisualMaterialCfg(
                    uid="dynamic_obstacle_orange",
                    base_color=[1.0, 0.18, 0.03, 1.0],
                    metallic=0.1,
                    roughness=0.35,
                ),
            ),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=list(OBSTACLE_START_POSITION),
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    # Initialize GPU physics before planning or recording so the first visible
    # frame and the initial planning context share the same settled state.
    sim.update(step=10)
    motion_gen = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=robot.uid,
                # The coarse default voxel fit under-covers the hand and
                # fingertips. A denser fit plus modest padding matches the
                # physical gripper without making the arm path infeasible.
                auto_gen=CuroboAutoGenCfg(
                    fit_type="morphit",
                    sphere_density=COLLISION_SPHERE_FIT_DENSITY,
                    collision_sphere_buffer=ROBOT_COLLISION_BUFFER,
                ),
                world=CuroboWorldCfg(
                    rigid_objects=[obstacle],
                    obstacle_representation="cuboid",
                    dynamic_obstacle_names=[OBSTACLE_UID],
                    multi_env=args.num_envs > 1,
                ),
            )
        )
    )
    scene_provider = RigidObjectSceneProvider(
        {OBSTACLE_UID: obstacle},
        collision_entity_ids=(OBSTACLE_UID,),
    )
    adapter = SimulationExecutionAdapter(
        sim,
        robot,
        control_dt=COMMAND_CYCLE_TIME,
        scene_provider=scene_provider,
    )

    current_pose = robot.compute_fk(
        qpos=robot.get_qpos(name=CONTROL_PART),
        name=CONTROL_PART,
        to_matrix=True,
    )
    target_pose = current_pose.clone()
    target_pose[:, :3, 3] += torch.tensor(
        [0.22, 0.24, 0.12],
        dtype=target_pose.dtype,
        device=target_pose.device,
    )
    engine = AtomicActionEngine(motion_generator=motion_gen)
    invocation = engine.make_invocation(
        "move_end_effector",
        EndEffectorPoseGoal(target_pose),
        control_parts={"primary": {"motion": CONTROL_PART}},
        motion_policy=MotionPolicy(
            strategy="motion_gen",
            sample_count=SAMPLE_COUNT,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            action_timeout=30.0,
        ),
        tracking_policy=TrackingPolicy.joint_position(
            in_flight_max_abs_error=TRACKING_ERROR_THRESHOLD,
            terminal_max_abs_error=TRACKING_ERROR_THRESHOLD,
        ),
        invocation_id="dynamic-obstacle-demo",
    )
    task_state = TaskState.empty(robot.get_qpos().shape[0], robot.device)
    session = engine.start((invocation,), adapter.observe(task_state))
    initial_eef_path = _command_eef_positions(
        robot,
        session.active_commands,
        control_part=CONTROL_PART,
    )
    blocking_obstacle_pose, blocking_waypoint_index = _blocking_obstacle_pose(
        obstacle.get_local_pose(to_matrix=True),
        initial_eef_path,
        path_fraction=BLOCKING_PATH_FRACTION,
    )
    logger.log_info(
        "Initial path prepared: obstacle will move onto waypoint "
        f"{blocking_waypoint_index}/{initial_eef_path.shape[1] - 1} at XYZ="
        f"{blocking_obstacle_pose[:, :3, 3].detach().cpu().tolist()}."
    )
    runner = ExecutionRunner(
        session,
        adapter,
        adapter,
        clock=adapter,
        # cuRobo can supply a trajectory duration, which takes precedence over
        # engine fallback timing. Keep a runner-side floor so the simulated
        # controller receives enough feedback cycles to follow every waypoint.
        cfg=ExecutionRunnerCfg(minimum_cycle_time=COMMAND_CYCLE_TIME),
    )

    initial_obstacle_pose = obstacle.get_local_pose(to_matrix=True).clone()
    draw_axis_marker(
        sim,
        "dynamic_obstacle_initial_pose",
        initial_obstacle_pose,
        axis_len=0.12,
    )
    _draw_eef_path(
        sim,
        "initial_eef_path",
        initial_eef_path,
        stride=TRAJECTORY_MARKER_STRIDE,
        axis_len=0.035,
        axis_size=0.0015,
    )
    _publish_path_overlays(sim, initial_eef_path)
    publish_tutorial_scene(sim, args)
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Small markers show the initial path. Press Enter, then watch the "
        "obstacle block it and the larger markers reveal the replanned detour...",
    )
    obstacle_moved = False
    observed_events: set[ExecutionEventKind] = set()
    replanned_eef_path: torch.Tensor | None = None
    replan_detour: torch.Tensor | None = None
    replan_clearance: torch.Tensor | None = None
    pace_obstacle_motion = bool(getattr(args, "viser", False)) or (
        not args.headless and not args.auto_play
    )

    def on_step(step: RunnerStep) -> None:
        nonlocal obstacle_moved, replanned_eef_path, replan_detour, replan_clearance
        if (
            not args.no_obstacle_motion
            and not obstacle_moved
            and step.command_count >= MOVE_AFTER_COMMAND
        ):
            start_pose = obstacle.get_local_pose(to_matrix=True).clone()
            logger.log_warning(
                f"Moving the collision obstacle over {OBSTACLE_MOVE_DURATION:.2f} s "
                f"after {step.command_count} accepted commands; start XYZ="
                f"{start_pose[:, :3, 3].detach().cpu().tolist()}."
            )
            moved_pose = _animate_obstacle_to_pose(
                obstacle,
                adapter,
                start_pose,
                target_pose=blocking_obstacle_pose,
                duration=OBSTACLE_MOVE_DURATION,
                pace_wall_time=pace_obstacle_motion,
            )
            obstacle_moved = True
            draw_axis_marker(
                sim,
                "dynamic_obstacle_moved_pose",
                moved_pose,
                axis_len=0.12,
            )
            logger.log_warning(
                "Obstacle animation completed at XYZ="
                f"{moved_pose[:, :3, 3].detach().cpu().tolist()}; the next "
                "scene snapshot should invalidate the active trajectory."
            )
        if step.tick is None:
            return
        for event in step.tick.events:
            observed_events.add(event.kind)
            if event.kind in {
                ExecutionEventKind.COLLISION_WORLD_CHANGED,
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.TRACKING_DIVERGED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
            }:
                rows = event.env_mask.nonzero(as_tuple=False).flatten().tolist()
                logger.log_info(
                    f"Execution event {event.kind.value}: env rows={rows}; "
                    f"{event.message}"
                )
            if (
                event.kind is ExecutionEventKind.REPLANNED
                and replanned_eef_path is None
                and ExecutionEventKind.COLLISION_WORLD_CHANGED in observed_events
            ):
                replanned_eef_path = _command_eef_positions(
                    robot,
                    session.active_commands,
                    control_part=CONTROL_PART,
                )
                replan_detour = _maximum_path_deviation(
                    replanned_eef_path,
                    initial_eef_path,
                )
                replan_clearance = _minimum_cuboid_clearance(
                    replanned_eef_path,
                    blocking_obstacle_pose,
                    size=OBSTACLE_SIZE,
                )
                _draw_eef_path(
                    sim,
                    "replanned_eef_path",
                    replanned_eef_path,
                    stride=TRAJECTORY_MARKER_STRIDE,
                    axis_len=0.055,
                    axis_size=0.0025,
                )
                _publish_path_overlays(
                    sim,
                    initial_eef_path,
                    replanned_eef_path,
                )
                logger.log_info(
                    "Replanned avoidance path maximum deviation from the "
                    f"initial path: {replan_detour.detach().cpu().tolist()} m; "
                    "minimum TCP-to-cube clearance: "
                    f"{replan_clearance.detach().cpu().tolist()} m.",
                    color="green",
                )

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="dynamic_obstacle_recovery_auto_play",
        look_at=(
            (1.20, -1.20, 0.95),
            (0.52, 0.06, 0.43),
            (0.0, 0.0, 1.0),
        ),
    )
    try:
        if recording_started:
            adapter.sleep(AUTO_PLAY_LEAD_IN_DURATION)
        result = runner.run_until_blocked(on_step=on_step)
        adapter.sleep(POST_EXECUTION_HOLD_DURATION)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if result.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(f"Closed-loop execution failed: {result.message}")
    if not args.no_obstacle_motion:
        if not obstacle_moved:
            raise RuntimeError("Execution completed before the obstacle could move.")
        if ExecutionEventKind.COLLISION_WORLD_CHANGED not in observed_events:
            raise RuntimeError("Obstacle motion did not invalidate the trajectory.")
        if ExecutionEventKind.REPLANNED not in observed_events:
            raise RuntimeError("Obstacle motion did not trigger replanning.")
        if (
            replanned_eef_path is None
            or replan_detour is None
            or replan_clearance is None
        ):
            raise RuntimeError("Replanned trajectory was not captured.")
        if (replan_detour < MINIMUM_REPLAN_DETOUR).any().item():
            raise RuntimeError(
                "Replanned trajectory did not detour around the moved obstacle: "
                f"deviation={replan_detour.detach().cpu().tolist()} m."
            )
        if (replan_clearance < MINIMUM_REPLAN_CLEARANCE).any().item():
            raise RuntimeError(
                "Replanned trajectory did not keep sufficient TCP clearance from "
                "the moved obstacle: "
                f"clearance={replan_clearance.detach().cpu().tolist()} m."
            )
    final_eef_position = robot.compute_fk(
        qpos=robot.get_qpos(name=CONTROL_PART),
        name=CONTROL_PART,
        to_matrix=True,
    )[:, :3, 3]
    final_eef_error = torch.linalg.vector_norm(
        final_eef_position - target_pose[:, :3, 3],
        dim=1,
    )
    if (final_eef_error > MAXIMUM_FINAL_EEF_ERROR).any().item():
        raise RuntimeError(
            "Execution stopped too far from the Cartesian goal: "
            f"error={final_eef_error.detach().cpu().tolist()} m."
        )
    logger.log_info(
        f"Execution completed after {result.command_count} accepted commands; "
        f"final EEF error={final_eef_error.detach().cpu().tolist()} m.",
        color="green",
    )

    serve_tutorial_scene(sim, args)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
