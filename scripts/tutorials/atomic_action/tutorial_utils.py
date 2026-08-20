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

"""Shared helpers for atomic-action tutorial scripts."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Collection, Sequence
from typing import Literal

import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ObjectSemantics,
    TimedTrajectory,
)
from embodichain.lab.sim.cfg import LightCfg, MarkerCfg, RenderCfg, RobotCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import (
    CuroboPlannerCfg,
    MotionGenCfg,
    MotionGenerator,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.robots import FrankaPandaCfg, URRobotCfg
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger

RECORD_WIDTH = 640
RECORD_HEIGHT = 480
VIEWER_WIDTH = 1600
VIEWER_HEIGHT = 900
AUTO_PLAY_RECORD_FPS = 20
AUTO_PLAY_RECORD_MAX_MEMORY = 2048
DEFAULT_AUTO_PLAY_LOOK_AT = (
    (-1.25, -1.15, 0.95),
    (-0.25, -0.02, 0.25),
    (0.0, 0.0, 1.0),
)
DEFAULT_AXIS_LEN = 0.06
DEFAULT_AXIS_SIZE = 0.003

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"
GRIPPER_MAX_OPEN_WIDTH = 0.100
GRIPPER_MIN_OPEN_WIDTH = 0.003
GRIPPER_FINGER_LENGTH = 0.10
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
DEFAULT_GRIPPER_CLOSE_QPOS = 0.024
DEFAULT_TUTORIAL_LIGHT_POS = (1.0, 0.0, 3.0)
_FRANKA_TUTORIAL_BASE_ROTATION = (0.0, 0.0, 180.0)
_DEFAULT_GRIPPER_TCP_Z = 0.17
_GRIPPER_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, _DEFAULT_GRIPPER_TCP_Z),
    (0.0, 0.0, 0.0, 1.0),
)
TOP_DOWN_EEF_ROTATION = (
    (-0.0539, -0.9985, -0.0022),
    (-0.9977, 0.0540, -0.0401),
    (0.0401, 0.0000, -0.9992),
)

TutorialCliFeature = Literal[
    "debug_state",
    "diagnose_plan",
    "grasp_sampling",
    "headless_play",
    "visualize_axes",
]
TutorialRobot = Literal["ur5", "franka"]
TUTORIAL_ROBOTS: tuple[TutorialRobot, ...] = ("ur5", "franka")


def create_tutorial_argument_parser(
    description: str,
    *,
    features: Collection[TutorialCliFeature] = (),
    default_device: str | None = None,
    default_renderer: str | None = None,
) -> argparse.ArgumentParser:
    """Create a launcher parser with the shared atomic-tutorial switches."""
    parser = argparse.ArgumentParser(description=description)
    add_env_launcher_args_to_parser(parser)
    defaults = {}
    if default_device is not None:
        defaults["device"] = default_device
    if default_renderer is not None:
        defaults["renderer"] = default_renderer
    if defaults:
        parser.set_defaults(**defaults)

    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--robot",
        choices=TUTORIAL_ROBOTS,
        default="ur5",
        help="Robot construction to use (default: ur5).",
    )
    if "debug_state" in features:
        parser.add_argument(
            "--debug_state",
            action="store_true",
            help="Log robot and object state during execution.",
        )
    if "diagnose_plan" in features:
        parser.add_argument(
            "--diagnose_plan",
            action="store_true",
            help="Plan and print diagnostics without playing the trajectory.",
        )
    if "grasp_sampling" in features:
        parser.add_argument("--n_sample", type=int, default=10000)
        parser.add_argument("--force_reannotate", action="store_true")
    if "headless_play" in features:
        parser.add_argument(
            "--headless_play",
            action="store_true",
            help="Execute trajectories without opening the viewer window.",
        )
    if "visualize_axes" in features:
        parser.add_argument(
            "--no_vis_eef_axis",
            action="store_true",
            help="Skip drawing target coordinate-frame markers.",
        )
    return parser


def make_ur5_solver_cfg(tcp_z: float) -> URSolverCfg:
    """Create the UR5 arm solver cfg used by atomic-action tutorials."""
    cfg = URSolverCfg(
        ur_type="ur5",
        end_link_name="ee_link",
        root_link_name="base_link",
        tcp=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, tcp_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    cfg.urdf_path = None
    return cfg


def create_tutorial_simulation(
    args: argparse.Namespace,
    *,
    arena_space: float = 2.5,
    light_pos: Sequence[float] = DEFAULT_TUTORIAL_LIGHT_POS,
) -> SimulationManager:
    """Create the shared simulation setup used by atomic-action tutorials.

    Args:
        args: Parsed launcher arguments containing environment count, device,
            and renderer selections.
        arena_space: Spacing between parallel simulation arenas in meters.
        light_pos: Position of the scene's key light.

    Returns:
        A simulation manager with the tutorial key light configured.
    """
    width, height = get_tutorial_window_size(args)
    sim = SimulationManager(
        SimulationManagerCfg(
            width=width,
            height=height,
            headless=True,
            num_envs=args.num_envs,
            sim_device=args.device,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_dt=1.0 / 100.0,
            arena_space=arena_space,
            visualization=visualization_cfg_from_args(args),
        )
    )
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=list(light_pos),
        )
    )
    return sim


def run_tutorial(main: Callable[[], None]) -> None:
    """Run a tutorial entry point and release its simulation at top level.

    ``SimulationManager.destroy`` defers native cleanup because action and
    planner locals may still retain wrapped C++ objects. Calling it only after
    ``main`` has unwound makes the cleanup deterministic and avoids native
    teardown during Python interpreter finalization.

    Args:
        main: Zero-argument tutorial entry point.
    """
    interrupted = False
    try:
        try:
            main()
        except KeyboardInterrupt:
            # Handle Ctrl+C before native cleanup.  An active traceback keeps
            # main() locals (including borrowed C++ material wrappers) alive;
            # destroying World first would make their later destructors unsafe.
            interrupted = True
            logger.log_info("Tutorial interrupted; shutting down cleanly.")
    finally:
        if SimulationManager.is_instantiated():
            sim = SimulationManager.get_instance()
            if not getattr(sim, "_is_constructed", False):
                SimulationManager.reset(getattr(sim, "instance_id", 0))
            else:
                if sim.is_window_recording():
                    sim.stop_window_record()
                sim.wait_window_record_saves()
                sim.destroy(exit_process=False)
                SimulationManager.flush_cleanup_queue()

    if interrupted:
        raise SystemExit(130)


def add_ur5_gripper_robot(
    sim: SimulationManager,
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
    tcp_z: float = _DEFAULT_GRIPPER_TCP_Z,
) -> Robot:
    """Add the standard UR5 plus PGI gripper tutorial robot.

    Args:
        sim: Simulation manager that owns the robot.
        init_pos: Root position of the robot in its arena.

    Returns:
        The added robot instance.
    """
    return sim.add_robot(
        cfg=create_ur5_gripper_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
            tcp_z=tcp_z,
        )
    )


def add_franka_panda_robot(
    sim: SimulationManager,
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
) -> Robot:
    """Add a Franka arm with the standard PGI tutorial gripper.

    Args:
        sim: Simulation manager that owns the robot.
        init_pos: Root position of the robot in its arena.
        init_qpos: Optional full robot joint configuration.

    Returns:
        The added robot instance.
    """
    return sim.add_robot(
        cfg=create_franka_panda_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
        )
    )


def add_tutorial_robot(
    sim: SimulationManager,
    robot_type: TutorialRobot,
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
) -> Robot:
    """Add a selected tutorial robot with the shared PGI gripper.

    Args:
        sim: Simulation manager that owns the robot.
        robot_type: Tutorial robot family to construct.
        init_pos: Root position of the robot in its arena.
        init_qpos: Optional full robot joint configuration.

    Returns:
        The added robot instance.

    Raises:
        ValueError: If ``robot_type`` is not supported.
    """
    return sim.add_robot(
        cfg=create_tutorial_robot_cfg(
            robot_type,
            init_pos=init_pos,
            init_qpos=init_qpos,
        )
    )


def create_toppra_motion_generator(robot: Robot) -> MotionGenerator:
    """Create the standard TOPPRA motion generator for a tutorial robot.

    Args:
        robot: Robot whose trajectories will be planned.

    Returns:
        The configured motion generator.
    """
    return MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )


def create_curobo_motion_generator(robot: Robot) -> MotionGenerator:
    """Create a cuRobo-backed motion generator for a tutorial robot.

    Args:
        robot: Robot whose trajectories will be planned.

    Returns:
        The configured motion generator with an empty external collision world.
    """
    return MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=CuroboPlannerCfg(robot_uid=robot.uid))
    )


def get_hand_open_close_qpos(
    robot: Robot,
    *,
    hand_control_part: str = "hand",
    close_qpos: float = DEFAULT_GRIPPER_CLOSE_QPOS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the open limit and a safe closed position for a PGI gripper.

    Args:
        robot: Robot containing the gripper control part.
        hand_control_part: Name of the gripper control part.
        close_qpos: Desired per-joint closed position, clamped to joint limits.

    Returns:
        Open and closed joint-position tensors on the robot device.
    """
    hand_limits = robot.get_qpos_limits(name=hand_control_part)[0].to(
        device=robot.device, dtype=torch.float32
    )
    return hand_limits[:, 0], torch.clamp(
        torch.full_like(hand_limits[:, 1], close_qpos),
        min=hand_limits[:, 0],
        max=hand_limits[:, 1],
    )


def create_antipodal_semantics(
    obj: RigidObject,
    *,
    label: str,
    n_sample: int,
    force_reannotate: bool,
) -> ObjectSemantics:
    """Describe a rigid object using the standard PGI antipodal-grasp setup.

    Args:
        obj: Rigid object that will be grasped.
        label: Human-readable object category.
        n_sample: Number of grasp-pair samples to generate.
        force_reannotate: Whether to ignore cached grasp annotations.

    Returns:
        Object semantics with mesh data stored only on its affordance.
    """
    vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    triangles = obj.get_triangles(env_ids=[0])[0]
    return ObjectSemantics(
        label=label,
        geometry={},
        affordance=AntipodalAffordance(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            gripper_collision_cfg=GripperCollisionCfg(
                max_open_length=GRIPPER_MAX_OPEN_WIDTH,
                finger_length=GRIPPER_FINGER_LENGTH,
                y_thickness=GRIPPER_Y_THICKNESS,
                root_z_width=GRIPPER_ROOT_Z_WIDTH,
                open_check_margin=0.03,
                point_sample_dense=0.012,
            ),
            generator_cfg=GraspGeneratorCfg(
                viser_port=11801,
                antipodal_sampler_cfg=AntipodalSamplerCfg(
                    n_sample=n_sample,
                    max_length=GRIPPER_MAX_OPEN_WIDTH,
                    min_length=GRIPPER_MIN_OPEN_WIDTH,
                ),
                is_partial_annotate=False,
                is_filter_ground_collision=False,
            ),
            force_reannotate=force_reannotate,
        ),
        entity=obj,
    )


def make_top_down_eef_pose(position: torch.Tensor) -> torch.Tensor:
    """Build the standard downward-facing TCP pose at ``position``.

    Args:
        position: Three-dimensional position for the tool center point.

    Returns:
        A homogeneous end-effector pose.
    """
    pose = torch.eye(4, dtype=torch.float32, device=position.device)
    pose[:3, :3] = torch.tensor(
        TOP_DOWN_EEF_ROTATION,
        dtype=torch.float32,
        device=position.device,
    )
    pose[:3, 3] = position
    return pose


def format_tensor(tensor: torch.Tensor) -> str:
    """Format a tensor compactly for tutorial log messages.

    Args:
        tensor: Tensor to render for a human-readable log message.

    Returns:
        A rounded Python-list representation.
    """
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def make_eef_pose_at(robot: Robot, position: torch.Tensor) -> torch.Tensor:
    """Reuse the current TCP orientation at a requested position.

    Args:
        robot: Robot whose current arm pose supplies the orientation.
        position: Position tensor with shape ``(3,)`` or ``(num_envs, 3)``.

    Returns:
        A single or batched homogeneous pose matching ``position``.
    """
    pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"), name="arm", to_matrix=True
    ).clone()
    if position.dim() == 1:
        pose = pose[0]
    pose[..., :3, 3] = position
    return pose


def initialize_pre_pick_robot_pose(
    robot: Robot,
    obj: RigidObject,
    hand_open: torch.Tensor,
    *,
    height: float = 0.36,
) -> None:
    """Set a tutorial robot at a deterministic open-gripper pose above an object.

    Args:
        robot: Robot to initialize.
        obj: Object below the pre-pick TCP pose.
        hand_open: Open gripper joint positions.
        height: World-Z coordinate for the pre-pick TCP pose.

    Raises:
        RuntimeError: If the pre-pick pose is unreachable.
    """
    position = obj.get_local_pose(to_matrix=True)[:, :3, 3].clone()
    position[:, 2] = height
    ik_success, arm_qpos = robot.compute_ik(
        pose=make_eef_pose_at(robot, position),
        joint_seed=robot.get_qpos(name="arm"),
        name="arm",
    )
    if not torch.all(ik_success):
        raise RuntimeError("Failed to initialize the robot at the pre-pick pose.")

    hand_qpos = hand_open.unsqueeze(0).repeat(robot.get_qpos().shape[0], 1)
    for target in (False, True):
        robot.set_qpos(arm_qpos, name="arm", target=target)
        robot.set_qpos(hand_qpos, name="hand", target=target)
    robot.clear_dynamics()


def prepare_tutorial_scene(
    sim: SimulationManager,
    args: argparse.Namespace,
    prompt: str,
) -> bool:
    """Open an interactive scene and optionally wait before planning.

    Args:
        sim: Simulation manager used by the tutorial.
        args: Parsed tutorial arguments.
        prompt: Prompt displayed for interactive runs.

    Returns:
        Whether the caller should retain later interactive pauses.
    """
    wait_for_user = should_wait_for_tutorial_input(args)
    if should_open_tutorial_window(args):
        sim.open_window()
    if wait_for_user:
        input(prompt)
    return wait_for_user


def publish_tutorial_scene(sim: SimulationManager, args: argparse.Namespace) -> None:
    """Immediately push the current scene to the Viser browser.

    Tutorials build their scene and then run slow grasp annotation and motion
    planning before the next :meth:`SimulationManager.update`. Without this call
    the browser stays empty during that gap. It is a no-op unless ``--viser`` is
    set, so callers can always invoke it right after assembling the scene.

    Args:
        sim: Simulation manager whose scene should be published.
        args: Parsed tutorial arguments.
    """
    if getattr(args, "viser", False):
        sim.capture_visualization_safely(force=True)


def serve_tutorial_scene(
    sim: SimulationManager,
    args: argparse.Namespace,
    *,
    message: str = "Viser browser preview open. Press Ctrl+C to exit.",
) -> bool:
    """Keep stepping the simulation so the Viser browser stays live.

    After a tutorial replays its trajectory the process would otherwise exit and
    tear the browser scene down. When ``--viser`` is set this blocks with a
    physics-stepping loop until the user interrupts (``Ctrl+C``); otherwise it is
    a no-op, so callers can always invoke it just before cleanup.

    Args:
        sim: Simulation manager to keep stepping.
        args: Parsed tutorial arguments.
        message: Informational message logged before the keep-alive loop.

    Returns:
        Whether a keep-alive loop was run.
    """
    if not getattr(args, "viser", False):
        return False
    logger.log_info(message, color="green")
    try:
        while True:
            sim.update(step=1)
    except KeyboardInterrupt:
        pass
    return True


def replay_trajectory(
    sim: SimulationManager,
    robot: Robot,
    trajectory: TimedTrajectory | torch.Tensor,
    args: argparse.Namespace,
    *,
    video_prefix: str,
    hold_steps: int,
    trajectory_sim_steps: int | None = None,
    hold_sim_steps: int = 2,
    joint_ids: list[int] | None = None,
    on_trajectory_step: Callable[[int, int], None] | None = None,
    look_at: tuple[Sequence[float], Sequence[float], Sequence[float]] | None = None,
    record: bool = True,
) -> None:
    """Replay a trajectory, optionally record it, and hold its final pose.

    Args:
        sim: Simulation manager to step and record.
        robot: Robot receiving full-DOF trajectory positions.
        trajectory: Timed full-robot trajectory, or a legacy position tensor
            with shape ``(num_envs, n_steps, dof)``.
        args: Parsed tutorial arguments controlling auto-play recording.
        video_prefix: Output video filename prefix.
        hold_steps: Number of final-pose simulation updates after the trajectory.
        trajectory_sim_steps: Optional fixed physics steps per waypoint. When
            omitted for a ``TimedTrajectory``, its arrival intervals determine
            the synchronized physics-step count. Legacy tensors default to four.
        hold_sim_steps: Physics steps for each final-pose update.
        joint_ids: Optional joint IDs when controlling a robot subset.
        on_trajectory_step: Optional callback run after each trajectory update.
        look_at: Optional recorder camera pose for auto-play videos.
        record: Whether to start auto-play recording for this replay.
    """
    if trajectory_sim_steps is not None and trajectory_sim_steps <= 0:
        raise ValueError("trajectory_sim_steps must be greater than zero.")
    timed = trajectory if isinstance(trajectory, TimedTrajectory) else None
    positions = timed.positions if timed is not None else trajectory
    if not isinstance(positions, torch.Tensor) or positions.dim() != 3:
        raise ValueError("trajectory positions must have shape (B, N, D).")
    if positions.shape[1] == 0:
        raise ValueError("trajectory must contain at least one waypoint.")

    recording_started = (
        start_auto_play_recording(
            sim,
            args,
            video_prefix=video_prefix,
            look_at=DEFAULT_AUTO_PLAY_LOOK_AT if look_at is None else look_at,
        )
        if record
        else False
    )
    try:
        total_steps = positions.shape[1]
        for step_idx in range(total_steps):
            if joint_ids is None:
                robot.set_qpos(positions[:, step_idx, :])
            else:
                robot.set_qpos(positions[:, step_idx, :], joint_ids=joint_ids)
            waypoint_sim_steps = trajectory_sim_steps
            if waypoint_sim_steps is None and timed is not None:
                next_index = min(step_idx + 1, total_steps - 1)
                duration = float(timed.dt[:, next_index].max().item())
                step_ratio = duration / float(sim.sim_config.physics_dt)
                nearest_step_count = round(step_ratio)
                waypoint_sim_steps = max(
                    1,
                    (
                        nearest_step_count
                        if math.isclose(
                            step_ratio,
                            nearest_step_count,
                            rel_tol=1.0e-6,
                            abs_tol=1.0e-9,
                        )
                        else math.ceil(step_ratio)
                    ),
                )
            sim.update(step=4 if waypoint_sim_steps is None else waypoint_sim_steps)
            if on_trajectory_step is not None:
                on_trajectory_step(step_idx, total_steps)
            time.sleep(1e-2)

        final_qpos = positions[:, -1, :]
        for _ in range(hold_steps):
            if joint_ids is None:
                robot.set_qpos(final_qpos)
            else:
                robot.set_qpos(final_qpos, joint_ids=joint_ids)
            sim.update(step=hold_sim_steps)
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)


def make_clear_dynamics_callback(
    obj: RigidObject,
    clear_after_step: int,
) -> Callable[[int, int], None]:
    """Create a one-shot replay callback that freezes an attached object."""
    dynamics_cleared = False

    def clear_dynamics(step_idx: int, _: int) -> None:
        nonlocal dynamics_cleared
        if not dynamics_cleared and step_idx + 1 >= clear_after_step:
            obj.clear_dynamics()
            dynamics_cleared = True

    return clear_dynamics


def get_tutorial_window_size(args: argparse.Namespace) -> tuple[int, int]:
    """Return the viewer window size used by atomic-action tutorials."""
    return VIEWER_WIDTH, VIEWER_HEIGHT


def should_open_tutorial_window(args: argparse.Namespace) -> bool:
    """Return whether an interactive viewer window should be opened."""
    return not (
        getattr(args, "headless", False)
        or getattr(args, "viser", False)
        or getattr(args, "diagnose_plan", False)
        or getattr(args, "headless_play", False)
    )


def should_wait_for_tutorial_input(args: argparse.Namespace) -> bool:
    """Return whether the tutorial should pause for terminal input."""
    return not (
        getattr(args, "auto_play", False)
        or getattr(args, "headless", False)
        or getattr(args, "viser", False)
        or getattr(args, "diagnose_plan", False)
        or getattr(args, "headless_play", False)
    )


def start_auto_play_recording(
    sim: SimulationManager,
    args: argparse.Namespace,
    video_prefix: str,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_AUTO_PLAY_LOOK_AT,
) -> bool:
    """Start recording for ``--auto_play`` tutorial runs."""
    if not getattr(args, "auto_play", False):
        return False

    original_width = sim.sim_config.width
    original_height = sim.sim_config.height
    try:
        sim.sim_config.width = RECORD_WIDTH
        sim.sim_config.height = RECORD_HEIGHT
        if not sim.start_window_record(
            fps=AUTO_PLAY_RECORD_FPS,
            max_memory=AUTO_PLAY_RECORD_MAX_MEMORY,
            video_prefix=video_prefix,
            look_at=look_at,
            use_sim_time=True,
        ):
            raise RuntimeError("Failed to start auto_play recording.")
    finally:
        sim.sim_config.width = original_width
        sim.sim_config.height = original_height
    return True


def stop_auto_play_recording(
    sim: SimulationManager,
    recording_started: bool,
) -> None:
    """Stop recording and wait until the mp4 has been written."""
    if not recording_started:
        return

    if sim.is_window_recording():
        sim.stop_window_record()
    sim.wait_window_record_saves()


def draw_axis_marker(
    sim: SimulationManager,
    name: str,
    xpos: torch.Tensor,
    axis_len: float = DEFAULT_AXIS_LEN,
    axis_size: float = DEFAULT_AXIS_SIZE,
    arena_index: int = -1,
) -> None:
    """Draw a named coordinate-frame marker for a semantic tutorial target."""
    arena_offsets = sim.arena_offsets
    num_envs = arena_offsets.shape[0]

    # Normalize a single (4, 4) pose to (1, 4, 4) so it is not mistaken for a
    # per-environment batch when num_envs happens to equal 4.
    if xpos.dim() == 2:
        xpos = xpos.unsqueeze(0)

    if num_envs == xpos.shape[0]:
        # add arena offsets to xpos
        draw_xpos = xpos.clone()
        draw_xpos[:, :3, 3] += arena_offsets
    else:
        draw_xpos = xpos
    sim.draw_marker(
        cfg=MarkerCfg(
            name=name,
            marker_type="axis",
            axis_xpos=draw_xpos,
            axis_size=axis_size,
            axis_len=axis_len,
            arena_index=arena_index,
        )
    )


def broadcast_pose_batch(pose: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Expand a single pose to ``(num_envs, 4, 4)`` when needed."""
    if pose.dim() == 2 and pose.shape == (4, 4):
        return pose.unsqueeze(0).repeat(num_envs, 1, 1)
    if pose.dim() == 3 and pose.shape[1:] == (4, 4):
        if pose.shape[0] == num_envs:
            return pose
        if pose.shape[0] == 1:
            return pose.repeat(num_envs, 1, 1)
    raise ValueError(
        "Expected a pose with shape "
        f"(4, 4), (1, 4, 4), or ({num_envs}, 4, 4) for num_envs={num_envs}, "
        f"got {tuple(pose.shape)}."
    )


def broadcast_waypoint_pose_batch(pose: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Expand waypoint poses to ``(num_envs, n_waypoint, 4, 4)`` when needed."""
    if pose.dim() == 3 and pose.shape[1:] == (4, 4):
        return pose.unsqueeze(0).repeat(num_envs, 1, 1, 1)
    if pose.dim() == 4 and pose.shape[2:] == (4, 4):
        if pose.shape[0] == num_envs:
            return pose
        if pose.shape[0] == 1:
            return pose.repeat(num_envs, 1, 1, 1)
    raise ValueError(
        "Expected waypoint poses with shape "
        f"(n_waypoint, 4, 4), (1, n_waypoint, 4, 4), or ({num_envs}, n_waypoint, 4, 4) "
        f"for num_envs={num_envs}, got {tuple(pose.shape)}."
    )


def clone_local_pose_from_first_env(entity) -> torch.Tensor:
    """Copy the first environment's local pose onto every environment."""
    pose = entity.get_local_pose(to_matrix=True)
    if pose.dim() != 3 or pose.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected entity local pose with shape (num_envs, 4, 4), got {tuple(pose.shape)}."
        )
    shared_pose = pose[:1].clone().repeat(pose.shape[0], 1, 1)
    entity.set_local_pose(shared_pose)
    return shared_pose


def create_ur5_gripper_robot_cfg(
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
    tcp_z: float = _DEFAULT_GRIPPER_TCP_Z,
) -> RobotCfg:
    """Build a UR5 arm + DH_PGI_140_80 gripper robot configuration.

    The arm is taken from :class:`~embodichain.lab.sim.robots.ur_robot.URRobotCfg`
    so the URDF, joint names and drive defaults match the canonical UR family
    config. The gripper and tool-center-point offset are added on top to match
    the existing atomic-action tutorial setups.

    .. attention::
        :class:`~embodichain.lab.sim.cfg.URDFCfg` defaults to upper-casing joint
        names during multi-component assembly. The override dict passed to
        :meth:`URRobotCfg.from_dict` sets
        ``urdf_cfg.name_case = {"joint": "lower", "link": "lower"}`` so the
        assembled robot keeps the source URDF's lowercase joint names
        (``joint1``..``joint6`` and ``gripper_finger1_joint_1``), matching the
        control parts produced by
        :class:`~embodichain.lab.sim.robots.ur_robot.URRobotCfg`.

    Args:
        init_pos: Initial root position of the robot in the arena.

    Returns:
        A fully populated :class:`~embodichain.lab.sim.cfg.RobotCfg`.
    """
    qpos = (
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0]
        if init_qpos is None
        else list(init_qpos)
    )
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur5",
            "uid": "UR5",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": GRIPPER_URDF_PATH,
                    }
                ],
            },
            "control_parts": {
                "hand": [GRIPPER_HAND_JOINT_PATTERN],
            },
            "drive_pros": {
                "stiffness": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e3,
                },
                "damping": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e2,
                },
                "max_effort": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e4,
                },
            },
            "solver_cfg": {
                "arm": {
                    "tcp": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, tcp_z],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            },
            "init_qpos": qpos,
            "init_pos": init_pos,
        }
    )


def create_franka_panda_robot_cfg(
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
) -> RobotCfg:
    """Build a Franka arm + PGI gripper configuration for the tutorials.

    The shared scenes place manipulation targets on the robot's negative-X
    side, so the base is rotated 180 degrees. The arm-only Franka URDF is
    assembled with the same DH_PGI_140_80 component, hand control part, drive
    properties, open state, and TCP offset used by the UR5 tutorial robot.

    Args:
        init_pos: Initial root position of the robot in the arena.
        init_qpos: Optional full robot joint configuration.

    Returns:
        A fully populated :class:`~embodichain.lab.sim.cfg.RobotCfg`.
    """
    overrides = {
        "robot_type": "panda",
        "uid": "FrankaPanda",
        "init_pos": init_pos,
        "init_rot": _FRANKA_TUTORIAL_BASE_ROTATION,
        "urdf_cfg": {
            "components": [
                {
                    "component_type": "arm",
                    "urdf_path": "Franka/Panda/Panda.urdf",
                },
                {
                    "component_type": "hand",
                    "urdf_path": GRIPPER_URDF_PATH,
                },
            ],
        },
        "control_parts": {"hand": [GRIPPER_HAND_JOINT_PATTERN]},
        "drive_pros": {
            "stiffness": {GRIPPER_HAND_JOINT_PATTERN: 1e3},
            "damping": {GRIPPER_HAND_JOINT_PATTERN: 1e2},
            "max_effort": {GRIPPER_HAND_JOINT_PATTERN: 1e4},
        },
        "solver_cfg": {
            "arm": {
                "end_link_name": "fr3_link8",
                "tcp": _GRIPPER_TCP,
            }
        },
    }
    if init_qpos is not None:
        overrides["init_qpos"] = list(init_qpos)
    cfg = FrankaPandaCfg.from_dict(overrides)
    if init_qpos is None:
        cfg.init_qpos[-2:] = [0.0, 0.0]
    for drive_values in (
        cfg.drive_pros.stiffness,
        cfg.drive_pros.damping,
        cfg.drive_pros.max_effort,
    ):
        drive_values.pop("fr3_finger_joint[1-2]", None)
    return cfg


def create_tutorial_robot_cfg(
    robot_type: TutorialRobot,
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
) -> RobotCfg:
    """Build a selected tutorial arm with the common PGI gripper contract.

    Args:
        robot_type: Tutorial robot family to construct.
        init_pos: Initial root position of the robot in its arena.
        init_qpos: Optional full robot joint configuration.

    Returns:
        A UR5 or Franka robot configuration exposing ``arm`` and ``hand``.

    Raises:
        ValueError: If ``robot_type`` is not supported.
    """
    if robot_type == "ur5":
        return create_ur5_gripper_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
        )
    if robot_type == "franka":
        return create_franka_panda_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
        )
    raise ValueError(
        f"Unsupported tutorial robot {robot_type!r}; expected one of {TUTORIAL_ROBOTS}."
    )


__all__ = [
    "DEFAULT_AUTO_PLAY_LOOK_AT",
    "DEFAULT_AXIS_LEN",
    "DEFAULT_AXIS_SIZE",
    "DEFAULT_GRIPPER_CLOSE_QPOS",
    "DEFAULT_TUTORIAL_LIGHT_POS",
    "GRIPPER_HAND_JOINT_PATTERN",
    "GRIPPER_URDF_PATH",
    "TOP_DOWN_EEF_ROTATION",
    "TutorialCliFeature",
    "TutorialRobot",
    "TUTORIAL_ROBOTS",
    "add_franka_panda_robot",
    "add_tutorial_robot",
    "add_ur5_gripper_robot",
    "broadcast_pose_batch",
    "broadcast_waypoint_pose_batch",
    "clone_local_pose_from_first_env",
    "create_antipodal_semantics",
    "create_curobo_motion_generator",
    "create_franka_panda_robot_cfg",
    "create_toppra_motion_generator",
    "create_tutorial_argument_parser",
    "create_tutorial_robot_cfg",
    "create_tutorial_simulation",
    "create_ur5_gripper_robot_cfg",
    "format_tensor",
    "get_hand_open_close_qpos",
    "initialize_pre_pick_robot_pose",
    "make_ur5_solver_cfg",
    "make_eef_pose_at",
    "make_clear_dynamics_callback",
    "make_top_down_eef_pose",
    "get_tutorial_window_size",
    "prepare_tutorial_scene",
    "publish_tutorial_scene",
    "replay_trajectory",
    "run_tutorial",
    "serve_tutorial_scene",
    "should_open_tutorial_window",
    "should_wait_for_tutorial_input",
    "start_auto_play_recording",
    "stop_auto_play_recording",
    "draw_axis_marker",
]
