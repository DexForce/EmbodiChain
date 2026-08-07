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

"""cuRobo V2 collision-aware planning through the atomic-action interface.

The demo creates one or more copies of the selected robot and a kinematic
cuboid represented in both DexSim and cuRobo. With multiple environments, each
obstacle receives a small reproducible XY/yaw perturbation and cuRobo allocates
an independent collision world for each environment. The demo then executes a
batched ``MoveEndEffector`` action through :class:`AtomicActionEngine`, replays
the returned full-robot-DoF trajectories, and reports the final TCP error for
every environment.

Run from the repository root::

    python examples/sim/planners/curobo_planner.py --headless
    python examples/sim/planners/curobo_planner.py --headless --num_envs 4
    python examples/sim/planners/curobo_planner.py --headless --device cuda:1

Requirements: an NVIDIA CUDA device and the CUDA-matched EmbodiChain cuRobo V2
extra installed in the active environment.  Installation instructions:
https://nvlabs.github.io/curobo/latest/getting-started/installation.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Prefer the in-repo source over any installed (possibly stale) embodichain
# package, so this example exercises the current code. The demo relies on the
# cuRobo adapter's URDF-based robot-YAML auto-generation, which lives in the
# source tree and may not be present in an older installed copy.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.visualization import (
    VisualizationCfg,
    visualization_cfg_from_args,
)
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RenderCfg, RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg, Robot, RigidObject
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
import numpy as np
from embodichain.lab.sim.robots import FrankaPandaCfg, URRobotCfg, DexforceW1Cfg
from embodichain.lab.sim.shapes import CubeCfg

__all__ = ["main"]


DEFAULT_RECORD_FPS = 20
DEFAULT_RECORD_MAX_MEMORY = 2048
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_OBSTACLE_XY_PERTURBATION = 0.02
DEFAULT_OBSTACLE_YAW_PERTURBATION_DEG = 5.0
DEFAULT_RANDOM_SEED = 0
DEFAULT_RECORD_LOOK_AT = (
    (1.8, -1.8, 1.35),
    (0.35, 0.10, 0.40),
    (0.0, 0.0, 1.0),
)
CUROBO_INSTALL_URL = (
    "https://nvlabs.github.io/curobo/latest/getting-started/installation.html"
)


def parse_args() -> argparse.Namespace:
    """Parse the interactive/headless playback and recording controls."""
    parser = argparse.ArgumentParser(
        description="Run cuRobo V2 through EmbodiChain AtomicActionEngine."
    )
    add_env_launcher_args_to_parser(parser)
    # This standalone example does not merge a gym config after parsing, so
    # override the launcher's ``None`` sentinel with a concrete single-world
    # default.
    parser.set_defaults(arena_space=2.0, num_envs=1)
    # Backward-compatible aliases used by older versions of this example.
    parser.add_argument(
        "--step-repeat",
        type=int,
        default=4,
        help="Simulation updates for each planned trajectory waypoint.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=20,
        help="Simulation updates to hold before and after trajectory playback.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            "cuRobo planning attempts per request. Lower values are faster; "
            "increase this if a harder scene fails to find a path."
        ),
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=DEFAULT_RECORD_FPS,
        help="Output video FPS for automatic headless recording.",
    )
    parser.add_argument(
        "--record-save-path",
        type=str,
        default=None,
        help="Optional MP4 output path for headless recording.",
    )
    parser.add_argument(
        "--disable-record",
        action="store_true",
        help="Disable automatic offscreen recording in headless mode.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        help="Robot type for the cuRobo demo (franka, ur, w1).",
    )
    parser.add_argument(
        "--obstacle-xy-perturbation",
        type=float,
        default=DEFAULT_OBSTACLE_XY_PERTURBATION,
        help=(
            "Maximum per-axis XY obstacle position perturbation in meters when "
            "num_envs > 1."
        ),
    )
    parser.add_argument(
        "--obstacle-yaw-perturbation-deg",
        type=float,
        default=DEFAULT_OBSTACLE_YAW_PERTURBATION_DEG,
        help=(
            "Maximum absolute obstacle yaw perturbation in degrees when "
            "num_envs > 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for per-environment obstacle perturbations.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable in-process cuRobo CUDA graphs with renderer-compatible "
            "thread-local stream capture (default: enabled; use "
            "--no-cuda-graph to disable)."
        ),
    )
    return parser.parse_args()


def _resolve_device(device: str, gpu_id: int) -> str:
    """Resolve launcher device syntax to an explicit simulation device."""
    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device value {device!r}.") from exc
    if resolved.type != "cuda":
        return str(resolved)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested, but CUDA is not available."
        )
    index = gpu_id if resolved.index is None else resolved.index
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; torch reports "
            f"{torch.cuda.device_count()} device(s)."
        )
    return f"cuda:{index}"


def _check_runtime(curobo_gpu_id: int) -> None:
    """Raise clear errors before allocating the CUDA simulation scene."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuRobo V2 requires a CUDA-capable NVIDIA GPU, but CUDA is not "
            "available. This demo cannot run on CPU."
        )
    try:
        import curobo  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cuRobo V2 is not installed. From the EmbodiChain repository root, "
            "install the extra matching the CUDA environment: "
            '`uv pip install ".[curobo-cu12]"` for CUDA 12.x or '
            '`uv pip install ".[curobo-cu13]"` for CUDA 13.x '
            f"(see {CUROBO_INSTALL_URL})."
        ) from exc
    if curobo_gpu_id < 0 or curobo_gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"cuRobo CUDA device index {curobo_gpu_id} is unavailable; torch "
            f"reports {torch.cuda.device_count()} device(s)."
        )


def _build_scene(
    headless: bool,
    robot_type: str = "franka",
    device: str = "cuda:0",
    num_envs: int = 1,
    renderer: str = "auto",
    arena_space: float = 2.0,
    gpu_id: int = 0,
    visualization: VisualizationCfg | None = None,
) -> tuple[SimulationManager, Robot, RigidObject, torch.Tensor, str]:
    """Create the batched robot scene with an identical cuboid in each arena."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device=device,
            num_envs=num_envs,
            arena_space=arena_space,
            gpu_id=gpu_id,
            render_cfg=RenderCfg(renderer=renderer),
            visualization=visualization or VisualizationCfg(),
        )
    )
    if robot_type == "franka":
        control_part = "arm"
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict(
                {
                    "uid": "franka",
                    "robot_type": "panda",
                    "init_qpos": [0.0, -0.5, 0.0, -2.3, 0.0, 1.8, 0.741, 0.04, 0.04],
                }
            )
        )
        demo_block_size = [0.18, 0.3, 0.36]
        demo_block_position = (0.40, 0.0, 0.18)

        target_xpos = torch.tensor(
            [
                [
                    [9.9896e-01, 4.3707e-02, -1.2806e-02, 6.5e-01],
                    [4.3759e-02, -9.9903e-01, 3.7920e-03, 8.5299e-04],
                    [-1.2628e-02, -4.3484e-03, -9.9991e-01, 2.0e-01],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ]
            ],
            device=robot.device,
        )
    elif robot_type == "ur":
        control_part = "arm"
        hand_urdf_path = get_data_path(
            "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
        )
        hand_attach_xpos = np.eye(4)
        try:
            from scipy.spatial.transform import Rotation as _Rotation
        except ImportError as exc:  # pragma: no cover - exercised only without SciPy
            raise ImportError(
                "The '--robot ur' demo path requires SciPy. Install it with "
                "`pip install scipy`."
            ) from exc
        hand_attach_xpos[:3, :3] = _Rotation.from_rotvec(
            [90, 0, 0], degrees=True
        ).as_matrix()
        robot = sim.add_robot(
            cfg=URRobotCfg.from_dict(
                {
                    "robot_type": "ur10",
                    "uid": "ur10_with_brainco",
                    "urdf_cfg": {
                        "components": [
                            {
                                "component_type": "hand",
                                "urdf_path": hand_urdf_path,
                                "transform": hand_attach_xpos,
                            },
                        ]
                    },
                    "control_parts": {
                        "hand": [
                            "LEFT_HAND_THUMB1",
                            "LEFT_HAND_THUMB2",
                            "LEFT_HAND_INDEX",
                            "LEFT_HAND_MIDDLE",
                            "LEFT_HAND_RING",
                            "LEFT_HAND_PINKY",
                        ],
                    },
                    "drive_pros": {
                        "stiffness": {"LEFT_[A-Z|_]+[0-9]?": 1e2},
                        "damping": {"LEFT_[A-Z|_]+[0-9]?": 1e1},
                        "max_effort": {"LEFT_[A-Z|_]+[0-9]?": 1e3},
                        "drive_type": "force",
                    },
                    "solver_cfg": {"arm": {"tcp": np.eye(4)}},
                    "init_qpos": [
                        0.0,
                        -np.pi / 2,
                        -np.pi / 2,
                        2.5,
                        -np.pi / 2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.5,
                        -0.00016,
                        -0.00010,
                        -0.00013,
                        -0.00009,
                        0.0,
                    ],
                }
            )
        )
        demo_block_size = [0.18, 0.3, 0.36]
        demo_block_position = (0.60, 0.0, 0.18)
        target_xpos = torch.tensor(
            [
                [
                    [9.9896e-01, 4.3707e-02, -1.2806e-02, 8.5e-01],
                    [4.3759e-02, -9.9903e-01, 3.7920e-03, 8.5299e-04],
                    [-1.2628e-02, -4.3484e-03, -9.9991e-01, 3.0e-01],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ]
            ],
            device=robot.device,
        )
    elif robot_type == "w1":
        control_part = "right_arm"
        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
            }
        )
        cfg.solver_cfg["left_arm"].tcp = np.array(
            [
                [1.0, 0.0, 0.0, 0.012],
                [0.0, 1.0, 0.0, 0.04],
                [0.0, 0.0, 1.0, 0.11],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        cfg.solver_cfg["right_arm"].tcp = np.array(
            [
                [1.0, 0.0, 0.0, 0.012],
                [0.0, 1.0, 0.0, -0.04],
                [0.0, 0.0, 1.0, 0.11],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        cfg.init_qpos = [
            1.0000e00,
            -2.0000e00,
            1.0000e00,
            0.0000e00,
            -2.6921e-05,
            -2.6514e-03,
            -1.5708e00,
            1.4575e00,
            -7.8540e-01,
            1.2834e-01,
            1.5708e00,
            -2.2310e00,
            -7.8540e-01,
            1.4461e00,
            -1.5708e00,
            1.6716e00,
            7.8540e-01,
            7.6745e-01,
            0.0000e00,
            3.8108e-01,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            1.5000e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            1.5000e00,
            6.9974e-02,
            7.3950e-02,
            6.6574e-02,
            6.0923e-02,
            0.0000e00,
            6.7342e-02,
            7.0862e-02,
            6.3684e-02,
            5.7822e-02,
            0.0000e00,
        ]
        robot = sim.add_robot(cfg=cfg)

        demo_block_size = [0.2, 0.2, 0.2]
        demo_block_position = (0.36, -0.15, 0.88)
        target_xpos = torch.tensor(
            [
                [
                    [2.2020e-03, 3.4217e-01, 9.3964e-01, 4.6395e-01],
                    [1.5398e-04, -9.3964e-01, 3.4217e-01, -1.7e-01],
                    [1.0000e00, -6.0877e-04, -2.1218e-03, 6.80e-01],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ]
            ],
            device=robot.device,
        )

        # sim.open_window()
        # # sim.update(50)
        # current_qpos = robot.get_qpos(name=control_part)
        # current_xpos = robot.compute_fk(name=control_part, qpos=current_qpos, to_matrix=True)
        # print(f"Current {control_part} TCP pose:\n{current_xpos}")
        # import ipdb; ipdb.set_trace()

    else:
        raise ValueError(f"Unknown robot type '{robot_type}' for cuRobo demo.")

    if robot is None:
        raise RuntimeError(f"Failed to add robot '{robot_type}' to the cuRobo demo.")
    target_xpos = _resolve_batched_target(target_xpos, robot.num_instances)
    if robot_type == "w1":
        # Keep the W1-specific IK diagnostic batched so it remains useful when
        # checking solver and cuRobo reachability across multiple environments.
        is_success, ik_qpos = robot.compute_ik(pose=target_xpos, name=control_part)
        print(f"robot compute ik success: {is_success}, ik_qpos: {ik_qpos}")

    # This object is also exported into the cuRobo collision world below via
    # CuroboWorldCfg.rigid_objects, so the simulator and planner share geometry
    # automatically (no hand-authored collision YAML to keep in sync).
    demo_block = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="demo_block",
            shape=CubeCfg(size=demo_block_size),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=demo_block_position,
            init_rot=(0.0, 0.0, 0.0),
        )
    )

    return sim, robot, demo_block, target_xpos, control_part


def _resolve_batched_target(target: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Return a homogeneous target pose for every simulation environment."""
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    if target.shape == (4, 4):
        target = target.unsqueeze(0)
    if target.shape == (1, 4, 4):
        return target.repeat(num_envs, 1, 1)
    if target.shape == (num_envs, 4, 4):
        return target
    raise ValueError(
        "Target pose must have shape (4, 4), (1, 4, 4), or "
        f"({num_envs}, 4, 4); got {tuple(target.shape)}."
    )


def _sample_perturbed_obstacle_poses(
    nominal_poses: torch.Tensor,
    *,
    xy_perturbation: float,
    yaw_perturbation_deg: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply bounded XY translation and local-yaw noise to batched poses."""
    if nominal_poses.dim() != 3 or nominal_poses.shape[-2:] != (4, 4):
        raise ValueError(
            "nominal_poses must have shape (B, 4, 4), got "
            f"{tuple(nominal_poses.shape)}."
        )
    if xy_perturbation < 0.0:
        raise ValueError("xy_perturbation must be non-negative.")
    if yaw_perturbation_deg < 0.0:
        raise ValueError("yaw_perturbation_deg must be non-negative.")

    num_envs = nominal_poses.shape[0]
    if num_envs == 1:
        return nominal_poses.clone()

    # Sample on CPU so one generator works regardless of the simulation device.
    unit_noise = (
        2.0
        * torch.rand(
            num_envs,
            3,
            generator=generator,
            dtype=torch.float32,
        )
        - 1.0
    ).to(nominal_poses.device)
    perturbed = nominal_poses.clone()
    perturbed[:, :2, 3] += unit_noise[:, :2] * xy_perturbation

    yaw = torch.deg2rad(unit_noise[:, 2] * yaw_perturbation_deg)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    yaw_rotation = torch.zeros(
        num_envs,
        3,
        3,
        dtype=nominal_poses.dtype,
        device=nominal_poses.device,
    )
    yaw_rotation[:, 0, 0] = cos_yaw
    yaw_rotation[:, 0, 1] = -sin_yaw
    yaw_rotation[:, 1, 0] = sin_yaw
    yaw_rotation[:, 1, 1] = cos_yaw
    yaw_rotation[:, 2, 2] = 1.0
    perturbed[:, :3, :3] = torch.bmm(
        nominal_poses[:, :3, :3],
        yaw_rotation,
    )
    return perturbed


def _perturb_obstacles(
    obstacles: list[RigidObject],
    *,
    num_envs: int,
    xy_perturbation: float,
    yaw_perturbation_deg: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Perturb every obstacle and return its per-environment current poses."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    obstacle_poses: dict[str, torch.Tensor] = {}
    for obstacle in obstacles:
        nominal_poses = obstacle.get_local_pose(to_matrix=True)
        if nominal_poses.shape[0] != num_envs:
            raise ValueError(
                f"Obstacle '{obstacle.uid}' has {nominal_poses.shape[0]} poses, "
                f"expected {num_envs}."
            )
        perturbed_poses = _sample_perturbed_obstacle_poses(
            nominal_poses,
            xy_perturbation=xy_perturbation,
            yaw_perturbation_deg=yaw_perturbation_deg,
            generator=generator,
        )
        if num_envs > 1:
            obstacle.set_local_pose(perturbed_poses)
        obstacle_poses[obstacle.uid] = perturbed_poses
    return obstacle_poses


def _start_headless_recording(sim: SimulationManager, args: argparse.Namespace) -> bool:
    """Start the fixed-pose offscreen recorder for a headless demo run."""
    if not args.headless or args.disable_record:
        return False
    if not sim.start_window_record(
        save_path=args.record_save_path,
        fps=args.record_fps,
        max_memory=DEFAULT_RECORD_MAX_MEMORY,
        video_prefix="curobo_planner_headless",
        look_at=DEFAULT_RECORD_LOOK_AT,
        use_sim_time=True,
    ):
        raise RuntimeError("Failed to start cuRobo demo headless recording.")
    print("[INFO]: Headless offscreen recording enabled.")
    print(
        "[INFO]: The MP4 output path is reported by "
        "`SimulationManager.start_window_record()`."
    )
    return True


def _replay_full_dof_trajectory(
    sim: SimulationManager,
    robot: Robot,
    trajectory: torch.Tensor,
    *,
    step_repeat: int,
) -> None:
    """Replay the engine's ``(B, N, robot.dof)`` trajectory in DexSim."""
    expected_batch = robot.num_instances
    if trajectory.dim() != 3 or trajectory.shape[0] != expected_batch:
        raise ValueError(
            "Expected an environment-batched trajectory with shape "
            f"({expected_batch}, N, robot.dof), got {tuple(trajectory.shape)}."
        )
    if trajectory.shape[-1] != robot.dof:
        raise ValueError(
            "AtomicActionEngine must return full-robot DoF positions; got "
            f"{trajectory.shape[-1]} DoF for a {robot.dof}-DoF robot."
        )

    all_joint_ids = list(range(robot.dof))
    for waypoint_idx in range(trajectory.shape[1]):
        waypoint = trajectory[:, waypoint_idx]
        # Synchronize current state as well as the drive target.  Updating a
        # target alone makes the viewer show controller lag instead of the
        # collision-free cuRobo waypoint being replayed.
        robot.set_qpos(
            qpos=waypoint,
            joint_ids=all_joint_ids,
            target=False,
        )
        robot.set_qpos(
            qpos=waypoint,
            joint_ids=all_joint_ids,
            target=True,
        )
        sim.update(step=step_repeat)


def _final_tcp_errors(
    robot: Robot, target: torch.Tensor, control_part: str
) -> torch.Tensor:
    """Return the Cartesian position error for every simulator environment."""
    final_qpos = robot.get_qpos(name=control_part)
    final_pose = robot.compute_fk(
        qpos=final_qpos,
        name=control_part,
        to_matrix=True,
    )
    target = _resolve_batched_target(target, robot.num_instances).to(final_pose)
    return torch.linalg.vector_norm(
        final_pose[:, :3, 3] - target[:, :3, 3],
        dim=-1,
    )


def main() -> None:
    """Plan and replay one batched collision-aware end-effector action."""
    args = parse_args()
    if args.step_repeat < 1:
        raise ValueError("--step-repeat must be at least 1.")
    if args.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")
    if args.record_fps < 1:
        raise ValueError("--record-fps must be at least 1.")
    if args.num_envs < 1:
        raise ValueError("--num_envs must be at least 1.")
    if args.obstacle_xy_perturbation < 0.0:
        raise ValueError("--obstacle-xy-perturbation must be non-negative.")
    if args.obstacle_yaw_perturbation_deg < 0.0:
        raise ValueError("--obstacle-yaw-perturbation-deg must be non-negative.")
    sim_device = _resolve_device(args.device, args.gpu_id)
    resolved_device = torch.device(sim_device)
    effective_gpu_id = (
        resolved_device.index if resolved_device.type == "cuda" else int(args.gpu_id)
    )
    assert effective_gpu_id is not None
    _check_runtime(effective_gpu_id)
    sim: SimulationManager | None = None
    try:
        sim, robot, demo_block, target_xpos, control_part = _build_scene(
            args.headless,
            args.robot,
            sim_device,
            args.num_envs,
            args.renderer,
            args.arena_space,
            effective_gpu_id,
            visualization_cfg_from_args(args),
        )
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()

        obstacles = [demo_block]
        obstacle_poses = _perturb_obstacles(
            obstacles,
            num_envs=args.num_envs,
            xy_perturbation=args.obstacle_xy_perturbation,
            yaw_perturbation_deg=args.obstacle_yaw_perturbation_deg,
            seed=args.seed,
        )
        use_independent_worlds = args.num_envs > 1
        if use_independent_worlds:
            for name, poses in obstacle_poses.items():
                yaw_deg = torch.rad2deg(torch.atan2(poses[:, 1, 0], poses[:, 0, 0]))
                print(
                    f"{name} perturbed pose by environment: "
                    f"XY={poses[:, :2, 3].tolist()}, "
                    f"yaw_deg={yaw_deg.tolist()}"
                )

        # Delay viewer/recorder startup until the robot and every obstacle have
        # been loaded and placed at their final initial poses.
        if not args.headless:
            sim.open_window()
        _start_headless_recording(sim, args)
        if args.hold_steps:
            sim.update(step=args.hold_steps)

        motion_generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=CuroboPlannerCfg(
                    robot_uid=robot.uid,
                    world=CuroboWorldCfg(
                        rigid_objects=obstacles,
                        obstacle_representation="sphere",
                        dynamic_obstacle_names=(
                            [obstacle.uid for obstacle in obstacles]
                            if use_independent_worlds
                            else []
                        ),
                        multi_env=use_independent_worlds,
                    ),
                    max_attempts=args.max_attempts,
                    use_cuda_graph=args.cuda_graph,
                    cuda_device=f"cuda:{effective_gpu_id}",
                )
            )
        )
        # visualize arm collision
        # motion_generator.planner.visualize_collision_models("arm")
        # motion_generator.planner.visualize_obstacle_collision_model()
        engine = AtomicActionEngine(motion_generator)
        engine.register(
            MoveEndEffector(
                motion_generator,
                MoveEndEffectorCfg(
                    motion_source="motion_gen",
                    control_part=control_part,
                    plan_opts=CuroboPlanOptions(
                        dynamic_obstacle_poses=(
                            obstacle_poses if use_independent_worlds else None
                        ),
                        max_attempts=args.max_attempts,
                    ),
                    # sample_interval sets the returned trajectory's waypoint count.
                    # cuRobo's own collision-checked samples are arc-length resampled
                    # to this count; set CuroboPlannerCfg.preserve_plan_samples=True
                    # above to keep cuRobo's raw samples (count from interpolation_dt).
                    sample_interval=30,
                ),
            ),
            name="move_end_effector",
        )

        initial_qpos = robot.get_qpos(name=control_part)
        initial_xpos = robot.compute_fk(
            qpos=initial_qpos,
            name=control_part,
            to_matrix=True,
        )
        plan_start = time.perf_counter()
        success, trajectory, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target_xpos))]
        )
        planning_duration = time.perf_counter() - plan_start

        print(f"cuRobo atomic-action success by environment: {success.tolist()}")
        print(f"full-DoF trajectory shape: {tuple(trajectory.shape)}")
        print(f"[warm-up] atomic-action planning duration: {planning_duration:.3f} s")

        if not bool(success.all().item()):
            failed_env_ids = torch.nonzero(~success, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "cuRobo failed to find a collision-free trajectory for "
                f"environment(s) {failed_env_ids}."
            )

        _replay_full_dof_trajectory(
            sim,
            robot,
            trajectory,
            step_repeat=args.step_repeat,
        )
        if args.hold_steps:
            sim.update(step=args.hold_steps)
        final_errors = _final_tcp_errors(robot, target_xpos, control_part)
        print(f"final TCP position error by environment: {final_errors.tolist()} m")
        print(f"maximum final TCP position error: {final_errors.max().item():.4f} m")

        plan_start = time.perf_counter()
        success, trajectory, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=initial_xpos))]
        )
        planning_duration = time.perf_counter() - plan_start
        print(f"cuRobo return-action success by environment: {success.tolist()}")
        print(f"full-DoF trajectory shape: {tuple(trajectory.shape)}")
        print(f"[Runtime]atomic-action planning duration: {planning_duration:.3f} s")
        if not bool(success.all().item()):
            failed_env_ids = torch.nonzero(~success, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "cuRobo failed to plan the return trajectory for "
                f"environment(s) {failed_env_ids}."
            )
        _replay_full_dof_trajectory(
            sim,
            robot,
            trajectory,
            step_repeat=args.step_repeat,
        )
        if not args.headless:
            input("Press Enter to exit the cuRobo demo...")
    finally:
        # Guarantee GPU sim resources and the recorder subprocess are released
        # even when planning/replay raises mid-demo.
        if sim is not None:
            if sim.is_window_recording():
                sim.stop_window_record()
                sim.wait_window_record_saves()
            sim.destroy()
            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
