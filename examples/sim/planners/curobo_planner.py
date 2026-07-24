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

The demo creates a single Franka Panda and a static cuboid that is represented
both in DexSim and in the cuRobo collision world.  It then executes a
``MoveEndEffector`` action through :class:`AtomicActionEngine`, replays the
returned full-robot-DoF trajectory, and reports the final TCP error.

Run from the repository root::

    python examples/sim/planners/curobo_planner.py --headless

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
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg, Robot, RigidObject
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
import numpy as np
from scipy.spatial.transform import Rotation as R
from embodichain.lab.sim.robots import FrankaPandaCfg, URRobotCfg, DexforceW1Cfg
from embodichain.lab.sim.shapes import CubeCfg

__all__ = ["main"]


DEFAULT_RECORD_FPS = 20
DEFAULT_RECORD_MAX_MEMORY = 2048
DEFAULT_MAX_ATTEMPTS = 2
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the simulation viewer.",
    )
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
    return parser.parse_args()


def _check_runtime() -> None:
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


def _build_scene(
    headless: bool, robot_type: str = "franka"
) -> tuple[SimulationManager, Robot, RigidObject, torch.Tensor, str]:
    """Create the one-environment Franka scene with its shared cuboid."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=headless,
            sim_device="cuda",
            num_envs=1,
            arena_space=2.0,
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
        hand_attach_xpos[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
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
                    [-1.2628e-02, -4.3484e-03, -9.9991e-01, 4.0e-01],
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

        # robot compute ik success in example
        is_success, ik_qpos = robot.compute_ik(pose=target_xpos, name=control_part)
        print(f"robot compute ik success: {is_success}, ik_qpos: {ik_qpos}")

        # sim.open_window()
        # # sim.update(50)
        # current_qpos = robot.get_qpos(name=control_part)
        # current_xpos = robot.compute_fk(name=control_part, qpos=current_qpos, to_matrix=True)
        # print(f"Current {control_part} TCP pose:\n{current_xpos}")
        # import ipdb; ipdb.set_trace()

    else:
        raise ValueError(f"Unknown robot type '{robot_type}' for cuRobo demo.")

    if robot is None:
        raise RuntimeError(f"Failed to add robot '{robot.uid}' to the cuRobo demo.")
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
    if trajectory.dim() != 3 or trajectory.shape[0] != 1:
        raise ValueError(
            "This single-environment demo expected a (1, N, robot.dof) "
            f"trajectory, got {tuple(trajectory.shape)}."
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


def _final_tcp_error(robot: Robot, target: torch.Tensor, control_part: str) -> float:
    """Return the Cartesian position error of the simulator's final TCP pose."""
    final_qpos = robot.get_qpos(name=control_part)
    final_pose = robot.compute_fk(
        qpos=final_qpos,
        name=control_part,
        to_matrix=True,
    )
    # Accept either a single (4, 4) pose or a batched (B, 4, 4) target.
    target_pos = target[0, :3, 3] if target.dim() == 3 else target[:3, 3]
    return float(torch.linalg.vector_norm(final_pose[0, :3, 3] - target_pos))


def main() -> None:
    """Plan and replay one collision-aware atomic end-effector action."""
    args = parse_args()
    if args.step_repeat < 1:
        raise ValueError("--step-repeat must be at least 1.")
    if args.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")
    if args.record_fps < 1:
        raise ValueError("--record-fps must be at least 1.")
    _check_runtime()
    # Spawn the cuRobo worker now so its ~5s Python+torch startup overlaps with
    # the simulation build below instead of blocking the first plan.

    sim: SimulationManager | None = None
    # try:
    sim, robot, demo_block, target_xpos, control_part = _build_scene(
        args.headless, args.robot
    )
    CuroboPlanner.prewarm(robot.uid)
    if not args.headless:
        sim.open_window()
    _start_headless_recording(sim, args)
    if args.hold_steps:
        sim.update(step=args.hold_steps)

    motion_generator = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=robot.uid,
                world=CuroboWorldCfg(rigid_objects=[demo_block]),
                max_attempts=args.max_attempts,
            )
        )
    )
    engine = AtomicActionEngine(motion_generator)
    engine.register(
        MoveEndEffector(
            motion_generator,
            MoveEndEffectorCfg(
                motion_source="motion_gen",
                planner_type="curobo",
                control_part=control_part,
                sample_interval=80,
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

    print(f"cuRobo atomic-action success: {bool(success.item())}")
    print(f"full-DoF trajectory shape: {tuple(trajectory.shape)}")
    print(f"[warm-up] atomic-action planning duration: {planning_duration:.3f} s")

    if not bool(success.item()):
        raise RuntimeError("cuRobo failed to find a collision-free trajectory.")

    _replay_full_dof_trajectory(
        sim,
        robot,
        trajectory,
        step_repeat=args.step_repeat,
    )
    if args.hold_steps:
        sim.update(step=args.hold_steps)
    print(
        f"final TCP position error: {_final_tcp_error(robot, target_xpos, control_part):.4f} m"
    )

    plan_start = time.perf_counter()
    success, trajectory, _ = engine.run(
        [("move_end_effector", EndEffectorPoseTarget(xpos=initial_xpos))]
    )
    planning_duration = time.perf_counter() - plan_start
    print(f"cuRobo atomic-action success: {bool(success.item())}")
    print(f"full-DoF trajectory shape: {tuple(trajectory.shape)}")
    print(f"[Runtime]atomic-action planning duration: {planning_duration:.3f} s")
    _replay_full_dof_trajectory(
        sim,
        robot,
        trajectory,
        step_repeat=args.step_repeat,
    )
    if sim.is_window_recording():
        sim.stop_window_record()
        sim.wait_window_record_saves()
    sim.destroy()
    SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
