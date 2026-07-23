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
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg, Robot, RigidObject
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.sim.shapes import CubeCfg

__all__ = ["main"]


ROBOT_UID = "curobo_franka"
CONTROL_PART = "arm"
DEMO_BLOCK_DIMS = [0.18, 0.3, 0.36]
DEMO_BLOCK_POS = (0.40, 0.0, 0.18)
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


def _build_scene(headless: bool) -> tuple[SimulationManager, Robot, RigidObject]:
    """Create the one-environment Franka scene with its shared cuboid."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=headless,
            sim_device="cuda",
            num_envs=1,
            arena_space=2.0,
        )
    )

    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict(
            {
                "uid": ROBOT_UID,
                "robot_type": "panda",
                "init_qpos": [0.0, -0.5, 0.0, -2.3, 0.0, 1.8, 0.741, 0.04, 0.04],
            }
        )
    )

    if robot is None:
        raise RuntimeError(f"Failed to add robot '{ROBOT_UID}' to the cuRobo demo.")
    # This object is also exported into the cuRobo collision world below via
    # CuroboWorldCfg.rigid_objects, so the simulator and planner share geometry
    # automatically (no hand-authored collision YAML to keep in sync).
    demo_block = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="demo_block",
            shape=CubeCfg(size=DEMO_BLOCK_DIMS),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=DEMO_BLOCK_POS,
            init_rot=(0.0, 0.0, 0.0),
        )
    )

    return sim, robot, demo_block


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


def _final_tcp_error(robot: Robot, target: torch.Tensor) -> float:
    """Return the Cartesian position error of the simulator's final TCP pose."""
    final_qpos = robot.get_qpos(name=CONTROL_PART)
    final_pose = robot.compute_fk(
        qpos=final_qpos,
        name=CONTROL_PART,
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
    CuroboPlanner.prewarm(ROBOT_UID)

    sim: SimulationManager | None = None
    # try:
    sim, robot, demo_block = _build_scene(args.headless)
    if not args.headless:
        sim.open_window()
    _start_headless_recording(sim, args)
    if args.hold_steps:
        sim.update(step=args.hold_steps)

    motion_generator = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=ROBOT_UID,
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
                control_part=CONTROL_PART,
                sample_interval=80,
            ),
        ),
        name="move_end_effector",
    )

    initial_qpos = robot.get_qpos(name=CONTROL_PART)
    initial_xpos = robot.compute_fk(
        qpos=initial_qpos,
        name=CONTROL_PART,
        to_matrix=True,
    )
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
    print(f"final TCP position error: {_final_tcp_error(robot, target_xpos):.4f} m")

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
