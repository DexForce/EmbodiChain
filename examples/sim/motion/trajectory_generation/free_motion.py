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

"""Collect one measured UR5 free-motion episode with real physics and cuRobo.

Run from the repository root with CUDA, cuRobo and LeRobot installed::

    python examples/sim/motion/trajectory_generation/free_motion.py --output /tmp/ur5-free-motion

Save a five-second offscreen camera preview::

    python examples/sim/motion/trajectory_generation/free_motion.py --output /tmp/ur5-video --record-video --duration 5 --joint-displacement 0.4

The pure-arm UR5 is fixed 0.5 m above the ground, without a gripper or held
object. A registered cuboid models the simulator's implicit ground collider.
Physics uses CPU; cuRobo and the optional offscreen camera use CUDA. Video is
streamed to preview.mp4 at 20 fps, including the initial and terminal observation.
No desktop window is required. This is a free-motion benchmark, not a PickUp task.
All expert gates remain enabled, and a rejected episode stays out of LeRobot.
Gravity remains enabled. ``--robot panda`` is a physical negative example:
its moving gripper/mimic coordinates violate the current locked-hand model,
so collection rejects the evidence and exits with status 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.motion.motion_generator import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.motion.planners import CuroboPlannerCfg, CuroboWorldCfg
from embodichain.lab.sim.motion.expansion import (
    CandidateTrajectoryBatch,
    SceneCase,
    TrajectoryGenerationJobCfg,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.robots import FrankaPandaCfg, URRobotCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.visualization import VisualizationCfg
from embodichain.lab.trajectory_generation.execution import QposRolloutExecutor
from embodichain.lab.trajectory_generation.initial_state import (
    FixedSceneHost,
    InitialStateProfile,
)
from embodichain.lab.trajectory_generation.integrations.planning import (
    EnvRowMotionPlanner,
)
from embodichain.lab.trajectory_generation.integrations.sim import (
    SimInitialStateAdapter,
)
from embodichain.lab.trajectory_generation.runner import (
    GenerationRunner,
    MotionLimitsProfile,
)
from embodichain.lab.trajectory_generation.sinks import LeRobotEpisodeSink

__all__ = ["run_free_motion", "main"]


def run_free_motion(
    output_dir: Path,
    *,
    robot_type: str = "ur5",
    cuda_device: int = 0,
    record_video: bool = False,
    duration_s: float = 1.0,
    joint_displacement: float = 0.08,
) -> Mapping[str, object]:
    """Run one bounded collection and return the persisted final audit.

    Args:
        output_dir: Dedicated local LeRobot collection directory.
        robot_type: ``ur5`` for the pure-arm benchmark; ``panda`` for the
            rejection example with a moving articulated gripper.
        cuda_device: GPU used by cuRobo and the headless simulator engine.
        record_video: Stream actual RGB observations to ``preview.mp4`` in the
            output directory. The numeric LeRobot observations retain their schema.
        duration_s: Motion duration in seconds, a multiple of 0.05, up to 30.
        joint_displacement: Positive first-arm-joint displacement in radians.

    Returns:
        The Runner report, including committed counts and validation diagnostics.
    """
    if robot_type not in {"ur5", "panda"}:
        raise ValueError("robot_type must be ur5 or panda")
    physics_dt, control_dt = 0.005, 0.05
    if not math.isfinite(duration_s) or not control_dt <= duration_s <= 30:
        raise ValueError("duration_s must be between 0.05 and 30 seconds")
    steps = round(duration_s / control_dt)
    if not math.isclose(steps * control_dt, duration_s, rel_tol=0, abs_tol=1e-8):
        raise ValueError("duration_s must be a multiple of the 0.05 s control period")
    if not math.isfinite(joint_displacement) or not 0 < joint_displacement <= 0.5:
        raise ValueError("joint_displacement must be in (0, 0.5] radians")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            num_envs=1,
            physics_dt=physics_dt,
            gpu_id=cuda_device,
            visualization=VisualizationCfg(),
        )
    )
    host = sink = generator = video_writer = None
    try:
        cfg_type = URRobotCfg if robot_type == "ur5" else FrankaPandaCfg
        robot_values = {
            "uid": f"generation_{robot_type}",
            "robot_type": robot_type,
            "init_pos": [0.0, 0.0, 0.5],
            "enable_gravity": True,
        }
        if robot_type == "ur5":
            robot_values["init_qpos"] = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        robot = sim.add_robot(cfg=cfg_type.from_dict(robot_values))
        # SimulationManager creates an unregistered 1000 x 1000 x 100 m ground
        # cuboid centered at z=-50.001. This explicit matching collider gives
        # the trusted profile and cuRobo an owned representation of that floor.
        floor = sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="ground_proxy",
                shape=CubeCfg(size=(1000.0, 1000.0, 100.0)),
                attrs=RigidBodyAttributesCfg(),
                body_type="kinematic",
                init_pos=(0.0, 0.0, -50.001),
            )
        )
        camera = (
            sim.add_sensor(
                CameraCfg(
                    uid="preview_camera",
                    visualization_role="record",
                    width=640,
                    height=480,
                    intrinsics=(520.0, 520.0, 320.0, 240.0),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(0.6, -1.3, 1.35),
                        target=(-0.2, 0.0, 1.0),
                    ),
                )
            )
            if record_video
            else None
        )
        initial = torch.tensor([robot.cfg.init_qpos], device=robot.device)
        zeros = torch.zeros_like(initial)
        arm_ids = tuple(robot.get_joint_ids("arm"))

        def prepare() -> None:
            robot.set_gravity(True)
            robot.set_qpos(initial, target=False)
            robot.set_qvel(zeros, target=False)
            robot.set_qpos(initial, target=True)
            robot.set_qvel(zeros, target=True)
            robot.set_qf(zeros)

        def signature() -> str:
            return json.dumps(
                {
                    "robot": robot.cfg.to_dict(),
                    "physics": sim.sim_config.physics_config.to_dict(),
                    "floor_size": (1000.0, 1000.0, 100.0),
                    "floor_center": (0.0, 0.0, -50.001),
                    "camera": None if camera is None else camera.cfg.to_dict(),
                    "render": sim.sim_config.render_cfg.to_dict(),
                },
                sort_keys=True,
                default=str,
            )

        def verify(cases: tuple[SceneCase, ...]) -> ValidationResult:
            passed = (
                len(cases) == 1
                and torch.allclose(robot.get_qpos(), initial, atol=1e-6, rtol=0)
                and torch.allclose(robot.get_qvel(), zeros, atol=1e-6, rtol=0)
            )
            return ValidationResult(
                (
                    ValidationCheck(
                        "task_initial_state",
                        "passed" if passed else "failed",
                        "Neutral full-joint position and zero initial joint velocity.",
                    ),
                )
            )

        host = FixedSceneHost(
            SimInitialStateAdapter(sim, robot),
            InitialStateProfile(
                profile_id="fixed_scene_initial_state",
                prepare=prepare,
                signature=signature,
                verify=verify,
                physics_dt=physics_dt,
                settling_steps=0,
            ),
        )
        generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=CuroboPlannerCfg(
                    robot_uid=robot.uid,
                    cuda_device=cuda_device,
                    use_cuda_graph=False,
                    world=CuroboWorldCfg(
                        rigid_objects={floor.uid: floor},
                        dynamic_obstacle_names=[floor.uid],
                        obstacle_representation="cuboid",
                    ),
                )
            )
        )
        planner = EnvRowMotionPlanner(
            generator,
            control_part="arm",
            held_object_ids=(),
            max_joint_step=0.01,
            max_validation_samples=max(128, steps + 1),
        )

        def task_validator(
            candidate: CandidateTrajectoryBatch,
            observations: Mapping[str, torch.Tensor],
            actions: torch.Tensor,
            timestamps: torch.Tensor,
        ) -> ValidationResult:
            measured = observations["joint_positions"]
            expected = candidate.positions[0, : int(candidate.valid_length[0])].to(
                measured
            )
            endpoint_error = float((measured[-1] - expected[-1]).abs().max())
            tracking_error = float((measured - expected).abs().max())
            movement = float(measured[-1, arm_ids[0]] - measured[0, arm_ids[0]])
            gripper_ids = [index for index in range(robot.dof) if index not in arm_ids]
            gripper_drift = (
                float((measured[:, gripper_ids] - expected[:, gripper_ids]).abs().max())
                if gripper_ids
                else 0.0
            )
            passed = (
                endpoint_error <= 0.01
                and tracking_error <= 0.02
                and movement >= 0.875 * joint_displacement
            )
            return ValidationResult(
                (
                    ValidationCheck(
                        "task_success",
                        "passed" if passed else "failed",
                        "Actual endpoint, measured tracking, and observed arm displacement.",
                        {
                            "endpoint_error": endpoint_error,
                            "tracking_error": tracking_error,
                            "measured_displacement": movement,
                            "max_gripper_drift": gripper_drift,
                        },
                    ),
                )
            )

        def observe() -> Mapping[str, torch.Tensor]:
            if camera is not None:
                camera.update()
                rgb = (
                    camera.get_data()["color"][0, ..., :3]
                    .cpu()
                    .contiguous()
                    .numpy()
                    .copy()
                )
                video_writer.append_data(rgb)
            return {
                "joint_positions": robot.get_qpos(),
                "joint_velocities": robot.get_qvel(),
            }

        executor = QposRolloutExecutor(
            host,
            control_dt=control_dt,
            observe=observe,
            validator=task_validator,
            max_episode_bytes=128 * 1024,
        )
        sink = LeRobotEpisodeSink(
            Path(output_dir), fps=20, max_episode_bytes=128 * 1024
        )
        if record_video:
            import imageio.v2 as imageio

            video_writer = imageio.get_writer(
                str(sink.root / "preview.mp4"),
                format="FFMPEG",
                fps=20,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart"],
            )
        limits = MotionLimitsProfile(
            torch.tensor(
                [1.0 if index in arm_ids else 0.1 for index in range(robot.dof)]
            ),
            torch.tensor(
                [5.0 if index in arm_ids else 1.0 for index in range(robot.dof)]
            ),
        )
        runner = GenerationRunner(
            TrajectoryGenerationJobCfg.from_mapping(
                {
                    "collection": {
                        "target_committed_episodes": 1,
                        "max_proposals": 1,
                        "max_rollout_attempts": 1,
                        "max_wall_time_s": 180.0,
                    }
                }
            ),
            host,
            planner,
            executor,
            sink,
            motion_limits=limits,
        )
        time = torch.linspace(0.0, 1.0, steps + 1)
        smooth = 10 * time**3 - 15 * time**4 + 6 * time**5
        positions = initial.repeat(steps + 1, 1)
        positions[:, arm_ids[0]] += joint_displacement * smooth
        dt = torch.full((steps + 1,), control_dt, dtype=torch.float64)
        dt[0] = 0
        template = TrajectoryTemplate(
            "handwritten_qpos",
            f"v1-duration={duration_s!r}-displacement={joint_displacement!r}",
            "reference_0",
            tuple(robot.joint_names),
            positions,
            dt,
            (TrajectoryPhase("free", 0, steps + 1),),
            validator_id="task_success",
            controlled_joint_indices=arm_ids,
        )
        case = SceneCase(
            f"{robot_type}_free_motion",
            "reference_start",
            f"{robot_type}_elevated_ground_gravity_enabled",
            "free_arm_motion",
            robot.uid,
        )
        return runner.run((case,), (template,))
    finally:
        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            if sink is not None:
                sink.close()
            if host is not None:
                host.close()
            if generator is not None:
                generator.planner.close()
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Parse the output location and run the measured free-motion benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=("ur5", "panda"), default="ur5")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save preview.mp4 using an offscreen RGB camera",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Motion duration in seconds (multiple of 0.05, max 30)",
    )
    parser.add_argument(
        "--joint-displacement",
        type=float,
        default=0.08,
        help="First arm joint displacement in radians (0 < value <= 0.5)",
    )
    args = parser.parse_args()
    report = run_free_motion(
        args.output,
        robot_type=args.robot,
        cuda_device=args.cuda_device,
        record_video=args.record_video,
        duration_s=args.duration,
        joint_displacement=args.joint_displacement,
    )
    video_path = args.output / "preview.mp4"
    print(
        json.dumps(
            {
                "counts": dict(report["counts"]),
                "target_reached": report["target_reached"],
                "report": str(args.output / "generation_report.json"),
                "video": str(video_path) if video_path.is_file() else None,
            }
        )
    )
    if not report["target_reached"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
