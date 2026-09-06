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

"""Collect contact-validated parallel cube PickUp episodes and an optional video.

Run from the repository root::

    python examples/sim/motion/trajectory_generation/cube_pickup_collection.py --output /tmp/cube-experts --episodes 8 --record-video

Requires the simulator, LeRobot, python-fcl, trimesh and yourdfpy. CPU physics
is stepped at 200 Hz; a CUDA renderer is needed for the optional 20 Hz preview.
Four rows share one physical clock. The complete batch restores its initial
robot/object state between rounds. Only validated, sealed and read-back episodes
count towards the requested target. The video also shows rejected attempts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    RecoveryPolicy,
    TrackingPolicy,
)
from embodichain.lab.sim.motion.trajectory_augmentation import (
    SceneCase,
    TrajectoryGenerationJobCfg,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.execution import QposRolloutExecutor
from embodichain.lab.trajectory_generation.initial_state import (
    FixedSceneHost,
    InitialStateProfile,
)
from embodichain.lab.trajectory_generation.integrations.sim import (
    SimInitialStateAdapter,
)
from embodichain.lab.trajectory_generation.integrations.atomic import (
    export_pickup_templates,
)
from embodichain.lab.trajectory_generation.integrations.atomic_runtime import (
    PickUpRuntimeSource,
)
from embodichain.lab.trajectory_generation.integrations.contact import (
    PickUpContactProfile,
    PickUpMotionValidator,
)
from embodichain.lab.trajectory_generation.runner import (
    GenerationRunner,
    MotionLimitsProfile,
)
from embodichain.lab.trajectory_generation.sinks import LeRobotEpisodeSink
from embodichain.utils.math import look_at_to_pose
from examples.sim.motion.trajectory_generation.cube_grasp_parallel import (
    _prepare_cube_scene,
    _cube_pickup_program,
    _preview_frame,
    _YAW,
    _EYE,
    _TARGET,
    _INTRINSICS,
    _CONTROL_DT,
    _PHYSICS_DT,
)

__all__ = ["run_cube_pickup_collection", "main"]


def run_cube_pickup_collection(
    output_dir: Path,
    *,
    episodes: int = 8,
    seed: int = 13,
    cuda_device: int = 0,
    record_video: bool = False,
    runtime: bool = False,
) -> dict[str, object]:
    """Run a bounded four-row PickUp collection with real contact acceptance.

    Args:
        output_dir: New or empty local collection directory.
        episodes: Number of committed episodes requested, between 1 and 64.
        seed: Local augmentation random seed.
        cuda_device: GPU used by the simulator renderer.
        record_video: Stream synchronized four-panel observations to preview.mp4.
        runtime: Execute candidates through observed Atomic ExecutionRunner
            tracking and effect verification before accepting expert data.

    Returns:
        The persisted generation audit with committed count and target outcome.
    """
    if type(episodes) is not int or not 1 <= episodes <= 64:
        raise ValueError("episodes must be an integer between 1 and 64")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output_dir must be new or empty")
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "source": {"kind": "atomic", "source_id": "atomic_pickup"},
            "augmentation": {
                "seed": seed,
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": "joint_residual",
                        "joint_offset_scale": 0.025,
                    }
                },
            },
            "collection": {
                "target_committed_episodes": episodes,
                "max_proposals": max(32, episodes * 8),
                "max_rollout_attempts": max(16, episodes * 4),
                "max_wall_time_s": 1200.0,
            },
            "validation": {"path_length_ratio_max": 2.0},
        }
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            num_envs=4,
            physics_dt=_PHYSICS_DT,
            arena_space=4.0,
            gpu_id=cuda_device,
        )
    )
    generator = host = sink = writer = None
    try:
        robot, cube, camera, initial, grasps, hand_open, hand_close = (
            _prepare_cube_scene(sim, arm_stiffness=2e5, hand_open_margin=0.001)
        )
        generator, engine, invocations = _cube_pickup_program(
            robot, cube, grasps, hand_open, hand_close
        )
        compiled = engine.compile(
            invocations, engine.initial_context(control_dt=_CONTROL_DT)
        )
        templates = export_pickup_templates(compiled, robot, control_dt=_CONTROL_DT)
        initial_objects = {
            uid: sim.get_rigid_object(uid).get_local_pose(to_matrix=True).clone()
            for uid in sim.get_rigid_object_uid_list()
        }
        zero = torch.zeros_like(initial)
        prepares = 0

        def prepare() -> None:
            nonlocal prepares
            prepares += 1
            robot.set_qpos(initial, target=False)
            robot.set_qvel(zero, target=False)
            robot.set_qpos(initial, target=True)
            robot.set_qvel(zero, target=True)
            robot.set_qf(zero)
            for uid, pose in initial_objects.items():
                obj = sim.get_rigid_object(uid)
                obj.set_local_pose(pose)
                obj.clear_dynamics()

        def signature() -> str:
            return json.dumps(
                {
                    "robot": robot.cfg.to_dict(),
                    "scene": {
                        uid: sim.get_rigid_object(uid).cfg.to_dict()
                        for uid in initial_objects
                    },
                    "physics": sim.sim_config.physics_config.to_dict(),
                },
                sort_keys=True,
                default=str,
            )

        def verify(cases) -> ValidationResult:
            passed = torch.allclose(
                robot.get_qpos(), initial, atol=1e-6, rtol=0
            ) and all(
                torch.allclose(
                    sim.get_rigid_object(uid).get_local_pose(to_matrix=True),
                    pose,
                    atol=1e-6,
                    rtol=0,
                )
                for uid, pose in initial_objects.items()
            )
            return ValidationResult(
                (
                    ValidationCheck(
                        "task_initial_state",
                        "passed" if passed else "failed",
                        "Identical prepared cube, table and full robot state",
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
                physics_dt=_PHYSICS_DT,
                settling_steps=0,
            ),
        )
        # The PGI fingertips extend past the 0.15 m TCP. Their contact entry
        # region includes the cube half-height and that physical overhang.
        planner = PickUpMotionValidator(
            sim, robot, profile=PickUpContactProfile(approach_contact_distance=0.06)
        )
        frames = 0
        preview_round, round_frame = 0, 0
        trails = [[] for _ in range(4)]
        view = torch.linalg.inv(look_at_to_pose(_EYE, _TARGET)).squeeze(0).numpy()
        if record_video:
            from PIL import ImageFont

            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 17)
            except OSError:
                font = ImageFont.load_default()

        def observe() -> dict[str, torch.Tensor]:
            nonlocal frames, trails, preview_round, round_frame
            if record_video:
                if preview_round != prepares:
                    preview_round, round_frame = prepares, 0
                    trails = [[] for _ in range(4)]
                index = round_frame
                observations = planner.observations()
                tcp = observations["tcp_pose"].cpu().numpy()
                for row in range(4):
                    point = view @ np.r_[tcp[row, :3, 3], 1.0]
                    if point[2] > 0:
                        fx, fy, cx, cy = _INTRINSICS
                        trails[row].append(
                            (
                                fx * point[0] / point[2] + cx,
                                fy * point[1] / point[2] + cy,
                            )
                        )
                phase = next(
                    p.phase_id
                    for p in templates[0].phases
                    if p.start_index <= index < p.stop_index
                )
                camera.update()
                rgb = camera.get_data()["color"][..., :3].cpu().numpy().copy()
                frame = _preview_frame(
                    rgb,
                    trails,
                    index * _CONTROL_DT,
                    f"round {prepares}: {phase}",
                    (
                        observations["object_pose"][:, 2, 3]
                        - initial_objects[cube.uid][:, 2, 3]
                    )
                    .cpu()
                    .numpy(),
                    font,
                    path_labels=("augmented path",) * 4,
                )
                writer.append_data(frame)
                frames += 1
                round_frame += 1
            return {
                "joint_positions": robot.get_qpos(),
                "joint_velocities": robot.get_qvel(),
            }

        arm_ids = tuple(robot.get_joint_ids("arm"))

        def task_validator(
            candidate, observations, actions, timestamps
        ) -> ValidationResult:
            measured = observations["joint_positions"]
            expected = candidate.positions[0, : len(measured)].to(measured)
            endpoint = float(
                (measured[-1, arm_ids] - expected[-1, arm_ids]).abs().max()
            )
            tracking = float((measured[:, arm_ids] - expected[:, arm_ids]).abs().max())
            # Physical grasp and contact checks are independent mandatory gates
            # owned by planner; this check proves the commanded arm path executed.
            return ValidationResult(
                (
                    ValidationCheck(
                        "task_success",
                        "passed" if endpoint <= 0.05 and tracking <= 0.08 else "failed",
                        "Measured arm endpoint and tracking",
                        {
                            "endpoint_error_rad": endpoint,
                            "tracking_error_rad": tracking,
                        },
                    ),
                )
            )

        budget = 1024 * 1024
        # Ground the runtime request in the selected reference's full-URDF TCP.
        # The analytic IK target can differ from that geometry (the UR5 asset
        # includes a wrist offset); pending effects must describe the qualified
        # joint branch that will actually execute.
        qualified_grasps = robot.compute_fk(
            qpos=torch.stack(
                [t.positions[t.phases[2].start_index - 1, arm_ids] for t in templates]
            ),
            name="arm",
            to_matrix=True,
        )

        def runtime_invocation(binding):
            reference = invocations[1]
            invocation = engine.make_invocation(
                "pick_up",
                replace(reference.goal, grasp_xpos=qualified_grasps),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=reference.motion_policy,
                skill_options=reference.skill_options,
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=0.08,
                    terminal_max_abs_error=0.05,
                ),
                recovery_policy=RecoveryPolicy(max_replans=1, max_action_retries=0),
                invocation_id=f"pickup_batch_{prepares}",
            )
            # The closing hand presses into the cube. Its physical contract is
            # bilateral contact and held-object stability, while arm positions
            # retain the ordinary runtime feedback thresholds.
            return replace(
                invocation,
                binding=ActionBinding(
                    invocation.binding.owner_id,
                    tuple(
                        (
                            replace(endpoint, tracking_channels={})
                            if endpoint.endpoint_id == "grasp"
                            else endpoint
                        )
                        for endpoint in invocation.binding.endpoints
                    ),
                ),
            )

        executor = QposRolloutExecutor(
            host,
            control_dt=_CONTROL_DT,
            observe=observe,
            validator=task_validator,
            max_episode_bytes=budget,
            contact_validator=planner,
            runtime_source=(
                PickUpRuntimeSource(
                    engine,
                    invocation_factory=runtime_invocation,
                    templates=templates,
                )
                if runtime
                else None
            ),
        )
        sink = LeRobotEpisodeSink(output_dir, fps=20, max_episode_bytes=budget)
        if record_video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(output_dir / "preview.mp4"),
                format="FFMPEG",
                fps=20,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart"],
            )
        cases = tuple(
            SceneCase(
                f"cube_grasp_row_{row}",
                f"fixed_yaw_{round(np.degrees(_YAW[row]))}",
                "cube_table_v1",
                "cube_pickup",
                "ur5_dh_pgi",
            )
            for row in range(4)
        )
        limits = MotionLimitsProfile(
            torch.full((robot.dof,), 1.5), torch.full((robot.dof,), 15.0)
        )
        runner = GenerationRunner(
            cfg, host, planner, executor, sink, motion_limits=limits
        )
        report = dict(runner.run(cases, templates))
        (output_dir / "pickup_report.json").write_text(
            json.dumps(
                {
                    "committed": report["counts"]["committed"],
                    "target_reached": report["target_reached"],
                    "prepared_batches": prepares,
                    "video_frames": frames,
                    "source": (
                        "Atomic ExecutionRunner with fresh PickUp candidate plans"
                        if runtime
                        else "offline AtomicActionEngine MoveEndEffector + PickUp export"
                    ),
                    "contact_profile": planner.profile.to_dict(),
                },
                indent=2,
            )
        )
        return report
    finally:
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()
        if host is not None:
            host.close()
        if generator is not None:
            generator.planner.close()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run the collection CLI and exit unsuccessfully if its target was not met."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Use observed atomic candidate execution and effect verification",
    )
    args = parser.parse_args()
    report = None
    try:
        report = run_cube_pickup_collection(
            args.output,
            episodes=args.episodes,
            seed=args.seed,
            cuda_device=args.cuda_device,
            record_video=args.record_video,
            runtime=args.runtime,
        )
    except Exception as error:
        diagnostic = traceback.TracebackException.from_exception(error)
        # Release completed rollout frames before interpreter shutdown; their
        # locals can otherwise retain native scene objects through a traceback.
        traceback.clear_frames(error.__traceback__)
        print("".join(diagnostic.format()), file=sys.stderr, end="")
    if report is None:
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "target_reached": report["target_reached"],
                "counts": dict(report["counts"]),
            }
        )
    )
    if not report["target_reached"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
