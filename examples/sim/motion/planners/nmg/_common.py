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

"""Shared download, planning and fixed-cadence playback for NMG examples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

from embodichain.cli.sim import add_sim_args_to_parser
from embodichain.data.assets import download_neural_planner_checkpoint

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.planners import PlanResult
    from embodichain.lab.sim.objects import Robot
    from examples.sim.motion.planners.nmg._visuals import TrajectoryMarkers

# Pin immutable model bytes; a moving main branch must not change a demo.
_REVISION = "ddca0e456e097ba42c05e1269febf73fa2835541"
_SHA256 = {
    "franka": "2361954fda7b6f0d21445ebfb5099d31bdd67199bdadcecba71e0835f2157ad5",
    "piper": "223a02d7f1c3db1ea96352404f623ebfde7e1b00f3bc3d7a186a634dfea5ecf0",
}
_FPS = 25
_PHASE_HOLD_SECONDS = 0.5


def _parse_args(robot_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"NMG v1 {robot_name} example")
    add_sim_args_to_parser(parser)
    parser.set_defaults(arena_space=2.0)
    parser.add_argument(
        "--scenario",
        choices=["free-motion", "closed-loop-perturbation"],
        default="free-motion",
        help="Replay a timed free-space plan or drive the arm with live NMG feedback.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Use a local copy of the published v1 ONNX file.",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use only the Hugging Face cache."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify weights without starting simulation.",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Plan and report geometric arrival errors only.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the viewer open after playback; Ctrl+C exits.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the demonstration report as JSON."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Record simulated playback to a 720p MP4, including in headless mode.",
    )
    return parser.parse_args()


def _model_path(robot_name: str, args: argparse.Namespace) -> Path:
    path = args.model_path or Path(
        download_neural_planner_checkpoint(
            filename=f"v1/{robot_name}.onnx",
            revision=_REVISION,
            local_files_only=args.offline,
        )
    )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != _SHA256[robot_name]:
        raise ValueError(
            f"The model at {path} is not the pinned NMG v1 {robot_name} release (SHA256 mismatch)."
        )
    print(f"NMG v1 {robot_name}: {path} (SHA256 {digest})")
    return path


def _play_free_motion(
    sim: SimulationManager,
    robot: Robot,
    result: PlanResult,
    waypoint_indices: Sequence[int],
    joint_ids: Sequence[int],
    control_dt: float,
    markers: TrajectoryMarkers | None,
) -> dict[str, object]:
    """Replay rest-to-rest phases and sample the measured TCP for display."""
    from embodichain.compute.trajectory import retime_to_control_grid
    from embodichain.lab.sim.motion.execution import (
        JointTrajectoryPlaybackCfg,
        play_joint_trajectory,
    )

    previous = 0
    playback_seconds = 0.0
    tracking_errors = []
    for slot, end in enumerate(waypoint_indices):
        phase_dt = result.dt[:, previous : end + 1].clone()
        phase_dt[:, 0] = 0
        phase_positions = result.positions[:, previous : end + 1]
        if markers is not None:
            # Regrid once so the source callback samples actual control steps.
            phase_positions, _, phase_dt, _ = retime_to_control_grid(
                phase_positions, phase_dt, control_dt
            )
            markers.show_targets(slot, active=slot)
        executed = play_joint_trajectory(
            sim,
            robot,
            positions=phase_positions,
            dt=phase_dt,
            joint_ids=joint_ids,
            cfg=JointTrajectoryPlaybackCfg(
                control_dt=control_dt,
                joint_command_mode="position_velocity",
                preserve_source_grid=markers is not None,
            ),
            on_step=markers.observe if markers is not None else None,
        )
        if markers is not None:
            markers.show_targets(slot + 1)
        # Advance holds at the physics cadence so recording covers the settle.
        for _ in range(round(_PHASE_HOLD_SECONDS / sim.sim_config.physics_dt)):
            sim.update(step=1)
        measured = robot.get_qpos(name="arm")
        if markers is not None:
            markers.record_tip(measured)
        tracking_errors.append(float((measured - result.positions[:, end]).abs().max()))
        playback_seconds += float(executed.dt.sum()) + _PHASE_HOLD_SECONDS
        previous = end
    return {
        "simulation_playback": "completed",
        "playback_duration_s": playback_seconds,
        "measured_tracking_error_rad": tracking_errors,
    }


def run(
    robot_name: str,
    *,
    offsets: list[list[float]],
    modalities: list[str],
    look_at: tuple[tuple[float, float, float], ...],
) -> None:
    """Download the release and run one free-space robot demonstration."""
    args = _parse_args(robot_name)
    if args.video and not args.download_only:
        if args.video.suffix.lower() != ".mp4":
            raise ValueError("--video must end in .mp4.")
        args.video = args.video.with_suffix(".mp4")
        if args.no_playback:
            raise ValueError("--video requires playback; omit --no-playback.")
        if args.renderer == "no-render":
            raise ValueError("--video requires a renderer; omit --renderer no-render.")
        if args.renderer == "auto":
            args.renderer = "fast-rt"
    elif args.headless and not args.web and args.renderer == "auto":
        args.renderer = "no-render"
    model_path = _model_path(robot_name, args)
    if args.download_only:
        return
    if args.num_envs != 1:
        raise ValueError("These demonstrations use one robot; pass --num-envs 1.")

    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend
    from embodichain.lab.visualization import visualization_cfg_from_args
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenCfg,
        MotionGenOptions,
        MotionGenerator,
    )
    from embodichain.lab.sim.motion.planners import NeuralPlannerCfg, PlanState
    from embodichain.lab.sim.motion.planners.neural_planner import NeuralPlanOptions
    from embodichain.utils.math import quat_error_magnitude, quat_from_matrix
    from examples.sim.motion.planners.nmg._closed_loop import (
        FINAL_HOLD_SECONDS,
        MAX_STEPS_PER_TARGET,
        play_closed_loop,
    )
    from examples.sim.motion.planners.nmg._robots import make_robot_cfg
    from examples.sim.motion.planners.nmg._visuals import TrajectoryMarkers

    # Avoid machine-sized Torch/ORT thread pools for the small actor batch.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(
        args.device or (f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{args.gpu_id}")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            width=1280,
            height=720,
            device=str(device),
            num_envs=1,
            arena_space=args.arena_space,
            gpu_id=device.index if device.type == "cuda" else args.gpu_id,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_cfg=physics_cfg_for_backend(args.physics),
            visualization=visualization_cfg_from_args(args),
        )
    )
    try:
        cfg = make_robot_cfg(robot_name)
        robot = sim.add_robot(cfg=cfg)
        sim.prepare()
        if not args.headless or args.web:
            sim.open_viewer()
        joints = robot.get_joint_ids("arm")
        joint_names = list(cfg.control_parts["arm"])
        expected_names = (
            [f"fr3_joint{i}" for i in range(1, 8)]
            if robot_name == "franka"
            else [f"joint{i}" for i in range(1, 7)]
        )
        if joint_names != expected_names:
            raise ValueError("Robot joint order differs from the published NMG policy.")
        start = torch.tensor(
            cfg.init_qpos[: len(joints)], device=robot.device, dtype=torch.float32
        ).unsqueeze(0)
        robot.set_qpos(start, joint_ids=joints, target=False)
        robot.set_qpos(start, joint_ids=joints)

        # FK constructs reachable targets. These are authored demonstration
        # cases, not training data, a benchmark or a success-rate estimate.
        goal_qpos = (
            start[:, None]
            + torch.tensor(offsets, dtype=start.dtype, device=start.device)[None]
        )
        limits = robot.get_qpos_limits(name="arm")
        if bool(
            (
                (goal_qpos < limits[:, None, :, 0])
                | (goal_qpos > limits[:, None, :, 1])
            ).any()
        ):
            raise ValueError("An authored demo target violates joint limits.")
        goal_poses = torch.stack(
            [
                robot.compute_fk(qpos=goal_qpos[:, i], name="arm", to_matrix=True)
                for i in range(len(modalities))
            ],
            dim=1,
        )
        targets = [
            (
                PlanState.from_qpos(goal_qpos[:, i])
                if kind == "joint"
                else PlanState.from_xpos(goal_poses[:, i])
            )
            for i, kind in enumerate(modalities)
        ]
        control_dt = 4 * sim.sim_config.physics_dt
        generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=NeuralPlannerCfg(
                    robot_uid=robot.uid,
                    sim_instance_id=sim.instance_id,
                    control_part="arm",
                    onnx_model_path=str(model_path),
                    num_arm_joints=len(joints),
                    num_waypoints=12,
                    max_steps=30 * len(modalities),
                    action_scale=0.2,
                    dt=control_dt,
                    onnx_providers=(
                        ["CPUExecutionProvider"]
                        if device.type == "cpu"
                        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    ),
                    onnx_cuda_use_tf32=False,
                    preserve_waypoint_arrivals=True,
                    constraints={"velocity": 0.6, "acceleration": 1.5},
                )
            )
        )
        result = generator.generate(
            target_states=targets,
            options=MotionGenOptions(
                plan_opts=NeuralPlanOptions(
                    control_part="arm",
                    start_qpos=start,
                    waypoint_modalities=modalities,
                )
            ),
        )
        report = {
            "robot": robot_name,
            "scenario": args.scenario,
            "revision": _REVISION,
            "sha256": _SHA256[robot_name],
            "modalities": modalities,
            "geometric_success": result.success.cpu().tolist(),
            "simulation_playback": "not_run",
            "arrivals": [],
        }
        raw = result.policy_rollout_positions
        arrivals = result.policy_waypoint_indices
        for slot, kind in enumerate(modalities):
            index = int(arrivals[0, slot])
            entry = {
                "environment": 0,
                "waypoint": slot,
                "modality": kind,
                "rollout_index": index,
            }
            if index >= 0:
                q = raw[:, index]
                pose = robot.compute_fk(qpos=q, name="arm", to_matrix=True)
                entry["position_error_m"] = float(
                    torch.linalg.vector_norm(
                        pose[0, :3, 3] - goal_poses[0, slot, :3, 3]
                    )
                )
                entry["rotation_error_rad"] = float(
                    quat_error_magnitude(
                        quat_from_matrix(pose[:, :3, :3]),
                        quat_from_matrix(goal_poses[:, slot, :3, :3]),
                    )[0]
                )
                entry["joint_max_error_rad"] = float(
                    (q[0] - goal_qpos[0, slot]).abs().max()
                )
            report["arrivals"].append(entry)
        succeeded = result.is_all_success()
        waypoint_indices = (
            result.waypoint_indices[0].cpu().tolist() if succeeded else []
        )
        markers = None
        if waypoint_indices and (
            args.video or args.web or (not args.headless and sim.has_native_renderer)
        ):
            markers = TrajectoryMarkers(
                sim,
                robot,
                goal_poses[0],
                result.xpos_list[0],
                waypoint_indices,
                modalities,
            )

        # A geometric failure is reported, never passed off as a completed motion.
        if succeeded and not args.no_playback:
            feedback = args.scenario == "closed-loop-perturbation"
            if args.video:
                if feedback:
                    recording_seconds = (
                        MAX_STEPS_PER_TARGET * len(targets) + 1
                    ) * control_dt + FINAL_HOLD_SECONDS
                else:
                    recording_seconds = float(result.duration[0]) + len(targets) * (
                        _PHASE_HOLD_SECONDS + control_dt
                    )
                # The recorder holds RGB frames in memory. Budget for the full
                # sequence, including the control-grid rounding and settles.
                frame_count = math.ceil(recording_seconds * _FPS) + 1
                memory_mb = math.ceil(
                    frame_count
                    * sim.sim_config.width
                    * sim.sim_config.height
                    * 3
                    / 2**20
                )
                if not sim.start_window_record(
                    save_path=str(args.video),
                    fps=_FPS,
                    max_memory=memory_mb,
                    look_at=look_at,
                    use_sim_time=True,
                ):
                    raise RuntimeError("Could not start the demo video recorder.")
            if feedback:
                playback = play_closed_loop(
                    sim,
                    generator.planner,
                    targets,
                    modalities,
                    joints,
                    control_dt,
                    markers,
                )
            else:
                playback = _play_free_motion(
                    sim, robot, result, waypoint_indices, joints, control_dt, markers
                )
            report.update(playback)
            if args.video:
                if not sim.stop_window_record():
                    raise RuntimeError("No video frames were captured.")
                sim.wait_window_record_saves()
                if not args.video.is_file() or args.video.stat().st_size == 0:
                    raise RuntimeError(f"Video encoding did not produce {args.video}.")
                report["video"] = str(args.video.resolve())
        report["duration_s"] = result.duration.cpu().tolist()
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        if not succeeded:
            raise RuntimeError(
                "NMG did not reach every demo target; see the arrival report."
            )
        if report["simulation_playback"] == "failed":
            raise RuntimeError("NMG feedback did not reach every demo target.")
        if args.keep_open:
            while True:
                sim.update()
                time.sleep(0.02)
    finally:
        if sim.is_window_recording():
            sim.stop_window_record()
        sim.wait_window_record_saves()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
