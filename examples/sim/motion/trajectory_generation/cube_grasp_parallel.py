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

"""Show grasp-pose and path augmentation on four synchronized cube pickups.

Run from the repository root with the simulation and video dependencies::

    python examples/sim/motion/trajectory_generation/cube_grasp_parallel.py --output /tmp/cube-grasp-parallel

The four real physics rows share the same initial robot and cube poses. A cube
symmetry supplies two grasp orientations; joint residuals supply two transit
paths per orientation. Existing Atomic Skills plan the final approach, close,
and lift. The cube moves only through physical contact with the gripper.

This example saves a four-panel MP4, measured rollout arrays, and a grasp-result
report. It does not use the free-motion-only GenerationRunner or certify a
contact-aware expert dataset. Planning uses the existing UR IK solver and the
MotionGenerator ik_interp strategy, without a collision-planning guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    EndEffectorPoseGoal,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    create_simulation_atomic_action_engine,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.motion.expansion import (
    TrajectoryPhase,
    TrajectoryTemplate,
    joint_residual,
    rotate_grasp_about_object_axis,
    validate_motion_limits,
)
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import VisualizationCfg
from embodichain.utils.math import look_at_to_pose
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_antipodal_semantics,
    create_toppra_motion_generator,
    create_ur5_gripper_robot_cfg,
    get_hand_open_close_qpos,
)

__all__ = ["run_cube_grasp_parallel", "main"]

_FPS = 20
_CONTROL_DT = 1 / _FPS
_PHYSICS_DT = 0.005
_WIDTH, _HEIGHT = 640, 480
_HEADER = 48
_YAW = (0.0, 0.0, math.pi / 2, math.pi / 2)
_COLORS = ((79, 183, 244), (250, 180, 63), (177, 143, 245), (80, 213, 160))
_EYE, _TARGET = (-1.0, -1.0, 1.12), (-0.28, -0.08, 0.56)
_INTRINSICS = (570.0, 570.0, 320.0, 240.0)


def _grasp_results(
    cube_poses: np.ndarray, tcp_poses: np.ndarray, hold_start: int
) -> list[dict[str, object]]:
    """Check sustained lift and object stability relative to the actual TCP."""
    results = []
    for row in range(4):
        cube_hold = cube_poses[row, hold_start:]
        tcp_hold = tcp_poses[row, hold_start:]
        relative = np.linalg.inv(tcp_hold) @ cube_hold
        min_lift = float((cube_hold[:, 2, 3] - cube_poses[row, 0, 2, 3]).min())
        drift = float(
            np.linalg.norm(relative[:, :3, 3] - relative[0, :3, 3], axis=-1).max()
        )
        distance = float(np.linalg.norm(relative[:, :3, 3], axis=-1).max())
        results.append(
            {
                "row": row,
                "yaw_degrees": round(math.degrees(_YAW[row])),
                "path": "residual" if row % 2 else "reference",
                "success": min_lift >= 0.12 and drift <= 0.01 and distance <= 0.06,
                "min_hold_lift_m": min_lift,
                "hold_relative_drift_m": drift,
                "max_cube_tcp_distance_m": distance,
            }
        )
    return results


def _preview_frame(
    rgb: np.ndarray,
    trails: list[list[tuple[float, float]]],
    elapsed: float,
    phase: str,
    lifts: np.ndarray,
    font: ImageFont.ImageFont,
    path_labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Tile camera frames from the same simulation tick with measured TCP trails."""
    preview = Image.new("RGB", (2 * _WIDTH, 2 * (_HEIGHT + _HEADER)), (18, 23, 32))
    for row in range(4):
        panel = Image.fromarray(rgb[row])
        draw = ImageDraw.Draw(panel)
        if len(trails[row]) > 1:
            draw.line(trails[row], fill=_COLORS[row], width=3)
        left, top = (row % 2) * _WIDTH, (row // 2) * (_HEIGHT + _HEADER)
        preview.paste(panel, (left, top + _HEADER))
        draw = ImageDraw.Draw(preview)
        label = (
            path_labels[row]
            if path_labels is not None
            else ("reference path" if row % 2 == 0 else "augmented path")
        )
        draw.text(
            (left + 12, top + 3),
            f"{row + 1}  |  grasp {round(math.degrees(_YAW[row]))} deg  |  {label}",
            fill=_COLORS[row],
            font=font,
        )
        draw.text(
            (left + 12, top + 25),
            f"{elapsed:5.2f} s  |  {phase}  |  cube lift {lifts[row] * 100:5.1f} cm",
            fill=(229, 234, 241),
            font=font,
        )
    return np.asarray(preview)


def _prepare_cube_scene(
    sim: SimulationManager, *, arm_stiffness: float = 5e4, hand_open_margin: float = 0.0
):
    """Share the four-row physical scene and initial state between examples."""
    robot_cfg = create_ur5_gripper_robot_cfg(init_pos=(0.0, 0.0, 0.3), tcp_z=0.15)
    robot_cfg.drive_pros.stiffness["arm"] = arm_stiffness
    robot = sim.add_robot(cfg=robot_cfg)
    sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="bench",
            body_type="kinematic",
            init_pos=(-0.2, -0.08, 0.15),
            shape=CubeCfg(
                size=(0.9, 0.65, 0.3),
                visual_material=VisualMaterialCfg(
                    uid="bench_material", base_color=[0.16, 0.20, 0.27, 1.0]
                ),
            ),
        )
    )
    cube = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=VisualMaterialCfg(
                    uid="cube_material", base_color=[0.95, 0.23, 0.04, 1.0]
                ),
            ),
            init_pos=(-0.42, -0.08, 0.325),
            attrs=RigidBodyAttributesCfg(
                mass=0.05, dynamic_friction=0.97, static_friction=0.99
            ),
        )
    )
    camera = sim.add_sensor(
        CameraCfg(
            uid="comparison_camera",
            visualization_role="record",
            width=_WIDTH,
            height=_HEIGHT,
            far=2.5,
            intrinsics=_INTRINSICS,
            extrinsics=CameraCfg.ExtrinsicsCfg(eye=_EYE, target=_TARGET),
        )
    )
    # Settle the identical cubes before fixing the reference state. No
    # object pose, gravity, or dynamics are changed during the rollout.
    sim.update(step=100)
    object_poses = cube.get_local_pose(to_matrix=True).clone()
    torch.testing.assert_close(
        object_poses, object_poses[0:1].expand_as(object_poses), atol=1e-5, rtol=0
    )
    reference_grasp = torch.eye(4, device=robot.device)
    reference_grasp[:3, :3] = reference_grasp.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    reference_grasp[:3, 3] = object_poses[0, :3, 3]
    grasps = rotate_grasp_about_object_axis(
        object_poses[0],
        reference_grasp,
        axis=torch.tensor([0.0, 0.0, 1.0]),
        angles=torch.tensor(_YAW),
    )
    start_pose = reference_grasp.repeat(4, 1, 1)
    start_pose[:, :3, 3] += start_pose.new_tensor([0.12, 0.0, 0.34])
    success, arm_qpos = robot.compute_ik(
        pose=start_pose, joint_seed=robot.get_qpos(name="arm"), name="arm"
    )
    if not success.all():
        raise RuntimeError("The common initial TCP pose is unreachable")
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    hand_open = hand_open + hand_open_margin
    initial_qpos = torch.tensor(robot.cfg.init_qpos, device=robot.device).repeat(4, 1)
    initial_qpos[:, robot.get_joint_ids("arm")] = arm_qpos
    initial_qpos[:, robot.get_joint_ids("hand")] = hand_open
    for child, parent, multiplier, offset in zip(
        robot.mimic_ids,
        robot.mimic_parents,
        robot.mimic_multipliers,
        robot.mimic_offsets,
    ):
        initial_qpos[:, child] = initial_qpos[:, parent] * multiplier + offset
    for target in (False, True):
        robot.set_qpos(initial_qpos, target=target)
    robot.clear_dynamics()

    return robot, cube, camera, initial_qpos, grasps, hand_open, hand_close


def _compile_cube_pickup(robot, cube, grasps, hand_open, hand_close):
    """Compile the same atomic transit and PickUp for preview and collection."""
    generator = create_toppra_motion_generator(robot)
    engine = create_simulation_atomic_action_engine(
        motion_generator=generator,
        scene_entities=(cube,),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open, grasp=hand_close
            )
        },
    )
    pregrasp = grasps.clone()
    pregrasp[:, 2, 3] += 0.16
    compiled = engine.compile(
        (
            engine.make_invocation(
                "move_end_effector",
                EndEffectorPoseGoal(pregrasp),
                control_parts={"primary": {"motion": "arm"}},
                motion_policy=MotionPolicy(strategy="ik_interp", sample_count=81),
            ),
            engine.make_invocation(
                "pick_up",
                GraspGoal(
                    create_antipodal_semantics(cube, label="cube"),
                    grasp_xpos=grasps,
                ),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=MotionPolicy(strategy="ik_interp", sample_count=141),
                skill_options=PickUpOptions(
                    pre_grasp_distance=0.16,
                    lift_height=0.18,
                    hand_interp_steps=30,
                    grasp_settle_steps=20,
                ),
            ),
        ),
        engine.initial_context(control_dt=_CONTROL_DT),
    )
    return generator, compiled


def run_cube_grasp_parallel(
    output_dir: Path, *, seed: int = 13, cuda_device: int = 0
) -> dict[str, object]:
    """Run four simultaneous physical pickups and save their measured evidence.

    Args:
        output_dir: New or empty directory for the MP4, arrays, and report.
        seed: Local random seed for both residual variants; no global RNG reset.
        cuda_device: GPU used by the offscreen renderer.

    Returns:
        The persisted report with per-row grasp outcomes and augmentation factors.

    Raises:
        ValueError: If the output directory is occupied or the seed is invalid.
        RuntimeError: If initialization, IK, or sampled motion-limit checks fail.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output_dir must be new or empty")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            num_envs=4,
            physics_dt=_PHYSICS_DT,
            arena_space=4.0,
            gpu_id=cuda_device,
            visualization=VisualizationCfg(),
        )
    )
    generator = writer = None
    try:
        robot, cube, camera, initial_qpos, grasps, hand_open, hand_close = (
            _prepare_cube_scene(sim)
        )
        generator, compiled = _compile_cube_pickup(
            robot, cube, grasps, hand_open, hand_close
        )
        if not compiled.plan_success.all():
            raise RuntimeError(
                f"IK planning failed in rows: {compiled.plan_success.tolist()}"
            )
        transit_stop = compiled.action_waypoint_offset(1)
        positions = compiled.trajectory.positions.clone()
        # Atomic concatenation retains the next action's zero-time start.
        # Here that duplicate becomes one explicit control-period hold so all
        # four rows and their camera frames share a uniform execution clock.
        torch.testing.assert_close(
            positions[:, transit_stop],
            positions[:, transit_stop - 1],
            atol=1e-5,
            rtol=0,
        )
        intervals = compiled.trajectory.dt.clone()
        intervals[:, transit_stop] = _CONTROL_DT
        baseline = positions.clone()
        arm_ids = tuple(robot.get_joint_ids("arm"))
        limits = robot.get_qpos_limits()[0]
        motion_checks = []
        for row in range(4):
            template = TrajectoryTemplate(
                "cube_pickup",
                "v1",
                f"yaw_{round(math.degrees(_YAW[row]))}",
                tuple(robot.joint_names),
                positions[row],
                intervals[row],
                (
                    TrajectoryPhase(
                        "transit",
                        0,
                        transit_stop,
                        allowed_operators=("joint_residual",),
                    ),
                    TrajectoryPhase(
                        "pickup", transit_stop, positions.shape[1], kind="contact"
                    ),
                ),
                allowed_operators=("joint_residual",),
                controlled_joint_indices=arm_ids,
            )
            if row % 2:
                template = joint_residual(
                    template,
                    joint_limits=limits,
                    normalized_scale=0.025,
                    generator=torch.Generator().manual_seed(seed),
                )
            check = validate_motion_limits(
                template,
                velocity_limits=torch.full((robot.dof,), 1.5),
                acceleration_limits=torch.full((robot.dof,), 15.0),
            )
            if not check.accepted:
                raise RuntimeError(f"Motion limits failed in row {row}: {check}")
            motion_checks.append(dict(check.checks[0].metrics))
            positions[row] = template.positions
        # The augmentation is confined to transit: all closing/contact/lift
        # commands and the endpoint into the pre-grasp corridor are preserved.
        assert torch.equal(
            positions[:, transit_stop - 1 :], baseline[:, transit_stop - 1 :]
        )
        torch.testing.assert_close(positions[:, 0], robot.get_qpos(), atol=1e-5, rtol=0)
        count = positions.shape[1]
        hold_frames = 30
        positions = torch.cat(
            (positions, positions[:, -1:].repeat(1, hold_frames, 1)), dim=1
        )
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        writer = imageio.get_writer(
            str(output_dir / "preview.mp4"),
            format="FFMPEG",
            fps=_FPS,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        view = torch.linalg.inv(look_at_to_pose(_EYE, _TARGET)).squeeze(0).numpy()
        trails: list[list[tuple[float, float]]] = [[] for _ in range(4)]
        qpos_samples, cube_samples, tcp_samples, timestamps = [], [], [], []
        start_time = sim.simulation_time
        close_start = compiled.segment(1, "close").start
        lift_start = compiled.segment(1, "lift").start

        for index in range(positions.shape[1]):
            if index:
                robot.set_qpos(positions[:, index])
                sim.update(step=round(_CONTROL_DT / _PHYSICS_DT))
            elapsed = sim.simulation_time - start_time
            if not math.isclose(elapsed, index * _CONTROL_DT, abs_tol=1e-7):
                raise RuntimeError(
                    "The measured simulation clock diverged from the video clock"
                )
            qpos_samples.append(robot.get_qpos().cpu().numpy().copy())
            cube_samples.append(
                cube.get_local_pose(to_matrix=True).cpu().numpy().copy()
            )
            tcp = robot.compute_fk(
                robot.get_qpos(name="arm"), name="arm", to_matrix=True
            )
            tcp_samples.append(tcp.cpu().numpy().copy())
            timestamps.append(elapsed)
            for row in range(4):
                point = view @ np.r_[tcp_samples[-1][row, :3, 3], 1.0]
                if point[2] > 0:
                    fx, fy, cx, cy = _INTRINSICS
                    trails[row].append(
                        (fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy)
                    )
            phase = (
                "transit"
                if index < transit_stop
                else (
                    "approach"
                    if index < close_start
                    else (
                        "close gripper"
                        if index < lift_start
                        else "lift" if index < count else "hold"
                    )
                )
            )
            camera.update()
            rgb = camera.get_data()["color"][..., :3].cpu().numpy().copy()
            writer.append_data(
                _preview_frame(
                    rgb,
                    trails,
                    elapsed,
                    phase,
                    cube_samples[-1][:, 2, 3] - cube_samples[0][:, 2, 3],
                    font,
                )
            )

        cubes = np.stack(cube_samples, axis=1)
        tcps = np.stack(tcp_samples, axis=1)
        results = _grasp_results(cubes, tcps, count)
        deviations = [
            float(
                np.linalg.norm(
                    tcps[row, :transit_stop, :3, 3]
                    - tcps[row - 1, :transit_stop, :3, 3],
                    axis=-1,
                ).max()
            )
            for row in (1, 3)
        ]
        report = {
            "success": all(result["success"] for result in results),
            "num_envs": 4,
            "seed": seed,
            "normalized_residual_scale": 0.025,
            "fps": _FPS,
            "frame_count": len(timestamps),
            "physical_duration_s": timestamps[-1],
            "transit_stop": transit_stop,
            "hold_start": count,
            "contact_commands_preserved": True,
            "max_path_pair_separation_m": deviations,
            "planned_motion_limits": motion_checks,
            "results": results,
            "validation_scope": "physical sustained lift and TCP-relative position stability; no contact-aware collision or expert-dataset certification",
        }
        np.savez_compressed(
            output_dir / "rollout.npz",
            timestamps=np.asarray(timestamps),
            commanded_qpos=positions.cpu().numpy(),
            measured_qpos=np.stack(qpos_samples, axis=1),
            cube_poses=cubes,
            tcp_poses=tcps,
            grasp_poses=grasps.cpu().numpy(),
            reference_qpos=baseline.cpu().numpy(),
        )
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if generator is not None:
                generator.planner.close()
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run the parallel comparison and return a failing exit code for failed grasps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    report = run_cube_grasp_parallel(
        args.output, seed=args.seed, cuda_device=args.cuda_device
    )
    print(json.dumps(report))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
