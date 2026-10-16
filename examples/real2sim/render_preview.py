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

"""Render one external qpos recording and two unqualified motion variants.

This is a kinematic inspection example: no physics stepping, object attachment,
physical-success claim or EpisodeSink commit. The qualified Real2SimSourceAdapter
continues to reject unverified input. The example uses the public residual
operator directly on an explicitly diagnostic template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import pytorch_kinematics as pk

from embodichain.data import get_data_path
from embodichain.data_pipeline.real2sim import (
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
)
from embodichain.data_pipeline.real2sim.normalization import (
    Real2SimNormalizationCfg,
    normalize_real2sim,
)
from embodichain.compute.trajectory import differentiate_positions
from embodichain.lab.sim.motion.expansion import TrajectoryPhase, TrajectoryTemplate
from embodichain.lab.sim.motion.expansion.operators import joint_residual

__all__: list[str] = []

ARM_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
COLORS = [(230, 210, 70), (65, 195, 255), (215, 125, 245)]  # BGR


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(
    frame: np.ndarray,
    message: str,
    xy: tuple[int, int],
    color=(235, 240, 245),
    scale=0.55,
) -> None:
    cv2.putText(
        frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _decode_video(path: Path) -> np.ndarray:
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    if (probe["width"], probe["height"]) != (640, 360):
        raise ValueError(
            "This optional recording example expects a 640x360 source video"
        )
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, np.uint8).reshape(-1, 360, 640, 3)


def _build_paths(args: argparse.Namespace, arm_path: Path):
    with np.load(args.input, allow_pickle=False) as data:
        q = data["joint_positions"].copy()
        timestamps = data["timestamps"].copy()
        observed_pose = (
            data["source_gripper_pose_xyzw"].copy()
            if "source_gripper_pose_xyzw" in data
            else None
        )
    if not 5 <= args.free_stop < len(q):
        raise ValueError(
            "--free-stop must leave an interior free window and protected suffix"
        )
    root = ET.parse(arm_path).getroot()
    limits = np.array(
        [
            [
                float(root.find(f"joint[@name='{name}']/limit").get(k))
                for k in ("lower", "upper")
            ]
            for name in ARM_NAMES
        ]
    )
    velocity_limits = np.array(
        [
            float(root.find(f"joint[@name='{name}']/limit").get("velocity"))
            for name in ARM_NAMES
        ]
    )
    raw = Real2SimTrajectory(
        source_id="external_motion_preview",
        source_revision=_sha(args.input),
        unit_id=args.input.stem,
        robot_id="Franka-arm",
        robot_revision=_sha(arm_path),
        joint_names=ARM_NAMES,
        joint_units=("rad",) * 7,
        positions=q,
        timestamps=timestamps,
        time_unit="ms",
        calibration_revision="preview-unverified",
        annotation_revision="manual-conservative-free-window-v1",
        phases=(
            SourcePhase("free", 0, args.free_stop, "free", ("joint_residual",)),
            SourcePhase("protected_suffix", args.free_stop, len(q), "contact"),
        ),
        provenance={
            "scope": "kinematic diagnostic only",
            "annotation_note": "Conservative free prefix reviewed from source images; suffix combines approach/contact/transport/release and is locked, not finely segmented.",
        },
    )
    layout = RobotJointLayout(
        "Franka-arm", _sha(arm_path), ARM_NAMES, ("rad",) * 7, q[0], ARM_NAMES
    )
    unit = normalize_real2sim(
        raw, layout, Real2SimNormalizationCfg(control_dt=0.01, allow_unverified=True)
    )
    free = unit.phases[0]
    # Protect two samples at each original endpoint for differentiation stencils.
    template = TrajectoryTemplate(
        source_id=raw.source_id,
        source_revision=unit.revision,
        template_id=raw.unit_id,
        joint_names=ARM_NAMES,
        positions=torch.from_numpy(unit.positions.copy()),
        dt=torch.from_numpy(unit.dt.copy()),
        controlled_joint_indices=tuple(range(7)),
        phases=(
            TrajectoryPhase(
                "diagnostic_free",
                free.start + 1,
                free.stop - 1,
                "free",
                ("joint_residual",),
            ),
        ),
        allowed_operators=("joint_residual",),
    )
    paths = [template]
    for seed in (7, 11):
        paths.append(
            joint_residual(
                template,
                joint_limits=torch.from_numpy(limits),
                normalized_scale=args.scale,
                generator=torch.Generator().manual_seed(seed),
            )
        )
    qpos = np.stack([p.positions.numpy() for p in paths])
    qvel = np.stack(
        [
            differentiate_positions(p.positions[None], p.dt[None])[0].numpy()
            for p in paths
        ]
    )
    protected = np.ones(len(unit.positions), dtype=bool)
    protected[free.start + 2 : free.stop - 2] = False
    if not np.array_equal(
        qpos[1:, protected],
        np.broadcast_to(qpos[0, protected], qpos[1:, protected].shape),
    ):
        raise AssertionError("Expansion changed protected qpos")
    if not np.array_equal(
        qvel[1:, free.stop - 1 :],
        np.broadcast_to(qvel[0, free.stop - 1 :], qvel[1:, free.stop - 1 :].shape),
    ):
        raise AssertionError("Expansion changed protected suffix velocity")
    chain = pk.build_serial_chain_from_urdf(arm_path.read_text(), "fr3_link8").to(
        dtype=torch.float64
    )
    if tuple(chain.get_joint_parameter_names()) != ARM_NAMES:
        raise ValueError("Asset FK joint order differs from the source declaration")
    fk = (
        chain.forward_kinematics(torch.from_numpy(qpos.reshape(-1, 7)))
        .get_matrix()
        .numpy()
        .reshape(3, -1, 4, 4)
    )
    source_fk = chain.forward_kinematics(torch.from_numpy(q)).get_matrix().numpy()
    consistency = (
        None
        if observed_pose is None
        else float(
            np.sqrt(
                np.mean(
                    np.sum((source_fk[:, :3, 3] - observed_pose[:, :3]) ** 2, axis=1)
                )
            )
        )
    )
    diagnostics = []
    for i in range(3):
        diagnostics.append(
            {
                "name": ["nominal", "variant_a", "variant_b"][i],
                "seed": [None, 7, 11][i],
                "max_joint_deviation_rad": float(np.abs(qpos[i] - qpos[0]).max()),
                "max_flange_deviation_m": float(
                    np.linalg.norm(fk[i, :, :3, 3] - fk[0, :, :3, 3], axis=1).max()
                ),
                "joint_limits_pass": bool(
                    ((qpos[i] >= limits[:, 0]) & (qpos[i] <= limits[:, 1])).all()
                ),
                "velocity_limits_pass": bool(
                    (np.abs(qvel[i]) <= velocity_limits).all()
                ),
                "physical_validation": "not_run",
                "collision_validation": "not_run",
            }
        )
    if not all(
        d["joint_limits_pass"] and d["velocity_limits_pass"] for d in diagnostics
    ):
        raise ValueError(f"Kinematic limit checks failed: {diagnostics}")
    return unit, qpos, qvel, fk, protected, diagnostics, consistency


def _render(args: argparse.Namespace) -> None:
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        RobotCfg,
        URDFCfg,
        RenderCfg,
        LightCfg,
        DefaultPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.sim.material import VisualMaterialCfg

    output = args.output.resolve()
    source_dir = args.input.resolve().parent
    if output == source_dir or source_dir in output.parents:
        raise ValueError("Output must be outside the recording directory")
    output.mkdir(parents=True, exist_ok=False)
    arm_path = Path(get_data_path("Franka/Panda/Panda.urdf"))
    hand_path = Path(get_data_path("Robotiq2F85/Robotiq2F85.urdf"))
    unit, qpos, qvel, fk, protected, diagnostics, consistency = _build_paths(
        args, arm_path
    )
    frames = _decode_video(args.video)
    if len(frames) not in (len(unit.source.positions), len(unit.source.positions) - 1):
        raise ValueError(
            "Expected N or N-1 source video frames; supply explicit synchronization for other layouts"
        )
    assembly = URDFCfg(
        components=[
            {
                "component_type": "arm",
                "urdf_path": str(arm_path),
                "transform": np.eye(4),
            },
            {
                "component_type": "hand",
                "urdf_path": str(hand_path),
                "transform": np.eye(4),
            },
        ],
        fname="franka_robotiq",
        fpath_prefix=str(output / "robot"),
    )
    cfg = RobotCfg(
        uid="preview_franka",
        urdf_cfg=assembly,
        control_parts={"arm": list(ARM_NAMES), "eef": ["finger_joint"]},
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=3,
            arena_space=20.0,
            physics_cfg=DefaultPhysicsCfg(),
            render_cfg=RenderCfg(renderer="hybrid", tone_mapping_enabled=True),
            startup_summary="off",
            robot_ik_gizmo=None,
        )
    )
    writer = None
    try:
        sim.set_emission_light([0.92, 0.95, 1.0], 30.0)
        sim.set_ground_plane_visibility(False)
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="preview_stage",
                body_type="static",
                init_pos=(0.0, 0.0, -0.025),
                shape=CubeCfg(
                    size=[8.0, 8.0, 0.04],
                    visual_material=VisualMaterialCfg(
                        uid="preview_stage_material",
                        base_color=[0.12, 0.16, 0.20, 1.0],
                        roughness=0.85,
                    ),
                ),
            )
        )
        robot = sim.add_robot(cfg)
        camera = sim.add_sensor(
            CameraCfg(
                uid="preview_camera",
                width=640,
                height=360,
                far=4.0,
                intrinsics=(620.0, 620.0, 320.0, 180.0),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(1.20, -1.45, 1.00),
                    target=(0.32, 0.0, 0.34),
                    up=(0.0, 0.0, 1.0),
                ),
            )
        )
        sim.add_light(
            LightCfg(
                uid="key",
                light_type="direction",
                intensity=8.0,
                direction=(-0.5, 0.4, -1.0),
            )
        )
        sim.prepare()
        ids = [robot.joint_names.index(n) for n in ARM_NAMES]
        # Fixed-open gripper is explicit: the recorded scalar has no verified unit mapping.
        full = torch.zeros(
            (3, len(robot.joint_names)), dtype=torch.float32, device=robot.device
        )
        duration = float(unit.time_map.execution_times[-1])
        render_times = np.arange(math.ceil(duration * args.fps) + 1) / args.fps
        if args.max_frames:
            render_times = render_times[: args.max_frames]
        execution_indices = np.minimum(
            np.rint(render_times / 0.01).astype(int), len(unit.positions) - 1
        )
        source_query = unit.time_map.source_query_times[execution_indices]
        source_indices = np.abs(
            unit.source.seconds[:, None] - source_query[None, :]
        ).argmin(axis=0)
        video_indices = np.clip(
            source_indices + args.video_state_offset, 0, len(frames) - 1
        )
        np.savez_compressed(
            output / "trajectories.npz",
            qpos=qpos,
            qvel=qvel,
            flange_fk=fk,
            protected_mask=protected,
            dt=unit.dt,
            render_times_s=render_times,
            execution_indices=execution_indices,
            source_indices=source_indices,
            video_indices=video_indices,
            **{
                "map_" + k: getattr(unit.time_map, k)
                for k in unit.time_map.__dataclass_fields__
            },
        )
        report = {
            "scope": "kinematic_motion_preview",
            "physics_stepped": False,
            "qualified_episode": False,
            "normalization": unit.report_dict(),
            "diagnostics": diagnostics,
            "source_video_sha256": _sha(args.video),
            "arm_asset_sha256": _sha(arm_path),
            "gripper_asset_sha256": _sha(hand_path),
            "assembled_asset": str(Path(assembly.fpath).relative_to(output)),
            "assembled_asset_sha256": _sha(Path(assembly.fpath)),
            "source_pose_vs_flange_position_rmse_m": consistency,
            "source_pose_is_independent_measurement": False,
            "free_source_samples": [0, args.free_stop],
            "joint_residual_scale": args.scale,
            "gripper_policy": "fixed open; scalar encoding and real mount calibration unverified",
            "mount_policy": "identity flange-to-gripper mount for preview only",
            "scene_policy": "neutral ground; no reconstructed objects or object animation",
            "preview_camera": {
                "intrinsics": camera.cfg.intrinsics,
                "eye": camera.cfg.extrinsics.eye,
                "target": camera.cfg.extrinsics.target,
                "source_camera_calibrated": False,
            },
            "video_state_offset": args.video_state_offset,
            "video_alignment_verified": False,
            "video_sync_policy": "state timestamp mapping; nearest frame; endpoint clamp for N-1 video",
            "render_frames": len(render_times),
            "fps": args.fps,
            "rendered_duration_s": len(render_times) / args.fps,
            "readback_max_error_rad": 0.0,
            "flange_readback_max_error_m": 0.0,
        }
        writer = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                "1280x900",
                "-r",
                str(args.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output / "comparison.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        allxy = fk[:, :, :2, 3].reshape(-1, 2)
        low, high = allxy.min(0) - 0.03, allxy.max(0) + 0.03

        def plot_path(panel, index, tick):
            origin = np.array([440, 290])
            span = np.array([160, -100])
            cv2.rectangle(panel, (430, 172), (630, 335), (26, 32, 40), -1)
            _text(panel, "Flange path (XY)", (440, 193), scale=0.42)
            for j in (0, index):
                points = origin + (fk[j, :, :2, 3] - low) / (high - low) * span
                cv2.polylines(
                    panel,
                    [points.astype(np.int32)],
                    False,
                    (95, 100, 110) if j == 0 else COLORS[index],
                    1,
                    cv2.LINE_AA,
                )
            now = origin + (fk[index, tick, :2, 3] - low) / (high - low) * span
            cv2.circle(panel, tuple(now.astype(int)), 4, COLORS[index], -1)
            _text(
                panel,
                f"delta {np.linalg.norm(fk[index,tick,:3,3]-fk[0,tick,:3,3])*1000:.1f} mm",
                (440, 318),
                COLORS[index],
                0.43,
            )

        for frame_number, tick in enumerate(execution_indices):
            full[:, ids] = torch.from_numpy(qpos[:, tick]).to(full)
            robot.set_qpos(full, target=False)
            with sim.render_frame():
                # Camera.update() consumes native buffers; explicitly publish direct
                # articulation writes without taking a positive-duration physics step.
                sim.sync_render_state()
                camera.update()
                rendered = camera.get_data()["color"].cpu().numpy()[..., :3].copy()
            error = float(
                np.max(np.abs(robot.get_qpos().cpu().numpy()[:, ids] - qpos[:, tick]))
            )
            report["readback_max_error_rad"] = max(
                report["readback_max_error_rad"], error
            )
            flange_error = float(
                np.linalg.norm(
                    robot.get_link_pose("fr3_link8").cpu().numpy()[:, :3]
                    - fk[:, tick, :3, 3],
                    axis=1,
                ).max()
            )
            report["flange_readback_max_error_m"] = max(
                report["flange_readback_max_error_m"], flange_error
            )
            if error > 1e-5 or flange_error > 1e-4:
                raise RuntimeError(
                    f"Preview state disagrees with source/FK: qpos={error}, flange={flange_error}"
                )
            panels = []
            for i in range(4):
                panel = np.full((410, 640, 3), (22, 28, 36), np.uint8)
                picture = (
                    frames[video_indices[frame_number]]
                    if i == 0
                    else cv2.cvtColor(rendered[i - 1], cv2.COLOR_RGB2BGR)
                )
                panel[38:398] = picture
                title = [
                    "REAL RECORDING / timestamp aligned",
                    "NOMINAL / recorded arm qpos",
                    "VARIANT A / seed 7",
                    "VARIANT B / seed 11",
                ][i]
                _text(
                    panel,
                    title,
                    (18, 26),
                    (235, 240, 245) if i == 0 else COLORS[i - 1],
                    0.62,
                )
                if i:
                    plot_path(panel, i - 1, tick)
                    _text(
                        panel,
                        "KINEMATIC | fixed-open gripper",
                        (12, 385),
                        (30, 205, 250),
                        0.5,
                    )
                panels.append(panel)
            canvas = np.full((900, 1280, 3), (18, 23, 30), np.uint8)
            canvas[:410] = np.concatenate(panels[:2], axis=1)
            canvas[410:820] = np.concatenate(panels[2:], axis=1)
            free_time = float(unit.time_map.execution_times[unit.phases[0].stop - 1])
            stage = (
                "FREE PREFIX / augmentation enabled"
                if tick < unit.phases[0].stop
                else "PROTECTED SUFFIX / identical arm motion"
            )
            _text(
                canvas,
                f"t={render_times[frame_number]:05.2f}s  |  {stage}",
                (20, 845),
                (220, 230, 240),
                0.6,
            )
            _text(
                canvas,
                "Motion inspection only: no grasp/contact validation; source video offset and mount remain uncalibrated.",
                (20, 870),
                (170, 185, 205),
                0.46,
            )
            x0, x1 = 20, 1260
            cv2.line(canvas, (x0, 889), (x1, 889), (65, 75, 85), 5)
            cv2.line(
                canvas,
                (x0, 889),
                (int(x0 + (x1 - x0) * free_time / duration), 889),
                COLORS[1],
                5,
            )
            cv2.circle(
                canvas,
                (
                    int(
                        x0
                        + (x1 - x0)
                        * min(render_times[frame_number], duration)
                        / duration
                    ),
                    889,
                ),
                6,
                (245, 245, 245),
                -1,
            )
            writer.stdin.write(canvas.tobytes())
            if frame_number in (
                0,
                min(60, len(execution_indices) - 1),
                len(execution_indices) - 1,
            ):
                cv2.imwrite(str(output / f"frame_{frame_number:04d}.jpg"), canvas)
            if frame_number % 90 == 0:
                print(f"Rendered {frame_number+1}/{len(execution_indices)}", flush=True)
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("Video encoder failed")
        writer = None
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"output": str(output), "diagnostics": diagnostics}, indent=2),
            flush=True,
        )
    finally:
        if writer is not None:
            writer.stdin.close()
            writer.wait()
        sim.destroy(exit_process=False)
    del robot, camera, sim
    SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Parse one optional recording and render a non-qualifying preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--free-stop",
        type=int,
        required=True,
        help="Reviewed exclusive source index; remaining motion is locked",
    )
    parser.add_argument("--scale", type=float, default=0.025)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-state-offset", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Bounded renderer smoke check; zero renders the whole recording",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 0:
        parser.error("fps must be positive and max-frames nonnegative")
    _render(args)


if __name__ == "__main__":
    main()
