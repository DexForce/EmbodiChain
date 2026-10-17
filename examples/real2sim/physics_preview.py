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

"""Measured physical interaction preview for one external qpos recording.

Uses the shared fixed-cadence playback host. Objects are initialized only at
reset and then move exclusively through gravity/contact; no attachment or
object-pose playback is used. Scene geometry and hardware calibration remain
estimates, so this diagnostic never grants real2sim source admission.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation

from embodichain.data import get_data_path
from embodichain.data_pipeline.real2sim.gripper import (
    GripperPositionCalibration,
    gripper_release_duration,
)
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    RenderCfg,
    LightCfg,
    DefaultPhysicsCfg,
    RigidObjectCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    MassPropertiesCfg,
    RigidBodyMaterialCfg,
    CollisionPropertiesCfg,
    DefaultRigidBodyPropertiesCfg,
    ArticulationRootPropertiesCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sensors import CameraCfg, ContactSensorCfg
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)
from embodichain.lab.sim.motion.execution import (
    play_joint_trajectory,
    JointTrajectoryPlaybackCfg,
)
from examples.real2sim.render_preview import (
    ARM_NAMES,
    _build_paths,
    _decode_video,
    _text,
)

__all__: list[str] = []
CONTROL_DT = 0.01
PHYSICS_DT = 0.001


def _physics(friction: float, mass: float | None = None) -> RigidBodyPhysicsCfg:
    return RigidBodyPhysicsCfg(
        mass_props=None if mass is None else MassPropertiesCfg(mass=mass),
        collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        material_props=RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction, restitution=0.0
        ),
        rigid_props=DefaultRigidBodyPropertiesCfg(
            linear_damping=0.02, angular_damping=0.05, enable_ccd=True
        ),
    )


def _model_geometry(arm_path: Path, hand_path: Path, q: np.ndarray, driver: float):
    arm = pk.build_serial_chain_from_urdf(arm_path.read_bytes(), "fr3_link8").to(
        dtype=torch.float64
    )
    flange = arm.forward_kinematics(torch.from_numpy(q[None])).get_matrix()[0].numpy()
    hand = pk.build_chain_from_urdf(hand_path.read_bytes()).to(dtype=torch.float64)
    joints = [
        j
        for j in ET.parse(hand_path).getroot().findall("joint")
        if j.get("type") != "fixed"
    ]
    values = {
        j.get("name"): torch.tensor(
            [
                driver
                * (
                    float(j.find("mimic").get("multiplier", "1"))
                    if j.find("mimic") is not None
                    else 1.0
                )
            ],
            dtype=torch.float64,
        )
        for j in joints
    }
    fk = hand.forward_kinematics(values)
    pads = []
    for name in ("left_inner_finger", "right_inner_finger"):
        pads.append(
            (fk[name].get_matrix()[0].numpy() @ np.array([-0.0078, 0, 0.033, 1]))[:3]
        )
    center = (flange @ np.r_[np.mean(pads, axis=0), 1])[:3]
    yaw = math.atan2(flange[1, 0], flange[0, 0])
    return center, yaw, flange


def _box(
    sim, uid, size, pos, color, *, yaw=0.0, dynamic=False, mass=None, friction=0.8
):
    return sim.add_rigid_object(
        RigidObjectCfg(
            uid=uid,
            body_type="dynamic" if dynamic else "static",
            init_pos=tuple(float(v) for v in pos),
            init_rot=(0.0, 0.0, float(np.rad2deg(yaw))),
            shape=CubeCfg(
                size=list(size),
                visual_material=VisualMaterialCfg(
                    uid=(
                        "physics_bowl_mat" if uid.startswith("bowl_") else uid + "_mat"
                    ),
                    base_color=[*color, 1.0],
                    roughness=0.75,
                ),
            ),
            attrs=_physics(friction, mass),
        )
    )


class _ObservedPlayback:
    """Observe the existing playback host without changing its control clock."""

    def __init__(
        self,
        sim,
        robot,
        block,
        sensor,
        targets,
        initial_z,
        bowl_center,
        table_z,
        cameras,
        writer,
        fps,
        label,
        *,
        open_driver_rad=0.0,
        release_tolerance_rad=0.03,
    ):
        self.sim = sim
        self.sim_config = sim.sim_config
        self.robot = robot
        self.block = block
        self.sensor = sensor
        self.targets = targets
        self.initial_z = initial_z
        self.bowl_center = bowl_center
        self.table_z = table_z
        self.cameras = cameras
        self.writer = writer
        self.fps = fps
        self.label = label
        self.open_driver_rad = open_driver_rad
        self.release_tolerance_rad = release_tolerance_rad
        self.samples = []
        self.time = 0.0
        self.frame_count = 0
        self.actor_names = {}

    def update(self, physics_dt, steps):
        self.sim.update(physics_dt, steps, render_final_step=False)
        self.time += physics_dt * steps
        index = min(len(self.samples), len(self.targets) - 1)
        pose = self.block.get_local_pose()[0].cpu().numpy().copy()
        q = self.robot.get_qpos()[0].cpu().numpy().copy()
        self.sensor.update()
        contact = self.sensor.get_data()
        valid = contact["is_valid"].cpu().numpy().astype(bool)
        pairs = contact["user_ids"].cpu().numpy()[valid]
        names = []
        for pair in pairs:
            labels = []
            for actor in pair:
                if actor >= 0:
                    actor = int(actor)
                    if actor not in self.actor_names:
                        info = self.sensor.get_actor_info(actor)
                        self.actor_names[actor] = info.path + (
                            "/" + info.link_name if info.link_name else ""
                        )
                    labels.append(self.actor_names[actor])
            names.extend(labels)
        left = any("left_inner_finger" in x or "left_outer_finger" in x for x in names)
        right = any(
            "right_inner_finger" in x or "right_outer_finger" in x for x in names
        )
        impulse = contact["impulse"].cpu().numpy()[valid]
        target = self.targets[index]
        active = self.active_ids
        error = float(np.abs(q[active] - target).max())
        flange = self.robot.get_link_pose("fr3_link8")[0].cpu().numpy().copy()
        self.samples.append(
            {
                "t": self.time,
                "pose": pose,
                "flange": flange,
                "qpos": q,
                "target": target.copy(),
                "bilateral": left and right,
                "left": left,
                "right": right,
                "contact_count": len(pairs),
                "impulse": float(impulse.max()) if len(impulse) else 0.0,
                "tracking_error": error,
            }
        )
        if self.writer is not None:
            while self.frame_count / self.fps <= self.time + 1e-10:
                with self.sim.render_frame():
                    self.sim.sync_render_state()
                    images = []
                    for camera in self.cameras:
                        camera.update()
                        images.append(
                            cv2.cvtColor(
                                camera.get_data()["color"][0, :, :, :3].cpu().numpy(),
                                cv2.COLOR_RGB2BGR,
                            )
                        )
                canvas = np.zeros((450, 640, 3), np.uint8)
                canvas[:] = (22, 28, 36)
                canvas[40:400] = images[1]
                canvas[46:158, 6:206] = cv2.resize(images[0], (200, 112))
                _text(canvas, self.label, (16, 27), scale=0.65)
                height = (pose[2] - self.initial_z) * 1000
                _text(
                    canvas,
                    f"t={self.time:05.2f}s   block lift {height:+.0f} mm",
                    (12, 421),
                    scale=0.55,
                )
                _text(
                    canvas,
                    f"contact L/R: {int(left)}/{int(right)}   q error: {error:.3f} rad",
                    (12, 444),
                    scale=0.48,
                )
                self.writer.stdin.write(canvas.tobytes())
                self.frame_count += 1
        if len(self.samples) % 300 == 0:
            print(
                f"{self.label}: t={self.time:.2f}s lift={pose[2]-self.initial_z:.3f} contacts={left}/{right} err={error:.4f}",
                flush=True,
            )


def _resting_checks(poses, times, block_size, bowl_center, table_z, bowl_radius):
    positions = poses[:, :3]
    final = positions[-1]
    tail = poses[times >= times[-1] - 1.0]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = (signs * np.array(block_size) / 2) @ Rotation.from_quat(
        poses[-1, 3:]
    ).as_matrix().T + final
    inner_radius = bowl_radius - 0.004
    inside = bool(
        (
            np.linalg.norm(corners[:, :2] - bowl_center[:2], axis=1)
            <= inner_radius + 0.001
        ).all()
        and corners[:, 2].min() >= table_z + 0.008 - 0.002
        and corners[:, 2].max() <= table_z + 0.06 + 0.002
    )
    speed = float(
        np.linalg.norm(np.diff(tail[:, :3], axis=0) / CONTROL_DT, axis=1).max()
    )
    rotations = Rotation.from_quat(tail[:, 3:])
    angle = float((rotations * rotations[-1].inv()).magnitude().max())
    angular_speed = float(
        (rotations[1:] * rotations[:-1].inv()).magnitude().max() / CONTROL_DT
    )
    stable = bool(
        np.linalg.norm(tail[:, :3] - final, axis=1).max() < 0.005
        and speed < 0.03
        and angle < np.deg2rad(3)
        and angular_speed < 0.2
    )
    return {
        "final_inside_container": inside,
        "stable_last_second": stable,
        "max_final_second_object_speed_m_s": speed,
        "max_final_second_rotation_rad": angle,
        "max_final_second_angular_speed_rad_s": angular_speed,
    }


def _summary(observer, block_size, bowl_radius):
    data = observer.samples
    positions = np.array([s["pose"][:3] for s in data])
    times = np.array([s["t"] for s in data])
    heights = positions[:, 2] - observer.initial_z
    bilateral = np.array([s["bilateral"] for s in data])
    start = None
    sustained = 0.0
    for timestamp, valid in zip(times, bilateral & (heights > 0.04)):
        if valid:
            if start is None:
                start = timestamp
            sustained = max(sustained, float(timestamp - start))
        else:
            start = None
    final = positions[-1]
    resting = _resting_checks(
        np.array([s["pose"] for s in data]),
        times,
        block_size,
        observer.bowl_center,
        observer.table_z,
        bowl_radius,
    )
    inside = resting["final_inside_container"]
    stable = resting["stable_last_second"]
    q = np.array([s["qpos"] for s in data])[:, observer.active_ids]
    targets = np.array([s["target"] for s in data])
    measured_open = (
        np.abs(q[:, -1] - observer.open_driver_rad) <= observer.release_tolerance_rad
    )
    released_for = gripper_release_duration(times, measured_open, bilateral)
    release_pass = released_for >= 1.0 - 1e-9
    return {
        "max_block_lift_m": float(heights.max()),
        "continuous_bilateral_lift_s": sustained,
        "grasp_lift_pass": sustained >= 1.0,
        "final_inside_container": bool(inside),
        "stable_last_second": bool(stable),
        "continuous_release_s": released_for,
        "release_pass": release_pass,
        "physical_task_pass": bool(
            sustained >= 1.0 and inside and stable and release_pass
        ),
        "max_joint_tracking_error_rad": float(max(s["tracking_error"] for s in data)),
        "arm_tracking_rmse_rad": float(
            np.sqrt(np.mean((q[:, :7] - targets[:, :7]) ** 2))
        ),
        "max_arm_tracking_error_rad": float(np.abs(q[:, :7] - targets[:, :7]).max()),
        **resting,
        "final_block_xyz": final.tolist(),
        "physical_validation_scope": "estimated_proxy_scene",
        "real2sim_qualified": False,
    }


def _compose(args, output, unit, results):
    source_frames = _decode_video(args.source_video)
    captures = [cv2.VideoCapture(str(output / f"candidate_{i}.mp4")) for i in range(3)]
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
    if len(set(counts)) != 1:
        raise RuntimeError("Candidate video clocks differ")
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
            "1280x960",
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
            str(output / "physical_comparison.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in range(counts[0]):
            time = frame / args.fps
            tick = min(int(round(time / CONTROL_DT)), len(unit.positions) - 1)
            index = int(
                np.abs(
                    unit.source.seconds - unit.time_map.source_query_times[tick]
                ).argmin()
            )
            source = np.full((450, 640, 3), (22, 28, 36), np.uint8)
            source[40:400] = source_frames[min(index, len(source_frames) - 1)]
            _text(source, "REAL RECORDING / state-clock mapping", (16, 27), scale=0.62)
            _text(
                source,
                f"t={min(time,float(unit.source.seconds[-1])):05.2f}s  source frame {index}",
                (12, 422),
                scale=0.55,
            )
            _text(
                source, "Camera frame offset remains unverified", (12, 444), scale=0.48
            )
            panels = []
            for cap in captures:
                ok, panel = cap.read()
                if not ok:
                    raise RuntimeError("Candidate frame decode failed")
                panels.append(panel)
            canvas = np.full((960, 1280, 3), (18, 23, 30), np.uint8)
            canvas[:450] = np.concatenate([source, panels[0]], axis=1)
            canvas[450:900] = np.concatenate(panels[1:], axis=1)
            phase = (
                "FREE MOTION VARIANTS"
                if tick < unit.phases[0].stop
                else "PROTECTED MOTION / PHYSICAL CONTACT"
            )
            _text(
                canvas, f"PHYSICS 1 ms | CONTROL 10 ms | {phase}", (16, 925), scale=0.57
            )
            _text(
                canvas,
                f"Estimated foreground + uncalibrated mount | measured arm qpos; explicit gripper preload {args.grip_preload:.3f} rad",
                (16, 950),
                (175, 195, 210),
                0.47,
            )
            writer.stdin.write(canvas.tobytes())
            if frame in (0, 255, counts[0] - 1):
                cv2.imwrite(str(output / f"comparison_{frame:04d}.jpg"), canvas)
        card = np.full((960, 1280, 3), (18, 23, 30), np.uint8)
        _text(card, "MEASURED PHYSICAL DIAGNOSTIC", (70, 120), scale=1.1)
        _text(
            card,
            "Same scene, reset-only object initialization, no object attachment",
            (70, 180),
            scale=0.72,
        )
        for i, name in enumerate(
            ("Nominal", "Variant A / seed 7", "Variant B / seed 11")
        ):
            rows = [r for r in results if r["candidate"] == i]
            passed = sum(r["physical_task_pass"] for r in rows)
            _text(
                card,
                f"{name}:  {passed}/{len(rows)} passed",
                (80, 300 + 150 * i),
                (225, 230, 235),
                0.95,
            )
            _text(
                card,
                f"Lift {min(r['max_block_lift_m'] for r in rows)*1000:.0f} mm | hold {min(r['continuous_bilateral_lift_s'] for r in rows):.2f} s | released {min(r['continuous_release_s'] for r in rows):.2f} s | stable 1 s",
                (80, 345 + 150 * i),
                (160, 195, 215),
                0.67,
            )
        _text(
            card,
            "Proxy geometry and model-based gripper mapping; real2sim calibration remains unverified.",
            (70, 850),
            (210, 190, 130),
            0.62,
        )
        for _ in range(2 * args.fps):
            writer.stdin.write(card.tobytes())
    finally:
        for cap in captures:
            cap.release()
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("Comparison video encoding failed")


def _run(args):
    output = args.output.resolve()
    source = args.input.resolve()
    if output == source.parent or source.parent in output.parents:
        raise ValueError("Output must be outside the original recording directory")
    output.mkdir(parents=True, exist_ok=False)
    arm = Path(get_data_path("Franka/Panda/Panda.urdf"))
    hand = Path(get_data_path("Robotiq2F85/Robotiq2F85.urdf"))
    unit, arm_paths, _, flange_paths, protected, diagnostics, _ = _build_paths(
        args, arm
    )
    calibration = GripperPositionCalibration(
        **json.loads(args.gripper_calibration.read_text())
    )
    open_driver = float(
        np.interp(
            calibration.max_width_m,
            calibration.widths_m,
            calibration.driver_positions_rad,
        )
    )
    driver_range = abs(
        calibration.driver_positions_rad[-1] - calibration.driver_positions_rad[0]
    )
    if not 0 < args.release_open_tolerance < driver_range / 2:
        raise ValueError(
            "Release tolerance must be smaller than half the calibrated driver range"
        )
    with np.load(source, allow_pickle=False) as raw:
        scalar = raw["source_gripper_position"].copy()
    if len(scalar) != len(unit.source.positions) or len(scalar) <= 160:
        raise ValueError(
            "This recording profile requires aligned source samples 105 and 160"
        )
    driver = np.interp(
        unit.time_map.source_query_times,
        unit.source.seconds,
        calibration.decode(scalar),
    )
    # Measurement positions are not force commands: explicitly record a local
    # squeeze preload while closure is nonzero; all three candidates share it.
    driver_targets = driver + args.grip_preload * (driver > 0.03)
    driver_targets = np.minimum(driver_targets, 0.725)
    contact_index = int(
        np.argmin(np.abs(unit.source.seconds - unit.source.seconds[105]))
    )
    center, yaw, _ = _model_geometry(
        arm,
        hand,
        unit.source.positions[contact_index],
        float(calibration.decode(scalar)[105]),
    )
    center[2] += args.base_height
    width = calibration.max_width_m * (1 - float(scalar[105]))
    block_size = (width, 0.05, 0.045)
    table_z = float(center[2] - block_size[2] / 2)
    bowl_center, _, _ = _model_geometry(
        arm, hand, unit.source.positions[160], float(calibration.decode(scalar)[160])
    )
    bowl_center[2] = table_z
    bowl_radius = 0.085
    joint_tree = [
        j for j in ET.parse(hand).getroot().findall("joint") if j.get("type") != "fixed"
    ]
    arm_stiff = {name: 1e4 for name in ARM_NAMES}
    arm_damp = {name: 600.0 for name in ARM_NAMES}
    arm_effort = {name: (87.0 if i < 4 else 12.0) for i, name in enumerate(ARM_NAMES)}
    for j in joint_tree:
        name = j.get("name")
        active = name == "finger_joint"
        arm_stiff[name] = args.grip_stiffness if active else 0.0
        arm_damp[name] = 8.0 if active else 0.0
        arm_effort[name] = args.grip_effort if active else 100.0
    assembly = URDFCfg(
        components=[
            {"component_type": "arm", "urdf_path": str(arm), "transform": np.eye(4)},
            {"component_type": "hand", "urdf_path": str(hand), "transform": np.eye(4)},
        ],
        fname="physics_franka_robotiq",
        fpath_prefix=str(output / "robot"),
    )
    cfg = RobotCfg(
        uid="franka",
        urdf_cfg=assembly,
        init_pos=(0.0, 0.0, args.base_height),
        control_parts={"arm": list(ARM_NAMES), "eef": ["finger_joint"]},
        root_props=ArticulationRootPropertiesCfg(
            fixed_base=True,
            self_collision_enabled=False,
            min_position_iters=16,
            min_velocity_iters=4,
        ),
        attrs=_physics(args.friction),
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="force",
            target_mode="position",
            stiffness=arm_stiff,
            damping=arm_damp,
            max_effort=arm_effort,
        ),
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=1,
            physics_cfg=DefaultPhysicsCfg(physics_dt=PHYSICS_DT, enable_ccd=True),
            render_cfg=RenderCfg(
                renderer="no-render" if args.no_render else "hybrid",
                tone_mapping_enabled=True,
            ),
            robot_ik_gizmo=None,
            startup_summary="off",
            visualization=visualization_cfg_from_args(args),
        )
    )
    all_results = []
    try:
        if not args.no_render:
            sim.set_emission_light([0.92, 0.95, 1.0], 30.0)
            sim.set_ground_plane_visibility(False)
            sim.add_light(
                LightCfg(
                    uid="key",
                    light_type="direction",
                    intensity=8.0,
                    direction=(-0.5, 0.4, -1.0),
                )
            )
        _box(
            sim,
            "table",
            (1.3, 1.0, 0.04),
            (0.35, 0, table_z - 0.02),
            (0.18, 0.30, 0.34),
            friction=args.friction,
        )
        _box(
            sim,
            "bowl_floor",
            (0.18, 0.18, 0.008),
            (bowl_center[0], bowl_center[1], table_z + 0.004),
            (0.8, 0.86, 0.9),
        )
        for i in range(24):
            angle = 2 * np.pi * i / 24
            _box(
                sim,
                f"bowl_wall_{i}",
                (0.008, 2 * bowl_radius * np.sin(np.pi / 24) + 0.001, 0.06),
                (
                    bowl_center[0] + bowl_radius * np.cos(angle),
                    bowl_center[1] + bowl_radius * np.sin(angle),
                    table_z + 0.03,
                ),
                (0.8, 0.86, 0.9),
                yaw=angle,
            )
        block = _box(
            sim,
            "block",
            block_size,
            center,
            (0.95, 0.72, 0.05),
            yaw=yaw,
            dynamic=True,
            mass=args.mass,
            friction=args.friction,
        )
        robot = sim.add_robot(cfg)
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="block_contacts",
                rigid_uid_list=["block"],
                filter_need_both_actor=False,
                max_contacts_per_env=128,
            )
        )
        cameras = []
        if not args.no_render:
            for uid, eye, target, fx in [
                ("overview", (1.20, -1.45, 1.0), (0.32, 0, 0.30), 620.0),
                ("contact_view", (0.86, -0.76, 0.52), (0.50, -0.035, 0.045), 620.0),
            ]:
                cameras.append(
                    sim.add_sensor(
                        CameraCfg(
                            uid=uid,
                            width=640,
                            height=360,
                            intrinsics=(fx, fx, 320.0, 180.0),
                            extrinsics=CameraCfg.ExtrinsicsCfg(
                                eye=(eye[0], eye[1], eye[2] + args.base_height),
                                target=(
                                    target[0],
                                    target[1],
                                    target[2] + args.base_height,
                                ),
                                up=(0.0, 0.0, 1.0),
                            ),
                        )
                    )
                )
        sim.prepare()
        active_names = (*ARM_NAMES, "finger_joint")
        active_ids = [robot.joint_names.index(n) for n in active_names]
        target_paths = np.concatenate(
            [
                arm_paths,
                np.broadcast_to(
                    driver_targets[None, :, None], (3, len(driver_targets), 1)
                ),
            ],
            axis=2,
        )
        settle_count = 150
        target_paths = np.concatenate(
            [target_paths, np.repeat(target_paths[:, -1:, :], settle_count, axis=1)],
            axis=1,
        )
        dt = np.r_[0.0, np.full(target_paths.shape[1] - 1, CONTROL_DT)]
        settings = {
            "source_revision": unit.revision,
            "source_input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "normalization": unit.report_dict(),
            "gripper_calibration": json.loads(args.gripper_calibration.read_text()),
            "physics_dt": PHYSICS_DT,
            "control_dt": CONTROL_DT,
            "full_joint_names": robot.joint_names,
            "active_joint_names": active_names,
            "active_joint_ids": active_ids,
            "geometry_estimated": True,
            "robot_base_height": args.base_height,
            "table_top_z": table_z,
            "block_initial_xyz": center.tolist(),
            "block_yaw": yaw,
            "block_size": block_size,
            "bowl_center": bowl_center.tolist(),
            "bowl_radius": bowl_radius,
            "mass": args.mass,
            "friction": args.friction,
            "grip_preload_rad": args.grip_preload,
            "grip_effort_nm": args.grip_effort,
            "grip_stiffness": args.grip_stiffness,
            "object_state_writes": "reset only",
            "attachments": False,
            "robot_control": "PD joint targets through shared play_joint_trajectory",
            "calibration_status": "unverified; estimated scene and hardware mapping",
            "source_diagnostics": diagnostics,
            "acceptance": {
                "minimum_lift_m": 0.04,
                "continuous_bilateral_hold_s": 1.0,
                "final_stability_s": 1.0,
                "minimum_trailing_release_s": 1.0,
                "release_open_driver_rad": open_driver,
                "release_open_tolerance_rad": args.release_open_tolerance,
                "release_evidence": "measured open driver and no bilateral grasp contact",
                "final_position_spread_m": 0.005,
                "final_speed_limit_m_s": 0.03,
                "final_angle_limit_rad": float(np.deg2rad(3)),
                "final_angular_speed_rad_s": 0.2,
                "containment": "all oriented box corners above container floor and inside rim",
            },
        }
        (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        np.savez_compressed(
            output / "candidate_targets.npz",
            targets=target_paths,
            dt=dt,
            raw_driver=driver,
            command_driver=driver_targets,
            protected_mask=protected,
        )
        for candidate in args.cases:
            for repeat in range(args.repeats):
                sim.reset_objects_state()
                initial = torch.zeros((1, len(robot.joint_names)), dtype=torch.float32)
                initial[:, active_ids] = torch.from_numpy(
                    target_paths[candidate, 0]
                ).float()
                for joint in joint_tree:
                    mimic = joint.find("mimic")
                    if mimic is not None:
                        initial[
                            :, robot.joint_names.index(joint.get("name"))
                        ] = driver_targets[0] * float(
                            mimic.get("multiplier", "1")
                        ) + float(
                            mimic.get("offset", "0")
                        )
                robot.set_state(
                    qpos=initial,
                    target_qpos=initial,
                    qvel=torch.zeros_like(initial),
                    target_qvel=torch.zeros_like(initial),
                    clear_dynamics=True,
                )
                for _ in range(60):
                    sim.update(PHYSICS_DT, 10, render_final_step=False)
                initial_z = float(block.get_local_pose()[0, 2])
                writer = None
                label = ["NOMINAL", "VARIANT A / seed 7", "VARIANT B / seed 11"][
                    candidate
                ]
                if not args.no_render and repeat == 0:
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
                            "640x450",
                            "-r",
                            str(args.fps),
                            "-i",
                            "pipe:0",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-crf",
                            "20",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(output / f"candidate_{candidate}.mp4"),
                        ],
                        stdin=subprocess.PIPE,
                    )
                observer = _ObservedPlayback(
                    sim,
                    robot,
                    block,
                    sensor,
                    target_paths[candidate],
                    initial_z,
                    bowl_center,
                    table_z,
                    cameras,
                    writer,
                    args.fps,
                    label,
                    open_driver_rad=open_driver,
                    release_tolerance_rad=args.release_open_tolerance,
                )
                observer.active_ids = active_ids
                try:
                    play_joint_trajectory(
                        observer,
                        robot,
                        positions=torch.from_numpy(
                            target_paths[candidate : candidate + 1]
                        ).float(),
                        dt=torch.from_numpy(dt[None]).float(),
                        joint_ids=active_ids,
                        cfg=JointTrajectoryPlaybackCfg(control_dt=CONTROL_DT),
                    )
                finally:
                    if writer is not None:
                        writer.stdin.close()
                        if writer.wait() != 0:
                            raise RuntimeError("Physical preview encoder failed")
                result = _summary(observer, block_size, bowl_radius)
                result.update(candidate=candidate, repeat=repeat, physics_stepped=True)
                all_results.append(result)
                np.savez_compressed(
                    output / f"trace_{candidate}_{repeat}.npz",
                    t=[s["t"] for s in observer.samples],
                    object_pose=[s["pose"] for s in observer.samples],
                    flange=[s["flange"] for s in observer.samples],
                    qpos=[s["qpos"] for s in observer.samples],
                    target=[s["target"] for s in observer.samples],
                    bilateral=[s["bilateral"] for s in observer.samples],
                    left=[s["left"] for s in observer.samples],
                    right=[s["right"] for s in observer.samples],
                    contacts=[s["contact_count"] for s in observer.samples],
                    impulse=[s["impulse"] for s in observer.samples],
                )
                (output / "results.json").write_text(
                    json.dumps(all_results, indent=2) + "\n"
                )
                (output / "contact_actors.json").write_text(
                    json.dumps(observer.actor_names, indent=2) + "\n"
                )
                print(json.dumps(result), flush=True)
    finally:
        sim.destroy(exit_process=False)
    del robot, block, sensor, cameras, sim
    if (
        not args.no_render
        and args.source_video is not None
        and set(args.cases) == {0, 1, 2}
    ):
        _compose(args, output, unit, all_results)


def main() -> None:
    """Run bounded B=1 physical comparisons using estimated foreground geometry."""
    p = argparse.ArgumentParser(description=__doc__)
    add_viser_args_to_parser(p)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-video", type=Path)
    p.add_argument("--gripper-calibration", type=Path, required=True)
    p.add_argument("--free-stop", type=int, default=60)
    p.add_argument("--scale", type=float, default=0.025)
    p.add_argument("--base-height", type=float, default=0.75)
    p.add_argument("--mass", type=float, default=0.08)
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--grip-preload", type=float, default=0.015)
    p.add_argument("--grip-effort", type=float, default=2.0)
    p.add_argument("--grip-stiffness", type=float, default=600.0)
    p.add_argument("--release-open-tolerance", type=float, default=0.03)
    p.add_argument("--cases", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    physical_values = (
        args.mass,
        args.friction,
        args.grip_preload,
        args.grip_effort,
        args.grip_stiffness,
        args.base_height,
        args.release_open_tolerance,
    )
    if (
        not all(math.isfinite(value) for value in physical_values)
        or args.repeats < 1
        or args.fps < 1
        or args.mass <= 0
        or args.friction < 0
        or args.grip_preload < 0
        or args.grip_effort <= 0
        or args.grip_stiffness <= 0
        or args.release_open_tolerance <= 0
        or args.base_height < 0.1
    ):
        p.error("Invalid physical parameters")
    try:
        _run(args)
    finally:
        # _run has unwound its observer/camera/robot references before native cleanup.
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
