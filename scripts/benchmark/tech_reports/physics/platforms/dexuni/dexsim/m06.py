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
"""Preview a physical, kinematic Franka T-shirt fold from left to right.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m06.py
This is a scene preview, not a performance measurement or platform comparison.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

__all__ = ["main"]

TECH = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold"
)
CONFIG = TECH / "assets/physics/dexuni/m06/m06_config.json"
JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8)) + (
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)


def write_json(path: Path, value: Any) -> None:
    """Write human-readable preview evidence."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    """Hash a source or frozen input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(position: Any) -> np.ndarray:
    """Make a world matrix in metres."""
    value = np.eye(4, dtype=np.float32)
    value[:3, 3] = position
    return value


def prepare_mesh(config: dict[str, Any], output: Path) -> dict[str, np.ndarray]:
    """Preserve the reference garment topology, including its front and back."""
    vertices, triangles = [], []
    section = ""
    for line in Path(config["cloth"]["mesh_source"]).read_text().splitlines():
        if line in ("shirt_vertices", "shirt_indices"):
            section = line
        elif ":" in line:
            values = line.split(":", 1)[1].split()[:3]
            if section == "shirt_vertices":
                vertices.append([float(x) for x in values])
            elif section == "shirt_indices":
                triangles.append([int(x) for x in values])
    local = np.asarray(vertices, np.float32) * config["cloth"]["scale"]
    local -= (local.min(axis=0) + local.max(axis=0)) * 0.5
    local[:, :2] *= -1
    faces = np.asarray(triangles, np.int32)
    world = local + np.asarray(config["cloth"]["center"], np.float32)
    uv = local[:, :2].copy()
    uv = (uv - uv.min(0)) / np.maximum(np.ptp(uv, axis=0), 1e-8) * 2
    left = np.flatnonzero(world[:, 0] < world[:, 0].min() + 0.018)
    right = np.flatnonzero(world[:, 0] > world[:, 0].max() - 0.018)
    torso = np.flatnonzero(
        (np.abs(world[:, 0]) < 0.06) & (np.abs(world[:, 1] + 0.5) < 0.12)
    )
    cross = np.cross(
        local[faces[:, 1]] - local[faces[:, 0]], local[faces[:, 2]] - local[faces[:, 0]]
    )
    normals = np.zeros_like(local)
    for column in range(3):
        np.add.at(normals, faces[:, column], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    arrays = dict(
        vertices=local,
        triangles=faces,
        uv=uv.astype(np.float32),
        normals=normals,
        initial_positions=world,
        left_ids=left,
        right_ids=right,
        torso_ids=torso,
    )
    np.savez_compressed(output / "inputs/cloth_mesh.npz", **arrays)
    return arrays


def motion_keys(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Author one grasp, one fold and one release with a clear retreat."""
    motion, robot = config["motion"], config["robot"]
    park = np.asarray(motion["park_tcp"])
    grip = np.asarray(motion["left_grasp"])
    placed = np.asarray(motion["left_place"])
    above = grip + [0, 0, motion["lift_height"]]
    over = placed.copy()
    over[2] = max(above[2], placed[2] + 0.04)
    retreat = placed + [0, 0, 0.10]
    opened, closed = robot["open_finger"], robot["closed_finger"]
    keys = [
        (0.0, park, opened, "initial"),
        (motion["settle_seconds"], park, opened, "settle"),
        (1.5, above, opened, "approach left sleeve"),
        (1.5, grip, opened, "descend"),
        (0.8, grip, closed, "close gripper"),
        (1.3, above, closed, "lift sleeve"),
        (2.0, over, closed, "fold left to right"),
        (1.2, placed, closed, "lower folded sleeve"),
        (0.8, placed, opened, "release"),
        (0.8, retreat, opened, "retreat vertically"),
        (1.2, park, opened, "retreat"),
        (motion["observe_seconds"], park, opened, "observe"),
    ]
    return (
        np.cumsum([k[0] for k in keys]),
        np.stack([k[1] for k in keys]),
        np.asarray([k[2] for k in keys]),
        [k[3] for k in keys],
    )


def prepare_trajectory(
    config: dict[str, Any], output: Path, device: str
) -> dict[str, np.ndarray]:
    """Use DexSim IK, then retime its sampled path to the fixed command grid."""
    import warp as wp
    from scipy.spatial.transform import Rotation
    from dexsim.kit.ik import (
        IKSolver,
        JointLimitTarget,
        KinematicModel,
        PositionTarget,
        RotationTarget,
    )

    robot = config["robot"]
    model = KinematicModel.from_urdf(
        str(output / "inputs/franka/fr3_franka_hand.dexsim.urdf"),
        floating=False,
        hide_visuals=True,
        collapse_fixed_joints=False,
    )
    starts, dofs = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    coords = {
        label.rsplit("/", 1)[-1]: int(starts[j])
        for j, label in enumerate(model.joint_label)
        if starts[j + 1] > starts[j]
    }
    body = next(
        i
        for i, label in enumerate(model.body_label)
        if label.endswith("/" + robot["tcp_body"])
    )
    offset = np.asarray(robot["tcp_offset"])
    target_rotation = Rotation.from_quat(robot["tcp_rotation_xyzw"])
    position = PositionTarget(
        body,
        wp.vec3(*offset),
        wp.array([wp.vec3(0, 0, 0.5)], dtype=wp.vec3, device=device),
    )
    rotation = RotationTarget(
        body,
        wp.quat_identity(),
        wp.array([wp.vec4(*robot["tcp_rotation_xyzw"])], dtype=wp.vec4, device=device),
    )
    limits = JointLimitTarget(
        model.joint_limit_lower, model.joint_limit_upper, weight=10.0
    )
    mask = np.zeros(model.joint_dof_count, bool)
    for j, label in enumerate(model.joint_label):
        if label.rsplit("/", 1)[-1] in JOINTS[:7]:
            mask[int(dofs[j]) : int(dofs[j + 1])] = True
    solver = IKSolver(
        model, 1, [position, rotation, limits], lambda_initial=0.03, joint_dof_mask=mask
    )
    row = model.joint_q.numpy().astype(np.float32)
    for name, value in zip(JOINTS[:7], robot["seed_q"]):
        row[coords[name]] = value
    q = wp.array(row[None, :], device=device)
    times, points, fingers, phases = motion_keys(config)
    base = np.asarray(robot["base_position"])
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    samples, errors, rotation_errors = [], [], []
    state = model.state()
    for segment in range(len(times) - 1):
        distance = float(np.linalg.norm(points[segment + 1] - points[segment]))
        count = max(2, int(np.ceil(distance / 0.006)) + 1)
        rows = []
        for fraction in np.linspace(0, 1, count):
            point = (1 - fraction) * points[segment] + fraction * points[segment + 1]
            position.set_target_position(0, wp.vec3(*(point - base)))
            solver.step(q, q, iterations=180 if not samples and not rows else 40)
            result = q.numpy()[0]
            if np.any(result < lower - 1e-5) or np.any(result > upper + 1e-5):
                raise RuntimeError("IK violates joint limits")
            model.forward_kinematics(wp.array(result, device=device), state)
            body_pose = state.body_q.numpy()[body]
            achieved_rotation = Rotation.from_quat(body_pose[3:7])
            achieved = body_pose[:3] + achieved_rotation.apply(offset) + base
            errors.append(float(np.linalg.norm(achieved - point)))
            rotation_errors.append(
                float((achieved_rotation * target_rotation.inv()).magnitude())
            )
            rows.append([result[coords[name]] for name in JOINTS[:7]])
        samples.append(np.asarray(rows))
    if max(errors) > 0.0005 or max(rotation_errors) > 0.003:
        raise RuntimeError(
            f"IK error: {max(errors):.6f} m, {max(rotation_errors):.6f} rad"
        )
    frame_times = (
        np.arange(round(times[-1] / config["solver"]["dt"]) + 1)
        * config["solver"]["dt"]
    )
    targets, desired, phase_ids = [], [], []
    for time in frame_times:
        segment = min(
            int(np.searchsorted(times, time, side="right")) - 1, len(times) - 2
        )
        u = np.clip(
            (time - times[segment]) / (times[segment + 1] - times[segment]), 0, 1
        )
        smooth = u**3 * (10 - 15 * u + 6 * u * u)
        source = samples[segment]
        arm = [
            np.interp(smooth, np.linspace(0, 1, len(source)), col) for col in source.T
        ]
        finger = (1 - smooth) * fingers[segment] + smooth * fingers[segment + 1]
        targets.append([*arm, finger, finger])
        desired.append((1 - smooth) * points[segment] + smooth * points[segment + 1])
        phase_ids.append(segment + 1)
    arrays = dict(
        joint_targets=np.asarray(targets, np.float32),
        time=frame_times,
        desired_tcp=np.asarray(desired, np.float32),
        phase_ids=np.asarray(phase_ids),
        phase_names=np.asarray(phases),
        key_times=times,
        key_tcp=points,
        ik_position_errors=np.asarray(errors),
        ik_rotation_errors=np.asarray(rotation_errors),
        joint_names=np.asarray(JOINTS),
    )
    np.savez_compressed(output / "inputs/trajectory.npz", **arrays)
    print(
        f"IK ready: {len(targets)} frames; error={max(errors)*1000:.4f} mm", flush=True
    )
    return arrays


def add_scene(
    world: Any,
    config: dict[str, Any],
    output: Path,
    mesh: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    video: bool,
) -> Any:
    """Declare all resources and controls through the spawn facade."""
    import dexsim
    from dexsim.engine.newton_physics import (
        ContactMaterialSchedule,
        KinematicJointTrajectoryControl,
    )

    builder = dexsim.spawn.SceneBuilder(world)
    builder.materials.update(
        {
            "shirt": dexsim.spawn.MaterialDesc(
                name="m06_shirt",
                base_color=(0.85, 0.23, 0.16, 1),
                base_color_map=str(output / "inputs/mianbu.png"),
                roughness=0.85,
                double_sided=True,
            ),
            "table": dexsim.spawn.MaterialDesc(
                name="m06_table", base_color=(0.42, 0.48, 0.52, 1), roughness=0.9
            ),
        }
    )
    contact = config["contact"]
    collision = dexsim.spawn.NewtonCollisionDesc(
        ke=contact["shape_ke"], kd=contact["shape_kd"], mu=contact["table_friction"]
    )
    for name, center, size in (
        ("table", (0, -0.5, 0.10), (0.82, 0.82, 0.20)),
        ("robot_stand", (-0.52, -0.5, 0.06), (0.22, 0.22, 0.12)),
    ):
        geometry = dexsim.spawn.GeometryDesc.cube(size)
        shape_collision = dexsim.spawn.CollisionDesc.from_geometry(geometry)
        shape_collision.newton = collision
        builder.add_object(
            dexsim.spawn.ObjectDesc(
                name=name,
                pose=pose(center),
                per_env=False,
                renders=[
                    dexsim.spawn.RenderDesc.from_geometry(
                        geometry, material_ref="table"
                    )
                ],
                collisions=[shape_collision],
                physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
            )
        )
    builder.add_plane(
        "ground", 20, physics=dexsim.spawn.RigidBodyPhysicsDesc.static(), per_env=False
    )
    builder.add_urdf(
        "franka",
        str(output / "inputs/franka/fr3_franka_hand.dexsim.urdf"),
        pose=pose(config["robot"]["base_position"]),
        fix_root_link=True,
        per_env=False,
        newton_collision=collision,
    )
    cloth = config["cloth"]
    geometry = dexsim.spawn.GeometryDesc.mesh(
        vertices=mesh["vertices"],
        triangles=mesh["triangles"],
        normals=mesh["normals"],
        uv_coords=mesh["uv"],
    )
    builder.add_clothbody(
        "shirt",
        dexsim.spawn.RenderDesc.from_geometry(geometry, material_ref="shirt"),
        pose=pose(cloth["center"]),
        particle_radius=cloth["particle_radius"],
        per_env=False,
        physics=dexsim.spawn.ClothPhysicsDesc(
            surface_density=cloth["density"],
            tri_ke=cloth["tri_ke"],
            tri_ka=cloth["tri_ka"],
            tri_kd=cloth["tri_kd"],
            edge_ke=cloth["edge_ke"],
            edge_kd=cloth["edge_kd"],
            edge_rest_angle=cloth.get("edge_rest_angle"),
            contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                stiffness=contact["particle_ke"],
                damping=contact["particle_kd"],
                dynamic_friction=contact["particle_friction"],
            ),
        ),
    )
    builder.add_runtime_control(
        KinematicJointTrajectoryControl(
            "franka",
            trajectory["joint_targets"],
            fps=1 / config["solver"]["dt"],
            joint_names=JOINTS,
            initialize_pose=True,
        )
    )
    builder.add_runtime_control(
        ContactMaterialSchedule(
            "franka",
            {
                "dynamic_friction": ((0.0, contact["gripper_friction"]),),
                "stiffness": ((0.0, contact["gripper_ke"]),),
                "damping": ((0.0, contact["gripper_kd"]),),
            },
            link_names=("fr3_hand", "fr3_leftfinger", "fr3_rightfinger"),
        )
    )
    if video:
        builder.add_light(
            dexsim.spawn.LightDesc(
                name="key_light",
                light_type=dexsim.render.LightType.POINT,
                pose=pose((1, -0.2, 3)),
                intensity=70,
                shadow=False,
            )
        )
        for name, resolution in (("perspective", (960, 720)), ("overhead", (720, 720))):
            builder.add_camera(
                dexsim.spawn.CameraDesc(
                    name, resolution=resolution, near=0.01, far=10, headless=True
                )
            )
    return builder.finalize()


def observations(
    states: dict[str, np.ndarray], mesh: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Describe fixed material groups without substituting gripper motion."""
    times, particles = states["time"], states["particles"]
    start = int(np.argmin(np.abs(times - 2.0)))
    left, right = particles[:, mesh["left_ids"]].mean(1), particles[
        :, mesh["right_ids"]
    ].mean(1)
    return {
        "all_states_finite": all(
            bool(np.isfinite(value).all()) for value in states.values()
        ),
        "recorded_rows": len(times),
        "actual_end_time": float(times[-1]),
        "left_sleeve_start_xyz": left[start].tolist(),
        "left_sleeve_end_xyz": left[-1].tolist(),
        "left_sleeve_peak_lift_m": float(left[start:, 2].max() - left[start, 2]),
        "left_sleeve_final_rightward_motion_m": float(left[-1, 0] - left[start, 0]),
        "right_sleeve_final_motion_m": float(np.linalg.norm(right[-1] - right[start])),
        "minimum_node_z": float(particles[..., 2].min()),
        "geometry_only": True,
        "user_scene_approved": False,
    }


def write_report(
    output: Path, observation: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Write one three-table report, explicitly leaving performance unmeasured."""
    text = f"""# M06 DexSim 场景预览

Franka 从左往右折一次 T 恤。运动学关节，原始衣服拓扑；没有固定或附着节点。对齐不是验收要求。当前视频等待用户观看。

## Time & Memory

| 平台 | 性能耗时 | 显存 | 范围 |
| --- | --- | --- | --- |
| DexSim / DexUni | 未测量 | 未测量 | 独立场景预览 |

## Success & Other Metrics

| 有限状态 | 实际时长 | 左袖峰值抬升 | 左袖最终向右位移 | 人工确认 |
| --- | --- | --- | --- | --- |
| {observation['all_states_finite']} | {observation['actual_end_time']:.3f} s | {observation['left_sleeve_peak_lift_m']*1000:.1f} mm | {observation['left_sleeve_final_rightward_motion_m']*1000:.1f} mm | 待用户观看 |

## Leaderboard

| 平台 | 结论 |
| --- | --- |
| DexSim / DexUni | 未做性能排名；仅原生预览 |

[视频](preview/dexsim.mp4) · [实际状态](preview/states.npz) · [配置](inputs/config.json) · [运行信息](manifest.json)

袖口指标使用创建时固定的材料节点组。它们不是完整折叠成功率或布面无贯穿证明。CUDA Graph 实际状态见 manifest 的 effective 字段。采样帧与实际仿真时间见 preview/frame_times.json。
"""
    (output / "REPORT.md").write_text(text)


def run_preview(
    config: dict[str, Any],
    output: Path,
    mesh: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    """Run one finite trajectory, optionally recording two synchronized views."""
    import dexsim
    import newton
    import warp as wp
    from dexsim.engine.newton_physics import DexUniSolverCfg, NewtonCfg

    solver = config["solver"]
    world_cfg = dexsim.WorldConfig()
    world_cfg.use_default_physics = False
    world_cfg.open_windows = False
    world_cfg.renderer = dexsim.types.Renderer.OFFLINERT
    world_cfg.raytrace_config.spp = 1
    world_cfg.newton_cfg = NewtonCfg(
        device=args.device,
        dt=solver["dt"],
        num_substeps=solver["num_substeps"],
        solver_cfg=DexUniSolverCfg(
            joint_mode="kinematic",
            vbd_options={
                "iterations": solver["vbd_iterations"],
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": solver["self_contact_radius"],
                "particle_self_contact_margin": solver["self_contact_margin"],
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.005,
                "particle_vertex_contact_buffer_size": 32,
                "particle_edge_contact_buffer_size": 40,
                "particle_collision_detection_interval": -1,
            },
            collision_options={
                "soft_contact_margin": config["contact"]["soft_contact_margin"]
            },
        ),
    )
    world_cfg.newton_cfg.use_cuda_graph = solver["use_cuda_graph"]
    manifest = {
        "task": "M06",
        "status": "running",
        "scope": "native_scene_preview_only",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "user_scene_approved": False,
        "other_platforms_implemented": False,
        "performance_measurements_started": False,
        "python": sys.executable,
        "versions": {
            "dexsim": dexsim.__version__,
            "newton": newton.__version__,
            "warp": wp.__version__,
        },
        "config_sha256": digest(output / "inputs/config.json"),
        "source_sha256": digest(Path(__file__)),
        "requested": solver,
    }
    write_json(output / "manifest.json", manifest)
    world = dexsim.World(world_cfg)
    world.set_manual_update(True)
    scene, cameras = None, []
    robot, shirt = None, None
    rows: dict[str, list[np.ndarray]] = {
        key: [] for key in ("time", "particles", "qpos", "tcp")
    }
    frame_times = []
    last = len(trajectory["time"]) - 1
    if args.steps is not None:
        last = min(last, args.steps)
    try:
        scene = add_scene(world, config, output, mesh, trajectory, not args.no_video)
        robot, shirt = scene.get_articulation("franka"), scene.get_cloth_particle_set(
            "shirt"
        )
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont

            for name, eye, target, up in (
                ("perspective", (1.25, 0.65, 1.35), (-0.15, -0.5, 0.36), (0, 0, 1)),
                ("overhead", (0, -0.5, 1.40), (0, -0.5, 0.20), (0, 1, 0)),
            ):
                camera = scene.get_camera(name)
                width = 960 if name == "perspective" else 720
                camera.set_perspective(48.0, width / 720, 0.01, 10)
                camera.look_at(eye, target, up)
                camera.open_camera()
                cameras.append(camera)
            scene.set_environment_light_intensity(70)
            font = ImageFont.truetype("DejaVuSans.ttf", 21)
        with ExitStack() as stack:
            writer = None
            if cameras:
                writer = stack.enter_context(
                    imageio.get_writer(
                        str(output / "preview/dexsim.mp4"),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        macro_block_size=2,
                    )
                )
            for frame in range(last + 1):
                capture = bool(cameras) and frame > 0 and frame % 2 == 0
                if frame:
                    world.update(solver["dt"], sync_to_dexsim=capture)
                particles = shirt.get_particle_positions().numpy()
                qpos = robot.get_current_qpos()
                matrix = robot.get_link_pose(config["robot"]["tcp_body"])
                tcp = matrix[:3, 3] + matrix[:3, :3] @ np.asarray(
                    config["robot"]["tcp_offset"]
                )
                if not np.isfinite(particles).all() or not np.isfinite(qpos).all():
                    raise RuntimeError(f"Nonfinite state at frame {frame}")
                if frame % 6 == 0 or frame == last:
                    for key, value in (
                        ("time", scene.get_simulation_time()),
                        ("particles", particles),
                        ("qpos", qpos),
                        ("tcp", tcp),
                    ):
                        rows[key].append(np.array(value, copy=True))
                if capture:
                    views = []
                    for camera in cameras:
                        camera.render()
                        views.append(np.asarray(camera.get_rgb_map())[..., :3].copy())
                    panel = Image.fromarray(np.concatenate(views, axis=1))
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, 1680, 45), fill="black")
                    phase = str(
                        trajectory["phase_names"][trajectory["phase_ids"][frame]]
                    )
                    draw.text(
                        (12, 11),
                        f"DexSim | M06 Franka T-shirt fold LEFT -> RIGHT | t={scene.get_simulation_time():.2f}s | {phase}",
                        font=font,
                        fill="white",
                    )
                    writer.append_data(np.asarray(panel))
                    frame_times.append(float(scene.get_simulation_time()))
                    if frame == 2 or frame == last - last % 2:
                        panel.save(
                            output
                            / f"preview/{'first' if frame == 2 else 'last'}_frame.jpg"
                        )
                if frame % 120 == 0:
                    left = particles[mesh["left_ids"]].mean(0)
                    print(
                        f"frame={frame}/{last} t={scene.get_simulation_time():.2f} sleeve={left.round(4)}",
                        flush=True,
                    )
        states = {key: np.stack(value) for key, value in rows.items()}
        observation = observations(states, mesh)
        write_json(output / "preview/observations.json", observation)
        write_json(
            output / "preview/frame_times.json", {"fps": 30, "actual_time": frame_times}
        )
        manifest.update(
            status="completed",
            effective=scene.get_solver_diagnostics(),
            observations=observation,
            full_trajectory=last == len(trajectory["time"]) - 1,
            video_frames=len(frame_times),
        )
        write_report(output, observation, manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if rows["time"]:
            np.savez_compressed(
                output / "preview/states.npz",
                **{key: np.stack(value) for key, value in rows.items()},
            )
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        for camera in cameras:
            camera.close_camera()
        cameras.clear()
        robot, shirt = None, None
        if scene is not None:
            scene.close()


def main() -> None:
    """Prepare frozen inputs and generate the requested native preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-inputs", action="store_true")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni" not in output.parents:
        raise ValueError(
            "Preview outputs must be inside outputs/benchmarks/tech_reports/dexuni"
        )
    for name in ("inputs", "source", "preview"):
        (output / name).mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    if not args.reuse_inputs:
        write_json(output / "inputs/config.json", config)
        if not (output / "inputs/franka").exists():
            shutil.copytree(
                Path(config["robot"]["urdf"]).parent, output / "inputs/franka"
            )
        shutil.copy2(config["cloth"]["texture_source"], output / "inputs/mianbu.png")
        mesh = prepare_mesh(config, output)
        trajectory = prepare_trajectory(config, output, args.device)
    else:
        config = json.loads((output / "inputs/config.json").read_text())
        with np.load(output / "inputs/cloth_mesh.npz") as data:
            mesh = {key: data[key] for key in data.files}
        with np.load(output / "inputs/trajectory.npz") as data:
            trajectory = {key: data[key] for key in data.files}
    shutil.copy2(__file__, output / "source/m06.py")
    for path in (
        Path(
            "/home/oem/code/engine/dexsim/examples/python/physics/multiphys/demo_dexuni_franka_fold_tshirt.py"
        ),
        Path(
            "/home/oem/code/repos/newton/newton/examples/cloth/example_cloth_franka.py"
        ),
    ):
        shutil.copy2(path, output / "source" / path.name)
    write_json(
        output / "inputs/manifest.json",
        {
            "sha256": {
                str(path.relative_to(output / "inputs")): digest(path)
                for path in (output / "inputs").rglob("*")
                if path.is_file() and path.name != "manifest.json"
            }
        },
    )
    if not args.prepare_only:
        run_preview(config, output, mesh, trajectory, args)


if __name__ == "__main__":
    main()
