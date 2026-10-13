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
"""EmbodiChain pouring demo using the public fluid and rigid-boundary APIs.

The robot and mug replay the recorded MuJoCo grasp trajectory. Water is simulated
by DFSPH with volume-map boundaries, without reaction forces on the robot.
Volume maps use baked mesh SDFs; the additional SDF penalty force is disabled.
The manager owns fluid declaration, state binding and render publication.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from dexsim.engine.newton_physics import SPHSolverCfg

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    LightCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidObjectCfg,
    FluidObjectCfg,
    FluidSpawnCfg,
    FluidRenderCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.visualization import VisualizationCfg
from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg
from embodichain.lab.sim.sensors import CameraCfg

__all__ = ["main"]

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAJECTORY = Path(__file__).parent / "assets/franka_dfsph_pour/trajectory.pt"
CUP_ASSET = ROOT / (
    "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/assets/cup.glb"
)
RECEIVER_CENTER = np.array([-0.755, -0.137, 0.50])
BOWL_OUTER_RADIUS = 0.097
BOWL_INNER_RADIUS = 0.0915
BOWL_SPHERE_Z = 0.088
BOWL_HEIGHT = 0.066
BOWL_BOTTOM_Z = 0.006


def _pose(position: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = position
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def _add_boundary(
    sim: SimulationManager,
    name: str,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    *,
    resolution: int = 192,
    material: VisualMaterialCfg | None = None,
    visual_asset: Path | None = None,
    render_enabled: bool = True,
):
    if material is not None:
        material = material.copy()
        material.uid = f"{name}_material"
    return sim.add_rigid_object(
        RigidObjectCfg.from_dict(
            {
                "uid": name,
                "body_type": "kinematic",
                "init_local_pose": pose,
                "render_enabled": render_enabled,
                "shape": {
                    "shape_type": "Mesh",
                    "vertices": np.asarray(mesh.vertices, dtype=np.float32),
                    "triangles": np.asarray(mesh.faces, dtype=np.int32),
                    "visual_material": None if material is None else material.to_dict(),
                    "collision": {
                        "approximation": "sdf",
                        "force_sdf": False,
                        "sdf_resolution": resolution,
                        "sdf_narrow_band_range": [-0.025, 0.025],
                        "sdf_padding": 0.012,
                        "sdf_texture_format": "uint16",
                    },
                },
                "visual_shape": (
                    None
                    if visual_asset is None
                    else {"shape_type": "Mesh", "fpath": str(visual_asset)}
                ),
                "attrs": {
                    "collision_props": {
                        "backend": "newton",
                        "has_particle_collision": True,
                    },
                    "material_props": {"backend": "newton", "dynamic_friction": 0.2},
                },
            }
        )
    )


def _particle_summary(
    positions: np.ndarray,
    cup_pose: np.ndarray,
    spacing: float,
    velocities: np.ndarray,
) -> dict[str, float | int]:
    local = (positions - cup_pose[:3, 3]) @ cup_pose[:3, :3]
    radius = 0.031 + (local[:, 2] + 0.044) * (0.0075 / 0.094)
    in_mug = (
        (np.linalg.norm(local[:, :2], axis=1) < radius + 0.003)
        & (local[:, 2] > -0.049)
        & (local[:, 2] < 0.053)
    )
    receiver = positions - RECEIVER_CENTER
    bowl_radius = np.sqrt(
        np.maximum(0.0, BOWL_INNER_RADIUS**2 - (receiver[:, 2] - BOWL_SPHERE_Z) ** 2)
    )
    in_receiver = (
        (np.linalg.norm(receiver[:, :2], axis=1) < bowl_radius + spacing)
        & (receiver[:, 2] > BOWL_BOTTOM_Z - spacing)
        & (receiver[:, 2] < BOWL_HEIGHT + spacing)
    )
    summary = {
        "in_mug": int(in_mug.sum()),
        "in_receiver": int(in_receiver.sum()),
        "below_table": int((positions[:, 2] < 0.49).sum()),
        "outside_containers": int((~(in_mug | in_receiver)).sum()),
        "received_ml": float(in_receiver.sum() * spacing**3 * 1e6),
        "min_z": float(positions[:, 2].min()),
    }
    if in_receiver.any():
        bowl_speed = np.linalg.norm(velocities[in_receiver], axis=1)
        summary["bowl_speed_rms_m_s"] = float(np.sqrt(np.mean(bowl_speed**2)))
        summary["bowl_speed_p95_m_s"] = float(np.quantile(bowl_speed, 0.95))
        summary["bowl_height_p99_m"] = float(
            np.quantile(receiver[in_receiver, 2], 0.99)
        )
    return summary


def main() -> None:
    """Replay the arm and simulate water in the same EmbodiChain world."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/franka_pour_cup/dfsph_stable_bowl",
    )
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--spacing", type=float, default=0.0015)
    parser.add_argument("--volume-map-resolution", type=int, default=192)
    parser.add_argument("--viscosity", type=float, default=0.0006)
    parser.add_argument("--surface-tension", type=float, default=0.002)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--physics-hz", type=int, default=1440)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--renderer", choices=["rt", "hybrid"], default="rt")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if min(args.speed, args.spacing, args.fps, args.substeps, args.physics_hz) <= 0:
        parser.error("speed, spacing, fps, substeps and physics-hz must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.trajectory, map_location="cpu", weights_only=False)
    joints = data["states"]["robot", "qpos"][0].numpy()
    cup_states = data["states"]["rigid_objects", "cup", "pose"][0].numpy()
    recorded_times = np.arange(len(joints)) * float(data["meta"]["dt"])
    joint_curve = PchipInterpolator(recorded_times, joints)
    position_curve = PchipInterpolator(recorded_times, cup_states[:, :3])
    orientation_curve = Slerp(recorded_times, Rotation.from_quat(cup_states[:, 3:]))
    if not 0.0 <= args.start_time < recorded_times[-1]:
        parser.error("start-time must lie inside the recorded trajectory")
    duration = 1.0 + (recorded_times[-1] - args.start_time) / args.speed + 1.0
    if args.seconds is not None:
        duration = min(duration, args.seconds)
    # Update the moving boundary at every solver step. With coarse pose writes
    # and many solver substeps, DFSPH estimates an impulsive boundary velocity.
    dt = 1.0 / args.physics_hz
    frame_stride = round(1.0 / (dt * args.fps))
    if not np.isclose(frame_stride * dt * args.fps, 1.0):
        parser.error("fps must divide physics-hz")
    spacing = args.spacing
    grid = np.arange(-0.04, 0.0401, spacing)
    xx, yy, zz = np.meshgrid(grid, grid, np.arange(-0.036, 0.044, spacing))
    local_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside_radius = 0.031 + (local_points[:, 2] + 0.044) * (0.0075 / 0.094)
    local_points = local_points[
        np.linalg.norm(local_points[:, :2], axis=1) < inside_radius - 0.003
    ]
    initial_pose = _pose(
        position_curve(args.start_time), orientation_curve(args.start_time).as_matrix()
    )
    points = (local_points @ initial_pose[:3, :3].T + initial_pose[:3, 3]).astype(
        np.float32
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(),
            width=1280,
            height=960,
            num_envs=1,
            arena_space=2.5,
            device="cuda",
            physics_dt=dt,
            render_cfg=RenderCfg(renderer=args.renderer, spp=8),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=args.substeps,
                collision_cfg=None,
                solver_cfg=SPHSolverCfg(
                    sph_method="dfsph",
                    particle_diameter=spacing,
                    particle_support_radius=2.25 * spacing,
                    rest_density=1000.0,
                    viscosity=args.viscosity,
                    surface_tension_coefficient=args.surface_tension,
                    surface_normal_threshold=0.08,
                    pressure_relaxation=0.2,
                    divergence_max_iterations=10,
                    density_max_iterations=20,
                    max_neighbors=192,
                    max_velocity=2.0,
                    volume_map_boundary_enabled=True,
                    volume_map_resolution=args.volume_map_resolution,
                    volume_map_quadrature_samples=6,
                    volume_map_volume_scale=0.8,
                    sdf_boundary_enabled=False,
                ),
            ),
        )
    )
    writer = None
    close_writer = None
    try:
        robot_cfg = FrankaPandaCfg.from_dict(
            {
                "uid": "FrankaPanda",
                "init_rot": [0.0, 0.0, 180.0],
                "init_qpos": joint_curve(args.start_time).tolist(),
                "attrs": {
                    "collision_props": {
                        "backend": "newton",
                        "has_particle_collision": False,
                    }
                },
            }
        )
        robot = sim.add_robot(robot_cfg)
        # VisualMaterialCfg colors use sRGB; preserve the native demo
        # linear baseColor [0.88, 0.96, 1.0] after conversion.
        water_material = VisualMaterialCfg(
            base_color=[0.9452768, 0.982207, 1.0, 1.0],
            roughness=0.025,
            metallic=0.0,
            ior=1.333,
            transmission=0.96,
            volume_thickness=0.04,
            volume_attenuation_distance=0.6,
            volume_attenuation_color=(0.80, 0.92, 1.0),
        )
        _add_boundary(
            sim,
            "pour_table",
            trimesh.creation.box([0.42, 0.70, 0.50]),
            _pose(np.array([-0.91, -0.08, 0.25])),
            resolution=96,
            material=VisualMaterialCfg(
                base_color=[0.23, 0.29, 0.34, 1.0], roughness=0.6, metallic=0.5, ior=1.0
            ),
        )
        _add_boundary(
            sim,
            "fluid_floor",
            trimesh.creation.box([2.5, 2.5, 0.10]),
            _pose(np.array([-0.5, 0.0, -0.055])),
            resolution=128,
            render_enabled=False,
        )
        mug_mesh = trimesh.creation.revolve(
            [
                [0.0, -0.05],
                [0.035, -0.05],
                [0.0425, 0.05],
                [0.0385, 0.05],
                [0.031, -0.044],
                [0.0, -0.044],
            ],
            sections=128,
        )
        mug = _add_boundary(
            sim, "liquid_mug", mug_mesh, initial_pose, visual_asset=CUP_ASSET
        )
        if args.substeps > 1:
            boundary_times = (
                np.arange(round(duration / dt) * args.substeps) * dt / args.substeps
            )
            source_times = np.clip(
                args.start_time + (boundary_times - 1.0) * args.speed,
                args.start_time,
                recorded_times[-1],
            )
            boundary_poses = np.concatenate(
                (
                    position_curve(source_times),
                    orientation_curve(source_times).as_quat(),
                ),
                axis=1,
            )
            sim.register_kinematic_rigid_trajectory(
                "liquid_mug", boundary_poses[None], fps=args.physics_hz * args.substeps
            )
        outer_z = np.linspace(0.0, BOWL_HEIGHT, 64)
        inner_z = np.linspace(BOWL_HEIGHT, BOWL_BOTTOM_Z, 64)
        outer_r = np.sqrt(BOWL_OUTER_RADIUS**2 - (outer_z - BOWL_SPHERE_Z) ** 2)
        inner_r = np.sqrt(BOWL_INNER_RADIUS**2 - (inner_z - BOWL_SPHERE_Z) ** 2)
        bowl_profile = np.vstack(
            [
                [0.0, 0.0],
                np.column_stack([outer_r, outer_z]),
                np.column_stack([inner_r, inner_z]),
                [0.0, BOWL_BOTTOM_Z],
            ]
        )
        bowl_mesh = trimesh.creation.revolve(bowl_profile, sections=192)
        _add_boundary(
            sim,
            "receiver",
            bowl_mesh,
            _pose(RECEIVER_CENTER),
            material=VisualMaterialCfg(
                base_color=[0.82, 0.85, 0.88, 1.0],
                roughness=0.22,
                metallic=0.5,
                ior=1.0,
            ),
        )
        fluid = sim.add_fluid_object(
            FluidObjectCfg(
                uid="water",
                density=1000.0,
                particle_spacing=spacing,
                particle_radius=0.30 * spacing,
                spawn=FluidSpawnCfg(kind="points", positions=points),
                render=FluidRenderCfg(
                    mode="mesh_surface",
                    render_radius=0.30 * spacing,
                    material=water_material,
                    clip_below_z=0.49,
                    mesh_surface={
                        "gpu_enabled": True,
                        "cell_size": 0.54 * spacing,
                        "max_res": 512,
                        "kernel_radius": 1.01 * spacing,
                        "iso_density": 0.57,
                        "density_smooth_iters": 2,
                        "aabb_trim_fraction": 0.0,
                        "min_component_tris": 32,
                        "laplacian_iters": 4,
                        "laplacian_alpha": 0.11,
                        "update_period": 1,
                    },
                ),
            )
        )
        sim.add_light(
            LightCfg(
                uid="key", light_type="sun", intensity=4.0, direction=[-0.3, 0.2, -1.0]
            )
        )
        sim.add_light(
            LightCfg(
                uid="fill",
                light_type="rect",
                intensity=6.0,
                init_pos=[-0.6, -0.4, 1.4],
                direction=[0.0, 0.0, -1.0],
                rect_width=0.8,
                rect_height=0.8,
            )
        )
        sim.prepare()
        diagnostics = sim.get_fluid_diagnostics("water")
        boundary_info = {
            "method": "volume_map",
            "volume_map_enabled": diagnostics["volume_map_enabled"],
            "sdf_penalty_enabled": diagnostics["sdf_penalty_enabled"],
            "volume_map_resolution": args.volume_map_resolution,
            "boundary_shape_count": diagnostics["boundary_shape_count"],
            "rest_density_kg_m3": diagnostics["rest_density_kg_m3"],
            "viscosity": diagnostics["viscosity"],
        }
        print(json.dumps(boundary_info), flush=True)
        camera = sim.add_sensor(
            CameraCfg(
                uid="demo_camera",
                width=1280,
                height=960,
                intrinsics=[1100.0, 1100.0, 640.0, 480.0],
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=[-1.4, -1.4, 1.2],
                    target=[-0.43, -0.03, 0.47],
                    up=[0.0, 0.0, 1.0],
                ),
            )
        )
        close_camera = sim.add_sensor(
            CameraCfg(
                uid="water_camera",
                width=960,
                height=720,
                intrinsics=[1150.0, 1150.0, 480.0, 360.0],
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=[-1.12, -0.73, 0.96],
                    target=[-0.76, -0.07, 0.61],
                    up=[0.0, 0.0, 1.0],
                ),
            )
        )
        if not args.headless:
            sim.open_window()
        writer = imageio.get_writer(
            args.output / "franka_dfsph_pour.mp4",
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        close_writer = imageio.get_writer(
            args.output / "water_closeup.mp4",
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        metrics = []
        started = time.perf_counter()
        previous_cup_pose = initial_pose
        print(
            f"DFSPH: {len(points)} particles, {len(points) * spacing**3 * 1e6:.1f} mL, "
            f"{duration:.2f} s",
            flush=True,
        )
        for step in range(round(duration / dt)):
            elapsed = step * dt
            source_time = float(
                np.clip(
                    args.start_time + (elapsed - 1.0) * args.speed,
                    args.start_time,
                    recorded_times[-1],
                )
            )
            robot.set_qpos(
                torch.as_tensor(
                    joint_curve(source_time), dtype=torch.float32, device=sim.device
                ).unsqueeze(0),
                target=False,
            )
            cup_pose = _pose(
                position_curve(source_time), orientation_curve(source_time).as_matrix()
            )
            if args.substeps == 1:
                mug.set_kinematic_state(
                    torch.from_numpy(cup_pose).unsqueeze(0),
                    torch.from_numpy(
                        (cup_pose[:3, 3] - previous_cup_pose[:3, 3]) / dt
                    ).unsqueeze(0),
                    torch.from_numpy(
                        Rotation.from_matrix(
                            cup_pose[:3, :3] @ previous_cup_pose[:3, :3].T
                        ).as_rotvec()
                        / dt
                    ).unsqueeze(0),
                )
            previous_cup_pose = cup_pose
            sim.update(step=1, render_final_step=False)
            if step % frame_stride == 0:
                with sim.render_frame():
                    sim.sync_render_state()
                    camera.update()
                    close_camera.update()
                    rgb = camera.get_data()["color"][0, :, :, :3].cpu().numpy()
                    close_rgb = (
                        close_camera.get_data()["color"][0, :, :, :3].cpu().numpy()
                    )
                writer.append_data(rgb)
                close_writer.append_data(close_rgb)
                if step % args.physics_hz == 0:
                    cv2.imwrite(
                        str(args.output / f"frame_{int(elapsed):02d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(args.output / f"close_{int(elapsed):02d}.png"),
                        cv2.cvtColor(close_rgb, cv2.COLOR_RGB2BGR),
                    )
            if step % args.physics_hz == 0 or step == round(duration / dt) - 1:
                if args.substeps > 1:
                    cup_pose = mug.get_local_pose(to_matrix=True)[0].cpu().numpy()
                positions = fluid.data.particle_pos_w[0].cpu().numpy()
                if not np.isfinite(positions).all():
                    raise RuntimeError("Non-finite DFSPH particle state")
                velocities = fluid.data.particle_vel_w[0].cpu().numpy()
                summary = _particle_summary(positions, cup_pose, spacing, velocities)
                summary["time_s"] = elapsed + dt
                speed = np.linalg.norm(velocities, axis=1)
                density = (
                    sim.get_fluid_diagnostics("water")["particle_density"][0]
                    .cpu()
                    .numpy()
                )
                summary["speed_p99_m_s"] = float(np.quantile(speed, 0.99))
                summary["density_p99_kg_m3"] = float(np.quantile(density, 0.99))
                metrics.append(summary)
                np.savez_compressed(
                    args.output / f"particles_{int(elapsed):02d}.npz",
                    positions=positions,
                    velocities=velocities,
                    cup_pose=cup_pose,
                    time_s=elapsed + dt,
                )
                print(json.dumps(summary), flush=True)
        report = {
            "physics": "Newton SPH / DFSPH",
            "rigid_motion": "kinematic replay of verified MuJoCo trajectory",
            "coupling": "one-way moving volume-map boundaries; no fluid reaction on robot",
            "boundary": boundary_info,
            "bowl": {
                "center_m": RECEIVER_CENTER.tolist(),
                "outer_diameter_m": float(2.0 * outer_r[-1]),
                "height_m": BOWL_HEIGHT,
                "shape": "curved ceramic bowl with flat foot",
            },
            "trajectory": str(args.trajectory.resolve()),
            "particle_count": len(points),
            "spacing_m": spacing,
            "source_start_time_s": args.start_time,
            "rest_density_kg_m3": diagnostics["rest_density_kg_m3"],
            "viscosity": diagnostics["viscosity"],
            "surface_tension_coefficient": args.surface_tension,
            "render_aabb_trim_fraction": 0.0,
            "render_only_cull_below_z_m": 0.49,
            "nominal_volume_ml": len(points) * spacing**3 * 1e6,
            "simulation_dt": dt,
            "substeps": args.substeps,
            "duration_s": duration,
            "wall_seconds": time.perf_counter() - started,
            "samples": metrics,
        }
        (args.output / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"Saved {args.output}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if close_writer is not None:
            close_writer.close()
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    main()
