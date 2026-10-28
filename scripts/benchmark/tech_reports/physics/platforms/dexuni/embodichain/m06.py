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
"""Run the M06 T-shirt fold with DexUni's accelerated surface policy.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m06.py
Uses frozen native motion and materials; --measure records an authorized episode.
The default surface-fast policy uses eight base sweeps and adaptive fallback.
Pass --solver-policy ordinary to reproduce the original 15-sweep budget.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Literal

import numpy as np

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO))

from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m06 import (
    DEFAULT_OUTPUT as TASK,
    JOINTS,
    digest,
    observations,
    write_json,
)

__all__ = ["main"]


def load_inputs() -> (
    tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]
):
    """Verify and read the exact inputs approved with the native video."""
    approval = json.loads((TASK / "native_approval.json").read_text())
    if (
        not approval["approved"]
        or digest(TASK / "inputs/config.json") != approval["config_sha256"]
    ):
        raise ValueError("Approved M06 configuration is missing or has changed")
    if digest(TASK / "preview/dexsim.mp4") != approval["video_sha256"]:
        raise ValueError("Approved native preview video has changed")
    manifest = json.loads((TASK / "inputs/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if digest(TASK / "inputs" / name) != expected:
            raise ValueError(f"Frozen M06 input changed: {name}")
    config = json.loads((TASK / "inputs/config.json").read_text())
    with np.load(TASK / "inputs/cloth_mesh.npz") as data:
        mesh = {key: data[key] for key in data.files}
    with np.load(TASK / "inputs/trajectory.npz") as data:
        trajectory = {key: data[key] for key in data.files}
    return config, mesh, trajectory


def manager_config(
    config: dict[str, Any],
    device: str,
    video: bool,
    *,
    solver_policy: Literal["surface-fast", "ordinary"] = "surface-fast",
) -> Any:
    """Keep the frozen scene and select accelerated or ordinary surface sweeps."""
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RenderCfg

    solver = config["solver"]
    return SimulationManagerCfg(
        num_envs=1,
        device=device,
        headless=True,
        width=960,
        height=720,
        enable_entity_gizmo=False,
        robot_ik_gizmo=None,
        render_cfg=RenderCfg(renderer="rt" if video else "no-render", spp=1),
        physics_cfg=NewtonPhysicsCfg(
            device=device,
            physics_dt=solver["dt"],
            num_substeps=solver["num_substeps"],
            use_cuda_graph=solver["use_cuda_graph"],
            collision_cfg=None,
            solver_cfg={
                "solver_type": "dexuni",
                "joint_mode": "kinematic",
                "vbd_preset": (
                    "surface-fast" if solver_policy == "surface-fast" else None
                ),
                # Match Isaac Lab's color schedule. Bending forces remain enabled.
                "particle_coloring_include_bending": False,
                "particle_coloring_balance_colors": False,
                "vbd_options": {
                    # Keep the preset's multilevel correction and adaptive fallback.
                    "iterations": (
                        8
                        if solver_policy == "surface-fast"
                        else solver["vbd_iterations"]
                    ),
                    # Cache substep invariants while retaining current contact forces.
                    "particle_enable_surface_cache": True,
                    "particle_enable_truncation_cache": True,
                    "particle_enable_self_contact": True,
                    "particle_self_contact_radius": solver["self_contact_radius"],
                    "particle_self_contact_margin": solver["self_contact_margin"],
                    "particle_topological_contact_filter_threshold": 1,
                    "particle_rest_shape_contact_exclusion_radius": 0.005,
                    "particle_vertex_contact_buffer_size": 32,
                    "particle_edge_contact_buffer_size": 40,
                    "particle_collision_detection_interval": -1,
                },
                "collision_options": {
                    "soft_contact_margin": config["contact"]["soft_contact_margin"]
                },
            },
        ),
    )


def build_scene(
    sim: Any,
    config: dict[str, Any],
    mesh: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    video: bool,
) -> tuple[Any, Any, list[Any]]:
    """Create every physical object and camera through SimulationManager."""
    from dexsim.engine.newton_physics import KinematicJointTrajectoryControl
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        ArticulationRootPropertiesCfg,
        LightCfg,
        NewtonCollisionPropertiesCfg,
        NewtonRigidBodyMaterialCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
        SurfaceDeformableObjectCfg,
        SurfaceDeformablePhysicsCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.sim.shapes import CubeCfg, MeshCfg

    contact = config["contact"]
    # NewtonCollisionDesc in the approved native scene uses 1 mm for both.
    attrs = RigidBodyPhysicsCfg(
        collision_props=NewtonCollisionPropertiesCfg(margin=0.001, gap=0.001),
        material_props=NewtonRigidBodyMaterialCfg(
            ke=contact["shape_ke"],
            kd=contact["shape_kd"],
            dynamic_friction=contact["table_friction"],
        ),
    )
    for name, center, size in (
        ("table", (0, -0.5, 0.10), (0.82, 0.82, 0.20)),
        ("robot_stand", (-0.52, -0.5, 0.06), (0.22, 0.22, 0.12)),
    ):
        sim.add_rigid_object(
            RigidObjectCfg(
                uid=name,
                body_type="static",
                init_pos=center,
                attrs=attrs,
                shape=CubeCfg(
                    size=list(size),
                    visual_material=VisualMaterialCfg(
                        uid=name, base_color=[0.42, 0.48, 0.52, 1], roughness=0.9
                    ),
                ),
            )
        )
    robot = sim.add_articulation(
        ArticulationCfg(
            uid="franka",
            fpath=str(TASK / "inputs/franka/fr3_franka_hand.dexsim.urdf"),
            init_pos=tuple(config["robot"]["base_position"]),
            init_qpos=trajectory["joint_targets"][0].tolist(),
            build_pk_chain=False,
            asset_physics_mode="overlay",
            attrs=attrs,
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True, newton_mimic_stabilizer=False
            ),
        )
    )
    sim.add_runtime_control(
        lambda arena, _index: KinematicJointTrajectoryControl(
            f"{arena}/franka",
            trajectory["joint_targets"],
            joint_names=JOINTS,
            fps=1 / config["solver"]["dt"],
            initialize_pose=True,
        )
    )
    sim.register_contact_material_schedule(
        "franka",
        {
            "dynamic_friction": ((0.0, contact["gripper_friction"]),),
            "stiffness": ((0.0, contact["gripper_ke"]),),
            "damping": ((0.0, contact["gripper_kd"]),),
        },
        link_names=("fr3_hand", "fr3_leftfinger", "fr3_rightfinger"),
    )
    cloth = config["cloth"]
    shirt = sim.add_deformable_object(
        SurfaceDeformableObjectCfg(
            uid="shirt",
            init_pos=tuple(cloth["center"]),
            particle_radius=cloth["particle_radius"],
            shape=MeshCfg(
                vertices=mesh["vertices"],
                triangles=mesh["triangles"],
                normals=mesh["normals"],
                uv_coords=mesh["uv"],
                visual_material=VisualMaterialCfg(
                    uid="m06_shirt",
                    base_color=[0.85, 0.23, 0.16, 1],
                    base_color_texture=str(TASK / "inputs/mianbu.png"),
                    roughness=0.85,
                ),
            ),
            attrs=SurfaceDeformablePhysicsCfg(
                density=cloth["density"],
                edge_rest_angle=cloth["edge_rest_angle"],
                surface_props={
                    key: cloth[key]
                    for key in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                },
                contact_material={
                    "stiffness": contact["particle_ke"],
                    "damping": contact["particle_kd"],
                    "dynamic_friction": contact["particle_friction"],
                },
            ),
        )
    )
    cameras = []
    if video:
        sim.add_light(
            LightCfg(
                uid="key_light",
                light_type="point",
                init_pos=(1, -0.2, 3),
                intensity=70,
                enable_shadow=False,
            )
        )
        sim.set_emission_light([1, 1, 1], 70)
        for name, width, eye, target, up in (
            ("perspective", 960, (1.25, 0.65, 1.35), (-0.15, -0.5, 0.36), (0, 0, 1)),
            ("overhead", 720, (0, -0.5, 1.4), (0, -0.5, 0.2), (0, 1, 0)),
        ):
            focal = 720 / 2 / np.tan(np.deg2rad(48) / 2)
            cameras.append(
                sim.add_sensor(
                    CameraCfg(
                        uid=name,
                        width=width,
                        height=720,
                        near=0.01,
                        far=10,
                        intrinsics=(focal, focal, width / 2, 360),
                        enable_depth=False,
                        enable_mask=False,
                        extrinsics=CameraCfg.ExtrinsicsCfg(
                            eye=eye, target=target, up=up
                        ),
                    )
                )
            )
    sim.prepare()
    if video:
        for material in shirt.get_visual_material_inst():
            if material is None:
                raise RuntimeError("Cloth render material was not bound")
            material.mat.set_double_sided(True)
        # CameraCfg initialization is retained; verify the finalized intrinsics.
        for camera in cameras:
            expected = np.array(
                [[focal, 0, camera.cfg.width / 2], [0, focal, 360], [0, 0, 1]]
            )
            if not np.allclose(
                camera.get_intrinsics()[0].cpu().numpy(), expected, atol=1e-3
            ):
                import torch

                camera.set_intrinsics(
                    torch.tensor(expected, device=sim.device, dtype=torch.float32)
                )
    return robot, shirt, cameras


def write_report(output: Path, result: dict[str, Any]) -> None:
    """Write one scene report with explicitly unmeasured performance fields."""
    o = result["observations"]
    (output / "REPORT.md").write_text(f"""# M06 EmbodiChain 预览

沿用用户确认的 DexSim 衣服、轨迹和材料。Franka 运动学控制，只从左往右折一次，在桌面上方 95 mm 松爪，随后垂直退开。

求解策略：{result['effective']['vbd_preset'] or 'ordinary'}；基础迭代：{result['effective']['vbd']['iterations']}。surface-fast 保留多层校正、自适应回退和默认 5 微米/子步位移阈值。

## Time & Memory

| 平台 | 性能耗时 | 显存 | 范围 |
| --- | --- | --- | --- |
| EmbodiChain / DexUni | 未测量 | 未测量 | 场景预览 |

## Success & Other Metrics

| 有限状态 | 实际时长 | 袖口抬升 | 袖口向右位移 | Graph | 人工确认 |
| --- | --- | --- | --- | --- | --- |
| {o['all_states_finite']} | {o['actual_end_time']:.3f} s | {o['left_sleeve_peak_lift_m']*1000:.1f} mm | {o['left_sleeve_final_rightward_motion_m']*1000:.1f} mm | {result['effective']['cuda_graph_status']} | 待观看 |

## Leaderboard

| 平台 | 结论 |
| --- | --- |
| EmbodiChain / DexUni | 尚未测量性能或排名 |

[视频](embodichain.mp4) · [实际状态](states.npz) · [验证](manifest.json)。袖口指标是固定材料节点组的几何描述，未作为精确对齐或无贯穿结论。
""")


def run(
    args: argparse.Namespace,
    config: dict[str, Any],
    mesh: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
) -> None:
    """Verify the physical fold and optionally record both camera sensors."""
    import dexsim
    import embodichain
    import newton
    import warp as wp
    from embodichain.lab.sim import SimulationManager

    output = args.out_dir.resolve()
    if TASK not in output.parents or output.parts[len(TASK.parts)] not in {
        "preview",
        "history",
    }:
        raise ValueError("M06 outputs must be under the task's preview/ or history/")
    if (output / "manifest.json").exists():
        raise FileExistsError("Preserve the existing M06 attempt before rerunning")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir(exist_ok=True)
    paths = [
        Path(__file__),
        REPO / "embodichain/lab/sim/cfg/deformable.py",
        REPO / "embodichain/lab/sim/spawn/descriptors.py",
        REPO / "embodichain/lab/sim/sim_manager.py",
    ]
    for path in paths:
        shutil.copy2(path, source / path.name)
    manifest: dict[str, Any] = {
        "task": "M06",
        "platform": "embodichain",
        "status": "running",
        "scope": "scene_preview_only",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.executable,
        "source_sha256": {str(path.relative_to(REPO)): digest(path) for path in paths},
        "config_sha256": digest(TASK / "inputs/config.json"),
        "trajectory_sha256": digest(TASK / "inputs/trajectory.npz"),
        "versions": {
            "embodichain": embodichain.__version__,
            "dexsim": dexsim.__version__,
            "newton": newton.__version__,
            "warp": wp.__version__,
        },
        "native_scene_approved": True,
        "embodichain_scene_approved": False,
        "performance_measured": False,
        "frozen_input_solver": config["solver"],
        "requested": {
            **config["solver"],
            "vbd_preset": (
                "surface-fast" if args.solver_policy == "surface-fast" else None
            ),
            "vbd_iterations": (
                8
                if args.solver_policy == "surface-fast"
                else config["solver"]["vbd_iterations"]
            ),
        },
    }
    write_json(output / "manifest.json", manifest)
    sim = None
    robot, shirt, cameras = None, None, []
    states: dict[str, list[np.ndarray]] = {
        key: [] for key in ("time", "qpos", "tcp", "particles")
    }
    frame_times = []
    try:
        sim = SimulationManager(
            manager_config(
                config,
                args.device,
                not args.no_video,
                solver_policy=args.solver_policy,
            )
        )
        robot, shirt, cameras = build_scene(
            sim, config, mesh, trajectory, not args.no_video
        )
        ids = [robot.joint_names.index(name) for name in JOINTS]
        initial = shirt.data.nodal_pos_w[0].cpu().numpy()
        if not np.allclose(initial, mesh["initial_positions"], atol=1e-7):
            raise RuntimeError(
                "Cloth initial topology/positions differ from the approved scene"
            )
        inverse_mass = (
            sim.spawn_result.get_particle_set("arena_0/shirt")
            .get_inverse_masses()
            .numpy()
        )
        if not np.isfinite(inverse_mass).all() or np.any(inverse_mass <= 0):
            raise RuntimeError(
                "M06 requires all cloth nodes to have finite positive inverse masses"
            )
        np.savez_compressed(
            output / "initial_model.npz",
            positions=initial,
            inverse_mass=inverse_mass,
            triangles=mesh["triangles"],
        )
        last = len(trajectory["time"]) - 1
        if args.steps is not None:
            last = min(last, args.steps)
        with ExitStack() as stack:
            writer = None
            if cameras:
                import imageio.v2 as imageio
                from PIL import Image, ImageDraw, ImageFont

                font = ImageFont.truetype("DejaVuSans.ttf", 21)
                writer = stack.enter_context(
                    imageio.get_writer(
                        str(output / "embodichain.mp4"),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        macro_block_size=2,
                    )
                )
            for frame in range(last + 1):
                if frame:
                    sim.update(
                        physics_dt=config["solver"]["dt"],
                        step=1,
                        render_final_step=False,
                    )
                actual_time = sim.spawn_result.get_simulation_time()
                if not np.isclose(actual_time, trajectory["time"][frame], atol=1e-9):
                    raise RuntimeError(
                        f"Actual simulation time differs at frame {frame}"
                    )
                particles = shirt.data.nodal_pos_w[0].cpu().numpy()
                qpos = robot.get_qpos()[0, ids].cpu().numpy()
                matrix = (
                    robot.get_link_pose(config["robot"]["tcp_body"], to_matrix=True)[0]
                    .cpu()
                    .numpy()
                )
                tcp = matrix[:3, 3] + matrix[:3, :3] @ np.asarray(
                    config["robot"]["tcp_offset"]
                )
                if not all(np.isfinite(x).all() for x in (particles, qpos, tcp)):
                    raise RuntimeError(f"Nonfinite physical state at frame {frame}")
                if not np.allclose(qpos, trajectory["joint_targets"][frame], atol=1e-5):
                    raise RuntimeError(
                        f"Kinematic joint tracking failed at frame {frame}"
                    )
                if frame % 6 == 0 or frame == last:
                    for key, value in (
                        ("time", actual_time),
                        ("particles", particles),
                        ("qpos", qpos),
                        ("tcp", tcp),
                    ):
                        states[key].append(np.array(value, copy=True))
                if cameras and frame > 0 and frame % 2 == 0:
                    with sim.render_frame():
                        sim.sync_render_state()
                        views = []
                        for camera in cameras:
                            camera.update()
                            views.append(
                                camera.get_data()["color"][0, ..., :3]
                                .cpu()
                                .numpy()
                                .copy()
                            )
                    panel = Image.fromarray(np.concatenate(views, axis=1))
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, 1680, 45), fill="black")
                    phase = trajectory["phase_names"][trajectory["phase_ids"][frame]]
                    draw.text(
                        (12, 11),
                        f"EmbodiChain | M06 Franka T-shirt LEFT -> RIGHT | t={actual_time:.2f}s | {phase}",
                        font=font,
                        fill="white",
                    )
                    writer.append_data(np.asarray(panel))
                    frame_times.append(float(actual_time))
                    if frame == 2 or frame == last - last % 2:
                        panel.save(
                            output / f"{'first' if frame==2 else 'last'}_frame.jpg"
                        )
                if frame % 120 == 0:
                    print(
                        f"frame={frame}/{last} t={actual_time:.2f} sleeve={particles[mesh['left_ids']].mean(0).round(4)}",
                        flush=True,
                    )
        arrays = {key: np.stack(value) for key, value in states.items()}
        observation = observations(arrays, mesh)
        expected = np.stack(
            [
                np.interp(arrays["time"], trajectory["time"], column)
                for column in trajectory["desired_tcp"].T
            ],
            axis=-1,
        )
        tracking = float(np.linalg.norm(arrays["tcp"] - expected, axis=-1).max())
        if tracking > 0.0005:
            raise RuntimeError(
                f"Recorded TCP tracking error {tracking} m exceeds the IK tolerance"
            )
        if last == len(trajectory["time"]) - 1 and (
            observation["left_sleeve_peak_lift_m"] < 0.05
            or observation["left_sleeve_final_rightward_motion_m"] < 0.15
        ):
            raise RuntimeError(
                "Physical sleeve motion is insufficient to reproduce the approved fold"
            )
        manifest.update(
            status="completed",
            effective=sim.spawn_result.get_solver_diagnostics(),
            observations=observation,
            full_trajectory=last == len(trajectory["time"]) - 1,
            video_frames=len(frame_times),
            all_nodes_dynamic=True,
            max_sampled_tcp_error_m=tracking,
            release_tcp_height_above_table_m=config["motion"]["left_place"][2] - 0.2,
        )
        if manifest["effective"]["cuda_graph_status"] != "captured":
            raise RuntimeError("Expected a captured CUDA Graph")
        write_json(output / "frame_times.json", {"fps": 30, "actual_time": frame_times})
        write_report(output, manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if states["time"]:
            np.savez_compressed(
                output / "states.npz",
                **{key: np.stack(value) for key, value in states.items()},
            )
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        robot, shirt, cameras = None, None, []
        if sim is not None:
            sim.destroy(exit_process=False)
    print(f"EmbodiChain scene ready: {output}", flush=True)


def measure(args: argparse.Namespace) -> None:
    """Run one authorized no-camera performance episode through the scene facade."""
    import warp as wp
    from embodichain.lab.sim import SimulationManager
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.m06_measurement import (
        measurement_directory,
        record_episode,
    )

    if args.steps is not None or args.solver_policy != "surface-fast":
        raise ValueError("Formal M06 requires the full surface-fast episode")
    config, mesh, trajectory = load_inputs()
    output = measurement_directory("embodichain", args.repeat)
    start = time.perf_counter()
    settings = manager_config(config, args.device, False)
    settings.startup_summary = "off"
    sim = SimulationManager(settings)
    robot = shirt = None
    try:
        robot, shirt, _ = build_scene(sim, config, mesh, trajectory, False)
        ids = [robot.joint_names.index(name) for name in JOINTS]
        inverse_mass = (
            sim.spawn_result.get_particle_set("arena_0/shirt")
            .get_inverse_masses()
            .numpy()
        )
        if not np.isfinite(inverse_mass).all() or np.any(inverse_mass <= 0):
            raise RuntimeError("Every M06 cloth node must remain dynamic")
        np.savez_compressed(
            output / "initial_model.npz",
            inverse_mass=inverse_mass,
            positions=shirt.data.nodal_pos_w[0].cpu().numpy(),
            triangles=mesh["triangles"],
        )
        init_s = time.perf_counter() - start

        def readings() -> dict[str, np.ndarray]:
            """Read actual backend time, named joints and garment nodes."""
            matrix = (
                robot.get_link_pose(config["robot"]["tcp_body"], to_matrix=True)[0]
                .cpu()
                .numpy()
            )
            return {
                "time": np.asarray(sim.spawn_result.get_simulation_time()),
                "particles": shirt.data.nodal_pos_w[0].cpu().numpy(),
                "qpos": robot.get_qpos()[0, ids].cpu().numpy(),
                "tcp": matrix[:3, 3]
                + matrix[:3, :3] @ np.asarray(config["robot"]["tcp_offset"]),
            }

        if not np.allclose(
            readings()["particles"], mesh["initial_positions"], atol=1e-7
        ):
            raise RuntimeError("Frozen cloth initial positions changed")

        def diagnostics() -> dict[str, Any]:
            """Record effective preset flags and the actual outer-step count."""
            effective = sim.spawn_result.get_solver_diagnostics()
            effective.update(
                outer_steps=effective["step_index"],
                simulated_elapsed_s=effective["sim_time"],
                vbd_iterations=effective["vbd"]["iterations"],
            )
            if not (
                effective["vbd_preset"] == "surface-fast"
                and effective["vbd"]["particle_enable_batched_jacobi"]
                and effective["vbd"]["particle_chebyshev_enabled"]
                and effective["vbd"]["multilevel_active"]
                and effective["vbd"]["surface_cache"]
                and effective["vbd"]["truncation_cache"]
                and effective["features"]["bending_solve_enabled"]
                and effective["vbd"]["particle_enable_self_contact"]
            ):
                raise RuntimeError("The measured fast surface policy is not active")
            return effective

        record_episode(
            "embodichain",
            output,
            config,
            mesh,
            trajectory,
            lambda: sim.update(
                physics_dt=config["solver"]["dt"], step=1, render_final_step=False
            ),
            readings,
            lambda: wp.synchronize_device(args.device),
            diagnostics,
            init_s,
        )
    finally:
        robot = shirt = None
        sim.destroy(exit_process=False)


def main() -> None:
    """Load approved inputs, execute the scene and release deferred resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=TASK / "preview/embodichain")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--solver-policy",
        choices=("surface-fast", "ordinary"),
        default="surface-fast",
        help="Accelerated eight-sweep preset, or the original ordinary solve budget",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    config, mesh, trajectory = load_inputs()
    try:
        if args.measure:
            measure(args)
        else:
            run(args, config, mesh, trajectory)
    finally:
        from embodichain.lab.sim import SimulationManager

        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
