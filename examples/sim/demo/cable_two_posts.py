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
"""Contact-driven two-post cable preview through EmbodiChain's manager.

The Cartesian gripper is a prescribed articulation. Only its fingers are
controlled: the cable has one fixed root and no moving attachments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from scipy.interpolate import make_interp_spline

FPS = 60
DURATION = 14.0
RADIUS = 0.008
POST_RADIUS = 0.085
POSTS = ((0.0, 0.0), (0.0, 0.4))
# t, x, y, z, finger centre distance, phase
WAYPOINTS = (
    (0.0, 0.0, -0.28, 0.020, 0.045, "APPROACH"),
    (0.5, 0.0, -0.28, 0.020, 0.045, "APPROACH"),
    (1.2, 0.0, -0.28, 0.020, 0.0125, "GRIP"),
    (2.0, 0.0, -0.25, 0.035, 0.0125, "LIFT"),
    (3.3, 0.17, -0.12, 0.035, 0.0125, "WRAP POST 1"),
    (4.7, 0.17, 0.10, 0.035, 0.0125, "WRAP POST 1"),
    (5.6, 0.09, 0.20, 0.035, 0.0125, "BETWEEN POSTS"),
    (6.8, -0.17, 0.24, 0.035, 0.0125, "WRAP POST 2"),
    (8.2, -0.17, 0.49, 0.035, 0.0125, "WRAP POST 2"),
    (9.2, 0.0, 0.73, 0.035, 0.0125, "WRAP POST 2"),
    (10.2, 0.0, 0.73, 0.035, 0.0125, "HOLD"),
    (10.7, 0.0, 0.73, 0.035, 0.045, "RELEASE"),
    (11.5, 0.0, 0.73, 0.22, 0.045, "RETREAT"),
    (14.0, 0.0, 0.73, 0.22, 0.045, "OBSERVE"),
)


def control_at(t: float) -> tuple[np.ndarray, str]:
    """Sample a smooth deterministic Cartesian gripper command."""
    for a, b in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        if t <= b[0]:
            u = np.clip((t - a[0]) / (b[0] - a[0]), 0, 1)
            u = u * u * (3 - 2 * u)
            values = (1 - u) * np.array(a[1:5]) + u * np.array(b[1:5])
            return np.array([*values[:3], values[3], values[3]], np.float32), b[5]
    return np.array([0, 0.73, 0.22, 0.045, 0.045], np.float32), "OBSERVE"


def initial_points(segments: int) -> np.ndarray:
    """Sample one continuous slack shape uniformly by arc length."""
    waypoints = np.array(
        [
            [0, -0.52],
            [-0.36, -0.65],
            [-0.57, -0.50],
            [-0.40, -0.34],
            [-0.12, -0.43],
            [0, -0.39],
            [0, -0.28],
        ]
    )
    spline = make_interp_spline(np.arange(len(waypoints)), waypoints, k=2)
    dense = spline(np.linspace(0, len(waypoints) - 1, 3000))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    sample = np.linspace(0, arc[-1], segments + 1)
    return np.column_stack(
        [
            np.interp(sample, arc, dense[:, 0]),
            np.interp(sample, arc, dense[:, 1]),
            np.full_like(sample, 0.011),
        ]
    ).astype(np.float32)


def write_stage(path: Path) -> None:
    """Author fixed posts and a five-axis parallel-jaw gripper in one URDF."""
    parts = ['<?xml version="1.0"?><robot name="two_post_stage">']

    def link(
        name: str, geometry: str = "", xyz: str = "0 0 0", color: str = ".3 .4 .5 1"
    ) -> None:
        parts.append(
            f'<link name="{name}"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial>'
        )
        if geometry:
            parts.append(
                f'<visual><origin xyz="{xyz}"/><geometry>{geometry}</geometry><material name="{name}_mat"><color rgba="{color}"/></material></visual>'
            )
            parts.append(
                f'<collision><origin xyz="{xyz}"/><geometry>{geometry}</geometry></collision>'
            )
        parts.append("</link>")

    def joint(
        name: str,
        parent: str,
        child: str,
        kind: str = "fixed",
        xyz: str = "0 0 0",
        axis: str = "1 0 0",
    ) -> None:
        parts.append(
            f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/><axis xyz="{axis}"/><limit lower="-3" upper="3" effort="1000" velocity="10"/></joint>'
        )

    link("base", '<box size="1.6 1.8 .04"/>', "-.15 .10 -.02", ".15 .20 .25 1")
    for i, (x, y) in enumerate(POSTS):
        link(
            f"post_{i+1}",
            f'<cylinder radius="{POST_RADIUS}" length=".22"/>',
            "0 0 .11",
            ".15 .60 .73 1",
        )
        joint(f"post_{i+1}_fixed", "base", f"post_{i+1}", xyz=f"{x} {y} 0")
    link("anchor", '<box size=".055 .055 .026"/>', "0 0 0", ".95 .57 .12 1")
    joint("anchor_fixed", "base", "anchor", xyz="0 -.52 .013")
    for name, parent, axis in (
        ("x", "base", "1 0 0"),
        ("y", "slide_x", "0 1 0"),
        ("z", "slide_y", "0 0 1"),
    ):
        link(f"slide_{name}")
        joint(f"slide_{name}_joint", parent, f"slide_{name}", "prismatic", axis=axis)
    link("palm", '<box size=".13 .08 .04"/>', "0 0 .085", ".7 .75 .8 1")
    joint("palm_fixed", "slide_z", "palm")
    for name, axis in (("left", "1 0 0"), ("right", "-1 0 0")):
        link(f"finger_{name}", '<box size=".014 .09 .08"/>', "0 0 .025", ".87 .9 .94 1")
        joint(f"finger_{name}_joint", "palm", f"finger_{name}", "prismatic", axis=axis)
    parts.append("</robot>")
    path.write_text("\n".join(parts))


def run(args: argparse.Namespace) -> None:
    """Build with EmbodiChain, step cable physics, and record camera observations."""
    import dexsim
    import newton
    import warp as wp
    from dexsim.spawn import (
        CableDesc,
        CablePhysicsDesc,
        MaterialDesc,
        NewtonCollisionDesc,
    )
    from dexsim.engine.newton_physics import MJVBDV2SolverCfg
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        DLSSCfg,
        RigidBodyPhysicsCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    urdf = out / "two_post_stage.urdf"
    write_stage(urdf)
    points = initial_points(args.segments)
    frames = round(min(args.duration, DURATION) * FPS)
    times = np.arange(frames + 1) / FPS
    controls = np.array([control_at(t)[0] for t in times])
    np.savez(out / "inputs.npz", points=points, controls=controls, times=times)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="compact",
            robot_ik_gizmo=None,
            render_cfg=RenderCfg(
                renderer="hybrid", spp=4, dlss=DLSSCfg(dlss_enabled=False)
            ),
            physics_cfg=NewtonPhysicsCfg(
                physics_dt=1 / FPS,
                num_substeps=args.substeps,
                solver_cfg=MJVBDV2SolverCfg(
                    joint_mode="kinematic",
                    vbd_options={
                        "iterations": args.iterations,
                        "rigid_contact_history": True,
                        "friction_epsilon": 0.001,
                    },
                    collision_options={"contact_matching": "latest"},
                ),
                collision_cfg=None,
                sync_to_renderer=False,
                use_cuda_graph=True,
            ),
        )
    )
    video = camera = cable = None
    poses, speeds, slips = [], [], []
    start = time.perf_counter()
    try:
        sim.add_articulation(
            ArticulationCfg(
                uid="stage",
                fpath=str(urdf),
                init_qpos=controls[0],
                asset_physics_mode="overlay",
                attrs=RigidBodyPhysicsCfg.from_dict(
                    {
                        "material_props": {
                            "static_friction": 1.2,
                            "dynamic_friction": 1.0,
                            "restitution": 0.0,
                        },
                        "collision_props": {
                            "contact_offset": 0.002,
                            "rest_offset": 0.0,
                        },
                    }
                ),
            )
        )
        sim.register_kinematic_joint_trajectory("stage", controls[None], fps=FPS)
        sim.add_cable(
            CableDesc(
                name="cable",
                points=points,
                radius=RADIUS,
                per_env=True,
                fixed_segments=(0,),
                collision_filter_span=1,
                collision=NewtonCollisionDesc(mu=1.2, margin=0, gap=0.002),
                physics=CablePhysicsDesc(
                    density=800,
                    stretch_stiffness=1e5,
                    shear_stiffness=1e5,
                    bend_stiffness=0.012,
                    twist_stiffness=0.012,
                    stretch_damping=1,
                    shear_damping=1,
                    bend_damping=0.002,
                    twist_damping=0.002,
                ),
                render_material=MaterialDesc(
                    name="orange_cable",
                    base_color=(1.0, 0.25, 0.035, 1.0),
                    roughness=0.6,
                ),
            )
        )
        if not args.no_video:
            camera = sim.add_sensor(
                CameraCfg(
                    uid="overview",
                    width=1280,
                    height=960,
                    intrinsics=(1050.0, 1050.0, 640.0, 480.0),
                    enable_color=True,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(-0.12, -0.60, 2.30), target=(-0.16, 0.05, 0.025)
                    ),
                )
            )
        sim.prepare()
        cable = sim.get_cable("cable")
        print("READY", sim.physics.solver_type, cable.segment_count, flush=True)
        if camera is not None:
            video = cv2.VideoWriter(
                str(out / "preview_raw.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (1280, 960),
            )
            if not video.isOpened():
                raise RuntimeError("Cannot open video writer")
        phase = None
        initial_tip = cable.get_segment_poses()[-1, :3, 3]
        tip_offset = initial_tip - controls[0, :3]
        for frame, t in enumerate(times):
            if frame:
                sim.update(step=1, render_final_step=False)
            p = cable.get_segment_poses()
            v = cable.get_segment_velocities()
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                raise RuntimeError(f"Non-finite cable state at {t}")
            poses.append(p)
            speeds.append(np.sqrt(np.mean(np.sum(v[:, :3] ** 2, axis=1))))
            slips.append(
                float(np.linalg.norm(p[-1, :3, 3] - controls[frame, :3] - tip_offset))
            )
            current = control_at(t)[1]
            if current != phase:
                print(
                    f"{t:.2f}s {current}, tip={p[-1,:3,3]}, slip={slips[-1]:.4f}",
                    flush=True,
                )
                phase = current
            if camera is not None and frame % 2 == 0:
                with sim.render_frame(capture_camera_images=False):
                    sim.sync_render_state()
                    camera.update()
                rgb = camera.get_data()["color"][0, ..., :3].cpu().numpy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr, (0, 0), (1280, 88), (25, 29, 34), -1)
                cv2.putText(
                    bgr,
                    "P-11 | CABLE AROUND TWO POSTS",
                    (28, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    f"{t:05.2f} s    {current}    |    EmbodiChain + cable DexSim",
                    (28, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (170, 200, 215),
                    1,
                    cv2.LINE_AA,
                )
                video.write(bgr)
                if frame % 60 == 0:
                    cv2.imwrite(str(out / f"frame_{frame:04d}.jpg"), bgr)
        wp.synchronize()
        np.savez_compressed(
            out / "states.npz",
            times=times,
            poses=poses,
            speed_rms=speeds,
            grip_tip_error=slips,
        )
        metrics = {
            "demo_only": True,
            "grasp": "finger frictional contact; no endpoint attachment",
            "fixed_root_segment": 0,
            "dexsim_commit": dexsim.__commit_id__,
            "dexsim_path": dexsim.__file__,
            "newton": newton.__version__,
            "warp": wp.__version__,
            "solver": sim.physics.solver_type,
            "segments": args.segments,
            "initial_length_m": float(
                np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            ),
            "dt": 1 / FPS,
            "substeps": args.substeps,
            "iterations": args.iterations,
            "duration_s": float(times[-1]),
            "wall_s": time.perf_counter() - start,
            "max_grip_tip_error_before_release_m": (
                float(np.max(np.array(slips)[(times >= 1.3) & (times <= 10.2)]))
                if frames >= 78
                else None
            ),
            "final_speed_rms_m_s": float(speeds[-1]),
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        }
        (out / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        if video is not None:
            video.release()
        cable = camera = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    if not args.no_video:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out / "preview_raw.mp4"),
                "-c:v",
                "libx264",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out / "cable_two_posts.mp4"),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    run(parser.parse_args())
