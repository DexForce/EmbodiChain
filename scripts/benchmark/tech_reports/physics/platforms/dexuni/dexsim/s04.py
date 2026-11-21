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
"""Record the native W1 rigid/soft squeeze-grasp scene for user acceptance.

Run: /home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/s04.py
This entry point records a scene preview only, without performance measurements.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

TECH = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp"
)
REFERENCE = Path("/home/oem/code/engine/dexsim/examples/python/physics/multiphys")
ROBOT = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing/w1_v030")
__all__ = ["main"]


def _write_json(path: Path, value: object) -> None:
    """Write readable scene provenance with NumPy array conversion."""
    path.write_text(
        json.dumps(value, indent=2, default=lambda item: item.tolist()) + "\n"
    )


def _phase(time: float) -> str:
    """Describe the fixed reference motion for the video caption."""
    for end, name in (
        (2.8, "approach"),
        (4.3, "close"),
        (5.8, "lift"),
        (7.3, "hold"),
        (8.8, "lower"),
        (9.6, "release"),
        (10.6, "retreat"),
    ):
        if time <= end:
            return name
    return "observe"


def main() -> None:
    """Freeze the reference scene and record one preview in its fixed directory."""
    output = DEFAULT_OUTPUT
    source = output / "source/reference"
    source.mkdir(parents=True, exist_ok=True)
    reference_names = (
        "demo_dexuni_w1_squeeze_grasp.py",
        "dexuni_w1_two_way_utils.py",
        "demo_dexuni_w1_bag_packing.py",
        "dexuni_asset_utils.py",
    )
    for name in reference_names:
        shutil.copy2(REFERENCE / name, source / name)
    robot_root = output / "inputs/w1_v030"
    if not robot_root.exists():
        shutil.copytree(ROBOT, robot_root)
    shutil.copy2(__file__, output / "source/s04.py")
    sys.path.insert(0, str(source))
    reference = importlib.import_module("demo_dexuni_w1_squeeze_grasp")
    args = reference.Example.create_parser().parse_args()
    args.headless = True
    args.w1_urdf = robot_root / "robot.urdf"
    if args.coupling != "two_way":
        raise ValueError("The requested preview uses dynamic two-way coupling")

    import dexsim
    import imageio.v2 as imageio
    import newton
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont
    from dexsim.engine.newton_physics import DexUniSolverCfg, NewtonCfg
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    width, height = 1280, 720
    eye = (1.65, -1.25, 1.7)
    target = (0.35, 0.0, 1.0)
    config = dexsim.WorldConfig()
    config.use_default_physics = False
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.OFFLINERT
    config.newton_cfg = NewtonCfg(
        device=args.device,
        dt=1 / 60,
        num_substeps=args.substeps,
        solver_cfg=DexUniSolverCfg(
            mujoco_articulation_paths=("w1",),
            joint_mode="dynamic",
            contact_mode="full",
            coupling="two_way",
            coupling_options={
                "mass_scale": args.proxy_mass_scale,
                "proxy_relaxation": args.proxy_relaxation,
            },
            vbd_options={
                "iterations": args.iterations,
                "friction_epsilon": 1e-4,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
            },
            collision_options={
                "broad_phase": "nxn",
                "include_static_kinematic_pairs": False,
                "soft_contact_margin": 0.003,
            },
        ),
    )
    config.newton_cfg.use_cuda_graph = not args.no_cuda_graph

    class PreviewScene(reference.Example):
        """Add one camera and neutral table appearance to the reference scene."""

        def add_box(self, builder, label, position, half, color, **options):
            if label == "worktop" or label.startswith("table_leg"):
                color = (0.5, 0.5, 0.5)
            return super().add_box(builder, label, position, half, color, **options)

        def build_task(self, builder):
            super().build_task(builder)
            builder.add_light(
                dexsim.spawn.LightDesc(
                    name="key_light",
                    light_type=dexsim.render.LightType.POINT,
                    pose=reference.pose_at((1.5, -1.0, 3.0)),
                    intensity=50.0,
                    shadow=False,
                )
            )
            builder.add_camera(
                dexsim.spawn.CameraDesc(
                    "preview",
                    resolution=(width, height),
                    near=0.01,
                    far=10.0,
                    headless=True,
                )
            )

    (output / "preview").mkdir(exist_ok=True)
    manifest = {
        "task": "S04",
        "mode": "native_scene_preview_only",
        "status": "recording",
        "user_scene_approved": False,
        "performance_measurements_started": False,
        "other_platforms_implemented": False,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "dexsim_version": dexsim.__version__,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "requested": {
            "substeps": args.substeps,
            "vbd_iterations": args.iterations,
            "cuda_graph": not args.no_cuda_graph,
            "joint_mode": "dynamic",
            "coupling": "two_way",
        },
        "robot": "W1 V030 parallel grippers, URDF finger effort limit 10 N",
        "scene": {
            "table_z": 0.86,
            "rigid_size": [0.048, 0.048, 0.09],
            "soft_size": [0.06, 0.06, 0.08],
            "lift_m": 0.15,
            "soft_material": {
                "density": 250,
                "k_mu": 6000,
                "k_lambda": 15000,
                "k_damp": 0.001,
            },
            "finger_kp": args.finger_kp,
            "finger_kd": args.finger_kd,
            "particle_radius": 0.003,
        },
        "camera": {
            "eye": eye,
            "target": target,
            "resolution": [width, height],
            "fps": 30,
        },
        "reference_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in reference_names
        },
        "robot_sha256": hashlib.sha256(args.w1_urdf.read_bytes()).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)
    world = dexsim.World(config)
    world.set_manual_update(True)
    scene = None
    camera = None
    states: dict[str, list] = {}
    try:
        scene = PreviewScene(world, args)
        camera = scene.scene.get_camera("preview")
        vfov = 2 * np.arctan(np.tan(np.deg2rad(65) / 2) * height / width)
        camera.set_perspective(float(np.rad2deg(vfov)), width / height, 0.01, 10.0)
        camera.look_at(eye, target, (0, 0, 1))
        camera.open_camera()
        backend = get_newton_backend(world)
        model = backend.model
        arrays = {
            name: getattr(model, name).numpy()
            for name in (
                "particle_q",
                "particle_mass",
                "particle_flags",
                "particle_radius",
                "tet_indices",
                "tet_materials",
                "tri_indices",
            )
            if getattr(model, name, None) is not None
        }
        np.savez_compressed(output / "inputs/soft_mesh.npz", **arrays)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        with imageio.get_writer(
            str(output / "preview/dexsim.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for frame in range(args.steps + 1):
                if frame:
                    scene.step()
                scene.check_finite()
                particles = (
                    scene.scene.get_particle_set("soft_cube")
                    .get_particle_positions()
                    .numpy()
                )
                row = {
                    "time": scene.sim_time,
                    "qpos": scene.robot.get_current_qpos(),
                    "target_q": scene._public_joint_row(),
                    "particles": particles,
                    "rigid_pose": scene.scene.get_object(
                        "rigid_block"
                    ).get_world_pose(),
                    "tcp": scene.tcp_positions(),
                }
                for key, value in row.items():
                    states.setdefault(key, []).append(np.array(value, copy=True))
                if frame % 2 == 0:
                    backend.sync_to_dexsim(world)
                    world._flush_render_updates()
                    camera.render()
                    image = Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3])
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, width, 42), fill="black")
                    draw.text(
                        (14, 10),
                        f"DexSim | S04 rigid / soft grasp | t={scene.sim_time:.2f}s | {_phase(scene.sim_time)}",
                        font=font,
                        fill="white",
                    )
                    writer.append_data(np.asarray(image))
                    if frame == 0:
                        image.save(output / "preview/first_frame.jpg")
        manifest.update(
            status="completed",
            frames=args.steps,
            video_frames=args.steps // 2 + 1,
            effective=scene.scene.get_solver_diagnostics(),
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if states:
            np.savez_compressed(
                output / "preview/states.npz",
                **{key: np.stack(values) for key, values in states.items()},
            )
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(output / "manifest.json", manifest)
        if camera is not None:
            camera.close_camera()
        if scene is not None:
            scene.scene.close()
    print(f"Scene preview ready: {output / 'preview/dexsim.mp4'}", flush=True)


if __name__ == "__main__":
    main()
