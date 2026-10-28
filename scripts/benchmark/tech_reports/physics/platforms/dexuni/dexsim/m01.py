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
"""Record the native W1 dual-arm soft-beam bend scene (M01, task 5).

Run: /home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m01.py
Pilot: add ``--pilot --out <scratch dir>`` to save a few still frames instead of the video.
This entry point records a scene preview only, without performance measurements.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m01_common import (
    PHASES,
    beam_metrics,
    phase as _phase,
)

TECH = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend"
)
REFERENCE = Path("/home/oem/code/engine/dexsim/examples/python/physics/multiphys")
ROBOT = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing/w1_v030")
REFERENCE_NAMES = (
    "dexuni_w1_two_way_utils.py",
    "demo_dexuni_w1_bag_packing.py",
    "dexuni_asset_utils.py",
)
__all__ = ["main"]

# Scene (metres). The beam lies along world Y on the worktop.
BEAM_SIZE = np.array((0.04, 0.46, 0.04))  # X, Y, Z
BEAM_X = 0.48
GRIP_Y = 0.20  # grip points 30 mm from each end -> 400 mm apart
SHORTEN = 0.025  # each end moves inward -> 350 mm apart (12.5%)
TILT_DEG = 10.0  # each end rotates inward about X
LIFT = 0.15
CARRY_X = -0.10  # target marked area, towards the robot
GROUP_HALF = 0.01  # half width (Y) of measured node groups
# Bottom nodes rest about 4.5 mm above the worktop (particle radius plus
# contact margin); spawning lower makes the beam pop up at the start.
SPAWN_GAP = 0.005
# Downward gripper with fingers closing along world X.
DOWNWARD_X_FINGERS = np.array((-np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
# Each gripper leans back towards the robot; the left one is turned 180 deg
# about vertical (same parallel-jaw grasp). An IK-only sweep of the full motion
# showed this keeps every arm joint >= 0.4 rad from its limit; straight-down
# grippers hit a wrist limit near the grasp height and jerked the arms.
GRIPPER_PITCH_DEG = 15.0
SIDE_YAW_DEG = (180.0, 0.0)  # left, right


def _write_json(path: Path, value: object) -> None:
    """Write readable provenance with NumPy conversion."""
    path.write_text(
        json.dumps(value, indent=2, default=lambda item: np.asarray(item).tolist())
        + "\n"
    )


def side_rotation(side: int, tilt_deg: float) -> np.ndarray:
    """XYZW grip orientation of one side (0 left, 1 right) with inward tilt."""
    from scipy.spatial.transform import Rotation

    sign = (1.0, -1.0)[side]
    turned = (
        Rotation.from_euler("x", sign * tilt_deg, degrees=True)
        * Rotation.from_euler("y", -GRIPPER_PITCH_DEG, degrees=True)
        * Rotation.from_euler("z", SIDE_YAW_DEG[side], degrees=True)
        * Rotation.from_quat(DOWNWARD_X_FINGERS)
    )
    return turned.as_quat().astype(np.float32)


def build_scene_class(reference, utils):
    """Create the M01 scene on top of the frozen W1 two-way reference scene."""
    import dexsim
    import warp as wp

    table_z = utils.TABLE_Z

    class BeamScene(utils.W1TwoWayScene):
        """Two W1 grippers clamp, bend, carry and release one soft beam."""

        gripper_rotation = DOWNWARD_X_FINGERS
        home_offset = np.array((0.0, 0.0, 0.12))
        minimum_task_frames = 960

        def __init__(self, world, args):
            captured = []
            original = utils.RotationTarget
            # The base class already solves IK while building, so set these first.
            self.rotations = captured
            self.tilt = 0.0

            def capture(*items, **options):
                # IK checks exact objective types, so keep the original class.
                target = original(*items, **options)
                captured.append(target)
                return target

            utils.RotationTarget = capture
            try:
                super().__init__(world, args)
            finally:
                utils.RotationTarget = original

        def initial_targets(self):
            z = table_z + self.args.grasp_height
            return np.array(((BEAM_X, GRIP_Y, z), (BEAM_X, -GRIP_Y, z)))

        def build_task(self, builder):
            self.grasp = self.initial_targets()
            for name, x, color in (
                ("start_mark", BEAM_X, (0.35, 0.55, 0.35)),
                ("target_mark", BEAM_X + CARRY_X, (0.55, 0.40, 0.30)),
            ):
                self.add_box(
                    builder,
                    name,
                    (x, 0.0, table_z - 0.0045),
                    (0.035, 0.25, 0.005),
                    color,
                )
            physics = dexsim.spawn.SoftBodyPhysicsDesc(
                volume_density=self.args.density,
                k_mu=self.args.k_mu,
                k_lambda=self.args.k_lambda,
                k_damp=self.args.k_damp,
                contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                    stiffness=2e4, damping=10.0, dynamic_friction=1.0
                ),
            )
            builder.materials["soft_beam"] = dexsim.spawn.MaterialDesc(
                name="soft_beam", base_color=(0.9, 0.4, 0.2, 1.0)
            )
            builder.add_softbody(
                "soft_beam",
                dexsim.spawn.RenderDesc.from_geometry(
                    dexsim.spawn.GeometryDesc.cube(tuple(BEAM_SIZE)),
                    visual_surface_resolution=8,
                    material_ref="soft_beam",
                ),
                pose=utils.pose_at(
                    (BEAM_X, 0.0, table_z + SPAWN_GAP + BEAM_SIZE[2] / 2)
                ),
                meshing=dexsim.spawn.SoftBodyMeshingDesc(
                    voxel_resolution=self.args.voxel_resolution
                ),
                physics=physics,
                particle_radius=0.003,
                per_env=False,
            )
            self.soft_paths.append("soft_beam")

        def _keyframes(self):
            above = self.grasp + np.array((0.0, 0.0, 0.08))
            lifted = self.grasp + np.array((0.0, 0.0, LIFT))
            bent = lifted + np.array(((0.0, -SHORTEN, 0.0), (0.0, SHORTEN, 0.0)))
            carried = lifted + np.array((CARRY_X, 0.0, 0.0))
            placed = self.grasp + np.array((CARRY_X, 0.0, 0.0))
            retreat = placed + np.array((0.0, 0.0, 0.10))
            o, s = self.args.finger_open, self.args.finger_close
            return [
                (0.0, self.home, o, 0.0),
                (0.5, self.home, o, 0.0),
                (2.0, above, o, 0.0),
                (3.0, self.grasp, o, 0.0),
                (4.0, self.grasp, s, 0.0),
                (4.4, self.grasp, s, 0.0),
                (6.0, lifted, s, 0.0),
                (6.5, lifted, s, 0.0),
                (8.0, bent, s, TILT_DEG),
                (10.0, bent, s, TILT_DEG),
                (11.5, lifted, s, 0.0),
                (13.5, carried, s, 0.0),
                (14.5, placed, s, 0.0),
                (15.2, placed, o, 0.0),
                (16.0, retreat, o, 0.0),
            ]

        def plan(self, time):
            targets, finger, tilt = utils.smoothstep(self._keyframes(), time)
            self.tilt = float(tilt)
            return targets, float(finger)

        def _solve_ik(self, targets, finger, *, iterations=12):
            # Left end (+Y) turns +tilt about X, right end -tilt: both inward.
            for side, objective in enumerate(self.rotations):
                objective.set_target_rotation(
                    0, wp.vec4(*side_rotation(side, self.tilt))
                )
            super()._solve_ik(targets, finger, iterations=iterations)

        def tracks_freely(self, time):
            return time <= 3.0 or time >= 15.2

        def particles(self):
            return (
                self.scene.get_particle_set("soft_beam")
                .get_particle_positions()
                .numpy()
            )

    return BeamScene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--steps", type=int, default=960)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--no-cuda-graph", action="store_true")
    for name, default in (
        ("arm-kp", 12000.0),
        ("arm-kd", 350.0),
        ("finger-kp", 1000.0),
        ("finger-kd", 20.0),
        ("finger-armature", 0.3),
        ("finger-damping", 50.0),
        ("proxy-relaxation", 0.5),
        ("proxy-mass-scale", 4.0),
        ("gravity-compensation", 1.0),
        ("density", 40.0),
        ("k-mu", 2.0e4),
        ("k-lambda", 5.0e4),
        ("k-damp", 1.0e-3),
        ("grasp-height", 0.03),
        ("finger-open", 0.05),
        ("finger-close", 0.0),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--voxel-resolution", type=int, default=48)
    return parser


def main() -> None:
    """Freeze the reference scene and record one preview."""
    args = _parser().parse_args()
    args.headless = True
    args.coupling = "two_way"
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source/reference"
    source.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_NAMES:
        shutil.copy2(REFERENCE / name, source / name)
    robot_root = output / "inputs/w1_v030"
    if not robot_root.exists():
        shutil.copytree(ROBOT, robot_root)
    shutil.copy2(__file__, output / "source/m01.py")
    args.w1_urdf = robot_root / "robot.urdf"
    sys.path.insert(0, str(source))
    utils = importlib.import_module("dexuni_w1_two_way_utils")

    import dexsim
    import imageio.v2 as imageio
    import newton
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont
    from dexsim.engine.newton_physics import DexUniSolverCfg, NewtonCfg
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    width, height = 1280, 720
    eye = (2.05, -1.05, 1.62)
    target = (0.36, 0.0, 0.98)
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
    BeamScene = build_scene_class(None, utils)

    class PreviewScene(BeamScene):
        """Add one camera, a light and neutral table colours."""

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
                    pose=utils.pose_at((1.5, -1.0, 3.0)),
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

    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    manifest = {
        "task": "M01",
        "mode": "native_scene_preview_only",
        "status": "recording",
        "user_scene_approved": False,
        "performance_measurements_started": False,
        "other_platforms_implemented": False,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executable": sys.executable,
        "dexsim_version": dexsim.__version__,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "robot": "W1 V030 parallel grippers, URDF finger effort limit 10 N",
        "scene": {
            "table_z": utils.TABLE_Z,
            "beam_size_xyz": BEAM_SIZE,
            "beam_center_xy": [BEAM_X, 0.0],
            "spawn_gap": SPAWN_GAP,
            "grip_y": GRIP_Y,
            "shorten_each_end": SHORTEN,
            "tilt_deg": TILT_DEG,
            "gripper_pitch_deg": GRIPPER_PITCH_DEG,
            "side_yaw_deg": SIDE_YAW_DEG,
            "lift": LIFT,
            "carry_x": CARRY_X,
            "phases": PHASES,
        },
        "camera": {"eye": eye, "target": target, "resolution": [width, height]},
        "reference_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in REFERENCE_NAMES
        },
        "robot_sha256": hashlib.sha256(args.w1_urdf.read_bytes()).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)
    world = dexsim.World(config)
    world.set_manual_update(True)
    scene = camera = writer = None
    states: dict[str, list] = {}
    metrics = []
    try:
        scene = PreviewScene(world, args)
        camera = scene.scene.get_camera("preview")
        vfov = 2 * np.arctan(np.tan(np.deg2rad(60) / 2) * height / width)
        camera.set_perspective(float(np.rad2deg(vfov)), width / height, 0.01, 10.0)
        camera.look_at(eye, target, (0, 0, 1))
        camera.open_camera()
        backend = get_newton_backend(world)
        model = backend.model
        rest = scene.particles()
        groups = {
            "left": np.flatnonzero(np.abs(rest[:, 1] - GRIP_Y) < GROUP_HALF),
            "right": np.flatnonzero(np.abs(rest[:, 1] + GRIP_Y) < GROUP_HALF),
            "middle": np.flatnonzero(np.abs(rest[:, 1]) < GROUP_HALF),
        }
        (output / "inputs").mkdir(exist_ok=True)
        np.savez_compressed(
            output / "inputs/soft_mesh.npz",
            **{
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
            },
            **{f"group_{k}": v for k, v in groups.items()},
        )
        manifest["mesh"] = {
            "particles": int(rest.shape[0]),
            "tetrahedra": int(model.tet_count),
            "group_sizes": {k: int(v.size) for k, v in groups.items()},
        }
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        if not args.pilot:
            writer = imageio.get_writer(
                str(preview / "dexsim.mp4"),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
        stills = {round(t * 60) for t in (0.0, 4.4, 4.9, 6.5, 9.0, 13.5, 15.0, 16.0)}
        for frame in range(args.steps + 1):
            if frame:
                scene.step()
            scene.check_finite()
            particles = scene.particles()
            row = beam_metrics(particles, groups)
            row.update(
                frame=frame,
                time=scene.sim_time,
                phase=_phase(scene.sim_time),
                finger_q=scene.finger_q(),
                tcp=scene.tcp_positions(),
                tcp_target=getattr(scene, "targets", scene.home),
                tilt_deg=scene.tilt,
            )
            row["tcp_error_m"] = np.linalg.norm(
                np.asarray(row["tcp"]) - np.asarray(row["tcp_target"]), axis=1
            )
            metrics.append(row)
            for key, value in (
                ("time", scene.sim_time),
                ("qpos", scene.robot.get_current_qpos()),
                ("target_q", scene._public_joint_row()),
                ("particles", particles),
                ("tcp", scene.tcp_positions()),
            ):
                states.setdefault(key, []).append(np.array(value, copy=True))
            if frame % 60 == 0:
                print(
                    f"[M01] t={scene.sim_time:5.2f} {row['phase']:9s} sag={row['sag_m'] * 1e3:6.1f}mm "
                    f"ends={row['end_distance_m'] * 1e3:6.1f}mm lowest={(row['lowest_z_m'] - utils.TABLE_Z) * 1e3:6.1f}mm "
                    f"finger={np.round(row['finger_q'] * 1e3, 1).tolist()}",
                    flush=True,
                )
            if (writer is not None and frame % 2 == 0) or frame in stills:
                backend.sync_to_dexsim(world)
                world._flush_render_updates()
                camera.render()
                image = Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 42), fill="black")
                draw.text(
                    (14, 10),
                    f"DexSim | M01 soft beam bend | t={scene.sim_time:.2f}s | {row['phase']} | sag {row['sag_m'] * 1e3:.1f} mm",
                    font=font,
                    fill="white",
                )
                if writer is not None and frame % 2 == 0:
                    writer.append_data(np.asarray(image))
                if frame in stills:
                    image.save(preview / f"still_{scene.sim_time:05.2f}s.jpg")
        manifest.update(
            status="completed",
            frames=args.steps,
            effective=scene.scene.get_solver_diagnostics(),
            peak_free_tcp_error_m=scene.peak_free_tcp_error,
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
        if states:
            np.savez_compressed(
                preview / "states.npz",
                **{key: np.stack(values) for key, values in states.items()},
            )
        if metrics:
            _write_json(preview / "metrics.json", metrics)
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(output / "manifest.json", manifest)
        if camera is not None:
            camera.close_camera()
        if scene is not None:
            scene.scene.close()
    print(f"Scene preview ready: {preview}", flush=True)


if __name__ == "__main__":
    main()
