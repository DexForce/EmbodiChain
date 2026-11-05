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
"""Preview W1 picking four independent cloths with one-way surface-fast DexUni.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m03.py
This native preview has no timing measurements. Every layer remains dynamic.
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
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m02 import (
    TrajectoryBaker,
)

__all__ = ["main"]

TECH = Path(__file__).resolve().parents[4]
OUTPUT = (
    TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place"
)
CONFIG = TECH / "assets/physics/dexuni/m03/m03_config.json"
PHASE_LABELS = {
    "hold": "hold 2s",
    "transport": "transport +150mm",
    "advance": "advance above target",
    "advance_hold": "stop swing",
    "land": "land far edge",
    "land_hold": "settle far edge",
    "lay": "lay stack flat",
}


def write_json(path: Path, value: Any) -> None:
    """Write detached preview metadata."""
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda v: np.asarray(v).tolist(),
        )
        + "\n"
    )


def digest(path: Path) -> str:
    """Hash one source or input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(xyz: Any) -> np.ndarray:
    """Translate a Spawn world matrix."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = xyz
    return matrix


def phase(time: float, config: dict[str, Any]) -> str:
    """Read the current fixed motion phase."""
    return next(
        (
            PHASE_LABELS.get(name, name)
            for name, end in config["motion"]["timing"].items()
            if time <= end + 1e-9
        ),
        "observe 2s",
    )


class LayerTrajectoryBaker(TrajectoryBaker):
    """Reuse W1 IK with horizontal fingers closing vertically around the stack."""

    def solve(
        self, targets: np.ndarray, finger: float, pitch: float, iterations: int
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        wp = self.wp
        for objective, point in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*point))
        # Both local Z axes point into the near edge along +X. Local Y is
        # vertical (the Pika prismatic axis); opposite roll keeps cameras apart.
        for side, objective in enumerate(self.rotations):
            sign = 1 if side == 0 else -1
            matrix = np.array(((0, 0, 1), (sign, 0, 0), (0, sign, 0)))
            rotation = (
                Rotation.from_euler(
                    "z", -sign * getattr(self, "side_yaw_deg", 0.0), degrees=True
                )
                * Rotation.from_euler("y", pitch, degrees=True)
                * Rotation.from_matrix(matrix)
            )
            objective.set_target_rotation(0, wp.vec4(*rotation.as_quat()))
        self.ik.step(self.q, self.q, iterations=iterations)
        q = np.clip(self.q.numpy()[0], self.lower, self.upper)
        for side in ("LEFT", "RIGHT"):
            for k in (1, 2):
                q[self.coords[f"{side}_FINGER{k}_JOINT"]] = finger
        self.q.assign(q.reshape(1, -1))
        return q


def prepare_inputs(
    config: dict[str, Any], output: Path, device: str
) -> dict[str, np.ndarray]:
    """Bake actual IK commands and independent equal-mass flat layer meshes."""
    import warp as wp

    robot_root = output / "inputs/w1_v030"
    if not robot_root.exists():
        shutil.copytree(Path(config["robot"]["urdf"]).parent, robot_root)
    cloth, motion, robot = config["cloth"], config["motion"], config["robot"]
    n = int(round(cloth["size"][0] / cloth["spacing"])) + 1
    axis = np.linspace(-cloth["size"][0] / 2, cloth["size"][0] / 2, n)
    xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    vertices = np.c_[xy + cloth["center_xy"], np.full(len(xy), cloth["bottom_z"])]
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            if (i < (n - 1) // 2) != (j < (n - 1) // 2):
                faces.extend(((a, a + n, a + n + 1), (a, a + n + 1, a + 1)))
            else:
                faces.extend(((a, a + n, a + 1), (a + n, a + n + 1, a + 1)))
    initial = np.stack(
        [vertices + (0, 0, k * cloth["layer_spacing"]) for k in range(cloth["layers"])]
    ).astype(np.float32)
    np.savez_compressed(
        output / "inputs/cloth_mesh.npz",
        initial_positions=initial,
        triangles=np.asarray(faces, np.int32),
    )
    points = lambda x, z: np.array(
        ((x, motion["grasp_y"], z), (x, -motion["grasp_y"], z))
    )
    grasp = points(motion["grasp_x"], motion["grasp_z"])
    grasp[:, 1] *= (
        float(motion.get("tcp_grasp_y", motion["grasp_y"])) / motion["grasp_y"]
    )
    lifted = points(motion["lift_x"], motion["lift_z"])
    moved = points(motion["transport_x"], motion["lift_z"])
    lay_start = points(motion["lay_start_x"], motion["lay_start_z"])
    above_lay = points(motion["lay_start_x"], motion["lift_z"])
    placed = points(motion["place_x"], motion["place_z"])
    home = grasp + np.asarray(motion.get("home_offset", (-0.09, 0, 0.12)))
    approach = grasp + np.asarray(motion.get("approach_offset", (-0.06, 0, 0)))
    release = placed + np.asarray(motion.get("release_offset", (-0.10, 0, 0)))
    retreat = placed + np.asarray(motion.get("retreat_offset", (-0.10, 0, 0.12)))
    o, c = robot["open_finger"], robot["closed_finger"]
    released = robot["release_finger"]
    returned = float(robot.get("retreat_finger", released))
    timing = motion["timing"]
    keyframes = [
        (0, home, o),
        (timing["settle"], home, o),
        (timing["approach"], approach, o),
        (timing["insert"], grasp, o),
        (timing["close"], grasp, c),
        (timing["lift"], lifted, c),
        (timing["hold"], lifted, c),
        (timing["transport"], moved, c),
        (timing["advance"], above_lay, c),
        (timing["advance_hold"], above_lay, c),
        (timing["land"], lay_start, c),
        (timing["land_hold"], lay_start, c),
        (timing["lay"], placed, c),
        (timing["release"], release, released),
        (timing["retreat"], retreat, returned),
        (motion["duration"], retreat, returned),
    ]
    # Tilt only near the support. Keeping the elevated transport horizontal
    # avoids the wrist limit encountered by a constant-pitch trajectory.
    grasp_pitch = float(motion.get("grasp_pitch_deg", 0.0))
    place_pitch = float(motion.get("place_pitch_deg", 0.0))
    release_pitch = float(motion.get("release_pitch_deg", place_pitch))
    retreat_pitch = float(motion.get("retreat_pitch_deg", place_pitch))
    pitches = (
        [grasp_pitch] * 5
        + [0.0] * 7
        + [place_pitch, release_pitch, retreat_pitch, retreat_pitch]
    )
    travel_yaw = float(motion.get("side_yaw_deg", 0.0))
    grasp_yaw = float(motion.get("grasp_yaw_deg", travel_yaw))
    yaws = [travel_yaw] * 2 + [grasp_yaw] * 3 + [travel_yaw] * 11
    time = (
        np.arange(round(motion["duration"] / config["solver"]["dt"]) + 1)
        * config["solver"]["dt"]
    )
    qrows, target_rows, actual_rows, gaps = [], [], [], []
    base = np.asarray(robot["base_position"])
    with wp.ScopedDevice(device):
        baker = LayerTrajectoryBaker(
            robot_root / "robot.urdf",
            SimpleNamespace(TCP=np.asarray(robot["tcp_offset"])),
            device,
        )
        baker.side_yaw_deg = yaws[0]
        baker.solve(home - base, o, pitches[0], 500)
        for frame, t in enumerate(time):
            interval = next(
                (i for i, b in enumerate(keyframes[1:]) if t <= b[0] + 1e-9),
                len(keyframes) - 2,
            )
            a, b = keyframes[interval : interval + 2]
            u = np.clip((t - a[0]) / (b[0] - a[0]), 0, 1)
            u = u**3 * (10 - 15 * u + 6 * u**2)
            target, finger = (1 - u) * a[1] + u * b[1], (1 - u) * a[2] + u * b[2]
            pitch = (1 - u) * pitches[interval] + u * pitches[interval + 1]
            baker.side_yaw_deg = (1 - u) * yaws[interval] + u * yaws[interval + 1]
            q = baker.solve(
                target - base, float(finger), float(pitch), 150 if frame == 0 else 30
            )
            qrows.append([q[baker.coords[name]] for name in baker.names])
            target_rows.append(target)
            actual_rows.append(baker.tcp() + base)
            gaps.append(baker.camera_gap())
            if frame % 240 == 0:
                print(f"bake {frame}/{len(time)-1}", flush=True)
        commands = dict(
            time=time,
            joint_names=np.asarray(baker.names),
            targets=np.asarray(qrows, np.float32),
            tcp_target=np.asarray(target_rows),
            tcp_baked=np.asarray(actual_rows),
        )
        q = np.asarray(qrows)
        arms = [
            baker.names.index(f"{s}_J{j}")
            for s in ("LEFT", "RIGHT")
            for j in range(1, 8)
        ]
        error = np.linalg.norm(commands["tcp_target"] - commands["tcp_baked"], axis=2)
        checks = dict(
            max_tcp_error_m=float(error.max()),
            max_arm_joint_step_rad=float(np.abs(np.diff(q[:, arms], axis=0)).max()),
            min_camera_gap_m=float(min(gaps)),
        )
        worst = np.unravel_index(
            np.abs(np.diff(q[:, arms], axis=0)).argmax(), (len(q) - 1, len(arms))
        )
        checks.update(
            worst_joint=baker.names[arms[worst[1]]], worst_frame=int(worst[0] + 1)
        )
        np.savez_compressed(output / "inputs/joint_commands.npz", **commands)
        write_json(output / "inputs/trajectory_checks.json", checks)
        if checks["max_tcp_error_m"] > 0.004 or checks["max_arm_joint_step_rad"] > 0.04:
            raise RuntimeError(f"M03 offline IK rejected: {checks}")
    return commands


def build_scene(
    world: Any,
    config: dict[str, Any],
    output: Path,
    commands: dict[str, np.ndarray],
    mesh: dict[str, np.ndarray],
    video: bool,
) -> Any:
    """Declare all collision supports, four cloth resources and kinematic W1."""
    import dexsim
    from dexsim.engine.newton_physics import (
        KinematicJointTrajectoryControl,
        ContactMaterialSchedule,
    )

    builder = dexsim.spawn.SceneBuilder(world)
    contact, cloth, support = config["contact"], config["cloth"], config["support"]
    builder.add_urdf(
        "w1",
        str(output / "inputs/w1_v030/robot.urdf"),
        pose=pose(config["robot"]["base_position"]),
        per_env=False,
        newton_collision=dexsim.spawn.NewtonCollisionDesc(
            ke=contact["shape_ke"], kd=contact["shape_kd"], mu=0.6, gap=0.002
        ),
    )
    builder.add_runtime_control(
        KinematicJointTrajectoryControl(
            "w1",
            commands["targets"],
            fps=1 / config["solver"]["dt"],
            joint_names=commands["joint_names"].tolist(),
            initialize_pose=True,
        )
    )
    builder.add_runtime_control(
        ContactMaterialSchedule(
            "w1",
            {
                "dynamic_friction": ((0, contact["gripper_friction"]),),
                "stiffness": ((0, contact["gripper_ke"]),),
                "damping": ((0, contact["gripper_kd"]),),
            },
            link_names=tuple(
                f"{side}_finger{k}" for side in ("left", "right") for k in (1, 2)
            ),
        )
    )

    def box(
        name: str, center: tuple, size: tuple, color: tuple, collision: bool = True
    ) -> None:
        geometry = dexsim.spawn.GeometryDesc.cube(size)
        desc = dexsim.spawn.CollisionDesc.from_geometry(geometry)
        desc.newton = dexsim.spawn.NewtonCollisionDesc(
            ke=contact["shape_ke"],
            kd=contact["shape_kd"],
            mu=contact["table_friction"],
            gap=0.002,
        )
        builder.materials[name] = dexsim.spawn.MaterialDesc(
            name=name, base_color=(*color, 1)
        )
        obj = builder.add_cube(
            name,
            size,
            pose=pose(center),
            physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
            collisions=[desc] if collision else [],
            per_env=False,
        )
        obj.renders[0].material_ref = name

    tz, pz = support["table_z"], support["pickup_z"]
    box("table", (0.49, 0, tz - 0.025), (0.68, 1.2, 0.05), (0.45, 0.44, 0.41))
    for x in (0.2, 0.78):
        for y in (-0.53, 0.53):
            box(
                f"leg_{x}_{y}",
                (x, y, (tz - 0.05) / 2),
                (0.05, 0.05, tz - 0.05),
                (0.22, 0.24, 0.27),
            )
    # Solid rear support plus three front strips leave two open channels
    # centered at +/-130 mm. No collision mask is changed during the episode.
    xe = support["channel_end_x"]
    box(
        "pickup_rear",
        ((xe + 0.79) / 2, 0, (tz + pz) / 2),
        (0.79 - xe, 0.36, pz - tz),
        (0.62, 0.58, 0.48),
    )
    half = support["channel_width"] / 2
    gy = config["motion"]["grasp_y"]
    segments = ((-0.18, -gy - half), (-gy + half, gy - half), (gy + half, 0.18))
    for i, (a, b) in enumerate(segments):
        box(
            f"pickup_front_{i}",
            ((0.265 + xe) / 2, (a + b) / 2, (tz + pz) / 2),
            (xe - 0.265, b - a, pz - tz),
            (0.62, 0.58, 0.48),
        )
    cx, cy = support["target_center_xy"]
    hs = support["target_size"] / 2
    for i, y in enumerate((cy - hs, cy + hs)):
        box(
            f"target_x_{i}",
            (cx, y, pz + 0.0005),
            (2 * hs, 0.004, 0.001),
            (0.12, 0.85, 0.75),
            False,
        )
    for i, x in enumerate((cx - hs, cx + hs)):
        box(
            f"target_y_{i}",
            (x, cy, pz + 0.0005),
            (0.004, 2 * hs, 0.001),
            (0.12, 0.85, 0.75),
            False,
        )
    for k in range(cloth["layers"]):
        name = f"cloth_{k}"
        builder.materials[name] = dexsim.spawn.MaterialDesc(
            name=name,
            base_color=(*cloth["colors"][k], 1),
            roughness=0.8,
            double_sided=True,
        )
        builder.add_cloth_object(
            dexsim.spawn.ClothDesc(
                name=name,
                mesh=dexsim.spawn.RenderDesc.from_geometry(
                    dexsim.spawn.GeometryDesc.mesh(
                        vertices=mesh["initial_positions"][k],
                        triangles=mesh["triangles"],
                    ),
                    material_ref=name,
                ),
                physics=dexsim.spawn.ClothPhysicsDesc(
                    surface_density=cloth["surface_density"],
                    tri_ke=cloth["tri_ke"],
                    tri_ka=cloth["tri_ka"],
                    tri_kd=cloth["tri_kd"],
                    tri_drag=0,
                    tri_lift=0,
                    edge_ke=cloth["edge_ke"],
                    edge_kd=cloth["edge_kd"],
                    contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                        stiffness=contact["particle_ke"],
                        damping=contact["particle_kd"],
                        dynamic_friction=contact["particle_friction"],
                    ),
                ),
                particle_radius=cloth["particle_radius"],
                per_env=False,
            )
        )
    if video:
        builder.add_light(
            dexsim.spawn.LightDesc(
                name="key",
                light_type=dexsim.render.LightType.POINT,
                pose=pose((1, -1, 3)),
                intensity=70,
                shadow=False,
            )
        )
        for name, resolution in (("perspective", (960, 720)), ("detail", (720, 720))):
            builder.add_camera(
                dexsim.spawn.CameraDesc(
                    name, resolution=resolution, near=0.01, far=10, headless=True
                )
            )
    return builder.finalize()


def observations(
    states: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    """Track each independent layer and report candidate criteria unfiltered."""
    time, particles = states["time"], states["particles"]
    timing = config["motion"]["timing"]
    reference = int(np.argmin(np.abs(time - timing["settle"])))
    center = particles.mean(2)
    minimum = particles[..., 2].min(2)
    hold = (time >= timing["lift"] - 1e-8) & (time <= timing["hold"] + 1e-8)
    target = np.asarray(config["support"]["target_center_xy"])
    half = config["support"]["target_size"] / 2
    layers = []
    for k in range(config["cloth"]["layers"]):
        final_error = float(np.linalg.norm(center[-1, k, :2] - target))
        fraction = float(
            (np.abs(particles[-1, k, :, :2] - target) <= half).all(1).mean()
        )
        clearance = float(minimum[hold, k].min() - config["support"]["pickup_z"])
        layers.append(
            dict(
                layer=f"cloth_{k}",
                hold_min_clearance_m=clearance,
                reference_centroid=center[reference, k],
                final_centroid=center[-1, k],
                final_x_motion_m=float(center[-1, k, 0] - center[reference, k, 0]),
                final_target_xy_error_m=final_error,
                final_inside_fraction=fraction,
                clears_50mm_for_2s=clearance >= 0.05,
                candidate_placement_pass=final_error <= 0.02 and fraction >= 0.95,
            )
        )
    relative = center[hold] - center[hold, :1]
    return dict(
        all_states_finite=all(bool(np.isfinite(v).all()) for v in states.values()),
        actual_end_time=float(time[-1]),
        layers=layers,
        hold_max_relative_centroid_change_m=float(
            np.linalg.norm(relative - relative[:1], axis=2).max()
        ),
        all_four_clear=all(r["clears_50mm_for_2s"] for r in layers),
        all_four_candidate_placement=all(r["candidate_placement_pass"] for r in layers),
        candidate_thresholds_only=True,
        user_scene_approved=False,
    )


def run(
    config: dict[str, Any],
    output: Path,
    commands: dict[str, np.ndarray],
    mesh: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    """Execute one full native scene and optionally render synchronized views."""
    import dexsim
    import newton
    import warp as wp
    from dexsim.engine.newton_physics import DexUniSolverCfg, NewtonCfg

    solver = config["solver"]
    wc = dexsim.WorldConfig()
    wc.use_default_physics, wc.open_windows = False, False
    wc.renderer = dexsim.types.Renderer.OFFLINERT
    wc.raytrace_config.spp = 1
    wc.newton_cfg = NewtonCfg(
        device=args.device,
        dt=solver["dt"],
        num_substeps=solver["num_substeps"],
        solver_cfg=DexUniSolverCfg(
            mujoco_articulation_paths=("w1",),
            joint_mode=solver["joint_mode"],
            coupling=solver["coupling"],
            contact_mode=solver["contact_mode"],
            vbd_preset=solver["vbd_preset"],
            particle_coloring_include_bending=solver[
                "particle_coloring_include_bending"
            ],
            particle_coloring_balance_colors=solver["particle_coloring_balance_colors"],
            vbd_options={
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": solver["self_contact_radius"],
                "particle_self_contact_margin": solver["self_contact_margin"],
                "particle_rest_shape_contact_exclusion_radius": solver[
                    "rest_shape_exclusion_radius"
                ],
                "particle_topological_contact_filter_threshold": 1,
                "particle_vertex_contact_buffer_size": 64,
                "particle_edge_contact_buffer_size": 64,
                "particle_collision_detection_interval": -1,
            },
            collision_options={
                "soft_contact_margin": config["contact"]["soft_contact_margin"]
            },
        ),
    )
    wc.newton_cfg.use_cuda_graph = solver["use_cuda_graph"]
    manifest = dict(
        task="M03",
        status="running",
        scope="native_scene_preview_only",
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        python=sys.executable,
        versions=dict(
            dexsim=dexsim.__version__, newton=newton.__version__, warp=wp.__version__
        ),
        user_scene_approved=False,
        performance_measurements_started=False,
        requested=solver,
        config_sha256=digest(output / "inputs/config.json"),
        source_sha256=digest(Path(__file__)),
    )
    write_json(output / "manifest.json", manifest)
    world = dexsim.World(wc)
    world.set_manual_update(True)
    scene, robot, layers, cameras = None, None, [], []
    rows = {name: [] for name in ("time", "particles", "velocities", "qpos", "tcp")}
    frames = []
    last = len(commands["time"]) - 1
    if args.steps is not None:
        last = min(last, args.steps)
    try:
        scene = build_scene(world, config, output, commands, mesh, not args.no_video)
        robot = scene.get_articulation("w1")
        layout = {
            item.name.rsplit("/", 1)[-1]: item.dof_start
            for item in robot.joint_dof_layout
        }
        joint_ids = [layout[name] for name in commands["joint_names"].tolist()]
        manifest["joint_state_order"] = commands["joint_names"].tolist()
        layers = [scene.get_cloth_particle_set(f"cloth_{k}") for k in range(4)]
        inv = np.stack([layer.get_inverse_masses().numpy() for layer in layers])
        if np.any(inv <= 0) or not np.isfinite(inv).all():
            raise RuntimeError("All four cloths must have dynamic nodes")
        masses = (1 / inv).sum(1)
        if not np.allclose(masses, 0.018, atol=1e-7):
            raise RuntimeError(f"Layer masses are not 18g: {masses}")
        manifest["initial_model"] = dict(
            paths=[layer.path for layer in layers],
            nodes_per_layer=[layer.particle_count for layer in layers],
            masses_kg=masses,
            total_mass_kg=float(masses.sum()),
            pinned_nodes=int((inv <= 0).sum()),
        )
        np.savez_compressed(
            output / "preview/initial_model.npz",
            inverse_mass=inv,
            particles=np.stack(
                [layer.get_particle_positions().numpy() for layer in layers]
            ),
            triangles=mesh["triangles"],
        )
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            # Publish the initialized t=0 joint pose without a physics step.
            get_newton_backend(world).sync_to_dexsim(world)
            world._flush_render_updates()

            for name, eye, target, up in (
                ("perspective", (1.7, -1.45, 1.65), (0.4, 0, 1.04), (0, 0, 1)),
                ("detail", (0.68, -0.68, 1.40), (0.44, 0, 1.08), (0, 0, 1)),
            ):
                camera = scene.get_camera(name)
                camera.set_perspective(
                    48, (960 if name == "perspective" else 720) / 720, 0.01, 10
                )
                camera.look_at(eye, target, up)
                camera.open_camera()
                cameras.append(camera)
            scene.set_environment_light_intensity(70)
            font = ImageFont.truetype("DejaVuSans.ttf", 21)
        with ExitStack() as stack:
            writer = (
                stack.enter_context(
                    imageio.get_writer(
                        str(output / "preview/dexsim.mp4"),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        macro_block_size=2,
                    )
                )
                if cameras
                else None
            )
            for frame in range(last + 1):
                capture = bool(cameras) and frame % 2 == 0
                if frame:
                    world.update(solver["dt"], sync_to_dexsim=capture)
                particles = np.stack(
                    [layer.get_particle_positions().numpy() for layer in layers]
                )
                velocities = np.stack(
                    [layer.get_particle_velocities().numpy() for layer in layers]
                )
                q = robot.get_current_qpos()[joint_ids]
                tcp = np.stack(
                    [
                        (
                            robot.get_link_pose(f"{side}_gripper_base")
                            @ np.r_[config["robot"]["tcp_offset"], 1]
                        )[:3]
                        for side in ("left", "right")
                    ]
                )
                t = scene.get_simulation_time()
                if not all(
                    np.isfinite(v).all() for v in (particles, velocities, q, tcp)
                ):
                    raise RuntimeError(f"Nonfinite state at frame {frame}")
                if abs(t - commands["time"][frame]) > 1e-7:
                    raise RuntimeError(f"Actual clock mismatch at {frame}: {t}")
                for key, value in (
                    ("time", t),
                    ("particles", particles),
                    ("velocities", velocities),
                    ("qpos", q),
                    ("tcp", tcp),
                ):
                    rows[key].append(np.array(value, copy=True))
                if capture:
                    views = []
                    for camera in cameras:
                        camera.render()
                        views.append(np.asarray(camera.get_rgb_map())[..., :3].copy())
                    panel = Image.fromarray(np.concatenate(views, 1))
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, panel.width, 45), fill="black")
                    draw.text(
                        (12, 11),
                        f"DexSim | M03 four independent cloths | one-way / surface-fast | t={t:.2f}s | {phase(t, config)}",
                        font=font,
                        fill="white",
                    )
                    writer.append_data(np.asarray(panel))
                    frames.append(t)
                    if frame in {0, last} | {
                        round(t / solver["dt"])
                        for t in config["motion"]["timing"].values()
                    }:
                        panel.save(output / f"preview/stage_{t:05.2f}.jpg")
                if frame % 120 == 0:
                    print(
                        f"frame={frame}/{last} t={t:.2f} min_z={particles[...,2].min(1).round(4)} centers={particles.mean(1).round(3)}",
                        flush=True,
                    )
        states = {key: np.stack(value) for key, value in rows.items()}
        observation = (
            observations(states, config)
            if last >= round(config["motion"]["timing"]["hold"] / solver["dt"])
            else {}
        )
        effective = scene.get_solver_diagnostics()
        if last and not (
            effective["cuda_graph_status"] == "captured"
            and not effective["features"]["two_way_coupling_enabled"]
            and effective["features"]["backend"] == "mjvbd_kinematic_soft"
            and effective["vbd_preset"] == "surface-fast"
            and effective["vbd"]["particle_enable_batched_jacobi"]
            and effective["vbd"]["particle_chebyshev_enabled"]
            and effective["vbd"]["multilevel_active"]
            and effective["vbd"]["surface_cache"]
            and effective["vbd"]["truncation_cache"]
            and effective["features"]["bending_solve_enabled"]
            and effective["vbd"]["particle_enable_self_contact"]
        ):
            raise RuntimeError(f"Fast surface policy is not effective: {effective}")
        manifest.update(
            status="completed",
            effective=effective,
            observations=observation,
            full_trajectory=last == len(commands["time"]) - 1,
            video_frames=len(frames),
            max_tcp_replay_error_m=float(
                np.linalg.norm(
                    states["tcp"] - commands["tcp_baked"][: last + 1], axis=2
                ).max()
            ),
            max_named_joint_replay_error=float(
                np.abs(states["qpos"] - commands["targets"][: last + 1]).max()
            ),
        )
        if (
            manifest["max_named_joint_replay_error"] > 1e-6
            or manifest["max_tcp_replay_error_m"] > 1e-6
        ):
            raise RuntimeError(
                "Kinematic replay does not match the baked named trajectory"
            )
        write_json(output / "preview/observations.json", observation)
        write_json(
            output / "preview/frame_times.json", dict(fps=30, actual_time=frames)
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if rows["time"]:
            np.savez_compressed(
                output / "preview/states.npz",
                **{k: np.stack(v) for k, v in rows.items()},
            )
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        for camera in cameras:
            camera.close_camera()
        cameras.clear()
        layers.clear()
        robot = None
        if scene is not None:
            scene.close()


def main() -> None:
    """Prepare one current task directory and run the native preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
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
            "M03 output must be inside outputs/benchmarks/tech_reports/dexuni"
        )
    for directory in ("inputs", "source", "preview"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (
            output / "inputs/config.json" if args.reuse_inputs else args.config
        ).read_text()
    )
    if not args.reuse_inputs:
        write_json(output / "inputs/config.json", config)
        commands = prepare_inputs(config, output, args.device)
    else:
        with np.load(output / "inputs/joint_commands.npz") as data:
            commands = {k: data[k] for k in data.files}
    with np.load(output / "inputs/cloth_mesh.npz") as data:
        mesh = {k: data[k] for k in data.files}
    shutil.copy2(__file__, output / "source/m03.py")
    import scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m02 as m02

    shutil.copy2(m02.__file__, output / "source/m02_trajectory_reference.py")
    import scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common as m02_common

    shutil.copy2(m02_common.__file__, output / "source/m02_common_reference.py")
    write_json(
        output / "source/manifest.json",
        dict(sha256={p.name: digest(p) for p in (output / "source").glob("*.py")}),
    )
    write_json(
        output / "inputs/manifest.json",
        dict(
            sha256={
                str(p.relative_to(output / "inputs")): digest(p)
                for p in (output / "inputs").rglob("*")
                if p.is_file() and p.name != "manifest.json"
            }
        ),
    )
    if not args.prepare_only:
        run(config, output, commands, mesh, args)


if __name__ == "__main__":
    main()
