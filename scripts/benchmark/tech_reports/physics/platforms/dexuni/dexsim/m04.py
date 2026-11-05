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
"""Preview M04: one physical W1/Pika hem-to-neckline T-shirt fold."""

from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    CONFIG,
    OUTPUT,
    ASSETS,
    REFERENCE,
    shirt_mesh,
    orientation,
    motion,
    phase,
    supports,
    write_json,
    digest,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m02 import (
    TrajectoryBaker,
)

__all__ = ["main"]


def pose(xyz: Any) -> np.ndarray:
    """Make a translated world matrix."""
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = xyz
    return result


class EdgeBaker(TrajectoryBaker):
    """Orient the current Pika fingers for a physical cloth pinch."""

    def solve(
        self, targets: np.ndarray, finger: float, pitch: float, iterations: int
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        wp = self.wp
        for objective, point in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*point))
        for side, objective in enumerate(self.rotations):
            sign = 1 if side == 0 else -1
            matrix = np.array(((0, 0, 1), (sign, 0, 0), (0, sign, 0)))
            rotation = (
                Rotation.from_euler("z", -sign * self.side_yaw_deg, degrees=True)
                * Rotation.from_euler("y", pitch, degrees=True)
                * Rotation.from_matrix(matrix)
                * Rotation.from_euler("z", getattr(self, "roll_deg", 0.0), degrees=True)
            )
            objective.set_target_rotation(0, wp.vec4(*rotation.as_quat()))
        self.ik.step(self.q, self.q, iterations=iterations)
        q = np.clip(self.q.numpy()[0], self.lower, self.upper)
        for side in ("LEFT", "RIGHT"):
            for k in (1, 2):
                q[self.coords[f"{side}_FINGER{k}_JOINT"]] = finger
        self.q.assign(q.reshape(1, -1))
        return q


def prepare(config: dict[str, Any], output: Path, device: str) -> None:
    """Freeze the unchanged garment, current W1 asset and checked named IK."""
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    robot = inputs / "w1_v030"
    if robot.exists():
        shutil.rmtree(robot)
    shutil.copytree(Path(config["robot"]["urdf"]).parent, robot)
    if digest(robot / "robot.urdf") != digest(Path(config["robot"]["urdf"])):
        raise RuntimeError("M04 must use the original unmodified Pika asset")
    garment = inputs / "shirt"
    if garment.exists():
        shutil.rmtree(garment)
    garment.mkdir()
    shutil.copy2(ASSETS / "shirt_mesh.txt", garment / "shirt_mesh.txt")
    shutil.copytree(Path(config["cloth"]["material_dir"]), garment / "FabricGray")
    mesh = shirt_mesh(config)
    np.savez_compressed(inputs / "cloth_mesh.npz", **mesh)
    write_json(inputs / "config.json", config)
    time = (
        np.arange(round(config["motion"]["phases"][-1][0] / config["solver"]["dt"]) + 1)
        * config["solver"]["dt"]
    )
    base = np.array(config["robot"]["base_position"])
    rows = []
    actual = []
    desired = []
    gaps = []
    with wp.ScopedDevice(device):
        baker = EdgeBaker(
            robot / "robot.urdf",
            SimpleNamespace(TCP=np.array(config["robot"]["tcp_offset"])),
            device,
        )
        baker.side_yaw_deg = config["motion"]["side_yaw_deg"]
        baker.roll_deg = config["motion"].get("roll_deg", 0.0)
        initial = baker.q.numpy()
        for name, value in config["robot"]["ik_seed"].items():
            initial[0, baker.coords[name]] = value
        baker.q.assign(initial)
        target, finger = motion(0, config)
        baker.solve(target - base, finger, config["motion"]["pitch_deg"], 600)
        for frame, t in enumerate(time):
            target, finger = motion(float(t), config)
            pitch, yaw = orientation(float(t), config)
            baker.side_yaw_deg = yaw
            q = baker.solve(
                target - base,
                finger,
                pitch,
                120 if frame == 0 else 30,
            )
            rows.append([q[baker.coords[name]] for name in baker.names])
            desired.append(target)
            actual.append(baker.tcp() + base)
            gaps.append(baker.camera_gap())
            if frame % 240 == 0:
                print(f"bake={frame}/{len(time)-1}", flush=True)
        rows = np.array(rows, np.float32)
        errors = np.linalg.norm(np.array(actual) - np.array(desired), axis=2)
        arms = [
            baker.names.index(f"{side}_J{k}")
            for side in ("LEFT", "RIGHT")
            for k in range(1, 8)
        ]
        check = dict(
            max_tcp_error_m=float(errors.max()),
            max_arm_joint_step_rad=float(np.abs(np.diff(rows[:, arms], axis=0)).max()),
            min_hand_camera_gap_m=float(min(gaps)),
        )
        write_json(inputs / "trajectory_checks.json", check)
        np.savez_compressed(
            inputs / "joint_commands.npz",
            time=time,
            joint_names=np.array(baker.names),
            targets=rows,
            tcp_target=np.array(desired),
            tcp_baked=np.array(actual),
        )
        if check["max_tcp_error_m"] > 0.002 or check["max_arm_joint_step_rad"] > 0.04:
            raise RuntimeError(f"M04 offline trajectory rejected: {check}")


def shape_damping(stiffness: float) -> float:
    """Translate the specified reference demo's legacy proportional damping."""
    return 0.5 * (0.05 + 1e-6) * (3e5 + stiffness) - 0.05


def build(
    world: Any,
    config: dict[str, Any],
    output: Path,
    commands: dict[str, np.ndarray],
    mesh: dict[str, np.ndarray],
    video: bool,
) -> tuple[Any, Any]:
    """Build unpinned reference T-shirt, fixed table and colliding W1/Pika."""
    import dexsim
    from scipy.spatial.transform import Rotation
    from dexsim.engine.newton_physics import (
        KinematicJointTrajectoryControl,
        ContactMaterialSchedule,
        ParticleDragControl,
        FilmDragControl,
    )

    c = config["contact"]
    r = config["robot"]
    builder = dexsim.spawn.SceneBuilder(world)
    robot = builder.add_urdf(
        "w1",
        str(output / "inputs/w1_v030/robot.urdf"),
        pose=pose(r["base_position"]),
        per_env=False,
        newton_collision=dexsim.spawn.NewtonCollisionDesc(
            ke=c["robot_ke"],
            kd=shape_damping(c["robot_ke"]),
            mu=c["open_mu"],
            gap=c["gap"],
            margin=c["margin"],
            force_sdf=True,
        ),
    )
    robot.enable_self_collision = True
    urdf = ET.parse(output / "inputs/w1_v030/robot.urdf").getroot()
    robot.filtered_link_pairs.extend(
        (link.get("name"), link.get("name"))
        for link in urdf.findall("link")
        if link.findall("collision")
    )
    control = KinematicJointTrajectoryControl(
        "w1",
        commands["targets"],
        fps=1 / config["solver"]["dt"],
        joint_names=commands["joint_names"].tolist(),
        initialize_pose=True,
    )
    builder.add_runtime_control(control)
    ends = {name: end for end, name in config["motion"]["phases"]}
    # Ease friction only after the accepted folding/hold motion has finished.
    # The SDK consumes step keyframes; sample the quintic ramp at control rate.
    release_start = ends["hold folded hem"]
    release_ramp = c.get("release_ramp_s", 0.0)
    if not 0 <= release_ramp <= ends["release"] - release_start:
        raise ValueError("Release friction ramp must fit inside the opening phase")
    fractions = np.linspace(
        0, 1, max(2, round(release_ramp / config["solver"]["dt"]) + 1)
    )
    release_track = tuple(
        (
            release_start + float(u) * release_ramp,
            float(
                c["grasp_mu"]
                + (c["release_mu"] - c["grasp_mu"]) * u**3 * (10 - 15 * u + 6 * u * u)
            ),
        )
        for u in fractions
    )
    builder.add_runtime_control(
        ContactMaterialSchedule(
            "w1",
            {
                "dynamic_friction": (
                    (0, c["open_mu"]),
                    (ends["descend"], c["grasp_mu"]),
                    *release_track,
                ),
                "stiffness": ((0, c["robot_ke"]),),
                "damping": ((0, shape_damping(c["robot_ke"])),),
            },
            link_names=tuple(
                f"{side}_finger{k}" for side in ("left", "right") for k in (1, 2)
            ),
        )
    )
    for group, (name, center, size) in enumerate(supports(config), start=1):
        geometry = dexsim.spawn.GeometryDesc.cube(size)
        col = dexsim.spawn.CollisionDesc.from_geometry(geometry)
        col.newton = dexsim.spawn.NewtonCollisionDesc(
            ke=c["support_ke"],
            kd=shape_damping(c["support_ke"]),
            mu=c["table_mu"],
            gap=c["gap"],
            margin=c["margin"],
            collision_group=group,
        )
        builder.add_cube(
            name,
            size,
            pose=pose(center),
            physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
            collisions=[col],
            material=dexsim.spawn.MaterialDesc(
                name=name,
                base_color=(
                    (0.53, 0.48, 0.39, 1) if name == "table" else (0.30, 0.31, 0.33, 1)
                ),
            ),
            per_env=False,
        )
    cloth = config["cloth"]
    cloth_pose = pose(cloth["position"])
    cloth_pose[:3, :3] = Rotation.from_euler(
        "z", cloth["yaw_deg"], degrees=True
    ).as_matrix()
    vertices, triangles = mesh["vertices"], mesh["triangles"]
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    uv = vertices[:, :2] - vertices[:, :2].min(0)
    uv /= np.maximum(uv.max(0), 1e-8)
    uv[:, 1] = 1 - uv[:, 1]
    visual = dexsim.spawn.GeometryDesc.mesh(
        vertices=vertices,
        triangles=triangles,
        normals=normals,
        uv_coords=uv * 2,
    )
    texture_dir = output / "inputs/shirt/FabricGray"
    builder.materials["cloth_default"] = dexsim.spawn.MaterialDesc(
        name="cloth_default",
        base_color=(1, 1, 1, 1),
        base_color_map=str(texture_dir / "FabricGray_Color.png"),
        normal_map=str(texture_dir / "FabricGray_Normal.png"),
        roughness_map=str(texture_dir / "FabricGray_Roughness.png"),
        roughness=0.8,
        metallic=0.0,
        double_sided=True,
    )
    builder.add_clothbody(
        "cloth",
        dexsim.spawn.GeometryDesc.mesh(
            vertices=mesh["vertices"], triangles=mesh["triangles"]
        ),
        visual_mesh=dexsim.spawn.RenderDesc.from_geometry(
            visual, material_ref="cloth_default"
        ),
        visual_binding_mode="nearest_vertex",
        pose=cloth_pose,
        particle_radius=cloth["particle_radius"],
        per_env=False,
        physics=dexsim.spawn.ClothPhysicsDesc(
            surface_density=cloth["surface_density"],
            tri_ke=cloth["tri_ke"],
            tri_ka=cloth["tri_ka"],
            tri_kd=cloth["tri_kd"],
            edge_ke=cloth["edge_ke"],
            edge_kd=cloth["edge_kd"],
            edge_rest_angle=cloth["edge_rest_angle"],
            contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                stiffness=c["particle_ke"],
                damping=c["particle_kd"],
                dynamic_friction=c["particle_mu"],
            ),
        ),
    )
    if cloth.get("air_drag"):
        builder.add_runtime_control(FilmDragControl("cloth", **cloth["air_drag"]))
    elif cloth.get("air_drag_rate", 0.0) > 0:
        builder.add_runtime_control(ParticleDragControl(cloth["air_drag_rate"]))
    if video:
        builder.add_light(
            dexsim.spawn.LightDesc(
                name="key",
                light_type=dexsim.render.LightType.POINT,
                pose=pose((1.5, -1, 3)),
                intensity=60,
                shadow=False,
            )
        )
        builder.add_camera(
            dexsim.spawn.CameraDesc(
                "preview", resolution=(1280, 720), near=0.01, far=10, headless=True
            )
        )
    return builder.finalize(), control


def run(config: dict[str, Any], output: Path, args: argparse.Namespace) -> None:
    """Record actual kinematic poses, cloth geometry and one main view."""
    import dexsim, newton, warp as wp
    from dexsim.engine.newton_physics import NewtonCfg, DexUniSolverCfg

    commands = dict(np.load(output / "inputs/joint_commands.npz"))
    mesh = dict(np.load(output / "inputs/cloth_mesh.npz"))
    c = config["contact"]
    solver = config["solver"]
    wc = dexsim.WorldConfig()
    wc.use_default_physics = False
    wc.open_windows = False
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
            particle_coloring="newton_mcs",
            vbd_options={
                "particle_enable_multilevel_correction": False,
                "particle_jacobi_polish_iterations": 1,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": solver["self_contact_radius"],
                "particle_self_contact_margin": solver["self_contact_margin"],
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
                **solver.get("vbd_options", {}),
            },
            collision_options={"soft_contact_margin": c["soft_margin"]},
        ),
    )
    wc.newton_cfg.use_cuda_graph = solver["use_cuda_graph"]
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    # A scratch rerun must not leave media or states from a different candidate.
    for pattern in ("dexsim.mp4", "stage_*.jpg", "states.npz", "rigid_contacts.json"):
        for stale in preview.glob(pattern):
            stale.unlink()
    manifest = dict(
        task="M04",
        status="running",
        scope="native_scene_preview_only; no performance measurements",
        user_scene_approved=False,
        performance_measurements_started=False,
        config=config,
        joint_state_order=commands["joint_names"].tolist(),
        versions=dict(
            dexsim=dexsim.__version__, newton=newton.__version__, warp=wp.__version__
        ),
        executable=sys.executable,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_json(output / "manifest.json", manifest)
    last = len(commands["time"]) - 1
    if args.until is not None:
        last = min(last, round(args.until / solver["dt"]))
    world = dexsim.World(wc)
    world.set_manual_update(True)
    scene = camera = writer = None
    rows = {}
    contact_records = []
    video_count = 0
    try:
        scene, control = build(world, config, output, commands, mesh, not args.no_video)
        world.enable_physics(True)
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(world)
        manifest["effective_proxy_body_feedback"] = getattr(
            backend.solver.vbd_solver, "proxy_body_feedback", None
        )
        vbd_model = backend.solver.vbd_solver.model
        colors = np.full(vbd_model.particle_count, -1, dtype=np.int32)
        for color, group in enumerate(vbd_model.particle_color_groups):
            colors[group.numpy()] = color
        stencils = vbd_model.edge_indices.numpy()
        stencils = stencils[(stencils >= 0).all(axis=1)]
        repeats = sum(len(np.unique(colors[row])) != 4 for row in stencils)
        manifest["particle_coloring_audit"] = {
            "colors": len(vbd_model.particle_color_groups),
            "interior_bending_stencils": len(stencils),
            "same_color_bending_stencils": int(repeats),
        }
        labels = vbd_model.shape_label
        indices = [i for i, name in enumerate(labels) if "finger" in name]
        manifest["initialized_contact_materials"] = {
            "scope": "read-only diagnostic of the VBD owning model",
            "shapes": [
                {
                    "label": labels[i],
                    **{
                        field: float(getattr(vbd_model, field).numpy()[i])
                        for field in (
                            "shape_material_ke",
                            "shape_material_kd",
                            "shape_material_mu",
                            "shape_margin",
                        )
                    },
                }
                for i in indices
            ],
            "particle_contact_materials": vbd_model.dexuni.contact_materials.numpy(),
        }
        robot = scene.get_articulation("w1")
        cloth = scene.get_particle_set("cloth")
        import torch
        from scipy.spatial.transform import Rotation

        batch = scene.create_articulation_batch([robot])
        link_names = list(batch.link_names_per_articulation[0])
        pose_buffer = torch.empty(
            (1, batch.link_width, 7), device=args.device, dtype=torch.float32
        )
        hand_names = [
            f"{side}_{name}"
            for side in ("left", "right")
            for name in ("gripper_base", "finger1", "finger2")
        ]
        hand_indices = [link_names.index(name) for name in hand_names]
        batch_validation = {"checked_frames": 0, "max_matrix_error": 0.0}
        manifest["link_pose_order"] = link_names
        query = (
            scene.create_contact_query([robot], match="any", capacity=8192)
            if args.contacts
            else None
        )
        layout = {
            item.name.rsplit("/", 1)[-1]: item.dof_start
            for item in robot.joint_dof_layout
        }
        ids = [layout[name] for name in commands["joint_names"]]
        inv = cloth.get_inverse_masses().numpy()
        if np.any(inv <= 0):
            raise RuntimeError("M04 must use dynamic unpinned garment")
        manifest["initial_model"] = dict(
            nodes=len(inv), mass_kg=float((1 / inv).sum()), pinned_nodes=0
        )
        np.savez_compressed(
            preview / "initial_model.npz",
            inverse_mass=inv,
            particles=cloth.get_particle_positions().numpy(),
            triangles=mesh["triangles"],
        )
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            get_newton_backend(world).sync_to_dexsim(world)
            world._flush_render_updates()
            camera = scene.get_camera("preview")
            camera.set_perspective(42, 1280 / 720, 0.01, 10)
            camera.look_at((1.45, -1.05, 1.85), (0.47, 0, 1.05), (0, 0, 1))
            camera.open_camera()
            scene.set_environment_light_intensity(70)
            writer = imageio.get_writer(
                str(preview / "dexsim.mp4"),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        for frame in range(last + 1):
            capture = (
                camera is not None
                and frame % 2 == 0
                and commands["time"][frame] >= config["motion"]["pre_roll_s"]
            )
            if frame:
                if solver["joint_mode"] == "dynamic":
                    control.set_target(commands["targets"][frame])
                world.update(solver["dt"], sync_to_dexsim=capture)
            p = cloth.get_particle_positions().numpy()
            vel = cloth.get_particle_velocities().numpy()
            q = robot.get_current_qpos()[ids]
            if batch.fetch_link_pose(pose_buffer) != 1:
                raise RuntimeError("M04 link-pose batch read failed")
            vectors = pose_buffer[0].cpu().numpy()
            link_poses = np.broadcast_to(np.eye(4), (len(link_names), 4, 4)).copy()
            link_poses[:, :3, :3] = Rotation.from_quat(vectors[:, :4]).as_matrix()
            link_poses[:, :3, 3] = vectors[:, 4:]
            if frame in (0, last) or any(
                abs(commands["time"][frame] - end) < 1e-7
                for end, _ in config["motion"]["phases"]
            ):
                scalar = np.array([robot.get_link_pose(name) for name in link_names])
                error = float(np.max(np.abs(link_poses - scalar)))
                if error > 1e-5:
                    raise RuntimeError(f"M04 batched pose mismatch: {error}")
                batch_validation["checked_frames"] += 1
                batch_validation["max_matrix_error"] = max(
                    batch_validation["max_matrix_error"], error
                )
            hand = link_poses[hand_indices]
            tcp = np.array(
                [
                    (hand[i] @ np.r_[config["robot"]["tcp_offset"], 1])[:3]
                    for i in (0, 3)
                ]
            )
            t = scene.get_simulation_time()
            if any(abs(t - end) < 1e-7 for end, _ in config["motion"]["phases"]):
                manifest.setdefault("contact_material_samples", []).append(
                    dict(
                        time=t,
                        vbd_finger_mu=[
                            float(vbd_model.shape_material_mu.numpy()[i])
                            for i in indices
                        ],
                        model_finger_mu_ptr=int(backend.model.shape_material_mu.ptr),
                        vbd_finger_mu_ptr=int(vbd_model.shape_material_mu.ptr),
                    )
                )
            values = dict(
                time=t,
                particles=p,
                velocities=vel,
                qpos=q,
                tcp=tcp,
                hand_poses=hand,
                link_poses=link_poses,
            )
            if any(not np.isfinite(v).all() for v in values.values()):
                raise RuntimeError(f"Nonfinite state at frame{frame}")
            if abs(t - commands["time"][frame]) > 1e-7:
                raise RuntimeError(f"Actual clock mismatch: {t}")
            for key, value in values.items():
                rows.setdefault(key, []).append(np.array(value, copy=True))
            if query is not None and frame and frame % 30 == 0:
                contact = query.fetch()
                pairs = {}
                for data, actors in zip(
                    contact.data[: contact.count].numpy(),
                    contact.actor_ids[: contact.count].numpy(),
                    strict=True,
                ):
                    names = [
                        (
                            str(actor)
                            if actor < 0
                            else f"{query.actor_info(int(actor)).path}/{query.actor_info(int(actor)).link_name}"
                        )
                        for actor in actors
                    ]
                    key = " | ".join(names)
                    pairs[key] = min(pairs.get(key, float("inf")), float(data[-1]))
                contact_records.append(
                    dict(time=t, pairs=pairs, dropped=contact.dropped_count)
                )
            if capture:
                camera.render()
                image = Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 42), fill="black")
                draw.text(
                    (12, 10),
                    f"DexSim | M04 T-shirt fold | {solver['coupling']} / {solver['joint_mode']} / {solver['vbd_preset']} | t={max(0.0,t-config['motion']['pre_roll_s']):.2f}s | {phase(t,config)}",
                    font=font,
                    fill="white",
                )
                writer.append_data(np.asarray(image))
                video_count += 1
                if frame in (0, last) or any(
                    abs(t - end) < 1e-7 for end, _ in config["motion"]["phases"]
                ):
                    image.save(preview / f"stage_{t:05.2f}.jpg")
            if frame % 120 == 0:
                print(
                    f"frame={frame}/{last} t={t:.2f} center={p.mean(0).round(4)} TCPz={tcp[:,2].round(4)}",
                    flush=True,
                )
        if writer is not None:
            writer.close()
            writer = None
        states = {key: np.stack(value) for key, value in rows.items()}
        np.savez_compressed(preview / "states.npz", **states)
        if query is not None:
            write_json(preview / "rigid_contacts.json", contact_records)
        effective = scene.get_solver_diagnostics()
        if (
            effective["cuda_graph_status"] != "captured"
            or effective["features"]["two_way_coupling_enabled"]
            != (solver["coupling"] == "two_way")
            or effective["vbd_preset"] != solver["vbd_preset"]
        ):
            raise RuntimeError("M04 graph/coupling not active")
        manifest.update(
            status=(
                "completed"
                if last == len(commands["time"]) - 1
                else "diagnostic_truncated"
            ),
            outer_steps=last,
            video_frames=video_count,
            actual_simulation_time_s=float(states["time"][-1]),
            effective=effective,
            batched_readback_validation=batch_validation,
            input_sha256={
                name: digest(output / "inputs" / name)
                for name in (
                    "config.json",
                    "cloth_mesh.npz",
                    "joint_commands.npz",
                    "w1_v030/robot.urdf",
                )
            },
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        write_json(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if writer is not None:
            writer.close()
        if camera is not None:
            camera.close_camera()
        if scene is not None:
            scene.close()


def main() -> None:
    """Bake or preview one current candidate in the canonical task directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bake-only", action="store_true")
    parser.add_argument("--reuse-inputs", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--until", type=float)
    parser.add_argument("--contacts", action="store_true")
    args = parser.parse_args()
    if args.until is not None and args.out.resolve() == OUTPUT.resolve():
        parser.error("Truncated previews require a scratch --out")
    config = copy.deepcopy(CONFIG)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.reuse_inputs:
        preparing = dict(
            task="M04",
            status="preparing",
            scope="native_scene_preview_only; no performance measurements",
            config=config,
        )
        write_json(args.out / "manifest.json", preparing)
        try:
            prepare(config, args.out, args.device)
        except BaseException as error:
            write_json(
                args.out / "manifest.json",
                {**preparing, "status": "preparation_failed", "error": repr(error)},
            )
            raise
        write_json(args.out / "manifest.json", {**preparing, "status": "baked_only"})
    else:
        frozen = json.loads((args.out / "inputs/config.json").read_text())
        if frozen != config:
            raise ValueError("Reused M04 inputs differ from requested candidate")
    if args.bake_only:
        return
    run(config, args.out, args)


if __name__ == "__main__":
    main()
