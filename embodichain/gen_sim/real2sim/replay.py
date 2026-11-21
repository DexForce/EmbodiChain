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
"""Physically replay a prepared Real2Sim package and export its full comparison.

Call ``replay_recording`` or use the packaged replay worker CLI.
Only initial object transforms are consumed. Source camera poses stay in world
coordinates, independently of the registered robot base transform.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Iterator
import xml.etree.ElementTree as ET

import numpy as np

from embodichain.data_pipeline.real2sim.adaptation.media import (
    export_comparison,
    create_video_stream as _stream,
    encode_video_frame as _encode,
)

__all__ = ["replay_recording", "export_native_inputs"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_native_inputs(source: Path, run: Path) -> Path:
    """Package explicit registered scene and refined commands, preserving clocks.

    Args:
        source: Prepared recording with robot metadata, original trajectory
            arrays, and source videos.
        run: Run directory containing ``scene.json``, a native scene export
            receipt and ``trajectory/trajectory_refined.npz``.

    Returns:
        New ``run/native_inputs`` directory. Videos and the native scene export
        are symlinked; source calibration arrays are preserved alongside the
        refined joints. Keep the referenced recording and export available.

    Raises:
        ValueError: Native Scene Engine export or RH20T timing is invalid.
        FileExistsError: The native input directory already exists.
    """
    source, run = source.resolve(), run.resolve()
    output = run / "native_inputs"
    output.mkdir(parents=True, exist_ok=False)
    meta = _read(source / "robot.json")
    meta["urdf"] = str((source / meta["urdf"]).resolve())
    _write(output / "robot.json", meta)
    scene = _read(run / "scene.json")
    for obj in scene["objects"]:
        obj["path"] = str((source / obj["path"]).resolve())
    _write(output / "scene.json", scene)
    receipt = run / "scene_export_source.json"
    native = (
        Path(_read(receipt)["path"])
        if receipt.exists()
        else source / "native_scene_export"
    )
    if not (native / "scene_config.json").is_file():
        native = run / "scene_engine/scene_export"
    if not (native / "scene_config.json").is_file():
        raise ValueError("Missing native Scene Engine export")
    (output / "native_scene_export").symlink_to(
        native.resolve(), target_is_directory=True
    )
    for video in source.glob("original_*.mp4"):
        (output / video.name).symlink_to(video.resolve())
    with np.load(source / "trajectory_original.npz", allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
        observation_times = arrays["timestamps"].copy()
    with np.load(
        run / "trajectory/trajectory_refined.npz", allow_pickle=False
    ) as refined:
        arrays.update({key: refined[key] for key in refined.files})
    if "acquisition_timestamps" in arrays and meta.get("robot_type", "").startswith(
        "rh20t-"
    ):
        acquisition = arrays["acquisition_timestamps"]
        if (
            acquisition.shape != (len(arrays["qpos"]),)
            or not len(acquisition)
            or not np.isfinite(acquisition).all()
            or not np.all(np.diff(acquisition) > 0)
        ):
            raise ValueError(
                "RH20T acquisition timestamps must be finite, match samples and increase"
            )
        arrays["observation_video_timestamps"] = observation_times
        arrays["timestamps"] = acquisition - acquisition[0]
    np.savez_compressed(output / "trajectory_refined.npz", **arrays)
    return output


@contextmanager
def _resources() -> Iterator[None]:
    lock_path = Path.home() / ".cache/embodichain/real2sim-gpu0.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.setpriority(os.PRIO_PROCESS, 0, max(15, os.getpriority(os.PRIO_PROCESS, 0)))
        cpus = set(sorted(os.sched_getaffinity(0))[:2])
        for thread in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(thread.name), cpus)
            except ProcessLookupError:
                pass
        yield


def _robot_model(source: Path, visual: Path | None, output: Path) -> Path:
    """Replace only visuals; keep original joints, inertias and collision shapes."""
    output.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(source)
    root = tree.getroot()
    if not root.findall("./link/collision"):
        raise ValueError("Physical robot URDF has no collision geometry")
    asset_dirs = {}

    def local_assets(node: ET.Element, model: Path) -> None:
        for resource in node.findall(".//mesh") + node.findall(".//texture"):
            path = (model.parent / resource.get("filename")).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.parent not in asset_dirs:
                folder = f"robot_assets_{len(asset_dirs)}"
                (output / folder).symlink_to(path.parent, target_is_directory=True)
                asset_dirs[path.parent] = folder
            resource.set("filename", str(Path(asset_dirs[path.parent]) / path.name))

    local_assets(root, source)
    if visual is not None:
        replacement = {
            link.get("name"): link
            for link in ET.parse(visual).getroot().findall("link")
        }
        for link in root.findall("link"):
            if link.get("name") not in replacement:
                raise ValueError(f"Visual URDF lacks link {link.get('name')}")
            for node in link.findall("visual"):
                link.remove(node)
            for node in replacement[link.get("name")].findall("visual"):
                node = copy.deepcopy(node)
                local_assets(node, visual)
                link.append(node)
    target = output / "robot.urdf"
    tree.write(target, encoding="unicode")
    return target


def _execute(
    source: Path,
    output: Path,
    *,
    visual_urdf: Path | None,
    diagnostic_initial: bool,
    renderer: str,
) -> None:
    import av
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation
    from embodichain.compute.trajectory import retime_to_control_grid
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationRootPropertiesCfg,
        CollisionPropertiesCfg,
        DefaultPhysicsCfg,
        JointDrivePropertiesCfg,
        LightCfg,
        MeshCfg,
        MeshCollisionCfg,
        RenderCfg,
        RigidBodyMaterialCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
        RobotCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.sensors.camera import CameraCfg
    from embodichain.lab.sim.sensors.contact import (
        ArticulationContactFilterCfg,
        ContactSensorCfg,
    )
    from embodichain.lab.sim.motion.execution import (
        JointTrajectoryPlaybackCfg,
        play_joint_trajectory,
    )
    from embodichain.lab.sim.scene.query import SceneView
    from embodichain.lab.sim.scene.validation import SettlingCfg, settle_scene

    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    videos, assets, diagnostics = (
        output / name for name in ("videos", "assets", "diagnostics")
    )
    for directory in (videos, assets / "scene", diagnostics):
        directory.mkdir(parents=True)
    scene, meta = _read(source / "scene.json"), _read(source / "robot.json")
    native = _read(source / "native_scene_export/scene_config.json")
    entries = {
        item["uid"]: item
        for kind in ("background", "rigid_object", "articulation")
        for item in native.get(kind, [])
    }
    background = {item["uid"] for item in native.get("background", [])}
    with np.load(source / "trajectory_refined.npz", allow_pickle=False) as data:
        trajectory = {key: data[key] for key in data.files}
    qpos, names = trajectory["qpos"].copy(), trajectory["joint_names"].tolist()
    joint_limit_clips = {}
    for joint in ET.parse(source / meta["urdf"]).getroot().findall("joint"):
        limit, name = joint.find("limit"), joint.get("name")
        if name not in names or limit is None or joint.get("type") == "continuous":
            continue
        index = names.index(name)
        original = qpos[:, index].copy()
        qpos[:, index] = np.clip(
            original,
            float(limit.get("lower", "-inf")),
            float(limit.get("upper", "inf")),
        )
        delta = qpos[:, index] - original
        if np.any(delta):
            joint_limit_clips[name] = dict(
                max_abs=float(abs(delta).max()),
                initial_delta=float(delta[0]),
                samples=int(np.count_nonzero(delta)),
            )
    times = trajectory["timestamps"] - trajectory["timestamps"][0]
    if (
        not np.isfinite(qpos).all()
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError(
            "Finite joints and strictly increasing source timestamps required"
        )
    urdf = _robot_model(source / meta["urdf"], visual_urdf, assets / "robot")
    base = np.asarray(scene.get("robot_base_pose_world", np.eye(4)), dtype=float)
    if base.shape != (4, 4) or not np.isfinite(base).all():
        raise ValueError("Invalid robot_base_pose_world")
    report = dict(
        status="running",
        source=str(source),
        robot_base_pose_world=base.tolist(),
        object_pose_commands_after_initialization=0,
        physical_success_verified=False,
        joint_limit_clips=joint_limit_clips,
    )
    sim, video, overview_video = None, None, None
    try:
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                width=640,
                height=480,
                render_cfg=RenderCfg(
                    renderer=renderer, spp=1, min_bounces=1, max_bounces=3
                ),
                physics_dt=1 / 600,
                physics_cfg=DefaultPhysicsCfg(
                    gravity=meta.get("gravity_m_s2", [0, 0, -9.81])
                ),
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
            )
        )
        objects, world_bounds = {}, []
        for obj in scene["objects"]:
            uid, matrix = obj["id"], np.asarray(obj["transform"], dtype=float)
            entry = entries[uid]
            mesh_scene = trimesh.load(
                source / obj["path"], force="scene", process=False
            )
            if len(mesh_scene.graph.nodes_geometry) != 1:
                raise ValueError(f"{uid}: expected one registered textured mesh")
            node = next(iter(mesh_scene.graph.nodes_geometry))
            node_pose, mesh_name = mesh_scene.graph[node]
            mesh = mesh_scene.geometry[mesh_name].copy()
            mesh.apply_transform(node_pose)
            world_bounds.append(trimesh.transform_points(mesh.vertices, matrix))
            scale, pose = np.linalg.norm(matrix[:3, :3], axis=0), matrix.copy()
            pose[:3, :3] /= scale
            if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
                mesh.apply_transform(matrix)
                pose, scale = np.eye(4), np.ones(3)
            material = mesh.visual.material
            texture = getattr(material, "baseColorTexture", None)
            texture_path = assets / "scene" / f"{uid}_texture.png"
            if texture is not None:
                texture.save(texture_path)
            color = getattr(material, "baseColorFactor", None)
            visual_material = VisualMaterialCfg(
                base_color=(
                    [1.0] * 4 if color is None else (np.asarray(color) / 255).tolist()
                ),
                base_color_texture=str(texture_path) if texture is not None else None,
                metallic=float(getattr(material, "metallicFactor", None) or 0),
                roughness=float(getattr(material, "roughnessFactor", None) or 0.7),
            )
            collision = MeshCollisionCfg.from_dict(
                entry.get("shape", {}).get("collision")
                or {"max_convex_hull_num": entry.get("max_convex_hull_num", 32)}
            )
            cfg = RigidObjectCfg.from_dict(
                dict(
                    uid=uid,
                    attrs=entry.get("attrs", {}),
                    body_type=(
                        "static"
                        if uid in background
                        else entry.get("body_type", "dynamic")
                    ),
                    asset_physics_mode="overlay",
                )
            )
            cfg.shape = MeshCfg(
                vertices=mesh.vertices,
                triangles=mesh.faces,
                normals=mesh.vertex_normals,
                uv_coords=mesh.visual.uv,
                collision=collision,
                visual_material=visual_material,
            )
            cfg.init_local_pose, cfg.body_scale = pose, tuple(scale)
            objects[uid] = sim.add_rigid_object(cfg)
        stiffness, damping, effort = {}, {}, {}
        definitions = {
            j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")
        }
        for name in names:
            joint = definitions[name]
            if meta["robot_type"] != "cobotmagic":
                finger = (
                    joint.get("type") == "prismatic"
                    or "gripper" in name
                    or "finger" in name
                )
                stiffness[name], damping[name] = (2000, 70) if finger else (30000, 600)
                limit = joint.find("limit")
                effort[name] = (
                    max(0.1, float(limit.get("effort", "40" if finger else "100")))
                    if limit is not None
                    else 100
                )
        robot_cfg = RobotCfg(
            uid="robot",
            fpath=str(urdf),
            init_local_pose=base,
            build_pk_chain=False,
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True, min_position_iters=32, min_velocity_iters=8
            ),
            joint_drive_props=JointDrivePropertiesCfg(
                drive_type="force",
                stiffness=stiffness,
                damping=damping,
                max_effort=effort,
            ),
            attrs=RigidBodyPhysicsCfg(
                collision_props=CollisionPropertiesCfg(
                    contact_offset=0.001, rest_offset=0.0
                ),
                material_props=RigidBodyMaterialCfg(
                    static_friction=0.95, dynamic_friction=0.9
                ),
            ),
        )
        if meta["robot_type"] == "cobotmagic":
            from embodichain_robots import CobotMagicCfg
            from embodichain.utils.string import resolve_matching_names_values

            preset = CobotMagicCfg.from_dict({})
            drives = preset.joint_drive_props.to_dict()
            # Preserve the recording's URDF, assembly and joint identities.
            # Only map the model-owned drive rules to its uppercase spelling.
            for field, rules in drives.items():
                if isinstance(rules, dict):
                    indices, _, values = resolve_matching_names_values(
                        rules, [name.lower() for name in names]
                    )
                    if len(indices) != len(names):
                        raise ValueError(
                            f"CobotMagic preset lacks {field} for input joints"
                        )
                    drives[field] = {
                        names[i]: value for i, value in zip(indices, values)
                    }
            robot_cfg.joint_drive_props = JointDrivePropertiesCfg.from_dict(drives)
            robot_cfg.attrs = copy.deepcopy(preset.attrs)
        robot = sim.add_robot(robot_cfg)
        if renderer != "no-render":
            sim.add_light(
                LightCfg(
                    uid="key", light_type="sun", direction=(-0.4, -0.5, -1), intensity=3
                )
            )
        sim.prepare()
        _write(
            diagnostics / "physics_config.json",
            dict(
                robot_model=str(urdf),
                drive_source=(
                    "embodichain_robots.CobotMagicCfg"
                    if meta["robot_type"] == "cobotmagic"
                    else "legacy replay settings and source URDF effort limits"
                ),
                joint_names=list(robot.joint_names),
                drives={
                    key: value.cpu().numpy().tolist()
                    for key, value in zip(
                        (
                            "stiffness",
                            "damping",
                            "max_effort",
                            "max_velocity",
                            "friction",
                            "armature",
                        ),
                        robot.get_joint_drive(),
                    )
                },
                objects={
                    uid: dict(
                        body_type=body.cfg.body_type,
                        mass_kg=body.get_mass().cpu().numpy().tolist(),
                        inertia_kg_m2=body.get_inertia().cpu().numpy().tolist(),
                        mesh=str(body.cfg.shape.fpath),
                    )
                    for uid, body in objects.items()
                },
            ),
        )
        sim.spawn_result.get_actor("default_plane").enable_collision(False)
        sim.set_ground_plane_visibility(False)
        contacts = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=list(objects),
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid="robot", link_name_list=list(robot.link_names)
                    )
                ],
                max_contacts_per_env=128,
            )
        )
        collision_links = [
            link.get("name")
            for link in ET.parse(urdf).getroot().findall("link")
            if link.find("collision") is not None
        ]
        tracked_ids = torch.cat(
            [contacts.get_actor_ids(uid) for uid in objects]
            + [contacts.get_actor_ids("robot", collision_links)],
            dim=1,
        )
        tracked_ids_cpu = tracked_ids[0].cpu().numpy()
        counterpart_ids = {uid: contacts.get_actor_ids(uid) for uid in objects}
        histories = {
            uid: contacts.create_history(
                "scene_contact_" + uid,
                tracked_ids,
                counterpart_ids=ids,
            )
            for uid, ids in counterpart_ids.items()
        }
        counterpart_ids_cpu = {
            uid: int(ids[0, 0]) for uid, ids in counterpart_ids.items()
        }
        ids = [robot.joint_names.index(name) for name in names]
        positions, _, intervals, _ = retime_to_control_grid(
            torch.tensor(qpos[None], dtype=torch.float32, device=robot.device),
            torch.tensor(
                np.diff(times, prepend=0)[None],
                dtype=torch.float64,
                device=robot.device,
            ),
            1 / 30,
        )
        robot.set_qpos(
            torch.tensor(qpos[0:1], dtype=torch.float32, device=robot.device),
            joint_ids=ids,
            target=False,
        )
        robot.set_qpos(positions[:, 0], joint_ids=ids)
        camera, overview_camera = None, None
        if renderer != "no-render":
            with av.open(str(source / "original_front.mp4")) as src:
                width, height = src.streams.video[0].width, src.streams.video[0].height
            intrinsic = trajectory["front_K"].copy()
            intrinsic[0] *= 640 / width
            intrinsic[1] *= 480 / height
            pose = trajectory["front_camera"][0]
            camera = sim.add_sensor(
                CameraCfg(
                    uid="source_front",
                    width=640,
                    height=480,
                    intrinsics=(
                        intrinsic[0, 0],
                        intrinsic[1, 1],
                        intrinsic[0, 2],
                        intrinsic[1, 2],
                    ),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        pos=tuple(pose[:3, 3]),
                        quat=tuple(Rotation.from_matrix(pose[:3, :3]).as_quat()),
                    ),
                )
            )
            video = av.open(str(videos / "simulation.mp4"), "w")
            stream = _stream(video, 640, 480)
            # Frame the scene and commanded robot workspace through the existing
            # camera API. Source-camera footage remains independently available.
            import pytorch_kinematics as pk

            chain = pk.build_chain_from_urdf(urdf.read_bytes()).to(dtype=torch.float64)
            samples = qpos[:: max(1, len(qpos) // 100)]
            links = chain.forward_kinematics(
                {
                    name: torch.as_tensor(samples[:, i], dtype=torch.float64)
                    for i, name in enumerate(names)
                }
            )
            for link in links.values():
                centers = link.get_matrix().numpy()[:, :3, 3]
                world_bounds.append(trimesh.transform_points(centers, base))
            points = np.concatenate(world_bounds)
            lower, upper = points.min(axis=0), points.max(axis=0)
            target = (lower + upper) / 2
            direction = trajectory["front_camera"][0, :3, 3] - target
            direction[2] = max(
                abs(direction[2]), float(np.linalg.norm(direction[:2])) * 0.6, 1e-3
            )
            direction /= np.linalg.norm(direction)
            right = np.cross(-direction, [0, 0, 1])
            if np.linalg.norm(right) < 1e-8:
                right = np.array([1.0, 0.0, 0.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, -direction)
            relative = points - target
            projected_radius = np.maximum(
                abs(relative @ right) * 312 / 240,
                abs(relative @ up) * 312 / 180,
            )
            distance = max(
                float(np.max(relative @ direction + projected_radius * 1.2)), 0.1
            )
            eye = target + direction * distance
            overview_camera = sim.add_sensor(
                CameraCfg(
                    uid="overview",
                    width=480,
                    height=360,
                    intrinsics=(312, 312, 240, 180),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=tuple(eye),
                        target=tuple(target),
                        up=tuple(up),
                    ),
                )
            )
            _write(
                diagnostics / "overview_camera.json",
                dict(
                    eye=eye.tolist(),
                    target=target.tolist(),
                    up=up.tolist(),
                    source="scene geometry and commanded robot link bounds",
                    width=480,
                    height=360,
                    fps=10,
                ),
            )
            overview_video = av.open(str(videos / "overview.mp4"), "w")
            overview_stream = _stream(overview_video, 480, 360)
        observed, poses, timestamps, contact_rows, render_times = [], [], [], [], []
        interval_contacts = []
        started = time.monotonic()
        if diagnostic_initial:
            import pytorch_kinematics as pk

            chain = pk.build_chain_from_urdf((source / meta["urdf"]).read_bytes()).to(
                dtype=torch.float64
            )
            fk = chain.forward_kinematics(
                {
                    name: torch.tensor([qpos[0, i]], dtype=torch.float64)
                    for i, name in enumerate(names)
                }
            )
            link_checks = {}
            for link in robot.link_names:
                actual = robot.get_link_pose(link, to_matrix=True)[0].cpu().numpy()
                expected = base @ fk[link].get_matrix()[0].numpy()
                link_checks[link] = dict(
                    actual=actual.tolist(),
                    expected=expected.tolist(),
                    max_abs_error=float(abs(actual - expected).max()),
                )
            _write(
                diagnostics / "initial_native_poses.json",
                dict(
                    base_actual=robot.get_local_pose(to_matrix=True)[0]
                    .cpu()
                    .numpy()
                    .tolist(),
                    base_expected=base.tolist(),
                    qpos_actual=robot.get_qpos()[0, ids].cpu().numpy().tolist(),
                    qpos_expected=qpos[0].tolist(),
                    links=link_checks,
                ),
            )

        def finite_state() -> None:
            values = [
                robot.get_qpos(),
                robot.get_qvel(),
                robot.body_data.body_link_pose,
                robot.body_data.body_link_vel,
            ]
            values.extend(
                body.get_local_pose(to_matrix=True) for body in objects.values()
            )
            values.extend(
                body.body_data.vel
                for uid, body in objects.items()
                if uid not in background
            )
            if any(not bool(torch.isfinite(value).all()) for value in values):
                raise FloatingPointError(
                    f"Non-finite physical state at {sim.simulation_time:.6f}s"
                )

        class CheckedSimulation:
            sim_config = sim.sim_config
            startup_steps = 0

            def startup_check(self, dt: float) -> None:
                self.startup_steps += 1
                if diagnostic_initial and self.startup_steps <= 3:
                    contacts.update()
                    data = contacts.get_data()
                    valid = data["is_valid"][0].cpu().numpy().astype(bool)
                    sample = {
                        key: data[key][0].cpu().numpy()[valid]
                        for key in ("user_ids", "position", "impulse", "distance")
                    }
                    np.savez_compressed(
                        diagnostics / f"startup_contacts_{self.startup_steps}.npz",
                        **sample,
                    )
                    actors = {
                        int(actor) for pair in sample["user_ids"] for actor in pair
                    }
                    _write(
                        diagnostics / f"startup_actors_{self.startup_steps}.json",
                        {
                            str(actor): str(contacts.get_actor_info(actor))
                            for actor in actors
                        },
                    )
                if self.startup_steps <= 20 and not bool(
                    torch.isfinite(robot.get_qpos()).all()
                ):
                    np.savez_compressed(
                        diagnostics / "failed_initial_state.npz",
                        qpos=robot.get_qpos().cpu().numpy(),
                        time=sim.simulation_time,
                    )
                    raise FloatingPointError(
                        f"Non-finite initial joint state at {sim.simulation_time:.6f}s"
                    )

            def update(self, dt: float, steps: int) -> None:
                # Reuse the native sensor's GPU reductions. Read their interval
                # aggregates only at the control observation boundary.
                contacts.begin_control_step()

                def after_step(physics_dt: float) -> None:
                    contacts.update_physics_step(physics_dt)
                    if self.startup_steps < 20:
                        self.startup_check(physics_dt)

                sim.update(dt, steps, after_substep=after_step)

        checked_sim = CheckedSimulation()
        # Registration preserves the generated layout, but its scale and gravity
        # frame can change physical support. Initialize in the actual replay
        # world, holding the first robot command, before starting the source clock.
        # Keep this motion as a separate, unedited artifact; never teleport bodies
        # or reuse an equilibrium measured with different collision geometry.
        initialization_times, initialization_poses, initialization_joints = [], [], []
        initialization_video = (
            av.open(str(videos / "initialization.mp4"), "w")
            if overview_camera is not None
            else None
        )
        initialization_stream = (
            _stream(initialization_video, 480, 360)
            if initialization_video is not None
            else None
        )
        initialization_frames = 0
        initialization_steps = 0

        def record_initialization() -> None:
            nonlocal initialization_frames
            finite_state()
            now = float(sim.simulation_time)
            if initialization_times and abs(now - initialization_times[-1]) < 1e-9:
                return
            initialization_times.append(now)
            initialization_poses.append(
                np.stack(
                    [
                        body.get_local_pose(to_matrix=True)[0].cpu().numpy().copy()
                        for body in objects.values()
                    ]
                )
            )
            initialization_joints.append(robot.get_qpos()[0, ids].cpu().numpy().copy())
            if (
                initialization_video is not None
                and sim.simulation_time + 1e-7 >= initialization_frames / 10
            ):
                overview_camera.update()
                _encode(
                    initialization_video,
                    initialization_stream,
                    overview_camera.get_data()["color"][0].cpu().numpy()[..., :3],
                    initialization_frames,
                )
                initialization_frames += 1

        def initialize_step(count: int) -> None:
            nonlocal initialization_steps
            checked_sim.update(sim.sim_config.physics_dt, count)
            initialization_steps += count
            if initialization_steps % 20 == 0:
                record_initialization()

        record_initialization()
        try:
            stability = settle_scene(
                SceneView(sim),
                initialize_step,
                SettlingCfg(
                    mode="convergence",
                    max_steps=3000,
                    linear_threshold=0.001,
                    angular_threshold=0.01,
                    joint_threshold=0.01,
                    consecutive_steps=120,
                ),
            )
            record_initialization()
            initialization = dict(
                passed=stability.passed,
                duration_s=float(sim.simulation_time),
                method="native settle_scene with first joint target held",
                object_names=list(objects),
                initial_object_poses=initialization_poses[0].tolist(),
                settled_object_poses=initialization_poses[-1].tolist(),
                checks=[
                    dict(
                        name=c.name,
                        status=c.status,
                        message=c.message,
                        metrics=dict(c.metrics),
                    )
                    for c in stability.checks
                ],
            )
            _write(diagnostics / "initialization.json", initialization)
            report["initialization"] = initialization
            if not stability.passed:
                raise RuntimeError(
                    "Native initialization did not reach velocity convergence"
                )
        finally:
            np.savez_compressed(
                diagnostics / "initialization.npz",
                timestamps=initialization_times,
                object_names=list(objects),
                object_poses=initialization_poses,
                qpos_observed=initialization_joints,
            )
            if initialization_video is not None:
                for packet in initialization_stream.encode():
                    initialization_video.mux(packet)
                initialization_video.close()
        playback_epoch = float(sim.simulation_time)

        def observe(index: int, count: int) -> None:
            finite_state()
            now = float(sim.simulation_time) - playback_epoch
            if timestamps and abs(now - timestamps[-1]) < 1e-9:
                return
            observed.append(robot.get_qpos()[0, ids].cpu().numpy().copy())
            poses.append(
                np.stack(
                    [
                        body.get_local_pose(to_matrix=True)[0].cpu().numpy()
                        for body in objects.values()
                    ]
                )
            )
            timestamps.append(now)
            contacts.update()
            data = contacts.get_data()
            valid = data["is_valid"][0].cpu().numpy().astype(bool)
            row = {
                key: data[key][0].cpu().numpy()[valid].tolist()
                for key in ("user_ids", "position", "impulse", "distance")
            }
            if any(not np.isfinite(np.asarray(value)).all() for value in row.values()):
                raise FloatingPointError("Non-finite contact state")
            contact_rows.append(dict(timestamp=now, **row))
            found = torch.stack([h.found[0] for h in histories.values()]).cpu().numpy()
            peaks = (
                torch.stack([h.peak_force[0] for h in histories.values()]).cpu().numpy()
            )
            pairs = {}
            for i, uid in enumerate(histories):
                for j in np.flatnonzero(found[i]):
                    actor, other = int(tracked_ids_cpu[j]), counterpart_ids_cpu[uid]
                    key = tuple(sorted((actor, other)))
                    force = peaks[i, j] if actor == key[0] else -peaks[i, j]
                    if key not in pairs or np.linalg.norm(force) > np.linalg.norm(
                        pairs[key]
                    ):
                        pairs[key] = force
            interval_contacts.append(
                dict(
                    timestamp=now,
                    start_timestamp=timestamps[-2] if len(timestamps) > 1 else now,
                    dropped_contact_rows=contacts.evidence_dropped_contacts,
                    user_ids=[list(pair) for pair in pairs],
                    peak_force_on_first_actor_n=[
                        force.tolist() for force in pairs.values()
                    ],
                )
            )
            if camera is not None and now + 1e-7 >= len(render_times) / 10:
                camera.update()
                if not render_times:
                    av.VideoFrame.from_ndarray(
                        np.ascontiguousarray(
                            camera.get_data()["color"][0].cpu().numpy()[..., :3]
                        ),
                        format="rgb24",
                    ).to_image().save(diagnostics / "initial.png")
                _encode(
                    video,
                    stream,
                    camera.get_data()["color"][0].cpu().numpy()[..., :3],
                    len(render_times),
                )
                overview_camera.update()
                _encode(
                    overview_video,
                    overview_stream,
                    overview_camera.get_data()["color"][0].cpu().numpy()[..., :3],
                    len(render_times),
                )
                render_times.append(now)
            if index % 150 == 0:
                print(
                    f"Physics {now:.2f}/{times[-1]:.2f}s; wall {time.monotonic()-started:.1f}s",
                    flush=True,
                )

        observe(0, len(times))
        play_joint_trajectory(
            checked_sim,
            robot,
            positions=positions,
            dt=intervals,
            joint_ids=ids,
            cfg=JointTrajectoryPlaybackCfg(
                control_dt=1 / 30,
                joint_command_mode="position_velocity",
                preserve_source_grid=True,
            ),
            on_step=observe,
        )
        np.savez_compressed(
            diagnostics / "physics_rollout.npz",
            timestamps=timestamps,
            joint_names=names,
            qpos_observed=observed,
            qpos_command=positions[0].cpu().numpy(),
            command_timestamps=np.cumsum(intervals[0].cpu().numpy()),
            object_names=list(objects),
            object_poses=poses,
            source_timestamps=times,
            source_frame_indices=trajectory.get(
                "source_frame_indices", np.arange(len(times))
            ),
        )
        actor_ids = {
            int(actor)
            for row in (*contact_rows, *interval_contacts)
            for pair in row["user_ids"]
            for actor in pair
        }
        _write(
            diagnostics / "contacts.json",
            dict(
                actors={
                    str(actor): str(contacts.get_actor_info(actor))
                    for actor in actor_ids
                },
                samples=contact_rows,
                sampling=dict(
                    physics_dt_s=sim.sim_config.physics_dt,
                    control_dt_s=1 / 30,
                    samples="last native physics substep at each observation",
                    interval_samples="scene-object contacts found over all native substeps in the control interval",
                    peak_force_frame="arena coordinates, force on first actor, newtons",
                ),
                interval_samples=interval_contacts,
            ),
        )
        _write(diagnostics / "render_times.json", dict(timestamps=render_times))
        if video is not None:
            for packet in stream.encode():
                video.mux(packet)
            video.close()
            video = None
            for packet in overview_stream.encode():
                overview_video.mux(packet)
            overview_video.close()
            overview_video = None
            export_comparison(source, output)
        report.update(
            status="physics_executed_task_success_unverified",
            physics_samples=len(observed),
            duration_s=timestamps[-1],
            wall_s=time.monotonic() - started,
            frames=len(render_times),
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        if video is not None:
            video.close()
        if overview_video is not None:
            overview_video.close()
        _write(output / "report.json", report)
        if sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


def replay_recording(
    source: Path,
    output: Path,
    *,
    visual_urdf: Path | None = None,
    diagnostic_initial: bool = False,
    renderer: str = "rt",
) -> Path:
    """Replay prepared commands through native simulation with existing defaults.

    Args:
        source: Prepared native replay package.
        output: Fresh output directory for videos, runtime assets and diagnostics.
        visual_urdf: Optional robot visuals; physical joints/collisions are retained.
        diagnostic_initial: Export only the existing initialization diagnostic.
        renderer: Existing ``rt`` or ``no-render`` mode.

    Returns:
        Replay directory. Completion does not establish task success.
    """
    if renderer not in ("rt", "no-render"):
        raise ValueError("renderer must be rt or no-render")
    with _resources():
        _execute(
            source,
            output,
            visual_urdf=visual_urdf,
            diagnostic_initial=diagnostic_initial,
            renderer=renderer,
        )
    return output.resolve()


def main() -> None:
    """Execute prepared input under the shared GPU worker lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--visual-urdf", type=Path)
    parser.add_argument("--diagnostic-initial", action="store_true")
    parser.add_argument("--renderer", choices=("rt", "no-render"), default="rt")
    args = parser.parse_args()
    replay_recording(
        args.source,
        args.output,
        visual_urdf=args.visual_urdf,
        diagnostic_initial=args.diagnostic_initial,
        renderer=args.renderer,
    )


if __name__ == "__main__":
    main()
