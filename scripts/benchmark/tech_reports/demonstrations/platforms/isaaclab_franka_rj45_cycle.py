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
"""Replay actual Franka RJ45 joint targets on Isaac Lab's native Newton VBD."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import newton
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from franka_rj45_cycle_inputs import (
    JOINT_NAMES,
    load_asset_manifest,
    load_commands,
    load_measurements,
    load_scene,
    save_results,
    scene_inputs,
)
from franka_rope_wrap_inputs import file_info
from isaaclab_franka_rope_wrap import (
    _compensate_gravity,
    _initialize_rest_poses,
    _shape_config,
)

__all__: list[str] = []


@dataclass(frozen=True)
class _Scene:
    joints: list[int]
    robot_bodies: dict[str, int]
    robot_root_pose: np.ndarray
    tcp_body: int
    tcp_local: np.ndarray
    plug_bodies: dict[str, int]
    socket_body: int
    latch_joint: int
    cable_bodies: list[int]
    cable_lengths: np.ndarray


def _transform(matrix: np.ndarray) -> wp.transform:
    return wp.transform(matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat())


def _matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def _add_articulation(
    builder: newton.ModelBuilder,
    path: Path,
    cfg: dict[str, Any],
    pose: np.ndarray,
    *,
    floating: bool,
    collapse: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    first_body, first_joint = builder.body_count, builder.joint_count
    default = builder.default_shape_cfg
    builder.default_shape_cfg = _shape_config(builder, cfg["attrs"])
    try:
        builder.add_urdf(
            str(path),
            xform=_transform(pose),
            floating=floating,
            collapse_fixed_joints=collapse,
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
            parse_visuals_as_colliders=False,
            force_position_velocity_actuation=True,
        )
    finally:
        builder.default_shape_cfg = default
    return (
        {
            str(builder.body_label[i]).rsplit("/", 1)[-1]: i
            for i in range(first_body, builder.body_count)
        },
        {
            str(builder.joint_label[i]).rsplit("/", 1)[-1]: i
            for i in range(first_joint, builder.joint_count)
        },
    )


def _tcp(urdf: Path, bodies: dict[str, int]) -> tuple[int, np.ndarray]:
    root = ElementTree.parse(urdf).getroot()
    parents = {
        joint.find("child").get("link"): joint for joint in root.findall("joint")
    }
    name, local = "fr3_hand_tcp", np.eye(4)
    while name not in bodies:
        joint = parents[name]
        if joint.get("type") != "fixed":
            raise ValueError(f"Isaac Lab URDF import lost dynamic link {name}.")
        offset = np.eye(4)
        origin = joint.find("origin")
        if origin is not None:
            offset[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            offset[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
        local = offset @ local
        name = joint.find("parent").get("link")
    return bodies[name], local


def _build_scene(
    builder: newton.ModelBuilder,
    env: dict[str, Any],
    robot: dict[str, Any],
    assets: dict[str, Path],
) -> _Scene:
    scene = env["simulation"]
    if env["enable_ground_plane"]:
        builder.add_ground_plane()
    for item in scene["background"]:
        box = item["shape"]
        if box["shape_type"] != "Cube":
            raise ValueError(f"Unsupported RJ45 fixture shape: {box['shape_type']}")
        hx, hy, hz = np.asarray(box["size"]) / 2
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(item["init_pos"], wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=_shape_config(builder, item["attrs"]),
            label=item["uid"],
        )
    robot_pose = np.eye(4)
    robot_pose[:3, 3] = robot["init_pos"]
    robot_bodies, joints_by_name = _add_articulation(
        builder, assets["robot"], robot, robot_pose, floating=False, collapse=True
    )
    joints = [joints_by_name[name] for name in JOINT_NAMES]
    drives = robot["joint_drive_props"]
    for name, joint, home in zip(JOINT_NAMES, joints, robot["init_qpos"], strict=True):
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        group = "eef" if name.startswith("fr3_finger_joint") else "arm"
        builder.joint_q[q] = builder.joint_target_q[q] = home
        builder.joint_target_qd[dof] = 0.0
        builder.joint_target_ke[dof] = drives["stiffness"][group]
        builder.joint_target_kd[dof] = drives["damping"][group]
        builder.joint_limit_ke[dof] = 1e5
        builder.joint_limit_kd[dof] = 10.0
    tcp_body, tcp_local = _tcp(assets["robot"], robot_bodies)
    shapes = {item["uid"]: item for item in scene["articulation"]}
    plug_cfg = shapes["plug"]
    plug_pose = np.asarray(plug_cfg["init_local_pose"])
    plug, plug_joints = _add_articulation(
        builder, assets["plug"], plug_cfg, plug_pose, floating=True, collapse=False
    )
    socket, _ = _add_articulation(
        builder,
        assets["socket"],
        shapes["socket"],
        np.asarray(shapes["socket"]["init_local_pose"]),
        floating=False,
        collapse=False,
    )
    latch_joint = plug_joints["latch_hinge"]
    latch_dof = builder.joint_qd_start[latch_joint]
    latch_q = builder.joint_q_start[latch_joint]
    drive = plug_cfg["joint_drive_props"]
    builder.joint_target_ke[latch_dof] = drive["stiffness"]
    builder.joint_target_kd[latch_dof] = drive["damping"]
    builder.joint_limit_ke[latch_dof] = drive["limit_stiffness"]
    builder.joint_limit_kd[latch_dof] = drive["limit_damping"]
    builder.joint_target_q[latch_q] = builder.joint_target_qd[latch_dof] = 0.0
    latch_attrs = plug_cfg["link_attrs"]["latch_contact"]["attrs"]["material_props"]
    collision_counts = dict.fromkeys(("plug", "latch", "socket"), 0)
    for i, body in enumerate(builder.shape_body):
        if body == plug["latch"]:
            builder.shape_material_ke[i] = latch_attrs["ke"]
        if int(builder.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            for name, link in (
                ("plug", plug["plug"]),
                ("latch", plug["latch"]),
                ("socket", socket["socket"]),
            ):
                if body == link:
                    collision_counts[name] += 1
    if collision_counts != {"plug": 4, "latch": 3, "socket": 8}:
        raise RuntimeError(f"RJ45 collision components differ: {collision_counts}")

    (cable,) = scene["cable_object"]
    attrs = cable["attrs"]
    points = np.asarray(cable["points"], dtype=np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    stiffness = {
        f"{term}_{mode}": attrs[f"{term}_{mode}"]
        for term in ("stretch", "shear", "bend", "twist")
        for mode in ("stiffness", "damping")
    }
    first_shape = builder.shape_count
    bodies, _ = builder.add_rod(
        points.tolist(),
        radius=cable["radius"],
        cfg=_shape_config(builder, attrs),
        body_frame_origin="com",
        label=cable["uid"],
        **stiffness,
    )
    span = cable["collision_filter_span"]
    cable_shapes = list(range(first_shape, builder.shape_count))
    if len(cable_shapes) != len(lengths) or len(bodies) != len(lengths):
        raise RuntimeError("The imported RJ45 cable topology differs from the scene.")
    for index, shape in enumerate(cable_shapes):
        for neighbor in cable_shapes[index + 1 : index + span + 1]:
            builder.add_shape_collision_filter_pair(shape, neighbor)
    for index in cable["fixed_segments"]:
        builder.body_flags[bodies[index]] = int(newton.BodyFlags.KINEMATIC)
    attach = cable["end_attachment"]
    child_local = np.eye(4)
    child_local[2, 3] = lengths[-1] / 2
    end_world = _matrix(np.asarray(builder.body_q[bodies[-1]])) @ child_local
    parent_local = np.linalg.inv(plug_pose) @ end_world
    np.testing.assert_allclose(parent_local[:3, 3], attach["position"], atol=1e-6)
    builder.add_joint_fixed(
        parent=plug[attach["link_name"]],
        child=bodies[-1],
        parent_xform=_transform(parent_local),
        child_xform=_transform(child_local),
        collision_filter_parent=True,
        label="cable/plug_attachment",
    )
    return _Scene(
        joints,
        robot_bodies,
        np.asarray([*robot["init_pos"], 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        tcp_body,
        tcp_local,
        plug,
        socket["socket"],
        latch_joint,
        bodies,
        lengths,
    )


def _contacts(
    contacts: newton.Contacts,
    shape_bodies: np.ndarray,
    poses: np.ndarray,
    scene: _Scene,
) -> tuple[bool, bool, float, int]:
    raw_count = int(contacts.rigid_contact_count.numpy()[0])
    count = min(raw_count, contacts.rigid_contact_max)
    bodies = [
        shape_bodies[shapes.numpy()[:count]]
        for shapes in (contacts.rigid_contact_shape0, contacts.rigid_contact_shape1)
    ]
    tracked = [*scene.robot_bodies.values(), *scene.plug_bodies.values()]
    select = np.isin(bodies[0], tracked) | np.isin(bodies[1], tracked)
    if not select.any():
        return False, False, 0.0, raw_count - count
    first, second = (body[select] for body in bodies)
    world = []
    for body, points in zip(
        (first, second),
        (contacts.rigid_contact_point0, contacts.rigid_contact_point1),
        strict=True,
    ):
        locations = points.numpy()[:count][select]
        moving = body >= 0
        if moving.any():
            locations[moving] = (
                Rotation.from_quat(poses[body[moving], 3:]).apply(locations[moving])
                + poses[body[moving], :3]
            )
        world.append(locations)
    normal = contacts.rigid_contact_normal.numpy()[:count][select]
    distance = np.sum((world[1] - world[0]) * normal, axis=1)
    for field in (contacts.rigid_contact_margin0, contacts.rigid_contact_margin1):
        distance -= field.numpy()[:count][select]
    engaged = distance <= 0
    latch = scene.plug_bodies["latch"]
    other = np.r_[
        first[(second == latch) & engaged], second[(first == latch) & engaged]
    ]
    fingers = [scene.robot_bodies[f"fr3_{side}finger"] for side in ("left", "right")]
    return (
        bool(np.isin(other, fingers).any()),
        bool((other == scene.socket_body).any()),
        float(np.maximum(-distance, 0).max(initial=0)),
        raw_count - count,
    )


def _replay(
    sim: Any,
    manager: Any,
    scene: _Scene,
    config: dict[str, Any],
    home: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    measurements: SimpleNamespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    model, control = manager.get_model(), manager.get_control()
    q_indices = model.joint_q_start.numpy()[scene.joints]
    dof_indices = model.joint_qd_start.numpy()[scene.joints]
    target_indices = model.joint_target_q_start.numpy()[scene.joints]
    latch_q = model.joint_q_start.numpy()[scene.latch_joint]
    limits = (model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy())
    latch_dof = model.joint_qd_start.numpy()[scene.latch_joint]
    targets, speeds = control.joint_target_q.numpy(), control.joint_target_qd.numpy()
    measured_q, measured_qd = wp.clone(model.joint_q), wp.clone(model.joint_qd)
    parts_per_control = measurements.samples_per_control
    steps = len(positions)
    frames = steps + 1
    data = {
        "time_s": np.arange(frames) / config["control_hz"],
        "robot_q": np.empty((frames, len(home)), dtype=np.float32),
        "robot_qd": np.empty((frames, len(home)), dtype=np.float32),
        "robot_root_pose": np.broadcast_to(scene.robot_root_pose, (frames, 7)),
        "plug_pose": np.empty((frames, 7), dtype=np.float32),
        "latch_angle_rad": np.empty(frames, dtype=np.float32),
        "cable_body_pose": np.empty(
            (frames, len(scene.cable_bodies), 7), dtype=np.float32
        ),
        "cable_segment_lengths": scene.cable_lengths,
        "command_q": np.empty_like(positions),
        "command_qd": np.empty_like(velocities),
    }
    rows = []
    measure = measurements.measure_frame
    insertion_axis = np.asarray(measurements.parameters["insertion_rotation"])[:, 2]
    socket_mouth = np.asarray(measurements.parameters["socket_mouth_m"])
    # Match the source's declared contact sampling boundaries. The measurement
    # buffer never participates in the solve.
    pipeline = newton.CollisionPipeline(
        model, broad_phase="explicit", rigid_contact_max=4096, reduce_contacts=False
    )
    contacts = pipeline.contacts()
    shape_bodies = model.shape_body.numpy()
    anchor = manager.get_state().body_q.numpy()[scene.cable_bodies[0], :3].copy()

    def write_command(q: np.ndarray, qd: np.ndarray) -> None:
        targets[target_indices] = q
        speeds[dof_indices] = qd
        control.joint_target_q.assign(targets)
        control.joint_target_qd.assign(speeds)

    def observe() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = manager.get_state()
        newton.eval_ik(model, state, measured_q, measured_qd)
        return state.body_q.numpy(), measured_q.numpy(), measured_qd.numpy()

    def capture(
        frame: int,
        poses: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        *,
        finger: bool,
        socket: bool,
        penetration: float,
        dropped: int,
        latch_limit: float,
        press_samples: list[dict[str, float]],
    ) -> None:
        data["robot_q"][frame] = q[q_indices]
        data["robot_qd"][frame] = qd[dof_indices]
        data["plug_pose"][frame] = poses[scene.plug_bodies["plug"]]
        data["latch_angle_rad"][frame] = q[latch_q]
        data["cable_body_pose"][frame] = poses[scene.cable_bodies]
        if not (
            np.isfinite(poses).all() and np.isfinite(q).all() and np.isfinite(qd).all()
        ):
            raise FloatingPointError(
                f"Nonfinite state at replay control frame {frame}."
            )
        rows.append(
            measure(
                frame=frame,
                plug_pose=_matrix(data["plug_pose"][frame]),
                tcp_pose=_matrix(poses[scene.tcp_body]) @ scene.tcp_local,
                robot_qpos=q[q_indices],
                latch_angle_rad=float(q[latch_q]),
                cable_states=data["cable_body_pose"][frame],
                cable_lengths=scene.cable_lengths,
                anchor_baseline=anchor,
                finger_latch_contact=finger,
                latch_socket_contact=socket,
                latch_limit_violation_rad=latch_limit,
                penetration_m=penetration,
                dropped_contacts=dropped,
                press_samples=press_samples,
            )
        )

    write_command(home, np.zeros_like(home))
    for _ in range(
        round(config["pre_roll_s"] * config["control_hz"]) * parts_per_control
    ):
        sim.step(render=False)
    current = observe()
    capture(
        0,
        *current,
        finger=False,
        socket=False,
        penetration=0,
        dropped=0,
        latch_limit=0,
        press_samples=[],
    )
    for index, (q, qd) in enumerate(zip(positions, velocities, strict=True)):
        write_command(q, qd)
        data["command_q"][index] = control.joint_target_q.numpy()[target_indices]
        data["command_qd"][index] = control.joint_target_qd.numpy()[dof_indices]
        finger, socket, penetration, dropped, latch_limit = False, False, 0.0, 0, 0.0
        press_samples = []
        for _ in range(parts_per_control):
            sim.step(render=False)
            current = observe()
            pipeline.collide(manager.get_state(), contacts)
            f, s, p, d = _contacts(contacts, shape_bodies, current[0], scene)
            finger |= f
            socket |= s
            penetration = max(penetration, p)
            dropped += d
            angle = float(current[1][latch_q])
            if f:
                plug_position = current[0][scene.plug_bodies["plug"], :3]
                press_samples.append(
                    {
                        "depth_m": float(
                            insertion_axis @ (plug_position - socket_mouth)
                        ),
                        "latch_angle_deg": float(np.rad2deg(angle)),
                    }
                )
            latch_limit = max(
                latch_limit,
                angle - float(limits[1][latch_dof]),
                float(limits[0][latch_dof]) - angle,
            )
        capture(
            index + 1,
            *current,
            finger=finger,
            socket=socket,
            penetration=penetration,
            dropped=dropped,
            latch_limit=latch_limit,
            press_samples=press_samples,
        )
        if (index + 1) % 600 == 0:
            print(f"Completed {index + 1}/{steps} recorded controls", flush=True)
    np.testing.assert_array_equal(data["command_q"], positions)
    np.testing.assert_array_equal(data["command_qd"], velocities)
    return data, rows


def _run(
    device: str,
    config: dict[str, Any],
    env: dict[str, Any],
    robot: dict[str, Any],
    assets: dict[str, Path],
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    measurements: SimpleNamespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
    from isaaclab_newton.physics.vbd_manager_cfg import VBDSolverCfg

    @configclass
    class SolverCfg(VBDSolverCfg):
        friction_epsilon: float = config["friction_epsilon"]
        rigid_contact_history: bool = config["rigid_contact_history"]
        rigid_contact_hard: bool = config["rigid_contact_hard"]
        rigid_body_contact_buffer_size: int = 512
        rigid_joint_linear_ke: float = 1e5
        rigid_joint_angular_ke: float = 1e5

    @configclass
    class CollisionCfg(NewtonCollisionPipelineCfg):
        contact_matching: str = "sticky"
        contact_matching_pos_threshold: float = config["contact_match_distance_m"]
        rigid_contact_max: int = 4096

    launch_args = {"physics": "newton_vbd", "visualizer": [], "device": device}
    newton.use_coord_layout_targets = True
    with launch_simulation(PhysicsCfg(), launch_args) as physics:
        physics.num_substeps = measurements.substeps_per_sample
        physics.collision_decimation = 1
        physics.use_cuda_graph = wp.get_device(device).is_cuda
        physics.solver_cfg = SolverCfg(iterations=config["iterations"])
        physics.collision_cfg = CollisionCfg()
        sim = SimulationContext(
            SimulationCfg(
                dt=1 / measurements.diagnostic_rate_hz,
                device=device,
                physics=physics,
            )
        )
        try:
            builder = NewtonManager.create_builder()
            scene = _build_scene(builder, env, robot, assets)
            _initialize_rest_poses(builder)
            ids = wp.array(
                list(scene.robot_bodies.values()), dtype=wp.int32, device=device
            )

            def compensate(state: newton.State) -> None:
                wp.launch(
                    _compensate_gravity,
                    dim=len(scene.robot_bodies),
                    inputs=[
                        ids,
                        NewtonManager.get_model().body_mass,
                        robot["root_props"]["newton_gravity_compensation"],
                        state.body_f,
                    ],
                    device=device,
                )

            NewtonManager.set_builder(builder)
            NewtonManager.register_state_force_callback(compensate)
            sim.reset()
            return _replay(
                sim,
                NewtonManager,
                scene,
                config,
                np.asarray(robot["init_qpos"], dtype=np.float32),
                positions,
                velocities,
                measurements=measurements,
            )
        finally:
            sim.clear_instance()


def main() -> None:
    """Simulate the shared commands and report acceptance from fresh observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    recording = args.recording.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    assets = load_asset_manifest(args.asset_manifest)
    config, env, robot = load_scene()
    # Use the evaluator from the checkout that owns this scene and its YAML.
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    measurements = load_measurements(config)
    positions, velocities, source = load_commands(recording, config)
    inputs = scene_inputs(recording, assets)
    output.mkdir(parents=True, exist_ok=False)
    data, rows = _run(
        args.device,
        config,
        env,
        robot,
        assets,
        positions,
        velocities,
        measurements=measurements,
    )
    acceptance = measurements.evaluate(rows)
    save_results(
        output,
        data,
        {
            "status": "completed",
            "backend": "Isaac Lab native Newton VBD",
            "device": args.device,
            "device_name": wp.get_device(args.device).name,
            "versions": {
                name: version(name)
                for name in (
                    "isaaclab",
                    "isaaclab-newton",
                    "newton",
                    "warp-lang",
                    "torch",
                )
            },
            "inputs": inputs,
            "script": file_info(Path(__file__)),
            "source_episode": source,
            "joint_names": JOINT_NAMES,
            "control_hz": config["control_hz"],
            "control_steps": len(positions),
            "physics_steps_per_control": config["substeps"],
            "contact_samples_per_control": measurements.samples_per_control,
            "pre_roll_s": config["pre_roll_s"],
            "solver_iterations": config["iterations"],
            "robot_drive": "Native VBD position and velocity targets with authored PD gains",
            "cable_segments": config["cable_segments"],
            "position_and_velocity_readbacks_equal_recorded_actions": True,
            "task_success": acceptance["passed"],
            "acceptance": acceptance,
        },
        rows,
    )


if __name__ == "__main__":
    main()
