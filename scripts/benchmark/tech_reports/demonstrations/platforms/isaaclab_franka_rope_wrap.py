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
"""Run the recorded FrankaRopeWrap-v1 commands on Isaac Lab's native Newton VBD.

Run in the Isaac Lab Newton Python environment with TensorDict and USD support.
Record one successful episode first with EmbodiChain's standard run-task
entrypoint and task.franka.yaml. Pass the resulting .pt file and the
FrankaRopePads/Panda/Fr3RopePadsV1.urdf asset with its adjacent meshes.

The replay initializes from the authored scene and performs a fresh pre-roll.
Each control interval uses the recorded position and velocity targets exactly
once. The output archives the native command readbacks and newly simulated
states, plus input hashes and a geometry summary. The geometry summary does
not implement the EmbodiChain acceptance graph.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import newton
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from franka_rope_wrap_inputs import (
    JOINT_NAMES,
    file_info,
    geometry_summary,
    input_metadata,
    load_commands,
    load_scene,
    sha256,
)

__all__: list[str] = []


def _load_cable(path: Path, expected_segments: int, radius: float) -> np.ndarray:
    """Read the authored, untransformed centerline of the packaged cable."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
    if (
        UsdGeom.GetStageMetersPerUnit(stage) != 1.0
        or UsdGeom.GetStageUpAxis(stage) != "Z"
    ):
        raise ValueError("Expected the cable USD to use meters and the Z-up axis.")
    curve = UsdGeom.BasisCurves.Get(stage, "/World/cable")
    if (
        not curve
        or curve.GetTypeAttr().Get() != "linear"
        or curve.GetWrapAttr().Get() != "nonperiodic"
        or list(curve.GetCurveVertexCountsAttr().Get()) != [expected_segments + 1]
        or UsdGeom.XformCache().GetLocalToWorldTransform(curve.GetPrim())
        != Gf.Matrix4d(1)
    ):
        raise ValueError(
            "Expected one untransformed linear cable with the configured topology."
        )
    thicknesses = curve.GetPrim().GetAttribute("physics:thicknesses").Get()
    if (
        thicknesses is None
        or len(thicknesses) != 1
        or not math.isclose(thicknesses[0], 2 * radius, rel_tol=1e-6)
    ):
        raise ValueError("Cable USD thickness does not match the configured diameter.")
    points = np.array(curve.GetPointsAttr().Get(), dtype=np.float32)
    if (
        points.shape != (expected_segments + 1, 3)
        or not np.isfinite(points).all()
        or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0)
    ):
        raise ValueError("Cable USD must contain finite points and nonzero segments.")
    return points


@dataclass(frozen=True)
class _Scene:
    joint_ids: list[int]
    robot_body_ids: list[int]
    cable_body_ids: list[int]
    cable_lengths: np.ndarray


def _shape_config(
    builder: newton.ModelBuilder, attrs: dict[str, Any]
) -> newton.ModelBuilder.ShapeConfig:
    collision = attrs["collision_props"]
    material = attrs["material_props"]
    shape = builder.default_shape_cfg.copy()
    shape.density = attrs.get("density", 0.0)
    shape.margin, shape.gap = collision["margin"], collision["gap"]
    shape.mu = material["dynamic_friction"]
    shape.ke, shape.kd = material["ke"], material["kd"]
    return shape


def _build_scene(
    builder: newton.ModelBuilder,
    env: dict[str, Any],
    robot: dict[str, Any],
    urdf: Path,
    points: np.ndarray,
    radius: float,
) -> _Scene:
    # The report scene uses analytic colliders and the flat-pad Franka URDF.
    if env["enable_ground_plane"]:
        builder.add_ground_plane()
    for item in env["simulation"]["background"]:
        shape = item["shape"]
        orientation = Rotation.from_euler(
            "xyz", item.get("init_rot", (0.0, 0.0, 0.0)), degrees=True
        ).as_quat()
        pose = wp.transform(item["init_pos"], orientation)
        material = _shape_config(builder, item["attrs"])
        if shape["shape_type"] == "Cube":
            hx, hy, hz = (size / 2 for size in shape["size"])
            builder.add_shape_box(
                body=-1,
                xform=pose,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=material,
                label=item["uid"],
            )
        elif shape["shape_type"] == "Cylinder":
            builder.add_shape_cylinder(
                body=-1,
                xform=pose,
                radius=shape["radius"],
                half_height=shape["half_height"],
                cfg=material,
                label=item["uid"],
            )
        else:
            raise ValueError(f"Unsupported scene primitive: {shape['shape_type']}")

    first_body, first_joint = builder.body_count, builder.joint_count
    default_shape = builder.default_shape_cfg
    builder.default_shape_cfg = _shape_config(builder, robot["attrs"])
    try:
        builder.add_urdf(
            str(urdf),
            floating=False,
            xform=wp.transform(robot["init_pos"], wp.quat_identity()),
            parse_visuals_as_colliders=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_position_velocity_actuation=True,
        )
    finally:
        builder.default_shape_cfg = default_shape
    robot_bodies = list(range(first_body, builder.body_count))
    if any(builder.body_mass[index] <= 0 for index in robot_bodies):
        raise ValueError("The collapsed Franka articulation contains a massless link.")
    joint_ids_by_name = {
        str(builder.joint_label[index]).rsplit("/", 1)[-1]: index
        for index in range(first_joint, builder.joint_count)
    }
    joint_ids = [joint_ids_by_name[name] for name in JOINT_NAMES]
    drives = robot["joint_drive_props"]
    for name, joint, home in zip(
        JOINT_NAMES, joint_ids, robot["init_qpos"], strict=True
    ):
        q_index = builder.joint_q_start[joint]
        dof_index = builder.joint_qd_start[joint]
        group = "eef" if name.startswith("fr3_finger_joint") else "arm"
        builder.joint_q[q_index] = builder.joint_target_q[q_index] = home
        builder.joint_target_qd[dof_index] = 0.0
        builder.joint_target_ke[dof_index] = drives["stiffness"][group]
        builder.joint_target_kd[dof_index] = drives["damping"][group]
        builder.joint_limit_ke[dof_index] = 1.0e5
        builder.joint_limit_kd[dof_index] = 10.0

    cable = env["simulation"]["usd_scene"]["asset_cfgs"]["/World/cable"]
    attrs = cable["attrs"]
    first_shape = builder.shape_count
    cable_bodies, _ = builder.add_rod(
        points.tolist(),
        radius=radius,
        cfg=_shape_config(builder, attrs),
        stretch_stiffness=attrs["stretch_stiffness"],
        stretch_damping=attrs["stretch_damping"],
        shear_stiffness=attrs["shear_stiffness"],
        shear_damping=attrs["shear_damping"],
        bend_stiffness=attrs["bend_stiffness"],
        bend_damping=attrs["bend_damping"],
        twist_stiffness=attrs["twist_stiffness"],
        twist_damping=attrs["twist_damping"],
        body_frame_origin="com",
        label="/World/cable",
    )
    shapes = list(range(first_shape, builder.shape_count))
    if len(shapes) != len(cable_bodies):
        raise RuntimeError("Expected one cable collider per segment.")
    span = cable["collision_filter_span"]
    for segment, shape in enumerate(shapes):
        for neighbor in shapes[segment + 1 : segment + span + 1]:
            builder.add_shape_collision_filter_pair(shape, neighbor)
    for segment in cable["fixed_segments"]:
        builder.body_flags[cable_bodies[segment]] = int(newton.BodyFlags.KINEMATIC)
    return _Scene(
        joint_ids,
        robot_bodies,
        cable_bodies,
        np.linalg.norm(np.diff(points, axis=0), axis=1),
    )


def _initialize_rest_poses(builder: newton.ModelBuilder) -> None:
    # VBD integrates body coordinates. Set the robot rest transforms from its
    # authored home joints before finalizing the model used by Isaac Lab.
    with wp.ScopedDevice("cpu"):
        model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        builder.body_q = [
            wp.transform(pose[:3], pose[3:]) for pose in state.body_q.numpy()
        ]


@wp.kernel
def _compensate_gravity(
    ids: wp.array(dtype=wp.int32),
    masses: wp.array(dtype=float),
    scale: float,
    forces: wp.array(dtype=wp.spatial_vector),
):
    body = ids[wp.tid()]
    force = wp.vec3(0.0, 0.0, 9.81 * scale * masses[body])
    wp.atomic_add(forces, body, wp.spatial_vector(force, wp.vec3(0.0, 0.0, 0.0)))


def _replay(
    sim: Any,
    manager: Any,
    scene: _Scene,
    home: np.ndarray,
    pre_roll_steps: int,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, np.ndarray]:
    model, control = manager.get_model(), manager.get_control()
    joints = scene.joint_ids
    q_indices = model.joint_q_start.numpy()[joints]
    dof_indices = model.joint_qd_start.numpy()[joints]
    target_indices = model.joint_target_q_start.numpy()[joints]
    targets, speeds = control.joint_target_q.numpy(), control.joint_target_qd.numpy()
    measured_q, measured_qd = wp.clone(model.joint_q), wp.clone(model.joint_qd)
    steps = len(positions)
    assigned_q, assigned_qd = np.empty_like(positions), np.empty_like(velocities)
    robot_q = np.empty((steps + 1, home.size), dtype=np.float32)
    robot_qd = np.empty_like(robot_q)
    rope = np.empty((steps + 1, len(scene.cable_body_ids), 7), dtype=np.float32)

    def write_command(q: np.ndarray, qd: np.ndarray) -> None:
        targets[target_indices] = q
        speeds[dof_indices] = qd
        control.joint_target_q.assign(targets)
        control.joint_target_qd.assign(speeds)

    def capture(frame: int) -> None:
        state = manager.get_state()
        newton.eval_ik(model, state, measured_q, measured_qd)
        robot_q[frame] = measured_q.numpy()[q_indices]
        robot_qd[frame] = measured_qd.numpy()[dof_indices]
        rope[frame] = state.body_q.numpy()[scene.cable_body_ids]
        if not all(
            np.isfinite(value[frame]).all() for value in (robot_q, robot_qd, rope)
        ):
            raise FloatingPointError(
                f"Nonfinite replay state at control frame {frame}."
            )

    write_command(home, np.zeros_like(home))
    for _ in range(pre_roll_steps):
        sim.step(render=False)
    capture(0)
    for index, (q, qd) in enumerate(zip(positions, velocities, strict=True)):
        write_command(q, qd)
        assigned_q[index] = control.joint_target_q.numpy()[target_indices]
        assigned_qd[index] = control.joint_target_qd.numpy()[dof_indices]
        sim.step(render=False)
        capture(index + 1)
        if (index + 1) % 600 == 0:
            print(f"Completed {index + 1}/{steps} recorded controls", flush=True)
    np.testing.assert_array_equal(assigned_q.view(np.uint32), positions.view(np.uint32))
    np.testing.assert_array_equal(
        assigned_qd.view(np.uint32), velocities.view(np.uint32)
    )
    return {
        "command_q": assigned_q,
        "command_qd": assigned_qd,
        "robot_q": robot_q,
        "robot_qd": robot_qd,
        "cable_body_pose": rope,
        "cable_segment_lengths": scene.cable_lengths,
    }


def _run(
    args: argparse.Namespace,
    config: dict[str, Any],
    env: dict[str, Any],
    robot: dict[str, Any],
    points: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, np.ndarray]:
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
        # The native VBD solver integrates the robot's internal joints as well.
        rigid_joint_linear_ke: float = 1e5
        rigid_joint_angular_ke: float = 1e5

    @configclass
    class CollisionCfg(NewtonCollisionPipelineCfg):
        contact_matching: str = "sticky"
        contact_matching_pos_threshold: float = config["contact_match_distance_m"]
        rigid_contact_max: int = 4096

    hz = config["control_hz"]
    launch_args = {"physics": "newton_vbd", "visualizer": [], "device": args.device}
    newton.use_coord_layout_targets = True
    with launch_simulation(PhysicsCfg(), launch_args) as physics:
        physics.num_substeps = config["substeps"]
        physics.collision_decimation = 1
        physics.use_cuda_graph = True
        physics.solver_cfg = SolverCfg(iterations=config["iterations"])
        physics.collision_cfg = CollisionCfg()
        sim = SimulationContext(
            SimulationCfg(dt=1 / hz, device=args.device, physics=physics)
        )
        try:
            builder = NewtonManager.create_builder()
            scene = _build_scene(
                builder, env, robot, args.robot_urdf, points, config["cable_radius_m"]
            )
            _initialize_rest_poses(builder)
            ids = wp.array(scene.robot_body_ids, dtype=wp.int32, device=args.device)

            def gravity_compensation(state: Any) -> None:
                wp.launch(
                    _compensate_gravity,
                    dim=len(scene.robot_body_ids),
                    inputs=[
                        ids,
                        NewtonManager.get_model().body_mass,
                        robot["root_props"]["newton_gravity_compensation"],
                        state.body_f,
                    ],
                    device=args.device,
                )

            NewtonManager.set_builder(builder)
            NewtonManager.register_state_force_callback(gravity_compensation)
            sim.reset()
            return _replay(
                sim,
                NewtonManager,
                scene,
                np.asarray(robot["init_qpos"], dtype=np.float32),
                round(config["pre_roll_s"] * hz),
                positions,
                velocities,
            )
        finally:
            sim.clear_instance()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording", required=True, type=Path, help="Trusted run-task .pt artifact"
    )
    parser.add_argument(
        "--robot-urdf", required=True, type=Path, help="Resolved Fr3RopePadsV1.urdf"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New result directory"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.recording = args.recording.resolve(strict=True)
    args.robot_urdf = args.robot_urdf.resolve(strict=True)
    config, env, robot, usd = load_scene()
    points = _load_cable(usd, config["cable_segments"], config["cable_radius_m"])
    positions, velocities, source = load_commands(
        args.recording,
        control_hz=config["control_hz"],
        substeps=config["substeps"],
        expected_steps=round(config["duration_s"] * config["control_hz"]),
    )
    urdf = ElementTree.parse(args.robot_urdf)
    for finger in ("fr3_leftfinger", "fr3_rightfinger"):
        box = urdf.find(
            f"./link[@name='{finger}']/collision[@name='{finger}_tip_collision']/geometry/box"
        )
        if (
            box is None
            or list(map(float, box.attrib["size"].split()))
            != config["finger_pad_size_m"]
        ):
            parser.error(f"Expected the calibrated flat pad on {finger}.")
    inputs = input_metadata(args.recording, args.robot_urdf)
    args.output.mkdir(parents=True, exist_ok=False)
    results = _run(args, config, env, robot, points, positions, velocities)
    report = {
        "status": "completed",
        "backend": "Isaac Lab native Newton VBD",
        "device": args.device,
        "device_name": wp.get_device(args.device).name,
        "versions": {
            name: version(name)
            for name in ("isaaclab", "isaaclab-newton", "newton", "warp-lang", "torch")
        },
        "inputs": inputs,
        "script": file_info(Path(__file__)),
        "source_episode": source,
        "joint_names": JOINT_NAMES,
        "control_hz": config["control_hz"],
        "control_steps": len(positions),
        "physics_steps_per_control": config["substeps"],
        "pre_roll_s": config["pre_roll_s"],
        "solver_iterations": config["iterations"],
        "solver_rigid_joint_angular_ke": 1e5,
        "cable_segments": config["cable_segments"],
        "position_and_velocity_readbacks_equal_recorded_actions": True,
        "task_success": None,
        "final_geometry": geometry_summary(
            results["cable_body_pose"],
            results["cable_segment_lengths"],
            config["post_center_xy_m"],
            config["control_hz"],
        ),
    }
    trajectory = args.output / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        time_s=np.arange(len(positions) + 1) / config["control_hz"],
        **results,
    )
    report["trajectory_sha256"] = sha256(trajectory)
    summary = args.output / "result.json"
    summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Completed {len(positions)} controls. Result: {summary}")


if __name__ == "__main__":
    _main()
