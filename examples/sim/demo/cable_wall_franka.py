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
"""Preview a Franka top-feed winding trajectory with gravity and cable contact.

The commanded path makes three turns per post. Frictional grasp can still slip;
this preview does not certify successful wrapping or P-11 acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

FPS = 60
CONTROL_FPS = 30
TIME_SCALE = 3.5
DURATION = 41.0 * TIME_SCALE
BASE = np.array([0.25, 0.0, 2.4])
POSTS = np.array([[0.75, -0.20, 2.75], [0.75, 0.20, 2.75]])
POST_RADIUS = 0.05
POST_HALF_LENGTH = 0.05
RADIUS = 0.004
ORBIT_RADIUS = 0.16
CABLE_LENGTH = 3.10
CLOSED_FINGER = 0.0155
FEED_FINGER = 0.0155
BEND_STIFFNESS = 0.0003
ANCHOR = np.array([0.76, -0.40, 2.92])
JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8)) + (
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)
HAND_QUAT = Rotation.from_euler("x", np.pi).as_quat()
TOOL_POINT = np.array([0.13, 0.0, 0.0])


def smooth(t: float) -> float:
    u = float(np.clip(t, 0, 1))
    return u * u * (3 - 2 * u)


def orbit(post: int, u: float) -> np.ndarray:
    theta = np.pi / 4 - 6 * np.pi * u
    return np.array(
        [
            0.76 - 0.042 * u,
            POSTS[post, 1] + ORBIT_RADIUS * np.cos(theta),
            POSTS[post, 2] + ORBIT_RADIUS * np.sin(theta),
        ]
    )


def target_at(t: float) -> tuple[np.ndarray, float, str, tuple[float, float]]:
    """Three complete end-effector revolutions at each post, then release."""
    t = t / TIME_SCALE
    if t <= 1:
        return orbit(0, 0), FEED_FINGER, "THREAD / SETTLE", (0.0, 0.0)
    if t <= 16:
        u = smooth((t - 1) / 15)
        return orbit(0, u), FEED_FINGER, "WIND + FEED POST 1", (3 * u, 0.0)
    if t <= 19:
        u = smooth((t - 16) / 3)
        p = (1 - u) * orbit(0, 1) + u * orbit(1, 0)
        p[2] += 0.10 * np.sin(np.pi * u)
        return p, FEED_FINGER, "TRANSFER + FEED", (3.0, 0.0)
    if t <= 34:
        u = smooth((t - 19) / 15)
        return orbit(1, u), FEED_FINGER, "WIND + FEED POST 2", (3.0, 3 * u)
    if t <= 36:
        return orbit(1, 1), CLOSED_FINGER, "HOLD", (3.0, 3.0)
    if t <= 36.6:
        return (
            orbit(1, 1),
            CLOSED_FINGER + (0.04 - CLOSED_FINGER) * smooth((t - 36) / 0.6),
            "RELEASE",
            (3.0, 3.0),
        )
    p = orbit(1, 1) + np.array([-0.07 * smooth((t - 36.6) / 1.4), 0, 0])
    return p, 0.04, "OBSERVE GRAVITY", (3.0, 3.0)


def initial_points(segments: int) -> np.ndarray:
    """Short working lead passes through the hand; the remaining tail is free."""
    grip = orbit(0, 0)
    # Enter the guide from the wall side and feed out toward the observer.
    # The hanging tail then stays in front of both post tips.
    u = np.linspace(0, 1, 2000)[:, None]
    a, b = ANCHOR, ANCHOR + np.array([0.02, 0.16, 0.06])
    c, d = grip + np.array([0.035, 0, 0]), grip
    lead = (
        (1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * b + 3 * (1 - u) * u**2 * c + u**3 * d
    )
    lead_length = np.linalg.norm(np.diff(lead, axis=0), axis=1).sum()
    outlet = grip - np.array([0.19, 0, 0])
    straight = np.linspace(grip, outlet, 500)
    theta = np.linspace(0, np.pi / 2, 500)
    bend = outlet + np.column_stack(
        (-0.035 * np.sin(theta), np.zeros(500), -0.035 * (1 - np.cos(theta)))
    )
    tail_length = CABLE_LENGTH - lead_length - 0.19 - 0.035 * np.pi / 2
    tail = bend[-1] + np.column_stack(
        (np.zeros(2000), np.zeros(2000), -np.linspace(0, tail_length, 2000))
    )
    dense = np.vstack((lead[:-1], straight[:-1], bend[:-1], tail))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    return np.column_stack(
        [
            np.interp(np.linspace(0, arc[-1], segments + 1), arc, dense[:, axis])
            for axis in range(3)
        ]
    ).astype(np.float32)


def write_feed_gripper(source: Path, path: Path) -> None:
    """Add two visible grooved finger pads, open along the cable feed axis.

    The rectangular channel closes transversely around the rope and routes the
    hanging tail in front of the post tips.
    Nothing constrains a cable node or assigns its material position.
    """
    root = ET.parse(source).getroot()
    mesh_root = path.parent / "meshes"
    source_mesh_root = (source.parent / "meshes").resolve()
    if not mesh_root.exists():
        mesh_root.symlink_to(source_mesh_root, target_is_directory=True)
    if mesh_root.resolve() != source_mesh_root:
        raise ValueError("Output meshes path conflicts with the original Franka assets")
    for mesh in root.findall(".//mesh"):
        if not (path.parent / mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))
    for name in ("fr3_leftfinger", "fr3_rightfinger"):
        # Right finger is authored with a half turn around local Z.
        center = 0.07 if name == "fr3_leftfinger" else -0.07
        boxes = [
            ((center, 0.002, 0.045), (0.180, 0.004, 0.035)),
            ((center, -0.00575, 0.0275), (0.180, 0.0195, 0.004)),
            ((center, -0.00575, 0.0625), (0.180, 0.0195, 0.004)),
        ]
        for index, (xyz, size) in enumerate(boxes):
            pad_name = f"{name}_feed_pad_{index}"
            link = ET.SubElement(root, "link", name=pad_name)
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=".0005")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="1e-7",
                iyy="1e-7",
                izz="1e-7",
                ixy="0",
                ixz="0",
                iyz="0",
            )
            joint = ET.SubElement(root, "joint", name=pad_name + "_fixed", type="fixed")
            ET.SubElement(joint, "parent", link=name)
            ET.SubElement(joint, "child", link=pad_name)
            ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)), rpy="0 0 0")
            for purpose in ("collision", "visual"):
                item = ET.SubElement(link, purpose, name=f"feed_pad_{index}")
                geometry = ET.SubElement(item, "geometry")
                ET.SubElement(geometry, "box", size=" ".join(map(str, size)))
                if purpose == "visual":
                    mat = ET.SubElement(item, "material", name="feed_pad")
                    ET.SubElement(mat, "color", rgba=".85 .58 .12 1")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def material_at_gripper(
    poses: np.ndarray, rest_lengths: np.ndarray, grip: np.ndarray
) -> tuple[float, float, int]:
    """Closest point on complete capsule axes, expressed in material arc length."""
    axis = poses[:, :3, 2]
    starts = poses[:, :3, 3] - 0.5 * rest_lengths[:, None] * axis
    local = np.clip(np.sum((grip - starts) * axis, axis=1), 0, rest_lengths)
    distances = np.linalg.norm(starts + local[:, None] * axis - grip, axis=1)
    index = int(np.argmin(distances))
    return (
        float(rest_lengths[:index].sum() + local[index]),
        float(distances[index]),
        index,
    )


def summarize_feeding(
    times: np.ndarray,
    poses: np.ndarray,
    feed_points: np.ndarray,
    material_s: np.ndarray,
    distances: np.ndarray,
    segment_ids: np.ndarray,
    windings: np.ndarray,
    post_penetrations: np.ndarray,
) -> dict[str, object]:
    """Preview diagnostics; retain failure and incomplete runs as such."""
    pre = times < 36 * TIME_SCALE
    lost = (distances > 0.02) & pre
    result = {
        "scope": "demo diagnostics, not P-11 physical acceptance",
        "grip_loss_episodes": int(np.sum(lost & ~np.r_[False, lost[:-1]])),
        "feed_after_settle_m": float(
            material_s[np.flatnonzero(pre)[-1]]
            - material_s[np.argmin(abs(times - TIME_SCALE))]
        ),
        "max_material_jump_per_frame_m": (
            float(np.max(abs(np.diff(material_s[pre])))) if pre.sum() > 1 else 0.0
        ),
        "complete": bool(times[-1] >= DURATION - 1 / FPS),
        "maximum_sampled_post_penetration_m": float(post_penetrations.max()),
        "wraps_before_release": None,
        "near_post_turns_before_release": None,
    }
    if times[-1] >= 35 * TIME_SCALE:
        k = int(np.argmin(abs(times - 35 * TIME_SCALE)))
        line = np.vstack((poses[k, : segment_ids[k] + 1, :3, 3], feed_points[k]))
        wraps = []
        contact_turns = []
        for j, post in enumerate(POSTS):
            angles = np.arctan2(line[:, 2] - post[2], line[:, 1] - post[1])
            endpoint_delta = np.angle(np.exp(1j * (angles[-1] - angles[0])))
            wraps.append(int(round(windings[k, j] - endpoint_delta / (2 * np.pi))))
            radial = np.linalg.norm(line[:, 1:] - post[1:], axis=1)
            near = (
                (abs(line[:, 0] - post[0]) < POST_HALF_LENGTH)
                & (radial < POST_RADIUS + RADIUS + 0.018)
                & (radial > POST_RADIUS - 0.006)
            )
            increments = np.angle(np.exp(1j * np.diff(angles)))
            contact_turns.append(
                float(abs(np.sum(increments[near[:-1] & near[1:]])) / (2 * np.pi))
            )
        result["wraps_before_release"] = wraps
        result["near_post_turns_before_release"] = contact_turns
        result["preview_checks_passed"] = bool(
            result["complete"]
            and not lost.any()
            and all(x == -3 for x in wraps)
            and min(contact_turns) >= 2.7
            and post_penetrations.max() < 0.01
        )
    else:
        result["preview_checks_passed"] = False
    return result


def sampled_post_penetration(poses: np.ndarray, rest_lengths: np.ndarray) -> np.ndarray:
    """Seven axial samples per capsule against ideal finite cylinders.

    This catches gross crossings at every saved frame; it is not an exact
    mesh-distance calculation or a continuous collision proof.
    """
    offsets = np.linspace(-0.5, 0.5, 7)
    samples = (
        poses[:, None, :3, 3]
        + rest_lengths[:, None, None] * offsets[None, :, None] * poses[:, None, :3, 2]
    ).reshape(-1, 3)
    depths = []
    for post in POSTS:
        d = np.column_stack(
            (
                abs(samples[:, 0] - post[0]) - POST_HALF_LENGTH,
                np.linalg.norm(samples[:, 1:] - post[1:], axis=1) - POST_RADIUS,
            )
        )
        sdf = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
        depths.append(max(0.0, float(RADIUS - sdf.min())))
    return np.asarray(depths)


def write_stage(path: Path) -> None:
    """Wall, normal-axis posts, visible fixed root and arm pedestal."""
    root = ET.Element("robot", name="wall_fixture")

    def link(
        name: str,
        kind: str,
        dims: dict[str, str],
        xyz: np.ndarray,
        rpy: str,
        color: str,
    ) -> None:
        body = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(body, "inertial")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(
            inertial, "inertia", ixx=".1", iyy=".1", izz=".1", ixy="0", ixz="0", iyz="0"
        )
        for purpose in ("visual", "collision"):
            item = ET.SubElement(body, purpose)
            ET.SubElement(item, "origin", xyz=" ".join(map(str, xyz)), rpy=rpy)
            geo = ET.SubElement(item, "geometry")
            ET.SubElement(geo, kind, **dims)
            if purpose == "visual":
                mat = ET.SubElement(item, "material", name=name + "_mat")
                ET.SubElement(mat, "color", rgba=color)
        if name != "wall":
            joint = ET.SubElement(root, "joint", name=name + "_fixed", type="fixed")
            ET.SubElement(joint, "parent", link="wall")
            ET.SubElement(joint, "child", link=name)

    link(
        "wall",
        "box",
        {"size": ".05 1.65 3.25"},
        np.array([0.825, 0, 1.625]),
        "0 0 0",
        ".17 .22 .28 1",
    )
    for n, p in enumerate(POSTS, 1):
        link(
            f"post_{n}",
            "cylinder",
            {"radius": str(POST_RADIUS), "length": str(2 * POST_HALF_LENGTH)},
            p,
            "0 1.570796326794897 0",
            ".10 .57 .68 1",
        )
    link(
        "anchor",
        "box",
        {"size": ".06 .042 .042"},
        ANCHOR + np.array([0.015, 0, 0]),
        "0 0 0",
        ".96 .57 .10 1",
    )
    link(
        "pedestal",
        "box",
        {"size": f".32 .32 {BASE[2]}"},
        np.array([BASE[0], 0, BASE[2] / 2]),
        "0 0 0",
        ".35 .39 .43 1",
    )
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def robot_fk(urdf: Path, q: np.ndarray) -> dict[str, np.ndarray]:
    """Independent XML FK, with named joints and unmodified asset limits."""
    root = ET.parse(urdf).getroot()
    qmap = dict(zip(JOINT_NAMES, q, strict=True))
    transforms = {"base": np.eye(4)}
    transforms["base"][:3, 3] = BASE
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                local[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            motion = np.eye(4)
            value = qmap.get(joint.get("name"), 0.0)
            axis = joint.find("axis")
            if axis is not None:
                vector = np.fromstring(axis.get("xyz"), sep=" ")
                if joint.get("type") == "revolute":
                    motion[:3, :3] = Rotation.from_rotvec(vector * value).as_matrix()
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = vector * value
            transforms[child] = transforms[parent] @ local @ motion
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise RuntimeError("URDF joint tree is disconnected")
    return transforms


def make_trajectory(
    urdf: Path, out: Path, duration: float, q7: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """Use EmbodiChain FEP IK; independently check FK and asset joint limits."""
    import torch
    from embodichain.lab.sim.motion.solvers import FEPSolverCfg

    times = np.arange(round(duration * CONTROL_FPS) + 1) / CONTROL_FPS
    targets = np.array([target_at(t)[0] for t in times])
    ik = FEPSolverCfg(
        urdf_path=str(urdf),
        root_link_name="base",
        end_link_name="fr3_hand_tcp",
        redundancy_search=False,
    ).init_solver("cpu")
    seed = torch.tensor([[0, -0.6, 0, -2.2, 0, 1.6, q7]], dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = torch.tensor(
        Rotation.from_quat(HAND_QUAT).as_matrix(), dtype=torch.float32
    )
    qs = []
    errors = []
    rot_errors = []
    for index, t in enumerate(times):
        target, finger, _, _ = target_at(t)
        tcp_target = target - Rotation.from_quat(HAND_QUAT).apply(TOOL_POINT)
        pose[:3, 3] = torch.tensor(tcp_target - BASE, dtype=torch.float32)
        valid, seed = ik.get_ik(pose, seed)
        if not bool(valid.all()):
            raise RuntimeError(f"FEP IK failed at {t:.3f}s")
        q = np.r_[seed.numpy().reshape(-1), finger, finger]
        fk = robot_fk(urdf, q)["fr3_hand_tcp"]
        errors.append(np.linalg.norm(fk[:3, 3] + fk[:3, :3] @ TOOL_POINT - target))
        rot_errors.append(
            (
                Rotation.from_matrix(fk[:3, :3]).inv() * Rotation.from_quat(HAND_QUAT)
            ).magnitude()
        )
        qs.append(q)
        if index % 150 == 0:
            print("IK", t, "error_m", errors[-1], flush=True)
    qs = np.array(qs, dtype=np.float32)
    root = ET.parse(urdf).getroot()
    limits = {
        j.get("name"): j.find("limit")
        for j in root.findall("joint")
        if j.get("name") in JOINT_NAMES
    }
    lower = np.array([float(limits[n].get("lower")) for n in JOINT_NAMES])
    upper = np.array([float(limits[n].get("upper")) for n in JOINT_NAMES])
    limit_violation = float(max(np.max(lower - qs), np.max(qs - upper), 0))
    joint_speed = np.abs(np.diff(qs[:, :7], axis=0)) * CONTROL_FPS
    speed_limits = np.array([float(limits[n].get("velocity")) for n in JOINT_NAMES[:7]])
    stats = {
        "ik_solver": f"EmbodiChain FEP, fixed q7={q7} rad",
        "max_position_error_m": float(max(errors)),
        "max_rotation_error_rad": float(max(rot_errors)),
        "joint_limit_violation_rad": limit_violation,
        "max_joint_step_rad": float(np.abs(np.diff(qs[:, :7], axis=0)).max()),
        "max_joint_speed_rad_s": float(joint_speed.max()),
        "max_speed_limit_ratio": float((joint_speed / speed_limits).max()),
    }
    (out / "ik_checks.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez(
        out / "robot_trajectory.npz",
        times=times,
        qpos=qs,
        joint_names=JOINT_NAMES,
        tcp_targets=targets,
        hand_quaternion=HAND_QUAT,
        tool_point=TOOL_POINT,
        position_error=errors,
    )
    print("IK_CHECKS", stats, flush=True)
    if (
        max(errors) > 0.004
        or max(rot_errors) > 0.04
        or limit_violation > 0.001
        or stats["max_speed_limit_ratio"] > 1
    ):
        raise RuntimeError(
            "Franka trajectory failed independent FK / joint-limit checks"
        )
    return times, qs


def run(args: argparse.Namespace) -> None:
    import dexsim, newton, warp as wp
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
    from embodichain.lab.sim.cfg.articulation import LinkPhysicsOverrideCfg
    from embodichain.lab.sim.sensors import CameraCfg

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_urdf = args.robot.resolve()
    urdf = out / "franka_feed_gripper.urdf"
    write_feed_gripper(source_urdf, urdf)
    duration = min(args.duration, DURATION)
    if args.trajectory:
        saved = np.load(args.trajectory)
        qs = saved["qpos"]
        control_times = saved["times"]
        qs[:, -2:] = np.array([target_at(t)[1] for t in control_times])[:, None]
        if control_times[-1] < duration:
            raise ValueError("Trajectory shorter than requested duration")
        expected = np.array([target_at(t)[0] for t in control_times])
        if not np.allclose(saved["tcp_targets"], expected, atol=1e-6, rtol=0):
            raise ValueError(
                "Cached trajectory does not match the current winding path"
            )
        if (
            "hand_quaternion" not in saved
            or "tool_point" not in saved
            or not np.allclose(saved["hand_quaternion"], HAND_QUAT, atol=1e-6, rtol=0)
            or not np.allclose(saved["tool_point"], TOOL_POINT, atol=1e-6, rtol=0)
        ):
            raise ValueError(
                "Cached trajectory does not match the current feed tool pose"
            )
        checks = args.trajectory.parent / "ik_checks.json"
        if checks.is_file():
            (out / "ik_checks.json").write_bytes(checks.read_bytes())
    else:
        control_times, qs = make_trajectory(urdf, out, duration, args.ik_q7)
    if args.plan_only:
        return
    np.savez(
        out / "robot_trajectory.npz",
        times=control_times,
        qpos=qs,
        joint_names=JOINT_NAMES,
        tcp_targets=np.array([target_at(t)[0] for t in control_times]),
        hand_quaternion=HAND_QUAT,
        tool_point=TOOL_POINT,
    )
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    config = {
        "length_m": CABLE_LENGTH,
        "radius_m": RADIUS,
        "post_centers": POSTS.tolist(),
        "post_radius_m": POST_RADIUS,
        "post_half_length_m": POST_HALF_LENGTH,
        "anchor": ANCHOR.tolist(),
        "franka_base": BASE.tolist(),
        "hand_quaternion_xyzw": HAND_QUAT.tolist(),
        "tool_point_in_tcp_m": TOOL_POINT.tolist(),
        "orbit_radius_m": ORBIT_RADIUS,
        "time_scale": TIME_SCALE,
        "feed_finger_q_m": FEED_FINGER,
        "hold_finger_q_m": CLOSED_FINGER,
        "bend_stiffness": BEND_STIFFNESS,
        "gripper_mu": 0.08,
        "rope_mu": 0.15,
        "stage_mu": 0.15,
        "contact_ke": 1e5,
        "joint_coefficient_scale": args.segments / 256,
        "rigid_body_contact_buffer_size": 512,
        "physical_tip_attachment": False,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stage = out / "wall_fixture.urdf"
    write_stage(stage)
    points = initial_points(args.segments)
    rest_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    np.savez(out / "inputs.npz", points=points, posts=POSTS, anchor=ANCHOR)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=not args.viewer,
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
                        "rigid_body_contact_buffer_size": 512,
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
    video = camera = detail_camera = cable = robot = None
    poses = []
    speeds = []
    free_tip_distances = []
    actual_tcp = []
    actual_feed_point = []
    grip_material_s = []
    grip_distances = []
    grip_segments = []
    prefix_windings = []
    post_penetrations = []
    start = time.perf_counter()
    try:
        material = RigidBodyPhysicsCfg.from_dict(
            {
                "material_props": {
                    "backend": "newton",
                    "dynamic_friction": 0.15,
                    "ke": 1e5,
                    "kd": 100.0,
                },
                "collision_props": {"contact_offset": 0.001, "rest_offset": 0.0},
            }
        )
        grip_material = RigidBodyPhysicsCfg.from_dict(
            {
                "material_props": {
                    "backend": "newton",
                    "dynamic_friction": 0.08,
                    "ke": 2e4,
                    "kd": 100.0,
                },
                "collision_props": {"contact_offset": 0.001, "rest_offset": 0.0},
            }
        )
        sim.add_articulation(
            ArticulationCfg(
                uid="wall_fixture",
                fpath=str(stage),
                asset_physics_mode="overlay",
                attrs=material,
            )
        )
        robot = sim.add_articulation(
            ArticulationCfg(
                uid="franka",
                fpath=str(urdf),
                init_pos=tuple(BASE),
                init_qpos=qs[0],
                asset_physics_mode="overlay",
                attrs=material,
                link_attrs={
                    "finger_pads": LinkPhysicsOverrideCfg(
                        link_names_expr=[
                            "fr3_leftfinger",
                            "fr3_rightfinger",
                            "fr3_.*finger_feed_pad_.*",
                        ],
                        attrs=grip_material,
                    )
                },
            )
        )
        sim.register_kinematic_joint_trajectory("franka", qs[None], fps=CONTROL_FPS)
        sim.add_cable(
            CableDesc(
                name="cable",
                points=points,
                radius=RADIUS,
                render_mode="capsules",
                per_env=True,
                fixed_segments=(0,),
                collision_filter_span=1,
                collision=NewtonCollisionDesc(
                    mu=0.15, ke=1e5, kd=100.0, margin=0, gap=0.001
                ),
                physics=CablePhysicsDesc(
                    density=800,
                    stretch_stiffness=5e5 * args.segments / 256,
                    shear_stiffness=5e5 * args.segments / 256,
                    bend_stiffness=BEND_STIFFNESS * args.segments / 256,
                    twist_stiffness=BEND_STIFFNESS * args.segments / 256,
                    stretch_damping=args.segments / 256,
                    shear_damping=args.segments / 256,
                    bend_damping=0.001 * args.segments / 256,
                    twist_damping=0.001 * args.segments / 256,
                ),
                render_material=MaterialDesc(
                    name="orange_cable",
                    base_color=(1.0, 0.26, 0.025, 1.0),
                    roughness=0.6,
                ),
            )
        )
        if not args.no_video:
            camera = sim.add_sensor(
                CameraCfg(
                    uid="wall_view",
                    width=1600,
                    height=1200,
                    intrinsics=(1180.0, 1180.0, 800.0, 600.0),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(-1.7, 1.0, 3.3), target=(0.65, 0, 2.45)
                    ),
                )
            )
            detail_camera = sim.add_sensor(
                CameraCfg(
                    uid="contact_detail",
                    width=760,
                    height=560,
                    intrinsics=(800.0, 800.0, 380.0, 280.0),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(0.20, -0.55, 3.15), target=(0.75, 0, 2.75)
                    ),
                )
            )
        sim.prepare()
        cable = sim.get_cable("cable")
        print("ROBOT_JOINT_ORDER", robot.joint_names, flush=True)
        assert [x.lower() for x in robot.joint_names] == list(JOINT_NAMES)
        if camera is not None:
            video = cv2.VideoWriter(
                str(out / "preview_raw.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (1600, 1200),
            )
            if not video.isOpened():
                raise RuntimeError("Video writer unavailable")
        frames = round(duration * FPS)
        times = np.arange(frames + 1) / FPS
        phase = None
        for frame, t in enumerate(times):
            if frame:
                sim.update(step=1, render_final_step=args.viewer and camera is None)
            p = cable.get_segment_poses()
            v = cable.get_segment_velocities()
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                raise RuntimeError(f"Nonfinite state at {t}")
            poses.append(p)
            post_penetrations.append(sampled_post_penetration(p, rest_lengths))
            speeds.append(np.sqrt(np.mean(np.sum(v[:, :3] ** 2, axis=1))))
            target, finger, current, turns = target_at(t)
            hand_pose = (
                robot.get_link_pose("fr3_hand_tcp", to_matrix=True)
                .detach()
                .cpu()
                .numpy()[0]
            )
            actual_tcp.append(hand_pose[:3, 3])
            feed_point = hand_pose[:3, 3] + hand_pose[:3, :3] @ TOOL_POINT
            actual_feed_point.append(feed_point)
            material_s, grip_distance, grip_segment = material_at_gripper(
                p, rest_lengths, feed_point
            )
            grip_material_s.append(material_s)
            grip_distances.append(grip_distance)
            grip_segments.append(grip_segment)
            working_line = np.vstack((p[: grip_segment + 1, :3, 3], feed_point))
            angles = [
                np.unwrap(
                    np.arctan2(working_line[:, 2] - c[2], working_line[:, 1] - c[1])
                )
                for c in POSTS
            ]
            net = [float((x[-1] - x[0]) / (2 * np.pi)) for x in angles]
            prefix_windings.append(net)
            free_tip_distances.append(
                float(np.linalg.norm(p[-1, :3, 3] - hand_pose[:3, 3]))
            )
            if current != phase or frame % 300 == 0:
                print(
                    f"{t:.2f}s {current} material_s={material_s:.3f} grip_axis_distance={grip_distance:.4f} working_winding={net}",
                    flush=True,
                )
                phase = current
            if camera is not None and frame % 7 == 0:
                with sim.render_frame(capture_camera_images=False):
                    sim.sync_render_state()
                    camera.update()
                    detail_camera.update()
                rgb = camera.get_data()["color"][0, ..., :3].cpu().numpy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detail_rgb = detail_camera.get_data()["color"][0, ..., :3].cpu().numpy()
                bgr[625:1185, 825:1585] = cv2.cvtColor(detail_rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr, (823, 623), (1587, 1187), (190, 190, 190), 2)
                cv2.putText(
                    bgr,
                    "CONTACT DETAIL",
                    (845, 652),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(bgr, (0, 0), (1600, 124), (24, 29, 35), -1)
                cv2.putText(
                    bgr,
                    "P-11 | FRANKA FR3 | SLIDING FEED | 3 TURNS / POST",
                    (25, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (240, 245, 250),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    f"{t:06.2f}s  {current} | Commanded turns: {turns[0]:.2f} / {turns[1]:.2f} | Gravity: 9.81 m/s2 | {TIME_SCALE:g}x replay",
                    (25, 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.63,
                    (180, 207, 220),
                    1,
                    cv2.LINE_AA,
                )
                grasp_label = (
                    f"MATERIAL AT GRIP: {material_s:.3f} m | TAIL: {CABLE_LENGTH-material_s:.3f} m | CONTACT ONLY / NO ATTACHMENT"
                    if t < 36 * TIME_SCALE
                    else "GRIP OPEN / FREE RELEASE | ROPE MOTION FROM GRAVITY AND CONTACT"
                )
                cv2.putText(
                    bgr,
                    grasp_label,
                    (25, 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.63,
                    (110, 195, 255),
                    1,
                    cv2.LINE_AA,
                )
                video.write(bgr)
                if frame % 420 == 0:
                    cv2.imwrite(str(out / f"frame_{frame:04d}.jpg"), bgr)
        wp.synchronize()
        poses = np.asarray(poses)
        feeding_checks = summarize_feeding(
            times,
            poses,
            np.asarray(actual_feed_point),
            np.asarray(grip_material_s),
            np.asarray(grip_distances),
            np.asarray(grip_segments),
            np.asarray(prefix_windings),
            np.asarray(post_penetrations),
        )
        (out / "feeding_checks.json").write_text(
            json.dumps(feeding_checks, indent=2) + "\n"
        )
        np.savez_compressed(
            out / "states.npz",
            times=times,
            poses=poses,
            speed_rms=speeds,
            free_tip_distance_to_tcp=free_tip_distances,
            actual_tcp=actual_tcp,
            actual_feed_point=actual_feed_point,
            grip_material_s=grip_material_s,
            grip_distance=grip_distances,
            grip_segment=grip_segments,
            working_prefix_winding=prefix_windings,
            sampled_post_penetration_m=post_penetrations,
        )
        metrics = {
            "demo_only": True,
            "acceptance_status": "not_validated; commanded turns are not measured rope wraps",
            "feeding_checks": feeding_checks,
            "robot_asset": str(urdf),
            "source_robot_asset": str(source_urdf),
            "robot_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "dexsim_commit": dexsim.__commit_id__,
            "solver": sim.physics.solver_type,
            "newton": newton.__version__,
            "warp": wp.__version__,
            "grasp": "sliding contact in visible grooved Franka pads; free material feed",
            "diagnostic_endpoint_attachment": False,
            "release_time_s": 36 * TIME_SCALE,
            "base_position": BASE.tolist(),
            "segments": args.segments,
            "length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            "radius_m": RADIUS,
            "bend_stiffness": BEND_STIFFNESS,
            "initial_material_at_grip_m": grip_material_s[0],
            "final_material_at_grip_m": grip_material_s[-1],
            "max_grip_centerline_distance_before_release_m": float(
                np.max(np.array(grip_distances)[times <= 36 * TIME_SCALE])
            ),
            "working_prefix_winding_final": prefix_windings[-1],
            "dt": 1 / FPS,
            "substeps": args.substeps,
            "iterations": args.iterations,
            "duration_s": duration,
            "wall_s": time.perf_counter() - start,
            "final_free_tip_distance_to_tcp_m": float(free_tip_distances[-1]),
            "final_speed_rms_m_s": float(speeds[-1]),
        }
        (out / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        if video is not None:
            video.release()
        robot = cable = camera = detail_camera = None
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
                str(out / "cable_wall_franka.mp4"),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--segments", type=int, default=128)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the native simulation viewer while running",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--ik-q7", type=float, default=0.75)
    args = parser.parse_args()
    if (
        args.duration <= 0
        or args.segments < 2
        or args.substeps < 1
        or args.iterations < 1
    ):
        parser.error(
            "duration, substeps and iterations must be positive; segments must be >= 2"
        )
    run(args)
