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
"""Check M04 actual robot poses against the fixed table and reference garment.

Install python-fcl in the current task .tool_deps directory (diagnostics only).
This checks sampled collision meshes, not continuous motion or clearance distance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    supports,
)


def write_json(path: Path, value: object) -> None:
    """Save the sampled geometric audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


__all__ = ["main"]


def origin(element: ET.Element | None) -> np.ndarray:
    """Return a URDF origin matrix."""
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return matrix


def main() -> None:
    """Report exact mesh intersections for every requested recorded FK pose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trajectory-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--distance",
        action="store_true",
        help="Also calculate exact positive cloth/robot distance (optional diagnostic).",
    )
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--fcl-path",
        type=Path,
        default=Path("/tmp/m05_fcl"),
    )
    args = parser.parse_args()
    if args.stride < 1 or args.substeps < 1:
        parser.error("stride and substeps must be positive")
    sys.path.insert(0, str(args.fcl_path))
    import trimesh
    from trimesh.collision import CollisionManager

    inputs = args.run / "inputs"
    cfg = json.loads((inputs / "config.json").read_text())
    manifest = (
        json.loads((args.run / "manifest.json").read_text())
        if (args.run / "manifest.json").exists()
        else {}
    )
    commands = np.load(inputs / "joint_commands.npz")
    states = None if args.trajectory_only else np.load(args.run / "preview/states.npz")
    times = commands["time"] if states is None else states["time"]
    qrows = commands["targets"] if states is None else states["qpos"]
    asset = inputs / "w1_v030"
    urdf = ET.parse(asset / "robot.urdf").getroot()
    joints = list(urdf.findall("joint"))
    links = [link.get("name") for link in urdf.findall("link")]
    roots = {name: name for name in links}

    def root(name: str) -> str:
        while roots[name] != name:
            name = roots[name]
        return name

    for joint in joints:
        if joint.get("type") == "fixed":
            roots[root(joint.find("child").get("link"))] = root(
                joint.find("parent").get("link")
            )
    adjacent = {
        tuple(
            sorted(
                (root(j.find("parent").get("link")), root(j.find("child").get("link")))
            )
        )
        for j in joints
    }
    robot = CollisionManager()
    cloth_obstacles = CollisionManager()
    members = {}
    for link in urdf.findall("link"):
        geometries = link.findall("collision")
        if not geometries and (
            "camera" in link.get("name") or "stand" in link.get("name")
        ):
            geometries = link.findall("visual")
        for i, collision in enumerate(geometries):
            element = collision.find("geometry/mesh")
            if element is not None:
                mesh = trimesh.load(asset / element.get("filename"), force="mesh")
                mesh.apply_scale(np.fromstring(element.get("scale", "1 1 1"), sep=" "))
            elif (element := collision.find("geometry/box")) is not None:
                mesh = trimesh.creation.box(
                    extents=np.fromstring(element.get("size"), sep=" ")
                )
            else:
                raise ValueError("Expected a mesh or box collision geometry")
            mesh.apply_transform(origin(collision.find("origin")))
            name = f"{link.get('name')}/{i}"
            members[name] = link.get("name")
            robot.add_object(name, mesh)
            if "finger" not in link.get("name"):
                cloth_obstacles.add_object(name, mesh)
    obstacles = CollisionManager()
    for name, center, size in supports(cfg):
        matrix = np.eye(4)
        matrix[:3, 3] = center
        mesh = trimesh.creation.box(extents=size)
        obstacles.add_object(name, mesh, matrix)
    origins = {j.get("name"): origin(j.find("origin")) for j in joints}
    records = {}
    cloth_records = {}
    cloth_support_records = {}
    cloth_distance = []
    cloth_triangles = (
        None if states is None else np.load(inputs / "cloth_mesh.npz")["triangles"]
    )
    base = np.eye(4)
    base[:3, 3] = cfg["robot"]["base_position"]
    total = (len(times) - 1) * args.substeps + 1
    for frame in range(0, total, args.stride):
        outer = min(frame // args.substeps, len(times) - 2)
        weight = frame / args.substeps - outer
        target = (1 - weight) * qrows[outer] + weight * qrows[outer + 1]
        values = dict(zip(commands["joint_names"], target))
        transforms = {"base_link": base}
        pending = joints.copy()
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in transforms]
            if not ready:
                raise ValueError("Disconnected URDF")
            for joint in ready:
                axis_element = joint.find("axis")
                axis = np.fromstring(
                    (
                        axis_element.get("xyz", "1 0 0")
                        if axis_element is not None
                        else "1 0 0"
                    ),
                    sep=" ",
                )
                value = float(values.get(joint.get("name"), 0.0))
                motion = np.eye(4)
                if joint.get("type") in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif joint.get("type") == "prismatic":
                    motion[:3, 3] = axis * value
                transforms[joint.find("child").get("link")] = (
                    transforms[joint.find("parent").get("link")]
                    @ origins[joint.get("name")]
                    @ motion
                )
                pending.remove(joint)
        if states is not None and "link_poses" in states:
            if args.substeps != 1:
                raise ValueError(
                    "Actual link matrices must not be linearly interpolated"
                )
            transforms = dict(
                zip(
                    manifest["link_pose_order"],
                    states["link_poses"][frame],
                    strict=True,
                )
            )
        for name, link in members.items():
            robot.set_transform(name, transforms[link])
            if "finger" not in link:
                cloth_obstacles.set_transform(name, transforms[link])
        if states is not None:
            cloth = trimesh.Trimesh(
                states["particles"][frame], cloth_triangles, process=False
            )
            _, names = cloth_obstacles.in_collision_single(cloth, return_names=True)
            _, support_names = obstacles.in_collision_single(cloth, return_names=True)
            for name in support_names:
                cloth_support_records.setdefault(name, []).append(frame)
            if args.distance:
                cloth_distance.append(
                    max(0, cloth_obstacles.min_distance_single(cloth))
                )
            for name in names:
                cloth_records.setdefault(name, []).append(frame)
        _, internal = robot.in_collision_internal(return_names=True)
        _, external = robot.in_collision_other(obstacles, return_names=True)
        pairs = []
        for a, b in internal:
            pair_roots = tuple(sorted((root(members[a]), root(members[b]))))
            if pair_roots[0] != pair_roots[1] and pair_roots not in adjacent:
                pairs.append(("self", a, b))
        pairs += [("support", a, b) for a, b in external]
        for kind, a, b in pairs:
            key = f"{kind}: {a} | {b}"
            if key not in records:
                records[key] = dict(kind=kind, pair=[a, b], frames=[])
            records[key]["frames"].append(frame)
        if frame % 60 == 0:
            print("mesh check", frame, "pairs", len(pairs), flush=True)
    if states is not None and args.distance:
        np.savez_compressed(
            args.out.parent / "cloth_robot_clearance.npz",
            time=states["time"][:: args.stride],
            distance_m=np.array(cloth_distance),
        )
    report = dict(
        input_sha256={
            str(p.relative_to(args.run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                inputs / "config.json",
                inputs / "joint_commands.npz",
                inputs / "w1_v030/robot.urdf",
                inputs / "cloth_mesh.npz",
            )
        },
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method="FCL authored collision triangles plus visual-only cameras/stands; fixed components and directly joined components excluded from self checks; sampled poses only, no continuous collision or positive-clearance guarantee",
        pose_source=(
            "baked targets"
            if states is None
            else (
                "actual link matrices"
                if "link_poses" in states
                else "FK of actual joint positions"
            )
        ),
        states_sha256=(
            None
            if states is None
            else hashlib.sha256(
                (args.run / "preview/states.npz").read_bytes()
            ).hexdigest()
        ),
        stride=args.stride,
        substeps=args.substeps,
        sample_dt_s=cfg["solver"]["dt"] * args.stride / args.substeps,
        positive_distance_calculated=args.distance,
        checked_frames=len(range(0, total, args.stride)),
        intersections=list(records.values()),
        cloth_support_check=(
            None
            if states is None
            else dict(
                method="FCL recorded cloth triangles against the authored rectangular tabletop and table-leg box meshes",
                intersections=[
                    dict(
                        geometry=name,
                        first_frame=min(frames),
                        last_frame=max(frames),
                        frame_count=len(frames),
                    )
                    for name, frames in cloth_support_records.items()
                ],
            )
        ),
        cloth_robot_check=(
            None
            if states is None
            else dict(
                method="FCL recorded cloth triangles against every authored non-finger robot collision mesh; intentional finger grip contact excluded, arms/body/gripper bases included",
                intersections=[
                    dict(
                        geometry=name,
                        first_frame=min(frames),
                        last_frame=max(frames),
                        frame_count=len(frames),
                    )
                    for name, frames in cloth_records.items()
                ],
                min_surface_distance_m=(
                    float(min(cloth_distance)) if cloth_distance else None
                ),
            )
        ),
    )
    write_json(args.out, report)
    print(
        json.dumps(
            [
                {
                    **r,
                    "frames": [min(r["frames"]), max(r["frames"])],
                    "count": len(r["frames"]),
                }
                for r in records.values()
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
