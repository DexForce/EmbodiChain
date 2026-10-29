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
"""Check recorded M03 FK poses with exact triangle intersections using python-fcl.

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
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    boxes,
    write_json,
)

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
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--fcl-path",
        type=Path,
        default=Path(__file__).resolve().parents[5].parents[2]
        / "outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place/.tool_deps",
    )
    args = parser.parse_args()
    if args.stride < 1 or args.substeps < 1:
        parser.error("stride and substeps must be positive")
    sys.path.insert(0, str(args.fcl_path))
    import trimesh
    from trimesh.collision import CollisionManager

    cfg = json.loads((args.inputs / "config.json").read_text())
    commands = np.load(args.inputs / "joint_commands.npz")
    asset = args.inputs / "w1_v030"
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
    members = {}
    for link in urdf.findall("link"):
        for i, collision in enumerate(link.findall("collision")):
            element = collision.find("geometry/mesh")
            if element is None:
                raise ValueError("Expected mesh-only robot collision geometry")
            mesh = trimesh.load(asset / element.get("filename"), force="mesh")
            mesh.apply_scale(np.fromstring(element.get("scale", "1 1 1"), sep=" "))
            mesh.apply_transform(origin(collision.find("origin")))
            name = f"{link.get('name')}/{i}"
            members[name] = link.get("name")
            robot.add_object(name, mesh)
    supports = CollisionManager()
    for name, center, size, _, collision in boxes(cfg):
        if collision:
            matrix = np.eye(4)
            matrix[:3, 3] = center
            supports.add_object(name, trimesh.creation.box(extents=size), matrix)
    origins = {j.get("name"): origin(j.find("origin")) for j in joints}
    records = {}
    base = np.eye(4)
    base[:3, 3] = cfg["robot"]["base_position"]
    total = (len(commands["time"]) - 1) * args.substeps + 1
    for frame in range(0, total, args.stride):
        outer = min(frame // args.substeps, len(commands["time"]) - 2)
        weight = frame / args.substeps - outer
        target = (1 - weight) * commands["targets"][outer] + weight * commands[
            "targets"
        ][outer + 1]
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
        for name, link in members.items():
            robot.set_transform(name, transforms[link])
        _, internal = robot.in_collision_internal(return_names=True)
        _, external = robot.in_collision_other(supports, return_names=True)
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
        if frame % 240 == 0:
            print("mesh check", frame, "pairs", len(pairs), flush=True)
    report = dict(
        input_sha256={
            str(p.relative_to(args.inputs)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.inputs / "config.json",
                args.inputs / "joint_commands.npz",
                args.inputs / "w1_v030/robot.urdf",
            )
        },
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method="FCL exact collision triangles; fixed components and directly joined components excluded from self checks; sampled poses only, no positive-clearance guarantee",
        stride=args.stride,
        substeps=args.substeps,
        sample_dt_s=float(cfg["solver"]["dt"]) / args.substeps,
        checked_frames=len(range(0, total, args.stride)),
        intersections=list(records.values()),
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
