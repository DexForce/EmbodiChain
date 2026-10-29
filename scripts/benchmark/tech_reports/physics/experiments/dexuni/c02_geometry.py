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
"""Audit nonadjacent robot links and table against actual recorded C02 poses.

Run with /home/oem/code/engine/dexsim/.env-3.12/bin/python; existing /tmp/m05_fcl supplies FCL.
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

__all__ = ["main"]


def origin(element: ET.Element | None) -> np.ndarray:
    """Read one URDF collision origin."""
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return matrix


def main() -> None:
    """Save every sampled collision, excluding only mechanical adjacency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    sys.path.insert(0, "/tmp/m05_fcl")
    import trimesh
    from trimesh.collision import CollisionManager

    with np.load(run / "preview/states.npz") as a:
        s = dict(a)
    manifest = json.loads((run / "manifest.json").read_text())
    names = manifest["link_state_order"]
    asset = run / "inputs/w1_v030"
    urdf = ET.parse(asset / "robot.urdf").getroot()
    joints = list(urdf.findall("joint"))
    roots = {link.get("name"): link.get("name") for link in urdf.findall("link")}

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
            e = collision.find("geometry/mesh")
            assert e is not None
            mesh = trimesh.load(asset / e.get("filename"), force="mesh")
            mesh.apply_scale(np.fromstring(e.get("scale", "1 1 1"), sep=" "))
            mesh.apply_transform(origin(collision.find("origin")))
            name = f"{link.get('name')}/{i}"
            members[name] = link.get("name")
            assert members[name] in names
            robot.add_object(name, mesh)
    table = CollisionManager()
    matrix = np.eye(4)
    matrix[:3, 3] = manifest["table_centre"]
    table.add_object(
        "table",
        trimesh.creation.box(extents=np.array(manifest["table_half"]) * 2),
        matrix,
    )
    self_hits = []
    table_hits = []
    for frame, t in enumerate(s["time"]):
        pose = s["links"][frame]
        transforms = np.repeat(np.eye(4)[None], len(names), 0)
        transforms[:, :3, :3] = Rotation.from_quat(pose[:, :4]).as_matrix()
        transforms[:, :3, 3] = pose[:, 4:]
        for name, link in members.items():
            robot.set_transform(name, transforms[names.index(link)])
        hit, pairs = robot.in_collision_internal(return_names=True)
        relevant = []
        for a, b in pairs:
            aa, bb = root(members[a]), root(members[b])
            if aa != bb and tuple(sorted((aa, bb))) not in adjacent:
                relevant.append([a, b])
        if relevant:
            self_hits.append(dict(frame=frame, time_s=float(t), pairs=sorted(relevant)))
        hit, pairs = robot.in_collision_other(table, return_names=True)
        if hit:
            table_hits.append(
                dict(
                    frame=frame, time_s=float(t), pairs=sorted([list(p) for p in pairs])
                )
            )
        if frame % 600 == 0:
            print("C02 actual geometry", frame, "/", len(s["time"]) - 1, flush=True)
    result = dict(
        method="FCL authored collision triangles and actual named link poses, all saved outer steps; same rigid group and direct mechanical neighbors excluded; no continuous-collision guarantee",
        samples=len(s["time"]),
        self_collision_samples=self_hits,
        table_collision_samples=table_hits,
        collision_shape_count=len(members),
        states_sha256=hashlib.sha256(
            (run / "preview/states.npz").read_bytes()
        ).hexdigest(),
    )
    (run / "analysis/robot_geometry.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("self hits", len(self_hits), "table hits", len(table_hits), flush=True)


if __name__ == "__main__":
    main()
