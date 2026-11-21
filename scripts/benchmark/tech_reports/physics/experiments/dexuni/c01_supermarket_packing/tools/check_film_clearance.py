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
"""Check C01 hand and worktop clearance from film using recorded FCL geometry.

Run: python check_film_clearance.py --run <run> --stride 3
Uses actual named joint states; reports sampled geometry, not continuous clearance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["main"]


def _origin(element: ET.Element | None) -> np.ndarray:
    """Read one URDF translation and fixed rotation."""
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return matrix


def main() -> None:
    """Save sampled film intersections and distances to authored hands and worktop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    sys.path.insert(0, "/tmp/m05_fcl")
    import trimesh
    from trimesh.collision import CollisionManager

    run = args.run
    with np.load(run / "preview/states.npz") as d:
        time, qpos, bag, tcp = (d[k] for k in ("time", "qpos", "bag", "tcp"))
    with np.load(run / "inputs/goods_meshes.npz") as d:
        triangles = d["bag_triangles"]
    with np.load(run / "inputs/joint_commands.npz") as d:
        names = d["joint_names"].tolist()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["joint_state_order"] == names
    asset = run / "inputs/w1_v030"
    urdf = ET.parse(asset / "robot.urdf").getroot()
    joints = list(urdf.findall("joint"))
    origins = {j.get("name"): _origin(j.find("origin")) for j in joints}
    parents = {j.find("child").get("link") for j in joints}
    root = next(
        link.get("name")
        for link in urdf.findall("link")
        if link.get("name") not in parents
    )
    utils = ast.parse((run / "source/reference/dexuni_w1_two_way_utils.py").read_text())
    tcp_node = next(
        n
        for n in utils.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TCP" for t in n.targets)
    )
    tcp_local = np.r_[ast.literal_eval(tcp_node.value.args[0]), 1.0]
    hands = CollisionManager()
    members = {}
    for link in urdf.findall("link"):
        for i, collision in enumerate(link.findall("collision")):
            element = collision.find("geometry/mesh")
            if element is None or not any(
                part in element.get("filename")
                for part in ("left_hand/", "right_hand/")
            ):
                continue
            mesh = trimesh.load(asset / element.get("filename"), force="mesh")
            mesh.apply_scale(np.fromstring(element.get("scale", "1 1 1"), sep=" "))
            mesh.apply_transform(_origin(collision.find("origin")))
            name = f"{link.get('name')}/{i}"
            members[name] = link.get("name")
            hands.add_object(name, mesh)
    assert members
    common_spec = importlib.util.spec_from_file_location(
        "recorded_c01_common", run / "source/c01_common.py"
    )
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    worktop = CollisionManager()
    for name, centre, size in common.worktop_pieces():
        if name.startswith("table_"):
            transform = np.eye(4)
            transform[:3, 3] = centre
            worktop.add_object(name, trimesh.creation.box(extents=size), transform)
    distances = []
    collisions = []
    worktop_distances = []
    worktop_collisions = []
    worst_tcp = 0.0
    frames = list(range(0, len(time), args.stride))
    if frames[-1] != len(time) - 1:
        frames.append(len(time) - 1)
    for frame in frames:
        values = dict(zip(names, qpos[frame], strict=True))
        transforms = {root: np.eye(4)}
        pending = joints.copy()
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in transforms]
            if not ready:
                raise ValueError("Disconnected URDF")
            for j in ready:
                axis = j.find("axis")
                axis = np.fromstring(
                    axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "
                )
                value = float(values.get(j.get("name"), 0.0))
                motion = np.eye(4)
                if j.get("type") in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif j.get("type") == "prismatic":
                    motion[:3, 3] = axis * value
                transforms[j.find("child").get("link")] = (
                    transforms[j.find("parent").get("link")]
                    @ origins[j.get("name")]
                    @ motion
                )
                pending.remove(j)
        fk = np.array(
            [
                (transforms[f"{s}_gripper_base"] @ tcp_local)[:3]
                for s in ("left", "right")
            ]
        )
        worst_tcp = max(worst_tcp, float(np.linalg.norm(fk - tcp[frame], axis=1).max()))
        if worst_tcp > 2e-5:
            raise ValueError(f"Recorded pose/FK mismatch {worst_tcp}")
        for name, link in members.items():
            hands.set_transform(name, transforms[link])
        film = trimesh.Trimesh(bag[frame], triangles, process=False)
        hit, pairs = hands.in_collision_single(film, return_names=True)
        distances.append(float(hands.min_distance_single(film)))
        if hit:
            collisions.append(
                dict(frame=frame, time_s=float(time[frame]), hand_shapes=sorted(pairs))
            )
        hit, pairs = worktop.in_collision_single(film, return_names=True)
        worktop_distances.append(float(worktop.min_distance_single(film)))
        if hit:
            worktop_collisions.append(
                dict(frame=frame, time_s=float(time[frame]), shapes=sorted(pairs))
            )
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    np.savez_compressed(
        output / "film_clearance.npz",
        time=time[frames],
        distance_m=np.asarray(distances),
        worktop_distance_m=np.asarray(worktop_distances),
    )
    result = dict(
        method="FCL authored hand collision meshes versus recorded film triangles; actual named qpos FK checked against recorded TCP; discrete poses only, no continuous guarantee",
        sample_count=len(frames),
        stride=args.stride,
        hand_shapes=members,
        max_fk_tcp_error_m=worst_tcp,
        minimum_surface_distance_m=min(distances),
        collision_samples=collisions,
        worktop_minimum_surface_distance_m=min(worktop_distances),
        worktop_collision_samples=worktop_collisions,
        worktop_scope="Recorded common layout's three worktop slab boxes versus film triangles at the same discrete poses",
        states_sha256=hashlib.sha256(
            (run / "preview/states.npz").read_bytes()
        ).hexdigest(),
    )
    (output / "film_clearance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
