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
"""Measure sampled block/hand triangle penetration from M05 states using FCL.

Run: native Python check_grasp.py --run <directory> --out <report.json>
Reported FCL depths are sampled geometric contact depths, not a force estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_inflatable_pouch_load.tools.check_clearance import (
    origin,
)

__all__ = ["main"]
LINKS = ("left_gripper_base", "left_finger1", "left_finger2")


def main() -> None:
    """Check collision and visual hand meshes against the recorded rigid cube."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fcl-path", type=Path, default=Path("/tmp/m05_fcl"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.fcl_path))
    import trimesh
    from trimesh.collision import CollisionManager

    run = args.run
    cfg = json.loads((run / "manifest.json").read_text())
    state = np.load(run / "preview/states.npz")
    asset = run / "inputs/w1_v030"
    urdf = ET.parse(asset / "robot.urdf").getroot()
    joints = list(urdf.findall("joint"))
    origins = {j.get("name"): origin(j.find("origin")) for j in joints}
    managers = {}
    members = {}
    for kind in ("collision", "visual"):
        manager = CollisionManager()
        names = []
        for link in urdf.findall("link"):
            if link.get("name") not in LINKS:
                continue
            for i, element in enumerate(link.findall(kind)):
                geometry = element.find("geometry/mesh")
                mesh = trimesh.load(asset / geometry.get("filename"), force="mesh")
                mesh.apply_scale(np.fromstring(geometry.get("scale", "1 1 1"), sep=" "))
                mesh.apply_transform(origin(element.find("origin")))
                name = f"{link.get('name')}/{i}"
                names.append(name)
                manager.add_object(name, mesh)
        managers[kind] = manager
        members[kind] = names
    cube = trimesh.creation.box(extents=[cfg["scene"]["block_size_m"]] * 3)
    data = []
    for frame in range(0, len(state["time"]), args.stride):
        if "left_hand_poses" in state:
            poses = dict(zip(LINKS, state["left_hand_poses"][frame]))
        else:
            values = dict(zip(cfg["joint_state_order"], state["qpos"][frame]))
            poses = {"base_link": np.eye(4)}
            poses["base_link"][:3, 3] = cfg["scene"].get(
                "robot_base_position", [0, 0, 0]
            )
            pending = joints.copy()
            while pending:
                ready = [j for j in pending if j.find("parent").get("link") in poses]
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
                    q = values.get(joint.get("name"), 0.0)
                    motion = np.eye(4)
                    if joint.get("type") in ("revolute", "continuous"):
                        motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                    elif joint.get("type") == "prismatic":
                        motion[:3, 3] = axis * q
                    poses[joint.find("child").get("link")] = (
                        poses[joint.find("parent").get("link")]
                        @ origins[joint.get("name")]
                        @ motion
                    )
                    pending.remove(joint)
        row = {"time": float(state["time"][frame])}
        for kind, manager in managers.items():
            for name in members[kind]:
                manager.set_transform(name, poses[name.split("/")[0]])
            hit, contacts = manager.in_collision_single(
                cube, transform=state["block_pose"][frame], return_data=True
            )
            row[kind + "_penetration_m"] = (
                max([float(c.depth) for c in contacts], default=0.0) if hit else 0.0
            )
        data.append(row)
    phases = dict((name, end) for end, name in cfg["scene"]["phases"])
    carry = [
        r for r in data if phases["block settle"] <= r["time"] <= phases["block carry"]
    ]
    phase_maxima = {}
    for name, start, end in (
        ("grasp_close", phases["block descend"], phases["block settle"]),
        ("lift_and_carry", phases["block settle"], phases["block carry"]),
        ("lower_and_hold", phases["block carry"], phases["block load hold"]),
        ("release_and_retreat", phases["block load hold"], phases["left retreat"]),
    ):
        selected = [r for r in data if start - 1e-8 <= r["time"] <= end + 1e-8]
        phase_maxima[name] = {
            "range_s": [start, end],
            "collision_penetration_m": max(
                r["collision_penetration_m"] for r in selected
            ),
            "visual_penetration_m": max(r["visual_penetration_m"] for r in selected),
        }
    bag_mesh = np.load(run / "inputs/bag_mesh.npz")
    bag_gaps = []
    last_second = np.flatnonzero(state["time"] >= phases["observe"] - 1 - 1e-8)
    for frame in last_second[:: args.stride]:
        bag_manager = CollisionManager()
        bag_manager.add_object(
            "bag",
            trimesh.Trimesh(
                vertices=state["bag"][frame], faces=bag_mesh["triangles"], process=False
            ),
        )
        bag_gaps.append(
            float(
                bag_manager.min_distance_single(
                    cube, transform=state["block_pose"][frame]
                )
            )
        )
    report = {
        "method": "FCL sampled actual hand collision/visual triangles vs recorded rigid cube; maximum returned contact penetration depth",
        "stride": args.stride,
        "states": len(data),
        "carry_range_s": [phases["block settle"], phases["block carry"]],
        "carry_max_collision_penetration_m": max(
            r["collision_penetration_m"] for r in carry
        ),
        "carry_max_visual_penetration_m": max(r["visual_penetration_m"] for r in carry),
        "full_episode_max_visual_penetration_m": max(
            r["visual_penetration_m"] for r in data
        ),
        "phase_maxima": phase_maxima,
        "loaded_block_bag_closest_surface_gap_m": {
            "median": float(np.median(bag_gaps)),
            "min": min(bag_gaps),
            "max": max(bag_gaps),
        },
        "samples": data,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
