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

"""Private URDF point kinematics oracle; no MuJoCo in the kinematic model."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
REFERENCE = ROOT / "alignment-point-response-pgs"


class PointKinematics:
    def __init__(self, root=REFERENCE):
        self.asset = json.loads((root / "robot/robot.json").read_text())
        self.contract = json.loads((root / "physical-contract.json").read_text())
        self.names = self.asset["joint_names"]
        self.bindings = {x["name"]: x for x in self.asset["source_frames"]}
        xml = ET.parse(root / "robot/robot.urdf").getroot()
        pending = list(xml.findall("joint"))
        children = {j.find("child").attrib["link"] for j in pending}
        (self.root_name,) = {x.attrib["name"] for x in xml.findall("link")} - children
        known = {self.root_name}
        self.joints = []
        while pending:
            remaining = []
            for j in pending:
                parent = j.find("parent").attrib["link"]
                if parent not in known:
                    remaining.append(j)
                    continue
                child = j.find("child").attrib["link"]
                origin = j.find("origin")
                attrs = {} if origin is None else origin.attrib
                offset = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
                rotation = Rotation.from_euler(
                    "xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
                axis = j.find("axis")
                axis = np.fromstring(
                    "1 0 0" if axis is None else axis.attrib["xyz"], sep=" "
                )
                index = (
                    None
                    if j.attrib["type"] == "fixed"
                    else self.names.index(j.attrib["name"])
                )
                self.joints.append((parent, child, offset, rotation, axis, index))
                known.add(child)
            assert len(remaining) < len(pending), "invalid URDF tree"
            pending = remaining
        eq = self.contract["equality"]
        self.points = []
        for i, kind in enumerate(eq["type"]):
            if kind != 0:
                continue
            ends = []
            for k, key in enumerate(("obj1id", "obj2id")):
                binding = self.bindings[self.contract["body_names"][eq[key][i]]]
                frame = np.array(binding["transform"])
                local = (
                    frame[:3, :3] @ np.array(eq["data"][i][k * 3 : k * 3 + 3])
                    + frame[:3, 3]
                )
                ends.append((binding["link"], local, frame[:3, :3]))
            self.points.append((i, ends))

    def evaluate(self, q, v):
        root = np.array(self.asset["root_pose"])
        zero = np.zeros(3)
        # Rotation, origin position, linear/angular velocity and their bias
        # accelerations with generalized acceleration fixed to zero.
        frames = {self.root_name: (root[:3, :3], root[:3, 3], zero, zero, zero, zero)}
        for parent, child, offset, origin, axis, index in self.joints:
            r, p, lv, av, la, aa = frames[parent]
            arm = r @ offset
            world_axis = r @ origin @ axis
            speed = 0 if index is None else v[index]
            angle = 0 if index is None else q[index]
            frames[child] = (
                r @ origin @ Rotation.from_rotvec(axis * angle).as_matrix(),
                p + arm,
                lv + np.cross(av, arm),
                av + world_axis * speed,
                la + np.cross(aa, arm) + np.cross(av, np.cross(av, arm)),
                aa + np.cross(av, world_axis * speed),
            )
        result = []
        for index, endpoints in self.points:
            points = []
            for link, local, joint_rotation in endpoints:
                r, p, lv, av, la, aa = frames[link]
                arm = r @ local
                points.append(
                    (
                        p + arm,
                        lv + np.cross(av, arm),
                        la + np.cross(aa, arm) + np.cross(av, np.cross(av, arm)),
                        r @ joint_rotation,
                        av,
                    )
                )
            a, b = points
            result.append(
                {
                    "index": index,
                    "residual": b[0] - a[0],
                    "velocity": b[1] - a[1],
                    "Jdotv": b[2] - a[2],
                    "rotation_a": a[3],
                    "omega_a": a[4],
                }
            )
        return result


def validate_source():
    import h5py
    import mujoco
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    model = PointKinematics()
    with h5py.File(ROOT / "alignment/official-queries.h5") as f:
        q, v = f["qpos"][:], f["qvel"][:]
    env = CubeEnv(
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
    )
    rows = []
    try:
        env.reset(seed=42)
        for i in range(len(q)):
            env.set_state(q[i], v[i])
            mujoco.mj_forward(env.model, env.data)
            for point in model.evaluate(q[i, :14], v[i, :14]):
                ids = np.flatnonzero(
                    (env.data.efc_type == 0) & (env.data.efc_id == point["index"])
                )
                assert len(ids) == 3
                K, B, d = env.data.efc_KBIP[ids, :3].T
                drift = (
                    -env.data.efc_aref[ids]
                    - B * env.data.efc_vel[ids]
                    - K * d * env.data.efc_pos[ids]
                )
                rows.append(
                    {
                        "query": i,
                        "equality": point["index"],
                        "position_error": float(
                            abs(point["residual"] + env.data.efc_pos[ids]).max()
                        ),
                        "velocity_error": float(
                            abs(point["velocity"] + env.data.efc_vel[ids]).max()
                        ),
                        "Jdotv_error": float(abs(point["Jdotv"] + drift).max()),
                    }
                )
        report = {
            "scope": "100 real-robot point rows, all 50 official q/qvel queries; independent URDF recursive world velocity/bias acceleration",
            "mujoco": mujoco.__version__,
            "rows": rows,
            "maximum": {
                key: max(r[key] for r in rows)
                for key in ("position_error", "velocity_error", "Jdotv_error")
            },
        }
        (
            ROOT / "alignment/point-step-response/robot-kinematics-source.json"
        ).write_text(json.dumps(report, indent=2) + "\n")
        print(report["maximum"], flush=True)
        assert report["maximum"]["position_error"] < 1e-10
        assert report["maximum"]["velocity_error"] < 1e-10
        assert report["maximum"]["Jdotv_error"] < 1e-10
    finally:
        env.close()


if __name__ == "__main__":
    validate_source()
