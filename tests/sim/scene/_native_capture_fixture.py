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

"""Named analytic FK and native SDK-shaped read-only test ports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

URDF = """<robot name="native_fixture">
<link name="base"/><link name="mount"/><link name="hand_base"/>
<link name="negative_finger_pad"/><link name="positive_finger_pad"/>
<joint name="rail" type="prismatic"><parent link="base"/><child link="mount"/></joint>
<joint name="mounting" type="fixed"><parent link="mount"/><child link="hand_base"/></joint>
<joint name="finger_left" type="prismatic"><parent link="hand_base"/><child link="negative_finger_pad"/><mimic joint="finger_right" multiplier="-1"/></joint>
<joint name="finger_right" type="prismatic"><parent link="hand_base"/><child link="positive_finger_pad"/></joint>
</robot>"""


class Box:
    def __init__(self, half, pose=None):
        self.half_extents = np.asarray(half)
        self.local_pose = np.eye(4) if pose is None else np.asarray(pose)


class Convex:
    def __init__(self, vertices, faces, scale, pose):
        self.vertices, self.triangles = vertices, faces
        self.scale, self.local_pose = scale, pose


class Body:
    def __init__(self, shapes, rest=0.0002, contact=0.003, enabled=True):
        self.shapes, self.rest, self.contact, self.enabled = (
            shapes,
            rest,
            contact,
            enabled,
        )

    def is_collision_enable(self):
        return self.enabled

    def get_shape_count(self):
        return len(self.shapes)

    def get_shape_geometry(self, index):
        return self.shapes[index]

    def get_rest_offset(self):
        return self.rest

    def get_contact_offset(self):
        return self.contact

    def destroy(self):
        raise AssertionError("Queries never destroy borrowed native bodies.")


class Robot:
    uid = "robot"
    joint_names = ["rail", "finger_left", "finger_right"]

    def __init__(self, path):
        self.qpos = torch.tensor([[0.3, -0.11, 0.11]])
        self.pose = torch.eye(4)[None]
        self.pose[0, 0, 3] = 0.2
        tcp = torch.eye(4)
        tcp[2, 3] = 0.1
        self.cfg = NS(
            fpath=str(path),
            control_parts={"arm": ["rail"], "grip": ["finger_right", "finger_left"]},
            solver_cfg={"arm": NS(end_link_name="mount", tcp=tcp.tolist())},
        )
        self.calls, self.reads = [], 0

    def get_qpos(self):
        self.reads += 1
        return self.qpos

    def get_local_pose(self, *, to_matrix):
        assert to_matrix
        return self.pose

    def fk(self, qpos, link_names):
        output = []
        for name in link_names:
            pose = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(
                len(qpos), 1, 1
            )
            pose[:, 0, 3] = qpos[:, 0]
            if name == "negative_finger_pad":
                pose[:, 0, 3] += qpos[:, 1]
                pose[:, 2, 3] = 0.1
            if name == "positive_finger_pad":
                pose[:, 0, 3] += qpos[:, 2]
                pose[:, 2, 3] = 0.1
            output.append(pose)
        return torch.stack(output, 1)

    def compute_fk(self, *, qpos, link_names, qpos_joint_names):
        self.calls.append(
            {
                "names": list(link_names),
                "joint_names": list(qpos_joint_names),
                "qpos": qpos.tolist(),
            }
        )
        return self.fk(qpos, link_names)

    def set_qpos(self, *args, **kwargs):
        raise AssertionError("No robot writes in native capture.")

    def update(self, *args, **kwargs):
        raise AssertionError("No simulation update in native capture.")


class NativeRobot:
    arena_name = "arena0"
    path = "/arena0/robot"

    def __init__(self, robot):
        self.robot = robot
        self.bodies = {
            name: Body((Box((0.01, 0.01, 0.01)),))
            for name in ["hand_base", "negative_finger_pad", "positive_finger_pad"]
        }
        self.render_articulation = NS(get_physical_body=self.bodies.get)

    def get_name(self):
        return "robot"

    def get_link_names(self):
        return list(self.bodies)

    def get_link_pose(self, name):
        return (
            self.robot.pose[0] @ self.robot.fk(self.robot.qpos, [name])[0, 0]
        ).numpy()


class Rigid:
    arena_name = "arena0"

    def __init__(self, uid, body, xyz):
        self.uid, self.body = uid, body
        self.path = "/arena0/" + uid
        self.pose = torch.eye(4)[None]
        self.pose[0, :3, 3] = torch.tensor(xyz)

    def get_name(self):
        return self.uid

    def native(self):
        return NS(get_physical_body=lambda: self.body)

    def get_local_pose(self, *, to_matrix):
        assert to_matrix
        return self.pose


class Simulation:
    num_envs = 1
    is_default_backend = True

    def __init__(self, robot):
        self.robot = robot
        self.native_robot = NativeRobot(robot)
        self.rigids = {
            "source": Rigid(
                "source",
                Body((Box((0.07, 0.07, 0.07)),), rest=0.0003, contact=0.004),
                (0.5, 0, 0.1),
            ),
            "support": Rigid(
                "support",
                Body((Box((0.5, 0.5, 0.025)),), rest=0.0, contact=0.02),
                (0.5, 0, -0.2),
            ),
        }
        self.handles = {
            obj.path: obj for obj in [self.native_robot, *self.rigids.values()]
        }
        self.spawn_result = NS(
            get_paths=lambda: list(self.handles), get=lambda path: self.handles[path]
        )

    def get_robot(self, uid):
        return self.robot if uid == self.robot.uid else None

    def get_rigid_object_uid_list(self):
        return list(self.rigids)

    def get_rigid_object(self, uid):
        return self.rigids[uid]

    def step(self, *args, **kwargs):
        raise AssertionError("No physics steps in native capture.")


def setup_case(directory: Path):
    path = directory / "hand.urdf"
    path.write_text(URDF)
    robot = Robot(path)
    simulation = Simulation(robot)
    cfg = RobotHandGeometryCfg(
        motion_part="arm",
        hand_part="grip",
        hand_root_link="hand_base",
        pad_link_names=("positive_finger_pad", "negative_finger_pad"),
        open_qpos=(99, -99),
        closed_qpos=(-99, 99),
        sample_count=65,
        body_max_edge=0.01,
        pad_max_edge=0.004,
        max_subdivisions=10,
    )
    return (
        robot,
        simulation,
        cfg,
        torch.tensor([[0.12, -0.12]]),
        torch.tensor([[0.06, -0.06]]),
    )


def parts_record(query):
    def record(array):
        value = np.asarray(array)
        return {
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }

    return [
        {
            "name": p.name,
            "pad": p.pad,
            "rest_offset": p.rest_offset,
            "contact_offset": p.contact_offset,
            "tcp_poses": record(p.tcp_poses),
            "meshes": [
                {"vertices": record(m.vertices), "faces": record(m.faces)}
                for m in p.meshes
            ],
        }
        for p in query.parts
    ]
