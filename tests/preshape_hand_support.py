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

"""Real URDF/FK hand geometry fixture; no scripted grasp acceptance."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
import trimesh
import yourdfpy
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg


class PreshapeRobot:
    def __init__(self, path):
        self.model = yourdfpy.URDF.load(
            str(path),
            load_meshes=True,
            build_scene_graph=True,
            build_collision_scene_graph=True,
        )
        self.joint_names = ["rail", "finger_left", "finger_right"]
        self.link_names = list(self.model.link_map)
        self.qpos = torch.tensor([[0.4, 0.0, 0.0]])
        self.reads = []
        self.fk_calls = []
        tcp = torch.eye(4)
        tcp[2, 3] = 0.1
        self.cfg = SimpleNamespace(
            fpath=str(path),
            control_parts={"custom_hand": ["finger_right", "finger_left"]},
            solver_cfg={
                "custom_arm": SimpleNamespace(end_link_name="mount", tcp=tcp.tolist())
            },
        )
        self.device = torch.device("cpu")
        self.uid = "qualified_fixture"
        self.num_instances = 1

    def get_qpos(self):
        self.reads.append("qpos")
        return self.qpos.clone()

    def get_parent_joint_chain(self, link):
        parents = {joint.child: joint for joint in self.model.robot.joints}
        result = []
        while link in parents:
            joint = parents[link]
            result.append(
                SimpleNamespace(
                    name=joint.name,
                    parent_link_name=joint.parent,
                    child_link_name=joint.child,
                )
            )
            link = joint.parent
        return list(reversed(result))

    def compute_fk(self, *, qpos, link_names, qpos_joint_names):
        self.fk_calls.append(
            {
                "qpos": qpos.tolist(),
                "links": list(link_names),
                "joints": list(qpos_joint_names),
            }
        )
        rows = []
        for row in qpos:
            self.model.update_cfg(
                {name: float(value) for name, value in zip(qpos_joint_names, row)}
            )
            rows.append(
                np.stack([self.model.get_transform(link) for link in link_names])
            )
        return torch.tensor(np.stack(rows), dtype=qpos.dtype, device=qpos.device)

    def get_link_vert_face(self, link):
        meshes = []
        for collision in self.model.link_map[link].collisions:
            mesh = trimesh.creation.box(extents=collision.geometry.box.size)
            if collision.origin is not None:
                mesh.apply_transform(collision.origin)
            meshes.append(mesh)
        if not meshes:
            return torch.empty(0, 3), torch.empty(0, 3, dtype=torch.long)
        mesh = trimesh.util.concatenate(meshes)
        return (
            torch.tensor(mesh.vertices.copy()).float(),
            torch.tensor(mesh.faces.copy()).long(),
        )


def preshape_fixture(tmp_path):
    source = json.loads(
        (
            Path(__file__).parent / "sim/utility/fixtures/hand_geometry_f1.json"
        ).read_text()
    )["urdf"]
    source = source.replace(
        '<axis xyz="1 0 0"/>',
        '<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>',
    )
    path = tmp_path / "hand.urdf"
    path.write_text(source)
    robot = PreshapeRobot(path)
    cfg = RobotHandGeometryCfg(
        motion_part="custom_arm",
        hand_part="custom_hand",
        hand_root_link="hand_base",
        pad_link_names=("negative_finger_pad", "positive_finger_pad"),
        open_qpos=(0.06, -0.06),
        closed_qpos=(0.01, -0.01),
        sample_count=65,
        body_max_edge=0.01,
        pad_max_edge=0.004,
        max_subdivisions=10,
    )
    return robot, cfg
