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

from __future__ import annotations
from types import SimpleNamespace as NS
import numpy as np
import torch
import yourdfpy
from scipy.spatial.transform import Rotation


def distal_fixture(directory, mode="nominal"):
    sections = ['<robot name="distal"><link name="base"/>']
    names = []
    for side, x, y in (("left", -0.1, 0.0), ("right", 0.1, 0.02)):
        sections.append(
            f'<link name="{side}_root"/><joint name="{side}_mount" type="fixed"><parent link="base"/><child link="{side}_root"/><origin xyz="{x} {y} .3"/></joint>'
        )
        previous = f"{side}_root"
        for i, axis in ((1, "1 0 0"), (2, "0 1 0")):
            link = f"{side}_{i}"
            joint = f"{side}_joint{i}"
            names.append(joint)
            sections.append(
                f'<link name="{link}"><collision><geometry><box size=".09 .06 .05"/></geometry></collision></link><joint name="{joint}" type="prismatic"><parent link="{previous}"/><child link="{link}"/><axis xyz="{axis}"/><limit lower="-1" upper="1" effort="10" velocity="10"/></joint>'
            )
            previous = link
        sections.append(
            f'<link name="{side}_hand"><collision><geometry><box size=".08 .08 .04"/></geometry></collision></link><joint name="{side}_hand_mount" type="fixed"><parent link="{previous}"/><child link="{side}_hand"/><origin xyz="0 0 .03"/></joint>'
        )
    sections.append("</robot>")
    path = directory / "distal.urdf"
    path.write_text("".join(sections))
    model = yourdfpy.URDF.load(
        path,
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    qpos = torch.tensor([[0.0, 0.02, 0.01, -0.01]])
    model.update_cfg(dict(zip(names, qpos[0].tolist())))
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("z", 0.3).as_matrix()
    world[:3, 3] = [0.3, -0.2, 0.4]
    tcp = np.eye(4)
    tcp[2, 3] = 0.05
    solvers = {
        side: NS(
            root_link_name=f"{side}_root",
            end_link_name=f"{side}_2",
            get_tcp=lambda: tcp.copy(),
        )
        for side in ("left", "right")
    }
    ids = {"left": [0, 1], "right": [2, 3]}
    reads = []

    def positions():
        reads.append("qpos")
        return qpos + 0.01 if mode == "changed" and len(reads) > 1 else qpos

    def link(name, **kwargs):
        value = world @ model.get_transform(name)
        if mode == "calibration" and name == "left_2":
            value[0, 3] += 0.001
        if mode == "hand_offset" and name.endswith("_hand"):
            value[2, 3] += 0.005
        return torch.tensor(value)[None]

    robot = NS(
        cfg=NS(fpath=str(path)),
        joint_names=names,
        control_parts={"left": object(), "right": object()},
        get_qpos=positions,
        get_local_pose=lambda **kwargs: torch.tensor(world)[None],
        get_solver=lambda part: solvers.get(part),
        get_joint_ids=lambda part: ids[part],
        get_link_pose=link,
        compute_fk=lambda qpos, name, **kwargs: torch.tensor(
            world @ model.get_transform(solvers[name].end_link_name) @ tcp
        )[None],
    )
    if mode == "ambiguous":
        robot.control_parts["third"] = object()
        solvers["third"] = solvers["right"]
        ids["third"] = [2, 3]
    actual = robot.compute_fk(qpos[:, :2], name="left").float()
    stage = actual.clone()
    stage[:, 2, 3] += 0.1
    final = actual.clone()
    final[:, 0, 3] += 0.1
    held = NS(semantics=NS(entity_id="object"), object_to_eef=torch.eye(4)[None])
    context = NS(
        robot=NS(qpos=qpos, root_pose=torch.tensor(world)[None]),
        scene=NS(entities={"object": NS(pose=actual, confidence=1.0)}),
        task=NS(get_held_object=lambda key: held),
    )
    target = NS(control_part="left", joint_ids=(0, 1))
    endpoint = NS(task_state_key="left", require_target=lambda expected: target)
    request = NS(
        goal=NS(object_target_pose=torch.stack((stage, final), 1)),
        binding=NS(endpoint=lambda *args: endpoint),
    )
    return robot, context, request
