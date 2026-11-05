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

"""Analytic native-shaped ports; no simulator, provider or native allocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import numpy as np
import torch
import trimesh
from pxr import Usd, UsdGeom

from embodichain.lab.sim.scene.control_source import capture_revolute_source_seal
from embodichain.lab.sim.scene.revolute_geometry import load_twist_geometry
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

knob_source = runpy.run_path(
    str(Path(__file__).parents[1] / "scene" / "_control_source_fixture.py")
)["knob_source"]


def revolute_query_inputs(
    tmp_path, *, mesh_kind="box", root_height=0.8, table_uid="table", ring_count=32
):
    config = knob_source(tmp_path)
    source = tmp_path / "knob.usda"
    stage = Usd.Stage.Open(str(source))
    grip = UsdGeom.Mesh.Get(stage, "/Control/knob/grip")
    count = ring_count
    angles = np.arange(count) * (2 * np.pi / count)
    ring = np.column_stack((0.03 * np.cos(angles), 0.03 * np.sin(angles)))
    vertices = np.concatenate(
        [np.column_stack((ring, np.full(count, z))) for z in (0.01, 0.03)]
    )
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            (
                (index, following, count + following),
                (index, count + following, count + index),
            )
        )
    faces.extend((0, index + 1, index) for index in range(1, count - 1))
    faces.extend(
        (count, count + index, count + index + 1) for index in range(1, count - 1)
    )
    faces = np.asarray(faces)
    grip.GetPointsAttr().Set(vertices.tolist())
    grip.GetFaceVertexCountsAttr().Set([3] * len(faces))
    grip.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
    stage.GetRootLayer().Save()
    metadata_path = source.with_suffix(".twist.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    binding = inspect_revolute_control(config, 90)
    geometry = load_twist_geometry(source, binding)
    seal = capture_revolute_source_seal(source, binding)

    robot_path = tmp_path / "robot.urdf"
    tool = '<box size="0.02 0.02 0.02"/>'
    if mesh_kind != "box":
        vertices = (
            [[0.1, 0, 0.03], [0, 0.1, 0.03], [0.11, 0.01, 0.03]]
            if mesh_kind == "separated_projection"
            else [[-0.005, -0.005, 0.03], [0.005, -0.005, 0.03], [0, 0.005, 0.03]]
        )
        trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False).export(
            tmp_path / "tool.ply"
        )
        tool = '<mesh filename="tool.ply"/>'
    robot_path.write_text(f"""<robot name="explicit_fixture">
      <link name="anchor"/>
      <link name="wrist"><collision><geometry>{tool}</geometry></collision></link>
      <link name="clamp_root"><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>
      <link name="pad_a"><collision><geometry><box size="0.01 0.01 0.02"/></geometry></collision></link>
      <link name="pad_b"><collision><geometry><box size="0.01 0.01 0.02"/></geometry></collision></link>
      <joint name="drive" type="prismatic"><parent link="anchor"/><child link="wrist"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="mount" type="fixed"><parent link="wrist"/><child link="clamp_root"/></joint>
      <joint name="jaw_a" type="prismatic"><parent link="clamp_root"/><child link="pad_a"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="jaw_b" type="prismatic"><parent link="clamp_root"/><child link="pad_b"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    </robot>""")
    names = ("wrist", "clamp_root", "pad_a", "pad_b")
    root = torch.eye(4)[None]
    root[:, 2, 3] = root_height
    trace = []
    bodies = {name: NS(get_shape_count=lambda: 1) for name in names}
    robot_native = NS(
        render_articulation=NS(get_physical_body=lambda name: bodies[name])
    )

    def fk(*, qpos, link_names, qpos_joint_names):
        assert tuple(qpos_joint_names) == ("drive", "jaw_a", "jaw_b")
        trace.append(
            {
                "rows": len(qpos),
                "names": list(link_names),
                "qpos_sha256": hashlib.sha256(qpos.numpy().tobytes()).hexdigest(),
            }
        )
        poses = torch.eye(4).repeat(len(qpos), len(link_names), 1, 1)
        poses[:, :, 0, 3] = qpos[:, :1]
        for index, name in enumerate(link_names):
            if name in ("pad_a", "pad_b"):
                poses[:, index, 1, 3] = -0.04 if name == "pad_a" else 0.04
                poses[:, index, 2, 3] = 0.1
        return poses

    robot = NS(
        uid="fixture_robot",
        num_instances=1,
        dof=3,
        joint_names=["drive", "jaw_a", "jaw_b"],
        cfg=NS(
            fpath=str(robot_path),
            body_scale=(1.0, 1.0, 1.0),
            control_parts={"manipulator": ("drive",), "clamp": ("jaw_a", "jaw_b")},
            solver_cfg={
                "manipulator": NS(
                    root_link_name="anchor", end_link_name="wrist", tcp=np.eye(4)
                )
            },
        ),
        body_data=NS(entities=[robot_native]),
        get_joint_ids=lambda part: [0] if part == "manipulator" else [1, 2],
        compute_fk=fk,
        get_local_pose=lambda **kwargs: root.clone(),
        get_parent_joint_chain=lambda name: [NS(name="drive")],
        get_link_physical_attr=lambda names, **kwargs: [
            NS(contact_offset=0.002, rest_offset=0.0) for _ in names
        ],
    )
    art_root = torch.eye(4)[None]
    art_root[:, 2, 3] = 0.8
    joint = NS(
        origin_pose=np.eye(4), target_pose=np.eye(4), axis=np.array([0.0, 0.0, 1.0])
    )
    art_native = NS(
        get_root_link_name=lambda: binding.parent,
        render_articulation=NS(get_joint_info=lambda name: joint),
    )
    art = NS(
        uid=binding.object_id,
        cfg=NS(
            fpath=str(source),
            body_scale=(1.0, 1.0, 1.0),
            body_scale_mass_policy="native",
        ),
        body_data=NS(entities=[art_native]),
        joint_names=[binding.joint],
        get_qpos=lambda: torch.zeros(1, 1),
        get_link_pose=lambda *args, **kwargs: art_root.clone(),
        get_link_physical_attr=lambda *args, **kwargs: [
            NS(contact_offset=0.001, rest_offset=0.0)
        ],
    )
    table_body = NS(get_shape_count=lambda: 1)
    table_native = NS(
        native=lambda: NS(get_physical_body=lambda: table_body),
        get_physical_attr=lambda: NS(contact_offset=0.001, rest_offset=0.0),
    )
    table = NS(
        uid=table_uid,
        num_instances=1,
        body_type="static",
        cfg=NS(shape=NS(fpath=None), source=None),
        body_data=NS(entities=[table_native]),
        get_local_pose=lambda **kwargs: torch.eye(4)[None],
        get_collision_shapes=lambda **kwargs: [
            NS(
                local_pose=np.eye(4),
                vertices=None,
                triangles=None,
                half_extents=torch.tensor([0.5, 0.5, 0.05]),
            )
        ],
    )
    simulation = NS(
        num_envs=1,
        is_default_backend=True,
        get_robot=lambda uid: robot if uid == robot.uid else None,
        get_articulation=lambda uid: art if uid == art.uid else None,
        get_rigid_object=lambda uid: table if uid == table.uid else None,
    )
    hand = RobotHandGeometryCfg(
        motion_part="manipulator",
        hand_part="clamp",
        hand_root_link="clamp_root",
        pad_link_names=("pad_a", "pad_b"),
        open_qpos=(0.04, -0.04),
        closed_qpos=(0.0, 0.0),
        sample_count=65,
        body_max_edge=0.01,
        pad_max_edge=0.004,
        max_subdivisions=1,
    )
    positions = torch.zeros(1, 130, 3)
    if mesh_kind == "box":
        positions[0, :, 0] = torch.linspace(0.0, 0.13, 130)
    return NS(
        simulation=simulation,
        robot=robot,
        art=art,
        table=table,
        binding=binding,
        geometry=geometry,
        seal=seal,
        hand=hand,
        names=names,
        positions=positions,
        frames=list(range(70)) + list(range(80, 100)) + list(range(110, 130)),
        root=root,
        parent=art_root.clone(),
        child=art_root.clone(),
        trace=trace,
        native_bodies=bodies,
        table_body=table_body,
    )
