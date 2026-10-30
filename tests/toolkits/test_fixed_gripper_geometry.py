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

"""Fixed mesh geometry and bounded collision reductions, without simulation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from embodichain.toolkits.graspkit.pg_grasp import (
    FixedGripperCollisionGeometry,
    GripperCollisionCfg,
    GripperCollisionChecker,
)


def test_urdf_collision_geometry_respects_joint_mesh_and_tcp_frames(
    tmp_path: Path,
) -> None:
    mesh = trimesh.creation.box(extents=(0.01, 0.02, 0.03))
    mesh.export(tmp_path / "finger.stl")
    urdf = tmp_path / "hand.urdf"
    urdf.write_text("""<robot name="fixture">
      <link name="hand"/><link name="tcp"/>
      <link name="finger"><collision><origin xyz="0.01 0 0"/>
        <geometry><mesh filename="finger.stl" scale="2 1 1"/></geometry>
      </collision></link>
      <joint name="tcp_joint" type="fixed"><parent link="hand"/><child link="tcp"/>
        <origin xyz="0 0 0.1"/>
      </joint>
      <joint name="finger_joint" type="prismatic"><parent link="hand"/><child link="finger"/>
        <origin xyz="0 0 0.06" rpy="0 0 1.5707963267948966"/>
        <axis xyz="0 1 0"/><limit lower="0" upper="0.04"/>
      </joint></robot>""")
    grasp_from_tcp = np.array(
        [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]
    )
    arguments = dict(
        root_link="hand",
        tcp_link="tcp",
        collision_links=("finger",),
        joint_positions={"finger_joint": 0.04},
        grasp_from_tcp=grasp_from_tcp,
        point_spacing=0.012,
    )
    geometry = FixedGripperCollisionGeometry.from_urdf(urdf, **arguments)
    # Rz(-90) cancels the joint's Rz(+90); the physical joint opens along
    # canonical +Y. Mesh scale/offset and the TCP subtraction remain explicit.
    expected = np.array([[-0.0, 0.03, -0.055], [0.02, 0.05, -0.025]])
    np.testing.assert_allclose(geometry.points.amin(0), expected[0], atol=2e-9)
    np.testing.assert_allclose(geometry.points.amax(0), expected[1], atol=2e-9)
    assert len(geometry.provenance["files_sha256"]) == 2
    arguments["joint_positions"] = {"finger_joint": 0.041}
    with pytest.raises(ValueError, match="Invalid fixed opening"):
        FixedGripperCollisionGeometry.from_urdf(urdf, **arguments)


@pytest.mark.parametrize("kind", ["prismatic", "revolute", "continuous"])
def test_urdf_missing_joint_axis_defaults_to_positive_x(
    tmp_path: Path, kind: str
) -> None:
    trimesh.creation.box(extents=(0.01, 0.02, 0.03)).export(tmp_path / "finger.stl")
    path = tmp_path / "default_axis.urdf"
    path.write_text(f"""<robot name="fixture"><link name="base"/>
      <link name="finger"><collision><origin xyz="0 0.02 0"/>
        <geometry><mesh filename="finger.stl"/></geometry></collision></link>
      <joint name="drive" type="{kind}"><parent link="base"/><child link="finger"/>
        <limit lower="-2" upper="2" effort="10" velocity="1"/>
      </joint></robot>""")
    geometry = FixedGripperCollisionGeometry.from_urdf(
        path,
        root_link="base",
        tcp_link="base",
        collision_links=("finger",),
        joint_positions={"drive": 0.04 if kind == "prismatic" else np.pi / 2},
        grasp_from_tcp=np.eye(4),
        point_spacing=0.012,
    )
    bounds = np.array(geometry.provenance["parts"][0]["bounds_m"])
    center = [0.04, 0.02, 0] if kind == "prismatic" else [0, 0, 0.02]
    extents = [0.01, 0.02, 0.03] if kind == "prismatic" else [0.01, 0.03, 0.02]
    np.testing.assert_allclose(bounds.mean(0), center, atol=2e-9)
    np.testing.assert_allclose(bounds[1] - bounds[0], extents, atol=2e-9)


def test_fixed_collision_chunks_match_full_reduction_and_bound_storage() -> None:
    coordinates = torch.linspace(-0.3, 0.3, 1025)
    points = torch.stack((coordinates, coordinates.flip(0), coordinates * 0.5), dim=1)
    geometry = FixedGripperCollisionGeometry(
        points, {"scope": "synthetic_test", "grasp_from_tcp": np.eye(4).tolist()}
    )
    checker = object.__new__(GripperCollisionChecker)
    checker.device = torch.device("cpu")
    # Public sampling accepts float64 object geometry but emits float32 poses.
    checker.obj_mesh_verts = torch.zeros((1, 3), dtype=torch.float64)
    checker.cfg = GripperCollisionCfg()
    checker.surface_points = geometry.points
    shapes = []

    def query(batch_points, collision_threshold):
        shapes.append(batch_points.shape)
        distance = batch_points.abs().amax(-1) - 0.1
        return distance <= collision_threshold, distance

    checker._checker = SimpleNamespace(query_batch_points=query)
    poses = torch.eye(4).repeat(258, 1, 1)[::2]  # noncontiguous, final partial chunk
    poses[:, 0, 3] = torch.linspace(0, 0.5, len(poses))
    flags, distance = checker.query(
        torch.eye(4), poses, is_filter_ground_collision=False
    )
    expected = (points[None] + poses[:, None, :3, 3]).abs().amax(-1) - 0.1
    torch.testing.assert_close(distance, expected.amin(1), rtol=0, atol=0)
    torch.testing.assert_close(flags, (expected <= 0).any(1))
    assert flags.any() and (~flags).any()
    assert max(shape[0] for shape in shapes) == 128
    assert max(shape[1] for shape in shapes) == 512
    assert len(shapes) == 6  # two candidate blocks, three point blocks
    # Far from the object but below its support plane: ground affects both
    # collision state and distance, rather than returning object state alone.
    ground_flags, ground_distance = checker.query(
        torch.eye(4), poses[-1:], is_filter_ground_collision=True
    )
    assert not flags[-1] and ground_flags[0]
    torch.testing.assert_close(ground_distance, points[:, 2].amin().reshape(1))


def test_revolute_mimic_open_state_and_virtual_tcp(tmp_path: Path) -> None:
    trimesh.creation.box(extents=(0.01, 0.01, 0.01)).export(tmp_path / "finger.stl")
    collision = '<collision><origin xyz="0.02 0 0"/><geometry><mesh filename="finger.stl"/></geometry></collision>'
    path = tmp_path / "rotary.urdf"
    path.write_text(f"""<robot name="rotary"><link name="base"/>
      <link name="left">{collision}</link><link name="right">{collision}</link>
      <joint name="drive" type="revolute"><parent link="base"/><child link="left"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>
      <joint name="follower" type="revolute"><parent link="base"/><child link="right"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper="2"/>
        <mimic joint="drive" multiplier="-1" offset="0"/></joint></robot>""")
    base_from_tcp = np.eye(4)
    base_from_tcp[2, 3] = 0.1
    kwargs = dict(
        root_link="base",
        tcp_link="base",
        collision_links=("left", "right"),
        joint_positions={"drive": np.pi / 2},
        grasp_from_tcp=base_from_tcp,
        link_from_tcp=base_from_tcp,
        point_spacing=0.012,
    )
    geometry = FixedGripperCollisionGeometry.from_urdf(path, **kwargs)
    assert geometry.provenance["joint_positions"] == {
        "drive": np.pi / 2,
        "follower": -np.pi / 2,
    }
    for part, expected_y in zip(geometry.provenance["parts"], [0.02, -0.02]):
        bounds = np.array(part["bounds_m"])
        np.testing.assert_allclose(bounds.mean(0), [0, expected_y, 0], atol=1e-9)
        np.testing.assert_allclose(bounds[1] - bounds[0], [0.01, 0.01, 0.01], atol=1e-9)
    world_grasp = torch.eye(4)
    world_grasp[:3, :3] = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    world_grasp[:3, 3] = torch.tensor([0.2, 0.3, 0.4])
    world_tcp = geometry.canonical_to_tcp_poses(world_grasp)
    torch.testing.assert_close(world_tcp[:3, 3], torch.tensor([0.2, 0.2, 0.4]))
    kwargs["joint_positions"] = {"drive": np.pi / 2, "follower": np.pi / 2}
    with pytest.raises(ValueError, match="contradicts mimic"):
        FixedGripperCollisionGeometry.from_urdf(path, **kwargs)
