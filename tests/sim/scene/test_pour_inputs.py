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
"""Pour input adapters retain native read order, ownership and explicit gravity."""

from __future__ import annotations
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import trimesh
from embodichain.lab.sim.scene.pour_inputs import (
    capture_pour_vertices,
    capture_pour_faces,
    configured_pour_gravity,
    observed_pour_arm_origin,
)
from embodichain.lab.sim.scene.pour_region import PourRegionGeometry


def test_native_pour_inputs_copy_exact_b1_data_without_pose_or_scale_reapplication():
    vertices = torch.tensor(
        [[[-0.1, 0.0, 0.0], [0.2, 0.3, 0.4], [0.1, -0.2, 0.1]]], dtype=torch.float32
    )
    faces = torch.tensor([[[2, 0, 1]]], dtype=torch.int64)
    calls = []

    def read_vertices(**kwargs):
        calls.append(("vertices", kwargs))
        return vertices

    def read_faces(**kwargs):
        calls.append(("faces", kwargs))
        return faces

    entity = NS(num_instances=1, get_vertices=read_vertices, get_triangles=read_faces)
    points = capture_pour_vertices(entity)
    triangles = capture_pour_faces(entity)
    assert points.dtype == np.float32 and triangles.dtype == np.int64
    assert (
        points.tobytes() == vertices[0].numpy().tobytes()
        and triangles.tobytes() == faces[0].numpy().tobytes()
    )
    assert calls == [
        ("vertices", {"env_ids": [0], "scale": True}),
        ("faces", {"env_ids": [0]}),
    ]
    vertices.fill_(99)
    faces.fill_(0)
    assert points[0, 0] == pytest.approx(-0.1) and triangles.tolist() == [[2, 0, 1]]
    assert not points.flags.writeable
    entity.num_instances = 2
    with pytest.raises(ValueError, match="one native instance"):
        capture_pour_vertices(entity)
    assert len(calls) == 2


def test_pour_gravity_and_root_are_read_from_exact_declared_owners():
    simulation = NS(
        physics_backend="default",
        num_envs=1,
        sim_config=NS(physics_cfg=NS(gravity=(0.0, 0.0, -9.81))),
    )
    assert configured_pour_gravity(simulation) == (0.0, 0.0, -9.81)
    simulation.physics_backend = "newton"
    with pytest.raises(ValueError, match="Default/B1"):
        configured_pour_gravity(simulation)
    simulation.physics_backend = "default"
    simulation.sim_config.physics_cfg.gravity = None
    with pytest.raises(ValueError, match="nonzero"):
        configured_pour_gravity(simulation)
    pose = torch.eye(4)[None]
    pose[0, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    calls = []
    robot = NS(
        get_solver=lambda part: NS(root_link_name="configured_root"),
        get_link_pose=lambda **kwargs: (calls.append(kwargs) or pose),
    )
    ids = torch.tensor([0])
    value = observed_pour_arm_origin(robot, "custom_motion", ids)
    np.testing.assert_allclose(value, [0.1, 0.2, 0.3])
    assert calls[0]["link_name"] == "configured_root" and calls[0]["env_ids"] is ids
    pose.fill_(99)
    np.testing.assert_allclose(value, [0.1, 0.2, 0.3])


def test_receiver_snapshot_retains_immutable_geometry_and_fresh_pose_queries():
    mesh = trimesh.creation.box((0.4, 0.3, 0.05))
    source = trimesh.creation.box((0.06, 0.08, 0.15))
    vertices = {
        "source": np.asarray(source.vertices).copy(),
        "receiver": np.asarray(mesh.vertices).copy(),
    }
    faces = {"receiver": np.asarray(mesh.faces).copy()}
    snapshot = PourRegionGeometry(vertices, faces, (0.0, 0.0, -9.81), 0.03)
    pose = np.eye(4)
    pose[2, 3] = 0.4
    initial = snapshot.staging_pose(
        "source", "receiver", pose, pose, np.array([-0.5, 0.0, 0.0]), clearance_m=0.05
    )
    vertices["source"][:] = 99
    faces["receiver"][:] = 0
    repeated = snapshot.staging_pose(
        "source", "receiver", pose, pose, np.array([-0.5, 0.0, 0.0]), clearance_m=0.05
    )
    np.testing.assert_array_equal(initial, repeated)
    with pytest.raises(ValueError):
        snapshot.vertices["source"].setflags(write=True)
    changed = pose.copy()
    changed[0, 3] = 0.2
    translated = snapshot.staging_pose(
        "source",
        "receiver",
        pose,
        changed,
        np.array([-0.5, 0.0, 0.0]),
        clearance_m=0.05,
    )
    assert translated[0, 3] != initial[0, 3]
