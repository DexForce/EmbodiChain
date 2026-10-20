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

"""Native rigid-scene rest clearance is not zero mesh penetration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program.pick_native_scene import (
    NativeRigidRestScene,
    build_native_rigid_scene,
)


def _geometry():
    table = trimesh.collision.CollisionManager()
    table.add_object(
        "table",
        trimesh.creation.box([1.0, 1.0, 0.1]),
        transform=trimesh.transformations.translation_matrix([0.0, 0.0, -0.05]),
    )
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 2, 3] = [0.00425, 0.00575, 0.002]
    part = SimpleNamespace(
        name="pad",
        meshes=(trimesh.creation.box([0.02, 0.06, 0.0075]),),
        tcp_poses=poses,
        rest_offset=0.001,
    )
    return part, table


def test_positive_mesh_gap_inside_actual_rest_envelope_is_rejected():
    part, table = _geometry()
    result = NativeRigidRestScene((part,), [("table", 0.0, table)]).inspect(
        np.eye(4), 1
    )
    assert not result["valid"]
    assert 0 < result["mesh_distance_m"] < result["native_pair_rest_offset_m"]
    assert result["link"] == "pad" and result["scene_uid"] == "table"
    assert result["minimum_rest_clearance_m"] < 0
    assert result["sample_index"] == 0


def test_nominal_precontact_count_does_not_check_postcontact_virtual_closure():
    part, table = _geometry()
    part.tcp_poses[:, 2, 3] += 0.002
    screen = NativeRigidRestScene((part,), [("table", 0.0, table)])
    assert screen.inspect(np.eye(4), 2)["valid"]
    assert not screen.inspect(np.eye(4), 3)["valid"]
    assert NativeRigidRestScene((part,), []).inspect(np.eye(4), 3)["valid"]


@pytest.mark.parametrize(
    "damage", ["pose", "count_zero", "count_long", "target_rest", "hand_rest"]
)
def test_native_scene_geometry_fails_closed(damage):
    part, table = _geometry()
    pose, count, rest = np.eye(4), 1, 0.0
    if damage == "pose":
        pose[0, 0] = np.nan
    elif damage == "count_zero":
        count = 0
    elif damage == "count_long":
        count = 4
    elif damage == "target_rest":
        rest = float("nan")
    else:
        part.rest_offset = -0.001
    with pytest.raises(ValueError):
        NativeRigidRestScene((part,), [("table", rest, table)]).inspect(pose, count)


def test_scene_snapshot_excludes_only_the_exact_bound_source(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import pick_contact

    part, _ = _geometry()
    body = SimpleNamespace(get_rest_offset=lambda: 0.001)
    source = SimpleNamespace(
        get_name=lambda: "payload",
        arena_name="env0",
        native=lambda: pytest.fail("source already has its own native qualifier"),
    )
    target = SimpleNamespace(
        get_name=lambda: "table",
        arena_name="env0",
        native=lambda: SimpleNamespace(get_physical_body=lambda: body),
    )
    foreign = SimpleNamespace(
        get_name=lambda: "table",
        arena_name="env1",
        native=lambda: pytest.fail("foreign arena"),
    )
    handles = [source, target, foreign]
    simulation = SimpleNamespace(
        spawn_result=SimpleNamespace(
            get_paths=lambda: list(range(3)), get=lambda index: handles[index]
        ),
        get_rigid_object_uid_list=lambda: ["payload", "table"],
        get_rigid_object=lambda uid: SimpleNamespace(
            get_local_pose=lambda **kwargs: torch.eye(4)[None]
        ),
    )
    copied = Mock(return_value=(trimesh.creation.box([1.0, 1.0, 0.1]),))
    monkeypatch.setattr(pick_contact, "_native_meshes", copied)
    screen = SimpleNamespace(parts=(part,), identity={"simulation_uid": "payload"})
    result = build_native_rigid_scene(simulation, screen)
    copied.assert_called_once_with(body)
    assert len(result.targets) == 1 and result.targets[0][0] == "table"
    assert simulation.get_rigid_object_uid_list() == ["payload", "table"]
