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
"""Static source mounts and scene affordances preserve source declarations."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
import trimesh
from embodichain.gen_sim.task_engine.scene_binding_declarations import (
    resolve_scene_binding_declaration,
)


def _mesh(tmp_path, name, size, role="rigid_object"):
    mesh = trimesh.creation.box(extents=size)
    mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    path = tmp_path / f"{name}.glb"
    mesh.export(path)
    return dict(
        uid=name,
        runtime_uid=name,
        role=role,
        shape={"shape_type": "Mesh", "fpath": str(path)},
        init_pos=[0.0, 0.0, 0.5],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0, 1.0, 1.0],
    )


def test_scene_affordances_keep_custom_resource_and_ordered_inside_reservations(
    tmp_path,
):
    table = _mesh(tmp_path, "table", [2.0, 2.0, 0.1], "background")
    tray = _mesh(tmp_path, "tray", [0.6, 0.6, 0.02])
    first = _mesh(tmp_path, "first", [0.08, 0.08, 0.1])
    second = _mesh(tmp_path, "second", [0.08, 0.08, 0.1])
    scene = SimpleNamespace(
        planner_objects=(table, tray, first, second), articulations=()
    )
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.5, 0.5]
    embodiment = {
        "simulation": {
            "init_pos": [0.0, 0.0, 0.0],
            "init_rot": [0.0, 0.0, 0.0],
            "urdf_cfg": {
                "components": [
                    {"component_type": "custom_motion", "transform": transform.tolist()}
                ]
            },
        },
        "skill_profile": {
            "resources": [
                {
                    "resource_id": "custom_resource",
                    "endpoints": [
                        {"endpoint_id": "motion", "control_part": "custom_motion"}
                    ],
                }
            ]
        },
    }
    graph = {
        "targets": {},
        "nodes": [
            {
                "id": name,
                "call": {
                    "kind": "place",
                    "object": name,
                    "inside": f"inside__tray__{name}",
                    "resources": {"primary": "custom_resource"},
                },
            }
            for name in ("first", "second")
        ],
    }
    before = deepcopy((graph, embodiment, scene.planner_objects))
    result = resolve_scene_binding_declaration(
        graph, scene, embodiment, program_id="program"
    )
    assert [item["entity_id"] for item in result["rigid_objects"]] == [
        "first",
        "second",
        "table",
        "tray",
    ]
    target = next(
        item for item in result["rigid_objects"] if item["entity_id"] == "tray"
    )
    inside = target["affordances"][1:]
    assert [item["entity_id"] for item in inside] == [
        "inside__tray__first",
        "inside__tray__second",
    ]
    assert all(item["release_clearance"] == 0.12 for item in inside)
    p, q = (np.asarray(item["object_target_pose"]).reshape(4, 4) for item in inside)
    assert p[1, 3] > 0 and np.linalg.norm(p[:2, 3] - q[:2, 3]) > 0.1
    assert p[2, 3] == pytest.approx(0.061, abs=1e-7)
    assert (graph, embodiment, scene.planner_objects) == before
    graph["nodes"][0]["call"]["resources"]["primary"] = "missing"
    with pytest.raises(ValueError, match="resource.*not declared"):
        resolve_scene_binding_declaration(
            graph, scene, embodiment, program_id="program"
        )


@pytest.fixture(autouse=True)
def forbid_runtime_construction(monkeypatch):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args, **kwargs):
        pytest.fail("placement declarations constructed a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(BaseSolver, "__init__", forbidden)
