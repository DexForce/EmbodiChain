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

"""An explicit undirected vertical goal must not weaken ordinary upright goals."""

from __future__ import annotations

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.align_held import (
    alignment_target_axis,
)
from embodichain.gen_sim.task_engine._task_program.stability import (
    StabilityConstraint,
    _constraint_axis_alignment,
    _constraint_pose_flags,
)
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.task_spec import task_spec_from_candidate
from .test_task_spec_generation import candidate, selector, step


@pytest.mark.parametrize("direction_free", [False, True])
def test_alignment_and_acceptance_share_the_direction_contract(direction_free: bool):
    source = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    target = alignment_target_axis(source, direction_free=direction_free)
    assert target[:, 2].tolist() == (
        [1.0, -1.0, 1.0] if direction_free else [1.0, 1.0, 1.0]
    )
    cfg = StabilityConstraint(
        "payload",
        "upright",
        local_axis=(0.0, 0.0, 1.0),
        axis_direction_free=direction_free,
    )
    poses = torch.eye(4)[None].repeat(3, 1, 1)
    poses[1, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses[2, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    assert _constraint_pose_flags(
        cfg, poses, torch.eye(4)[None].repeat(3, 1, 1)
    ).tolist() == [True, direction_free, False]
    assert _constraint_axis_alignment(cfg, poses).tolist() == (
        [1.0, 1.0, 0.0] if direction_free else [1.0, -1.0, 0.0]
    )


@pytest.mark.parametrize("orientation", ["upright", "vertical"])
def test_vertical_intent_preserves_ordinary_pick_and_serializes_its_alignment(
    orientation,
):
    value = candidate(
        step(
            "E1",
            "payload",
            target=selector("tray"),
            relation="inside",
            orientation_goal=orientation,
        )
    )
    task_id = value["draft"]["task_id"]
    step_id = value["draft"]["steps"][0]["id"]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": task_id,
        "candidate_id": value["candidate_id"],
        "reference_bindings": {
            f"{step_id}.object": ["payload"],
            f"{step_id}.target": ["tray"],
        },
        "role_bindings": {},
    }
    scene = [
        {"uid": uid, "init_pos": [0.0, y, 0.8]}
        for uid, y in (("payload", -0.2), ("tray", 0.2))
    ]
    graph = SemanticTaskPlanner().plan(value, bindings, scene)
    calls = [node["call"] for node in graph["nodes"]]
    align = [call for call in calls if call.get("call_id") == "gen_sim.align_held"]
    assert align
    assert all(
        call["arguments"].get("axis_direction_free", False)
        is (orientation == "vertical")
        for call in align
    )
    assert any(call["kind"] == "pick" for call in calls) is (orientation == "vertical")
    spec = task_spec_from_candidate(value)
    assert any(
        goal.name == ("orientation" if orientation == "vertical" else "object_upright")
        for goal in spec.goal
    )


def test_undirected_axis_policy_is_explicit_and_typed():
    with pytest.raises(ValueError, match="boolean"):
        StabilityConstraint(
            "payload", "upright", local_axis=(0.0, 0.0, 1.0), axis_direction_free=1
        )
    with pytest.raises(ValueError, match="declared local axis"):
        StabilityConstraint("payload", "placement", axis_direction_free=True)
    with pytest.raises(ValueError, match="E1 only"):
        candidate(step("E4", "payload", orientation_goal="vertical"))


@pytest.mark.parametrize("angle,ordinary", [(0.0, True), (180.0, False), (90.0, False)])
def test_already_upright_source_keeps_standard_acquisition(angle, ordinary):
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _preserve_already_upright_pick,
    )

    value = candidate(
        step(
            "E1",
            "payload",
            target=selector("tray"),
            relation="inside",
            orientation_goal="upright",
        )
    )
    sid = value["draft"]["steps"][0]["id"]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": value["draft"]["task_id"],
        "candidate_id": value["candidate_id"],
        "reference_bindings": {f"{sid}.object": ["payload"], f"{sid}.target": ["tray"]},
        "role_bindings": {},
    }
    objects = {
        uid: {
            "uid": uid,
            "init_pos": [0.0, y, 0.8],
            "init_rot": [angle, 0.0, 0.0],
            "shape": {"shape_type": "Cube", "size": [0.04, 0.04, 0.1]},
        }
        for uid, y in (("payload", -0.2), ("tray", 0.2))
    }
    graph = SemanticTaskPlanner().plan(value, bindings, list(objects.values()))
    result = _preserve_already_upright_pick(graph, objects)
    assert (result["nodes"][0]["call"]["kind"] == "pick") is ordinary
    assert result["nodes"][1:] == graph["nodes"][1:]
    assert graph["nodes"][0]["call"]["kind"] == "registered"
