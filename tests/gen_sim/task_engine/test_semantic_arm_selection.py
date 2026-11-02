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

from copy import deepcopy

import pytest

from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from .test_agent import _selector, _step


@pytest.mark.parametrize(
    "support_y,target_y,required,expected",
    [
        (-0.3, -0.3, "auto", "left"),
        (0.3, 0.3, "auto", "right"),
        (-0.3, 0.3, "auto", "right"),
        (-0.3, -0.3, "right_arm", "right"),
    ],
)
@pytest.mark.parametrize("history", ["plain", "moved_target", "nested_support"])
@pytest.mark.parametrize("reuse", [False, True])
def test_pour_arm_reuse_respects_relocated_placement_endpoints(
    support_y, target_y, required, expected, history, reuse
):
    first = _step(step_id="stage", reference="cup")
    first.update(
        task_type="E1",
        orientation_goal="none",
        relation="on",
        target=_selector("scene_ref", reference="pad"),
    )
    pour = _step(step_id="pour", reference="bottle")
    pour.update(
        task_type="E3",
        orientation_goal="none",
        required_arm="right_arm",
        relation="above",
        target=_selector("step_result", step_id="stage"),
        depends_on=["stage"],
    )
    last = _step(step_id="collect", reference="cup")
    last.update(
        task_type="E1",
        orientation_goal="none",
        relation="inside",
        required_arm=required,
        object=_selector("step_result", step_id="stage"),
        target=_selector("scene_ref", reference="tray"),
        depends_on=["pour"],
    )
    steps = [first, pour, last]
    if history != "plain":
        before = _step(
            step_id="prepare", reference="tray" if history == "moved_target" else "pad"
        )
        before.update(
            task_type="E1",
            orientation_goal="none",
            relation="on",
            target=_selector("scene_ref", reference="static_support"),
        )
        first["depends_on"] = ["prepare"]
        steps.insert(0, before)
        if history == "nested_support":
            moved = deepcopy(before)
            moved.update(id="move_pad", depends_on=["stage"])
            pour["depends_on"] = ["move_pad"]
            steps.insert(2, moved)
    result = InstructionDraftResult(
        intent={"steps": steps},
        model="injected_caller",
        attempts=1,
        latency_seconds=0.01,
        normalizations=(),
    )
    candidate = TaskAgent(interpreter=lambda *a, **kw: result).generate(
        "relocated_pour", "Collect a previously placed pour receiver", candidate_count=1
    )["candidates"][0]
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "relocated_pour",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {
            f'{step["id"]}.{role}': [step[role]["reference"]]
            for step in candidate["draft"]["steps"]
            for role in ("object", "target")
            if step[role]["kind"] == "scene_ref"
        },
        "role_bindings": {},
    }
    scene = [
        {"uid": uid, "init_pos": [0.0, y, 0.7]}
        for uid, y in (
            ("cup", 0.2),
            ("pad", support_y),
            ("bottle", 0.3),
            ("tray", target_y),
            ("static_support", 0.3 if history == "moved_target" else -0.3),
        )
    ]
    planner = SemanticTaskPlanner()
    planner.reuse_pour_resource = reuse
    graph = planner.plan(candidate, bindings, scene)
    calls = [
        node["call"]
        for node in graph["nodes"]
        if node["task_instance_id"] == candidate["draft"]["steps"][-1]["id"]
    ]
    if required == "auto":
        if not reuse:
            expected = "right" if 0.2 + target_y >= 0 else "left"
        elif history != "plain":
            expected = "right"
    assert all(call["resources"] == {"primary": expected} for call in calls)
    assert scene[0]["init_pos"] == [0.0, 0.2, 0.7]
