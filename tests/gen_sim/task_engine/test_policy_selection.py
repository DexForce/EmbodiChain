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

"""Typed rigid-inside policy selection keeps other callers and recipes unchanged."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
    bind_contact_revisions,
)
from embodichain.gen_sim.task_engine._task_program.policy_selection import (
    select_manipulation_policy,
)
from embodichain.lab.task_program.semantics import (
    Pick,
    RegisteredSemanticCall,
    SceneObjectRef,
)


def _inputs():
    pick = {
        "id": "acquire",
        "task_type": "E1",
        "task_instance_id": "transfer",
        "depends_on": [],
        "call": {"kind": "pick", "object": "item", "resources": {"primary": "left"}},
    }
    place = {
        "id": "release",
        "task_type": "E1",
        "task_instance_id": "transfer",
        "depends_on": ["acquire"],
        "call": {
            "kind": "place",
            "object": "item",
            "inside": "inside__tray__item",
            "resources": {"primary": "left"},
        },
    }
    graph = {"nodes": [pick, place]}
    candidate = {
        "draft": {
            "steps": [
                {
                    "id": "transfer",
                    "task_type": "E1",
                    "object": {"kind": "scene_object"},
                }
            ]
        }
    }
    scene = SimpleNamespace(
        planner_objects=[
            {
                "runtime_uid": "item",
                "role": "rigid_object",
                "attributes": {
                    "final_world_aabb": {"min": [0, 0, 0], "max": [0.1, 0.1, 0.2]}
                },
            },
            {"runtime_uid": "tray", "role": "rigid_object"},
        ]
    )
    return graph, candidate, scene


@pytest.mark.parametrize(
    "profile, selected",
    [
        ("default", False),
        ("composition_experiment", False),
        ("contact_experiment", True),
        ("auto", True),
    ],
)
def test_pad_quality_permission_is_profile_and_recipe_scoped(profile, selected) -> None:
    graph, candidate, scene = _inputs()
    graph["nodes"].append(
        {
            "id": "button",
            "task_type": "E9",
            "task_instance_id": "control",
            "depends_on": [],
            "call": {"kind": "registered", "arguments": {"object": "panel"}},
        }
    )
    before = deepcopy(graph)
    result = select_manipulation_policy(
        graph, candidate, scene, runtime_profile=profile
    )
    assert result.get("pick_contact_revisions", {}) == (
        {"acquire": 9} if selected else {}
    )
    assert graph == before
    assert result["world_yaw_free_places"] == ["release"]


@pytest.mark.parametrize(
    "exclusion",
    [
        "missing_bounds",
        "horizontal",
        "authored",
        "orientation",
        "confined",
        "previously_moved",
    ],
)
def test_pad_quality_does_not_override_existing_scopes(exclusion) -> None:
    graph, candidate, scene = _inputs()
    subject = scene.planner_objects[0]
    if exclusion == "missing_bounds":
        subject.pop("attributes")
    elif exclusion == "horizontal":
        subject["attributes"]["final_world_aabb"]["max"] = [0.2, 0.1, 0.1]
    elif exclusion == "authored":
        graph["nodes"][0]["call"]["grasp"] = "explicit"
    elif exclusion == "orientation":
        candidate["draft"]["steps"][0]["orientation_goal"] = "upright"
    elif exclusion == "confined":
        scene.planner_objects.append(
            {
                "role": "articulation",
                "attributes": {
                    "final_world_aabb": {"min": [-1, -1, -1], "max": [1, 1, 1]}
                },
            }
        )
    else:
        graph["nodes"][0]["depends_on"] = ["earlier"]
        graph["nodes"].insert(
            0,
            {
                "id": "earlier",
                "task_type": "E2",
                "task_instance_id": "prior",
                "depends_on": [],
                "call": {"kind": "registered", "arguments": {"object": "item"}},
            },
        )
    result = select_manipulation_policy(
        graph, candidate, scene, runtime_profile="contact_experiment"
    )
    assert result.get("pick_contact_revisions", {}) == {}
    if exclusion == "confined":
        assert result["measured_rigid_calls"] == ["acquire", "release"]


def test_native_nine_binds_ordinary_pick_but_not_handover_alias() -> None:
    segment = SimpleNamespace(
        name="acquire",
        segment_id="s",
        calls=[
            SimpleNamespace(
                call=Pick(object=SceneObjectRef("item")), segment_call_index=0
            )
        ],
    )
    program = SimpleNamespace(program_id="p", iter_segments=lambda: [segment])
    assert bind_contact_revisions({"acquire": 9}, program) == (("p/s:0", 9),)
    segment.calls[0].call = RegisteredSemanticCall(
        call_id="gen_sim.pick.handover_source", arguments={"object": "item"}
    )
    with pytest.raises(ValueError, match="native8 handover source"):
        bind_contact_revisions({"acquire": 9}, program)
    assert bind_contact_revisions({"acquire": 8}, program) == (("p/s:0", 8),)
