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


def test_explicit_e2_held_target_keeps_source_world_pose_priority_over_reference():
    from embodichain.gen_sim.task_engine.pour_declarations import (
        resolve_pour_lowerer_declarations,
    )

    pose = dict(position=[0.1, 0.2, 0.8], quaternion_xyzw=[0, 0, 0, 1])
    graph = dict(
        nodes=[
            dict(
                task_type="E2",
                call=dict(
                    kind="registered",
                    call_id="simulation.move_held_object",
                    arguments=dict(
                        object="payload", target="upright", reference="unread"
                    ),
                ),
            )
        ],
        targets=dict(upright=dict(kind="cyclic_pose", values=[pose])),
    )
    assert resolve_pour_lowerer_declarations(graph, {}, {}, ()) == [
        dict(
            kind="move_held_object",
            effect_contract="terminal_attachment",
            routes=[
                dict(
                    object_id="payload",
                    target_id="upright",
                    pose={"kind": "pose", **pose},
                )
            ],
        )
    ]


def test_pour_return_selection_preserves_frozen_source_calls():
    import json
    from pathlib import Path
    from embodichain.gen_sim.task_engine.pour_declarations import (
        _pour_return_call_declarations,
    )

    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    # Frozen B0 invocation_policy.pour_return_call_declarations selects the
    # same E3 return for static, revision-1 and revision-2 geometry; all other
    # source recipes in this input inventory have no Pour-return override.
    for case in cases:
        expected = (
            ["step_01__call_05"]
            if case["source_case"] == "e3-dual_franka-originalenv-v2"
            else []
        )
        assert _pour_return_call_declarations(case["graph"]) == expected


def test_pour_return_selection_keeps_each_group_subject_and_occurrence():
    from copy import deepcopy
    from embodichain.gen_sim.task_engine.pour_declarations import (
        _pour_return_call_declarations,
    )

    def pour(identifier, group, subject="payload", task_type="E3"):
        return dict(
            id=identifier,
            task_type=task_type,
            task_instance_id=group,
            call=dict(
                kind="registered",
                call_id="simulation.pour",
                arguments=dict(object=subject),
            ),
        )

    def place(identifier, group, subject="payload", *, absolute=True):
        return dict(
            id=identifier,
            task_type="E3",
            task_instance_id=group,
            call=dict(kind="place", object=subject, at="target" if absolute else None),
        )

    graph = dict(
        nodes=[
            place("before_pour", "first"),
            pour("pour_first", "first"),
            place("other_subject", "first", "other"),
            place("other_group_before_pour", "second"),
            place("relative_place", "first", absolute=False),
            place("return_first", "first"),
            place("same_call_repeated_without_pour", "first"),
            pour("pour_again", "first"),
            pour("pour_second", "second"),
            place("return_first_again", "first"),
            place("return_second", "second"),
            pour("not_source_pour_recipe", "third", task_type="E1"),
            place("no_matching_pour", "third"),
        ]
    )
    before = deepcopy(graph)
    assert _pour_return_call_declarations(graph) == [
        "return_first",
        "return_first_again",
        "return_second",
    ]
    assert graph == before
