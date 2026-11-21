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
import json
from pathlib import Path
import pytest
import yaml

from embodichain.gen_sim.task_engine.program_declarations import (
    ProgramSegmentBinding,
    build_program_declaration,
)


def _frozen():
    fixture = json.loads(
        (Path(__file__).parent / "data/e1_bundle_declarations.json").read_text()
    )["files"]
    return json.loads(fixture["semantic_task_graph.json"]), yaml.safe_load(
        fixture["task_program/program.yaml"]
    )


def test_explicit_acceptance_copy_preserves_frozen_program_and_inputs():
    graph, expected = _frozen()
    before = deepcopy((graph, expected))
    place, park = expected["program"]["items"][1:]
    binding = ProgramSegmentBinding(
        place["name"],
        post=tuple(place["post"]),
        validators=tuple(place["validators"]),
        terminal_post=tuple(park["post"]),
        terminal_validators=tuple(park["validators"]),
    )
    actual = build_program_declaration(
        graph, expected["program_id"], bindings=(binding,)
    )
    assert actual == expected
    assert (graph, expected) == before
    actual["program"]["items"][1]["post"][0]["preset"] = "caller-copy"
    assert binding.post[0]["preset"] == "rigid_object"


def test_explicit_deferral_moves_every_check_without_changing_calls():
    graph, expected = _frozen()
    place = expected["program"]["items"][1]
    binding = ProgramSegmentBinding(
        place["name"],
        post=tuple(place["post"]),
        validators=tuple(place["validators"]),
        terminal_post=tuple(place["post"]),
        terminal_validators=tuple(place["validators"]),
        defer_to_terminal=True,
    )
    actual = build_program_declaration(
        graph, expected["program_id"], bindings=(binding,)
    )
    assert "post" not in actual["program"]["items"][1]
    assert actual["program"]["items"][2]["post"] == place["post"]
    assert actual["program"]["items"][2]["validators"] == place["validators"]
    assert [x["steps"] for x in actual["program"]["items"]] == [
        x["steps"] for x in expected["program"]["items"]
    ]
    with pytest.raises(ValueError, match="retain every"):
        build_program_declaration(
            graph,
            expected["program_id"],
            bindings=(
                ProgramSegmentBinding(
                    place["name"], post=tuple(place["post"]), defer_to_terminal=True
                ),
            ),
        )


def test_bindings_cannot_change_occurrences_or_semantic_calls():
    graph, program = _frozen()
    node = graph["nodes"][0]
    binding = ProgramSegmentBinding(node["id"])
    for bindings in ((binding, binding), (ProgramSegmentBinding("absent"),)):
        with pytest.raises(ValueError, match="unique existing"):
            build_program_declaration(graph, program["program_id"], bindings=bindings)
    with pytest.raises(ValueError, match="exact.*projection"):
        build_program_declaration(
            graph,
            program["program_id"],
            bindings=(ProgramSegmentBinding(node["id"], call={}),),
        )


def test_explicit_empty_checks_survive_and_terminal_checks_are_not_dropped():
    graph, program = _frozen()
    last = graph["nodes"][-1]["id"]
    result = build_program_declaration(
        graph,
        program["program_id"],
        bindings=(ProgramSegmentBinding(last, validators=()),),
    )
    assert result["program"]["items"][-1]["validators"] == []
    with pytest.raises(ValueError, match="declared directly"):
        build_program_declaration(
            graph,
            program["program_id"],
            bindings=(
                ProgramSegmentBinding(
                    last,
                    terminal_post=(
                        {
                            "kind": "wait_stable",
                            "entity": "cylinder",
                            "preset": "rigid_object",
                        },
                    ),
                ),
            ),
        )
