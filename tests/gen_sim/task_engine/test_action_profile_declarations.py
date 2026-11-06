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
from pathlib import Path
from copy import deepcopy
import json
import pytest
from embodichain.gen_sim.task_engine.action_profile_declarations import (
    resolve_action_profile_declaration,
)

_SOURCE = json.loads(
    (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
)


@pytest.mark.parametrize("case", _SOURCE["cases"], ids=lambda row: row["source_case"])
def test_source_recipe_option_and_monitor_defaults_are_preserved(case):
    before = deepcopy(case)
    actual = resolve_action_profile_declaration(
        case["graph"],
        default_resources=case["expected"]["defaults"],
        registered_pick_options=case["pick_options"],
    )
    assert actual == case["expected"]
    assert case == before
    actual["action_options"]["pick"]["lift_height"] = 99
    assert case == before


def test_explicit_custom_resource_names_are_not_rewritten():
    actual = resolve_action_profile_declaration(
        {"nodes": []},
        default_resources={"pick_up": {"primary": "tool_resource"}},
        registered_pick_options={},
    )
    assert actual["defaults"] == {"pick_up": {"primary": "tool_resource"}}
    assert actual["effect_monitors"] == {}
    with pytest.raises(ValueError, match="exact declared alias"):
        resolve_action_profile_declaration(
            {"nodes": []},
            default_resources={},
            registered_pick_options={"gen_sim.pick.unused": {"kind": "pick_up"}},
        )


@pytest.mark.parametrize(
    "source_kind,release,adaptive",
    [
        ("ordinary", False, True),
        ("constrained", False, False),
        ("end", False, True),
        ("end", True, False),
    ],
)
def test_handover_profile_tracks_declared_grasp_provenance(
    source_kind, release, adaptive
):
    from embodichain.gen_sim.task_engine.alignment_declarations import (
        resolve_handover_profile_declarations,
    )

    pick = {"kind": "pick", "object": "payload", "resources": {"primary": "left"}}
    lowerers = []
    if source_kind != "ordinary":
        pick = {
            "kind": "registered",
            "call_id": "gen_sim.pick.selected",
            "arguments": {"object": "payload", "target": "goal"},
            "resources": {"primary": "left"},
        }
        lowerers = [
            {
                "kind": "pick",
                "call_id": "gen_sim.pick.selected",
                "routes": [
                    {
                        "object_id": "payload",
                        "target_id": "goal",
                        **(
                            {"grasp_region": "upper_half"}
                            if source_kind == "constrained"
                            else {}
                        ),
                    }
                ],
            }
        ]
    nodes = [{"id": "acquire", "call": pick}]
    if release:
        nodes.append(
            {
                "id": "release",
                "call": {
                    "kind": "place",
                    "object": "payload",
                    "resources": {"primary": "left"},
                },
            }
        )
    nodes.append(
        {
            "id": "transfer",
            "call": {
                "kind": "hand_over",
                "object": "payload",
                "resources": {"source": "left", "destination": "right"},
            },
        }
    )
    options = resolve_handover_profile_declarations({"nodes": nodes}, lowerers)[
        "transfer"
    ]
    assert options["adaptive_receive"] is adaptive
    assert options["require_receiver_scene"] is adaptive
    assert options["receive_region"] == ("supported_end" if adaptive else "mesh_extent")
    assert options["existing_hold_hand_source"] == "grasp_command"
