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

"""Independent cohort acceptance follows producing actions, not quote layout."""

from __future__ import annotations
from copy import deepcopy
import pytest
from .image_case_acceptance import check_image_case


def _drawer_plan():
    instruction = "打开抽屉，取出里面的苹果放入抽屉左侧的果盘，再把桌上的橙子放入同一果盘，最后关闭抽屉。"
    definitions = [
        ("open", "open_step", "registered", "drawer", None, None),
        ("take_apple", "apple_step", "pick", "apple", "drawer", "inside"),
        ("place_apple", "apple_step", "place", "apple", "bowl", "inside"),
        ("take_orange", "orange_step", "pick", "orange", None, None),
        ("place_orange", "orange_step", "place", "orange", "bowl", "inside"),
        ("close", "close_step", "registered", "drawer", None, None),
    ]
    nodes, annotations = [], {}
    for nid, step, kind, obj, reference, relation in definitions:
        call = {"kind": kind, "object": obj}
        if kind == "registered":
            call["call_id"] = "simulation.articulation_link_slide"
        elif kind == "place":
            call["inside"] = reference
        nodes.append({"id": nid, "task_instance_id": step, "call": call})
        annotations[nid] = {
            "object": obj,
            "reference": reference,
            "relation": relation,
            "intent": {},
        }
    quotes = [
        "打开抽屉",
        "取出里面的苹果",
        "放入抽屉左侧的果盘",
        "再把桌上的橙子",
        "放入同一果盘",
        "最后关闭抽屉",
    ]
    return {
        "semantic_task_graph": {"instruction": instruction, "nodes": nodes},
        "semantic_graph_validation": {
            "source": {
                "annotations": annotations,
                "requirements": [
                    {"quote": q, "node_ids": [n["id"]]} for q, n in zip(quotes, nodes)
                ],
                "candidate": {"draft": {"steps": []}},
            },
            "coverage": {
                "semantic_status": "valid",
                "runtime_evidence_status": "not_run",
                "execution_ready": False,
            },
        },
    }


def test_split_subject_quote_and_pickup_origin_do_not_fail_valid_placements():
    assert check_image_case("task1172", _drawer_plan()) == []


@pytest.mark.parametrize(
    "damage", ["missing_placement", "different_destination", "different_subject"]
)
def test_cohort_oracle_still_rejects_missing_or_incorrect_placements(damage):
    artifacts = deepcopy(_drawer_plan())
    graph = artifacts["semantic_task_graph"]
    source = artifacts["semantic_graph_validation"]["source"]
    if damage == "missing_placement":
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != "place_orange"]
        source["requirements"] = [
            r for r in source["requirements"] if "place_orange" not in r["node_ids"]
        ]
    elif damage == "different_destination":
        source["annotations"]["place_orange"]["reference"] = "different_bowl"
        next(n for n in graph["nodes"] if n["id"] == "place_orange")["call"][
            "inside"
        ] = "different_bowl"
    else:
        source["annotations"]["place_orange"]["object"] = "apple"
        next(n for n in graph["nodes"] if n["id"] == "place_orange")["call"][
            "object"
        ] = "apple"
        # The orange quote selects the pickup, not this unrelated producer.
        source["requirements"] = [
            r for r in source["requirements"] if "place_orange" not in r["node_ids"]
        ]
    failures = check_image_case("task1172", artifacts)
    assert failures
    assert any(
        "橙子入盘" in failure or "same container" in failure for failure in failures
    )
