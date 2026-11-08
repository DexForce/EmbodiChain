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

"""Independent, hand-authored acceptance checks for the pinned image cohort."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

__all__ = ["check_image_case"]


def check_image_case(case: str, artifacts: dict[str, Any]) -> list[str]:
    """Check source clauses, restoration and case-specific identity constraints.

    Args:
        case: Dataset task ID, task1171 through task1190.
        artifacts: Persisted planner artifacts, never used to create expectations.
    Returns:
        Acceptance failures; an empty list denotes semantic-only acceptance.
    """
    expected = json.loads(
        (Path(__file__).parent / "data/image_task_expectations.json").read_text()
    )["cases"][case]
    graph = artifacts["semantic_task_graph"]
    source = artifacts["semantic_graph_validation"]["source"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    instruction = graph["instruction"]
    failures = []
    for check in expected:
        selected = set()
        for fragment in check["quote_fragments"]:
            pos = instruction.find(fragment)
            if pos < 0:
                failures.append("Oracle fragment missing from source: " + fragment)
                continue
            end = pos + len(fragment)
            for requirement in source["requirements"]:
                start = instruction.find(requirement["quote"])
                stop = start + len(requirement["quote"])
                if min(end, stop) > max(pos, start):
                    selected.update(requirement["node_ids"])
        actual = {
            nodes[n]["call"].get("call_id", nodes[n]["call"]["kind"]) for n in selected
        }
        if not actual & set(check["call_alternatives"]):
            failures.append("Missing original obligation: " + check["label"])
    annotations = source["annotations"]
    if "原位" in instruction:
        restored = [
            nid
            for nid, ann in annotations.items()
            if ann["intent"].get("return_to_origin") is True
            and nodes[nid]["call"].get("call_id", nodes[nid]["call"]["kind"])
            in {"place", "simulation.place_relative"}
        ]
        if not restored:
            failures.append("Missing source restoration and release")
        if not any(
            b["kind"] == "source_pose" for b in artifacts["binding_requirements"]
        ):
            failures.append("Missing original-pose binding")
    candidate = source["candidate"]
    if any(
        step["task_type"] == "E3" and step.get("pour_goal_mode") != "motion"
        for step in candidate["draft"]["steps"]
    ):
        failures.append(
            "The pinned cohort requests pouring motions, not unobserved liquid transfer"
        )
    if case in {"task1171", "task1172", "task1179"}:
        targets = {
            annotations[n["id"]]["reference"]
            for n in nodes.values()
            if annotations[n["id"]]["relation"] == "inside"
        }
        if len(targets) != 1:
            failures.append(
                "Objects requested in the same container must retain one destination identity"
            )
    if case == "task1182":
        pouring = next(
            (
                n
                for n in nodes.values()
                if n["call"].get("call_id") == "simulation.pour"
            ),
            None,
        )
        if pouring:
            pouring_source = annotations[pouring["id"]]["object"]
            recipient = annotations[pouring["id"]]["reference"]
            contained = [
                annotations[n["id"]]["object"]
                for n in nodes.values()
                if annotations[n["id"]]["relation"] == "inside"
                and n["call"].get("call_id", n["call"]["kind"])
                in {"place", "simulation.place_relative"}
            ]
            last = next(
                (
                    n
                    for n in reversed(list(nodes.values()))
                    if n["call"].get("call_id", n["call"]["kind"])
                    in {"place", "simulation.place_relative"}
                ),
                None,
            )
            if (
                contained != [recipient]
                or last is None
                or annotations[last["id"]]["object"] != pouring_source
            ):
                failures.append(
                    "Move the pouring recipient into the tray, then reacquire the original source for the final mat placement"
                )
    if case == "task1175":
        goals = {a["intent"].get("orientation_goal") for a in annotations.values()}
        if not {"vertical", "upright"} <= goals:
            failures.append(
                "Undirected can vertical and upright bottle must remain distinct"
            )
    if case == "task1185" and not any(
        n["call"].get("arguments", {}).get("relation") == "right_of"
        for n in nodes.values()
    ):
        failures.append("Missing right_of relation")
    if case in {"task1181", "task1186", "task1188", "task1190"}:
        expected_pair = (
            ("left", "right") if case in {"task1188", "task1190"} else ("right", "left")
        )
        transfers = [n for n in nodes.values() if n["call"]["kind"] == "hand_over"]
        if any(
            (
                n["call"]["resources"].get("source"),
                n["call"]["resources"].get("destination"),
            )
            != expected_pair
            for n in transfers
        ):
            failures.append("Incorrect handover direction")
    if case == "task1181":
        destinations = {
            a["reference"]
            for a in annotations.values()
            if a["reference"] is not None and a["relation"] in {"on", "inside"}
        }
        if len(destinations) < 3:
            failures.append(
                "Serving, bottle and utensil mats must have distinct identities"
            )
    coverage = artifacts["semantic_graph_validation"]["coverage"]
    if (
        coverage["semantic_status"] != "valid"
        or coverage["runtime_evidence_status"] != "not_run"
        or coverage["execution_ready"] is not False
    ):
        failures.append("Invalid phase-specific verdict")
    return failures
