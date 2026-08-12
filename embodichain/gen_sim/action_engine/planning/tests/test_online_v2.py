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
from threading import Barrier

import pytest
import torch

import embodichain.gen_sim.action_engine.planning.online as online_module
import embodichain.gen_sim.action_engine.planning.vision as vision_module
from embodichain.gen_sim.action_engine.domain import public_task_spec
from embodichain.gen_sim.action_engine.planning import (
    CameraObservation,
    SceneObservation,
    analyze_visual_scene,
    fuse_seed_graphs,
    plan_candidates_parallel,
    plan_online_seed_graph,
    select_seed_graph,
    validate_visual_facts,
)
from embodichain.gen_sim.action_engine.tasks import TaskFactory, instantiate_seed_graph


def _task(level: str, *, reasoning: str | None = None):
    factory = TaskFactory(41, executable_only=True)
    for index in range(100):
        task, requirements = factory.generate(level, index)
        if reasoning is None or task["reasoning_type"] == reasoning:
            bindings = {
                item["role_id"]: f"uid_{item['role_id']}"
                for item in requirements["objects"]
            }
            return task, requirements, bindings
    raise AssertionError(f"No deterministic {reasoning!r} task found.")


def test_online_planner_sees_public_task_and_returns_complete_seed_graph() -> None:
    task, _, bindings = _task("L4", reasoning="visual_semantics")
    offline = instantiate_seed_graph(task, bindings)
    body = {key: deepcopy(offline[key]) for key in ("nodes", "task_groups", "success")}
    for node in body["nodes"]:
        node.pop("contract")
    for group in body["task_groups"]:
        group.pop("contract")
    visual_move = next(
        node for node in body["nodes"] if node["atomic_action"] == "MoveHeldObject"
    )
    visual_move["target_binding"] = {
        "kind": "visual_constraint",
        "camera_uid": "front",
        "normalized_keypoint": [0.2, 0.3],
    }
    camera = CameraObservation(
        "front",
        torch.zeros((8, 8, 3), dtype=torch.uint8),
        None,
        None,
        None,
    )
    observation = SceneObservation(
        (camera,),
        tuple({"uid": uid} for uid in bindings.values()),
    )
    uid = next(iter(bindings.values()))
    facts = {
        "entities": [
            {
                "uid": uid,
                "camera_uid": "front",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "keypoints": {"center": [0.2, 0.3]},
                "confidence": 0.9,
            }
        ],
        "relations": [],
        "confidence": 0.9,
    }
    prompts = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return body

    graph, observed_facts = plan_online_seed_graph(
        public_task_spec(task),
        observation,
        visual_facts=facts,
        graph_caller=caller,
    )

    assert graph["planner_route"] == "online"
    assert observed_facts == facts
    assert "oracle" not in prompts[0]
    assert '"task_instances"' not in prompts[0]
    assert '"E4"' in prompts[0]
    assert "Transfer one held object" in prompts[0]
    assert graph["metadata"]["oracle_exposed"] is False
    assert any(
        node["target_binding"]["kind"] == "visual_constraint" for node in graph["nodes"]
    )


def test_offline_and_online_candidates_plan_concurrently_with_isolated_views() -> None:
    task, _, bindings = _task("L1")
    offline = instantiate_seed_graph(task, bindings)
    online = deepcopy(offline)
    online["planner_route"] = "online"
    barrier = Barrier(2)
    views = {}

    def offline_planner(*, task_spec):
        views["offline"] = task_spec
        barrier.wait(timeout=2.0)
        return offline

    def online_planner(*, task_spec):
        views["online"] = task_spec
        barrier.wait(timeout=2.0)
        return online

    pair = plan_candidates_parallel(
        task,
        offline_planner=offline_planner,
        online_planner=online_planner,
    )

    assert "oracle" in views["offline"]
    assert "oracle" not in views["online"]
    assert pair.offline["planner_route"] == "offline"
    assert pair.online["planner_route"] == "online"


def test_visual_facts_reject_unknown_uid_and_out_of_range_keypoint() -> None:
    value = {
        "entities": [
            {
                "uid": "unknown",
                "camera_uid": "front",
                "bbox": [0.0, 0.0, 1.2, 1.0],
                "keypoints": {},
                "confidence": 1.0,
            }
        ],
        "relations": [],
        "confidence": 1.0,
    }
    with pytest.raises(ValueError, match="unknown UID"):
        validate_visual_facts(value, known_uids={"known"}, camera_uids={"front"})


def test_visual_facts_reject_visible_entity_without_image_evidence() -> None:
    value = {
        "entities": [
            {
                "uid": "known",
                "camera_uid": "front",
                "visible": True,
                "confidence": 0.9,
            }
        ],
        "relations": [],
        "confidence": 0.9,
    }

    with pytest.raises(ValueError, match="bbox or keypoint"):
        validate_visual_facts(value, known_uids={"known"}, camera_uids={"front"})


def test_visual_facts_reject_non_numeric_image_coordinates() -> None:
    value = {
        "entities": [
            {
                "uid": "known",
                "camera_uid": "front",
                "bbox": ["0.1", 0.2, 0.3, 0.4],
                "confidence": 0.9,
            }
        ],
        "relations": [],
        "confidence": 0.9,
    }

    with pytest.raises(ValueError, match="must be numeric"):
        validate_visual_facts(value, known_uids={"known"}, camera_uids={"front"})


def test_visual_fact_caller_receives_rgb_depth_and_calibration_evidence() -> None:
    task, _, bindings = _task("L1")
    uid = next(iter(bindings.values()))
    observation = SceneObservation(
        (
            CameraObservation(
                "front",
                torch.zeros((4, 5, 3), dtype=torch.uint8),
                torch.linspace(0.0, 1.0, 20, dtype=torch.float32).reshape(4, 5),
                torch.eye(3),
                torch.eye(4),
            ),
        ),
        ({"uid": uid},),
    )
    captured = {}

    def caller(**kwargs):
        captured.update(kwargs)
        return {
            "entities": [
                {
                    "uid": uid,
                    "camera_uid": "front",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "confidence": 0.9,
                }
            ],
            "relations": [],
            "confidence": 0.9,
        }

    facts = analyze_visual_scene(observation, task, caller=caller)

    assert facts["entities"][0]["uid"] == uid
    assert len(captured["images"]) == 2
    assert '"depth_image_index": 1' in captured["prompt"]
    assert '"intrinsics": [[1.0, 0.0, 0.0]' in captured["prompt"]


def test_production_online_graph_caller_receives_reset_time_multiview_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = SceneObservation(
        (
            CameraObservation(
                "front",
                torch.zeros((4, 5, 3), dtype=torch.uint8),
                torch.zeros((4, 5), dtype=torch.float32),
                torch.eye(3),
                torch.eye(4),
            ),
        ),
        ({"uid": "known"},),
    )
    captured = {}

    def caller(**kwargs):
        captured.update(kwargs)
        return {"nodes": [], "task_groups": [], "success": {}}

    monkeypatch.setattr(vision_module, "_default_structured_caller", caller)
    monkeypatch.setattr(vision_module, "_vlm_model", lambda model: f"resolved:{model}")

    result = online_module._default_graph_caller(
        prompt="plan",
        schema={"type": "object"},
        model="mimo",
        observation=observation,
    )

    assert result == {"nodes": [], "task_groups": [], "success": {}}
    assert captured["model"] == "resolved:mimo"
    assert len(captured["images"]) == 2


def test_visual_facts_reject_unstructured_entity_fields() -> None:
    value = {
        "entities": [
            {
                "uid": "known",
                "camera_uid": "front",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "semantic_label": "can",
                "confidence": 0.9,
            }
        ],
        "relations": [],
        "confidence": 0.9,
    }

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_visual_facts(value, known_uids={"known"}, camera_uids={"front"})


def test_selection_prefers_exact_offline_and_l4_online() -> None:
    task, _, bindings = _task("L1")
    offline = instantiate_seed_graph(task, bindings)
    online = deepcopy(offline)
    online["planner_route"] = "online"
    selected, evaluations = select_seed_graph(
        offline,
        online,
        task,
        known_objects=set(bindings.values()) | {"table"},
        exact_template_match=True,
    )
    assert selected["metadata"]["selected_from"] == "offline"
    assert evaluations["offline"].score > evaluations["online"].score

    l4, _, l4_bindings = _task("L4", reasoning="logic")
    l4_offline = instantiate_seed_graph(l4, l4_bindings)
    l4_online = deepcopy(l4_offline)
    l4_online["planner_route"] = "online"
    selected, _ = select_seed_graph(
        l4_offline,
        l4_online,
        l4,
        known_objects=set(l4_bindings.values()) | {"table"},
        visual_confidence=0.95,
    )
    assert selected["metadata"]["selected_from"] == "online"


def test_fusion_keeps_whole_task_groups() -> None:
    task, _, bindings = _task("L2")
    offline = instantiate_seed_graph(task, bindings)
    online = deepcopy(offline)
    online["planner_route"] = "online"
    routes = {
        group["id"]: ("offline" if index % 2 == 0 else "online")
        for index, group in enumerate(offline["task_groups"])
    }
    fused = fuse_seed_graphs(offline, online, routes)

    assert fused["planner_route"] == "fused"
    assert all(
        all(node_id.startswith(routes[group["id"]]) for node_id in group["node_ids"])
        for group in fused["task_groups"]
    )
