# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import pytest

from scripts.benchmark.task_program.new3_pick import (
    SCENES,
    SELECTION,
    UPRIGHT,
    graph_for,
    physical_verdict,
    read,
    native_wall_failure,
)


def test_sixty_predeclared_distinct_rigid_objects() -> None:
    assert len(SELECTION) == 20
    assert sum(map(len, SELECTION.values())) == 60
    declared = set()
    for task, uids in SELECTION.items():
        objects = read(SCENES / f"task{task}/scene_export/scene_config.json")[
            "rigid_object"
        ]
        assert len(set(uids)) == 3
        assert set(uids) <= {o["uid"] for o in objects if o["body_type"] == "dynamic"}
        declared.update((task, uid) for uid in uids)
    assert UPRIGHT <= declared


@pytest.mark.parametrize("upright,expected", [(False, "E1"), (True, "E3")])
def test_policy_context_is_explicit(upright: bool, expected: str) -> None:
    from embodichain.gen_sim.task_engine.semantic_graph import (
        validate_semantic_task_graph,
    )

    graph = graph_for("test", "exact_uid", "right", upright)
    assert validate_semantic_task_graph(graph) == graph
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["task_type"] == expected
    assert graph["nodes"][0]["call"] == {
        "kind": "pick",
        "object": "exact_uid",
        "resources": {"primary": "right"},
    }
    assert graph["success"]["kind"] == "diagnostic_pick_only"


def valid_evidence() -> dict:
    return dict(
        completed=True,
        acquisition=True,
        resource_ok=True,
        final_lift=0.10,
        terminal_lift=0.085,
        contact_count=6,
        terminal_contacts=True,
        complete_trace=True,
    )


def test_complete_native_evidence_passes() -> None:
    assert physical_verdict(**valid_evidence())


@pytest.mark.parametrize(
    "phase,category",
    [("preparation", "planning_wall_budget"), ("execution", "execution_wall_budget")],
)
def test_native_guard_is_not_reported_as_acquisition_failure(phase, category):
    runtime = {
        "segments": [
            {
                "metadata": {
                    "runtime": {
                        "failures": [
                            {
                                "phase": phase,
                                "message": "TimeoutError: e3_broad_native_wall_guard_180_seconds",
                            }
                        ]
                    }
                }
            }
        ]
    }
    result = native_wall_failure(runtime, "", exit_code=2)
    assert result["category"] == category


@pytest.mark.parametrize(
    "log,code",
    [
        (
            "Exception ignored in: weakref callback\nTimeoutError: e3_broad_native_wall_guard_180_seconds",
            0,
        ),
        ("timeout: sending signal INT to command python", 124),
        ("timeout: sending signal KILL to command python", 137),
        ("", 124),
        ("/usr/bin/timeout: \u6b63\u5728\u53d1\u9001\u4fe1\u53f7 KILL", 137),
    ],
)
def test_wall_guard_cannot_be_swallowed_into_eventual_success(log, code):
    failure = native_wall_failure({}, log, exit_code=code)
    assert failure and failure["category"] == "native_wall_budget"
    assert not physical_verdict(**{**valid_evidence(), "resource_ok": failure is None})


def test_action_timeout_is_not_a_native_worker_wall_budget():
    assert (
        native_wall_failure({}, "Action attempt timeout exceeded.", exit_code=2) is None
    )


@pytest.mark.parametrize("old_version", [None, "stale-evaluator"])
def test_collector_reassesses_stale_cache_without_overwriting_original(
    monkeypatch, tmp_path, old_version
):
    import json
    from scripts.benchmark.task_program import new3_pick as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "runs/case"
    output.mkdir(parents=True)
    (output / "controller_receipt.json").write_text(
        json.dumps({"status": "returned", "exit_code": 0})
    )
    (output / "native.log").write_text("TimeoutError: native_wall_guard_180_seconds")
    cached = output / "pick_result.json"
    cached.write_text(
        json.dumps(
            {
                "accepted": True,
                "raw_hashes": {
                    str(output / "native.log"): module.sha(output / "native.log")
                },
                **(
                    {"evaluation_digest": old_version}
                    if old_version is not None
                    else {}
                ),
            }
        )
    )
    original = cached.read_bytes()
    row = {"id": "case", "source_uid": "payload", "upright_quality": False}
    result = module.collect(row, {})
    assert not result["accepted"]
    assert result["failure_category"] == "native_wall_budget"
    assert result["evaluation_digest"] == module.evaluation_digest()
    assert cached.read_bytes() == original
    assert len(list(output.glob("evaluations/*/pick_result.json"))) == 1
    assert module.collect(row, {}) == result


@pytest.mark.parametrize("upright", [False, True])
@pytest.mark.parametrize("resource", ["left", "right"])
def test_producer_effort_calibration_is_pick_specific(
    upright: bool, resource: str
) -> None:
    from types import SimpleNamespace
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _calibrate_task_gripper_effort,
    )

    embodiment = {
        "simulation": {
            "joint_drive_props": {"max_effort": {"left_eef": 500.0, "right_eef": 500.0}}
        },
        "skill_profile": {
            "resources": [
                {
                    "resource_id": side,
                    "endpoints": [
                        {"endpoint_id": "grasp", "control_part": side + "_eef"}
                    ],
                }
                for side in ("left", "right")
            ]
        },
    }
    scene = SimpleNamespace(
        rigid_objects=[{"uid": "target", "attrs": {"mass_props": {"mass": 0.005}}}]
    )
    _calibrate_task_gripper_effort(
        embodiment, scene, graph_for("test", "target", resource, upright)
    )
    limits = embodiment["simulation"]["joint_drive_props"]["max_effort"]
    assert limits[resource + "_eef"] == (500.0 if upright else 0.5)
    assert limits[("right" if resource == "left" else "left") + "_eef"] == 500.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("completed", False),
        ("acquisition", False),
        ("resource_ok", False),
        ("final_lift", None),
        ("final_lift", 0.079),
        ("terminal_lift", None),
        ("terminal_lift", 0.079),
        ("contact_count", 2),
        ("terminal_contacts", False),
        ("complete_trace", False),
    ],
)
def test_no_success_without_every_evidence_layer(key: str, value: object) -> None:
    evidence = valid_evidence()
    evidence[key] = value
    assert not physical_verdict(**evidence)
