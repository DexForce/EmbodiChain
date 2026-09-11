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


def test_failed_task_spec_attempt_freezes_unavailable_evidence_and_rejects_reuse(
    tmp_path,
):
    from types import SimpleNamespace
    import json
    from embodichain.gen_sim.task_engine._task_spec import (
        binding_for_graph,
        record_failure,
        require_fresh_evidence_output,
    )

    candidate, bindings, objects, template, _ = planning_inputs()
    binding = binding_for_graph(
        template, SemanticTaskPlanner().plan(candidate, bindings, objects)
    )
    failure = {"type": "RuntimeError", "message": "planner failed after safe-stop"}
    env = SimpleNamespace(sim=SimpleNamespace(get_rigid_object=lambda uid: None))
    require_fresh_evidence_output(tmp_path)
    record_failure(env, binding, None, tmp_path, failure, num_envs=2)
    evidence = json.loads((tmp_path / "task_evaluation.json").read_text())
    assert evidence["accepted"] == [False, False]
    assert evidence["final"]["status"] == ["unavailable", "unavailable"]
    assert evidence["execution_failure"] == failure
    with pytest.raises(ValueError, match="fresh evidence"):
        require_fresh_evidence_output(tmp_path)


def test_task_spec_observes_live_final_state_and_records_failure_before_acceptance(
    tmp_path,
):
    from types import SimpleNamespace
    import json
    import torch
    from embodichain.gen_sim.task_engine._task_spec import (
        binding_for_graph,
        observe,
        final_acceptance,
    )

    candidate, bindings, objects, template, instance = planning_inputs()
    graph = SemanticTaskPlanner().plan(candidate, bindings, objects)
    binding = binding_for_graph(template, graph)
    poses = torch.eye(4).repeat(2, 1, 1)
    fallen = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    poses[:, :3, :3] = fallen
    obj = SimpleNamespace(get_local_pose=lambda **kwargs: poses)
    env = SimpleNamespace(
        num_envs=2, sim=SimpleNamespace(get_rigid_object=lambda uid: obj)
    )
    initial = observe(env, binding, scope="initial", output=tmp_path)
    assert initial["goal_satisfied"] == [False, False]
    # Both program rows succeeded, but the second object's final goal did not.
    poses[0] = torch.eye(4)
    result = SimpleNamespace(
        success=(True, True), to_metadata=lambda: {"success": [True, True]}
    )
    accepted = final_acceptance(env, binding, initial, result, tmp_path)
    assert accepted == (True, False)
    report = json.loads((tmp_path / "task_evaluation.json").read_text())
    assert report["task_success"] == [True, False]
    assert report["program_success"] == [True, True]
    assert report["certificate_status"] == "unavailable"
    assert report["initial"]["goal_satisfied"] == [False, False]


def test_task_spec_binding_rejects_drift_and_orphaned_sidecars(tmp_path):
    from embodichain.gen_sim.task_engine._task_spec import (
        binding_for_graph,
        read_binding,
        write_evidence,
    )

    candidate, bindings, objects, template, _ = planning_inputs()
    graph = SemanticTaskPlanner().plan(candidate, bindings, objects)
    binding = binding_for_graph(template, graph)
    path = tmp_path / "task_spec_binding.json"
    ref = write_evidence(path, binding)
    fingerprint = {
        "schema_version": "semantic_integration_fingerprint/v3",
        "task_spec_binding": ref,
    }
    assert read_binding(tmp_path, graph, fingerprint) == binding
    with pytest.raises(ValueError, match="schema v3"):
        read_binding(
            tmp_path, graph, {"schema_version": "semantic_integration_fingerprint/v2"}
        )
    changed = deepcopy(binding)
    changed["entity_id"] = "another_can"
    write_evidence(path, changed)
    with pytest.raises(ValueError, match="drifted"):
        read_binding(tmp_path, graph, fingerprint)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_binding(tmp_path, graph, fingerprint)


import pytest

from embodichain.task_spec import semantic_hash, scene_instance_hash
from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.semantic_graph import validate_semantic_task_graph
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)


def planning_inputs(arm="auto"):
    selector = {
        "kind": "none",
        "step_id": "",
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }
    step = {
        "id": "orient",
        "task_type": "E2",
        "object": dict(selector, kind="scene_ref", reference="can"),
        "target": selector,
        "relation": "none",
        "required_arm": arm,
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "upright",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }
    candidate = TaskAgent(
        interpreter=lambda *_args, **_kwargs: InstructionDraftResult(
            intent={"steps": [step]},
            model="injected",
            attempts=1,
            latency_seconds=0,
            normalizations=(),
        )
    ).generate("orient", "Make the can upright", candidate_count=1)["candidates"][0]
    binding = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "orient",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {"step_01.object": ["can"]},
        "role_bindings": {},
    }
    objects = [{"runtime_uid": "can", "init_pos": [0, -0.2, 0.7]}]
    template = {
        "schema_version": "taskspec/template/v0.1",
        "semantic_version": "0.1",
        "roles": {"item": {"kind": "object"}},
        "init": [],
        "invariants": [],
        "requirements": [],
        "goal": [
            {
                "predicate": "upright",
                "object": "item",
                "max_tilt": {"value": 0.1, "unit": "rad"},
            }
        ],
    }
    template["semantic_hash"] = semantic_hash(template)

    def ref(uri):
        return {"uri": uri, "content_hash": "a" * 64}

    instance = {
        "schema_version": "taskspec/scene_instance/v0.1",
        "template_hash": template["semantic_hash"],
        "roles": {"role_0": {"entity_id": "can", "asset": ref("can.glb")}},
        "scene": ref("scene.yaml"),
        "embodiment": ref("robot.yaml"),
        "initial_state": ref("observed_state.json"),
        "evidence": [ref("initial_check.json")],
    }
    instance["content_hash"] = scene_instance_hash(instance)
    return candidate, binding, objects, template, instance


def test_planner_keeps_legacy_graph_and_adds_explicit_task_spec_provenance():
    candidate, binding, objects, template, instance = planning_inputs()
    planner = SemanticTaskPlanner()
    legacy = planner.plan(candidate, binding, objects)
    graph = planner.plan(
        candidate, binding, objects, task_template=template, scene_instance=instance
    )
    assert legacy["schema_version"] == "semantic_task_graph/v1"
    assert "task_spec" not in legacy
    assert graph["schema_version"] == "semantic_task_graph/v2"
    assert graph["task_spec"] == {
        "template_hash": template["semantic_hash"],
        "instance_hash": instance["content_hash"],
        "legacy_plan_hash": candidate["semantic_hash"],
        "status": "candidate",
    }
    assert graph["nodes"] == legacy["nodes"]
    assert validate_semantic_task_graph(graph) == graph
    changed = deepcopy(graph)
    changed["task_spec"]["status"] = "succeeded"
    with pytest.raises(ValueError):
        validate_semantic_task_graph(changed)


def test_two_recipe_choices_share_task_identity_but_keep_legacy_plan_hash():
    graphs = []
    for arm in ("left_arm", "right_arm"):
        candidate, binding, objects, template, instance = planning_inputs(arm)
        graphs.append(
            SemanticTaskPlanner().plan(
                candidate,
                binding,
                objects,
                task_template=template,
                scene_instance=instance,
            )
        )
    assert (
        graphs[0]["task_spec"]["template_hash"]
        == graphs[1]["task_spec"]["template_hash"]
    )
    assert (
        graphs[0]["task_spec"]["legacy_plan_hash"]
        != graphs[1]["task_spec"]["legacy_plan_hash"]
    )
    assert graphs[0]["nodes"] != graphs[1]["nodes"]


def test_task_spec_requires_paired_inputs_and_matching_physical_binding():
    candidate, binding, objects, template, instance = planning_inputs()
    with pytest.raises(ValueError):
        SemanticTaskPlanner().plan(candidate, binding, objects, task_template=template)
    instance["roles"]["role_0"]["entity_id"] = "other_can"
    instance["content_hash"] = scene_instance_hash(instance)
    with pytest.raises(ValueError, match="binding"):
        SemanticTaskPlanner().plan(
            candidate, binding, objects, task_template=template, scene_instance=instance
        )


def test_task_spec_bundle_export_is_gated_before_creating_artifacts(tmp_path):
    candidate, binding, objects, template, instance = planning_inputs()
    graph = SemanticTaskPlanner().plan(
        candidate,
        binding,
        objects,
        task_template=template,
        scene_instance=instance,
    )
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="evaluation"):
        generate_task_program_bundle(
            graph, None, destination, robot_profile="dual_franka"
        )
    assert not destination.exists()


def test_imported_task_spec_graph_cannot_bypass_runtime_evaluation_gate(tmp_path):
    import json
    from embodichain.gen_sim.task_engine._bundle_runner import execute_bundle

    candidate, binding, objects, template, instance = planning_inputs()
    graph = SemanticTaskPlanner().plan(
        candidate,
        binding,
        objects,
        task_template=template,
        scene_instance=instance,
    )
    bundle = tmp_path / "imported_bundle"
    (bundle / "task_program").mkdir(parents=True)
    (bundle / "semantic_task_graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )
    # V2 must be rejected before even attempting to decode deployment files.
    for name in (
        "task_program_deployment.yaml",
        "task_program/program.yaml",
        "integration_fingerprint.json",
    ):
        (bundle / name).write_text("{}", encoding="utf-8")
    output = tmp_path / "execution"
    with pytest.raises(ValueError, match="evaluation"):
        execute_bundle(bundle, execution_output=output)
    assert not output.exists()
