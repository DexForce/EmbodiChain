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

from embodichain.task_spec import (
    semantic_hash,
    validate_task_template,
    validate_scene_instance,
    validate_action_witness,
    validate_expansion_manifest,
    validate_validation_certificate,
)
from embodichain.task_spec._json import hash_json

_DIGEST = "a" * 64


def ref(name):
    return {"uri": name, "content_hash": _DIGEST}


def instance():
    value = {
        "schema_version": "taskspec/scene_instance/v0.1",
        "template_hash": _DIGEST,
        "roles": {"role_0": {"entity_id": "cube_01", "asset": ref("cube.glb")}},
        "scene": ref("scene.yaml"),
        "embodiment": ref("robot.yaml"),
        "initial_state": ref("observed_initial_state.json"),
        "evidence": [ref("initial_inspection.json")],
    }
    value["content_hash"] = hash_json(value)
    return value


def witness(status="candidate"):
    return {
        "schema_version": "taskspec/action_witness/v0.1",
        "template_hash": _DIGEST,
        "instance_hash": instance()["content_hash"],
        "status": status,
        "plan": {
            "graph": ref("graph.json"),
            "program": ref("program.yaml"),
            "integration": ref("integration.yaml"),
            "execution_policy": ref("execution_policy.yaml"),
            "constraints": ref("constraints.json"),
            "integration_fingerprint": _DIGEST,
        },
        "execution": None,
        "evidence": [],
    }


def check(status="pass"):
    return {
        "id": "final_goal",
        "status": status,
        "scope": "task_goal",
        "inputs": {
            "template": _DIGEST,
            "instance": instance()["content_hash"],
            "witness": _DIGEST,
        },
        "checker": {"name": "goal_checker", "version": "1"},
        "predicate": {"name": "relative_position", "version": "1"},
        "parameters": {"margin_m": 0.1},
        "metrics": {"distance_m": 0.12},
        "env_id": 0,
        "episode_id": "episode_1",
        "evidence": [ref("task_evaluation.json")],
    }


def certificate(status="pass"):
    return {
        "schema_version": "taskspec/certificate/v0.1",
        "template_hash": _DIGEST,
        "instance_hash": instance()["content_hash"],
        "witness_id": ref("witness.json"),
        "checks": [check(status)],
        "policy": {"id": "demo", "version": "1", "required_checks": ["final_goal"]},
    }


def test_instance_recomputes_content_identity_and_owns_copied_grounding():
    value = instance()
    decoded = validate_scene_instance(value)
    decoded["roles"]["role_0"]["entity_id"] = "another"
    assert value["roles"]["role_0"]["entity_id"] == "cube_01"
    for field in ("roles", "scene", "embodiment", "initial_state", "template_hash"):
        changed = instance()
        if field == "roles":
            changed[field]["role_0"]["entity_id"] = "cube_02"
        elif field == "template_hash":
            changed[field] = "b" * 64
        else:
            changed[field]["content_hash"] = "b" * 64
        with pytest.raises(ValueError, match="hash"):
            validate_scene_instance(changed)


@pytest.mark.parametrize(
    "field", ["template_hash", "scene", "roles", "initial_state", "evidence"]
)
def test_instance_requires_typed_references_and_grounding(field):
    value = instance()
    value[field] = None
    with pytest.raises(ValueError):
        validate_scene_instance(value)


def test_two_candidates_share_task_identity_without_sharing_program():
    first = witness()
    second = deepcopy(first)
    second["plan"]["program"] = {"uri": "alternate.yaml", "content_hash": "b" * 64}
    decoded = [validate_action_witness(item) for item in (first, second)]
    assert decoded[0]["template_hash"] == decoded[1]["template_hash"]
    assert decoded[0]["plan"]["program"] != decoded[1]["plan"]["program"]


def test_successful_witness_requires_actual_execution_and_task_success():
    value = witness("succeeded")
    with pytest.raises(ValueError):
        validate_action_witness(value)
    value["execution"] = {
        "env_id": 0,
        "episode_id": "episode_1",
        "program_success": True,
        "task_success": True,
        "runtime_result": ref("runtime.json"),
        "task_evaluation": ref("goal.json"),
        "trajectory": [ref("trajectory.npz")],
    }
    value["evidence"] = [ref("execution_report.json")]
    assert validate_action_witness(value)["status"] == "succeeded"
    value["execution"]["task_success"] = False
    with pytest.raises(ValueError):
        validate_action_witness(value)


@pytest.mark.parametrize("status", ["not_run", "unavailable", "failed", "unsupported"])
def test_nonpass_required_checks_are_preserved_as_nonpass(status):
    from embodichain.task_spec.contracts import certificate_passed

    value = validate_validation_certificate(certificate(status))
    assert value["checks"][0]["status"] == status
    assert not certificate_passed(value)


def test_missing_required_checks_and_evidence_cannot_be_pass():
    from embodichain.task_spec.contracts import certificate_passed

    assert certificate_passed(certificate())
    value = certificate()
    value["checks"] = []
    assert not certificate_passed(value)
    for field in (
        "checker",
        "inputs",
        "predicate",
        "evidence",
        "parameters",
        "metrics",
        "env_id",
        "episode_id",
    ):
        value = certificate()
        value["checks"][0].pop(field)
        with pytest.raises(ValueError):
            validate_validation_certificate(value)
    value = certificate()
    value["checks"][0]["evidence"] = []
    with pytest.raises(ValueError):
        validate_validation_certificate(value)


def test_duplicate_checks_and_mismatched_input_identity_are_rejected():
    value = certificate()
    value["checks"].append(deepcopy(value["checks"][0]))
    with pytest.raises(ValueError):
        validate_validation_certificate(value)
    value = certificate()
    value["checks"][0]["inputs"]["instance"] = "b" * 64
    with pytest.raises(ValueError):
        validate_validation_certificate(value)


def expansion():
    return {
        "schema_version": "taskspec/expansion/v0.1",
        "parent": ref("parent.json"),
        "changes": [{"scope": "trajectory", "child": ref("child.json")}],
        "operator": {"name": "retime", "version": "1", "parameters": {"factor": 1.2}},
        "seed": 7,
        "semantic_effect": "preserve",
        "invalidates": ["planning", "execution", "task_goal"],
    }


def test_expansion_keeps_scope_and_semantic_effect_independent():
    assert validate_expansion_manifest(expansion())["semantic_effect"] == "preserve"
    for field, bad in [
        ("seed", True),
        ("semantic_effect", "semantic_preserving"),
        ("changes", [{"scope": "invalid", "child": ref("child")}]),
    ]:
        value = expansion()
        value[field] = bad
        with pytest.raises(ValueError):
            validate_expansion_manifest(value)


def test_json_safe_parameters_reject_nested_callables():
    value = expansion()
    value["operator"]["parameters"]["callback"] = lambda: True
    with pytest.raises(ValueError):
        validate_expansion_manifest(value)


def test_certificate_checks_bind_witness_and_cannot_mix_environment_rows():
    value = certificate()
    value["checks"][0]["inputs"]["witness"] = "b" * 64
    with pytest.raises(ValueError, match="identit"):
        validate_validation_certificate(value)
    value = certificate()
    another = deepcopy(value["checks"][0])
    another.update(id="other_goal", env_id=1)
    value["checks"].append(another)
    with pytest.raises(ValueError, match="env/episode"):
        validate_validation_certificate(value)


def test_instance_and_witness_can_be_validated_against_their_owners():
    from embodichain.task_spec.contracts import scene_instance_hash

    template = {
        "schema_version": "taskspec/template/v0.1",
        "semantic_version": "0.1",
        "roles": {"cube": {"kind": "object"}},
        "init": [],
        "requirements": [],
        "invariants": [],
        "goal": [
            {
                "predicate": "upright",
                "object": "cube",
                "max_tilt": {"value": 0.1, "unit": "rad"},
            }
        ],
    }
    template["semantic_hash"] = semantic_hash(template)
    scene = instance()
    scene["template_hash"] = template["semantic_hash"]
    scene["content_hash"] = scene_instance_hash(scene)
    assert validate_scene_instance(scene, template=template) == scene
    action = witness()
    action.update(
        template_hash=scene["template_hash"], instance_hash=scene["content_hash"]
    )
    assert validate_action_witness(action, instance=scene) == action
    action["instance_hash"] = "b" * 64
    with pytest.raises(ValueError, match="instance"):
        validate_action_witness(action, instance=scene)
    scene["roles"]["role_1"] = {"entity_id": "extra", "asset": ref("extra.glb")}
    scene["content_hash"] = scene_instance_hash(scene)
    with pytest.raises(ValueError, match="roles"):
        validate_scene_instance(scene, template=template)


def test_canonical_role_binding_distinguishes_opposite_grounded_goals():
    from embodichain.task_spec import canonical_role_map, scene_instance_hash

    first = {
        "schema_version": "taskspec/template/v0.1",
        "semantic_version": "0.1",
        "roles": {"a": {"kind": "object"}, "b": {"kind": "object"}},
        "init": [],
        "invariants": [],
        "requirements": [],
        "goal": [
            {
                "predicate": "relative_position",
                "object": "a",
                "reference": "b",
                "relation": "left_of",
                "margin": {"value": 0.1, "unit": "m"},
            }
        ],
    }
    second = deepcopy(first)
    second["goal"][0].update(object="b", reference="a")
    assert semantic_hash(first) == semantic_hash(second)
    scenes = []
    physical = {"a": "physical_A", "b": "physical_B"}
    for task in (first, second):
        task["semantic_hash"] = semantic_hash(task)
        labels = canonical_role_map(task)
        scene = instance()
        scene["template_hash"] = task["semantic_hash"]
        scene["roles"] = {
            labels[author]: {"entity_id": entity, "asset": ref(entity + ".glb")}
            for author, entity in physical.items()
        }
        scene["content_hash"] = scene_instance_hash(scene)
        scenes.append(validate_scene_instance(scene, template=task))
    assert scenes[0]["content_hash"] != scenes[1]["content_hash"]
    assert scenes[0]["roles"] != scenes[1]["roles"]
