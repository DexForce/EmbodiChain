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

from embodichain.gen_sim.task_engine import (
    BoundTaskDraft,
    TaskInterpretationError,
    bind_task_draft,
    decode_task_draft,
    interpret_task_candidates,
)
from embodichain.lab.gym.envs.expert_program import PickCfg, PlaceCfg
from embodichain.lab.sim.skills import (
    SceneArticulationRef,
    SceneEntityManifest,
    SceneManifest,
    SceneObjectRef,
    SemanticValidationError,
)


def _projection() -> dict[str, object]:
    return {
        "schema_version": "semantic_integration_planner_projection/v1",
        "integration_fingerprint": "b" * 64,
        "scene_registry_id": "generated-scene",
        "robot_profile_id": "test-robot",
        "scene": {"entities": []},
        "semantic_calls": [
            {"call_id": "pick", "schema_version": 1},
            {"call_id": "place", "schema_version": 1},
        ],
        "robot": {"resources": []},
        "providers": {},
    }


def _draft() -> dict[str, object]:
    return {
        "schema_version": "task_draft/v1",
        "candidate_id": "candidate_01",
        "integration_fingerprint": "b" * 64,
        "task_spec": {
            "schema_version": "task_spec/v1",
            "task_id": "place_cube",
            "level": "L1",
            "instruction": "Place the cube on the tray",
            "reasoning_type": "none",
            "task_instances": [
                {
                    "id": "e1_0",
                    "task_type": "E1",
                    "params": {
                        "object_role": "cube_role",
                        "target_role": "tray_role",
                        "relation": "on",
                    },
                    "depends_on": [],
                    "role": "primary",
                }
            ],
            "success": {
                "kind": "object_supported_by",
                "object": "cube_role",
                "support": "tray_role",
            },
            "oracle": {},
            "metadata": {},
        },
        "scene_requirements": {
            "schema_version": "scene_requirements/v1",
            "task_id": "place_cube",
            "roles": [
                {
                    "role_id": "cube_role",
                    "reference": "the_cube",
                    "expected_type": "object",
                },
                {
                    "role_id": "tray_role",
                    "reference": "the_tray",
                    "expected_type": "object",
                },
            ],
        },
        "semantic_call_candidates": [
            {
                "task_instance_id": "e1_0",
                "calls": [
                    {"kind": "pick", "object": "cube_role"},
                    {"kind": "place", "object": "cube_role", "on": "tray_role"},
                ],
                "confidence": 0.9,
            }
        ],
        "model_provenance": {"model_id": "structured-test", "attempt": 1},
    }


def _scene_manifest() -> SceneManifest:
    return SceneManifest(
        (
            SceneEntityManifest(
                ref=SceneObjectRef("cube"),
                aliases=("the_cube",),
                semantic_type="cube",
            ),
            SceneEntityManifest(
                ref=SceneObjectRef("tray"),
                aliases=("the_tray",),
                semantic_type="tray",
            ),
        )
    )


def test_task_draft_uses_catalog_projection_and_canonical_call_codec() -> None:
    draft = decode_task_draft(_draft(), planner_projection=_projection())

    calls = draft.semantic_call_candidates[0].calls
    assert type(calls[0]) is PickCfg
    assert type(calls[1]) is PlaceCfg
    assert draft.integration_fingerprint == _projection()["integration_fingerprint"]
    assert decode_task_draft(draft.to_dict(), planner_projection=_projection()) == draft


def test_scene_roles_bind_once_to_typed_canonical_manifest_refs() -> None:
    draft = decode_task_draft(_draft(), planner_projection=_projection())
    bound = bind_task_draft(draft, _scene_manifest())

    assert type(bound) is BoundTaskDraft
    assert bound.role_bindings == {
        "cube_role": SceneObjectRef("cube"),
        "tray_role": SceneObjectRef("tray"),
    }
    calls = bound.semantic_call_candidates[0].calls
    assert calls[0].object == "cube"
    assert calls[1].object == "cube"
    assert calls[1].on == "tray"


def test_scene_binding_rejects_unknown_and_mistyped_references_explicitly() -> None:
    unknown = _draft()
    unknown["scene_requirements"]["roles"][0]["reference"] = "missing"  # type: ignore[index]
    with pytest.raises(SemanticValidationError, match="Unknown scene entity"):
        bind_task_draft(
            decode_task_draft(unknown, planner_projection=_projection()),
            _scene_manifest(),
        )

    mistyped = _draft()
    scene = SceneManifest(
        (
            SceneEntityManifest(
                ref=SceneArticulationRef("cube"), aliases=("the_cube",)
            ),
            SceneEntityManifest(ref=SceneObjectRef("tray"), aliases=("the_tray",)),
        )
    )
    with pytest.raises(SemanticValidationError, match="SceneArticulationRef"):
        bind_task_draft(
            decode_task_draft(mistyped, planner_projection=_projection()),
            scene,
        )


def test_model_and_deterministic_drafts_share_the_same_local_decoder() -> None:
    source = _draft()

    def caller(**kwargs: object) -> object:
        candidate = deepcopy(source)
        index = int(kwargs["candidate_index"])
        candidate["candidate_id"] = f"candidate_{index + 1:02d}"
        if index == 1:
            candidate["model_provenance"] = {"nested": {"qpos": [0.0]}}
        return candidate

    result = interpret_task_candidates(
        "Place the cube on the tray",
        caller=caller,
        planner_projection=_projection(),
        candidate_count=3,
    )

    assert len(result.candidates) == 1
    assert len(result.errors) == 1
    assert "forbidden" in result.errors[0]


def test_interpretation_fails_when_every_candidate_is_invalid() -> None:
    with pytest.raises(TaskInterpretationError, match="All task interpretation"):
        interpret_task_candidates(
            "Place the cube on the tray",
            caller=lambda **_: {"invalid": True},
            planner_projection=_projection(),
            candidate_count=2,
        )


def test_task_draft_rejects_catalog_mismatch_and_robot_routing_fields() -> None:
    mismatch = _draft()
    mismatch["integration_fingerprint"] = "c" * 64
    with pytest.raises(ValueError, match="does not match catalog"):
        decode_task_draft(mismatch, planner_projection=_projection())

    routed = _draft()
    routed["model_provenance"] = {"required_arm": "left_arm"}
    with pytest.raises(ValueError, match="forbidden"):
        decode_task_draft(routed, planner_projection=_projection())
