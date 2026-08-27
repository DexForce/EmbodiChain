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
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine import cli
from embodichain.gen_sim.task_engine import _bundle_runner as bundle_runner
from embodichain.gen_sim.task_engine.orchestration.artifacts import (
    ArtifactTransaction,
)
from embodichain.gen_sim.task_engine.orchestration.contracts import (
    BINDING_REPORT_SCHEMA,
    ROLE_BINDINGS_SCHEMA,
    SCENE_MANIFEST_SCHEMA,
    SCENE_REQUEST_SCHEMA,
    SUCCESS_SPEC_SCHEMA,
    TASK_CANDIDATE_SET_SCHEMA,
    TASK_DRAFT_SCHEMA,
    canonical_hash,
)
from embodichain.gen_sim.task_engine.orchestration.coordinator import (
    TaskEngineCoordinator,
)
from embodichain.gen_sim.task_engine.orchestration.scene_adapter import (
    SceneAdaptation,
)
from embodichain.gen_sim.task_engine.orchestration.scene_source import SceneSourceRef
from embodichain.gen_sim.action_engine.generation.artifacts import artifact_paths
from embodichain.gen_sim.action_engine.generation.models import PreparedScene
from embodichain.gen_sim.action_engine.protocol import (
    AGENT_CONFIG_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
)
from embodichain.gen_sim.action_engine.runtime import (
    ExecutionReport,
    build_execution_provenance,
)
from embodichain.gen_sim.task_engine.scene import SceneEngineV1Adapter

_UPRIGHT_CAN_INSTRUCTION = "test-instruction"


def _candidate_set() -> dict:
    selector = {
        "kind": "scene_ref",
        "step_id": "",
        "reference": "red can",
        "quantifier": "one",
        "count": 0,
    }
    none_selector = {
        "kind": "none",
        "step_id": "",
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }
    step = {
        "id": "upright",
        "task_type": "E2",
        "object": selector,
        "target": none_selector,
        "relation": "none",
        "required_arm": "auto",
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
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "upright_can",
        "instruction": _UPRIGHT_CAN_INSTRUCTION,
        "steps": [step],
    }
    candidate = {
        "candidate_id": "candidate_01",
        "draft": draft,
        "scene_request": {
            "schema_version": SCENE_REQUEST_SCHEMA,
            "task_id": "upright_can",
            "references": [
                {
                    "reference_id": "upright.object",
                    "step_id": "upright",
                    "role": "object",
                    "reference": "red can",
                    "quantifier": "one",
                    "count": 0,
                    "source_structure": "rigid_object",
                    "affordances": ["graspable", "orientable"],
                    "initial_state": {"orientation": "fallen"},
                    "attributes": {},
                }
            ],
        },
        "success_spec": {
            "schema_version": SUCCESS_SPEC_SCHEMA,
            "task_id": "upright_can",
            "op": "all",
            "terms": [{"step_id": "upright", "type": "object_upright"}],
        },
        "semantic_hash": canonical_hash([step]),
        "vote_count": 1,
        "attempts": 1,
        "normalizations": [],
    }
    return {
        "schema_version": TASK_CANDIDATE_SET_SCHEMA,
        "task_id": "upright_can",
        "instruction": _UPRIGHT_CAN_INSTRUCTION,
        "candidates": [candidate],
        "requested_candidate_count": 1,
        "valid_response_count": 1,
        "errors": [],
    }


def _candidate_set_with_alternative() -> dict:
    candidates = _candidate_set()
    alternative = deepcopy(candidates["candidates"][0])
    alternative["candidate_id"] = "candidate_02"
    alternative["draft"]["steps"][0]["required_arm"] = "left_arm"
    alternative["semantic_hash"] = canonical_hash(alternative["draft"]["steps"])
    candidates["candidates"].append(alternative)
    candidates["requested_candidate_count"] = 2
    candidates["valid_response_count"] = 2
    return candidates


def _prepared_scene(tmp_path: Path) -> PreparedScene:
    scene_path = tmp_path / "scene_config.json"
    scene_path.write_text("{}", encoding="utf-8")
    scene_object = {
        "uid": "red_can",
        "source_uid": "red_can",
        "role": "rigid_object",
        "name": "red can",
        "description": "A red can.",
        "category": "can",
        "color": "red",
        "position": [0.0, 0.0, 0.5],
        "affordances": ["graspable", "orientable"],
        "initial_state": {"orientation": "fallen"},
        "attributes": {},
    }
    return PreparedScene(
        source_config_path=scene_path,
        scene_dir=tmp_path,
        planner_objects=(scene_object,),
        background=(),
        rigid_objects=(),
        articulations=(),
        uid_map={"red_can": "red_can"},
        table_top_z=None,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )


def _adaptation(tmp_path: Path, *, status: str = "bound") -> SceneAdaptation:
    candidates = _candidate_set()
    candidate = candidates["candidates"][0]
    selected_id = candidate["candidate_id"] if status == "bound" else ""
    role_bindings = (
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "upright_can",
            "candidate_id": "candidate_01",
            "reference_bindings": {"upright.object": ["red_can"]},
            "role_bindings": {},
        }
        if status == "bound"
        else None
    )
    return SceneAdaptation(
        scene_manifest={
            "schema_version": SCENE_MANIFEST_SCHEMA,
            "scene_id": "scene",
            "source_format": "test",
            "robot_profile": "dual_franka",
            "objects": [
                {
                    "uid": "red_can",
                    "role": "rigid_object",
                    "name": "red can",
                    "description": "A red can.",
                    "category": "can",
                    "color": "red",
                    "affordances": ["graspable", "orientable"],
                    "initial_state": {"orientation": "fallen"},
                    "attributes": {},
                }
            ],
        },
        role_bindings=role_bindings,
        binding_report={
            "schema_version": BINDING_REPORT_SCHEMA,
            "task_id": "upright_can",
            "status": status,
            "selected_candidate_id": selected_id,
            "selection_reason": "test",
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "semantic_hash": candidate["semantic_hash"],
                    "status": "resolved" if status == "bound" else status,
                    "references": [
                        {
                            "reference_id": "upright.object",
                            "status": (
                                "resolved" if status == "bound" else "ambiguous"
                            ),
                            "confidence": 1.0,
                            "candidate_uids": ["red_can"],
                            "selected_uids": (["red_can"] if status == "bound" else []),
                            "reasons": [],
                        }
                    ],
                    "reasons": [],
                }
            ],
        },
        selected_candidate=deepcopy(candidate) if status == "bound" else None,
        prepared_scene=_prepared_scene(tmp_path),
        source_config_path=tmp_path / "scene_config.json",
        conservative_scene_graph={
            "schema_version": "embodichain.conservative-scene-graph/v1",
            "scene_id": "scene",
            "nodes": [
                {
                    "uid": "red_can",
                    "parent_uid": "unknown",
                    "parent_relation": "unknown",
                    "orientation": "unknown",
                    "source": "test",
                }
            ],
            "relations": [],
        },
    )


def _adaptation_with_alternative(tmp_path: Path) -> SceneAdaptation:
    candidate_set = _candidate_set_with_alternative()
    adaptation = _adaptation(tmp_path)
    alternative = candidate_set["candidates"][1]
    alternative_audit = deepcopy(adaptation.binding_report["candidates"][0])
    alternative_audit["candidate_id"] = "candidate_02"
    alternative_audit["semantic_hash"] = alternative["semantic_hash"]
    alternative_bindings = {
        **deepcopy(adaptation.role_bindings),
        "candidate_id": "candidate_02",
    }
    return replace(
        adaptation,
        binding_report={
            **deepcopy(adaptation.binding_report),
            "candidates": [
                *deepcopy(adaptation.binding_report["candidates"]),
                alternative_audit,
            ],
        },
        candidate_bindings={"candidate_02": alternative_bindings},
    )


def test_artifact_transaction_rolls_back_and_preserves_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "kept.txt").write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fail"):
        with ArtifactTransaction(output, overwrite=True) as transaction:
            assert transaction.staging_dir is not None
            (transaction.staging_dir / "partial.txt").write_text(
                "partial", encoding="utf-8"
            )
            raise RuntimeError("fail before commit")

    assert (output / "kept.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "partial.txt").exists()


def test_prepare_rejects_output_overlapping_read_only_source(tmp_path: Path) -> None:
    source = tmp_path / "gym_project"
    source.mkdir()
    coordinator = TaskEngineCoordinator(
        task_agent=object(),
        scene_adapter=SimpleNamespace(robot_profile="franka"),
        action_agent=object(),
        feasibility_broker=object(),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        coordinator.prepare(
            "task",
            "Pick up the object.",
            source,
            source / "task_run",
            overwrite=True,
        )


def test_unbound_prepare_publishes_only_audit_artifacts(tmp_path: Path) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path, status="ambiguous")
    task_agent = SimpleNamespace(generate=lambda *args, **kwargs: candidates)
    scene_adapter = SimpleNamespace(
        robot_profile="franka",
        adapt=lambda *args, **kwargs: adaptation,
    )
    action_agent = SimpleNamespace(
        plan=lambda *_args, **_kwargs: pytest.fail("Action Agent must not run")
    )
    coordinator = TaskEngineCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        action_agent=action_agent,
        bundle_generator=lambda *_args, **_kwargs: pytest.fail(
            "legacy generator must not run"
        ),
    )

    result = coordinator.prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "bundle",
        candidate_count=1,
    )

    assert result.status == "ambiguous"
    assert (result.output_dir / "task_candidate_set.json").is_file()
    assert (result.output_dir / "binding_report.json").is_file()
    assert not (result.output_dir / "scene_manifest.json").exists()
    assert not (result.output_dir / "role_bindings.json").exists()
    assert not (result.output_dir / "grounded_task_plan.json").exists()
    assert not (result.output_dir / FAST_GYM_CONFIG_FILENAME).exists()


def test_prepare_reuses_precomputed_candidates_without_rerunning_task_agent(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path, status="ambiguous")
    task_agent = SimpleNamespace(
        generate=lambda *args, **kwargs: pytest.fail("Task Agent must not rerun")
    )
    coordinator = TaskEngineCoordinator(
        task_agent=task_agent,
        scene_adapter=SimpleNamespace(
            robot_profile="franka",
            adapt=lambda *args, **kwargs: adaptation,
        ),
        action_agent=object(),
    )

    result = coordinator.prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "candidate-reuse",
        candidate_set=candidates,
        force_most_likely=True,
    )

    assert result.status == "ambiguous"
    assert result.candidate_set == candidates


def test_prepare_inherits_adapter_robot_profile_for_raw_scene_path(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path, status="ambiguous")
    captured: dict[str, object] = {}

    def adapt(_candidates, source, **_kwargs):
        captured["source"] = source
        return adaptation

    coordinator = TaskEngineCoordinator(
        task_agent=SimpleNamespace(
            generate=lambda *args, **kwargs: pytest.fail("Task Agent must not rerun")
        ),
        scene_adapter=SimpleNamespace(robot_profile="ur10", adapt=adapt),
        action_agent=object(),
    )

    result = coordinator.prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "ur10-bundle",
        candidate_set=candidates,
    )

    assert result.status == "ambiguous"
    assert isinstance(captured["source"], SceneSourceRef)
    assert captured["source"].robot_profile == "ur10"


def test_contradicted_feasibility_publishes_audit_without_planning(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path)
    static_manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        adaptation.prepared_scene,
        source_format="test",
        robot_profile="dual_franka",
    )
    adaptation = replace(
        adaptation,
        static_scene_manifest=static_manifest,
    )
    task_agent = SimpleNamespace(generate=lambda *args, **kwargs: candidates)
    scene_adapter = SimpleNamespace(
        robot_profile="franka",
        adapt=lambda *args, **kwargs: adaptation,
    )
    registry = SimpleNamespace(
        catalog=lambda: {
            name: {
                "runtime_available": name != "PickUp",
                "unavailable_reason": (
                    "PickUp disabled for test." if name == "PickUp" else None
                ),
            }
            for name in ("PickUp", "MoveHeldObject", "Place")
        }
    )
    action_agent = SimpleNamespace(
        registry=registry,
        plan=lambda *_args, **_kwargs: pytest.fail("Action Agent must not plan"),
    )

    result = TaskEngineCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        action_agent=action_agent,
        bundle_generator=lambda *_args, **_kwargs: pytest.fail(
            "legacy generator must not run"
        ),
    ).prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "infeasible-bundle",
        candidate_count=1,
    )

    assert result.status == "infeasible"
    assert result.feasibility_report is not None
    assert result.feasibility_report["status"] == "contradicted"
    assert result.feasibility_report["remediation_class"] == "action_capability"
    assert result.artifacts.static_scene_manifest.is_file()
    assert result.artifacts.feasibility_report.is_file()
    assert not result.artifacts.grounded_task_plan.exists()


def test_feasibility_contradiction_falls_back_to_next_resolved_candidate(
    tmp_path: Path,
) -> None:
    candidate_set = _candidate_set()
    first = candidate_set["candidates"][0]
    second = deepcopy(first)
    second["candidate_id"] = "candidate_02"
    second["semantic_hash"] = "b" * 64
    candidate_set["candidates"].append(second)
    adaptation = _adaptation(tmp_path)
    second_audit = deepcopy(adaptation.binding_report["candidates"][0])
    second_audit["candidate_id"] = "candidate_02"
    second_audit["semantic_hash"] = "b" * 64
    binding_report = deepcopy(adaptation.binding_report)
    binding_report["candidates"].append(second_audit)
    second_bindings = {
        **deepcopy(adaptation.role_bindings),
        "candidate_id": "candidate_02",
    }
    adaptation = replace(
        adaptation,
        binding_report=binding_report,
        candidate_bindings={"candidate_02": second_bindings},
        static_scene_manifest={},
    )

    class _Broker:
        @staticmethod
        def assess(candidate, *_args, **_kwargs):
            return {
                "status": (
                    "runtime_probe"
                    if candidate["candidate_id"] == "candidate_02"
                    else "contradicted"
                )
            }

    registry = SimpleNamespace(catalog=lambda: {})
    coordinator = TaskEngineCoordinator(
        action_agent=SimpleNamespace(registry=registry),
        feasibility_broker=_Broker(),
    )

    updated, selected, bindings, report = coordinator._fallback_feasible_candidate(
        candidate_set,
        adaptation,
        first,
        adaptation.role_bindings,
        {"status": "contradicted"},
    )

    assert selected["candidate_id"] == "candidate_02"
    assert bindings["candidate_id"] == "candidate_02"
    assert report["status"] == "runtime_probe"
    assert updated.binding_report["selected_candidate_id"] == "candidate_02"
    assert (
        "static feasibility contradicted candidate_01"
        in updated.binding_report["selection_reason"]
    )


def test_bound_prepare_uses_sidecar_and_publishes_complete_bundle(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path)
    task_agent = SimpleNamespace(generate=lambda *args, **kwargs: candidates)
    scene_adapter = SimpleNamespace(
        robot_profile="franka",
        adapt=lambda *args, **kwargs: adaptation,
    )
    graph = {"graph": "planned"}
    action_agent = SimpleNamespace(plan=lambda _plan: deepcopy(graph))
    generator_calls = []

    def generator(_scene, output, **kwargs):
        generator_calls.append(kwargs)
        task_spec_path = Path(kwargs["task_spec"])
        assert task_spec_path.is_file()
        assert (task_spec_path.parent / "scene_requirements.json").is_file()
        paths = artifact_paths(output)
        for path in (
            paths.gym_config,
            paths.agent_config,
            paths.task_spec,
            paths.scene_requirements,
            paths.seed_task_graph,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            value = graph if path == paths.seed_task_graph else {}
            path.write_text(json.dumps(value), encoding="utf-8")
        paths.seed_task_graph_png.write_bytes(b"png")
        return paths

    result = TaskEngineCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        action_agent=action_agent,
        bundle_generator=generator,
    ).prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "bundle",
        candidate_count=1,
    )

    assert result.bound
    assert generator_calls
    assert generator_calls[0]["gripper_model"] == "pgi"
    assert generator_calls[0]["ik_solver"] == "auto"
    assert not (result.output_dir / ".task_engine_input").exists()
    grounded = json.loads(
        (result.output_dir / "grounded_task_plan.json").read_text(encoding="utf-8")
    )
    assert grounded["success_spec"]["terms"] == [
        {"step_id": "task_01", "type": "object_upright"}
    ]
    assert (result.output_dir / "seed_task_graph.json").is_file()


def test_prepare_falls_back_after_candidate_action_planning_failure(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set_with_alternative()
    adaptation = _adaptation_with_alternative(tmp_path)
    planned_candidates: list[str] = []
    graph = {"graph": "planned"}

    def plan(grounded_plan):
        candidate_id = grounded_plan["selected_candidate_id"]
        planned_candidates.append(candidate_id)
        if candidate_id == "candidate_01":
            raise ValueError(
                "SeedGraph TaskGroup 'task_04' requires unavailable state "
                "{'predicate': 'arm_free', 'arm': 'right_arm'}."
            )
        return deepcopy(graph)

    def generator(_scene, output, **_kwargs):
        paths = artifact_paths(output)
        for path in (
            paths.gym_config,
            paths.agent_config,
            paths.task_spec,
            paths.scene_requirements,
            paths.seed_task_graph,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            value = graph if path == paths.seed_task_graph else {}
            path.write_text(json.dumps(value), encoding="utf-8")
        paths.seed_task_graph_png.write_bytes(b"png")
        return paths

    result = TaskEngineCoordinator(
        task_agent=SimpleNamespace(generate=lambda *args, **kwargs: candidates),
        scene_adapter=SimpleNamespace(
            robot_profile="franka",
            adapt=lambda *args, **kwargs: adaptation,
        ),
        action_agent=SimpleNamespace(plan=plan),
        bundle_generator=generator,
    ).prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "fallback-bundle",
        candidate_count=2,
    )

    assert result.bound
    assert result.selected_candidate_id == "candidate_02"
    assert planned_candidates == ["candidate_01", "candidate_02"]
    assert (
        "candidate_01 failed action_planning"
        in result.adaptation.binding_report["selection_reason"]
    )
    assert not result.artifacts.preparation_failure.exists()


def test_prepare_publishes_failure_context_when_all_candidates_fail_planning(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set_with_alternative()
    adaptation = _adaptation_with_alternative(tmp_path)
    output = tmp_path / "failed-bundle"
    output.mkdir()
    (output / "stale.txt").write_text("old", encoding="utf-8")

    result = TaskEngineCoordinator(
        task_agent=SimpleNamespace(generate=lambda *args, **kwargs: candidates),
        scene_adapter=SimpleNamespace(
            robot_profile="franka",
            adapt=lambda *args, **kwargs: adaptation,
        ),
        action_agent=SimpleNamespace(
            plan=lambda _plan: (_ for _ in ()).throw(
                ValueError(
                    "SeedGraph TaskGroup 'task_04' requires unavailable state "
                    "{'predicate': 'arm_free', 'arm': 'right_arm'}."
                )
            )
        ),
        bundle_generator=lambda *_args, **_kwargs: pytest.fail(
            "bundle generation must not run"
        ),
    ).prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        output,
        candidate_count=2,
        overwrite=True,
    )

    assert result.status == "planning_failed"
    assert not result.bound
    assert result.artifacts.preparation_failure.is_file()
    assert not (result.output_dir / "stale.txt").exists()
    failure = json.loads(
        result.artifacts.preparation_failure.read_text(encoding="utf-8")
    )
    assert failure["schema_version"] == "action_engine_preparation_failure_v1"
    assert failure["task_id"] == "upright_can"
    assert failure["selected_candidate_id"] == "candidate_01"
    assert [attempt["candidate_id"] for attempt in failure["attempts"]] == [
        "candidate_01",
        "candidate_02",
    ]
    for index, attempt in enumerate(failure["attempts"]):
        candidate_id = f"candidate_{index + 1:02d}"
        assert attempt["stage"] == "action_planning"
        assert attempt["draft"] == candidates["candidates"][index]["draft"]
        assert attempt["bindings"]["candidate_id"] == candidate_id
        assert attempt["grounded_task_plan"]["selected_candidate_id"] == candidate_id
        assert "unbound_action_plan" in attempt
        assert "action_graph" in attempt
        assert attempt["error"]["type"] == "ValueError"
        assert "arm_free" in attempt["error"]["message"]


def test_private_bundle_runner_forwards_arguments_without_leaking_sys_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / AGENT_CONFIG_FILENAME).write_text(
        json.dumps({"task_name": "task"}), encoding="utf-8"
    )
    (bundle / FAST_GYM_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    captured = []

    def fake_cli() -> None:
        import sys

        captured.append(list(sys.argv))

    import embodichain.gen_sim.action_engine.cli as legacy_cli

    monkeypatch.setattr(
        legacy_cli,
        "run_agent",
        SimpleNamespace(cli=fake_cli),
        raising=False,
    )
    import sys

    original = sys.argv
    assert bundle_runner.main(["--bundle", str(bundle), "--seed", "7"]) == 0

    assert sys.argv is original
    assert captured[0][-2:] == ["--seed", "7"]
    assert str(bundle / AGENT_CONFIG_FILENAME) in captured[0]


@pytest.mark.parametrize(
    ("mode", "image", "scene", "edit"),
    [
        ("image", "input.png", None, None),
        ("image-edit", "input.png", None, "move the cup left"),
        ("scene", None, "gym_project", None),
        ("scene-edit", None, "gym_project", "move the cup left"),
    ],
)
def test_unified_cli_accepts_exactly_four_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    image: str | None,
    scene: str | None,
    edit: str | None,
) -> None:
    captured = {}

    class FakeWorkflow:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request, **kwargs):
            captured["request"] = request
            captured["kwargs"] = kwargs
            output = Path(request["output_dir"])
            return SimpleNamespace(
                status="succeeded",
                succeeded=True,
                failure_class=None,
                output_dir=output,
                manifest_path=output / "run_manifest.json",
                final_bundle=output / "final" / "bundle",
            )

    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "TaskEngineWorkflow", FakeWorkflow)
    arguments = [
        "--mode",
        mode,
        "--task-id",
        "task",
        "--instruction",
        "place the cup",
        "--output-root",
        str(tmp_path / "history"),
        "--base-seed",
        "9",
    ]
    if image is not None:
        arguments.extend(["--image", str(tmp_path / image)])
    if scene is not None:
        arguments.extend(["--scene", str(tmp_path / scene)])
    if edit is not None:
        arguments.extend(["--scene-edit", edit])
    if mode == "image":
        arguments.append("--dataset_saving")

    assert cli.main(arguments) == 0

    request = captured["request"]
    assert request["image_path"] == (None if image is None else str(tmp_path / image))
    assert request["gym_project"] == (None if scene is None else str(tmp_path / scene))
    assert request["scene_edit_prompt"] == edit
    assert captured["kwargs"]["base_seed"] == 9
    assert captured["kwargs"]["dataset_saving"] is (mode == "image")
    assert captured["kwargs"]["failure_policy"] == "stop"
    assert captured["kwargs"]["execute"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded"
    assert payload["run_id"].replace("_", "").isdigit()
    assert len(payload["run_id"]) == 15
    assert Path(payload["output_dir"]).parent == tmp_path / "history"


def test_unified_cli_explicit_planner_mode_overrides_packaged_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class FakeWorkflow:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request, **kwargs):
            captured.update(kwargs)
            output = Path(request["output_dir"])
            return SimpleNamespace(
                status="prepared",
                succeeded=False,
                failure_class=None,
                output_dir=output,
                manifest_path=output / "run_manifest.json",
                final_bundle=output / "final" / "bundle",
            )

    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "TaskEngineWorkflow", FakeWorkflow)

    assert (
        cli.main(
            [
                "prepare",
                "--mode",
                "image",
                "--task-id",
                "task",
                "--instruction",
                "place the cup",
                "--image",
                str(tmp_path / "input.png"),
                "--output-root",
                str(tmp_path / "history"),
                "--planner-mode",
                "ik_interp",
                "--ik-solver",
                "pytorch",
            ]
        )
        == 0
    )

    planning_cfg = captured["planning_cfg"]
    assert planning_cfg.planner == {"mode": "ik_interp"}
    assert planning_cfg.ik_solver == "pytorch"
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_unified_cli_reuses_history_root_without_modifying_prior_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = tmp_path / "task1008"
    source = (
        history
        / "20260820_105939"
        / "attempts"
        / "scene_0001"
        / "scene_revision"
        / "scene_export"
    )
    source.mkdir(parents=True)
    marker = source / "scene_config.json"
    marker.write_text('{"source": "unchanged"}\n', encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request, **_kwargs):
            captured["request"] = request
            output = Path(request["output_dir"])
            return SimpleNamespace(
                status="succeeded",
                succeeded=True,
                failure_class=None,
                output_dir=output,
                manifest_path=output / "run_manifest.json",
                final_bundle=output / "final" / "bundle",
            )

    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "TaskEngineWorkflow", FakeWorkflow)

    assert (
        cli.main(
            [
                "--mode",
                "scene",
                "--task-id",
                "task1008",
                "--scene",
                str(source),
                "--instruction",
                "place the cup on the book",
                "--output-root",
                str(history),
            ]
        )
        == 0
    )

    output_dir = Path(captured["request"]["output_dir"])
    assert output_dir.parent == history
    assert output_dir != source
    assert marker.read_text(encoding="utf-8") == '{"source": "unchanged"}\n'
    assert list(history.glob(".*.reserve")) == []


def test_unified_cli_rejects_history_root_inside_source_before_reservation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene_export"
    source.mkdir()
    output_root = source / "new_runs"

    with pytest.raises(ValueError, match="read-only source"):
        cli.main(
            [
                "--mode",
                "scene",
                "--task-id",
                "task",
                "--scene",
                str(source),
                "--instruction",
                "place the cup",
                "--output-root",
                str(output_root),
            ]
        )

    assert not output_root.exists()


def test_unified_cli_rejects_mode_input_mismatch(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "--mode",
                "image",
                "--task-id",
                "task",
                "--instruction",
                "place the cup",
                "--image",
                str(tmp_path / "input.png"),
                "--scene",
                str(tmp_path / "scene"),
                "--output-root",
                str(tmp_path / "history"),
            ]
        )


def test_public_cli_exposes_prepare_run_and_run_all_modes() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "prepare" in help_text
    assert "run-all" in help_text
    assert "run" in help_text
    assert "--overwrite" not in help_text
    assert "--run-after-prepare" not in help_text
    arguments = parser.parse_args(
        [
            "prepare",
            "--mode",
            "image",
            "--task-id",
            "task",
            "--instruction",
            "place the cup",
            "--image",
            "input.png",
            "--output-root",
            "history",
            "--dataset_saving",
        ]
    )
    assert arguments.command == "prepare"
    assert arguments.dataset_saving is True
    assert arguments.failure_policy == "stop"
    assert arguments.planner_mode is None
    assert arguments.ik_solver is None
    assert arguments.show_grasp_poses is False


def test_run_all_cli_accepts_grasp_pose_visualization() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "run-all",
            "--mode",
            "scene",
            "--task-id",
            "task",
            "--instruction",
            "move the tray",
            "--scene",
            "scene",
            "--output-root",
            "history",
            "--show-grasp-poses",
        ]
    )

    assert arguments.show_grasp_poses is True


def test_prepare_cli_stops_before_simulator_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeWorkflow:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, request, **kwargs):
            captured.update(kwargs)
            output = Path(request["output_dir"])
            return SimpleNamespace(
                status="prepared",
                succeeded=False,
                failure_class=None,
                output_dir=output,
                manifest_path=output / "run_manifest.json",
                final_bundle=output / "final" / "bundle",
            )

    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "TaskEngineWorkflow", FakeWorkflow)

    result = cli.main(
        [
            "prepare",
            "--mode",
            "image",
            "--task-id",
            "task",
            "--instruction",
            "place the cup",
            "--image",
            str(tmp_path / "input.png"),
            "--output-root",
            str(tmp_path / "history"),
        ]
    )

    assert result == 0
    assert captured["execute"] is False


def test_run_cli_executes_an_existing_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    class Executor:
        def __call__(self, _bundle, output, **kwargs):
            Path(output).mkdir()
            assert kwargs["num_envs"] == 2
            assert kwargs["failure_policy"] == "continue"
            return {
                "status": "failed",
                "environments": [
                    {"success": True},
                    {"success": False},
                ],
            }

    monkeypatch.setattr(cli, "SubprocessActionExecutor", Executor)

    result = cli.main(
        [
            "run",
            "--bundle",
            str(bundle),
            "--output-root",
            str(tmp_path / "history"),
            "--num-envs",
            "2",
            "--failure-policy",
            "continue",
        ]
    )

    assert result == 0


def test_private_bundle_runner_publishes_rejected_preflight_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / AGENT_CONFIG_FILENAME).write_text(
        json.dumps({"task_name": "task"}), encoding="utf-8"
    )
    (bundle / FAST_GYM_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    report = ExecutionReport(
        task_id="task",
        plan_hash="0" * 64,
        action_graph_hash="1" * 64,
        status="rejected",
        run_id="preflight",
        episode_id="0",
        provenance=build_execution_provenance(),
        environments=(
            {
                "env_id": "0",
                "success": False,
                "semantic_success": {},
                "action_count": 0,
                "retry_count": 0,
                "recovery_count": 0,
                "revision_count": 0,
                "failures": [],
            },
        ),
        error="ValueError: planning-only action",
    )
    monkeypatch.setattr(
        bundle_runner,
        "_preflight_bundle",
        lambda *args, **kwargs: report,
    )

    assert bundle_runner.main(["--bundle", str(bundle)]) == 2
    payload = json.loads((bundle / "execution_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["action_count"] == 0
