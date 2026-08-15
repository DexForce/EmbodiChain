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

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.collaboration import cli
from embodichain.gen_sim.collaboration.artifacts import (
    ArtifactTransaction,
)
from embodichain.gen_sim.collaboration.contracts import (
    BINDING_REPORT_SCHEMA,
    ROLE_BINDINGS_SCHEMA,
    SCENE_MANIFEST_SCHEMA,
    SCENE_REQUEST_SCHEMA,
    SUCCESS_SPEC_SCHEMA,
    TASK_CANDIDATE_SET_SCHEMA,
    TASK_DRAFT_SCHEMA,
    canonical_hash,
)
from embodichain.gen_sim.collaboration.coordinator import (
    CollaborationCoordinator,
)
from embodichain.gen_sim.collaboration.scene_adapter import (
    SceneAdaptation,
)
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
from embodichain.gen_sim.scene_bridge import SceneEngineV1Adapter


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
        "instruction": "扶正红色易拉罐。",
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
        "instruction": "扶正红色易拉罐。",
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


def test_unbound_prepare_publishes_only_audit_artifacts(tmp_path: Path) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path, status="ambiguous")
    task_agent = SimpleNamespace(generate=lambda *args, **kwargs: candidates)
    scene_adapter = SimpleNamespace(adapt=lambda *args, **kwargs: adaptation)
    action_agent = SimpleNamespace(
        plan=lambda *_args, **_kwargs: pytest.fail("Action Agent must not run")
    )
    coordinator = CollaborationCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        action_agent=action_agent,
        bundle_generator=lambda *_args, **_kwargs: pytest.fail(
            "legacy generator must not run"
        ),
    )

    result = coordinator.prepare(
        "upright_can",
        "扶正红色易拉罐。",
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
    scene_adapter = SimpleNamespace(adapt=lambda *args, **kwargs: adaptation)
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

    result = CollaborationCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        action_agent=action_agent,
        bundle_generator=lambda *_args, **_kwargs: pytest.fail(
            "legacy generator must not run"
        ),
    ).prepare(
        "upright_can",
        "扶正红色易拉罐。",
        tmp_path / "scene_config.json",
        tmp_path / "infeasible-bundle",
        candidate_count=1,
    )

    assert result.status == "infeasible"
    assert result.feasibility_report is not None
    assert result.feasibility_report["status"] == "contradicted"
    assert result.collaboration_artifacts.static_scene_manifest.is_file()
    assert result.collaboration_artifacts.feasibility_report.is_file()
    assert not result.collaboration_artifacts.grounded_task_plan.exists()


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
    coordinator = CollaborationCoordinator(
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
    scene_adapter = SimpleNamespace(adapt=lambda *args, **kwargs: adaptation)
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

    result = CollaborationCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        action_agent=action_agent,
        bundle_generator=generator,
    ).prepare(
        "upright_can",
        "扶正红色易拉罐。",
        tmp_path / "scene_config.json",
        tmp_path / "bundle",
        candidate_count=1,
    )

    assert result.bound
    assert generator_calls
    assert not (result.output_dir / ".collaboration_input").exists()
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

    result = CollaborationCoordinator(
        task_agent=SimpleNamespace(generate=lambda *args, **kwargs: candidates),
        scene_adapter=SimpleNamespace(adapt=lambda *args, **kwargs: adaptation),
        action_agent=SimpleNamespace(plan=plan),
        bundle_generator=generator,
    ).prepare(
        "upright_can",
        "扶正红色易拉罐。",
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
    assert not result.collaboration_artifacts.preparation_failure.exists()


def test_prepare_publishes_failure_context_when_all_candidates_fail_planning(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set_with_alternative()
    adaptation = _adaptation_with_alternative(tmp_path)
    output = tmp_path / "failed-bundle"
    output.mkdir()
    (output / "stale.txt").write_text("old", encoding="utf-8")

    result = CollaborationCoordinator(
        task_agent=SimpleNamespace(generate=lambda *args, **kwargs: candidates),
        scene_adapter=SimpleNamespace(adapt=lambda *args, **kwargs: adaptation),
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
        "扶正红色易拉罐。",
        tmp_path / "scene_config.json",
        output,
        candidate_count=2,
        overwrite=True,
    )

    assert result.status == "planning_failed"
    assert not result.bound
    assert result.collaboration_artifacts.preparation_failure.is_file()
    assert not (result.output_dir / "stale.txt").exists()
    failure = json.loads(
        result.collaboration_artifacts.preparation_failure.read_text(encoding="utf-8")
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
        assert attempt["error"]["type"] == "ValueError"
        assert "arm_free" in attempt["error"]["message"]


def test_run_bundle_forwards_arguments_without_leaking_sys_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert cli.main(["run", "--bundle", str(bundle), "--seed", "7"]) == 0

    assert sys.argv is original
    assert captured[0][-2:] == ["--seed", "7"]
    assert str(bundle / AGENT_CONFIG_FILENAME) in captured[0]


def test_prepare_prints_the_next_run_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle with spaces"
    result = SimpleNamespace(
        status="bound",
        bound=True,
        selected_candidate_id="candidate_01",
        output_dir=output_dir,
        collaboration_artifacts=SimpleNamespace(
            grounded_task_plan=output_dir / "grounded_task_plan.json",
            preparation_failure=output_dir / "preparation_failure.json",
        ),
    )

    class FakeCoordinator:
        def __init__(self, **_kwargs) -> None:
            pass

        def prepare(self, *_args, **_kwargs):
            return result

    monkeypatch.setattr(cli, "ScenePackageStore", lambda *_args: object())
    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "CollaborationCoordinator", FakeCoordinator)

    assert (
        cli.main(
            [
                "prepare",
                "--task-id",
                "task",
                "--instruction",
                "place the carrot",
                "--scene",
                str(tmp_path / "scene"),
                "--output",
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_command"] == cli._bundle_run_command(output_dir)
    command = shlex.split(payload["run_command"])
    assert command[:3] == [
        "python",
        "-m",
        "embodichain.gen_sim.collaboration",
    ]
    assert command[-4:] == [
        "--bundle",
        str(output_dir.resolve()),
        "--filter_dataset_saving",
        "--headless",
    ]


def test_prepare_can_run_the_bound_bundle_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bundle"
    result = SimpleNamespace(
        status="bound",
        bound=True,
        selected_candidate_id="candidate_01",
        output_dir=output_dir,
        collaboration_artifacts=SimpleNamespace(
            grounded_task_plan=output_dir / "grounded_task_plan.json",
            preparation_failure=output_dir / "preparation_failure.json",
        ),
    )

    class FakeCoordinator:
        def __init__(self, **_kwargs) -> None:
            pass

        def prepare(self, *_args, **_kwargs):
            return result

    run_args: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "ScenePackageStore", lambda *_args: object())
    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "CollaborationCoordinator", FakeCoordinator)
    monkeypatch.setattr(cli, "_run", lambda args: run_args.append(args) or 0)

    assert (
        cli.main(
            [
                "prepare",
                "--task-id",
                "task",
                "--instruction",
                "place the carrot",
                "--scene",
                str(tmp_path / "scene"),
                "--output",
                str(output_dir),
                "--run-after-prepare",
            ]
        )
        == 0
    )
    assert len(run_args) == 1
    assert run_args[0].bundle == output_dir
    assert run_args[0].run_args == ["--filter_dataset_saving", "--headless"]


def test_run_bundle_publishes_rejected_preflight_report(
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
    monkeypatch.setattr(cli, "_preflight_bundle", lambda *args, **kwargs: report)

    assert cli.main(["run", "--bundle", str(bundle)]) == 2
    payload = json.loads((bundle / "execution_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["action_count"] == 0
