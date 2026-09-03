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
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.semantic_planner import (
    UnsupportedSemanticCapabilityError,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    TaskProgramBundlePaths,
)

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


def test_artifact_transaction_relocates_paths_in_json_mapping_keys(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    with ArtifactTransaction(output) as transaction:
        assert transaction.staging_dir is not None
        staging = transaction.staging_dir.resolve().as_posix()
        (transaction.staging_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "config_path": f"{staging}/scene/scene_config.json",
                    "asset_sha256": {f"{staging}/scene/asset.usdc": "hash"},
                }
            ),
            encoding="utf-8",
        )
        transaction.commit()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["config_path"]
        == f"{output.resolve().as_posix()}/scene/scene_config.json"
    )
    assert list(manifest["asset_sha256"]) == [
        f"{output.resolve().as_posix()}/scene/asset.usdc"
    ]


def test_prepare_rejects_output_overlapping_read_only_source(tmp_path: Path) -> None:
    source = tmp_path / "gym_project"
    source.mkdir()
    coordinator = TaskEngineCoordinator(
        task_agent=object(),
        scene_adapter=SimpleNamespace(robot_profile="franka"),
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
    coordinator = TaskEngineCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        semantic_planner=SimpleNamespace(
            plan=lambda *_args, **_kwargs: pytest.fail(
                "Semantic Task Planner must not run"
            )
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
    assert not (result.output_dir / "semantic_task_graph.json").exists()
    assert not (result.output_dir / "task_program_deployment.yaml").exists()


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


def test_bound_prepare_publishes_semantic_task_program_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path)
    task_agent = SimpleNamespace(generate=lambda *args, **kwargs: candidates)
    scene_adapter = SimpleNamespace(
        robot_profile="franka",
        adapt=lambda *args, **kwargs: adaptation,
    )
    graph = {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "upright_can",
        "instruction": _UPRIGHT_CAN_INSTRUCTION,
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {},
        "nodes": [
            {
                "id": "upright__call_01",
                "call": {
                    "kind": "pick",
                    "object": "red_can",
                    "resources": {"primary": "left"},
                },
                "depends_on": [],
                "task_instance_id": "upright",
                "task_type": "E2",
                "role": "primary",
            }
        ],
        "task_groups": [
            {
                "id": "upright",
                "task_type": "E2",
                "node_ids": ["upright__call_01"],
                "depends_on": [],
                "success": {"type": "object_upright"},
            }
        ],
        "success": {"kind": "all_task_groups"},
    }
    generated_calls: list[dict[str, object]] = []

    def generate_bundle(planned_graph, _scene, output, **kwargs):
        generated_calls.append({"graph": planned_graph, **kwargs})
        root = Path(output)
        paths = TaskProgramBundlePaths(
            root=root,
            deployment=root / "task_program_deployment.yaml",
            program=root / "task_program/program.yaml",
            integration=root / "task_program/integration.yaml",
            scene=root / "components/scene.yaml",
            embodiment=root / "components/embodiment.yaml",
            execution_policy=root / "components/execution_policy.yaml",
            semantic_task_graph=root / "semantic_task_graph.json",
            integration_fingerprint=root / "integration_fingerprint.json",
        )
        for path in (
            paths.deployment,
            paths.program,
            paths.integration,
            paths.scene,
            paths.embodiment,
            paths.execution_policy,
            paths.semantic_task_graph,
            paths.integration_fingerprint,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        return deepcopy(planned_graph), paths

    monkeypatch.setattr(
        "embodichain.gen_sim.task_engine.orchestration.coordinator.generate_task_program_bundle",
        generate_bundle,
    )

    result = TaskEngineCoordinator(
        task_agent=task_agent,
        scene_adapter=scene_adapter,
        semantic_planner=SimpleNamespace(plan=lambda *_args, **_kwargs: graph),
    ).prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        tmp_path / "bundle",
        candidate_count=1,
    )

    assert result.bound
    assert generated_calls == [
        {
            "graph": graph,
            "robot_profile": "dual_franka",
            "max_episodes": None,
            "max_episode_steps": None,
        }
    ]
    assert result.semantic_task_graph == graph
    assert (result.output_dir / "semantic_task_graph.json").is_file()
    assert (result.output_dir / "task_program_deployment.yaml").is_file()
    assert not (result.output_dir / "grounded_task_plan.json").exists()
    assert not (result.output_dir / "seed_task_graph.json").exists()


def test_prepare_publishes_semantic_planning_failure_context(
    tmp_path: Path,
) -> None:
    candidates = _candidate_set()
    adaptation = _adaptation(tmp_path)
    output = tmp_path / "failed-bundle"
    output.mkdir()
    (output / "stale.txt").write_text("old", encoding="utf-8")

    def unsupported(*_args, **_kwargs):
        raise UnsupportedSemanticCapabilityError(
            "Task type E2 has no phase-one Semantic Call route."
        )

    result = TaskEngineCoordinator(
        task_agent=SimpleNamespace(generate=lambda *args, **kwargs: candidates),
        scene_adapter=SimpleNamespace(
            robot_profile="franka",
            adapt=lambda *args, **kwargs: adaptation,
        ),
        semantic_planner=SimpleNamespace(plan=unsupported),
    ).prepare(
        "upright_can",
        _UPRIGHT_CAN_INSTRUCTION,
        tmp_path / "scene_config.json",
        output,
        candidate_count=1,
        overwrite=True,
    )

    assert result.status == "planning_failed"
    assert not result.bound
    assert result.artifacts.preparation_failure.is_file()
    assert not (result.output_dir / "stale.txt").exists()
    failure = json.loads(
        result.artifacts.preparation_failure.read_text(encoding="utf-8")
    )
    assert failure["schema_version"] == "semantic_task_preparation_failure/v1"
    assert failure["task_id"] == "upright_can"
    assert failure["selected_candidate_id"] == "candidate_01"
    assert failure["status"] == "unsupported_semantic_capability"
    assert failure["attempts"] == [
        {
            "candidate_id": "candidate_01",
            "planner_route": "offline",
            "status": "failed",
            "error": {
                "type": "UnsupportedSemanticCapabilityError",
                "message": "Task type E2 has no phase-one Semantic Call route.",
            },
        }
    ]


def test_private_bundle_runner_forwards_arguments_without_leaking_sys_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    execution_output = tmp_path / "execution"
    captured: dict[str, object] = {}

    def fake_execute_bundle(path, forwarded, *, execution_output):
        captured.update(
            {
                "path": path,
                "forwarded": list(forwarded),
                "execution_output": execution_output,
            }
        )
        return 7

    monkeypatch.setattr(bundle_runner, "execute_bundle", fake_execute_bundle)
    import sys

    original = sys.argv
    assert (
        bundle_runner.main(
            [
                "--bundle",
                str(bundle),
                "--execution-output",
                str(execution_output),
                "--seed",
                "7",
            ]
        )
        == 7
    )

    assert sys.argv is original
    assert captured == {
        "path": str(bundle),
        "forwarded": ["--seed", "7"],
        "execution_output": str(execution_output),
    }


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


@pytest.mark.parametrize(
    "lower_layer_option",
    [
        ["--planner-mode", "ik_interp"],
        ["--ik-solver", "pytorch"],
        ["--show-grasp-poses"],
    ],
)
def test_unified_cli_rejects_lower_layer_execution_options(
    tmp_path: Path,
    lower_layer_option: list[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
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
                *lower_layer_option,
            ]
        )


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
    assert not hasattr(arguments, "planner_mode")
    assert not hasattr(arguments, "ik_solver")
    assert not hasattr(arguments, "show_grasp_poses")


def test_run_all_cli_accepts_runtime_window_option() -> None:
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
            "--open-window",
        ]
    )

    assert arguments.open_window is True


def test_run_all_cli_forwards_open_window(
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
                status="succeeded",
                succeeded=True,
                failure_class=None,
                output_dir=output,
                manifest_path=output / "run_manifest.json",
                final_bundle=output / "final" / "bundle",
            )

    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "TaskEngineWorkflow", FakeWorkflow)

    result = cli.main(
        [
            "run-all",
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
            "--open-window",
        ]
    )

    assert result == 0
    assert captured["execute"] is True
    assert captured["open_window"] is True


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
            assert kwargs["open_window"] is True
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
            "--open-window",
        ]
    )

    assert result == 0
