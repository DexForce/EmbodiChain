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

"""One selected physical model reaches binding, inspection and generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_frozen_physical_input import _freeze_reference
from .test_support_bundle import _placement_graph, support_scene
from .test_parallel_workflow import (
    _candidate_set as workflow_candidates,
    _Coordinator,
    _Executor,
    _selection,
    _TaskAgent,
)
from .orchestration.test_scene_adapter import _candidate, _candidate_set
from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
    load_frozen_physical_input,
)
from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    TaskEngineWorkflowCfg,
)
from embodichain.gen_sim.task_engine.orchestration.scene_adapter import SceneAdapter
from embodichain.gen_sim.task_engine.orchestration.scene_source import (
    SceneSourceRef,
    scene_revision_id,
)
from embodichain.gen_sim.task_engine.scene.final_inspection import inspect_final_scene
from embodichain.gen_sim.task_engine.scene_backend import SceneEngineBackend
from embodichain.gen_sim.task_engine.workflow import (
    TaskEngineWorkflow,
    _load_physical_reference,
)
from embodichain.gen_sim.task_engine.workflow_contracts import TASK_RUN_REQUEST_SCHEMA
from embodichain.utils.utility import save_config


@pytest.fixture
def physical_case(support_scene, tmp_path):
    scene = deepcopy(support_scene)
    source = tmp_path / "source" / "scene_config.json"
    source.parent.mkdir()
    scene = replace(scene, source_config_path=source, scene_dir=source.parent)
    for item in (*scene.rigid_objects, *scene.planner_objects):
        if item.get("runtime_uid", item.get("uid")) == "bottle":
            item.setdefault("attrs", {}).setdefault("mass_props", {})["mass"] = 0.01

    def write_source():
        save_config(
            source,
            {
                "format": "embodichain.scene-export/v1",
                "scene_id": "physical_selection_fixture",
                "background": list(scene.background),
                "rigid_object": list(scene.rigid_objects),
                "articulation": [],
            },
        )

    write_source()
    reference = tmp_path / "reference"
    _freeze_reference(scene, _placement_graph(), reference)
    # The explicit model may differ; neither source geometry nor equivalence is implied.
    for item in (*scene.rigid_objects, *scene.planner_objects):
        if item.get("runtime_uid", item.get("uid")) == "bottle":
            item["init_pos"] = [0.4, -0.2, 0.85]
    write_source()
    physical = load_frozen_physical_input(scene, reference)
    return scene, physical, reference


def _request(scene, output):
    return {
        "schema_version": TASK_RUN_REQUEST_SCHEMA,
        "task_id": "place_can",
        "task_instruction": "Place the can on the table.",
        "image_path": None,
        "gym_project": str(scene.source_config_path),
        "scene_edit_prompt": None,
        "output_dir": str(output),
    }


def test_binding_reads_selected_geometry_and_physical_inspection(
    physical_case, monkeypatch
):
    scene, physical, _ = physical_case
    seen = {}
    adapter = SceneAdapter(physical_input=physical)

    def select(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            role_bindings=None,
            binding_report={"status": "ambiguous"},
            selected_candidate=None,
            candidate_bindings={},
        )

    monkeypatch.setattr(adapter, "_select_candidates", select)
    result = adapter.adapt(
        _candidate_set([_candidate("c", "bottle")]), scene.source_config_path
    )

    bottle = next(v for v in seen["scene_objects"] if v["runtime_uid"] == "bottle")
    assert bottle["init_pos"][0] == pytest.approx(0.1)
    bounds = bottle["attributes"]["final_world_aabb"]
    assert (bounds["min"][0] + bounds["max"][0]) / 2 == pytest.approx(0.1)
    assert result.physical_input is physical
    physical.assert_unchanged()


@pytest.mark.parametrize("relabel", [False, True])
def test_selected_input_rejects_inspection_of_original_source(physical_case, relabel):
    scene, physical, _ = physical_case
    stale = inspect_final_scene(
        scene.source_config_path,
        revision_id=scene_revision_id(scene.source_config_path),
    )
    if relabel:
        stale["scene_revision_id"] = physical.physical_digest
    with pytest.raises(ValueError, match="adapted scene revision"):
        SceneAdapter(physical_input=physical).adapt(
            _candidate_set([_candidate("c", "bottle")]),
            scene.source_config_path,
            final_inspection=stale,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"z_rotation_degrees": 90.0},
        {"body_scale_policy": "multiply", "body_scale": (2.0, 2.0, 2.0)},
    ],
)
def test_selected_input_rejects_source_transform_overrides(physical_case, overrides):
    scene, physical, _ = physical_case
    with pytest.raises(ValueError, match="owns source transforms"):
        SceneAdapter(physical_input=physical).adapt(
            _candidate_set([_candidate("c", "bottle")]),
            SceneSourceRef(scene.source_config_path, **overrides),
        )


@pytest.mark.parametrize("execute", [False, True])
def test_workflow_uses_one_model_before_selection_and_preparation(
    physical_case, tmp_path, monkeypatch, execute
):
    scene, physical, reference = physical_case
    candidates = workflow_candidates()
    coordinator = _Coordinator(["bound"])
    executor = _Executor([[True]])

    class Backend(SceneEngineBackend):
        def select(self, analysis, candidate_set, scene_adapter, **kwargs):
            self.selected = scene_adapter.physical_input
            assert (
                self.selected.manifest()["components_digest"]
                == physical.manifest()["components_digest"]
            )
            return _selection(candidate_set)

        def inspect(self, *args, **kwargs):
            pytest.fail(
                "Original-source inspection must not qualify a different physical model"
            )

    from embodichain.gen_sim.task_engine.orchestration import visual_grounding

    monkeypatch.setattr(visual_grounding, "visual_grounding_available", lambda: False)
    backend = Backend()
    result = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=backend,
        coordinator=coordinator,
        action_executor=executor,
    ).run(
        _request(scene, tmp_path / "run"),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=1),
        planning_cfg=TaskEnginePlanningCfg(
            physical_reference=str(reference), max_episode_steps=30000
        ),
        execution_cfg=TaskEngineExecutionCfg(),
        execute=execute,
    )

    assert result.status == ("succeeded" if execute else "prepared")
    assert coordinator.kwargs[0]["physical_input"] is backend.selected
    inspection = coordinator.kwargs[0]["final_inspection"]
    assert inspection["scene_revision_id"] == backend.selected.physical_digest
    audit = json.loads((result.output_dir / "selected_physical_input.json").read_text())
    assert audit["original_scene_equivalence"] == "not_claimed_explicit_physical_input"
    assert "init_pos" in audit["changed_instance_fields"]["bottle"]
    assert audit["joint_drive_props"]["max_effort"]["left_eef"] == 0.5
    assert executor.calls == int(execute)


def test_selected_model_never_uses_original_source_visual_fallback(
    physical_case, tmp_path, monkeypatch
):
    scene, _, reference = physical_case
    candidates = workflow_candidates()
    coordinator = _Coordinator(["bound"])

    class Backend(SceneEngineBackend):
        def select(self, analysis, candidate_set, scene_adapter, **kwargs):
            return replace(
                _selection(candidate_set), selected_candidate=None, role_bindings=None
            )

    from embodichain.gen_sim.task_engine.orchestration import visual_grounding
    from embodichain.gen_sim.task_engine.orchestration import visual_evidence

    monkeypatch.setattr(visual_grounding, "visual_grounding_available", lambda: True)
    monkeypatch.setattr(
        visual_evidence,
        "render_scene_visual_evidence",
        lambda *a, **k: pytest.fail("Wrong physical world rendered"),
    )
    result = TaskEngineWorkflow(
        task_agent=_TaskAgent(candidates),
        scene_backend=Backend(),
        coordinator=coordinator,
    ).run(
        _request(scene, tmp_path / "run"),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=1),
        planning_cfg=TaskEnginePlanningCfg(
            physical_reference=str(reference), max_episode_steps=30000
        ),
        execution_cfg=TaskEngineExecutionCfg(),
        execute=False,
    )
    assert coordinator.calls == 0
    audit = json.loads((result.output_dir / "visual_grounding_status.json").read_text())
    assert audit["status"] == "skipped_physical_reference"


def test_direct_coordinator_cannot_overwrite_its_selected_reference(physical_case):
    from embodichain.gen_sim.task_engine.orchestration.coordinator import (
        TaskEngineCoordinator,
    )

    scene, physical, reference = physical_case
    coordinator = TaskEngineCoordinator(
        task_agent=SimpleNamespace(
            generate=lambda *a, **k: pytest.fail("Model must not run")
        )
    )
    with pytest.raises(ValueError, match="must not overlap"):
        coordinator.prepare(
            "task",
            "instruction",
            scene.source_config_path,
            reference,
            physical_input=physical,
            overwrite=True,
        )
    physical.assert_unchanged()


def test_cli_does_not_compare_source_image_to_a_selected_model(
    physical_case, tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import cli, config as config_module
    from embodichain.gen_sim.task_engine.scene import visual_consistency
    from embodichain.utils.utility import load_config

    scene, _, reference = physical_case
    config = load_config(Path(config_module.__file__).with_name("defaults.yaml"))
    config["planning"].update(physical_reference=str(reference), fit_grasp_assets=False)
    path = tmp_path / "selected.yaml"
    save_config(path, config)
    monkeypatch.setattr(
        visual_consistency,
        "review_scene_image",
        lambda *a, **k: pytest.fail("Wrong world visual check"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "prepare",
                "--mode",
                "scene",
                "--task-id",
                "t",
                "--instruction",
                "move",
                "--scene",
                str(scene.source_config_path),
                "--output-root",
                str(tmp_path / "runs"),
                "--config",
                str(path),
                "--reference-image",
                str(tmp_path / "reference.png"),
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("change", ["image", "edit", "overlap", "budget"])
def test_invalid_physical_selection_fails_before_any_model_calls(
    physical_case, tmp_path, change
):
    scene, _, reference = physical_case
    request = _request(scene, tmp_path / "run")
    planning = TaskEnginePlanningCfg(
        physical_reference=str(reference), max_episode_steps=30000
    )
    if change == "image":
        request.update(image_path=str(tmp_path / "image.png"), gym_project=None)
    elif change == "edit":
        request["scene_edit_prompt"] = "move the bottle"
    elif change == "overlap":
        request["output_dir"] = str(reference / "overwrite")
    else:
        planning.max_episode_steps = 99
    with pytest.raises(ValueError, match="physical_reference"):
        _load_physical_reference(request, planning, "dual_franka")
