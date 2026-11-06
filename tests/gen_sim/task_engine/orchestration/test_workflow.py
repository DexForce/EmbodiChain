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

"""Software workflow integration uses fixed declarations and real bundle owners."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from embodichain.gen_sim.task_engine.bundle_loading import (
    load_current_task_program_bundle,
)
from embodichain.gen_sim.task_engine.bundle_deployment import prepare_configured_bundle
from embodichain.gen_sim.task_engine.bundle_io import write_bundle_atomic
from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    TaskEngineWorkflowCfg,
)
from embodichain.gen_sim.task_engine.orchestration.scene_adapter import SceneAdapter
from embodichain.gen_sim.task_engine.workflow import TaskEngineWorkflow
from embodichain.gen_sim.task_engine.workflow_contracts import TASK_RUN_REQUEST_SCHEMA

from .test_scene_adapter import _candidate_set, _placement_candidate


def _fixed_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, executor=None):
    from embodichain.gen_sim.task_engine import semantic_graph_html
    from embodichain.lab.sim.sim_manager import SimulationManager

    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **k: pytest.fail("declaration workflow created simulation"),
    )
    # Display/model translation has separate coverage; this fixture supplies no
    # provider and exercises only the executable orchestration/declaration path.
    monkeypatch.setattr(semantic_graph_html, "_finalize_run_html", lambda *a, **k: None)
    scene = tmp_path / "scene"
    scene.mkdir()
    for uid, extents in (("table", [1.5, 1.0, 0.04]), ("red_can", [0.05, 0.05, 0.1])):
        mesh = trimesh.creation.box(extents=extents)
        mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
        mesh.export(scene / f"{uid}.glb")

    def item(uid):
        table = uid == "table"
        return dict(
            uid=uid,
            name=uid,
            category="table" if table else "can",
            affordances=(
                ["support_surface"]
                if table
                else ["graspable", "placeable", "orientable"]
            ),
            shape=dict(shape_type="Mesh", fpath=str(scene / f"{uid}.glb")),
            init_pos=[0.0, 0.0, 0.56 if table else 0.65],
            init_rot=[0.0, 0.0, 0.0],
            body_scale=[1.0, 1.0, 1.0],
        )

    (scene / "scene_config.json").write_text(
        json.dumps(
            dict(
                format="embodichain.scene-export/v1",
                scene_id="fixed-workflow",
                background=[item("table")],
                rigid_object=[item("red_can")],
                articulation=[],
            )
        )
    )
    candidates = _candidate_set([_placement_candidate()])
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(candidates)

    def ground(**kwargs):
        return dict(
            bindings=[
                dict(
                    reference_id="upright.object",
                    status="resolved",
                    uids=["red_can"],
                    confidence=0.95,
                ),
                dict(
                    reference_id="upright.target",
                    status="resolved",
                    uids=["table"],
                    confidence=0.95,
                ),
            ]
        )

    workflow = TaskEngineWorkflow(
        task_agent=SimpleNamespace(generate=generate),
        scene_adapter=SceneAdapter(grounding_caller=ground),
        action_executor=executor,
    )
    request = dict(
        schema_version=TASK_RUN_REQUEST_SCHEMA,
        task_id="upright_can",
        task_instruction="test-instruction",
        image_path=None,
        gym_project=str(scene),
        scene_edit_prompt=None,
        output_dir=str(tmp_path / "run"),
    )
    cfg = dict(
        workflow_cfg=TaskEngineWorkflowCfg(max_action_attempts=2),
        planning_cfg=TaskEnginePlanningCfg(candidate_count=1),
        execution_cfg=TaskEngineExecutionCfg(),
    )
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in scene.iterdir()
    }
    return workflow, request, cfg, calls, hashes


def test_fixed_workflow_publishes_identical_final_bundle_without_relocating_inputs(
    tmp_path, monkeypatch
):
    workflow, request, cfg, calls, hashes = _fixed_workflow(tmp_path, monkeypatch)
    result = workflow.run(request, **cfg, execute=False)
    assert result.status == "prepared" and not result.succeeded
    assert len(calls) == 1
    prepared = tmp_path / "run/attempts/scene_0001/bundle"
    source = load_current_task_program_bundle(prepared)
    final = load_current_task_program_bundle(result.final_bundle)
    assert source.snapshot.identity.digest == final.snapshot.identity.digest
    assert dict(source.snapshot.files) == dict(final.snapshot.files)
    assert source.task_spec is not None
    assert (
        json.loads(source.snapshot.files["semantic_graph_validation.json"])["coverage"][
            "status"
        ]
        == "covered"
    )
    for name in source.snapshot.identity.to_dict()["external_inputs"]:
        assert Path(name).is_file()
    assert hashes == {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in hashes
    }
    manifest = json.loads(result.manifest_path.read_bytes())
    assert len(manifest["attempts"]) == 1
    assert manifest["attempts"][0]["action_attempts"] == []
    assert manifest["task_spec"]["bundle_companions"]["html"] == "semantic_graph.html"
    from embodichain.gen_sim.task_engine import cli

    assert cli.main(["validate-graph", "--bundle", str(result.final_bundle)]) == 0
    files = dict(source.snapshot.files)
    audit = json.loads(files["semantic_graph_validation.json"])
    audit["coverage"]["issues"].append({"code": "unverified", "owner": "goal_001"})
    files["semantic_graph_validation.json"] = json.dumps(audit).encode()
    identity = source.snapshot.identity.to_dict()
    stale = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in identity["external_inputs"]
        },
        implementation_revisions=identity["implementation_revisions"],
    )
    write_bundle_atomic(tmp_path / "stale-audit-bundle", stale)
    with pytest.raises(SystemExit) as rejected:
        cli.main(["validate-graph", "--bundle", str(tmp_path / "stale-audit-bundle")])
    assert rejected.value.code == 2


def test_bounded_attempts_never_accept_an_unverified_success_report(
    tmp_path, monkeypatch
):
    calls = []

    def invalid_report(bundle, output, **options):
        calls.append((Path(bundle), Path(output), options))
        return {"environments": [{"success": True}]}

    workflow, request, cfg, generation, _ = _fixed_workflow(
        tmp_path, monkeypatch, executor=invalid_report
    )
    result = workflow.run(request, **cfg, base_seed=31)
    assert result.status == "failed" and result.failure_class == "action_execution"
    assert result.final_bundle is None
    assert len(generation) == 1
    assert [row[2]["seed"] for row in calls] == [31, 32]
    assert [row[1].name for row in calls] == ["action_0001", "action_0002"]
    assert calls[0][0] == calls[1][0]
    manifest = json.loads(result.manifest_path.read_bytes())
    assert len(manifest["attempts"]) == 1
    attempts = manifest["attempts"][0]["action_attempts"]
    assert [row["successful_environments"] for row in attempts] == [0, 0]
    assert all(row["error"]["type"] == "ValueError" for row in attempts)
    load_current_task_program_bundle(calls[0][0])


def test_cancellation_retains_the_prepared_bundle_and_interruption_receipt(
    tmp_path, monkeypatch
):
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt("cancelled at action boundary")

    workflow, request, cfg, _, _ = _fixed_workflow(
        tmp_path, monkeypatch, executor=cancel
    )
    with pytest.raises(KeyboardInterrupt, match="action boundary"):
        workflow.run(request, **cfg)
    root = tmp_path / "run"
    loaded = load_current_task_program_bundle(root / "attempts/scene_0001/bundle")
    assert loaded.task_spec is not None
    assert not (root / "run_manifest.json").exists()
    assert (
        json.loads((root / "workflow_interruption.json").read_bytes())["status"]
        == "interrupted"
    )
