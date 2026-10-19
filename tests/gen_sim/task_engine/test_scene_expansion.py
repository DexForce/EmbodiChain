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

from dataclasses import replace
import json

import numpy as np
import pytest
import trimesh

from embodichain.gen_sim.task_spec import (
    TaskSpec,
    TaskRequirement,
    RoleSpec,
    Predicate,
    scene_layout_requirement,
    validate_task_spec,
)
from embodichain.gen_sim.task_engine.scene_expansion import (
    SceneExpansionExecutionCfg,
    compile_scene_layout_requirement,
    evaluate_scene_task_spec,
)
from embodichain.lab.sim.scene.expansion import (
    SceneLayoutInput,
    generate_scene_variants,
)


def source() -> SceneLayoutInput:
    meshes = {
        "bench": trimesh.creation.box([1.0, 1.0, 0.1]),
        "cube": trimesh.creation.box([0.05, 0.05, 0.05]),
        "goal": trimesh.creation.box([0.06, 0.06, 0.02]),
    }
    poses = {uid: np.eye(4) for uid in meshes}
    poses["cube"][:3, 3] = [-0.2, 0, 0.075]
    poses["goal"][:3, 3] = [0.2, 0.2, 0.06]
    return SceneLayoutInput(
        "e1",
        "bench",
        poses,
        {uid: mesh.vertices for uid, mesh in meshes.items()},
        {uid: mesh.faces for uid, mesh in meshes.items()},
        {"cube": "bench", "goal": "bench"},
    )


def spec(value: dict | None = None) -> TaskSpec:
    return TaskSpec(
        task_id="move",
        instruction="place the cube",
        roles=(
            RoleSpec("subject"),
            RoleSpec("destination", kind="target"),
            RoleSpec("surface", kind="surface"),
        ),
        goal=(
            Predicate(
                "semantic_goal",
                arguments={"subject": "subject", "reference": "destination"},
            ),
        ),
        requirements=(
            TaskRequirement(
                consumer="expansion",
                key="scene_layout",
                value=value
                or {
                    "movable_roles": ["subject"],
                    "fixed_roles": ["destination", "surface"],
                    "xy_offset_bounds_m": {"subject": [[-0.02, -0.03], [0.02, 0.03]]},
                },
            ),
        ),
    )


def test_normative_roles_and_ranges_are_canonical_and_runtime_free() -> None:
    task = spec()
    assert scene_layout_requirement(task).movable_roles == ("subject",)
    assert TaskSpec.from_dict(task.to_dict()).semantic_hash == task.semantic_hash
    changed = spec({"movable_roles": ["subject"], "fixed_roles": ["destination"]})
    assert changed.semantic_hash != task.semantic_hash
    with pytest.raises(ValueError, match="runtime settings"):
        spec({"movable_roles": ["subject"], "device": "cuda:0"})
    with pytest.raises(ValueError, match="undeclared"):
        validate_task_spec(spec({"movable_roles": ["missing"]}))


@pytest.mark.parametrize(
    "bounds",
    [
        [[-0.02, -0.03], [0.02, 0.03]],
        [[0.0, -0.02], [0.0, 0.02]],
        [[0.0, 0.0], [0.0, 0.0]],
    ],
)
def test_task_spec_compiles_role_permissions_and_root_offsets(bounds) -> None:
    task = spec(
        {
            "movable_roles": ["subject"],
            "fixed_roles": ["destination", "surface"],
            "xy_offset_bounds_m": {"subject": bounds},
        }
    )
    prepared = compile_scene_layout_requirement(
        task,
        source(),
        {"subject": ["cube"], "destination": ["goal"], "surface": ["bench"]},
    )
    result = generate_scene_variants(prepared, candidate_count=8)
    assert result.status == "completed"
    for variant in result.variants:
        assert [change.entity_uid for change in variant.pose_changes] == ["cube"]
        delta = np.asarray(variant.pose_changes[0].arena_pose)[:2, 3] - [-0.2, 0]
        assert np.all(delta >= np.array(bounds)[0] - 1e-9)
        assert np.all(delta <= np.array(bounds)[1] + 1e-9)


def test_group_root_yaw_is_reported_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported.*yaw"):
        compile_scene_layout_requirement(
            spec(
                {
                    "movable_roles": ["subject"],
                    "yaw_delta_range_deg": {"subject": [-30, 30]},
                }
            ),
            source(),
            {"subject": ["cube"]},
        )


def test_missing_measurement_cannot_accept_program_completion() -> None:
    graph = {
        "nodes": [{"id": "place", "task_type": "E1"}],
        "task_groups": [{"id": "move", "node_ids": ["place"]}],
    }
    missing = evaluate_scene_task_spec(
        spec(), graph, {"completed": True, "segments": []}, num_envs=1
    )
    assert missing[0]["certificate_ready"] is False
    assert missing[0]["acceptance_status"] == "unverified"
    failed = evaluate_scene_task_spec(
        spec(),
        graph,
        {
            "completed": True,
            "segments": [{"name": "place", "successes": [False], "active": [True]}],
        },
        num_envs=1,
    )
    assert failed[0]["certificate_ready"] is False
    assert failed[0]["acceptance_status"] == "failed"
    passed = evaluate_scene_task_spec(
        spec(),
        graph,
        {
            "completed": True,
            "segments": [{"name": "place", "successes": [True], "active": [True]}],
        },
        num_envs=1,
    )
    assert passed[0]["certificate_ready"] is True


def make_e1_candidates() -> dict:
    from embodichain.gen_sim.task_engine.agent import TaskAgent
    from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult

    def selector(reference: str) -> dict:
        return {
            "kind": "scene_ref",
            "step_id": "",
            "reference": reference,
            "quantifier": "one",
            "count": 0,
        }

    step = {
        "id": "move",
        "task_type": "E1",
        "object": selector("cube"),
        "target": selector("table"),
        "relation": "on",
        "required_arm": "auto",
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "preserve",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }

    def interpreter(*args, **kwargs):
        return InstructionDraftResult(
            intent={"steps": [step]},
            model="fixture",
            attempts=1,
            latency_seconds=0.0,
            normalizations=(),
        )

    return TaskAgent(interpreter=interpreter).generate(
        "e1-layout", "pick and place the cube on the table", candidate_count=1
    )


def make_e1_export(root) -> object:
    export = root / "scene_export"
    export.mkdir(parents=True)
    table = trimesh.creation.box([1.4, 0.1, 1.0])
    table.apply_translation([0.0, 0.7, 0.0])
    cube = trimesh.creation.box([0.05, 0.05, 0.05])
    entries = []
    for uid, mesh, position in (
        ("table", table, [0.0, 0.0, 0.0]),
        ("cube", cube, [0.0, 0.15, 0.776]),
    ):
        path = export / "mesh_assets" / uid / (uid + ".glb")
        path.parent.mkdir(parents=True)
        mesh.export(path)
        entries.append(
            {
                "uid": uid,
                "is_articulated": False,
                "shape": {"shape_type": "Mesh", "fpath": str(path.relative_to(export))},
                "body_type": "kinematic" if uid == "table" else "dynamic",
                "init_pos": position,
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
                "name": uid,
                "category": "table" if uid == "table" else "cube",
            }
        )
    (export / "scene_config.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "scene_id": "e1-layout",
                "background": entries[:1],
                "rigid_object": entries[1:],
                "articulation": [],
            }
        )
    )
    (export / "scene_graph.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "object_id": uid,
                        "parent_id": None if uid == "table" else "table",
                        "parent_relation": None if uid == "table" else "on",
                        "table_region": None,
                        "pose_description": None,
                    }
                    for uid in ("table", "cube")
                ],
                "relations": [],
            }
        )
    )
    return export


def make_e1_workflow():
    from embodichain.gen_sim.task_engine.workflow import TaskEngineWorkflow
    from embodichain.gen_sim.task_engine.orchestration.scene_adapter import SceneAdapter

    def grounder(**kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "step_01.object",
                    "status": "resolved",
                    "uids": ["cube"],
                    "confidence": 1.0,
                },
                {
                    "reference_id": "step_01.target",
                    "status": "resolved",
                    "uids": ["table"],
                    "confidence": 1.0,
                },
            ]
        }

    return TaskEngineWorkflow(scene_adapter=SceneAdapter(grounding_caller=grounder))


def e1_spec(candidates: dict) -> TaskSpec:
    from embodichain.gen_sim.task_engine.task_spec import task_spec_from_candidate

    initial = task_spec_from_candidate(candidates["candidates"][0])
    moving = next(role.name for role in initial.roles if role.kind == "object")
    fixed = next(
        role.name for role in initial.roles if role.kind in ("target", "surface")
    )
    return replace(
        initial,
        requirements=(
            *initial.requirements,
            TaskRequirement(
                consumer="expansion",
                key="scene_layout",
                value={
                    "movable_roles": [moving],
                    "fixed_roles": [fixed],
                    "xy_offset_bounds_m": {moving: [[-0.015, -0.015], [0.015, 0.015]]},
                },
            ),
        ),
    )


def test_task_spec_drives_independent_variant_bundles_without_models(
    tmp_path, monkeypatch
) -> None:
    from embodichain.gen_sim.task_engine.config import (
        TaskEngineWorkflowCfg,
        TaskEnginePlanningCfg,
        TaskEngineExecutionCfg,
    )
    from embodichain.gen_sim.task_engine.workflow_contracts import (
        TASK_RUN_REQUEST_SCHEMA,
    )

    export = make_e1_export(tmp_path / "source")
    candidates = make_e1_candidates()
    normative = e1_spec(candidates)
    request = {
        "schema_version": TASK_RUN_REQUEST_SCHEMA,
        "task_id": candidates["task_id"],
        "task_instruction": candidates["instruction"],
        "image_path": None,
        "gym_project": str(export),
        "scene_edit_prompt": None,
        "output_dir": str(tmp_path / "result"),
    }
    result = make_e1_workflow().run(
        request,
        task_spec=normative,
        candidate_set=candidates,
        scene_expansion_execution=SceneExpansionExecutionCfg(
            candidate_count=2, max_attempts=100
        ),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=1, max_action_attempts=1),
        planning_cfg=TaskEnginePlanningCfg(candidate_count=1),
        execution_cfg=TaskEngineExecutionCfg(),
        execute=False,
    )
    assert result.status == "prepared", result.manifest_path.read_text()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["task_spec"]["semantic_hash"] == normative.semantic_hash
    assert manifest["generated_count"] == 2
    bundles = list(result.output_dir.glob("variants/*/task_run/final/bundle"))
    assert len(bundles) == 2
    for bundle in bundles:
        assert (
            TaskSpec.from_dict(
                json.loads((bundle / "task_spec.json").read_text())
            ).semantic_hash
            == normative.semantic_hash
        )
        frozen = json.loads((bundle / "scene_variant.json").read_text())
        assert frozen["variant_id"] in {
            item["variant_id"] for item in manifest["variants"]
        }
        from embodichain.gen_sim.task_engine._bundle_runner import (
            _verify_task_spec_bundle,
        )

        assert _verify_task_spec_bundle(bundle, normative, frozen)
        from embodichain.gen_sim.task_engine._bundle_runner import (
            _verify_source,
            _verify_integration_fingerprint,
        )

        _verify_source(bundle)
        _verify_integration_fingerprint(
            bundle,
            bundle / "task_program_deployment.yaml",
            json.loads((bundle / "semantic_task_graph.json").read_text()),
            json.loads((bundle / "integration_fingerprint.json").read_text()),
        )
        from embodichain.gen_sim.task_engine._bundle_runner import execute_bundle

        spec_path = bundle / "task_spec.json"
        preserved_spec = spec_path.read_bytes()
        spec_path.unlink()
        try:
            with pytest.raises(ValueError, match="requires its authoritative TaskSpec"):
                execute_bundle(
                    bundle, execution_output=tmp_path / "rejected_missing_spec"
                )
        finally:
            spec_path.write_bytes(preserved_spec)
        frozen["layout"]["poses"]["cube"][0][3] += 0.01
        with pytest.raises(ValueError, match="identity"):
            _verify_task_spec_bundle(bundle, normative, frozen)
    from embodichain.gen_sim.task_engine import scene_expansion

    monkeypatch.setattr(
        scene_expansion,
        "generate_scene_variants",
        lambda *a, **k: pytest.fail("Replay must never sample."),
    )
    replay = make_e1_workflow().run(
        {**request, "output_dir": str(tmp_path / "replay")},
        task_spec=normative,
        candidate_set=candidates,
        scene_expansion_execution=SceneExpansionExecutionCfg(
            candidate_count=2,
            max_attempts=100,
            layout_dir=result.output_dir / "scene_layouts",
        ),
        workflow_cfg=TaskEngineWorkflowCfg(max_scene_attempts=1, max_action_attempts=1),
        planning_cfg=TaskEnginePlanningCfg(candidate_count=1),
        execution_cfg=TaskEngineExecutionCfg(),
        execute=False,
    )
    replay_manifest = json.loads(replay.manifest_path.read_text())
    assert replay.status == "prepared"
    assert replay_manifest["source"] == "saved"
    assert [item["variant_id"] for item in manifest["variants"]] == [
        item["variant_id"] for item in replay_manifest["variants"]
    ]


def test_supported_initial_relations_are_measured_per_row_and_goals_are_preserved() -> (
    None
):
    from types import SimpleNamespace
    import torch
    from embodichain.gen_sim.task_engine.scene_expansion import (
        _observe_initial_conditions,
    )
    from embodichain.gen_sim.task_engine.task_spec import _preserves_task_spec

    source_input = source()
    initial = replace(
        spec(),
        init=(
            Predicate(
                "relation",
                arguments={
                    "subject": "subject",
                    "reference": "surface",
                    "relation": "on",
                },
            ),
        ),
    )
    assert _preserves_task_spec(initial, spec())
    changed = replace(
        initial,
        goal=(
            Predicate(
                "semantic_goal",
                arguments={"subject": "destination", "reference": "subject"},
            ),
        ),
    )
    assert not _preserves_task_spec(changed, spec())
    entities = {}
    for uid in source_input.poses:
        poses = np.repeat(np.asarray(source_input.poses[uid])[None], 2, axis=0).copy()
        if uid == "cube":
            poses[1, 0, 3] = 0.8
        entities[uid] = SimpleNamespace(
            get_local_pose=lambda *, to_matrix, value=poses: torch.tensor(value),
            get_vertices=lambda *, scale, env_ids, value=source_input.vertices[
                uid
            ]: np.asarray(value)[None],
            get_triangles=lambda *, env_ids, value=source_input.faces[uid]: np.asarray(
                value
            )[None],
        )
    sim = SimpleNamespace(num_envs=2, get_rigid_object=lambda uid: entities.get(uid))
    observations = _observe_initial_conditions(
        initial,
        sim,
        {"subject": ["cube"], "surface": ["bench"]},
        source_input.parent_by_uid,
    )
    key = initial.init[0].predicate_id
    assert observations == ({key: True}, {key: False})


def test_initial_unknown_evidence_cannot_receive_a_certificate() -> None:
    initial = replace(
        spec(),
        init=(
            Predicate(
                "contact", arguments={"subject": "subject", "reference": "surface"}
            ),
        ),
    )
    graph = {
        "nodes": [{"id": "pick", "task_type": "E1"}],
        "task_groups": [{"id": "step", "node_ids": ["pick"]}],
    }
    runtime = {"segments": [{"name": "pick", "active": [True], "successes": [True]}]}
    report = evaluate_scene_task_spec(initial, graph, runtime, num_envs=1)[0]
    assert report["acceptance_status"] == "unverified"
    assert report["certificate_ready"] is False


def test_native_mesh_aliases_preserve_assets_and_external_texture_paths(
    tmp_path,
) -> None:
    from types import SimpleNamespace
    from embodichain.lab.sim.shapes import MeshCfg
    from embodichain.gen_sim.task_engine._bundle_runner import _alias_runtime_meshes

    source_dir = tmp_path / ("nested_" * 25)
    source_dir.mkdir()
    (source_dir / "mesh.gltf").write_text("asset")
    (source_dir / "texture.png").write_text("pixels")
    shape = MeshCfg(fpath=str(source_dir / "mesh.gltf"))
    cfg = SimpleNamespace(background=[SimpleNamespace(shape=shape)], rigid_object=[])
    alias_dir = tmp_path / "aliases"
    alias_dir.mkdir()
    _alias_runtime_meshes(cfg, alias_dir)
    from pathlib import Path

    alias = Path(shape.fpath)
    assert alias.read_text() == "asset"
    assert alias.with_name("texture.png").read_text() == "pixels"
    assert len(str(alias)) < len(str(source_dir / "mesh.gltf"))
