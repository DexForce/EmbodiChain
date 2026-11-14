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

"""Cross-engine binding, strict deployment composition and execution receipts."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import trimesh
import yaml

from embodichain.gen_sim._run_config import _RunCfg, _load_run_config
from embodichain.gen_sim._workflow import _run
from embodichain.gen_sim.services import ChatResponse, GenSimServices, GenSimServicesCfg

pytestmark = pytest.mark.no_sim

_ROOT = Path(__file__).resolve().parents[2]


class _ScriptedChat:
    def __init__(self, *responses: dict) -> None:
        self.responses = deque(responses)

    def complete(self, request) -> ChatResponse:
        return ChatResponse(text=json.dumps(self.responses.popleft()))


def _services(
    *, missing_object: bool = False, required_color: str | None = None
) -> GenSimServices:
    def selector(reference: str = "") -> dict:
        return dict(
            kind="scene_ref" if reference else "none",
            step_id="",
            reference=reference,
            quantifier="one",
            count=0,
        )

    step = dict(
        id="step_01",
        task_type="E1",
        object=selector("cube"),
        target=selector("table"),
        relation="on",
        required_arm="auto",
        transfer_arm="none",
        receive_arm="none",
        orientation_goal="none",
        target_state="none",
        target_setting=0,
        layout="none",
        axis="none",
        direction="none",
        terminal_behavior="none",
        depends_on=[],
    )
    grounding = {
        "objects": [
            dict(
                id="source",
                description="red cube",
                bbox=[100, 200, 300, 500],
                provenance="observed",
            ),
            dict(
                id="table",
                description="table",
                bbox=[100, 100, 900, 900],
                provenance="observed",
            ),
        ],
        "bindings": {"step_01.object": ["source"], "step_01.target": ["table"]},
    }
    plan = {
        "nodes": [
            dict(
                id="pick_cube",
                step_id="step_01",
                call=dict(kind="pick", object="source", resources={}),
                depends_on=[],
                semantic=dict(
                    object="source",
                    reference="table",
                    relation="on",
                    intent={"relation": "on"},
                ),
            ),
            dict(
                id="place_cube",
                step_id="step_01",
                call=dict(
                    kind="place", object="source", on="table_surface", resources={}
                ),
                depends_on=["pick_cube"],
                semantic=dict(
                    object="source",
                    reference="table",
                    relation="on",
                    intent={"relation": "on"},
                ),
            ),
        ],
        "requirements": [
            dict(
                id="r1",
                quote="Put the cube on the table",
                node_ids=["pick_cube", "place_cube"],
            )
        ],
    }
    if required_color is not None:
        grounding["role_properties"] = {"role_001": {"color": required_color}}
        grounding["clauses"] = [
            dict(
                id="required_color",
                quote=required_color,
                target="role_001",
                origin="explicit",
                resolution="resolved",
                constraint={
                    "kind": "role_property",
                    "key": "color",
                    "value": required_color,
                },
            )
        ]
        plan["requirements"][0]["quote"] = f"Put the {required_color} cube on the table"
    return GenSimServices(
        GenSimServicesCfg(),
        injected={
            "task_interpretation": _ScriptedChat({"steps": [step]}),
            "task_image_grounding": _ScriptedChat(
                grounding,
                {
                    "objects": {
                        "source": None if missing_object else "cube_native",
                        "table": "table",
                    },
                    "resources": {"default_primary": "primary_manipulator"},
                },
            ),
            "task_semantic_planning": _ScriptedChat(plan),
        },
    )


def _write_scene(root: Path) -> None:
    export = root / "scene_export"
    export.mkdir(parents=True)
    entries = []
    for uid, extents, position, body_type in (
        ("table", [0.6, 0.04, 0.6], [0.45, 0.0, 0.30], "static"),
        ("cube_native", [0.04, 0.04, 0.04], [0.40, -0.12, 0.34], "dynamic"),
    ):
        trimesh.creation.box(extents).export(export / (uid + ".glb"))
        entries.append(
            dict(
                uid=uid,
                category="table" if uid == "table" else "cube",
                name=uid,
                description=uid,
                shape={"shape_type": "Mesh", "fpath": uid + ".glb"},
                init_pos=position,
                init_rot=[0, 0, 0],
                body_scale=[1, 1, 1],
                body_type=body_type,
                attrs={"mass_props": {"mass": 0.05}},
                max_convex_hull_num=1,
            )
        )
    (export / "scene_config.json").write_text(
        json.dumps(
            dict(
                format="embodichain.scene-export/v1",
                scene_id="test-scene",
                background=entries[:1],
                rigid_object=entries[1:],
                articulation=[],
            )
        )
    )


@pytest.fixture
def run_cfg(tmp_path: Path) -> _RunCfg:
    image = tmp_path / "input.png"
    Image.new("RGB", (100, 80), "white").save(image)
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        yaml.safe_dump(
            {
                "physics": "default",
                "env": {"sim_steps_per_control": 4},
                "render_cfg": {"renderer": "no-render"},
                "enable_sensor": False,
                "embodiment": {
                    "component": str(
                        _ROOT
                        / "extensions/embodichain_tasks/configs/components/embodiments/franka_panda.yaml"
                    )
                },
                "max_episode_steps": 1000,
            }
        )
    )
    policy = yaml.safe_load(
        (
            _ROOT
            / "extensions/embodichain_tasks/configs/components/execution_policies/motion_gen_verified.yaml"
        ).read_text()
    )
    policy["requires"]["embodiment_contract"] = "single_arm_parallel_gripper"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy))
    return _RunCfg(
        image=image,
        instruction="Put the cube on the table",
        output=tmp_path / "run",
        runtime=runtime,
        execution_policy=policy_path,
    )


@pytest.mark.parametrize("success", [True, False])
def test_workflow_generates_binds_and_preserves_actual_execution_verdict(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline import generate
    from embodichain.gen_sim import _run_execution
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.task_program.integrations.composition import (
        load_configured_task_program_deployment,
    )

    if not success:
        run_cfg.instruction += " \n"
        run_cfg.task_id = " gensim_task "

    monkeypatch.setattr(
        generate,
        "generate_scene_from_image",
        lambda image, output, **kwargs: _write_scene(output),
    )
    executed = []

    def execute(path: Path, **kwargs) -> dict:
        physical = resolve_gym_components(
            yaml.safe_load(path.read_text()), base_dir=path.parent
        )
        deployment = load_configured_task_program_deployment(
            task_program=physical.config["task_program"],
            skill_profile=physical.embodiment_skill_profile,
            base_dir=path.parent,
        )
        program = deployment.load_program()
        physical.validate_scene_bindings(deployment.scene_binding)
        assert program.instruction == run_cfg.instruction.strip()
        assert (
            deployment.scene_binding["rigid_objects"][0]["simulation_uid"]
            == "cube_native"
        )
        executed.append(path)
        return {
            "status": "succeeded" if success else "failed",
            "completed": success,
            "all_success": success,
            "effect_assurance": "verified",
        }

    monkeypatch.setattr(_run_execution, "_execute_deployment", execute)
    with _services() as services:
        manifest = _run(run_cfg, services=services)
        assert services.get("task_semantic_planning") is not None
    assert len(executed) == 1
    assert manifest["task_id"] == run_cfg.task_id.strip()
    assert manifest["instruction"] == run_cfg.instruction.strip()
    assert manifest["execution_ready"] is True
    assert manifest["status"] == ("succeeded" if success else "failed")
    assert manifest["runtime_evidence_status"] == manifest["status"]
    assert manifest["task_spec_verdict"] == "not_evaluated"
    receipt = json.loads((run_cfg.output / "execution_report.json").read_text())
    assert receipt["all_success"] is success
    import re

    def review_payload(path):
        return json.loads(
            re.search(
                r'<script id="graph-data" type="application/json">(.*?)</script>',
                path.read_text(),
                re.S,
            )[1]
        )["review"]

    assert (
        review_payload(run_cfg.output / "task_review.html")["attempt"]["status"]
        == receipt["status"]
    )
    assert (
        review_payload(run_cfg.output / "task" / "semantic_task_graph.html")["attempt"]
        is None
    )
    from embodichain.gen_sim.task_engine._review import write_review

    plan_root = run_cfg.output / "task"
    from embodichain.gen_sim.workflow import load_task_plan, revise_task_plan

    plan = load_task_plan(plan_root)
    graph = json.loads((plan_root / "semantic_task_graph.json").read_text())
    source = json.loads((plan_root / "semantic_graph_validation.json").read_text())[
        "source"
    ]
    proposal = {
        "nodes": [
            dict(
                id=n["id"],
                step_id=n["task_instance_id"],
                call=n["call"],
                depends_on=n["depends_on"],
                semantic=source["annotations"][n["id"]],
            )
            for n in graph["nodes"]
        ],
        "requirements": source["requirements"],
    }
    proposal["nodes"][0]["semantic"]["intent"][
        "description"
    ] = "different deployment intent"
    revised_plan = revise_task_plan(
        plan_root,
        proposal,
        run_cfg.output / "revised-plan",
        expected_spec_digest=plan["spec_digest"],
        expected_graph_digest=plan["semantic_graph_sha256"],
        expected_plan_digest=plan["plan_digest"],
    )
    assert (
        load_task_plan(revised_plan)["semantic_graph_sha256"]
        == plan["semantic_graph_sha256"]
    )
    different_plan = run_cfg.output / "different-plan.html"
    write_review(revised_plan, different_plan, run_cfg.output)
    assert "error" in review_payload(different_plan)["attempt"]

    feedback_path = run_cfg.output / "attempt_feedback.json"
    feedback = json.loads(feedback_path.read_text())
    feedback["attempt_id"] = "stale"
    feedback_path.write_text(json.dumps(feedback))
    alternate = run_cfg.output / "stale-review.html"
    write_review(run_cfg.output / "task", alternate, run_cfg.output)
    assert "error" in review_payload(alternate)["attempt"]
    unbound = json.loads((run_cfg.output / "task/run_manifest.json").read_text())
    assert unbound["execution_ready"] is False
    binding = yaml.safe_load(
        (run_cfg.output / "deployment/integration.yaml").read_text()
    )
    destination = binding["scene_binding"]["rigid_objects"][1]["affordances"][0]
    pose = np.array(destination["object_target_pose"]).reshape(4, 4)
    # Table is 4 cm thick, cube is 4 cm tall; the owner adds its 1 mm landing gap.
    assert pose[2, 3] == pytest.approx(0.041, abs=1e-7)


def test_unresolved_native_binding_retains_failure_and_never_executes(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline import generate
    from embodichain.gen_sim import _run_execution

    monkeypatch.setattr(
        generate,
        "generate_scene_from_image",
        lambda image, output, **kwargs: _write_scene(output),
    )
    execute = Mock(side_effect=AssertionError("Unbound task executed"))
    monkeypatch.setattr(_run_execution, "_execute_deployment", execute)
    with (
        _services(missing_object=True) as services,
        pytest.raises(ValueError, match="Unresolved"),
    ):
        _run(run_cfg, services=services)
    execute.assert_not_called()
    manifest = json.loads((run_cfg.output / "run_manifest.json").read_text())
    assert (manifest["status"], manifest["stage"], manifest["execution_ready"]) == (
        "failed",
        "binding",
        False,
    )
    assert (run_cfg.output / "task/semantic_task_graph.json").is_file()
    assert (run_cfg.output / "service_calls.json").is_file()


@pytest.mark.parametrize(
    "color,stale_evidence",
    [
        (None, None),
        ("blue", None),
        ("red", None),
        ("red", "mesh"),
        ("red", "deployment_mesh"),
        ("red", "source"),
        ("red", "sha256"),
    ],
)
def test_default_scene_route_enforces_frozen_role_properties(
    run_cfg: _RunCfg,
    monkeypatch: pytest.MonkeyPatch,
    color: str | None,
    stale_evidence: str | None,
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline import generate
    from embodichain.gen_sim import _run_execution

    run_cfg.instruction = "Put the red cube on the table"

    def scene(image, output, **kwargs):
        from embodichain.gen_sim.scene_engine.core.scene import Scene
        from embodichain.gen_sim.scene_engine.core.scene_object import (
            SceneObject,
            ObjectPhysics,
        )
        from embodichain.gen_sim.scene_engine.core.scene_graph import (
            SceneGraph,
            SceneGraphNode,
        )
        from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import (
            SceneExporter,
        )

        _write_scene(output)
        path = output / "scene_export/scene_config.json"
        value = json.loads(path.read_text())
        objects = []
        for entry in [*value["background"], *value["rigid_object"]]:
            mesh_path = path.parent / entry["shape"]["fpath"]
            if entry["uid"] == "cube_native" and color is not None:
                mesh = trimesh.load_mesh(mesh_path, process=False)
                mesh.visual.vertex_colors = (
                    [255, 0, 0, 255] if color == "red" else [0, 0, 255, 255]
                )
                mesh.export(mesh_path)
            x, y, z = entry["init_pos"]
            objects.append(
                SceneObject(
                    id=entry["uid"],
                    kind="table" if entry["uid"] == "table" else "asset",
                    category=entry["category"],
                    name=entry["name"],
                    description="red cube",
                    simready_glb_path=str(mesh_path),
                    pos=[x, z, -y],
                    rot=[0, 0, 0],
                    scale=[1, 1, 1],
                    physics=ObjectPhysics(
                        body_type="kinematic" if entry["uid"] == "table" else "dynamic",
                        attrs=entry["attrs"],
                        max_convex_hull_num=1,
                    ),
                )
            )
        SceneExporter(
            scene=Scene(objects=objects),
            scene_graph=SceneGraph(
                nodes=[
                    SceneGraphNode(object_id="table", parent_id=None),
                    SceneGraphNode(
                        object_id="cube_native", parent_id="table", parent_relation="on"
                    ),
                ]
            ),
            output_root=output,
        ).export()

    if stale_evidence is not None:
        # Reuse an exported workspace, exactly as --scene does, after its proof
        # source, digest or actual deployment mesh has changed.
        prepared = run_cfg.output.parent / "prepared_scene"
        scene(None, prepared)
        config_path = prepared / "scene_export/scene_config.json"
        config = json.loads(config_path.read_text())
        asset = config["rigid_object"][0]
        proof = asset["property_evidence"]["color"]
        mesh_path = config_path.parent / asset["shape"]["fpath"]
        if stale_evidence == "mesh":
            mesh = trimesh.load_mesh(mesh_path, process=False)
            mesh.visual.vertex_colors = [0, 0, 255, 255]
            mesh.export(mesh_path)
        elif stale_evidence == "deployment_mesh":
            # The legacy runtime loads shape.fpath even if mesh_path is present.
            asset["mesh_path"] = asset["shape"]["fpath"]
            blue_path = mesh_path.with_name("deployed-blue.glb")
            blue = trimesh.load_mesh(mesh_path, process=False)
            blue.visual.vertex_colors = [0, 0, 255, 255]
            blue.export(blue_path)
            asset["shape"]["fpath"] = blue_path.relative_to(
                config_path.parent
            ).as_posix()
        elif stale_evidence == "source":
            other = mesh_path.with_name("other-red.glb")
            other.write_bytes(mesh_path.read_bytes())
            proof["source"] = other.relative_to(config_path.parent).as_posix()
        else:
            proof["sha256"] = "0" * 64
        config_path.write_text(json.dumps(config))
        run_cfg.scene = prepared

    monkeypatch.setattr(generate, "generate_scene_from_image", scene)
    execute = Mock(
        return_value=dict(status="succeeded", completed=True, all_success=True)
    )
    monkeypatch.setattr(_run_execution, "_execute_deployment", execute)
    with _services(required_color="red") as services:
        if color == "red" and stale_evidence is None:
            assert _run(run_cfg, services=services)["execution_ready"] is True
        else:
            expected = (
                "asset_property_unverified"
                if color is None or stale_evidence is not None
                else "asset_property_mismatch"
            )
            with pytest.raises(ValueError, match=expected):
                _run(run_cfg, services=services)
            role_report = json.loads(
                (run_cfg.output / "scene_role_report.json").read_text()
            )
            assert role_report["issues"][0]["role_id"] == "role_001"
            assert "property:color" not in role_report["roles"]["role_001"]["verified"]
    assert execute.call_count == int(color == "red" and stale_evidence is None)


def test_unstructured_role_clause_is_not_covered_by_another_property(
    run_cfg, monkeypatch
):
    from embodichain.gen_sim.scene_engine.pipeline import generate
    from embodichain.gen_sim import _run_execution

    instruction = "Put the red waterproof cube on the table"
    run_cfg.instruction = instruction

    def scene(image, output, **kwargs):
        _write_scene(output)
        path = output / "scene_export/scene_config.json"
        value = json.loads(path.read_text())
        value["rigid_object"][0]["properties"] = {"color": "red", "waterproof": False}
        path.write_text(json.dumps(value))

    monkeypatch.setattr(generate, "generate_scene_from_image", scene)
    execute = Mock(side_effect=AssertionError("Unverified qualifier executed"))
    monkeypatch.setattr(_run_execution, "_execute_deployment", execute)
    with _services(required_color="red") as services:
        grounding = services.get("task_image_grounding").responses[0]
        grounding["clauses"].append(
            dict(
                id="waterproof",
                quote="waterproof",
                target="role_001",
                origin="explicit",
                resolution="pending_binding",
            )
        )
        services.get("task_semantic_planning").responses[0]["requirements"][0][
            "quote"
        ] = instruction
        with pytest.raises(ValueError, match="unverified_instruction_qualifier"):
            _run(run_cfg, services=services)
    execute.assert_not_called()
    plan = json.loads(
        (run_cfg.output / "task/semantic_graph_validation.json").read_text()
    )
    assert {r["clause_id"]: r["status"] for r in plan["qualifier_coverage"]} == {
        "required_color": "covered",
        "waterproof": "unverified",
    }


def test_annotation_change_after_planning_rejects_binding_before_execution(
    run_cfg: _RunCfg,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline import generate
    from embodichain.gen_sim import _run_execution

    def changed_scene(image, output, **kwargs):
        _write_scene(output)
        root = run_cfg.output / "task"
        path = root / "semantic_graph_validation.json"
        value = json.loads(path.read_text())
        annotation = value["source"]["annotations"]["pick_cube"]
        annotation["intent"]["description"] = "revised after planning"
        for binding in value["source"]["bindings"]:
            if binding["node_id"] == "pick_cube":
                binding["value"] = annotation
        path.write_text(json.dumps(value))
        manifest_path = root / "task_plan_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(generate, "generate_scene_from_image", changed_scene)
    execute = Mock(side_effect=AssertionError("A stale deployment executed"))
    monkeypatch.setattr(_run_execution, "_execute_deployment", execute)
    with (
        _services() as services,
        pytest.raises(ValueError, match="Stale task plan revision"),
    ):
        _run(run_cfg, services=services)
    execute.assert_not_called()
    assert not (run_cfg.output / "deployment").exists()


def _provider_scene(
    root: Path, plan: dict, *, affordances: list[str] | None, properties: dict | None
) -> Path:
    """Use real measured GLB geometry with the v1 provider metadata boundary."""
    _write_scene(root)
    entries = json.loads((root / "scene_export/scene_config.json").read_text())
    package = root / "scene.usdz"
    package.write_bytes(b"provider-package-not-opened-by-static-preflight")
    objects = []
    for entry in [*entries["background"], *entries["rigid_object"]]:
        mesh = root / "scene_export" / entry["shape"]["fpath"]
        item = dict(
            uid=entry["uid"],
            kind="rigid",
            name=entry["name"],
            description=entry["description"],
            category=entry["category"],
            source_prim_path="/Scene/" + entry["uid"],
            mesh_path=mesh.relative_to(root).as_posix(),
            mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
            pose=dict(
                position=entry["init_pos"],
                rotation_euler_xyz_degrees=entry["init_rot"],
                scale=entry["body_scale"],
            ),
            physics=dict(body_type=entry["body_type"], attrs=entry["attrs"]),
        )
        if entry["uid"] == "cube_native":
            if affordances is not None:
                item["affordances"] = affordances
            if properties is not None:
                item["properties"] = properties
        objects.append(item)
    manifest = dict(
        schema_version="embodichain.generated-scene/v1",
        scene_id="provider-scene",
        source={"image_sha256": plan["image_sha256"]},
        requirements_sha256=hashlib.sha256(
            (Path(plan["root"]) / "scene_requirements.json").read_bytes()
        ).hexdigest(),
        coordinates=dict(editable_frame="Y-up", runtime_frame="Z-up", units="m"),
        runtime_asset=dict(
            path=package.name, sha256=hashlib.sha256(package.read_bytes()).hexdigest()
        ),
        objects=objects,
        graph=dict(
            nodes=[dict(object_id=o["uid"], parent_id=None) for o in objects],
            relations=[],
        ),
        artifacts={
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [package, *(root / "scene_export").glob("*.glb")]
        },
    )
    (root / "generated_scene.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize(
    "affordances,color,required_color,expected",
    [
        (None, None, None, "accepted"),
        ([], None, None, "missing_asset_capability"),
        (["graspable", "placeable"], "red", "red", "accepted"),
        (None, "blue", "red", "asset_property_mismatch"),
        (None, None, "red", "asset_property_unverified"),
    ],
)
def test_provider_v1_qualification_preserves_optional_metadata_and_frozen_properties(
    run_cfg: _RunCfg,
    affordances,
    color,
    required_color,
    expected,
) -> None:
    from embodichain.gen_sim.workflow import (
        load_task_plan,
        plan_image_task,
        prepare_task_deployment,
        qualify_task_scene,
    )

    instruction = (
        "Put the cube on the table"
        if required_color is None
        else "Put the red cube on the table"
    )
    with _services(required_color=required_color) as services:
        task = plan_image_task(
            run_cfg.image,
            instruction,
            run_cfg.output.parent / "plan",
            services=services,
        )
        plan = load_task_plan(task)
        scene = _provider_scene(
            run_cfg.output.parent / "provider-scene",
            plan,
            affordances=affordances,
            properties=None if color is None else {"color": color},
        )
        qualification = qualify_task_scene(
            task,
            scene,
            run_cfg.runtime,
            run_cfg.execution_policy,
            run_cfg.output.parent / "qualified",
            services=services,
        )
    report = json.loads((qualification / "scene_qualification.json").read_text())
    assert report["status"] == ("accepted" if expected == "accepted" else "rejected")
    if expected == "accepted":
        assert not report["issues"]
        if affordances is None:
            assert set(report["roles"]["role_001"]["pending"]) == {
                "graspable",
                "placeable",
            }
        deployment = prepare_task_deployment(
            task,
            scene,
            qualification,
            run_cfg.runtime,
            run_cfg.execution_policy,
            run_cfg.output.parent / "prepared",
        )
        assert deployment.is_file()
    else:
        assert report["issues"][0]["code"] == expected


def test_provider_proxy_color_cannot_certify_deployed_usdz(run_cfg: _RunCfg) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdUtils
    from embodichain.gen_sim.workflow import (
        load_generated_scene,
        load_task_plan,
        plan_image_task,
        qualify_task_scene,
    )
    from embodichain.gen_sim._run_binding import (
        _bind_deployment,
        _load_runtime,
        _physical_generated_scene,
    )

    with _services(required_color="red") as services:
        task = plan_image_task(
            run_cfg.image,
            "Put the red cube on the table",
            run_cfg.output.parent / "plan",
            services=services,
        )
        plan = load_task_plan(task)
        scene = _provider_scene(
            run_cfg.output.parent / "provider-scene",
            plan,
            affordances=None,
            properties={"color": "red"},
        )
        manifest_path = scene / "generated_scene.json"
        manifest = json.loads(manifest_path.read_text())
        obj = next(o for o in manifest["objects"] if o["uid"] == "cube_native")
        proxy_path = scene / obj["mesh_path"]
        proxy = trimesh.load_mesh(proxy_path, process=False)
        proxy.visual.vertex_colors = [255, 0, 0, 255]
        proxy.export(proxy_path)
        obj["mesh_sha256"] = hashlib.sha256(proxy_path.read_bytes()).hexdigest()
        obj["property_evidence"] = {
            "color": dict(
                value="red", source=obj["mesh_path"], sha256=obj["mesh_sha256"]
            )
        }
        # A real blue USD prim is loaded while the correctly hashed proxy is red.
        layer = scene / "blue.usda"
        stage = Usd.Stage.CreateNew(str(layer))
        for item in manifest["objects"]:
            cube = UsdGeom.Cube.Define(stage, item["source_prim_path"])
            cube.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])
        stage.GetRootLayer().Save()
        package = scene / manifest["runtime_asset"]["path"]
        package.unlink()
        assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(package))
        manifest["runtime_asset"]["sha256"] = hashlib.sha256(
            package.read_bytes()
        ).hexdigest()
        for relative in manifest["artifacts"]:
            manifest["artifacts"][relative] = hashlib.sha256(
                (scene / relative).read_bytes()
            ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        verified = load_generated_scene(scene)
        physical = _physical_generated_scene(scene, verified, scene / "copied")
        source = physical["rigid_object"][0]["source"]
        loaded = Usd.Stage.Open(source["file_path"])
        assert UsdGeom.Cube(
            loaded.GetPrimAtPath(source["source_prim_path"])
        ).GetDisplayColorAttr().Get()[0] == (0.0, 0.0, 1.0)
        qualification = qualify_task_scene(
            task,
            scene,
            run_cfg.runtime,
            run_cfg.execution_policy,
            run_cfg.output.parent / "qualified",
            services=services,
        )
    report = json.loads((qualification / "scene_qualification.json").read_text())
    assert report["status"] == "rejected"
    assert report["issues"][0]["code"] == "asset_property_unverified"
    assert "property:color" not in report["roles"]["role_001"]["verified"]
    # Binding must recheck the runtime identity even if bindings were preselected.
    run_cfg.output.mkdir()
    runtime, policy, profile = _load_runtime(run_cfg)
    with pytest.raises(ValueError, match="asset_property_unverified"):
        _bind_deployment(
            run_cfg,
            scene,
            task,
            runtime,
            policy,
            profile,
            services=None,
            expected_plan_digest=plan["plan_digest"],
            generated_scene=verified,
            resolved_bindings=report["bindings"],
        )
    assert not (run_cfg.output / "deployment").exists()


def test_attempt_review_retains_per_environment_evidence(tmp_path: Path) -> None:
    from embodichain.gen_sim.task_engine._review import _attempt
    from embodichain.gen_sim.task_spec import canonical_hash

    identity = dict(
        spec_digest="spec",
        graph_digest="graph",
        plan_digest="plan",
        bundle_digest="bundle",
        scene_digest="scene",
        program_digest="program",
        policy_sha256="policy",
        run_id="run",
        attempt_id="attempt",
    )
    event = dict(
        kind="phase_effect_gate_failed",
        message="env 1 release failed",
        timestamp=2.0,
        env_mask=[False, True],
    )
    report = dict(
        status="failed",
        segments=[
            dict(
                name="place",
                success=False,
                failure_reason="release_failed",
                start_step=0,
                end_step=60,
                active=[False, True],
                successes=[False, False],
                start_steps=[0, 5],
                end_steps=[0, 30],
                failure_reasons=[None, "truncated"],
                metadata={"runtime": {"env_ids": [1], "events": [event]}},
            )
        ],
    )
    feedback = dict(
        schema_version="embodichain.attempt-feedback/v1",
        **identity,
        execution="failed",
        physical_verdict="unavailable",
        execution_report_digest=canonical_hash(report),
    )
    for name, value in {
        "run_manifest": identity,
        "attempt_feedback": feedback,
        "execution_report": report,
    }.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    result = _attempt(
        {
            "spec_digest": "spec",
            "semantic_graph_sha256": "graph",
            "plan_digest": "plan",
        },
        tmp_path,
        {"place"},
    )
    row = result["calls"]["place"]
    assert row["env_ids"] == [0, 1]
    assert row["failure_reasons"][1] == "truncated"
    assert (row["start_steps"][1], row["end_steps"][1]) == (5, 30)
    assert row["issues"][0]["env_mask"] == [False, True]


def test_run_configuration_resolves_owner_paths_and_cli_overrides(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = run_cfg.image.parent / "job.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": "embodichain.gensim-run/v1",
                "image": "input.png",
                "runtime": "runtime.yaml",
                "execution_policy": "policy.yaml",
                "instruction": run_cfg.instruction,
                "output": "configured-output",
            }
        )
    )
    consumer = source.parent / "consumer"
    consumer.mkdir()
    monkeypatch.chdir(consumer)
    cfg = _load_run_config(argparse.Namespace(config=source, output=Path("cli-output")))
    assert cfg.image == run_cfg.image
    assert cfg.runtime == run_cfg.runtime
    assert cfg.output == consumer / "cli-output"
    data = yaml.safe_load(source.read_text())
    data["unknown"] = True
    source.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Unsupported"):
        _load_run_config(argparse.Namespace(config=source))


def test_existing_run_is_not_overwritten(run_cfg: _RunCfg) -> None:
    run_cfg.output.mkdir()
    marker = run_cfg.output / "keep.txt"
    marker.write_text("existing run")
    with _services() as services, pytest.raises(FileExistsError):
        _run(run_cfg, services=services)
    assert marker.read_text() == "existing run"


def test_runtime_mesh_handoff_bakes_axis_and_scale_without_changing_source(
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim._run_binding import _physical_scene

    scene = tmp_path / "scene"
    _write_scene(scene)
    export = scene / "scene_export"
    source = export / "table.glb"
    before = source.read_bytes()
    data = json.loads((export / "scene_config.json").read_text())
    data["background"][0]["body_scale"] = [2.0, 3.0, 4.0]
    result = _physical_scene(export, data, tmp_path / "runtime-assets")
    table = result["background"][0]
    mesh = trimesh.load_scene(table["shape"]["fpath"])
    vertices = np.concatenate(
        [geometry.vertices for geometry in mesh.geometry.values()]
    )
    # Y-up scale is applied first; source Y thickness becomes native Z thickness.
    # Native coordinates themselves must be Z-up: a scene-node rotation is
    # insufficient when native graphics and collision loaders differ.
    assert np.ptp(vertices, axis=0) == pytest.approx([1.2, 2.4, 0.12], abs=1e-6)
    assert table["body_scale"] == [1.0, 1.0, 1.0]
    assert source.read_bytes() == before


def test_runtime_mesh_sidecars_preserve_each_assets_texture(tmp_path: Path) -> None:
    from embodichain.gen_sim._run_binding import _physical_scene

    scene = tmp_path / "scene"
    _write_scene(scene)
    export = scene / "scene_export"
    colors = {"table": (255, 0, 0), "cube_native": (0, 0, 255)}
    for uid, color in colors.items():
        box = trimesh.creation.box([0.1, 0.1, 0.1])
        box.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.zeros((len(box.vertices), 2)),
            material=trimesh.visual.material.SimpleMaterial(
                image=Image.new("RGB", (2, 2), color), name="shared_material_name"
            ),
        )
        box.export(export / (uid + ".glb"))
    data = json.loads((export / "scene_config.json").read_text())
    result = _physical_scene(export, data, tmp_path / "native")
    for entry in result["background"] + result["rigid_object"]:
        native = trimesh.load_scene(entry["shape"]["fpath"])
        geometry = next(iter(native.geometry.values()))
        assert (
            geometry.visual.material.image.convert("RGB").getpixel((0, 0))
            == colors[entry["uid"]]
        )


@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
def test_execution_commits_only_success_and_always_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from embodichain.gen_sim._run_execution import _execute_deployment
    from embodichain.lab.gym.envs import demo
    from embodichain.lab.gym.utils import gym_utils, registration
    from embodichain.lab.sim import SimulationManager

    path = tmp_path / "task.yaml"
    path.write_text("id: execution-test\n")
    env = Mock()
    native_cfg = SimpleNamespace(
        sim_cfg=SimpleNamespace(device="cuda:0", headless=False)
    )
    monkeypatch.setattr(gym_utils, "config_to_cfg", lambda *args, **kwargs: native_cfg)
    monkeypatch.setattr(registration, "discover_task_packages", lambda: None)
    monkeypatch.setattr(registration, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        registration,
        "get_env_spec",
        lambda _: SimpleNamespace(make=lambda **kwargs: env),
    )
    cleanup = Mock()
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", cleanup)
    result = SimpleNamespace(
        completed=outcome == "success",
        all_success=outcome == "success",
        to_metadata=lambda: {},
    )
    execute = Mock(
        return_value=result,
        side_effect=(
            RuntimeError("execution failed") if outcome == "exception" else None
        ),
    )
    monkeypatch.setattr(demo, "execute_demo_episode", execute)
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="execution failed"):
            _execute_deployment(path, device=None, headless=True)
        assert env.reset.call_count == 1
    else:
        report = _execute_deployment(path, device=None, headless=True)
        assert report["all_success"] is (outcome == "success")
        assert env.reset.call_args.kwargs == {
            "options": {"save_data": outcome == "success"}
        }
    assert env.reset.call_args_list[0].kwargs == {"options": {"save_data": False}}
    assert native_cfg.sim_cfg.device == "cuda:0"
    env.close.assert_called_once_with(exit_process=False)
    cleanup.assert_called_once_with()


def test_run_cli_propagates_native_failure_as_nonzero_exit(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.cli.main import main
    from embodichain.gen_sim import _run_config, _workflow

    monkeypatch.setattr(_run_config, "_load_run_config", lambda args: run_cfg)
    monkeypatch.setattr(_workflow, "_run", lambda cfg: {"status": "failed"})
    with pytest.raises(SystemExit) as error:
        main(["gensim", "run", "--config", "run.yaml"])
    assert error.value.code == 1
