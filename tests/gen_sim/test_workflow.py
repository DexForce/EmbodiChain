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


def _selector(reference: str = "") -> dict:
    return dict(
        kind="scene_ref" if reference else "none",
        step_id="",
        reference=reference,
        quantifier="one",
        count=0,
    )


def _step(**changes) -> dict:
    step = dict(
        id="step_01",
        task_type="E1",
        object=_selector("cube"),
        target=_selector("table"),
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
    return {**step, **changes}


def _services(
    *,
    missing_object: bool = False,
    required_color: str | None = None,
    step: dict | None = None,
    grounding: dict | None = None,
    plan: dict | None = None,
) -> GenSimServices:
    default_grounding = {
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
    default_plan = {
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
    grounding = grounding or default_grounding
    native = {"source": None if missing_object else "cube_native", "table": "table"}
    plan = plan or default_plan
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
            "task_interpretation": _ScriptedChat({"steps": [step or _step()]}),
            "task_image_grounding": _ScriptedChat(
                grounding,
                {
                    "objects": {
                        item["id"]: native[item["id"]] for item in grounding["objects"]
                    },
                    "resources": {"default_primary": "primary_manipulator"},
                },
            ),
            "task_semantic_planning": _ScriptedChat(plan or default_plan),
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

    # Releasing above the resting pose is opt-in; cover both settings.
    clearance = 0.03 if success else 0.0
    run_cfg = _RunCfg(
        **{
            **run_cfg.to_dict(),
            "binding": {**run_cfg.binding, "release_clearance": clearance},
        }
    )

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
    # Table is 4 cm thick, cube is 4 cm tall; the owner adds its 1 mm landing
    # gap, and an opt-in release clearance lifts only the release pose. The
    # validator still expects the resting pose.
    assert pose[2, 3] == pytest.approx(0.041 + clearance, abs=1e-7)
    program = json.loads((run_cfg.output / "deployment/program.json").read_text())
    (validator,) = program["program"]["items"][1]["validators"]
    assert validator["displacement"][2] == pytest.approx(0.041, abs=1e-7)


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
        ("red", "external_buffer"),
        ("red", "external_texture"),
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
        elif stale_evidence == "external_buffer":
            import struct

            data = mesh_path.read_bytes()
            size = struct.unpack_from("<I", data, 12)[0]
            document = json.loads(data[20 : 20 + size])
            binary = bytearray(data[28 + size :])
            document["buffers"][0]["uri"] = "colors.bin"
            header = json.dumps(document).encode()
            header += b" " * (-len(header) % 4)
            mesh_path.write_bytes(
                struct.pack(
                    "<4sIIII", b"glTF", 2, 20 + len(header), len(header), 0x4E4F534A
                )
                + header
            )
            # Simulate a pre-fix red proof for this external-buffer GLB.
            proof["sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            buffer = mesh_path.with_name("colors.bin")
            index = document["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]
            accessor = document["accessors"][index]
            view = document["bufferViews"][accessor["bufferView"]]
            start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            for row in range(accessor["count"]):
                offset = start + row * view.get("byteStride", 4)
                binary[offset : offset + 4] = bytes([0, 0, 255, 255])
            buffer.write_bytes(binary)
            assert trimesh.load_mesh(mesh_path, process=False).visual.vertex_colors[
                0
            ].tolist() == [0, 0, 255, 255]
        elif stale_evidence == "external_texture":
            import struct
            from PIL import Image

            textured = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
            textured.visual = trimesh.visual.TextureVisuals(
                uv=np.zeros((len(textured.vertices), 2)),
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[255, 0, 0, 255],
                    baseColorTexture=Image.new("RGB", (2, 2), "blue"),
                ),
            )
            data = textured.export(file_type="glb")
            size = struct.unpack_from("<I", data, 12)[0]
            document = json.loads(data[20 : 20 + size])
            document["images"][0] = {"uri": "blue.png"}
            header = json.dumps(document).encode()
            header += b" " * (-len(header) % 4)
            binary = data[20 + size :]
            mesh_path.write_bytes(
                struct.pack(
                    "<4sIIII",
                    b"glTF",
                    2,
                    20 + len(header) + len(binary),
                    len(header),
                    0x4E4F534A,
                )
                + header
                + binary
            )
            # A pre-fix proof measured the fallback factor while PNG was missing.
            proof["sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            Image.new("RGB", (2, 2), "blue").save(mesh_path.with_name("blue.png"))
            assert (
                trimesh.load_mesh(
                    mesh_path, process=False
                ).visual.material.baseColorTexture
                is not None
            )
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


def _bound_deployment(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch, services: GenSimServices
) -> tuple[dict, dict]:
    """Run planning and binding with a stubbed execution receipt."""
    from embodichain.gen_sim.scene_engine.pipeline import generate
    from embodichain.gen_sim import _run_execution

    monkeypatch.setattr(
        generate,
        "generate_scene_from_image",
        lambda image, output, **kwargs: _write_scene(output),
    )
    monkeypatch.setattr(
        _run_execution,
        "_execute_deployment",
        lambda path, **kwargs: {
            "status": "succeeded",
            "completed": True,
            "all_success": True,
            "effect_assurance": "verified",
        },
    )
    with services:
        _run(run_cfg, services=services)
    root = run_cfg.output / "deployment"
    return (
        json.loads((root / "program.json").read_text()),
        yaml.safe_load((root / "integration.yaml").read_text()),
    )


def _node(node_id: str, call: dict, depends_on: list[str], **semantic) -> dict:
    return dict(
        id=node_id,
        step_id="step_01",
        call=call,
        depends_on=depends_on,
        semantic=dict(object="source", **semantic),
    )


@pytest.mark.parametrize("configured", [False, True])
def test_relative_placement_validators_follow_the_selected_lowerer(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    instruction = "Put the cube right of the table"
    semantic = dict(
        reference="table", relation="right_of", intent={"relation": "right_of"}
    )
    place = dict(
        kind="registered",
        call_id="simulation.place_relative",
        arguments={"object": "source", "reference": "table", "relation": "right_of"},
        resources={},
    )
    plan = {
        "nodes": [
            _node(
                "pick_cube",
                dict(kind="pick", object="source", resources={}),
                [],
                **semantic,
            ),
            _node("place_cube", place, ["pick_cube"], **semantic),
        ],
        "requirements": [
            dict(id="r1", quote=instruction, node_ids=["pick_cube", "place_cube"])
        ],
    }
    route = {
        "object_id": "source",
        "reference_entity_id": "table",
        "relation": "right_of",
        "world_displacement": [0.0, 0.30, 0.005],
    }
    services = (
        {
            "registered_semantic_lowerers": [
                {"kind": "place_relative", "routes": [route]}
            ]
        }
        if configured
        else {}
    )
    run_cfg = _RunCfg(
        **{
            **run_cfg.to_dict(),
            "instruction": instruction,
            "binding": {"runtime_services": services},
        }
    )

    program, integration = _bound_deployment(
        run_cfg,
        monkeypatch,
        _services(step=_step(relation="right_of"), plan=plan),
    )

    validators = program["program"]["items"][1]["validators"]
    (lowerer,) = integration["runtime_services"]["registered_semantic_lowerers"]
    if configured:
        # The configured route owns its target; no measured validator contradicts it.
        assert lowerer["routes"] == [route]
        assert validators == []
    else:
        (validator,) = validators
        assert lowerer["routes"][0]["world_displacement"] == validator["displacement"]


@pytest.mark.parametrize("lift_height_m", [None, 0.10])
def test_held_without_reference_validates_its_terminal_pose(
    run_cfg: _RunCfg, monkeypatch: pytest.MonkeyPatch, lift_height_m: float | None
) -> None:
    intent = {"orientation_goal": "upright", "terminal_behavior": "hold"}
    instruction = "Hold the cube upright"
    if lift_height_m is not None:
        instruction = (
            "Lift the cube vertically by more than 10 centimetres and keep it held"
        )
        intent = {
            "direction": "up",
            "lift_height_m": lift_height_m,
            "terminal_behavior": "hold",
        }
    semantic = dict(
        reference=None,
        relation="none",
        intent=intent,
    )
    hold = dict(
        kind="registered",
        call_id="simulation.move_held_object",
        arguments={"target": "hold_pose"},
        resources={},
    )
    plan = {
        "nodes": [
            _node(
                "pick_cube",
                dict(kind="pick", object="source", resources={}),
                [],
                **semantic,
            ),
            _node("hold_cube", hold, ["pick_cube"], **semantic),
        ],
        "requirements": [
            dict(id="r1", quote=instruction, node_ids=["pick_cube", "hold_cube"])
        ],
    }
    grounding = {
        "objects": [
            dict(
                id="source",
                description="red cube",
                bbox=[100, 200, 300, 500],
                provenance="observed",
            )
        ],
        "bindings": {"step_01.object": ["source"]},
    }
    step = _step(
        task_type="E2" if lift_height_m is None else "E1",
        target=_selector(),
        relation="none",
        **intent,
    )
    run_cfg = _RunCfg(**{**run_cfg.to_dict(), "instruction": instruction})

    program, integration = _bound_deployment(
        run_cfg, monkeypatch, _services(step=step, grounding=grounding, plan=plan)
    )

    # A lift validator would measure from the transport start, after Pick has
    # already lifted the object; the held pose itself is the terminal goal.
    near, *other = program["program"]["items"][1]["validators"]
    assert near["kind"] == "object_near_target"
    assert [item["kind"] for item in other] == (
        ["object_upright"] if lift_height_m is None else []
    )
    (lowerer,) = integration["runtime_services"]["registered_semantic_lowerers"]
    pose = lowerer["routes"][0]["pose"]
    assert program["targets"][near["target"]] == {
        "kind": "pose",
        "position": pose["position"],
        "quaternion_xyzw": pose["quaternion_xyzw"],
    }
    if lift_height_m is not None:
        # Match the initial object origin of the synthetic scene.
        x, y, z = pose["position"]
        assert [x, y] == pytest.approx([0.40, -0.12])
        assert z - near["position_tolerance"] > 0.34 + lift_height_m


def test_run_config_rejects_output_inside_the_reused_task(run_cfg: _RunCfg) -> None:
    task = run_cfg.image.parent / "plan"
    task.mkdir()
    (task / "task_plan_manifest.json").write_text("{}")
    source = run_cfg.image.parent / "job.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": "embodichain.gensim-run/v1",
                "image": "input.png",
                "runtime": "runtime.yaml",
                "execution_policy": "policy.yaml",
                "instruction": run_cfg.instruction,
                "task": "plan",
                "output": "plan/new-run",
            }
        )
    )
    with pytest.raises(ValueError, match="source task must not overlap"):
        _load_run_config(argparse.Namespace(config=source))


@pytest.mark.parametrize("changed", ["instruction", "image"])
def test_reused_task_must_plan_the_run_instruction_and_image(
    run_cfg: _RunCfg, changed: str
) -> None:
    from embodichain.gen_sim.workflow import plan_image_task

    with _services() as services:
        task = plan_image_task(
            run_cfg.image,
            run_cfg.instruction,
            run_cfg.image.parent / "plan",
            task_id=run_cfg.task_id,
            services=services,
        )
    image = run_cfg.image.parent / "other.png"
    Image.new("RGB", (100, 80), "black").save(image)
    change = (
        {"instruction": "Put the cube inside the box"}
        if changed == "instruction"
        else {"image": image}
    )
    cfg = _RunCfg(**{**run_cfg.to_dict(), "task": task, **change})
    with pytest.raises(ValueError, match="different instruction or image"):
        _run(cfg)
    assert not cfg.output.exists()


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


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        ("right_of", [0.0, 0.12, 0.005]),
        ("left_of", [0.0, -0.12, 0.005]),
        ("front_of", [-0.12, 0.0, 0.005]),
        ("on", [0.0, 0.0, 0.105]),
    ],
)
def test_relative_placement_offsets_follow_shared_world_axes(
    tmp_path: Path, relation: str, expected: list[float]
) -> None:
    from embodichain.gen_sim._run_binding import _relative_displacement

    # 0.1 m cubes; the side gap is 0.02 m and the release clearance 0.005 m.
    trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(tmp_path / "cube.glb")
    entry = {"shape": {"fpath": "cube.glb"}, "init_rot": [0, 0, 0]}
    displacement = _relative_displacement(entry, entry, relation, tmp_path)
    assert displacement == pytest.approx(expected, abs=1e-6)


def test_directional_relations_follow_the_source_camera_view(
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim._run_binding import (
        _relation_axes,
        _relative_displacement,
    )

    # The RoboTwin head camera looks along +y with its left along -x, so an
    # instruction's "right of" is world +x and "in front of" is world -y.
    camera = {"left": [-1.0, 0.0, 0.0], "forward": [0.0, 0.6, -0.8]}
    axes = _relation_axes({"camera": camera})
    trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(tmp_path / "cube.glb")
    entry = {"shape": {"fpath": "cube.glb"}, "init_rot": [0, 0, 0]}

    right = _relative_displacement(entry, entry, "right_of", tmp_path, axes=axes)
    front = _relative_displacement(entry, entry, "front_of", tmp_path, axes=axes)
    assert right == pytest.approx([0.12, 0.0, 0.005], abs=1e-6)
    assert front == pytest.approx([0.0, -0.12, 0.005], abs=1e-6)


def _export_z_up(mesh: trimesh.Trimesh, path: Path) -> None:
    """Write a body-frame Z-up mesh as the Y-up GLB a scene export holds."""
    from embodichain.gen_sim.scene_engine.randomization._geometry import (
        _Y_UP_TO_Z_UP,
    )

    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices) @ _Y_UP_TO_Z_UP
    mesh.export(path)


def test_axis_alignment_frame_points_local_z_at_the_narrow_end(
    tmp_path: Path,
) -> None:
    from scipy.spatial.transform import Rotation

    from embodichain.gen_sim._run_binding import _align_body_frame, _z_up_mesh

    # A bottle lying along body x with its narrow neck toward -x.
    body = trimesh.creation.cylinder(radius=0.04, height=0.2)
    neck = trimesh.creation.cylinder(radius=0.012, height=0.06)
    neck.apply_translation([0.0, 0.0, 0.13])
    bottle = trimesh.util.concatenate([body, neck])
    bottle.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])
    )
    _export_z_up(bottle, tmp_path / "bottle.glb")
    entry = {"shape": {"fpath": "bottle.glb"}, "init_rot": [0.0, 0.0, 30.0]}
    # ObjectBaseCfg reads init_rot as extrinsic x-y-z; the aligned frame's
    # pitch is near 90 degrees, where an intrinsic reading would differ.

    def world(value: dict) -> np.ndarray:
        rotation = Rotation.from_euler("xyz", value["init_rot"], degrees=True)
        return rotation.apply(np.asarray(_z_up_mesh(value, tmp_path).vertices))

    before = world(entry)
    _align_body_frame(entry, tmp_path)

    rotation = Rotation.from_euler("xyz", entry["init_rot"], degrees=True)
    neck_direction = Rotation.from_euler("z", 30, degrees=True).apply([-1, 0, 0])
    assert rotation.apply([0, 0, 1]) == pytest.approx(neck_direction, abs=0.02)
    assert world(entry) == pytest.approx(before, abs=1e-6)


@pytest.mark.parametrize(("relation", "gap"), [("on", 0.002), ("above", 0.05)])
def test_held_tool_target_puts_its_wide_end_on_the_reference(
    tmp_path: Path, relation: str, gap: float
) -> None:
    from embodichain.gen_sim._run_binding import _held_contact_pose

    # A hammer along body x: a thin handle and a wide head at +x.
    handle = trimesh.creation.box(extents=[0.2, 0.02, 0.02])
    head = trimesh.creation.box(extents=[0.03, 0.03, 0.1])
    head.apply_translation([0.1, 0.0, 0.04])
    _export_z_up(trimesh.util.concatenate([handle, head]), tmp_path / "hammer.glb")
    trimesh.creation.box(extents=[0.04, 0.04, 0.04]).export(tmp_path / "cube.glb")
    # The hammer lies yawed by 90 degrees, its head toward world +y.
    hammer = {
        "uid": "tool_001",
        "shape": {"fpath": "hammer.glb"},
        "init_pos": [0.0, 0.0, 0.81],
        "init_rot": [0.0, 0.0, 90.0],
    }
    block = {
        "uid": "cube_001",
        "shape": {"fpath": "cube.glb"},
        "init_pos": [0.2, 0.1, 0.76],
        "init_rot": [0.0, 0.0, 30.0],
    }

    pose, displacement = _held_contact_pose(hammer, block, relation, tmp_path)

    assert (pose["kind"], pose["entity_id"]) == ("scene_entity", "cube_001")
    # The head (centered 0.1 along the handle, bottom z -0.01) sits on the block top.
    assert displacement == pytest.approx([0.0, -0.1, 0.03 + gap], abs=3e-3)
    # The runtime replaces the block rotation with world_orientation and then
    # applies relative_pose; the result is the hammer's initial orientation at
    # the displaced position.
    resolved = np.eye(4)
    resolved[:3, :3] = np.reshape(pose["world_orientation"], (3, 3))
    resolved[:3, 3] = block["init_pos"]
    resolved = resolved @ np.reshape(pose["relative_pose"], (4, 4))
    yaw = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]
    assert resolved[:3, :3] == pytest.approx(yaw, abs=1e-6)
    assert resolved[:3, 3] == pytest.approx(
        np.asarray(block["init_pos"]) + displacement, abs=1e-6
    )


def test_upright_hold_turns_the_grasped_side_toward_the_holding_arm(
    tmp_path: Path,
) -> None:
    from scipy.spatial.transform import Rotation

    from embodichain.gen_sim._run_binding import _align_body_frame, _held_contact_pose

    # A bottle lying along world x, its neck toward -x, picked from above by
    # an arm anchored in front-left of it.
    body = trimesh.creation.cylinder(radius=0.04, height=0.2)
    neck = trimesh.creation.cylinder(radius=0.012, height=0.06)
    neck.apply_translation([0.0, 0.0, 0.13])
    bottle = trimesh.util.concatenate([body, neck])
    bottle.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])
    )
    _export_z_up(bottle, tmp_path / "bottle.glb")
    entry = {
        "uid": "bottle_001",
        "shape": {"fpath": "bottle.glb"},
        "init_pos": [0.0, 0.0, 0.78],
        "init_rot": [0.0, 0.0, 0.0],
    }
    _align_body_frame(entry, tmp_path)
    anchor = [0.0, -0.4]

    pose, displacement = _held_contact_pose(
        entry, None, None, tmp_path, upright=True, toward=anchor
    )

    rotation = Rotation.from_quat(pose["quaternion_xyzw"])
    initial = Rotation.from_euler("xyz", entry["init_rot"], degrees=True)
    assert rotation.apply([0, 0, 1]) == pytest.approx([0, 0, 1], abs=1e-6)
    # The side that faced up, where a top grasp holds it, now faces the arm.
    grasped_side = rotation.apply(initial.inv().apply([0, 0, 1]))
    # The length axis is estimated from surface samples.
    assert grasped_side == pytest.approx([0, -1, 0], abs=0.01)
    # Held 10 cm above its initial footprint: the 0.04 m-radius body rested
    # with its bottom at z 0.74, so the upright base, 0.1 m below the body
    # center origin, is at 0.84.
    assert pose["position"][2] - 0.1 == pytest.approx(0.84, abs=0.005)
    assert displacement[2] == pytest.approx(pose["position"][2] - 0.78)

    # Holding it above the table it rests on lifts it in place rather than
    # carrying it to the table center.
    trimesh.creation.box(extents=[1.2, 0.05, 0.7]).export(tmp_path / "table.glb")
    table = {"uid": "table", "shape": {"fpath": "table.glb"}}
    table["init_pos"] = [0.4, 0.2, 0.715]
    pose, _ = _held_contact_pose(
        entry, table, "above", tmp_path, upright=True, toward=anchor
    )
    resolved = (
        np.asarray(table["init_pos"])
        + np.reshape(pose["world_orientation"], (3, 3))
        @ np.reshape(pose["relative_pose"], (4, 4))[:3, 3]
    )
    assert resolved[:2] == pytest.approx([0.0, 0.0], abs=0.01)
    assert resolved[2] - 0.1 == pytest.approx(0.74 + 0.05, abs=0.005)


def test_support_landing_prefers_the_robot_side_when_arms_are_anchored(
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim._run_binding import _placement

    # Y-up exports: a 1.2 x 0.7 m tabletop 5 cm thick, and a 5 cm cube.
    trimesh.creation.box(extents=[1.2, 0.05, 0.7]).export(tmp_path / "table.glb")
    trimesh.creation.box(extents=[0.05, 0.05, 0.05]).export(tmp_path / "cube.glb")
    table = {"shape": {"fpath": "table.glb"}, "init_pos": [0.0, 0.0, 0.715]}
    cube = {"shape": {"fpath": "cube.glb"}, "init_pos": [0.2, 0.1, 0.765]}

    _, centered = _placement(cube, table, tmp_path, [])
    _, front = _placement(cube, table, tmp_path, [], robot_xy=[0.0, -0.42])

    assert centered[:2] == pytest.approx([0.0, 0.0], abs=1e-6)
    # The landing is inset 15 cm from the robot-side rim at y = -0.35.
    assert front[:2] == pytest.approx([0.0, -0.2], abs=1e-6)


def test_resource_anchors_select_the_nearest_arm(run_cfg: _RunCfg) -> None:
    from embodichain.gen_sim._run_binding import _nearest_resource

    anchors = {"left": [-0.3, -0.42], "right": [0.3, -0.42]}
    assert _nearest_resource(anchors, [-0.17, -0.07, 0.74]) == "left"
    assert _nearest_resource(anchors, [0.15, 0.0, 0.74]) == "right"
    with pytest.raises(ValueError, match="resource_anchors"):
        _RunCfg(**{**run_cfg.to_dict(), "binding": {"resource_anchors": {"left": [0]}}})
    with pytest.raises(ValueError, match="release_clearance"):
        _RunCfg(**{**run_cfg.to_dict(), "binding": {"release_clearance": -0.01}})


def test_registered_calls_get_templates_keyed_by_call_id() -> None:
    from embodichain.gen_sim._run_binding import _register_call_options

    options = {"place": {"kind": "place", "hand_interp_steps": 12}}
    monitors: dict = {}
    call = {"kind": "registered", "call_id": "simulation.place_relative"}
    _register_call_options(
        options, monitors, call, "simulation.place_relative", "place", "place"
    )
    _register_call_options(
        options, monitors, {"kind": "pick"}, "pick", "pick", "pick_up"
    )

    assert options["simulation.place_relative"] == options["place"]
    assert options["simulation.place_relative"] is not options["place"]
    assert options["pick"] == {"kind": "pick_up"}
    assert set(monitors) == {"place", "simulation.place_relative", "pick"}


def test_configured_lowerers_take_precedence_over_automatic_ones() -> None:
    from embodichain.gen_sim._run_binding import _with_automatic_lowerers

    route = {"object_id": "a", "reference_entity_id": "b", "relation": "on"}
    configured = {"registered_semantic_lowerers": [{"kind": "place_relative"}]}
    services = _with_automatic_lowerers(configured, [route], ["a", "a"])
    assert services["registered_semantic_lowerers"] == [
        {"kind": "place_relative"},
        {"kind": "axis_align", "object_ids": ["a"]},
    ]
    assert configured == {"registered_semantic_lowerers": [{"kind": "place_relative"}]}
    assert _with_automatic_lowerers({}, [], []) == {}
    held = {"object_id": "a", "target_id": "t", "pose": {"kind": "scene_entity"}}
    assert _with_automatic_lowerers({}, [], [], [held]) == {
        "registered_semantic_lowerers": [{"kind": "move_held_object", "routes": [held]}]
    }


def test_room_adds_visual_shell_and_authored_floor_collider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import embodichain.data
    from embodichain.gen_sim._run_binding import _room_entries

    bundle = tmp_path / "WoodLab" / "wood_lab"
    (bundle / "collision").mkdir(parents=True)
    (bundle / "visual").mkdir()
    (bundle / "wood_lab.usda").write_text("#usda 1.0\n")
    (bundle / "collision" / "floor.usda").write_text(
        "double3 xformOp:translate = (0.7, 0, -0.03)\n"
        "double3 xformOp:scale = (5.6, 5.2, 0.06)\n"
    )
    # A Y-up room shell: 2 m tall, its floor at height 0.
    shell = trimesh.creation.box(extents=[4.0, 2.0, 3.0])
    shell.apply_translation([0.0, 1.0, 0.0])
    shell.export(bundle / "visual" / "room.glb")
    monkeypatch.setattr(
        embodichain.data, "get_data_path", lambda path: str(tmp_path / path)
    )

    room = _room_entries("WoodLab", tmp_path / "assets")

    assert room["floor"]["shape"] == {"shape_type": "Cube", "size": [5.6, 5.2, 0.06]}
    assert room["floor"]["init_pos"] == [0.7, 0.0, -0.03]
    assert room["floor"]["body_type"] == "static"
    # Only the room mesh draws the floor; the collider stays invisible.
    assert room["floor"]["render_enabled"] is False
    mesh = trimesh.load(room["visual"]["shape"]["fpath"], force="mesh")
    assert mesh.bounds[0] == pytest.approx([-2.0, -1.5, 0.0], abs=1e-6)
    assert mesh.bounds[1] == pytest.approx([2.0, 1.5, 2.0], abs=1e-6)


def test_run_config_rejects_invalid_room_and_recording(run_cfg: _RunCfg) -> None:
    with pytest.raises(ValueError, match="room must name"):
        _RunCfg(**{**run_cfg.to_dict(), "room": "wood lab"})
    with pytest.raises(TypeError, match="recording must be a boolean"):
        _RunCfg(**{**run_cfg.to_dict(), "recording": "yes"})


def test_episode_recording_keeps_every_camera_frame_and_state(tmp_path: Path) -> None:
    import torch
    from embodichain.gen_sim._run_execution import _EpisodeRecording

    pose = Mock()
    pose.get_local_pose.return_value = torch.eye(4).unsqueeze(0)
    sim = Mock()
    sim.get_rigid_object_uid_list.return_value = ["cube"]
    sim.get_rigid_object.return_value = pose
    recording = _EpisodeRecording(SimpleNamespace(step_dt=0.04, sim=sim), tmp_path)
    for value in (0, 255):
        recording.observe(
            {
                "sensor": {
                    "cam_head": {
                        "color": torch.full((1, 16, 16, 4), value, dtype=torch.uint8)
                    }
                },
                "robot": {"qpos": torch.zeros(1, 2)},
            }
        )

    manifest = recording.write({"completed": False})

    assert manifest["fps"] == pytest.approx(25.0)
    assert manifest["cameras"]["cam_head"] == {"frames": 2, "width": 16, "height": 16}
    assert (tmp_path / "cam_head.mp4").is_file()
    states = [
        json.loads(line)
        for line in (tmp_path / "states.jsonl").read_text().splitlines()
    ]
    assert [state["control_step"] for state in states] == [0, 1]
    assert states[1]["objects"] == {"cube": [0.0, 0.0, 0.0]}


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


@pytest.mark.parametrize("seen", [True, False])
def test_grounded_hybrid_placement_uses_scene_qualification(run_cfg, seen):
    from embodichain.gen_sim.workflow import (
        load_task_plan,
        qualify_task_scene,
        prepare_task_deployment,
    )
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid

    base = run_cfg.output.parent
    text = "Place the cube on the table."
    roles = ["cube_native", "table"]
    packet = dict(
        image_sha256=hashlib.sha256(run_cfg.image.read_bytes()).hexdigest(),
        grounding=dict(
            bindings={k: [k] for k in roles},
            objects=[
                dict(
                    id=k,
                    description=k,
                    bbox=[100, 100, 200, 200] if seen or k == "table" else None,
                    provenance=(
                        "observed" if seen or k == "table" else "runtime_required"
                    ),
                )
                for k in roles
            ],
        ),
    )
    ground = base / "ground.json"
    ground.write_text(json.dumps(packet))
    op = dict(
        template="E1", object=roles[0], reference=roles[1], relation="on", quote=text
    )
    with GenSimServices(
        GenSimServicesCfg(),
        injected={
            "task_image_grounding": _ScriptedChat(
                {"operations": [op]},
                {
                    o["id"]: {"description": o["description"], "bbox": o["bbox"]}
                    for o in packet["grounding"]["objects"]
                },
            ),
            "task_interpretation": _ScriptedChat({"issues": []}),
        },
    ) as services:
        publish_hybrid(
            text,
            "place",
            run_cfg.image,
            base / "plan",
            services,
            scene_grounding=ground if seen else None,
        )
    plan = load_task_plan(base / "plan")
    scene = _provider_scene(
        base / "provider", plan, affordances=["graspable", "placeable"], properties=None
    )
    q = qualify_task_scene(
        base / "plan",
        scene,
        run_cfg.runtime,
        run_cfg.execution_policy,
        base / "qualification",
    )
    verdict = json.loads((q / "scene_qualification.json").read_text())
    assert verdict["status"] == ("accepted" if seen else "ambiguous")
    if seen:
        path = prepare_task_deployment(
            base / "plan",
            scene,
            q,
            run_cfg.runtime,
            run_cfg.execution_policy,
            base / "prepared",
        )
        assert path.exists()
        manifest = json.loads((base / "prepared/deployment_manifest.json").read_text())
        assert (
            manifest["input_revisions"]["graph_digest"] == plan["semantic_graph_sha256"]
        )
        assert manifest["runtime_evidence_status"] == "not_run"
    else:
        with pytest.raises(ValueError, match="accepted qualification"):
            prepare_task_deployment(
                base / "plan",
                scene,
                q,
                run_cfg.runtime,
                run_cfg.execution_policy,
                base / "prepared",
            )


def test_dual_arm_scene_binding_rejects_resource_alias(monkeypatch):
    from embodichain.gen_sim._run_binding import _ground_ids
    from embodichain.gen_sim.services import compat

    # Even a valid object match must not allow a resolver to alias both arms.
    monkeypatch.setattr(
        compat,
        "structured_call",
        lambda **kw: {
            "objects": {"fruit": "apple"},
            "resources": {"right": "left", "left": "left"},
        },
    )
    with pytest.raises(ValueError, match="distinct declared sides"):
        _ground_ids(
            {"objects": [{"id": "fruit", "provenance": "observed"}]},
            {"transfer": {"object": "fruit"}},
            {
                "nodes": [
                    {"call": {"resources": {"source": "right", "destination": "left"}}}
                ]
            },
            {"apple": {}},
            {"resources": [{"resource_id": "left"}, {"resource_id": "right"}]},
            services=None,
        )


def _dual_runtime(run_cfg):
    runtime = yaml.safe_load(run_cfg.runtime.read_text())
    runtime["embodiment"] = {
        "component": str(
            _ROOT
            / "extensions/embodichain_tasks/configs/components/embodiments/dual_franka_robotiq_arg2f_140_handover.yaml"
        )
    }
    run_cfg.runtime.write_text(yaml.safe_dump(runtime))
    policy = yaml.safe_load(run_cfg.execution_policy.read_text())
    policy["requires"]["embodiment_contract"] = "dual_arm_parallel_gripper"
    run_cfg.execution_policy.write_text(yaml.safe_dump(policy))


def test_legacy_configured_coordinated_target_survives_binding(run_cfg, monkeypatch):
    from embodichain.gen_sim.workflow import plan_image_task, load_task_plan
    from embodichain.gen_sim._run_binding import _bind_deployment, _load_runtime

    _dual_runtime(run_cfg)
    step = _step(task_type="E5", required_arm="none", terminal_behavior="place")
    call = dict(
        kind="registered",
        call_id="simulation.coordinated_transport",
        arguments=dict(object="source", target="tray_target"),
        resources=dict(left="left", right="right"),
    )
    proposed = dict(
        nodes=[
            dict(
                id="carry",
                step_id="step_01",
                call=call,
                depends_on=[],
                semantic=dict(
                    object="source",
                    reference="table",
                    relation="on",
                    intent=dict(relation="on", terminal_behavior="place"),
                ),
            )
        ],
        requirements=[dict(id="source", quote=run_cfg.instruction, node_ids=["carry"])],
    )
    route = dict(
        object_id="source", target_id="tray_target", world_displacement=[0.1, 0.0, 0.0]
    )
    run_cfg.binding = {
        "runtime_services": {
            "registered_semantic_lowerers": [
                dict(kind="coordinated_transport", routes=[route])
            ]
        },
        "profile": {
            "action_options": {
                "simulation.coordinated_transport": dict(
                    kind="coordinated_pickment", release=True
                )
            }
        },
    }
    run_cfg.output.mkdir()
    with _services(step=step, plan=proposed) as services:
        task = plan_image_task(
            run_cfg.image,
            run_cfg.instruction,
            run_cfg.output / "task",
            services=services,
        )
        scene = run_cfg.output / "scene"
        _write_scene(scene)
        from embodichain.gen_sim.services import compat

        monkeypatch.setattr(
            compat,
            "structured_call",
            lambda **kw: dict(
                objects={"source": "cube_native", "table": "table"},
                resources={"left": "left", "right": "right"},
            ),
        )
        runtime, policy, profile = _load_runtime(run_cfg)
        path = _bind_deployment(
            run_cfg,
            scene,
            task,
            runtime,
            policy,
            profile,
            services=services,
            expected_plan_digest=load_task_plan(task)["plan_digest"],
        )
    program = json.loads((path.parent / "program.json").read_text())
    assert (
        program["program"]["items"][0]["steps"]["call"]["arguments"]["target"]
        == "tray_target"
    )
    integration = yaml.safe_load((path.parent / "integration.yaml").read_text())
    assert integration["runtime_services"]["registered_semantic_lowerers"][0][
        "routes"
    ] == [route]


@pytest.mark.parametrize("different_pose", [False, True])
def test_repeated_coordinated_release_has_unique_routes(run_cfg, different_pose):
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid
    from embodichain.gen_sim.workflow import (
        load_task_plan,
        qualify_task_scene,
        prepare_task_deployment,
    )

    _dual_runtime(run_cfg)
    base = run_cfg.output.parent
    text = "Using both arms, place the cube on the table, then repeat once."
    ops = [
        dict(
            template="E5",
            object="cube_native",
            reference="table",
            relation="on",
            quote=text,
        )
    ] * 2
    packet = dict(
        image_sha256=hashlib.sha256(run_cfg.image.read_bytes()).hexdigest(),
        grounding=dict(
            objects=[
                dict(
                    id=k, description=k, bbox=[10, 10, 200, 200], provenance="observed"
                )
                for k in ("cube_native", "table")
            ],
            bindings={k: [k] for k in ("cube_native", "table")},
        ),
    )
    ground = base / "ground.json"
    ground.write_text(json.dumps(packet))
    with GenSimServices(
        GenSimServicesCfg(),
        injected={
            "task_image_grounding": _ScriptedChat(dict(operations=ops)),
            "task_interpretation": _ScriptedChat(dict(issues=[])),
        },
    ) as services:
        publish_hybrid(
            text,
            "repeat",
            run_cfg.image,
            base / "plan",
            services,
            scene_grounding=ground,
        )
    plan = load_task_plan(base / "plan")
    scene = _provider_scene(base / "scene", plan, affordances=None, properties=None)
    q = qualify_task_scene(
        base / "plan",
        scene,
        run_cfg.runtime,
        run_cfg.execution_policy,
        base / "qualification",
    )
    report_path = q / "scene_qualification.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "accepted"
    assert report["placements"]["call_2"] == report["placements"]["call_1"]
    if different_pose:
        # A separately qualified occurrence may choose another valid site.
        report["placements"]["call_2"]["object_target_pose"][3] += 0.05
        report["placements"]["call_2"]["displacement"][0] += 0.05
        report_path.write_text(json.dumps(report))
    path = prepare_task_deployment(
        base / "plan",
        scene,
        q,
        run_cfg.runtime,
        run_cfg.execution_policy,
        base / "deployment",
    )
    integration = yaml.safe_load((path.parent / "integration.yaml").read_text())
    routes = integration["runtime_services"]["registered_semantic_lowerers"][0][
        "routes"
    ]
    assert len(routes) == (2 if different_pose else 1)
    selectors = {(r["object_id"], r["target_id"]) for r in routes}
    assert len(selectors) == len(routes)
