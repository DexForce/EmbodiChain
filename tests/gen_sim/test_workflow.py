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


def _services(*, missing_object: bool = False) -> GenSimServices:
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
        assert program.instruction == run_cfg.instruction
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
    assert manifest["execution_ready"] is True
    assert manifest["status"] == ("succeeded" if success else "failed")
    assert manifest["runtime_evidence_status"] == manifest["status"]
    assert manifest["task_spec_verdict"] == "not_evaluated"
    receipt = json.loads((run_cfg.output / "execution_report.json").read_text())
    assert receipt["all_success"] is success
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
