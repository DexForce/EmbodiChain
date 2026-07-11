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

import ast
from contextlib import nullcontext
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    DEFAULT_TARGET_BODY_SCALE,
)
from embodichain.gen_sim.action_agent_pipeline.gym_project_api.image2tabletop_client import (
    _require_server,
    wait_for_job,
)

_ACTION_AGENT_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain/gen_sim/action_agent_pipeline"
)


def test_image2tabletop_wait_for_job_times_out_without_polling() -> None:
    with pytest.raises(TimeoutError, match="did not complete"):
        wait_for_job("http://example.test", "job-1", poll_interval=0.1, timeout_s=0.0)


def test_image2tabletop_server_requires_http_scheme() -> None:
    with pytest.raises(ValueError, match="http\\(s\\) URL"):
        _require_server("localhost:4523")


def test_action_agent_config_cli_imports() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import (
        generate_action_agent_config,
    )

    assert callable(generate_action_agent_config.cli)


def test_run_agent_reset_randomization_is_disabled_for_single_env() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import run_agent

    gym_config = {"num_envs": 1, "rigid_object": [{"uid": "apple"}]}

    run_agent._add_vectorized_reset_randomization(gym_config)

    assert "env" not in gym_config


def test_run_agent_reset_randomization_configures_parallel_envs() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import run_agent

    gym_config = {
        "num_envs": 4,
        "rigid_object": [{"uid": "apple"}],
        "env": {
            "dataset": {
                "recorder": {"func": "record_camera_data", "params": {}},
                "metadata": {"task": "demo"},
            }
        },
    }

    run_agent._add_vectorized_reset_randomization(gym_config)

    assert gym_config["env"]["dataset"] == {"metadata": {"task": "demo"}}
    events = gym_config["env"]["events"]
    assert events["init_apple_pose"]["params"] == {
        "entity_cfg": {"uid": "apple"},
        "position_range": [[-0.04, -0.04, 0.0], [0.04, 0.04, 0.0]],
        "rotation_range": [[0.0, 0.0, -45.0], [0.0, 0.0, 45.0]],
        "relative_position": True,
    }
    assert events["random_table_height"]["params"] == {
        "anchor_uid": "table",
        "height_delta_range": [[-0.05], [0.05]],
    }


def test_generate_config_cli_auto_applies_prompt2scene_alignment(
    monkeypatch,
    tmp_path,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import (
        generate_action_agent_config,
    )
    from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
        DEFAULT_PROMPT2SCENE_MESH_X_ROTATION_DEGREES,
        DEFAULT_PROMPT2SCENE_SCENE_Z_ROTATION_DEGREES,
    )

    gym_export = tmp_path / "prompt2scene" / "demo111" / "gym_export"
    (gym_export / "scene_state").mkdir(parents=True)
    (gym_export / "scene_state" / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    output_dir = tmp_path / "configs" / "demo111"
    captured = {}

    def fake_generate_action_agent_config_from_project(**kwargs):
        captured.update(kwargs)
        return _fake_generated_config_paths(output_dir)

    monkeypatch.setattr(
        generate_action_agent_config,
        "generate_action_agent_config_from_project",
        fake_generate_action_agent_config_from_project,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_action_agent_config",
            "--gym_project",
            str(gym_export),
            "--output_dir",
            str(output_dir),
            "--task_name",
            "Demo111",
            "--task_description",
            "用双臂把两边的东西放到篮子里",
            "--robot-profile",
            "franka",
            "--target_body_scale",
            "1.0",
            "--overwrite",
        ],
    )

    generate_action_agent_config.cli()

    assert captured["robot_profile"] == "franka"
    assert captured["preserve_source_scene_geometry"] is True
    assert captured["load_source_meshes_directly"] is True
    assert captured["source_scene_z_rotation_degrees"] == (
        DEFAULT_PROMPT2SCENE_SCENE_Z_ROTATION_DEGREES
    )
    assert captured["source_mesh_x_rotation_degrees"] == (
        DEFAULT_PROMPT2SCENE_MESH_X_ROTATION_DEGREES
    )


def test_generate_config_cli_defaults_to_prompt2scene_scale_multiplier(
    monkeypatch,
    tmp_path,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import (
        generate_action_agent_config,
    )

    gym_export = tmp_path / "prompt2scene" / "demo131" / "gym_export"
    (gym_export / "scene_state").mkdir(parents=True)
    (gym_export / "scene_state" / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    output_dir = tmp_path / "configs" / "demo131"
    captured = {}

    def fake_generate_action_agent_config_from_project(**kwargs):
        captured.update(kwargs)
        return _fake_generated_config_paths(output_dir)

    monkeypatch.setattr(
        generate_action_agent_config,
        "generate_action_agent_config_from_project",
        fake_generate_action_agent_config_from_project,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_action_agent_config",
            "--gym_project",
            str(gym_export),
            "--output_dir",
            str(output_dir),
            "--task_name",
            "Demo131",
            "--task_description",
            "用左臂把倒下的瓶子扶正",
            "--overwrite",
        ],
    )

    generate_action_agent_config.cli()

    assert captured["target_body_scale"] == DEFAULT_TARGET_BODY_SCALE
    assert captured["source_scene_body_scale_mode"] == "multiply"


def test_generate_config_cli_respects_explicit_prompt2scene_alignment_overrides(
    monkeypatch,
    tmp_path,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import (
        generate_action_agent_config,
    )

    gym_export = tmp_path / "prompt2scene" / "demo111" / "gym_export"
    (gym_export / "scene_state").mkdir(parents=True)
    (gym_export / "scene_state" / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    output_dir = tmp_path / "configs" / "demo111"
    captured = {}

    def fake_generate_action_agent_config_from_project(**kwargs):
        captured.update(kwargs)
        return _fake_generated_config_paths(output_dir)

    monkeypatch.setattr(
        generate_action_agent_config,
        "generate_action_agent_config_from_project",
        fake_generate_action_agent_config_from_project,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_action_agent_config",
            "--gym_project",
            str(gym_export),
            "--output_dir",
            str(output_dir),
            "--source_scene_z_rotation_degrees",
            "0",
            "--source_mesh_x_rotation_degrees",
            "0",
            "--no-preserve-source-scene-geometry",
            "--no-load-source-meshes-directly",
        ],
    )

    generate_action_agent_config.cli()

    assert captured["preserve_source_scene_geometry"] is False
    assert captured["load_source_meshes_directly"] is False
    assert captured["source_scene_z_rotation_degrees"] == 0.0
    assert captured["source_mesh_x_rotation_degrees"] == 0.0


def _fake_generated_config_paths(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        gym_config=output_dir / "fast_gym_config.json",
        agent_config=output_dir / "agent_config.json",
        task_prompt=output_dir / "task_prompt.txt",
        task_graph=output_dir / "task_graph.json",
        basic_background=output_dir / "basic_background.txt",
        atom_actions=output_dir / "atom_actions.txt",
        summary=None,
    )


def test_action_agent_config_generation_imports() -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation import action_agent_config

    assert callable(action_agent_config.generate_action_agent_config_from_project)
    assert action_agent_config.GeneratedActionAgentConfigPaths.__name__ == (
        "GeneratedActionAgentConfigPaths"
    )


def test_source_scene_xy_positions_track_body_scale_multiplier() -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
        _maybe_apply_source_scene_xy_scale,
    )
    from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
        _SceneObject,
    )

    table_anchor = [0.1, -0.2, 0.5]
    gym_config = {
        "background": [
            {
                "uid": "table",
                "init_pos": list(table_anchor),
                "body_scale": [1.3, 1.3, 1.3],
            }
        ],
        "rigid_object": [
            {
                "uid": "cup",
                "init_pos": [0.4, 0.0, 0.9],
                "body_scale": [1.3, 1.3, 1.3],
            },
            {
                "uid": "wide",
                "init_pos": [-0.2, 0.2, 0.8],
                "body_scale": [2.0, 0.5, 1.0],
            },
        ],
    }
    source_objects_by_runtime_uid = {
        "table": _SceneObject(
            source_uid="table_0",
            source_role="background",
            config={"body_scale": [1.0, 1.0, 1.0]},
        ),
        "cup": _SceneObject(
            source_uid="cup_0",
            source_role="rigid_object",
            config={"body_scale": [1.0, 1.0, 1.0]},
        ),
        "wide": _SceneObject(
            source_uid="wide_0",
            source_role="rigid_object",
            config={"body_scale": [1.0, 1.0, 1.0]},
        ),
    }

    _maybe_apply_source_scene_xy_scale(
        gym_config,
        source_objects_by_runtime_uid,
        source_scene_body_scale_mode="multiply",
    )

    assert gym_config["background"][0]["init_pos"] == pytest.approx(table_anchor)
    assert gym_config["rigid_object"][0]["init_pos"] == pytest.approx([0.49, 0.06, 0.9])
    assert gym_config["rigid_object"][1]["init_pos"] == pytest.approx([-0.5, 0.0, 0.8])


def test_source_scene_xy_positions_preserve_mode_stays_unchanged() -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
        _maybe_apply_source_scene_xy_scale,
    )
    from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
        _SceneObject,
    )

    original_position = [0.4, 0.0, 0.9]
    gym_config = {
        "rigid_object": [
            {
                "uid": "cup",
                "init_pos": list(original_position),
                "body_scale": [1.3, 1.3, 1.3],
            }
        ]
    }

    _maybe_apply_source_scene_xy_scale(
        gym_config,
        {
            "cup": _SceneObject(
                source_uid="cup_0",
                source_role="rigid_object",
                config={"body_scale": [1.0, 1.0, 1.0]},
            )
        },
        source_scene_body_scale_mode="preserve",
    )

    assert gym_config["rigid_object"][0]["init_pos"] == original_position


def test_prompt2scene_prompt_is_independent_from_task_description() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import build_parser

    args = build_parser().parse_args(
        [
            "--use-prompt2scene",
            "--prompt2scene-prompt",
            "move the can left",
            "--task_description",
            "put the can into the pot",
        ]
    )

    assert args.prompt2scene_prompt == "move the can left"
    assert args.task_description == "put the can into the pot"


def test_pipeline_parser_accepts_headless() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import build_parser

    args = build_parser().parse_args(["--headless"])

    assert args.headless is True


def test_pipeline_parser_defaults_to_target_body_scale() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import build_parser

    args = build_parser().parse_args([])

    assert args.target_body_scale == DEFAULT_TARGET_BODY_SCALE
    assert args.surface_release_clearance == DEFAULT_SURFACE_RELEASE_CLEARANCE


def test_pipeline_parser_accepts_surface_release_clearance() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import build_parser

    args = build_parser().parse_args(["--surface-release-clearance", "0.05"])

    assert args.surface_release_clearance == pytest.approx(0.05)


def test_run_agent_command_passes_headless(monkeypatch, tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import agent_run_stage

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(agent_run_stage.subprocess, "run", fake_run)

    return_code = agent_run_stage.run_agent_command(
        task_name="Demo111",
        gym_config=tmp_path / "fast_gym_config.json",
        agent_config=tmp_path / "agent_config.json",
        regenerate=True,
        headless=True,
    )

    assert return_code == 0
    assert "--headless" in captured["command"]
    assert "--regenerate" in captured["command"]
    assert captured["kwargs"]["check"] is False


def test_action_agent_python_modules_declare_all() -> None:
    missing_all = []
    for path in sorted(_ACTION_AGENT_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        has_all = any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and (
                any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in getattr(node, "targets", [])
                )
                or (
                    isinstance(getattr(node, "target", None), ast.Name)
                    and node.target.id == "__all__"
                )
            )
            for node in tree.body
        )
        if not has_all:
            missing_all.append(path.relative_to(_ACTION_AGENT_PACKAGE_ROOT).as_posix())

    assert missing_all == []


def test_glb_io_is_shared_by_generation_modules() -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation import (
        action_agent_config,
        glb_io,
        mesh_frame_normalization,
    )

    assert action_agent_config.read_glb is glb_io.read_glb
    assert mesh_frame_normalization.read_glb is glb_io.read_glb


def test_gltf_normalized_accessor_components_are_scaled() -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
        _iter_gltf_accessor_vec3,
    )

    doc = {
        "bufferViews": [{"buffer": 0, "byteOffset": 0}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5121,
                "type": "VEC3",
                "count": 2,
                "normalized": True,
            }
        ],
    }
    vertices = list(
        _iter_gltf_accessor_vec3(
            doc,
            bytes([0, 127, 255, 255, 0, 127]),
            0,
        )
    )

    assert vertices[0] == pytest.approx((0.0, 127.0 / 255.0, 1.0))
    assert vertices[1] == pytest.approx((1.0, 0.0, 127.0 / 255.0))


def test_prompt2geometry_cleanup_only_removes_new_known_outputs(tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.prompt2geometry.pipeline import (
        _cleanup_output_root,
        _snapshot_output_root_entries,
    )

    output_root = tmp_path / "prompt2geometry"
    output_root.mkdir()
    preexisting_dir = output_root / "zimage"
    preexisting_dir.mkdir()
    (preexisting_dir / "keep.txt").write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_link = output_root / "sam3d"
    outside_link.symlink_to(outside, target_is_directory=True)
    preexisting = _snapshot_output_root_entries(output_root)

    request_path = output_root / "prompt2geometry_request.json"
    request_path.write_text("{}", encoding="utf-8")
    result_path = output_root / "apple.glb"
    result_path.write_text("glb", encoding="utf-8")

    _cleanup_output_root(
        output_root,
        keep_path=result_path,
        preexisting_paths=preexisting,
    )

    assert result_path.is_file()
    assert (preexisting_dir / "keep.txt").is_file()
    assert outside.is_dir()
    assert outside_link.is_symlink()
    assert not request_path.exists()


def test_dimension_estimation_uses_finite_retry(monkeypatch) -> None:
    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.prompt2geometry.dimensions import (
        estimate_real_dimensions,
    )

    class BadClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, *, messages):
            self.calls += 1
            return {"length_m": -1}

    client = BadClient()
    monkeypatch.setattr(
        "embodichain.gen_sim.action_agent_pipeline.gym_project_api.prompt2geometry.dimensions.time.sleep",
        lambda _: None,
    )

    with pytest.raises(ValueError, match="after 3 attempts"):
        estimate_real_dimensions(object_prompt="apple", client=client)

    assert client.calls == 3


def test_prompt2geometry_config_does_not_stringify_none_llm_values(
    monkeypatch,
    tmp_path,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.prompt2geometry import (
        config as prompt2geometry_config,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "services": {
                    "zimage": {"base_url": "http://zimage.test"},
                    "sam3": {"base_url": "http://sam3.test"},
                    "sam3d": {"base_url": "http://sam3d.test"},
                },
                "llm": {
                    "openai_compatible": {
                        "api_key": None,
                        "model": None,
                        "base_url": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prompt2geometry_config,
        "get_openai_compatible_llm_config",
        lambda **kwargs: {},
    )

    cfg = prompt2geometry_config.load_prompt2geometry_config(config_path)

    assert cfg.llm_api_key == ""
    assert cfg.llm_model == ""
    assert cfg.llm_base_url == ""


def test_coacd_cache_generation_uses_obj_suffixed_temp_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation import coacd_cache

    captured = {}
    open3d_module = ModuleType("open3d")
    open3d_module.t = SimpleNamespace(
        io=SimpleNamespace(read_triangle_mesh=lambda path: "mesh")
    )
    meshproc_module = ModuleType("dexsim.kit.meshproc")
    meshproc_module.convex_decomposition_coacd = lambda mesh, max_convex_hull_num: (
        True,
        ["convex-part"],
    )
    utility_module = ModuleType("dexsim.kit.meshproc.utility")

    def fake_mesh_list_to_file(path, mesh_list):
        captured["temp_path"] = Path(path)
        captured["mesh_list"] = mesh_list
        Path(path).write_text("cache", encoding="utf-8")

    utility_module.mesh_list_to_file = fake_mesh_list_to_file
    monkeypatch.setitem(sys.modules, "open3d", open3d_module)
    monkeypatch.setitem(sys.modules, "dexsim", ModuleType("dexsim"))
    monkeypatch.setitem(sys.modules, "dexsim.kit", ModuleType("dexsim.kit"))
    monkeypatch.setitem(sys.modules, "dexsim.kit.meshproc", meshproc_module)
    monkeypatch.setitem(sys.modules, "dexsim.kit.meshproc.utility", utility_module)

    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    cache_path = tmp_path / "cache_16.obj"

    coacd_cache._generate_coacd_cache(mesh_path, cache_path, 16)

    assert captured["temp_path"].suffix == ".obj"
    assert captured["temp_path"].name.startswith("cache_16.tmp.")
    assert captured["mesh_list"] == ["convex-part"]
    assert cache_path.read_text(encoding="utf-8") == "cache"
    assert not captured["temp_path"].exists()


def test_create_openai_client_uses_per_client_proxy(monkeypatch) -> None:
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_openai_client,
    )

    class _FakeHttpClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeOpenAI:
        last_kwargs = None

        def __init__(self, **kwargs) -> None:
            _FakeOpenAI.last_kwargs = kwargs

    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=_FakeHttpClient))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    create_openai_client(
        config={
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "proxy_url": "http://proxy.test:8080",
        }
    )

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    http_client = _FakeOpenAI.last_kwargs["http_client"]
    assert http_client.kwargs == {
        "proxy": "http://proxy.test:8080",
        "trust_env": False,
    }


def test_create_chat_openai_uses_per_client_proxy(monkeypatch) -> None:
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import create_chat_openai

    class _FakeHttpClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=_FakeHttpClient))
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=_FakeChatOpenAI),
    )

    chat_model = create_chat_openai(
        config={
            "api_key": "test-key",
            "model": "test-model",
            "proxy_url": "http://proxy.test:8080",
        },
        usage_stage="test",
    )

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    http_client = chat_model._inner.kwargs["http_client"]
    assert http_client.kwargs == {
        "proxy": "http://proxy.test:8080",
        "trust_env": False,
    }


def test_image2scene_pipeline_passes_client_url(monkeypatch, tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.image2scene_stage import (
        run_image2scene_pipeline,
    )

    root = tmp_path / "image2scene"
    script = root / "demo_api/client/image2scene_pipeline.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    image = tmp_path / "demo.jpg"
    image.write_bytes(b"image")
    gen_config = root / "gen_config.json"
    gen_config.write_text(
        json.dumps(
            {
                "DEFAULT_TABLE_TYPE": "",
                "DEFAULT_API_KEY": "key",
                "DEFAULT_MODEL": "model",
                "DEFAULT_BASE_URL": "https://llm.test/v1",
                "DEFAULT_CLIENT_URL": "",
            }
        ),
        encoding="utf-8",
    )
    merged_output = root / "merged.json"
    merged_output.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "embodichain.gen_sim.action_agent_pipeline.cli.image2scene_stage.subprocess.run",
        fake_run,
    )

    run_image2scene_pipeline(
        SimpleNamespace(
            background="a vase",
            image2scene_root=str(root),
            image=str(image),
            image_name=None,
            image2scene_download_dir="./downloads",
            image2scene_output_root="./generated",
            image2scene_gen_config="./gen_config.json",
            image2scene_llm_config="./gen_config.json",
            image2scene_extract_dir=None,
            image2scene_merged_output="./merged.json",
            server="http://stage-a.test:4523",
            image2scene_client_url=None,
            poll_interval=0.1,
        )
    )

    command = captured["command"]
    gen_config_index = command.index("--gen-config") + 1
    runtime_config = Path(command[gen_config_index])
    assert runtime_config != gen_config
    assert runtime_config.parent.name == ".image2scene_runtime"
    assert (
        json.loads(runtime_config.read_text(encoding="utf-8"))["DEFAULT_CLIENT_URL"]
        == "http://stage-a.test:4523"
    )


def test_image2scene_runtime_gen_config_injects_client_url(tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.image2scene_stage import (
        _stage_b_gen_config_with_client_url,
    )

    gen_config = tmp_path / "gen_config.json"
    gen_config.write_text(
        json.dumps(
            {
                "DEFAULT_TABLE_TYPE": "",
                "DEFAULT_API_KEY": "key",
                "DEFAULT_MODEL": "model",
                "DEFAULT_BASE_URL": "https://llm.test/v1",
                "DEFAULT_CLIENT_URL": "",
            }
        ),
        encoding="utf-8",
    )

    runtime_config = _stage_b_gen_config_with_client_url(
        gen_config,
        "http://mesatask.test:4523/",
        tmp_path,
    )

    assert runtime_config != gen_config
    assert (
        json.loads(gen_config.read_text(encoding="utf-8"))["DEFAULT_CLIENT_URL"] == ""
    )
    assert (
        json.loads(runtime_config.read_text(encoding="utf-8"))["DEFAULT_CLIENT_URL"]
        == "http://mesatask.test:4523"
    )


def test_prompt2scene_stage_returns_exported_gym_config(monkeypatch, tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import prompt2scene_stage

    output_root = tmp_path / "prompt2scene"
    llm_config = tmp_path / "llm_config.json"
    image_path = tmp_path / "scene.jpg"
    llm_config.write_text("{}", encoding="utf-8")
    image_path.write_bytes(b"fake image")
    gym_config = output_root / "gym_export/gym_config.json"
    captured = {}

    def fake_run_prompt2scene(request, *, llm_cfg):
        captured["request"] = request
        captured["llm_cfg"] = llm_cfg
        gym_config.parent.mkdir(parents=True)
        gym_config.write_text("{}", encoding="utf-8")
        return SimpleNamespace(gym_config_path=gym_config)

    class FakePrompt2SceneInput:
        @classmethod
        def from_cli_args(cls, *, image_path, prompt, output_root, gravity_settle_mode):
            return SimpleNamespace(
                image_path=image_path,
                prompt=prompt,
                gravity_settle_mode=gravity_settle_mode,
                output_root=output_root.expanduser().resolve(),
            )

    monkeypatch.setattr(
        prompt2scene_stage,
        "_load_prompt2scene_components",
        lambda: (fake_run_prompt2scene, FakePrompt2SceneInput),
    )
    monkeypatch.setattr(
        prompt2scene_stage,
        "build_prompt2scene_llm_config",
        lambda path: captured.update(llm_config_path=path) or "llm-cfg",
    )
    monkeypatch.setattr(
        prompt2scene_stage,
        "write_prompt2scene_client_config",
        lambda _: tmp_path / "prompt2scene_client_config.json",
    )
    monkeypatch.setattr(
        prompt2scene_stage,
        "use_prompt2scene_client_config",
        lambda _: nullcontext(),
    )

    result = prompt2scene_stage.run_prompt2scene_stage(
        SimpleNamespace(
            prompt2scene_text=None,
            prompt2scene_prompt="move the can left",
            prompt2scene_output_root=str(output_root),
            prompt2scene_llm_config=str(llm_config),
            prompt2scene_gravity_settle_mode="physics",
            image=str(image_path),
            image_name=None,
        )
    )

    assert result == gym_config
    assert captured["llm_config_path"] == llm_config
    assert captured["llm_cfg"] == "llm-cfg"
    assert captured["request"].image_path == image_path.resolve()
    assert captured["request"].prompt == "move the can left"
    assert captured["request"].gravity_settle_mode == "physics"
    assert captured["request"].output_root == output_root.resolve()


def test_prompt2scene_stage_edit_only_does_not_use_default_image(
    monkeypatch,
    tmp_path,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import prompt2scene_stage

    output_root = tmp_path / "prompt2scene"
    gym_config = output_root / "gym_export/gym_config.json"
    gym_config.parent.mkdir(parents=True)
    gym_config.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_run_prompt2scene(request, *, llm_cfg):
        captured["request"] = request
        captured["llm_cfg"] = llm_cfg
        return SimpleNamespace(gym_config_path=None)

    class FakePrompt2SceneInput:
        @classmethod
        def from_cli_args(cls, *, image_path, prompt, output_root, gravity_settle_mode):
            return SimpleNamespace(
                image_path=image_path,
                prompt=prompt,
                gravity_settle_mode=gravity_settle_mode,
                output_root=output_root.expanduser().resolve(),
            )

    monkeypatch.setattr(
        prompt2scene_stage,
        "_load_prompt2scene_components",
        lambda: (fake_run_prompt2scene, FakePrompt2SceneInput),
    )
    monkeypatch.setattr(
        prompt2scene_stage,
        "build_prompt2scene_llm_config",
        lambda path: captured.update(llm_config_path=path) or "llm-cfg",
    )
    monkeypatch.setattr(
        prompt2scene_stage,
        "write_prompt2scene_client_config",
        lambda _: tmp_path / "prompt2scene_client_config.json",
    )
    monkeypatch.setattr(
        prompt2scene_stage,
        "use_prompt2scene_client_config",
        lambda _: nullcontext(),
    )

    result = prompt2scene_stage.run_prompt2scene_stage(
        SimpleNamespace(
            prompt2scene_text=None,
            prompt2scene_prompt="move the can left",
            prompt2scene_output_root=str(output_root),
            prompt2scene_llm_config=None,
            prompt2scene_gravity_settle_mode="geometry",
            image=None,
            image_name=None,
        )
    )

    assert result == gym_config
    assert captured["llm_config_path"] is None
    assert captured["llm_cfg"] == "llm-cfg"
    assert captured["request"].image_path is None
    assert captured["request"].prompt == "move the can left"
    assert captured["request"].output_root == output_root.resolve()


def test_prompt2scene_stage_rejects_text_input(tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import prompt2scene_stage

    with pytest.raises(ValueError, match="--prompt2scene-prompt"):
        prompt2scene_stage.run_prompt2scene_stage(
            SimpleNamespace(
                prompt2scene_text="a tabletop scene with bread and a basket",
                prompt2scene_prompt=None,
                prompt2scene_output_root=str(tmp_path / "prompt2scene"),
                prompt2scene_llm_config=None,
                prompt2scene_gravity_settle_mode="geometry",
                image=None,
                image_name=None,
            )
        )


def test_prompt2scene_source_record_includes_request_fields(tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_records import (
        build_pipeline_record,
    )

    repo_root = tmp_path
    source_dir = repo_root / "gym_project/prompt2scene/demo/gym_export"
    source_dir.mkdir(parents=True)
    source_config = source_dir / "gym_config.json"
    source_config.write_text("{}", encoding="utf-8")
    output_dir = repo_root / "gym_project/action_agent_pipeline/configs/demo"
    generated_gym_config = output_dir / "fast_gym_config.json"
    generated_agent_config = output_dir / "agent_config.json"
    task_prompt = output_dir / "task_prompt.txt"
    basic_background = output_dir / "basic_background.txt"
    atom_actions = output_dir / "atom_actions.txt"
    for path in (
        generated_gym_config,
        generated_agent_config,
        task_prompt,
        basic_background,
        atom_actions,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    record = build_pipeline_record(
        args=SimpleNamespace(
            task_name="Demo_Text",
            task_description="place bread into basket",
            image_name=None,
            image=None,
            use_image2scene=False,
            prompt2scene_output_root=str(repo_root / "gym_project/prompt2scene/demo"),
            prompt2scene_llm_config=str(
                repo_root / "embodichain/gen_sim/prompt2scene/configs/llm_config.json"
            ),
            prompt2scene_text=None,
            prompt2scene_prompt="move the bread left",
            prompt2scene_gravity_settle_mode="physics",
            prompt2scene_scene_z_rotation_degrees=-90.0,
            target_body_scale=0.8,
            target_body_scale_mode="multiply",
            inside_container_slot_distance_scale=1.0,
            surface_release_clearance=0.05,
            target_replacement1=None,
            target_replacement2=None,
            sync_replacement_names=False,
            reuse_target_replacements=True,
            acd_method="vhacd",
            overwrite_config=True,
            regenerate=True,
            skip_run_agent=False,
            headless=True,
        ),
        resolution=SimpleNamespace(path=source_config, mode="prompt2scene"),
        generated_paths=SimpleNamespace(
            output_dir=output_dir,
            gym_config=generated_gym_config,
            agent_config=generated_agent_config,
            task_prompt=task_prompt,
            basic_background=basic_background,
            atom_actions=atom_actions,
            summary={},
        ),
        history_path=repo_root / "history.json",
        target_replacements=[],
        repo_root=repo_root,
        schema_version=1,
    )

    assert record["source_mode"] == "prompt2scene"
    assert record["source_gym_config"] == (
        "gym_project/prompt2scene/demo/gym_export/gym_config.json"
    )
    assert record["prompt2scene_output_root"] == "gym_project/prompt2scene/demo"
    assert record["prompt2scene_llm_config"] == (
        "embodichain/gen_sim/prompt2scene/configs/llm_config.json"
    )
    assert "prompt2scene_text" not in record
    assert record["prompt2scene_prompt"] == "move the bread left"
    assert "prompt2scene_existing_gym_project" not in record
    assert record["prompt2scene_gravity_settle_mode"] == "physics"
    assert record["prompt2scene_scene_z_rotation_degrees"] == -90.0
    assert "prompt2scene_mesh_x_rotation_degrees" not in record
    assert record["target_body_scale_mode"] == "multiply"
    assert record["surface_release_clearance"] == pytest.approx(0.05)
    assert record["acd_method"] == "vhacd"
    assert record["headless"] is True


@pytest.mark.parametrize(
    "path_kind",
    ["output_root", "gym_export", "gym_config"],
)
def test_existing_gym_project_rejects_prompt2scene_prompt(
    tmp_path,
    path_kind,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import project_resolution

    output_root = tmp_path / "prompt2scene/demo"
    gym_export = output_root / "gym_export"
    gym_config = gym_export / "gym_config.json"
    scene_state = gym_export / "scene_state/result.json"
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text("{}", encoding="utf-8")
    gym_config.write_text("{}", encoding="utf-8")
    input_path = {
        "output_root": output_root,
        "gym_export": gym_export,
        "gym_config": gym_config,
    }[path_kind]

    with pytest.raises(ValueError, match="--use-prompt2scene"):
        project_resolution.resolve_gym_project(
            SimpleNamespace(
                use_image2scene=False,
                use_prompt2scene=False,
                use_existing_gym_project=True,
                base_task_name=None,
                base_history_index=None,
                gym_project=str(input_path),
                prompt2scene_prompt="move the can right",
            )
        )


@pytest.mark.parametrize(
    "path_kind",
    ["output_root", "gym_export", "gym_config"],
)
def test_existing_gym_project_detects_prompt2scene_export(
    tmp_path,
    path_kind,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import project_resolution

    output_root = tmp_path / "prompt2scene/demo"
    gym_export = output_root / "gym_export"
    gym_config = gym_export / "gym_config.json"
    scene_state = gym_export / "scene_state/result.json"
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text("{}", encoding="utf-8")
    gym_config.write_text("{}", encoding="utf-8")
    input_path = {
        "output_root": output_root,
        "gym_export": gym_export,
        "gym_config": gym_config,
    }[path_kind]

    resolution = project_resolution.resolve_gym_project(
        SimpleNamespace(
            use_image2scene=False,
            use_prompt2scene=False,
            use_existing_gym_project=True,
            base_task_name=None,
            base_history_index=None,
            gym_project=str(input_path),
            prompt2scene_prompt=None,
        )
    )

    assert resolution.path == input_path.resolve()
    assert resolution.mode == "prompt2scene_existing_gym_project"


def test_existing_gym_project_without_scene_state_stays_generic(tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import project_resolution

    gym_project = tmp_path / "image2tabletop_project"
    gym_project.mkdir()
    (gym_project / "gym_config.json").write_text("{}", encoding="utf-8")

    resolution = project_resolution.resolve_gym_project(
        SimpleNamespace(
            use_image2scene=False,
            use_prompt2scene=False,
            use_existing_gym_project=True,
            base_task_name=None,
            base_history_index=None,
            gym_project=str(gym_project),
            prompt2scene_prompt=None,
        )
    )

    assert resolution.path == gym_project.resolve()
    assert resolution.mode == "existing_gym_project"


@pytest.mark.parametrize(
    (
        "target_body_scale",
        "target_body_scale_mode",
        "expected_source_scene_body_scale_mode",
        "expected_target_body_scale",
    ),
    [
        (None, None, "multiply", DEFAULT_TARGET_BODY_SCALE),
        (0.8, None, "multiply", 0.8),
        (1.0, "absolute", "absolute", 1.0),
        (0.5, "preserve", "preserve", 0.5),
    ],
)
@pytest.mark.parametrize(
    "resolution_mode",
    ["prompt2scene", "prompt2scene_existing_gym_project"],
)
def test_prompt2scene_pipeline_handles_target_scale(
    monkeypatch,
    tmp_path,
    resolution_mode,
    target_body_scale,
    target_body_scale_mode,
    expected_source_scene_body_scale_mode,
    expected_target_body_scale,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import pipeline_runner

    captured = {}

    def fake_resolve_gym_project(args):
        return SimpleNamespace(
            path=tmp_path / "gym_config.json",
            mode=resolution_mode,
        )

    class FakeTargetReplacementSpec:
        pass

    class FakeGeneratedPaths:
        output_dir = tmp_path / "configs"
        gym_config = tmp_path / "configs/fast_gym_config.json"
        agent_config = tmp_path / "configs/agent_config.json"
        task_prompt = tmp_path / "configs/task_prompt.txt"
        basic_background = tmp_path / "configs/basic_background.txt"
        atom_actions = tmp_path / "configs/atom_actions.txt"
        summary = {}

    def fake_generate_action_agent_config_from_project(**kwargs):
        captured.update(kwargs)
        return FakeGeneratedPaths()

    monkeypatch.setattr(
        pipeline_runner,
        "resolve_gym_project",
        fake_resolve_gym_project,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_target_replacements",
        lambda args, spec_cls, path: [],
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_task_description_for_generation",
        lambda args: args.task_description,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "configure_llm_usage_tracking",
        lambda args: SimpleNamespace(),
    )
    monkeypatch.setattr(pipeline_runner, "write_llm_usage_summary", lambda paths: None)
    monkeypatch.setattr(
        pipeline_runner, "write_pipeline_manifests", lambda **kwargs: {}
    )
    monkeypatch.setitem(
        sys.modules,
        "embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config",
        SimpleNamespace(
            TargetReplacementSpec=FakeTargetReplacementSpec,
            generate_action_agent_config_from_project=(
                fake_generate_action_agent_config_from_project
            ),
        ),
    )

    result = pipeline_runner.run_pipeline(
        SimpleNamespace(
            task_name="Demo",
            task_description="move cup",
            config_output_dir=str(tmp_path / "configs"),
            target_body_scale=target_body_scale,
            target_body_scale_mode=target_body_scale_mode,
            inside_container_slot_distance_scale=1.0,
            surface_release_clearance=0.05,
            prompt2scene_scene_z_rotation_degrees=-90.0,
            sync_replacement_names=False,
            reuse_target_replacements=True,
            acd_method="vhacd",
            overwrite_config=True,
            skip_run_agent=True,
            regenerate=True,
        )
    )

    assert result == 0
    assert captured["source_scene_body_scale_mode"] == (
        expected_source_scene_body_scale_mode
    )
    assert captured["preserve_source_scene_geometry"] is True
    assert captured["load_source_meshes_directly"] is True
    assert captured["source_scene_z_rotation_degrees"] == -90.0
    assert "source_mesh_x_rotation_degrees" not in captured
    assert captured["target_body_scale"] == expected_target_body_scale
    assert captured["surface_release_clearance"] == pytest.approx(0.05)
    assert captured["acd_method"] == "vhacd"


def test_pipeline_runner_forwards_headless_to_run_agent(monkeypatch, tmp_path) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import pipeline_runner

    captured = {}

    def fake_resolve_gym_project(args):
        return SimpleNamespace(
            path=tmp_path / "gym_config.json",
            mode="existing_gym_project",
        )

    class FakeTargetReplacementSpec:
        pass

    class FakeGeneratedPaths:
        output_dir = tmp_path / "configs"
        gym_config = tmp_path / "configs/fast_gym_config.json"
        agent_config = tmp_path / "configs/agent_config.json"
        task_prompt = tmp_path / "configs/task_prompt.txt"
        task_graph = tmp_path / "configs/task_graph.json"
        basic_background = tmp_path / "configs/basic_background.txt"
        atom_actions = tmp_path / "configs/atom_actions.txt"
        summary = {}

    monkeypatch.setattr(
        pipeline_runner,
        "resolve_gym_project",
        fake_resolve_gym_project,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_target_replacements",
        lambda args, spec_cls, path: [],
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_task_description_for_generation",
        lambda args: args.task_description,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "configure_llm_usage_tracking",
        lambda args: SimpleNamespace(),
    )
    monkeypatch.setattr(pipeline_runner, "write_llm_usage_summary", lambda paths: None)
    monkeypatch.setattr(
        pipeline_runner, "write_pipeline_manifests", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_agent_command",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    monkeypatch.setitem(
        sys.modules,
        "embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config",
        SimpleNamespace(
            TargetReplacementSpec=FakeTargetReplacementSpec,
            generate_action_agent_config_from_project=lambda **kwargs: FakeGeneratedPaths(),
        ),
    )

    result = pipeline_runner.run_pipeline(
        SimpleNamespace(
            task_name="Demo111",
            task_description="move cup",
            config_output_dir=str(tmp_path / "configs"),
            target_body_scale=None,
            target_body_scale_mode=None,
            inside_container_slot_distance_scale=1.0,
            prompt2scene_scene_z_rotation_degrees=-90.0,
            sync_replacement_names=False,
            reuse_target_replacements=True,
            acd_method="vhacd",
            overwrite_config=True,
            skip_run_agent=False,
            regenerate=True,
            headless=True,
        )
    )

    assert result == 0
    assert captured["headless"] is True
    assert captured["task_name"] == "Demo111"


def test_batch_new_pipeline_command_uses_pipeline_scale_defaults(
    tmp_path: Path,
) -> None:
    module = _load_local_batch_run_action_agent_videos()

    command = module._build_new_pipeline_command(
        args=SimpleNamespace(
            python="python",
            new_target_body_scale=None,
            new_target_body_scale_mode=None,
            new_pipeline_input="image-name",
            no_overwrite_config=False,
            no_regenerate=False,
        ),
        task=module.TaskSpec(
            number=13,
            name="13_Move Can Pot",
            description="用左臂把罐子放到锅左边",
        ),
        prompt2scene_output_root=tmp_path / "prompt2scene/demo13",
        config_dir=tmp_path / "configs/demo13",
    )

    assert "--target_body_scale" not in command
    assert "--target_body_scale_mode" not in command


def test_batch_new_pipeline_command_passes_explicit_scale_mode(
    tmp_path: Path,
) -> None:
    module = _load_local_batch_run_action_agent_videos()

    command = module._build_new_pipeline_command(
        args=SimpleNamespace(
            python="python",
            new_target_body_scale=0.8,
            new_target_body_scale_mode="multiply",
            new_pipeline_input="prompt-text",
            no_overwrite_config=True,
            no_regenerate=True,
        ),
        task=module.TaskSpec(
            number=13,
            name="13_Move Can Pot",
            description="用左臂把罐子放到锅左边",
        ),
        prompt2scene_output_root=tmp_path / "prompt2scene/demo13",
        config_dir=tmp_path / "configs/demo13",
    )

    assert command[command.index("--target_body_scale") + 1] == "0.8"
    assert command[command.index("--target_body_scale_mode") + 1] == "multiply"
    assert command[command.index("--prompt2scene-text") + 1] == "用左臂把罐子放到锅左边"


def _load_local_batch_run_action_agent_videos():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "gym_project"
        / "batch_run_action_agent_videos.py"
    )
    if not module_path.is_file():
        pytest.skip("local ignored gym_project batch script is not available")
    spec = importlib.util.spec_from_file_location(
        "batch_run_action_agent_videos",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentic_gen_sim_env_api_and_compat_alias() -> None:
    from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware import (
        agent_env,
    )

    assert agent_env.AtomicActionsAgentEnv is agent_env.AgenticGenSimEnv
    assert agent_env.EmbodiedEnv in agent_env.AgenticGenSimEnv.__mro__
    assert len(agent_env.AgenticGenSimEnv.__bases__) == 1
    assert agent_env.AgenticGenSimEnv.__bases__[0] is agent_env.EmbodiedEnv


def test_agentic_gen_sim_env_splits_agent_kwargs(monkeypatch) -> None:
    from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware import (
        agent_env,
    )

    captured = {}

    def fake_env_init(self, cfg=None, **kwargs) -> None:
        captured["cfg"] = cfg
        captured["env_kwargs"] = kwargs
        self.cfg = SimpleNamespace(ignore_terminations=False)

    def fake_init_agents(self, **kwargs) -> None:
        captured["agent_kwargs"] = kwargs

    monkeypatch.setattr(agent_env.EmbodiedEnv, "__init__", fake_env_init)
    monkeypatch.setattr(agent_env.AgenticGenSimEnv, "_init_agents", fake_init_agents)

    agent_env.AgenticGenSimEnv(
        cfg="cfg",
        agent_config={"Agent": {}},
        task_name="Task",
        agent_config_path="agent_config.json",
        num_envs=1,
    )

    assert captured["cfg"] == "cfg"
    assert captured["env_kwargs"] == {"num_envs": 1}
    assert captured["agent_kwargs"] == {
        "agent_config": {"Agent": {}},
        "task_name": "Task",
        "agent_config_path": "agent_config.json",
    }
