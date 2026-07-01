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
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.gym_project_api.image2tabletop_client import (
    wait_for_job,
)

_ACTION_AGENT_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain/gen_sim/action_agent_pipeline"
)


def test_image2tabletop_wait_for_job_times_out_without_polling() -> None:
    with pytest.raises(TimeoutError, match="did not complete"):
        wait_for_job("http://example.test", "job-1", poll_interval=0.1, timeout_s=0.0)


def test_action_agent_config_cli_imports() -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import (
        generate_action_agent_config,
    )

    assert callable(generate_action_agent_config.cli)


def test_action_agent_config_generation_imports() -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation import action_agent_config

    assert callable(action_agent_config.generate_action_agent_config_from_project)
    assert action_agent_config.GeneratedActionAgentConfigPaths.__name__ == (
        "GeneratedActionAgentConfigPaths"
    )


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
    llm_config.write_text("{}", encoding="utf-8")
    gym_config = output_root / "gym_export/gym_config.json"
    captured = {}

    def fake_load_llm_config(path):
        captured["llm_config_path"] = path
        return "llm-cfg"

    def fake_run_prompt2scene(request, *, llm_cfg):
        captured["request"] = request
        captured["llm_cfg"] = llm_cfg
        gym_config.parent.mkdir(parents=True)
        gym_config.write_text("{}", encoding="utf-8")
        return SimpleNamespace(gym_config_path=gym_config)

    class FakePrompt2SceneInput:
        @classmethod
        def from_cli_args(cls, *, image_path, text, output_root):
            return SimpleNamespace(
                image_path=image_path,
                text=text,
                output_root=output_root.expanduser().resolve(),
            )

    monkeypatch.setattr(
        prompt2scene_stage,
        "_load_prompt2scene_components",
        lambda: (fake_load_llm_config, fake_run_prompt2scene, FakePrompt2SceneInput),
    )

    result = prompt2scene_stage.run_prompt2scene_stage(
        SimpleNamespace(
            prompt2scene_text="a tabletop scene with bread and a basket",
            prompt2scene_output_root=str(output_root),
            prompt2scene_llm_config=str(llm_config),
            image=None,
            image_name=None,
        )
    )

    assert result == gym_config
    assert captured["llm_config_path"] == llm_config
    assert captured["llm_cfg"] == "llm-cfg"
    assert captured["request"].text == "a tabletop scene with bread and a basket"
    assert captured["request"].output_root == output_root.resolve()


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
            prompt2scene_text="a tabletop scene with bread and a basket",
            prompt2scene_scene_z_rotation_degrees=-90.0,
            prompt2scene_mesh_x_rotation_degrees=90.0,
            target_body_scale=0.8,
            target_body_scale_mode="multiply",
            target_replacement1=None,
            target_replacement2=None,
            sync_replacement_names=False,
            reuse_target_replacements=True,
            prewarm_coacd_cache=True,
            overwrite_config=True,
            regenerate=True,
            skip_run_agent=False,
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
    assert record["prompt2scene_text"] == "a tabletop scene with bread and a basket"
    assert record["prompt2scene_scene_z_rotation_degrees"] == -90.0
    assert record["prompt2scene_mesh_x_rotation_degrees"] == 90.0
    assert record["target_body_scale_mode"] == "multiply"


@pytest.mark.parametrize(
    (
        "target_body_scale",
        "target_body_scale_mode",
        "expected_source_scene_body_scale_mode",
        "expected_target_body_scale",
    ),
    [
        (None, None, "preserve", 1.0),
        (0.8, None, "multiply", 0.8),
        (1.0, "absolute", "absolute", 1.0),
        (0.5, "preserve", "preserve", 0.5),
    ],
)
def test_prompt2scene_pipeline_handles_target_scale(
    monkeypatch,
    tmp_path,
    target_body_scale,
    target_body_scale_mode,
    expected_source_scene_body_scale_mode,
    expected_target_body_scale,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.cli import pipeline_runner

    captured = {}

    def fake_resolve_gym_project(args):
        return SimpleNamespace(path=tmp_path / "gym_config.json", mode="prompt2scene")

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
            prompt2scene_scene_z_rotation_degrees=-90.0,
            prompt2scene_mesh_x_rotation_degrees=90.0,
            sync_replacement_names=False,
            reuse_target_replacements=True,
            prewarm_coacd_cache=False,
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
    assert captured["source_scene_z_rotation_degrees"] == -90.0
    assert captured["source_mesh_x_rotation_degrees"] == 90.0
    assert captured["target_body_scale"] == expected_target_body_scale


def test_batch_new_pipeline_command_preserves_prompt2scene_scale_by_default(
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
