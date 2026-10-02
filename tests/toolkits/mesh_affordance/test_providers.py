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

"""Model selection and credential isolation for the Codex provider switch."""

from __future__ import annotations

import json
import os
import sys

import pytest

from embodichain.toolkits.mesh_affordance import _codex
from embodichain.toolkits.mesh_affordance._providers import resolve_provider

_SECRET = "fake-provider-secret-for-tests"


@pytest.fixture
def provider_file(tmp_path):
    path = tmp_path / "private.json"
    path.write_text(
        json.dumps({"base_url": "https://api.deepseek.com", "api_key": _SECRET})
    )
    return path


def test_default_openai_provider_does_not_require_credentials():
    assert resolve_provider("openai", None, None) == ("gpt-6-astra", None)
    assert resolve_provider("openai", "explicit-model", None) == (
        "explicit-model",
        None,
    )


def test_deepseek_model_defaults_and_explicit_override(provider_file):
    assert resolve_provider("deepseek", None, str(provider_file))[0] == "deepseek-flash"
    settings = json.loads(provider_file.read_text())
    settings["model"] = "deepseek-v4-flash-vision-exp"
    provider_file.write_text(json.dumps(settings))
    assert (
        resolve_provider("deepseek", None, str(provider_file))[0] == settings["model"]
    )
    assert (
        resolve_provider("deepseek", "deepseek-flash", str(provider_file))[0]
        == "deepseek-flash"
    )


@pytest.mark.parametrize(
    "model", ["deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]
)
def test_known_text_only_models_fail_before_inference(provider_file, model):
    with pytest.raises(ValueError, match="vision"):
        resolve_provider("deepseek", model, str(provider_file))


@pytest.mark.parametrize(
    "contents",
    [
        "not-json-with-fake-provider-secret-for-tests",
        json.dumps({"api_key": _SECRET}),
        json.dumps(
            {"base_url": "https://example.com?key=" + _SECRET, "api_key": _SECRET}
        ),
        json.dumps(
            {"base_url": "https://user:" + _SECRET + "@example.com", "api_key": _SECRET}
        ),
        json.dumps(
            {
                "base_url": "https://api.deepseek.com",
                "api_key": _SECRET,
                "unknown": _SECRET,
            }
        ),
    ],
)
def test_invalid_credentials_never_appear_in_errors(provider_file, contents):
    provider_file.write_text(contents)
    with pytest.raises(ValueError) as error:
        resolve_provider("deepseek", None, str(provider_file))
    assert _SECRET not in str(error.value)


def test_missing_deepseek_config_is_actionable():
    with pytest.raises(ValueError, match="provider-config"):
        resolve_provider("deepseek", None, None)


def _evidence(directory, provider_file):
    directory.mkdir()
    (directory / "evidence.json").write_text(
        json.dumps(
            {
                "object_description": "cup",
                "task_description": "pour",
                "images": ["view.png"],
                "config": {"provider_config": str(provider_file)},
            }
        )
    )


def test_deepseek_uses_codex_images_and_response_schema_without_key_in_argv(
    tmp_path, provider_file, monkeypatch
):
    directory = tmp_path / "run"
    _evidence(directory, provider_file)
    before = os.environ.get("EMBODICHAIN_DEEPSEEK_API_KEY")

    def fake_process(command, directory, stem, timeout, prompt=None, *, env=None):
        assert env["EMBODICHAIN_DEEPSEEK_API_KEY"] == _SECRET
        assert os.environ.get("EMBODICHAIN_DEEPSEEK_API_KEY") == before
        assert _SECRET not in json.dumps(command)
        assert _SECRET not in prompt
        assert str(provider_file) not in prompt
        assert 'model_provider="deepseek"' in command
        assert 'model_providers.deepseek.wire_api="responses"' in command
        assert 'model_providers.deepseek.base_url="https://api.deepseek.com"' in command
        assert (
            'model_providers.deepseek.env_key="EMBODICHAIN_DEEPSEEK_API_KEY"' in command
        )
        assert (
            'shell_environment_policy.filters.EMBODICHAIN_DEEPSEEK_API_KEY="exclude"'
            in command
        )
        assert command[command.index("--model") + 1] == "deepseek-flash"
        assert "--image" in command and "--output-schema" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        (directory / "response.raw.txt").write_text('{"test": "ok"}')

    monkeypatch.setattr(_codex, "run_process", fake_process)
    assert _codex.run_codex(
        directory,
        "deepseek-flash",
        "codex",
        30,
        provider="deepseek",
        provider_config=str(provider_file),
    ) == {"test": "ok"}
    for path in directory.iterdir():
        assert _SECRET not in path.read_text()


def test_provider_failure_logs_are_redacted(tmp_path, provider_file, monkeypatch):
    directory = tmp_path / "run"
    _evidence(directory, provider_file)

    def fake_process(command, directory, stem, timeout, prompt=None, *, env=None):
        (directory / "codex.stderr.log").write_text(
            "Invalid key: " + env["EMBODICHAIN_DEEPSEEK_API_KEY"]
        )
        (directory / "response.raw.txt").write_text(json.dumps({"message": _SECRET}))
        raise RuntimeError("Provider rejected authentication")

    monkeypatch.setattr(_codex, "run_process", fake_process)
    with pytest.raises(RuntimeError, match="authentication"):
        _codex.run_codex(
            directory,
            "deepseek-flash",
            "codex",
            30,
            provider="deepseek",
            provider_config=str(provider_file),
        )
    for name in ("codex.stderr.log", "response.raw.txt"):
        assert _SECRET not in (directory / name).read_text()
        assert "[REDACTED]" in (directory / name).read_text()


def test_process_receives_provider_environment_without_mutating_parent(tmp_path):
    environment = os.environ.copy()
    environment["MESH_PROVIDER_TEST"] = "child-only"
    _codex.run_process(
        [sys.executable, "-c", "import os; print(os.environ['MESH_PROVIDER_TEST'])"],
        tmp_path,
        "environment",
        10,
        env=environment,
    )
    assert (tmp_path / "environment.stdout.log").read_text().strip() == "child-only"
    assert "MESH_PROVIDER_TEST" not in os.environ


def test_pipeline_routes_deepseek_and_records_provider_without_secrets(
    tmp_path, provider_file, monkeypatch
):
    import hashlib
    import numpy as np
    from embodichain.toolkits.mesh_affordance import (
        MeshAffordanceCfg,
        score_mesh_affordance,
    )
    from embodichain.toolkits.mesh_affordance import _pipeline

    directory = tmp_path / "pipeline"
    directory.mkdir()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    np.savez(
        directory / "geometry.npz", vertices=vertices, faces=faces, face_patch_ids=[0]
    )
    cfg = MeshAffordanceCfg(
        provider="deepseek",
        provider_config=str(provider_file),
        codex_executable=sys.executable,
    )
    manifest = {
        "config": cfg.to_dict(),
        "images": ["view.png"],
        "patches": [{"patch_id": 0}],
        "geometry_sha256": hashlib.sha256(
            (directory / "geometry.npz").read_bytes()
        ).hexdigest(),
        "mesh_sha256": "test-mesh",
        "vertex_index_contract": "obj_v_line_order",
    }
    (directory / "evidence.json").write_text(json.dumps(manifest))
    (directory / "view.png").write_bytes(b"fake image")

    def fake_codex(directory, model, executable, timeout, **kwargs):
        assert model == "deepseek-flash"
        assert kwargs == {"provider": "deepseek", "provider_config": str(provider_file)}
        return {
            "task_interpretation": "pour",
            "assumptions": [],
            "regions": [
                {
                    "name": "handle",
                    "patch_ids": [0],
                    "score": 0.9,
                    "confidence": 0.9,
                    "reason": "test",
                }
            ],
        }

    monkeypatch.setattr(_pipeline, "run_codex", fake_codex)
    monkeypatch.setattr(_pipeline, "_render", lambda *args: None)
    result = score_mesh_affordance(directory, cfg)
    np.testing.assert_allclose(result.scores, [0.9, 0.9, 0.9])
    report = json.loads((directory / "report.json").read_text())
    assert report["provider"] == "deepseek" and report["model"] == "deepseek-flash"
    assert report["harness"] == "codex"
    assert _SECRET not in (directory / "report.json").read_text()
    assert _SECRET not in (directory / "evidence.json").read_text()


def test_cli_provider_switch_resets_saved_model_without_overriding_explicit_model(
    tmp_path, provider_file, monkeypatch
):
    from types import SimpleNamespace
    import numpy as np
    from embodichain.toolkits.mesh_affordance import __main__ as cli

    directory = tmp_path / "prepared"
    directory.mkdir()
    (directory / "evidence.json").write_text(
        json.dumps({"config": {"model": "gpt-6-astra"}})
    )
    seen = []

    def fake_score(output, cfg):
        seen.append(resolve_provider(cfg.provider, cfg.model, cfg.provider_config)[0])
        return SimpleNamespace(
            graspable_mask=np.zeros(1, dtype=bool),
            scores=np.zeros(1),
            part_mask=None,
            output_dir=directory,
        )

    monkeypatch.setattr(cli, "score_mesh_affordance", fake_score)
    args = [
        "mesh-affordance",
        "score",
        "--output",
        str(directory),
        "--provider",
        "deepseek",
        "--provider-config",
        str(provider_file),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli._main()
    monkeypatch.setattr(sys, "argv", args + ["--model", "deepseek-v4-flash-vision-exp"])
    cli._main()
    assert seen == ["deepseek-flash", "deepseek-v4-flash-vision-exp"]
