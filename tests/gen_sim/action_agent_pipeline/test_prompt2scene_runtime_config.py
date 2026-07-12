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

import json
from pathlib import Path

import pytest

from embodichain.gen_sim.action_agent_pipeline.utils import (
    prompt2scene_runtime_config,
)

_SERVICE_PORTS = {
    "PROMPT2SCENE_SAM3_SEGMENTATION_PORT": "5024",
    "PROMPT2SCENE_SAM3D_GENERATION_PORT": "5029",
    "PROMPT2SCENE_ZIMAGE_PORT": "5023",
}


def test_build_prompt2scene_llm_config_uses_shared_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "llm_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {
                    "openai_compatible": {
                        "default_query": {"provider": "test"},
                        "max_attempts": 4,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prompt2scene_runtime_config,
        "get_openai_compatible_llm_config",
        lambda **_: {
            "api_key": "shared-key",
            "model": "shared-model",
            "base_url": "https://llm.example.test/v1",
        },
    )

    cfg = prompt2scene_runtime_config.build_prompt2scene_llm_config(config_path)

    assert cfg.api_key == "shared-key"
    assert cfg.model == "shared-model"
    assert cfg.base_url == "https://llm.example.test/v1"
    assert cfg.default_query == {"provider": "test"}
    assert cfg.max_attempts == 4


def test_write_prompt2scene_client_config_resolves_host_and_ports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_config_path = _write_source_client_config(tmp_path)
    monkeypatch.setattr(
        prompt2scene_runtime_config,
        "load_local_env_values",
        lambda: {
            "PROMPT2SCENE_SERVICE_HOST": "services.example.test",
            **_SERVICE_PORTS,
        },
    )

    runtime_path = prompt2scene_runtime_config.write_prompt2scene_client_config(
        tmp_path / "output",
        source_config_path=source_config_path,
    )
    runtime_config = json.loads(runtime_path.read_text(encoding="utf-8"))

    assert runtime_config["sam3_segmentation"]["base_url"] == (
        "http://services.example.test:5024"
    )
    assert runtime_config["sam3d_generation"]["base_url"] == (
        "http://services.example.test:5029"
    )
    assert runtime_config["zimage"]["base_url"] == "http://services.example.test:5023"
    assert (
        json.loads(source_config_path.read_text(encoding="utf-8"))["zimage"]["base_url"]
        == "http://placeholder.test"
    )


def test_write_prompt2scene_client_config_prefers_shell_base_urls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_config_path = _write_source_client_config(tmp_path)
    monkeypatch.setattr(
        prompt2scene_runtime_config,
        "load_local_env_values",
        lambda: {"PROMPT2SCENE_SERVICE_HOST": "local.example.test", **_SERVICE_PORTS},
    )
    monkeypatch.setenv(
        "PROMPT2SCENE_ZIMAGE_BASE_URL",
        "https://override.example.test:7443/",
    )

    runtime_path = prompt2scene_runtime_config.write_prompt2scene_client_config(
        tmp_path / "output",
        source_config_path=source_config_path,
    )
    runtime_config = json.loads(runtime_path.read_text(encoding="utf-8"))

    assert runtime_config["zimage"]["base_url"] == "https://override.example.test:7443"


def test_write_prompt2scene_client_config_rejects_missing_service_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_config_path = _write_source_client_config(tmp_path)
    monkeypatch.setattr(
        prompt2scene_runtime_config, "load_local_env_values", lambda: {}
    )

    with pytest.raises(ValueError, match="PROMPT2SCENE_SAM3_SEGMENTATION_BASE_URL"):
        prompt2scene_runtime_config.write_prompt2scene_client_config(
            tmp_path / "output",
            source_config_path=source_config_path,
        )


def test_write_prompt2scene_client_config_rejects_invalid_base_url_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_config_path = _write_source_client_config(tmp_path)
    monkeypatch.setattr(
        prompt2scene_runtime_config,
        "load_local_env_values",
        lambda: {
            "PROMPT2SCENE_SERVICE_HOST": "services.example.test",
            **_SERVICE_PORTS,
        },
    )
    monkeypatch.setenv(
        "PROMPT2SCENE_ZIMAGE_BASE_URL",
        "http://services.example.test:not-a-port",
    )

    with pytest.raises(ValueError, match="invalid port"):
        prompt2scene_runtime_config.write_prompt2scene_client_config(
            tmp_path / "output",
            source_config_path=source_config_path,
        )


def test_use_prompt2scene_client_config_restores_default_after_exception(
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim.prompt2scene.agent_tools.clients import config

    original_path = config.DEFAULT_CLIENT_CONFIG_PATH

    with pytest.raises(RuntimeError, match="expected failure"):
        with prompt2scene_runtime_config.use_prompt2scene_client_config(
            tmp_path / "runtime_client_config.json"
        ):
            assert (
                config.DEFAULT_CLIENT_CONFIG_PATH
                == (tmp_path / "runtime_client_config.json").resolve()
            )
            raise RuntimeError("expected failure")

    assert config.DEFAULT_CLIENT_CONFIG_PATH == original_path


def _write_source_client_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "client_config.json"
    config_path.write_text(
        json.dumps(
            {
                "sam3_segmentation": {"base_url": "http://placeholder.test"},
                "sam3d_generation": {"base_url": "http://placeholder.test"},
                "zimage": {"base_url": "http://placeholder.test"},
            }
        ),
        encoding="utf-8",
    )
    return config_path
