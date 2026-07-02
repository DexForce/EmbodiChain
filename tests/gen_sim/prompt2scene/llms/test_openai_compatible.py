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

from embodichain.gen_sim.prompt2scene.llms import openai_compatible


def test_load_llm_config_reuses_simready_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "llm_config.json"
    _write_llm_config(
        config_path,
        api_key="",
        model="",
        base_url="",
        max_attempts=2,
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="env-key"',
                'OPENAI_BASE_URL="https://example.test/v1/"',
                'OPENAI_MODEL="env-model"',
                'OPENAI_MAX_ATTEMPTS="4"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(openai_compatible, "SIMREADY_LLM_ENV_PATH", env_path)
    _clear_openai_env(monkeypatch)

    cfg = openai_compatible.load_llm_config(config_path)

    assert cfg.api_key == "env-key"
    assert cfg.model == "env-model"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.max_attempts == 4


def test_load_llm_config_prefers_shell_env_over_simready_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "llm_config.json"
    _write_llm_config(
        config_path,
        api_key="json-key",
        model="json-model",
        base_url="https://json.example.test/v1",
        max_attempts=2,
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="file-key"',
                'OPENAI_BASE_URL="https://file.example.test/v1"',
                'OPENAI_MODEL="file-model"',
                'OPENAI_MAX_ATTEMPTS="4"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(openai_compatible, "SIMREADY_LLM_ENV_PATH", env_path)
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://shell.example.test/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    monkeypatch.setenv("OPENAI_MAX_ATTEMPTS", "5")

    cfg = openai_compatible.load_llm_config(config_path)

    assert cfg.api_key == "shell-key"
    assert cfg.model == "shell-model"
    assert cfg.base_url == "https://shell.example.test/v1"
    assert cfg.max_attempts == 5


def _write_llm_config(
    path: Path,
    *,
    api_key: str,
    model: str,
    base_url: str,
    max_attempts: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "llm": {
                    "openai_compatible": {
                        "api_key": api_key,
                        "model": model,
                        "base_url": base_url,
                        "default_query": {},
                        "max_attempts": max_attempts,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _clear_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MAX_ATTEMPTS", raising=False)
