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

from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine import interpretation as interpretation_module


def _write_dotenv(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=dotenv-key",
                "OPENAI_BASE_URL=https://dotenv.example/v1",
                "OPENAI_MODEL=dotenv-model",
            )
        ),
        encoding="utf-8",
    )


def _clear_process_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "LLM_URL",
        "TASK_ENGINE_LLM_MODEL",
        "ACTION_ENGINE_LLM_MODEL",
        "OPENAI_MODEL",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_partial_process_transport_does_not_mix_with_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    _write_dotenv(env_path)
    monkeypatch.setattr(interpretation_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(
        interpretation_module,
        "_GEN_CONFIG_PATH",
        tmp_path / "missing.json",
    )
    _clear_process_provider(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-process-key")

    settings = interpretation_module._load_llm_settings(model=None)

    assert settings["api_key"] == "dotenv-key"
    assert settings["base_url"] == "https://dotenv.example/v1"
    assert settings["model"] == "dotenv-model"


def test_complete_process_transport_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    _write_dotenv(env_path)
    monkeypatch.setattr(interpretation_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(
        interpretation_module,
        "_GEN_CONFIG_PATH",
        tmp_path / "missing.json",
    )
    _clear_process_provider(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://process.example/v1/")
    monkeypatch.setenv("TASK_ENGINE_LLM_MODEL", "process-model")

    settings = interpretation_module._load_llm_settings(model=None)

    assert settings["api_key"] == "process-key"
    assert settings["base_url"] == "https://process.example/v1"
    assert settings["model"] == "process-model"
