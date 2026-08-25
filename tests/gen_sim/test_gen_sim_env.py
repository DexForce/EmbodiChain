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

from embodichain.gen_sim import env as gen_sim_env


def test_missing_default_env_file_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "embodichain" / "gen_sim" / "env.py"
    monkeypatch.delenv("EMBODICHAIN_ENV_FILE", raising=False)
    monkeypatch.setattr(gen_sim_env, "__file__", str(module_path))

    assert gen_sim_env.find_gen_sim_env_file() is None
    assert gen_sim_env.load_gen_sim_env({}) is None


def test_missing_configured_env_file_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "missing.env"
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(env_path))

    assert gen_sim_env.find_gen_sim_env_file() == env_path.resolve()
    assert gen_sim_env.load_gen_sim_env({}) is None


def test_shell_values_take_precedence_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_MODEL=dotenv-model\nOPENAI_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(env_path))
    target_env = {"OPENAI_MODEL": "shell-model"}

    assert gen_sim_env.load_gen_sim_env(target_env) == env_path.resolve()
    assert target_env == {
        "OPENAI_MODEL": "shell-model",
        "OPENAI_API_KEY": "dotenv-key",
    }
