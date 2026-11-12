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
from embodichain.gen_sim.environment import _read_gen_sim_env_values
from embodichain.gen_sim.services.legacy import _load_env_file
from embodichain.gen_sim.services.config import read_env
from embodichain.gen_sim.services.errors import ServiceConfigurationError


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


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("export OPENAI_MODEL=test-model # comment", "test-model"),
        ('OPENAI_MODEL="model # literal"', "model # literal"),
    ],
)
def test_all_env_readers_agree_on_export_comments_and_quotes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entry: str,
    expected: str,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(entry + "\n", encoding="utf-8")
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(env_path))
    loaded: dict[str, str] = {}

    gen_sim_env.load_gen_sim_env(loaded)

    assert loaded["OPENAI_MODEL"] == expected
    assert _read_gen_sim_env_values("OPENAI_MODEL", env_path=env_path) == {
        "OPENAI_MODEL": expected
    }
    assert _load_env_file(env_path) == {"OPENAI_MODEL": expected}
    assert read_env(env_path) == {"OPENAI_MODEL": expected}


def test_shared_parser_preserves_each_readers_duplicate_key_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_MODEL=first\nOPENAI_MODEL=last\n", encoding="utf-8")
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(env_path))
    loaded: dict[str, str] = {}

    gen_sim_env.load_gen_sim_env(loaded)

    assert loaded["OPENAI_MODEL"] == "first"
    assert _read_gen_sim_env_values("OPENAI_MODEL", env_path=env_path) == {
        "OPENAI_MODEL": "last"
    }
    assert _load_env_file(env_path) == {"OPENAI_MODEL": "last"}
    assert read_env(env_path) == {"OPENAI_MODEL": "last"}


def test_service_env_reader_preserves_configuration_error_boundary(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("INVALID-KEY=value\n", encoding="utf-8")

    with pytest.raises(ServiceConfigurationError, match="Invalid environment variable"):
        read_env(env_path)
