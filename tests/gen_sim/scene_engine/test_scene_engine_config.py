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

from embodichain.gen_sim.scene_engine.cli import start
from embodichain.gen_sim.scene_engine.configs import environment


def test_read_scene_engine_env_values_reads_requested_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text('OPENAI_MODEL="test-model"\nUNRELATED_VALUE=ignored\n')
    monkeypatch.setattr(environment, "_SCENE_ENGINE_ENV_PATH", env_path)

    assert environment.read_scene_engine_env_values("OPENAI_MODEL") == {
        "OPENAI_MODEL": "test-model"
    }


def test_read_scene_engine_env_values_reports_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_MODEL=test-model\n")
    monkeypatch.setattr(environment, "_SCENE_ENGINE_ENV_PATH", env_path)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        environment.read_scene_engine_env_values("OPENAI_MODEL", "OPENAI_API_KEY")


def test_scene_engine_help_exposes_only_runtime_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        start.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--image" in output
    assert "--output_root" in output
    assert "gen_sim/.env" in output
    assert "--config" not in output


def test_scene_engine_cli_forwards_validated_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    captured: dict[str, Path] = {}

    def generate_scene(*, image_path: Path, output_root: Path) -> None:
        captured["image_path"] = image_path
        captured["output_root"] = output_root

    monkeypatch.setattr(start, "generate_scene_from_image", generate_scene)
    output_root = tmp_path / "output"

    start.cli_scene_engine(image_path, output_root)

    assert captured == {
        "image_path": image_path.resolve(),
        "output_root": output_root.resolve(),
    }


@pytest.mark.parametrize("image_name", ["missing.png", "scene.gif"])
def test_scene_engine_cli_rejects_invalid_image_inputs(
    tmp_path: Path,
    image_name: str,
) -> None:
    image_path = tmp_path / image_name
    if image_path.suffix == ".gif":
        image_path.write_bytes(b"gif")

    with pytest.raises((FileNotFoundError, ValueError)):
        start.cli_scene_engine(image_path, tmp_path / "output")
