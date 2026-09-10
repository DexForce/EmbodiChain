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
    assert "--prompt2scene_scene_z_rotation_degrees" in output
    assert "--prompt2scene_mesh_x_rotation_degrees" in output
    assert "--target_body_scale_mode" in output
    assert "gen_sim/.env" in output
    assert "--config" not in output


def test_scene_engine_cli_forwards_validated_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    captured: dict[str, object] = {}

    def generate_scene(
        *,
        image_path: Path,
        output_root: Path,
        scene_z_rotation_degrees: float,
    ) -> None:
        captured["image_path"] = image_path
        captured["output_root"] = output_root
        captured["scene_z_rotation_degrees"] = scene_z_rotation_degrees

    monkeypatch.setattr(start, "generate_scene_from_image", generate_scene)
    output_root = tmp_path / "output"

    start.cli_scene_engine(
        image_path,
        output_root,
        scene_z_rotation_degrees=180.0,
    )

    assert captured == {
        "image_path": image_path.resolve(),
        "output_root": output_root.resolve(),
        "scene_z_rotation_degrees": 180.0,
    }


def test_scene_engine_main_accepts_legacy_direct_glb_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    captured: dict[str, object] = {}

    def cli_scene_engine(
        image: str,
        output_root: str,
        *,
        edit_prompt: str | None,
        scene_z_rotation_degrees: float,
    ) -> None:
        captured.update(
            image=image,
            output_root=output_root,
            edit_prompt=edit_prompt,
            scene_z_rotation_degrees=scene_z_rotation_degrees,
        )

    monkeypatch.setattr(start, "cli_scene_engine", cli_scene_engine)

    start.main(
        [
            "--image",
            str(image_path),
            "--output_root",
            str(tmp_path / "output"),
            "--target_body_scale_mode",
            "preserve",
            "--prompt2scene_scene_z_rotation_degrees",
            "180",
            "--prompt2scene_mesh_x_rotation_degrees",
            "0",
        ]
    )

    assert captured["scene_z_rotation_degrees"] == 180.0


def test_scene_engine_main_rejects_legacy_mesh_x_rotation(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        start.main(
            [
                "--output_root",
                str(tmp_path / "output"),
                "--prompt2scene_mesh_x_rotation_degrees",
                "90",
            ]
        )

    assert exc_info.value.code == 2


def test_scene_engine_cli_edits_existing_output_without_an_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def edit_scene(*, output_root: Path, edit_prompt: str) -> None:
        captured["output_root"] = output_root
        captured["edit_prompt"] = edit_prompt

    monkeypatch.setattr(start, "edit_scene", edit_scene)
    output_root = tmp_path / "existing_output"

    start.cli_scene_engine(None, output_root, edit_prompt="move the cup right")

    assert captured == {
        "output_root": output_root.resolve(),
        "edit_prompt": "move the cup right",
    }


def test_scene_engine_cli_generates_then_edits_when_both_inputs_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    call_order: list[str] = []

    def generate_scene(
        *,
        image_path: Path,
        output_root: Path,
        scene_z_rotation_degrees: float,
    ) -> None:
        call_order.append("generate")

    def edit_scene(*, output_root: Path, edit_prompt: str) -> None:
        call_order.append("edit")

    monkeypatch.setattr(start, "generate_scene_from_image", generate_scene)
    monkeypatch.setattr(start, "edit_scene", edit_scene)

    start.cli_scene_engine(
        image_path,
        tmp_path / "output",
        edit_prompt="move the cup right",
    )

    assert call_order == ["generate", "edit"]


def test_scene_engine_cli_requires_an_image_or_edit_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--image, --edit_prompt, or both"):
        start.cli_scene_engine(None, tmp_path / "output")


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
