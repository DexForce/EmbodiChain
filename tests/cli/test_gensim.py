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

"""Public GenSim command grammar, compatibility and dependency boundaries."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from embodichain.cli.main import main

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    "command",
    [
        ["gensim"],
        ["gensim", "services"],
        ["gensim", "asset"],
        ["gensim", "scene"],
        ["gensim", "task"],
        ["gensim", "asset", "ingest"],
        ["gensim", "scene", "generate"],
        ["gensim", "scene", "edit"],
        ["gensim", "scene", "expand"],
        ["gensim", "scene", "preview"],
        ["gensim", "task", "plan"],
        ["gensim", "services", "validate"],
        ["gensim", "services", "doctor"],
        ["gensim", "run"],
        ["gensim", "ui"],
        ["asset-engine"],
        ["scene-engine"],
        ["task-engine"],
        ["preview-scene"],
        ["scene-randomization"],
    ],
)
def test_help_works_without_site_packages_or_service_configuration(
    command: list[str], tmp_path: Path
) -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "embodichain", *command, "--help"],
        env={
            **os.environ,
            "EMBODICHAIN_MODEL_SERVICES_CONFIG": str(tmp_path / "missing.json"),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: embodichain " + " ".join(command) in result.stdout


def test_asset_ingest_and_legacy_command_share_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.asset_engine.cli import start

    ingest = Mock()
    monkeypatch.setattr(start, "cli_ingest_single", ingest)
    main(
        [
            "gensim",
            "asset",
            "ingest",
            "--input",
            "raw asset",
            "--output",
            "simready",
            "--category",
            "cup",
        ]
    )
    main(
        [
            "asset-engine",
            "--input_dir",
            "raw asset",
            "--output_root",
            "simready",
            "--category",
            "cup",
        ]
    )
    assert ingest.call_args_list == [(("raw asset", "simready", "cup"),)] * 2


@pytest.mark.parametrize(
    "operation,image,prompt",
    [("generate", "scene.png", None), ("edit", None, "move the cup")],
)
def test_scene_operations_preserve_legacy_inputs(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    image: str | None,
    prompt: str | None,
) -> None:
    from embodichain.gen_sim.scene_engine.cli import start

    dispatch = Mock()
    monkeypatch.setattr(start, "cli_scene_engine", dispatch)
    args = ["gensim", "scene", operation]
    legacy = ["scene-engine", "--output_root", "scene root"]
    if image is not None:
        args += ["--image", image, "--output", "scene root"]
        legacy += ["--image", image]
    if prompt is not None:
        args += ["--scene", "scene root", "--prompt", prompt]
        legacy += ["--edit_prompt", prompt]
    main(args)
    main(legacy)
    assert dispatch.call_count == 2
    for call in dispatch.call_args_list:
        assert call.args == (image, "scene root")
        assert call.kwargs == {
            "edit_prompt": prompt,
            "scene_z_rotation_degrees": 180.0,
            "scene_randomization": None,
        }


@pytest.mark.parametrize(
    "command",
    [
        ["gensim", "scene", "randomize"],
        ["gensim", "scene", "generate", "--output", "out"],
        ["gensim", "scene", "edit", "--scene", "out"],
        ["gensim", "asset", "ingest", "--bogus"],
    ],
)
def test_invalid_operation_fails_before_loading_runtime(command: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "embodichain", *command],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr


def test_ui_launch_uses_the_active_python_and_preserves_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.cli import gensim

    launch = Mock()
    monkeypatch.setattr(gensim.os, "execv", launch)
    monkeypatch.setenv("GRADIO_SERVER_NAME", "old-host")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7000")
    monkeypatch.setenv("PYTHONPATH", "inherited-path")
    main(["gensim", "ui", "--host", "127.0.0.1", "--port", "7861"])
    args = launch.call_args.args
    assert args[0] == sys.executable
    assert args[1][0] == sys.executable
    assert Path(args[1][1]).name == "gradio_app.py"
    assert Path(args[1][1]).is_file()
    assert os.environ["GRADIO_SERVER_NAME"] == "127.0.0.1"
    assert os.environ["GRADIO_SERVER_PORT"] == "7861"
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [
        str(Path(__file__).resolve().parents[2]),
        "inherited-path",
    ]
