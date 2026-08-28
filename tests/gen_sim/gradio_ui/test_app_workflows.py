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
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

import app_workflows  # noqa: E402
import app_processes  # noqa: E402


class _FakeProcess:
    """Minimal subprocess stand-in for Scene Engine workflow tests."""

    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def _write_scene_export(
    scene_store: Path,
    *,
    scene_name: str,
    scene_id: str,
) -> Path:
    scene_root = scene_store / scene_name
    scene_export_root = scene_root / "scene_export"
    scene_export_root.mkdir(parents=True)
    (scene_export_root / "scene_config.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "scene_id": scene_id,
            }
        ),
        encoding="utf-8",
    )
    return scene_root


def test_saved_scene_choices_lists_complete_exports_newest_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older_root = _write_scene_export(
        tmp_path,
        scene_name="older-scene",
        scene_id="scene-engine-older",
    )
    newer_root = _write_scene_export(
        tmp_path,
        scene_name="newer-scene",
        scene_id="scene-engine-newer",
    )
    oldest_timestamp_s = 1_700_000_000
    newest_timestamp_s = oldest_timestamp_s + 60
    os.utime(
        older_root / "scene_export" / "scene_config.json",
        (oldest_timestamp_s, oldest_timestamp_s),
    )
    os.utime(
        newer_root / "scene_export" / "scene_config.json",
        (newest_timestamp_s, newest_timestamp_s),
    )
    incomplete_root = tmp_path / "incomplete-scene" / "scene_export"
    incomplete_root.mkdir(parents=True)
    (incomplete_root / "scene_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_workflows, "GEN_SIM_SCENE_ROOT", tmp_path)

    choices = app_workflows.saved_scene_choices()

    assert [value for _label, value in choices] == ["newer-scene", "older-scene"]


def test_scene_edit_command_targets_selected_export_and_prompt(tmp_path: Path) -> None:
    prompt = "move the blue cup to the front-center"

    command = app_workflows._build_scene_edit_command(tmp_path, prompt)

    assert command[-4:] == ["--output_root", str(tmp_path), "--edit_prompt", prompt]
    assert "--image" not in command


def test_scene_edit_command_rejects_empty_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Enter an edit instruction"):
        app_workflows._build_scene_edit_command(tmp_path, "   ")


def test_scene_edit_workflow_does_not_start_for_empty_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_scene_export(
        tmp_path,
        scene_name="selected-scene",
        scene_id="scene-engine-selected",
    )
    monkeypatch.setattr(app_workflows, "GEN_SIM_SCENE_ROOT", tmp_path)

    def fail_if_started(_command: list[str]):
        raise AssertionError("The Scene Engine process must not start.")

    monkeypatch.setattr(app_workflows, "start_pipeline", fail_if_started)
    request = SimpleNamespace(session_hash="empty-edit-prompt-session")
    try:
        updates = list(app_workflows.run_scene_edit("selected-scene", "   ", request))
    finally:
        app_workflows._scene_runs.reset(request.session_hash, force=True)
        app_workflows.runtime_registry.reset(request.session_hash)

    assert len(updates) == 1
    assert "Enter an edit instruction" in updates[0][1]


def test_scene_edit_workflow_restarts_selected_scene_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_name = "selected-scene"
    edit_prompt = "add a red cup to the tabletop center"
    _write_scene_export(
        tmp_path,
        scene_name=scene_name,
        scene_id="scene-engine-selected",
    )
    monkeypatch.setattr(app_workflows, "GEN_SIM_SCENE_ROOT", tmp_path)
    commands: list[list[str]] = []

    def start_fake_pipeline(command: list[str]) -> _FakeProcess:
        commands.append(command)
        return _FakeProcess(0 if len(commands) == 1 else None)

    monkeypatch.setattr(app_workflows, "start_pipeline", start_fake_pipeline)
    monkeypatch.setattr(
        app_workflows,
        "read_process_output",
        lambda _process, _queue, _log_path=None: None,
    )
    monkeypatch.setattr(app_workflows, "_wait_for_viser", lambda _port, _process: True)
    preview_port = 54_321
    monkeypatch.setattr(
        app_workflows,
        "_select_available_port",
        lambda _preferred_port: preview_port,
    )
    monkeypatch.setattr(app_processes, "kill_process_group", lambda _process: None)
    request = SimpleNamespace(session_hash="successful-edit-session")
    try:
        updates = list(app_workflows.run_scene_edit(scene_name, edit_prompt, request))
    finally:
        app_workflows._scene_runs.reset(request.session_hash, force=True)
        app_workflows.runtime_registry.reset(request.session_hash)

    assert commands[0][-2:] == ["--edit_prompt", edit_prompt]
    assert "--viser" in commands[1]
    assert updates[-1][0] == 100
    assert "Scene edited successfully" in updates[-1][1]
    assert scene_name in updates[-1][3]


@pytest.mark.parametrize(
    ("log_line", "expected_phase"),
    [
        ("Starting Edit Understanding", "edit_understanding"),
        ("Starting Objects Preparation", "edit_asset_preparation"),
        ("Starting Layout Generation", "edit_layout"),
        ("Starting Scene Export", "gym_export"),
    ],
)
def test_scene_edit_logs_map_to_progress_phases(
    log_line: str,
    expected_phase: str,
) -> None:
    assert app_workflows._scene_engine_phase_from_log(log_line, "started") == (
        expected_phase
    )
