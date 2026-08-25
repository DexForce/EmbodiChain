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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.video_archive import (
    _archive_task_recording,
    _archive_task_video,
)

SOURCE_STEM = "episode_0_record_cam_audience_view"


class record_camera_data:
    def __init__(self, save_path: Path) -> None:
        self._name = "record_cam_audience_view"
        self._save_path = save_path


def _env(recorder: record_camera_data | None = None) -> SimpleNamespace:
    event_manager = SimpleNamespace(
        _mode_functor_cfgs={
            "interval": (
                [SimpleNamespace(func=recorder)] if recorder is not None else []
            )
        }
    )
    env = SimpleNamespace(event_manager=event_manager)
    env.unwrapped = env
    return env


def _write_source(directory: Path, extension: str, content: bytes = b"video") -> Path:
    source = directory / f"{SOURCE_STEM}{extension}"
    source.write_bytes(content)
    return source


def test_archive_task_video_renames_source_and_preserves_extension(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, ".webm")

    destination = _archive_task_video(
        tmp_path,
        source_stem=SOURCE_STEM,
        task_id="task2_1",
    )

    assert destination == tmp_path / "task2_1.webm"
    assert destination.read_bytes() == b"video"
    assert not source.exists()


@pytest.mark.parametrize("task_id", ["../task2_1", "task2/1", r"task2\1", ".."])
def test_archive_task_video_rejects_path_characters(
    tmp_path: Path,
    task_id: str,
) -> None:
    source = _write_source(tmp_path, ".mp4")

    with pytest.raises(ValueError, match="Invalid task ID"):
        _archive_task_video(tmp_path, source_stem=SOURCE_STEM, task_id=task_id)

    assert source.is_file()


def test_archive_task_video_reports_missing_source_with_task_and_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError) as error:
        _archive_task_video(
            tmp_path,
            source_stem=SOURCE_STEM,
            task_id="task2_1",
        )

    message = str(error.value)
    assert "task2_1" in message
    assert str(tmp_path / f"{SOURCE_STEM}.<extension>") in message


def test_archive_task_video_overwrites_existing_target(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, ".mp4", b"new")
    destination = tmp_path / "task2_1.mp4"
    destination.write_bytes(b"existing")

    result = _archive_task_video(
        tmp_path,
        source_stem=SOURCE_STEM,
        task_id="task2_1",
    )

    assert result == destination
    assert destination.read_bytes() == b"new"
    assert not source.exists()


def test_consecutive_tasks_keep_independent_videos(tmp_path: Path) -> None:
    for task_id, content in (("task2_1", b"first"), ("task2_2", b"second")):
        _write_source(tmp_path, ".mp4", content)
        _archive_task_video(
            tmp_path,
            source_stem=SOURCE_STEM,
            task_id=task_id,
        )

    assert (tmp_path / "task2_1.mp4").read_bytes() == b"first"
    assert (tmp_path / "task2_2.mp4").read_bytes() == b"second"
    assert not (tmp_path / f"{SOURCE_STEM}.mp4").exists()


def test_task_recording_uses_runtime_recorder_path(tmp_path: Path) -> None:
    recorder = record_camera_data(tmp_path)
    source = _write_source(tmp_path, ".mkv")

    destination = _archive_task_recording(_env(recorder), "task2_1")

    assert destination == tmp_path / "task2_1.mkv"
    assert destination.read_bytes() == b"video"
    assert not source.exists()


def test_task_recording_is_noop_when_recording_is_disabled() -> None:
    assert _archive_task_recording(_env(), "task2_1") is None
