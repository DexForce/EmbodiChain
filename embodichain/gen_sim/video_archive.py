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

"""Rename one completed GenSim recording to its task ID."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

__all__: list[str] = []


def _archive_task_recording(env: Any, task_id: str) -> Path | None:
    """Archive the generated audience video from a completed GenSim task.

    Args:
        env: Completed GenSim environment whose final reset flushed recording.
        task_id: ID of the task that produced the recording.

    Returns:
        Archived video path, or ``None`` when video recording is disabled.

    Raises:
        RuntimeError: If configured recorders do not identify one task video.
        ValueError: If the task ID can escape the video directory.
        FileNotFoundError: If the expected source recording does not exist.
    """
    manager = getattr(env.unwrapped, "event_manager", None)
    mode_cfgs = getattr(manager, "_mode_functor_cfgs", {})
    recorders: list[Any] = []
    for configured in mode_cfgs.values():
        for functor_cfg in configured:
            functor = getattr(functor_cfg, "func", None)
            if getattr(type(functor), "__name__", "") in {
                "record_camera_data",
                "record_camera_data_async",
            }:
                recorders.append(functor)
    if not recorders:
        return None

    audience = [
        recorder
        for recorder in recorders
        if getattr(recorder, "_name", None) == "record_cam_audience_view"
    ]
    if len(audience) == 1:
        recorder = audience[0]
    elif len(recorders) == 1:
        recorder = recorders[0]
    else:
        raise RuntimeError(
            "GenSim task video archival found multiple camera recorders without "
            "one audience recorder."
        )
    recorder_name = str(getattr(recorder, "_name", "")).strip()
    save_path = getattr(recorder, "_save_path", None)
    if not recorder_name or not isinstance(save_path, (str, Path)):
        raise RuntimeError(
            f"Cannot archive video for task {task_id!r}: camera recorder does not "
            "expose its output path and name."
        )
    return _archive_task_video(
        save_path,
        source_stem=f"episode_0_{recorder_name}",
        task_id=task_id,
    )


def _archive_task_video(
    video_directory: str | Path,
    *,
    source_stem: str,
    task_id: str,
) -> Path:
    """Move a completed recording to ``<task_id>.<original extension>``.

    Args:
        video_directory: Directory containing the completed recording.
        source_stem: Source file name without its video extension.
        task_id: ID of the task that produced the recording.

    Returns:
        Path to the archived recording.

    Raises:
        ValueError: If the task ID can escape the video directory.
        FileNotFoundError: If the expected source recording does not exist.
        RuntimeError: If more than one source extension matches.
    """
    _validate_task_id(task_id)
    directory = Path(video_directory).expanduser().resolve()
    source_prefix = f"{source_stem}."
    candidates = (
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.startswith(source_prefix)
        )
        if directory.is_dir()
        else []
    )
    expected = directory / f"{source_stem}.<extension>"
    if not candidates:
        raise FileNotFoundError(
            f"Cannot archive video for task {task_id!r}: "
            f"expected source video at {expected}."
        )
    if len(candidates) != 1:
        matches = ", ".join(path.as_posix() for path in candidates)
        raise RuntimeError(
            f"Cannot archive video for task {task_id!r}: expected exactly one "
            f"source video at {expected}, found {matches}."
        )

    source = candidates[0]
    extension = source.name[len(source_stem) :]
    destination = directory / f"{task_id}{extension}"
    source.replace(destination)
    return destination


def _validate_task_id(task_id: str) -> None:
    if (
        not isinstance(task_id, str)
        or not task_id
        or task_id in {".", ".."}
        or "/" in task_id
        or "\\" in task_id
        or "\x00" in task_id
    ):
        raise ValueError(
            f"Invalid task ID {task_id!r}: task IDs must be non-empty file names "
            "without path separators."
        )


def _main(argv: Sequence[str] | None = None) -> int:
    """Run task-video archival as a standalone GenSim command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-directory", required=True)
    parser.add_argument("--source-stem", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    try:
        destination = _archive_task_video(
            args.video_directory,
            source_stem=args.source_stem,
            task_id=args.task_id,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
