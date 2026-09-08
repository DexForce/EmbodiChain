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

"""Media helpers used by the Action engine."""

from __future__ import annotations

from pathlib import Path

from app_config import OUTPUTS_DIR, VIDEO_SUFFIXES

__all__ = ["latest_audience_output_video"]


def _collect_audience_output_videos() -> list[Path]:
    if not OUTPUTS_DIR.is_dir():
        return []
    videos = [
        path
        for path in OUTPUTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    audience_videos = [
        path
        for path in videos
        if "audience" in path.relative_to(OUTPUTS_DIR).as_posix().lower()
    ]
    return audience_videos or videos


def latest_audience_output_video(min_mtime_ns: int | None = None) -> Path | None:
    """Return the newest DexSim video created after the requested time."""
    latest_path: Path | None = None
    latest_mtime = -1
    for path in _collect_audience_output_videos():
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if min_mtime_ns is not None and mtime < min_mtime_ns:
            continue
        if mtime > latest_mtime:
            latest_path = path
            latest_mtime = mtime
    return latest_path
