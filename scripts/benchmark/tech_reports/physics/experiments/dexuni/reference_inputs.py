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
"""Verify published runtime inputs independently of optional recordings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import ASSETS

__all__ = ["verify_reference_inputs", "verify_reference"]
_INDEX = ASSETS / "runtime_checks.json"


def verify_reference_inputs(root: Path) -> None:
    """Check required input and reference bytes against the reviewed snapshot.

    Args:
        root: Restored task directory, retaining its published folder name.

    Raises:
        ValueError: The task is unknown or a required file has changed.
        FileNotFoundError: Required runtime data has not been restored.

    Video files are optional evidence and are never needed for simulation.
    This check runs during preparation, outside timed simulation regions.
    """
    tasks = json.loads(_INDEX.read_text())
    if root.name not in tasks:
        raise ValueError(f"Unknown published reference: {root.name}")
    for relative, expected in tasks[root.name]["files"].items():
        path = root / relative
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if digest != expected:
            raise ValueError(f"Published runtime input changed: {relative}")


def verify_reference(root: Path, video_sha256: str) -> None:
    """Validate local recording evidence or the published runtime snapshot.

    Args:
        root: Reference run directory.
        video_sha256: Original accepted recording identity retained as provenance.

    Existing local campaigns keep their recording check. Public input-only
    restores instead verify every required input against the reviewed manifest.
    """
    video = root / "preview/dexsim.mp4"
    if video.is_file():
        if hashlib.sha256(video.read_bytes()).hexdigest() != video_sha256:
            raise ValueError("Approved native preview changed")
    else:
        verify_reference_inputs(root)
