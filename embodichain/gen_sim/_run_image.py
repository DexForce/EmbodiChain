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

"""Select the grounded image for a GenSim run and preserve cleanup provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .services.runtime import GenSimServices

__all__: list[str] = []


def _find_previous_cleanup(
    task: Path, plan: dict[str, Any], camera: dict[str, str]
) -> Path:
    """Locate the preprocessing snapshot that produced a reused Task Plan."""
    root = task.parent
    try:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        manifest = {}
    if (
        not isinstance(manifest, dict)
        or task.name != "task"
        or manifest.get("schema_version") != "embodichain.gensim-run/v1"
        or manifest.get("source_image_sha256") != camera["sha256"]
        or manifest.get("image_sha256") != plan["image_sha256"]
        or manifest.get("plan_digest") != plan["plan_digest"]
        or not isinstance(manifest.get("stages"), dict)
        or manifest.get("stages", {}).get("image_preprocessing")
        not in {"succeeded", "reused"}
        or not isinstance(manifest.get("image_preprocessing"), dict)
        or manifest.get("image_preprocessing", {}).get("view") != camera["view"]
    ):
        raise ValueError(
            "The reused Task Plan has no matching cleanup snapshot for this raw image. "
            "Use the prior run's task directory with its preprocessing artifacts, "
            "or pass the Task Plan's exact image without input_camera."
        )
    return root / "image_preprocessing"


def _prepare_image(
    source: Path,
    output: Path,
    services: GenSimServices,
    *,
    previous_cleanup: Path | None,
    expected_sha256: str | None,
) -> Path:
    """Publish a fresh cleanup or copy and verify the previously grounded bytes."""
    if previous_cleanup is None:
        from ._robot_cleanup import clean_robot_image

        clean_robot_image(source, output, services=services)
        return output / "input.png"

    # Copy only the outputs owned by the standalone tool into the new run.
    output.mkdir()
    for name in (
        "original.png",
        "input.png",
        "robot-mask.png",
        "arms-mask.png",
        "grippers-mask.png",
        "camera-support-mask.png",
        "editor-response.png",
        "preparation.json",
    ):
        shutil.copyfile(previous_cleanup / name, output / name)
    cleaned = output / "input.png"
    if hashlib.sha256(cleaned.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(
            "Reused cleanup image does not match the Task Plan's image hash"
        )

    import numpy as np
    from PIL import Image

    with (
        Image.open(source) as raw,
        Image.open(output / "original.png") as original,
        Image.open(cleaned) as edited,
        Image.open(output / "robot-mask.png") as mask,
    ):
        if not raw.size == original.size == edited.size == mask.size:
            raise ValueError("Reused cleanup images and mask have different dimensions")
        source_rgb = np.asarray(raw.convert("RGB"))
        if not np.array_equal(source_rgb, np.asarray(original.convert("RGB"))):
            raise ValueError("Reused cleanup original differs from the run's raw image")
        changed = np.any(source_rgb != np.asarray(edited.convert("RGB")), axis=2)
        inside = np.asarray(mask.convert("L")) != 0

    report = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or not changed.any()
        or np.any(changed & ~inside)
        or report.get("outside_mask_pixels_checked") != int((~inside).sum())
        or report.get("outside_mask_pixels_changed") != 0
        or report.get("inside_mask_pixels_changed") != int((changed & inside).sum())
    ):
        raise ValueError("Reused cleanup failed the pixel audit")
    return cleaned
