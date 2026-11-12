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
"""Image editing for reconstruction; source observations remain unchanged."""

from __future__ import annotations
from pathlib import Path
from typing import Protocol
import math
import requests
from PIL import Image
from ._artifacts import digest, write_json

__all__ = ["ImageEditor", "HttpImageEditor", "edit_scene_image"]


class ImageEditor(Protocol):
    """Caller-configured image editing capability, independent of model vendor."""

    def edit(self, image: Path, prompt: str, output: Path) -> None:
        """Write an edited image.

        Args:
            image: Source image accepted by the provider.
            prompt: Reconstruction cleanup constraints.
            output: Destination image file.
        """
        ...


class HttpImageEditor:
    """Invoke the existing Qwen image-edit HTTP service without automatic retries."""

    def __init__(self, base_url: str, *, timeout_s: float = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def edit(self, image: Path, prompt: str, output: Path) -> None:
        """Send one image edit; ambiguous POST failures are returned to the caller.

        Args:
            image: Provider-sized PNG input.
            prompt: Image-edit instruction.
            output: Destination for the returned PNG bytes.
        """
        with Image.open(image) as opened:
            width, height = opened.size
        with requests.Session() as session:
            session.trust_env = False
            with image.open("rb") as handle:
                response = session.post(
                    self.base_url + "/generate_image_by_image",
                    files={"image": (image.name, handle, "image/png")},
                    data={
                        "prompt": prompt,
                        "width": str(width),
                        "height": str(height),
                        "seed": "42",
                    },
                    timeout=(15, self.timeout_s),
                )
            response.raise_for_status()
            output.write_bytes(response.content)


_SCENE_PROMPT = """Remove both robot arms and their complete grippers, including
fingers, jaws, linkages and attached cables. Reconstruct the table where they were.
Keep the original worktable and non-robot objects against a plain gray background.
Reduce shadows, glare and texture noise using diffuse lighting.
Preserve camera framing, tabletop outline and object positions, shapes, colors and openings.
Retain held objects. Do not add or move objects or perform the task.
"""


def edit_scene_image(
    source: Path,
    output: Path,
    editor: ImageEditor,
    *,
    prompt: str = _SCENE_PROMPT,
) -> Path:
    """Remove robot/background and lighting clutter, preserving image coordinates.

    The service receives padded 32-aligned input; output padding is removed and
    resized back to the original pixel grid for calibrated RGB-D registration.
    Generated hidden surfaces remain estimates, never observation evidence.

    Args:
        source: Original or observation-recovered image.
        output: New directory for edit inputs, outputs and provenance.
        editor: Caller-provided image-edit service.
        prompt: Image-edit instruction; override to compare preprocessing models or styles.

    Returns:
        Cleaned reconstruction image path, restored to the source pixel grid.
    """
    output.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    # Low-resolution robot recordings lose small task objects when edited at
    # 320 px. Use the model's normal image scale, then restore the source grid.
    service_long_side = min(1024, max(768, max(image.size)))
    scale = service_long_side / max(image.size)
    resized_size = tuple(max(1, round(side * scale)) for side in image.size)
    canvas_size = tuple(max(256, math.ceil(side / 32) * 32) for side in resized_size)
    offset = tuple(
        (large - small) // 2 for large, small in zip(canvas_size, resized_size)
    )
    canvas = Image.new("RGB", canvas_size, (128, 128, 128))
    canvas.paste(image.resize(resized_size, Image.Resampling.LANCZOS), offset)
    prepared = output / "edit_input.png"
    canvas.save(prepared)
    (output / "prompt.txt").write_text(prompt)
    receipt = dict(
        source_sha256=digest(source),
        source_size=list(image.size),
        service_size=list(canvas_size),
        padding_offset_xy=list(offset),
        resized_size=list(resized_size),
        reconstruction_only=True,
        hidden_surfaces="generative estimates",
        status="submitting",
    )
    write_json(output / "receipt.json", receipt)
    generated = output / "generated.png"
    try:
        editor.edit(prepared, prompt, generated)
        with Image.open(generated) as edited:
            if edited.size != canvas_size:
                raise ValueError("Image service returned unexpected dimensions")
            cropped = edited.convert("RGB").crop(
                (
                    offset[0],
                    offset[1],
                    offset[0] + resized_size[0],
                    offset[1] + resized_size[1],
                )
            )
            target = output / "scene_input.png"
            cropped.resize(image.size, Image.Resampling.LANCZOS).save(target)
    except Exception as error:
        receipt.update(status="failed_or_unknown", error=str(error))
        write_json(output / "receipt.json", receipt)
        raise
    receipt.update(status="completed", output_sha256=digest(target))
    write_json(output / "receipt.json", receipt)
    return target
