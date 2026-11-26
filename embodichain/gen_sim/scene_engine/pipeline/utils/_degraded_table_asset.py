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

import math
from pathlib import Path
from typing import Protocol

from PIL import Image

from embodichain.gen_sim.scene_engine.core.scene_graph import TABLE_OBJECT_ID
from embodichain.gen_sim.scene_engine.pipeline.utils._image_segmentation_utils import (
    build_mask_candidates,
    decode_rle_mask,
    invert_mask_if_foreground_is_off_center,
)
from embodichain.gen_sim.services.compat import (
    GeometryClient,
    SegmentationClient,
    reconstruct,
)
from embodichain.gen_sim.services.contracts import GeometryRequest
from embodichain.gen_sim.services.errors import CapabilityError

__all__ = ["generate_degraded_table_asset"]

_REFERENCE_MATTE_RGB = (128, 128, 128)
_EDIT_SIZE = (768, 768)
_EDIT_SEED = 0


class _TableImageEditor(Protocol):
    def generate_image_by_image(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        size: tuple[int, int],
        seed: int,
    ) -> Path: ...


def generate_degraded_table_asset(
    *,
    description: str,
    support_width_depth: tuple[float, float],
    reference_image_path: Path,
    output_root: Path,
    image_generation_client: _TableImageEditor,
    image_segmentation_client: SegmentationClient,
    geometry_generation_client: GeometryClient,
) -> Path:
    """Complete one table using its isolated observed RGBA tabletop crop.

    Args:
        description: Observed table description from scene understanding.
        support_width_depth: Width and depth of the old horizontal support bounds.
        reference_image_path: Masked RGBA crop of the observed table.
        output_root: Directory for the edit input, prompt, output, mask and geometry.
        image_generation_client: Borrowed image-to-image client.
        image_segmentation_client: Borrowed semantic mask client.
        geometry_generation_client: Borrowed masked-image reconstruction client.

    Returns:
        Raw standalone GLB in asset-local coordinates. Its scene fit is
        determined later from the old tabletop geometry.
    """
    description = description.strip()
    width, depth = support_width_depth
    if not description or not all(
        math.isfinite(span) and span > 0 for span in (width, depth)
    ):
        raise ValueError("Table recovery requires a description and positive spans.")
    edit_image = getattr(image_generation_client, "generate_image_by_image", None)
    if not callable(edit_image):
        raise CapabilityError(
            "Degraded table recovery requires an image-to-image editing service."
        )
    prompt = (
        "Use the supplied masked tabletop crop as the appearance reference. "
        "Complete this same table as a product photograph of exactly one freestanding "
        "table. Match the observed tabletop color, material, finish, and texture, "
        "including any visible grain direction and pattern. Prefer the observed "
        "appearance if it differs from the description. Neutral gray masked areas "
        "are unobserved pixels. Continue the observed material into missing tabletop "
        "areas. Below the flat tabletop, add a plausible full-height base and sturdy "
        "legs compatible with the observed tabletop. "
        f"The tabletop's horizontal width-to-depth ratio is about {width / depth:.2f}. "
        "Show the entire table and its supporting structure from a clear three-quarter "
        "view, centered with space around every edge against a plain contrasting "
        "studio background. Keep the tabletop empty and omit other objects, robots, "
        "people, text, prominent cast shadows, and cropping. "
        f"The observed table was described as follows. {description}"
    )
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_path = output_root / "reference_table.png"
    _save_rgb_reference(reference_image_path, reference_path)
    (output_root / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    image_path = edit_image(
        image_path=reference_path,
        prompt=prompt,
        output_path=output_root / "generated_table.png",
        size=_EDIT_SIZE,
        seed=_EDIT_SEED,
    )
    mask_path = _isolate_generated_table(
        image_path=image_path,
        output_path=output_root / "generated_table_mask.png",
        client=image_segmentation_client,
    )
    geometry_root = output_root / "geometry"
    geometry_root.mkdir(exist_ok=True)
    result = reconstruct(
        geometry_generation_client,
        GeometryRequest(image_path, ((TABLE_OBJECT_ID, mask_path),), geometry_root),
    )
    if len(result.objects) != 1:
        raise ValueError("Table recovery requires exactly one reconstructed asset.")
    mesh_path = result.objects[0].mesh_path
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Table recovery did not produce a GLB: {mesh_path}")
    return mesh_path


def _save_rgb_reference(source_path: Path, output_path: Path) -> None:
    """Preserve visible surface pixels and cover the masked background."""
    with Image.open(source_path) as source:
        if "A" not in source.getbands():
            raise ValueError(
                "The tabletop reference must have a visible-pixel alpha mask."
            )
        rgba = source.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getbbox() is None:
        raise ValueError("The tabletop reference mask has no visible pixels.")
    matte = Image.new("RGB", rgba.size, _REFERENCE_MATTE_RGB)
    Image.composite(rgba.convert("RGB"), matte, alpha).save(output_path)


def _isolate_generated_table(
    *, image_path: Path, output_path: Path, client: SegmentationClient
) -> Path:
    """Keep the largest valid foreground in the isolated table image."""
    with Image.open(image_path) as image:
        image_size = image.size
    pixel_count = math.prod(image_size)
    for prompt in ("table", "freestanding table with legs"):
        best_mask = None
        best_area = 0
        for candidate in build_mask_candidates(
            client.segment_single_object(image_path=image_path, prompt=prompt)
        ):
            # Isolated generation centers the table. Correct a background mask
            # before comparing its area with the possible whole-table masks.
            if max(image_size) >= 6:
                candidate = invert_mask_if_foreground_is_off_center(candidate)
            mask = decode_rle_mask(candidate.mask_rle)
            if mask.size != image_size:
                raise ValueError(
                    "Generated table mask dimensions must match its image."
                )
            area = mask.histogram()[255]
            if best_area < area < pixel_count:
                best_mask, best_area = mask, area
        if best_mask is not None:
            best_mask.save(output_path)
            return output_path
    raise ValueError("Generated table has no usable foreground mask.")
