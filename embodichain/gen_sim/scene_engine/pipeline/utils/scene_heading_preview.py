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

from collections.abc import Mapping
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageOps

from embodichain.gen_sim._blender_runtime import _run_blender_job
from embodichain.gen_sim.scene_engine.pipeline.utils._image_segmentation_utils import (
    _load_label_font_at_size,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
)

__all__: list[str] = []

SCENE_HEADING_CANDIDATES_FILENAME = "scene_heading_candidates.png"
_VIEW_SIZE = 768
_HEADING_ANGLES = (0, 90, 180, 270)


def render_scene_heading_comparison(
    *,
    image_path: Path,
    geometry_root: Path,
    output_path: Path,
    candidate_layouts: Mapping[int, list[dict[str, object]]] | None = None,
    baseline_layouts: list[dict[str, object]] | None = None,
    flipped_layouts: list[dict[str, object]] | None = None,
    proposed_yaw_degrees: int | None = None,
) -> Path:
    """Save shared-camera heading candidates and their reference comparison.

    Four-angle candidates use a two-by-two sheet and label the program proposal
    when supplied. The original RGB is placed beside the sheet in the full
    comparison. The baseline/flipped pair uses a horizontal layout.
    """
    if candidate_layouts is None:
        if baseline_layouts is None or flipped_layouts is None:
            raise ValueError("Scene heading comparison requires candidate layouts.")
        angles = (0, 180)
        layouts = [baseline_layouts, flipped_layouts]
    else:
        if set(candidate_layouts) != set(_HEADING_ANGLES):
            raise ValueError("Scene heading candidates must cover 0, 90, 180 and 270.")
        angles = _HEADING_ANGLES
        layouts = [candidate_layouts[angle] for angle in angles]
    object_ids = _layout_ids(layouts[0])
    if any(_layout_ids(candidate) != object_ids for candidate in layouts[1:]):
        raise ValueError("Scene heading candidates must use the same ordered ids.")

    with Image.open(image_path) as source:
        reference = ImageOps.contain(
            ImageOps.exif_transpose(source).convert("RGB"), (_VIEW_SIZE, _VIEW_SIZE)
        )
    with tempfile.TemporaryDirectory(prefix="embodichain-heading-views-") as directory:
        candidate_paths = [Path(directory) / f"{angle}.png" for angle in angles]
        transforms = [
            [layout_object_to_transform_matrix(layout).tolist() for layout in candidate]
            for candidate in layouts
        ]
        transform_payload = (
            {"baseline_transforms": transforms[0], "flipped_transforms": transforms[1]}
            if candidate_layouts is None
            else {"candidate_transforms": transforms}
        )
        _run_blender_job(
            "scene_heading",
            {
                "glb_paths": [
                    str((geometry_root / f"{object_id}.glb").resolve())
                    for object_id in object_ids
                ],
                **transform_payload,
                "output_paths": [str(path) for path in candidate_paths],
                "resolution": _VIEW_SIZE,
            },
            timeout=240.0,
        )
        candidates = []
        for path in candidate_paths:
            with Image.open(path) as rendered:
                candidates.append(rendered.convert("RGB"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_panels = [
        (
            (
                f"{angle}\u00b0 | Program proposal"
                if angle == proposed_yaw_degrees
                else f"{angle}\u00b0"
            ),
            rendered,
        )
        for angle, rendered in zip(angles, candidates, strict=True)
    ]
    candidate_sheet = _compose_heading_panels(
        candidate_panels, columns=2 if candidate_layouts is not None else None
    )
    candidate_sheet.save(
        output_path.with_name(SCENE_HEADING_CANDIDATES_FILENAME), format="PNG"
    )
    if candidate_layouts is None:
        comparison = _compose_heading_panels(
            [("Original image", reference), *candidate_panels]
        )
    else:
        reference_panel = _compose_heading_panels([("Original image", reference)])
        comparison = Image.new(
            "RGB", (_VIEW_SIZE + candidate_sheet.width, candidate_sheet.height), "white"
        )
        comparison.paste(
            reference_panel, (0, (comparison.height - reference_panel.height) // 2)
        )
        comparison.paste(candidate_sheet, (_VIEW_SIZE, 0))
    comparison.save(output_path, format="PNG")
    return output_path


def _layout_ids(layouts: list[dict[str, object]]) -> list[str]:
    ids = []
    for layout in layouts:
        object_id = layout.get("id")
        if (
            not isinstance(object_id, str)
            or not object_id
            or Path(object_id).name != object_id
        ):
            raise ValueError("Scene heading comparison requires valid mesh ids.")
        ids.append(object_id)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Scene heading comparison requires distinct mesh ids.")
    return ids


def _compose_heading_panels(
    panels: list[tuple[str, Image.Image]], *, columns: int | None = None
) -> Image.Image:
    header_height = 40
    columns = len(panels) if columns is None else columns
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * _VIEW_SIZE, rows * (_VIEW_SIZE + header_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = _load_label_font_at_size(20)
    for index, (title, panel) in enumerate(panels):
        x = (index % columns) * _VIEW_SIZE
        y = (index // columns) * (_VIEW_SIZE + header_height)
        draw.text((x + 12, y + 10), title, fill="#17232e", font=font)
        canvas.paste(
            panel,
            (
                x + (_VIEW_SIZE - panel.width) // 2,
                y + header_height + (_VIEW_SIZE - panel.height) // 2,
            ),
        )
    return canvas
