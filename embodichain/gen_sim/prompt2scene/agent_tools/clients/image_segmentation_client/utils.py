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
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.schemas import (
    ImageSegmentationCandidate,
)
from embodichain.gen_sim.prompt2scene.utils.log import log_info

__all__ = [
    "apply_mask_to_alpha",
    "bbox_iou",
    "decode_rle_mask",
    "draw_labeled_bboxes",
    "draw_numbered_bboxes",
    "draw_numbered_masks",
    "is_usable_segmentation_candidate",
    "save_candidate_rgba_and_mask",
    "sort_segments_by_bbox",
]


def decode_rle_mask(mask_rle: dict[str, Any]) -> Image.Image:
    """Decode an uncompressed SAM3 RLE mask into a grayscale PIL image."""
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not _is_size_pair(size):
        raise ValueError("SAM3 mask_rle requires size=[height, width].")
    if not isinstance(counts, list):
        raise ValueError("SAM3 mask_rle counts must be an uncompressed list.")

    height = int(size[0])
    width = int(size[1])
    expected_pixels = height * width
    starts_with = int(mask_rle.get("starts_with", 0))
    value = 255 if starts_with else 0
    pixels = bytearray(expected_pixels)
    offset = 0

    for count_value in counts:
        count = int(count_value)
        if count < 0:
            raise ValueError("SAM3 mask_rle counts must be non-negative.")
        next_offset = offset + count
        if next_offset > expected_pixels:
            raise ValueError("SAM3 mask_rle counts exceed the expected image size.")
        if value:
            pixels[offset:next_offset] = b"\xff" * count
        offset = next_offset
        value = 0 if value else 255

    if offset != expected_pixels:
        raise ValueError(
            "SAM3 mask_rle counts do not cover the expected image size: "
            f"{offset} != {expected_pixels}."
        )
    return Image.frombytes("L", (width, height), bytes(pixels))


def apply_mask_to_alpha(
    image_path: str | Path,
    mask: Image.Image,
) -> Image.Image:
    """Return an RGBA image whose alpha channel is the provided mask."""
    image = Image.open(image_path).convert("RGBA")
    alpha = mask.convert("L")
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.Resampling.NEAREST)
    image.putalpha(alpha)
    return image


def save_candidate_rgba_and_mask(
    *,
    image_path: str | Path,
    candidate: ImageSegmentationCandidate,
    output_dir: str | Path,
    prefix: str | None = None,
) -> dict[str, str]:
    """Save one candidate's mask image and RGBA image for SAM3D input."""
    if candidate.mask_rle is None:
        raise ValueError(f"Candidate {candidate.candidate_id} has no mask_rle.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = prefix or candidate.candidate_id
    mask_path = output_dir / f"{filename_prefix}_mask.png"
    rgba_path = output_dir / f"{filename_prefix}_rgba.png"

    mask = decode_rle_mask(candidate.mask_rle)
    mask.save(mask_path)
    rgba = apply_mask_to_alpha(image_path, mask)
    rgba.save(rgba_path)
    log_info(f"SAM3 mask written: {mask_path}")
    log_info(f"SAM3 RGBA image written: {rgba_path}")
    return {
        "mask_path": str(mask_path),
        "rgba_path": str(rgba_path),
    }


def draw_numbered_bboxes(
    *,
    image_path: str | Path,
    segments: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Draw numbered bounding boxes for visual segmentation verification."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_label_font(image.width)
    for index, segment in enumerate(segments, start=1):
        _draw_bbox_label(
            draw=draw,
            bbox_xyxy=segment["bbox_xyxy"],
            label=str(index),
            font=font,
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def draw_numbered_masks(
    *,
    image_path: str | Path,
    segments: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Draw numbered segmentation masks for visual segmentation verification."""
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    font = _load_label_font(image.width)
    colors = [
        (255, 64, 64, 110),
        (64, 160, 255, 110),
        (64, 220, 120, 110),
        (255, 190, 64, 110),
        (190, 96, 255, 110),
        (255, 96, 190, 110),
    ]

    for index, segment in enumerate(segments, start=1):
        mask_rle = segment.get("mask_rle")
        if mask_rle is None:
            continue
        mask = decode_rle_mask(mask_rle)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        color = colors[(index - 1) % len(colors)]
        color_layer = Image.new("RGBA", image.size, color)
        transparent = Image.new("RGBA", image.size)
        overlay.alpha_composite(Image.composite(color_layer, transparent, mask))
        _draw_mask_label(
            draw=draw_overlay,
            segment=segment,
            mask=mask,
            label=str(index),
            font=font,
        )

    result = Image.alpha_composite(image, overlay).convert("RGB")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def draw_labeled_bboxes(
    *,
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Draw labeled bounding boxes for final segmentation visualization."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_label_font(image.width)
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        label = str(box["label"])
        _draw_bbox_label(
            draw=draw,
            bbox_xyxy=[x1, y1, x2, y2],
            label=label,
            font=font,
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def sort_segments_by_bbox(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort segments by top-left image position, then by descending score."""
    return sorted(
        segments,
        key=lambda segment: (
            float(segment["bbox_xyxy"][1]),
            float(segment["bbox_xyxy"][0]),
            -float(segment["score"]),
        ),
    )


def bbox_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute IoU for two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def is_usable_segmentation_candidate(
    candidate: ImageSegmentationCandidate,
) -> bool:
    """Return whether a candidate has the fields needed by downstream stages."""
    return candidate.mask_rle is not None and len(candidate.bbox_xyxy) == 4


def _is_size_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
    )


def _load_label_font(image_width: int) -> ImageFont.ImageFont:
    font_size = max(24, image_width // 80)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _draw_bbox_label(
    *,
    draw: ImageDraw.ImageDraw,
    bbox_xyxy: list[float],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    draw.rectangle((x1, y1, x2, y2), outline="red", width=6)
    label_box = draw.textbbox((x1, y1), label, font=font)
    padding = 8
    draw.rectangle(
        (
            label_box[0] - padding,
            label_box[1] - padding,
            label_box[2] + padding,
            label_box[3] + padding,
        ),
        fill="red",
    )
    draw.text((x1, y1), label, fill="white", font=font)


def _draw_mask_label(
    *,
    draw: ImageDraw.ImageDraw,
    segment: dict[str, Any],
    mask: Image.Image,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = mask.getbbox()
    if bbox is None:
        x1, y1, x2, y2 = segment["bbox_xyxy"]
        x = float(x1 + x2) * 0.5
        y = float(y1 + y2) * 0.5
    else:
        x1, y1, x2, y2 = bbox
        x = float(x1 + x2) * 0.5
        y = float(y1 + y2) * 0.5

    label_box = draw.textbbox((x, y), label, font=font)
    padding = 8
    draw.rectangle(
        (
            label_box[0] - padding,
            label_box[1] - padding,
            label_box[2] + padding,
            label_box[3] + padding,
        ),
        fill="red",
    )
    draw.text((x, y), label, fill="white", font=font)
