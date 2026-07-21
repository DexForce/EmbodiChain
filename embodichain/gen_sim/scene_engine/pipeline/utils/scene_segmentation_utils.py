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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class MaskCandidate:
    """One numbered mask candidate returned by the Image Segmentation Server."""

    index: int
    mask_rle: dict[str, Any]


def build_mask_candidates(mask_rles: list[dict[str, Any]]) -> list[MaskCandidate]:
    return [
        MaskCandidate(index=index, mask_rle=mask_rle)
        for index, mask_rle in enumerate(mask_rles, start=1)
    ]


def decode_rle_mask(mask_rle: dict[str, Any]) -> Image.Image:
    """Decode an uncompressed RLE mask into a binary image."""

    # Check the return value's format.
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(value, int) and value > 0 for value in size)
    ):
        raise ValueError("Image Segmentation Server RLE needs size=[height, width].")
    if not isinstance(counts, list):
        raise ValueError("Image Segmentation Server RLE counts must be a list.")

    height, width = size
    pixel_count = height * width
    starts_with = mask_rle.get("starts_with", 0)
    if starts_with not in (0, 1, False, True):
        raise ValueError("Image Segmentation Server RLE starts_with must be 0 or 1.")

    pixels = bytearray(pixel_count)
    is_foreground = bool(
        starts_with
    )  # True for white foreground, False for black background.
    offset = 0  # How many pixels have been filled so far.
    for raw_count in counts:
        if isinstance(raw_count, bool):
            raise ValueError(
                "Image Segmentation Server RLE counts must contain integers."
            )
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Image Segmentation Server RLE counts must contain integers."
            ) from exc
        if count < 0 or offset + count > pixel_count:
            raise ValueError(
                "Image Segmentation Server RLE counts do not match its declared size."
            )
        if is_foreground:
            pixels[offset : offset + count] = (
                b"\xff" * count
            )  # Write white pixels for the foreground.
        offset += count
        is_foreground = not is_foreground

    if offset != pixel_count:
        raise ValueError("Image Segmentation Server RLE does not cover the image.")
    return Image.frombytes("L", (width, height), bytes(pixels))


def union_overlapping_mask_candidates(
    candidates: list[MaskCandidate],
    *,
    min_iou: float = 0.8,
) -> list[MaskCandidate]:
    """Union candidate masks with IOU >= min_iou into one mask candidate."""
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be greater than 0 and at most 1.")
    if not candidates:
        return []

    masks = [decode_rle_mask(candidate.mask_rle) for candidate in candidates]
    image_size = masks[0].size
    for mask in masks:
        _require_image_size(mask, image_size)

    parents = list(
        range(len(candidates))
    )  # Initialize the Union-Find data structure for candidates.
    for first_index, first_mask in enumerate(masks):
        for second_index in range(first_index + 1, len(masks)):
            if _mask_iou(first_mask, masks[second_index]) >= min_iou:
                _union_parent(
                    parents, first_index, second_index
                )  # Union the two candidates into one.

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        # Put all the index of the same parent into one group.
        grouped_indices.setdefault(_find_parent(parents, index), []).append(index)

    merged_candidates: list[MaskCandidate] = []
    for merged_index, member_indices in enumerate(grouped_indices.values(), start=1):
        merged_mask = masks[member_indices[0]]
        for member_index in member_indices[1:]:
            # Union the masks of the same group into one mask (lighter = union).
            merged_mask = ImageChops.lighter(merged_mask, masks[member_index])
        merged_candidates.append(
            MaskCandidate(
                index=merged_index,
                mask_rle=_encode_binary_mask_rle(merged_mask),
            )
        )
    return merged_candidates


def save_binary_mask(
    candidate: MaskCandidate,
    *,
    image_size: tuple[int, int],
    output_path: str | Path,
) -> Path:
    """Save one candidate as a white-foreground, black-background PNG mask."""
    mask = decode_rle_mask(candidate.mask_rle)
    _require_image_size(mask, image_size)  # Check whether the image size == mask size.

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(resolved_output_path)
    return resolved_output_path


def render_image_without_masks(
    *,
    image_path: str | Path,
    mask_paths: list[str | Path],
    output_path: str | Path,
    removed_color: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Replace all the other masks with gray color."""
    image = Image.open(image_path).convert("RGB")
    ignored_mask = Image.new("L", image.size, 0)
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("L")
        _require_image_size(mask, image.size)
        ignored_mask = ImageChops.lighter(ignored_mask, mask)

    removed_layer = Image.new("RGB", image.size, removed_color)
    result = Image.composite(removed_layer, image, ignored_mask)
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(resolved_output_path)
    return resolved_output_path


def render_numbered_mask_candidates(
    *,
    image_path: str | Path,
    candidates: list[MaskCandidate],
    output_path: str | Path,
    mask_style: str = "fill",
) -> Path:
    """Overlay numbered mask candidates on their source image.
    Notice that:
        - mask_style can be either "fill" or "outline".
        - The label font and its background scale with the source image resolution.
    """
    if mask_style not in {"fill", "outline"}:
        raise ValueError("mask_style must be 'fill' or 'outline'.")

    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    colors = (
        (239, 83, 80, 160),
        (66, 165, 245, 160),
        (102, 187, 106, 160),
        (255, 202, 40, 160),
        (171, 71, 188, 160),
        (38, 198, 218, 160),
    )

    decoded_masks: list[tuple[MaskCandidate, Image.Image]] = []
    for candidate in candidates:
        mask = decode_rle_mask(candidate.mask_rle)
        _require_image_size(mask, image.size)
        decoded_masks.append((candidate, mask))
        color_layer = Image.new(
            "RGBA", image.size, colors[(candidate.index - 1) % len(colors)]
        )
        transparent_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        rendered_mask = (  # If weuse outline, then need to do some another processings.
            mask if mask_style == "fill" else _mask_outer_outline(mask, image.size)
        )
        overlay.alpha_composite(
            Image.composite(color_layer, transparent_layer, rendered_mask)
        )

    draw = ImageDraw.Draw(overlay)  # Initialize a draw object.
    font = _load_label_font(image.size)
    for candidate, mask in decoded_masks:
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError(
                f"Image Segmentation Server candidate {candidate.index} has an empty mask."
            )
        _draw_number_label(
            draw=draw,
            label=str(candidate.index),
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            font=font,
        )

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(resolved_output_path)
    return resolved_output_path


def _require_image_size(mask: Image.Image, image_size: tuple[int, int]) -> None:
    if mask.size != image_size:
        raise ValueError(
            "Image Segmentation Server mask size does not match the input image: "
            f"{mask.size} != {image_size}."
        )


def _mask_outer_outline(mask: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    """Use dilation and subtraction to get the outer outline of a binary mask."""
    outline_width = max(1, round(min(image_size) / 400))
    dilated_mask = mask.filter(ImageFilter.MaxFilter(outline_width * 2 + 1))
    return ImageChops.subtract(dilated_mask, mask)


def _mask_iou(first_mask: Image.Image, second_mask: Image.Image) -> float:
    """Compute the Intersection over Union (IoU) of two binary masks."""
    _require_image_size(second_mask, first_mask.size)
    intersection = ImageChops.multiply(first_mask, second_mask)
    union = ImageChops.lighter(first_mask, second_mask)
    union_pixels = union.histogram()[255]
    if union_pixels == 0:
        return 0.0
    return intersection.histogram()[255] / union_pixels


def _encode_binary_mask_rle(mask: Image.Image) -> dict[str, Any]:
    binary_mask = mask.convert("L").point(
        lambda value: 255 if value else 0
    )  # Force translate an image into a binary mask.
    width, height = binary_mask.size
    counts: list[int] = []
    current_value = 0
    run_length = 0
    for value in binary_mask.tobytes():
        value = 255 if value else 0
        if value == current_value:
            run_length += 1
            continue
        counts.append(run_length)
        current_value = value
        run_length = 1
    counts.append(run_length)
    return {
        "size": [height, width],
        "counts": counts,
        "starts_with": 0,
    }


def _find_parent(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union_parent(parents: list[int], first_index: int, second_index: int) -> None:
    first_root = _find_parent(parents, first_index)
    second_root = _find_parent(parents, second_index)
    if first_root != second_root:
        parents[second_root] = first_root


def _load_label_font(image_size: tuple[int, int]) -> ImageFont.ImageFont:
    font_size = max(16, round(min(image_size) / 32))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _draw_number_label(
    *,
    draw: ImageDraw.ImageDraw,
    label: str,
    center: tuple[float, float],
    font: ImageFont.ImageFont,
) -> None:
    """Draw a numbered label with red background and white text at the given center position."""
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    padding = max(4, round(max(label_width, label_height) / 4))
    x = center[0] - label_width / 2
    y = center[1] - label_height / 2
    draw.rectangle(
        (
            x - padding,
            y - padding,
            x + label_width + padding,
            y + label_height + padding,
        ),
        fill=(220, 0, 0, 255),
        outline=(255, 255, 255, 255),
        width=max(1, padding // 3),
    )
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)
