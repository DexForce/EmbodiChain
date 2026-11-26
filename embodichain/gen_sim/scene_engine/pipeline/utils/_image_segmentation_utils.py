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
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from embodichain.utils.logger import log_warning

__all__ = [
    "MaskCandidate",
    "build_mask_candidates",
    "decode_rle_mask",
    "invert_mask_if_foreground_is_off_center",
    "union_overlapping_mask_candidates",
    "save_binary_mask",
    "save_visible_rgba_crop",
    "render_image_without_masks",
    "render_numbered_mask_candidates",
    "render_asset_mask_id_overlay",
]


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


def invert_mask_if_foreground_is_off_center(candidate: MaskCandidate) -> MaskCandidate:
    """Invert a mask when its foreground is less concentrated at image center.

    This heuristic is intended for generated single-object images, where the
    object is expected near the center and SAM3 may return its background.
    """
    mask = decode_rle_mask(candidate.mask_rle)
    width, height = mask.size
    left, top = width // 6, height // 6
    right, bottom = width - left, height - top
    center_mask = mask.crop((left, top, right, bottom))

    center_foreground_ratio = _foreground_ratio(center_mask)
    total_foreground_pixels = _foreground_pixel_count(mask)
    outside_foreground_pixels = total_foreground_pixels - _foreground_pixel_count(
        center_mask
    )
    outside_pixel_count = width * height - center_mask.width * center_mask.height
    outside_foreground_ratio = outside_foreground_pixels / outside_pixel_count
    if center_foreground_ratio >= outside_foreground_ratio:
        return candidate

    inverted_mask = mask.point(lambda value: 0 if value else 255)
    return MaskCandidate(
        index=candidate.index,
        mask_rle=_encode_binary_mask_rle(inverted_mask),
    )


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


def save_visible_rgba_crop(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    output_size: tuple[int, int] = (512, 512),
    padding_ratio: float = 0.1,
) -> Path:
    """Save a fixed-size RGBA crop of one object's visible source-image pixels."""
    if len(output_size) != 2 or not all(
        isinstance(value, int) and value > 0 for value in output_size
    ):
        raise ValueError("RGBA crop output_size must contain two positive integers.")
    if padding_ratio < 0:
        raise ValueError("RGBA crop padding_ratio must be non-negative.")

    with Image.open(image_path) as loaded_image:
        image = loaded_image.convert("RGB")
    with Image.open(mask_path) as loaded_mask:
        mask = loaded_mask.convert("L")
    _require_image_size(mask, image.size)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Cannot save an RGBA crop for an empty binary mask.")

    left, top, right, bottom = bbox
    padding = math.ceil(max(right - left, bottom - top) * padding_ratio)
    crop_bounds = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    image_crop = image.crop(crop_bounds)
    mask_crop = mask.crop(crop_bounds)
    scale = min(
        output_size[0] / image_crop.width,
        output_size[1] / image_crop.height,
    )
    resized_size = (
        max(1, round(image_crop.width * scale)),
        max(1, round(image_crop.height * scale)),
    )
    rgba_crop = image_crop.resize(resized_size, Image.Resampling.LANCZOS).convert(
        "RGBA"
    )
    rgba_crop.putalpha(mask_crop.resize(resized_size, Image.Resampling.NEAREST))

    rgba_canvas = Image.new("RGBA", output_size, (0, 0, 0, 0))
    paste_xy = (
        (output_size[0] - resized_size[0]) // 2,
        (output_size[1] - resized_size[1]) // 2,
    )
    rgba_canvas.alpha_composite(rgba_crop, dest=paste_xy)
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba_canvas.save(resolved_output_path)
    return resolved_output_path


def render_image_without_masks(
    *,
    image_path: str | Path,
    mask_paths: list[str | Path],
    output_path: str | Path,
    removed_color: tuple[int, int, int] = (128, 128, 128),
) -> tuple[Path, Image.Image]:
    """Gray masked regions and return the image path with their combined mask."""
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
    return resolved_output_path, ignored_mask


def render_numbered_mask_candidates(
    *,
    image_path: str | Path,
    candidates: list[MaskCandidate],
    output_path: str | Path,
    mask_style: str = "fill",
    label_avoid_mask: Image.Image | None = None,
    include_candidate_details: bool = False,
    detail_image_path: str | Path | None = None,
) -> Path:
    """Render candidate IDs, optionally with right-side leaders and pixel cards.

    Args:
        image_path: Source image used for the overview. It may have known assets
            grayed out when choosing a table mask.
        candidates: Masks and their selectable IDs. No candidate is removed or
            renumbered by this renderer.
        output_path: Destination PNG path.
        mask_style: Either ``"fill"`` or ``"outline"`` for the overview.
        label_avoid_mask: Regions to avoid when placing legacy in-image labels.
            Detail sheets use an external legend instead of in-image labels.
        include_candidate_details: Connect every mask to a right-side numbered
            legend and append one independently labeled card per
            mask. Included source pixels retain their RGB appearance; excluded
            regions, including holes, are uniformly gray. Cards preserve nearby
            context so missing visible parts remain apparent.
        detail_image_path: Optional unmodified source for the detail cards. Its
            dimensions must match the overview image. Defaults to ``image_path``.
            Only used when ``include_candidate_details`` is true.

    Returns:
        Absolute path to the rendered image.
    """
    if mask_style not in {"fill", "outline"}:
        raise ValueError("mask_style must be 'fill' or 'outline'.")

    if include_candidate_details:
        return _render_mask_selection_sheet(
            image_path=image_path,
            detail_image_path=detail_image_path,
            candidates=candidates,
            output_path=output_path,
            mask_style=mask_style,
            label_avoid_mask=label_avoid_mask,
        )

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
        rendered_mask = (
            mask if mask_style == "fill" else _mask_outer_outline(mask, image.size)
        )
        overlay.alpha_composite(
            Image.composite(color_layer, transparent_layer, rendered_mask)
        )

    draw = ImageDraw.Draw(overlay)  # Initialize a draw object.
    font = _load_label_font(image.size)
    if label_avoid_mask is not None:
        label_blocked_mask = label_avoid_mask.convert("L")
        _require_image_size(label_blocked_mask, image.size)
    else:
        label_blocked_mask = None
    label_occupied_mask = Image.new("L", image.size, 0)
    for candidate, mask in decoded_masks:
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError(
                f"Image Segmentation Server candidate {candidate.index} has an empty mask."
            )
        center = (
            ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if label_blocked_mask is None
            else _find_label_center_inside_bbox(
                candidate_mask=mask,
                candidate_bbox=bbox,
                blocked_mask=ImageChops.lighter(
                    label_blocked_mask, label_occupied_mask
                ),
                label=str(candidate.index),
                font=font,
            )
        )
        if center is None:
            # Keep every candidate selectable even when its AABB is entirely
            # occluded. This is preferable to silently omitting its number.
            log_warning(
                "Could not place table-candidate label %s outside masked objects "
                "while keeping it inside the candidate AABB; using the AABB center.",
                candidate.index,
            )
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        label_bounds = _number_label_bounds(
            draw=draw, label=str(candidate.index), center=center, font=font
        )
        _draw_number_label(
            draw=draw,
            label=str(candidate.index),
            center=center,
            font=font,
        )
        if label_blocked_mask is not None:
            ImageDraw.Draw(label_occupied_mask).rectangle(label_bounds, fill=255)

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(resolved_output_path)
    return resolved_output_path


_SELECTION_COLORS = (
    (239, 83, 80),
    (66, 165, 245),
    (102, 187, 106),
    (255, 202, 40),
    (171, 71, 188),
    (38, 198, 218),
)
_SELECTION_GRAY = (200, 203, 208)


def _preview_typography(image_size: tuple[int, int]) -> dict[str, int]:
    """Derive readable preview dimensions from the rendered overview, as AgenTable does."""
    number = min(40, max(24, round(max(image_size) * 0.035)))
    legend = max(18, round(number * 0.7))
    padding = number // 2
    return {
        "number": number,
        "legend": legend,
        "title": max(20, round(number * 0.8)),
        "padding": padding,
        "line_height": math.ceil(legend * 1.35),
        "line_width": max(1, round(number / 10)),
        "badge_padding": max(4, round(number / 5)),
        "card_width": 12 * number,
        "card_height": 8 * number,
    }


def _render_mask_selection_sheet(
    *,
    image_path: str | Path,
    detail_image_path: str | Path | None,
    candidates: list[MaskCandidate],
    output_path: str | Path,
    mask_style: str,
    label_avoid_mask: Image.Image | None,
) -> Path:
    """Keep all candidate IDs and show each exact included region separately."""
    indices = [candidate.index for candidate in candidates]
    if any(type(index) is not int or index < 1 for index in indices):
        raise ValueError("Mask candidate IDs must be positive integers.")
    if len(indices) != len(set(indices)):
        raise ValueError("Mask candidate IDs must be unique.")
    with Image.open(image_path) as loaded_image:
        image = loaded_image.convert("RGB")
    if detail_image_path is None:
        detail_image = image
    else:
        with Image.open(detail_image_path) as loaded_image:
            detail_image = loaded_image.convert("RGB")
        _require_image_size(detail_image, image.size)

    decoded_masks = []
    for candidate in candidates:
        mask = decode_rle_mask(candidate.mask_rle)
        _require_image_size(mask, image.size)
        if mask.getbbox() is None:
            raise ValueError(
                f"Image Segmentation Server candidate {candidate.index} has an empty mask."
            )
        decoded_masks.append((candidate, mask))

    overview_width = min(960, image.width)
    overview_size = (
        overview_width,
        max(1, round(image.height * overview_width / image.width)),
    )
    typography = _preview_typography(overview_size)
    overview = image.resize(overview_size, Image.Resampling.LANCZOS)
    overview_masks = []
    for candidate, mask in decoded_masks:
        visible_mask = mask.resize(overview_size, Image.Resampling.NEAREST)
        if visible_mask.getbbox() is None:
            # A tiny candidate may vanish in a thumbnail, but must remain
            # selectable. Preserve a single display pixel at its source anchor.
            anchor = _selection_mask_anchor(mask)
            visible_mask.putpixel(
                (
                    min(
                        overview_width - 1,
                        round(anchor[0] * overview_width / image.width),
                    ),
                    min(
                        overview_size[1] - 1,
                        round(anchor[1] * overview_size[1] / image.height),
                    ),
                ),
                255,
            )
        overview_masks.append((candidate, visible_mask))
    if label_avoid_mask is not None:
        _require_image_size(label_avoid_mask, image.size)
    overview = _render_mask_overview_with_legend(
        image=overview,
        masks=[(candidate.index, mask) for candidate, mask in overview_masks],
        labels={
            candidate.index: [f"Mask {candidate.index}"] for candidate in candidates
        },
        mask_style=mask_style,
        color_labels=True,
    )

    padding, line_height = typography["padding"], typography["line_height"]
    header_height = 3 * padding + 2 * line_height
    card_width, card_height = typography["card_width"], typography["card_height"]
    columns = (
        0
        if not candidates
        else 1 if len(candidates) <= 4 else 2 if len(candidates) <= 10 else 3
    )
    rows = math.ceil(len(candidates) / columns) if columns else 0
    canvas = Image.new(
        "RGB",
        (
            overview.width + columns * card_width,
            header_height + max(overview.height, rows * card_height),
        ),
        (245, 246, 248),
    )
    canvas.paste(overview, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (padding, padding),
        "Candidate overview",
        fill=(23, 33, 45),
        font=_load_label_font_at_size(typography["title"]),
    )
    if candidates:
        draw.text(
            (overview.width + padding, padding),
            "Mask cards: RGB = included",
            fill=(23, 33, 45),
            font=_load_label_font_at_size(typography["legend"]),
        )
        draw.text(
            (overview.width + padding, 2 * padding + line_height),
            "Gray = excluded pixels",
            fill=(23, 33, 45),
            font=_load_label_font_at_size(typography["legend"]),
        )
    else:
        draw.text(
            (padding, 2 * padding + line_height),
            "No candidate masks returned.",
            fill=(23, 33, 45),
            font=_load_label_font_at_size(typography["legend"]),
        )
    for position, (candidate, mask) in enumerate(decoded_masks):
        column, row = position % columns, position // columns
        origin = (
            overview.width + column * card_width,
            header_height + row * card_height,
        )
        _draw_candidate_detail_card(
            canvas=canvas,
            image=detail_image,
            mask=mask,
            candidate_index=candidate.index,
            origin=origin,
            card_size=(card_width, card_height),
            typography=typography,
        )

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_output_path)
    return resolved_output_path


def _selection_mask_anchor(mask: Image.Image) -> tuple[int, int]:
    """Find an included pixel near the mask bounds' center without filling holes."""
    bounds = mask.getbbox()
    assert bounds is not None
    middle_x, middle_y = (bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2
    if mask.getpixel((middle_x, middle_y)):
        return middle_x, middle_y
    _, rows = mask.getprojection()
    selected_y = min(
        (y for y in range(bounds[1], bounds[3]) if rows[y]),
        key=lambda y: abs(y - middle_y),
    )
    pixels = mask.crop((0, selected_y, mask.width, selected_y + 1)).tobytes()
    selected_x = min(
        (x for x in range(bounds[0], bounds[2]) if pixels[x]),
        key=lambda x: abs(x - middle_x),
    )
    return selected_x, selected_y


def _render_mask_overview_with_legend(
    *,
    image: Image.Image,
    masks: list[tuple[int, Image.Image]],
    labels: dict[int, list[str]],
    mask_style: str,
    color_labels: bool = False,
) -> Image.Image:
    """Keep source scale and connect every included-pixel anchor to a right legend."""
    typography = _preview_typography(image.size)
    overlay = Image.new("RGBA", image.size)
    anchors = {}
    for display_id, mask in masks:
        anchors[display_id] = _selection_mask_anchor(mask)
        color = _SELECTION_COLORS[(display_id - 1) % len(_SELECTION_COLORS)]
        rendered_mask = (
            mask
            if mask_style == "fill"
            else _mask_outer_outline(
                mask, image.size, outline_width=typography["line_width"]
            )
        )
        overlay.alpha_composite(
            Image.composite(
                Image.new(
                    "RGBA", image.size, (*color, 110 if mask_style == "fill" else 255)
                ),
                Image.new("RGBA", image.size),
                rendered_mask,
            )
        )

    font = _load_label_font_at_size(typography["legend"])
    badge_font = _load_label_font_at_size(typography["number"])
    padding, line_height = typography["padding"], typography["line_height"]
    badge_padding = typography["badge_padding"]
    badge_width = (
        max(
            typography["number"],
            math.ceil(
                max((badge_font.getlength(str(key)) for key in anchors), default=0)
            ),
        )
        + 2 * badge_padding
    )
    badge_height = typography["number"] + 2 * badge_padding
    if color_labels:
        label_width = math.ceil(
            max(
                (font.getlength(line) for lines in labels.values() for line in lines),
                default=0,
            )
        )
        legend_width = badge_width + 3 * padding + label_width
    else:
        legend_width = max(12 * typography["number"], badge_width + 7 * padding)
    # Sorting by anchor height reduces crossing while preserving selectable IDs.
    display_order = sorted(
        anchors, key=lambda key: (anchors[key][1], anchors[key][0], key)
    )
    rows = []
    next_y = padding
    text_width = legend_width - badge_width - 3 * padding
    for display_id in display_order:
        lines = []
        for label in labels[display_id]:
            lines.extend(_wrap_legend_text(label, font=font, maximum_width=text_width))
        height = max(badge_height, len(lines) * line_height)
        rows.append([display_id, next_y, height, lines])
        next_y += height + padding

    body_height = max(image.height, next_y)
    # Keep labels near their masks when room allows. Resolve collisions in
    # vertical order, then shift overflowing rows back into the fixed legend.
    previous_bottom = padding
    for row in rows:
        row[1] = max(previous_bottom, anchors[row[0]][1] - badge_height // 2)
        previous_bottom = row[1] + row[2] + padding
    next_top = body_height
    for row in reversed(rows):
        row[1] = min(row[1], next_top - row[2] - padding)
        next_top = row[1]

    canvas = Image.new(
        "RGB", (image.width + legend_width, body_height), (245, 246, 248)
    )
    canvas.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    draw = ImageDraw.Draw(canvas)
    # Draw all leaders before badges so lines never obscure another legend ID.
    for display_id, top, _, _ in rows:
        color = _SELECTION_COLORS[(display_id - 1) % len(_SELECTION_COLORS)]
        anchor = anchors[display_id]
        destination = (image.width + padding, top + badge_height // 2)
        line_width = typography["line_width"]
        draw.line((anchor, destination), fill="white", width=line_width + 2)
        draw.line((anchor, destination), fill=color, width=line_width)
        radius = max(2, round(typography["number"] / 12))
        draw.ellipse(
            (
                anchor[0] - radius,
                anchor[1] - radius,
                anchor[0] + radius,
                anchor[1] + radius,
            ),
            fill=color,
            outline="white",
        )
    for display_id, top, _, lines in rows:
        color = _SELECTION_COLORS[(display_id - 1) % len(_SELECTION_COLORS)]
        _draw_selection_label(
            draw,
            (
                image.width + padding,
                top,
                image.width + padding + badge_width,
                top + badge_height,
            ),
            str(display_id),
            color,
            badge_font,
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (
                    image.width + badge_width + 2 * padding,
                    top + line_index * line_height,
                ),
                line,
                font=font,
                fill=color if color_labels else (23, 33, 45),
            )
    return canvas


def _wrap_legend_text(
    label: str, *, font: ImageFont.ImageFont, maximum_width: int
) -> list[str]:
    """Wrap full IDs instead of truncating them or shrinking them over objects."""
    text = " ".join(label.split())
    lines: list[str] = []
    while text:
        split = len(text)
        while split > 1 and font.getlength(text[:split]) > maximum_width:
            split -= 1
        if split < len(text) and " " in text[:split]:
            split = text.rfind(" ", 0, split) or split
        lines.append(text[:split])
        text = text[split:].lstrip()
    return lines or [""]


def _draw_selection_label(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    font_size = getattr(font, "size", 24)
    draw.rounded_rectangle(
        bounds,
        radius=max(3, round(font_size / 6)),
        fill=color,
        outline="white",
        width=max(1, round(font_size / 12)),
    )
    text_bounds = draw.textbbox((0, 0), label, font=font)
    center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    draw.text(
        (
            center_x - (text_bounds[2] - text_bounds[0]) / 2 - text_bounds[0],
            center_y - (text_bounds[3] - text_bounds[1]) / 2 - text_bounds[1],
        ),
        label,
        fill=(23, 33, 45),
        font=font,
    )


def _draw_candidate_detail_card(
    *,
    canvas: Image.Image,
    image: Image.Image,
    mask: Image.Image,
    candidate_index: int,
    origin: tuple[int, int],
    card_size: tuple[int, int],
    typography: dict[str, int],
) -> None:
    """Composite after resizing so excluded holes retain an exact uniform gray."""
    bounds = mask.getbbox()
    assert bounds is not None
    padding = typography["padding"]
    header_height = typography["line_height"] + 2 * padding
    margin = max(padding, round(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 4))
    crop_bounds = (
        max(0, bounds[0] - margin),
        max(0, bounds[1] - margin),
        min(image.width, bounds[2] + margin),
        min(image.height, bounds[3] + margin),
    )
    crop, crop_mask = image.crop(crop_bounds), mask.crop(crop_bounds)
    available_size = (
        card_size[0] - 2 * padding,
        card_size[1] - header_height - padding,
    )
    scale = min(1.0, available_size[0] / crop.width, available_size[1] / crop.height)
    detail_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    crop = crop.resize(detail_size, Image.Resampling.LANCZOS)
    crop_mask = crop_mask.resize(detail_size, Image.Resampling.NEAREST)
    detail = Image.composite(
        crop, Image.new("RGB", detail_size, _SELECTION_GRAY), crop_mask
    )
    canvas.paste(
        detail,
        (
            origin[0] + (card_size[0] - detail.width) // 2,
            origin[1] + header_height + (available_size[1] - detail.height) // 2,
        ),
    )
    draw = ImageDraw.Draw(canvas)
    color = _SELECTION_COLORS[(candidate_index - 1) % len(_SELECTION_COLORS)]
    border, bar_width = max(2, padding // 3), max(3, round(typography["number"] / 5))
    draw.rectangle(
        (
            origin[0] + border,
            origin[1] + border,
            origin[0] + border + bar_width,
            origin[1] + card_size[1] - border,
        ),
        fill=color,
    )
    draw.text(
        (origin[0] + padding + bar_width, origin[1] + border),
        f"Mask {candidate_index} | selected pixels",
        font=_load_label_font_at_size(typography["legend"]),
        fill=color,
    )


def render_asset_mask_id_overlay(
    *,
    image_path: str | Path,
    asset_masks: list[tuple[str, str | Path]],
    output_path: str | Path,
    asset_names: dict[str, str] | None = None,
) -> Path:
    """Show all supplied scene masks with leaders to full IDs on the right.

    Short display numbers do not replace business IDs. The source image keeps
    its original pixel dimensions within the enlarged canvas, including small
    objects. Callers may supply the table alongside assets in ``asset_masks``.

    Args:
        image_path: Unmodified scene image.
        asset_masks: Ordered business-ID/mask-path pairs, including the table
            when its outline should appear in the overall scene view.
        output_path: Destination image path.
        asset_names: Optional readable names keyed by the same business IDs.

    Returns:
        Absolute path to the annotated scene image.
    """
    asset_ids = [asset_id for asset_id, _ in asset_masks]
    if any(not asset_id for asset_id in asset_ids):
        raise ValueError("Every asset mask must have a non-empty asset id.")
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Asset mask ids must be unique.")

    with Image.open(image_path) as loaded_image:
        image = loaded_image.convert("RGB")
    decoded_masks: list[tuple[int, Image.Image]] = []
    labels = {}
    for display_id, (asset_id, mask_path) in enumerate(asset_masks, start=1):
        with Image.open(mask_path) as loaded_mask:
            mask = loaded_mask.convert("L")
        _require_image_size(mask, image.size)
        if mask.getbbox() is None:
            raise ValueError(f"Asset mask {asset_id!r} is empty.")
        decoded_masks.append((display_id, mask))
        name = (asset_names or {}).get(asset_id)
        labels[display_id] = (
            [asset_id, name] if name and name != asset_id else [asset_id]
        )
    canvas = _render_mask_overview_with_legend(
        image=image,
        masks=decoded_masks,
        labels=labels,
        mask_style="outline",
    )

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_output_path)
    return resolved_output_path


def _require_image_size(mask: Image.Image, image_size: tuple[int, int]) -> None:
    if mask.size != image_size:
        raise ValueError(
            "Image Segmentation Server mask size does not match the input image: "
            f"{mask.size} != {image_size}."
        )


def _find_label_center_inside_bbox(
    *,
    candidate_mask: Image.Image,
    candidate_bbox: tuple[int, int, int, int],
    blocked_mask: Image.Image,
    label: str,
    font: ImageFont.ImageFont,
) -> tuple[float, float] | None:
    """Find the nearest label centre that stays in a candidate AABB and avoids masks."""
    probe_draw = ImageDraw.Draw(Image.new("RGBA", candidate_mask.size))
    bounds_at_origin = _number_label_bounds(
        draw=probe_draw, label=label, center=(0.0, 0.0), font=font
    )
    minimum_x = candidate_bbox[0] - bounds_at_origin[0]
    maximum_x = candidate_bbox[2] - 1 - bounds_at_origin[2]
    minimum_y = candidate_bbox[1] - bounds_at_origin[1]
    maximum_y = candidate_bbox[3] - 1 - bounds_at_origin[3]
    if minimum_x > maximum_x or minimum_y > maximum_y:
        return None

    # Expand each blocked region by the label's largest half-extent, so testing
    # one candidate centre guarantees the complete label rectangle stays clear.
    label_radius = max(
        -bounds_at_origin[0],
        bounds_at_origin[2],
        -bounds_at_origin[1],
        bounds_at_origin[3],
    )
    blocked_centres = blocked_mask.filter(ImageFilter.MaxFilter(2 * label_radius + 1))
    bbox_center = (
        (candidate_bbox[0] + candidate_bbox[2]) / 2,
        (candidate_bbox[1] + candidate_bbox[3]) / 2,
    )
    for require_candidate_mask in (True, False):
        for step in (max(1, round(min(candidate_mask.size) / 512)), 1):
            best_center: tuple[float, float] | None = None
            best_distance_squared = float("inf")
            for y_coordinate in range(minimum_y, maximum_y + 1, step):
                for x_coordinate in range(minimum_x, maximum_x + 1, step):
                    if blocked_centres.getpixel((x_coordinate, y_coordinate)):
                        continue
                    if require_candidate_mask and not candidate_mask.getpixel(
                        (x_coordinate, y_coordinate)
                    ):
                        continue
                    distance_squared = (x_coordinate - bbox_center[0]) ** 2 + (
                        y_coordinate - bbox_center[1]
                    ) ** 2
                    if distance_squared < best_distance_squared:
                        best_center = (float(x_coordinate), float(y_coordinate))
                        best_distance_squared = distance_squared
            if best_center is not None:
                return best_center
    return None


def _mask_outer_outline(
    mask: Image.Image,
    image_size: tuple[int, int],
    *,
    outline_width: int | None = None,
) -> Image.Image:
    """Use dilation and subtraction to get the outer outline of a binary mask."""
    # Asset-candidate images use outlines only; keep them visible at common
    # image resolutions without obscuring the original object appearance.
    if outline_width is None:
        outline_width = max(2, round(min(image_size) / 200))
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


def _foreground_pixel_count(mask: Image.Image) -> int:
    """Return the number of white pixels in one binary mask."""
    return mask.convert("L").histogram()[255]


def _foreground_ratio(mask: Image.Image) -> float:
    """Return the white-pixel ratio in one non-empty image region."""
    pixel_count = mask.width * mask.height
    if pixel_count == 0:
        raise ValueError("Mask region must contain at least one pixel.")
    return _foreground_pixel_count(mask) / pixel_count


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
    return _load_label_font_at_size(font_size)


def _load_asset_id_label_font(
    *,
    image_size: tuple[int, int],
    mask_bbox: tuple[int, int, int, int],
    label: str,
) -> ImageFont.ImageFont:
    """Choose an ID-label font constrained by both image and mask dimensions."""
    # The image sets the readable upper bound; the individual mask then caps it.
    image_font_size = min(32, max(8, round(min(image_size) / 48)))
    mask_width = mask_bbox[2] - mask_bbox[0]
    mask_height = mask_bbox[3] - mask_bbox[1]
    maximum_label_width = max(24, round(mask_width * 0.9))
    maximum_label_height = max(16, round(mask_height * 0.75))
    # Measure the complete text-and-background rectangle, not glyphs alone.
    probe_draw = ImageDraw.Draw(Image.new("RGBA", image_size))
    smallest_font = _load_label_font_at_size(6)
    for font_size in range(image_font_size, 5, -1):
        font = _load_label_font_at_size(font_size)
        label_bounds = _number_label_bounds(
            draw=probe_draw,
            label=label,
            center=(0.0, 0.0),
            font=font,
            minimum_padding=2,
        )
        if (
            label_bounds[2] - label_bounds[0] <= maximum_label_width
            and label_bounds[3] - label_bounds[1] <= maximum_label_height
        ):
            return font
        smallest_font = font
    return smallest_font


def _load_label_font_at_size(font_size: int) -> ImageFont.ImageFont:
    """Load the shared bold label font at one validated pixel size."""
    if font_size < 1:
        raise ValueError("Label font size must be positive.")
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
    minimum_padding: int = 4,
) -> None:
    """Draw a numbered label with red background and white text at the given center position."""
    label_bounds = _number_label_bounds(
        draw=draw,
        label=label,
        center=center,
        font=font,
        minimum_padding=minimum_padding,
    )
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    x = center[0] - label_width / 2
    y = center[1] - label_height / 2
    draw.rectangle(
        label_bounds,
        fill=(220, 0, 0, 255),
        outline=(255, 255, 255, 255),
        width=max(1, round(max(label_width, label_height) / 12)),
    )
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)


def _number_label_bounds(
    *,
    draw: ImageDraw.ImageDraw,
    label: str,
    center: tuple[float, float],
    font: ImageFont.ImageFont,
    minimum_padding: int = 4,
) -> tuple[int, int, int, int]:
    """Return the red label rectangle bounds for a label centre."""
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    if minimum_padding < 0:
        raise ValueError("Label minimum padding must be non-negative.")
    padding = max(minimum_padding, round(max(label_width, label_height) / 4))
    x = center[0] - label_width / 2
    y = center[1] - label_height / 2
    return (
        round(x - padding),
        round(y - padding),
        round(x + label_width + padding),
        round(y + label_height + padding),
    )
