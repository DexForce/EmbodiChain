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

"""Remove visible robot hardware from a scene reference image."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import label

from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    decode_rle_mask,
)
from embodichain.gen_sim.services.compat import SegmentationFacade
from embodichain.gen_sim.services.errors import CapabilityError

if TYPE_CHECKING:
    from embodichain.gen_sim.services.compat import SegmentationClient
    from embodichain.gen_sim.services.runtime import GenSimServices

__all__: list[str] = []

_Box = tuple[int, int, int, int]
_MOUNT_APPEARANCE = "black rectangular base at bottom center of image"
_MOUNT_EDIT = (
    "Remove the large black rectangular slab at the very bottom center of the image, "
    "including the silver U-shaped metal bracket on it. Replace the entire black and "
    "silver region that touches the bottom edge with uninterrupted tabletop that follows "
    "the neighboring grain and lighting. Also remove the two black robot arms and fingers "
    "extending from the bottom left and bottom right edges. Keep all loose tabletop objects "
    "and the camera viewpoint unchanged."
)


def _bottom_center_mount(mask: Image.Image) -> Image.Image | None:
    """Select the connected camera plate at the image edge for this framing."""
    width, height = mask.size
    # The plate must stay inside the lower central part of the image. Only use
    # components reaching the bottom edge so disconnected scene objects cannot
    # enter the composite even if SAM returns them in the same instance.
    bottom_rows = max(1, round(height * 0.01))
    components, _ = label(np.asarray(mask) != 0, structure=np.ones((3, 3), dtype=bool))
    candidates = []
    for component in np.unique(components[-bottom_rows:]):
        if component == 0:
            continue
        plate = Image.fromarray((components == component).astype(np.uint8) * 255)
        x0, y0, x1, _ = plate.getbbox()
        if width / 4 <= x0 < width / 2 < x1 <= 3 * width / 4 and y0 >= 3 * height / 4:
            candidates.append(plate)
    return candidates[0] if len(candidates) == 1 else None


def _segment_part(
    client: SegmentationClient,
    image_path: Path,
    size: tuple[int, int],
    prompts: tuple[str, ...],
    boxes: tuple[_Box, ...] = (),
    *,
    bottom_center_mount: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    started = perf_counter()
    for prompt in prompts:
        mask = Image.new("L", size)
        instance_boxes = []
        for rle in client.segment_single_object(image_path=image_path, prompt=prompt):
            instance = decode_rle_mask(rle)
            if instance.size != size:
                raise ValueError("SAM mask dimensions differ from the source image.")
            if bottom_center_mount:
                instance = _bottom_center_mount(instance)
                if instance is None:
                    continue
            if (bounds := instance.getbbox()) is not None:
                instance_boxes.append(bounds)
            mask = ImageChops.lighter(mask, instance)
        if bottom_center_mount and len(instance_boxes) != 1:
            continue
        if boxes:
            regions = [mask.crop(box) for box in boxes]
            if not all(region.getbbox() is not None for region in regions):
                continue
            restricted = Image.new("L", size)
            for box, region in zip(boxes, regions, strict=True):
                restricted.paste(region, box[:2])
            mask = restricted
        if mask.getbbox() is not None:
            return mask, {
                "prompt": prompt,
                "mask_pixels": int(np.count_nonzero(mask)),
                "instance_boxes": instance_boxes,
                "elapsed_seconds": perf_counter() - started,
            }
    if bottom_center_mount:
        raise ValueError("No unambiguous bottom-center camera support mask found.")
    raise ValueError(f"No segmentation mask found for: {', '.join(prompts)}")


def _gripper_regions(arm_boxes: list[_Box], image_width: int) -> tuple[_Box, ...]:
    """In the overhead front view the jaw assembly caps each rising arm."""
    left, right = ([], [])
    for box in arm_boxes:
        side = left if box[0] + box[2] < image_width else right
        side.append(box)
    if len(left) != 1 or len(right) != 1:
        raise ValueError(
            "Cannot locate both grippers from unambiguous left and right arm masks."
        )
    pad = max(1, round(image_width * 0.01))
    return tuple(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(image_width, x1 + pad),
            min(y1, y0 + max(1, round((y1 - y0) / 3))),
        )
        for x0, y0, x1, y1 in (left[0], right[0])
    )


def clean_robot_image(
    image_path: Path,
    output: Path,
    *,
    services: GenSimServices,
    gripper_boxes: tuple[_Box, ...] = (),
    gripper_prompt: str | None = None,
    camera_support_prompt: str | None = None,
    edit_prompt: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run separate masks, composite the edit, and write a pixel audit.

    Args:
        image_path: Source frame, decoded as RGB at its original resolution.
        output: New directory for the frame, masks, edit and audit.
        services: Borrowed GenSim run owning segmentation and image editing.
        gripper_boxes: Optional pixel boxes (x0, y0, x1, y1) in the source image.
            The gripper mask is limited to these boxes, and every box must
            contain a detection.
        gripper_prompt: Optional appearance description for these grippers.
        camera_support_prompt: Optional appearance description for a specific mount.
        edit_prompt: Optional replacement instructions for the image editor.
        seed: Seed passed to the image editor.

    Returns:
        Audit with the selected prompts, mask coverage, and RGB change counts.
    """
    started = perf_counter()
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    width, height = original.size
    for x0, y0, x1, y1 in gripper_boxes:
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Each gripper box must be nonempty and inside the image.")
    editor = services.get("image_generation")
    if not callable(getattr(editor, "generate_image_by_image", None)):
        raise CapabilityError(
            "The selected image service does not support image-to-image editing."
        )
    mount_prompts = (
        (camera_support_prompt,)
        if camera_support_prompt is not None
        else ("camera mounting bracket", "camera support", _MOUNT_APPEARANCE)
    )
    gripper_prompts = (
        (gripper_prompt,)
        if gripper_prompt is not None
        else (
            "robot gripper",
            "gripper",
            "parallel jaw gripper",
            "black triangular prongs",
        )
    )

    output.mkdir(parents=True, exist_ok=False)
    original_path = output / "original.png"
    original.save(original_path)
    segmenter = SegmentationFacade(services)
    arms, arm_report = _segment_part(
        segmenter, original_path, original.size, ("robot arm",)
    )
    box_source = "provided" if gripper_boxes else "arm_masks"
    if not gripper_boxes:
        gripper_boxes = _gripper_regions(arm_report["instance_boxes"], width)
    grippers, gripper_report = _segment_part(
        segmenter, original_path, original.size, gripper_prompts, gripper_boxes
    )
    mount, mount_report = _segment_part(
        segmenter,
        original_path,
        original.size,
        mount_prompts,
        bottom_center_mount=True,
    )
    parts = {}
    union = Image.new("L", original.size)
    for name, part, metadata in (
        ("arms", arms, arm_report),
        ("grippers", grippers, gripper_report),
        ("camera_support", mount, mount_report),
    ):
        filename = name.replace("_", "-") + "-mask.png"
        part.save(output / filename)
        parts[name] = {**metadata, "mask": filename}
        union = ImageChops.lighter(union, part)

    # SAM can omit the silver hardware within the accepted plate's silhouette.
    plate = np.asarray(mount) != 0
    filled = np.logical_and(
        np.maximum.accumulate(plate, axis=1),
        np.maximum.accumulate(plate[:, ::-1], axis=1)[:, ::-1],
    )
    mount_interior_pixels = int(np.count_nonzero(filled & ~plate))
    union = ImageChops.lighter(union, Image.fromarray(filled.astype(np.uint8) * 255))

    # Replace the foreground fully and feather only in the adjacent margin.
    edge = 2 if width >= 480 else 1
    core = union.filter(ImageFilter.MaxFilter(2 * edge + 1))
    mask = core.filter(ImageFilter.MaxFilter(4 * edge + 1))
    feather = core.filter(ImageFilter.MaxFilter(2 * edge + 1)).filter(
        ImageFilter.GaussianBlur(edge)
    )
    alpha = ImageChops.lighter(core, ImageChops.multiply(mask, feather))
    inside = np.asarray(mask) != 0
    fraction = float(inside.mean())
    if not 0 < fraction < 0.35:
        raise ValueError(f"Robot mask has an implausible area fraction: {fraction}")
    mask.save(output / "robot-mask.png")

    edit_width = max(256, min(1024, round(width / 32) * 32))
    edit_height = max(256, min(1024, round(height / 32) * 32))
    edit_size = (edit_width, edit_height)
    if edit_prompt is None:
        support = camera_support_prompt or "camera support and its mounting plate"
        edit_prompt = (
            _MOUNT_EDIT
            if mount_report["prompt"] == _MOUNT_APPEARANCE
            else (
                f"Remove the entire {support}, including all attached metal brackets "
                "and hardware. Replace the entire fixture with a seamless continuation "
                "of the neighboring surface and its texture. Also remove the robot arms, "
                "grippers and fingers. Preserve the positions, sizes and appearance "
                "of all loose scene objects and keep the camera viewpoint unchanged."
            )
        )
    edit_started = perf_counter()
    edit_path = editor.generate_image_by_image(
        image_path=original_path,
        prompt=edit_prompt,
        output_path=output / "editor-response.png",
        size=edit_size,
        seed=seed,
    )
    with Image.open(edit_path) as response:
        edited = response.convert("RGB").resize(original.size)
    edit_seconds = perf_counter() - edit_started
    cleaned = Image.composite(edited, original, alpha)
    changed = np.any(np.asarray(cleaned) != np.asarray(original), axis=2)
    outside_changed = int(np.count_nonzero(changed & ~inside))
    if outside_changed:
        raise RuntimeError(f"Edit changed {outside_changed} pixels outside the mask.")
    if not np.any(changed & (np.asarray(union) != 0)):
        raise ValueError("Image editor did not change any detected robot pixels.")
    cleaned.save(output / "input.png")

    report = {
        "method": "per-part text segmentation and masked image-edit composite",
        "source": str(image_path),
        "image_size": list(original.size),
        "parts": parts,
        "gripper_boxes": gripper_boxes,
        "gripper_box_source": box_source,
        "mount_interior_pixels_added": mount_interior_pixels,
        "robot_mask_pixels": int(np.count_nonzero(inside)),
        "robot_mask_fraction": fraction,
        "inside_mask_pixels_changed": int(np.count_nonzero(changed & inside)),
        "outside_mask_pixels_checked": int(np.count_nonzero(~inside)),
        "outside_mask_pixels_changed": outside_changed,
        "preserves_all_pixels_outside_robot_mask": True,
        "edit_prompt": edit_prompt,
        "edit_seed": seed,
        "edit_requested_size": edit_size,
        "edit_seconds": edit_seconds,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
