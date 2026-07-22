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

import json
from pathlib import Path
from typing import Any

__all__ = ["save_segmentation_outputs"]


def save_segmentation_outputs(
    *,
    image_path: Path,
    segmentation_result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Save local mask and transparent crop images from mask RLE output."""
    cv2 = _require_cv2()
    image_path = image_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Segmentation source image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read segmentation source image: {image_path}")

    segmentations = segmentation_result.get("segmentations", [])
    if not isinstance(segmentations, list):
        raise ValueError("Segmentation result key segmentations must be a list.")

    output_dir.mkdir(parents=True, exist_ok=True)
    local_segmentations = []
    height, width = image.shape[:2]
    used_stems: set[str] = set()

    for index, segmentation in enumerate(segmentations):
        if not isinstance(segmentation, dict):
            continue
        mask_rle = segmentation.get("mask_rle") or segmentation.get("segmentation")
        if not isinstance(mask_rle, dict):
            continue
        mask_bool = _decode_mask_rle(mask_rle).astype(bool)
        if mask_bool.shape[:2] != (height, width):
            raise ValueError(
                "Decoded mask shape does not match source image: "
                f"{mask_bool.shape[:2]} vs {(height, width)}"
            )

        bbox = _bbox_from_segmentation(segmentation, mask_bool)
        target_id = str(segmentation.get("target_id") or f"segment_{index}")
        phrase = str(segmentation.get("phrase") or target_id)
        file_stem = _unique_file_stem(_safe_name(target_id), used_stems)
        mask_path = output_dir / f"{file_stem}_mask.png"
        crop_path = output_dir / f"{file_stem}_crop.png"

        _save_mask(mask_bool, mask_path)
        _save_transparent_crop(image, mask_bool, bbox, crop_path)
        local_segmentations.append(
            {
                "target_id": target_id,
                "target_kind": segmentation.get("target_kind"),
                "phrase": phrase,
                "bbox_xyxy": [float(value) for value in bbox],
                "local_mask_path": str(mask_path),
                "local_crop_path": str(crop_path),
            }
        )

    manifest = {
        "output_dir": str(output_dir),
        "source_image_path": str(image_path),
        "segmentations": local_segmentations,
        "num_segmentations": len(local_segmentations),
    }
    manifest_path = output_dir / "segmentation_outputs.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _decode_mask_rle(rle: dict[str, Any]) -> Any:
    try:
        from pycocotools import mask as mask_util
    except ImportError:
        mask_util = None

    if mask_util is not None:
        try:
            return mask_util.decode(rle)
        except Exception:
            pass

    np = _require_numpy()
    size = rle.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Mask RLE must contain size [height, width].")
    height, width = int(size[0]), int(size[1])
    counts = rle.get("counts")
    if isinstance(counts, str):
        runs = _decode_compressed_coco_rle_counts(counts)
    elif isinstance(counts, list):
        runs = [int(value) for value in counts]
    else:
        raise ValueError("Mask RLE counts must be a string or list.")

    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    value = 0
    for run_length in runs:
        next_offset = min(offset + int(run_length), flat.size)
        if value == 1:
            flat[offset:next_offset] = 1
        offset = next_offset
        value = 1 - value
    return flat.reshape((height, width), order="F")


def _decode_compressed_coco_rle_counts(counts: str) -> list[int]:
    runs = []
    index = 0
    while index < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            char_value = ord(counts[index]) - 48
            index += 1
            value |= (char_value & 0x1F) << shift
            more = bool(char_value & 0x20)
            shift += 5
            if not more and (char_value & 0x10):
                value |= -1 << shift
        if len(runs) > 2:
            value += runs[-2]
        runs.append(value)
    return runs


def _bbox_from_segmentation(
    segmentation: dict[str, Any],
    mask_bool: Any,
) -> tuple[int, int, int, int]:
    bbox = segmentation.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(int(round(float(value))) for value in bbox)

    np = _require_numpy()
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _save_mask(mask_bool: Any, output_path: Path) -> None:
    cv2 = _require_cv2()
    cv2.imwrite(str(output_path), mask_bool.astype("uint8") * 255)


def _save_transparent_crop(
    image: Any,
    mask_bool: Any,
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    cv2 = _require_cv2()
    np = _require_numpy()
    x1, y1, x2, y2 = _clip_bbox(bbox, image=image)
    if x2 <= x1 or y2 <= y1:
        return
    crop_bgr = image[y1:y2, x1:x2]
    crop_mask = mask_bool[y1:y2, x1:x2].astype("uint8") * 255
    crop_bgra = np.dstack([crop_bgr, crop_mask])
    cv2.imwrite(str(output_path), crop_bgra)


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    *,
    image: Any,
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _safe_name(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value.strip().lower()
    )
    return safe or "object"


def _unique_file_stem(stem: str, used_stems: set[str]) -> str:
    if stem not in used_stems:
        used_stems.add(stem)
        return stem
    suffix = 1
    while f"{stem}_{suffix}" in used_stems:
        suffix += 1
    unique_stem = f"{stem}_{suffix}"
    used_stems.add(unique_stem)
    return unique_stem


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required to save segmentation outputs."
        ) from exc
    return cv2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required to save segmentation outputs.") from exc
    return np
