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
"""Bounded video decoding and observed-pixel deocclusion for reconstruction."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings
import av
import cv2
import numpy as np
from embodichain.utils import configclass

__all__ = [
    "RecoveryCfg",
    "RecoveredSceneImage",
    "decode_video_frames",
    "recover_scene_image",
]


@configclass
class RecoveryCfg:
    """Limits for direct pixel fusion from same-view reference observations."""

    mask_margin: int = 2
    color_tolerance: float = 20.0
    max_frames: int = 12
    max_working_bytes: int = 256 * 1024**2


@dataclass(frozen=True)
class RecoveredSceneImage:
    """Recovered pixels with unresolved region and actual donor provenance."""

    rgb: np.ndarray
    unresolved: np.ndarray
    donor_frames: np.ndarray
    report: dict


def decode_video_frames(
    video: Path, indices: list[int], *, max_bytes: int = 128 * 1024**2
) -> dict[int, np.ndarray]:
    """Decode selected frame IDs into bounded uint8 CPU arrays, without CUDA.

    Args:
        video: Source video file.
        indices: Zero-based decoded frame IDs.
        max_bytes: Maximum total decoded RGB storage.

    Returns:
        RGB arrays keyed by the requested frame IDs.
    """
    ids = sorted(set(indices))
    if not ids or ids[0] < 0 or any(type(i) is not int for i in ids):
        raise ValueError("Expected nonnegative integer decoded frame IDs")
    output = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if stream.width * stream.height * 3 * len(ids) > max_bytes:
            raise ValueError("Requested video frames exceed decoding memory budget")
        wanted = set(ids)
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                output[index] = frame.to_ndarray(format="rgb24")
            if index >= ids[-1]:
                break
    if len(output) != len(ids):
        raise ValueError("Requested frame is outside the video")
    return output


def _align(
    first: np.ndarray,
    current: np.ndarray,
    first_mask: np.ndarray,
    current_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # This deliberately supports only tiny fixed-camera drift, not parallax or
    # moving cameras. Refuse large shifts rather than warping scene geometry.
    stable = (~(first_mask | current_mask)).astype(np.uint8) * 255
    warp = np.eye(2, 3, dtype=np.float32)
    correlation, warp = cv2.findTransformECC(
        cv2.cvtColor(first, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(current, cv2.COLOR_RGB2GRAY),
        warp,
        cv2.MOTION_TRANSLATION,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5),
        stable,
        5,
    )
    if correlation < 0.9 or np.linalg.norm(warp[:, 2]) > 1.5:
        raise ValueError(f"Fixed-camera assumption failed: {correlation=}, {warp=}")
    size = (first.shape[1], first.shape[0])
    aligned = cv2.warpAffine(
        current, warp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    blocked = cv2.warpAffine(
        current_mask.astype(np.uint8),
        warp,
        size,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1,
    ).astype(bool)
    return (
        aligned,
        blocked,
        {"correlation": correlation, "first_to_source": warp.tolist()},
    )


def recover_scene_image(
    images: dict[int, np.ndarray],
    robot_masks: dict[int, np.ndarray],
    reference_frames: list[int],
    *,
    cfg: RecoveryCfg | None = None,
) -> RecoveredSceneImage:
    """Fill robot occlusion where same-view reference pixels are available.

    Unaligned frames are skipped. Unresolved pixels keep the first observation
    for subsequent image editing; recovery coverage never gates reconstruction.
    References should reveal task objects without changing the desired scene state.
    """
    cfg = cfg or RecoveryCfg()
    refs = sorted(set(reference_frames))
    if any(i <= 0 for i in refs) or cfg.mask_margin < 0 or cfg.color_tolerance <= 0:
        raise ValueError("Expected positive reference IDs and valid recovery settings")
    indices = [0, *refs]
    if len(indices) > cfg.max_frames:
        raise ValueError("Too many reference frames")
    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("Images must be uint8 RGB")
    # Conservative bound for source, aligned, float median/error and mask arrays.
    if first.size * len(indices) * 16 > cfg.max_working_bytes:
        raise ValueError("Recovery exceeds configured working memory budget")
    cv2.setNumThreads(2)
    masks = {}
    for index in indices:
        if images[index].shape != first.shape or images[index].dtype != np.uint8:
            raise ValueError("Reference images differ in shape or dtype")
        mask = np.asarray(robot_masks[index], dtype=bool)
        if mask.shape != first.shape[:2]:
            raise ValueError("Robot mask resolution differs from source")
        kernel = np.ones((cfg.mask_margin * 2 + 1,) * 2, np.uint8)
        masks[index] = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    if not masks[0].any():
        return RecoveredSceneImage(
            first.copy(),
            masks[0],
            np.zeros(first.shape[:2], np.int32),
            {"status": "no_robot_detected", "generative_completion": False},
        )
    aligned, blocked, alignment = {}, {}, {}
    for index in refs:
        try:
            aligned[index], blocked[index], alignment[str(index)] = _align(
                first, images[index], masks[0], masks[index]
            )
        except (ValueError, cv2.error) as error:
            alignment[str(index)] = {"status": "skipped", "reason": str(error)}
    refs = list(aligned)
    if not refs:
        return RecoveredSceneImage(
            first.copy(),
            masks[0],
            np.where(masks[0], -1, 0).astype(np.int32),
            {
                "status": "image_edit_required",
                "recovered_pixels": 0,
                "unresolved_pixels": int(masks[0].sum()),
                "alignment": alignment,
            },
        )
    values = np.stack([aligned[i] for i in refs]).astype(np.float32)
    valid = np.stack([~blocked[i] for i in refs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(valid[..., None], values, np.nan), axis=0)
    error = np.max(np.abs(values - median), axis=-1)
    consistent = valid & (error <= cfg.color_tolerance)
    recovered = masks[0] & (consistent.any(axis=0))
    choice = np.argmin(np.where(consistent, error, np.inf), axis=0)
    yy, xx = np.indices(recovered.shape)
    result = first.copy()
    result[recovered] = values[choice, yy, xx][recovered].astype(np.uint8)
    unresolved = masks[0] & ~recovered
    donor = np.full(recovered.shape, -1, np.int32)
    donor[recovered] = np.asarray(refs)[choice[recovered]]
    donor[~masks[0]] = 0
    report = dict(
        status="recovered",
        recovered_pixels=int(recovered.sum()),
        unresolved_pixels=int(unresolved.sum()),
        unchanged_outside_robot_region=bool(
            np.array_equal(result[~masks[0]], first[~masks[0]])
        ),
        alignment=alignment,
        generative_completion=False,
    )
    return RecoveredSceneImage(result, unresolved, donor, report)
