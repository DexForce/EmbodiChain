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

"""Owned CPU payloads shared by the live and standard-decoded showcase arms."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDictBase

__all__ = [
    "describe_raw_buffer",
    "normalize_online_batch",
    "pack_episode",
    "numeric_episode_digest",
    "payload_bytes",
]


def _path(key: str | tuple[str, ...]) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def describe_raw_buffer(raw: TensorDictBase) -> dict[str, Any]:
    """Describe numeric observation order and native image layout without copying.

    Args:
        raw: ODS rollout buffer or sampled context with nested ``obs``.

    Returns:
        JSON-compatible metadata used by both transports. State leaves are
        sorted by their complete path; image features retain camera UIDs.
    """
    batch_dims = raw.batch_dims
    state_features: list[dict[str, Any]] = []
    cameras: dict[str, Any] = {}
    for key, tensor in sorted(
        raw["obs"].items(include_nested=True, leaves_only=True),
        key=lambda pair: _path(pair[0]),
    ):
        path = _path(key)
        shape = list(tensor.shape[batch_dims:])
        if len(path) == 3 and path[0] == "sensor" and path[-1] == "color":
            if tensor.dtype != torch.uint8 or len(shape) != 3 or shape[-1] != 4:
                raise ValueError("Native camera color must be HWC RGBA uint8")
            cameras[path[1]] = {"path": list(path), "shape": shape}
        elif path[0] == "sensor":
            raise ValueError(f"Unsupported sensor feature: {path}")
        else:
            if tensor.dtype == torch.bool or not tensor.is_floating_point():
                raise ValueError(f"State observation must be floating point: {path}")
            state_features.append(
                {"path": list(path), "shape": shape, "width": math.prod(shape)}
            )
    action_shape = list(raw["actions"].shape[batch_dims:])
    if not state_features or len(action_shape) != 1:
        raise ValueError("Require numeric state features and flat action vectors")
    return {
        "format": "ods-data-stream-payload-v1",
        "state_features": state_features,
        "state_width": sum(feature["width"] for feature in state_features),
        "action_width": action_shape[0],
        "cameras": cameras,
        "image_output": "RGB float32 channel-first [0,1]",
    }


def _numeric(raw: TensorDictBase, metadata: Mapping[str, Any]) -> torch.Tensor:
    shape = tuple(raw.batch_size)
    return torch.cat(
        [
            raw[("obs", *feature["path"])].reshape(*shape, feature["width"])
            for feature in metadata["state_features"]
        ],
        -1,
    ).to(device="cpu", dtype=torch.float32)


def _provenance(raw: TensorDictBase) -> dict[str, torch.Tensor]:
    keys = raw.keys()
    if "frame_ids" in keys:
        ids = raw["frame_ids"]
    else:
        ids = torch.stack(
            [raw[key] for key in ("ods_run_id", "ods_episode_id", "ods_frame_index")],
            -1,
        )
    result = {"frame_ids": ids.to(device="cpu", dtype=torch.int64).clone()}
    for canonical, native in (
        ("generations", "ods_generation"),
        ("published_at_monotonic_ns", "ods_published_at_monotonic_ns"),
    ):
        key = canonical if canonical in keys else native
        result[canonical] = raw[key].to(device="cpu", dtype=torch.int64).clone()
    if "continuity_id" not in keys:
        raise ValueError("The showcase requires explicit continuity_id")
    result["continuity_id"] = (
        raw["continuity_id"].to(device="cpu", dtype=torch.int64).clone()
    )
    return result


def normalize_online_batch(
    raw: TensorDictBase, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert one owned ODS context to the common RGB float32 contract.

    Args:
        raw: Engine or OnlineDataSource context with batch axes ``[B,L]``.
        metadata: Result of :func:`describe_raw_buffer`.

    Returns:
        Owned CPU tensors, including zero images/numerics and -1 source padding.
        The caller includes this conversion in measured delivery time.
    """
    if raw.batch_dims != 2:
        raise ValueError("A sampled context must have [batch,time] axes")
    valid = raw["valid"].to(device="cpu", dtype=torch.bool).clone()
    result: dict[str, Any] = {
        "state": _numeric(raw, metadata),
        "actions": raw["actions"].to(device="cpu", dtype=torch.float32).clone(),
        "valid": valid,
        "loss_mask": raw["loss_mask"].to(device="cpu", dtype=torch.bool).clone(),
        **_provenance(raw),
        "images": {},
    }
    for name, feature in metadata["cameras"].items():
        rgb = raw[("obs", *feature["path"])][..., :3].movedim(-1, -3)
        image = torch.empty(rgb.shape, device="cpu", dtype=torch.float32)
        image.copy_(rgb).div_(255)
        image.masked_fill_(~valid[..., None, None, None], 0)
        result["images"][name] = image
    for key in ("state", "actions"):
        result[key].masked_fill_(~valid[..., None], 0)
    for key in (
        "frame_ids",
        "generations",
        "continuity_id",
        "published_at_monotonic_ns",
    ):
        mask = valid[..., None] if key == "frame_ids" else valid
        result[key].masked_fill_(~mask, -1)
    return result


def pack_episode(rawrow: TensorDictBase, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one accepted cache row into a bounded writer-owned CPU episode.

    Args:
        rawrow: One row with a single time batch axis and immutable provenance.
        metadata: Shared ordered observation and camera description.

    Returns:
        An owned episode trimmed to real frames, with RGB uint8 HWC images.
        Interior invalid gaps are rejected instead of silently retiming data.
    """
    if rawrow.batch_dims != 1:
        raise ValueError("pack_episode requires one time-axis row")
    valid = rawrow["valid"].bool()
    positions = valid.nonzero().flatten()
    if not positions.numel():
        raise ValueError("Cannot persist an empty accepted episode")
    start, stop = int(positions[0]), int(positions[-1]) + 1
    if not valid[start:stop].all():
        raise ValueError("Accepted episode contains interior invalid gaps")
    row = rawrow[start:stop]
    episode: dict[str, Any] = {
        "state": _numeric(row, metadata).clone(),
        "actions": row["actions"].to(device="cpu", dtype=torch.float32).clone(),
        "valid": torch.ones(stop - start, dtype=torch.bool),
        **_provenance(row),
        "images": {},
    }
    for name, feature in metadata["cameras"].items():
        episode["images"][name] = (
            row[("obs", *feature["path"])][..., :3].cpu().contiguous().clone()
        )
    for key in ("state", "actions"):
        if not torch.isfinite(episode[key]).all():
            raise ValueError(f"Nonfinite accepted {key}")
    ids = episode["frame_ids"]
    if (ids[:, :2] != ids[0, :2]).any() or (ids[1:, 2] != ids[:-1, 2] + 1).any():
        raise ValueError("Accepted episode must retain one contiguous source identity")
    if (ids < 0).any() or (episode["continuity_id"] < 0).any():
        raise ValueError("Accepted frames require valid source and continuity IDs")
    generations = episode["generations"]
    if (generations < 1).any() or (generations != generations[0]).any():
        raise ValueError("An accepted episode must belong to one positive generation")
    return episode


def numeric_episode_digest(episode: Mapping[str, Any]) -> str:
    """Hash accepted numeric payload independently of run IDs and clocks.

    Args:
        episode: Packed accepted episode.

    Returns:
        SHA256 covering ordered state/actions, lengths and continuity boundaries.
    """
    digest = hashlib.sha256()
    for name in ("state", "actions", "continuity_id"):
        value = episode[name].contiguous()
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def payload_bytes(payload: Mapping[str, Any]) -> int:
    """Count tensor payload bytes, including image padding.

    Args:
        payload: Canonical batch or packed episode.

    Returns:
        Logical tensor bytes; views and temporaries are not a peak-memory model.
    """
    return sum(
        (
            value.numel() * value.element_size()
            if isinstance(value, torch.Tensor)
            else payload_bytes(value) if isinstance(value, Mapping) else 0
        )
        for value in payload.values()
    )
