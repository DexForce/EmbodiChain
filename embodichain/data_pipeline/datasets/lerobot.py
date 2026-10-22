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

"""Explicit, bounded numeric loading through the optional official LeRobot SDK."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from embodichain.utils import configclass
from .sources import TensorDataSource
from .schema import validate_data_schema

__all__ = ["LeRobotFieldMappingCfg", "LeRobotDataSource"]


@configclass
class LeRobotFieldMappingCfg:
    """Declare SDK columns and optional EmbodiChain provenance explicitly.

    Args:
        observation_key: Numeric vector column mapped to ``obs``.
        action_key: Numeric vector column mapped to ``actions``.
        provenance_key: Optional int64 vector ``[origin, episode, frame, generation]``.
        metadata_keys: Optional canonical scalar fields mapped to SDK columns;
            supports continuity/segment IDs, segment acceptance and language IDs.

    Column mapping is a declaration of semantics by the caller, not automatic
    inference from a matching width. Image/video sequence decoding is not part
    of this resident numeric backend.
    """

    observation_key: str = "observation.state"
    action_key: str = "action"
    provenance_key: str | None = None
    metadata_keys: dict[str, str] = {}

    def __post_init__(self) -> None:
        if not isinstance(self.metadata_keys, dict):
            raise ValueError("metadata_keys must be a field-to-column mapping")
        columns = [self.observation_key, self.action_key, *self.metadata_keys.values()]
        if self.provenance_key is not None:
            columns.append(self.provenance_key)
        if not all(isinstance(key, str) and key for key in columns) or len(
            set(columns)
        ) != len(columns):
            raise ValueError("Mapped LeRobot columns must be distinct nonempty strings")
        if set(self.metadata_keys) - {
            "continuity_id",
            "segment_id",
            "segment_accepted",
            "task_index",
            "subtask_index",
            "accepted_at_ns",
            "published_at_monotonic_ns",
        }:
            raise ValueError("Unsupported LeRobot scalar metadata mapping")


def _column_batch(
    dataset: Any, keys: list[str], start: int, count: int
) -> dict[str, torch.Tensor]:
    backing = getattr(dataset, "hf_dataset", None)
    if backing is not None and callable(getattr(backing, "__getitem__", None)):
        # Select columns before indexing so unrelated images/videos are not decoded.
        if hasattr(backing, "select_columns"):
            views = getattr(dataset, "_embodichain_numeric_views", None)
            if views is None:
                views = {}
                dataset._embodichain_numeric_views = views
            columns = tuple(keys)
            if columns not in views:
                views[columns] = backing.select_columns(keys)
            view = views[columns]
        else:
            view = backing
        payload = view[start : start + count]
    else:
        if any(
            feature.get("dtype") in {"image", "video"}
            for feature in dataset.features.values()
        ):
            raise ValueError(
                "Numeric loading of visual datasets requires SDK backing-column access"
            )
        items = [dataset[index] for index in range(start, start + count)]
        payload = {key: [item[key] for item in items] for key in keys}
    result = {}
    for key in keys:
        value = payload[key]
        if isinstance(value, (list, tuple)):
            if not value or not all(isinstance(item, torch.Tensor) for item in value):
                raise ValueError(f"LeRobot batch {key} must contain tensors")
            value = torch.stack(value)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim == 0
            or value.shape[0] != count
            or value.device.type != "cpu"
        ):
            raise ValueError(f"LeRobot batch {key} dtype/shape/device mismatch")
        result[key] = value
    return result


def _feature(
    dataset: Any, key: str, definition: Mapping[str, Any]
) -> tuple[int, torch.dtype]:
    feature = dataset.features.get(key, {})
    shape, kind = feature.get("shape"), feature.get("dtype")
    supported = {
        name: getattr(torch, name)
        for name in (
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
        )
    }
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] < 1
        or kind not in supported
    ):
        raise ValueError(f"LeRobot feature {key} must be a numeric vector")
    names = definition.get("names")
    if (
        not isinstance(names, list)
        or len(names) != shape[0]
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"Schema ordered names differ from LeRobot feature {key}")
    if feature.get("names") is not None and feature["names"] != names:
        raise ValueError(f"LeRobot feature {key} ordered names mismatch")
    if definition.get("dtype", kind) != kind:
        raise ValueError(f"LeRobot feature {key} dtype mismatch")
    return shape[0], supported[kind]


class LeRobotDataSource(TensorDataSource):
    """Load numeric episodes once and serve owned resident context batches.

    Args:
        root: Existing local LeRobot dataset directory.
        schema: Explicit ordered observation/action schema. Actions require a
            nonempty ``semantics`` or ``representation`` and positive ``step_dt``.
        memory_limit_bytes: Positive hard limit for owned resident tensors.
            SDK caches and bounded read/validation staging are separate costs.
        source_id: Persisted dataset/origin identity; used for generated frame
            IDs when the dataset does not already record provenance.
        mapping: Explicit numeric SDK columns and optional source metadata.
        repo_id: Dataset identifier required by the installed SDK.
        max_episode_steps: Optional fixed padded capacity. Inferred from source
            frame indices otherwise; specify it for exact ODS RNG pairing.
        episode_frames: Optional original valid frame indices, including gaps
            and empty padded rows. SDK episode order must match nonempty rows.
        read_batch_size: Maximum numeric SDK rows per read.
        language: Optional source-owned task/subtask ID-to-text dictionaries.

    Uses the official SDK lazily and never imports simulation modules. Missing
    backing-column access selects scalar SDK fallback; integrity failures do
    not. External datasets need explicit semantic mapping, not a showcase
    sidecar. Timestamps must match the declared fixed-rate action grid.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        schema: Mapping[str, Any],
        memory_limit_bytes: int,
        source_id: str,
        mapping: LeRobotFieldMappingCfg | None = None,
        repo_id: str = "local/dataset",
        max_episode_steps: int | None = None,
        episode_frames: Sequence[Sequence[int]] | None = None,
        read_batch_size: int = 1024,
        language: Mapping[str, Mapping[int | str, str]] | None = None,
    ) -> None:
        if type(memory_limit_bytes) is not int or memory_limit_bytes < 1:
            raise ValueError("memory_limit_bytes must be a positive integer")
        if type(read_batch_size) is not int or read_batch_size < 1:
            raise ValueError("read_batch_size must be a positive integer")
        if max_episode_steps is not None and (
            type(max_episode_steps) is not int or max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        mapping = mapping or LeRobotFieldMappingCfg()
        mapping.__post_init__()
        validate_data_schema(schema)
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a nonempty persisted dataset identity")
        # The SDK transform creates tensors without an explicit device.
        with torch.device("cpu"):
            data, storage = self._load(
                Path(root),
                schema,
                memory_limit_bytes,
                source_id,
                mapping,
                repo_id,
                max_episode_steps,
                episode_frames,
                read_batch_size,
            )
        metadata: dict[str, Any] = {"storage": storage}
        if language is not None:
            metadata["language"] = language
        super().__init__(
            data, schema=schema, source_id=source_id, copy_data=False, metadata=metadata
        )

    @staticmethod
    def _load(
        root: Path,
        schema: Mapping[str, Any],
        memory_limit: int,
        source_id: str,
        mapping: LeRobotFieldMappingCfg,
        repo_id: str,
        capacity: int | None,
        episode_frames: Sequence[Sequence[int]] | None,
        batch_size: int,
    ) -> tuple[TensorDict, dict[str, Any]]:
        if not root.is_dir():
            raise FileNotFoundError(
                f"Local LeRobot dataset directory does not exist: {root}"
            )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
        backing = getattr(dataset, "hf_dataset", None)
        if (
            backing is not None
            and callable(getattr(backing, "__getitem__", None))
            and len(backing) != len(dataset)
        ):
            raise ValueError("LeRobot backing frame count differs from SDK dataset")
        if not math.isclose(
            1 / dataset.fps, schema["action"]["step_dt"], rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError("LeRobot fps differs from schema action step_dt")
        obs_width, obs_dtype = _feature(
            dataset, mapping.observation_key, schema.get("observation", {})
        )
        act_width, act_dtype = _feature(dataset, mapping.action_key, schema["action"])
        if not torch.empty((), dtype=act_dtype).is_floating_point():
            raise ValueError("LeRobot action vectors must be floating point")
        fields = {
            "obs": ((obs_width,), obs_dtype),
            "actions": ((act_width,), act_dtype),
            "valid": ((), torch.bool),
            "frame_ids": ((3,), torch.int64),
        }
        if mapping.provenance_key is not None:
            fields["generations"] = ((), torch.int64)
        for key in mapping.metadata_keys:
            fields[key] = ((), torch.bool if key == "segment_accepted" else torch.int64)
        bytes_per_frame = sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in fields.values()
        )
        # Even unpadded allocation cannot fit; reject before scanning row payloads.
        if len(dataset) * bytes_per_frame > memory_limit:
            raise MemoryError("LeRobot resident tensors exceed memory_limit_bytes")
        metadata_keys = ["episode_index", "frame_index", "timestamp"]
        if mapping.provenance_key is not None:
            metadata_keys.append(mapping.provenance_key)
        layout: list[tuple[int, int, int, Sequence[int] | None]] = []
        largest_frame = -1
        if episode_frames is not None:
            cursor = 0
            for row, indices in enumerate(episode_frames):
                if any(
                    type(index) is not int or index < 0 for index in indices
                ) or list(indices) != sorted(set(indices)):
                    raise ValueError(
                        "episode_frames must contain ordered unique nonnegative indices"
                    )
                if indices:
                    layout.append((row, len(layout), len(indices), indices))
                    largest_frame = max(largest_frame, indices[-1])
                    cursor += len(indices)
            if cursor != len(dataset):
                raise ValueError("LeRobot frame count differs from episode_frames")
            rows = len(episode_frames)
        else:
            previous_episode = -1
            count = 0
            for start in range(0, len(dataset), batch_size):
                batch = _column_batch(
                    dataset, metadata_keys, start, min(batch_size, len(dataset) - start)
                )
                if (
                    batch["episode_index"].dtype != torch.int64
                    or batch["frame_index"].dtype != torch.int64
                ):
                    raise ValueError("LeRobot episode/frame dtype mismatch")
                if mapping.provenance_key is not None:
                    provenance = batch[mapping.provenance_key]
                    if (
                        provenance.shape != (len(batch["episode_index"]), 4)
                        or provenance.dtype != torch.int64
                    ):
                        raise ValueError("LeRobot provenance dtype/shape mismatch")
                    frames = provenance[:, 2]
                else:
                    frames = batch["frame_index"]
                if (frames < 0).any():
                    raise ValueError("LeRobot source frame indices must be nonnegative")
                largest_frame = max(largest_frame, int(frames.max()))
                episodes, lengths = batch["episode_index"].unique_consecutive(
                    return_counts=True
                )
                offset = 0
                for episode, size in zip(episodes.tolist(), lengths.tolist()):
                    if episode != previous_episode:
                        if episode <= previous_episode:
                            raise ValueError("LeRobot episode/frame order changed")
                        if previous_episode >= 0:
                            layout.append((len(layout), previous_episode, count, None))
                        previous_episode, count = episode, 0
                    if not torch.equal(
                        batch["frame_index"][offset : offset + size],
                        torch.arange(count, count + size),
                    ):
                        raise ValueError("LeRobot episode/frame order changed")
                    count += size
                    offset += size
            if previous_episode >= 0:
                layout.append((len(layout), previous_episode, count, None))
            rows = len(layout)
        if not rows or largest_frame < 0:
            raise ValueError("LeRobot dataset has no numeric episodes")
        steps = capacity if capacity is not None else largest_frame + 1
        if steps <= largest_frame:
            raise ValueError("max_episode_steps cannot truncate source frame indices")
        resident_bytes = rows * steps * bytes_per_frame
        if resident_bytes > memory_limit:
            raise MemoryError(
                "LeRobot padded resident tensors exceed memory_limit_bytes"
            )
        tensors = {
            key: torch.zeros(rows, steps, *shape, dtype=dtype)
            for key, (shape, dtype) in fields.items()
        }
        source_number = int.from_bytes(
            hashlib.sha256(source_id.encode()).digest()[:8], "little"
        ) & ((1 << 63) - 1)
        keys = [
            *metadata_keys,
            mapping.observation_key,
            mapping.action_key,
            *mapping.metadata_keys.values(),
        ]
        cursor = 0
        for row, sdk_episode, count, indices in layout:
            previous_source_frame = -1
            for offset in range(0, count, batch_size):
                size = min(batch_size, count - offset)
                batch = _column_batch(dataset, keys, cursor, size)
                for key, width, dtype in (
                    (mapping.observation_key, obs_width, obs_dtype),
                    (mapping.action_key, act_width, act_dtype),
                ):
                    if width == 1 and batch[key].shape == (size,):
                        batch[key] = batch[key].unsqueeze(-1)
                    if batch[key].shape != (size, width) or batch[key].dtype != dtype:
                        raise ValueError(
                            f"LeRobot batch {key} dtype/shape/device mismatch"
                        )
                if (
                    batch["episode_index"].dtype != torch.int64
                    or batch["frame_index"].dtype != torch.int64
                    or not (batch["episode_index"] == sdk_episode).all()
                    or not torch.equal(
                        batch["frame_index"], torch.arange(offset, offset + size)
                    )
                ):
                    raise ValueError("LeRobot episode/frame order changed")
                timestamps = batch["timestamp"].reshape(-1)
                expected_time = (
                    torch.arange(offset, offset + size) * schema["action"]["step_dt"]
                )
                if not torch.allclose(
                    timestamps.float(), expected_time, atol=1e-4, rtol=1e-5
                ):
                    raise ValueError(
                        "LeRobot timestamps differ from the declared action grid"
                    )
                selected = (
                    torch.tensor(indices[offset : offset + size], dtype=torch.int64)
                    if indices is not None
                    else batch["frame_index"]
                )
                if mapping.provenance_key is not None:
                    provenance = batch[mapping.provenance_key]
                    if (
                        provenance.shape != (size, 4)
                        or provenance.dtype != torch.int64
                        or (provenance < 0).any()
                    ):
                        raise ValueError(
                            "LeRobot provenance dtype/shape/value mismatch"
                        )
                    if indices is not None and not torch.equal(
                        provenance[:, 2], selected
                    ):
                        raise ValueError(
                            "LeRobot source frame indices differ from episode_frames"
                        )
                    selected = provenance[:, 2]
                if (
                    (selected < 0).any()
                    or (selected >= steps).any()
                    or selected[0] <= previous_source_frame
                    or (selected[1:] <= selected[:-1]).any()
                ):
                    raise ValueError("LeRobot source frame order/capacity mismatch")
                previous_source_frame = int(selected[-1])
                if mapping.provenance_key is not None:
                    tensors["frame_ids"][row, selected] = provenance[:, :3]
                    tensors["generations"][row, selected] = provenance[:, 3]
                else:
                    tensors["frame_ids"][row, selected, 0] = source_number
                    tensors["frame_ids"][row, selected, 1] = sdk_episode
                    tensors["frame_ids"][row, selected, 2] = selected
                tensors["obs"][row, selected] = batch[mapping.observation_key]
                tensors["actions"][row, selected] = batch[mapping.action_key]
                tensors["valid"][row, selected] = True
                for key, sdk_key in mapping.metadata_keys.items():
                    value = batch[sdk_key]
                    if value.shape == (size, 1):
                        value = value.squeeze(-1)
                    if value.shape != (size,) or value.dtype != fields[key][1]:
                        raise ValueError(f"LeRobot metadata {key} dtype/shape mismatch")
                    tensors[key][row, selected] = value
                cursor += size
        backing = getattr(dataset, "hf_dataset", None)
        bulk = backing is not None and callable(getattr(backing, "__getitem__", None))
        storage = {
            "format": "lerobot",
            "loader": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
            "lerobot_version": version("lerobot"),
            "root": str(root.resolve()),
            "per_minibatch_lerobot_io": False,
            "resident_tensor_bytes": resident_bytes,
            "memory_limit_bytes": memory_limit,
            "numeric_read_batch_size": batch_size,
            "numeric_read_mode": (
                "sdk_hf_dataset_batch" if bulk else "sdk_scalar_fallback"
            ),
            "scalar_fallback_reason": (
                None if bulk else "SDK backing dataset API absent"
            ),
            "sdk_cache_files": [
                str(Path(item["filename"]).resolve())
                for item in getattr(backing, "cache_files", [])
            ]
            or None,
        }
        return TensorDict(tensors, batch_size=[rows, steps], device="cpu"), storage
