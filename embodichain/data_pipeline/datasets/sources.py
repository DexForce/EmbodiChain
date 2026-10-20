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

"""Live ODS and resident CPU implementations of the training-batch contract."""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict, TensorDictBase

from ._sampling import _context_indices, _draw_budgeted_rows, _pad_context
from .sequence import DataBudgetExhaustedError, DataUnavailableError, SequenceSpecCfg
from .schema import validate_data_schema

if TYPE_CHECKING:
    from embodichain.data_pipeline.engine.data import OnlineDataEngine

__all__ = ["OnlineDataSource", "TensorDataSource"]


def _check_features(data: TensorDictBase, schema: Mapping[str, Any]) -> None:
    for key, section in (("obs", "observation"), ("actions", "action")):
        value = data[key]
        definition = schema.get(section, {})
        if isinstance(value, TensorDictBase):
            if "names" in definition:
                raise ValueError(
                    f"Flat schema {section}.names cannot describe nested tensors"
                )
            if not list(value.keys(include_nested=True, leaves_only=True)):
                raise ValueError(f"Schema {section} requires nonempty numeric features")
            dtype = definition.get("dtype")
            if dtype is not None and any(
                str(leaf.dtype).removeprefix("torch.") != dtype
                for leaf in value.values(include_nested=True, leaves_only=True)
            ):
                raise ValueError(f"Schema {section} dtype mismatch")
            continue
        names = definition.get("names")
        if names is not None and (
            value.ndim != data.batch_dims + 1 or len(names) != value.shape[-1]
        ):
            raise ValueError(f"Schema {section}.names must name all ordered features")
        dtype = definition.get("dtype")
        if dtype is not None and dtype != str(value.dtype).removeprefix("torch."):
            raise ValueError(f"Schema {section} dtype mismatch")


class OnlineDataSource:
    """Adapt a started ODS engine without taking ownership of its lifecycle.

    Args:
        engine: Caller-owned engine. Enable provenance for portable frame IDs.
        schema: Explicit schema of the returned observations and actions.
        observation_transform: Optional row-local feature function. It runs on
            the owned copy, after the producer lock has been released.

    Refill, waits, start and stop remain caller operations. Sampling propagates
    empty-window, quota and worker errors immediately. The adapter preserves
    additional continuity/language/segment fields when present.
    """

    def __init__(
        self,
        engine: OnlineDataEngine,
        *,
        schema: Mapping[str, Any],
        observation_transform: (
            Callable[[Any], torch.Tensor | TensorDictBase] | None
        ) = None,
    ) -> None:
        validate_data_schema(schema)
        self.engine = engine
        self.schema = copy.deepcopy(dict(schema))
        self.source_id = str(engine.run_id)
        self.observation_transform = observation_transform
        action = self.schema.get("action", {})
        if not isinstance(action, Mapping):
            raise ValueError("Schema action must be a mapping")
        semantics = action.get("semantics")
        if semantics is not None and semantics != engine.cfg.action_semantics:
            raise ValueError(
                "Schema action semantics differ from the ODS recording contract"
            )
        terms = action.get("terms", [])
        if not isinstance(terms, (list, tuple)) or not all(
            isinstance(term, Mapping) and isinstance(term.get("term", {}), Mapping)
            for term in terms
        ):
            raise ValueError("Schema action terms must be ordered descriptor mappings")
        contracts = tuple(term.get("term", {}).get("contract") for term in terms)
        if contracts and contracts != tuple(engine.cfg.action_contracts):
            raise ValueError("Schema action term order differs from the ODS contract")

    def sample_context(
        self,
        batch_size: int,
        spec: SequenceSpecCfg,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Sample an owned canonical batch under the engine's normal checks.

        Args:
            batch_size: Number of contexts to deliver atomically.
            spec: History/future lengths, segment mode and generation filter.
            generator: Optional sampling RNG independent of model randomness.

        Returns:
            Canonical TensorDict with ``obs/actions/valid/loss_mask`` and any
            available ``frame_ids/generations`` and segment/language fields.
            Optional acceptance Unix nanoseconds and publication host-monotonic
            nanoseconds remain provenance, not inputs to feature inference.
        """
        spec._check()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        raw = self.engine.sample_batch_with_context(
            batch_size,
            spec.history_length,
            spec.prediction_horizon,
            generator=generator,
            sampling_mode=spec.sampling_mode,
            generations=spec.generations,
        )
        keys = [key for key in raw.keys() if not key.startswith("ods_")]
        result = raw.select(*keys)
        if self.observation_transform is not None:
            result["obs"] = self.observation_transform(raw["obs"])
        if "ods_run_id" in raw.keys():
            result["frame_ids"] = torch.stack(
                [
                    raw[key]
                    for key in ("ods_run_id", "ods_episode_id", "ods_frame_index")
                ],
                -1,
            )
        if "ods_generation" in raw.keys():
            result["generations"] = raw["ods_generation"]
        for native, canonical in (
            ("ods_accepted_at_ns", "accepted_at_ns"),
            ("ods_published_at_monotonic_ns", "published_at_monotonic_ns"),
        ):
            if native in raw.keys():
                result[canonical] = raw[native]
        _check_features(result, self.schema)
        return _pad_context(result, raw["valid"], raw["loss_mask"])

    def resolve_language(self, batch: TensorDict) -> dict[str, object]:
        """Resolve copied task/subtask IDs through the engine's text registry.

        Args:
            batch: Copied batch containing task/subtask index tensors.

        Returns:
            Overall and segment texts with the batch's nested shape.
        """
        return self.engine.resolve_language(batch)


class TensorDataSource:
    """Sample causal sequences from resident, explicitly described CPU data.

    Args:
        data: TensorDict with ``[episode, time]`` batch axes, ``obs``, floating
            ``actions`` and boolean ``valid``. Optional continuity, segment,
            language and provenance fields are preserved.
        schema: Ordered observation/action semantics. No feature conversion is
            inferred from vector dimensions.
        source_id: Stable source dataset/run identity. Persist this for datasets
            without recorded ``frame_ids``; it seeds their numeric source ID.
        copy_data: Clone the supplied data by default. False transfers storage
            responsibility to the caller, which must not mutate it afterwards.
        metadata: Optional detached storage/provenance diagnostics.

    Each returned batch owns its storage. Optional source quotas are owner-only
    and cannot be reset. ``data`` is exposed for inspection and fitting frozen
    training statistics; treat its tensors as immutable after construction.
    """

    def __init__(
        self,
        data: TensorDictBase,
        *,
        schema: Mapping[str, Any],
        source_id: str,
        copy_data: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validate_data_schema(schema)
        if data.batch_dims != 2 or any(size < 1 for size in data.batch_size):
            raise ValueError(
                "Resident data must have nonempty [episode, time] batch axes"
            )
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a nonempty persisted dataset identity")
        if not {"obs", "actions", "valid"} <= set(data.keys()):
            raise ValueError("Resident data requires obs, actions and valid")
        for _, value in data.items(include_nested=True, leaves_only=True):
            if not isinstance(value, torch.Tensor):
                raise ValueError("Resident source payloads must be numeric tensors")
            if value.device.type != "cpu":
                raise ValueError("Resident tensors must be on CPU")
        if data["valid"].dtype != torch.bool or data["valid"].shape != data.batch_size:
            raise ValueError("valid must be bool [episode, time]")
        if (
            not data["valid"].any()
            or not isinstance(data["actions"], torch.Tensor)
            or not data["actions"].is_floating_point()
        ):
            raise ValueError("Require valid observations and floating-point actions")
        if data["actions"].ndim != data.batch_dims + 1 or data["actions"].shape[-1] < 1:
            raise ValueError("actions must have a nonempty flat feature axis")
        _check_features(data, schema)
        for key, value in data.items(include_nested=True, leaves_only=True):
            if (
                key == "actions"
                or key == "obs"
                or isinstance(key, tuple)
                and key[0] == "obs"
            ):
                for row in range(data.batch_size[0]):
                    for start in range(0, data.batch_size[1], 1024):
                        mask = data["valid"][row, start : start + 1024]
                        if not torch.isfinite(
                            value[row, start : start + 1024][mask]
                        ).all():
                            raise ValueError(
                                "Valid observations/actions must be finite"
                            )
        self.data = data.clone() if copy_data else data
        self.schema = copy.deepcopy(dict(schema))
        self.source_id = source_id
        self.metadata = copy.deepcopy(dict(metadata or {}))
        if "frame_ids" not in self.data.keys():
            rows, steps = self.data.batch_size
            source_number = int.from_bytes(
                hashlib.sha256(source_id.encode()).digest()[:8], "little"
            ) & ((1 << 63) - 1)
            self.data["frame_ids"] = torch.stack(
                (
                    torch.full(
                        (rows, steps), source_number, dtype=torch.int64, device="cpu"
                    ),
                    torch.arange(rows, device="cpu")[:, None].expand(rows, steps),
                    torch.arange(steps, device="cpu")[None].expand(rows, steps),
                ),
                -1,
            )
        ids = self.data["frame_ids"]
        if ids.dtype != torch.int64 or ids.shape != (*self.data.batch_size, 3):
            raise ValueError("frame_ids must be int64 [episode, time, 3]")
        identities = set()
        for row in range(data.batch_size[0]):
            selected = ids[row, self.data["valid"][row]]
            if not len(selected):
                continue
            identity = tuple(selected[0, :2].tolist())
            if (
                (selected < 0).any()
                or not (selected[:, :2] == selected[0, :2]).all()
                or identity in identities
            ):
                raise ValueError(
                    "Each source identity must own one distinct episode row"
                )
            if (selected[1:, 2] <= selected[:-1, 2]).any():
                raise ValueError("Source frame IDs must increase within each episode")
            identities.add(identity)
        for key in (
            "continuity_id",
            "segment_id",
            "task_index",
            "subtask_index",
            "generations",
            "accepted_at_ns",
            "published_at_monotonic_ns",
        ):
            if key in self.data.keys() and (
                self.data[key].dtype != torch.int64
                or self.data[key].shape != self.data.batch_size
            ):
                raise ValueError(f"{key} must be int64 [episode, time]")
        if "segment_accepted" in self.data.keys() and (
            self.data["segment_accepted"].dtype != torch.bool
            or self.data["segment_accepted"].shape != self.data.batch_size
        ):
            raise ValueError("segment_accepted must be bool [episode, time]")
        adjacent = self.data["valid"][:, 1:] & self.data["valid"][:, :-1]
        continuity = self.data.get("continuity_id")
        if continuity is not None:
            adjacent &= (continuity[:, 1:] == continuity[:, :-1]) & (
                continuity[:, 1:] >= 0
            )
        if ((ids[:, 1:, 2] != ids[:, :-1, 2] + 1) & adjacent).any():
            raise ValueError(
                "Source frame gaps require invalid frames or an explicit continuity boundary"
            )
        self._rows = self.data["valid"].any(1).nonzero().flatten()
        self._sampling_owner = (os.getpid(), threading.get_ident())
        self._sampling_started = False
        self._quota: dict[str, Any] | None = None

    def resolve_language(self, batch: TensorDict) -> dict[str, object]:
        """Resolve IDs using source-owned dictionaries in metadata['language'].

        Args:
            batch: Copied task/subtask indices; select real frames before
                resolving since padding and unknown IDs are rejected.

        Returns:
            Nested task/subtask texts with the corresponding index shapes.
        """
        dictionaries = self.metadata.get("language", {})
        result = {}
        for kind in ("task", "subtask"):
            if kind not in dictionaries or kind + "_index" not in batch.keys():
                raise ValueError(
                    "Source does not provide language dictionaries and indices"
                )
            lookup = {int(key): value for key, value in dictionaries[kind].items()}
            ids = batch[kind + "_index"]

            def resolve(values: object) -> object:
                if isinstance(values, list):
                    return [resolve(item) for item in values]
                if values not in lookup or not isinstance(lookup[values], str):
                    raise ValueError("Unknown or padded source-local language ID")
                return lookup[values]

            result[kind] = resolve(ids.tolist())
        return result

    def configure_sampling_budget(self, budget: int, hard_limit: int) -> None:
        """Assign a permanent per-source chunk allowance before first sampling.

        The stateful sampler belongs to its creating process and thread. Copying
        it into DataLoader workers cannot duplicate allowances. Valid rows must
        each identify one distinct immutable ``(run, episode)`` source. Empty
        padded rows receive no allowance. There is no reset or refill operation.

        Args:
            budget: Maximum delivered contexts for each retained source episode.
            hard_limit: Explicit protection ceiling; budget must not exceed it.
        """
        self._require_sampling_owner()
        if self._quota is not None or self._sampling_started:
            raise RuntimeError("Configure the sampling budget once, before sampling")
        if (
            any(
                type(n) is not int or not 0 < n <= (1 << 63) - 1
                for n in (budget, hard_limit)
            )
            or budget > hard_limit
        ):
            raise ValueError("Require positive int64 budget <= hard_limit")
        if any(
            t.device.type != "cpu"
            for _, t in self.data.items(include_nested=True, leaves_only=True)
        ):
            raise ValueError("Offline sampling budgets require CPU corpus tensors")
        identities = torch.full(
            (len(self.data["valid"]), 2), -1, dtype=torch.int64, device="cpu"
        )
        seen = set()
        generations = {}
        for row in self._rows.tolist():
            ids = self.data["frame_ids"][row, self.data["valid"][row]]
            identity = tuple(ids[0, :2].tolist())
            if (ids[:, :2] < 0).any() or not (ids[:, :2] == ids[0, :2]).all():
                raise ValueError(
                    "Each valid row must contain one immutable source identity"
                )
            if identity in seen:
                raise ValueError("Duplicate source identity across corpus rows")
            seen.add(identity)
            identities[row] = ids[0, :2]
            values = (
                self.data.get("generations")[row, self.data["valid"][row]].unique()
                if self.data.get("generations") is not None
                else None
            )
            generations[row] = (
                int(values[0]) if values is not None and len(values) == 1 else None
            )
        self._quota = {
            "budget": budget,
            "hard_limit": hard_limit,
            "identities": identities,
            "generations": generations,
            "delivered": torch.zeros(
                len(self.data["valid"]), dtype=torch.int64, device="cpu"
            ),
            "clock_origin_monotonic_ns": time.monotonic_ns(),
        }

    def _require_sampling_owner(self) -> None:
        if self._sampling_owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError(
                "Budgeted TensorDataSource sampling is owner-process/thread only"
            )

    def get_sampling_budget_snapshot(self) -> dict[str, Any]:
        """Return detached delivery counts, using one context as one chunk.

        Returns:
            Per-episode allowances/deliveries, total remaining credits and
            continuity-qualified eligible remaining credits. A sequence's
            generation or segment filter can restrict eligibility further.
        """
        quota = self._quota
        if quota is None:
            return {
                "enabled": False,
                "rows": [],
                "remaining_chunks": None,
                "unlocked_remaining_chunks": None,
                "total_delivered_chunks": None,
                "clock_origin_monotonic_ns": None,
            }
        self._require_sampling_owner()
        rows = []
        for row in self._rows.tolist():
            run_id, episode_id = quota["identities"][row].tolist()
            delivered = int(quota["delivered"][row])
            rows.append(
                {
                    "slot": row,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "generation": quota["generations"][row],
                    "assigned": quota["budget"],
                    "delivered": delivered,
                    "remaining": quota["budget"] - delivered,
                    "locked": False,
                }
            )
        remaining = sum(row["remaining"] for row in rows)
        valid = self.data["valid"]
        continuity = self.data.get("continuity_id")
        eligible = (valid if continuity is None else valid & (continuity >= 0)).any(1)
        unlocked = sum(row["remaining"] for row in rows if eligible[row["slot"]])
        return {
            "enabled": True,
            "rows": rows,
            "remaining_chunks": remaining,
            "unlocked_remaining_chunks": unlocked,
            "total_delivered_chunks": sum(row["delivered"] for row in rows),
            "episode_sample_budget": quota["budget"],
            "episode_sample_hard_limit": quota["hard_limit"],
            "clock_origin_monotonic_ns": quota["clock_origin_monotonic_ns"],
        }

    @staticmethod
    def _budget_exhausted() -> None:
        raise DataBudgetExhaustedError(
            "Insufficient offline source quota for a full batch"
        )

    def _draw_rows(
        self, eligible: torch.Tensor, batch_size: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        if self._quota is None:
            return eligible[
                torch.randint(
                    len(eligible), (batch_size,), generator=generator, device="cpu"
                )
            ]
        self._require_sampling_owner()
        remaining = self._quota["budget"] - self._quota["delivered"][eligible]
        if sum(remaining.tolist()) < batch_size:
            self._budget_exhausted()
        return _draw_budgeted_rows(eligible, remaining, batch_size, generator=generator)

    def _check_quota_rows(self, rows: torch.Tensor) -> torch.Tensor | None:
        if self._quota is None:
            return None
        self._require_sampling_owner()
        counts = torch.bincount(rows.reshape(-1), minlength=len(self.data["valid"]))
        if bool((counts > self._quota["budget"] - self._quota["delivered"]).any()):
            self._budget_exhausted()
        return counts

    def _finish_sample(self, rows: torch.Tensor, result: TensorDict) -> TensorDict:
        counts = self._check_quota_rows(rows)
        if counts is not None:
            self._quota["delivered"] += counts
        self._sampling_started = True
        return result

    def sample_context(
        self,
        batch_size: int,
        spec: SequenceSpecCfg,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Draw complete owned contexts, charging quotas only after copying.

        Args:
            batch_size: Positive number of contexts with replacement.
            spec: Window lengths, boundary behavior and generation filter.
            generator: Optional CPU sampling RNG.

        Returns:
            Canonical TensorDict of shape ``[B, H + F - 1]`` with validity and
            supervision masks. Gaps and continuity/segment boundaries pad.
        """
        spec._check()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        valid = self.data["valid"]
        if spec.generations is not None:
            if "generations" not in self.data.keys():
                raise ValueError("Source has no generation metadata")
            valid = valid & torch.isin(
                self.data["generations"],
                torch.tensor(spec.generations, dtype=torch.int64, device="cpu"),
            )
        continuity = self.data.get("continuity_id")
        eligible_valid = valid if continuity is None else valid & (continuity >= 0)
        segments = None
        if spec.sampling_mode == "segment":
            if not {"segment_id", "segment_accepted"} <= set(self.data.keys()):
                raise ValueError(
                    "Segment context requires segment_id and segment_accepted metadata"
                )
            segments = self.data["segment_id"]
            eligible_valid = (
                eligible_valid & (segments >= 0) & self.data["segment_accepted"]
            )
            valid = eligible_valid
        eligible = eligible_valid.any(1).nonzero().flatten()
        if not len(eligible):
            raise DataUnavailableError("No valid offline context anchor")
        rows = self._draw_rows(eligible, batch_size, generator)
        anchors = torch.multinomial(
            eligible_valid[rows].float(), 1, replacement=True, generator=generator
        ).flatten()
        times, mask, loss_mask = _context_indices(
            valid,
            continuity,
            rows,
            anchors,
            spec.history_length,
            spec.prediction_horizon,
            segments,
        )
        result = self._select(rows[:, None], times)
        _pad_context(result, mask, loss_mask)
        return self._finish_sample(rows, result)

    def _select(self, rows: torch.Tensor, times: torch.Tensor) -> TensorDict:
        if self._quota is not None:
            self._require_sampling_owner()
        result = self.data[rows, times]
        if self._quota is not None:
            expected = self._quota["identities"][rows]
            if not torch.all(
                (result["frame_ids"][..., :2] == expected).all(-1) | ~result["valid"]
            ):
                raise RuntimeError("Source identity changed after quota assignment")
        return result
