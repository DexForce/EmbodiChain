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

"""Owned visual training blocks, causal geometry and bounded source sampling."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase

from examples.ods.simple_bc_demo.models import pose_targets
from examples.ods.simple_bc_demo.observations import (
    _geometry_statistics,
    policy_observations,
)

from .payload import _numeric, _provenance

__all__ = [
    "unpack_geometry",
    "geometry_from_raw",
    "make_statistics",
    "BalancedAnchorSampler",
    "RetainedBlock",
    "build_offline_plan",
]

Plan = list[tuple[int, int]]


def unpack_geometry(state: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
    """Reconstruct measured geometry from the canonical ordered numeric state."""
    if state.shape[-1] != metadata["state_width"] or state.dtype != torch.float32:
        raise ValueError("Canonical float32 state width differs from payload metadata")
    obs = TensorDict({}, batch_size=state.shape[:-1], device=state.device)
    offset = 0
    for feature in metadata["state_features"]:
        width = feature["width"]
        obs[tuple(feature["path"])] = state[..., offset : offset + width].reshape(
            *state.shape[:-1], *feature["shape"]
        )
        offset += width
    if offset != state.shape[-1]:
        raise ValueError("State feature widths do not cover the canonical state")
    return policy_observations(obs)


def geometry_from_raw(raw: TensorDictBase) -> torch.Tensor:
    """Pack current measured features; optional invalid frames remain neutral."""
    return policy_observations(raw["obs"], valid_mask=raw.get("valid", None))


def make_statistics(
    episodes: list[TensorDictBase], metadata: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Freeze statistics using only valid physical frames in the initial block."""
    if not episodes:
        raise ValueError("Statistics require a nonempty initial block")
    geometry, targets = [], []
    for episode in episodes:
        values = geometry_from_raw(episode)[episode["valid"]]
        if not len(values):
            raise ValueError("Statistics cannot include an empty accepted episode")
        if episode["actions"].shape[-1] != metadata["action_width"]:
            raise ValueError("Recorded action width differs from payload metadata")
        geometry.append(values)
        targets.append(pose_targets(values, episode["actions"][episode["valid"]]))
    values, target = torch.cat(geometry), torch.cat(targets)
    mean, std = _geometry_statistics(values)
    target_mean = target.mean(0)
    target_std = target.std(0, correction=0).clamp_min(0.05)
    target_mean[3:], target_std[3:] = 0, 1
    result = {
        "proprio_mean": mean[:27].clone(),
        "proprio_std": std[:27].clone(),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Initial physical frames produced nonfinite statistics")
    return result


class BalancedAnchorSampler:
    """One in-flight batch, balanced sources and per-source anchor permutations.

    Records contain ``ordinal``, ``length``, ``source_id`` and optionally exact
    ``legal_anchors``. Permutations are exhausted before reshuffling. Reservations
    protect K before delivery; training is acknowledged only by the caller.
    """

    def __init__(
        self,
        episodes: list[dict[str, Any]],
        *,
        history_length: int = 2,
        prediction_horizon: int = 32,
        seed: int = 29001,
        episode_budget: int | None = None,
    ) -> None:
        if not episodes or any(
            type(v) is not int or v < 1 for v in (history_length, prediction_horizon)
        ):
            raise ValueError("Sampling requires episodes and positive H/F")
        if episode_budget is not None and (
            type(episode_budget) is not int or episode_budget < 1
        ):
            raise ValueError("episode_budget must be a positive integer or None")
        self.records = copy.deepcopy(episodes)
        self.budget = episode_budget
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(episodes), generator=self.generator).tolist()
        self.cursor = 0
        self.anchors: dict[int, list[int]] = {}
        self.permutations: dict[int, list[int]] = {}
        self.positions: dict[int, int] = {}
        for index, episode in enumerate(self.records):
            ordinal = episode.setdefault("ordinal", episode.get("index", index))
            if type(ordinal) is not int or ordinal in self.anchors:
                raise ValueError("Episode ordinals must be unique integers")
            anchors = list(
                episode.get(
                    "legal_anchors",
                    range(
                        history_length - 1, episode["length"] - prediction_horizon + 1
                    ),
                )
            )
            if (
                not anchors
                or len(set(anchors)) != len(anchors)
                or any(
                    type(a) is not int
                    or a < history_length - 1
                    or a + prediction_horizon > episode["length"]
                    for a in anchors
                )
            ):
                raise ValueError(
                    "Each source requires distinct complete-window anchors"
                )
            self.anchors[ordinal] = anchors
            self.permutations[ordinal] = []
            self.positions[ordinal] = 0
        self.delivered = Counter({ordinal: 0 for ordinal in self.anchors})
        self.reserved = self.delivered.copy()
        self.trained = self.delivered.copy()
        self._pending: Plan | None = None
        self._delivered: Plan | None = None

    def reserve_exact(self, plan: Plan) -> Plan:
        """Reserve a caller-owned serial trace without changing its anchor order."""
        if self._pending is not None or self._delivered is not None:
            raise RuntimeError("A previous batch is still reserved or not trained")
        if not plan or any(
            ordinal not in self.anchors or anchor not in self.anchors[ordinal]
            for ordinal, anchor in plan
        ):
            raise ValueError("Exact plan contains an unknown source or illegal anchor")
        counts = Counter(ordinal for ordinal, _ in plan)
        if self.budget is not None and any(
            self.delivered[ordinal] + count > self.budget
            for ordinal, count in counts.items()
        ):
            raise StopIteration(
                "Insufficient protected source credits for a complete batch"
            )
        self.reserved.update(counts)
        self._pending = list(plan)
        return plan

    def reserve(self, batch_size: int) -> Plan:
        """Reserve a balanced complete batch without spending delivery credits."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self._pending is not None or self._delivered is not None:
            raise RuntimeError("A previous batch is still reserved or not trained")
        if (
            self.budget is not None
            and sum(self.budget - value for value in self.delivered.values())
            < batch_size
        ):
            raise StopIteration(
                "Insufficient protected source credits for a complete batch"
            )
        plan: Plan = []
        counts: Counter[int] = Counter()
        while len(plan) < batch_size:
            row = self.order[self.cursor % len(self.order)]
            self.cursor += 1
            ordinal = self.records[row]["ordinal"]
            if (
                self.budget is not None
                and self.delivered[ordinal] + counts[ordinal] >= self.budget
            ):
                continue
            if self.positions[ordinal] >= len(self.permutations[ordinal]):
                indices = torch.randperm(
                    len(self.anchors[ordinal]), generator=self.generator
                ).tolist()
                self.permutations[ordinal] = [
                    self.anchors[ordinal][index] for index in indices
                ]
                self.positions[ordinal] = 0
            anchor = self.permutations[ordinal][self.positions[ordinal]]
            self.positions[ordinal] += 1
            plan.append((ordinal, anchor))
            counts[ordinal] += 1
        return self.reserve_exact(plan)

    def cancel(self, plan: Plan) -> None:
        """Release a failed materialization without spending delivery credits."""
        if self._pending != plan:
            raise RuntimeError("Cancellation differs from the reserved batch")
        self.reserved.subtract(ordinal for ordinal, _ in plan)
        self._pending = None

    def mark_delivered(self, plan: Plan) -> None:
        """Commit delivery only after the entire batch has materialized."""
        if self._pending != plan:
            raise RuntimeError("Delivery differs from the reserved batch")
        self.reserved.subtract(ordinal for ordinal, _ in plan)
        self.delivered.update(ordinal for ordinal, _ in plan)
        self._pending, self._delivered = None, list(plan)

    def mark_trained(self, plan: Plan) -> None:
        """Acknowledge exactly one successfully completed optimizer update."""
        if self._delivered != plan:
            raise RuntimeError("Training acknowledgement differs from delivery")
        self.trained.update(ordinal for ordinal, _ in plan)
        self._delivered = None

    def snapshot(self) -> dict[str, Any]:
        """Return immutable-source delivery/training counters for retirement."""
        return {
            "episode_budget": self.budget,
            "reserved_chunks": sum(self.reserved.values()),
            "delivered_chunks": sum(self.delivered.values()),
            "trained_chunks": sum(self.trained.values()),
            "episodes": [
                {
                    "ordinal": row["ordinal"],
                    "source_id": list(row["source_id"]),
                    "reserved": self.reserved[row["ordinal"]],
                    "delivered": self.delivered[row["ordinal"]],
                    "trained": self.trained[row["ordinal"]],
                }
                for row in self.records
            ],
        }


class RetainedBlock:
    """Own at most one producer block; gather only causal image histories.

    Ownership of the supplied raw episodes transfers to this object. Returned
    batches have independent storage and remain valid after the block retires.
    """

    def __init__(
        self,
        episodes: list[TensorDictBase],
        metadata: Mapping[str, Any],
        *,
        history_length: int = 2,
        prediction_horizon: int = 32,
        seed: int = 29001,
        episode_budget: int | None = 128,
        episode_ordinals: list[int] | None = None,
    ) -> None:
        if not episodes or (
            episode_ordinals is not None and len(episode_ordinals) != len(episodes)
        ):
            raise ValueError("A retained block requires aligned owned episodes")
        self.metadata = copy.deepcopy(dict(metadata))
        self.history_length, self.prediction_horizon = (
            history_length,
            prediction_horizon,
        )
        self.length = history_length + prediction_horizon - 1
        self.episodes: dict[int, dict[str, Any]] = {}
        records = []
        for index, raw in enumerate(episodes):
            if raw.batch_dims != 1 or raw["actions"].device.type != "cpu":
                raise ValueError("Retained episodes require one CPU time axis")
            valid = raw["valid"].bool()
            length = int(valid.sum())
            if not length or not torch.equal(valid, torch.arange(len(valid)) < length):
                raise ValueError("Accepted episodes require a nonempty valid prefix")
            provenance = _provenance(raw)
            ids = provenance["frame_ids"][:length]
            continuity = provenance["continuity_id"][:length]
            if (
                (ids < 0).any()
                or (ids[:, :2] != ids[0, :2]).any()
                or (ids[1:, 2] != ids[:-1, 2] + 1).any()
                or (provenance["generations"][:length] < 1).any()
                or (
                    provenance["generations"][:length] != provenance["generations"][0]
                ).any()
            ):
                raise ValueError(
                    "Accepted source frames must be contiguous and immutable"
                )
            windows = (
                continuity.unfold(0, self.length, 1)
                if length >= self.length
                else torch.empty((0, self.length), dtype=torch.int64)
            )
            legal = (
                ((windows == windows[:, :1]).all(-1) & (windows >= 0).all(-1))
                .nonzero()
                .flatten()
                + history_length
                - 1
            )
            ordinal = (
                int(ids[0, 1]) if episode_ordinals is None else episode_ordinals[index]
            )
            if ordinal in self.episodes:
                raise ValueError("Retained episode ordinals must be unique")
            record = {
                "ordinal": ordinal,
                "length": length,
                "source_id": ids[0, :2].tolist(),
                "generation": int(provenance["generations"][0]),
                "legal_anchors": legal.tolist(),
            }
            records.append(record)
            self.episodes[ordinal] = {
                "raw": raw,
                "state": _numeric(raw, metadata),
                "geometry": geometry_from_raw(raw),
                **provenance,
            }
        self.sampler = BalancedAnchorSampler(
            records,
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            seed=seed,
            episode_budget=episode_budget,
        )
        self.records = self.sampler.records

    def next_plan(self, batch_size: int) -> Plan:
        """Reserve the next batch; the first batch must cover every source."""
        if not sum(self.sampler.delivered.values()) and (
            batch_size < len(self.records) or batch_size % len(self.records)
        ):
            raise ValueError("First retained batch must evenly cover every source")
        return self.sampler.reserve(batch_size)

    def materialize(self, plan: Plan, *, reserve: bool = False) -> dict[str, Any]:
        """Gather independent numeric windows and only H RGB image frames."""
        if reserve:
            if not sum(self.sampler.delivered.values()) and (
                Counter(ordinal for ordinal, _ in plan)
                != Counter(
                    {
                        record["ordinal"]: len(plan) // len(self.records)
                        for record in self.records
                    }
                )
            ):
                raise ValueError("First exact batch must evenly cover every source")
            self.sampler.reserve_exact(plan)
        if self.sampler._pending != plan:
            raise RuntimeError("Materialization requires an exact reserved plan")
        try:
            numeric: dict[str, list[torch.Tensor]] = {
                name: []
                for name in (
                    "state",
                    "actions",
                    "geometry",
                    "frame_ids",
                    "generations",
                    "continuity_id",
                    "published_at_monotonic_ns",
                )
            }
            images = {
                name: torch.empty(
                    (len(plan), self.history_length, 3, *feature["shape"][:2]),
                    dtype=torch.float32,
                )
                for name, feature in self.metadata["cameras"].items()
            }
            divisor = torch.tensor(255.0, dtype=torch.float32, device="cpu")
            for batch, (ordinal, anchor) in enumerate(plan):
                episode = self.episodes[ordinal]
                first = anchor - self.history_length + 1
                selection = slice(first, anchor + self.prediction_horizon)
                for name in numeric:
                    value = (
                        episode["raw"]["actions"]
                        if name == "actions"
                        else episode[name]
                    )
                    numeric[name].append(value[selection])
                for name, feature in self.metadata["cameras"].items():
                    rgb = episode["raw"][("obs", *feature["path"])][
                        first : anchor + 1, ..., :3
                    ].movedim(-1, -3)
                    if (
                        rgb.dtype != torch.uint8
                        or rgb.shape != images[name][batch].shape
                    ):
                        raise ValueError(
                            "Native camera layout differs from payload metadata"
                        )
                    torch.div(rgb, divisor, out=images[name][batch])
            result = {name: torch.stack(values) for name, values in numeric.items()}
            result["images"] = images
            result["valid"] = torch.ones((len(plan), self.length), dtype=torch.bool)
            result["loss_mask"] = (
                (torch.arange(self.length) >= self.history_length - 1)
                .expand(len(plan), -1)
                .clone()
            )
            self.sampler.mark_delivered(plan)
            return result
        except BaseException:
            self.sampler.cancel(plan)
            raise

    def mark_trained(self, plan: Plan) -> None:
        """Commit training only after the host observes a completed update."""
        self.sampler.mark_trained(plan)

    def snapshot(self) -> dict[str, Any]:
        """Return lifetime quota counters, including retired-source identity."""
        return self.sampler.snapshot()


def build_offline_plan(
    manifest: Mapping[str, Any],
    updates: int,
    batch_size: int,
    seed: int,
    *,
    history_length: int = 2,
    prediction_horizon: int = 32,
) -> list[Plan]:
    """Build finite uncapped plans using the same balanced/permutation rule."""
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer")
    sampler = BalancedAnchorSampler(
        list(manifest["episodes"]),
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        seed=seed,
    )
    result = []
    for _ in range(updates):
        plan = sampler.reserve(batch_size)
        sampler.mark_delivered(plan)
        sampler.mark_trained(plan)
        result.append(plan)
    return result
