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

"""Single-producer ACT sampler with optional in-memory replay and a commit journal."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import random
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from embodichain.data_pipeline.engine.data import OnlineDataEngine

MAX_EPISODES = 96
ANCHORS_PER_EPISODE = 32
MAX_HISTORY_DRAWS = 8


class GroupReplay:
    """Own the retired minibatches and the successfully committed step ledger."""

    def __init__(
        self,
        engine: OnlineDataEngine,
        seed: int,
        enabled: bool,
        batch_size: int,
        horizon: int,
        fields: Sequence[str | tuple[str, ...]],
        generator: torch.Generator,
    ) -> None:
        self.engine = engine
        self.enabled = enabled
        self.batch_size = batch_size
        self.horizon = horizon
        self.fields = fields
        self.generator = generator
        self.frame_rng = torch.Generator(device=generator.device).manual_seed(
            seed + 3_100_000
        )
        self.history_rng = random.Random(seed + 3_200_000)
        self.episodes: dict[int, TensorDict] = {}
        self.steps: list[tuple[list[int], list[int]]] = []
        self.release_before_step: list[int] = []
        self.release_monotonic: list[float] = []

    def refill(self) -> int | None:
        """Keep bounded, owned samples only when an occupied group is released."""
        if not self.enabled:
            return self.engine.try_refill_group()
        generation, retired = self.engine.try_refill_group_with_anchors(
            ANCHORS_PER_EPISODE,
            self.horizon,
            anchor_fields=self.fields,
            generator=self.frame_rng,
        )
        for episode in retired:
            episode_id = int(episode["ods_episode_id"][0])
            self.episodes[episode_id] = episode
        while len(self.episodes) > MAX_EPISODES:
            del self.episodes[next(iter(self.episodes))]
        return generation

    @contextmanager
    def draw(self) -> Iterator[TensorDict]:
        """Mix live and distinct retired episodes, accounting after a successful update."""
        count = min(MAX_HISTORY_DRAWS, self.batch_size - 1, len(self.episodes))
        with self.engine.group_batch(
            self.batch_size - count,
            self.horizon,
            anchor_fields=self.fields,
            generator=self.generator,
        ) as live:
            ids = self.history_rng.sample(list(self.episodes), count) if count else []
            parts = [live]
            for episode_id in ids:
                offset = self.history_rng.randrange(ANCHORS_PER_EPISODE)
                parts.append(self.episodes[episode_id][offset : offset + 1])
            yield torch.cat(parts) if ids else live
        self.steps.append((live["ods_episode_id"].tolist(), ids))

    def evidence(self) -> dict:
        """Return per-step episode IDs, release boundaries and retained tensor bytes."""
        byte_count = sum(
            t.numel() * t.element_size()
            for episode in self.episodes.values()
            for t in episode.values(True, True)
        )
        return dict(
            steps=self.steps,
            release_before_step=self.release_before_step,
            release_monotonic=self.release_monotonic,
            history_bytes=byte_count,
        )
