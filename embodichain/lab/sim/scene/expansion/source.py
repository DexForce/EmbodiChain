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

"""Bounded fixed candidates for scene expansion hosts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

from .cfg import SceneExpansionCfg
from .contracts import ScenePoseChange, SceneVariant, _identifier

__all__ = ["FixedSceneCandidateSource"]


class FixedSceneCandidateSource:
    """Snapshot a bounded prefix of authored proposals for repeatable iteration.

    Args:
        cfg: Proposal budget and allowed physical entity UIDs.
        parent_scene_id: Identity of the scene the proposals modify.
        candidates: Iterable of pose-change iterables. At most
            ``cfg.max_candidates`` entries are consumed; later entries are
            deliberately untouched. An empty pose-change iterable describes
            an explicit nominal proposal.
        metadata: Immutable string key/value provenance added to each proposal.

    The source owns no simulator, retries, or execution state. Configuration
    is revalidated at this boundary and no mutable caller configuration is
    retained. Every fresh iteration returns the same immutable proposals.
    """

    def __init__(
        self,
        cfg: SceneExpansionCfg,
        parent_scene_id: str,
        candidates: Iterable[Iterable[ScenePoseChange]],
        *,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if not isinstance(cfg, SceneExpansionCfg):
            raise TypeError("cfg must be a SceneExpansionCfg")
        cfg.validate()
        validated = cfg.copy()
        _identifier(parent_scene_id, "parent_scene_id")
        # Validate source metadata even if no candidates are available.
        source_metadata = SceneVariant(
            parent_scene_id, validated.seed, 0, (), metadata
        ).metadata
        permitted = set(validated.movable_entity_ids)
        variants = []
        for ordinal, changes in enumerate(islice(candidates, validated.max_candidates)):
            variant = SceneVariant(
                parent_scene_id,
                validated.seed,
                ordinal,
                tuple(changes),
                source_metadata,
            )
            unsupported = {
                change.entity_uid for change in variant.pose_changes
            } - permitted
            if unsupported:
                raise ValueError(
                    f"candidate changes entities not declared movable: {sorted(unsupported)}"
                )
            variants.append(variant)
        self._variants = tuple(variants)

    @property
    def variants(self) -> tuple[SceneVariant, ...]:
        """Return the bounded, immutable snapshot of candidate proposals.

        Returns:
            Candidates in source order, with consecutive zero-based ordinals.
        """
        return self._variants

    def __iter__(self) -> Iterator[SceneVariant]:
        """Return a fresh iterator over the candidate snapshot.

        Returns:
            Iterator independent of previous iterations or caller mutations.
        """
        return iter(self._variants)

    def __len__(self) -> int:
        """Return the number of snapshotted candidates.

        Returns:
            Candidate count, bounded by the source configuration.
        """
        return len(self._variants)
