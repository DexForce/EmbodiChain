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

"""Coverage reservation policy separated from the session lifecycle."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Protocol, runtime_checkable
import torch
from .contracts import SceneCase, ExpertEpisode

__all__ = ["ExpansionCoverage", "EpisodeCoverage"]


@runtime_checkable
class ExpansionCoverage(Protocol):
    """Session-owned coverage reservations; implementations never step or write data."""

    def register_case(
        self,
        case: SceneCase,
        joint_limits: torch.Tensor,
        joint_names: tuple[str, ...],
        reference: float | None,
    ) -> None:
        """Register fixed normalization conditions, rejecting unsupported references."""
        ...

    def reserve(self, episode: ExpertEpisode) -> bool:
        """Reserve measured coverage until the final write receipt arrives."""
        ...

    def confirm(self, case_id: str, commit_id: str) -> None:
        """Confirm a pending coverage reservation once."""
        ...

    def release(self, case_id: str, commit_id: str) -> None:
        """Release a failed write reservation."""
        ...

    def reference(self, case: SceneCase) -> float | None:
        """Return the registered optional domain guidance reference."""
        ...

    def snapshot(self) -> tuple[Mapping[str, int], Mapping[str, Mapping[int, int]]]:
        """Return independent committed coverage and optional domain-band counts."""
        ...


class EpisodeCoverage:
    """Count confirmed episode identities without applying trajectory deduplication."""

    def __init__(self) -> None:
        self._cases: dict[str, set[str]] = {}
        self._pending: dict[str, set[str]] = {}

    def register_case(
        self,
        case: SceneCase,
        joint_limits: torch.Tensor,
        joint_names: tuple[str, ...],
        reference: float | None,
    ) -> None:
        """Register a case; numerical guidance needs an explicit domain policy."""
        if reference is not None:
            raise ValueError(
                "manipulability_reference requires a trajectory coverage policy"
            )
        self._cases.setdefault(case.scene_case_id, set())
        self._pending.setdefault(case.scene_case_id, set())

    def reserve(self, episode: ExpertEpisode) -> bool:
        """Reserve an accepted episode without evaluating its trajectory."""
        self._pending[episode.identity.scene_case_id].add(episode.commit_id)
        return True

    def confirm(self, case_id: str, commit_id: str) -> None:
        """Confirm the episode associated with a valid session receipt."""
        self._pending[case_id].remove(commit_id)
        self._cases[case_id].add(commit_id)

    def release(self, case_id: str, commit_id: str) -> None:
        """Discard the failed submission's coverage reservation."""
        self._pending[case_id].discard(commit_id)

    def reference(self, case: SceneCase) -> float | None:
        """Episode counting has no numerical guidance reference."""
        return None

    def snapshot(self) -> tuple[Mapping[str, int], Mapping[str, Mapping[int, int]]]:
        """Return independent committed episode counts per case."""
        return {key: len(values) for key, values in self._cases.items()}, {
            key: {} for key in self._cases
        }
