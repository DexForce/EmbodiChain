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

"""Measured trajectory coverage adapter for the source-independent session."""

from __future__ import annotations
from collections.abc import Mapping
import math
import torch
from .cfg import TrajectoryAugmentationCfg
from .coverage import CoverageIndex, describe_trajectory
from .manipulability import ManipulabilityBands
from embodichain.lab.expansion.contracts import ExpertEpisode, SceneCase

__all__ = ["TrajectorySessionCoverage"]


class TrajectorySessionCoverage:
    """Keep legacy phase geometry, timing and manipulability admission unchanged."""

    def __init__(self, cfg: TrajectoryAugmentationCfg) -> None:
        self._cfg = TrajectoryAugmentationCfg.from_mapping(cfg.to_dict())
        self._cases = {}
        self._coverage = {}
        self._bands = {}

    def register_case(
        self,
        case: SceneCase,
        joint_limits: torch.Tensor,
        joint_names: tuple[str, ...],
        reference: float | None,
    ) -> None:
        """Register domain normalization using independently owned numerical limits."""
        factor = self._cfg.factors.manipulability
        if factor.enabled:
            if (
                type(reference) not in (float, int)
                or not math.isfinite(reference)
                or reference <= 0
            ):
                raise ValueError(
                    "Banded coverage requires a positive manipulability_reference."
                )
            reference = float(reference)
        elif reference is not None:
            raise ValueError(
                "manipulability_reference requires factors.manipulability.enabled."
            )
        key = (case.scene_case_id, case.initial_state_id)
        self._cases[key] = (case, joint_limits.clone(), joint_names, reference)
        if case.scene_case_id not in self._coverage:
            coverage = self._cfg.coverage
            self._coverage[case.scene_case_id] = CoverageIndex(
                geometry_tolerance=coverage.joint_dedup_normalized_tol,
                target_per_geometry=coverage.target_per_cell,
                target_per_band=factor.target_per_band if factor.enabled else None,
            )
        if factor.enabled:
            self._bands[key] = ManipulabilityBands(
                factor.band_edges, reference=reference
            )

    def reference(self, case: SceneCase) -> float | None:
        """Return the exact registered initial state's guidance reference."""
        return self._cases[(case.scene_case_id, case.initial_state_id)][3]

    def _measured_band(self, episode: ExpertEpisode) -> int | None:
        """Classify measured manipulability; planned scores are never accepted.

        The normalizer belongs to the episode's own initial state, so rollouts
        started from different reference trajectories are ranked against the
        reference each one was registered with.
        """
        identity = episode.identity
        bands = self._bands.get((identity.scene_case_id, identity.initial_state_id))
        if bands is None:
            return None
        scores = episode.observations.get("manipulability")
        if (
            scores is None
            or scores.ndim != 1
            or scores.shape != episode.timestamps.shape
            or bool((scores < 0).any())
        ):
            raise ValueError(
                "Banded coverage requires one non-negative measured manipulability "
                "value per observation."
            )
        return bands.band_of(float(scores.min()))

    def reserve(self, episode: ExpertEpisode) -> bool:
        identity = episode.identity
        limits = self._cases[(identity.scene_case_id, identity.initial_state_id)][1]
        descriptor = describe_trajectory(
            episode.observations["joint_positions"],
            episode.timestamps,
            limits,
            phases=episode.phases,
            samples_per_phase=self._cfg.coverage.geometry_samples_per_phase,
        )
        return self._coverage[identity.scene_case_id].reserve(
            episode.commit_id,
            identity.geometry_family_id,
            descriptor,
            band=self._measured_band(episode),
        )

    def confirm(self, case_id: str, commit_id: str) -> None:
        """Confirm trajectory coverage after persistence acknowledgement."""
        self._coverage[case_id].confirm(commit_id)

    def release(self, case_id: str, commit_id: str) -> None:
        """Release trajectory coverage after a failed write."""
        self._coverage[case_id].release(commit_id)

    def snapshot(self) -> tuple[Mapping[str, int], Mapping[str, Mapping[int, int]]]:
        """Return legacy geometry and manipulability counts."""
        return {k: v.geometry_count for k, v in self._coverage.items()}, {
            k: v.band_counts for k, v in self._coverage.items()
        }
