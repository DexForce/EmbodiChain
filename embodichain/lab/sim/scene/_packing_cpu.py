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

"""CPU support-constrained greedy packing policy with tensor broad phases."""

from __future__ import annotations
import numpy as np
import torch
from embodichain.compute.geometry import aabb_overlap_depth, aabb_containment_violation
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from embodichain.utils import configclass
from embodichain.utils.logger import log_info, log_warning
from ._placement_cpu import GroupSupportClamp, SupportGeometry

__all__ = ["AABBLayoutConfig", "AABBLayoutOptimizer"]


@configclass
class AABBLayoutConfig:
    """Controls for support-constrained pairwise AABB separation."""

    margin_m: float = 0.0  # Required clearance between each AABB and the boundary.
    aabb_clearance_m: float = 1e-6  # Required clearance between AABB pairs.
    max_rounds: int = 64  # Maximum greedy pair-separation passes.
    split_samples: int = 9  # Candidate splits between the two overlapping AABBs.


class AABBLayoutOptimizer:
    """Greedily separate XY AABBs while checking the full support polygon."""

    def __init__(self, *, config: AABBLayoutConfig | None = None) -> None:
        """Configure deterministic greedy separation.

        Args:
            config: Boundary, clearance and search budget parameters.
        """
        self.config = config if config is not None else AABBLayoutConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Recheck mutable controls at each public solve boundary."""
        self.config.validate()
        if (
            not np.isfinite([self.config.margin_m, self.config.aabb_clearance_m]).all()
            or min(self.config.margin_m, self.config.aabb_clearance_m) < 0
        ):
            raise ValueError("Margins and clearances must be finite and non-negative.")
        if (
            type(self.config.max_rounds) is not int
            or type(self.config.split_samples) is not int
            or self.config.max_rounds <= 0
            or self.config.split_samples < 2
        ):
            raise ValueError(
                "max_rounds must be a positive integer and split_samples an integer >= 2."
            )

    def solve(
        self, support_region: SupportGeometry, aabbs_by_id: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Separate AABBs within a support polygon using greedy pair proposals.

        Args:
            support_region: Allowed XY region.
            aabbs_by_id: Finite four-corner AABBs per logical object.

        Returns:
            Object UID to XY translation mapping.

        Raises:
            ValueError: Invalid input or no placement found within the budget.
        """
        self._validate_config()
        # Check inputs just like the previous GroupSupportClamp step would have done.
        aabbs_by_id = GroupSupportClamp._validate_aabbs(aabbs_by_id)
        raw_support = GroupSupportClamp._coerce_support_geometry(support_region)
        if raw_support is None:
            log_warning("AABB overlap optimization failed: invalid support geometry.")
            raise ValueError("Support region is invalid.")
        safe_support = (
            raw_support
            if self.config.margin_m == 0.0
            else GroupSupportClamp._polygonal_geometry(
                raw_support.buffer(-self.config.margin_m)
            )
        )
        if safe_support is None or safe_support.is_empty:
            log_warning(
                "AABB overlap optimization failed: support region is empty after "
                f"applying a {self.config.margin_m:.4f} m boundary margin."
            )
            raise ValueError("Support region is empty after applying layout margin.")

        asset_ids = sorted(aabbs_by_id)
        base_aabbs = np.stack([aabbs_by_id[asset_id] for asset_id in asset_ids])
        offsets = np.zeros((len(asset_ids), 2), dtype=float)
        if not self._all_contained(safe_support, base_aabbs, offsets):
            # Independently project each AABB into the rectangular optimization region.
            offsets = self._project_aabbs_inside_rectangle(safe_support, base_aabbs)
            if not self._all_contained(safe_support, base_aabbs, offsets):
                raise ValueError("Projected AABBs are not contained in support.")
        initial_overlaps = self._overlaps(base_aabbs, offsets)
        projected_asset_count = int(np.count_nonzero(np.any(offsets != 0.0, axis=1)))
        log_info(
            "Support-constrained AABB overlap optimization started: "
            f"assets={len(asset_ids)}, initial_overlaps={len(initial_overlaps)}, "
            f"initial_projections={projected_asset_count}, "
            f"boundary_margin={self.config.margin_m:.4f} m, "
            f"aabb_clearance={self.config.aabb_clearance_m:.4f} m, "
            f"max_rounds={self.config.max_rounds}."
        )
        if not initial_overlaps:  # Return directly if there are no overlaps to resolve.
            log_info("AABB overlap optimization succeeded without pair separation.")
            return {uid: offsets[index].copy() for index, uid in enumerate(asset_ids)}

        for round_index in range(self.config.max_rounds):
            # Check whether any overlaps remain.
            overlaps = self._overlaps(base_aabbs, offsets)
            if not overlaps:
                log_info(
                    "AABB overlap optimization succeeded after "
                    f"{round_index} rounds."
                )
                return {
                    uid: offsets[index].copy() for index, uid in enumerate(asset_ids)
                }
            log_info(
                "AABB overlap optimization round "
                f"{round_index + 1}/{self.config.max_rounds}: "
                f"remaining_overlaps={len(overlaps)}."
            )
            moved = False
            # Choose a pair once.
            for _, first_index, second_index in overlaps:
                # If the overlaps is handeled by a previous pair, skip it.
                if not self._overlaps(base_aabbs, offsets, (first_index, second_index)):
                    continue
                candidates = self._separation_candidates(
                    base_aabbs=base_aabbs,
                    offsets=offsets,
                    first_index=first_index,
                    second_index=second_index,
                    safe_support=safe_support,
                )
                if candidates:
                    # A local separation must not blindly create a new
                    # collision with a third asset.  Prefer candidates with
                    # no new collisions; when every placement causes one,
                    # retain the least-colliding state before considering
                    # displacement from the generated layout.
                    offsets = min(
                        candidates,
                        key=lambda candidate: self._candidate_score(
                            base_aabbs=base_aabbs,
                            current_offsets=offsets,
                            candidate_offsets=candidate,
                        ),
                    )  # Choose the best candidate with score.
                    moved = True
                else:
                    log_warning(
                        "No support-valid axis-aligned separation candidate for "
                        f"overlapping AABBs {asset_ids[first_index]!r} and "
                        f"{asset_ids[second_index]!r}."
                    )
            if not moved:
                break

        if not self._overlaps(base_aabbs, offsets):
            return {uid: offsets[index].copy() for index, uid in enumerate(asset_ids)}

        unresolved_pairs = [
            f"{asset_ids[first_index]}/{asset_ids[second_index]}"
            for _, first_index, second_index in self._overlaps(base_aabbs, offsets)
        ]
        log_warning(
            "Unable to resolve all asset AABB overlaps inside the detected support "
            "region; unresolved pairs="
            f"{unresolved_pairs}."
        )
        raise ValueError(
            "Asset AABB overlap cannot be resolved within the search budget while keeping all assets "
            "inside the detected table support region."
        )

    @staticmethod
    def _project_aabbs_inside_rectangle(
        support: Polygon | MultiPolygon,
        base_aabbs: np.ndarray,
    ) -> np.ndarray:
        """Return minimum per-AABB offsets that place AABBs in a rectangle."""
        if not isinstance(support, Polygon) or support.interiors:
            raise ValueError(
                "Initial AABB projection requires an axis-aligned rectangular "
                "support region."
            )
        minimum_x, minimum_y, maximum_x, maximum_y = support.bounds
        rectangle = Polygon(
            [
                (minimum_x, minimum_y),
                (maximum_x, minimum_y),
                (maximum_x, maximum_y),
                (minimum_x, maximum_y),
            ]
        )
        if not support.equals(rectangle):
            raise ValueError(
                "Initial AABB projection requires an axis-aligned rectangular "
                "support region."
            )

        aabb_minimums, aabb_maximums = base_aabbs.min(axis=1), base_aabbs.max(axis=1)
        half_extents = (aabb_maximums - aabb_minimums) / 2.0
        support_minimum = np.array([minimum_x, minimum_y], dtype=float)
        support_maximum = np.array([maximum_x, maximum_y], dtype=float)
        valid_center_minimums = support_minimum + half_extents
        valid_center_maximums = support_maximum - half_extents
        if np.any(valid_center_minimums > valid_center_maximums):
            raise ValueError(
                "An asset AABB is larger than the rectangular support region."
            )

        centers = (aabb_minimums + aabb_maximums) / 2.0
        # A center must stay inset from each boundary by its AABB half extent.
        projected_centers = np.clip(
            centers, valid_center_minimums, valid_center_maximums
        )
        return projected_centers - centers

    def _separation_candidates(
        self,
        *,
        base_aabbs: np.ndarray,
        offsets: np.ndarray,
        first_index: int,
        second_index: int,
        safe_support: Polygon | MultiPolygon,
    ) -> list[np.ndarray]:
        # Get current aabbs.
        current_aabbs = base_aabbs + offsets[:, None, :]
        minimums, maximums = current_aabbs.min(axis=1), current_aabbs.max(axis=1)
        candidates: list[np.ndarray] = []
        for axis in (0, 1):
            directions_and_distances = (
                (
                    -1.0,
                    maximums[first_index, axis]
                    + self.config.aabb_clearance_m
                    - minimums[second_index, axis],
                ),
                (
                    1.0,
                    maximums[second_index, axis]
                    + self.config.aabb_clearance_m
                    - minimums[first_index, axis],
                ),
            )
            for first_direction, required_distance in directions_and_distances:
                if required_distance <= 0.0:
                    continue
                for fraction in np.linspace(0.0, 1.0, self.config.split_samples):
                    candidate = offsets.copy()
                    first_move = required_distance * float(fraction)
                    candidate[first_index, axis] += first_direction * first_move
                    candidate[second_index, axis] -= first_direction * (
                        required_distance - first_move
                    )
                    if not self._overlaps(
                        base_aabbs, candidate, (first_index, second_index)
                    ) and self._all_contained(safe_support, base_aabbs, candidate):
                        candidates.append(candidate)
        return candidates

    def _overlaps(
        self,
        base_aabbs: np.ndarray,
        offsets: np.ndarray,
        only_pair: tuple[int, int] | None = None,
    ) -> list[tuple[float, int, int]]:
        return sorted(
            [
                (min(overlap_x, overlap_y), first_index, second_index)
                for overlap_x, overlap_y, first_index, second_index in self._overlap_details(
                    base_aabbs, offsets, only_pair
                )
            ],
            reverse=True,
        )

    def _overlap_details(
        self,
        base_aabbs: np.ndarray,
        offsets: np.ndarray,
        only_pair: tuple[int, int] | None = None,
    ) -> list[tuple[float, float, int, int]]:
        """Return positive XY penetration extents, including requested clearance."""
        if only_pair is not None:
            # Candidate search requests one pair repeatedly. Select before both
            # geometry reduction and tensor conversion, independent of scene size.
            selected = np.asarray(only_pair)
            current_aabbs = base_aabbs[selected] + offsets[selected, None, :]
            first = np.array([0])
            second = np.array([1])
            report_first, report_second = (np.array([index]) for index in only_pair)
        else:
            current_aabbs = base_aabbs + offsets[:, None, :]
            first, second = np.triu_indices(len(current_aabbs), k=1)
            report_first, report_second = first, second
        minimums, maximums = current_aabbs.min(axis=1), current_aabbs.max(axis=1)
        margin = self.config.aabb_clearance_m / 2.0
        bounds = np.stack((minimums - margin, maximums + margin), axis=1)
        # Evaluate only requested pairs, once in a batch; omit diagonal and
        # symmetric duplicates rather than allocating an N-by-N depth array.
        depths = aabb_overlap_depth(
            torch.from_numpy(bounds[first]), torch.from_numpy(bounds[second])
        ).numpy()
        active = np.flatnonzero(np.all(depths > 1e-9, axis=1))
        return [
            (
                float(depths[index, 0]),
                float(depths[index, 1]),
                int(report_first[index]),
                int(report_second[index]),
            )
            for index in active
        ]

    def _candidate_score(
        self,
        *,
        base_aabbs: np.ndarray,
        current_offsets: np.ndarray,
        candidate_offsets: np.ndarray,
    ) -> tuple[int, int, float, float]:
        """Rank a valid pair-separation candidate by global collision impact.

        The first term is deliberately based on *new* overlap pairs: this
        keeps a pairwise correction from simply transferring its collision to
        a nearby third asset.  If every candidate causes a new collision, the
        remaining terms prefer fewer total overlaps, less total penetration,
        then less layout displacement.
        """
        current_pairs = {
            (first_index, second_index)
            for _, _, first_index, second_index in self._overlap_details(
                base_aabbs, current_offsets
            )
        }
        candidate_details = self._overlap_details(base_aabbs, candidate_offsets)
        candidate_pairs = {
            (first_index, second_index)
            for _, _, first_index, second_index in candidate_details
        }
        new_overlap_count = len(candidate_pairs - current_pairs)
        total_penetration_area = sum(
            overlap_x * overlap_y for overlap_x, overlap_y, _, _ in candidate_details
        )
        total_squared_displacement = float(
            np.einsum("ij,ij->", candidate_offsets, candidate_offsets)
        )
        return (
            new_overlap_count,
            len(candidate_pairs),
            total_penetration_area,
            total_squared_displacement,
        )

    @staticmethod
    def _all_contained(
        support: Polygon | MultiPolygon,
        base_aabbs: np.ndarray,
        offsets: np.ndarray,
    ) -> bool:
        bounds = (
            np.stack((base_aabbs.min(axis=1), base_aabbs.max(axis=1)), axis=1)
            + offsets[:, None, :]
        )
        region_bounds = np.asarray(support.bounds, dtype=float).reshape(2, 2)
        if bool(
            aabb_containment_violation(
                torch.from_numpy(bounds), torch.from_numpy(region_bounds)
            ).any()
        ):
            return False
        return all(
            support.covers(
                affinity.translate(
                    GroupSupportClamp._aabb_polygon(corners),
                    xoff=float(offset[0]),
                    yoff=float(offset[1]),
                )
            )
            for corners, offset in zip(base_aabbs, offsets)
        )
