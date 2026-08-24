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


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon

from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp import (
    AssetsGroupSupportClamp,
    SupportGeometry,
)
from embodichain.utils.logger import log_info, log_warning


@dataclass(frozen=True)
class AssetsSupportLayoutOptimizerConfig:
    """Controls for support-constrained pairwise AABB separation."""

    margin_m: float = 0.0  # Required clearance between each AABB and the boundary.
    aabb_clearance_m: float = 1e-6  # Required clearance between AABB pairs.
    max_rounds: int = 64  # Maximum greedy pair-separation passes.
    split_samples: int = 9  # Candidate splits between the two overlapping AABBs.


class AssetsSupportLayoutOptimizer:
    """Greedily separate AABBs while retaining arbitrary support containment.

    This reuses the previous packing algorithm's pairwise strategy: detect an
    overlap, try the two separating directions on both XY axes, and choose the
    lowest-displacement valid push.  Unlike the old path, every candidate is
    validated against the actual Polygon/MultiPolygon support region instead
    of a largest internal rectangle.
    """

    def __init__(
        self,
        *,
        support_region: SupportGeometry,
        assets_aabb_2d_z_up_world_corners_by_id: dict[str, np.ndarray],
        assets_layout: list[dict[str, object]],
        debug_output_root: str | Path | None = None,
        config: AssetsSupportLayoutOptimizerConfig | None = None,
    ) -> None:
        self.support_region = support_region
        self.assets_aabb_2d_z_up_world_corners_by_id = (
            assets_aabb_2d_z_up_world_corners_by_id
        )
        self.assets_layout = assets_layout
        self.debug_output_root = (
            Path(debug_output_root).expanduser().resolve()
            if debug_output_root is not None
            else None
        )
        self.refined_assets_layout: list[dict[str, object]] | None = None
        self.config = (
            config if config is not None else AssetsSupportLayoutOptimizerConfig()
        )
        # Check config.
        if self.config.margin_m < 0.0:
            raise ValueError("margin_m must be non-negative.")
        if self.config.aabb_clearance_m < 0.0:
            raise ValueError("aabb_clearance_m must be non-negative.")
        if self.config.max_rounds <= 0 or self.config.split_samples < 2:
            raise ValueError(
                "max_rounds must be positive and split_samples at least two."
            )

    def optimize(self) -> list[dict[str, object]]:
        """Resolve pairwise AABB overlap and return updated y-up layouts."""
        self.refined_assets_layout = None
        # Check inputs just like the previous AssetsGroupSupportClamp step would have done.
        aabbs_by_id = AssetsGroupSupportClamp._validate_aabbs(
            self.assets_aabb_2d_z_up_world_corners_by_id
        )
        raw_support = AssetsGroupSupportClamp._coerce_support_geometry(
            self.support_region
        )
        if raw_support is None:
            log_warning("AABB overlap optimization failed: invalid support geometry.")
            raise ValueError("Support region is invalid.")
        safe_support = (
            raw_support
            if self.config.margin_m == 0.0
            else AssetsGroupSupportClamp._polygonal_geometry(
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
            self.refined_assets_layout = self._apply_offsets_to_y_up_layouts(
                asset_ids=asset_ids,
                offsets=offsets,
            )
            return self.refined_assets_layout

        for round_index in range(self.config.max_rounds):
            # Check whether any overlaps remain.
            overlaps = self._overlaps(base_aabbs, offsets)
            if not overlaps:
                log_info(
                    "AABB overlap optimization succeeded after "
                    f"{round_index} rounds."
                )
                self.refined_assets_layout = self._apply_offsets_to_y_up_layouts(
                    asset_ids=asset_ids,
                    offsets=offsets,
                )
                return self.refined_assets_layout
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
            "Asset AABB overlap cannot be resolved while keeping all assets "
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
        if np.any(valid_center_minimums > valid_center_maximums + 1e-9):
            raise ValueError(
                "An asset AABB is larger than the rectangular support region."
            )

        centers = (aabb_minimums + aabb_maximums) / 2.0
        # A center must stay inset from each boundary by its AABB half extent.
        projected_centers = np.clip(
            centers, valid_center_minimums, valid_center_maximums
        )
        return projected_centers - centers

    def _apply_offsets_to_y_up_layouts(
        self, *, asset_ids: list[str], offsets: np.ndarray
    ) -> list[dict[str, object]]:
        """Write independent z-up XY offsets back to the stored y-up layouts."""
        received_ids = {str(layout.get("id")) for layout in self.assets_layout}
        expected_ids = set(asset_ids)
        if received_ids != expected_ids:
            raise ValueError(
                "Asset layouts and optimized AABBs must have identical ids."
            )
        updated_layouts: list[dict[str, object]] = []
        offsets_by_id = {
            asset_id: offsets[index] for index, asset_id in enumerate(asset_ids)
        }
        for layout in self.assets_layout:
            asset_id = str(layout["id"])
            position = layout.get("pos")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(
                    "Each asset layout must contain a three-value pos list."
                )
            dx, dy = offsets_by_id[asset_id]
            updated_layout = dict(layout)
            updated_position = [float(value) for value in position]
            updated_position[0] += float(dx)
            updated_position[2] -= float(dy)
            updated_layout["pos"] = updated_position
            updated_layouts.append(updated_layout)
        return updated_layouts

    def save_overlap_optimization_debug_images(self) -> bool:
        """Optionally save diagnostics for the most recent optimization."""
        if self.refined_assets_layout is None:
            self.optimize()
        assert self.refined_assets_layout is not None
        if self.debug_output_root is None:
            raise ValueError(
                "A debug_output_root is required when saving overlap-optimization "
                "debug images."
            )

        initial_aabbs_by_id = AssetsGroupSupportClamp._validate_aabbs(
            self.assets_aabb_2d_z_up_world_corners_by_id
        )
        raw_support = AssetsGroupSupportClamp._coerce_support_geometry(
            self.support_region
        )
        if raw_support is None:
            raise ValueError("Cannot render overlap optimization for invalid support.")
        original_positions_by_id = {
            str(layout["id"]): layout["pos"] for layout in self.assets_layout
        }
        refined_positions_by_id = {
            str(layout["id"]): layout["pos"] for layout in self.refined_assets_layout
        }
        if set(original_positions_by_id) != set(initial_aabbs_by_id) or set(
            refined_positions_by_id
        ) != set(initial_aabbs_by_id):
            raise ValueError(
                "Asset layouts and optimized AABBs must have identical ids."
            )

        translated_aabbs_by_id: dict[str, np.ndarray] = {}
        moved = False
        for asset_id, corners in initial_aabbs_by_id.items():
            original_position = original_positions_by_id[asset_id]
            refined_position = refined_positions_by_id[asset_id]
            if (
                not isinstance(original_position, list)
                or not isinstance(refined_position, list)
                or len(original_position) != 3
                or len(refined_position) != 3
            ):
                raise ValueError(
                    "Each asset layout must contain a three-value pos list."
                )
            delta_xy = np.array(
                [
                    float(refined_position[0]) - float(original_position[0]),
                    float(original_position[2]) - float(refined_position[2]),
                ]
            )
            translated_aabbs_by_id[asset_id] = corners + delta_xy
            moved = moved or not np.allclose(delta_xy, 0.0)

        path = self.debug_output_root / "assets_aabb_overlap_optimization_2d.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            1, 2, figsize=(14, 7), dpi=160, constrained_layout=True
        )
        self._draw_overlap_state(
            axes[0],
            raw_support,
            initial_aabbs_by_id,
            "Before AABB overlap optimization",
        )
        self._draw_overlap_state(
            axes[1],
            raw_support,
            translated_aabbs_by_id,
            (
                "After AABB overlap optimization"
                if moved
                else "After AABB overlap optimization (already non-overlapping)"
            ),
        )
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return True

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
        current_aabbs = base_aabbs + offsets[:, None, :]
        minimums, maximums = current_aabbs.min(axis=1), current_aabbs.max(axis=1)
        pairs = (
            [only_pair]
            if only_pair is not None
            else [
                (first_index, second_index)
                for first_index in range(len(current_aabbs))
                for second_index in range(first_index + 1, len(current_aabbs))
            ]
        )
        overlaps: list[tuple[float, float, int, int]] = []
        for first_index, second_index in pairs:
            overlap_x = (
                min(maximums[first_index, 0], maximums[second_index, 0])
                - max(minimums[first_index, 0], minimums[second_index, 0])
                + self.config.aabb_clearance_m
            )
            overlap_y = (
                min(maximums[first_index, 1], maximums[second_index, 1])
                - max(minimums[first_index, 1], minimums[second_index, 1])
                + self.config.aabb_clearance_m
            )
            if overlap_x > 1e-9 and overlap_y > 1e-9:
                overlaps.append((overlap_x, overlap_y, first_index, second_index))
        return overlaps

    def _draw_overlap_state(
        self,
        axis: plt.Axes,
        support: Polygon | MultiPolygon,
        aabbs_by_id: dict[str, np.ndarray],
        title: str,
    ) -> None:
        """Draw a state and highlight every AABB pair that still overlaps."""
        validated_aabbs = AssetsGroupSupportClamp._validate_aabbs(aabbs_by_id)
        asset_ids = sorted(validated_aabbs)
        aabbs = np.stack([validated_aabbs[asset_id] for asset_id in asset_ids])
        zero_offsets = np.zeros((len(asset_ids), 2), dtype=float)
        overlaps = self._overlaps(aabbs, zero_offsets)
        overlapping_indices = {
            index
            for _, first_index, second_index in overlaps
            for index in (first_index, second_index)
        }

        AssetsGroupSupportClamp._draw_support(axis, support, title, "darkorange")
        for index, asset_id in enumerate(asset_ids):
            corners = validated_aabbs[asset_id]
            polygon = AssetsGroupSupportClamp._aabb_polygon(corners)
            boundary = np.asarray(polygon.exterior.coords)
            is_overlapping = index in overlapping_indices
            color = "firebrick" if is_overlapping else "seagreen"
            axis.fill(
                boundary[:, 0],
                boundary[:, 1],
                facecolor=color,
                edgecolor=color,
                linewidth=2.0 if is_overlapping else 1.0,
                alpha=0.32,
            )
            axis.text(
                *corners.mean(axis=0),
                asset_id,
                ha="center",
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

        for _, first_index, second_index in overlaps:
            first_center = aabbs[first_index].mean(axis=0)
            second_center = aabbs[second_index].mean(axis=0)
            axis.plot(
                [first_center[0], second_center[0]],
                [first_center[1], second_center[1]],
                color="firebrick",
                linestyle="--",
                linewidth=1.3,
            )
        axis.set_title(f"{title}\nremaining AABB overlaps: {len(overlaps)}")

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
        return all(
            support.covers(
                affinity.translate(
                    AssetsGroupSupportClamp._aabb_polygon(corners),
                    xoff=float(offset[0]),
                    yoff=float(offset[1]),
                )
            )
            for corners, offset in zip(base_aabbs, offsets)
        )
