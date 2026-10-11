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

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from embodichain.compute.geometry.aabb import (
    AABBLayoutOptimizer,
    AABBLayoutConfig as AssetsSupportLayoutOptimizerConfig,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp import (
    AssetsGroupSupportClamp,
    SupportGeometry,
)

__all__ = ["AssetsSupportLayoutOptimizerConfig", "AssetsSupportLayoutOptimizer"]


class AssetsSupportLayoutOptimizer(AABBLayoutOptimizer):
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
        super().__init__(config=self.config)

    def optimize(self) -> list[dict[str, object]]:
        """Resolve overlap through shared geometry and adapt offsets to Y-up."""
        self.refined_assets_layout = None
        offsets_by_id = self.solve(
            self.support_region, self.assets_aabb_2d_z_up_world_corners_by_id
        )
        asset_ids = sorted(offsets_by_id)
        self.refined_assets_layout = self._apply_offsets_to_y_up_layouts(
            asset_ids=asset_ids,
            offsets=np.stack([offsets_by_id[uid] for uid in asset_ids]),
        )
        return self.refined_assets_layout

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
