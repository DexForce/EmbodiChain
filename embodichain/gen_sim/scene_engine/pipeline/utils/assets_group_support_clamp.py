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

from embodichain.lab.sim.scene._placement_cpu import (
    GroupSupportClamp,
    GroupSupportClampConfig as AssetsGroupSupportClampConfig,
    SupportGeometry,
)
from embodichain.utils.logger import log_info, log_warning

__all__ = [
    "SupportGeometry",
    "AssetsGroupSupportClampConfig",
    "AssetsGroupSupportClamp",
]


class AssetsGroupSupportClamp(GroupSupportClamp):
    """Find a small shared XY shift that places all AABBs on a support region.

    Each AABB gets a feasible-centre map obtained by binary erosion of the safe
    support mask.  A candidate translation is valid only when it is feasible
    for every asset, then it must pass exact Shapely containment.  This supports
    concave polygons, holes, and disconnected ``MultiPolygon`` regions.
    """

    def __init__(
        self,
        *,
        support_region: SupportGeometry,
        assets_aabb_2d_z_up_world_corners_by_id: dict[str, np.ndarray],
        assets_layout: list[dict[str, object]],
        debug_output_root: str | Path | None = None,
        config: AssetsGroupSupportClampConfig | None = None,
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
        self.config = config if config is not None else AssetsGroupSupportClampConfig()
        super().__init__(config=self.config)

    def clamp(self) -> list[dict[str, object]]:
        """Return y-up layouts after one rigid, support-valid XY translation."""
        self._validate_config()
        self.refined_assets_layout = None
        # Validate input aabbs.
        aabbs_by_id = self._validate_aabbs(self.assets_aabb_2d_z_up_world_corners_by_id)

        # Validate and coerce the support region into a usable polygonal geometry.
        raw_support = self._coerce_support_geometry(self.support_region)
        if raw_support is None:
            log_warning("Asset-group support clamp failed: invalid support geometry.")
            raise ValueError(
                "Asset-group support clamp requires a valid support region."
            )
        # Re coerce the support region with a margin to get the safe support region.
        safe_support = (
            raw_support
            if self.config.margin_m == 0.0
            else self._polygonal_geometry(raw_support.buffer(-self.config.margin_m))
        )
        if safe_support is None or safe_support.is_empty:
            log_warning(
                "Asset-group support clamp failed: support region is empty after "
                f"applying a {self.config.margin_m:.4f} m boundary margin."
            )
            raise ValueError("Asset-group support clamp has no usable support area.")

        # Get all the translated layouts, and store them for later debug rendering.
        delta_xy = self._find_clamp_delta(
            safe_support=safe_support,
            aabbs_by_id=aabbs_by_id,
        )
        if delta_xy is None:
            log_warning(
                "Asset-group support clamp failed: no shared translation can place "
                f"all {len(aabbs_by_id)} AABBs inside the support region."
            )
            raise ValueError(
                "Asset clutter cannot be placed completely on the detected table "
                "support region."
            )
        # Translate the layouts and store them for later debug rendering.
        refined_assets_layout = self._apply_delta_to_y_up_layouts(
            delta_xy=delta_xy,
            expected_ids=set(aabbs_by_id),
        )
        self.refined_assets_layout = refined_assets_layout
        if np.allclose(
            delta_xy, 0.0
        ):  # Judge whether the translation is zero, if so, no need to apply optimization.
            log_info(
                "All asset AABBs are already fully inside the detected table "
                "support region; no planar group optimization was applied."
            )
        else:
            log_info(
                "Applied rigid asset-group support optimization with "
                f"delta_xy={delta_xy.tolist()} m; all AABBs passed "
                "exact support containment."
            )
        return refined_assets_layout

    def _apply_delta_to_y_up_layouts(
        self, *, delta_xy: np.ndarray, expected_ids: set[str]
    ) -> list[dict[str, object]]:
        """Apply a successful common z-up XY translation to stored layouts."""
        received_ids = {str(layout.get("id")) for layout in self.assets_layout}
        if received_ids != expected_ids:
            raise ValueError("Asset layouts and clamped AABBs must have identical ids.")

        dx, dy = delta_xy
        translated_layouts: list[dict[str, object]] = []
        for layout in self.assets_layout:
            position = layout.get("pos")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(
                    "Each asset layout must contain a three-value pos list."
                )
            translated_layout = dict(layout)
            translated_position = [float(value) for value in position]
            translated_position[0] += float(dx)
            translated_position[2] -= float(dy)
            translated_layout["pos"] = translated_position
            translated_layouts.append(translated_layout)
        return translated_layouts

    def save_group_clamp_debug_images(self) -> bool:
        """Optionally save diagnostics for the support-valid group translation."""
        if self.refined_assets_layout is None:
            self.clamp()
        assert self.refined_assets_layout is not None

        initial_aabbs_by_id = self._validate_aabbs(
            self.assets_aabb_2d_z_up_world_corners_by_id
        )
        raw_support = self._coerce_support_geometry(self.support_region)
        if raw_support is None:
            raise ValueError(
                "Asset-group support clamp requires a valid support region."
            )
        safe_support = (
            raw_support
            if self.config.margin_m == 0.0
            else self._polygonal_geometry(raw_support.buffer(-self.config.margin_m))
        )
        if safe_support is None or safe_support.is_empty:
            raise ValueError("Asset-group support clamp has no usable support area.")

        original_positions_by_id = {
            str(layout["id"]): layout["pos"] for layout in self.assets_layout
        }
        refined_positions_by_id = {
            str(layout["id"]): layout["pos"] for layout in self.refined_assets_layout
        }
        if set(original_positions_by_id) != set(initial_aabbs_by_id) or set(
            refined_positions_by_id
        ) != set(initial_aabbs_by_id):
            raise ValueError("Asset layouts and clamped AABBs must have identical ids.")
        first_asset_id = next(iter(initial_aabbs_by_id))
        original_position = original_positions_by_id[first_asset_id]
        refined_position = refined_positions_by_id[first_asset_id]
        if (
            not isinstance(original_position, list)
            or not isinstance(refined_position, list)
            or len(original_position) != 3
            or len(refined_position) != 3
        ):
            raise ValueError("Each asset layout must contain a three-value pos list.")
        delta_xy = np.array(
            [
                float(refined_position[0]) - float(original_position[0]),
                float(original_position[2]) - float(refined_position[2]),
            ]
        )
        translated_aabbs_by_id = {
            asset_id: corners + delta_xy
            for asset_id, corners in initial_aabbs_by_id.items()
        }
        if self.debug_output_root is None:
            raise ValueError(
                "A debug_output_root is required when saving group-clamp "
                "debug images."
            )
        self._render_support_debug(
            raw_support=raw_support,
            safe_support=safe_support,
            output_path=self.debug_output_root / "table_support_region_safe_2d.png",
        )
        self._render_clamp_debug(
            raw_support=raw_support,
            initial_aabbs_by_id=initial_aabbs_by_id,
            translated_aabbs_by_id=translated_aabbs_by_id,
            delta_xy=delta_xy,
            output_path=self.debug_output_root / "assets_group_support_clamp_2d.png",
        )
        return True

    def _render_support_debug(
        self,
        *,
        raw_support: Polygon | MultiPolygon,
        safe_support: Polygon | MultiPolygon | None,
        output_path: str | Path,
    ) -> Path:
        path = self._resolve_png_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            1, 2, figsize=(14, 7), dpi=160, constrained_layout=True
        )
        self._draw_support(axes[0], raw_support, "Input support region", "darkorange")
        title = f"Safe support (margin={self.config.margin_m:.3f} m)"
        if safe_support is None or safe_support.is_empty:
            axes[1].set_title(f"{title}\n(infeasible)")
            axes[1].set_aspect("equal", adjustable="box")
        else:
            self._draw_support(axes[1], safe_support, title, "seagreen")
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return path

    def _render_clamp_debug(
        self,
        *,
        raw_support: Polygon | MultiPolygon,
        initial_aabbs_by_id: dict[str, np.ndarray],
        translated_aabbs_by_id: dict[str, np.ndarray] | None,
        delta_xy: np.ndarray | None,
        output_path: str | Path,
    ) -> Path:
        path = self._resolve_png_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            1, 2, figsize=(14, 7), dpi=160, constrained_layout=True
        )
        self._draw_state(
            axes[0], raw_support, initial_aabbs_by_id, "Before group clamp", "royalblue"
        )
        if translated_aabbs_by_id is not None and delta_xy is not None:
            action = "no movement" if np.allclose(delta_xy, 0.0) else "translated"
            self._draw_state(
                axes[1],
                raw_support,
                translated_aabbs_by_id,
                f"After group clamp ({action})\nΔxy=({delta_xy[0]:+.3f}, {delta_xy[1]:+.3f}) m",
                "seagreen",
            )
        else:
            self._draw_state(
                axes[1],
                raw_support,
                initial_aabbs_by_id,
                "No feasible group translation",
                "firebrick",
            )
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return path

    @classmethod
    def _draw_support(
        cls, axis: plt.Axes, support: Polygon | MultiPolygon, title: str, color: str
    ) -> None:
        for polygon in cls._polygon_components(support):
            exterior = np.asarray(polygon.exterior.coords)
            axis.fill(
                exterior[:, 0],
                exterior[:, 1],
                facecolor=color,
                edgecolor="saddlebrown",
                alpha=0.35,
            )
            for hole in polygon.interiors:
                hole_points = np.asarray(hole.coords)
                axis.fill(
                    hole_points[:, 0],
                    hole_points[:, 1],
                    facecolor="white",
                    edgecolor="saddlebrown",
                    alpha=1.0,
                )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (z-up world)")
        axis.set_ylabel("y (z-up world)")
        axis.set_title(title)
        axis.autoscale_view()

    @classmethod
    def _draw_state(
        cls,
        axis: plt.Axes,
        support: Polygon | MultiPolygon,
        aabbs_by_id: dict[str, np.ndarray],
        title: str,
        color: str,
    ) -> None:
        cls._draw_support(axis, support, title, "darkorange")
        for asset_id, corners in sorted(aabbs_by_id.items()):
            polygon = cls._aabb_polygon(corners)
            boundary = np.asarray(polygon.exterior.coords)
            axis.fill(
                boundary[:, 0],
                boundary[:, 1],
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
            )
            axis.text(
                *corners.mean(axis=0),
                asset_id,
                ha="center",
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    @staticmethod
    def _resolve_png_output_path(output_path: str | Path) -> Path:
        path = Path(output_path).expanduser().resolve()
        return path if path.suffix.lower() == ".png" else path.with_suffix(".png")
