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
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from embodichain.utils.logger import log_info, log_warning

SupportGeometry: TypeAlias = Polygon | MultiPolygon


@dataclass(frozen=True)
class AssetsGroupSupportClampConfig:
    """Numerical controls for rigid group placement on a support region."""

    margin_m: float = 0.0  # Required clearance between each AABB and the boundary.
    grid_resolution_m: float = 0.005  # Raster cell size for the coarse search.


@dataclass(frozen=True)
class _GridTransform:
    """World/grid conversion for a centre-sampled regular XY raster."""

    x_coordinates: np.ndarray  # X coordinate of each grid-column centre.
    y_coordinates: np.ndarray  # Y coordinate of each grid-row centre.
    resolution_m: float  # Uniform spacing between neighbouring cell centres.

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y_coordinates), len(self.x_coordinates)

    def world_to_nearest_pixel(self, point_xy: np.ndarray) -> tuple[int, int]:
        column = int(np.rint((point_xy[0] - self.x_coordinates[0]) / self.resolution_m))
        row = int(np.rint((point_xy[1] - self.y_coordinates[0]) / self.resolution_m))
        return row, column

    def pixel_to_world(self, row: int, column: int) -> np.ndarray:
        return np.array([self.x_coordinates[column], self.y_coordinates[row]])


class AssetsGroupSupportClamp:
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
        # Check.
        if self.config.margin_m < 0.0:
            raise ValueError("margin_m must be non-negative.")
        if self.config.grid_resolution_m <= 0.0:
            raise ValueError("grid_resolution_m must be positive.")

    def clamp(self) -> list[dict[str, object]]:
        """Return y-up layouts after one rigid, support-valid XY translation."""
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

    def _find_clamp_delta(
        self,
        *,
        safe_support: Polygon | MultiPolygon,
        aabbs_by_id: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """Find one common z-up XY translation, or return ``None``.

        The initial position is always checked exactly first.  Otherwise, grid
        candidates are considered in increasing translation distance and the
        first vector-valid placement is returned.
        """

        # Get all aabbs's clutter center as anchor, to keep the internal
        # relative layouts unchanged.
        anchor_xy = self._group_anchor(aabbs_by_id)
        log_info(
            "Asset-group support clamp started: "
            f"assets={len(aabbs_by_id)}, support_area={safe_support.area:.4f} m^2, "
            f"boundary_margin={self.config.margin_m:.4f} m, "
            f"grid_resolution={self.config.grid_resolution_m:.4f} m."
        )

        # Check whether the initial position is already valid, which is a common case.
        zero_translation = np.zeros(2, dtype=float)
        if self._is_exactly_contained(safe_support, aabbs_by_id, zero_translation):
            log_info(
                "Asset-group support clamp succeeded without movement: all AABBs "
                "are exactly contained by the safe support region."
            )
            return zero_translation

        # Rasterize the support region and compute feasible-centre maps for each AABB.s
        transform, support_mask = self._rasterize_support(safe_support)

        # Compute feasible-centre maps with cacheing to avoid repeated binary erosion
        # for identical AABB half-extents.
        feasible_maps_by_id, centres_by_id = self._feasible_maps(
            support_mask=support_mask,
            transform=transform,
            aabbs_by_id=aabbs_by_id,
        )
        candidate_pixels = np.argwhere(support_mask)
        if len(candidate_pixels) == 0:
            return None

        candidate_world = np.asarray(
            [
                transform.pixel_to_world(int(row), int(column))
                for row, column in candidate_pixels
            ]
        )
        candidate_deltas = candidate_world - anchor_xy
        candidate_order = np.argsort(
            np.einsum("ij,ij->i", candidate_deltas, candidate_deltas), kind="stable"
        )
        for candidate_rank, candidate_index in enumerate(candidate_order, start=1):
            delta_xy = candidate_deltas[candidate_index]
            if not self._grid_translation_is_feasible(
                delta_xy=delta_xy,
                transform=transform,
                feasible_maps_by_id=feasible_maps_by_id,
                centres_by_id=centres_by_id,
            ):
                continue
            if self._is_exactly_contained(safe_support, aabbs_by_id, delta_xy):
                log_info(
                    "Asset-group support clamp succeeded after evaluating "
                    f"{candidate_rank}/{len(candidate_order)} grid candidates: "
                    f"delta_xy=({delta_xy[0]:+.4f}, {delta_xy[1]:+.4f}) m."
                )
                return delta_xy
        return None

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

    def _feasible_maps(
        self,
        *,
        support_mask: np.ndarray,
        transform: _GridTransform,
        aabbs_by_id: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute each AABB's rasterized feasible-centre map and current XY centre."""
        cached_feasible_maps: dict[tuple[int, int], np.ndarray] = {}
        feasible_maps_by_id: dict[str, np.ndarray] = {}
        centres_by_id: dict[str, np.ndarray] = {}
        for asset_id, corners in aabbs_by_id.items():
            minimum = corners.min(axis=0)
            maximum = corners.max(axis=0)
            half_extent = (maximum - minimum) / 2.0
            kernel_key = tuple(
                np.ceil(half_extent / transform.resolution_m).astype(int)
            )
            if kernel_key not in cached_feasible_maps:
                cached_feasible_maps[kernel_key] = binary_erosion(
                    support_mask,
                    structure=self._footprint_kernel(kernel_key),
                    border_value=0,
                )
            feasible_maps_by_id[asset_id] = cached_feasible_maps[kernel_key]
            centres_by_id[asset_id] = (minimum + maximum) / 2.0
        return feasible_maps_by_id, centres_by_id

    @staticmethod
    def _footprint_kernel(kernel_key: tuple[int, int]) -> np.ndarray:
        half_width_pixels, half_height_pixels = kernel_key
        return np.ones(
            (2 * half_height_pixels + 1, 2 * half_width_pixels + 1), dtype=bool
        )

    @staticmethod
    def _grid_translation_is_feasible(
        *,
        delta_xy: np.ndarray,
        transform: _GridTransform,
        feasible_maps_by_id: dict[str, np.ndarray],
        centres_by_id: dict[str, np.ndarray],
    ) -> bool:
        height, width = transform.shape
        # Sorting by feasible map population is a cheap early-rejection order:
        # small legal regions are most likely to reject a candidate quickly.
        ordered_assets = sorted(
            centres_by_id.items(),
            key=lambda item: int(feasible_maps_by_id[item[0]].sum()),
        )
        for asset_id, centre_xy in ordered_assets:
            row, column = transform.world_to_nearest_pixel(centre_xy + delta_xy)
            if row < 0 or row >= height or column < 0 or column >= width:
                return False
            if not feasible_maps_by_id[asset_id][row, column]:
                return False
        return True

    def _rasterize_support(
        self, support: Polygon | MultiPolygon
    ) -> tuple[_GridTransform, np.ndarray]:
        """Rasterize a support region into a boolean XY grid and return the transform."""
        minimum_x, minimum_y, maximum_x, maximum_y = support.bounds
        resolution = self.config.grid_resolution_m
        x_coordinates = np.arange(
            np.floor(minimum_x / resolution) * resolution,
            np.ceil(maximum_x / resolution) * resolution + resolution / 2.0,
            resolution,
        )
        y_coordinates = np.arange(
            np.floor(minimum_y / resolution) * resolution,
            np.ceil(maximum_y / resolution) * resolution + resolution / 2.0,
            resolution,
        )
        x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        mask = np.zeros(len(points), dtype=bool)
        for polygon in self._polygon_components(support):
            component_mask = self._points_in_polygon(points, polygon.exterior.coords)
            for hole in polygon.interiors:
                component_mask &= ~self._points_in_polygon(points, hole.coords)
            mask |= component_mask
        return (
            _GridTransform(
                x_coordinates, y_coordinates, resolution
            ),  # Keeps the transform for later world/grid conversions
            mask.reshape(
                x_grid.shape
            ),  # Keeps the boolean mask of the support region in grid form
        )

    @staticmethod
    def _points_in_polygon(points: np.ndarray, coordinates: object) -> np.ndarray:
        from matplotlib.path import Path as MatplotlibPath

        return MatplotlibPath(np.asarray(coordinates)).contains_points(
            points, radius=1e-12
        )

    @staticmethod
    def _validate_aabbs(
        aabbs_by_id: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Validate the input AABB(s)."""
        if not aabbs_by_id:
            raise ValueError("At least one asset 2D AABB is required.")
        validated: dict[str, np.ndarray] = {}
        for asset_id, corners in aabbs_by_id.items():
            if not isinstance(asset_id, str) or not asset_id:
                raise ValueError("Each asset AABB id must be a non-empty string.")
            corner_array = np.asarray(corners, dtype=float)
            if corner_array.shape != (4, 2) or not np.isfinite(corner_array).all():
                raise ValueError(
                    f"Asset {asset_id!r} must have four finite XY corners."
                )
            validated[asset_id] = corner_array
        return validated

    @classmethod
    def _coerce_support_geometry(
        cls, support_region: SupportGeometry
    ) -> Polygon | MultiPolygon | None:
        # Check the input type.
        if isinstance(support_region, (Polygon, MultiPolygon)):
            geometry = support_region
        else:
            log_warning(
                "Unsupported support region type: " f"{type(support_region).__name__}."
            )
            return None
        return cls._polygonal_geometry(geometry)

    @staticmethod
    def _polygonal_geometry(geometry: object) -> Polygon | MultiPolygon | None:
        if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty:
            return None
        repaired = geometry if geometry.is_valid else geometry.buffer(0)
        if not repaired.is_valid:
            log_warning("Support polygon repair did not produce a valid geometry.")
            return None
        if isinstance(repaired, (Polygon, MultiPolygon)):
            return repaired
        if isinstance(repaired, GeometryCollection):
            polygons = [item for item in repaired.geoms if isinstance(item, Polygon)]
            return MultiPolygon(polygons) if polygons else None
        return None

    @staticmethod
    def _group_anchor(aabbs_by_id: dict[str, np.ndarray]) -> np.ndarray:
        all_corners = np.concatenate(list(aabbs_by_id.values()), axis=0)
        return (all_corners.min(axis=0) + all_corners.max(axis=0)) / 2.0

    @staticmethod
    def _is_exactly_contained(
        support: Polygon | MultiPolygon,
        aabbs_by_id: dict[str, np.ndarray],
        delta_xy: np.ndarray,
    ) -> bool:
        return all(
            support.covers(
                affinity.translate(
                    AssetsGroupSupportClamp._aabb_polygon(corners),
                    xoff=float(delta_xy[0]),
                    yoff=float(delta_xy[1]),
                )
            )
            for corners in aabbs_by_id.values()
        )

    @staticmethod
    def _aabb_polygon(corners: np.ndarray) -> Polygon:
        """Build a non-self-intersecting footprint regardless of corner order."""
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        return Polygon(
            [
                (minimum[0], minimum[1]),
                (maximum[0], minimum[1]),
                (maximum[0], maximum[1]),
                (minimum[0], maximum[1]),
            ]
        )

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
    def _polygon_components(geometry: Polygon | MultiPolygon) -> list[Polygon]:
        return [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)

    @staticmethod
    def _resolve_png_output_path(output_path: str | Path) -> Path:
        path = Path(output_path).expanduser().resolve()
        return path if path.suffix.lower() == ".png" else path.with_suffix(".png")
