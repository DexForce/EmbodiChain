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

"""Numerical rigid-group placement on polygonal support regions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from scipy.ndimage import binary_erosion
from shapely import affinity, intersects_xy
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from embodichain.utils import configclass
from embodichain.utils.logger import log_info, log_warning

SupportGeometry: TypeAlias = Polygon | MultiPolygon
__all__ = [
    "SupportGeometry",
    "GroupSupportClampConfig",
    "GroupSupportClamp",
    "group_height_offset",
]


@configclass
class GroupSupportClampConfig:
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


class GroupSupportClamp:
    """Search shared translations using raster proposals and exact containment."""

    def __init__(self, *, config: GroupSupportClampConfig | None = None) -> None:
        """Set margin and raster resolution for support placement.

        Args:
            config: Search parameters.
        """
        self.config = config if config is not None else GroupSupportClampConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Reject mutated invalid configuration before any geometric operation."""
        self.config.validate()
        if (
            not np.isfinite([self.config.margin_m, self.config.grid_resolution_m]).all()
            or self.config.margin_m < 0
            or self.config.grid_resolution_m <= 0
        ):
            raise ValueError(
                "margin_m must be finite and non-negative; grid_resolution_m must be finite and positive."
            )

    def find_translation(
        self, support_region: SupportGeometry, aabbs_by_id: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Find a shared XY translation preserving all relative positions.

        Args:
            support_region: Polygonal support, including holes/disconnected pieces.
            aabbs_by_id: Four XY corners per object in the same frame.

        Returns:
            A shared XY translation, or ``None`` when the bounded raster search
            found no feasible placement. This does not prove infeasibility.
        """
        self._validate_config()
        aabbs = self._validate_aabbs(aabbs_by_id)
        support = self._coerce_support_geometry(support_region)
        if support is None:
            raise ValueError("Group placement requires a valid support region.")
        safe = (
            support
            if self.config.margin_m == 0.0
            else self._polygonal_geometry(support.buffer(-self.config.margin_m))
        )
        if safe is None or safe.is_empty:
            return None
        return self._find_clamp_delta(safe_support=safe, aabbs_by_id=aabbs)

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
        # A group center can lie in a hole or between disconnected supports.
        # Anchor candidates to a real member footprint instead; translating its
        # center generates the same shared offsets without an invalid constraint
        # that the empty space between members must itself be supported.
        anchor_id = min(
            feasible_maps_by_id,
            key=lambda uid: int(feasible_maps_by_id[uid].sum()),
        )
        anchor_xy = centres_by_id[anchor_id]
        candidate_pixels = np.argwhere(feasible_maps_by_id[anchor_id])
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
        return intersects_xy(
            Polygon(np.asarray(coordinates)), points[:, 0], points[:, 1]
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
            if np.any(np.ptp(corner_array, axis=0) <= 0.0):
                raise ValueError(f"Asset {asset_id!r} must have positive XY extents.")
            validated[asset_id] = corner_array.copy()
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
                    GroupSupportClamp._aabb_polygon(corners),
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

    @staticmethod
    def _polygon_components(geometry: Polygon | MultiPolygon) -> list[Polygon]:
        return [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)


def group_height_offset(
    support_top_z: float, bounds: np.ndarray, *, clearance_m: float = 0.02
) -> float:
    """Compute one vertical translation for an entire group of Z-up AABBs.

    Args:
        support_top_z: Desired support height in metres.
        bounds: Object minima/maxima, shape ``(N, 2, 3)``.
        clearance_m: Non-negative gap above support.

    Returns:
        Shared translation along Z, preserving relative object poses.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.ndim != 3
        or bounds.shape[1:] != (2, 3)
        or len(bounds) == 0
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError(
            "bounds must contain finite ordered AABBs with shape (N, 2, 3)."
        )
    if not np.isfinite([support_top_z, clearance_m]).all() or clearance_m < 0:
        raise ValueError(
            "Support height must be finite and clearance finite and non-negative."
        )
    return float(support_top_z + clearance_m - bounds[:, 0, 2].min())
