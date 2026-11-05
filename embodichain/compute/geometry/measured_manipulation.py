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

"""Numeric pad projection and occupied-floor proposals for measured manipulation."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import math
import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import polylabel

__all__ = [
    "PadProjectionStroke",
    "PreshapeProjectionFit",
    "fit_projected_preshape",
    "ProjectedFloorContents",
    "project_floor_contents",
    "OccupiedFloorProposal",
    "propose_occupied_floor_landing",
]


@dataclass(frozen=True, slots=True, eq=False)
class PadProjectionStroke:
    """Numeric mesh/pose stroke with source per-sample bounded temporary storage.

    Args:
        vertices: Ordered body-local meshes, each finite (V,3).
        tcp_poses: Ordered link-to-TCP transforms on the original stroke (S,4,4).

    Arrays own immutable byte-backed snapshots. Sampling allocates only one
    transformed point cloud, retaining the source mesh order and NumPy dtype.
    """

    vertices: tuple[np.ndarray, ...]
    tcp_poses: np.ndarray

    def __post_init__(self) -> None:
        poses = np.asarray(self.tcp_poses)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or len(poses) < 2
            or not np.isfinite(poses).all()
        ):
            raise ValueError(
                "Pad projection requires its complete finite stroke pose grid."
            )
        owned = []
        if type(self.vertices) is not tuple or not self.vertices:
            raise ValueError("Pad projection requires ordered local meshes.")
        for value in self.vertices:
            value = np.asarray(value)
            if (
                value.ndim != 2
                or value.shape[1] != 3
                or not len(value)
                or not np.isfinite(value).all()
            ):
                raise ValueError("Pad projection requires finite local vertices.")
            owned.append(
                np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            )
        object.__setattr__(self, "vertices", tuple(owned))
        object.__setattr__(
            self,
            "tcp_poses",
            np.frombuffer(poses.tobytes(), dtype=poses.dtype).reshape(poses.shape),
        )

    def __len__(self) -> int:
        return len(self.tcp_poses)

    def sample(self, index: int) -> np.ndarray:
        """Return one source-ordered point cloud in TCP coordinates."""
        pose = self.tcp_poses[index]
        return np.vstack(
            [value @ pose[:3, :3].T + pose[:3, 3] for value in self.vertices]
        )


@dataclass(frozen=True, slots=True)
class PreshapeProjectionFit:
    """Source stroke selection and full target projected width, without a grip verdict."""

    fraction: float
    target_width_m: float
    fitting_indices: tuple[int, ...]


def fit_projected_preshape(
    target_points_tcp: np.ndarray,
    negative_pad_points: np.ndarray | PadProjectionStroke,
    positive_pad_points: np.ndarray | PadProjectionStroke,
    *,
    pad_rest_offsets: tuple[float, float],
    target_rest_offset: float,
) -> PreshapeProjectionFit:
    """Select the last captured stroke index enclosing the full target projection.

    Args:
        target_points_tcp: Complete target points in the selected grasp TCP frame (V,3).
        negative_pad_points: First ordered pad points (S,N,3), or its bounded
            source mesh/pose stroke without materializing all sample vertices.
        positive_pad_points: Second pad points/stroke on the same S grid.
        pad_rest_offsets: Original ordered finite native rest offsets in metres.
        target_rest_offset: Original nonnegative target native rest offset.

    Returns:
        Source fitting index/fraction and projected target width. The unchanged
        two-millimetre side reserve is added to each pair's native rest offsets.
        Missing/degenerate data or no fitting sample rejects without a proposal.
    """

    def count(value: np.ndarray | PadProjectionStroke) -> int:
        if isinstance(value, PadProjectionStroke):
            return len(value)
        if (
            value.ndim != 3
            or value.shape[2] != 3
            or value.shape[1] < 1
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "Measured preshape requires complete finite pad stroke geometry."
            )
        return len(value)

    def sample(value: np.ndarray | PadProjectionStroke, index: int) -> np.ndarray:
        return (
            value.sample(index)
            if isinstance(value, PadProjectionStroke)
            else value[index]
        )

    negative_count, positive_count = count(negative_pad_points), count(
        positive_pad_points
    )
    if (
        target_points_tcp.ndim != 2
        or target_points_tcp.shape[1] != 3
        or not len(target_points_tcp)
        or negative_count != positive_count
        or negative_count < 2
        or not np.isfinite(target_points_tcp).all()
        or len(pad_rest_offsets) != 2
        or any(
            not math.isfinite(value) or value < 0
            for value in (*pad_rest_offsets, target_rest_offset)
        )
    ):
        raise ValueError(
            "Measured preshape requires complete finite target/pad stroke geometry."
        )
    axis = sample(positive_pad_points, 0).mean(0) - sample(negative_pad_points, 0).mean(
        0
    )
    length = np.linalg.norm(axis)
    if not math.isfinite(float(length)) or length == 0:
        raise ValueError("Measured preshape pad axis is unavailable.")
    axis /= length
    projected = target_points_tcp @ axis
    fitting = []
    for index in range(negative_count):
        negative, positive = (
            sample(negative_pad_points, index) @ axis,
            sample(positive_pad_points, index) @ axis,
        )
        gaps = (projected.min() - negative.max(), positive.min() - projected.max())
        if all(
            gap >= rest + target_rest_offset + 0.002
            for gap, rest in zip(gaps, pad_rest_offsets, strict=True)
        ):
            fitting.append(index)
    if not fitting:
        raise ValueError("No native preshape contains the complete target projection.")
    return PreshapeProjectionFit(
        fitting[-1] / (negative_count - 1),
        float(np.ptp(projected)),
        tuple(fitting),
    )


@dataclass(frozen=True, slots=True)
class OccupiedFloorProposal:
    """Parent-frame displacement and original reservation measurements."""

    displacement_parent: tuple[float, float, float] | None
    observed_contents: tuple[str, ...]
    selected_root_xy: tuple[float, float] | None = None
    floor_boundary_gap_m: float | None = None
    content_gaps_m: tuple[tuple[str, float], ...] = ()

    def metadata(self) -> dict[str, object]:
        """Return source-shaped owned proposal diagnostics, never physical evidence."""
        if self.displacement_parent is None:
            return {"observed_contents": list(self.observed_contents)}
        return dict(
            observed_contents=list(self.observed_contents),
            selected_root_xy=list(self.selected_root_xy),
            floor_boundary_gap_m=self.floor_boundary_gap_m,
            content_gaps_m=dict(self.content_gaps_m),
            required_reservation_m=0.005,
            release_height_unchanged=True,
        )


def propose_occupied_floor_landing(
    floor: Polygon,
    floor_height: float,
    target_vertices_oriented: np.ndarray,
    target_position_parent: np.ndarray,
    contents_parent: Sequence[tuple[str, np.ndarray]],
) -> OccupiedFloorProposal:
    """Preserve the source full-footprint five-millimetre occupied-floor proposal.

    Args:
        floor: Exact parent-local qualified floor region.
        floor_height: Original parent-local support height.
        target_vertices_oriented: Source subject vertices after release rotation (V,3).
        target_position_parent: Original release translation in the parent frame (3,).
        contents_parent: Fresh already-transformed object geometry, in original
            sorted identity order; no simulation or task state enters Compute.

    Returns:
        Unchanged release when no object occupies this floor, otherwise the source
        polylabel proposal and exact boundary/content gaps. No valid reserved
        region raises; height, orientation and success standards remain unchanged.
    """
    return project_floor_contents(floor, floor_height, contents_parent).propose(
        target_vertices_oriented, target_position_parent
    )


@dataclass(frozen=True, slots=True)
class ProjectedFloorContents:
    """Already-projected floor occupants; no runtime or target callback is retained."""

    floor: Polygon
    contents: tuple[tuple[str, Polygon], ...]

    def propose(
        self, target_vertices_oriented: np.ndarray, target_position_parent: np.ndarray
    ) -> OccupiedFloorProposal:
        """Select the original reserved release after required target operands exist."""
        if not self.contents:
            return OccupiedFloorProposal(None, ())
        return _propose_reserved_floor(
            self.floor, self.contents, target_vertices_oriented, target_position_parent
        )


def project_floor_contents(
    floor: Polygon,
    floor_height: float,
    contents_parent: Sequence[tuple[str, np.ndarray]],
) -> ProjectedFloorContents:
    """Project ordered observed contents before a release target is evaluated.

    Args:
        floor: Qualified original parent-local floor region.
        floor_height: Original local support height.
        contents_parent: Ordered source geometry already transformed to this frame.

    Returns:
        Exact occupancy selection, retaining the source no-contents short circuit.
        This geometric projection is not a physical release or stability verdict.
    """
    occupied = []
    for source, points in contents_parent:
        polygon = MultiPoint(points[:, :2]).convex_hull
        if floor.covers(polygon) and -0.01 <= points[:, 2].min() - floor_height <= 0.05:
            occupied.append((source, polygon))
    return ProjectedFloorContents(floor, tuple(occupied))


def _propose_reserved_floor(
    floor, occupied, target_vertices_oriented, target_position_parent
):
    margin = 0.005
    footprint = MultiPoint(target_vertices_oriented[:, :2]).convex_hull
    safe = floor.buffer(-margin)
    allowed = safe
    for x, y in footprint.exterior.coords[:-1]:
        allowed = allowed.intersection(translate(safe, xoff=-float(x), yoff=-float(y)))
    for _, polygon in occupied:
        pairs = (
            np.asarray(polygon.exterior.coords[:-1])[:, None]
            - np.asarray(footprint.exterior.coords[:-1])[None]
        )
        forbidden = MultiPoint(pairs.reshape(-1, 2)).convex_hull.buffer(
            margin / math.cos(math.pi / 64), quad_segs=16
        )
        allowed = allowed.difference(forbidden)
    regions = (
        [allowed]
        if allowed.geom_type == "Polygon"
        else [g for g in getattr(allowed, "geoms", ()) if g.geom_type == "Polygon"]
    )
    regions = [g for g in regions if not g.is_empty and g.area > 0]
    if not regions:
        raise ValueError("No measured inside landing retains both 5mm reservations.")
    choices = [polylabel(p, tolerance=0.001) for p in regions]
    point = max(choices, key=lambda p: allowed.boundary.distance(p))
    placed = translate(footprint, xoff=point.x, yoff=point.y)
    if not safe.covers(placed) or any(
        placed.distance(p) < margin - 1e-8 for _, p in occupied
    ):
        raise ValueError("Measured inside proposal failed full-footprint reservations.")
    return OccupiedFloorProposal(
        (
            float(point.x - target_position_parent[0]),
            float(point.y - target_position_parent[1]),
            0.0,
        ),
        tuple(uid for uid, _ in occupied),
        (point.x, point.y),
        floor.boundary.distance(placed),
        tuple((uid, placed.distance(p)) for uid, p in occupied),
    )
