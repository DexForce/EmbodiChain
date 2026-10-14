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

"""CPU support-surface selection, connectivity and polygon construction."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from embodichain.compute.geometry import triangle_geometry
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh
from embodichain.utils import configclass
from embodichain.utils.logger import log_info, log_warning

__all__ = ["SupportSurfaceConfig", "DetectedSupportSurface", "SupportSurfaceDetector"]


@configclass
class SupportSurfaceConfig:
    """Parameters for conservative single-level table-top detection."""

    normal_z_min: float = 0.95  # Minimum z component for an upward face normal.
    min_surface_area_m2: float = 0.01  # Minimum projected area for a candidate level.
    max_face_height_span_m: float = (
        0.01  # Maximum within-face z variation for flatness.
    )
    height_level_tolerance_m: float = 0.005  # Maximum z difference within one level.


@dataclass(frozen=True)
class DetectedSupportSurface:
    """Detected main table support surface in z-up world coordinates."""

    top_z: float  # Highest z value among the selected support-surface triangles.
    vertices: np.ndarray  # Full z-up table vertex array referenced by ``faces``.
    faces: np.ndarray  # Indices of triangles selected as the main support surface.
    support_polygon: Polygon  # Largest valid outer support contour in z-up XY.
    optimization_rectangle: Polygon  # Axis-aligned rectangle fully inside the contour.


class SupportSurfaceDetector:
    """Detect the largest main horizontal outer contour from Z-up geometry."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        config: SupportSurfaceConfig | None = None,
    ) -> None:
        """Prepare support detection from Z-up vertex and triangle arrays.

        Args:
            vertices: Finite vertex positions in metres, shape ``(N, 3)``.
            faces: Triangle vertex indices, shape ``(M, 3)``.
            config: Surface selection thresholds.
        """
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("vertices must be finite with shape (N, 3).")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
        ):
            raise ValueError(
                "faces must be integer triangle indices with shape (M, 3)."
            )
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("faces contain invalid vertex indices.")
        self.mesh = trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )
        self.config = config if config is not None else SupportSurfaceConfig()
        self.support_region: DetectedSupportSurface | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate mutable thresholds before each detection pass."""
        self.config.validate()
        if not np.isfinite(
            [
                self.config.normal_z_min,
                self.config.min_surface_area_m2,
                self.config.max_face_height_span_m,
                self.config.height_level_tolerance_m,
            ]
        ).all():
            raise ValueError("Support detection thresholds must be finite.")
        if not 0.0 < self.config.normal_z_min <= 1.0:
            raise ValueError("normal_z_min must be in (0, 1].")
        if (
            min(
                self.config.min_surface_area_m2,
                self.config.max_face_height_span_m,
                self.config.height_level_tolerance_m,
            )
            <= 0.0
        ):
            raise ValueError("Area, height span and height tolerance must be positive.")

    def detect(self) -> DetectedSupportSurface:
        """Detect the main upward-facing support surface of a z-up table mesh."""
        self._validate_config()
        table_world_mesh = self.mesh

        # Check.
        if len(table_world_mesh.vertices) < 3 or len(table_world_mesh.faces) == 0:
            raise ValueError("Table mesh must contain at least one triangle.")

        mesh = table_world_mesh.copy()
        # Material or UV seams can duplicate vertices along one physical table top.
        mesh.merge_vertices(digits_vertex=7)

        # Repair normals if possible.
        self._repair_normals_if_possible(mesh)

        face_vertices = mesh.vertices[mesh.faces]
        face_height_ranges = np.ptp(face_vertices[:, :, 2], axis=1)
        face_normals, _, valid_faces = triangle_geometry(
            torch.from_numpy(face_vertices)
        )
        # Check: 1. range of z values of each triangle; 2. upward-facing triangles.
        candidate_face_indices = np.flatnonzero(
            valid_faces.numpy()
            & (face_normals[:, 2].numpy() >= self.config.normal_z_min)
            & (face_height_ranges <= self.config.max_face_height_span_m)
        )
        if len(candidate_face_indices) == 0:
            raise ValueError(
                "Table mesh has no near-horizontal upward-facing triangles that "
                "satisfy max_face_height_span_m."
            )

        # Select the best support level among the candidate triangles.
        selected_faces = self._select_main_support_level(
            mesh=mesh,
            candidate_face_indices=candidate_face_indices,
            full_table_hull_area=self._convex_hull_area(
                mesh.vertices[:, :2], name="Table"
            ),
        )
        selected_vertices = face_vertices[selected_faces]
        vertices = mesh.vertices.copy()
        faces = mesh.faces[selected_faces].copy()
        support_polygon = self._extract_largest_support_polygon(vertices[faces, :2])
        self.support_region = DetectedSupportSurface(
            top_z=float(selected_vertices[:, :, 2].max()),
            vertices=vertices,
            faces=faces,
            support_polygon=support_polygon,
            optimization_rectangle=self._largest_inscribed_rectangle(support_polygon),
        )
        return self.support_region

    @staticmethod
    def _repair_normals_if_possible(mesh: trimesh.Trimesh) -> None:
        if not mesh.is_watertight:
            log_warning(
                "Table mesh is not watertight; skipped outward-normal repair and "
                "will use its input face normals for support detection."
            )
            return
        if mesh.is_volume:
            log_info("Table mesh normals already form a valid outward-facing volume.")
            return
        try:
            trimesh.repair.fix_normals(mesh, multibody=True)
        except Exception as exc:
            log_warning(
                "Table mesh normal repair raised "
                f"{exc}; support detection will use the resulting face normals."
            )
            return
        if mesh.is_volume:
            log_info("Table mesh normals were repaired into an outward-facing volume.")
            return
        log_warning(
            "Table mesh is watertight but normals could not be repaired into a "
            "valid outward-facing volume; support detection will use the resulting "
            "face normals."
        )
        return

    def _select_main_support_level(
        self,
        *,
        mesh: trimesh.Trimesh,
        candidate_face_indices: np.ndarray,
        full_table_hull_area: float,
    ) -> np.ndarray:
        # Find adj.
        adjacency = self._face_adjacency(mesh)
        # Use BFS to group the connect components.
        components = self._connected_components(
            set(int(index) for index in candidate_face_indices), adjacency
        )

        # Sort components by their top z value, descending.
        components_by_height = sorted(
            (
                (
                    float(mesh.vertices[mesh.faces[list(component)], 2].max()),
                    component,
                )
                for component in components
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        # Group components into levels by their top z value, within the height_level_tolerance_m.
        levels: list[tuple[float, list[set[int]]]] = []
        for component_top_z, component in components_by_height:
            for level_index, (level_top_z, level_components) in enumerate(levels):
                if (
                    level_top_z - component_top_z
                    <= self.config.height_level_tolerance_m
                ):
                    level_components.append(component)
                    levels[level_index] = (level_top_z, level_components)
                    break
            else:
                levels.append((component_top_z, [component]))

        best_level_faces: np.ndarray | None = None
        best_hull_gap = np.inf
        best_top_z = -np.inf
        best_projected_support_area = 0.0
        hull_gap_tolerance = max(full_table_hull_area * 1e-6, 1e-9)
        for level_top_z, level_components in levels:
            level_indices = np.asarray(
                sorted(
                    face_index
                    for component in level_components
                    for face_index in component
                ),
                dtype=int,
            )
            level_triangles = mesh.vertices[mesh.faces[level_indices]]
            level_projected_support_area = self._projected_triangle_area(
                level_triangles[:, :, :2]
            )
            if level_projected_support_area < self.config.min_surface_area_m2:
                continue
            level_hull_area = self._convex_hull_area(
                level_triangles[:, :, :2].reshape(-1, 2),
                name="Candidate support level",
            )
            # Compute the gap between convex hull and triangle projection area,
            # for selecting the best level among multiple candidates which avoids
            # small area which have the largest z value.
            level_hull_gap = max(0.0, full_table_hull_area - level_hull_area)
            if (
                best_level_faces is None
                or level_hull_gap < best_hull_gap - hull_gap_tolerance
                or (
                    abs(level_hull_gap - best_hull_gap) <= hull_gap_tolerance
                    and level_top_z > best_top_z + self.config.height_level_tolerance_m
                )
                or (
                    abs(level_hull_gap - best_hull_gap) <= hull_gap_tolerance
                    and abs(level_top_z - best_top_z)
                    <= self.config.height_level_tolerance_m
                    and level_projected_support_area > best_projected_support_area
                )
            ):
                best_level_faces = level_indices
                best_hull_gap = level_hull_gap
                best_top_z = level_top_z
                best_projected_support_area = level_projected_support_area
        if best_level_faces is None:
            raise ValueError(
                "No upward-facing support level meets min_surface_area_m2."
            )
        return best_level_faces

    @staticmethod
    def _convex_hull_area(points: np.ndarray, *, name: str) -> float:
        """Compute the area of the convex hull of a set of XY points."""
        unique_points = np.unique(np.asarray(points, dtype=float), axis=0)
        if (
            unique_points.ndim != 2
            or unique_points.shape[1] != 2
            or len(unique_points) < 3
        ):
            raise ValueError(f"{name} must contain at least three unique XY points.")
        try:
            return float(ConvexHull(unique_points).volume)
        except QhullError as exc:
            raise ValueError(f"{name} XY projection is degenerate.") from exc

    @staticmethod
    def _projected_triangle_area(triangles_xy: np.ndarray) -> float:
        """Compute the total area of triangles projected onto the XY plane."""
        triangles = np.zeros((*triangles_xy.shape[:-1], 3), dtype=float)
        triangles[..., :2] = triangles_xy
        _, areas, _ = triangle_geometry(torch.from_numpy(triangles))
        return float(areas.sum())

    @classmethod
    def _extract_largest_support_polygon(cls, triangles_xy: np.ndarray) -> Polygon:
        triangles = np.zeros((*triangles_xy.shape[:-1], 3), dtype=float)
        triangles[..., :2] = triangles_xy
        _, areas, valid = triangle_geometry(torch.from_numpy(triangles))
        selected = (valid & (areas > 1e-12)).numpy()
        projected_triangles = [Polygon(triangle) for triangle in triangles_xy[selected]]
        if not projected_triangles:
            raise ValueError(
                "Selected support surface has no non-degenerate XY triangles."
            )
        merged_region = unary_union(projected_triangles)
        if merged_region.geom_type == "Polygon":
            polygons = [merged_region]
        else:
            polygons = [
                geometry
                for geometry in merged_region.geoms
                if geometry.geom_type == "Polygon"
            ]
        if not polygons:
            raise ValueError(
                "Could not create a 2D support region from the selected triangles."
            )
        if len(polygons) > 1:
            log_warning(
                "Detected multiple disconnected outer support contours; using only "
                "the largest one for the single-contour support-region output."
            )
        largest_polygon = max(polygons, key=lambda polygon: polygon.area)
        if largest_polygon.is_empty or not largest_polygon.is_valid:
            raise ValueError("The merged 2D support region is not a valid polygon.")
        boundary_xy = np.asarray(largest_polygon.exterior.coords, dtype=float)
        if len(boundary_xy) < 4 or not np.isfinite(boundary_xy).all():
            raise ValueError("The merged 2D support contour is degenerate.")
        return Polygon(boundary_xy)

    @staticmethod
    def _largest_inscribed_rectangle(polygon: Polygon) -> Polygon:
        """Find a conservative axis-aligned rectangle contained by the support contour."""
        coordinates = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        x_values = np.unique(coordinates[:, 0])
        y_values = np.unique(coordinates[:, 1])
        # Keep the search bounded for highly tessellated support contours.
        if len(x_values) > 48:
            x_values = x_values[np.linspace(0, len(x_values) - 1, 48, dtype=int)]
        if len(y_values) > 48:
            y_values = y_values[np.linspace(0, len(y_values) - 1, 48, dtype=int)]

        best_rectangle: Polygon | None = None
        best_area = 0.0
        for x_index, minimum_x in enumerate(x_values[:-1]):
            for maximum_x in x_values[x_index + 1 :]:
                if maximum_x <= minimum_x:
                    continue
                for y_index, minimum_y in enumerate(y_values[:-1]):
                    for maximum_y in y_values[y_index + 1 :]:
                        if maximum_y <= minimum_y:
                            continue
                        rectangle = Polygon(
                            [
                                (minimum_x, minimum_y),
                                (maximum_x, minimum_y),
                                (maximum_x, maximum_y),
                                (minimum_x, maximum_y),
                            ]
                        )
                        area = rectangle.area
                        if area > best_area and polygon.covers(rectangle):
                            best_rectangle = rectangle
                            best_area = area
        if best_rectangle is None:
            raise ValueError(
                "Support contour has no non-degenerate inscribed rectangle."
            )
        return best_rectangle

    @staticmethod
    def _face_adjacency(mesh: trimesh.Trimesh) -> dict[int, set[int]]:
        """Build a face adjacency dictionary for the mesh."""
        adjacency: dict[int, set[int]] = {}
        for first, second in mesh.face_adjacency:
            first_index = int(first)
            second_index = int(second)
            adjacency.setdefault(first_index, set()).add(second_index)
            adjacency.setdefault(second_index, set()).add(first_index)
        return adjacency

    @staticmethod
    def _connected_components(
        faces: set[int], adjacency: dict[int, set[int]]
    ) -> list[set[int]]:
        """Group upward-facing candidate triangles into edge-connected surface components with BFS."""
        unvisited = set(faces)
        components: list[set[int]] = []
        while unvisited:
            component: set[int] = set()
            queue = deque([unvisited.pop()])
            while queue:
                face_index = queue.popleft()
                component.add(face_index)
                for neighbor in adjacency.get(face_index, set()):
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components
