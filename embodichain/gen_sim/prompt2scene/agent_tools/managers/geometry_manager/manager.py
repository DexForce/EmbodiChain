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

"""Geometry manager for mesh I/O, transforms, and tabletop detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager.schemas import (
    AlignToAxisRequest,
    AlignToAxisResult,
    AlignXYLongAxisRequest,
    AlignXYLongAxisResult,
    CenterMeshRequest,
    CenterMeshResult,
    ConvertUpAxisRequest,
    ConvertUpAxisResult,
    DetectTabletopRequest,
    DetectTabletopResult,
    ExportMeshRequest,
    ExportMeshResult,
    LoadMeshRequest,
    LoadMeshResult,
    NormalizeRequest,
    NormalizeResult,
    PlaceAbovePlaneRequest,
    PlaceAbovePlaneResult,
    SupportPlaneCandidate,
)

__all__ = ["GeometryManager"]

DEFAULT_INPUT_UP_AXIS = [0.0, 1.0, 0.0]
DEFAULT_UP_AXIS = [0.0, 0.0, 1.0]


class GeometryManager:
    """Manager for mesh geometry operations.

    Provides typed methods for mesh I/O, axis conversion, bounding-box
    transforms, tabletop plane detection, and PCA alignment, following
    the same pattern as service clients.
    """

    @staticmethod
    def compose_json_matrices(*values: Any) -> list[list[float]]:
        from . import utils as geometry_utils

        return geometry_utils._compose_json_matrices(*values)

    @staticmethod
    def compose_simready_to_aligned_matrix(
        *, raw_to_aligned_matrix: Any, raw_to_simready_matrix: Any
    ) -> list[list[float]]:
        from . import utils as geometry_utils

        return geometry_utils._compose_simready_to_aligned_matrix(
            raw_to_aligned_matrix=raw_to_aligned_matrix,
            raw_to_simready_matrix=raw_to_simready_matrix,
        )

    @staticmethod
    def decompose_transform_matrix(matrix_value: Any) -> dict[str, Any]:
        from . import utils as geometry_utils

        return geometry_utils._decompose_transform_matrix(matrix_value)

    @staticmethod
    def support_normal_flip_transform(**kwargs: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._support_normal_flip_transform(**kwargs)

    @staticmethod
    def z_yaw_transform(yaw_degrees: float) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._z_yaw_transform(yaw_degrees)

    @staticmethod
    def z_up_to_glb_y_up_transform() -> Any:
        from . import utils as geometry_utils

        return geometry_utils._z_up_to_glb_y_up_transform()

    @staticmethod
    def copy_scene_with_transform(scene: Any, transform: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._copy_scene_with_transform(scene, transform)

    @staticmethod
    def matrix_from_json(value: Any, *, name: str) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._matrix_from_json(value, name=name)

    @staticmethod
    def load_scene_with_transform(**kwargs: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._load_scene_with_transform(**kwargs)

    @staticmethod
    def estimate_support_normal(mesh: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._estimate_support_normal(mesh)

    @staticmethod
    def rotation_between_vectors(source: Any, target: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._rotation_between_vectors(source, target)

    @staticmethod
    def transform_point(transform: Any, point: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._transform_point(transform, point)

    @staticmethod
    def aabb_center(bounds: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._aabb_center(bounds)

    @staticmethod
    def xy_aabb_center(bounds: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._xy_aabb_center(bounds)

    @staticmethod
    def xy_aabb_size(bounds: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._xy_aabb_size(bounds)

    @staticmethod
    def aabb_bottom_to_xy_plane_transform(bounds: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._aabb_bottom_to_xy_plane_transform(bounds)

    @staticmethod
    def scale_transform(scale: float) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._scale_transform(scale)

    @staticmethod
    def compose_sam3d_multi_object_transform(**kwargs: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._compose_sam3d_multi_object_transform(**kwargs)

    @staticmethod
    def detect_table_fit_support_quad(
        mesh: Any,
        *,
        target_aspect: float,
    ) -> dict[str, Any]:
        from . import utils as geometry_utils

        return geometry_utils._detect_table_fit_support_quad(
            mesh,
            target_aspect=target_aspect,
        )

    @staticmethod
    def load_table_fit_scene_internal_z(path: Path, *, trimesh: Any, y_to_z: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._load_table_fit_scene_internal_z(
            path,
            trimesh=trimesh,
            y_to_z=y_to_z,
        )

    @staticmethod
    def table_fit_scene_union_bounds(scenes: list[Any], *, trimesh: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._table_fit_scene_union_bounds(scenes, trimesh=trimesh)

    @staticmethod
    def table_fit_bounds_xy_manifest(
        bounds: Any,
        *,
        unit_scale: float,
    ) -> dict[str, Any]:
        from . import utils as geometry_utils

        return geometry_utils._table_fit_bounds_xy_manifest(
            bounds,
            unit_scale=unit_scale,
        )

    @staticmethod
    def table_fit_uniform_xy_scale_transform(**kwargs: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._table_fit_uniform_xy_scale_transform(**kwargs)

    @staticmethod
    def table_fit_uniform_scale_transform(**kwargs: Any) -> Any:
        from . import utils as geometry_utils

        return geometry_utils._table_fit_uniform_scale_transform(**kwargs)

    @staticmethod
    def table_fit_safe_positive_ratio(numerator: float, denominator: float) -> float:
        from . import utils as geometry_utils

        return geometry_utils._table_fit_safe_positive_ratio(numerator, denominator)

    @staticmethod
    def load_mesh(request: LoadMeshRequest) -> LoadMeshResult:
        """Load a GLB/mesh file as one Trimesh object."""
        mesh_path = request.mesh_path.expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        loaded = trimesh.load(mesh_path, force=None)
        if isinstance(loaded, trimesh.Scene):
            geometries = [
                g
                for g in loaded.dump(concatenate=False)
                if hasattr(g, "vertices") and hasattr(g, "faces")
            ]
            if not geometries:
                raise ValueError(f"Scene contains no mesh geometry: {mesh_path}")
            return LoadMeshResult(mesh=trimesh.util.concatenate(geometries))
        return LoadMeshResult(mesh=loaded)

    @staticmethod
    def export_mesh(request: ExportMeshRequest) -> ExportMeshResult:
        """Export a mesh and return the resolved output path."""
        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request.mesh.export(output_path)
        if not output_path.is_file():
            raise FileNotFoundError(f"Mesh was not written: {output_path}")
        return ExportMeshResult(output_path=output_path)


    @staticmethod
    def convert_up_axis(request: ConvertUpAxisRequest) -> ConvertUpAxisResult:
        """Convert a mesh from one up-axis convention to another."""
        mesh = GeometryManager._align_vector_to_axis(
            request.mesh,
            source_axis=request.input_up_axis or DEFAULT_INPUT_UP_AXIS,
            target_axis=request.output_up_axis or DEFAULT_UP_AXIS,
        )
        return ConvertUpAxisResult(mesh=mesh)

    @staticmethod
    def center_by_bbox(request: CenterMeshRequest) -> CenterMeshResult:
        """Center a mesh by its bounding box."""
        GeometryManager._validate_mesh(request.mesh)

        bounds = np.asarray(request.mesh.bounds, dtype=float)
        if bounds.shape != (2, 3):
            raise ValueError("Mesh bounds must have shape (2, 3).")

        bbox_center = (bounds[0] + bounds[1]) * 0.5
        centered = request.mesh.copy()
        centered.apply_translation(-bbox_center)
        return CenterMeshResult(
            mesh=centered,
            bbox_center=[float(v) for v in bbox_center],
        )

    @staticmethod
    def align_to_axis(request: AlignToAxisRequest) -> AlignToAxisResult:
        """Rotate a mesh so a source vector aligns to a target axis."""
        mesh = GeometryManager._align_vector_to_axis(
            request.mesh,
            source_axis=request.source_axis,
            target_axis=request.target_axis,
        )
        return AlignToAxisResult(mesh=mesh)

    @staticmethod
    def place_above_plane(
        request: PlaceAbovePlaneRequest,
    ) -> PlaceAbovePlaneResult:
        """Translate a mesh so its AABB bottom is above the XY plane."""
        if request.clearance < 0.0:
            raise ValueError("clearance must be non-negative.")

        bounds = np.asarray(request.mesh.bounds, dtype=float)
        if bounds.shape != (2, 3):
            raise ValueError("Mesh bounds must have shape (2, 3).")

        min_z = float(bounds[0][2])
        placed = request.mesh.copy()
        placed.apply_translation([0.0, 0.0, request.clearance - min_z])
        return PlaceAbovePlaneResult(mesh=placed)

    @staticmethod
    def normalize(request: NormalizeRequest) -> NormalizeResult:
        """Scale a mesh so its longest bounding-box axis equals target_size."""
        if request.target_size <= 0.0:
            raise ValueError("target_size must be positive.")

        extents = np.asarray(
            request.mesh.bounding_box_oriented.primitive.extents, dtype=float
        )
        scale_factor = request.target_size / float(np.max(extents))
        normalized = request.mesh.copy()
        normalized.apply_scale(scale_factor)
        return NormalizeResult(mesh=normalized, scale_factor=scale_factor)

    @staticmethod
    def mesh_aabb_size(mesh: Any) -> Any:
        """Return a mesh AABB size vector."""
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3):
            raise ValueError("Mesh bounds must have shape (2, 3).")
        size = bounds[1] - bounds[0]
        if np.any(size <= 0.0):
            raise ValueError(f"Mesh AABB size must be positive, got {size.tolist()}.")
        return size

    @staticmethod
    def mesh_pca_bbox_size(mesh: Any) -> Any:
        """Return bbox extents in the mesh PCA frame.

        This is used for metric-scale estimation because it is less sensitive
        to arbitrary object yaw/tilt than a world-axis AABB.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            return GeometryManager.mesh_aabb_size(mesh)

        centered = vertices - np.mean(vertices, axis=0)
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
            return GeometryManager.mesh_aabb_size(mesh)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order]
        if np.linalg.det(axes) < 0.0:
            axes[:, -1] *= -1.0

        projected = centered @ axes
        size = projected.max(axis=0) - projected.min(axis=0)
        if np.any(size <= 0.0) or not np.all(np.isfinite(size)):
            return GeometryManager.mesh_aabb_size(mesh)
        return size

    @staticmethod
    def mesh_metric_bbox_size(mesh: Any) -> Any:
        """Return the bbox size used by metric-scale estimation."""
        return GeometryManager.mesh_pca_bbox_size(mesh)

    @staticmethod
    def bbox_ratio(size: Any) -> Any:
        """Return bbox dimensions normalized by the largest axis."""
        size = np.asarray(size, dtype=np.float64)
        max_size = float(np.max(size))
        if max_size <= 0.0:
            raise ValueError("bbox size max must be positive.")
        return size / max_size

    @staticmethod
    def best_axis_bbox_scale_match(
        *,
        source_size_cm: Any,
        target_size_cm: Any,
    ) -> dict[str, Any]:
        """Match target bbox axes to source axes and return a scale candidate."""
        source = np.asarray(source_size_cm, dtype=np.float64)
        target = np.asarray(target_size_cm, dtype=np.float64)
        if source.shape != (3,) or target.shape != (3,):
            raise ValueError("source_size_cm and target_size_cm must have shape (3,).")
        if np.any(source <= 0.0) or np.any(target <= 0.0):
            raise ValueError("source_size_cm and target_size_cm must be positive.")

        source_ratio = GeometryManager.bbox_ratio(source)
        best: dict[str, Any] | None = None
        for permutation in [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]:
            target_perm = target[list(permutation)]
            target_ratio = GeometryManager.bbox_ratio(target_perm)
            ratio_error = GeometryManager._mean_abs_log_ratio_error(
                source_ratio,
                target_ratio,
            )
            per_axis_scale = target_perm / source
            candidate = {
                "target_permutation": list(permutation),
                "source_size_cm": source.tolist(),
                "target_size_cm_original_order": target.tolist(),
                "target_size_cm_matched_to_source_axes": target_perm.tolist(),
                "source_ratio": source_ratio.tolist(),
                "target_ratio_matched": target_ratio.tolist(),
                "per_axis_scale": per_axis_scale.tolist(),
                "scale_factor": float(np.median(per_axis_scale)),
                "shape_ratio_error": float(ratio_error),
            }
            if best is None or ratio_error < float(best["shape_ratio_error"]):
                best = candidate
        if best is None:
            raise ValueError("Failed to match bbox axes.")
        return best

    @staticmethod
    def scene_to_mesh(scene: Any, *, trimesh: Any | None = None) -> Any:
        """Convert a trimesh Scene or mesh-like object to one mesh."""
        trimesh_module = globals()["trimesh"]
        if trimesh is not None:
            trimesh_module = trimesh
        if isinstance(scene, trimesh_module.Trimesh):
            return scene
        dumped = scene.dump(concatenate=True)
        if isinstance(dumped, trimesh_module.Trimesh):
            return dumped
        meshes = [
            item for item in dumped if isinstance(item, trimesh_module.Trimesh)
        ]
        if not meshes:
            raise ValueError("Scene contains no mesh geometry.")
        return trimesh_module.util.concatenate(meshes)

    @staticmethod
    def detect_tabletop(
        request: DetectTabletopRequest,
    ) -> DetectTabletopResult:
        """Detect the most likely tabletop plane in a mesh."""
        candidates = GeometryManager._find_support_plane_candidates(
            request.mesh,
            normal_angle_tol_deg=request.normal_angle_tol_deg,
            plane_distance_tol=request.plane_distance_tol,
            min_area_ratio=request.min_area_ratio,
            max_candidates=request.max_candidates,
        )
        selected = GeometryManager._select_tabletop_plane(candidates)
        oriented_normal = GeometryManager._orient_plane_normal(
            request.mesh,
            plane_normal=selected.normal,
            plane_center=selected.center,
        )
        return DetectTabletopResult(
            selected=selected,
            oriented_normal=oriented_normal,
            candidates=candidates,
        )


    @staticmethod
    def align_xy_long_axis(
        request: AlignXYLongAxisRequest,
    ) -> AlignXYLongAxisResult:
        """Rotate a table so its XY-projected long axis aligns with the Y axis."""
        vertices = np.asarray(request.mesh.vertices, dtype=float)
        xy_vertices = GeometryManager._select_xy_vertices(
            request.mesh, vertices, request.face_indices
        )
        if xy_vertices.shape[0] < 2:
            raise ValueError(
                "Mesh must contain at least two vertices for PCA alignment."
            )

        centered_xy = xy_vertices - np.mean(xy_vertices, axis=0)
        covariance = centered_xy.T @ centered_xy / max(centered_xy.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(np.linalg.norm(long_axis)) == 0.0:
            raise ValueError("PCA long axis is degenerate.")

        axis_angle = float(np.arctan2(long_axis[1], long_axis[0]))
        rotation_angle = GeometryManager._minimal_angle_to_align_axis(
            axis_angle, np.pi / 2.0
        )
        rotation = GeometryManager._z_axis_rotation_transform(rotation_angle)
        aligned = request.mesh.copy()
        aligned.apply_transform(rotation)
        return AlignXYLongAxisResult(
            mesh=aligned,
            yaw_angle_degrees=float(np.rad2deg(rotation_angle)),
        )


    @staticmethod
    def _align_vector_to_axis(
        mesh: Any,
        *,
        source_axis: list[float],
        target_axis: list[float],
    ) -> Any:
        source = GeometryManager._normalize(
            np.asarray(source_axis, dtype=float)
        )
        target = GeometryManager._normalize(
            np.asarray(target_axis, dtype=float)
        )
        if np.linalg.norm(source) == 0:
            raise ValueError("source_axis must be non-zero.")
        if np.linalg.norm(target) == 0:
            raise ValueError("target_axis must be non-zero.")

        transform = GeometryManager._rotation_transform_between_vectors(
            source, target
        )
        aligned = mesh.copy()
        aligned.apply_transform(transform)
        return aligned


    @staticmethod
    def _find_support_plane_candidates(
        mesh: Any,
        *,
        normal_angle_tol_deg: float = 8.0,
        plane_distance_tol: float | None = None,
        min_area_ratio: float = 0.02,
        max_candidates: int = 24,
    ) -> list[SupportPlaneCandidate]:
        GeometryManager._validate_mesh(mesh)

        normals = np.asarray(mesh.face_normals, dtype=float)
        centers = np.asarray(mesh.triangles_center, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        vertices = np.asarray(mesh.vertices, dtype=float)
        total_area = float(np.sum(areas))
        if total_area <= 0:
            raise ValueError("Mesh has no positive face area.")

        if plane_distance_tol is None:
            extent = float(
                np.linalg.norm(np.asarray(mesh.extents, dtype=float))
            )
            plane_distance_tol = max(extent * 0.01, 1e-4)

        cos_tol = float(np.cos(np.deg2rad(normal_angle_tol_deg)))
        min_area = total_area * min_area_ratio
        order = np.argsort(-areas)
        used = np.zeros(len(areas), dtype=bool)
        candidates: list[SupportPlaneCandidate] = []

        for seed_index in order:
            if used[seed_index]:
                continue
            seed_normal = GeometryManager._normalize(normals[seed_index])
            if np.linalg.norm(seed_normal) == 0:
                used[seed_index] = True
                continue

            seed_center = centers[seed_index]
            seed_offset = float(np.dot(seed_normal, seed_center))
            normal_match = normals @ seed_normal >= cos_tol
            offsets = centers @ seed_normal
            plane_match = np.abs(offsets - seed_offset) <= plane_distance_tol
            face_mask = normal_match & plane_match & ~used
            face_indices = np.flatnonzero(face_mask)
            if len(face_indices) == 0:
                used[seed_index] = True
                continue

            used[face_indices] = True
            area = float(np.sum(areas[face_indices]))
            if area < min_area:
                continue

            weighted_normal = GeometryManager._normalize(
                np.sum(
                    normals[face_indices] * areas[face_indices, None], axis=0
                ),
            )
            center = (
                np.sum(
                    centers[face_indices] * areas[face_indices, None], axis=0
                )
                / area
            )
            candidate = GeometryManager._build_candidate(
                normal=weighted_normal,
                center=center,
                area=area,
                face_indices=face_indices,
                vertices=vertices,
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:max_candidates]

    @staticmethod
    def _select_tabletop_plane(
        candidates: list[SupportPlaneCandidate],
    ) -> SupportPlaneCandidate:
        if not candidates:
            raise ValueError("No support-plane candidates were found.")
        return max(candidates, key=lambda c: c.score)

    @staticmethod
    def _orient_plane_normal(
        mesh: Any,
        *,
        plane_normal: list[float],
        plane_center: list[float],
    ) -> list[float]:
        GeometryManager._validate_mesh(mesh)

        normal = GeometryManager._normalize(
            np.asarray(plane_normal, dtype=float)
        )
        center = np.asarray(plane_center, dtype=float)
        if np.linalg.norm(normal) == 0:
            raise ValueError("plane_normal must be non-zero.")

        vertices = np.asarray(mesh.vertices, dtype=float)
        signed_distances = (vertices - center) @ normal
        positive_mask = signed_distances > 1e-6
        negative_mask = signed_distances < -1e-6
        positive_score = float(np.sum(np.abs(signed_distances[positive_mask])))
        negative_score = float(np.sum(np.abs(signed_distances[negative_mask])))

        if positive_score > negative_score:
            normal = -normal
        return [float(v) for v in normal]

    @staticmethod
    def _build_candidate(
        *,
        normal: Any,
        center: Any,
        area: float,
        face_indices: Any,
        vertices: Any,
    ) -> SupportPlaneCandidate:
        signed_distances = (vertices - center) @ normal
        below_mask = signed_distances < -1e-6
        above_mask = signed_distances > 1e-6
        below_count = int(np.count_nonzero(below_mask))
        above_count = int(np.count_nonzero(above_mask))
        below_score = float(np.sum(np.abs(signed_distances[below_mask])))
        above_score = float(np.sum(np.abs(signed_distances[above_mask])))

        smaller_score = min(below_score, above_score)
        larger_score = max(below_score, above_score)
        asymmetry_score = min(
            (larger_score + 1e-9) / (smaller_score + 1e-9), 10.0
        )
        score = float(area * asymmetry_score)
        return SupportPlaneCandidate(
            normal=[float(v) for v in normal],
            center=[float(v) for v in center],
            area=area,
            face_indices=[int(i) for i in face_indices],
            below_vertex_count=below_count,
            above_vertex_count=above_count,
            below_area_score=below_score,
            above_area_score=above_score,
            score=score,
        )


    @staticmethod
    def _select_xy_vertices(
        mesh: Any,
        vertices: Any,
        face_indices: list[int] | None,
    ) -> Any:
        if face_indices is None:
            return vertices[:, :2]

        faces = np.asarray(mesh.faces, dtype=int)
        selected_faces = faces[np.asarray(face_indices, dtype=int)]
        selected_vertex_indices = np.unique(selected_faces.reshape(-1))
        return vertices[selected_vertex_indices, :2]

    @staticmethod
    def _minimal_angle_to_align_axis(
        source_angle: float, target_angle: float
    ) -> float:
        candidates = [
            GeometryManager._wrap_to_pi(target_angle - source_angle),
            GeometryManager._wrap_to_pi(
                target_angle + 3.141592653589793 - source_angle
            ),
        ]
        return min(candidates, key=abs)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        two_pi = 2.0 * 3.141592653589793
        return (angle + 3.141592653589793) % two_pi - 3.141592653589793

    @staticmethod
    def _z_axis_rotation_transform(angle: float) -> Any:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return transform


    @staticmethod
    def _rotation_transform_between_vectors(
        source: Any, target: Any
    ) -> Any:
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        transform = np.eye(4)
        if dot > 1.0 - 1e-8:
            return transform
        if dot < -1.0 + 1e-8:
            axis = GeometryManager._orthogonal_axis(source)
            rotation = GeometryManager._axis_angle_rotation(axis, np.pi)
        else:
            axis = GeometryManager._normalize(np.cross(source, target))
            angle = float(np.arccos(dot))
            rotation = GeometryManager._axis_angle_rotation(axis, angle)
        transform[:3, :3] = rotation
        return transform

    @staticmethod
    def _axis_angle_rotation(axis: Any, angle: float) -> Any:
        axis = GeometryManager._normalize(axis)
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=float,
        )

    @staticmethod
    def _orthogonal_axis(vector: Any) -> Any:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(vector, axis))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        return GeometryManager._normalize(np.cross(vector, axis))

    @staticmethod
    def _normalize(vector: Any) -> Any:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    @staticmethod
    def _mean_abs_log_ratio_error(lhs: Any, rhs: Any) -> float:
        eps = 1.0e-6
        lhs = np.maximum(np.asarray(lhs, dtype=np.float64), eps)
        rhs = np.maximum(np.asarray(rhs, dtype=np.float64), eps)
        return float(np.mean(np.abs(np.log(lhs / rhs))))

    @staticmethod
    def _validate_mesh(mesh: Any) -> None:
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise ValueError("Loaded geometry is not a mesh.")
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Mesh must contain vertices and faces.")
