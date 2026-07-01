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

from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager.manager import (
    DEFAULT_INPUT_UP_AXIS,
    DEFAULT_UP_AXIS,
    GeometryManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager.schemas import (
    AlignToAxisRequest,
    CenterMeshRequest,
    ConvertUpAxisRequest,
    DetectTabletopRequest,
    ExportMeshRequest,
    LoadMeshRequest,
    NormalizeRequest,
    PlaceAbovePlaneRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.matplotlib_manager.manager import (
    MatplotlibManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.matplotlib_manager.schemas import (
    RenderSupportRegionRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager.schemas import (
    MakeAssetSimreadyRequest,
    MakeAssetSimreadyResult,
    MakeTableSimreadyRequest,
    MakeTableSimreadyResult,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.manager import (
    SimulationManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.schemas import (
    GravityDropRequest,
)


class SimreadyManager:
    """Prepare generated GLB assets for simulation placement."""

    def __init__(
        self,
        *,
        geometry_manager: GeometryManager | None = None,
        simulation_manager: SimulationManager | None = None,
        matplotlib_manager: MatplotlibManager | None = None,
    ) -> None:
        self.geometry_manager = geometry_manager or GeometryManager()
        self.simulation_manager = simulation_manager or SimulationManager()
        self.matplotlib_manager = matplotlib_manager or MatplotlibManager()

    def make_asset_simready(
        self,
        request: MakeAssetSimreadyRequest,
    ) -> MakeAssetSimreadyResult:
        input_path = request.input_path.expanduser().resolve()
        output_path = request.output_path.expanduser().resolve()
        if output_path.suffix.lower() != ".glb":
            raise ValueError("Sim-ready asset output_path must be a .glb file.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_up_axis = _request_axis(request.input_up_axis, DEFAULT_INPUT_UP_AXIS)
        raw_to_simready = np.eye(4, dtype=np.float64)
        geom = self.geometry_manager
        sim = self.simulation_manager

        mesh = geom.load_mesh(LoadMeshRequest(mesh_path=input_path)).mesh

        transform = _axis_conversion_transform(input_up_axis, DEFAULT_UP_AXIS)
        raw_to_simready = transform @ raw_to_simready
        mesh = geom.convert_up_axis(
            ConvertUpAxisRequest(
                mesh=mesh,
                input_up_axis=input_up_axis,
                output_up_axis=DEFAULT_UP_AXIS,
            )
        ).mesh

        center_result = geom.center_by_bbox(CenterMeshRequest(mesh=mesh))
        mesh = center_result.mesh
        transform = _translation_transform(-np.asarray(center_result.bbox_center))
        raw_to_simready = transform @ raw_to_simready

        transform = _place_above_plane_transform(mesh, request.ground_clearance)
        raw_to_simready = transform @ raw_to_simready
        mesh = geom.place_above_plane(
            PlaceAbovePlaneRequest(mesh=mesh, clearance=request.ground_clearance)
        ).mesh

        pre_gravity_mesh = geom.convert_up_axis(
            ConvertUpAxisRequest(
                mesh=mesh,
                input_up_axis=DEFAULT_UP_AXIS,
                output_up_axis=DEFAULT_INPUT_UP_AXIS,
            )
        ).mesh
        pre_gravity_path = output_path.with_name(f".{output_path.stem}_pre_gravity.glb")
        geom.export_mesh(
            ExportMeshRequest(mesh=pre_gravity_mesh, output_path=pre_gravity_path)
        )
        try:
            gravity_result = sim.run_gravity_simulation(
                GravityDropRequest(glb_path=pre_gravity_path, max_convex_hull_num=32)
            )

            gravity_transform = _as_transform(gravity_result.final_pose)
            settled_mesh = mesh.copy()
            settled_mesh.apply_transform(gravity_transform)
            raw_to_simready = gravity_transform @ raw_to_simready
            transform = _center_aabb_bottom_xy_at_origin_transform(settled_mesh)
            settled_mesh.apply_transform(transform)
            raw_to_simready = transform @ raw_to_simready

            transform = _center_aabb_bottom_xy_at_origin_transform(settled_mesh)
            raw_to_simready = transform @ raw_to_simready
            final_mesh = _center_aabb_bottom_xy_at_origin(settled_mesh)

            normalize_result = geom.normalize(NormalizeRequest(mesh=final_mesh))
            final_mesh = normalize_result.mesh
            transform = _scale_transform(normalize_result.scale_factor)
            raw_to_simready = transform @ raw_to_simready

            transform = _place_above_plane_transform(final_mesh, request.ground_clearance)
            raw_to_simready = transform @ raw_to_simready
            final_mesh = geom.place_above_plane(
                PlaceAbovePlaneRequest(
                    mesh=final_mesh,
                    clearance=request.ground_clearance,
                )
            ).mesh

            transform = _axis_conversion_transform(DEFAULT_UP_AXIS, DEFAULT_INPUT_UP_AXIS)
            raw_to_simready = transform @ raw_to_simready
            final_mesh = geom.convert_up_axis(
                ConvertUpAxisRequest(
                    mesh=final_mesh,
                    input_up_axis=DEFAULT_UP_AXIS,
                    output_up_axis=DEFAULT_INPUT_UP_AXIS,
                )
            ).mesh

            geom.export_mesh(ExportMeshRequest(mesh=final_mesh, output_path=output_path))
        finally:
            pre_gravity_path.unlink(missing_ok=True)

        return MakeAssetSimreadyResult(
            output_path=output_path,
            transform_matrix=raw_to_simready.tolist(),
        )

    def make_table_simready(
        self,
        request: MakeTableSimreadyRequest,
    ) -> MakeTableSimreadyResult:
        input_path = request.input_path.expanduser().resolve()
        output_path = request.output_path.expanduser().resolve()
        if output_path.suffix.lower() != ".glb":
            raise ValueError("Sim-ready table output_path must be a .glb file.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_up_axis = _request_axis(request.input_up_axis, DEFAULT_INPUT_UP_AXIS)
        up_axis = _request_axis(request.up_axis, DEFAULT_UP_AXIS)
        raw_to_simready = np.eye(4, dtype=np.float64)
        geom = self.geometry_manager
        sim = self.simulation_manager
        mpl = self.matplotlib_manager

        mesh = geom.load_mesh(LoadMeshRequest(mesh_path=input_path)).mesh

        transform = _axis_conversion_transform(input_up_axis, DEFAULT_UP_AXIS)
        raw_to_simready = transform @ raw_to_simready
        mesh = geom.convert_up_axis(
            ConvertUpAxisRequest(
                mesh=mesh,
                input_up_axis=input_up_axis,
                output_up_axis=DEFAULT_UP_AXIS,
            )
        ).mesh

        center_result = geom.center_by_bbox(CenterMeshRequest(mesh=mesh))
        mesh = center_result.mesh
        transform = _translation_transform(-np.asarray(center_result.bbox_center))
        raw_to_simready = transform @ raw_to_simready

        detect_result = geom.detect_tabletop(DetectTabletopRequest(mesh=mesh))

        transform = _axis_conversion_transform(detect_result.oriented_normal, up_axis)
        raw_to_simready = transform @ raw_to_simready
        mesh = geom.align_to_axis(
            AlignToAxisRequest(
                mesh=mesh,
                source_axis=detect_result.oriented_normal,
                target_axis=up_axis,
            )
        ).mesh

        transform = _place_above_plane_transform(mesh, request.ground_clearance)
        raw_to_simready = transform @ raw_to_simready
        mesh = geom.place_above_plane(
            PlaceAbovePlaneRequest(mesh=mesh, clearance=request.ground_clearance)
        ).mesh

        pre_gravity_mesh = geom.convert_up_axis(
            ConvertUpAxisRequest(
                mesh=mesh,
                input_up_axis=DEFAULT_UP_AXIS,
                output_up_axis=DEFAULT_INPUT_UP_AXIS,
            )
        ).mesh
        pre_gravity_path = output_path.with_name(f".{output_path.stem}_pre_gravity.glb")
        geom.export_mesh(
            ExportMeshRequest(mesh=pre_gravity_mesh, output_path=pre_gravity_path)
        )
        try:
            gravity_result = sim.run_gravity_simulation(
                GravityDropRequest(glb_path=pre_gravity_path, max_convex_hull_num=16)
            )

            gravity_transform = _as_transform(gravity_result.final_pose)
            settled_mesh = mesh.copy()
            settled_mesh.apply_transform(gravity_transform)
            raw_to_simready = gravity_transform @ raw_to_simready
            transform = _center_aabb_bottom_xy_at_origin_transform(settled_mesh)
            settled_mesh.apply_transform(transform)
            raw_to_simready = transform @ raw_to_simready

            settled_detect = geom.detect_tabletop(
                DetectTabletopRequest(mesh=settled_mesh)
            )

            mpl.render_selected_support_region(
                RenderSupportRegionRequest(
                    mesh=settled_mesh,
                    face_indices=settled_detect.selected.face_indices,
                    output_path=output_path.with_name(
                        f"{output_path.stem}_support_region.png"
                    ),
                )
            )

            transform = _center_aabb_bottom_xy_at_origin_transform(settled_mesh)
            raw_to_simready = transform @ raw_to_simready
            final_mesh = _center_aabb_bottom_xy_at_origin(settled_mesh)

            normalize_result = geom.normalize(NormalizeRequest(mesh=final_mesh))
            final_mesh = normalize_result.mesh
            transform = _scale_transform(normalize_result.scale_factor)
            raw_to_simready = transform @ raw_to_simready

            transform = _place_above_plane_transform(final_mesh, request.ground_clearance)
            raw_to_simready = transform @ raw_to_simready
            final_mesh = geom.place_above_plane(
                PlaceAbovePlaneRequest(
                    mesh=final_mesh,
                    clearance=request.ground_clearance,
                )
            ).mesh

            transform = _axis_conversion_transform(DEFAULT_UP_AXIS, DEFAULT_INPUT_UP_AXIS)
            raw_to_simready = transform @ raw_to_simready
            final_mesh = geom.convert_up_axis(
                ConvertUpAxisRequest(
                    mesh=final_mesh,
                    input_up_axis=DEFAULT_UP_AXIS,
                    output_up_axis=DEFAULT_INPUT_UP_AXIS,
                )
            ).mesh

            geom.export_mesh(ExportMeshRequest(mesh=final_mesh, output_path=output_path))
        finally:
            pre_gravity_path.unlink(missing_ok=True)

        return MakeTableSimreadyResult(
            output_path=output_path,
            transform_matrix=raw_to_simready.tolist(),
        )


def _request_axis(value: list[float] | None, default: tuple[float, float, float]) -> list[float]:
    if value is not None:
        return list(value)
    return list(default)


def _center_aabb_bottom_xy_at_origin(mesh: Any) -> Any:
    bounds = mesh.bounds
    bottom_center_x = (float(bounds[0][0]) + float(bounds[1][0])) * 0.5
    bottom_center_y = (float(bounds[0][1]) + float(bounds[1][1])) * 0.5
    centered = mesh.copy()
    centered.apply_translation([-bottom_center_x, -bottom_center_y, 0.0])
    return centered


def _axis_conversion_transform(source_axis: list[float], target_axis: list[float]) -> np.ndarray:
    source = _normalize(np.asarray(source_axis, dtype=np.float64))
    target = _normalize(np.asarray(target_axis, dtype=np.float64))
    return _rotation_between_vectors(source, target)


def _place_above_plane_transform(mesh: Any, clearance: float) -> np.ndarray:
    min_z = float(mesh.bounds[0][2])
    return _translation_transform(np.array([0.0, 0.0, clearance - min_z]))


def _center_aabb_bottom_xy_at_origin_transform(mesh: Any) -> np.ndarray:
    bounds = mesh.bounds
    bottom_center_x = (float(bounds[0][0]) + float(bounds[1][0])) * 0.5
    bottom_center_y = (float(bounds[0][1]) + float(bounds[1][1])) * 0.5
    return _translation_transform(np.array([-bottom_center_x, -bottom_center_y, 0.0]))


def _translation_transform(translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return transform


def _scale_transform(scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= float(scale)
    return transform


def _as_transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("Expected a 4x4 transform matrix.")
    return transform


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    transform = np.eye(4, dtype=np.float64)
    if dot > 1.0 - 1e-8:
        return transform
    if dot < -1.0 + 1e-8:
        axis = _orthogonal_axis(source)
        rotation = _axis_angle_rotation(axis, np.pi)
    else:
        axis = _normalize(np.cross(source, target))
        angle = float(np.arccos(dot))
        rotation = _axis_angle_rotation(axis, angle)
    transform[:3, :3] = rotation
    return transform


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
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
        dtype=np.float64,
    )


def _orthogonal_axis(vector: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(vector, axis))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _normalize(np.cross(vector, axis))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
