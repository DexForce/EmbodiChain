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


METRIC_SCALE_ENABLED = True

from .utils import (
    _as_transform,
    _axis_angle_rotation,
    _axis_conversion_transform,
    _center_aabb_bottom_xy_at_origin,
    _center_aabb_bottom_xy_at_origin_transform,
    _normalize,
    _orthogonal_axis,
    _place_above_plane_transform,
    _request_axis,
    _rotation_between_vectors,
    _scale_transform,
    _translation_transform,
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
                GravityDropRequest(glb_path=pre_gravity_path, max_convex_hull_num=16)
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
                GravityDropRequest(glb_path=pre_gravity_path, max_convex_hull_num=8)
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
    @staticmethod
    def estimate_metric_scales(request):
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager.schemas import (
            EstimateMetricScalesRequest,
            EstimateMetricScalesResult,
        )
        from embodichain.gen_sim.prompt2scene.llms.llm_output import (
            call_structured_json_model_step,
        )
        from embodichain.gen_sim.prompt2scene.utils.io import write_json

        object_payload = SimreadyManager.build_object_payload(request.objects)
        raw_model_output_path = (
            request.raw_output_path.expanduser().resolve()
            if request.raw_output_path is not None
            else None
        )
        raw_model_output = call_structured_json_model_step(
            llm=request.llm,
            schema=request.schema,
            messages=request.messages,
            context=request.context,
            attempt_count=0,
            raw_output_writer=(
                (lambda payload: write_json(raw_model_output_path, payload))
                if raw_model_output_path is not None
                else None
            ),
        )
        object_scales = SimreadyManager.apply_model_output(
            object_payload=object_payload,
            raw_model_output=raw_model_output,
            method=request.method,
        )
        return EstimateMetricScalesResult(
            status="ok",
            object_scales=object_scales,
            object_payload=object_payload,
            raw_model_output=raw_model_output,
        )


    @staticmethod
    def build_object_payload(objects):
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
            GeometryManager,
            LoadMeshRequest,
        )
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager.schemas import (
            MetricScaleObjectInput,
        )

        geom = GeometryManager()
        payload = []
        for obj in objects:
            mesh = geom.load_mesh(LoadMeshRequest(mesh_path=obj.mesh_path)).mesh
            normalized_bbox_size_m = GeometryManager.mesh_metric_bbox_size(mesh)
            payload.append({
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "object_description": obj.object_description,
                "normalized_bbox_method": "pca_bbox",
                "normalized_bbox_size_m": normalized_bbox_size_m.tolist(),
                "normalized_bbox_ratio": GeometryManager.bbox_ratio(
                    normalized_bbox_size_m
                ).tolist(),
            })
        return payload


    @staticmethod
    def object_prompt_payload(objects):
        return [
            {
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "object_description": obj.object_description,
            }
            for obj in objects
        ]


    @staticmethod
    def apply_model_output(*, object_payload, raw_model_output, method):
        import numpy as np

        from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
            GeometryManager,
        )

        model_by_id = {
            str(item.get("object_id", "")): item
            for item in raw_model_output.get("object_scales", [])
            if isinstance(item, dict)
        }
        estimates = []
        for p in object_payload:
            oid = str(p.get("object_id", ""))
            model_item = model_by_id.get(oid)
            if model_item is None:
                estimates.append(SimreadyManager.failure(
                    object_id=oid,
                    reason="missing_object_scale_from_model",
                    method=method,
                ))
                continue
            estimates.append(SimreadyManager.select_candidate(
                object_id=oid,
                object_name=str(p.get("object_name", "")),
                object_description=str(p.get("object_description", "")),
                bbox_dims_cm=model_item.get("bbox_dims_cm", []),
                confidence=float(model_item.get("confidence", 0.0)),
                reason=str(model_item.get("reason", "")),
                normalized_bbox_size_m=np.asarray(
                    p["normalized_bbox_size_m"], dtype=np.float64
                ),
                method=method,
            ))
        return estimates


    @staticmethod
    def apply_to_objects(*, objects, object_scales):
        scale_by_id = {str(item.get("object_id", "")): item for item in object_scales}
        for obj in objects:
            oid = str(obj.get("id", ""))
            if oid in scale_by_id:
                obj["metric_scale"] = scale_by_id[oid]


    @staticmethod
    def select_candidate(*, object_id, object_name, object_description, bbox_dims_cm, confidence, reason, normalized_bbox_size_m, method):
        import numpy as np
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
            GeometryManager,
        )
        try:
            selected = SimreadyManager.compute_from_bbox_dims(
                bbox_dims_cm=bbox_dims_cm,
                confidence=confidence,
                reason=reason,
                normalized_bbox_size_m=normalized_bbox_size_m,
            )
        except (TypeError, ValueError):
            return SimreadyManager.failure(
                object_id=object_id,
                reason="invalid_bbox_dims_cm",
                method=method,
            )
        nbs_cm = np.asarray(normalized_bbox_size_m, dtype=np.float64) * 100.0
        return {
            "status": "ok",
            "method": method,
            "object_id": object_id,
            "object_name": object_name,
            "object_description": object_description,
            "normalized_bbox_method": "pca_bbox",
            "normalized_bbox_size_m": normalized_bbox_size_m.tolist(),
            "normalized_bbox_size_cm": nbs_cm.tolist(),
            "normalized_bbox_ratio": GeometryManager.bbox_ratio(
                normalized_bbox_size_m
            ).tolist(),
            "bbox_dims_cm": selected["bbox_dims_cm"],
            "axis_match": selected["axis_match"],
            "scale_factor": selected["scale_factor"],
            "confidence": selected["confidence"],
            "reason": selected["reason"],
            "unit_note": "scale_factor is not baked into this GLB.",
        }


    @staticmethod
    def compute_from_bbox_dims(*, bbox_dims_cm, confidence, reason, normalized_bbox_size_m):
        import numpy as np
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
            GeometryManager,
        )
        dims_cm = np.asarray([float(v) for v in bbox_dims_cm], dtype=np.float64)
        if dims_cm.shape != (3,) or np.any(dims_cm <= 0.0):
            raise ValueError("bbox_dims_cm must contain three positive values.")
        nbs_cm = np.asarray(normalized_bbox_size_m, dtype=np.float64) * 100.0
        axis_match = GeometryManager.best_axis_bbox_scale_match(
            source_size_cm=nbs_cm,
            target_size_cm=dims_cm,
        )
        return {
            "bbox_dims_cm": dims_cm.tolist(),
            "axis_match": axis_match,
            "scale_factor": float(axis_match["scale_factor"]),
            "confidence": confidence,
            "reason": reason,
        }


    @staticmethod
    def failure(*, object_id, reason, method):
        return {
            "status": "failed",
            "method": method,
            "object_id": object_id,
            "scale_factor": 1.0,
            "reason": reason,
        }


    @staticmethod
    def set_for_all_objects(*, objects, status, reason, method):
        for obj in objects:
            obj["metric_scale"] = {
                "status": status,
                "method": method,
                "object_id": str(obj.get("id", "")),
                "scale_factor": 1.0,
                "reason": reason,
            }


    @staticmethod
    def compute_global_from_object_scenes(request):
        import numpy as np
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
            GeometryManager,
        )
        from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager.schemas import (
            GlobalMetricScaleRequest,
        )

        if not METRIC_SCALE_ENABLED:
            return {
                "status": "disabled",
                "method": "metric_scale_disabled",
                "scale_factor": 1.0,
                "object_count": len(request.objects),
                "used_count": 0,
                "skipped_count": len(request.objects),
                "used": [],
                "skipped": [
                    {"id": str(item.get("id", "")), "reason": "metric_scale_disabled"}
                    for item in request.objects
                ],
                "unit_note": "Metric scale is disabled; GLBs keep simready size.",
            }

        used = []
        skipped = []
        object_by_id = {str(item.get("id", "")): item for item in request.objects}
        for object_id, scene in request.object_scenes:
            item = object_by_id.get(object_id)
            if item is None:
                skipped.append({"id": object_id, "reason": "missing_object_record"})
                continue
            ms = item.get("metric_scale")
            if not isinstance(ms, dict):
                skipped.append({"id": object_id, "reason": "missing_metric_scale"})
                continue
            if ms.get("status") != "ok":
                skipped.append({"id": object_id, "reason": str(ms.get("status") or "not_ok")})
                continue
            sf = float(ms.get("scale_factor", 1.0))
            if not np.isfinite(sf) or sf <= 0.0:
                skipped.append({"id": object_id, "reason": "invalid_simready_scale_factor"})
                continue
            try:
                srs = np.asarray([float(v) for v in ms.get("normalized_bbox_size_m", [])], dtype=np.float64)
            except (TypeError, ValueError):
                skipped.append({"id": object_id, "reason": "invalid_normalized_bbox_size_m"})
                continue
            if srs.shape != (3,) or np.any(srs <= 0.0):
                skipped.append({"id": object_id, "reason": "invalid_normalized_bbox_size_m"})
                continue
            cs = np.asarray(
                GeometryManager.mesh_metric_bbox_size(
                    GeometryManager.scene_to_mesh(scene)
                ),
                dtype=np.float64,
            )
            if cs.shape != (3,) or np.any(cs <= 0.0):
                skipped.append({"id": object_id, "reason": "invalid_current_scene_bbox"})
                continue
            geo_ratio = np.sort(cs) / np.sort(srs)
            geo_scale = float(np.median(geo_ratio))
            if not np.isfinite(geo_scale) or geo_scale <= 0.0:
                skipped.append({"id": object_id, "reason": "non_positive_geo_scale"})
                continue
            effective = sf / geo_scale
            if not np.isfinite(effective) or effective <= 0.0:
                skipped.append({"id": object_id, "reason": "non_positive_effective_scale"})
                continue
            used.append({
                "id": object_id,
                "effective_scale": effective,
                "scale_factor_simready": sf,
                "geo_scale": geo_scale,
                "simready_bbox_size_m": srs.tolist(),
                "simready_bbox_size_cm": (srs * 100.0).tolist(),
                "current_scene_bbox_size_m": cs.tolist(),
                "current_scene_bbox_size_cm": (cs * 100.0).tolist(),
                "target_bbox_dims_cm": ms.get("bbox_dims_cm"),
                "confidence": ms.get("confidence"),
            })

        if not used:
            return {
                "status": "fallback",
                "method": "simready_reference_geo_ratio_mean_with_clamp",
                "scale_factor": 1.0,
                "raw_scale_factor": 1.0,
                "was_clamped": False,
                "clamp": {"min": request.min_scale, "max": request.max_scale},
                "object_count": len(request.objects),
                "used_count": 0,
                "skipped_count": len(skipped),
                "used": [],
                "skipped": skipped,
                "unit_note": "No valid metric scale available.",
            }

        raw = float(np.mean([u["effective_scale"] for u in used]))
        sf = float(np.clip(raw, request.min_scale, request.max_scale))
        return {
            "status": "ok",
            "method": "simready_reference_geo_ratio_mean_with_clamp",
            "scale_factor": sf,
            "raw_scale_factor": raw,
            "was_clamped": bool(sf != raw),
            "clamp": {"min": request.min_scale, "max": request.max_scale},
            "object_count": len(request.objects),
            "used_count": len(used),
            "skipped_count": len(skipped),
            "used": used,
            "skipped": skipped,
            "unit_note": (
                f"Global scale via per-object metric scale / geo ratio, "
                f"clamped to [{request.min_scale:.2f}, {request.max_scale:.2f}]."
            ),
        }
