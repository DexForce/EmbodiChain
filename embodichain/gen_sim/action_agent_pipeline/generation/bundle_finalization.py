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

"""Finalize generated mesh settings, summaries, and artifact writes."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    write_config_bundle as _write_config_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    GeneratedActionAgentConfigPaths,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_geometry_baking import (
    GlbGeometryNormalizer,
    bake_body_scale_into_glbs,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    iter_mesh_object_configs,
)

__all__ = ["_finalize_and_write_bundle", "_validate_acd_method"]


def _finalize_and_write_bundle(
    bundle: dict[str, Any],
    *,
    output_dir: Path,
    mesh_normalizer: GlbGeometryNormalizer,
    acd_method: str,
    overwrite: bool,
) -> GeneratedActionAgentConfigPaths:
    """Finalize runtime configuration and its matching diagnostic records.

    Geometry metadata is attached before writing so both the executable JSON
    files and the review summary describe the same deterministic generation
    result. The seed graph remains generation provenance while the compiled
    task graph remains the sole runtime execution plan.
    """
    acd_method = _validate_acd_method(acd_method)
    _apply_acd_method(
        bundle["gym_config"],
        method=acd_method,
    )
    _attach_mesh_normalization_summary(bundle, mesh_normalizer)
    _attach_body_scale_bake_summary(bundle, output_dir)
    summary = bundle.setdefault("summary", {})
    summary["mesh_loading_mode"] = "baked_glb"
    summary["acd_method"] = acd_method
    summary.pop("convex_decomposition_method", None)
    return _write_config_bundle(
        output_dir=output_dir,
        bundle=bundle,
        overwrite=overwrite,
    )


def _validate_acd_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized != "vhacd":
        raise ValueError("acd_method must be 'vhacd'")
    return normalized


def _apply_acd_method(
    gym_config: dict[str, Any],
    *,
    method: str,
) -> None:
    for obj in _iter_generated_mesh_objects(gym_config):
        obj.pop("convex_decomposition_method", None)
        obj.pop("acd_method", None)
        shape = obj.get("shape")
        if isinstance(shape, MutableMapping):
            shape.pop("convex_decomposition_method", None)
            shape.pop("acd_method", None)

        max_convex_hull_num = int(obj.get("max_convex_hull_num", 1))
        if max_convex_hull_num > 1:
            obj["acd_method"] = method
            if isinstance(shape, MutableMapping):
                shape["acd_method"] = method


def _iter_generated_mesh_objects(
    gym_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return iter_mesh_object_configs(gym_config)


def _attach_body_scale_bake_summary(
    bundle: dict[str, Any],
    output_dir: Path,
) -> None:
    reports = bake_body_scale_into_glbs(
        bundle["gym_config"],
        output_dir=output_dir / "mesh_assets" / "baked_glb",
    )
    if reports:
        bundle.setdefault("summary", {})["body_scaled_meshes"] = reports


def _attach_mesh_normalization_summary(
    bundle: dict[str, Any],
    mesh_normalizer: GlbGeometryNormalizer,
) -> None:
    if mesh_normalizer is None:
        return
    reports = mesh_normalizer.reports
    if reports:
        bundle.setdefault("summary", {})["normalized_meshes"] = reports
