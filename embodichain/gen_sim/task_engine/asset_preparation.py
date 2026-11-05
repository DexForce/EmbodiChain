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

"""Semantic acquisition selection over shared source asset-preparation owners."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any
from embodichain.gen_sim.scene_engine.grasp_fit import (
    SourceGraspFitRequest,
    fit_source_grasp_declarations,
)
from embodichain.gen_sim.scene_engine.asset_normalization import (
    normalize_source_scene_assets,
)
from .orchestration.source_scene import PreparedScene

__all__ = ["fit_prepared_scene_assets", "normalize_prepared_scene_assets"]


def fit_prepared_scene_assets(
    scene: PreparedScene,
    graph: Mapping[str, Any],
    *,
    openings: Mapping[str, float],
    contact_clearances: Mapping[str, float] | None = None,
) -> tuple[PreparedScene, dict[str, Any]]:
    """Select acquired scene identities and delegate all asset-fit geometry.

    Args:
        scene: Explicit source physical/semantic identity correspondence.
        graph: Semantic acquisition declarations in source execution order.
        openings: Existing resource calibration selected by the embodiment owner.
        contact_clearances: Optional physical object margins from the Scene owner.

    Returns:
        Detached source scene and original v3 adaptation metadata. This adapter
        only collects semantic identities/provenance and updates their shared
        physical/planner declarations; it implements no geometric fitting rules.
    """
    scene.assert_source_assets()
    selected = {}
    for node in graph["nodes"]:
        call = node["call"]
        identity = str(call.get("call_id", ""))
        coordinated = identity in {
            "simulation.coordinated_hold",
            "simulation.coordinated_transport",
        }
        if not (
            call.get("kind") == "pick"
            or identity == "simulation.pick"
            or identity.startswith("gen_sim.pick.")
            or coordinated
        ):
            continue
        args = call.get("arguments", {})
        uid = str(call.get("object", args.get("object", ""))).strip()
        if not uid:
            raise ValueError("Grasp-fit acquisition has no object identifier.")
        resources = (
            tuple(str(value) for value in call["resources"].values())
            if coordinated
            else (str(call["resources"]["primary"]),)
        )
        missing = [value for value in resources if value not in openings]
        if missing:
            raise ValueError(f"Grasp-fit has no calibrated opening for {missing}.")
        opening = min(openings[value] for value in resources)
        previous = selected.get(uid)
        if previous is None:
            previous = selected[uid] = dict(
                opening=opening,
                fit_mode="minimum_axis" if coordinated else "single_arm_transverse",
                task_types=set(),
                resources=set(),
            )
        elif opening < float(previous["opening"]) - 1e-9:
            raise ValueError(
                f"Grasp-fit contract for {uid!r} becomes stricter in a later invocation; use a separate scene revision instead of rescaling an earlier action's asset."
            )
        previous["task_types"].add(str(node["task_type"]))
        previous["resources"].update(resources)
    if contact_clearances is not None:
        for uid in selected:
            if uid not in contact_clearances:
                raise ValueError(
                    f"Missing contact clearance for grasp-fit target {uid!r}."
                )
    fitted = fit_source_grasp_declarations(
        scene.rigid_objects,
        tuple(
            SourceGraspFitRequest(
                uid,
                value["opening"],
                0.0 if contact_clearances is None else contact_clearances[uid],
                value["fit_mode"],
                scene.asset_hashes.get(uid),
            )
            for uid, value in selected.items()
        ),
    )
    records = [
        dict(
            record,
            task_types=sorted(selected[record["object_id"]]["task_types"]),
            resources=sorted(selected[record["object_id"]]["resources"]),
        )
        for record in fitted.records
    ]
    status = (
        "not_applicable"
        if not records
        else (
            "scaled"
            if any(record["status"] == "scaled" for record in records)
            else "unchanged"
        )
    )
    report = dict(
        schema_version="gen_sim.asset-adaptation/v3",
        policy="semantic_grasp_fit/v3",
        enabled=True,
        minimum_scale=0.25,
        status=status,
        records=records,
    )
    if status != "scaled":
        return scene, report
    configs = {value["uid"]: value for value in fitted.rigid_objects}
    planner = []
    for value in scene.planner_objects:
        owned = deepcopy(value)
        runtime = configs.get(owned["runtime_uid"])
        if runtime is not None and owned.get("role") == "rigid_object":
            for key in ("body_scale", "init_pos", "init_local_pose"):
                if key in runtime:
                    owned[key] = deepcopy(runtime[key])
        planner.append(owned)
    return (
        replace(
            scene, rigid_objects=fitted.rigid_objects, planner_objects=tuple(planner)
        ),
        report,
    )


def normalize_prepared_scene_assets(
    scene: PreparedScene, cache_root: str
) -> PreparedScene:
    """Project a shared normalized asset result into matching semantic identities.

    Args:
        scene: Detached source scene with ordered physical/planner identities.
        cache_root: Derived-asset stage root, separate from final bundle publication.

    Returns:
        Updated scene with source-compatible geometry/hash/provenance metadata.
        Frozen physical snapshots must bypass this explicit source preparation.
    """
    scene.assert_source_assets()
    assets = normalize_source_scene_assets(
        background=scene.background,
        rigid_objects=scene.rigid_objects,
        articulations=scene.articulations,
        cache_root=cache_root,
    )
    by_uid = {
        value["uid"]: value for value in (*assets.background, *assets.rigid_objects)
    }
    planner = []
    for value in scene.planner_objects:
        owned = deepcopy(value)
        runtime = by_uid.get(str(value["runtime_uid"]))
        if runtime is not None:
            owned["shape"] = deepcopy(runtime.get("shape", {}))
            owned["body_scale"] = list(runtime.get("body_scale", [1.0] * 3))
        planner.append(owned)
    return replace(
        scene,
        planner_objects=tuple(planner),
        background=assets.background,
        rigid_objects=assets.rigid_objects,
        articulations=assets.articulations,
        asset_hashes={**scene.asset_hashes, **assets.asset_hashes},
        asset_provenance=assets.provenance,
    )
