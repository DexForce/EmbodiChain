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

"""Semantic selection and provenance over independent source-control physical owners."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from embodichain.compute.geometry.control_grasp_scale import select_control_grasp_scale
from embodichain.data.asset_dependencies import collect_declared_asset_files
from embodichain.lab.sim.scene.control_preparation import (
    ControlSourceScene,
    ControlSourceResizeRequest,
    prepare_control_source_scene,
)
from embodichain.lab.sim.scene.press_inputs import discover_press_source
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.scene.control_semantics import resolve_twist_semantics
from embodichain.lab.sim.scene.press_calibration import (
    PressSourceCalibrationRequest,
    calibrate_press_source_declarations,
)
from embodichain.lab.sim.scene.control_mass_source import (
    create_mass_source_copy,
    requires_derived_revolute_mass_source,
)
from embodichain_robots.task_program import resolve_task_hand_calibrations
from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model
from embodichain.utils.utility import load_config
from .orchestration.source_scene import PreparedScene

__all__ = ["PreparedControlPhysicalSources", "prepare_source_control_physics"]


@dataclass(frozen=True, slots=True)
class PreparedControlPhysicalSources:
    """Detached semantic scene projection and original physical-owner receipts."""

    scene: PreparedScene
    artifacts: Mapping[str, Any]
    input_files: tuple[Path, ...]
    mass_owned_uids: tuple[str, ...]


def _project(scene: PreparedScene, physical: ControlSourceScene) -> PreparedScene:
    configs = {
        value["uid"]: value
        for value in (
            *physical.background,
            *physical.rigid_objects,
            *physical.articulations,
        )
    }
    planner = []
    for item in scene.planner_objects:
        owned = deepcopy(item)
        cfg = configs.get(item["runtime_uid"])
        if cfg is not None:
            for field in ("body_scale", "init_pos", "init_local_pose", "fpath"):
                if field in cfg:
                    owned[field] = deepcopy(cfg[field])
        planner.append(owned)
    return replace(
        scene,
        background=physical.background,
        rigid_objects=physical.rigid_objects,
        articulations=physical.articulations,
        planner_objects=tuple(planner),
    )


def _source_relationships(scene, graph, has_twist):
    referenced = set()
    known = {
        str(value["runtime_uid"])
        for value in scene.planner_objects
        if value.get("runtime_uid")
    }

    def collect(value):
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                collect(nested)
        elif isinstance(value, str) and value in known:
            referenced.add(value)

    collect(graph)
    eligible = set()
    payloads = set()
    path = scene.source_config_path.with_name("scene_graph.json")
    if has_twist and path.is_file():
        nodes = load_config(path).get("nodes", ())

        def uid(value):
            return scene.uid_map.get(value, value)

        rigids = {str(value["uid"]) for value in scene.rigid_objects}
        arts = {str(value["uid"]) for value in scene.articulations}
        payloads = {
            (uid(value.get("parent_id")), uid(value.get("object_id")))
            for value in nodes
            if isinstance(value, dict)
            and value.get("parent_relation") in {"on", "inside"}
            and uid(value.get("parent_id")) in arts
            and uid(value.get("object_id")) in rigids
        }
        supports = {
            uid(value.get("parent_id")) for value in nodes if isinstance(value, dict)
        }
        eligible = {
            uid(value.get("object_id"))
            for value in nodes
            if isinstance(value, dict)
            and uid(value.get("parent_id")) == "table"
            and value.get("parent_relation") == "on"
        } - supports
    movable = {
        str(value["uid"])
        for value in scene.rigid_objects
        if str(value["uid"]) in eligible
        and str(value["uid"]) not in referenced
        and str(value["uid"]) != "table"
    }
    return movable, payloads, (path,) if has_twist and path.is_file() else ()


def prepare_source_control_physics(
    scene: PreparedScene,
    graph: Mapping[str, Any],
    embodiment: Mapping[str, Any],
    *,
    cache_root: str | Path,
    twist_support_lift_m: float = 0.001,
    twist_grip_depth_m: float | None = None,
    twist_mass_source: bool = False,
    press_extra_distance: float | None = None,
) -> PreparedControlPhysicalSources:
    """Select declared controls and delegate all source geometry/mass preparation.

    Args:
        scene: Normal source PreparedScene after optional rigid fit/normalization.
        graph: Validated semantic declarations selecting control subjects/settings.
        embodiment: Explicit model/hand/calibration declaration used by the host.
        cache_root: Derived physical asset stage, separate from final bundle output.
        twist_support_lift_m: Original requested bounded support-lift candidate.
        twist_grip_depth_m: Optional measured depth candidate.
        twist_mass_source: Explicit derived mass-source selection, with the source
            compatibility fallback when the Lab config lacks the old scale field.

    Returns:
        Detached scene, owned source receipts and every additional input file.
        Normative acceptance and runtime observation remain separate. Frozen
        snapshots never call this source adaptation stage.
    """
    nodes = {node["id"]: node for node in graph["nodes"]}
    configs = {value["uid"]: value for value in scene.articulations}
    presses = []
    press_metadata = {}
    twists = []
    for group in graph["task_groups"]:
        if group["task_type"] not in {"E8", "E9"}:
            continue
        call = nodes[group["node_ids"][0]]["call"]
        args = call.get("arguments", {})
        uid = args["object"]
        if uid not in configs:
            raise ValueError(
                "Control preparation requires its declared articulation owner."
            )
        if group["task_type"] == "E9":
            binding, mass = discover_press_source(dict(configs[uid]))
            presses.append((binding, mass))
            press_metadata.setdefault(
                uid,
                dict(
                    arm=call["resources"]["primary"],
                    variant="calibrated",
                    extra_press_distance=(
                        0.010 if press_extra_distance is None else press_extra_distance
                    ),
                    operation_id=(
                        group["id"]
                        if call["call_id"] != "gen_sim.press_prepare"
                        else ""
                    ),
                ),
            )
        else:
            binding = inspect_revolute_control(dict(configs[uid]), args["setting"])
            twists.append((binding, call["resources"]["primary"]))
    if type(twist_mass_source) is not bool:
        raise ValueError("E8 deployment mass source must be an explicit boolean.")
    if twist_mass_source and not twists:
        raise ValueError("Deployment mass source requires a standalone E8 deployment.")
    if twist_grip_depth_m is not None and not twists:
        raise ValueError("Explicit knob depth candidates require an E8 deployment.")
    if not presses and not twists:
        return PreparedControlPhysicalSources(scene, {}, (), ())
    physical = ControlSourceScene(
        scene.background, scene.rigid_objects, scene.articulations
    )
    artifacts = {}
    inputs = []
    twist_uids = {value.object_id for value, _ in twists}
    requests = tuple(
        ControlSourceResizeRequest(binding)
        for binding, _ in presses
        if binding.object_id not in twist_uids
    )
    physical, press_adaptation = prepare_control_source_scene(
        physical, requests, interaction_kind="prismatic"
    )
    scene = _project(scene, physical)
    if press_adaptation:
        artifacts["press_adaptation.json"] = {
            "schema_version": "gen_sim.press-adaptation/v1",
            "records": press_adaptation,
            "policy": "unit_scale_preserve_support_and_calibration",
        }
    movable, payloads, relations = _source_relationships(scene, graph, bool(twists))
    inputs.extend(relations)
    hands = {
        value.resource_id: value for value in resolve_task_hand_calibrations(embodiment)
    }
    generators = embodiment["skill_profile"]["runtime_services"][
        "grasp_pose_generators"
    ]
    scales = {}
    scale_audits = []
    twist_adaptation = []
    for binding, resource in twists:
        hand = hands[resource]
        model = resolve_parallel_jaw_gripper_model(
            generators[hand.grasp_target_id]["model"]
        )
        envelope = hand.nominal_envelope_declaration()["envelope"]
        selected, audit = select_control_grasp_scale(
            source_scale=binding.scale,
            source_grip_depth=binding.grip_depth,
            source_grip_width=binding.grip_width,
            finger_thickness=envelope["finger_thickness"],
            minimum_opening_width=model.min_opening_width,
            maximum_opening_width=model.max_opening_width,
            opening_margin=envelope["opening_margin"],
            calibration=hand.revolute_grip_calibration(),
            grip_depth_candidate=twist_grip_depth_m,
        )
        audit.update(
            object_id=binding.object_id,
            arm=resource,
            source_sha256=binding.source_sha256,
        )
        uid = binding.object_id
        if uid in scales:
            if scales[uid] != selected:
                raise ValueError(
                    "Repeated E8 operations require compatible gripper dimensions."
                )
            continue
        scales[uid] = selected
        scale_audits.append(audit)
        from embodichain.gen_sim.scene_engine.pipeline.utils.table_surface_layout_optimizer import (
            TableSurfaceLayoutOptimizerConfig,
        )

        physical, records = prepare_control_source_scene(
            physical,
            (ControlSourceResizeRequest(binding),),
            interaction_kind="revolute",
            relayout_uids=movable,
            source_payloads=payloads,
            interaction_scale=selected,
            interaction_scale_qualified=selected is not None,
            support_lift_m=twist_support_lift_m,
            relayout_margin_m=TableSurfaceLayoutOptimizerConfig().collision_margin_m,
        )
        twist_adaptation.extend(records)
    scene = _project(scene, physical)
    if twist_adaptation:
        artifacts["twist_adaptation.json"] = dict(
            schema_version="gen_sim.twist-adaptation/v1",
            policy="gripper_measured_scale_bounded_height_constrained_relayout",
            scale_candidate=scale_audits[0] if len(scale_audits) == 1 else None,
            scale_candidates=scale_audits,
            records=twist_adaptation,
        )
    mass_source = bool(twists) and (
        twist_mass_source or requires_derived_revolute_mass_source()
    )
    owned = deepcopy(list(scene.articulations))
    hashes = dict(scene.asset_hashes)
    mass_records = []
    lineages = {}
    if mass_source:
        for uid in dict.fromkeys(value.object_id for value, _ in twists):
            art = next(value for value in owned if value["uid"] == uid)
            groups = [
                art.get("attrs") or {},
                *(
                    (value.get("attrs") or {})
                    for value in (art.get("link_attrs") or {}).values()
                ),
            ]
            if any(value.get("mass_props") for value in groups):
                raise ValueError(
                    "E8 mass deployment cannot combine with mass overlays."
                )
            original_path = Path(art["fpath"])
            copied = create_mass_source_copy(
                Path(art["fpath"]),
                Path(cache_root) / "mass_sources",
                tuple(art["body_scale"]),
            )
            art["fpath"] = str(copied.runtime)
            art.pop("body_scale_mass_policy", None)
            hashes[uid] = copied.runtime_sha256
            lineages[uid] = copied.manifest_sha256
            inputs.extend((copied.source, copied.runtime, copied.manifest))
            for path in (original_path, copied.source, copied.runtime):
                inputs.extend(
                    collect_declared_asset_files(
                        {"fpath": str(path)}, base_dir=Path(cache_root)
                    )
                )
            mass_records.append(
                dict(
                    object_id=uid,
                    origin_source_sha256=copied.source_sha256,
                    runtime_sha256=copied.runtime_sha256,
                    mass_lineage_sha256=copied.manifest_sha256,
                    source_path=str(copied.source),
                    runtime_path=str(copied.runtime),
                    manifest_path=str(copied.manifest),
                    body_scale=copied.scale,
                    source_edited=False,
                    conversion_owner="lab_scene_source",
                )
            )
        physical = ControlSourceScene(
            scene.background, scene.rigid_objects, tuple(owned)
        )
        scene = _project(scene, physical)
        planner = deepcopy(list(scene.planner_objects))
        by_uid = {value["uid"]: value for value in scene.articulations}
        for item in planner:
            uid = item["runtime_uid"]
            if uid not in twist_uids:
                continue
            semantics = resolve_twist_semantics(by_uid[uid]["fpath"])
            values = {
                setting: inspect_revolute_control(
                    dict(by_uid[uid]), setting
                ).target_qpos
                for setting, _ in semantics.settings
            }
            binding = inspect_revolute_control(dict(by_uid[uid]), next(iter(values)))
            item.setdefault("attributes", {}).update(
                joint_settings={binding.joint: list(values.values())},
                twist_setting_labels={str(key): value for key, value in values.items()},
                twist_source_sha256=semantics.source_sha256,
                twist_semantics_sha256=semantics.semantics_sha256,
                twist_control_joint=binding.joint,
            )
        scene = replace(scene, planner_objects=tuple(planner), asset_hashes=hashes)
        artifacts["twist_mass_adaptation.json"] = dict(
            schema="gen_sim.twist-mass-adaptation/v1", records=mass_records
        )
    runtime = {value["uid"]: value for value in scene.articulations}
    requests = []
    for source, mass in presses:
        binding, moving = discover_press_source(dict(runtime[source.object_id]))
        requests.append(
            PressSourceCalibrationRequest(
                binding,
                source,
                mass,
                moving,
                mass_lineage_sha256=lineages.get(source.object_id),
            )
        )
    arts, calibration = calibrate_press_source_declarations(
        scene.articulations, tuple(requests)
    )
    for art in arts:
        if art["uid"] in twist_uids:
            art["asset_physics_mode"] = "overlay"
    if calibration:
        receipts = []
        for record in calibration:
            uid = record["object_id"]
            binding, moving = discover_press_source(dict(runtime[uid]))
            declared = asdict(binding)
            declared.pop("source_path", None)
            receipts.append(
                dict(
                    binding=declared,
                    moving_mass=moving,
                    **press_metadata[uid],
                    **{
                        key: value
                        for key, value in record.items()
                        if key != "object_id"
                    },
                )
            )
        artifacts["press_calibration.json"] = dict(routes=receipts)
    scene = replace(scene, articulations=arts)
    return PreparedControlPhysicalSources(
        scene, artifacts, tuple(dict.fromkeys(inputs)), tuple(lineages)
    )
