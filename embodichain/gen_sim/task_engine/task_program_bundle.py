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

"""Materialize one SemanticTaskGraph as a configured Task Program bundle."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from embodichain.lab.task_program.language import load_task_program
from embodichain.utils.utility import load_config, save_config

from .config import TaskEnginePlanningCfg
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph
from ._task_program.assembly import ADAPTER_CONTRACT, load_deployment
from ._task_program.articulation_binding import (
    PARK_CALL,
    SLIDE_CALL,
    WITHDRAW_CALL,
    discover_prismatic_parts,
    graph_bindings,
)
from ._task_program.articulation_slide import preset_id
from ._task_program.drawer_binding import DRAWER_TRANSPORT_CALL
from ._task_program.identity import bundle_registration_fingerprint
from ._task_program.held_command import HELD_COMMAND_REVISION, held_command_declarations
from ._task_program.physical_inputs import FrozenPhysicalInput
from ._task_program.invocation_policy import (
    cartesian_call_declarations,
    motion_sample_declarations,
    pour_receiver_declarations,
    pour_return_call_declarations,
)
from ._task_program.e6_clearance import (
    ARM_STIFFNESS,
    PASSIVE_FRICTION,
    PRISMATIC_LINK_MASS,
    PRESHAPE_FRACTION,
    RELEASE_RETREAT_DISTANCE,
)
from embodichain.gen_sim.task_spec import TaskSpec

__all__ = ["TaskProgramBundlePaths", "generate_task_program_bundle"]

_EMBODIMENT_COMPONENTS: Final = {
    "dual_franka": "dual_franka_robotiq_arg2f_140.yaml",
    "dual_franka_robotiq_arg2f_140": "dual_franka_robotiq_arg2f_140.yaml",
    # Experiment-only profile; the production default remains Robotiq.
    "dual_franka_pgi": "dual_franka_robotiq_arg2f_140.yaml",
}

# The generated ``move forward`` task intent has no metric distance.  Phase one
# supports only the dual-Franka embodiment, whose two-arm top-down tray grasp
# retains a reachable continuation over 0.14 m.  Keep this semantic route
# target in the integration builder: the Atomic Action must execute the exact
# grounded goal and must not silently clamp an unreachable caller request.
_DUAL_FRANKA_COORDINATED_TRANSPORT_DISTANCE: Final = 0.14
# This is a minimum: held-object handover never lowers the current object pose.
_DUAL_FRANKA_HANDOVER_CLEARANCE: Final = 0.10
_DUAL_FRANKA_TABLE_MOUNT_OFFSET: Final = 0.35
_LATERAL_RELATION_DISTANCE: Final = 0.10


def _apply_pgi_experiment_embodiment(payload: dict[str, Any]) -> None:
    """Replace generated dual-Franka hands with the PGI experiment asset."""
    simulation = payload["simulation"]
    for component in simulation["urdf_cfg"]["components"]:
        if component["component_type"] in {"left_hand", "right_hand"}:
            component["urdf_path"] = "DH_PGI_140_80/DH_PGI_140_80.urdf"
    # PGI keeps two prismatic finger joints per hand; only the first is the
    # commanded leader, but both positions must exist in the full qpos vector.
    simulation["init_qpos"] = list(simulation["init_qpos"][:14]) + [0.0] * 4
    control_parts = simulation["control_parts"]
    drive = simulation["joint_drive_props"]
    for side in ("left", "right"):
        control_parts[f"{side}_eef"] = [f"{side}_gripper_finger1_joint_1"]
        drive["stiffness"][f"{side}_eef"] = 1000.0
        drive["damping"][f"{side}_eef"] = 100.0
        drive["max_effort"][f"{side}_eef"] = 10000.0
    payload["embodiment_id"] = "dual_franka_pgi_experiment"
    payload["skill_profile"]["profile_id"] = "dual_franka_pgi_experiment"
    for preset in payload["skill_profile"]["command_presets"]:
        if preset["preset_id"] in {"left_parallel_gripper", "right_parallel_gripper"}:
            preset["commands"] = {"open": [0.0], "grasp": [0.024]}


_FRONT_RELATION_DISTANCE: Final = 0.18
# Leave enough free space around a placed object's support reference for the
# configured parallel-jaw fingers to close during a later semantic Pick.  A
# tall object aligned by E2 needs the larger margin because its side grasp
# sweeps the finger length through the support plane.  Both routes still come
# from scene geometry rather than task-owned robot poses.
_RELATION_CLEARANCE: Final = 0.02
_AXIS_ALIGNED_RELATION_CLEARANCE: Final = 0.04
_PLACEMENT_CLEARANCE: Final = 0.01
_RELATIVE_POSITION_TOLERANCE: Final = 0.05
_AXIS_ALIGN_CALL_ID: Final = "simulation.axis_align"
_COORDINATED_TRANSPORT_CALL_ID: Final = "simulation.coordinated_transport"
_COORDINATED_DETACH_DISTANCE: Final = 0.08
_COORDINATED_RETREAT_MARGIN: Final = 0.04
_PARK_CALL_ID: Final = "simulation.park"
_HANDOVER_SOURCE_PICK: Final = "gen_sim.pick.handover_source"
_HORIZONTAL_HANDOVER_SOURCE_PICK: Final = "gen_sim.pick.handover_horizontal_source"
_DEFAULT_PICK_LIFT_HEIGHT: Final = 0.16
_PLACE_RELATIVE_CALL_ID: Final = "simulation.place_relative"
_UPRIGHT_PLACE_CALL_ID: Final = "gen_sim.place_upright"
_STACK_PLACE_CALL_ID: Final = "gen_sim.stack_place"
_STACK_PICK_CALL_ID: Final = "gen_sim.stack_pick"


_MOVE_HELD_OBJECT_CALL_ID: Final = "simulation.move_held_object"
_ALIGN_HELD_CALL_ID: Final = "gen_sim.align_held"
_CLEAR_RELEASED_CALL_ID: Final = "gen_sim.clear_released"


_PICK_CALL_ID: Final = "simulation.pick"

_POUR_CALL_ID: Final = "simulation.pour"

_COORDINATED_HOLD_CALL_ID: Final = "simulation.coordinated_hold"


_UPRIGHT_RELEASE_CLEARANCE: Final = 0.01

_SLENDER_UPRIGHT_RELEASE_CLEARANCE: Final = 0.01

_UPRIGHT_STAGING_CLEARANCE: Final = 0.20

_E2_RELEASE_SAFETY_MARGIN: Final = 0.02


@dataclass(frozen=True, slots=True)
class TaskProgramBundlePaths:
    """Files composing one portable configured Task Program deployment.

    Attributes:
        root: Bundle root directory.
        deployment: Runnable Gym deployment configuration.
        program: Embodiment-independent Task Program source.
        integration: Scene and runtime-service integration configuration.
        scene: Physical scene component.
        embodiment: Robot, sensor, and skill-profile component.
        execution_policy: Canonical runtime execution-policy component.
        semantic_task_graph: Immutable source semantic graph.
        integration_fingerprint: Composed integration identity artifact.
    """

    root: Path
    deployment: Path
    program: Path
    integration: Path
    scene: Path
    embodiment: Path
    execution_policy: Path
    semantic_task_graph: Path
    integration_fingerprint: Path


def generate_task_program_bundle(
    graph: SemanticTaskGraph,
    prepared_scene: Any,
    output_dir: str | Path,
    *,
    robot_profile: str,
    max_episodes: int | None = None,
    max_episode_steps: int | None = None,
    fit_grasp_assets: bool = False,
    task_spec: TaskSpec | None = None,
    task_candidate: Mapping[str, Any] | None = None,
    role_bindings: Mapping[str, Any] | None = None,
    step_lineage: Mapping[str, Any] | None = None,
    twist_support_lift_m: float = 0.001,
    twist_grip_depth_m: float | None = None,
    twist_contact_feedback: bool = False,
    twist_mass_source: bool = False,
    support_contract_revision: int | None = None,
    pick_contact_revision: int | None = None,
    pick_contact_revisions: Mapping[str, int] | None = None,
    world_yaw_free_places: Sequence[str] | None = None,
    post_handover_places: Sequence[str] | None = None,
    verify_pick_phases: bool = False,
    pour_geometry_revision: int | None = None,
    pour_proxy_position_tolerance_m: float | None = None,
    upright_container_picks: Mapping[str, float] | None = None,
) -> tuple[SemanticTaskGraph, TaskProgramBundlePaths]:
    """Write, compose, and provider-free preflight one semantic deployment.

    Args:
        graph: Provider-free semantic task graph. Its provisional fingerprint
            is replaced with the exact composed integration fingerprint.
        prepared_scene: Scene Adapter output, or an explicit FrozenPhysicalInput
            whose scene must also be used before semantic planning. Frozen
            inputs bypass all graph-selected physical adaptation and calibration.
        output_dir: Fresh bundle staging directory.
        robot_profile: Task Engine robot-profile selector. Phase one supports
            only the canonical dual-Franka embodiment.
        max_episodes: Optional Gym episode limit.
        max_episode_steps: Optional Gym step limit. E8 caps this limit at 15000
            without increasing a smaller requested limit.
        twist_support_lift_m: E8 total extra support lift, selected from 1--4 mm.
            Live startup checks must qualify this dimensional candidate.
        twist_grip_depth_m: Explicit E8 whole-assembly depth candidate selected
            after a typed runtime mismatch. None preserves ordinary source sizing.
        twist_contact_feedback: Enable bounded precontact feedback only after
            a qualified exhausted no-contact attempt. Defaults to disabled.
        twist_mass_source: Opt in to the Default-only deployment MassAPI copy
            instead of the Lab mass conversion. This is also selected when the
            unmodified Lab loader does not expose the mass policy capability.
        support_contract_revision: Explicit native E1/E4 placement support contract; None
            preserves legacy bundles, while 1 requires scoped physical evidence.
        pick_contact_revision: Explicit native rigid Pick closing screen. None
            preserves legacy sampling; 1 rejects nonpad-rest-first closure;
            2 also rejects nonpad contact-halo entry before bilateral pad rest.
            3 retains revision 2 closing and screens nominal hand-opening
            geometry for an ordinary adjacent on-Place. 4 combines revision 1
            rest-order closing with diagnostic-only contact halo and revision 3
            opening screening. Opening screening is not physical release
            verification; all revisions retain native acquisition requirements.
            5 replaces whole-object aperture approximation with native open-pad
            rest clearance for local grasps. 6 also searches measured pad-depth
            alternatives within the existing four recovery-path budget. 7 adds
            ordinary precontact scene screening. 8 retains these gates and
            delegates source-proposal qualification to the native screen;
            shared GraspKit defaults and earlier revisions remain unchanged.
        verify_pick_phases: Explicit registered Pick acquisition/lift contract.
            False preserves legacy lowerer declarations and fingerprints.
        pour_geometry_revision: Explicit candidate-dependent tilt/restore geometry.
            Revision 1 preserves observed entry orientation for ordinary E3 with
            no requested orientation or tilt direction. Revision 2 selects an
            arm-side point inside the receiver projection with an explicit proxy
            position reserve. None preserves legacy consumers. Neither revision
            is a liquid-transfer observation contract.
        pour_proxy_position_tolerance_m: Required positive receiver-relative pivot
            position budget for revision 2, shared with its pose-proxy observer.
            Forbidden for other revisions; no tolerance is invented implicitly.
        pick_contact_revisions: Explicit Pick node IDs mapped to native closing
            revisions; registered handover-source Picks require revision 8.
            Unlisted calls retain ``pick_contact_revision`` or legacy behavior.
        world_yaw_free_places: Explicit ordinary inside Place node IDs whose object
            heading may change on real-held planning failure. Omitted preserves
            legacy behavior; hypothetical Pick previews never use this recovery.
        post_handover_places: Explicit receiver relative-on Place node IDs granted
            failure-only goal-conditioned recovery. Targets and grasp stay fixed.
        upright_container_picks: Explicit E3 Pick node IDs mapped to maximum pad-axis
            tilt above the Z-up horizontal plane, in degrees. Omitted preserves
            legacy behavior. The caller declares these as upright-container Picks.

    Returns:
        Final fingerprint-bound graph and all generated paths.

    Raises:
        ValueError: If the robot profile is unsupported or graph/scene
            integration cannot be composed and preflighted.
    """
    selected_graph = validate_semantic_task_graph(graph)
    if pour_geometry_revision is not None:
        if type(pour_geometry_revision) is not int or pour_geometry_revision not in (
            1,
            2,
        ):
            raise ValueError("Unsupported GenSim Pour geometry revision.")
        if pour_geometry_revision == 2 and (
            type(pour_proxy_position_tolerance_m) not in (int, float)
            or not math.isfinite(pour_proxy_position_tolerance_m)
            or pour_proxy_position_tolerance_m <= 0
        ):
            raise ValueError(
                "Pour region proxy position tolerance must be explicit, finite and positive."
            )
        from ._task_program.pour_continuation import supports_ordinary_pour

        if task_candidate is None or not supports_ordinary_pour(
            selected_graph, task_candidate
        ):
            raise ValueError(
                "Pour geometry revision needs an explicit ordinary E3 witness "
                "without requested orientation, axis or direction and the "
                "complete independent Pick/stage/Pour/return/Place recipe."
            )
        if pour_geometry_revision == 2 and any(
            node["task_type"] == "E3"
            and node["call"].get("call_id") == _MOVE_HELD_OBJECT_CALL_ID
            and "reference" in node["call"].get("arguments", {})
            and node["call"]["arguments"]["target"] in selected_graph["targets"]
            for node in selected_graph["nodes"]
        ):
            raise ValueError("Pour region cannot replace an explicit staging target.")
    if pour_geometry_revision != 2 and pour_proxy_position_tolerance_m is not None:
        raise ValueError(
            "Pour proxy position tolerance is owned only by geometry revision 2."
        )
    if type(verify_pick_phases) is not bool:
        raise TypeError("verify_pick_phases must be a boolean.")
    if type(fit_grasp_assets) is not bool:
        raise TypeError("fit_grasp_assets must be a boolean.")
    frozen = prepared_scene if isinstance(prepared_scene, FrozenPhysicalInput) else None
    if frozen is not None:
        frozen.assert_unchanged()
        if str(robot_profile).strip() != frozen.robot_profile:
            raise ValueError("Frozen physical input owns its robot-profile selection.")
        if (
            fit_grasp_assets
            or twist_mass_source
            or twist_grip_depth_m is not None
            or twist_support_lift_m != 0.001
        ):
            raise ValueError(
                "Frozen physical inputs cannot request physical adaptation."
            )
        for key, value in (
            ("max_episodes", max_episodes),
            ("max_episode_steps", max_episode_steps),
        ):
            if value is not None and value != frozen.deployment[key]:
                raise ValueError(f"Frozen physical input owns {key}.")
        prepared_scene = frozen.scene
    unsupported = sorted(
        {node["task_type"] for node in selected_graph["nodes"]}
        - {"E1", "E2", "E3", "E4", "E5", "E6", "E8", "E9"}
    )
    if unsupported:
        raise ValueError(f"Task Engine supports E1-E6 and E8-E9, not {unsupported}.")
    from ._task_program.drawer_binding import prepare_drawer_graph, drawer_routes

    selected_graph = prepare_drawer_graph(selected_graph, prepared_scene)
    articulation_bindings = graph_bindings(selected_graph, prepared_scene)
    from ._task_program.press_binding import (
        graph_routes,
        calibrate_scene,
        prepare_press_scene,
    )

    source_press_routes = graph_routes(selected_graph, prepared_scene)
    from ._task_program.twist_binding import graph_routes as twist_graph_routes

    source_twist_routes = twist_graph_routes(selected_graph, prepared_scene)
    if type(twist_contact_feedback) is not bool:
        raise ValueError("E8 contact feedback must be an explicit boolean.")
    if twist_contact_feedback and not source_twist_routes:
        raise ValueError("Contact feedback requires a standalone E8 deployment.")
    if type(twist_mass_source) is not bool:
        raise ValueError("E8 deployment mass source must be an explicit boolean.")
    if twist_mass_source and not source_twist_routes:
        raise ValueError("Deployment mass source requires a standalone E8 deployment.")
    if source_twist_routes and not twist_mass_source and frozen is None:
        from embodichain.lab.sim.cfg import ArticulationCfg

        twist_mass_source = (
            "body_scale_mass_policy" not in ArticulationCfg.__dataclass_fields__
        )
    if twist_grip_depth_m is not None and not source_twist_routes:
        raise ValueError("Explicit knob depth candidates require an E8 deployment.")
    normalized_profile = str(robot_profile).strip()
    try:
        embodiment_filename = _EMBODIMENT_COMPONENTS[normalized_profile]
    except KeyError as exc:
        raise ValueError(
            "Phase-one semantic bundle generation supports only dual_franka; "
            f"received robot_profile={normalized_profile!r}."
        ) from exc
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    component_root = root / "components"
    task_program_root = root / "task_program"
    component_root.mkdir(parents=True, exist_ok=True)
    task_program_root.mkdir(parents=True, exist_ok=True)

    from embodichain.gen_sim.task_engine.orchestration.scene_assets import (
        normalize_scene_assets,
    )

    project_root = Path(__file__).resolve().parents[3]
    embodiment_source = (
        project_root
        / "embodichain_tasks/configs/components/embodiments"
        / embodiment_filename
    )
    embodiment_payload = (
        deepcopy(frozen.embodiment)
        if frozen is not None
        else load_config(embodiment_source)
    )
    if frozen is None:
        if normalized_profile == "dual_franka_pgi":
            _apply_pgi_experiment_embodiment(embodiment_payload)
        _calibrate_task_gripper_opening(embodiment_payload)
    adaptation = None
    if fit_grasp_assets:
        from .orchestration.grasp_fit import fit_grasp_assets as fit_assets

        generators = embodiment_payload["skill_profile"]["runtime_services"][
            "grasp_pose_generators"
        ]
        openings = {
            resource: float(generators[f"{resource}_eef"]["model"]["max_opening_width"])
            for resource in ("left", "right")
        }
        prepared_scene, adaptation = fit_assets(
            prepared_scene,
            selected_graph,
            openings=openings,
            contact_clearances=_grasp_contact_clearances(
                prepared_scene, embodiment_payload
            ),
        )
    scene = (
        frozen.scene
        if frozen is not None
        else normalize_scene_assets(prepared_scene, root)
    )
    press_adaptation = ()
    if frozen is None:
        scene, press_adaptation = prepare_press_scene(scene, source_press_routes)
    scene_uids = {
        str(item.get("runtime_uid"))
        for item in scene.planner_objects
        if item.get("runtime_uid")
    }
    referenced_uids: set[str] = set()

    def _collect_scene_refs(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                _collect_scene_refs(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                _collect_scene_refs(nested)
        elif isinstance(value, str) and value in scene_uids:
            referenced_uids.add(value)

    _collect_scene_refs(selected_graph)
    eligible_table_children: set[str] = set()
    if source_twist_routes and frozen is None:
        graph_path = scene.source_config_path.with_name("scene_graph.json")
        if graph_path.is_file():
            source_graph = load_config(graph_path)
            nodes = source_graph.get("nodes", ())
            uid_map = scene.uid_map
            support_uids = {
                uid_map.get(node.get("parent_id"), node.get("parent_id"))
                for node in nodes
                if isinstance(node, dict)
            }
            # Missing support relationships do not authorize moving an object.
            eligible_table_children = {
                uid_map.get(node.get("object_id"), node.get("object_id"))
                for node in nodes
                if isinstance(node, dict)
                and uid_map.get(node.get("parent_id"), node.get("parent_id")) == "table"
                and node.get("parent_relation") == "on"
            } - support_uids
    movable_e8_uids = {
        str(item["uid"])
        for item in scene.rigid_objects
        if str(item["uid"]) in eligible_table_children
        and str(item["uid"]) not in referenced_uids
        and str(item["uid"]) != "table"
    }
    twist_scale_audits = []
    twist_adaptation = []
    if source_twist_routes and frozen is None:
        from ._task_program.twist_adaptation import select_twist_scale

        adapted_scales = {}
        for route in source_twist_routes:
            twist_scale, audit = select_twist_scale(
                route, embodiment_payload, grip_depth_candidate=twist_grip_depth_m
            )
            uid = route.binding.object_id
            if uid in adapted_scales:
                if adapted_scales[uid] != twist_scale:
                    raise ValueError(
                        "Repeated E8 operations require compatible gripper dimensions."
                    )
                continue
            adapted_scales[uid] = twist_scale
            twist_scale_audits.append(audit)
            scene, records = prepare_press_scene(
                scene,
                (route,),
                interaction="E8",
                relayout_uids=movable_e8_uids,
                interaction_scale=twist_scale,
                interaction_scale_qualified=twist_scale is not None,
                support_lift_m=twist_support_lift_m,
            )
            twist_adaptation.extend(records)
    if twist_adaptation:
        _write_json(
            root / "twist_adaptation.json",
            {
                "schema_version": "gen_sim.twist-adaptation/v1",
                "policy": "gripper_measured_scale_bounded_height_constrained_relayout",
                "scale_candidate": (
                    twist_scale_audits[0] if len(twist_scale_audits) == 1 else None
                ),
                "scale_candidates": twist_scale_audits,
                "records": twist_adaptation,
            },
        )
    mass_source_routes = ()
    if twist_mass_source:
        from ._task_program.twist_mass_source import create_mass_source_copy
        from ._task_program.twist_binding import enrich_twist_inventory

        articulations = deepcopy(list(scene.articulations))
        planner = deepcopy(list(scene.planner_objects))
        asset_hashes = dict(scene.asset_hashes)
        mass_records = []
        mass_source_routes = twist_graph_routes(selected_graph, scene)
        converted_uids = set()
        for route in mass_source_routes:
            if route.binding.object_id in converted_uids:
                continue
            converted_uids.add(route.binding.object_id)
            art = next(a for a in articulations if a["uid"] == route.binding.object_id)
            groups = [art.get("attrs") or {}]
            groups.extend(
                (group.get("attrs") or {})
                for group in (art.get("link_attrs") or {}).values()
            )
            if any(group.get("mass_props") for group in groups):
                raise ValueError(
                    "E8 mass deployment cannot combine with mass overlays."
                )
            mass_copy = create_mass_source_copy(
                Path(art["fpath"]), root / "mass_sources", tuple(art["body_scale"])
            )
            art["fpath"] = str(mass_copy.runtime)
            art.pop("body_scale_mass_policy", None)
            asset_hashes[art["uid"]] = mass_copy.runtime_sha256
            for item in planner:
                if item.get("runtime_uid") == art["uid"]:
                    item["fpath"] = art["fpath"]
            mass_records.append(
                {
                    "object_id": art["uid"],
                    "origin_source_sha256": mass_copy.source_sha256,
                    "runtime_sha256": mass_copy.runtime_sha256,
                    "mass_lineage_sha256": mass_copy.manifest_sha256,
                    "source_path": str(mass_copy.source.relative_to(root)),
                    "runtime_path": str(mass_copy.runtime.relative_to(root)),
                    "manifest_path": str(mass_copy.manifest.relative_to(root)),
                    "body_scale": mass_copy.scale,
                    "source_edited": False,
                    "conversion_owner": "gen_sim_deployment_source",
                }
            )
        enrich_twist_inventory(planner, articulations)
        scene = replace(
            scene,
            articulations=tuple(articulations),
            planner_objects=tuple(planner),
            asset_hashes=asset_hashes,
        )
        _write_json(
            root / "twist_mass_adaptation.json",
            {
                "schema": "gen_sim.twist-mass-adaptation/v1",
                "records": mass_records,
            },
        )
    press_routes = graph_routes(selected_graph, scene)
    twist_routes = twist_graph_routes(selected_graph, scene)
    if mass_source_routes:
        for original, derived in zip(mass_source_routes, twist_routes, strict=True):
            comparable = replace(
                derived,
                binding=replace(
                    derived.binding,
                    source_sha256=original.binding.source_sha256,
                    mass_lineage_sha256=original.binding.mass_lineage_sha256,
                ),
            )
            if comparable != original:
                raise ValueError(
                    "E8 derived mass source changed its numeric binding or roles."
                )
    if twist_contact_feedback:
        twist_routes = tuple(
            replace(route, feedback_enabled=True, grip_extra_m=0.002)
            for route in twist_routes
        )
    if press_adaptation:
        _write_json(
            root / "press_adaptation.json",
            {
                "schema_version": "gen_sim.press-adaptation/v1",
                "policy": "unit_scale_preserve_support_and_calibration",
                "records": press_adaptation,
            },
        )
    if adaptation is not None:
        provenance = {item["uid"]: item for item in scene.asset_provenance}
        for record in adaptation["records"]:
            runtime = provenance.get(record["object_id"])
            if record["status"] == "scaled" and (
                runtime is None or runtime["status"] not in {"generated", "reused"}
            ):
                raise ValueError(
                    f"Scaled grasp asset {record['object_id']!r} could not be baked."
                )
            if runtime is not None:
                record["runtime_sha256"] = runtime["runtime_sha256"]
                record["runtime_path"] = runtime["runtime_path"]
        _write_json(root / "asset_adaptation.json", adaptation)
    selected_graph = _refine_upright_targets(selected_graph, scene)
    selected_graph = _refine_coordinated_targets(selected_graph, scene)
    scene_objects = {item["runtime_uid"]: item for item in scene.planner_objects}
    for node in selected_graph["nodes"]:
        if node["task_type"] == "E4" and node["call"]["kind"] == "pick":
            call = node["call"]
            source = scene_objects[call["object"]]
            world_axis = _initial_rotation(source) @ np.asarray(
                _longest_local_axis(source)
            )
            node["call"] = {
                "kind": "registered",
                "call_id": (
                    f"{_HORIZONTAL_HANDOVER_SOURCE_PICK}.{call['object']}"
                    if abs(float(world_axis[2])) < 0.5
                    else _HANDOVER_SOURCE_PICK
                ),
                "arguments": {
                    "object": call["object"],
                    "target": f"{node['task_instance_id']}_source_grasp",
                },
                "resources": deepcopy(call["resources"]),
            }
        if node["call"].get("call_id") == _PICK_CALL_ID:
            node["call"]["call_id"] = f"gen_sim.pick.{node['task_instance_id']}"
    selected_graph = _add_handover_staging(selected_graph, scene)
    paths = TaskProgramBundlePaths(
        root=root,
        deployment=root / "task_program_deployment.yaml",
        program=task_program_root / "program.yaml",
        integration=task_program_root / "integration.yaml",
        scene=component_root / "scene.yaml",
        embodiment=component_root / "embodiment.yaml",
        execution_policy=component_root / "execution_policy.yaml",
        semantic_task_graph=root / "semantic_task_graph.json",
        integration_fingerprint=root / "integration_fingerprint.json",
    )
    policy_source = (
        project_root
        / "embodichain_tasks/configs/components/execution_policies"
        / "dual_arm_trajectory_verified.yaml"
    )
    if frozen is None:
        _bind_embodiment_to_scene(embodiment_payload, table_top_z=scene.table_top_z)
        _calibrate_task_gripper_effort(embodiment_payload, scene, selected_graph)
    if articulation_bindings and frozen is None:
        # Calibrate this generated E6 deployment, not shared robot defaults.
        stiffness = embodiment_payload["simulation"]["joint_drive_props"]["stiffness"]
        for resource in embodiment_payload["skill_profile"]["resources"]:
            for endpoint in resource.get("endpoints", []):
                if endpoint["endpoint_id"] == "motion":
                    stiffness[endpoint["control_part"]] = ARM_STIFFNESS
    save_config(paths.embodiment, embodiment_payload)
    drawers = drawer_routes(selected_graph, scene)
    policy_payload = (
        deepcopy(frozen.execution_policy)
        if frozen is not None
        else load_config(policy_source)
    )
    if frozen is None:
        policy_payload["tracking"]["consecutive_acceptances"] = 5
        policy_payload["tracking"]["terminal_settle_timeout"] = 3.0
    save_config(paths.execution_policy, policy_payload)

    program_id = _program_identifier(selected_graph["task_id"])
    scene_contract = f"{program_id}_scene_v1"
    selected_graph = _refine_e3_return_targets(selected_graph, scene)
    from ._task_program.drawer_binding import rewrite_drawer_close_resources

    selected_graph = rewrite_drawer_close_resources(
        selected_graph,
        drawer_bindings=frozenset(
            (route.binding.object_id, route.binding.part_id) for route in drawers
        ),
    )
    stability = _task_stability_payload(
        selected_graph,
        scene,
        embodiment_payload,
        support_contract_revision=support_contract_revision,
        pick_contact_revision=pick_contact_revision,
        pick_contact_revisions=pick_contact_revisions,
        world_yaw_free_places=world_yaw_free_places,
        post_handover_places=post_handover_places,
        upright_container_picks=upright_container_picks,
    )
    drawer_targets = {route.affordance for route in drawers}
    drawer_groups = frozenset(
        node["task_instance_id"]
        for node in selected_graph["nodes"]
        if node["call"].get("inside") in drawer_targets
        or node["call"].get("arguments", {}).get("target") in drawer_targets
    )
    stability["motion_samples"] = motion_sample_declarations(
        selected_graph, drawer_groups=drawer_groups
    )
    stability["cartesian_calls"] = cartesian_call_declarations(
        selected_graph, drawer_groups=drawer_groups
    )
    stability["pour_receivers"] = pour_receiver_declarations(selected_graph)
    pour_returns = pour_return_call_declarations(selected_graph)
    if pour_returns:
        stability["pour_return_calls"] = pour_returns
    held_commands = held_command_declarations(selected_graph)
    if held_commands:
        stability["held_command_sources"] = held_commands
        stability["held_command_revision"] = HELD_COMMAND_REVISION
    (
        stability["staging_calls"],
        stability["staging_retention_sources"],
    ) = _staging_contract_payload(selected_graph, scene, program_id)
    if drawers:
        stability["drawers"] = [route.payload() for route in drawers]
    save_config(
        paths.program,
        _program_payload(
            selected_graph,
            program_id,
            scene,
            stability_presets=stability["presets"],
        ),
    )
    _write_json(
        task_program_root / "constraints.json",
        stability,
    )
    save_config(
        paths.integration,
        _integration_payload(
            selected_graph,
            scene,
            program_id=program_id,
            scene_contract=scene_contract,
            collision_world=policy_payload["motion"]["strategy"] == "motion_gen",
            embodiment=embodiment_payload,
            twist_contact_feedback=twist_contact_feedback,
            verify_pick_phases=verify_pick_phases,
            pour_geometry_revision=pour_geometry_revision,
            pour_proxy_position_tolerance_m=pour_proxy_position_tolerance_m,
        ),
    )
    scene_payload = _scene_payload(scene, program_id=program_id)
    if frozen is not None:
        scene_payload["simulation"] = deepcopy(frozen.scene_simulation)
    else:
        for route in twist_routes:
            art = next(
                a
                for a in scene_payload["simulation"]["articulation"]
                if a["uid"] == route.binding.object_id
            )
            art["asset_physics_mode"] = "overlay"
        press_calibration = calibrate_scene(
            scene_payload, press_routes, source_routes=source_press_routes
        )
        if press_calibration:
            _write_json(root / "press_calibration.json", {"routes": press_calibration})
        bound_uids = {binding.object_id for binding in articulation_bindings.values()}
        for articulation in scene_payload["simulation"]["articulation"]:
            if articulation["uid"] not in bound_uids:
                continue
            articulation.setdefault("asset_physics_mode", "overlay")
            parts = discover_prismatic_parts(articulation)
            link_attrs = articulation.setdefault("link_attrs", {})
            mass_group = "gen_sim_prismatic_link_mass"
            if mass_group in link_attrs:
                raise ValueError(f"Generated E6 articulation reserves {mass_group!r}.")
            link_attrs[mass_group] = {
                "link_names_expr": sorted({part.link for part in parts}),
                "attrs": {"mass_props": {"mass": PRISMATIC_LINK_MASS}},
            }
            drive = articulation.setdefault("joint_drive_props", {})
            if drive.get("drive_type", "none") == "none":
                drive.setdefault("drive_type", "none")
                if "friction" not in drive:
                    drive["friction"] = {part.joint: PASSIVE_FRICTION for part in parts}
    save_config(paths.scene, scene_payload)
    episode_steps = int(max_episode_steps or TaskEnginePlanningCfg().max_episode_steps)
    if twist_routes and len(selected_graph["task_groups"]) == 1 and frozen is None:
        from ._task_program.twist_adaptation import MAX_E8_EPISODE_STEPS

        episode_steps = min(episode_steps, MAX_E8_EPISODE_STEPS)
    deployment_payload = {
        "id": f"GenSimTaskProgram-{program_id}-v1",
        "max_episodes": int(max_episodes or 1),
        "max_episode_steps": episode_steps,
        "num_envs": 1,
        "arena_space": 2.5,
        # Gym configs own the physics backend explicitly.  Generated
        # bundles retain the calibrated DexSim/Default path that the
        # legacy scene attributes and CCD setting were authored for.
        "physics": "default",
        "physics_config": {"enable_ccd": True},
        "env": {
            "sim_steps_per_control": 4,
            "events": {
                "settle_objects_on_reset": {
                    "func": "wait_for_dynamic_objects_to_settle",
                    "mode": "reset",
                    "params": {
                        "entity_cfgs": [
                            {"uid": str(item["uid"])}
                            for item in _task_settle_rigid_objects(
                                selected_graph,
                                scene,
                            )
                        ],
                        "min_steps": 10,
                        # Tall generated objects may need several seconds
                        # to finish a final low-energy roll after import.
                        "max_steps": 600,
                        "check_interval_steps": 2,
                        "required_stable_checks": 3,
                        "timeout_behavior": "raise",
                        # Plan from the measured settled pose. Teleporting
                        # afterward invalidates the settled contact state.
                        "restore_initial_xy": False,
                    },
                }
            },
            "extensions": {},
        },
        "scene": {"component": "components/scene.yaml"},
        "embodiment": {"component": "components/embodiment.yaml"},
        "task_program": {
            "program": "task_program/program.yaml",
            "integration": "task_program/integration.yaml",
            "execution_policy": "components/execution_policy.yaml",
        },
    }
    if frozen is not None:
        deployment_payload.update(deepcopy(frozen.deployment))
    save_config(paths.deployment, deployment_payload)
    if frozen is not None:
        frozen.assert_unchanged()
        _write_json(root / "physical_input.json", frozen.manifest())
        from ._task_program.physical_inputs import verify_physical_input_manifest

        verify_physical_input_manifest(root)

    embodiment = load_config(paths.embodiment)
    deployment = load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=embodiment["skill_profile"],
        base_dir=root,
    )
    fingerprint = deployment.integration.integration_fingerprint
    selected_graph["integration_fingerprint"] = fingerprint
    selected_graph = validate_semantic_task_graph(selected_graph)
    _write_json(paths.semantic_task_graph, selected_graph)
    _write_json(
        paths.integration_fingerprint,
        {
            "schema_version": "semantic_integration_fingerprint/v2",
            "adapter_contract": ADAPTER_CONTRACT,
            "integration_id": deployment.integration_id,
            "integration_fingerprint": fingerprint,
            "registration_fingerprint": bundle_registration_fingerprint(
                deployment.integration.registration
            ),
        },
    )

    program = load_task_program(
        paths.program,
        integration=deployment.selection,
        validation_context=deployment.integration.registration.catalog,
    )
    deployment.integration.registration.catalog.preflight(program)
    validation = None
    if task_spec is not None:
        if task_candidate is None or role_bindings is None or step_lineage is None:
            raise ValueError(
                "TaskSpec generation requires its source witness and expansion lineage."
            )
        from .semantic_graph_validation import write_graph_contract

        validation = write_graph_contract(
            root, selected_graph, task_spec, task_candidate, role_bindings, step_lineage
        )
    from .semantic_graph_html import write_semantic_task_graph_html

    write_semantic_task_graph_html(
        selected_graph,
        root / "semantic_graph.html",
        task_spec=task_spec,
        validation=validation,
    )
    if frozen is not None:
        frozen.assert_unchanged()
        verify_physical_input_manifest(root)
    return selected_graph, paths


def _refine_coordinated_targets(
    graph: SemanticTaskGraph, scene: Any
) -> SemanticTaskGraph:
    """Keep a carried object above its support at the declared terminal pose."""
    result = deepcopy(graph)
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    for node in result["nodes"]:
        call = node["call"]
        if call["kind"] != "registered" or call["call_id"] != _COORDINATED_HOLD_CALL_ID:
            continue
        displacement = list(call["arguments"]["world_displacement"])
        if abs(float(displacement[2])) > 1e-8:
            continue
        source = objects[str(call["arguments"]["object"])]
        bottom, top = _vertical_mesh_bounds(source, axis_aligned=False)
        displacement[2] = max(0.05, min(0.10, 0.5 * (top - bottom) + 0.03))
        call["arguments"]["world_displacement"] = displacement
    return validate_semantic_task_graph(result)


def _refine_e3_return_targets(
    graph: SemanticTaskGraph, scene: Any
) -> SemanticTaskGraph:
    """Ground E3 return poses at the settled tabletop support height."""
    if scene.table_top_z is None:
        return graph
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    result = deepcopy(graph)
    for node in result["nodes"]:
        if node.get("task_type") != "E3" or node["call"].get("kind") != "place":
            continue
        target = node["call"].get("at")
        if not isinstance(target, dict) or target.get("kind") != "target_ref":
            continue
        target_id = str(target["target"])
        values = result["targets"].get(target_id, {}).get("values", ())
        object_id = str(node["call"]["object"])
        source = objects.get(object_id)
        if source is None or len(values) != 1:
            continue
        bottom, _ = _vertical_mesh_bounds(source, axis_aligned=False)
        values[0]["position"][2] = float(scene.table_top_z) - bottom
    return validate_semantic_task_graph(result)


def _task_stability_payload(
    graph: SemanticTaskGraph,
    scene: Any,
    embodiment: dict[str, Any],
    *,
    support_contract_revision: int | None = None,
    pick_contact_revision: int | None = None,
    pick_contact_revisions: Mapping[str, int] | None = None,
    world_yaw_free_places: Sequence[str] | None = None,
    post_handover_places: Sequence[str] | None = None,
    upright_container_picks: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Declare task conditions without adding public validator or runtime types."""
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    if pick_contact_revision is not None and (
        type(pick_contact_revision) is not int
        or pick_contact_revision not in (1, 2, 3, 4, 5, 6, 7, 8)
    ):
        raise ValueError("Unsupported GenSim Pick contact revision.")
    if support_contract_revision is not None and (
        type(support_contract_revision) is not int or support_contract_revision != 1
    ):
        raise ValueError("Unsupported GenSim support contract revision.")
    upright: set[str] = set()
    oriented_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["task_type"] in {"E1", "E4"}
        and node["call"].get("call_id") == _ALIGN_HELD_CALL_ID
        and node["call"]["arguments"].get("target") == "current_object_pose"
    }
    terminal_nodes = {group["node_ids"][-1] for group in graph["task_groups"]}
    horizontal_handover_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["call"]
        .get("call_id", "")
        .startswith(_HORIZONTAL_HANDOVER_SOURCE_PICK + ".")
    }
    routes = {
        _route_key(r): r
        for r in _relative_place_route_payloads(graph, scene, settled=True)
    }
    upright_routes = {
        _route_key(r): r
        for r in _relative_place_route_payloads(
            graph, scene, settled=True, upright_only=True
        )
    }
    coordinated_on_routes = {
        _route_key(r): r
        for r in _relative_place_route_payloads(
            graph, scene, settled=True, coordinated_only=True
        )
    }
    motion_parts = {
        resource["resource_id"]: next(
            endpoint["control_part"]
            for endpoint in resource["endpoints"]
            if endpoint["endpoint_id"] == "motion"
        )
        for resource in embodiment["skill_profile"]["resources"]
    }
    presets: dict[str, Any] = {}
    coordinated_positions: dict[str, list[float]] = {}
    for node in graph["nodes"]:
        call = node["call"]
        entity = call.get("object", call.get("arguments", {}).get("object"))
        if call.get("call_id") not in {
            _COORDINATED_HOLD_CALL_ID,
            _COORDINATED_TRANSPORT_CALL_ID,
        }:
            coordinated_positions.pop(entity, None)
        if (
            call["kind"] == "hand_over"
            and node["id"] in terminal_nodes
            and node["task_instance_id"] in horizontal_handover_groups
        ):
            object_id = str(call["object"])
            axis = _longest_local_axis(objects[object_id])
            world_axis = (
                np.asarray([0.0, 0.0, 1.0])
                if object_id in upright
                else _initial_rotation(objects[object_id]) @ np.asarray(axis)
            )
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold",
                "entity": object_id,
                "local_axis": axis,
                "world_axis": world_axis.tolist(),
                "minimum_alignment": math.cos(math.pi / 18.0),
                "motion_parts": [motion_parts[call["resources"]["destination"]]],
            }
        if call["kind"] == "place" and node.get("task_instance_id") in oriented_groups:
            object_id = str(call["object"])
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "upright",
                "entity": object_id,
                "local_axis": _longest_local_axis(objects[object_id]),
            }
        if call["kind"] != "registered":
            continue
        arguments = call["arguments"]
        if call["call_id"] in {_AXIS_ALIGN_CALL_ID, _ALIGN_HELD_CALL_ID} or (
            node.get("task_type") == "E2"
            and call["call_id"] == _MOVE_HELD_OBJECT_CALL_ID
        ):
            upright.add(str(arguments["object"]))
        if call["call_id"] in {
            _PLACE_RELATIVE_CALL_ID,
            _STACK_PLACE_CALL_ID,
            _UPRIGHT_PLACE_CALL_ID,
        }:
            object_id = str(arguments["object"])
            reference_id = str(arguments["reference"])
            if (
                node["task_type"] == "E2"
                or node.get("task_instance_id") in oriented_groups
                or (
                    object_id in upright
                    and reference_id == "table"
                    and arguments["relation"] in {"on", "above"}
                )
            ):
                selected_routes = (
                    upright_routes
                    if call["call_id"] == _UPRIGHT_PLACE_CALL_ID
                    else routes
                )
                route = _lookup_route(
                    selected_routes,
                    node,
                    object_id,
                    reference_id,
                    arguments["relation"],
                )
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "upright",
                    "entity": object_id,
                    "local_axis": _longest_local_axis(objects[object_id]),
                    "reference": reference_id,
                    "displacement": route["world_displacement"],
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
            elif (
                arguments["relation"] in {"on", "above"}
                and object_id in upright
                and reference_id in upright
            ):
                bottom, _ = _vertical_mesh_bounds(objects[object_id], axis_aligned=True)
                _, top = _vertical_mesh_bounds(objects[reference_id], axis_aligned=True)
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "stack",
                    "entity": object_id,
                    "reference": reference_id,
                    "local_axis": _longest_local_axis(objects[object_id]),
                    "reference_axis": _longest_local_axis(objects[reference_id]),
                    "object_bottom": bottom,
                    "reference_top": top,
                    "reference_half_extents": [
                        max(
                            0.001,
                            _horizontal_half_extent(
                                objects[reference_id],
                                world_axis=axis,
                                axis_aligned=True,
                            )
                            - 0.002,
                        )
                        for axis in (0, 1)
                    ],
                    "minimum_alignment": math.cos(math.pi / 18.0),
                }
            else:
                route = _lookup_route(
                    routes,
                    node,
                    object_id,
                    reference_id,
                    arguments["relation"],
                )
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "placement",
                    "entity": object_id,
                    "reference": route.get("reference_link_id", reference_id),
                    "displacement": route["world_displacement"],
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
                if node["task_type"] == "E1" and arguments["relation"] == "on":
                    presets[f"gen_sim.{node['id']}.stable"].update(
                        kind="supported_placement",
                    )
                    if (
                        reference_id == "table"
                        and call["call_id"] == _PLACE_RELATIVE_CALL_ID
                        and set(arguments).issubset(
                            {"object", "reference", "relation", "stage_id"}
                        )
                    ):
                        presets[f"gen_sim.{node['id']}.stable"] = {
                            "kind": "supported_region",
                            "entity": object_id,
                            "reference": reference_id,
                        }
        elif (
            call["call_id"] == _ALIGN_HELD_CALL_ID
            and node["task_type"] == "E4"
            and node["id"] in terminal_nodes
        ):
            object_id = str(arguments["object"])
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold",
                "entity": object_id,
                "local_axis": _longest_local_axis(objects[object_id]),
                "motion_parts": [motion_parts[call["resources"]["primary"]]],
            }
        elif call["call_id"] in {
            _COORDINATED_HOLD_CALL_ID,
            _COORDINATED_TRANSPORT_CALL_ID,
        }:
            object_id = str(arguments["object"])
            if "reference" in arguments:
                reference_id = str(arguments["reference"])
                route = _lookup_route(
                    coordinated_on_routes,
                    node,
                    object_id,
                    reference_id,
                    arguments["relation"],
                )
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "placement",
                    "entity": object_id,
                    "reference": reference_id,
                    "displacement": route["world_displacement"],
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
                coordinated_positions[object_id] = [
                    value + delta
                    for value, delta in zip(
                        _position(objects[reference_id]),
                        route["world_displacement"],
                        strict=True,
                    )
                ]
                continue
            position = coordinated_positions.get(
                object_id, list(_position(objects[object_id]))
            )
            support = objects[object_id].get("attributes", {}).get("final_support", {})
            if (
                object_id not in coordinated_positions
                and support.get("parent_uid") == "table"
                and scene.table_top_z is not None
            ):
                # The skill offsets the live resting pose, not the import hover gap.
                bottom, _ = _vertical_mesh_bounds(
                    objects[object_id], axis_aligned=False
                )
                position[2] = float(scene.table_top_z) - bottom
            displacement = arguments["world_displacement"]
            position = [position[i] + float(displacement[i]) for i in range(3)]
            coordinated_positions[object_id] = position
            holding = call["call_id"] == _COORDINATED_HOLD_CALL_ID
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold" if holding else "placement",
                "entity": object_id,
                "target_position": position,
                "motion_parts": (
                    [
                        motion_parts[call["resources"][slot]]
                        for slot in ("left", "right")
                    ]
                    if holding
                    else []
                ),
                "position_tolerance": 0.05 if holding else _RELATIVE_POSITION_TOLERANCE,
            }
    if support_contract_revision == 1:
        from ._task_program.drawer_binding import inside_parts

        for node in graph["nodes"]:
            if node["task_type"] not in {"E1", "E4"}:
                continue
            call = node["call"]
            args = call.get("arguments", {})
            obj = call.get("object", args.get("object"))
            ref = None
            if call["kind"] == "place" and "inside" in call:
                ref, parsed_obj, part = inside_parts(call["inside"])
                if parsed_obj != obj or part:
                    continue
            elif call.get("call_id") in {
                _PLACE_RELATIVE_CALL_ID,
                _UPRIGHT_PLACE_CALL_ID,
                _STACK_PLACE_CALL_ID,
            } and args.get("relation") in {"on", "inside"}:
                ref = args.get("reference")
            if ref not in objects or objects[ref].get("role") == "articulation":
                continue
            key = f"gen_sim.{node['id']}.stable"
            cfg = dict(presets.get(key, {"kind": "supported_region", "entity": obj}))
            if cfg["kind"] in {"placement", "upright"}:
                cfg["kind"] = "supported_region"
            cfg.update(reference=ref, support_contract_revision=1)
            presets[key] = cfg
    result = {
        "schema_version": "gen_sim_task_constraints/v1",
        "presets": presets,
        "adaptive_pick": any(
            n["call"]["kind"] == "pick" and not n["call"].get("grasp")
            for n in graph["nodes"]
        ),
        "pick_purposes": {
            n["id"]: ("pour" if n["task_type"] == "E3" else "ordinary")
            for n in graph["nodes"]
            if n["call"]["kind"] == "pick" and not n["call"].get("grasp")
        },
    }
    if support_contract_revision is not None:
        result["support_contract_revision"] = support_contract_revision
    if pick_contact_revision is not None:
        result["pick_contact_revision"] = pick_contact_revision
    if pick_contact_revisions is not None:
        from ._task_program.adaptive_grasp import (
            contact_revision_declarations,
            is_handover_source_call_id,
        )

        if not isinstance(pick_contact_revisions, Mapping):
            raise ValueError("Pick contact revisions must be an explicit mapping.")
        revisions = contact_revision_declarations(dict(pick_contact_revisions))
        nodes = {n["id"]: n for n in graph["nodes"]}
        if any(
            name not in nodes
            or not (
                nodes[name]["call"]["kind"] == "pick"
                or revision == 8
                and nodes[name]["task_type"] == "E4"
                and nodes[name]["call"]["kind"] == "registered"
                and is_handover_source_call_id(nodes[name]["call"].get("call_id"))
            )
            for name, revision in revisions.items()
        ):
            raise ValueError(
                "Pick contact revision must name explicit Pick nodes or native8 E4 source Picks."
            )
        if revisions:
            result["pick_contact_revisions"] = revisions
    if world_yaw_free_places is not None:
        from ._task_program.place_yaw import declarations

        names = declarations(world_yaw_free_places)
        nodes = {n["id"]: n for n in graph["nodes"]}
        if any(
            name not in nodes
            or nodes[name]["call"]["kind"] != "place"
            or not nodes[name]["call"].get("inside")
            for name in names
        ):
            raise ValueError("World yaw freedom must name explicit inside Place nodes.")
        if names:
            result["world_yaw_free_places"] = list(names)
    if post_handover_places is not None:
        from ._task_program.place_branch import handover_place_declarations

        names = handover_place_declarations(post_handover_places)
        if names:
            result["post_handover_places"] = list(names)
    if upright_container_picks is not None:
        from ._task_program.pick_closure_quality import declarations

        if not isinstance(upright_container_picks, Mapping):
            raise ValueError("upright_container_picks must be an explicit mapping.")
        quality = declarations(dict(upright_container_picks))
        nodes = {n["id"]: n for n in graph["nodes"]}
        if quality and pick_contact_revision is None:
            raise ValueError(
                "Upright quality requires an explicit native Pick contact revision."
            )
        if any(
            name not in nodes
            or nodes[name]["call"]["kind"] != "pick"
            or nodes[name]["task_type"] != "E3"
            for name in quality
        ):
            raise ValueError("Upright quality must name explicit E3 Pick nodes.")
        if quality:
            result["upright_container_picks"] = {
                name: policy.max_tilt_degrees for name, policy in quality.items()
            }
    return result


def _program_payload(
    graph: SemanticTaskGraph,
    program_id: str,
    scene: Any | None = None,
    *,
    stability_presets: set[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    axis_by_object = (
        {}
        if scene is None
        else {
            str(item["runtime_uid"]): _longest_local_axis(item)
            for item in scene.planner_objects
        }
    )
    relative_routes = (
        {}
        if scene is None
        else {
            _route_key(route): route
            for route in _relative_place_route_payloads(graph, scene, settled=True)
        }
    )
    upright_routes = (
        {}
        if scene is None
        else {
            _route_key(route): route
            for route in _relative_place_route_payloads(
                graph, scene, settled=True, upright_only=True
            )
        }
    )
    articulated_references = {
        str(item["runtime_uid"])
        for item in (() if scene is None else scene.planner_objects)
        if item.get("role") == "articulation"
    }
    items = [
        _program_node(
            node,
            relative_routes=(
                upright_routes
                if node["call"].get("call_id") == _UPRIGHT_PLACE_CALL_ID
                else relative_routes
            ),
            axis_by_object=axis_by_object,
            task_stability=f"gen_sim.{node['id']}.stable"
            in (stability_presets or set()),
            task_stability_kind=(
                stability_presets.get(f"gen_sim.{node['id']}.stable", {}).get("kind")
                if isinstance(stability_presets, Mapping)
                and stability_presets.get(f"gen_sim.{node['id']}.stable", {}).get(
                    "entity"
                )
                == node["call"].get("arguments", {}).get("object")
                and stability_presets.get(f"gen_sim.{node['id']}.stable", {}).get(
                    "reference"
                )
                == node["call"].get("arguments", {}).get("reference")
                else None
            ),
            articulated_reference=node["call"].get("arguments", {}).get("reference")
            in articulated_references,
        )
        for node in graph["nodes"]
    ]
    by_name = {item["name"]: item for item in items}
    if isinstance(stability_presets, Mapping):
        for item in items:
            key = f"gen_sim.{item['name']}.stable"
            if stability_presets.get(key, {}).get("support_contract_revision") == 1:
                # The strong service owns this uninterrupted terminal window.
                item["post"] = [
                    p for p in item.get("post", ()) if p.get("preset") == key
                ]
    articulation_bindings = graph_bindings(graph, scene)
    for group in graph["task_groups"]:
        if group.get("task_type") == "E8":
            from ._task_program.twist_binding import graph_routes as twist_graph_routes

            route = next(
                r
                for r in twist_graph_routes(graph, scene)
                if r.operation_id == group["id"] or not r.operation_id
            )
            for index, node_id in enumerate(group["node_ids"]):
                phase = (
                    "ready"
                    if index == 0
                    else "turned" if index >= len(group["node_ids"]) - 2 else "chunk"
                )
                by_name[node_id]["post"] = [
                    {
                        "kind": "wait_stable",
                        "entity": route.binding.object_id,
                        "preset": route.preset(phase),
                    }
                ]
            continue
        if group.get("task_type") == "E9":
            from ._task_program.press_binding import graph_routes

            route = next(
                r
                for r in graph_routes(graph, scene)
                if r.operation_id == group["id"] or not r.operation_id
            )
            for index, node_id in enumerate(group["node_ids"]):
                by_name[node_id]["post"] = [
                    {
                        "kind": "wait_stable",
                        "entity": route.binding.object_id,
                        "preset": route.preset("ready" if index == 0 else "pressed"),
                    }
                ]
            continue
        if group.get("task_type") == "E6":
            first = next(n for n in graph["nodes"] if n["id"] == group["node_ids"][0])
            args = first["call"]["arguments"]
            key = (
                args["object"]
                if "part" not in args
                else f"{args['object']}::{args['part']}"
            )
            binding = articulation_bindings[key]
            target, tolerance = binding.target(args["state"]), binding.tolerance(
                args["state"]
            )
            for node_id in group["node_ids"]:
                by_name[node_id]["post"] = [
                    {
                        "kind": "wait_stable",
                        "entity": binding.object_id,
                        "preset": preset_id(binding, args["state"]),
                    }
                ]
                by_name[node_id]["validators"] = [
                    {
                        "kind": "articulation_joint_position",
                        "articulation": binding.object_id,
                        "joint": binding.joint,
                        "minimum_position": target - tolerance,
                        "maximum_position": target + tolerance,
                    }
                ]
            continue
        terminal = by_name[group["node_ids"][-1]]
        for node_id in reversed(group["node_ids"]):
            accepted = by_name[node_id]
            policies = [
                policy
                for policy in accepted.get("post", ())
                if policy["preset"].startswith("gen_sim.")
            ]
            if not policies:
                policies = list(accepted.get("post", ()))
            validators = accepted.get("validators", ())
            if not policies and not validators:
                continue
            if accepted is not terminal:
                remaining = group["node_ids"][group["node_ids"].index(node_id) + 1 :]
                cleanup_calls = [by_name[key]["steps"]["call"] for key in remaining]
                placement = accepted["steps"]["call"]
                if (
                    (
                        placement.get("kind") == "place"
                        or placement.get("call_id")
                        in {
                            _PLACE_RELATIVE_CALL_ID,
                            _UPRIGHT_PLACE_CALL_ID,
                            _STACK_PLACE_CALL_ID,
                        }
                    )
                    and (
                        any(
                            call.get("call_id") == _CLEAR_RELEASED_CALL_ID
                            for call in cleanup_calls
                        )
                        or isinstance(stability_presets, Mapping)
                        and stability_presets.get(f"gen_sim.{node_id}.stable", {}).get(
                            "support_contract_revision"
                        )
                        == 1
                    )
                    and all(
                        call.get("kind") == "registered"
                        and call.get("call_id")
                        in {_CLEAR_RELEASED_CALL_ID, _PARK_CALL_ID}
                        for call in cleanup_calls
                    )
                ):
                    # Release effects remain mandatory; final pose/stability
                    # checks must not prevent the declared clearance sequence.
                    policies = accepted.pop("post", [])
                    validators = accepted.pop("validators", [])
                # Cleanup is still physical work; it can invalidate placement.
                terminal.setdefault("post", []).extend(deepcopy(policies))
                terminal.setdefault("validators", []).extend(deepcopy(validators))
            break
    # Later recipes can disturb an earlier drawer. Check each joint's last
    # requested state again at the end, including after non-articulation work.
    final_joints: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        for validator in item.get("validators", ()):
            if validator["kind"] == "articulation_joint_position":
                final_joints[(validator["articulation"], validator["joint"])] = (
                    validator
                )
    if final_joints:
        terminal_validators = items[-1].setdefault("validators", [])
        for validator in final_joints.values():
            if validator not in terminal_validators:
                terminal_validators.append(deepcopy(validator))
    return {
        "program_id": program_id,
        "targets": deepcopy(graph["targets"]),
        "program": {"kind": "sequence", "items": items},
    }


def _staging_contract_payload(
    graph: SemanticTaskGraph,
    scene: Any | None,
    program_id: str,
) -> tuple[list[str], list[list[str]]]:
    """Emit explicit E2 staging ownership from graph dependencies.

    The runtime must consume this declaration; it must not infer a predecessor
    by scanning adjacent compiled segments.  Only a direct dependency with the
    same object/resource is eligible for retention authority.
    """
    scene_rigid_objects = () if scene is None else scene.rigid_objects
    rigid_ids = {
        str(item.get("runtime_uid", item.get("uid")))
        for item in scene_rigid_objects
        if item.get("runtime_uid", item.get("uid")) is not None
    }
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    node_positions = {
        str(node["id"]): index for index, node in enumerate(graph["nodes"])
    }
    staging_calls: list[str] = []
    retention_sources: list[list[str]] = []
    for node in graph["nodes"]:
        call = node.get("call", {})
        if (
            call.get("kind") != "registered"
            or call.get("call_id") != "gen_sim.align_held"
        ):
            continue
        args = dict(call.get("arguments", {}))
        entity = args.get("object")
        resource = dict(call.get("resources", {})).get("primary")
        if (
            entity not in rigid_ids
            or args.get("preserve_yaw") is not False
            or args.get("target") == "current_object_pose"
            or not isinstance(resource, str)
        ):
            continue
        node_position = node_positions[node["id"]]
        staging_id = (
            f"{program_id}/{program_id}:segment:$.program.items[{node_position}]:0"
        )
        staging_calls.append(staging_id)
        predecessors = [nodes[str(dep)] for dep in node.get("depends_on", ())]
        eligible = []
        for predecessor in predecessors:
            predecessor_call = predecessor.get("call", {})
            predecessor_args = dict(predecessor_call.get("arguments", {}))
            predecessor_resource = dict(predecessor_call.get("resources", {})).get(
                "primary"
            )
            predecessor_entity = predecessor_call.get(
                "object", predecessor_args.get("object")
            )
            kind = predecessor_call.get("kind")
            call_id = predecessor_call.get("call_id")
            if (
                predecessor_entity == entity
                and predecessor_resource == resource
                and (
                    kind == "pick"
                    or call_id == "gen_sim.align_held"
                    or (
                        isinstance(call_id, str) and call_id.startswith("gen_sim.pick.")
                    )
                )
            ):
                eligible.append(predecessor)
        if len(eligible) == 1:
            predecessor_position = node_positions[eligible[0]["id"]]
            predecessor_id = (
                f"{program_id}/{program_id}:segment:$.program.items["
                f"{predecessor_position}]:0"
            )
            predecessor_call = eligible[0].get("call", {})
            predecessor_call_id = predecessor_call.get("call_id")
            if predecessor_call.get("kind") == "pick" and not predecessor_call_id:
                predecessor_call_id = "pick"
            if isinstance(predecessor_call_id, str) and predecessor_call_id:
                retention_sources.append(
                    [
                        staging_id,
                        predecessor_id,
                        predecessor_call_id,
                        entity,
                        resource,
                    ]
                )
    return staging_calls, retention_sources


def _program_node(
    node: dict[str, Any],
    *,
    relative_routes: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
    axis_by_object: dict[str, list[float]] | None = None,
    task_stability: bool = False,
    task_stability_kind: str | None = None,
    articulated_reference: bool = False,
) -> dict[str, Any]:
    """Materialize one task node as one canonical runtime segment."""
    call = deepcopy(node["call"])
    if call["kind"] == "registered" and call["call_id"] in {
        _PLACE_RELATIVE_CALL_ID,
        _UPRIGHT_PLACE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        arguments = call["arguments"]
        stage_key = _route_key_for_node(
            node,
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )
        # Keep the historical three-argument program shape when a placement
        # occurs only once.  A stage selector is needed only when the same
        # object/reference/relation tuple has multiple occurrences; this also
        # keeps old consumers and serialized fixtures byte-for-byte stable.
        repeated_route = False
        if relative_routes is not None and stage_key in relative_routes:
            repeated_route = (
                sum(
                    1
                    for key in relative_routes
                    if len(key) == 4 and key[1:] == stage_key[1:]
                )
                > 1
            )
        if repeated_route:
            arguments["stage_id"] = stage_key[0]
    segment: dict[str, Any] = {
        "kind": "segment",
        "name": str(node["id"]),
        "steps": {"kind": "invoke", "call": call},
    }
    settle_entities: list[str] = []
    settle_preset = "rigid_object"
    if call["kind"] == "place":
        inside = call.get("inside")
        if inside is not None:
            from ._task_program.drawer_binding import inside_parts

            inside_parts(str(inside))
            settle_preset = "contained_rigid_object"
        settle_entities.append(str(call["object"]))
    elif call["kind"] == "registered" and call["call_id"] in {
        _COORDINATED_TRANSPORT_CALL_ID,
        _PLACE_RELATIVE_CALL_ID,
        _UPRIGHT_PLACE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        if call["call_id"] == _COORDINATED_TRANSPORT_CALL_ID:
            settle_preset = "transported_rigid_object"
        settle_entities.append(str(call["arguments"]["object"]))
    if settle_entities:
        segment["post"] = [
            {
                "kind": "wait_stable",
                "entity": settle_entity,
                "preset": settle_preset,
            }
            for settle_entity in dict.fromkeys(settle_entities)
        ]
    if call["kind"] == "registered" and call["call_id"] in {
        _PLACE_RELATIVE_CALL_ID,
        _UPRIGHT_PLACE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        arguments = call["arguments"]
        try:
            route = _lookup_route(
                relative_routes or {},
                node,
                str(arguments["object"]),
                str(arguments["reference"]),
                str(arguments["relation"]),
            )
        except KeyError as exc:
            raise ValueError(
                "Relative placement has no generated route for "
                f"{_route_key_for_node(node, str(arguments['object']), str(arguments['reference']), str(arguments['relation']))!r}."
            ) from exc
        if articulated_reference:
            # The shared validator is rigid-only. The GenSim post-policy uses
            # the same displacement/tolerance and observes either native root.
            if not task_stability:
                raise ValueError(
                    "Articulated references require task stability acceptance."
                )
        elif not (
            task_stability
            and task_stability_kind == "supported_region"
            and node.get("task_type") == "E1"
            and call["call_id"] == _PLACE_RELATIVE_CALL_ID
            and arguments.get("reference") == "table"
            and arguments.get("relation") == "on"
            and set(arguments).issubset({"object", "reference", "relation", "stage_id"})
        ):
            segment["validators"] = [
                {
                    "kind": "object_near_relative_target",
                    "object": route["object_id"],
                    "reference": route["reference_entity_id"],
                    "displacement": deepcopy(route["world_displacement"]),
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
            ]
    if (
        not task_stability
        and node.get("task_type") == "E2"
        and (
            call["kind"] == "place"
            or (
                call["kind"] == "registered"
                and call["call_id"] in {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID}
            )
        )
    ):
        raise ValueError("E2 release requires a task-owned upright stability preset.")
    if node.get("task_type") == "E3" and call["kind"] == "place":
        target = call.get("at")
        if type(target) is not dict or set(target) != {"kind", "target"}:
            raise ValueError("Generated E3 Place requires one exact return target.")
        segment["validators"] = [
            {
                "kind": "object_near_target",
                "object": str(call["object"]),
                "target": str(target["target"]),
                "position_tolerance": 0.06,
            }
        ]
    if task_stability:
        arguments = call.get("arguments", {})
        entity = call.get("object", arguments.get("object"))
        segment.setdefault("post", []).append(
            {
                "kind": "wait_stable",
                "entity": entity,
                "preset": f"gen_sim.{node['id']}.stable",
            }
        )
    return segment


def _integration_payload(
    graph: SemanticTaskGraph,
    scene: Any,
    *,
    program_id: str,
    scene_contract: str,
    collision_world: bool = False,
    embodiment: dict[str, Any] | None = None,
    twist_contact_feedback: bool = False,
    verify_pick_phases: bool = False,
    pour_geometry_revision: int | None = None,
    pour_proxy_position_tolerance_m: float | None = None,
) -> dict[str, Any]:
    scene_objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    articulation_bindings = graph_bindings(graph, scene)
    from ._task_program.press_binding import graph_routes, PREPARE_CALL, PRESS_CALL

    press_routes = graph_routes(graph, scene)
    from ._task_program.twist_binding import (
        graph_routes as twist_graph_routes,
        PREPARE_CALL as TWIST_PREPARE_CALL,
        TWIST_CALL,
    )

    twist_routes = twist_graph_routes(graph, scene)
    if twist_contact_feedback:
        twist_routes = tuple(
            replace(route, feedback_enabled=True, grip_extra_m=0.002)
            for route in twist_routes
        )
    from ._task_program.drawer_binding import drawer_routes, inside_parts

    drawers = {route.affordance: route for route in drawer_routes(graph, scene)}
    from ._task_program.spatial_reference import graph_references

    spatial_references = graph_references(graph, scene)
    drawer_poses: dict[str, list[float]] = {}
    if drawers:
        from ._task_program.drawer_geometry import placement_pose
        from .scene.articulation_geometry import _root_pose, read_articulation_geometry

        configs = {str(cfg["uid"]): cfg for cfg in scene.articulations}
        for route in drawers.values():
            cfg = configs[route.binding.object_id]
            local_link = (
                read_articulation_geometry(cfg["fpath"])
                .link_poses[route.binding.link]
                .copy()
            )
            local_link[:3, 3] *= route.binding.scale
            link = _root_pose(cfg) @ local_link
            source = scene_objects[route.object_id]
            target = placement_pose(
                _mesh_vertices(source),
                _root_pose(source)[:3, :3],
                link,
                np.asarray(route.lower),
                np.asarray(route.upper),
            )
            drawer_poses[route.affordance] = (
                (np.linalg.inv(link) @ target).reshape(-1).tolist()
            )
    referenced_objects: set[str] = set()
    inside_routes: list[tuple[str, str, str, str, dict[str, np.ndarray]]] = []
    on_routes: list[tuple[str, str, str]] = []
    coordinated_routes: list[tuple[str, str, tuple[float, float, float]]] = []
    coordinated_on_routes = {
        _route_key(r): r
        for r in _relative_place_route_payloads(graph, scene, coordinated_only=True)
    }
    coordinated_on_lowerer_routes: list[dict[str, Any]] = []
    coordinated_hold_routes: list[tuple[str, str, tuple[float, float, float]]] = []
    move_held_routes: list[dict[str, Any]] = []
    drawer_transport_routes: list[dict[str, Any]] = []
    pour_staging_routes: list[dict[str, Any]] = []
    upright_move_objects: set[str] = set()
    horizontal_handover_objects: set[str] = set()
    pick_routes: dict[str, list[dict[str, Any]]] = {}
    default_pick_options = {
        "kind": "pick_up",
        # Generic built-in Pick is intentionally neutral.  A later E2 or
        # HandOver must not rewrite the policy of an earlier Pick by scanning
        # the whole graph; constrained calls receive their own aliases below.
        "pick_object_part": "center",
        "pre_grasp_distance": 0.15,
        "lift_height": _DEFAULT_PICK_LIFT_HEIGHT,
        "approach_alignment_max_angle": 0.10,
        # Robotiq travels 0.7 rad with a 2 rad/s limit at 25 Hz; include
        # enough intervals for closure, with the same budget as release.
        "hand_interp_steps": 12,
        "grasp_settle_steps": 0,
        "grasp_commit_fraction": 1.0,
    }
    pick_options: dict[str, dict[str, Any]] = {}
    axis_align_objects: set[str] = set()
    pour_objects: set[str] = set()
    relative_lowerer_routes = _relative_place_route_payloads(graph, scene)
    upright_lowerer_routes = _relative_place_route_payloads(
        graph, scene, upright_only=True
    )
    supported_upright_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["call"].get("call_id") == _UPRIGHT_PLACE_CALL_ID
    }
    has_relative_place = False
    has_park_call = False
    has_articulation_park_call = False
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] in {"pick", "place", "hand_over"}:
            referenced_objects.add(str(call["object"]))
        if call["kind"] == "place" and "inside" in call:
            affordance = str(call["inside"])
            container_id, object_id, _ = inside_parts(affordance)
            if affordance not in drawers:
                referenced_objects.add(container_id)
                inside_routes.append(
                    (
                        affordance,
                        container_id,
                        object_id,
                        str(call.get("resources", {}).get("primary", "left")),
                        _planned_object_rotations(graph, before_node_id=node["id"]),
                    )
                )
        if call["kind"] == "place" and "on" in call:
            affordance = str(call["on"])
            parts = affordance.split("__")
            if len(parts) != 3 or parts[0] != "on":
                raise ValueError(f"Unsupported generated on affordance {affordance!r}.")
            support_id, object_id = parts[1], parts[2]
            referenced_objects.add(support_id)
            on_routes.append((affordance, support_id, object_id))
        if call["kind"] == "registered" and call["call_id"] in {
            "simulation.coordinated_transport",
            _COORDINATED_HOLD_CALL_ID,
        }:
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            referenced_objects.add(object_id)
            if "reference" in arguments:
                reference_id = str(arguments["reference"])
                referenced_objects.add(reference_id)
                route = coordinated_on_routes[
                    _route_key_for_node(
                        node, object_id, reference_id, arguments["relation"]
                    )
                ]
                coordinated_on_lowerer_routes.append(
                    {
                        **_public_route(route, include_stage_id=False),
                        "target_id": str(arguments["target"]),
                    }
                )
                continue
            displacement = tuple(
                float(value)
                for value in arguments.get(
                    "world_displacement",
                    (-_DUAL_FRANKA_COORDINATED_TRANSPORT_DISTANCE, 0.0, 0.0),
                )
            )
            route = (object_id, str(arguments["target"]), displacement)
            if call["call_id"] == _COORDINATED_HOLD_CALL_ID:
                coordinated_hold_routes.append(route)
            else:
                coordinated_routes.append(route)
        elif call["kind"] == "registered" and (
            call["call_id"] == _HANDOVER_SOURCE_PICK
            or call["call_id"].startswith(_HORIZONTAL_HANDOVER_SOURCE_PICK + ".")
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            call_id = call["call_id"]
            referenced_objects.add(object_id)
            pick_routes.setdefault(call_id, []).append(
                {
                    "object_id": object_id,
                    "target_id": str(arguments["target"]),
                }
            )
            pick_options[call_id] = {
                **default_pick_options,
                "pick_object_part": "center",
            }
            horizontal_handover_objects.add(object_id)
        elif call["kind"] == "registered" and (
            call["call_id"] == _PICK_CALL_ID
            or call["call_id"].startswith("gen_sim.pick.")
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            final_target_id = str(arguments["target"])
            final_pose = _single_target_pose(graph, final_target_id)
            table_top = scene.table_top_z
            if table_top is None or not math.isfinite(float(table_top)):
                raise ValueError(
                    "Upright pickup requires the scene's measured table_top_z."
                )
            referenced_objects.add(object_id)
            referenced_objects.add("table")
            call_id = call["call_id"]
            source = scene_objects[object_id]
            source_axis = _longest_local_axis(scene_objects[object_id])
            if object_id in upright_move_objects | axis_align_objects:
                world_axis = np.array([0.0, 0.0, 1.0])
            else:
                from scipy.spatial.transform import Rotation

                world_axis = Rotation.from_euler(
                    "XYZ",
                    scene_objects[object_id].get("init_rot", [0.0, 0.0, 0.0]),
                    degrees=True,
                ).apply(source_axis)
            approach = np.array([0.0, 0.0, -1.0]) - world_axis
            length = np.linalg.norm(approach)
            approach = (
                np.array([0.0, 0.0, -1.0]) if length <= 1e-6 else approach / length
            )
            if (
                node.get("task_type") == "E2"
                or node.get("task_instance_id") in supported_upright_groups
            ):
                # Uprighting does not require an axial entry. A vertical grasp
                # becomes a side grasp after rotation; keep the release filter.
                approach = np.array([0.0, 0.0, -1.0])
            options = {
                "kind": "pick_up",
                "hand_interp_steps": default_pick_options["hand_interp_steps"],
                "pre_grasp_distance": 0.08,
                "grasp_settle_steps": 16,
                "pick_object_part": "center",
                "approach_direction": approach.tolist(),
            }
            if call_id in pick_options and pick_options[call_id] != options:
                raise ValueError(
                    "Distinct Pick policies require task-scoped aliases; regenerate the bundle."
                )
            pick_options[call_id] = options
            pick_routes.setdefault(call_id, []).append(
                {
                    "object_id": object_id,
                    "target_id": final_target_id,
                    # E2 permits all declared world-yaw alternatives. Requiring
                    # the nominal yaw here would reject feasible upright grasps.
                    # Transport checks live reachability; yaw does not change
                    # the fingertip heights used by release-clearance screening.
                    "release_clearance_object_pose": {"kind": "pose", **final_pose},
                    "release_clearance_plane_z": (
                        float(final_pose["position"][2])
                        + _vertical_mesh_bounds(source, axis_aligned=True)[0]
                        - _upright_release_clearance(source)
                        if node.get("task_instance_id") in supported_upright_groups
                        else float(table_top)
                    ),
                    "release_clearance_safety_margin": _E2_RELEASE_SAFETY_MARGIN,
                    "grasp_region": "upper_half",
                }
            )
        elif call["kind"] == "registered" and call["call_id"] == _AXIS_ALIGN_CALL_ID:
            object_id = str(call["arguments"]["object"])
            referenced_objects.add(object_id)
            axis_align_objects.add(object_id)
        elif call["kind"] == "registered" and call["call_id"] in {
            _PLACE_RELATIVE_CALL_ID,
            _UPRIGHT_PLACE_CALL_ID,
            _STACK_PLACE_CALL_ID,
        }:
            arguments = call["arguments"]
            referenced_objects.add(str(arguments["object"]))
            referenced_objects.add(str(arguments["reference"]))
            has_relative_place = True
        elif call["kind"] == "registered" and call["call_id"] in {
            _MOVE_HELD_OBJECT_CALL_ID,
            DRAWER_TRANSPORT_CALL,
        }:
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            target_id = str(arguments["target"])
            referenced_objects.add(object_id)
            if call["call_id"] == DRAWER_TRANSPORT_CALL:
                from ._task_program.drawer_binding import TRANSPORT_CLEARANCE

                if (
                    target_id not in drawers
                    or drawers[target_id].object_id != object_id
                ):
                    raise ValueError(
                        "Drawer transport requires its exact declared object-target route."
                    )
                drawer_transport_routes.append(
                    {
                        "object_id": object_id,
                        "target_id": target_id,
                        "pose": {
                            "kind": "scene_entity",
                            "entity_id": target_id,
                            "relative_pose": _translation_pose(
                                0.0, 0.0, TRANSPORT_CLEARANCE
                            ),
                        },
                    }
                )
            elif node["task_type"] == "E2":
                upright_move_objects.add(object_id)
                pose = _single_target_pose(graph, target_id)
                move_held_routes.append(
                    {
                        "object_id": object_id,
                        "target_id": target_id,
                        "pose": {"kind": "pose", **pose},
                    }
                )
            elif "reference" not in arguments:
                pose = _single_target_pose(graph, target_id)
                move_held_routes.append(
                    {
                        "object_id": object_id,
                        "target_id": target_id,
                        "pose": {"kind": "pose", **pose},
                    }
                )
            else:
                reference = str(arguments["reference"])
                referenced_objects.add(reference)
                if pour_geometry_revision in (1, 2):
                    if node["task_type"] != "E3":
                        raise ValueError("Canonical Pour staging must belong to E3.")
                    pose = {}
                    pour_staging_routes.append(
                        {
                            "object_id": object_id,
                            "target_id": target_id,
                            "receiver_id": reference,
                            "clearance_m": _pour_gripper_radius(
                                embodiment,
                                call.get("resources", {}).get("primary", "left"),
                            )
                            + 0.015,
                        }
                    )
                else:
                    pose, _ = _pour_target_geometry(
                        scene_objects[object_id],
                        scene_objects[reference],
                        embodiment,
                        call.get("resources", {}).get("primary", "left"),
                    )
                move_held_routes.append(
                    {
                        "object_id": object_id,
                        "target_id": target_id,
                        "pose": {
                            "kind": "scene_entity",
                            "entity_id": reference,
                            **pose,
                        },
                    }
                )
        elif call["kind"] == "registered" and call["call_id"] == _POUR_CALL_ID:
            object_id = str(call["arguments"]["object"])
            referenced_objects.add(object_id)
            pour_objects.add(object_id)
        elif call["kind"] == "registered" and call["call_id"] == _ALIGN_HELD_CALL_ID:
            referenced_objects.add(str(call["arguments"]["object"]))
            upright_move_objects.add(str(call["arguments"]["object"]))
        elif call["kind"] == "registered" and call["call_id"] in {
            _STACK_PICK_CALL_ID,
            _CLEAR_RELEASED_CALL_ID,
        }:
            referenced_objects.add(str(call["arguments"]["object"]))
        elif call["kind"] == "registered" and call["call_id"] == "simulation.park":
            has_park_call = True
        elif call["kind"] == "registered" and call["call_id"] == PARK_CALL:
            has_articulation_park_call = True
        elif call["kind"] == "registered" and call["call_id"] in {
            SLIDE_CALL,
            WITHDRAW_CALL,
        }:
            if (
                node["task_type"] != "E6"
                or (
                    call["arguments"]["object"]
                    if "part" not in call["arguments"]
                    else f"{call['arguments']['object']}::{call['arguments']['part']}"
                )
                not in articulation_bindings
            ):
                raise ValueError("Articulation calls require an inspected E6 recipe.")
        elif call["kind"] == "registered" and call["call_id"] in {
            route.call_id(base)
            for route in press_routes
            for base in (PREPARE_CALL, PRESS_CALL)
        }:
            if node["task_type"] != "E9" or not press_routes:
                raise ValueError("Press calls require an inspected E9 recipe.")
        elif call["kind"] == "registered" and call["call_id"] in {
            route.call_id(base)
            for route in twist_routes
            for base in (TWIST_PREPARE_CALL, TWIST_CALL)
        }:
            if node["task_type"] != "E8" or not twist_routes:
                raise ValueError("Twist calls require an inspected E8 recipe.")
        elif call["kind"] == "registered":
            raise ValueError(
                f"Unsupported generated registered call {call['call_id']!r}."
            )

    # Physical obstacle ownership must not depend on the selected planner backend.
    referenced_objects.update(
        uid for uid, source in scene_objects.items() if source["role"] != "articulation"
    )
    rigid_bindings: list[dict[str, Any]] = []
    spatial_articulations: set[str] = set()
    for entity_id in sorted(referenced_objects | {"table"}):
        source = scene_objects.get(entity_id)
        if source is None:
            raise ValueError(
                f"Semantic graph references missing scene entity {entity_id!r}."
            )
        if source["role"] == "articulation":
            if any(container == entity_id for _, container, *_ in inside_routes) or any(
                support == entity_id for _, support, _ in on_routes
            ):
                raise ValueError(
                    "Articulated supports/containers require qualified link bindings."
                )
            spatial_articulations.add(entity_id)
            continue
        affordances: list[dict[str, Any]] = []
        if str(source["role"]) == "rigid_object":
            grasp_affordance = {
                "entity_id": f"{entity_id}_grasp",
                "kind": "antipodal_grasp",
            }
            if (
                entity_id
                in axis_align_objects
                | pour_objects
                | upright_move_objects
                | horizontal_handover_objects
            ):
                grasp_affordance["internal_axis"] = _longest_local_axis(source)
            affordances.append(grasp_affordance)
        occupied_landings: list[tuple[list[float], dict[str, Any]]] = []
        occupied_rotations: dict[str, np.ndarray | None] = {}
        inside_count = sum(
            1 for _, container_id, *_ in inside_routes if container_id == entity_id
        )
        for (
            affordance_id,
            container_id,
            object_id,
            resource,
            rotations,
        ) in inside_routes:
            if container_id != entity_id:
                continue
            lateral = 0.06 if "apple" in object_id else -0.06
            target = _container_target_pose(
                source,
                scene_objects[object_id],
                resource=resource,
                embodiment=embodiment,
                occupied=occupied_landings,
                child_rotation=rotations.get(object_id),
                planned_rotations=occupied_rotations,
            )
            if target is None and inside_count > 1:
                raise ValueError(
                    f"Multiple objects require a measured container floor: {entity_id!r}."
                )
            if target is not None:
                occupied_landings.append((target, scene_objects[object_id]))
                # A reservation keeps its release-time footprint after later alignment.
                occupied_rotations.setdefault(object_id, rotations.get(object_id))
            affordances.append(
                {
                    "entity_id": affordance_id,
                    "kind": "container",
                    "native_name": affordance_id,
                    "object_target_pose": (
                        target
                        if target is not None
                        else _translation_pose(lateral, 0.0, 0.008)
                    ),
                    "release_clearance": 0.12,
                }
            )
        for affordance_id, support_id, object_id in on_routes:
            if support_id != entity_id:
                continue
            child = scene_objects.get(object_id)
            if child is None:
                raise ValueError(
                    f"On relation references missing scene entity {object_id!r}."
                )
            affordances.append(
                {
                    "entity_id": affordance_id,
                    "kind": "support_surface",
                    "native_name": affordance_id,
                    "object_target_pose": _support_target_pose(
                        source,
                        child,
                        axis_aligned=entity_id
                        in (axis_align_objects | upright_move_objects),
                    ),
                }
            )
        rigid_bindings.append(
            {
                "entity_id": entity_id,
                "simulation_uid": entity_id,
                "dynamics": (
                    "dynamic" if str(source["role"]) == "rigid_object" else "static"
                ),
                "semantic_type": str(source.get("category") or entity_id),
                **(
                    {
                        "collision_role": (
                            "dynamic"
                            if str(source["role"]) == "rigid_object"
                            else "static"
                        )
                    }
                    if collision_world
                    else {}
                ),
                "affordances": affordances,
            }
        )

    def coordinated_lowerer_routes(
        routes: list[tuple[str, str, tuple[float, float, float]]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "object_id": object_id,
                "target_id": target_id,
                "world_displacement": list(displacement),
            }
            for object_id, target_id, displacement in routes
        ]

    lowerer_routes = (
        coordinated_lowerer_routes(coordinated_routes) + coordinated_on_lowerer_routes
    )
    hold_lowerer_routes = coordinated_lowerer_routes(coordinated_hold_routes)
    if len(pour_objects) > 1:
        raise ValueError("One generated bundle currently supports one Pour object.")
    from ._task_program.collision_bindings import additional_collision_links

    physical_links = additional_collision_links(
        getattr(scene, "articulations", ()),
        (
            *[r.payload() for r in spatial_references.values()],
            *[
                {
                    "entity_id": r.link_id,
                    "articulation_id": r.binding.object_id,
                    "native_link_name": r.binding.link,
                }
                for r in (*press_routes, *twist_routes)
            ],
            *[
                {
                    "entity_id": b.link_id,
                    "articulation_id": b.object_id,
                    "native_link_name": b.link,
                }
                for b in articulation_bindings.values()
            ],
        ),
    )
    return {
        "integration_id": f"{program_id}_integration_v1",
        "program_id": program_id,
        "requires": {
            "scene_contract": scene_contract,
            "embodiment_contract": "dual_arm_parallel_gripper",
        },
        "scene_binding": {
            "contract_id": scene_contract,
            "registry_id": f"{program_id}_scene_registry",
            "rigid_objects": rigid_bindings,
            "articulations": [
                {"entity_id": uid, "simulation_uid": uid}
                for uid in sorted(
                    {b.object_id for b in articulation_bindings.values()}
                    | spatial_articulations
                    | {r.binding.object_id for r in press_routes}
                    | {r.binding.object_id for r in twist_routes}
                    | {str(cfg["uid"]) for cfg in getattr(scene, "articulations", ())}
                )
            ],
            "links": [
                *physical_links,
                *[
                    r.payload()
                    for r in spatial_references.values()
                    if r.link_id
                    not in (
                        {b.link_id for b in articulation_bindings.values()}
                        | {r.link_id for r in (*press_routes, *twist_routes)}
                    )
                ],
                *[
                    {
                        "entity_id": r.link_id,
                        "articulation_id": r.binding.object_id,
                        "native_link_name": r.binding.link,
                    }
                    for r in {
                        r.link_id: r for r in (*press_routes, *twist_routes)
                    }.values()
                ],
                *[
                    {
                        "entity_id": b.link_id,
                        "articulation_id": b.object_id,
                        "native_link_name": b.link,
                        **(
                            {
                                "affordances": [
                                    {
                                        "entity_id": r.affordance,
                                        "kind": "container",
                                        "native_name": r.affordance,
                                        "object_target_pose": drawer_poses[
                                            r.affordance
                                        ],
                                        "release_clearance": 0.0,
                                    }
                                    for r in drawers.values()
                                    if r.binding == b
                                ]
                            }
                            if any(r.binding == b for r in drawers.values())
                            else {}
                        ),
                    }
                    for b in articulation_bindings.values()
                ],
            ],
        },
        "profile": {
            "defaults": {
                "pick_up": {"primary": "left"},
                "place": {"primary": "left"},
                "hand_over": {"source": "left", "destination": "right"},
                "axis_align": {"primary": "left"},
                "coordinated_pickment": {"left": "left", "right": "right"},
            },
            "action_options": {
                "pick": default_pick_options,
                "place": {
                    "kind": "place",
                    "hand_interp_steps": 12,
                    "release_settle_steps": 60,
                    "lift_height": 0.05,
                    "cartesian_waypoint_count": 2,
                    "preserve_current_object_orientation": True,
                },
                "hand_over": {
                    "kind": "hand_over",
                    "pre_grasp_distance": 0.15,
                    "lift_height": 0.0,
                    "hand_interp_steps": 16,
                    "hold_steps": 8,
                    "retreat_steps": 36,
                    "retreat_distance": 0.10,
                    "receive_pick_object_part": "center",
                    "release_at_target": False,
                    "arm_selection": "bound",
                },
                **pick_options,
                **(
                    {
                        call_id: {
                            "kind": "place",
                            "hand_interp_steps": 12,
                            "release_settle_steps": 60,
                            "lift_height": 0.10,
                            "cartesian_waypoint_count": 2,
                            "preserve_current_object_orientation": True,
                        }
                        for call_id in (
                            _PLACE_RELATIVE_CALL_ID,
                            *(
                                (_UPRIGHT_PLACE_CALL_ID,)
                                if upright_lowerer_routes
                                else ()
                            ),
                        )
                    }
                    if has_relative_place
                    else {}
                ),
                **(
                    {
                        _MOVE_HELD_OBJECT_CALL_ID: {
                            "kind": "move_held_object",
                        }
                    }
                    if move_held_routes
                    else {}
                ),
                **(
                    {DRAWER_TRANSPORT_CALL: {"kind": "move_held_object"}}
                    if drawer_transport_routes
                    else {}
                ),
                **(
                    {
                        _POUR_CALL_ID: {
                            "kind": "pour",
                            "rotate_angle": -1.0471975512,
                        }
                    }
                    if pour_objects
                    else {}
                ),
                **(
                    {
                        _COORDINATED_HOLD_CALL_ID: {
                            "kind": "coordinated_pickment",
                            "object_motion_keyframes": 8,
                            "pre_grasp_distance": 0.10,
                            "lift_height": 0.08,
                            "hand_interp_steps": 10,
                            "hold_steps": 4,
                            "release": False,
                            "approach_direction": [0.0, 0.0, -1.0],
                            "left_to_right_arm_direction": [0.0, 1.0, 0.0],
                            "middle_empty_ratio": 0.4,
                        }
                    }
                    if hold_lowerer_routes
                    else {}
                ),
                **(
                    {
                        _AXIS_ALIGN_CALL_ID: {
                            "kind": "axis_align",
                            "pre_grasp_distance": 0.15,
                            "lift_height": 0.16,
                            "hand_interp_steps": 12,
                            "grasp_settle_steps": 0,
                            "grasp_commit_fraction": 1.0,
                            "target_axis": [0.0, 0.0, 1.0],
                        }
                    }
                    if axis_align_objects
                    else {}
                ),
                **(
                    {"simulation.park": {"kind": "move_joints"}}
                    if has_park_call
                    else {}
                ),
                **(
                    {PARK_CALL: {"kind": "move_joints"}}
                    if has_articulation_park_call
                    else {}
                ),
                **(
                    {
                        SLIDE_CALL: {
                            "kind": "slide",
                            "approach_distance": 0.10,
                            "hand_interp_steps": 12,
                            "preshape_fraction": PRESHAPE_FRACTION,
                            "release_retreat_distance": RELEASE_RETREAT_DISTANCE,
                            "approach_along_grasp_axis": True,
                        },
                        WITHDRAW_CALL: {"kind": "move_end_effector"},
                    }
                    if articulation_bindings
                    else {}
                ),
                **(
                    {
                        "simulation.coordinated_transport": {
                            "kind": "coordinated_pickment",
                            "object_motion_keyframes": 8,
                            "pre_grasp_distance": 0.10,
                            "lift_height": 0.08,
                            "hand_interp_steps": 10,
                            "hold_steps": 4,
                            "release": True,
                            "release_steps": 10,
                            "retreat_distance": _COORDINATED_DETACH_DISTANCE
                            + _COORDINATED_RETREAT_MARGIN,
                            "retreat_steps": 12,
                            "approach_direction": [0.0, 0.0, -1.0],
                            "left_to_right_arm_direction": [0.0, 1.0, 0.0],
                            "middle_empty_ratio": 0.4,
                        }
                    }
                    if lowerer_routes
                    else {}
                ),
            },
            # Effect truth stays in Semantic Skill.  The generated task
            # planner only selects the canonical built-in monitor for the
            # curated effectful calls; it never computes a held relation or
            # synthesizes a successful postcondition itself.
            # HandOver recovery derives a curated Pick even when the original
            # acquisition uses a registered GenSim alias outside the Pick graph.
            "effect_monitors": {
                semantic_id: {
                    "monitor_id": "builtin.composite_effect",
                    "revision": "1",
                    "params": {
                        "consecutive_samples": 3,
                        # Generated cube/apple meshes admit equivalent grasp
                        # orientations.  Translation and the gripper/contact
                        # clause remain strict physical checks; orientation
                        # is intentionally relaxed for this calibrated scene.
                        "attached_translation_threshold": (
                            0.02
                            if semantic_id
                            in {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID}
                            else 0.06
                        ),
                        "attached_rotation_threshold": 3.0,
                        "detached_translation_threshold": (
                            0.03
                            if semantic_id
                            in {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID}
                            else (
                                _COORDINATED_DETACH_DISTANCE
                                if semantic_id
                                in {
                                    _COORDINATED_TRANSPORT_CALL_ID,
                                    _COORDINATED_HOLD_CALL_ID,
                                }
                                else 0.08
                            )
                        ),
                        "detached_rotation_threshold": 3.141592653589793,
                    },
                }
                for semantic_id in (
                    "pick",
                    "place",
                    "hand_over",
                    "simulation.coordinated_transport",
                    _COORDINATED_HOLD_CALL_ID,
                    _AXIS_ALIGN_CALL_ID,
                    _MOVE_HELD_OBJECT_CALL_ID,
                    *((_POUR_CALL_ID,) if pour_geometry_revision in (1, 2) else ()),
                    *pick_routes,
                    _PLACE_RELATIVE_CALL_ID,
                    _UPRIGHT_PLACE_CALL_ID,
                )
                if any(
                    node["call"].get("kind") == semantic_id
                    or node["call"].get("call_id") == semantic_id
                    or (
                        semantic_id == "pick"
                        and node["call"].get("kind") == "hand_over"
                    )
                    for node in graph["nodes"]
                )
            },
            "grounding_providers": {"hand_over": "simulation.configured_handover_pose"},
        },
        "runtime_services": {
            "handover_pose_providers": [
                {
                    "kind": "configured_pose",
                    # Keep the exchange above the tray rim.  This is a
                    # semantic object-space staging target; the Atomic Action
                    # derives all arm/EEF poses from it at runtime.
                    "final_position": [
                        0.0,
                        -0.08,
                        _handover_position_z(scene.table_top_z),
                    ],
                    # Task Program poses use EmbodiChain's public ``xyzw``
                    # quaternion order.  The former generated bundle wrote
                    # this 90-degree x rotation as ``wxyz``; keep the same
                    # orientation while crossing into the configured service.
                    "final_quaternion_xyzw": [
                        0.7071067812,
                        0.0,
                        0.0,
                        0.7071067812,
                    ],
                }
            ],
            "registered_semantic_lowerers": [
                *[
                    {"kind": kind, "route": route.payload()}
                    for route in twist_routes
                    for kind in ("twist_prepare", "twist")
                ],
                *[
                    {"kind": kind, "route": route.payload()}
                    for route in press_routes
                    for kind in ("press_prepare", "press")
                ],
                *(
                    [
                        {
                            "kind": kind,
                            "bindings": [
                                b.payload() for b in articulation_bindings.values()
                            ],
                        }
                        for kind in ("articulation_slide", "articulation_withdraw")
                    ]
                    if articulation_bindings
                    else []
                ),
                *[
                    {
                        "kind": "pick",
                        "call_id": call_id,
                        "routes": routes,
                        **({"verify_phases": True} if verify_pick_phases else {}),
                    }
                    for call_id, routes in pick_routes.items()
                ],
                *(
                    [
                        {
                            "kind": "place_relative",
                            "routes": _public_routes(
                                relative_lowerer_routes,
                                graph=graph,
                                call_ids=frozenset(
                                    {_PLACE_RELATIVE_CALL_ID, _STACK_PLACE_CALL_ID}
                                ),
                            ),
                        }
                    ]
                    if relative_lowerer_routes
                    else []
                ),
                *(
                    [
                        {
                            "kind": "place_upright",
                            "routes": _public_routes(
                                upright_lowerer_routes,
                                graph=graph,
                                call_ids=frozenset({_UPRIGHT_PLACE_CALL_ID}),
                            ),
                        }
                    ]
                    if upright_lowerer_routes
                    else []
                ),
                *(
                    [
                        {
                            "kind": (
                                (
                                    "pour_region_transport"
                                    if pour_geometry_revision == 2
                                    else "pour_transport"
                                )
                                if pour_staging_routes
                                else "move_held_object"
                            ),
                            "routes": move_held_routes,
                            **(
                                {
                                    "proxy_position_tolerance_m": pour_proxy_position_tolerance_m
                                }
                                if pour_geometry_revision == 2
                                else {}
                            ),
                            **(
                                {"staging_routes": pour_staging_routes}
                                if pour_staging_routes
                                else {}
                            ),
                        }
                    ]
                    if move_held_routes
                    else []
                ),
                *(
                    [{"kind": "drawer_transport", "routes": drawer_transport_routes}]
                    if drawer_transport_routes
                    else []
                ),
                *(
                    [
                        {
                            "kind": (
                                "retained_pour"
                                if pour_geometry_revision in (1, 2)
                                else "pour"
                            ),
                            "object_id": next(iter(pour_objects)),
                        }
                    ]
                    if pour_objects
                    else []
                ),
                *(
                    [
                        {
                            "kind": "axis_align",
                            "object_ids": sorted(axis_align_objects),
                        }
                    ]
                    if axis_align_objects
                    else []
                ),
                *([{"kind": "park"}] if has_park_call else []),
                *(
                    [{"kind": "articulation_park"}]
                    if has_articulation_park_call
                    else []
                ),
                *(
                    [{"kind": "coordinated_hold", "routes": hold_lowerer_routes}]
                    if hold_lowerer_routes
                    else []
                ),
                *(
                    [{"kind": "coordinated_transport", "routes": lowerer_routes}]
                    if lowerer_routes
                    else []
                ),
            ],
        },
    }


def _scene_payload(scene: Any, *, program_id: str) -> dict[str, Any]:
    def runtime_pose(config: Any) -> dict[str, Any]:
        from scipy.spatial.transform import Rotation

        result = deepcopy(config)
        if result.get("init_local_pose") is None:
            # PreparedScene uses intrinsic XYZ; dictionary decoders infer
            # extrinsic xyz. Serialize the matrix to preserve the planner pose.
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler(
                "XYZ", result.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
            ).as_matrix()
            pose[:3, 3] = result.get("init_pos", [0.0, 0.0, 0.0])
            result["init_local_pose"] = pose.tolist()
        return result

    def runtime_rigid(config: Any) -> dict[str, Any]:
        result = runtime_pose(config)
        max_hulls = int(result.pop("max_convex_hull_num", 1))
        acd_method = str(result.pop("acd_method", "visacd"))
        shape = result.get("shape")
        if isinstance(shape, dict) and shape.get("shape_type") == "Mesh":
            shape.pop("max_convex_hull_num", None)
            shape.pop("acd_method", None)
            shape.setdefault(
                "collision",
                {
                    "approximation": (
                        "convex_hull" if max_hulls == 1 else "convex_decomposition"
                    ),
                    **(
                        {"max_hulls": max_hulls, "acd_method": acd_method}
                        if max_hulls > 1
                        else {}
                    ),
                },
            )
        return result

    articulations = [runtime_pose(value) for value in scene.articulations]
    for articulation in articulations:
        for semantic_only_key in (
            "affordances",
            "attributes",
            "category",
            "description",
            "initial_state",
            "is_articulated",
            "name",
            "proxy_body_scale",
            "proxy_glb_fpath",
            "proxy_init_pos",
        ):
            articulation.pop(semantic_only_key, None)
        fixed_base = articulation.pop("fix_base", None)
        if fixed_base is not None:
            articulation.setdefault("root_props", {})["fixed_base"] = fixed_base
        if articulation.get("joint_drive_props") is not None:
            articulation.setdefault("asset_physics_mode", "overlay")
        suffix = Path(str(articulation.get("fpath", ""))).suffix.lower()
        if suffix in {".usd", ".usda", ".usdc"}:
            # pytorch-kinematics accepts URDF XML only.  These task skills use
            # DexSim's native link poses/geometry and do not need a PK chain.
            articulation["build_pk_chain"] = False
    return {
        "scene_id": f"{program_id}_generated_scene",
        "simulation": {
            "light": {
                "direct": [
                    {
                        "uid": "main_light",
                        "color": [0.6, 0.6, 0.6],
                        "intensity": 30.0,
                        "init_pos": [0.5, 0.0, 3.0],
                    }
                ]
            },
            "background": [runtime_rigid(value) for value in scene.background],
            "rigid_object": [runtime_rigid(value) for value in scene.rigid_objects],
            "rigid_object_group": [],
            "articulation": articulations,
        },
    }


def _task_settle_rigid_objects(
    graph: SemanticTaskGraph,
    scene: Any,
) -> tuple[dict[str, Any], ...]:
    """Return the immutable scene settling set, independent of task suffixes.

    All dynamic rigid bodies share one scene-level reset contract.  Selecting a
    subset from the full task graph made appending an unrelated action change
    the settled initial state of an existing prefix.
    """
    del graph
    return tuple(scene.rigid_objects)


def _add_handover_staging(graph: SemanticTaskGraph, scene: Any) -> SemanticTaskGraph:
    """Propose a table-inward transfer using existing skills after a fresh E4 pick.

    This is a geometric candidate, not a reachability certificate. Normal
    planning and effect gates remain responsible for rejecting invalid motion.
    Previously manipulated objects cannot use source-scene pose predictions.
    """
    from scipy.spatial.transform import Rotation

    result = deepcopy(graph)
    prior = []
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] != "hand_over" or not prior:
            prior.append(node)
            continue
        pick = prior[-1]
        pick_call = pick["call"]
        object_id = call["object"]
        eligible = (
            node["task_type"] == "E4"
            and pick["task_instance_id"] == node["task_instance_id"]
            and (
                pick_call.get("call_id") == _HANDOVER_SOURCE_PICK
                or pick_call.get("call_id", "").startswith(
                    _HORIZONTAL_HANDOVER_SOURCE_PICK + "."
                )
            )
            and pick_call.get("arguments", {}).get("object") == object_id
            and pick_call.get("resources", {}).get("primary")
            == call["resources"]["source"]
            and node["depends_on"] == [pick["id"]]
            and not any(
                earlier["call"].get(
                    "object", earlier["call"].get("arguments", {}).get("object")
                )
                == object_id
                for earlier in prior[:-1]
            )
        )
        prior.append(node)
        if not eligible:
            continue
        source = next(
            item for item in scene.planner_objects if item["runtime_uid"] == object_id
        )
        table = next(item for item in scene.background if item["uid"] == "table")
        if table.get("shape", {}).get("shape_type") == "Cube":
            center = np.asarray(table["init_pos"], dtype=float)[:2]
        else:
            vertices = np.asarray(_mesh_vertices(table))
            world = vertices @ Rotation.from_euler(
                "XYZ", table.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
            ).as_matrix().T + np.asarray(table["init_pos"])
            center = (world[:, :2].min(0) + world[:, :2].max(0)) * 0.5
        position = np.asarray(source["init_pos"], dtype=float).copy()
        position[:2] = (position[:2] + center) * 0.5
        position[2] += _DEFAULT_PICK_LIFT_HEIGHT
        quat = Rotation.from_euler(
            "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_quat()
        staging_id = f"{node['id']}__staging"
        target_id = f"{staging_id}_target"
        if target_id in result["targets"] or any(
            item["id"] == staging_id for item in result["nodes"]
        ):
            raise ValueError(
                "Generated handover staging identity collides with existing content."
            )
        result["targets"][target_id] = {
            "kind": "cyclic_pose",
            "values": [
                {
                    "position": position.tolist(),
                    "quaternion_xyzw": quat.tolist(),
                }
            ],
        }
        stage = deepcopy(node)
        stage["id"] = staging_id
        stage["call"] = {
            "kind": "registered",
            "call_id": _MOVE_HELD_OBJECT_CALL_ID,
            "arguments": {"object": object_id, "target": target_id},
            "resources": {"primary": call["resources"]["source"]},
        }
        index = next(
            i for i, item in enumerate(result["nodes"]) if item["id"] == node["id"]
        )
        result["nodes"][index]["depends_on"] = [staging_id]
        result["nodes"].insert(index, stage)
        group = next(
            item
            for item in result["task_groups"]
            if item["id"] == node["task_instance_id"]
        )
        group["node_ids"].insert(group["node_ids"].index(node["id"]), staging_id)
    return validate_semantic_task_graph(result)


def _grasp_contact_clearances(
    scene: Any, embodiment: dict[str, Any]
) -> dict[str, float]:
    """Reserve both contact envelopes for the Default-backed GenSim deployment."""
    from dexsim.spawn import DexsimCollisionDesc

    default_offset = DexsimCollisionDesc().contact_offset

    def contact_offset(attrs: Any, fallback: float | None) -> float:
        configured = ((attrs or {}).get("collision_props") or {}).get("contact_offset")
        value = fallback if configured is None else configured
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Grasp-fit requires a finite non-negative contact_offset.")
        return float(value)

    simulation = embodiment["simulation"]
    robot_offset = contact_offset(simulation.get("attrs"), default_offset)
    # Conservatively include explicit link overrides without importing a robot
    # or starting a simulator during provider-free bundle generation.
    robot_offset = max(
        [robot_offset]
        + [
            contact_offset(group.get("attrs"), robot_offset)
            for group in (simulation.get("link_attrs") or {}).values()
        ]
    )
    return {
        str(obj["uid"]): robot_offset + contact_offset(obj.get("attrs"), default_offset)
        for obj in scene.rigid_objects
    }


def _calibrate_task_gripper_effort(
    embodiment: dict[str, Any], scene: Any, graph: SemanticTaskGraph
) -> None:
    """Bound qualified <=10 g E1/E2 grips, including light E5 compositions.

    The canonical Robotiq keeps its complete closing range. Unknown/heavier
    payloads and other recipe families retain their declared actuator limits.
    This does not change the imported mechanism or any collision properties.
    """
    picks = []
    blocked_resources: set[str] = set()
    carriers: dict[str, set[str]] = {}
    for node in graph["nodes"]:
        call = node["call"]
        resources = set(call.get("resources", {}).values())
        if node["task_type"] not in {"E1", "E2", "E5"}:
            blocked_resources.update(resources)
            continue
        if node["task_type"] == "E5":
            if call.get("call_id") in {
                _COORDINATED_HOLD_CALL_ID,
                _COORDINATED_TRANSPORT_CALL_ID,
            }:
                for resource in resources:
                    carriers.setdefault(resource, set()).add(
                        call["arguments"]["object"]
                    )
            continue
        if call["kind"] == "pick":
            object_id = call["object"]
        elif call["kind"] == "registered" and (
            call["call_id"] == _PICK_CALL_ID
            or call["call_id"].startswith("gen_sim.pick.")
        ):
            object_id = call["arguments"]["object"]
        else:
            continue
        picks.append((object_id, call.get("resources", {}).get("primary", "left")))
    if not picks:
        return
    objects = {item["uid"]: item for item in scene.rigid_objects}
    limits = embodiment["simulation"]["joint_drive_props"].get("max_effort")
    if not isinstance(limits, dict):
        return
    grasp_parts = {
        resource["resource_id"]: endpoint["control_part"]
        for resource in embodiment["skill_profile"]["resources"]
        for endpoint in resource["endpoints"]
        if endpoint["endpoint_id"] == "grasp"
    }
    for resource in {resource for _, resource in picks}:
        if resource in blocked_resources:
            continue
        payloads = {uid for uid, owner in picks if owner == resource} | carriers.get(
            resource, set()
        )
        masses = [
            objects.get(uid, {}).get("attrs", {}).get("mass_props", {}).get("mass")
            for uid in payloads
        ]
        if any(
            isinstance(mass, bool)
            or not isinstance(mass, (int, float))
            or not math.isfinite(mass)
            or not 0.0 < mass <= 0.01
            for mass in masses
        ):
            continue
        part = grasp_parts[resource]
        if part in limits:
            limits[part] = min(float(limits[part]), 0.5)


def _calibrate_task_gripper_opening(embodiment: dict[str, Any]) -> None:
    """Bound this deployment's grasp proposals by the mounted pad clearance."""
    from ._task_program.twist_adaptation import GRIPPER_COLLISION_CALIBRATIONS

    generators = embodiment["skill_profile"]["runtime_services"][
        "grasp_pose_generators"
    ]
    for generator in generators.values():
        model = generator["model"]
        if model["model_id"] == "robotiq_arg2f_140":
            # At the profile's zero-angle open command, the assembled URDF pad
            # centers are 0.1360346 m apart, with 0.0075 m total pad thickness.
            # Round the resulting 0.1285346 m free gap down, never up.
            model["max_opening_width"] = min(
                float(model["max_opening_width"]),
                GRIPPER_COLLISION_CALIBRATIONS[model["model_id"]][
                    "maximum_opening_width"
                ],
            )


def _bind_embodiment_to_scene(
    embodiment: dict[str, Any],
    *,
    table_top_z: float | None,
) -> None:
    """Bind the generated deployment's robot mount to the current tabletop."""
    if table_top_z is None or not math.isfinite(float(table_top_z)):
        raise ValueError("Dual-Franka deployment requires a derived tabletop height.")
    simulation = embodiment.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Embodiment component has no simulation mapping.")
    # Keep the qualified GenSim tolerance local instead of relaxing shared IK.
    for solver in simulation.get("solver_cfg", {}).values():
        if solver.get("class_type") == "PytorchSolver":
            solver.setdefault("pos_eps", 5e-3)
    init_pos = simulation.get("init_pos")
    if not isinstance(init_pos, list) or len(init_pos) != 3:
        raise ValueError("Embodiment simulation.init_pos must contain three values.")
    simulation["init_pos"] = [
        float(init_pos[0]),
        float(init_pos[1]),
        float(table_top_z) - _DUAL_FRANKA_TABLE_MOUNT_OFFSET,
    ]


def _route_key(route: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return a stage-qualified route key for repeated long-horizon placements."""
    return (
        str(route.get("stage_id", "")),
        str(route["object_id"]),
        str(route["reference_entity_id"]),
        str(route["relation"]),
    )


def _route_key_for_node(
    node: Mapping[str, Any],
    object_id: str,
    reference_id: str,
    relation: str,
) -> tuple[str, str, str, str]:
    """Return the route key for one graph node occurrence."""
    arguments = node.get("call", {}).get("arguments", {})
    if "stage_id" in arguments:
        stage_id = arguments["stage_id"]
        if type(stage_id) is not str or not stage_id or stage_id != stage_id.strip():
            raise ValueError("Placement stage_id must be a non-empty exact identifier.")
    else:
        stage_id = str(node.get("task_instance_id", node.get("id", "")))
    return (stage_id, object_id, reference_id, relation)


def _lookup_route(
    routes: Mapping[tuple[str, ...], Mapping[str, Any]],
    node: Mapping[str, Any],
    object_id: str,
    reference_id: str,
    relation: str,
) -> Mapping[str, Any]:
    """Resolve stage-qualified routes with a legacy three-field fallback."""
    stage_key = _route_key_for_node(node, object_id, reference_id, relation)
    if stage_key in routes:
        return routes[stage_key]
    legacy_key = (object_id, reference_id, relation)
    if legacy_key in routes:
        return routes[legacy_key]
    empty_stage_key = ("", object_id, reference_id, relation)
    if empty_stage_key in routes:
        return routes[empty_stage_key]
    raise KeyError(stage_key)


def _public_route(
    route: Mapping[str, Any], *, include_stage_id: bool = True
) -> dict[str, Any]:
    """Return one route with optional stage occurrence metadata."""
    if include_stage_id:
        return dict(route)
    return {key: value for key, value in route.items() if key != "stage_id"}


def _public_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    graph: SemanticTaskGraph,
    call_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Keep v38 route declarations unless occurrence selection is required."""
    counts = Counter(_route_key(route)[1:] for route in routes)
    explicit_selectors = {
        _route_key_for_node(
            node,
            str(node["call"]["arguments"]["object"]),
            str(node["call"]["arguments"]["reference"]),
            str(node["call"]["arguments"]["relation"]),
        )
        for node in graph["nodes"]
        if node["call"].get("call_id") in call_ids
        and "stage_id" in node["call"]["arguments"]
    }
    return [
        _public_route(
            route,
            include_stage_id=(
                counts[_route_key(route)[1:]] > 1
                or _route_key(route) in explicit_selectors
            ),
        )
        for route in routes
    ]


def _relative_place_route_payloads(
    graph: SemanticTaskGraph,
    scene: Any,
    *,
    settled: bool = False,
    upright_only: bool = False,
    coordinated_only: bool = False,
) -> list[dict[str, Any]]:
    """Project each release using only orientation changes preceding that call."""
    axis_align_objects: set[str] = set()
    scene_objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    from ._task_program.spatial_reference import graph_references

    spatial_references = graph_references(graph, scene)
    actuated_references = {
        node["call"].get("arguments", {}).get("object")
        for node in graph["nodes"]
        if node["call"].get("call_id") == SLIDE_CALL
    }
    routes: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] != "registered":
            continue
        if call["call_id"] in {_AXIS_ALIGN_CALL_ID, _ALIGN_HELD_CALL_ID} or (
            node.get("task_type") == "E2"
            and call["call_id"] == _MOVE_HELD_OBJECT_CALL_ID
        ):
            axis_align_objects.add(str(call["arguments"]["object"]))
        selected_calls = (
            {_COORDINATED_TRANSPORT_CALL_ID}
            if coordinated_only
            else (
                {_UPRIGHT_PLACE_CALL_ID}
                if upright_only
                else {_PLACE_RELATIVE_CALL_ID, _STACK_PLACE_CALL_ID}
            )
        )
        if call["call_id"] not in selected_calls:
            continue
        arguments = call["arguments"]
        if coordinated_only and "reference" not in arguments:
            continue
        if coordinated_only and (
            arguments.get("relation") != "on"
            or arguments["reference"] == arguments["object"]
        ):
            raise ValueError(
                "Coordinated relative placement requires on a distinct rigid support."
            )
        stage_id = _route_key_for_node(
            node,
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )[0]
        selector = (
            stage_id,
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )
        _, object_id, reference_id, relation = selector
        spatial_reference = spatial_references.get(
            (reference_id, arguments.get("reference_part"))
        )
        if reference_id in actuated_references and spatial_reference is None:
            raise ValueError(
                f"Spatial reference {reference_id!r} changes joints in this program; "
                "use qualified link geometry instead of an authored root envelope."
            )
        if scene_objects.get(reference_id, {}).get(
            "role"
        ) == "articulation" and relation in {
            "on",
            "above",
        }:
            raise ValueError(
                "Articulated supports require qualified link bindings; root references support side relations only."
            )
        if reference_id == "table" and relation not in {"on", "above"}:
            raise ValueError(
                "A lateral table-relative placement has no supported landing surface. "
                "Bind the named relation anchor, not the surrounding tabletop region."
            )
        if node.get("task_type") == "E2":
            target_id = f"{node['task_instance_id']}_upright_target"
            target = _single_target_pose(graph, target_id)
            reference_position = _position(scene_objects[reference_id])
            displacement = [
                float(target["position"][index]) - reference_position[index]
                for index in range(3)
            ]
            if settled:
                # Release clearance is a command target, not a resting height.
                displacement[2] -= _upright_release_clearance(scene_objects[object_id])
        else:
            placement_objects = scene_objects
            if reference_id == "table" and relation == "on":
                # The planner view omits fixed-base and joint configuration.
                # Endpoint clearance needs the actual authored mechanisms.
                placement_objects = {
                    **scene_objects,
                    **{
                        str(config["uid"]): {
                            **scene_objects.get(str(config["uid"]), {}),
                            **config,
                            "runtime_uid": str(config["uid"]),
                            "role": "articulation",
                        }
                        for config in getattr(scene, "articulations", ())
                    },
                }
            displacement = _relative_world_displacement(
                relation,
                object_id=object_id,
                reference_id=reference_id,
                scene_objects=(
                    {
                        **placement_objects,
                        reference_id: spatial_reference.planner_object(),
                    }
                    if spatial_reference is not None
                    else placement_objects
                ),
                axis_align_objects=axis_align_objects,
                table_top_z=scene.table_top_z,
            )
            if spatial_reference is not None:
                if scene.table_top_z is None:
                    raise ValueError(
                        "Articulated side placement requires a measured tabletop."
                    )
                bottom, _ = _vertical_mesh_bounds(
                    scene_objects[object_id],
                    axis_aligned=object_id in axis_align_objects,
                )
                displacement[2] = (
                    float(scene.table_top_z)
                    - bottom
                    + _PLACEMENT_CLEARANCE
                    - float(spatial_reference.initial_pose[2, 3])
                )
            if settled and (
                call["call_id"] == _STACK_PLACE_CALL_ID or coordinated_only
            ):
                displacement[2] -= _PLACEMENT_CLEARANCE
        route = {
            "stage_id": stage_id,
            "object_id": object_id,
            "reference_entity_id": reference_id,
            "relation": relation,
            "world_displacement": displacement,
            **(
                {"reference_link_id": spatial_reference.link_id}
                if spatial_reference is not None
                else {}
            ),
            **(
                {"reference_part": spatial_reference.part_id}
                if spatial_reference is not None
                and spatial_reference.part_id is not None
                else {}
            ),
        }
        # Long-horizon programs may place one object/reference pair more than
        # once.  The graph task instance is the occurrence selector; do not
        # collapse routes from different stages.
        if selector in routes and routes[selector] != route:
            raise ValueError(
                f"Repeated placement {selector!r} requires stage-specific routes; "
                f"existing={routes[selector]!r}, new={route!r}."
            )
        routes[selector] = route
    return [routes[selector] for selector in sorted(routes)]


def _relative_world_displacement(
    relation: str,
    *,
    object_id: str,
    reference_id: str,
    scene_objects: dict[str, Any],
    axis_align_objects: set[str],
    table_top_z: float | None,
) -> list[float]:
    """Derive one world-frame semantic relation from trusted scene geometry."""
    try:
        obj = scene_objects[object_id]
        reference = scene_objects[reference_id]
    except KeyError as exc:
        raise ValueError(
            f"Relative placement references missing scene entity {exc.args[0]!r}."
        ) from exc
    if relation in {"on", "above"}:
        object_bottom, _ = _vertical_mesh_bounds(
            obj,
            axis_aligned=object_id in axis_align_objects,
        )
        if reference_id == "table":
            if table_top_z is None:
                raise ValueError(
                    "Relative placement on the table requires a derived tabletop height."
                )
            reference_position = _position(reference)
            reference_top = float(table_top_z) - reference_position[2]
            object_position = _position(obj)
            if relation == "on":
                object_position = _table_landing_position(
                    object_id,
                    scene_objects,
                    axis_aligned=object_id in axis_align_objects,
                    table_top_z=float(table_top_z),
                )
            x_offset = object_position[0] - reference_position[0]
            y_offset = object_position[1] - reference_position[1]
        else:
            _, reference_top = _vertical_mesh_bounds(
                reference,
                axis_aligned=reference_id in axis_align_objects,
            )
            x_offset = 0.0
            y_offset = 0.0
        return [
            x_offset,
            y_offset,
            reference_top - object_bottom + _PLACEMENT_CLEARANCE,
        ]

    diagonal_relations = {
        "front_left_of": ("front_of", "left_of"),
        "front_right_of": ("front_of", "right_of"),
        "back_left_of": ("behind", "left_of"),
        "back_right_of": ("behind", "right_of"),
    }
    if relation in diagonal_relations:
        components = [
            _relative_world_displacement(
                component,
                object_id=object_id,
                reference_id=reference_id,
                scene_objects=scene_objects,
                axis_align_objects=axis_align_objects,
                table_top_z=table_top_z,
            )
            for component in diagonal_relations[relation]
        ]
        return [components[0][0], components[1][1], components[0][2]]

    if table_top_z is None:
        raise ValueError(
            "Lateral placement requires a measured tabletop landing height."
        )
    object_bottom, _ = _vertical_mesh_bounds(
        obj, axis_aligned=object_id in axis_align_objects
    )
    object_support_z = float(table_top_z) - object_bottom
    reference_support_z = _support_origin_z(
        reference,
        axis_aligned=reference_id in axis_align_objects,
        table_top_z=table_top_z,
    )
    displacement = [
        0.0,
        0.0,
        object_support_z - reference_support_z + _PLACEMENT_CLEARANCE,
    ]
    if relation == "right_of":
        displacement[1] = _horizontal_relation_distance(
            obj,
            reference,
            world_axis=1,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_LATERAL_RELATION_DISTANCE,
        )
    elif relation == "left_of":
        displacement[1] = -_horizontal_relation_distance(
            obj,
            reference,
            world_axis=1,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_LATERAL_RELATION_DISTANCE,
            reference_direction=-1,
        )
    elif relation == "front_of":
        displacement[0] = -_horizontal_relation_distance(
            obj,
            reference,
            world_axis=0,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_FRONT_RELATION_DISTANCE,
            reference_direction=-1,
        )
    elif relation == "behind":
        displacement[0] = _horizontal_relation_distance(
            obj,
            reference,
            world_axis=0,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_FRONT_RELATION_DISTANCE,
        )
    else:
        raise ValueError(f"Unsupported relative placement relation {relation!r}.")
    return displacement


def _table_world_vertices(source: dict[str, Any]) -> np.ndarray:
    """Measure a finite native envelope without guessing missing geometry."""
    from .scene.articulation_geometry import _root_pose

    _position(source)
    pose = _root_pose(source)
    scale = np.asarray(source.get("body_scale", [1.0] * 3), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Table landing geometry requires finite positive scale.")
    if source.get("shape", {}).get("shape_type") == "Cube":
        import trimesh

        bounds = _local_bounds(source)
        if bounds is None:
            raise ValueError("Table landing requires measured primitive geometry.")
        points = (
            np.asarray(
                trimesh.creation.box(extents=np.subtract(bounds[1], bounds[0])).vertices
            )
            * scale
        )
    else:
        points = _mesh_vertices(source)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Table landing requires finite mesh geometry.")
    return points @ pose[:3, :3].T + pose[:3, 3]


def _table_obstacle_bounds(source: dict[str, Any]) -> np.ndarray:
    """Bound fixed rigid geometry and the full travel of qualified sliders.

    Expanding the aggregate by every slider's maximum authored displacement
    deliberately overestimates fixed links and independent drawers. Revolute,
    floating and unqualified mechanisms are not assumed static. This envelope
    certifies neither the robot's approach path nor runtime contact.
    """
    points = _table_world_vertices(source)
    bounds = np.stack((points.min(0), points.max(0)))
    if source.get("role") != "articulation":
        return bounds
    from pxr import Gf, UsdPhysics

    from ._task_program.articulation_binding import _fixed_base, _stage
    from .scene.articulation_geometry import read_articulation_geometry

    if not _fixed_base(source):
        raise ValueError("Table landing requires a fixed articulation obstacle base.")
    stage = _stage(source["fpath"])
    joints = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
    ]
    if any(
        not prim.IsA(UsdPhysics.FixedJoint) and not prim.IsA(UsdPhysics.PrismaticJoint)
        for prim in joints
    ):
        raise ValueError(
            "Table landing cannot qualify non-prismatic articulation joints."
        )
    if not any(prim.IsA(UsdPhysics.PrismaticJoint) for prim in joints):
        return bounds
    geometry = read_articulation_geometry(source["fpath"])
    scale = np.asarray(source.get("body_scale", [1.0] * 3), dtype=float)
    if not np.allclose(scale, scale[0], atol=1e-6, rtol=1e-6):
        raise ValueError("Table landing slider geometry requires uniform scale.")
    expansion = np.zeros(3)
    for prim in joints:
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        joint = UsdPhysics.PrismaticJoint(prim)
        parent, child = (
            geometry.link_poses[stage.GetPrimAtPath(targets[0]).GetName()]
            for targets in (
                joint.GetBody0Rel().GetTargets(),
                joint.GetBody1Rel().GetTargets(),
            )
        )
        anchor0 = parent[:3, 3] + parent[:3, :3] @ np.asarray(
            joint.GetLocalPos0Attr().Get(), dtype=float
        )
        anchor1 = child[:3, 3] + child[:3, :3] @ np.asarray(
            joint.GetLocalPos1Attr().Get(), dtype=float
        )
        axis_name = str(joint.GetAxisAttr().Get())
        if axis_name not in {"X", "Y", "Z"}:
            raise ValueError("Table landing slider requires an explicit XYZ axis.")
        quaternion = joint.GetLocalRot1Attr().Get()
        joint_rotation = np.asarray(
            Gf.Matrix3d(
                Gf.Quatd(quaternion.GetReal(), Gf.Vec3d(quaternion.GetImaginary()))
            )
        ).T
        axis = child[:3, :3] @ joint_rotation @ np.eye(3)["XYZ".index(axis_name)]
        limits = (
            np.asarray(
                [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()],
                dtype=float,
            )
            * scale[0]
        )
        if (
            not np.isfinite(axis).all()
            or not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
            or not np.isfinite(limits).all()
            or limits[0] >= limits[1]
        ):
            raise ValueError(
                "Table landing slider requires finite axis and joint limits."
            )
        authored_position = float(np.dot(anchor1 - anchor0, axis)) * scale[0]
        travel = float(np.max(np.abs(limits - authored_position)))
        expansion += np.abs(_initial_rotation(source) @ axis) * travel
    if not np.isfinite(expansion).all():
        raise ValueError("Table landing articulation travel is not finite.")
    return bounds + np.stack((-expansion, expansion))


def _table_landing_position(
    object_id: str,
    scene_objects: dict[str, Any],
    *,
    axis_aligned: bool,
    table_top_z: float,
) -> tuple[float, float, float]:
    """Keep a legal original XY; replace only an occupied bare-table landing.

    The measured triangle union retains holes. Relocations use an origin-centred
    conservative footprint radius; existing legal full-footprint placements do
    not acquire that extra support margin. Static rigid snapshots and qualified
    prismatic full-travel envelopes are endpoint evidence only.
    """
    import trimesh
    from shapely.affinity import translate
    from shapely.geometry import MultiPoint, Point, box
    from shapely.ops import nearest_points, unary_union

    from ._task_program.container_targets import _floor_region

    obj, table = scene_objects[object_id], scene_objects["table"]
    table_points = _table_world_vertices(table)
    if table["shape"].get("shape_type") == "Cube":
        loaded = trimesh.creation.box(extents=table["shape"]["size"])
    else:
        loaded = trimesh.load(str(table["shape"]["fpath"]), force="scene").to_geometry()
    loaded.vertices = table_points
    measured = _floor_region(loaded)
    if (
        measured is None
        or abs(float(table_points[:, 2].max()) - table_top_z) > 1e-4
        or abs(measured[1] - table_top_z) > _PLACEMENT_CLEARANCE
    ):
        raise ValueError("Table landing requires a measured matching planar support.")
    floor = measured[0]
    vertices = _mesh_vertices(obj)
    if axis_aligned:
        major = int(np.argmax(_dominant_local_axis(obj)))
        radius = float(np.linalg.norm(np.delete(vertices, major, axis=1), axis=1).max())
        footprint = box(-radius, -radius, radius, radius)
    else:
        vertices = vertices @ _initial_rotation(obj).T
        footprint = MultiPoint(vertices[:, :2]).convex_hull
    if footprint.is_empty or footprint.area <= 0 or not np.isfinite(vertices).all():
        raise ValueError("Table landing requires a finite full object footprint.")
    position = _position(obj)
    original = translate(footprint, xoff=position[0], yoff=position[1])
    if not floor.buffer(1e-9).covers(original):
        raise ValueError(
            "Original table landing footprint lacks full measured support."
        )
    bottom, top = _vertical_mesh_bounds(obj, axis_aligned=axis_aligned)
    release_top = table_top_z + top - bottom + _PLACEMENT_CLEARANCE
    excluded = []
    footprint_points = np.asarray(footprint.exterior.coords[:-1])
    for uid, obstacle in sorted(scene_objects.items()):
        if uid in {object_id, "table"}:
            continue
        bounds = _table_obstacle_bounds(obstacle)
        if bounds[1, 2] < table_top_z or bounds[0, 2] > release_top:
            continue
        corners = np.asarray(box(*bounds[0, :2], *bounds[1, :2]).exterior.coords[:-1])
        # Minkowski difference: every full payload footprint must clear the
        # measured obstacle envelope, not merely its origin or source support.
        forbidden = MultiPoint(
            (corners[:, None, :] - footprint_points[None, :, :]).reshape(-1, 2)
        ).convex_hull.buffer(_RELATION_CLEARANCE + 1e-6, join_style=2)
        excluded.append(forbidden)
    occupied = unary_union(excluded)
    if occupied.is_empty or not occupied.covers(Point(position[:2])):
        return position
    radius = float(np.linalg.norm(footprint_points, axis=1).max())
    available = floor.buffer(-radius - _RELATION_CLEARANCE).difference(occupied)
    if available.is_empty or available.area <= 1e-8:
        raise ValueError("Table support has no remaining legal landing space.")
    point = nearest_points(available, Point(position[:2]))[0]
    if not floor.covers(translate(footprint, xoff=point.x, yoff=point.y)):
        raise ValueError("Selected table landing lacks full measured support.")
    return float(point.x), float(point.y), position[2]


def _horizontal_relation_distance(
    obj: dict[str, Any],
    reference: dict[str, Any],
    *,
    world_axis: int,
    object_axis_aligned: bool,
    reference_axis_aligned: bool,
    minimum: float,
    reference_direction: int = 1,
) -> float:
    """Return geometry-aware center separation for one planar relation."""
    if reference.get("role") == "articulation":
        # An articulation root need not be at its geometry center. Measure the
        # outward edge in its actual root frame, not half the aggregate span.
        points = _mesh_vertices(reference) @ _initial_rotation(reference).T
        edge = float((points[:, world_axis] * reference_direction).max())
        return max(
            minimum,
            edge
            + _horizontal_half_extent(
                obj, world_axis=world_axis, axis_aligned=object_axis_aligned
            )
            + (
                _AXIS_ALIGNED_RELATION_CLEARANCE
                if object_axis_aligned
                else _RELATION_CLEARANCE
            ),
        )
    return max(
        minimum,
        _horizontal_half_extent(
            obj,
            world_axis=world_axis,
            axis_aligned=object_axis_aligned,
        )
        + _horizontal_half_extent(
            reference,
            world_axis=world_axis,
            axis_aligned=reference_axis_aligned,
        )
        + (
            _AXIS_ALIGNED_RELATION_CLEARANCE
            if object_axis_aligned
            else _RELATION_CLEARANCE
        ),
    )


def _horizontal_half_extent(
    source: dict[str, Any],
    *,
    world_axis: int,
    axis_aligned: bool,
) -> float:
    """Return a conservative horizontal half extent in the intended pose."""
    vertices = _mesh_vertices(source)
    if axis_aligned:
        dominant_axis = int(np.argmax(np.ptp(vertices, axis=0)))
        horizontal_extents = np.delete(np.ptp(vertices, axis=0), dominant_axis)
        return 0.5 * float(horizontal_extents.max())
    world_vertices = vertices @ _initial_rotation(source).T
    return 0.5 * float(np.ptp(world_vertices[:, world_axis]))


def _support_origin_z(
    source: dict[str, Any],
    *,
    axis_aligned: bool,
    table_top_z: float | None,
) -> float:
    """Return the object-origin height when resting on the scene table."""
    if not axis_aligned:
        return _position(source)[2]
    if table_top_z is None:
        raise ValueError("Axis-aligned placement requires a derived tabletop height.")
    bottom, _ = _vertical_mesh_bounds(source, axis_aligned=True)
    return float(table_top_z) - bottom + _PLACEMENT_CLEARANCE


def _dominant_local_axis(source: dict[str, Any]) -> list[float]:
    """Return the unique major mesh axis in DexSim's object-local basis."""
    vertices = _mesh_vertices(source)
    extents = np.ptp(vertices, axis=0)
    order = np.argsort(extents)
    major = int(order[-1])
    if extents[major] <= max(float(extents[order[-2]]) * 1.25, 1.0e-6):
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has no unique dominant "
            "local axis for semantic upright alignment."
        )
    axis = [0.0, 0.0, 0.0]
    axis[major] = 1.0
    return axis


def _pour_target_geometry(
    source: dict[str, Any],
    reference: dict[str, Any],
    embodiment: dict[str, Any] | None,
    resource: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the swept vessel and finger envelope above the receiving rim."""
    from scipy.spatial.transform import Rotation

    vertices = _mesh_vertices(source)
    pivot = (vertices.min(0) + vertices.max(0)) * 0.5
    radius = float(np.linalg.norm(vertices - pivot, axis=1).max())
    upright = np.asarray(_longest_local_axis(source))
    rotation = _initial_rotation(source)
    world_up = rotation @ upright
    vertical = np.array([0.0, 0.0, 1.0])
    cross = np.cross(world_up, vertical)
    sine = np.linalg.norm(cross)
    cosine = np.clip(world_up @ vertical, -1.0, 1.0)
    if sine > 1e-8:
        rotation = (
            Rotation.from_rotvec(cross / sine * math.atan2(sine, cosine)).as_matrix()
            @ rotation
        )
    elif cosine < 0:
        transverse = rotation[:, int(np.argmin(np.abs(upright)))]
        rotation = Rotation.from_rotvec(transverse * math.pi).as_matrix() @ rotation
    _, top = _vertical_mesh_bounds(reference, axis_aligned=False)
    finger_radius = _pour_gripper_radius(embodiment, resource)
    offset = np.array([0.0, 0.0, top + radius + finger_radius + 0.015])
    relative = np.eye(4)
    relative[:3, 3] = rotation.T @ offset - pivot
    return (
        {
            "relative_pose": relative.reshape(-1).tolist(),
            "world_orientation": rotation.reshape(-1).tolist(),
        },
        {"upright_axis": upright.tolist(), "pivot_local": pivot.tolist()},
    )


def _pour_gripper_radius(embodiment: dict[str, Any] | None, resource: str) -> float:
    """Retain the existing nominal hand reserve, not a collision certificate."""
    if embodiment is None:
        raise ValueError("Pour clearance requires the selected gripper geometry.")
    model = embodiment["skill_profile"]["runtime_services"]["grasp_pose_generators"][
        f"{resource}_eef"
    ]["model"]
    return float(
        np.linalg.norm(
            [
                (float(model["max_opening_width"]) + float(model["finger_thickness"]))
                * 0.5,
                float(model["finger_width"]) * 0.5,
                float(model["finger_length"]) * 0.5,
            ]
        )
    )


def _initial_rotation(source: dict[str, Any]) -> np.ndarray:
    """Use the authoritative matrix, or the prepared scene's intrinsic angles."""
    from scipy.spatial.transform import Rotation

    if source.get("init_local_pose") is not None:
        rotation = np.asarray(source["init_local_pose"], dtype=float)[:3, :3]
    else:
        rotation = Rotation.from_euler(
            "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix()
    return rotation


def _initial_local_up(source: dict[str, Any]) -> list[float]:
    """Preserve the source's vertical orientation without fixing its heading."""
    return _initial_rotation(source)[2].tolist()


def _vertical_mesh_bounds(
    source: dict[str, Any],
    *,
    axis_aligned: bool,
) -> tuple[float, float]:
    """Return bottom/top offsets for the intended object orientation."""
    vertices = _mesh_vertices(source)
    if axis_aligned:
        axis = np.asarray(_dominant_local_axis(source), dtype=np.float64)
        heights = vertices @ axis
    else:
        heights = vertices @ np.asarray(_initial_local_up(source))
    return float(heights.min()), float(heights.max())


def _mesh_vertices(source: dict[str, Any]) -> np.ndarray:
    """Load native geometry in the configured entity's runtime root frame."""
    if "_spatial_vertices" in source:
        return source["_spatial_vertices"]
    if source.get("role") == "articulation":
        from .scene.articulation_geometry import read_articulation_geometry

        path = source.get("fpath")
        if not path:
            raise ValueError(
                f"Articulated reference {source.get('runtime_uid')!r} has no native USD asset."
            )
        scale = np.asarray(source.get("body_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(
                "Articulated reference requires a finite positive body_scale."
            )
        return read_articulation_geometry(path).vertices * scale
    shape = source.get("shape")
    if not isinstance(shape, dict) or not shape.get("fpath"):
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has no mesh geometry."
        )
    try:
        import trimesh

        loaded = trimesh.load(str(shape["fpath"]), force="scene")
        geometry = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"Could not inspect scene mesh for {source.get('runtime_uid')!r}: {exc}"
        ) from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not vertices.size:
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has empty mesh geometry."
        )
    # DexSim converts glTF's Y-up vertices into its Z-up object-local basis.
    result = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    scale = np.asarray(source.get("body_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all():
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has invalid body_scale."
        )
    return result * scale


def _position(source: dict[str, Any]) -> tuple[float, float, float]:
    """Return one finite source position."""
    value = source.get("init_pos")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has no three-value init_pos."
        )
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has invalid init_pos."
        )
    return result


def _handover_position_z(table_top_z: float | None) -> float:
    """Place the dual-arm exchange at a scene-relative reachable clearance."""
    if table_top_z is None or not math.isfinite(float(table_top_z)):
        raise ValueError("Hand-over grounding requires a derived tabletop height.")
    return float(table_top_z) + _DUAL_FRANKA_HANDOVER_CLEARANCE


def _translation_pose(x: float, y: float, z: float) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        float(x),
        0.0,
        1.0,
        0.0,
        float(y),
        0.0,
        0.0,
        1.0,
        float(z),
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _program_identifier(task_id: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(task_id).strip().lower()
    ).strip("_")
    if not normalized:
        raise ValueError("task_id does not contain a usable program identifier.")
    return f"gen_sim_{normalized}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _single_target_pose(
    graph: SemanticTaskGraph,
    target_id: str,
) -> dict[str, Any]:
    """Return one exact generated target pose."""
    target = graph["targets"].get(target_id)
    if not isinstance(target, dict):
        raise ValueError(f"Missing generated target {target_id!r}.")
    values = target.get("values")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"Generated target {target_id!r} must contain one pose.")
    pose = values[0]
    if (
        not isinstance(pose, dict)
        or not isinstance(pose.get("position"), list)
        or len(pose["position"]) != 3
        or not isinstance(pose.get("quaternion_xyzw"), list)
        or len(pose["quaternion_xyzw"]) != 4
    ):
        raise ValueError(f"Generated target {target_id!r} has an invalid pose.")
    return pose


def _longest_local_axis(source: dict[str, Any]) -> list[float]:
    """Return the longest axis in the simulator rigid-body local frame."""
    source_axis = _longest_source_axis(source)
    shape = source.get("shape", {})
    path = shape.get("fpath") if isinstance(shape, dict) else None
    if path and Path(str(path)).suffix.lower() in {".glb", ".gltf"}:
        # glTF is Y-up. DexSim converts it to Z-up while importing the render
        # mesh, so semantic axes must follow the same source-to-body mapping.
        x_axis, y_axis, z_axis = source_axis
        return [x_axis, -z_axis, y_axis]
    return source_axis


def _longest_source_axis(source: dict[str, Any]) -> list[float]:
    """Return the longest axis in the source mesh coordinate frame."""
    extents = _local_extents(source)
    if extents is None:
        return [0.0, 0.0, 1.0]
    axis = [0.0, 0.0, 0.0]
    axis[max(range(3), key=extents.__getitem__)] = 1.0
    return axis


def _local_extents(source: dict[str, Any]) -> list[float] | None:
    """Return finite positive local geometry extents when available."""
    bounds = _local_bounds(source)
    if bounds is None:
        return None
    return [maximum - minimum for minimum, maximum in zip(*bounds, strict=True)]


def _local_bounds(source: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    """Return finite local geometry bounds when available."""
    shape = source.get("shape", {})
    if not isinstance(shape, dict):
        return None
    if shape.get("shape_type") == "Cube":
        size = shape.get("size")
        if isinstance(size, list) and len(size) == 3:
            extents = [float(value) for value in size]
            if min(extents) <= 0.0:
                return None
            return (
                [-value / 2.0 for value in extents],
                [value / 2.0 for value in extents],
            )
    path = shape.get("fpath")
    if not path:
        return None
    try:
        import trimesh

        geometry = trimesh.load(str(path), force="scene").to_geometry()
        bounds = [[float(value) for value in row] for row in geometry.bounds]
    except (OSError, TypeError, ValueError):
        return None
    if (
        len(bounds) != 2
        or any(len(row) != 3 for row in bounds)
        or any(maximum <= minimum for minimum, maximum in zip(*bounds, strict=True))
    ):
        return None
    return bounds[0], bounds[1]


def _upright_release_clearance(source: dict[str, Any]) -> float:
    """Return the E2 recipe's drop distance above the final support surface."""
    extents = _local_extents(source)
    if extents is None:
        raise ValueError("Upright release requires finite object extents.")
    ordered_extents = sorted(extents)
    slenderness = ordered_extents[-1] / max(ordered_extents[-2], 1.0e-6)
    return (
        _SLENDER_UPRIGHT_RELEASE_CLEARANCE
        if slenderness >= 2.5
        else _UPRIGHT_RELEASE_CLEARANCE
    )


def _refine_upright_targets(
    graph: SemanticTaskGraph,
    scene: Any,
) -> SemanticTaskGraph:
    """Use normalized mesh origins and the stage's declared supporting surface."""
    selected = deepcopy(graph)
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    table = objects.get("table")
    if table is None:
        return selected
    if scene.table_top_z is None:
        return selected
    table_top = float(scene.table_top_z)
    support_heights: dict[str, float] = {}
    upright_objects: set[str] = set()
    for node in selected["nodes"]:
        call = node["call"]
        if call.get("call_id") in {_ALIGN_HELD_CALL_ID, _AXIS_ALIGN_CALL_ID}:
            upright_objects.add(call["arguments"]["object"])
        if call.get("call_id") == _UPRIGHT_PLACE_CALL_ID:
            support_id = call["arguments"]["reference"]
            support = objects[support_id]
            _, top = _vertical_mesh_bounds(
                support, axis_aligned=support_id in upright_objects
            )
            support_heights[node["task_instance_id"]] = _position(support)[2] + top
    target_routes: dict[str, tuple[str, str]] = {}
    target_heights: dict[str, float] = {}
    upright_staging_targets: set[str] = set()
    transfer_sources = {
        (
            node["task_instance_id"],
            node["depends_on"][0],
            node["call"]["object"],
            node["call"]["resources"]["source"],
        )
        for node in selected["nodes"]
        if node["task_type"] == "E4"
        and node["call"]["kind"] == "hand_over"
        and len(node["depends_on"]) == 1
    }
    for node in selected["nodes"]:
        call = node["call"]
        # A transfer pose is not an upright landing target, including when an
        # already expanded E4 graph is regenerated with a new Pick policy.
        if (
            node.get("task_type") == "E4"
            and call.get("call_id") == _MOVE_HELD_OBJECT_CALL_ID
            and (
                node["task_instance_id"],
                node["id"],
                call.get("arguments", {}).get("object"),
                call.get("resources", {}).get("primary"),
            )
            in transfer_sources
        ):
            continue
        if (
            node.get("task_type") not in {"E1", "E2", "E4"}
            or call.get("kind") != "registered"
            or call.get("call_id")
            not in {
                _MOVE_HELD_OBJECT_CALL_ID,
                _PICK_CALL_ID,
                _ALIGN_HELD_CALL_ID,
                _CLEAR_RELEASED_CALL_ID,
            }
        ):
            continue
        arguments = call.get("arguments", {})
        object_id = str(arguments.get("object", ""))
        target_id = str(arguments.get("target", ""))
        resource = str(call.get("resources", {}).get("primary", ""))
        target_routes[target_id] = (object_id, resource)
        if node.get("task_type") == "E2":
            upright_staging_targets.add(target_id)
        support_height = support_heights.get(node["task_instance_id"], table_top)
        target_heights[target_id] = support_height
        # Place owns the final descent; its release target still needs the
        # same mesh-origin refinement when no separate final Move is emitted.
        target_routes[f"{node['task_instance_id']}_upright_target"] = (
            object_id,
            resource,
        )
        target_heights[f"{node['task_instance_id']}_upright_target"] = support_height
    for target_id, (object_id, resource) in target_routes.items():
        source = objects.get(object_id)
        target = selected["targets"].get(target_id) if target_id is not None else None
        bounds = _local_bounds(source) if source is not None else None
        if bounds is None or not isinstance(target, dict):
            continue
        source_axis = _longest_source_axis(source)
        axis = _longest_local_axis(source)
        axis_index = source_axis.index(1.0)
        local_minimum = bounds[0][axis_index]
        values = target.get("values")
        if not isinstance(values, list) or len(values) != 1:
            continue
        position = values[0].get("position")
        if not isinstance(position, list) or len(position) != 3:
            continue
        clearance = _upright_release_clearance(source)
        position[2] = target_heights[target_id] + clearance - local_minimum
        if target_id.endswith("_upright_staging_target"):
            if target_id in upright_staging_targets:
                # Bound every rotated vertex above the support instead of
                # imposing a fixed lift that can overextend an otherwise valid grasp.
                radius = float(np.linalg.norm(_mesh_vertices(source), axis=1).max())
                position[2] = max(
                    position[2],
                    target_heights[target_id] + radius + _E2_RELEASE_SAFETY_MARGIN,
                )
            else:
                position[2] += _UPRIGHT_STAGING_CLEARANCE
        values[0]["quaternion_xyzw"] = _upright_target_quaternion(
            source,
            axis,
            world_yaw=(math.pi if resource == "right" else 0.0),
        )
    return validate_semantic_task_graph(selected)


def _upright_target_quaternion(
    source: dict[str, Any],
    local_axis: list[float],
    *,
    world_yaw: float,
) -> list[float]:
    """Rotate the current object frame so its selected positive axis is world +Z."""
    import torch

    from embodichain.utils.math import (
        axis_angle_to_rotation_matrix,
        matrix_from_quat,
        quat_from_euler_xyz,
        quat_from_matrix,
    )

    init_rot = source.get("init_rot")
    if not isinstance(init_rot, list) or len(init_rot) != 3:
        raise ValueError(
            f"E2 object {source.get('runtime_uid')!r} requires three-value init_rot."
        )
    angles = torch.deg2rad(torch.tensor(init_rot, dtype=torch.float32))
    quaternion = quat_from_euler_xyz(angles[0], angles[1], angles[2])
    rotation = matrix_from_quat(quaternion)
    axis = torch.tensor(local_axis, dtype=torch.float32)
    source_axis = torch.nn.functional.normalize(rotation @ axis, dim=0)
    target_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    cross = torch.linalg.cross(source_axis, target_axis)
    dot = torch.clamp(torch.dot(source_axis, target_axis), -1.0, 1.0)
    cross_norm = torch.linalg.vector_norm(cross)
    if float(cross_norm) <= 1.0e-6:
        if float(dot) >= 0.0:
            delta = torch.eye(3, dtype=torch.float32)
        else:
            basis = torch.eye(3, dtype=torch.float32)[
                torch.argmin(torch.abs(source_axis))
            ]
            perpendicular = torch.nn.functional.normalize(
                torch.linalg.cross(source_axis, basis), dim=0
            )
            delta = axis_angle_to_rotation_matrix(perpendicular * math.pi)
    else:
        rotation_axis = cross / cross_norm
        angle = torch.atan2(cross_norm, dot)
        delta = axis_angle_to_rotation_matrix(rotation_axis * angle)
    target_rotation = delta @ rotation
    if world_yaw:
        target_rotation = (
            axis_angle_to_rotation_matrix(
                torch.tensor([0.0, 0.0, world_yaw], dtype=torch.float32)
            )
            @ target_rotation
        )
    target_quaternion = quat_from_matrix(target_rotation)
    return [float(value) for value in target_quaternion]


def _container_target_pose(
    container: dict[str, Any],
    child: dict[str, Any],
    *,
    resource: str,
    embodiment: dict[str, Any] | None,
    occupied: Sequence[tuple[list[float], dict[str, Any]]] = (),
    child_rotation: np.ndarray | None = None,
    planned_rotations: Mapping[str, np.ndarray | None] | None = None,
) -> list[float] | None:
    """Ground an arm-side landing on a measured, nearly level container floor."""
    if embodiment is None:
        return None
    import trimesh
    from scipy.spatial.transform import Rotation

    from ._task_program.container_targets import select_container_landing

    simulation = embodiment["simulation"]
    mount = next(
        (
            item
            for item in simulation.get("urdf_cfg", {}).get("components", ())
            if item["component_type"] == f"{resource}_arm"
        ),
        None,
    )
    if mount is None:
        return None

    def pose(source: dict[str, Any]) -> np.ndarray:
        if source.get("init_local_pose") is not None:
            return np.asarray(source["init_local_pose"], dtype=float)
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler(
            "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix()
        result[:3, 3] = _position(source)
        return result

    container_pose = pose(container)
    container_rotation = container_pose[:3, :3]
    if container_rotation[2, 2] < math.cos(math.radians(5.0)):
        return None
    robot_pose = pose(simulation)
    root_world = (
        robot_pose[:3, :3] @ np.asarray(mount["transform"], dtype=float)[:3, 3]
        + robot_pose[:3, 3]
    )
    root_local = container_rotation.T @ (root_world - container_pose[:3, 3])
    mesh = trimesh.load(str(container["shape"]["fpath"]), force="scene").to_geometry()
    mesh.vertices = _mesh_vertices(container)

    def local_vertices(
        item: dict[str, Any], rotation_override: np.ndarray | None = None
    ) -> np.ndarray:
        rotation = (
            pose(item)[:3, :3] if rotation_override is None else rotation_override
        )
        return _mesh_vertices(item) @ (container_rotation.T @ rotation).T

    child_vertices = local_vertices(child, child_rotation)
    planned_rotations = planned_rotations or {}
    occupied_footprints = [
        (
            float(target[3]),
            float(target[7]),
            float(
                np.linalg.norm(
                    local_vertices(
                        previous,
                        planned_rotations.get(str(previous.get("runtime_uid"))),
                    )[:, :2],
                    axis=1,
                ).max()
            ),
        )
        for target, previous in occupied
    ]
    translation = select_container_landing(
        mesh, child_vertices, root_local, occupied=occupied_footprints
    )
    return None if translation is None else _translation_pose(*translation)


def _planned_object_rotations(
    graph: SemanticTaskGraph, *, before_node_id: str | None = None
) -> dict[str, np.ndarray]:
    """Return declared orientations, optionally stopping before one call.

    Bundle geometry must follow explicit GenSim alignment targets rather than
    silently reverting to the source ``init_rot`` after an earlier E2/E4 call.
    This is a planning snapshot only; runtime still observes the released pose.
    """
    from scipy.spatial.transform import Rotation

    result: dict[str, np.ndarray] = {}
    for node in graph["nodes"]:
        if before_node_id is not None and node.get("id") == before_node_id:
            break
        call = node["call"]
        if call.get("kind") != "registered" or call.get("call_id") not in {
            _ALIGN_HELD_CALL_ID,
            _MOVE_HELD_OBJECT_CALL_ID,
        }:
            continue
        arguments = call.get("arguments", {})
        object_id = arguments.get("object")
        target_id = arguments.get("target")
        if not isinstance(object_id, str) or not isinstance(target_id, str):
            continue
        if target_id == "current_object_pose":
            continue
        target = graph["targets"].get(target_id)
        values = target.get("values") if isinstance(target, Mapping) else None
        pose_value = (
            values[0] if isinstance(values, list) and len(values) == 1 else None
        )
        quaternion = (
            pose_value.get("quaternion_xyzw")
            if isinstance(pose_value, Mapping)
            else None
        )
        if isinstance(quaternion, list) and len(quaternion) == 4:
            result[object_id] = Rotation.from_quat(
                np.asarray(quaternion, dtype=float)
            ).as_matrix()
    return result


def _support_target_pose(
    support: dict[str, Any],
    child: dict[str, Any],
    *,
    axis_aligned: bool,
) -> list[float]:
    """Build a conservative object-center target above a support entity."""
    support_extents = _local_extents(support) or [0.1, 0.1, 0.1]
    child_extents = _local_extents(child) or [0.05, 0.05, 0.05]
    child_half_height = max(child_extents) / 2.0
    translation = [0.0, 0.0, 0.0]
    if axis_aligned:
        normal = _longest_local_axis(support)
        support_bounds = _local_bounds(support)
        source_axis = _longest_source_axis(support)
        source_axis_index = source_axis.index(1.0)
        support_top = (
            support_extents[source_axis_index] / 2.0
            if support_bounds is None
            else support_bounds[1][source_axis_index]
        )
        distance = support_top + child_half_height + 0.01
        translation = [component * distance for component in normal]
    else:
        attributes = support.get("attributes", {})
        aabb = (
            attributes.get("final_world_aabb") if isinstance(attributes, dict) else None
        )
        position = support.get("init_pos", [0.0, 0.0, 0.0])
        if isinstance(aabb, dict) and isinstance(aabb.get("max"), list):
            surface_height = float(aabb["max"][2]) - float(position[2])
        else:
            surface_height = support_extents[2] / 2.0
        translation[2] = surface_height + child_half_height + 0.01
    return _translation_pose(*translation)
