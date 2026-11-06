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

"""Task Engine declaration generation over shared configured-runtime owners."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from importlib import metadata, util
from pathlib import Path
from typing import Any
from embodichain.gen_sim.task_spec import TaskSpec
from embodichain.lab.task_program import TaskProgramIntegrationCfg
from embodichain.lab.task_program.integrations import (
    compile_configured_task_program,
    task_program_fingerprint,
    PostPolicyPresetBinding,
)
from embodichain.lab.task_program.integrations.simulation.stability import (
    SupportSensorDeclarationCfg,
)
from .bundle_io import BundleSnapshot
from .bundle_inputs import SOURCE_INPUT_PROVENANCE_SCHEMA
from .bundle_loading import _digest as _source_file_digest, _verify_required_inputs
from .bundle_assembly import (
    TaskProgramBundlePaths,
    prepare_task_program_bundle,
    publish_task_program_bundle,
)
from .physical_input import FrozenPhysicalInput
from .orchestration.source_scene import PreparedScene
from .native_pick_declarations import (
    resolve_native_pick_revisions,
    resolve_native_pick_bindings,
)
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph
from .scene_binding_declarations import resolve_scene_binding_declaration
from .action_profile_declarations import (
    resolve_action_profile_declaration,
    source_pick_option_declaration,
)
from .alignment_declarations import (
    resolve_alignment_pick_declarations,
    resolve_alignment_lowerer_declarations,
    resolve_handover_profile_declarations,
)
from .placement_declarations import resolve_relative_placement_routes
from .stability_declarations import resolve_stability_conditions
from .semantic_program import build_semantic_program_declaration
from .robot_profiles import canonical_robot_profile
from .prismatic_declarations import resolve_prismatic_program_declarations
from .control_declarations import resolve_control_program_declarations
from .continuation_declarations import (
    _place_acquisition_purposes,
    resolve_source_continuation_declarations,
    source_pick_release_pairs,
    resolve_inside_yaw_declarations,
    resolve_measured_rigid_declarations,
    resolve_released_hand_declarations,
)
from .drawer_declarations import resolve_drawer_declarations, bind_drawer_occurrences
from .drawer_graph_declarations import (
    prepare_source_drawer_graph,
    rewrite_source_drawer_close_resources,
)
from .source_graph_declarations import (
    refine_source_manipulation_declarations,
    refine_source_return_declarations,
)
from .pour_declarations import (
    _pour_return_call_declarations,
    resolve_pour_lowerer_declarations,
    supports_ordinary_pour,
)
from .embodiment_declarations import (
    resolve_source_hand_services,
    declare_source_grip_evidence,
    declare_source_opening_bindings,
)

__all__ = [
    "TaskProgramBundlePaths",
    "generate_task_program_bundle",
    "prepare_generated_task_program_bundle",
    "task_program_implementation_revisions",
]


def _program_identifier(task_id: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(task_id).strip().lower()
    ).strip("_")
    if not normalized:
        raise ValueError("task_id does not contain a usable program identifier.")
    return f"gen_sim_{normalized}"


def _verify_source_inputs(inputs: Mapping[str, str]) -> None:
    """Recheck every source-owner file without replacing its original digest."""
    for name, expected in inputs.items():
        path = Path(name)
        if not path.is_file() or _source_file_digest(path) != expected:
            raise ValueError(f"Frozen source input changed: {name}")


def _verify_generated_input_inventory(
    snapshot: BundleSnapshot, *, base_dir: Path
) -> None:
    """Recheck sealed source hashes and actual service/physical dependencies."""
    inputs = snapshot.identity.to_dict()["external_inputs"]
    names = tuple(inputs)
    _verify_source_inputs(inputs)
    required = _verify_required_inputs(snapshot, base_dir=base_dir, names=names)
    _verify_source_inputs(inputs)
    if _verify_required_inputs(snapshot, base_dir=base_dir, names=names) != required:
        raise ValueError(
            "Frozen source runtime dependencies changed during validation."
        )


def task_program_implementation_revisions() -> dict[str, str]:
    """Hash current packaged implementation sources and dependency versions.

    Returns:
        Fresh owner revision mapping suitable for generation and independent
        loading checks. Paths are relative within each official package; run
        outputs and cached bytecode do not enter the implementation identity.
    """
    result = {}
    for name in ("embodichain", "embodichain_robots", "embodichain_tasks"):
        spec = util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(
                f"Selected implementation package {name!r} is unavailable."
            )
        root = Path(spec.origin).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}:
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        result[name + ".source_sha256"] = digest.hexdigest()
    for name in ("dexsim", "torch", "numpy", "scipy", "trimesh", "shapely"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _ordinary_integration(
    graph,
    physical,
    program_id,
    support_revision,
    *,
    embodiment=None,
    hands=(),
    press_extra_distance=None,
    twist_chunk_angles=None,
    pour_geometry_revision=None,
    pour_proxy_position_tolerance_m=None,
    verify_pick_phases=False,
    idle_routes=(),
):
    scene = physical.scene
    embodiment = physical.embodiment if embodiment is None else embodiment
    calls = [node["call"] for node in graph["nodes"]]
    permitted = {
        "simulation.place_relative",
        "simulation.park",
        "gen_sim.articulation_slide",
        "gen_sim.articulation_withdraw",
        "gen_sim.articulation_park",
        "simulation.coordinated_transport",
        "simulation.coordinated_hold",
        "simulation.move_held_object",
        "simulation.pour",
        "simulation.pick",
        "gen_sim.align_held",
        "gen_sim.clear_released",
        "gen_sim.clear_idle_arm",
        "gen_sim.drawer_transport",
    }
    if any(
        node["task_type"] not in {"E1", "E2", "E3", "E4", "E5", "E6", "E8", "E9"}
        or call["kind"] not in {"pick", "place", "hand_over", "registered"}
        or call["kind"] == "registered"
        and call["call_id"] not in permitted
        and not call["call_id"].startswith(
            ("gen_sim.press", "gen_sim.twist", "gen_sim.pick.")
        )
        for node, call in zip(graph["nodes"], calls, strict=True)
    ):
        raise ValueError(
            "This source recipe requires its shared alias/operation declaration resolver before bundle generation."
        )
    prismatic = resolve_prismatic_program_declarations(graph, scene)
    drawers = resolve_drawer_declarations(graph, scene, prismatic)
    controls = resolve_control_program_declarations(
        graph,
        scene,
        robot_uid=embodiment["simulation"]["uid"],
        hands=hands,
        table_uid="table",
        press_extra_distance=press_extra_distance,
        twist_chunk_angles=twist_chunk_angles,
    )
    scene_binding = resolve_scene_binding_declaration(
        graph,
        scene,
        embodiment,
        program_id=program_id,
        link_declarations=controls.link_declarations() + prismatic.link_declarations(),
        container_declarations={
            route.target_id: route.container_declaration() for route in drawers
        },
    )
    resources = {
        item["resource_id"] for item in embodiment["skill_profile"]["resources"]
    }
    if not {"left", "right"} <= resources:
        raise ValueError(
            "The generated source dual-arm recipe requires explicit left/right semantic resources."
        )
    defaults = {
        "pick_up": {"primary": "left"},
        "place": {"primary": "left"},
        "hand_over": {"source": "left", "destination": "right"},
        "axis_align": {"primary": "left"},
        "coordinated_pickment": {"left": "left", "right": "right"},
    }
    pick_options, pick_lowerers = resolve_alignment_pick_declarations(
        graph,
        table_top_z=scene.table_top_z,
        hands=hands,
        default_pick_options=source_pick_option_declaration(),
        source_objects={item["runtime_uid"]: item for item in scene.planner_objects},
    )
    profile = resolve_action_profile_declaration(
        graph,
        default_resources=defaults,
        registered_pick_options=pick_options,
        pour_geometry_revision=pour_geometry_revision,
    )
    routes = resolve_relative_placement_routes(graph, scene)
    counts = {}
    for route in routes:
        key = tuple(
            route[name] for name in ("object_id", "reference_entity_id", "relation")
        )
        counts[key] = counts.get(key, 0) + 1
    routes = [
        {
            key: value
            for key, value in route.items()
            if key != "stage_id"
            or counts[
                tuple(
                    route[name]
                    for name in ("object_id", "reference_entity_id", "relation")
                )
            ]
            > 1
        }
        for route in routes
    ]
    lowerers = prismatic.lowerer_declarations(table_uid="table") + list(
        controls.lowerers
    )
    for lowerer in pick_lowerers:
        lowerer["verify_phases"] = verify_pick_phases
    lowerers.extend(pick_lowerers)
    if idle_routes:
        lowerers.append(
            dict(kind="idle_arm_clearance", routes=deepcopy(list(idle_routes)))
        )
    drawer_targets = {route.target_id: route for route in drawers}
    drawer_transport = []
    for call in calls:
        if call.get("call_id") != "gen_sim.drawer_transport":
            continue
        args = call["arguments"]
        if (
            args["target"] not in drawer_targets
            or drawer_targets[args["target"]].object_id != args["object"]
        ):
            raise ValueError(
                "Drawer transport requires its exact declared payload/container."
            )
        drawer_transport.append(
            dict(
                object_id=args["object"],
                target_id=args["target"],
                pose=dict(
                    kind="scene_entity",
                    entity_id=args["target"],
                    relative_pose=[
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.12,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ),
            )
        )
    if drawer_transport:
        lowerers.append(dict(kind="drawer_transport", routes=drawer_transport))

    lowerers.extend(
        resolve_pour_lowerer_declarations(
            graph,
            {str(item["runtime_uid"]): item for item in scene.planner_objects},
            embodiment,
            hands,
            geometry_revision=pour_geometry_revision,
            proxy_position_tolerance_m=pour_proxy_position_tolerance_m,
        )
    )
    if any(
        lowerer.get("effect_contract") == "terminal_attachment" for lowerer in lowerers
    ):
        profile["action_options"]["simulation.move_held_object"][
            "report_retained_attachment"
        ] = True
    if controls.operations and not prismatic.bindings:
        lowerers.append({"kind": "articulation_park"})
    profile["action_options"].update(controls.action_options)
    coordinated_routes = resolve_relative_placement_routes(
        graph, scene, coordinated_only=True
    )
    by_stage = {
        (
            value["stage_id"],
            value["object_id"],
            value["reference_entity_id"],
            value["relation"],
        ): value
        for value in coordinated_routes
    }
    for identity, kind in (
        ("simulation.coordinated_transport", "coordinated_transport"),
        ("simulation.coordinated_hold", "coordinated_hold"),
    ):
        selected_routes = []
        for node in graph["nodes"]:
            call = node["call"]
            if call.get("call_id") != identity:
                continue
            args = call["arguments"]
            if "reference" in args:
                key = (
                    str(args.get("stage_id", node["task_instance_id"])),
                    str(args["object"]),
                    str(args["reference"]),
                    str(args["relation"]),
                )
                route = by_stage[key]
                selected_routes.append(
                    dict(
                        object_id=args["object"],
                        target_id=args["target"],
                        reference_entity_id=args["reference"],
                        relation=args["relation"],
                        world_displacement=route["world_displacement"],
                    )
                )
            else:
                selected_routes.append(
                    dict(
                        object_id=args["object"],
                        target_id=args["target"],
                        world_displacement=list(
                            args.get("world_displacement", (-0.14, 0.0, 0.0))
                        ),
                    )
                )
        if selected_routes:
            lowerers.append(dict(kind=kind, routes=selected_routes))
    if routes:
        lowerers.append(
            dict(
                kind="place_relative",
                routes=routes,
                grasp_transform_source="observed",
                effect_contract="terminal_release",
            )
        )
    if any(call.get("call_id") == "simulation.park" for call in calls):
        lowerers.append({"kind": "park"})
    motion_parts = {
        item["resource_id"]: next(
            endpoint["control_part"]
            for endpoint in item["endpoints"]
            if endpoint["endpoint_id"] == "motion"
        )
        for item in embodiment["skill_profile"]["resources"]
    }
    conditions = resolve_stability_conditions(
        graph,
        scene,
        motion_parts,
        support_contract_revision=support_revision,
        support_sensor_factory=lambda name: SupportSensorDeclarationCfg(
            uid="gen_sim_support_"
            + name.removeprefix("gen_sim.").removesuffix(".stable")
        ),
    )
    alignment_lowerers = resolve_alignment_lowerer_declarations(
        graph,
        axes={
            cfg.entity: cfg.local_axis
            for cfg in conditions.values()
            if cfg.local_axis is not None
        },
        hands=hands,
    )
    lowerers.extend(alignment_lowerers)
    if any(value["kind"] == "align_held" for value in alignment_lowerers):
        profile["action_options"]["gen_sim.align_held"] = {"kind": "move_held_object"}
        profile["effect_monitors"]["gen_sim.align_held"] = dict(
            monitor_id="builtin.composite_effect",
            revision="1",
            params=dict(
                consecutive_samples=3,
                attached_translation_threshold=0.06,
                attached_rotation_threshold=3.0,
                detached_translation_threshold=0.08,
                detached_rotation_threshold=math.pi,
            ),
        )
    if any(value["kind"] == "clear_released" for value in alignment_lowerers):
        profile["action_options"]["gen_sim.clear_released"] = {
            "kind": "move_end_effector"
        }
    for cfg in conditions.values():
        if cfg.kind == "placement":
            cfg.position_tolerance = 0.10
        if cfg.kind == "supported_placement":
            cfg.position_tolerance = 0.10
            cfg.support_tolerance = 0.03
            cfg.com_tolerance = 0.10
    settles = {
        "rigid_object": dict(
            linear_velocity_threshold=0.03,
            angular_velocity_threshold=0.2,
            min_steps=10,
            max_steps=240,
            check_interval_steps=2,
            required_stable_checks=3,
            measurement="reported_velocity",
        ),
        "contained_rigid_object": dict(
            linear_velocity_threshold=0.03,
            angular_velocity_threshold=1.0,
            min_steps=10,
            max_steps=240,
            check_interval_steps=2,
            required_stable_checks=3,
            measurement="pose_delta",
        ),
        "transported_rigid_object": dict(
            linear_velocity_threshold=0.03,
            angular_velocity_threshold=0.2,
            min_steps=10,
            max_steps=240,
            check_interval_steps=2,
            required_stable_checks=3,
            measurement="pose_delta",
        ),
        "articulation": dict(
            linear_velocity_threshold=0.02,
            angular_velocity_threshold=0.1,
            min_steps=10,
            max_steps=120,
            check_interval_steps=2,
            required_stable_checks=3,
        ),
    }
    settles.update({name: cfg.to_dict() for name, cfg in conditions.items()})
    settles.update(
        {
            name: {"kind": "joint_position_and_stability", **cfg.to_dict()}
            for name, cfg in prismatic.conditions.items()
        }
    )
    for name, condition in controls.conditions.items():
        value = condition.to_dict()
        family = value.pop("family")
        settles[name] = {"kind": family + "_control", **value}
    # Nested sensor config objects must serialize as ordinary owner declarations.
    for name, cfg in conditions.items():
        if cfg.support_sensor is not None:
            settles[name]["support_sensor"] = cfg.support_sensor.to_dict()
    top = scene.table_top_z
    if top is None:
        raise ValueError(
            "Generated handover provider requires the source tabletop height."
        )
    integration = dict(
        integration_id=f"{program_id}_integration_v1",
        program_id=program_id,
        requires=dict(
            scene_contract=scene_binding["contract_id"],
            embodiment_contract=embodiment["skill_profile"]["contract_id"],
        ),
        scene_binding=scene_binding,
        profile=profile,
        runtime_services=dict(
            handover_pose_providers=[
                dict(
                    kind="configured_pose",
                    final_position=[0.0, -0.08, float(top) + 0.10],
                    final_quaternion_xyzw=[0.7071067812, 0.0, 0.0, 0.7071067812],
                )
            ],
            registered_semantic_lowerers=lowerers,
            settle_presets=settles,
            **(
                {"control_operations": controls.runtime_declarations()}
                if controls.operations
                else {}
            ),
        ),
    )
    program = build_semantic_program_declaration(
        graph,
        program_id,
        scene,
        stability_conditions=conditions,
        operation_groups={**prismatic.operation_groups, **controls.operation_groups},
        relative_position_tolerance=0.10,
    )
    if controls.post_policy_presets:
        selection = TaskProgramIntegrationCfg(
            robot_profile=embodiment["skill_profile"]["profile_id"],
            scene_registry=scene_binding["registry_id"],
            runtime_preset=physical.execution_policy["preset_id"],
        )
        compiled = compile_configured_task_program(
            program, scene_binding=scene_binding, selection=selection
        )
        fingerprint = task_program_fingerprint(compiled)
        selected = set()
        bindings = []
        for segment in compiled.iter_segments():
            presets = controls.post_policy_presets.get(segment.name)
            if presets is None:
                continue
            if segment.name in selected or len(presets) != len(segment.post_policies):
                raise ValueError(
                    "Control post-policy selections require unique complete segment occurrences."
                )
            selected.add(segment.name)
            for index, preset_id in enumerate(presets):
                value = PostPolicyPresetBinding(
                    fingerprint,
                    f"{compiled.program_id}/{segment.segment_id}:post:{index}",
                    preset_id,
                )
                bindings.append(
                    {
                        "program_fingerprint": value.program_fingerprint,
                        "occurrence_id": value.occurrence_id,
                        "preset_id": value.preset_id,
                    }
                )
        if selected != set(controls.post_policy_presets):
            raise ValueError("Control post-policy selections contain unknown segments.")
        integration["runtime_services"]["post_policy_bindings"] = bindings
    return integration, program, controls.sensors, drawers


def prepare_generated_task_program_bundle(
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
    measured_rigid_calls: Sequence[str] | None = None,
    post_handover_places: Sequence[str] | None = None,
    verify_pick_phases: bool = False,
    pour_geometry_revision: int | None = None,
    pour_proxy_position_tolerance_m: float | None = None,
    upright_container_picks: Mapping[str, float] | None = None,
    idle_clearance_routes: Sequence[Mapping[str, Any]] | None = None,
    runtime_profile: str = "default",
    experimental_open_release: bool | None = None,
    native_pick_exploration: bool = False,
    press_extra_distance: float | None = None,
    implementation_revisions: Mapping[str, str] | None = None,
    external_input_digests: Mapping[str, str] | None = None,
    artifact_declarations: Mapping[str, Mapping[str, Any]] | None = None,
    asset_cache_root: str | Path | None = None,
) -> tuple[SemanticTaskGraph, BundleSnapshot]:
    """Prepare a sealed shared-runtime declaration snapshot without publishing it.

    All preparation, public preflight and identity checks precede atomic
    publication. No simulator, IK query or skill implementation is constructed.
    Missing TaskSpec emits the explicit non-certifying absence declaration.

    Args:
        graph: Source semantic graph with stable public call identities.
        prepared_scene: Normal source PreparedScene or explicit frozen physical declarations.
        output_dir: Fresh atomic publication destination.
        robot_profile: Qualified source profile or identity owned by the frozen component.
        max_episodes: Optional confirmation of the frozen episode limit.
        max_episode_steps: Optional confirmation of the frozen step limit.
        fit_grasp_assets: Optional source asset preparation; frozen inputs reject it.
        task_spec: Existing normative specification, or None for explicit absence.
        task_candidate: Original source witness required when a spec is supplied.
        role_bindings: Original grounded roles required with a normative spec.
        step_lineage: Planner-emitted expansion lineage required with a spec.
        twist_support_lift_m: Source physical adaptation request.
        twist_grip_depth_m: Source physical adaptation request.
        twist_contact_feedback: Retired unsupported feedback flag.
        twist_mass_source: Source physical adaptation request.
        support_contract_revision: Optional explicit strong-support contract.
        pick_contact_revision: Native Pick declaration selection.
        pick_contact_revisions: Per-source-node native Pick selections.
        world_yaw_free_places: Explicit inside Place recovery selections.
        measured_rigid_calls: Explicit measured rigid call selections.
        post_handover_places: Explicit receiver placement selections.
        verify_pick_phases: Explicit physical Pick phase qualification.
        pour_geometry_revision: Explicit Pour geometry selection.
        pour_proxy_position_tolerance_m: Explicit Pour proxy budget.
        upright_container_picks: Retired unsupported option.
        idle_clearance_routes: Explicit idle-clearance declarations.
        runtime_profile: Concrete declaration profile or existing auto selection.
        experimental_open_release: Explicit release-evidence selection.
        native_pick_exploration: Physical exploration request; frozen inputs reject it.
        press_extra_distance: Explicit Press source overtravel selection.
        implementation_revisions: Explicit current owner revisions, or current
            packaged source/dependency hashes when omitted.
        artifact_declarations: Additional finite JSON audit declarations sealed
            with the runtime files; reserved names cannot be replaced.
        asset_cache_root: Explicit external derived-asset preparation directory.
            The default remains beside output_dir in .task_engine_assets.
        external_input_digests: Independently frozen additional input files, such
            as grasp annotation caches. Each path/hash is checked before sealing;
            these declarations cannot override physical or calibration inputs.

    Returns:
        Final bound semantic graph and owned preflighted snapshot. Pending recipe
        resolvers reject before write; no unsupported capability is silently dropped.
    """
    if type(prepared_scene) is PreparedScene:
        prepared_scene.assert_source_assets()
    if type(twist_contact_feedback) is not bool or twist_contact_feedback:
        raise ValueError(
            "E8 contact feedback is unsupported; use a basic Twist deployment."
        )
    if upright_container_picks is not None and (
        not isinstance(upright_container_picks, Mapping) or upright_container_picks
    ):
        raise ValueError("upright_container_picks is unsupported in this deployment.")
    selected = validate_semantic_task_graph(graph)
    from .experiment_policy import (
        resolve_profile,
        experimental_release_enabled,
        validate_native_pick_exploration,
    )

    validate_native_pick_exploration(native_pick_exploration, robot_profile)
    requested_runtime_profile = runtime_profile
    runtime_profile = resolve_profile(runtime_profile, selected)
    experiment = runtime_profile != "default"
    release_enabled = experimental_release_enabled(
        runtime_profile, experimental_open_release
    )
    source_artifacts = {}
    frozen = type(prepared_scene) is FrozenPhysicalInput
    if frozen:
        physical = prepared_scene
        physical.assert_unchanged()
        if canonical_robot_profile(
            str(robot_profile).strip()
        ) != canonical_robot_profile(physical.robot_profile):
            raise ValueError("Frozen physical input owns its robot-profile selection.")
        if (
            fit_grasp_assets
            or native_pick_exploration
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
            if value is not None and value != physical.deployment[key]:
                raise ValueError(f"Frozen physical input owns {key}.")
        selected = prepare_source_drawer_graph(selected, physical.scene)
    elif type(prepared_scene) is PreparedScene:
        from .source_physical_input import prepare_source_physical_input

        selected = prepare_source_drawer_graph(selected, prepared_scene)
        prismatic = resolve_prismatic_program_declarations(selected, prepared_scene)
        prepared = prepare_source_physical_input(
            prepared_scene,
            selected,
            robot_profile=robot_profile,
            cache_root=(
                Path(output_dir).expanduser().absolute().parent / ".task_engine_assets"
                if asset_cache_root is None
                else Path(asset_cache_root).expanduser().absolute()
            ),
            runtime_profile=runtime_profile,
            fit_grasp_assets=fit_grasp_assets,
            twist_support_lift_m=twist_support_lift_m,
            twist_grip_depth_m=twist_grip_depth_m,
            twist_mass_source=twist_mass_source,
            press_extra_distance=press_extra_distance,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
            prismatic_uids=tuple(
                dict.fromkeys(value.source.object_id for value in prismatic.bindings)
            ),
        )
        physical = prepared.physical
        source_artifacts = dict(prepared.artifacts)
    else:
        raise ValueError(
            "Shared source generation requires PreparedScene or explicit frozen physical owner declarations."
        )
    if type(verify_pick_phases) is not bool:
        raise TypeError("verify_pick_phases must be a boolean.")
    native_revisions = resolve_native_pick_revisions(
        selected, pick_contact_revision, pick_contact_revisions
    )
    if pour_geometry_revision is not None and (
        type(pour_geometry_revision) is not int or pour_geometry_revision not in (1, 2)
    ):
        raise ValueError("Unsupported source Pour geometry revision.")
    if pour_geometry_revision == 2:
        if (
            type(pour_proxy_position_tolerance_m) not in (int, float)
            or not math.isfinite(pour_proxy_position_tolerance_m)
            or pour_proxy_position_tolerance_m <= 0
        ):
            raise ValueError(
                "Pour region requires an explicit finite positive proxy budget."
            )
    elif pour_proxy_position_tolerance_m is not None:
        raise ValueError("Pour proxy tolerance belongs only to revision 2.")
    if pour_geometry_revision is not None:
        if task_candidate is None or not supports_ordinary_pour(
            selected, task_candidate
        ):
            raise ValueError(
                "Pour geometry revision requires the original unconstrained five-role source witness."
            )
        if pour_geometry_revision == 2 and any(
            node["task_type"] == "E3"
            and node["call"].get("call_id") == "simulation.move_held_object"
            and "reference" in node["call"].get("arguments", {})
            and node["call"]["arguments"]["target"] in selected["targets"]
            for node in selected["nodes"]
        ):
            raise ValueError("Pour region cannot replace an explicit staging target.")

    selected = refine_source_manipulation_declarations(selected, physical.scene)
    from .experiment_declarations import (
        horizontal_handover_permissions,
        horizontal_on_pick_permissions,
        add_reacquisition_clearance,
        merge_contact_permissions,
        validate_post_handover_place_declarations,
        select_end_grip_post_handover_places,
    )

    generated_places, generated_revisions = (), {}
    if experiment:
        generated_places, generated_revisions = horizontal_handover_permissions(
            selected
        )
        if runtime_profile == "contact_experiment":
            generated_revisions = merge_contact_permissions(
                generated_revisions,
                horizontal_on_pick_permissions(
                    selected,
                    {
                        item["runtime_uid"]: item
                        for item in physical.scene.planner_objects
                    },
                ),
            )
        pick_contact_revisions = merge_contact_permissions(
            pick_contact_revisions, generated_revisions
        )
        native_revisions = resolve_native_pick_revisions(
            selected, pick_contact_revision, pick_contact_revisions
        )
        if idle_clearance_routes is not None:
            raise ValueError(
                "Experiment profiles own their generated idle clearance declarations."
            )
        selected, generated_idle = add_reacquisition_clearance(
            selected, physical.embodiment
        )
        idle_clearance_routes = generated_idle or None
    declared_places = () if post_handover_places is None else post_handover_places
    if type(declared_places) not in (list, tuple) or len(set(declared_places)) != len(
        declared_places
    ):
        raise ValueError("Post-handover Place requires unique explicit source nodes.")
    post_handover_nodes = validate_post_handover_place_declarations(
        selected,
        sorted(set(declared_places) | set(generated_places)),
        frozenset(item["uid"] for item in physical.scene.rigid_objects),
    )
    selected = refine_source_return_declarations(selected, physical.scene)
    if (
        any(
            node["call"].get("kind") == "place" and "inside" in node["call"]
            for node in selected["nodes"]
        )
        and physical.scene.articulations
    ):
        prismatic = resolve_prismatic_program_declarations(selected, physical.scene)
        source_drawers = resolve_drawer_declarations(
            selected, physical.scene, prismatic
        )
        selected = rewrite_source_drawer_close_resources(
            selected,
            drawer_bindings=frozenset(
                (route.source.object_id, route.source.part_id)
                for route in source_drawers
            ),
        )
    artifacts = deepcopy(source_artifacts)
    if requested_runtime_profile == "auto":
        artifacts["runtime_profile_selection.json"] = dict(
            schema_version="gen_sim_runtime_profile_selection/v1",
            requested="auto",
            selected=runtime_profile,
            rule=(
                "ordinary_rigid_recipe"
                if runtime_profile == "default"
                else (
                    "pour_recipe"
                    if runtime_profile != "contact_experiment"
                    else "non_pour_recipe"
                )
            ),
        )
    if task_spec is not None:
        if task_candidate is None or role_bindings is None or step_lineage is None:
            raise ValueError(
                "TaskSpec generation requires its source witness and expansion lineage."
            )
        from .semantic_graph_validation import validate_graph_coverage

        coverage = validate_graph_coverage(
            selected, task_spec, task_candidate, role_bindings, step_lineage
        )
        if coverage["status"] != "covered":
            raise ValueError(
                "TaskSpec source declarations are not covered by the semantic graph."
            )
        artifacts["semantic_graph_validation.json"] = {
            "schema_version": "semantic_graph_coverage/v1",
            "coverage": coverage,
            "source": {
                "candidate": deepcopy(dict(task_candidate)),
                "role_bindings": deepcopy(dict(role_bindings)),
            },
        }
    program_id = _program_identifier(selected["task_id"])
    runtime_embodiment = deepcopy(physical.embodiment)
    hand_inputs = {}
    runtime_embodiment, hands, hand_inputs = resolve_source_hand_services(
        runtime_embodiment
    )
    coarse_groups = set()
    if task_candidate is not None:
        coarse_steps = {
            step["id"]
            for step in task_candidate["draft"]["steps"]
            if step["task_type"] == "E8" and step.get("twist_goal_mode") == "coarse"
        }
        if coarse_steps and step_lineage is None:
            raise ValueError("Coarse Twist requires explicit source lineage.")
        coarse_groups = {
            entry["group_id"] for name in coarse_steps for entry in step_lineage[name]
        }
    integration, program, control_sensors, drawers = _ordinary_integration(
        selected,
        physical,
        program_id,
        support_contract_revision,
        embodiment=runtime_embodiment,
        hands=hands,
        press_extra_distance=press_extra_distance,
        twist_chunk_angles={identity: math.radians(20.0) for identity in coarse_groups},
        pour_geometry_revision=pour_geometry_revision,
        pour_proxy_position_tolerance_m=pour_proxy_position_tolerance_m,
        verify_pick_phases=verify_pick_phases,
        idle_routes=idle_clearance_routes or (),
    )
    sensor_ids = {sensor["uid"] for sensor in runtime_embodiment.get("sensor", ())}
    if sensor_ids & {sensor["uid"] for sensor in control_sensors}:
        raise ValueError(
            "Generated control sensor identity already exists in the frozen embodiment."
        )
    if control_sensors:
        runtime_embodiment["sensor"] = [
            *runtime_embodiment.get("sensor", ()),
            *control_sensors,
        ]
    policy = deepcopy(physical.execution_policy)
    motion = policy.setdefault("motion", {})
    motion.update(
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        acceleration_limit=None,
        sample_count_retries=[260, 320],
        require_final_velocity=False,
    )
    policy.setdefault("recovery", {})["planning_budget"] = {}
    if "presets" in policy:
        raise ValueError(
            "Frozen source policy already owns named presets; generation cannot overwrite them."
        )
    policy["presets"] = {}
    profile = integration["profile"]
    post_handover_nodes = select_end_grip_post_handover_places(
        selected,
        profile["action_options"],
        post_handover_nodes,
    )
    if runtime_profile == "contact_experiment":
        pick = profile["action_options"]["pick"]
        pick["hand_interp_steps"] = max(24, pick.get("hand_interp_steps", 0))
    profile["preset_options"] = {}
    yaw_nodes = resolve_inside_yaw_declarations(
        selected,
        world_yaw_free_places,
        scene_binding=integration["scene_binding"],
        effect_monitors=profile["effect_monitors"],
        verified=policy.get("effect_assurance") == "verified",
    )
    measured_nodes = resolve_measured_rigid_declarations(
        selected,
        measured_rigid_calls,
        scene_binding=integration["scene_binding"],
        native_revisions=native_revisions,
    )
    selection = TaskProgramIntegrationCfg(
        robot_profile=physical.embodiment["skill_profile"]["profile_id"],
        scene_registry=integration["scene_binding"]["registry_id"],
        runtime_preset=policy["preset_id"],
    )
    compiled = compile_configured_task_program(
        program, scene_binding=integration["scene_binding"], selection=selection
    )
    first_segment = next(iter(compiled.iter_segments()), None)
    if (
        first_segment is None
        or len(first_segment.calls) != 1
        or first_segment.parallel_block is not None
    ):
        raise ValueError(
            "Source initial-plan qualification requires one first serial occurrence."
        )
    first_occurrence = f"{compiled.program_id}/{first_segment.segment_id}:{first_segment.calls[0].segment_call_index}"
    nodes = {node["id"]: node for node in selected["nodes"]}
    occurrence_ids = {
        segment.name: f"{compiled.program_id}/{segment.segment_id}:0"
        for segment in compiled.iter_segments()
    }
    released_continuations = resolve_released_hand_declarations(
        selected,
        program_fingerprint=task_program_fingerprint(compiled),
        occurrences=occurrence_ids,
    )
    released_clearances = {
        row["clearance_invocation_id"] for row in released_continuations
    }
    released_parks = {
        identity
        for row in released_continuations
        for identity in row["park_invocation_ids"]
    }
    receivers, staged = [], {}
    for node in selected["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        key = (node["task_instance_id"], args.get("object"))
        if call.get("call_id") == "simulation.move_held_object" and "reference" in args:
            staged[key] = args["reference"]
        elif call.get("call_id") == "simulation.pour":
            if key not in staged:
                raise ValueError(
                    "Each Pour occurrence requires its preceding staging receiver."
                )
            receivers.append([occurrence_ids[node["id"]], staged[key]])
    for lowerer in integration["runtime_services"]["registered_semantic_lowerers"]:
        if lowerer["kind"] == "jaw_pour":
            lowerer["receivers"] = receivers

    drawer_rows, drawer_picks = bind_drawer_occurrences(
        selected, drawers, occurrence_ids
    )
    drawer_groups = {
        node["task_instance_id"]
        for node in selected["nodes"]
        if node["id"] in drawer_rows
        and any(
            row["operation"] in {"place", "transport"}
            for row in drawer_rows[node["id"]]
        )
    }
    drawer_targets = {route.target_id: route for route in drawers}
    staged_groups = {
        node.get("task_instance_id", node["id"])
        for node in selected["nodes"]
        if node["task_type"] in {"E2", "E4", "E6"}
        or node["call"]["kind"] == "hand_over"
        or node["call"].get("call_id") == "simulation.coordinated_transport"
        and node["call"].get("arguments", {}).get("relation") == "on"
    }
    cartesian_groups = {
        node.get("task_instance_id", node["id"])
        for node in selected["nodes"]
        if node["call"]["kind"] == "hand_over"
        or node["call"].get("call_id")
        in {"gen_sim.clear_released", "gen_sim.articulation_withdraw"}
    }
    staged_groups.update(drawer_groups)
    cartesian_groups.update(drawer_groups)
    staging_bindings = []
    rigid_ids = {item["uid"] for item in physical.scene.rigid_objects}
    for node in selected["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        if (
            call.get("call_id") != "gen_sim.align_held"
            or args.get("preserve_yaw") is not False
            or args.get("target") == "current_object_pose"
            or args.get("object") not in rigid_ids
        ):
            continue
        resource = call["resources"]["primary"]
        binding = dict(
            invocation_id=occurrence_ids[node["id"]],
            rigid_entity_id=args["object"],
            articulation_scene=bool(physical.scene.articulations),
        )
        predecessors = []
        for name in node.get("depends_on", ()):
            before = nodes[name]["call"]
            identity = before.get("call_id", before["kind"])
            if (
                before.get("object", before.get("arguments", {}).get("object"))
                == args["object"]
                and before.get("resources", {}).get("primary") == resource
                and (
                    identity in {"pick", "gen_sim.align_held"}
                    or identity.startswith("gen_sim.pick.")
                )
            ):
                predecessors.append((name, identity))
        if len(predecessors) == 1:
            name, identity = predecessors[0]
            binding.update(
                predecessor_invocation_id=occurrence_ids[name],
                predecessor_call_id=identity,
                resource_id=resource,
            )
        staging_bindings.append(binding)
    staging_ids = {item["invocation_id"] for item in staging_bindings}
    handover_options = resolve_handover_profile_declarations(
        selected, integration["runtime_services"]["registered_semantic_lowerers"]
    )
    idle_nodes = {
        name
        for row in (idle_clearance_routes or ())
        for name in (row["clear_segment"], row["restore_segment"])
    }
    budget_nodes = set(measured_nodes)
    if policy.get("effect_assurance") == "verified":
        budget_nodes.update(
            name
            for pair in source_pick_release_pairs(selected, profile["effect_monitors"])
            for name in pair
        )
    if pour_geometry_revision in (1, 2):
        budget_nodes.update(
            node["id"]
            for node in selected["nodes"]
            if node["call"].get("call_id") == "simulation.pour"
        )
    pour_returns = frozenset(_pour_return_call_declarations(selected))
    presets = {}
    place_bindings = []
    slide_preshape_bindings = []
    for segment in compiled.iter_segments():
        node = nodes[segment.name]
        call = node["call"]
        identity = call["kind"] if call["kind"] != "registered" else call["call_id"]
        name = f"gen_sim.{segment.name}.execution"
        presets[segment.name] = name
        minimum = (
            260 if node.get("task_instance_id", node["id"]) in staged_groups else 140
        )
        current = dict(
            motion=dict(sample_count=max(minimum, int(motion.get("sample_count") or 0)))
        )
        if policy.get("effect_assurance") == "verified":
            current["release_gate_evidence"] = "binary_only"
        if runtime_profile == "contact_experiment" and call["kind"] == "pick":
            current["motion"]["sample_count"] += 12
        if node["id"] in post_handover_nodes and experiment:
            current["motion"]["sample_count"] = 140
        options = deepcopy(profile["action_options"][identity])
        if identity == "pick":
            options.update(
                adaptive_grasp="pour" if node["task_type"] == "E3" else "ordinary",
                full_opening_clearance=True,
                require_qualified_approach=node["task_type"] != "E3",
            )
            if native_pick_exploration and node["task_type"] != "E3":
                options["native_scene_trial"] = True
            if node["task_type"] != "E3":
                current["recovery"] = dict(
                    planning_recovery=dict(kind="pick_bounded", revision=1)
                )
        elif options["kind"] == "pick_up":
            constrained = any(
                route.get("release_clearance_object_pose") is not None
                or route.get("grasp_region") is not None
                for lowerer in integration["runtime_services"][
                    "registered_semantic_lowerers"
                ]
                if lowerer["kind"] == "pick"
                and lowerer.get("call_id", "simulation.pick") == identity
                for route in lowerer["routes"]
            )
            if constrained:
                options["constrained_grasp"] = True
            else:
                options.update(
                    adaptive_grasp="handover_source",
                    full_opening_clearance=True,
                    require_qualified_approach=False,
                )
            if constrained:
                current["recovery"] = dict(
                    planning_recovery=dict(
                        kind="pick_bounded", revision=1, constrained_roll=True
                    )
                )
        elif identity == "gen_sim.align_held":
            options.update(lift_clearance=0.06, report_retained_attachment=True)
            if call["arguments"].get("preserve_yaw") is False:
                current["recovery"] = dict(
                    planning_recovery=dict(
                        kind="held_alignment_bounded",
                        revision=1,
                        staging=occurrence_ids[node["id"]] in staging_ids,
                    )
                )
        elif identity == "hand_over":
            options.update(handover_options[node["id"]])
            if options["adaptive_receive"]:
                current["recovery"] = dict(
                    planning_recovery=dict(kind="handover_bounded", revision=1)
                )
        elif identity in {"place", "simulation.place_relative"}:
            options.update(
                linear_segments=True,
                qualify_required_waypoints=(
                    node.get("task_instance_id", node["id"]) not in cartesian_groups
                ),
                monitor_scene_until_approach=True,
            )
            if node["id"] in pour_returns:
                # Source Pour-return policy revision 1 selected only this exact
                # occurrence; ordinary Place and the shared default keep theirs.
                options["lift_height"] = 0.12
            current["motion"].update(
                segment_sample_count_retries=[260, 320],
                segment_continuous_ik_recovery=True,
            )
            # Source Cartesian Place bypasses both the required-waypoint probe
            # and its bounded waypoint recovery, retaining ordinary nominal
            # planning and the independent segment motion retries below.
            if node.get("task_instance_id", node["id"]) not in cartesian_groups:
                current["recovery"] = dict(
                    planning_recovery=dict(
                        kind="place_bounded", revision=1, variant="waypoint"
                    ),
                )
            args = call.get("arguments", {})
            support = args.get("reference")
            if "inside" in call:
                support = call["inside"].split("__")[1]
            if "on" in call:
                support = call["on"].split("__")[1]
            place_bindings.append(
                dict(
                    invocation_id=f"{compiled.program_id}/{segment.segment_id}:0",
                    floor_entity_id="table",
                    support_owner_id=support,
                )
            )
        elif identity == "simulation.move_held_object":
            # Source routed targets are SceneEntityPose expressions or one 4x4
            # absolute pose. Its lift prelude only owned already batched (B,4,4)
            # targets from alignment lowering, so routed Pour transport remains
            # a direct path. Resolving expressions before this choice adds a lift.
            options.update(report_retained_attachment=True)
        elif identity in {
            "simulation.coordinated_transport",
            "simulation.coordinated_hold",
        }:
            options.update(continue_verified_holds=True, prefer_cargo_clearance=True)
        elif identity in {
            "gen_sim.articulation_slide",
            "gen_sim.articulation_withdraw",
            "gen_sim.articulation_park",
        }:
            kinds = {
                "gen_sim.articulation_slide": "slide_bounded",
                "gen_sim.articulation_withdraw": "articulation_withdrawal",
                "gen_sim.articulation_park": "articulation_park",
            }
            current["motion"].update(
                segment_sample_count_retries=(
                    [] if identity == "gen_sim.articulation_park" else [260, 320]
                ),
                segment_continuous_ik_recovery=False,
            )
            current["recovery"] = dict(
                planning_recovery=dict(kind=kinds[identity], revision=1)
            )
            if identity == "gen_sim.articulation_slide":
                options["qualify_hand_clearance"] = True
                current["recovery"]["planning_recovery"][
                    "preshape_aperture_recovery"
                ] = True
                resource = call["resources"]["primary"]
                hand = next(item for item in hands if item.resource_id == resource)
                slide_preshape_bindings.append(
                    dict(
                        invocation_id=f"{compiled.program_id}/{segment.segment_id}:0",
                        grasp_target_id=hand.grasp_target_id,
                        table_uid="table",
                    )
                )
        elif identity.startswith("gen_sim.press") or identity.startswith(
            "gen_sim.twist"
        ):
            if identity.startswith("gen_sim.press.") or identity == "gen_sim.press":
                current["motion"].update(
                    strategy="ik_interp",
                    sample_count=max(260, current["motion"]["sample_count"]),
                )
                current["recovery"] = dict(max_replans=0, max_action_retries=0)
            elif identity.startswith("gen_sim.twist.") or identity == "gen_sim.twist":
                current["motion"].update(
                    strategy="ik_interp",
                    sample_count=max(320, current["motion"]["sample_count"]),
                )
                current["recovery"] = dict(
                    max_replans=0,
                    max_action_retries=0,
                    planning_recovery=dict(kind="twist_bounded", revision=1),
                )
        if node["id"] in post_handover_nodes:
            options.update(
                linear_segments=True,
                qualify_required_waypoints=False,
                monitor_scene_until_approach=True,
            )
            current["motion"].update(
                segment_sample_count_retries=[260, 320],
                segment_continuous_ik_recovery=True,
            )
            current["recovery"] = dict(
                planning_recovery=dict(
                    kind="place_bounded", revision=1, variant="retained_end_grip"
                )
            )
            bound = next(
                row
                for row in place_bindings
                if row["invocation_id"] == occurrence_ids[node["id"]]
            )
            bound["support_owner_id"] = None
        if node["id"] in idle_nodes:
            options["require_free_motion"] = True
        if node["id"] in yaw_nodes:
            current["recovery"]["planning_recovery"]["inside_world_yaw"] = True
            row = next(
                value
                for value in place_bindings
                if value["invocation_id"] == occurrence_ids[node["id"]]
            )
            row.update(
                inside_subject_id=yaw_nodes[node["id"]][0],
                inside_container_id=yaw_nodes[node["id"]][1],
            )
        if node["id"] in measured_nodes:
            options[
                (
                    "measured_inside"
                    if "container_id" in measured_nodes[node["id"]]
                    else "measured_preshape"
                )
            ] = True
        if node["id"] in native_revisions:
            options["native_closing_revision"] = native_revisions[node["id"]]
            if not options.get("constrained_grasp", False):
                options["full_opening_clearance"] = native_revisions[node["id"]] < 5
            if options.get("adaptive_grasp") == "pour" or (
                options.get("adaptive_grasp") == "handover_source"
                and native_revisions[node["id"]] == 8
            ):
                current.setdefault("recovery", {})["planning_recovery"] = dict(
                    kind="pick_bounded", revision=1
                )
        if node["id"] in budget_nodes:
            current.setdefault("recovery", {})["planning_budget"] = dict(
                wall_seconds=90.0
            )
        if node["id"] in drawer_picks:
            options["qualify_downstream_targets"] = False
        if node["id"] in drawer_rows:
            row = drawer_rows[node["id"]][0]
            if row["operation"] == "close":
                options["drawer_containment"] = True
            else:
                route = drawer_targets[row["target_id"]]
                options["preserve_vertical"] = node["id"] in route.vertical_nodes
                if row["operation"] == "place":
                    options["drawer_placement"] = True
                else:
                    options["drawer_transport"] = True
                    options.pop("lift_clearance", None)
                    options["linear_segments"] = False
                    current["motion"].update(
                        strategy="motion_gen",
                        interpolation_mode="adaptive",
                        ik_seed_mode="batched",
                        segment_sample_count_retries=[],
                        segment_continuous_ik_recovery=False,
                    )
                    current.pop("recovery", None)
        if node.get("task_instance_id", node["id"]) in cartesian_groups and options[
            "kind"
        ] in {"pick_up", "place", "move_held_object", "move_end_effector", "hand_over"}:
            if (
                identity
                not in {
                    "gen_sim.articulation_withdraw",
                    "gen_sim.drawer_transport",
                }
                and node["id"] not in post_handover_nodes
            ):
                options["linear_segments"] = True
                current["motion"].update(
                    segment_sample_count_retries=[260, 320],
                    segment_continuous_ik_recovery=False,
                )
        occurrence = occurrence_ids[segment.name]
        if occurrence in released_clearances:
            options["linear_segments"] = True
            current["motion"].update(
                segment_sample_count_retries=[260, 320],
                segment_continuous_ik_recovery=False,
            )
            current.setdefault("recovery", {})["planning_recovery"] = dict(
                kind="released_clearance", revision=1
            )
        elif occurrence in released_parks:
            current.setdefault("recovery", {})["planning_recovery"] = dict(
                kind="released_park", revision=1
            )
        if occurrence == first_occurrence:
            current["motion"].update(
                require_final_velocity=True, final_velocity_rejection="mask_only"
            )
            current.setdefault("recovery", {})["retry_all_failed_plans"] = False
        policy["presets"][name] = current
        profile["preset_options"][name] = {"action_options": {identity: options}}
    effective_options = {
        node_id: next(
            iter(profile["preset_options"][preset]["action_options"].values())
        )
        for node_id, preset in presets.items()
    }
    purposes = _place_acquisition_purposes(
        selected, occurrences=occurrence_ids, options=effective_options
    )
    for node_id, occurrence in occurrence_ids.items():
        acquisition = purposes.get(occurrence)
        if acquisition is None or acquisition[1] not in {"pour", "ends"}:
            continue
        current = policy["presets"][presets[node_id]]
        recovery = current.get("recovery", {})
        rule = recovery.get("planning_recovery", {})
        # Source excludes these held purposes from ordinary waypoint screening
        # and recovery. Independent retained-end-grip/other declarations keep
        # their own dispatch, as do all motion retries and budget fields.
        if rule.get("kind") == "place_bounded" and rule.get("variant") == "waypoint":
            effective_options[node_id]["qualify_required_waypoints"] = False
            recovery.pop("planning_recovery")
            if not recovery:
                current.pop("recovery")
    continuations = resolve_source_continuation_declarations(
        selected,
        program_fingerprint=task_program_fingerprint(compiled),
        occurrences=occurrence_ids,
        options=effective_options,
        motions={
            node_id: {**motion, **policy["presets"][preset]["motion"]}
            for node_id, preset in presets.items()
        },
        recoveries={
            node_id: policy["presets"][preset]
            .get("recovery", {})
            .get("planning_recovery", {})
            for node_id, preset in presets.items()
        },
        staging_bindings=staging_bindings,
        deferred_picks=drawer_picks,
    )
    by_occurrence = {value: key for key, value in occurrence_ids.items()}
    for continuation in continuations:
        source = by_occurrence[continuation["source_invocation_id"]]
        policy["presets"][presets[source]]["recovery"]["planning_recovery"][
            "place_continuation"
        ] = True
    services = integration["runtime_services"]
    carrier_ids, excluded_ids = set(), set()
    native_owners = {
        item["entity_id"]: item.get("simulation_uid", item["entity_id"])
        for kind in ("rigid_objects", "articulations")
        for item in integration["scene_binding"].get(kind, ())
    }
    for node in selected["nodes"]:
        call = node["call"]
        subject = call.get("object", call.get("arguments", {}).get("object"))
        if isinstance(subject, str) and subject in native_owners:
            (carrier_ids if node["task_type"] == "E5" else excluded_ids).add(
                native_owners[subject]
            )
    if carrier_ids:
        services["cargo_envelope"] = dict(
            carrier_uids=sorted(carrier_ids),
            excluded_uids=sorted(excluded_ids),
        )
    if carrier_ids:
        from embodichain.lab.gym.utils.control_timing import resolve_control_timing
        from embodichain.lab.sim.cfg.physics import PhysicsBackendCfg
        from embodichain.lab.sim.sensors.contact.timeline import ContactTimelineCfg
        from embodichain.lab.sim.atomic_actions.grip_evidence import (
            DriverOpeningBinding,
        )
        from embodichain.lab.task_program.integrations.simulation.recorded_contacts import (
            SimulationRecordedContactCfg,
        )

        env_fields = physical.deployment.get("env", {})
        physics_fields = physical.deployment.get("physics_config", {}) or {}
        sim_fields = physical.deployment.get("sim_cfg", {}) or {}
        timing = resolve_control_timing(
            physics_fields.get(
                "physics_dt",
                sim_fields.get("physics_dt", PhysicsBackendCfg().physics_dt),
            ),
            env_fields.get("sim_steps_per_control", 4),
            target_control_frequency=env_fields.get("target_control_frequency"),
        )
        recorded = SimulationRecordedContactCfg(
            sensor_uid="gen_sim_task_spec_contacts",
            retention=ContactTimelineCfg(step_dt=timing.step_dt),
            pad_bindings=tuple((hand.hand_part, hand.pad_link_names) for hand in hands),
            opening_bindings=tuple(
                DriverOpeningBinding(
                    value["control_part"],
                    tuple(value["joint_names"]),
                    tuple(value["driver_indices"]),
                    tuple(value["open_qpos"]),
                    value["tolerance"],
                )
                for value in declare_source_opening_bindings(physical.embodiment, hands)
            ),
        ).snapshot()
        services["recorded_contacts"] = recorded.to_dict()
        services["recorded_contacts"]["opening_bindings"] = [
            asdict(value) for value in recorded.opening_bindings
        ]
        sensor_cfg = recorded.contact_sensor_cfg(
            robot_uid=runtime_embodiment["simulation"]["uid"],
            rigid_uids=[
                item["simulation_uid"]
                for item in integration["scene_binding"]["rigid_objects"]
            ],
        )
        sensors = runtime_embodiment.setdefault("sensor", [])
        if any(value["uid"] == sensor_cfg.uid for value in sensors):
            raise ValueError(
                "The recorded source contact sensor must have one declaration owner."
            )
        sensors.append(sensor_cfg.to_dict())
    services["control_part_evidence"] = declare_source_grip_evidence(
        runtime_embodiment,
        hands,
        integration["scene_binding"],
        experimental_open_release=release_enabled,
    )
    services["planning_services"] = dict(
        kind="simulation_geometry",
        revision=1,
        free_motion_query="discrete_mesh",
        capture_root_pose=True,
        capture_native_revision=True,
        hold_qpos_source="drive_target",
        observed_curobo=dict(max_voxel_count=4000000),
        collision_roster=dict(
            rigid_objects=[
                [item["uid"], item["uid"]]
                for item in (*physical.scene.background, *physical.scene.rigid_objects)
            ],
            articulation_links=[
                [item["entity_id"], item["articulation_id"], item["native_link_name"]]
                for item in integration["scene_binding"]["links"]
            ],
        ),
        place_scene_bindings=place_bindings,
    )
    if native_revisions:
        services["planning_services"]["pick_closing_bindings"] = (
            resolve_native_pick_bindings(
                selected,
                revisions=native_revisions,
                occurrences=occurrence_ids,
                options=effective_options,
                continuations=continuations,
                scene_binding=integration["scene_binding"],
            )
        )
    if measured_nodes:
        services["planning_services"]["measured_rigid_bindings"] = [
            dict(invocation_id=occurrence_ids[name], **values)
            for name, values in measured_nodes.items()
        ]
    if continuations:
        services["planning_services"]["place_continuation_bindings"] = continuations
    if released_continuations:
        services["planning_services"][
            "released_hand_continuations"
        ] = released_continuations
    if drawer_rows:
        services["planning_services"]["drawer_bindings"] = [
            row for rows in drawer_rows.values() for row in rows
        ]
    if staging_bindings:
        services["planning_services"]["held_staging_bindings"] = staging_bindings
    if any(options["adaptive_receive"] for options in handover_options.values()):
        services["planning_services"]["native_handover_scene"] = True
    if pour_geometry_revision is not None:
        projection_bindings = []
        for group in selected["task_groups"]:
            if group["task_type"] != "E3":
                continue
            roles = [
                nodes[name]
                for name in group["node_ids"]
                if nodes[name]["call"].get("call_id") != "simulation.park"
            ]
            projection_bindings.append(
                dict(
                    zip(
                        (
                            "pick_invocation_id",
                            "stage_invocation_id",
                            "pour_invocation_id",
                            "return_invocation_id",
                            "place_invocation_id",
                        ),
                        (occurrence_ids[node["id"]] for node in roles),
                        strict=True,
                    )
                )
            )
        services["planning_services"]["pour_projection_bindings"] = projection_bindings
    paired_calls = [
        node["call"]
        for node in selected["nodes"]
        if node["call"].get("call_id")
        in {"simulation.coordinated_transport", "simulation.coordinated_hold"}
    ]
    if paired_calls:
        by_resource = {hand.resource_id: hand for hand in hands}
        pairs = []
        for call in paired_calls:
            left = by_resource[
                call.get("resources", {}).get("left", "left")
            ].grasp_target_id
            right = by_resource[
                call.get("resources", {}).get("right", "right")
            ].grasp_target_id
            pair = (left, right)
            if pair not in pairs:
                pairs.append(pair)
        services["planning_services"]["capture_coordinated_cargo"] = True
        services["planning_services"]["coordinated_grasps"] = [
            dict(
                left_grasp_target_id=left,
                right_grasp_target_id=right,
                contact=dict(
                    contact_tolerance=0.0005,
                    minimum_normal_cosine=0.5,
                    normal_probe_distance=0.00001,
                    max_candidates=32,
                    candidate_chunk_size=32,
                    cargo_clearance_reserve=0.008,
                ),
            )
            for left, right in pairs
        ]
    if slide_preshape_bindings:
        services["planning_services"][
            "slide_preshape_bindings"
        ] = slide_preshape_bindings
    deployment = deepcopy(physical.deployment)
    if carrier_ids:
        deployment["record_trajectory"] = True
    deployment.update(
        id=f"{program_id}_v1",
        scene={"component": "components/scene.yaml"},
        embodiment={"component": "components/embodiment.yaml"},
        task_program=dict(
            program="task_program/program.yaml",
            integration="task_program/integration.yaml",
            execution_policy="components/execution_policy.yaml",
        ),
    )
    revisions = (
        task_program_implementation_revisions()
        if implementation_revisions is None
        else dict(implementation_revisions)
    )
    inputs = {**physical.input_files, **hand_inputs}
    additional_inputs = {}
    if external_input_digests is not None:
        if not isinstance(external_input_digests, Mapping):
            raise TypeError("external_input_digests must be a path/digest mapping.")
        for name, digest in external_input_digests.items():
            if (
                type(name) is not str
                or not name
                or not Path(name).is_absolute()
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "Additional inputs require absolute paths and SHA-256 digests."
                )
            if name in inputs and inputs[name] != digest:
                raise ValueError(
                    "Additional input conflicts with its existing owner declaration."
                )
            path = Path(name)
            if not path.is_file() or _source_file_digest(path) != digest:
                raise ValueError(f"Frozen additional input changed: {name}")
            inputs[name] = digest
            additional_inputs[name] = digest
    if artifact_declarations is not None and not isinstance(
        artifact_declarations, Mapping
    ):
        raise TypeError("Supplemental audit declarations must be a mapping.")
    for name, declaration in (artifact_declarations or {}).items():
        if type(name) is not str or not isinstance(declaration, Mapping):
            raise TypeError(
                "Supplemental audit declarations require named JSON mappings."
            )
        if name in artifacts:
            raise ValueError(f"Supplemental declaration already has an owner: {name}")
        artifacts[name] = deepcopy(dict(declaration))
    _verify_source_inputs(inputs)
    bound, snapshot = prepare_task_program_bundle(
        selected,
        task_spec,
        program=program,
        integration=integration,
        scene=dict(
            scene_id=f"{program_id}_generated_scene",
            simulation=deepcopy(physical.scene_simulation),
        ),
        embodiment=runtime_embodiment,
        execution_policy=policy,
        deployment=deployment,
        invocation_presets=presets,
        external_input_digests=inputs,
        implementation_revisions=revisions,
        base_dir=Path(output_dir).expanduser().resolve(),
        artifact_declarations=artifacts,
        source_input_provenance={
            "schema_version": SOURCE_INPUT_PROVENANCE_SCHEMA,
            "groups": {
                "physical": dict(physical.input_files),
                "hand_services": dict(hand_inputs),
                "additional": dict(additional_inputs),
            },
        },
    )
    physical.assert_unchanged()
    for name, digest in additional_inputs.items():
        path = Path(name)
        if not path.is_file() or _source_file_digest(path) != digest:
            raise ValueError(f"Frozen additional input changed: {name}")
    _verify_generated_input_inventory(
        snapshot, base_dir=Path(output_dir).expanduser().resolve()
    )
    return bound, snapshot


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
    measured_rigid_calls: Sequence[str] | None = None,
    post_handover_places: Sequence[str] | None = None,
    verify_pick_phases: bool = False,
    pour_geometry_revision: int | None = None,
    pour_proxy_position_tolerance_m: float | None = None,
    upright_container_picks: Mapping[str, float] | None = None,
    idle_clearance_routes: Sequence[Mapping[str, Any]] | None = None,
    runtime_profile: str = "default",
    experimental_open_release: bool | None = None,
    native_pick_exploration: bool = False,
    press_extra_distance: float | None = None,
    implementation_revisions: Mapping[str, str] | None = None,
    external_input_digests: Mapping[str, str] | None = None,
    artifact_declarations: Mapping[str, Mapping[str, Any]] | None = None,
    asset_cache_root: str | Path | None = None,
) -> tuple[SemanticTaskGraph, TaskProgramBundlePaths]:
    """Generate a shared-runtime deployment from source task and physical declarations.

    All preparation, public preflight and identity checks precede atomic
    publication. No simulator, IK query or skill implementation is constructed.
    Missing TaskSpec emits the explicit non-certifying absence declaration.

    Args:
        graph: Source semantic graph with stable public call identities.
        prepared_scene: Normal source PreparedScene or explicit frozen physical declarations.
        output_dir: Fresh atomic publication destination.
        robot_profile: Qualified source profile or identity owned by the frozen component.
        max_episodes: Optional confirmation of the frozen episode limit.
        max_episode_steps: Optional confirmation of the frozen step limit.
        fit_grasp_assets: Optional source asset preparation; frozen inputs reject it.
        task_spec: Existing normative specification, or None for explicit absence.
        task_candidate: Original source witness required when a spec is supplied.
        role_bindings: Original grounded roles required with a normative spec.
        step_lineage: Planner-emitted expansion lineage required with a spec.
        twist_support_lift_m: Source physical adaptation request.
        twist_grip_depth_m: Source physical adaptation request.
        twist_contact_feedback: Retired unsupported feedback flag.
        twist_mass_source: Source physical adaptation request.
        support_contract_revision: Optional explicit strong-support contract.
        pick_contact_revision: Native Pick declaration selection.
        pick_contact_revisions: Per-source-node native Pick selections.
        world_yaw_free_places: Explicit inside Place recovery selections.
        measured_rigid_calls: Explicit measured rigid call selections.
        post_handover_places: Explicit receiver placement selections.
        verify_pick_phases: Explicit physical Pick phase qualification.
        pour_geometry_revision: Explicit Pour geometry selection.
        pour_proxy_position_tolerance_m: Explicit Pour proxy budget.
        upright_container_picks: Retired unsupported option.
        idle_clearance_routes: Explicit idle-clearance declarations.
        runtime_profile: Concrete declaration profile or existing auto selection.
        experimental_open_release: Explicit release-evidence selection.
        native_pick_exploration: Physical exploration request; frozen inputs reject it.
        press_extra_distance: Explicit Press source overtravel selection.
        implementation_revisions: Explicit current owner revisions, or current
            packaged source/dependency hashes when omitted.
        artifact_declarations: Additional finite JSON audit declarations sealed
            with the runtime files; reserved names cannot be replaced.
        asset_cache_root: Explicit external derived-asset preparation directory.
            The default remains beside output_dir in .task_engine_assets.
        external_input_digests: Independently frozen additional input files, such
            as grasp annotation caches. Each path/hash is checked before sealing;
            these declarations cannot override physical or calibration inputs.

    Returns:
        Final bound semantic graph and canonical published paths. Pending recipe
        resolvers reject before write; no unsupported capability is silently dropped.
    """
    bound, snapshot = prepare_generated_task_program_bundle(
        graph,
        prepared_scene,
        output_dir,
        robot_profile=robot_profile,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        fit_grasp_assets=fit_grasp_assets,
        task_spec=task_spec,
        task_candidate=task_candidate,
        role_bindings=role_bindings,
        step_lineage=step_lineage,
        twist_support_lift_m=twist_support_lift_m,
        twist_grip_depth_m=twist_grip_depth_m,
        twist_contact_feedback=twist_contact_feedback,
        twist_mass_source=twist_mass_source,
        support_contract_revision=support_contract_revision,
        pick_contact_revision=pick_contact_revision,
        pick_contact_revisions=pick_contact_revisions,
        world_yaw_free_places=world_yaw_free_places,
        measured_rigid_calls=measured_rigid_calls,
        post_handover_places=post_handover_places,
        verify_pick_phases=verify_pick_phases,
        pour_geometry_revision=pour_geometry_revision,
        pour_proxy_position_tolerance_m=pour_proxy_position_tolerance_m,
        upright_container_picks=upright_container_picks,
        idle_clearance_routes=idle_clearance_routes,
        runtime_profile=runtime_profile,
        experimental_open_release=experimental_open_release,
        native_pick_exploration=native_pick_exploration,
        press_extra_distance=press_extra_distance,
        implementation_revisions=implementation_revisions,
        external_input_digests=external_input_digests,
        artifact_declarations=artifact_declarations,
        asset_cache_root=asset_cache_root,
    )
    return bound, publish_task_program_bundle(
        snapshot,
        output_dir,
        validate_inputs=lambda: _verify_generated_input_inventory(
            snapshot, base_dir=Path(output_dir).expanduser().resolve()
        ),
    )
