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
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib, json
from pathlib import Path
import numpy as np
import pytest
import trimesh
import yaml
from embodichain.gen_sim.task_engine.physical_input import FrozenPhysicalInput
from embodichain.gen_sim.task_engine.embodiment_declarations import (
    resolve_source_hand_services,
)
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.gen_sim.task_engine.bundle_deployment import load_configured_bundle
from embodichain.gen_sim.task_spec import canonical_hash
from embodichain.data.asset_dependencies import collect_declared_asset_files


def _frozen(tmp_path, *, slender=False, slender_pad=False):
    files = json.loads(
        (Path(__file__).parent / "data/e1_bundle_declarations.json").read_text()
    )["files"]
    values = {name: yaml.safe_load(content) for name, content in files.items()}
    simulation = values["components/scene.yaml"]["simulation"]
    assets = {}
    for item in simulation["rigid_object"]:
        size = (
            ([0.1, 0.1, 0.3] if slender else [0.1, 0.1, 0.1])
            if item["uid"] == "cylinder"
            else ([0.1, 0.1, 0.3] if slender_pad else [0.4, 0.4, 0.02])
        )
        mesh = trimesh.creation.box(extents=size)
        mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
        path = tmp_path / (item["uid"] + ".glb")
        mesh.export(path)
        item["shape"]["fpath"] = str(path)
        assets[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Freeze the same complete source inventory as the physical-input owner.
    # Keep every original physical/semantic value; the earlier hand-made map
    # recorded only the generated scene meshes and omitted robot dependencies.
    assets = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            collect_declared_asset_files(
                (simulation, values["components/embodiment.yaml"]), base_dir=tmp_path
            )
        )
    }
    planner = tuple(
        {
            **deepcopy(item),
            "runtime_uid": item["uid"],
            "source_uid": item["uid"],
            "role": role,
        }
        for role in ("background", "rigid_object")
        for item in simulation[role]
    )
    scene = PreparedScene(
        tmp_path / "source.json",
        tmp_path,
        planner,
        tuple(simulation["background"]),
        tuple(simulation["rigid_object"]),
        (),
        {},
        0.56,
        0.0,
        "preserve",
        (1.0, 1.0, 1.0),
        {},
    )
    policy = values["components/execution_policy.yaml"]
    policy.pop("presets")
    deployment = {
        key: value
        for key, value in values["task_program_deployment.yaml"].items()
        if key not in {"id", "scene", "embodiment", "task_program"}
    }
    frozen = FrozenPhysicalInput(
        scene,
        simulation,
        values["components/embodiment.yaml"],
        policy,
        deployment,
        "dual_franka",
        assets,
        "",
    )
    plain = asdict(scene)
    plain["source_config_path"] = str(scene.source_config_path)
    plain["scene_dir"] = str(scene.scene_dir)
    digest = canonical_hash(
        dict(
            scene=plain,
            scene_simulation=simulation,
            embodiment=frozen.embodiment,
            execution_policy=policy,
            deployment=deployment,
            robot_profile="dual_franka",
            input_files=assets,
        )
    )
    return values, replace(frozen, physical_digest=digest)


@pytest.mark.parametrize("with_frozen_cache", [False, True])
def test_original_generation_api_publishes_from_frozen_declarations_without_sim(
    tmp_path, monkeypatch, with_frozen_cache
):
    from embodichain.lab.sim.sim_manager import SimulationManager

    def forbidden(*args, **kwargs):
        raise AssertionError("generation constructed simulation")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    values, physical = _frozen(tmp_path)
    graph = values["semantic_task_graph.json"]
    before = deepcopy(graph)
    cache = tmp_path / "frozen_candidates.npy"
    cache.write_bytes(b"independently frozen candidate input")
    additional = (
        {str(cache): hashlib.sha256(cache.read_bytes()).hexdigest()}
        if with_frozen_cache
        else {}
    )
    output = tmp_path / "generated"
    revisions = {"embodichain": "test-source"}
    selected, paths = generate_task_program_bundle(
        graph,
        physical,
        output,
        robot_profile="dual_franka",
        implementation_revisions=revisions,
        external_input_digests=additional,
    )
    assert paths.root == output
    assert graph == before
    loaded = load_configured_bundle(
        output,
        external_input_digests={
            **physical.input_files,
            **resolve_source_hand_services(physical.embodiment)[2],
            **additional,
        },
        implementation_revisions=revisions,
    )
    assert loaded.task_spec is None
    provenance = json.loads(paths.source_input_provenance.read_text())
    assert provenance["schema_version"] == "gen_sim.source-input-provenance/v1"
    assert provenance["groups"] == {
        "physical": physical.input_files,
        "hand_services": resolve_source_hand_services(physical.embodiment)[2],
        "additional": additional,
    }
    assert (
        selected["integration_fingerprint"]
        == loaded.deployment.integration.integration_fingerprint
    )
    assert loaded.program.program_id == "gen_sim_demo"
    assert not (output / "task_program/constraints.json").exists()
    registration = loaded.deployment.integration.registration
    assert len(registration.invocation_policy_bindings) == 3
    policies = yaml.safe_load(
        (output / "components/execution_policy.yaml").read_text()
    )["presets"]
    first, *later = policies.values()
    assert all(
        preset["release_gate_evidence"] == "binary_only" for preset in policies.values()
    )
    assert first["motion"]["require_final_velocity"] is True
    assert first["motion"]["final_velocity_rejection"] == "mask_only"
    assert first["recovery"]["retry_all_failed_plans"] is False
    assert all(
        "require_final_velocity" not in value["motion"]
        and "retry_all_failed_plans" not in value.get("recovery", {})
        for value in later
    )

    assert registration.planning_services.hold_qpos_source == "drive_target"
    place = yaml.safe_load((paths.root / "task_program/integration.yaml").read_text())[
        "profile"
    ]["preset_options"]["gen_sim.step_01__call_02.execution"]["action_options"][
        "simulation.place_relative"
    ]
    oracle = json.loads(
        (Path(__file__).parent / "data/place_cartesian_selection_b0.json").read_text()
    )
    assert oracle["cartesian_calls"]["e1-franka-originalenv-v2"] == []
    assert place["qualify_required_waypoints"] is True
    assert place["lift_height"] == 0.1
    assert later[0]["recovery"]["planning_recovery"]["variant"] == "waypoint"
    (continuation,) = registration.planning_services.place_continuation_bindings
    assert (
        continuation.source_invocation_id
        == registration.invocation_policy_bindings[0].invocation_id
    )
    assert (
        continuation.place_invocation_id
        == registration.invocation_policy_bindings[1].invocation_id
    )

    evidence = registration.control_part_evidence_factory
    assert evidence.algorithm_revision == 1
    assert dict(evidence.object_bindings) == {"cylinder": "cylinder", "pad": "pad"}
    assert evidence.max_contacts_per_env == 1024
    assert evidence.trace_capacity == 4096
    assert (
        "control_part_evidence"
        not in yaml.safe_load((output / "components/embodiment.yaml").read_text())[
            "skill_profile"
        ]["runtime_services"]
    )
    with pytest.raises(FileExistsError):
        generate_task_program_bundle(
            graph,
            physical,
            output,
            robot_profile="dual_franka",
            implementation_revisions=revisions,
        )
    physical.assert_unchanged()


def test_generation_rejects_owner_drift_and_unavailable_modes_before_writing(tmp_path):
    values, physical = _frozen(tmp_path)
    graph = values["semantic_task_graph.json"]
    output = tmp_path / "must_not_exist"
    for kwargs in (
        {"fit_grasp_assets": True},
        {"max_episode_steps": 5},
        {"pick_contact_revision": 10},
        {"twist_contact_feedback": True},
    ):
        with pytest.raises(ValueError):
            generate_task_program_bundle(
                graph,
                physical,
                output,
                robot_profile="dual_franka",
                implementation_revisions={"embodichain": "test"},
                **kwargs,
            )
        assert not output.exists()
    drift = deepcopy(physical)
    drift.scene_simulation["rigid_object"][0]["init_pos"][0] += 1
    with pytest.raises(ValueError, match="data changed"):
        generate_task_program_bundle(
            graph,
            drift,
            output,
            robot_profile="dual_franka",
            implementation_revisions={"embodichain": "test"},
        )
    assert not output.exists()


def test_coordinated_recipe_declares_paired_grasp_and_native_evidence(tmp_path):
    _, physical = _frozen(tmp_path)
    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    graph = deepcopy(
        next(item["graph"] for item in cases if item["source_case"] == "e5-franka-r7")
    )
    # Reuse the frozen source call sequence with the test fixture's declared body.
    graph["nodes"][0]["call"]["arguments"]["object"] = "cylinder"
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "paired",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "test-source"},
        experimental_open_release=True,
    )
    declaration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    services = declaration["runtime_services"]
    (paired,) = services["planning_services"]["coordinated_grasps"]
    assert paired["left_grasp_target_id"] == "left_eef"
    assert paired["right_grasp_target_id"] == "right_eef"
    assert paired["contact"]["max_candidates"] == 32
    assert paired["contact"]["cargo_clearance_reserve"] == 0.008
    assert services["planning_services"]["capture_coordinated_cargo"] is True
    assert services["cargo_envelope"]["carrier_uids"] == ["cylinder"]
    recorded = services["recorded_contacts"]
    assert recorded["sensor_uid"] == "gen_sim_task_spec_contacts"
    assert recorded["retention"]["step_dt"] == 0.04
    assert len(recorded["opening_bindings"]) == 2
    embodiment = yaml.safe_load((paths.root / "components/embodiment.yaml").read_text())
    contact = next(
        value
        for value in embodiment["sensor"]
        if value["uid"] == recorded["sensor_uid"]
    )
    assert contact["filter_need_both_actor"] is False
    assert contact["max_contacts_per_env"] == 1024
    assert set(contact["rigid_uid_list"]) == {"cylinder", "pad", "table"}
    assert (
        yaml.safe_load((paths.root / "task_program_deployment.yaml").read_text())[
            "record_trajectory"
        ]
        is True
    )
    assert services["control_part_evidence"]["algorithm_revision"] == 2
    assert len(services["control_part_evidence"]["opening_bindings"]) == 2
    physical.assert_unchanged()


@pytest.mark.parametrize("defect", ["drift", "owner_conflict"])
def test_additional_frozen_inputs_cannot_drift_or_override_owner(tmp_path, defect):
    values, physical = _frozen(tmp_path)
    name = next(iter(physical.input_files))
    expected = "0" * 64
    if defect == "drift":
        path = tmp_path / "candidate.npy"
        path.write_bytes(b"changed frozen input")
        name = str(path)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="changed|conflicts"):
        generate_task_program_bundle(
            values["semantic_task_graph.json"],
            physical,
            output,
            robot_profile="dual_franka",
            implementation_revisions={"embodichain": "test"},
            external_input_digests={name: expected},
        )
    assert not output.exists()


def test_frozen_additional_input_is_rechecked_before_publication(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine import task_program_bundle as module

    values, physical = _frozen(tmp_path)
    cache = tmp_path / "candidates.npy"
    cache.write_bytes(b"frozen")
    inputs = {str(cache): hashlib.sha256(cache.read_bytes()).hexdigest()}
    prepare = module.prepare_task_program_bundle

    def drift_after_preflight(*args, **kwargs):
        result = prepare(*args, **kwargs)
        cache.write_bytes(b"changed")
        return result

    monkeypatch.setattr(module, "prepare_task_program_bundle", drift_after_preflight)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Frozen additional input changed"):
        generate_task_program_bundle(
            values["semantic_task_graph.json"],
            physical,
            output,
            robot_profile="dual_franka",
            external_input_digests=inputs,
            implementation_revisions={"embodichain": "test"},
        )
    assert not output.exists()


def _pour_bundle(
    tmp_path, revision, runtime_profile="default", *, proxy_tolerance=0.03
):
    _, physical = _frozen(tmp_path)
    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    source = next(
        item["graph"]
        for item in cases
        if item["source_case"] == "e3-dual_franka-originalenv-v2"
    )
    # The public source recipe is exercised on explicit fixture bodies, without
    # claiming that its physical feasibility is qualified by this static test.
    graph = json.loads(
        json.dumps(source)
        .replace('"bottle"', '"cylinder"')
        .replace('"receiver"', '"pad"')
    )
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "pour",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "test-source"},
        task_candidate={"draft": {"steps": [{"task_type": "E3"}]}},
        pour_geometry_revision=revision,
        runtime_profile=runtime_profile,
        pour_proxy_position_tolerance_m=proxy_tolerance if revision == 2 else None,
    )
    return graph, physical, paths


@pytest.mark.parametrize("runtime_profile", ["default", "auto"])
@pytest.mark.parametrize("revision", [None, 1, 2])
def test_pour_generation_preserves_static_and_opt_in_projection_modes(
    tmp_path, revision, runtime_profile
):
    graph, physical, paths = _pour_bundle(tmp_path, revision, runtime_profile)
    declaration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    services = declaration["runtime_services"]
    # Source pour-return policy revision 1 overrides only occurrence options.
    assert declaration["profile"]["action_options"]["place"]["lift_height"] == 0.05
    if runtime_profile == "auto":
        assert (
            json.loads((paths.root / "runtime_profile_selection.json").read_text())[
                "selected"
            ]
            == "composition_experiment"
        )
        assert services["control_part_evidence"]["algorithm_revision"] == 2
    lowerers = {item["kind"]: item for item in services["registered_semantic_lowerers"]}
    if revision is None:
        assert "jaw_pour" in lowerers
        assert lowerers["jaw_pour"]["receivers"][0][1] == "pad"
        assert "pour_projection_bindings" not in services["planning_services"]
        assert "world_orientation" in lowerers["move_held_object"]["routes"][0]["pose"]
    else:
        assert "retained_pour" in lowerers
        assert len(services["planning_services"]["pour_projection_bindings"]) == 1
        kind = "pour_transport" if revision == 1 else "pour_region_transport"
        assert lowerers[kind]["staging_routes"][0]["receiver_id"] == "pad"
    policy = yaml.safe_load(
        (paths.root / "components/execution_policy.yaml").read_text()
    )
    assert all(
        preset["release_gate_evidence"] == "binary_only"
        for preset in policy["presets"].values()
    )
    first_name = "gen_sim." + graph["nodes"][0]["id"] + ".execution"
    pick_options = declaration["profile"]["preset_options"][first_name][
        "action_options"
    ]["pick"]
    assert pick_options["require_qualified_approach"] is False
    assert "planning_recovery" not in policy["presets"][first_name].get("recovery", {})
    for node in graph["nodes"]:
        name = "gen_sim." + node["id"] + ".execution"
        budget = policy["presets"][name].get("recovery", {}).get("planning_budget", {})
        assert budget == (
            {"wall_seconds": 90.0}
            if revision is not None and node["call"].get("call_id") == "simulation.pour"
            else {}
        )
        if node["call"]["kind"] == "place":
            assert (
                declaration["profile"]["preset_options"][name]["action_options"][
                    "place"
                ]["lift_height"]
                == 0.12
            )
            assert (
                declaration["profile"]["preset_options"][name]["action_options"][
                    "place"
                ]["qualify_required_waypoints"]
                is False
            )
            assert "planning_recovery" not in policy["presets"][name].get(
                "recovery", {}
            )
    for node in graph["nodes"]:
        if node["call"].get("call_id") == "simulation.move_held_object":
            preset = "gen_sim." + node["id"] + ".execution"
            assert "planning_recovery" not in policy["presets"][preset].get(
                "recovery", {}
            )
            options = declaration["profile"]["preset_options"][preset][
                "action_options"
            ]["simulation.move_held_object"]
            # Both static/retained scene-relative staging and absolute return
            # routes bypassed the source batched-tensor-only lift prelude.
            assert options.get("lift_clearance") is None
            assert options["report_retained_attachment"] is True
    physical.assert_unchanged()


def test_generated_place_preset_keeps_phase_and_terminal_effects_separate(tmp_path):
    from lab.task_program.test_semantic_compiler import (
        _engine,
        _context,
        _held_context,
        _profile,
        _scene_registry,
    )
    from embodichain.lab.task_program.semantics import (
        BinaryEffectClause,
        Pick,
        Place,
        PoseRelationClause,
        PoseRelationExpectation,
        SceneObjectRef,
        SemanticPose,
        SceneManifest,
        SemanticIntegrationManifest,
    )
    from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler
    import torch

    graph, physical, paths = _pour_bundle(tmp_path, 2, proxy_tolerance=0.02)
    loaded = load_configured_bundle(
        paths.root,
        external_input_digests={
            **physical.input_files,
            **resolve_source_hand_services(physical.embodiment)[2],
        },
        implementation_revisions={"embodichain": "test-source"},
    )
    registration = loaded.deployment.integration.registration
    node = next(node for node in graph["nodes"] if node["call"]["kind"] == "place")
    selected = next(
        preset
        for preset in registration.robot_profile_binding.presets
        if preset.preset_id == "gen_sim." + node["id"] + ".execution"
    )
    # Reuse the production compiler's observed-scene test ports. The generated
    # public preset remains the option/monitor owner; no solver or Sim is run.
    registry, _ = _scene_registry()
    profile = replace(
        _profile(),
        presets={selected.preset_id: selected},
        default_preset=selected.preset_id,
    )
    engine = _engine(profile)
    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=profile,
        call_catalog=registration.call_catalog,
    )
    compiler = SemanticCallCompiler(manifest.bind(registry, engine))
    acquired = compiler.ground(
        compiler.analyze((Pick(object=SceneObjectRef("cube")),)),
        0,
        _context(registry),
    )
    held = _held_context(
        registry,
        acquired.invocation.goal.semantics,
        torch.eye(4).repeat(2, 1, 1),
    )
    grounded = compiler.ground(
        compiler.analyze(
            (
                Place(
                    object=SceneObjectRef("cube"),
                    at=SemanticPose((0.5, -0.2, 0.4), (0, 0, 0, 1)),
                ),
            )
        ),
        0,
        held,
    )
    (gate,) = grounded.effect_gates
    assert gate.gate_id == "source_released" and gate.segment_name == "retract"
    assert len(gate.effect_spec.clauses) == 1
    (constraint,) = gate.effect_spec.clauses
    assert type(constraint) is BinaryEffectClause and constraint.expected is False
    terminal_pose, terminal_constraint = grounded.effect_spec.clauses
    assert type(terminal_pose) is PoseRelationClause
    assert terminal_pose.expectation is PoseRelationExpectation.SEPARATED
    assert torch.equal(
        terminal_pose.baseline_object_to_endpoint, torch.eye(4).repeat(2, 1, 1)
    )
    assert type(terminal_constraint) is BinaryEffectClause
    assert terminal_constraint.expected is False
    (guard,) = grounded.effect_guards
    held_pose, held_constraint = guard.effect_spec.clauses
    assert type(held_pose) is PoseRelationClause
    assert held_pose.expectation is PoseRelationExpectation.MATCHED
    assert (
        type(held_constraint) is BinaryEffectClause and held_constraint.expected is True
    )
    acquired_pose, acquired_constraint = acquired.effect_gates[0].effect_spec.clauses
    assert type(acquired_pose) is PoseRelationClause
    assert acquired_pose.expectation is PoseRelationExpectation.MATCHED
    assert (
        type(acquired_constraint) is BinaryEffectClause
        and acquired_constraint.expected is True
    )
    physical.assert_unchanged()


def test_generated_pour_return_keeps_source_nominal_future_place(tmp_path, monkeypatch):
    from sim.atomic_actions._alignment_continuation_fixture import (
        continuation_fixture,
    )
    from embodichain.lab.sim.motion import mesh_collision
    import torch

    graph, physical, paths = _pour_bundle(tmp_path, 2, proxy_tolerance=0.02)
    loaded = load_configured_bundle(
        paths.root,
        external_input_digests={
            **physical.input_files,
            **resolve_source_hand_services(physical.embodiment)[2],
        },
        implementation_revisions={"embodichain": "test-source"},
    )
    node = next(node for node in graph["nodes"] if node["call"]["kind"] == "place")
    registration = loaded.deployment.integration.registration
    preset_id = "gen_sim." + node["id"] + ".execution"
    preset = next(
        item
        for item in registration.robot_profile_binding.presets
        if item.preset_id == preset_id
    )
    binding = next(
        item
        for item in registration.invocation_policy_bindings
        if item.preset_id == preset_id
    )
    # This is the effective public deployment value, after occurrence linking,
    # matching B0 InvocationPolicy._resolve for its exact Pour-return call.
    assert preset.action_option_template("place").lift_height == 0.12
    fixture_dir = tmp_path / "future_return"
    fixture_dir.mkdir()
    engine, invocation, context, session, check, calls, *_ = continuation_fixture(
        fixture_dir, "success"
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *args, **kwargs: check)
    assert engine.planning_services.place_scene_provider is None
    action_type = type(engine.actions["move_held_object"])
    original_complete = action_type.plan_complete_request
    previews = []

    def complete_with_future_return(
        action, request, current, *, build, external_nominal=False
    ):
        plan = original_complete(
            action, request, current, build=build, external_nominal=external_nominal
        )
        assert plan.plan_success.all()
        projected = replace(
            current.project(
                qpos=plan.joint_trajectory.positions[:, -1], task=current.task
            ),
            is_preview=True,
        )
        future = replace(
            session.resolve_place(projected),
            invocation_id=binding.invocation_id,
            skill_options=preset.action_option_template("place"),
            motion_policy=preset.motion_policy,
            recovery_policy=preset.recovery_policy,
        )
        preview = build.plan_preview(future, projected)
        # Observe the actual controlled Motion port, rather than only the
        # serialized option: its final retract target retains the source lift.
        retract_target = torch.as_tensor(calls[-1]["targets"][-1])
        torch.testing.assert_close(
            retract_target[:, 2, 3], future.goal.xpos[:, 2, 3] + 0.12
        )
        assert torch.equal(
            preview.joint_trajectory.positions[:, 0], projected.robot.qpos
        )
        assert torch.equal(
            preview.joint_trajectory.dt[:, 1:],
            preview.joint_trajectory.dt[:, 1:].new_full(
                preview.joint_trajectory.dt[:, 1:].shape,
                current.require_control_dt(),
            ),
        )
        assert tuple(segment.name for segment in preview.segments) == (
            "approach",
            "release",
            "retract",
        )
        previews.append(preview)
        return plan

    monkeypatch.setattr(
        action_type, "plan_complete_request", complete_with_future_return
    )
    plan = engine.plan(invocation, context)
    assert plan.plan_success.all()
    assert len(previews) == 1 and previews[0].plan_success.all()
    assert previews[0].invocation_id == binding.invocation_id
    assert dict(previews[0].expected_effects.held_object_updates) == {
        "source_arm": None
    }
    assert preset.action_option_template("place").qualify_required_waypoints is False
    assert preset.recovery_policy.planning_recovery is None
    assert preset.motion_policy.segment_sample_count_retries == (260, 320)
    assert preset.motion_policy.segment_continuous_ik_recovery is True
    physical.assert_unchanged()


def _alignment_bundle(tmp_path):
    _, physical = _frozen(tmp_path, slender=True)
    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    source = next(
        item["graph"]
        for item in cases
        if item["source_case"] == "e2-dual_franka-originalenv-v2"
    )
    graph = json.loads(
        json.dumps(source).replace('"interact_purple_soda_can"', '"cylinder"')
    )
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "alignment",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "test-source"},
        verify_pick_phases=True,
    )
    return graph, physical, paths


def test_alignment_recipe_selects_exact_staging_and_cartesian_occurrences(tmp_path):
    graph, physical, paths = _alignment_bundle(tmp_path)
    declaration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    services = declaration["runtime_services"]
    lowerers = {item["kind"]: item for item in services["registered_semantic_lowerers"]}
    assert (
        lowerers["pick"]["routes"][0]["release_clearance_plane_z"]
        == physical.scene.table_top_z
    )
    assert lowerers["pick"]["routes"][0]["release_clearance_safety_margin"] == 0.02
    assert lowerers["pick"]["verify_phases"] is True
    assert lowerers["align_held"]["verify_retention"] is True
    bindings = services["planning_services"]["held_staging_bindings"]
    assert len(bindings) == 2
    (continuation,) = services["planning_services"]["place_continuation_bindings"]
    assert continuation["kind"] == "held_alignment"
    assert continuation["source_invocation_id"] == bindings[-1]["invocation_id"]

    assert bindings[1]["predecessor_invocation_id"] == bindings[0]["invocation_id"]
    assert all(
        item["rigid_entity_id"] == "cylinder" and item["resource_id"] == "right"
        for item in bindings
    )
    selected = declaration["profile"]["preset_options"]
    oracle = json.loads(
        (Path(__file__).parent / "data/place_cartesian_selection_b0.json").read_text()
    )["cartesian_calls"]["e2-dual_franka-originalenv-v2"]
    for node in graph["nodes"][:-1]:
        options = next(
            iter(
                selected["gen_sim." + node["id"] + ".execution"][
                    "action_options"
                ].values()
            )
        )
        assert options["linear_segments"] is True
        if options["kind"] == "place":
            assert node["id"] in oracle
            assert options["qualify_required_waypoints"] is False
    assert not any(
        name.endswith("call_04.stable") and value.get("support_sensor")
        for name, value in services["settle_presets"].items()
    )
    physical.assert_unchanged()


def test_generated_cartesian_place_keeps_nominal_alignment_continuation(
    tmp_path, monkeypatch
):
    # Exercise the generated, publicly loaded future Place through the real
    # complete alignment planner. A flag-only assertion missed the incompatible
    # waypoint recovery declaration, which rejected this preview before IK.
    from embodichain.lab.sim.motion import mesh_collision
    from sim.atomic_actions._alignment_continuation_fixture import (
        continuation_fixture,
    )

    graph, physical, paths = _alignment_bundle(tmp_path)
    loaded = load_configured_bundle(
        paths.root,
        external_input_digests={
            **physical.input_files,
            **resolve_source_hand_services(physical.embodiment)[2],
        },
        implementation_revisions={"embodichain": "test-source"},
    )
    place_node = next(
        node
        for node in graph["nodes"]
        if node["call"].get("call_id") == "simulation.place_relative"
    )
    preset = next(
        item
        for item in loaded.deployment.integration.registration.robot_profile_binding.presets
        if item.preset_id == "gen_sim." + place_node["id"] + ".execution"
    )
    fixture_dir = tmp_path / "future_place"
    fixture_dir.mkdir()
    engine, invocation, context, session, check, *_ = continuation_fixture(
        fixture_dir, "success", enabled=True
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *args, **kwargs: check)
    original_resolve = session.resolve_place
    original_plan = session.plan
    previews = []

    def resolve_place(current):
        return replace(
            original_resolve(current),
            skill_options=preset.action_option_template("simulation.place_relative"),
            motion_policy=preset.motion_policy,
            recovery_policy=preset.recovery_policy,
        )

    def plan_place(request, current):
        result = original_plan(request, current)
        previews.append(result)
        return result

    session.resolve_place = resolve_place
    session.plan = plan_place
    result = engine.plan(invocation, context)
    assert result.plan_success.all(), dict(result.diagnostics.metadata)
    assert previews and all(plan.plan_success.all() for plan in previews)
    assert preset.recovery_policy.planning_recovery is None
    assert preset.motion_policy.segment_sample_count_retries == (260, 320)
    assert preset.motion_policy.segment_continuous_ik_recovery is False
    physical.assert_unchanged()


@pytest.mark.parametrize(
    "profile", ["default", "auto", "contact_experiment", "composition_experiment"]
)
def test_profiles_preserve_source_selection_and_evidence_defaults(tmp_path, profile):
    _, physical = _frozen(tmp_path)
    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    graph = deepcopy(
        next(item["graph"] for item in cases if item["source_case"] == "e5-franka-r7")
    )
    graph["nodes"][0]["call"]["arguments"]["object"] = "cylinder"
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "profile",
        robot_profile="dual_franka",
        runtime_profile=profile,
        implementation_revisions={"embodichain": "test-source"},
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    evidence = integration["runtime_services"]["control_part_evidence"]
    assert evidence["algorithm_revision"] == (1 if profile == "default" else 2)
    close = integration["profile"]["action_options"]["pick"]["hand_interp_steps"]
    assert close == (24 if profile in {"auto", "contact_experiment"} else 12)
    if profile == "auto":
        selected = json.loads(
            (paths.root / "runtime_profile_selection.json").read_text()
        )
        assert (
            selected["selected"] == "contact_experiment"
            and selected["rule"] == "non_pour_recipe"
        )
    physical.assert_unchanged()


def test_native_pick_declaration_uses_bound_final_place_and_keeps_other_calls_neutral(
    tmp_path,
):
    values, physical = _frozen(tmp_path)
    _, paths = generate_task_program_bundle(
        values["semantic_task_graph.json"],
        physical,
        tmp_path / "native",
        robot_profile="dual_franka",
        pick_contact_revision=8,
        implementation_revisions={"embodichain": "test-source"},
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    (binding,) = integration["runtime_services"]["planning_services"][
        "pick_closing_bindings"
    ]
    assert binding["object_id"] == "cylinder"
    assert binding["opening_reference_id"] == "pad"
    assert "opening_object_target" not in binding
    (continuation,) = integration["runtime_services"]["planning_services"][
        "place_continuation_bindings"
    ]
    assert binding["opening_place_invocation_id"] == continuation["place_invocation_id"]
    selected = [
        next(iter(value["action_options"].values()))
        for value in integration["profile"]["preset_options"].values()
    ]
    assert selected[0]["native_closing_revision"] == 8
    assert selected[0]["full_opening_clearance"] is False
    assert all("native_closing_revision" not in value for value in selected[1:])


@pytest.mark.parametrize("native", [False, True])
def test_handover_source_pick_does_not_select_ordinary_approach_or_unpaired_budget(
    tmp_path, native
):
    _, physical = _frozen(tmp_path, slender=True)
    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    graph = deepcopy(
        next(
            item["graph"]
            for item in cases
            if item["source_case"] == "e4-dual_franka-originalenv-v2"
        )
    )
    group = graph["task_groups"][-1]
    graph["nodes"] = [
        node for node in graph["nodes"] if node["id"] in group["node_ids"]
    ]
    graph["nodes"][0]["depends_on"] = []
    group["depends_on"] = []
    graph["task_groups"] = [group]
    graph["success"]["source"]["terms"] = [
        term
        for term in graph["success"]["source"]["terms"]
        if term["step_id"] == group["id"]
    ]
    graph = json.loads(
        json.dumps(graph).replace('"interact_orange_soda_can"', '"cylinder"')
    )
    name = graph["nodes"][0]["id"]
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "handover_source",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "test-source"},
        pick_contact_revisions={name: 8} if native else None,
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    policy = yaml.safe_load(
        (paths.root / "components/execution_policy.yaml").read_text()
    )
    preset = "gen_sim." + name + ".execution"
    options = next(
        iter(
            integration["profile"]["preset_options"][preset]["action_options"].values()
        )
    )
    assert options["require_qualified_approach"] is False
    assert options["full_opening_clearance"] is (not native)
    recovery = policy["presets"][preset].get("recovery", {})
    assert ("planning_recovery" in recovery) is native
    assert "planning_budget" not in recovery


@pytest.mark.parametrize(
    "robot_profile,native_trial", [("dual_franka", False), ("dual_ur10", True)]
)
def test_original_api_accepts_normal_prepared_scene_without_simulation(
    tmp_path, monkeypatch, robot_profile, native_trial
):
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.gen_sim.task_engine.embodiment_declarations import (
        resolve_source_hand_services,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("normal generation created simulation")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    values, physical = _frozen(tmp_path)
    scene = physical.scene
    graph = values["semantic_task_graph.json"]
    before = deepcopy(scene)
    output = tmp_path / "normal"
    _, paths = generate_task_program_bundle(
        graph,
        scene,
        output,
        robot_profile=robot_profile,
        native_pick_exploration=native_trial,
        max_episodes=1,
        max_episode_steps=10000,
        implementation_revisions={"embodichain": "source-test"},
    )
    import json

    envelope = json.loads((output / "integration_fingerprint.json").read_text())
    loaded = load_configured_bundle(
        output,
        external_input_digests=envelope["external_inputs"],
        implementation_revisions={"embodichain": "source-test"},
    )
    assert loaded.program.program_id == "gen_sim_demo"
    first = loaded.deployment.integration.registration.invocation_policy_bindings[0]
    profile = loaded.deployment.integration.registration.robot_profile_binding
    preset = next(
        value for value in profile.presets if value.preset_id == first.preset_id
    )
    assert preset.action_option_template("pick").native_scene_trial is native_trial
    assert scene == before
    assert paths.root == output
    assert not (output / "constraints.json").exists()
    assert (
        output.parent / ".task_engine_assets" / "mesh_assets" / "normalized"
    ).is_dir()


def test_explicit_post_handover_recipe_keeps_exact_receiver_release(tmp_path):
    _, physical = _frozen(tmp_path, slender=True)
    cases = json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
    graph = deepcopy(
        next(
            item["graph"]
            for item in cases
            if item["source_case"] == "e4-dual_franka-originalenv-v2"
        )
    )
    group = graph["task_groups"][-1]
    graph["nodes"] = [
        node for node in graph["nodes"] if node["id"] in group["node_ids"]
    ]
    graph["nodes"][0]["depends_on"] = []
    group["depends_on"] = []
    graph["task_groups"] = [group]
    graph["success"]["source"]["terms"] = [
        term
        for term in graph["success"]["source"]["terms"]
        if term["step_id"] == group["id"]
    ]
    graph = json.loads(
        json.dumps(graph).replace('"interact_orange_soda_can"', '"cylinder"')
    )
    transfer = graph["nodes"][-1]
    placed = deepcopy(transfer)
    placed.update(
        id="step_03__call_03",
        depends_on=[transfer["id"]],
        role="place",
        call=dict(
            kind="registered",
            call_id="simulation.place_relative",
            arguments=dict(object="cylinder", reference="pad", relation="on"),
            resources=dict(primary="right"),
        ),
    )
    graph["nodes"].append(placed)
    graph["task_groups"][0]["node_ids"].append(placed["id"])
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "after_transfer",
        robot_profile="dual_franka",
        runtime_profile="auto",
        implementation_revisions={"embodichain": "test-source"},
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    policy = yaml.safe_load(
        (paths.root / "components/execution_policy.yaml").read_text()
    )
    preset = "gen_sim." + placed["id"] + ".execution"
    selected = policy["presets"][preset]
    assert selected["motion"]["sample_count"] == 140
    assert selected["recovery"]["planning_recovery"]["variant"] == "retained_end_grip"
    options = integration["profile"]["preset_options"][preset]["action_options"][
        "simulation.place_relative"
    ]
    assert options["qualify_required_waypoints"] is False
    closing = integration["runtime_services"]["planning_services"][
        "pick_closing_bindings"
    ]
    assert len(closing) == 1
    assert (
        integration["runtime_services"]["planning_services"]["place_scene_bindings"][
            -1
        ]["support_owner_id"]
        is None
    )


def test_explicit_idle_clearance_routes_use_standard_free_move_options(tmp_path):
    values, physical = _frozen(tmp_path)
    graph = deepcopy(values["semantic_task_graph.json"])
    first, last = graph["nodes"][0], graph["nodes"][-1]
    clear, restore = deepcopy(first), deepcopy(last)
    clear.update(
        id="idle_clear",
        role="idle_clearance",
        call=dict(
            kind="registered",
            call_id="gen_sim.clear_idle_arm",
            arguments=dict(route="offset"),
            resources=dict(primary="right"),
        ),
    )
    restore.update(
        id="idle_restore",
        role="idle_restore",
        depends_on=[last["id"]],
        call=dict(
            kind="registered",
            call_id="simulation.park",
            arguments={},
            resources=dict(primary="right"),
        ),
    )
    first["depends_on"] = [clear["id"]]
    graph["nodes"] = [clear, *graph["nodes"], restore]
    graph["task_groups"][0]["node_ids"] = [
        clear["id"],
        *graph["task_groups"][0]["node_ids"],
        restore["id"],
    ]
    arm = physical.embodiment["simulation"]["control_parts"]["right_arm"]
    route = dict(
        route_id="offset",
        clear_segment="idle_clear",
        restore_segment="idle_restore",
        resource="right",
        joint_name=arm[0],
        offset_rad=0.6,
        search=True,
        protected_control_part="left_arm",
    )
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "idle",
        robot_profile="dual_franka",
        idle_clearance_routes=[route],
        implementation_revisions={"embodichain": "test-source"},
    )
    declaration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    selected = declaration["profile"]["preset_options"]
    for node, identity in (
        ("idle_clear", "gen_sim.clear_idle_arm"),
        ("idle_restore", "simulation.park"),
    ):
        assert (
            selected["gen_sim." + node + ".execution"]["action_options"][identity][
                "require_free_motion"
            ]
            is True
        )
    assert next(
        value
        for value in declaration["runtime_services"]["registered_semantic_lowerers"]
        if value["kind"] == "idle_arm_clearance"
    )["routes"] == [route]


def test_world_yaw_permission_is_exact_and_keeps_native_landing_owner(tmp_path):
    values, physical = _frozen(tmp_path)
    graph = deepcopy(values["semantic_task_graph.json"])
    release = graph["nodes"][1]
    release["call"] = {
        "kind": "place",
        "object": "cylinder",
        "inside": "inside__pad__cylinder",
        "resources": {"primary": "left"},
    }
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "yaw",
        robot_profile="dual_franka",
        world_yaw_free_places=[release["id"]],
        implementation_revisions={"embodichain": "test-source"},
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    policy = yaml.safe_load(
        (paths.root / "components/execution_policy.yaml").read_text()
    )
    preset = "gen_sim." + release["id"] + ".execution"
    assert (
        policy["presets"][preset]["recovery"]["planning_recovery"]["inside_world_yaw"]
        is True
    )
    binding = integration["runtime_services"]["planning_services"][
        "place_scene_bindings"
    ][0]
    assert (
        binding["inside_subject_id"] == "cylinder"
        and binding["inside_container_id"] == "inside__pad__cylinder"
    )
    with pytest.raises(ValueError, match="World yaw freedom"):
        generate_task_program_bundle(
            graph,
            physical,
            tmp_path / "bad_yaw",
            robot_profile="dual_franka",
            world_yaw_free_places=[graph["nodes"][0]["id"]],
            implementation_revisions={"embodichain": "test-source"},
        )


def test_prepare_generated_snapshot_keeps_publication_separate_and_seals_audits(
    tmp_path,
):
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        prepare_generated_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine.bundle_assembly import (
        publish_task_program_bundle,
    )

    values, physical = _frozen(tmp_path)
    output = tmp_path / "prepared"
    audit = {
        "preparation_audit.json": {"status": "prepared", "source": "explicit caller"}
    }
    before = deepcopy(audit)
    graph, snapshot = prepare_generated_task_program_bundle(
        values["semantic_task_graph.json"],
        physical,
        output,
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "test"},
        artifact_declarations=audit,
    )
    assert not output.exists()
    assert audit == before
    assert (
        json.loads(snapshot.files["preparation_audit.json"])
        == audit["preparation_audit.json"]
    )
    publish_task_program_bundle(snapshot, output)
    assert (output / "preparation_audit.json").is_file()
    for invalid in (
        {"task_spec.json": {"schema_version": "changed"}},
        {"source_input_provenance.json": {"schema_version": "injected"}},
    ):
        with pytest.raises(ValueError, match="canonical"):
            prepare_generated_task_program_bundle(
                values["semantic_task_graph.json"],
                physical,
                tmp_path / "invalid",
                robot_profile="dual_franka",
                implementation_revisions={"embodichain": "test"},
                artifact_declarations=invalid,
            )


def test_measured_rigid_generation_selects_exact_shared_atomic_owners(tmp_path):
    values, physical = _frozen(tmp_path)
    graph = deepcopy(values["semantic_task_graph.json"])
    release = graph["nodes"][1]
    release["call"] = {
        "kind": "place",
        "object": "cylinder",
        "inside": "inside__pad__cylinder",
        "resources": {"primary": "left"},
    }
    selected = [graph["nodes"][0]["id"], release["id"]]
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "measured",
        robot_profile="dual_franka",
        measured_rigid_calls=selected,
        implementation_revisions={"embodichain": "test-source"},
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    policy = yaml.safe_load(
        (paths.root / "components/execution_policy.yaml").read_text()
    )
    rows = integration["runtime_services"]["planning_services"][
        "measured_rigid_bindings"
    ]
    assert len(rows) == 2 and all(row["object_id"] == "cylinder" for row in rows)
    assert rows[1]["container_id"] == "inside__pad__cylinder"
    for node, identity, flag in (
        (selected[0], "pick", "measured_preshape"),
        (selected[1], "place", "measured_inside"),
    ):
        preset = "gen_sim." + node + ".execution"
        assert (
            integration["profile"]["preset_options"][preset]["action_options"][
                identity
            ][flag]
            is True
        )
        assert policy["presets"][preset]["recovery"]["planning_budget"] == {
            "wall_seconds": 90.0
        }
    with pytest.raises(ValueError, match="revision4"):
        generate_task_program_bundle(
            graph,
            physical,
            tmp_path / "bad_measured",
            robot_profile="dual_franka",
            measured_rigid_calls=selected,
            pick_contact_revision=8,
            implementation_revisions={"embodichain": "test-source"},
        )


@pytest.mark.parametrize("park_count", [1, 2])
def test_generated_released_cleanup_uses_exact_shared_suffix_authority(
    tmp_path, monkeypatch, park_count
):
    from embodichain.lab.sim.sim_manager import SimulationManager

    def forbidden(*args, **kwargs):
        raise AssertionError("declaration preflight constructed simulation")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    values, physical = _frozen(tmp_path)
    graph = deepcopy(values["semantic_task_graph.json"])
    pick, release, park = graph["nodes"]
    clear = deepcopy(park)
    clear.update(
        id="released_clearance",
        depends_on=[release["id"]],
        call=dict(
            kind="registered",
            call_id="gen_sim.clear_released",
            arguments=dict(object="cylinder", target="released_clearance_target"),
            resources=dict(primary="left"),
        ),
    )
    graph["targets"]["released_clearance_target"] = dict(
        kind="cyclic_pose",
        values=[dict(position=[0.2, 0.1, 0.8], quaternion_xyzw=[0, 0, 0, 1])],
    )
    park["depends_on"] = [clear["id"]]
    parks = [park]
    if park_count == 2:
        other = deepcopy(park)
        other.update(id="second_park", depends_on=[park["id"]])
        other["call"]["resources"]["primary"] = "right"
        parks.append(other)
    graph["nodes"] = [pick, release, clear, *parks]
    graph["task_groups"][0]["node_ids"] = [n["id"] for n in graph["nodes"]]
    _, paths = generate_task_program_bundle(
        graph,
        physical,
        tmp_path / "released",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "source-test"},
    )
    integration = yaml.safe_load(
        (paths.root / "task_program/integration.yaml").read_text()
    )
    policy = yaml.safe_load(
        (paths.root / "components/execution_policy.yaml").read_text()
    )
    (binding,) = integration["runtime_services"]["planning_services"][
        "released_hand_continuations"
    ]
    policies = {
        value["preset_id"]: value
        for value in integration["runtime_services"]["invocation_policy_bindings"]
    }
    assert (
        binding["release_invocation_id"]
        == policies["gen_sim." + release["id"] + ".execution"]["invocation_id"]
    )
    assert (
        binding["clearance_invocation_id"]
        == policies["gen_sim.released_clearance.execution"]["invocation_id"]
    )
    assert binding["park_invocation_ids"] == [
        policies["gen_sim." + node["id"] + ".execution"]["invocation_id"]
        for node in parks
    ]
    assert all(
        value["program_fingerprint"] == binding["program_fingerprint"]
        for value in policies.values()
    )
    clear_policy = policy["presets"]["gen_sim.released_clearance.execution"]
    assert clear_policy["recovery"]["planning_recovery"] == dict(
        kind="released_clearance", revision=1
    )
    assert clear_policy["motion"]["segment_sample_count_retries"] == [260, 320]
    assert clear_policy["motion"]["segment_continuous_ik_recovery"] is False
    for node in parks:
        assert policy["presets"]["gen_sim." + node["id"] + ".execution"]["recovery"][
            "planning_recovery"
        ] == dict(kind="released_park", revision=1)


@pytest.mark.parametrize(
    "value,error,message",
    [(1, TypeError, "boolean"), (True, ValueError, "only for dual UR10")],
)
def test_native_trial_public_preflight_retains_source_type_and_robot_scope(
    tmp_path, value, error, message
):
    values, physical = _frozen(tmp_path)
    with pytest.raises(error, match=message):
        generate_task_program_bundle(
            values["semantic_task_graph.json"],
            physical.scene,
            tmp_path / "invalid",
            robot_profile="dual_franka",
            native_pick_exploration=value,
        )
    assert not (tmp_path / "invalid").exists()


def test_normal_generation_refines_unrefined_source_before_alias_composition(
    tmp_path, monkeypatch
):
    from embodichain.lab.sim.sim_manager import SimulationManager

    def forbidden(*args, **kwargs):
        raise AssertionError("source target declaration started simulation")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    values, physical = _frozen(tmp_path, slender=True)
    graph = deepcopy(values["semantic_task_graph.json"])
    pick, release, park = graph["nodes"]
    pick["call"] = dict(
        kind="registered",
        call_id="simulation.pick",
        arguments=dict(object="cylinder", target="source_upright_target"),
        resources=dict(primary="left"),
    )
    align = deepcopy(pick)
    align.update(id="preserve_upright", depends_on=[pick["id"]])
    align["call"] = dict(
        kind="registered",
        call_id="gen_sim.align_held",
        arguments=dict(
            object="cylinder", target="current_object_pose", preserve_yaw=True
        ),
        resources=dict(primary="left"),
    )
    release["depends_on"] = [align["id"]]
    graph["nodes"] = [pick, align, release, park]
    graph["task_groups"][0]["node_ids"] = [n["id"] for n in graph["nodes"]]
    graph["targets"]["source_upright_target"] = dict(
        kind="cyclic_pose",
        values=[dict(position=[0.2, 0.1, 123.0], quaternion_xyzw=[0, 0, 0, 1])],
    )
    before = deepcopy(graph)
    selected, paths = generate_task_program_bundle(
        graph,
        physical.scene,
        tmp_path / "refined",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "source-test"},
    )
    assert selected["nodes"][0]["call"]["kind"] == "pick"
    assert selected["targets"]["source_upright_target"]["values"][0]["position"][
        2
    ] == pytest.approx(0.72, abs=1e-6)
    assert graph == before
    assert paths.root.is_dir()


def test_normal_generation_verifies_declared_asset_identity_before_source_geometry(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import task_program_bundle as module

    values, physical = _frozen(tmp_path)
    paths = {
        str(item["uid"]): Path(item["shape"]["fpath"])
        for item in physical.scene.rigid_objects
        if (item.get("shape") or {}).get("fpath")
    }
    prepared = replace(
        physical.scene,
        asset_hashes={
            uid: hashlib.sha256(path.read_bytes()).hexdigest()
            for uid, path in paths.items()
        },
    )
    next(iter(paths.values())).write_bytes(b"changed after source preparation")
    geometry_reads = []

    def geometry_before_identity(*args, **kwargs):
        geometry_reads.append(True)
        pytest.fail("source geometry was inspected before declared asset identity")

    monkeypatch.setattr(module, "prepare_source_drawer_graph", geometry_before_identity)
    with pytest.raises(
        ValueError, match="Source asset changed since scene preparation"
    ):
        module.prepare_generated_task_program_bundle(
            values["semantic_task_graph.json"],
            prepared,
            tmp_path / "bundle",
            robot_profile="dual_franka",
        )
    assert geometry_reads == []
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / ".task_engine_assets").exists()


@pytest.mark.parametrize("family", ["E1", "E4"])
def test_normal_native_marvin_bundle_uses_original_component_and_public_loaded_deployment(
    tmp_path, monkeypatch, family
):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

    def forbidden(*args, **kwargs):
        pytest.fail("bundle declaration created a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    values, reference = _frozen(tmp_path, slender=family == "E4")
    graph = values["semantic_task_graph.json"]
    if family == "E4":
        cases = json.loads(
            (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
        )["cases"]
        graph = deepcopy(
            next(
                item["graph"]
                for item in cases
                if item["source_case"] == "e4-dual_franka-originalenv-v2"
            )
        )
        group = graph["task_groups"][-1]
        graph["nodes"] = [
            node for node in graph["nodes"] if node["id"] in group["node_ids"]
        ]
        graph["nodes"][0]["depends_on"] = []
        group["depends_on"] = []
        graph["task_groups"] = [group]
        graph["success"]["source"]["terms"] = [
            term
            for term in graph["success"]["source"]["terms"]
            if term["step_id"] == group["id"]
        ]
        graph = json.loads(
            json.dumps(graph).replace('"interact_orange_soda_can"', '"cylinder"')
        )
    before = deepcopy(graph)
    output = tmp_path / "standard_marvin"
    _, paths = generate_task_program_bundle(
        graph,
        reference.scene,
        output,
        robot_profile="tianji_marvin",
        implementation_revisions={"embodichain": "native-standard-test"},
    )
    envelope = json.loads((output / "integration_fingerprint.json").read_text())
    loaded = load_configured_bundle(
        output,
        external_input_digests=envelope["external_inputs"],
        implementation_revisions={"embodichain": "native-standard-test"},
    )
    assert paths.root == output and graph == before
    embodiment = yaml.safe_load((output / "components/embodiment.yaml").read_text())
    assert embodiment["embodiment_id"] == "tianji_marvin"
    assert embodiment["simulation"]["class_type"] == "TianjiMarvin"
    registration = loaded.deployment.integration.registration
    assert registration.robot_profile_binding.profile_id == "tianji_marvin"
    bound_ids = [
        binding.invocation_id for binding in registration.invocation_policy_bindings
    ]
    expected_ids = {
        f"{loaded.program.program_id}/{segment.segment_id}:0"
        for segment in loaded.program.iter_segments()
    }
    assert set(bound_ids) == expected_ids and len(bound_ids) == len(expected_ids)
    evidence = registration.control_part_evidence_factory
    assert dict(evidence.pad_bindings) == {
        "left_eef": ("left_hand_finger_2_link", "left_hand_finger_1_link"),
        "right_eef": ("right_hand_finger_2_link", "right_hand_finger_1_link"),
    }


def test_source_hand_owner_input_is_rechecked_after_static_preflight(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import task_program_bundle as module

    values, physical = _frozen(tmp_path)
    calibration = tmp_path / "source_hand_calibration.yaml"
    calibration.write_bytes(
        yaml.safe_dump(physical.embodiment, sort_keys=False).encode()
    )
    expected = hashlib.sha256(calibration.read_bytes()).hexdigest()
    resolve = module.resolve_source_hand_services

    def declared(*args, **kwargs):
        owned, hands, inputs = resolve(*args, **kwargs)
        return owned, hands, {**inputs, str(calibration): expected}

    monkeypatch.setattr(module, "resolve_source_hand_services", declared)
    prepare = module.prepare_task_program_bundle

    def changed(*args, **kwargs):
        result = prepare(*args, **kwargs)
        calibration.write_bytes(b"changed source owner after preflight")
        return result

    monkeypatch.setattr(module, "prepare_task_program_bundle", changed)
    output = tmp_path / "rejected_hand"
    with pytest.raises(ValueError, match="Frozen source input changed"):
        generate_task_program_bundle(
            values["semantic_task_graph.json"],
            physical,
            output,
            robot_profile="dual_franka",
            implementation_revisions={"embodichain": "test"},
        )
    assert not output.exists()
