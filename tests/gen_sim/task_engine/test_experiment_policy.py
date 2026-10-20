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
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    load_task_engine_config,
)
from embodichain.gen_sim.task_engine.experiment_policy import program_completion_mask
from embodichain.gen_sim.task_engine._bundle_runner import _apply_task_spec_verdict
from embodichain.gen_sim.task_engine.workflow import _environment_successes


def runtime():
    return {
        "completed": True,
        "completed_by_env": [True],
        "success": [True],
        "lengths": [20],
        "segments": [
            {
                "success": True,
                "metadata": {
                    "segment_count": 1,
                    "runtime": {"failures": []},
                    "validation": {"accepted_mask": [True]},
                },
            }
        ],
        "task_spec_evaluation": {
            "environments": [
                {"env_id": 0, "status": "unavailable", "certificate_ready": False}
            ]
        },
    }


def test_experimental_acceptance_is_explicit_and_does_not_rewrite_taskspec():
    raw = runtime()
    saved = deepcopy(raw)
    assert _apply_task_spec_verdict(raw, [True], ["success"]) == (
        [False],
        ["task_spec_unverified"],
    )
    assert _apply_task_spec_verdict(
        raw, [True], ["success"], acceptance_mode="experiment"
    ) == ([True], ["success"])
    assert raw == saved


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "partial",
        "no_steps",
        "missing_segment",
        "failed_segment",
        "runtime_failure",
        "validator_failure",
    ],
)
def test_experimental_acceptance_never_waives_execution_failure(damage):
    raw = runtime()
    if damage == "empty":
        raw = {}
    elif damage == "partial":
        raw["completed"] = False
    elif damage == "no_steps":
        raw["lengths"] = [0]
    elif damage == "missing_segment":
        raw["segments"][0]["metadata"]["segment_count"] = 2
    elif damage == "failed_segment":
        raw["segments"][0]["success"] = False
    elif damage == "runtime_failure":
        raw["segments"][0]["metadata"]["runtime"]["failures"] = ["failed"]
    elif damage == "validator_failure":
        raw["segments"][0]["metadata"]["validation"]["accepted_mask"] = [False]
    assert program_completion_mask(raw, 1) == [False]


def test_workflow_experiment_policy_requires_runner_marker_and_semantic_success():
    report = {
        "status": "succeeded",
        "failure": None,
        "runtime_result": runtime(),
        "environments": [{"success": True, "semantic_success": {"step": True}}],
    }
    assert _environment_successes(report, acceptance_mode="experiment") == [False]
    report["runtime_result"]["acceptance_mode"] = "experiment"
    assert _environment_successes(
        report, required_semantic_steps=["step"], acceptance_mode="experiment"
    ) == [True]
    report["environments"][0]["semantic_success"]["step"] = False
    assert _environment_successes(
        report, required_semantic_steps=["step"], acceptance_mode="experiment"
    ) == [False]


def test_profile_and_acceptance_defaults_remain_legacy():
    assert TaskEngineExecutionCfg().acceptance_mode == "strict"
    assert TaskEnginePlanningCfg().runtime_profile == "default"
    with pytest.raises(ValueError):
        TaskEngineExecutionCfg(acceptance_mode="anything")
    with pytest.raises(ValueError):
        TaskEngineExecutionCfg(acceptance_mode="experiment", num_envs=2)
    with pytest.raises(ValueError):
        TaskEnginePlanningCfg(runtime_profile="task_special_case")


def test_packaged_defaults_enable_automatic_qualified_policy():
    workflow, planning, execution = load_task_engine_config()
    assert workflow.max_parallel_workers == 1
    assert workflow.max_scene_attempts == 1
    assert workflow.max_action_attempts == 1
    assert planning.runtime_profile == "auto"
    assert planning.experimental_open_release is False
    assert execution.acceptance_mode == "experiment"


def test_packaged_contact_defaults_reach_generated_bundle(support_scene, tmp_path):
    from .test_support_bundle import _placement_graph, _load_bundle
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config

    _, planning, _ = load_task_engine_config()
    _, paths = generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "default_contact",
        robot_profile="dual_franka",
        runtime_profile=planning.runtime_profile,
        experimental_open_release=planning.experimental_open_release,
    )
    _load_bundle(paths)
    assert load_config(paths.embodiment)["simulation"]["link_attrs"][
        "gen_sim_contact_pads"
    ]["attrs"]["material_props"] == {
        "static_friction": 2.85,
        "dynamic_friction": 2.7,
    }
    assert (
        load_config(paths.integration)["profile"]["action_options"]["pick"][
            "hand_interp_steps"
        ]
        == 24
    )
    constraints = load_config(paths.root / "task_program/constraints.json")
    assert constraints["runtime_profile"] == "contact_experiment"
    assert set(constraints["motion_samples"].values()) == {140, 152}
    assert load_config(paths.root / "runtime_profile_selection.json")["selected"] == (
        "contact_experiment"
    )


@pytest.mark.parametrize(
    ("task_type", "selected"),
    [
        ("E1", "contact_experiment"),
        ("E4", "contact_experiment"),
        ("E3", "composition_experiment"),
    ],
)
def test_auto_profile_uses_recipe_semantics_not_task_or_object_ids(task_type, selected):
    from embodichain.gen_sim.task_engine.experiment_policy import resolve_profile

    graph = {
        "task_id": "arbitrary",
        "nodes": [
            {"task_type": task_type, "call": {"kind": "pick", "object": "arbitrary"}}
        ],
    }
    saved = deepcopy(graph)
    assert resolve_profile("auto", graph) == selected
    assert graph == saved
    graph["task_id"] = "another-task"
    graph["nodes"][0]["call"]["object"] = "another-object"
    assert resolve_profile("auto", graph) == selected


@pytest.mark.parametrize(
    "profile", ["default", "contact_experiment", "composition_experiment"]
)
def test_explicit_profile_is_not_overridden_by_automatic_selection(profile):
    from embodichain.gen_sim.task_engine.experiment_policy import resolve_profile

    graph = {"nodes": [{"task_type": "E3", "call": {"call_id": "simulation.pour"}}]}
    assert resolve_profile(profile, graph) == profile


def test_auto_pour_bundle_keeps_original_materials_and_closure(tmp_path):
    from .test_pour_region_bundle import _case, _generate
    from .test_support_bundle import _load_bundle
    from embodichain.utils.utility import load_config

    scene, graph, candidate = _case(tmp_path)
    _, paths = _generate(
        tmp_path / "auto_pour",
        scene,
        graph,
        candidate,
        runtime_profile="auto",
        pour_geometry_revision=2,
        pour_proxy_position_tolerance_m=0.02,
    )
    _load_bundle(paths)
    assert "gen_sim_contact_pads" not in load_config(paths.embodiment)[
        "simulation"
    ].get("link_attrs", {})
    assert (
        load_config(paths.integration)["profile"]["action_options"]["pick"][
            "hand_interp_steps"
        ]
        == 12
    )
    constraints = load_config(paths.root / "task_program/constraints.json")
    assert constraints["runtime_profile"] == "composition_experiment"
    assert set(constraints["motion_samples"].values()) == {140}
    assert load_config(paths.root / "runtime_profile_selection.json")["selected"] == (
        "composition_experiment"
    )


def test_bundle_constraints_reject_an_unresolved_automatic_profile(
    support_scene, tmp_path
):
    from .test_support_bundle import _placement_graph, _load_bundle
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config, save_config

    _, paths = generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "unresolved",
        robot_profile="dual_franka",
    )
    path = paths.root / "task_program/constraints.json"
    constraints = load_config(path)
    constraints["runtime_profile"] = "auto"
    save_config(path, constraints)
    with pytest.raises(ValueError, match="concrete runtime profile"):
        _load_bundle(paths)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("profile", ["default", "auto"])
def test_authorized_receiver_place_does_not_inherit_transfer_cartesian_mode(
    tmp_path, combined, profile
):
    from .test_semantic_graph import _fresh_handover_graph, _prepared_axis_scene
    from .test_support_bundle import _load_bundle
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config

    scene = _prepared_axis_scene(tmp_path, source_extents=(0.24, 0.06, 0.06))
    graph = _fresh_handover_graph()
    graph["nodes"][0]["call"] = {
        "kind": "pick",
        "object": "bottle",
        "resources": {"primary": "left"},
    }
    graph["nodes"][1]["call"]["object"] = "bottle"
    graph["nodes"].append(
        {
            "id": "receiver_place",
            "task_type": "E4" if combined else "E1",
            "task_instance_id": "exchange" if combined else "landing",
            "role": "primary",
            "depends_on": ["place_cube"],
            "call": {
                "kind": "registered",
                "call_id": "simulation.place_relative",
                "arguments": {
                    "object": "bottle",
                    "reference": "table",
                    "relation": "on",
                },
                "resources": {"primary": "right"},
            },
        }
    )
    if combined:
        graph["task_groups"][0]["node_ids"].append("receiver_place")
    else:
        graph["task_groups"].append(
            {
                "id": "landing",
                "task_type": "E1",
                "node_ids": ["receiver_place"],
                "depends_on": ["exchange"],
                "success": {"kind": "call_completed"},
            }
        )
    _, paths = generate_task_program_bundle(
        graph,
        scene,
        tmp_path / "scoped",
        robot_profile="dual_franka",
        runtime_profile=profile,
    )
    _load_bundle(paths)
    cfg = load_config(paths.root / "task_program/constraints.json")
    assert ("receiver_place" in cfg["cartesian_calls"]) is (
        combined and profile == "default"
    )
    assert cfg["motion_samples"]["receiver_place"] == (
        260 if combined and profile == "default" else 140
    )
    assert "place_cube" in cfg["cartesian_calls"]
    assert cfg["motion_samples"]["place_cube"] == 260


@pytest.mark.parametrize("profile", ["contact_experiment", "composition_experiment"])
def test_contact_bundle_uses_production_decoder_without_task_ids(
    support_scene, tmp_path: Path, profile: str
):
    from .test_support_bundle import _placement_graph, _load_bundle
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config

    _, paths = generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "contact",
        robot_profile="dual_franka",
        runtime_profile=profile,
    )
    loaded = _load_bundle(paths)
    cfg = load_config(paths.embodiment)
    overlays = cfg["simulation"].get("link_attrs", {})
    if profile == "contact_experiment":
        assert overlays["gen_sim_contact_pads"]["attrs"] == {
            "material_props": {"static_friction": 2.85, "dynamic_friction": 2.7}
        }
    else:
        assert "gen_sim_contact_pads" not in overlays
    assert load_config(paths.integration)["profile"]["action_options"]["pick"][
        "hand_interp_steps"
    ] == (24 if profile == "contact_experiment" else 12)
    constraints = load_config(paths.root / "task_program/constraints.json")
    assert set(constraints["motion_samples"].values()) == (
        {140, 152} if profile == "contact_experiment" else {140}
    )
    assert loaded.integration.adapter_factory.constraints


from .test_support_bundle import support_scene


def test_handover_permissions_are_adjacent_occurrences_not_object_wide():
    from embodichain.gen_sim.task_engine.experiment_policy import (
        horizontal_handover_permissions,
    )

    graph = {
        "nodes": [
            {
                "id": "source",
                "call": {
                    "kind": "registered",
                    "call_id": "gen_sim.pick.handover_horizontal_source.payload",
                    "arguments": {"object": "payload"},
                },
            },
            {
                "id": "transfer",
                "call": {
                    "kind": "hand_over",
                    "object": "payload",
                    "resources": {"source": "right", "destination": "left"},
                },
            },
            {
                "id": "received_place",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {"object": "payload", "relation": "on"},
                    "resources": {"primary": "left"},
                },
            },
            {"id": "reacquire", "call": {"kind": "pick", "object": "payload"}},
            {
                "id": "ordinary_place",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {"object": "payload", "relation": "on"},
                    "resources": {"primary": "left"},
                },
            },
        ]
    }
    places, revisions = horizontal_handover_permissions(graph)
    assert places == ["received_place"]
    assert revisions == {"source": 8}


def test_contact_profile_cannot_modify_frozen_physical_components(
    support_scene, tmp_path
):
    from .test_support_bundle import _placement_graph
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
        load_frozen_physical_input,
    )

    graph = _placement_graph(reference="tray")
    _, paths = generate_task_program_bundle(
        graph, support_scene, tmp_path / "base", robot_profile="dual_franka"
    )
    frozen = load_frozen_physical_input(
        support_scene, paths.root, robot_profile="dual_franka"
    )
    with pytest.raises(ValueError, match="Frozen physical"):
        generate_task_program_bundle(
            graph,
            frozen,
            tmp_path / "invalid",
            robot_profile="dual_franka",
            runtime_profile="contact_experiment",
        )


def test_merged_contact_permissions_preserve_independent_occurrences():
    from embodichain.gen_sim.task_engine.experiment_policy import (
        merge_contact_revisions,
    )

    selected = {"upright_pick": 2}
    generated = {"handover_pick": 8}
    assert merge_contact_revisions(selected, generated) == {
        "handover_pick": 8,
        "upright_pick": 2,
    }
    assert merge_contact_revisions(selected, {}) == selected
    assert selected == {"upright_pick": 2}
    assert generated == {"handover_pick": 8}


def test_merged_contact_permissions_reject_incompatible_same_occurrence():
    from embodichain.gen_sim.task_engine.experiment_policy import (
        merge_contact_revisions,
    )

    assert merge_contact_revisions({"pick": 8}, {"pick": 8}) == {"pick": 8}
    with pytest.raises(ValueError, match="pick.*2.*8"):
        merge_contact_revisions({"pick": 2}, {"pick": 8})


@pytest.mark.parametrize("with_handover", [False, True])
def test_auto_profile_preserves_explicit_upright_contact_revision(
    support_scene, tmp_path, with_handover
):
    import trimesh

    from .test_support_bundle import _placement_graph, _load_bundle
    from .test_semantic_graph import _fresh_handover_graph
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.utils.utility import load_config

    # Exported assets are Y-up; keep a wide floor after runtime normalization.
    trimesh.creation.box(extents=(0.5, 0.02, 0.5)).export(
        support_scene.rigid_objects[-1]["shape"]["fpath"]
    )
    graph = _placement_graph(inside=True, reference="tray")
    expected = {"pick": 2}
    if with_handover:
        handover = _fresh_handover_graph()
        handover["nodes"][0]["call"].update(
            call_id="gen_sim.pick.handover_horizontal_source.bottle",
            arguments={"object": "bottle", "target": "source_grasp"},
        )
        handover["nodes"][0]["depends_on"] = ["place"]
        handover["nodes"][1]["call"]["object"] = "bottle"
        handover["task_groups"][0]["depends_on"] = ["support_group"]
        graph["nodes"].extend(handover["nodes"])
        graph["task_groups"].extend(handover["task_groups"])
        expected[handover["nodes"][0]["id"]] = 8
    _, paths = generate_task_program_bundle(
        graph,
        support_scene,
        tmp_path / "auto_revision",
        robot_profile="dual_franka",
        runtime_profile="auto",
        pick_contact_revisions={"pick": 2},
    )
    assert (
        load_config(paths.root / "task_program/constraints.json")[
            "pick_contact_revisions"
        ]
        == expected
    )
    assert len(
        _load_bundle(paths).integration.adapter_factory.pick_contact_revisions
    ) == len(expected)


@pytest.mark.parametrize(
    "profile", ["auto", "contact_experiment", "composition_experiment"]
)
def test_frozen_input_profile_rejected_during_configuration(profile):
    with pytest.raises(ValueError, match="physical_reference.*runtime_profile"):
        TaskEnginePlanningCfg(
            physical_reference="selected_bundle", runtime_profile=profile
        )


@pytest.mark.parametrize("invalid", [0, 1, "false", [], {}])
def test_release_override_rejects_non_boolean_configuration(invalid):
    with pytest.raises(TypeError, match="experimental_open_release"):
        TaskEnginePlanningCfg(experimental_open_release=invalid)


@pytest.mark.parametrize("override", [None, False, True])
def test_release_override_is_independent_of_contact_strategy_and_tolerances(
    support_scene, tmp_path, monkeypatch, override
):
    from .test_support_bundle import _placement_graph, _load_bundle
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine._task_program import grip_evidence
    from embodichain.utils.utility import load_config

    original = grip_evidence.with_native_grip_evidence
    observed = []

    def record(*args, **kwargs):
        observed.append(kwargs["experimental_open_release"])
        return original(*args, **kwargs)

    monkeypatch.setattr(grip_evidence, "with_native_grip_evidence", record)
    _, paths = generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "release",
        robot_profile="dual_franka",
        runtime_profile="auto",
        experimental_open_release=override,
    )
    _load_bundle(paths)
    expected = override if override is not None else True
    assert observed and all(value is expected for value in observed)
    constraints = load_config(paths.root / "task_program/constraints.json")
    if override is None:
        assert "experimental_open_release" not in constraints
    else:
        assert constraints["experimental_open_release"] is override
    placements = [
        cfg
        for cfg in constraints["presets"].values()
        if cfg.get("kind") == "supported_placement"
    ]
    assert placements
    assert all(
        (cfg["position_tolerance"], cfg["com_tolerance"], cfg["support_tolerance"])
        == (0.10, 0.10, 0.03)
        for cfg in placements
    )
    assert (
        load_config(paths.integration)["profile"]["action_options"]["pick"][
            "hand_interp_steps"
        ]
        == 24
    )


def test_program_completion_summary_cannot_replace_strict_failure():
    from embodichain.gen_sim.task_engine.experiment_policy import (
        program_completion_summary,
    )

    raw = runtime()
    strict = {"environments": [{"success": False}], "failure": None}
    experiment = {"environments": [{"success": True}], "failure": None}
    summary = program_completion_summary(
        raw, strict, experiment, acceptance_mode="strict"
    )
    assert summary["experimental_program_completed_by_env"] == [True]
    assert summary["strict_execution_success_by_env"] == [False]
    assert summary["task_spec_certificate_ready_by_env"] == [False]
    assert summary["affects_selected_success"] is False
    assert strict["environments"][0]["success"] is False


@pytest.mark.parametrize("failure", [False, True])
def test_program_completion_summary_keeps_execution_failure_separate(failure):
    from embodichain.gen_sim.task_engine.experiment_policy import (
        program_completion_summary,
    )

    raw = runtime()
    raw["task_spec_evaluation"]["environments"][0].update(
        status="satisfied", certificate_ready=True
    )
    report = {
        "environments": [{"env_id": 0, "success": not failure}],
        "failure": {"message": "execution failed"} if failure else None,
    }
    summary = program_completion_summary(raw, report, report, acceptance_mode="strict")
    assert summary["experimental_program_completed_by_env"] == [not failure]
    assert summary["strict_execution_success_by_env"] == [not failure]
    assert summary["task_spec_certificate_ready_by_env"] == [True]


@pytest.mark.parametrize("rejection", ["cargo", "semantic"])
def test_program_completion_summary_preserves_experiment_postconditions(rejection):
    from embodichain.gen_sim.task_engine.experiment_policy import (
        program_completion_summary,
    )

    strict = {"environments": [{"success": False}], "failure": None}
    experiment = {
        "environments": [
            {
                "success": rejection != "cargo",
                "semantic_success": {"step": rejection != "semantic"},
            }
        ],
        "failure": None,
    }
    summary = program_completion_summary(
        runtime(), strict, experiment, acceptance_mode="experiment"
    )
    assert summary["experimental_program_completed_by_env"] == [False]


def test_release_override_is_fingerprint_bound_and_strictly_decoded(
    support_scene, tmp_path
):
    import json

    from .test_support_bundle import _placement_graph, _load_bundle
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )

    graphs, deployments = [], []
    for override in (False, True):
        graph, paths = generate_task_program_bundle(
            _placement_graph(reference="tray"),
            support_scene,
            tmp_path / str(override),
            robot_profile="dual_franka",
            runtime_profile="auto",
            experimental_open_release=override,
        )
        graphs.append(graph)
        deployments.append(paths)
    assert graphs[0]["integration_fingerprint"] != graphs[1]["integration_fingerprint"]
    path = deployments[0].root / "task_program/constraints.json"
    payload = json.loads(path.read_text())
    payload["experimental_open_release"] = "false"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="experimental_open_release"):
        _load_bundle(deployments[0])
