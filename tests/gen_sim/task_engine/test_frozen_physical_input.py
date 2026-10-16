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

"""Explicit physical inputs remain invariant under semantic composition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from .test_articulation_slide import _graph as drawer_graph
from .test_articulation_slide import scene as drawer_scene
from .test_press import button
from .test_press_integration import graph as press_graph
from .test_press_integration import scene as press_scene
from .test_semantic_graph_validation import _plan as coverage_plan
from .test_support_bundle import _load_bundle, _placement_graph, support_scene
from .test_twist import graph as twist_graph
from .test_twist import knob_scene
from embodichain.gen_sim.task_engine import task_program_bundle as bundle
from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
    FrozenPhysicalInput,
    _effective_articulation,
    load_frozen_physical_input,
    verify_physical_input_manifest,
)
from embodichain.gen_sim.task_engine.contracts import canonical_hash
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.semantic_graph_validation import (
    verify_graph_contract,
)
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program.language import load_task_program
from embodichain.utils.utility import load_config, save_config


@pytest.mark.parametrize("mode", ["overlay", "preserve"])
def test_effective_articulation_preserves_selected_scale_precision(
    knob_scene: PreparedScene, mode: str
) -> None:
    config = deepcopy(knob_scene.articulations[0])
    config["root_props"] = {"fixed_base": config.pop("fix_base")}
    declared_scale = 0.470352381468
    config.update(body_scale=[declared_scale] * 3, asset_physics_mode=mode)
    original = deepcopy(config)

    effective = _effective_articulation(config)

    expected = declared_scale if mode == "overlay" else 1.0
    assert effective["body_scale"] == [expected] * 3
    assert config == original


def _combined_graph(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(first)
    for field in ("nodes", "task_groups"):
        result[field].extend(deepcopy(second[field]))
    return result


def _control_graph(family: str, scene: PreparedScene) -> dict[str, Any]:
    # Inventory enrichment is allowed during planning, not on the frozen input.
    view = deepcopy(scene)
    if family == "E6":
        return drawer_graph(view)
    if family == "E9":
        return press_graph(view)
    return twist_graph(view)


def _freeze_reference(
    scene: PreparedScene, graph: dict[str, Any], output: Path
) -> FrozenPhysicalInput:
    _, paths = bundle.generate_task_program_bundle(
        graph,
        scene,
        output,
        robot_profile="dual_franka",
        max_episode_steps=30000,
    )
    frozen = load_frozen_physical_input(scene, paths.root, robot_profile="dual_franka")
    assert isinstance(frozen, FrozenPhysicalInput)
    assert frozen.scene_simulation == load_config(paths.scene)["simulation"]
    assert frozen.embodiment == load_config(paths.embodiment)
    assert frozen.execution_policy == load_config(paths.execution_policy)
    frozen.assert_unchanged()
    return frozen


@pytest.fixture
def frozen_rigid(support_scene: PreparedScene, tmp_path: Path) -> FrozenPhysicalInput:
    scene = deepcopy(support_scene)
    for item in (*scene.rigid_objects, *scene.planner_objects):
        if item.get("runtime_uid", item.get("uid")) == "bottle":
            item.setdefault("attrs", {}).setdefault("mass_props", {})["mass"] = 0.01
    return _freeze_reference(scene, _placement_graph(), tmp_path / "legacy_reference")


def _assert_physical_oracle(paths: Any, frozen: FrozenPhysicalInput) -> None:
    assert load_config(paths.scene)["simulation"] == frozen.scene_simulation
    assert load_config(paths.embodiment) == frozen.embodiment
    assert load_config(paths.execution_policy) == frozen.execution_policy
    deployment = load_config(paths.deployment)
    physical = {
        key: value
        for key, value in deployment.items()
        if key not in {"id", "scene", "embodiment", "task_program"}
    }
    assert physical == frozen.deployment
    resolved = _resolve_gym_components(deployment, base_dir=paths.root)
    assert resolved.config["robot"] == frozen.embodiment["simulation"]
    loaded = _load_bundle(paths)
    program = load_task_program(
        paths.program,
        integration=loaded.selection,
        validation_context=loaded.integration.registration.catalog,
    )
    loaded.integration.registration.catalog.preflight(program)
    frozen.assert_unchanged()


@pytest.mark.parametrize("family", ["E6", "E9", "E8"])
@pytest.mark.parametrize("prepend", [False, True])
def test_frozen_physics_survives_control_presence_and_reordering(
    support_scene: PreparedScene,
    drawer_scene: PreparedScene,
    press_scene: PreparedScene,
    knob_scene: PreparedScene,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    prepend: bool,
) -> None:
    control_scene = {"E6": drawer_scene, "E9": press_scene, "E8": knob_scene}[family]
    articulations = deepcopy(control_scene.articulations)
    for articulation in articulations:
        articulation["init_pos"] = [-0.25, 0.25, 0.8]
        articulation.pop("init_local_pose", None)
    scene = replace(
        deepcopy(support_scene),
        articulations=articulations,
        planner_objects=(
            *deepcopy(support_scene.planner_objects),
            *(
                {**deepcopy(item), "runtime_uid": item["uid"], "role": "articulation"}
                for item in articulations
            ),
        ),
    )
    neighbor = _placement_graph(reference="tray")
    reference_graph = _combined_graph(neighbor, _control_graph(family, scene))
    frozen = _freeze_reference(scene, reference_graph, tmp_path / "legacy_reference")
    expected_planner = deepcopy(frozen.scene.planner_objects)
    original_digest = frozen.physical_digest
    control = _control_graph(family, frozen.scene)
    mixed = (
        _combined_graph(control, neighbor)
        if prepend
        else _combined_graph(neighbor, control)
    )
    captured: list[tuple[dict[str, Any], ...]] = []
    original_integration = bundle._integration_payload

    def integration_payload(
        graph: dict[str, Any], current_scene: PreparedScene, **kwargs: Any
    ) -> dict[str, Any]:
        captured.append(deepcopy(current_scene.planner_objects))
        return original_integration(graph, current_scene, **kwargs)

    monkeypatch.setattr(bundle, "_integration_payload", integration_payload)
    if family == "E8":
        from embodichain.gen_sim.task_engine._task_program import (
            press_binding,
            twist_adaptation,
            twist_mass_source,
        )

        def forbidden_adaptation(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Frozen E8 inputs cannot trigger physical adaptation.")

        original_prepare = press_binding.prepare_press_scene

        def prepare_press_scene(
            current_scene: PreparedScene, routes: tuple, **kwargs: Any
        ) -> Any:
            if routes:
                return forbidden_adaptation()
            return original_prepare(current_scene, routes, **kwargs)

        monkeypatch.setattr(press_binding, "prepare_press_scene", prepare_press_scene)
        monkeypatch.setattr(
            twist_adaptation, "select_twist_scale", forbidden_adaptation
        )
        monkeypatch.setattr(
            twist_mass_source, "create_mass_source_copy", forbidden_adaptation
        )
    for label, graph in (
        ("without_control", neighbor),
        ("control_only", control),
        ("mixed", mixed),
    ):
        _, paths = bundle.generate_task_program_bundle(
            graph, frozen, tmp_path / label, robot_profile="dual_franka"
        )
        _assert_physical_oracle(paths, frozen)
    assert len(captured) == 3
    assert all(planner == expected_planner for planner in captured)
    assert frozen.scene.planner_objects == expected_planner
    assert frozen.physical_digest == original_digest


def test_frozen_rigid_preserves_explicit_legacy_effort(
    frozen_rigid: FrozenPhysicalInput, tmp_path: Path
) -> None:
    assert (
        frozen_rigid.embodiment["simulation"]["joint_drive_props"]["max_effort"][
            "left_eef"
        ]
        == 0.5
    )
    _, paths = bundle.generate_task_program_bundle(
        _placement_graph(),
        frozen_rigid,
        tmp_path / "frozen",
        robot_profile="dual_franka",
    )
    _assert_physical_oracle(paths, frozen_rigid)


@pytest.mark.parametrize(
    "options",
    [
        {"fit_grasp_assets": True},
        {"twist_mass_source": True},
        {"twist_grip_depth_m": 0.01},
        {"twist_support_lift_m": 0.002},
        {"max_episodes": 2},
        {"max_episode_steps": 500},
    ],
)
def test_frozen_input_rejects_adaptation_and_budget_overrides(
    frozen_rigid: FrozenPhysicalInput, tmp_path: Path, options: dict[str, Any]
) -> None:
    with pytest.raises((ValueError, TypeError)):
        bundle.generate_task_program_bundle(
            _placement_graph(),
            frozen_rigid,
            tmp_path / "conflicting_input",
            robot_profile="dual_franka",
            **options,
        )
    frozen_rigid.assert_unchanged()


@pytest.mark.parametrize(
    "field",
    ["scene", "scene_simulation", "embodiment", "execution_policy", "deployment"],
)
def test_frozen_input_rejects_nested_data_drift(
    frozen_rigid: FrozenPhysicalInput, tmp_path: Path, field: str
) -> None:
    if field == "scene":
        frozen_rigid.scene.planner_objects[0]["init_pos"][0] += 0.01
    elif field == "scene_simulation":
        frozen_rigid.scene_simulation["background"][0]["init_pos"][0] += 0.01
    elif field == "embodiment":
        frozen_rigid.embodiment["simulation"]["joint_drive_props"]["max_effort"][
            "left_eef"
        ] = 500.0
    elif field == "execution_policy":
        frozen_rigid.execution_policy["motion"]["sample_count"] += 1
    else:
        frozen_rigid.deployment["env"]["sim_steps_per_control"] += 1
    with pytest.raises((ValueError, RuntimeError)):
        frozen_rigid.assert_unchanged()
    with pytest.raises((ValueError, RuntimeError)):
        bundle.generate_task_program_bundle(
            _placement_graph(),
            frozen_rigid,
            tmp_path / "changed",
            robot_profile="dual_franka",
        )


def test_frozen_input_rejects_asset_bytes_drift(
    frozen_rigid: FrozenPhysicalInput, tmp_path: Path
) -> None:
    mesh = Path(frozen_rigid.scene.rigid_objects[0]["shape"]["fpath"])
    assert mesh.is_relative_to(tmp_path)
    assert str(mesh) in frozen_rigid.input_files
    mesh.write_bytes(mesh.read_bytes() + b"changed_fixture_asset")
    with pytest.raises((ValueError, RuntimeError)):
        frozen_rigid.assert_unchanged()
    with pytest.raises((ValueError, RuntimeError)):
        bundle.generate_task_program_bundle(
            _placement_graph(),
            frozen_rigid,
            tmp_path / "changed_asset",
            robot_profile="dual_franka",
        )


@pytest.fixture
def covered_frozen_bundle(frozen_rigid: FrozenPhysicalInput, tmp_path: Path) -> tuple:
    source = coverage_plan()
    bindings = deepcopy(source.role_bindings)
    bindings["reference_bindings"]["step_01.object"] = ["bottle"]
    plan = SemanticTaskPlanner().plan_spec(
        source.candidate,
        bindings,
        frozen_rigid.scene.planner_objects,
        task_spec=source.task_spec,
    )
    graph, paths = bundle.generate_task_program_bundle(
        plan.graph,
        frozen_rigid,
        tmp_path / "covered_frozen",
        robot_profile="dual_franka",
        task_spec=plan.task_spec,
        task_candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        step_lineage=plan.lineage,
    )
    contract = verify_graph_contract(paths.root, graph)
    manifest = verify_physical_input_manifest(paths.root)
    assert contract["coverage"]["status"] == "covered"
    assert manifest["physical_digest"] == frozen_rigid.physical_digest
    assert contract["physical_input_digest"] == canonical_hash(manifest)
    return graph, paths


@pytest.mark.parametrize(
    "damage", ["robot", "scene", "policy", "env", "asset", "marker"]
)
def test_ordinary_contract_reader_rejects_frozen_output_drift(
    covered_frozen_bundle: tuple, frozen_rigid: FrozenPhysicalInput, damage: str
) -> None:
    graph, paths = covered_frozen_bundle
    if damage == "marker":
        (paths.root / "physical_input.json").unlink()
    elif damage == "asset":
        path = Path(frozen_rigid.scene.rigid_objects[0]["shape"]["fpath"])
        path.write_bytes(path.read_bytes() + b"changed_runtime_asset")
    else:
        path = {
            "robot": paths.embodiment,
            "scene": paths.scene,
            "policy": paths.execution_policy,
            "env": paths.deployment,
        }[damage]
        value = load_config(path)
        if damage == "robot":
            value["simulation"]["joint_drive_props"]["max_effort"]["left_eef"] = 1.0
        elif damage == "scene":
            value["simulation"]["rigid_object"][0]["attrs"]["mass_props"]["mass"] = 0.02
        elif damage == "policy":
            value["tracking"]["terminal_max_abs_error"] *= 2
        else:
            value["env"]["sim_steps_per_control"] *= 2
        save_config(path, value)
    with pytest.raises(ValueError):
        verify_graph_contract(paths.root, graph)


def test_legacy_without_physical_marker_remains_unmodified(
    frozen_rigid: FrozenPhysicalInput, tmp_path: Path
) -> None:
    root = tmp_path / "legacy_reference"
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert verify_physical_input_manifest(root) is None
    assert (
        verify_graph_contract(root, load_config(root / "semantic_task_graph.json"))
        is None
    )
    assert {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    frozen_rigid.assert_unchanged()


@pytest.mark.parametrize("damage", ["source_uid", "duplicate_uid", "role", "pose"])
def test_frozen_import_rejects_inconsistent_identity_role_or_pose(
    frozen_rigid: FrozenPhysicalInput, tmp_path: Path, damage: str
) -> None:
    scene = deepcopy(frozen_rigid.scene)
    reference = tmp_path / "legacy_reference"
    if damage == "source_uid":
        scene.planner_objects[0]["runtime_uid"] = "missing_source_identity"
    elif damage == "duplicate_uid":
        scene = replace(
            scene,
            planner_objects=(
                *scene.planner_objects,
                deepcopy(scene.planner_objects[0]),
            ),
        )
    elif damage == "role":
        scene.planner_objects[0]["role"] = "articulation"
    else:
        path = reference / "components/scene.yaml"
        value = load_config(path)
        value["simulation"]["rigid_object"][0]["init_pos"][0] += 0.01
        save_config(path, value)
    with pytest.raises(ValueError, match="identities|role|pose"):
        load_frozen_physical_input(scene, reference)
