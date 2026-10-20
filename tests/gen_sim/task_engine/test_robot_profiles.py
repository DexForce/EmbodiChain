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
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from embodichain.gen_sim.task_engine import robot_profiles as profiles
from embodichain.gen_sim.task_engine.cli import build_parser
from embodichain.utils.utility import load_config
from .test_press import button
from .test_press_integration import scene as press_scene


@pytest.mark.parametrize(
    "profile", ["dual_franka", "dual_ur10", "dual_tianji", "tianji_marvin"]
)
def test_cli_selects_robot_profile(profile: str) -> None:
    args = build_parser().parse_args(
        [
            "run-all",
            "--mode",
            "scene",
            "--task-id",
            "test",
            "--instruction",
            "test",
            "--scene",
            "scene",
            "--output-root",
            "output",
            "--robot-profile",
            profile,
        ]
    )
    assert args.robot_profile == profile


def test_franka_embodiment_is_not_modified(tmp_path: Path) -> None:
    expected = load_config(profiles._FRANKA_COMPONENT)
    actual = profiles._load_robotiq_embodiment("dual_franka", tmp_path)
    assert actual == expected
    assert not list(tmp_path.iterdir())


def test_unknown_profile_fails_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="robot_profile"):
        profiles._load_robotiq_embodiment("not_a_robot", tmp_path)
    assert not list(tmp_path.iterdir())


def test_ur10_preserves_complete_robotiq_contract(tmp_path: Path) -> None:
    original = profiles._load_robotiq_embodiment("dual_franka", tmp_path)
    selected = profiles._load_robotiq_embodiment("dual_ur10", tmp_path)
    for side in ("left", "right"):
        assert len(selected["simulation"]["control_parts"][f"{side}_arm"]) == 6
        assert (
            selected["simulation"]["control_parts"][f"{side}_eef"]
            == original["simulation"]["control_parts"][f"{side}_eef"]
        )
        solver = selected["simulation"]["solver_cfg"][f"{side}_arm"]
        assert solver["class_type"] == "URSolver"
        assert solver["ur_type"] == "ur10"
    assert (
        selected["skill_profile"]["runtime_services"]
        == original["skill_profile"]["runtime_services"]
    )
    commands = {p["preset_id"]: p for p in selected["skill_profile"]["command_presets"]}
    assert len(commands["left_arm_postures"]["commands"]["park"]) == 6
    assert len(selected["simulation"]["init_qpos"]) == 24
    assert profiles._load_robotiq_embodiment("dual_franka", tmp_path) == original


def test_new_mount_uses_arm_anchor_not_franka_root_offset(tmp_path: Path) -> None:
    selected = profiles._load_robotiq_embodiment("dual_ur10", tmp_path)
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _bind_embodiment_to_scene,
    )

    _bind_embodiment_to_scene(selected, table_top_z=0.8)
    assert selected["simulation"]["init_pos"][2] == pytest.approx(0.85)


def test_ur10_initial_and_park_tcp_face_the_table(tmp_path: Path) -> None:
    import torch
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.robots import URRobotCfg
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _bind_embodiment_to_scene,
    )

    selected = profiles._load_robotiq_embodiment("dual_ur10", tmp_path)
    _bind_embodiment_to_scene(selected, table_top_z=0.8)
    simulation = selected["simulation"]
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler(
        "XYZ", simulation["init_rot"], degrees=True
    ).as_matrix()
    base[:3, 3] = simulation["init_pos"]
    chain = URRobotCfg.from_dict({"robot_type": "ur10"}).build_pk_serial_chain()["arm"]
    presets = {
        p["control_part"]: p["commands"]
        for p in selected["skill_profile"]["command_presets"]
    }
    for index, side in enumerate(("left", "right")):
        part = f"{side}_arm"
        home = presets[part]["park"]
        assert simulation["init_qpos"][index:12:2] == home
        arm_base = base.copy()
        arm_base[:3, 3] += base[:3, :3] @ profiles._arm_base_position(selected, side)
        tcp = (
            arm_base
            @ chain.forward_kinematics(torch.tensor([home])).get_matrix()[0].numpy()
        )
        tcp = tcp @ np.asarray(simulation["solver_cfg"][part]["tcp"])
        toward_table = -arm_base[:2, 3]
        reach = tcp[:2, 3] - arm_base[:2, 3]
        assert (
            float(toward_table @ reach) > 0
        ), f"{side} UR10 points away from the table"


def test_new_profile_mutations_do_not_leak(tmp_path: Path) -> None:
    first = profiles._load_robotiq_embodiment("dual_ur10", tmp_path)
    expected = deepcopy(first)
    first["skill_profile"]["runtime_services"]["grasp_pose_generators"]["left_eef"][
        "model"
    ]["max_opening_width"] = 1.0
    assert profiles._load_robotiq_embodiment("dual_ur10", tmp_path) == expected


def test_ur10_fk_chain_matches_solver_and_gripper(tmp_path: Path) -> None:
    from embodichain.lab.sim.cfg import RobotCfg
    import pytorch_kinematics as pk

    selected = profiles._load_robotiq_embodiment("dual_ur10", tmp_path)
    cfg = RobotCfg.from_dict(selected["simulation"])
    path = Path(cfg.urdf_cfg.assemble_urdf())
    root = ET.parse(path).getroot()
    links = {link.get("name") for link in root.findall("link")}
    assert {
        "left_inner_finger_pad",
        "left_right_inner_finger_pad",
        "right_inner_finger_pad",
        "right_left_inner_finger_pad",
    } <= links
    for side in ("left", "right"):
        solver = cfg.solver_cfg[f"{side}_arm"]
        chain = pk.build_serial_chain_from_urdf(
            path.read_text(), solver.end_link_name, solver.root_link_name
        )
        assert chain.get_joint_parameter_names() == cfg.control_parts[f"{side}_arm"]
        assert np.asarray(solver.tcp).shape == (4, 4)


@pytest.fixture
def marvin_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic fourteen-joint body tests assembly, not real Marvin geometry."""
    from embodichain import data
    from embodichain.lab.sim.robots import tianji_marvin

    root = ET.Element("robot", name="marvin_fixture")
    ET.SubElement(root, "link", name="body")
    for side, label, lateral in (("left", "L", 0.25), ("right", "R", -0.25)):
        base = f"{side}_arm_base"
        ET.SubElement(root, "link", name=base)
        fixed = ET.SubElement(root, "joint", name=f"{side}_mount", type="fixed")
        ET.SubElement(fixed, "parent", link="body")
        ET.SubElement(fixed, "child", link=base)
        ET.SubElement(fixed, "origin", xyz=f"0 {lateral} 0.8", rpy="0 0 0")
        parent = base
        prefixes = (
            "SHOULDER_PITCH",
            "SHOULDER_ROLL",
            "ELBOW_PITCH",
            "ELBOW_YAW",
            "WRIST_PITCH",
            "WRIST_YAW",
            "WRIST_ROLL",
        )
        for index, prefix in enumerate(prefixes, 1):
            child = f"{side}_ee" if index == 7 else f"{side}_link_{index}"
            ET.SubElement(root, "link", name=child)
            joint = ET.SubElement(
                root, "joint", name=f"{prefix}_{label}_J{index}", type="revolute"
            )
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(joint, "origin", xyz="0 0 0.1", rpy="0 0 0")
            ET.SubElement(
                joint, "limit", lower="-3.14", upper="3.14", effort="100", velocity="2"
            )
            parent = child
    path = tmp_path / "robot_acd.urdf"
    ET.ElementTree(root).write(path, encoding="utf-8")
    original = data.get_data_path

    def resolve(value: str) -> str:
        return str(path) if value.startswith("TianjiMarvin/") else original(value)

    monkeypatch.setattr(data, "get_data_path", resolve)
    monkeypatch.setattr(tianji_marvin, "get_data_path", resolve)
    return path


def test_marvin_keeps_body_and_attaches_only_two_robotiq_hands(
    tmp_path: Path,
    marvin_source: Path,
) -> None:
    from embodichain.lab.sim.cfg import RobotCfg
    import pytorch_kinematics as pk

    selected = profiles._load_robotiq_embodiment(
        "tianji_marvin", tmp_path / "components"
    )
    cfg = RobotCfg.from_dict(selected["simulation"])
    original_hand = profiles._load_robotiq_embodiment("dual_franka", tmp_path)
    assert selected["simulation"].get("attrs") == original_hand["simulation"].get(
        "attrs"
    )
    original = ET.parse(marvin_source).getroot()
    root = ET.parse(cfg.fpath).getroot()
    links = {link.get("name") for link in root.findall("link")}
    assert {link.get("name") for link in original.findall("link")} <= links
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    assert (
        len([joint for joint in joints.values() if joint.get("type") != "fixed"]) == 26
    )
    assert (
        len([joint for joint in joints.values() if joint.find("mimic") is not None])
        == 10
    )
    for joint in original.findall("joint"):
        assembled = joints[joint.get("name")]
        assert ET.canonicalize(
            ET.tostring(assembled).decode(), strip_text=True
        ) == ET.canonicalize(ET.tostring(joint).decode(), strip_text=True)
    for side in ("left", "right"):
        connector = next(
            j
            for j in joints.values()
            if j.find("child").get("link") == f"{side}_robotiq_arg2f_base_link"
        )
        assert connector.find("parent").get("link") == f"{side}_ee"
        chain = pk.build_serial_chain_from_urdf(
            Path(cfg.fpath).read_bytes(), f"{side}_ee", f"{side}_arm_base"
        )
        assert chain.get_joint_parameter_names() == cfg.control_parts[f"{side}_arm"]
        assert len(cfg.control_parts[f"{side}_eef"]) == 6
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _bind_embodiment_to_scene,
    )

    _bind_embodiment_to_scene(selected, table_top_z=0.8)
    assert selected["simulation"]["init_pos"][2] == pytest.approx(0.05)
    assert profiles._profile_lateral_positions("tianji_marvin") == {
        "left": -0.25,
        "right": 0.25,
    }


def test_dual_tianji_alias_preserves_model_and_physical_identity(
    tmp_path: Path, marvin_source: Path
) -> None:
    legacy = profiles._load_robotiq_embodiment("tianji_marvin", tmp_path)
    model = Path(legacy["simulation"]["fpath"])
    original_model = model.read_bytes()
    selected = profiles._load_robotiq_embodiment("dual_tianji", tmp_path)

    assert selected == legacy
    assert model.read_bytes() == original_model
    assert selected["embodiment_id"] == "tianji_marvin_robotiq_arg2f_140"
    assert selected["skill_profile"]["profile_id"] == legacy["embodiment_id"]
    assert profiles._profile_lateral_positions("dual_tianji") == (
        profiles._profile_lateral_positions("tianji_marvin")
    )


def test_missing_marvin_frame_fails_closed(tmp_path: Path, marvin_source: Path) -> None:
    root = ET.parse(marvin_source).getroot()
    root.remove(root.find("link[@name='left_ee']"))
    ET.ElementTree(root).write(marvin_source)
    with pytest.raises(ValueError, match="arm/tool frames"):
        profiles._attach_marvin_hands(marvin_source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("profile", ["dual_tianji", "tianji_marvin"])
def test_asset_failure_names_profile_and_asset(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    from embodichain import data

    def unavailable(_value: str) -> str:
        raise RuntimeError("download unavailable")

    monkeypatch.setattr(data, "get_data_path", unavailable)
    with pytest.raises(ValueError, match=f"{profile}.*TianjiMarvin/robot_acd.urdf"):
        profiles._require_robot_assets(profile)
    profiles._require_robot_assets("dual_franka")


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("profile", ["dual_tianji", "tianji_marvin"])
def test_cli_source_preflight_precedes_providers_and_respects_frozen_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen: bool,
    profile: str,
) -> None:
    from types import SimpleNamespace
    from embodichain.gen_sim.task_engine import cli

    class ReachedWorkflow(Exception):
        pass

    def workflow(**_kwargs):
        if not frozen:
            pytest.fail(
                "Missing source assets must fail before workflow/provider setup"
            )
        raise ReachedWorkflow

    def require(_profile):
        if frozen:
            pytest.fail("Frozen physical references must not load registry assets")
        raise ValueError("missing robot source")

    monkeypatch.setattr(cli, "_require_robot_assets", require)
    monkeypatch.setattr(cli, "TaskEngineWorkflow", workflow)
    monkeypatch.setattr(
        cli,
        "load_task_engine_config",
        lambda _: (
            None,
            SimpleNamespace(physical_reference="reference" if frozen else None),
            None,
        ),
    )
    with pytest.raises(ReachedWorkflow if frozen else SystemExit) as error:
        cli.main(
            [
                "run-all",
                "--mode",
                "scene",
                "--task-id",
                "test",
                "--instruction",
                "test",
                "--scene",
                str(tmp_path / "source"),
                "--output-root",
                str(tmp_path / "runs"),
                "--robot-profile",
                profile,
            ]
        )
    if not frozen:
        assert error.value.code == 2


@pytest.mark.parametrize(
    "profile", ["dual_franka", "dual_ur10", "dual_tianji", "tianji_marvin"]
)
def test_new_robot_pour_return_and_park_compose(
    tmp_path: Path,
    marvin_source: Path,
    profile: str,
) -> None:
    from .test_pour_region_bundle import _case
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine._bundle_runner import (
        _verify_integration_fingerprint,
    )

    scene, graph, candidate = _case(tmp_path)
    selected, paths = generate_task_program_bundle(
        graph,
        scene,
        tmp_path / "bundle",
        robot_profile=profile,
        task_candidate=candidate,
        pour_geometry_revision=1,
    )
    deployment = _verify_integration_fingerprint(
        paths.root,
        paths.deployment,
        selected,
        load_config(paths.integration_fingerprint),
    )
    deployment.integration.registration.assert_unchanged()
    embodiment = load_config(paths.embodiment)
    commands = {
        p["preset_id"]: p for p in embodiment["skill_profile"]["command_presets"]
    }
    assert len(commands["left_arm_postures"]["commands"]["park"]) == (
        6 if profile == "dual_ur10" else 7
    )
    assert "home" in load_config(paths.program)["targets"]


@pytest.mark.parametrize(
    "profile", ["dual_franka", "dual_ur10", "dual_tianji", "tianji_marvin"]
)
def test_new_robot_press_prepare_and_contact_routes_compose(
    tmp_path: Path,
    marvin_source: Path,
    press_scene,
    profile: str,
) -> None:
    from .test_press_integration import graph
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine._bundle_runner import (
        _verify_integration_fingerprint,
    )

    selected, paths = generate_task_program_bundle(
        graph(press_scene),
        press_scene,
        tmp_path / "bundle",
        robot_profile=profile,
    )
    deployment = _verify_integration_fingerprint(
        paths.root,
        paths.deployment,
        selected,
        load_config(paths.integration_fingerprint),
    )
    deployment.integration.registration.assert_unchanged()
    integration = load_config(paths.integration)
    assert {
        entry["kind"]
        for entry in integration["runtime_services"]["registered_semantic_lowerers"]
    } >= {"press_prepare", "press"}


def test_profile_arm_preference_uses_actual_side_anchors() -> None:
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner

    planner = SemanticTaskPlanner()
    objects = {"item": {"init_pos": [0.0, 0.3, 0.8]}}
    assert planner._nearest_resource("item", objects) == "right"
    planner.resource_lateral_positions = {"left": 0.25, "right": -0.25}
    assert planner._nearest_resource("item", objects) == "left"
    assert (
        planner._nearest_resource("item", objects, occupied_resources=("left",))
        == "right"
    )


@pytest.mark.parametrize("profile", ["dual_tianji", "tianji_marvin"])
def test_frozen_profile_anchors_do_not_reload_registry(
    tmp_path: Path,
    marvin_source: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    from embodichain import data

    selected = profiles._load_robotiq_embodiment(
        "tianji_marvin", tmp_path / "components"
    )
    selected["simulation"]["init_rot"] = [0.0, 0.0, 0.0]
    selected["simulation"]["init_pos"] = [0.0, 1.0, 0.0]

    def unavailable(_value: str) -> str:
        pytest.fail("Frozen embodiment must not resolve registry assets")

    monkeypatch.setattr(data, "get_data_path", unavailable)
    assert profiles._profile_lateral_positions(profile, selected) == {
        "left": 1.25,
        "right": 0.75,
    }


@pytest.mark.parametrize(
    "profile", ["dual_franka", "dual_ur10", "dual_tianji", "tianji_marvin"]
)
def test_robot_bundle_publication_and_asset_identity(
    tmp_path: Path,
    marvin_source: Path,
    profile: str,
) -> None:
    from .test_semantic_graph import _prepared_axis_scene
    from .test_task_spec_generation import candidate, selector, step
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine.semantic_graph_validation import (
        verify_graph_contract,
    )
    from embodichain.gen_sim.task_engine.orchestration.artifacts import (
        ArtifactTransaction,
    )
    from embodichain.gen_sim.task_engine._bundle_runner import (
        _verify_integration_fingerprint,
    )

    scene = _prepared_axis_scene(tmp_path)
    source = candidate(step("E1", "bottle", target=selector("table"), relation="on"))
    sid = source["draft"]["steps"][0]["id"]
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {f"{sid}.object": ["bottle"], f"{sid}.target": ["table"]},
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(source, bindings, scene.planner_objects)
    with ArtifactTransaction(tmp_path / "published") as transaction:
        graph, paths = generate_task_program_bundle(
            plan.graph,
            scene,
            transaction.staging_dir,
            robot_profile=profile,
            task_spec=plan.task_spec,
            task_candidate=plan.candidate,
            role_bindings=plan.role_bindings,
            step_lineage=plan.lineage,
        )
        transaction.commit()
    bundle = tmp_path / "published"
    envelope = verify_graph_contract(bundle, graph)
    assert envelope["coverage"]["status"] == "covered"
    assert ("robot_asset_digest" in envelope) == (profile != "dual_franka")
    _verify_integration_fingerprint(
        bundle,
        bundle / "task_program_deployment.yaml",
        graph,
        load_config(bundle / "integration_fingerprint.json"),
    )
    if profile in {"dual_tianji", "tianji_marvin"}:
        selected = load_config(bundle / "components/embodiment.yaml")
        model = Path(selected["simulation"]["fpath"])
        assert model.is_relative_to(bundle)
        assert model.is_file()
        model.write_bytes(model.read_bytes() + b"\n<!-- changed robot source -->\n")
        with pytest.raises(ValueError, match="identity drifted"):
            verify_graph_contract(bundle, graph)


@pytest.mark.parametrize(
    "original,requested",
    [("tianji_marvin", "dual_tianji"), ("dual_tianji", "tianji_marvin")],
)
def test_tianji_aliases_reuse_frozen_inputs_without_rewriting_reference(
    tmp_path: Path, marvin_source: Path, original: str, requested: str
) -> None:
    from .test_semantic_graph import _prepared_axis_scene
    from .test_support_bundle import _placement_graph
    from .test_task_spec_generation import candidate, selector, step
    from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
        load_frozen_physical_input,
        verify_physical_input_manifest,
    )
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        generate_task_program_bundle,
    )
    from embodichain.gen_sim.task_engine.scene import SceneEngineV1Adapter
    from embodichain.gen_sim.task_engine.orchestration.scene_adapter import SceneAdapter
    from embodichain.gen_sim.task_engine.orchestration.scene_source import (
        SceneSourceRef,
    )
    from embodichain.utils.utility import save_config

    scene = replace(
        _prepared_axis_scene(tmp_path),
        source_config_path=tmp_path / "scene_config.json",
    )
    save_config(
        scene.source_config_path,
        {
            "format": "embodichain.scene-export/v1",
            "scene_id": "alias-test",
            "background": list(scene.background),
            "rigid_object": list(scene.rigid_objects),
        },
    )
    graph = _placement_graph()
    _, paths = generate_task_program_bundle(
        graph, scene, tmp_path / "reference", robot_profile=original
    )
    save_config(
        paths.root / "static_scene_manifest.json",
        SceneEngineV1Adapter().adapt_prepared_scene(
            scene, source_format="scene_export", robot_profile=original
        ),
    )
    before = {
        path: path.read_bytes() for path in paths.root.rglob("*") if path.is_file()
    }
    frozen = load_frozen_physical_input(scene, paths.root, robot_profile=original)

    def grounder(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": f"{step_id}.{role}",
                    "status": "resolved",
                    "uids": [uid],
                    "confidence": 1.0,
                }
                for role, uid in (("object", "bottle"), ("target", "table"))
            ]
        }

    candidates = [
        candidate(step("E1", "bottle", target=selector("table"), relation="on"))
    ]
    step_id = candidates[0]["draft"]["steps"][0]["id"]
    adaptation = SceneAdapter(grounding_caller=grounder).adapt(
        candidates,
        SceneSourceRef(scene.source_config_path, robot_profile=requested),
        physical_input=frozen,
    )
    assert adaptation.physical_input is frozen
    assert adaptation.scene_manifest["robot_profile"] == requested
    with pytest.raises(ValueError, match="owns the robot profile"):
        SceneAdapter(grounding_caller=grounder).adapt(
            candidates,
            SceneSourceRef(scene.source_config_path, robot_profile="dual_ur10"),
            physical_input=frozen,
        )
    _, output = generate_task_program_bundle(
        graph, frozen, tmp_path / "frozen", robot_profile=requested
    )
    assert load_config(output.embodiment) == frozen.embodiment
    assert load_config(output.scene)["simulation"] == frozen.scene_simulation
    assert verify_physical_input_manifest(output.root)["robot_profile"] == original
    reloaded = load_frozen_physical_input(scene, output.root, robot_profile=requested)
    assert reloaded.embodiment == frozen.embodiment
    assert reloaded.robot_profile == requested
    assert (
        load_frozen_physical_input(
            scene, paths.root, robot_profile=requested
        ).embodiment
        == frozen.embodiment
    )
    with pytest.raises(ValueError, match="robot profile"):
        load_frozen_physical_input(scene, output.root, robot_profile="dual_ur10")
    with pytest.raises(ValueError, match="robot-profile selection"):
        generate_task_program_bundle(
            graph, frozen, tmp_path / "invalid", robot_profile="dual_ur10"
        )
    frozen.assert_unchanged()
    assert all(path.read_bytes() == content for path, content in before.items())
