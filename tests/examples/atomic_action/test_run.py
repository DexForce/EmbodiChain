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

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from examples.sim.atomic_action import __main__ as launcher
from examples.sim.atomic_action.defaults import (
    ROBOTS,
    SCENES,
    _sample_defaults,
    _scene_args,
    AtomicSkillRunCfg,
)
from examples.sim.atomic_action.support import (
    NEWTON_GRASP_CONTACT_DAMPING,
    NEWTON_GRASP_CONTACT_STIFFNESS,
    NEWTON_GRASP_ROLLING_FRICTION,
    NEWTON_GRASP_TORSIONAL_FRICTION,
    NEWTON_NATIVE_CONTACT_DIMENSION,
    configure_newton_gripper_contacts,
    create_affordance_sampling_context,
    create_franka_panda_robot_cfg,
)
from embodichain.lab.sim.atomic_actions import TimedTrajectory
from embodichain.lab.sim.cfg import RobotCfg
from embodichain_robots.franka import panda as franka_panda
from embodichain.utils.string import resolve_matching_names

from examples.sim.atomic_action import scenario, tcp_presets
from examples.sim.atomic_action.scenario import (
    create_dual_tutorial_robot_cfg,
)
from examples.sim.atomic_action.support import (
    create_tutorial_robot_cfg,
    create_ur5_gripper_robot_cfg,
    create_ur10_robotiq_robot_cfg,
)
from embodichain.lab.sim.motion.solvers import FEPSolverCfg, PytorchSolverCfg
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners import TrapezoidalPlanner

MULTI_ROBOT_SCENES = {"pickup", "place", "move_joint", "handover"}
PGI_FINGER_LINKS = ("gripper_finger1_link_1", "gripper_finger2_link_1")
NATIVE_FRANKA_FINGER_LINKS = ("fr3_leftfinger", "fr3_rightfinger")


@pytest.fixture
def sample_franka_cfg(monkeypatch: pytest.MonkeyPatch) -> RobotCfg:
    # Configuration tests do not need to download or load the URDF assets.
    monkeypatch.setattr(franka_panda, "get_data_path", lambda path: path)
    return create_franka_panda_robot_cfg()


def test_franka_sample_fk_matches_assembled_pgi_tool_frame() -> None:
    pytest.importorskip("pytorch_kinematics")
    cfg = create_franka_panda_robot_cfg()
    chain = cfg.build_pk_serial_chain()["arm"]
    solver_cfg = cfg.solver_cfg["arm"]
    solver_cfg.urdf_path = cfg.urdf_cfg.assemble_urdf()
    solver_cfg.joint_names = cfg.control_parts["arm"]
    solver = solver_cfg.init_solver(device=torch.device("cpu"))
    # Compare the ready pose and a distinct arm configuration so a mismatched
    # flange/stock-hand chain cannot be hidden by the sample's TCP calibration.
    qpos = torch.tensor(
        [cfg.init_qpos[:7], [0.3, -0.2, 0.1, -2.0, 0.2, 2.5, 0.4]],
        dtype=torch.float32,
    )
    tcp = torch.as_tensor(solver_cfg.tcp, dtype=torch.float32)
    torch.testing.assert_close(
        solver.get_fk(qpos), chain.forward_kinematics(qpos).get_matrix() @ tcp
    )


def test_franka_sample_contact_override_matches_replacement_pgi_fingers(
    sample_franka_cfg: RobotCfg,
) -> None:
    contact = sample_franka_cfg.link_attrs["newton_gripper_contacts"]
    available_links = [
        "base_link",
        "fr3_link8",
        "gripper_base_link_1",
        *PGI_FINGER_LINKS,
        *NATIVE_FRANKA_FINGER_LINKS,
    ]
    _, matched_links = resolve_matching_names(contact.link_names_expr, available_links)
    assert matched_links == list(PGI_FINGER_LINKS)


def test_franka_sample_keeps_contact_attributes_and_native_preset_independent(
    sample_franka_cfg: RobotCfg,
) -> None:
    native_cfg = franka_panda.FrankaPandaCfg.from_dict({})
    native_contact = native_cfg.link_attrs["newton_gripper_contacts"]
    sample_contact = sample_franka_cfg.link_attrs["newton_gripper_contacts"]
    assert sample_contact.attrs.to_dict() == native_contact.attrs.to_dict()

    # A later sample construction and mutation must not change native defaults.
    create_franka_panda_robot_cfg()
    sample_contact.link_names_expr.clear()
    fresh_native = franka_panda.FrankaPandaCfg.from_dict({})
    for cfg in (native_cfg, fresh_native):
        contact = cfg.link_attrs["newton_gripper_contacts"]
        _, matched_links = resolve_matching_names(
            contact.link_names_expr,
            [*PGI_FINGER_LINKS, *NATIVE_FRANKA_FINGER_LINKS],
        )
        assert matched_links == list(NATIVE_FRANKA_FINGER_LINKS)


@pytest.mark.parametrize("is_newton_backend", (False, True))
def test_franka_sample_contacts_are_valid_for_both_backends(
    sample_franka_cfg: RobotCfg, is_newton_backend: bool
) -> None:
    original_contact = sample_franka_cfg.link_attrs["newton_gripper_contacts"].to_dict()
    configure_newton_gripper_contacts(
        SimpleNamespace(is_newton_backend=is_newton_backend), sample_franka_cfg
    )
    contact = sample_franka_cfg.link_attrs["newton_gripper_contacts"]
    _, matched_links = resolve_matching_names(
        contact.link_names_expr,
        ["gripper_base_link_1", *PGI_FINGER_LINKS],
    )
    assert matched_links == list(PGI_FINGER_LINKS)
    if not is_newton_backend:
        assert contact.to_dict() == original_contact
    else:
        assert contact.attrs.collision_props.condim == NEWTON_NATIVE_CONTACT_DIMENSION
        material = contact.attrs.material_props
        assert material.ke == NEWTON_GRASP_CONTACT_STIFFNESS
        assert material.kd == NEWTON_GRASP_CONTACT_DAMPING
        assert material.torsional_friction == NEWTON_GRASP_TORSIONAL_FRICTION
        assert material.rolling_friction == NEWTON_GRASP_ROLLING_FRICTION
        assert contact.attrs.mass_props.recompute_inertia is True


@pytest.mark.parametrize(
    "robot, point, expected_z",
    (
        ("ur5", "default", 0.17),
        ("ur5", "grasp", 0.15),
        ("ur5", "handover", 0.15),
        ("franka", "default", 0.15),
        ("franka", "grasp", 0.15),
        ("franka", "handover", 0.165),
        ("ur10", "default", 0.21),
        ("ur10", "grasp", 0.21),
        ("ur10", "handover", 0.21),
    ),
)
def test_fixture_tcp_presets_keep_existing_sample_calibrations(
    robot: str, point: str, expected_z: float
) -> None:
    pose = tcp_presets.sample_tool_tcp(robot, point=point)
    assert pose[2][3] == expected_z
    if robot == "ur10":
        assert pose[:2] == [[0, -1, 0, 0], [1, 0, 0, 0]]


def test_fixture_tcp_returns_independent_matrices() -> None:
    expected = tcp_presets.sample_tool_tcp("franka")
    original = tcp_presets.sample_tool_tcp("franka")
    original[2][3] = 99.0
    assert tcp_presets.sample_tool_tcp("franka") == expected
    assert tcp_presets.sample_tool_tcp("ur5")[2][3] == 0.17
    assert tcp_presets.sample_tool_tcp("ur10")[2][3] == 0.21


def test_franka_tcp_calibration_does_not_change_other_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model a full calibration change, including rotation and lateral offsets.
    calibrated = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, -0.02),
        (0.0, 0.0, 1.0, 0.19),
        (0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(franka_panda, "get_data_path", lambda path: path)
    monkeypatch.setattr(tcp_presets, "FRANKA_PGI_TCP", calibrated)
    cfg = create_tutorial_robot_cfg("franka", tcp_point="grasp")
    assert torch.equal(
        torch.as_tensor(cfg.solver_cfg["arm"].tcp), torch.tensor(calibrated)
    )
    assert torch.equal(
        torch.as_tensor(
            create_ur5_gripper_robot_cfg(tcp_point="grasp").solver_cfg["arm"].tcp
        ),
        torch.tensor(tcp_presets.UR5_PGI_GRASP_TCP),
    )
    assert torch.equal(
        torch.as_tensor(create_ur10_robotiq_robot_cfg().solver_cfg["arm"].tcp),
        torch.tensor(tcp_presets.UR10_ROBOTIQ_2F_140_TCP),
    )


def test_franka_explicit_tcp_z_is_applied_instead_of_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(franka_panda, "get_data_path", lambda path: path)
    original = tcp_presets.FRANKA_PGI_TCP
    cfg = create_franka_panda_robot_cfg(tcp_z=0.19)
    assert torch.as_tensor(cfg.solver_cfg["arm"].tcp)[2, 3] == pytest.approx(0.19)
    assert tcp_presets.FRANKA_PGI_TCP == original


def test_dual_franka_uses_fixture_pose_with_mounting_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrated = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, -0.02),
        (0.0, 0.0, 1.0, 0.19),
        (0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(franka_panda, "get_data_path", lambda path: path)
    monkeypatch.setattr(tcp_presets, "FRANKA_PGI_HANDOVER_TCP", calibrated)
    cfg = create_dual_tutorial_robot_cfg(
        robot_type="franka", uid="calibration_test", urdf_name="calibration_test"
    )
    for part in ("left_arm", "right_arm"):
        assert torch.equal(
            torch.as_tensor(cfg.solver_cfg[part].tcp), torch.tensor(calibrated)
        )


def test_legacy_dual_pgi_z_override_preserves_fixture_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrated = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, -0.02),
        (0.0, 0.0, 1.0, 0.19),
        (0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(franka_panda, "get_data_path", lambda path: path)
    monkeypatch.setattr(tcp_presets, "FRANKA_PGI_HANDOVER_TCP", calibrated)
    cfg = create_dual_tutorial_robot_cfg(
        robot_type="franka",
        uid="calibration_test",
        urdf_name="calibration_test",
        tcp_z=0.12,
    )
    expected = torch.tensor(calibrated)
    expected[2, 3] = 0.12
    assert torch.equal(torch.as_tensor(cfg.solver_cfg["left_arm"].tcp), expected)


@pytest.mark.parametrize("robot", ("ur5", "franka", "ur10"))
def test_dual_fixture_tcp_matches_selected_handover_calibration(robot: str) -> None:
    cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot, uid=f"test_{robot}", urdf_name=f"tcp_test_{robot}"
    )
    expected = torch.tensor(tcp_presets.sample_tool_tcp(robot, point="handover"))
    for part in ("left_arm", "right_arm"):
        assert torch.equal(torch.as_tensor(cfg.solver_cfg[part].tcp), expected)


def test_dual_franka_retains_analytical_preset_and_mounted_solver_frames(
    sample_franka_cfg: RobotCfg,
) -> None:
    cfg = create_dual_tutorial_robot_cfg(
        robot_type="franka", uid="test_fep", urdf_name="test_dual_fep"
    )
    native_solver = sample_franka_cfg.solver_cfg["arm"]
    for side in ("left", "right"):
        solver = cfg.solver_cfg[f"{side}_arm"]
        assert isinstance(solver, FEPSolverCfg)
        assert solver.redundancy_search is True
        assert solver.root_link_name == f"{side}_{native_solver.root_link_name}"
        assert solver.end_link_name == f"{side}_{native_solver.end_link_name}"
        assert solver.user_qpos_limits == native_solver.user_qpos_limits
    assert native_solver.tcp == tcp_presets.sample_tool_tcp("franka")


def test_dual_franka_preserves_authored_fep_options(
    monkeypatch: pytest.MonkeyPatch, sample_franka_cfg: RobotCfg
) -> None:
    native_solver = sample_franka_cfg.solver_cfg["arm"]
    native_solver.position_tolerance = 2e-5
    native_solver.rotation_tolerance = 3e-5
    native_solver.arm_angle_weight = 0.25
    native_solver.max_joint_step = 0.5
    monkeypatch.setattr(
        scenario, "create_tutorial_robot_cfg", lambda robot: sample_franka_cfg
    )
    cfg = create_dual_tutorial_robot_cfg(
        robot_type="franka", uid="test_fep_options", urdf_name="test_fep_options"
    )
    for solver in cfg.solver_cfg.values():
        assert isinstance(solver, FEPSolverCfg)
        assert solver.redundancy_search is True
        assert solver.position_tolerance == native_solver.position_tolerance
        assert solver.rotation_tolerance == native_solver.rotation_tolerance
        assert solver.arm_angle_weight == native_solver.arm_angle_weight
        assert solver.max_joint_step == native_solver.max_joint_step


@pytest.mark.parametrize("robot", ROBOTS)
def test_dual_robot_explicit_pytorch_selection_preserves_solver_contract(
    robot: str,
) -> None:
    # A non-default seed count proves the explicit numerical path is used.
    requested_samples = 11
    single = create_tutorial_robot_cfg(robot).solver_cfg["arm"]
    cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot,
        uid=f"test_{robot}_pytorch",
        urdf_name=f"test_{robot}_pytorch",
        solver="pytorch",
        pytorch_num_samples=requested_samples,
    )
    for side in ("left", "right"):
        solver = cfg.solver_cfg[f"{side}_arm"]
        assert isinstance(solver, PytorchSolverCfg)
        assert solver.num_samples == requested_samples
        assert solver.root_link_name == f"{side}_{single.root_link_name}"
        assert solver.end_link_name == f"{side}_{single.end_link_name}"
        assert solver.user_qpos_limits == single.user_qpos_limits
        assert solver.tcp == tcp_presets.sample_tool_tcp(robot, point="handover")


@pytest.mark.parametrize("robot", ROBOTS)
def test_handover_uses_numerical_franka_and_analytical_ur_defaults(
    monkeypatch: pytest.MonkeyPatch, robot: str
) -> None:
    from examples.sim.atomic_action import handover

    captured = {}
    expected_robot = object()
    sim = object()

    def add_robot(actual_sim, **kwargs):
        assert actual_sim is sim
        captured.update(kwargs)
        return expected_robot

    monkeypatch.setattr(handover, "add_dual_tutorial_robot", add_robot)
    assert handover.create_dual_robot(sim, robot) is expected_robot
    assert captured["robot_type"] == robot
    assert captured["gripper"] == "robotiq_2f_140"
    assert captured["solver"] == ("pytorch" if robot == "franka" else "ur")


@pytest.mark.parametrize("robot_type", ROBOTS)
def test_handover_robotiq_assembly_and_signed_hand_commands(robot_type: str) -> None:
    """Every selected arm mounts the requested asset with valid six-joint commands."""
    from examples.sim.atomic_action import handover
    from examples.sim.atomic_action.support import get_hand_open_close_qpos

    cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot_type,
        uid=f"test_{robot_type}_robotiq",
        urdf_name=f"test_{robot_type}_robotiq_handover",
        gripper="robotiq_2f_140",
        solver="pytorch" if robot_type == "franka" else "ur",
    )
    arm_dof = len(create_tutorial_robot_cfg(robot_type).control_parts["arm"])
    assert len(cfg.init_qpos) == 2 * arm_dof + 12
    assert cfg.init_qpos[-12:] == [0.0] * 12
    for side in ("left", "right"):
        assert cfg.urdf_cfg.components[f"{side}_hand"]["urdf_path"].endswith(
            "/Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"
        )
        assert cfg.solver_cfg[f"{side}_arm"].tcp == tcp_presets.sample_tool_tcp(
            robot_type, point="handover", gripper="robotiq_2f_140"
        )
    root = ET.parse(cfg.urdf_cfg.assemble_urdf()).getroot()
    joints = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    }
    assert len(joints) == len(cfg.init_qpos)
    for rules in (
        cfg.joint_drive_props.stiffness,
        cfg.joint_drive_props.damping,
        cfg.joint_drive_props.max_effort,
    ):
        assert all("gripper_finger" not in pattern for pattern in rules)
        patterns = [
            pattern
            for selector in rules
            for pattern in cfg.control_parts.get(selector, [selector])
        ]
        resolve_matching_names(patterns, list(joints))
    for side in ("left", "right"):
        part = f"{side}_eef"
        _, names = resolve_matching_names(cfg.control_parts[part], list(joints))
        assert len(names) == 6
        limits = torch.tensor(
            [
                [
                    float(joints[name].find("limit").get(key))
                    for key in ("lower", "upper")
                ]
                for name in names
            ]
        )[None]
        robot = SimpleNamespace(
            device=torch.device("cpu"),
            cfg=SimpleNamespace(control_parts={part: names}),
            get_qpos_limits=lambda name: limits,
        )
        opened, closed = get_hand_open_close_qpos(
            robot, hand_control_part=part, close_qpos=handover.HAND_CLOSE_QPOS
        )
        assert torch.equal(opened, torch.zeros(6))
        master_joint = f"{side}_finger_joint"
        expected = []
        for name in names:
            mimic = joints[name].find("mimic")
            if name == master_joint:
                assert mimic is None
                expected.append(handover.HAND_CLOSE_QPOS)
            else:
                assert mimic is not None and mimic.get("joint") == master_joint
                expected.append(
                    float(mimic.get("multiplier", "1")) * handover.HAND_CLOSE_QPOS
                    + float(mimic.get("offset", "0"))
                )
        expected = torch.tensor(expected)
        torch.testing.assert_close(closed, expected)


def test_handover_robotiq_tcp_is_independent_of_pgi_and_other_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrated = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, -0.02),
        (0.0, 0.0, 1.0, 0.22),
        (0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(tcp_presets, "FRANKA_ROBOTIQ_2F_140_HANDOVER_TCP", calibrated)
    cfg = create_dual_tutorial_robot_cfg(
        robot_type="franka",
        uid="robotiq_tcp_test",
        urdf_name="robotiq_tcp_test",
        gripper="robotiq_2f_140",
        tcp_z=0.12,  # The legacy PGI override does not shorten this Robotiq fixture.
    )
    for part in ("left_arm", "right_arm"):
        assert torch.equal(
            torch.as_tensor(cfg.solver_cfg[part].tcp), torch.tensor(calibrated)
        )
    sample_tool_pose = tcp_presets.sample_tool_tcp("franka")
    assert sample_tool_pose == [list(row) for row in tcp_presets.FRANKA_PGI_TCP]
    for robot_type in ("ur5", "ur10"):
        pose = tcp_presets.sample_tool_tcp(robot_type, gripper="robotiq_2f_140")
        assert pose[2][3] == 0.21


def test_handover_grasp_model_uses_robotiq_opening_and_owns_its_geometry() -> None:
    from examples.sim.atomic_action import handover
    from examples.sim.atomic_action.support import (
        create_parallel_jaw_grasp_pose_generator,
    )

    generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=32, force_refresh=False, gripper_model=handover.HANDOVER_GRIPPER_MODEL
    )
    assert generator.gripper_model.model_id == "robotiq_2f_140"
    assert generator.gripper_model.max_opening_width == 0.14
    geometry = generator.gripper_model
    geometry.max_opening_width = 0.12
    assert generator.gripper_model.max_opening_width == 0.14
    assert handover.HANDOVER_GRIPPER_MODEL.max_opening_width == 0.14
    assert (
        create_parallel_jaw_grasp_pose_generator(
            n_sample=32, force_refresh=False
        ).gripper_model.model_id
        == "dh_pgi_140_80"
    )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True))
def test_invalid_tcp_calibration_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        tcp_presets.sample_tool_tcp("franka", tcp_z=value)


@pytest.mark.parametrize("skill", SCENES)
def test_every_sample_defaults_to_one_reference_environment(skill: str) -> None:
    cfg = launcher._resolve_sample(
        launcher._create_parser().parse_args(["--skill", skill])
    )
    assert isinstance(cfg, AtomicSkillRunCfg)
    assert cfg.variants == 1
    assert not cfg.affordance_enabled and not cfg.trajectory_enabled
    assert _scene_args(cfg).num_envs == 1
    assert cfg.device is None
    assert _scene_args(cfg).device is None


@pytest.mark.parametrize(
    "physics, expected_device", [("default", "cpu"), ("newton", "cuda:0")]
)
def test_sample_omitted_device_preserves_backend_default(
    monkeypatch: pytest.MonkeyPatch, physics: str, expected_device: str
) -> None:
    from examples.sim.atomic_action import support

    captured = {}

    def create_sim(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(add_light=lambda cfg: None)

    monkeypatch.setattr(support, "SimulationManager", create_sim)
    args = launcher._create_parser().parse_args(
        ["--skill", "pickup", "--physics", physics]
    )
    cfg = launcher._resolve_sample(args)
    support.create_tutorial_simulation(_scene_args(cfg))
    assert captured["cfg"].device == expected_device


@pytest.mark.parametrize("device", ["cpu", "cuda:2"])
def test_sample_explicit_device_is_preserved(device: str) -> None:
    args = launcher._create_parser().parse_args(
        ["--skill", "pickup", "--physics", "newton", "--device", device]
    )
    cfg = launcher._resolve_sample(args)
    assert cfg.device == device
    assert _scene_args(cfg).device == device


@pytest.mark.parametrize("count", (2, 8))
@pytest.mark.parametrize("mode", ("affordance", "trajectory", "both"))
def test_variant_count_is_the_parallel_environment_count(count: int, mode: str) -> None:
    cfg = launcher._resolve_sample(
        launcher._create_parser().parse_args(
            ["--skill", "place", "--variants", str(count), "--expansion", mode]
        )
    )
    args = _scene_args(cfg)
    assert args.num_envs == count
    assert cfg.affordance_enabled == (mode in {"affordance", "both"})
    assert cfg.trajectory_enabled == (mode in {"trajectory", "both"})
    assert not any(
        name.startswith("_config") or name == "_recipes" for name in vars(args)
    )


@pytest.mark.parametrize(
    "arguments,message",
    (
        (["--variants", "0"], "variants must be positive"),
        (["--variants", "-1"], "variants must be positive"),
        (["--variants", "8", "--num-envs", "2"], "must equal"),
        (["--variants", "2", "--no-expansion"], "requires --variants 1"),
        (["--gpu-id", "-1"], "non-negative"),
        (["--seed", "-1"], "seed must"),
        (["--arena-space", "nan"], "finite"),
        (["--device", ""], "device must"),
        (["--save-video", "--renderer", "no-render"], "requires a renderer"),
    ),
)
def test_invalid_cli_settings_fail_before_simulation(
    arguments: list[str], message: str
) -> None:
    args = launcher._create_parser().parse_args(["--skill", "place", *arguments])
    with pytest.raises(ValueError, match=message):
        launcher._resolve_sample(args)


@pytest.mark.parametrize("skill", ("move_joint", "push_object"))
def test_samples_without_affordances_reject_affordance_only_mode(skill: str) -> None:
    args = launcher._create_parser().parse_args(
        ["--skill", skill, "--variants", "2", "--expansion", "affordance"]
    )
    with pytest.raises(ValueError, match="no affordance"):
        launcher._resolve_sample(args)


@pytest.mark.parametrize("skill", SCENES)
@pytest.mark.parametrize("robot", ROBOTS)
def test_sample_robot_compatibility_is_validated_before_scene_import(
    skill: str, robot: str
) -> None:
    if skill not in MULTI_ROBOT_SCENES and robot != "ur5":
        with pytest.raises(ValueError, match="requires robot"):
            _sample_defaults(skill, robot=robot)
    else:
        assert _sample_defaults(skill, robot=robot).robot == robot


def test_removed_cartesian_product_recipe_argument_is_rejected() -> None:
    with pytest.raises(SystemExit):
        launcher._create_parser().parse_args(
            ["--skill", "place", "--recipe-indices", "0"]
        )


def test_cli_saves_completed_run_and_automatic_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def run_sample(cfg, args, run):
        assert cfg.variants == 2
        assert _scene_args(cfg).num_envs == 2
        run.replays.append({"status": "executed", "measured_task_success": None})

    monkeypatch.setattr(launcher, "_run_sample", run_sample)
    launcher.cli(["--skill", "place", "--variants", "2", "--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "executed"
    assert manifest["num_envs"] == manifest["variants"] == 2
    assert manifest["enabled_factors"] == ["affordance", "trajectory"]
    assert manifest["measured_task_success"] is None


@pytest.mark.parametrize("error", (RuntimeError("planning failed"), SystemExit(130)))
def test_cli_persists_failure_and_interruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: BaseException
) -> None:
    def fail(cfg, args, run):
        raise error

    monkeypatch.setattr(launcher, "_run_sample", fail)
    with pytest.raises(type(error)):
        launcher.cli(["--skill", "place", "--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert type(error).__name__ in manifest["error"]


@pytest.mark.no_sim
def test_sample_solver_overrides_preserve_configured_host_fields(monkeypatch) -> None:
    from examples.sim.atomic_action import support

    captured = {}

    def create_sim(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(add_light=lambda cfg: None)

    monkeypatch.setattr(support, "SimulationManager", create_sim)
    args = SimpleNamespace(
        physics="newton",
        num_envs=2,
        device="cpu",
        renderer="no-render",
        gpu_id=3,
        arena_space=4.0,
        run=SimpleNamespace(),
    )
    support.create_tutorial_simulation(
        args, newton_solver_options={"vbd_options": {"rigid_contact_history": True}}
    )
    cfg = captured["cfg"]
    assert cfg.arena_space == 4.0
    assert cfg.gpu_id == 3
    assert cfg.physics_cfg.collision_cfg is None
    assert cfg.physics_cfg.solver_cfg["solver_type"] == "dexuni"
    assert cfg.physics_cfg.solver_cfg["mujoco_options"]["enable_multiccd"]
    assert cfg.physics_cfg.solver_cfg["vbd_options"]["rigid_contact_history"]


@pytest.mark.no_sim
def test_sample_newton_options_reject_other_backends_before_creation(
    monkeypatch,
) -> None:
    from examples.sim.atomic_action import support

    def unexpected(*args, **kwargs):
        pytest.fail("Simulation should not be created for incompatible options")

    monkeypatch.setattr(support, "SimulationManager", unexpected)
    with pytest.raises(ValueError, match="Newton backend"):
        support.create_tutorial_simulation(
            SimpleNamespace(physics="default"),
            newton_solver_options={"coupling": "two_way"},
        )


@pytest.mark.no_sim
def test_sample_friction_and_arm_damping_are_applied_before_spawn(monkeypatch) -> None:
    from examples.sim.atomic_action import support
    from embodichain.lab.sim.cfg import JointDrivePropertiesCfg

    cfg = RobotCfg(joint_drive_props=JointDrivePropertiesCfg(damping={"arm": 2000.0}))
    captured = {}

    def create_cfg(*args, **kwargs):
        captured.update(kwargs)
        return cfg

    monkeypatch.setattr(support, "create_tutorial_robot_cfg", create_cfg)
    sim = SimpleNamespace(is_newton_backend=True, add_robot=lambda cfg: cfg)
    result = support.add_tutorial_robot(
        sim, "ur5", tcp_point="grasp", newton_gripper_friction=4.0, arm_damping=1000.0
    )
    assert captured["tcp_point"] == "grasp"
    assert "newton_gripper_friction" not in captured
    assert "arm_damping" not in captured
    assert result.joint_drive_props.damping["arm"] == 1000.0
    material = result.link_attrs["newton_gripper_contacts"].attrs.material_props
    assert material.static_friction == material.dynamic_friction == 4.0


@pytest.mark.no_sim
def test_dual_sample_gripper_friction_retains_fixture_selection(monkeypatch) -> None:
    cfg = RobotCfg()
    captured = {}

    def create_cfg(**kwargs):
        captured.update(kwargs)
        return cfg

    monkeypatch.setattr(scenario, "create_dual_tutorial_robot_cfg", create_cfg)
    sim = SimpleNamespace(is_newton_backend=True, add_robot=lambda cfg: cfg)
    result = scenario.add_dual_tutorial_robot(
        sim,
        robot_type="ur5",
        uid="robot",
        urdf_name="fixture",
        gripper="robotiq_2f_140",
        newton_gripper_friction=4.0,
    )
    assert captured["gripper"] == "robotiq_2f_140"
    assert "newton_gripper_friction" not in captured
    material = result.link_attrs["newton_gripper_contacts"].attrs.material_props
    assert material.static_friction == material.dynamic_friction == 4.0
