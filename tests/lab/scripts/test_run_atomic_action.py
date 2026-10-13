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

import pytest
import torch

from embodichain.lab.scripts import run_atomic_action as launcher
from embodichain.lab.scripts._atomic_action.defaults import (
    ROBOTS,
    SCENES,
    _recipes,
    _sample_defaults,
    _scene_args,
)
from embodichain.lab.scripts._atomic_action.expansion import _augment, _template
from embodichain.lab.scripts._atomic_action.support import (
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
from embodichain.lab.sim.robots import franka_panda
from embodichain.utils.string import resolve_matching_names

from embodichain.lab.scripts._atomic_action import scenario, tcp_presets
from embodichain.lab.scripts._atomic_action.scenario import (
    create_dual_tutorial_robot_cfg,
)
from embodichain.lab.scripts._atomic_action.support import (
    create_tutorial_robot_cfg,
    create_ur5_gripper_robot_cfg,
    create_ur10_robotiq_robot_cfg,
)
from embodichain.lab.sim.motion.solvers import FEPSolverCfg, PytorchSolverCfg

MULTI_ROBOT_SCENES = {"pickup", "place", "move_joint", "handover"}
PGI_FINGER_LINKS = ("gripper_finger1_link_1", "gripper_finger2_link_1")
NATIVE_FRANKA_FINGER_LINKS = ("fr3_leftfinger", "fr3_rightfinger")


@pytest.fixture
def sample_franka_cfg(monkeypatch: pytest.MonkeyPatch) -> RobotCfg:
    # Configuration tests do not need to download or load the URDF assets.
    monkeypatch.setattr(franka_panda, "get_data_path", lambda path: path)
    return create_franka_panda_robot_cfg()


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
    from embodichain.lab.scripts._atomic_action import handover

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
    assert captured["solver"] == ("pytorch" if robot == "franka" else "ur")


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True))
def test_invalid_tcp_calibration_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        tcp_presets.sample_tool_tcp("franka", tcp_z=value)


@pytest.mark.parametrize("skill", SCENES)
def test_internal_scene_defaults_preserve_expansion_recipes(skill: str) -> None:
    cfg = _sample_defaults(skill)
    recipes = _recipes(cfg)
    expected = 4 if skill == "move_joint" else 16
    assert len(recipes) == expected
    assert recipes[0] == (0, 0)
    assert cfg["expansion"]["trajectory"]["spatial"]["operators"] == [
        "joint_residual",
        "via_points",
    ]


def test_sample_defaults_are_independent_between_runs() -> None:
    cfg = _sample_defaults("pickup")
    cfg["options"]["approach"] = "custom"
    cfg["expansion"]["trajectory"]["spatial"]["operators"].clear()
    cfg["runtime"]["headless"] = True
    fresh = _sample_defaults("pickup")
    assert fresh["options"]["approach"] == "top"
    assert fresh["expansion"]["trajectory"]["spatial"]["operators"] == [
        "joint_residual",
        "via_points",
    ]
    assert fresh["runtime"]["headless"] is False


def test_cli_requires_skill(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        launcher._create_parser().parse_args([])
    assert exc.value.code == 2
    assert "--skill" in capsys.readouterr().err


def test_cli_selector_defaults() -> None:
    args = launcher._create_parser().parse_args(["--skill", "place"])
    assert (args.skill, args.physics, args.robot, args.planner) == (
        "place",
        "default",
        "ur5",
        "trapezoidal",
    )
    cfg, selected = launcher._resolve_sample(args)
    assert cfg["runtime"]["num_envs"] == 16
    assert selected == tuple(range(16))


@pytest.mark.parametrize(
    "option, value",
    [
        ("--skill", "unknown"),
        ("--physics", "other"),
        ("--robot", "panda"),
        ("--planner", "other"),
    ],
)
def test_cli_rejects_unknown_selectors(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        launcher._create_parser().parse_args(["--skill", "place", option, value])
    assert exc.value.code == 2


@pytest.mark.parametrize("option", ("--config", "--list-skills"))
def test_removed_configuration_arguments_are_rejected(option: str) -> None:
    with pytest.raises(SystemExit) as exc:
        launcher._create_parser().parse_args(["--skill", "place", option])
    assert exc.value.code == 2


@pytest.mark.parametrize("skill", SCENES)
@pytest.mark.parametrize("robot", ROBOTS)
def test_skill_robot_compatibility(skill: str, robot: str) -> None:
    args = launcher._create_parser().parse_args(["--skill", skill, "--robot", robot])
    if skill not in MULTI_ROBOT_SCENES and robot != "ur5":
        with pytest.raises(ValueError, match="requires robot 'ur5'"):
            launcher._resolve_sample(args)
    else:
        cfg, _ = launcher._resolve_sample(args)
        scene_args = _scene_args(cfg, (0,), 0)
        assert cfg["skill"] == skill
        assert cfg["robot"] == scene_args.robot == robot


@pytest.mark.parametrize(
    "skill", tuple(skill for skill in SCENES if skill not in MULTI_ROBOT_SCENES)
)
@pytest.mark.parametrize("robot", ("franka", "ur10"))
def test_unsupported_robot_is_rejected_before_simulation(
    skill: str,
    robot: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_batch(*args, **kwargs):
        pytest.fail("unsupported robot reached simulator execution")

    monkeypatch.setattr(launcher, "_run_batch", unexpected_batch)
    output_dir = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        launcher.cli(
            ["--skill", skill, "--robot", robot, "--output-dir", str(output_dir)]
        )
    assert exc.value.code == 2
    assert "requires robot 'ur5'" in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.parametrize("physics", ("default", "newton"))
@pytest.mark.parametrize("planner", ("trapezoidal", "toppra", "curobo"))
def test_direct_selectors_reach_scene(physics: str, planner: str) -> None:
    args = launcher._create_parser().parse_args(
        [
            "--skill",
            "place",
            "--physics",
            physics,
            "--robot",
            "franka",
            "--planner",
            planner,
        ]
    )
    cfg, _ = launcher._resolve_sample(args)
    scene_args = _scene_args(cfg, (0,), 0)
    assert cfg["physics"] == scene_args.physics == physics
    assert cfg["robot"] == scene_args.robot == "franka"
    assert cfg["planner"] == scene_args.planner == planner


def test_cli_runtime_controls_are_retained(tmp_path: Path) -> None:
    args = launcher._create_parser().parse_args(
        [
            "--skill",
            "pickup",
            "--headless",
            "--num-envs",
            "2",
            "--recipe-indices",
            "2",
            "4",
            "--seed",
            "17",
            "--save-video",
            "--output-dir",
            str(tmp_path),
        ]
    )
    cfg, selected = launcher._resolve_sample(args)
    assert cfg["runtime"]["headless"] is True
    assert cfg["runtime"]["num_envs"] == 2
    assert selected == (2, 4)
    assert cfg["seed"] == 17
    assert cfg["output"]["save_video"] is True
    assert cfg["output"]["directory"] == str(tmp_path)


def test_nominal_replay_disables_both_expansion_paths() -> None:
    args = launcher._create_parser().parse_args(
        ["--skill", "place", "--no-expansion", "--num-envs", "1"]
    )
    cfg, selected = launcher._resolve_sample(args)
    assert selected == (0,)
    assert _recipes(cfg) == ((0, 0),)
    assert cfg["expansion"]["affordance"]["enabled"] is False
    assert cfg["expansion"]["trajectory"]["spatial"]["enabled"] is False
    assert cfg["runtime"]["num_envs"] == 1


@pytest.mark.parametrize(
    "options, message",
    [
        (["--num-envs", "0"], "positive"),
        (["--num-envs", "-1"], "positive"),
        (["--gpu-id", "-1"], "non-negative"),
        (["--arena-space", "nan"], "finite and positive"),
        (["--arena-space", "0"], "finite and positive"),
        (["--seed", "-1"], "seed"),
        (["--seed", str(2**32)], "seed"),
        (["--recipe-indices", "0", "0"], "unique"),
        (["--recipe-indices", "16"], "unique"),
        (["--save-video", "--renderer", "no-render"], "requires a renderer"),
    ],
)
def test_runtime_errors_are_rejected_before_execution(
    options: list[str], message: str
) -> None:
    args = launcher._create_parser().parse_args(["--skill", "place", *options])
    with pytest.raises(ValueError, match=message):
        launcher._resolve_sample(args)


@pytest.mark.parametrize("planner", ("trapezoidal", "toppra", "curobo"))
def test_sample_pour_retains_native_planner_sampling(planner: str) -> None:
    from embodichain.lab.scripts._atomic_action.pour import _create_pick_motion_policy

    policy = _create_pick_motion_policy(planner, fixed_sample_count=False)
    assert policy.plan_opts is None
    assert policy.strategy == "motion_gen"


def test_affordance_branch_identity_is_independent_of_slots() -> None:
    cfg = _sample_defaults("pickup")
    args = _scene_args(cfg, (8, 2, 4), 7)
    sampling = create_affordance_sampling_context(args)
    assert [sampling.branch_for(row, row_index=row) for row in range(3)] == [2, 0, 1]
    other = create_affordance_sampling_context(_scene_args(cfg, (8,), 0))
    assert torch.equal(
        sampling.generator("grasp").get_state(), other.generator("grasp").get_state()
    )


def _compiled(rows: int = 3) -> SimpleNamespace:
    # Two actions repeat their join at sample 40 with zero elapsed time.
    horizon = 71
    positions = torch.zeros(rows, horizon, 3)
    positions[:, :40, 0] = torch.linspace(0, 0.5, 40)
    positions[:, 40:, 0] = torch.linspace(0.5, 0.0, horizon - 40)
    positions[:, :, 2] = 0.04  # gripper is never an editable arm joint
    intervals = torch.full((rows, horizon), 0.01)
    intervals[:, 0] = 0
    intervals[:, 40] = 0
    segments = (
        (
            SimpleNamespace(name="approach", start=0, stop=25),
            SimpleNamespace(name="close", start=25, stop=40),
        ),
        (
            SimpleNamespace(name="approach", start=0, stop=20),
            SimpleNamespace(name="release", start=20, stop=31),
        ),
    )
    plans = tuple(
        SimpleNamespace(
            skill_id=skill, segments=group, diagnostics=SimpleNamespace(metadata={})
        )
        for skill, group in zip(("pick_up", "place"), segments)
    )

    def segment(index: int, name: str) -> SimpleNamespace:
        value = next(value for value in segments[index] if value.name == name)
        offset = 40 if index else 0
        return SimpleNamespace(
            name=name, start=value.start + offset, stop=value.stop + offset
        )

    return SimpleNamespace(
        trajectory=TimedTrajectory(
            positions, None, None, intervals, torch.arange(rows)
        ),
        action_plans=plans,
        segment=segment,
    )


class _Robot:
    joint_names = ("arm_0", "arm_1", "hand")

    def get_joint_ids(self, name: str) -> list[int]:
        assert name == "arm"
        return [0, 1]

    def get_qpos_limits(self) -> torch.Tensor:
        return torch.tensor([[-3.0, 3.0], [-3.0, 3.0], [0.0, 0.05]]).expand(3, -1, -1)


def test_both_spatial_operators_keep_contacts_hands_clock_and_action_joins() -> None:
    cfg = _sample_defaults("place")
    args = _scene_args(cfg, (0, 1, 2), 0)
    compiled = _compiled()
    expanded, records = _augment(compiled, _Robot(), args)
    assert records[0]["operator"] == "none"
    assert {record["operator"] for record in records[1:]} == {
        "joint_residual",
        "via_points",
    }
    assert torch.equal(expanded.dt, compiled.trajectory.dt)
    assert torch.equal(expanded.positions[0], compiled.trajectory.positions[0])
    assert torch.equal(
        expanded.positions[:, :, 2], compiled.trajectory.positions[:, :, 2]
    )
    for left, right in ((25, 40), (60, 71)):
        assert torch.equal(
            expanded.positions[:, left:right],
            compiled.trajectory.positions[:, left:right],
        )
    for endpoint in (0, 24, 39, 40, 59, 60, 70):
        assert torch.equal(
            expanded.positions[:, endpoint], compiled.trajectory.positions[:, endpoint]
        )
    assert not torch.equal(
        expanded.positions[1, :25, :2], compiled.trajectory.positions[1, :25, :2]
    )
    assert not torch.equal(
        expanded.positions[2, :25, :2], compiled.trajectory.positions[2, :25, :2]
    )


@pytest.mark.parametrize("join_dt", [0.0, 0.01])
@pytest.mark.parametrize("join_offset", [0.0, 1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pour_expansion_preserves_interior_rotated_target(
    join_dt: float, join_offset: float, dtype: torch.dtype
) -> None:
    """Both operators retain Pour's off-center target and still expand PickUp."""
    cfg = _sample_defaults("pour")
    compiled = _compiled()
    source = compiled.trajectory
    compiled.trajectory = TimedTrajectory(
        source.positions.to(dtype), None, None, source.dt.to(dtype), source.env_ids
    )
    pour_start = 40
    # A revolute joint encodes an out-and-back rotation. The target deliberately
    # occurs away from the segment midpoint and after the duplicate action join.
    poured_index = pour_start + 9
    horizon = compiled.trajectory.waypoint_count
    angle = cfg["options"]["rotate_angle"]
    positions = compiled.trajectory.positions
    positions[:, pour_start:, 0] = positions[:, pour_start - 1, 0:1]
    positions[:, pour_start : poured_index + 1, 1] = torch.linspace(
        0.0, angle, poured_index - pour_start + 1, dtype=dtype
    )
    positions[:, poured_index:, 1] = torch.linspace(
        angle, 0.0, horizon - poured_index, dtype=dtype
    )
    # The adapter permits near-identical zero-time joins within 1e-5. Dropping
    # this sample must not replace the protected Pour state with its predecessor.
    positions[:, pour_start, 0] += join_offset
    compiled.trajectory.dt[:, pour_start] = join_dt
    compiled.action_plans = (
        compiled.action_plans[0],
        SimpleNamespace(
            skill_id="pour",
            segments=(
                SimpleNamespace(name="pour", start=0, stop=horizon - pour_start),
            ),
            diagnostics=SimpleNamespace(metadata={}),
        ),
    )
    pickup_segment = compiled.segment

    def segment(index: int, name: str) -> SimpleNamespace:
        if index == 0:
            return pickup_segment(index, name)
        assert index == 1 and name == "pour"
        return SimpleNamespace(name=name, start=pour_start, stop=horizon)

    compiled.segment = segment
    expanded, records = _augment(compiled, _Robot(), _scene_args(cfg, (0, 1, 2), 0))

    assert {record["operator"] for record in records[1:]} == {
        "joint_residual",
        "via_points",
    }
    target = positions[0, poured_index]
    assert torch.equal(expanded.positions[:, poured_index], target.expand(3, -1))
    # With no exported leg boundary, protect the complete Pour interval rather
    # than guessing an anchor from time or a trajectory extremum.
    assert torch.equal(expanded.positions[:, pour_start:], positions[:, pour_start:])
    assert torch.equal(expanded.dt, compiled.trajectory.dt)
    for row in (1, 2):
        assert not torch.equal(expanded.positions[row, :25], positions[row, :25])


def test_trajectory_variant_is_stable_when_recipe_moves_to_another_batch() -> None:
    cfg = _sample_defaults("place")
    compiled = _compiled()
    together, _ = _augment(compiled, _Robot(), _scene_args(cfg, (0, 1, 2), 0))
    isolated, _ = _augment(_compiled(1), _Robot(), _scene_args(cfg, (2,), 9))
    assert torch.equal(together.positions[2], isolated.positions[0])


def test_spatial_variants_share_one_grasp_reference_per_branch() -> None:
    cfg = _sample_defaults("place")
    compiled = _compiled()
    # Model different feasible IK/grasp rankings on otherwise equal slots.
    compiled.trajectory.positions[1, :, 1] = 0.3
    compiled.trajectory.positions[2, :, 1] = 0.6
    expanded, records = _augment(compiled, _Robot(), _scene_args(cfg, (0, 1, 2), 0))
    assert [record["reference_row"] for record in records] == [0, 0, 0]
    assert torch.equal(
        expanded.positions[:, 25:40],
        compiled.trajectory.positions[0:1, 25:40].expand(3, -1, -1),
    )


def _compiled_slide_push() -> SimpleNamespace:
    """Model three pushes replanned from distinct post-pull robot/handle states."""
    phases = (
        ("approach", 25),
        ("reach", 15),
        ("close", 5),
        ("push", 25),
        ("open", 5),
        ("return", 25),
    )
    segments = []
    offset = 0
    for name, count in phases:
        segments.append(SimpleNamespace(name=name, start=offset, stop=offset + count))
        offset += count
    starts = torch.tensor([0.0, 0.12, -0.08])
    handles = torch.tensor([0.3, 0.47, 0.19])
    positions = torch.zeros(3, offset, 3)
    for segment in segments:
        count = segment.stop - segment.start
        if segment.name == "approach":
            x = torch.lerp(
                starts[:, None], handles[:, None] - 0.1, torch.linspace(0, 1, count)
            )
        elif segment.name == "reach":
            x = handles[:, None] - torch.linspace(0.1, 0, count)
        elif segment.name == "push":
            x = handles[:, None] - torch.linspace(0, 0.18, count)
        elif segment.name == "return":
            x = handles[:, None] - 0.18 + torch.linspace(0, 0.1, count)
        else:
            x = handles[:, None] - (0.18 if segment.name == "open" else 0)
        positions[:, segment.start : segment.stop, 0] = x
        if segment.name == "close":
            hand = torch.linspace(0, 0.04, count)
        elif segment.name == "open":
            hand = torch.linspace(0.04, 0, count)
        else:
            hand = 0.04 if segment.name == "push" else 0
        positions[:, segment.start : segment.stop, 2] = hand
    intervals = torch.tensor([0.01, 0.02, 0.04])[:, None].expand(3, offset).clone()
    intervals[:, 0] = 0
    metadata = {
        "affordance_sample": {
            "grasp": {"candidate_ids": [7, 7, 7], "reused": [False, False, False]},
            "measured_handle_x": handles.tolist(),
        }
    }

    def segment(action_index: int, name: str) -> SimpleNamespace:
        assert action_index == 0
        return next(value for value in segments if value.name == name)

    return SimpleNamespace(
        trajectory=TimedTrajectory(positions, None, None, intervals, torch.arange(3)),
        action_plans=(
            SimpleNamespace(
                skill_id="slide",
                segments=tuple(segments),
                diagnostics=SimpleNamespace(metadata=metadata),
            ),
        ),
        segment=segment,
    )


@pytest.mark.parametrize("recipes", [(0, 1, 2), (1, 0, 2)])
@pytest.mark.parametrize("hold_steps", [0, None])
def test_slide_push_variants_preserve_each_replanned_row(
    recipes: tuple[int, ...], hold_steps: int | None
) -> None:
    """Shared grasp identity must not replace live row starts, contacts or clocks."""
    cfg = _sample_defaults("slide")
    cfg["runtime"]["hold_steps"] = hold_steps
    compiled = _compiled_slide_push()
    source = compiled.trajectory
    expanded, records = _augment(compiled, _Robot(), _scene_args(cfg, recipes, 0))

    torch.testing.assert_close(
        expanded.positions[:, 0], source.positions[:, 0], rtol=0, atol=0
    )
    assert torch.equal(expanded.dt, source.dt)
    assert [record["reference_row"] for record in records] == [0, 1, 2]
    assert {record["operator"] for record in records} == {
        "none",
        "joint_residual",
        "via_points",
    }
    for segment in compiled.action_plans[0].segments:
        region = slice(segment.start, segment.stop)
        if segment.name in {"reach", "close", "push", "open"}:
            assert torch.equal(
                expanded.positions[:, region], source.positions[:, region]
            )
        else:
            for endpoint in (segment.start, segment.stop - 1):
                torch.testing.assert_close(
                    expanded.positions[:, endpoint],
                    source.positions[:, endpoint],
                    rtol=0,
                    atol=0,
                )
            for row, record in enumerate(records):
                if record["operator"] != "none":
                    assert not torch.equal(
                        expanded.positions[row, region, :2],
                        source.positions[row, region, :2],
                    )
    assert torch.equal(expanded.positions[:, :, 2], source.positions[:, :, 2])
    for row, record in enumerate(records):
        sample = record["affordance_samples"][0]
        assert sample["grasp"]["candidate_ids"] == 7
        assert (
            sample["measured_handle_x"]
            == compiled.action_plans[0].diagnostics.metadata["affordance_sample"][
                "measured_handle_x"
            ][row]
        )
        if record["operator"] == "none":
            assert torch.equal(expanded.positions[row], source.positions[row])


def test_zero_time_position_change_is_rejected() -> None:
    compiled = _compiled()
    compiled.trajectory.positions[0, 40, 0] += 0.1
    with pytest.raises(ValueError, match="zero time"):
        _template(compiled, 0, _Robot.joint_names, (0, 1), "place", 0)


def test_missing_distinct_variants_fail_without_reusing_nominal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.scripts._atomic_action import expansion

    cfg = _sample_defaults("place")
    fake = SimpleNamespace(
        variants=(SimpleNamespace(spatial_operator="none"),), rejected={"duplicates": 4}
    )
    monkeypatch.setattr(
        expansion, "expand_trajectory_variants", lambda *args, **kwargs: fake
    )
    with pytest.raises(RuntimeError, match="only 1 survived"):
        _augment(_compiled(1), _Robot(), _scene_args(cfg, (1,), 0))


def test_cli_batches_all_recipes_and_records_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = []

    def run_batch(cfg, indices, batch_id, cli_args, *, last):
        batches.append((indices, last))
        return [{"status": "executed", "batch_id": batch_id}]

    monkeypatch.setattr(launcher, "_run_batch", run_batch)
    launcher.cli(
        [
            "--skill",
            "place",
            "--num-envs",
            "3",
            "--recipe-indices",
            "0",
            "1",
            "2",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert batches == [((0, 1, 2), False), ((4,), True)]
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["status"] == "executed"
    assert len(manifest["replays"]) == 2
    assert manifest["measured_task_success"] is None


def test_cli_persists_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("IK failure")

    monkeypatch.setattr(launcher, "_run_batch", fail)
    with pytest.raises(RuntimeError, match="IK failure"):
        launcher.cli(
            [
                "--skill",
                "place",
                "--no-expansion",
                "--output-dir",
                str(tmp_path),
            ]
        )
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "IK failure" in manifest["error"]
