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

"""Component-owned robot construction and benchmark runtime integration."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from embodichain_benchmarks.motion_generation.config import (
    EmbodimentSpecCfg,
    RobotSpecCfg,
    SuiteCfg,
    load_suite,
    suite_to_dict,
)
from embodichain_benchmarks.motion_generation.embodiments import resolve_embodiment
from embodichain_benchmarks.motion_generation.robots.base import EmbodimentRobotProvider
from embodichain_benchmarks.motion_generation.runner import BenchmarkRunner

pytestmark = pytest.mark.no_sim


@pytest.fixture
def component(tmp_path: Path) -> Path:
    tcp = np.eye(4)
    tcp[2, 3] = 0.15
    data = {
        "embodiment_id": "custom_robot",
        "simulation": {
            "uid": "component_robot",
            "control_parts": {"custom_arm": ["joint1", "joint2"], "jaw": ["finger"]},
            "solver_cfg": {
                "custom_arm": {"class_type": "PytorchSolver", "tcp": tcp.tolist()},
            },
        },
        "sensor": [],
        "skill_profile": {
            "resources": [
                {
                    "resource_id": "primary",
                    "endpoints": [
                        {
                            "endpoint_id": "motion",
                            "control_part": "custom_arm",
                            "capabilities": ["motion.cartesian_pose"],
                        },
                        {
                            "endpoint_id": "grasp",
                            "control_part": "jaw",
                            "capabilities": ["interaction.grasp"],
                            "command_preset": "jaw_commands",
                        },
                    ],
                }
            ],
            "command_presets": [
                {
                    "preset_id": "jaw_commands",
                    "control_part": "jaw",
                    "commands": {"open": [0.0], "grasp": [0.024]},
                }
            ],
        },
    }
    path = tmp_path / "component.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _suite_data(component: str) -> dict:
    return {
        "name": "component_test",
        "embodiment": {"component": component},
        "planners": [{"id": "selected", "adapter": "curobo", "enabled": True}],
        "protocol": {"warmup_trials": 0, "measured_trials": 1},
        "tracks": [{"id": "atomic", "scenario": "atomic_skill", "config": {}}],
    }


def _resolve(component: Path, **kwargs):
    return resolve_embodiment(
        EmbodimentSpecCfg(component=str(component), **kwargs), RobotSpecCfg()
    )


def test_component_builds_robot_with_custom_parts_commands_and_tcp(
    component: Path,
) -> None:
    provider = EmbodimentRobotProvider(_resolve(component))
    cfg = provider.build_cfg()

    assert cfg.uid == "component_robot"
    assert provider.spec.id == "custom_robot"
    assert provider.control_part == "custom_arm"
    assert cfg.control_parts["jaw"] == ["finger"]
    assert cfg.solver_cfg["custom_arm"].tcp[2][3] == pytest.approx(0.15)
    assert provider.gripper_config == {
        "control_part": "jaw",
        "open_qpos": [0.0],
        "grasp_qpos": [0.024],
    }
    cfg.control_parts["jaw"].clear()
    assert provider.build_cfg().control_parts["jaw"] == ["finger"]


def test_component_applies_initial_state_overrides_and_changes_hash(
    component: Path,
) -> None:
    original = _resolve(component)
    resolved = _resolve(
        component, overrides={"uid": "overridden"}, initial_qpos=[0.1, 0.2, 0.0]
    )
    cfg = EmbodimentRobotProvider(resolved).build_cfg()

    assert cfg.uid == "overridden"
    assert cfg.init_qpos == [0.1, 0.2, 0.0]
    assert resolved.config_hash != original.config_hash
    with pytest.raises(ValueError, match="not both"):
        _resolve(component, overrides={"init_qpos": [0.0]}, initial_qpos=[0.1])


def test_relative_component_is_resolved_from_suite_not_cwd(
    component: Path, tmp_path: Path, monkeypatch
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(_suite_data(component.name)))
    other = tmp_path / "other"
    other.mkdir()
    (other / component.name).write_text("invalid: cwd_shadow")
    monkeypatch.chdir(other)

    suite = load_suite(str(suite_path))

    assert Path(suite.embodiment.component) == component
    assert (
        BenchmarkRunner(suite, suite.planners, device="cpu").suite.robot.id
        == "custom_robot"
    )


def test_component_and_inline_robot_cannot_both_own_runtime(component: Path) -> None:
    data = _suite_data(str(component))
    data["robot"] = {"provider": "ur5"}
    with pytest.raises(ValueError, match="owns robot"):
        SuiteCfg.from_dict(data)


def test_missing_owner_component_does_not_load_cwd_shadow(
    component: Path, tmp_path: Path, monkeypatch
) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    suite_path = owner / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(_suite_data(component.name)))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="owner"):
        load_suite(str(suite_path))


@pytest.mark.parametrize("endpoint", ["motion", "grasp"])
def test_invalid_endpoint_binding_fails_before_simulation(
    component: Path, endpoint: str
) -> None:
    resolved = _resolve(
        component, endpoint_bindings={"primary": {endpoint: "missing_part"}}
    )
    with pytest.raises(ValueError, match="control part"):
        EmbodimentRobotProvider(resolved)


def test_multiple_resources_need_explicit_selection(component: Path) -> None:
    data = yaml.safe_load(component.read_text())
    secondary = deepcopy(data["skill_profile"]["resources"][0])
    secondary["resource_id"] = "secondary"
    data["skill_profile"]["resources"].append(secondary)
    component.write_text(yaml.safe_dump(data))
    resolved = _resolve(component)

    with pytest.raises(ValueError, match="resource_id"):
        EmbodimentRobotProvider(resolved)
    assert EmbodimentRobotProvider(resolved, "secondary").control_part == "custom_arm"


def test_runner_uses_component_robot_and_preserves_input_suite(
    component: Path, monkeypatch
) -> None:
    from embodichain_benchmarks.motion_generation import runner as runner_module

    suite = SuiteCfg.from_dict(_suite_data(str(component)))
    runner = BenchmarkRunner(suite, suite.planners, device="cpu")
    factory = Mock()
    monkeypatch.setattr(runner_module, "SimulationManager", factory)

    sim, robot = runner._create_simulation(1)

    assert sim is factory.return_value
    assert robot is sim.add_robot.return_value
    cfg = sim.add_robot.call_args.kwargs["cfg"]
    assert cfg.uid == "component_robot"
    assert runner.control_part == "custom_arm"
    assert runner.suite.robot.id == "custom_robot"
    assert runner.suite.tracks[0].config["gripper"]["control_part"] == "jaw"
    assert "gripper" not in suite.tracks[0].config
    resolved = suite_to_dict(runner.suite)
    assert "robot" not in resolved
    assert SuiteCfg.from_dict(resolved).embodiment.component == str(component)


def test_gripper_and_domain_overrides_cannot_change_component_commands(
    component: Path,
) -> None:
    from embodichain_benchmarks.motion_generation.config import DomainCfg

    suite = SuiteCfg.from_dict(_suite_data(str(component)))
    suite.tracks[0].config["gripper"] = {"control_part": "wrong"}
    with pytest.raises(ValueError, match="conflicts"):
        BenchmarkRunner(suite, suite.planners, device="cpu")
    suite.tracks[0].config.clear()
    suite.tracks[0].domains = [
        DomainCfg(
            id="changed",
            version="1",
            kind="robustness",
            overrides={"gripper": {"grasp_qpos": [0.9]}},
        )
    ]
    runner = BenchmarkRunner(suite, suite.planners, device="cpu")
    with pytest.raises(ValueError, match="conflicts"):
        runner.robot_provider.configure_track(runner.suite.evaluation_tracks()[0])


def test_official_ur5_component_uses_robot_catalog(monkeypatch, tmp_path: Path) -> None:
    from embodichain.lab.sim import cfg as cfg_module
    from embodichain_robots.universal_robots import ur_robot

    def local_asset(path: str) -> str:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('<robot name="fixture"><link name="base_link"/></robot>')
        return str(target)

    monkeypatch.setattr(ur_robot, "get_data_path", local_asset)
    monkeypatch.setattr(cfg_module, "get_data_path", local_asset)
    provider = EmbodimentRobotProvider(
        _resolve(
            Path(
                "embodichain_tasks/configs/components/embodiments/ur5_dh_pgi_140_80.yaml"
            )
        )
    )
    cfg = provider.build_cfg()

    assert isinstance(cfg, ur_robot.URRobotCfg)
    assert len(cfg.control_parts["arm"]) == 6
    assert cfg.control_parts["eef"] == ["gripper_finger1_joint_1"]
    assert cfg.solver_cfg["arm"].tcp[2][3] == pytest.approx(0.15)
    assert provider.gripper_config["grasp_qpos"] == [0.024]


@pytest.mark.parametrize("component_owned", [False, True])
@pytest.mark.parametrize(
    "case_identity", ["missing", "matching", "robot_mismatch", "embodiment_mismatch"]
)
def test_suite_validates_case_identity_before_planning_and_persistence(
    component: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component_owned: bool,
    case_identity: str,
) -> None:
    from embodichain_benchmarks.motion_generation import runner as runner_module
    from embodichain_benchmarks.motion_generation.metrics.trajectory import (
        make_failure_outcomes,
    )
    from embodichain_benchmarks.motion_generation.models import BenchmarkCase
    from embodichain_benchmarks.motion_generation.planners.base import PlannerAdapter
    from embodichain_benchmarks.motion_generation.scenarios.base import (
        ScenarioEvaluation,
        ScenarioProvider,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanResult

    observed = []
    close_runtime = Mock()
    expected_id = "custom_robot" if component_owned else "franka_panda"
    expected_control_part = "custom_arm" if component_owned else "arm"

    class FakeScenario(ScenarioProvider):
        def batch_sizes(self, suite, track):
            return [1]

        def configure_runtime(self, simulation, robot, suite, track, control_part):
            assert control_part == expected_control_part
            if component_owned:
                assert track.config["gripper"]["control_part"] == "jaw"

        def close_runtime(self):
            close_runtime()

        def generate_cases(self, suite, track, robot, control_part, batch_size):
            return [
                BenchmarkCase(
                    suite_version=suite.suite_version,
                    track=track.id,
                    scenario_id="fixture",
                    case_id="fixture_case",
                    seed=11,
                    batch_size=1,
                    num_waypoints=1,
                    path_shape="fixture",
                    start_state_bin="initial",
                    start_qpos=torch.zeros(1, 2),
                    target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
                    reference_qpos=torch.zeros(1, 1, 2),
                    robot_id=(
                        "other_robot"
                        if case_identity == "robot_mismatch"
                        else expected_id
                    ),
                    embodiment_id=(
                        None
                        if case_identity == "missing"
                        else (
                            "other_embodiment"
                            if case_identity == "embodiment_mismatch"
                            else expected_id
                        )
                    ),
                )
            ]

        def evaluate_case(self, result, case, robot, control_part, suite, **kwargs):
            # This fixture exercises wiring and persistence, not physical success.
            return ScenarioEvaluation(
                outcomes=make_failure_outcomes(1, "fixture_failure")
            )

    class FakeAdapter(PlannerAdapter):
        capabilities = frozenset({"eef_waypoint"})

        def build(self):
            observed.append(
                (self.spec.id, self.context.robot.uid, self.context.control_part)
            )

        def plan(self, case):
            return PlanResult(
                success=torch.ones(1, dtype=torch.bool),
                positions=torch.zeros(1, 2, 2),
                dt=torch.tensor([[0.0, 0.1]]),
            )

    data = _suite_data(str(component))
    if not component_owned:
        data.pop("embodiment")
        data["robot"] = {"id": expected_id, "provider": "franka_panda"}
    suite = SuiteCfg.from_dict(data)
    robot = Mock(uid="component_robot")
    sim = Mock()
    sim.add_robot.return_value = robot
    factory = Mock(return_value=sim)
    monkeypatch.setattr(runner_module, "SimulationManager", factory)
    if not component_owned:
        robot_provider = Mock(control_part=expected_control_part)
        robot_provider.add_robot.return_value = robot
        monkeypatch.setattr(
            runner_module, "create_robot_provider", lambda spec: robot_provider
        )
    monkeypatch.setattr(
        runner_module, "create_scenario_provider", lambda name: FakeScenario()
    )
    monkeypatch.setattr(
        runner_module,
        "create_planner_adapter",
        lambda spec, context: FakeAdapter(spec, context),
    )

    runner = BenchmarkRunner(suite, suite.planners, device="cpu", output_root=tmp_path)
    if case_identity in {"robot_mismatch", "embodiment_mismatch"}:
        with pytest.raises(ValueError, match="conflicts with suite embodiment"):
            runner.run()
        assert not observed
        assert runner.cases == []
        assert runner.records == []
        assert not list(tmp_path.rglob("case_manifest.json"))
        close_runtime.assert_called_once_with()
        sim.destroy.assert_called_once_with(exit_process=False)
        factory.flush_cleanup_queue.assert_called_once()
        return
    result = runner.run()

    assert observed == [("selected", "component_robot", expected_control_part)]
    rows = [json.loads(line) for line in result.trials_path.read_text().splitlines()]
    assert {row["embodiment_id"] for row in rows} == {expected_id}
    assert {row["robot_id"] for row in rows} == {expected_id}
    assert {row["algorithm_id"] for row in rows} == {"selected"}
    manifest = json.loads((result.run_dir / "case_manifest.json").read_text())
    assert manifest["resolved_embodiment"]["embodiment_id"] == expected_id
    assert {case["embodiment_id"] for case in manifest["cases"]} == {expected_id}
    assert {case["robot_id"] for case in manifest["cases"]} == {expected_id}
    if component_owned:
        assert (
            manifest["resolved_embodiment"]["endpoint_bindings"]["primary"]["motion"]
            == "custom_arm"
        )
    assert (result.run_dir / "aggregates.json").is_file()
    assert "Leaderboard" in result.report_path.read_text()
    close_runtime.assert_called_once_with()
    sim.destroy.assert_called_once_with(exit_process=False)
    factory.flush_cleanup_queue.assert_called_once()


def test_example_has_nominal_and_position_domains() -> None:
    from embodichain_benchmarks.motion_generation.config import (
        resolve_atomic_batch_sizes,
    )

    suite = load_suite("atomic_ur5_embodiment")
    tracks = suite.evaluation_tracks()
    assert [track.domain.kind for track in tracks] == ["nominal", "robustness"]
    assert [resolve_atomic_batch_sizes(track) for track in tracks] == [[1], [2]]
    assert suite.planners[0].adapter == "curobo"
    assert "gripper" not in suite.tracks[0].config


@pytest.mark.parametrize("tuned", [False, True], ids=["frozen", "handover"])
def test_dual_component_binds_roles_commands_geometry_and_mirrored_domain(
    tuned: bool,
) -> None:
    """Both hands must come from the component, including after exchanging roles."""
    from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
        AtomicSkillScenario,
    )

    suite = load_suite("atomic_dual_franka_robotiq_hand_over")
    if not tuned:
        suite.embodiment.component = (
            "embodichain_tasks/configs/components/embodiments/"
            "dual_franka_robotiq_arg2f_140.yaml"
        )
    closure = 0.5 if tuned else 0.7
    tcp_z = 0.21 if tuned else 0.2
    grasp_qpos = [closure, -closure, closure, -closure, -closure, closure]
    runner = BenchmarkRunner(suite, suite.planners, device="cpu")
    provider = runner.robot_provider
    cfg = provider.build_cfg()
    track = runner.suite.tracks[0]
    assert provider.control_part == "right_arm"
    reordered = deepcopy(suite)
    reordered.embodiment.resource_bindings = {"destination": "left", "source": "right"}
    assert (
        BenchmarkRunner(reordered, reordered.planners, device="cpu").control_part
        == "right_arm"
    )
    assert track.config["control_parts"] == {
        "source": {"motion": "right_arm", "grasp": "right_eef"},
        "destination": {"motion": "left_arm", "grasp": "left_eef"},
    }
    assert [item["control_part"] for item in track.config["grippers"]] == [
        "right_eef",
        "left_eef",
    ]
    for item in track.config["grippers"]:
        assert item["grasp_qpos"] == grasp_qpos
    assert cfg.solver_cfg["right_arm"].tcp[2][3] == pytest.approx(tcp_z)
    assert cfg.joint_drive_props.stiffness["right_eef"] == (1000.0 if tuned else 50.0)
    assert cfg.init_qpos[12:14] == ([0.741, 0.741] if tuned else [0.0, 0.0])
    for arm in ("left_arm", "right_arm"):
        assert cfg.solver_cfg[arm].class_type == (
            "FEPSolver" if tuned else "PytorchSolver"
        )
        assert cfg.joint_drive_props.max_effort[arm] == (100000.0 if tuned else 10000.0)
    for hand in ("left_eef", "right_eef"):
        assert cfg.joint_drive_props.damping[hand] == (100.0 if tuned else 5.0)
        assert cfg.joint_drive_props.max_effort[hand] == (10000.0 if tuned else 500.0)
    assert "grippers" not in suite.tracks[0].config
    assert SuiteCfg.from_dict(
        suite_to_dict(runner.suite)
    ).embodiment.resource_bindings == {"source": "right", "destination": "left"}

    mirrored = deepcopy(track)
    mirrored.config["control_parts"] = {
        "source": {"motion": "left_arm", "grasp": "left_eef"},
        "destination": {"motion": "right_arm", "grasp": "right_eef"},
    }
    provider.configure_track(mirrored)
    assert mirrored.config["control_parts"]["source"]["motion"] == "left_arm"
    invalid = deepcopy(mirrored)
    invalid.config["control_parts"]["source"]["grasp"] = "right_eef"
    with pytest.raises(ValueError, match="control_parts"):
        provider.configure_track(invalid)
    invalid = deepcopy(track)
    invalid.config["grippers"][0]["grasp_qpos"] = [0.5] * 6
    with pytest.raises(ValueError, match="conflicts"):
        provider.configure_track(invalid)

    scenario = AtomicSkillScenario()
    scenario.suite = runner.suite
    scenario.track = track
    scenario.end_effector_part = "right_eef"
    scenario._gripper_states = {
        part: (torch.zeros(6), torch.zeros(6)) for part in ("right_eef", "left_eef")
    }
    scenario._configure_embodiment_services()
    assert scenario.grasp_model().model_id == "robotiq_arg2f_140"
    assert scenario.grasp_model().max_opening_width == pytest.approx(
        0.14 if tuned else 0.128
    )
    assert scenario.grasp_root_from_tcp()[2][3] == pytest.approx(tcp_z)
    assert set(scenario._hand_geometry_cfgs) == {"left_eef", "right_eef"}
    assert scenario._hand_geometry_cfgs["right_eef"].pad_link_names == (
        "right_left_inner_finger_pad",
        "right_inner_finger_pad",
    )
    assert tuple(scenario._hand_geometry_cfgs["right_eef"].closed_qpos) == tuple(
        grasp_qpos
    )
    # Carry the same explicit geometry into the actual benchmark engine.
    adapter = Mock()
    adapter.require_motion_generator.return_value.device = torch.device("cpu")
    scenario.prepare_planner(adapter, Mock())
    assert set(scenario.require_engine().hand_geometry_cfgs) == {
        "left_eef",
        "right_eef",
    }


@pytest.mark.parametrize(
    "bindings",
    [
        {"source": "left", "destination": "left"},
        {"source": "left", "destination": "missing"},
    ],
)
def test_component_rejects_invalid_coordinated_resource_selection(bindings) -> None:
    suite = load_suite("atomic_dual_franka_robotiq_hand_over")
    suite.embodiment.resource_bindings = bindings
    with pytest.raises(ValueError, match="distinct|Unknown"):
        BenchmarkRunner(suite, suite.planners, device="cpu")


def test_handover_generalization_preserves_qualified_population() -> None:
    """Package the measured 50-environment design without dropping hard domains."""
    from embodichain_benchmarks.motion_generation.config import (
        resolve_atomic_batch_sizes,
    )

    suite = load_suite("atomic_dual_franka_robotiq_hand_over_spatial_generalization")
    assert suite.embodiment.component.endswith(
        "dual_franka_robotiq_arg2f_140_handover.yaml"
    )
    expected = {
        "nominal": 1,
        "bilateral_role_mirror": 1,
        **{
            name: 8
            for name in (
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "task_frame_planar",
                "object_target_position",
                "target_relative_shell",
            )
        },
    }
    actual = {
        track.domain.id: sum(resolve_atomic_batch_sizes(track))
        * len(track.config["seeds"])
        for track in suite.evaluation_tracks()
    }
    assert actual == expected
    assert sum(actual.values()) == 50
    for track in suite.evaluation_tracks():
        assert track.config["physics"]["joint_tracking_scope"] == "motion_control_parts"
        assert track.config["physics"]["joint_tracking_tolerance_rad"] == 0.05
