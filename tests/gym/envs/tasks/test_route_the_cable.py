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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET
import math
import yaml
import pytest
import torch
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
    config_to_cfg,
)
from embodichain_tasks.manipulation.cable.route_the_cable import (
    measure_route,
    RouteTheCableEnv,
)
from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.cfg import CableObjectCfg
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_trajectory import (
    ExpertTrajectoryCfg,
    build_expert_action_spec,
)
from embodichain.lab.sim.shapes import CylinderCfg

pytestmark = pytest.mark.no_sim


@pytest.fixture(autouse=True)
def local_yam_config_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep configuration tests independent of external model publication."""
    from embodichain_robots import yam_i2rt

    robot = ET.Element("robot", name="yam_config_fixture")
    ET.SubElement(robot, "link", name="base_link")
    for side in ("right", "left"):
        parent = "base_link"
        for child, number in [
            (f"{side}_base", None),
            *[(f"{side}_link{i}", i) for i in range(1, 7)],
            (f"{side}_tcp", None),
        ]:
            ET.SubElement(robot, "link", name=child)
            joint = ET.SubElement(
                robot,
                "joint",
                name=f"{side}_joint{number}" if number else f"{child}_fixed",
                type="revolute" if number else "fixed",
            )
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            if number:
                ET.SubElement(joint, "axis", xyz="0 0 1")
                ET.SubElement(
                    joint, "limit", lower="-3", upper="3", effort="27", velocity="3"
                )
            parent = child
    path = tmp_path / "yam_i2rt.urdf"
    ET.ElementTree(robot).write(path, encoding="unicode")

    def resolve(name: str) -> str:
        assert name == "YamI2rt/yam_i2rt.urdf"
        return str(path)

    monkeypatch.setattr(yam_i2rt, "get_data_path", resolve)
    return path


def test_route_deployment_composes_registered_robot_cable_and_board():
    p = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml"
    )
    cfg = config_to_cfg(yaml.safe_load(p.read_text()), source_path=p)
    assert len(cfg.cable_object) == 1 and len(cfg.cable_object[0].points) == 51
    assert cfg.cable_object[0].fixed_segments == [0]
    points = torch.tensor(cfg.cable_object[0].points) + torch.tensor(
        cfg.cable_object[0].init_pos
    )
    torch.testing.assert_close(points[:, 2], torch.full((51,), 0.7535))
    assert bool(
        (
            (points[:, 0] >= 0.3025)
            & (points[:, 0] <= 0.8975)
            & (points[:, 1].abs() <= 0.65)
        ).all()
    )
    lengths = points.diff(dim=0).norm(dim=-1)
    cumulative = lengths.cumsum(0)
    distance = cumulative[-1] - 0.030
    index = int(torch.searchsorted(cumulative, distance))
    grasp = torch.lerp(
        points[index],
        points[index + 1],
        (distance - cumulative[index - 1]) / lengths[index],
    )
    torch.testing.assert_close(
        grasp[:2], torch.tensor([0.42, 0.21]), atol=0.002, rtol=0
    )
    assert lengths.max() <= 0.0121
    assert lengths.min() > 0.0115
    assert len(cfg.background) == 11
    assert cfg.cable_object[0].attrs.density == 350.0
    assert cfg.cable_object[0].attrs.stretch_stiffness == 2e5
    assert cfg.cable_object[0].attrs.material_props.kd == 0.1
    assert cfg.cable_object[0].attrs.material_props.dynamic_friction == 60.0
    assert cfg.robot.link_attrs["pads"].attrs.material_props.dynamic_friction == 60.0
    assert all(
        obj.attrs.material_props.dynamic_friction == 2.0
        for obj in cfg.background
        if obj.uid.startswith("spool_")
    )
    assert cfg.sim_cfg.physics_cfg.solver_cfg["coupling"] == "two_way"
    assert cfg.sim_cfg.physics_cfg.solver_cfg["vbd_options"]["rigid_contact_history"]
    assert cfg.robot.joint_drive_props.stiffness["right_joint[1-6]"] == 300.0
    assert cfg.robot.joint_drive_props.stiffness[".*_joint[78]"] == 6000.0
    assert cfg.robot.joint_drive_props.damping[".*_joint[78]"] == 120.0
    assert cfg.sim_cfg.render_cfg.renderer == "no-render"
    assert not cfg.sim_cfg.visualization.allow_commands
    assert cfg.robot.fpath is None
    assert Path(cfg.robot.urdf_cfg.assemble_urdf()).is_file()
    for side, chain in cfg.robot.build_pk_serial_chain().items():
        assert (
            len(chain.get_joint_parameter_names())
            == len(cfg.robot.control_parts[side])
            == 6
        )


def test_route_viser_launcher_keeps_expert_deployment_read_only():
    import argparse

    path = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml"
    )
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    args = parser.parse_args(["--gym_config", str(path), "--viser"])

    cfg, _, _ = build_env_cfg_from_args(args)

    assert cfg.sim_cfg.visualization.backend == "viser"
    assert not cfg.sim_cfg.visualization.allow_commands


def _goal_state():
    state = torch.zeros(1, 50, 13)
    state[..., 3] = -math.sqrt(0.5)
    state[..., 6] = math.sqrt(0.5)
    state[..., :3] = torch.tensor([0.42, 0, 0.7535])
    state[0, :2, :3] = torch.tensor([0.50, -0.26, 0.7535])
    state[0, 2:4, :3] = torch.tensor([0.55, -0.02, 0.7535])
    state[0, -3:, :3] = torch.tensor([0.586, 0.078, 0.7535])
    return state


def test_reference_goal_is_measured_and_rejects_moving_or_held_cable():
    state = _goal_state()
    lengths = torch.full((50,), 0.012)
    hands = torch.tensor([[[0, 0, 1], [0, 1, 1.0]]])
    origin = torch.zeros(1, 3)
    assert measure_route(state, lengths, hands, origin)["reference_routed"].item()
    moving = state.clone()
    moving[0, 20, 7] = 0.051
    assert not measure_route(moving, lengths, hands, origin)["reference_routed"].item()
    hands[:, 0] = state[:, 0, :3]
    assert not measure_route(state, lengths, hands, origin)["reference_routed"].item()


def test_route_metrics_remove_arena_offsets_once():
    state = _goal_state().expand(2, -1, -1).clone()
    origin = torch.tensor([[0, 0, 0], [3, 5, 0.0]])
    state[..., :3] += origin[:, None, :]
    hands = (
        torch.tensor([[[0, 0, 1], [0, 1, 1.0]]]).expand(2, -1, -1) + origin[:, None, :]
    )
    out = measure_route(state, torch.full((50,), 0.012), hands, origin)
    for value in out.values():
        torch.testing.assert_close(value[0], value[1])


def test_cylinder_uses_native_collision_geometry():
    desc, _ = rigid_desc_from_cfg(
        RigidObjectCfg(
            uid="spool",
            shape=CylinderCfg(radius=0.0125, height=0.0025),
            body_type="static",
        ),
        newton_solver_type="dexuni",
    )
    assert desc.collisions[0].radius == 0.0125
    assert desc.collisions[0].half_height == 0.00125


def test_cable_deployment_rejects_duplicate_component_ownership():
    p = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml"
    )
    data = yaml.safe_load(p.read_text())
    data["cable_object"] = []
    with pytest.raises(ValueError, match="cable_object"):
        config_to_cfg(data, source_path=p)


def test_strict_channel_rejects_end_that_only_meets_broad_region():
    state = _goal_state()
    lengths = torch.full((50,), 0.012)
    hands = torch.tensor([[[0, 0, 1], [0, 1, 1.0]]])
    origin = torch.zeros(1, 3)
    assert measure_route(state, lengths, hands, origin)["strict_port"].item()
    state[0, -1, 0] += 0.014
    metrics = measure_route(state, lengths, hands, origin)
    assert metrics["reference_routed"].item()
    assert not metrics["strict_port"].item()


def test_object_goal_stops_before_writing_an_unexecuted_command():
    from types import SimpleNamespace

    command = torch.full((1, 3), 0.5)
    env = SimpleNamespace(
        _command=command,
        _grasp=None,
        _max_slip=torch.zeros(1),
        cfg=SimpleNamespace(
            expert_trajectory=SimpleNamespace(joint_command_mode="position")
        ),
    )
    actions = RouteTheCableEnv._execute(
        env, torch.ones(1, 5, 3), [0, 1, 2], stop_when=lambda: torch.tensor([True])
    )
    assert list(actions) == []
    torch.testing.assert_close(command, torch.full_like(command, 0.5))


def _task_cfg(segments=50):
    return EmbodiedEnvCfg(
        cable_object=[
            CableObjectCfg(
                uid="cable",
                points=[[0, 0.012 * i, 0] for i in range(segments + 1)],
                radius=0.0035,
            )
        ]
    )


@pytest.mark.parametrize("segments", [0, 49, 51])
def test_wrong_cable_topology_fails_before_environment_construction(
    monkeypatch, segments
):
    parent_init = Mock()
    monkeypatch.setattr(EmbodiedEnv, "__init__", parent_init)
    with pytest.raises(ValueError, match="50-segment"):
        RouteTheCableEnv(_task_cfg(segments))
    parent_init.assert_not_called()


def test_missing_cable_fails_before_environment_construction(monkeypatch):
    parent_init = Mock()
    monkeypatch.setattr(EmbodiedEnv, "__init__", parent_init)
    with pytest.raises(ValueError, match="named 'cable'"):
        RouteTheCableEnv(EmbodiedEnvCfg())
    parent_init.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_post_base_failure_closes_dataset_and_queues_simulator_cleanup(
    monkeypatch, cleanup_fails
):
    simulator, dataset = Mock(), Mock()
    if cleanup_fails:
        dataset.finalize.side_effect = RuntimeError("recorder failure")

    def initialize_parent(env, cfg, **kwargs):
        env.cfg = cfg
        env.sim = simulator
        env.dataset_manager = dataset
        env._profiler = Mock()
        env._discard_pending_recordings = Mock()

    simulator.get_cable_object.return_value = SimpleNamespace(
        data=SimpleNamespace(num_segments=49)
    )
    monkeypatch.setattr(EmbodiedEnv, "__init__", initialize_parent)
    with pytest.raises(ValueError, match="50-segment"):
        RouteTheCableEnv(_task_cfg())
    dataset.finalize.assert_called_once_with()
    simulator.destroy.assert_called_once_with(exit_process=False)


def _command_env(mode):
    env = object.__new__(RouteTheCableEnv)
    env.cfg = EmbodiedEnvCfg(
        sim_steps_per_control=1,
        expert_trajectory=ExpertTrajectoryCfg(joint_command_mode=mode),
    )
    env.sim_cfg = SimpleNamespace(physics_dt=0.1)
    env._num_envs = 2
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._command = torch.zeros(2, 4)
    env._grasp = None
    env._max_slip = torch.zeros(2)
    env.robot = SimpleNamespace(get_qpos=lambda: torch.zeros(2, 4))
    env.active_joint_ids = [1, 3]
    env.action_manager = None
    env._traj_buffer = object()
    env._trajectory_action_kind = "expert"
    env._record_raw_actions = False
    env.expert_action_spec = build_expert_action_spec(
        joint_names=["a", "b"],
        joint_command_mode=mode,
    )
    return env


@pytest.mark.parametrize("mode", ["position", "position_velocity"])
def test_motion_commands_pass_environment_preprocessing_and_recording(mode):
    env = _command_env(mode)
    path = torch.zeros(2, 3, 4)
    path[:, :, 1] = torch.tensor([0.1, 0.2, 0.3])
    path[:, :, 3] = torch.tensor([-0.2, -0.4, -0.6])
    # Non-selected planner columns must not affect the inactive commands.
    path[:, :, 0] = 9
    actions = list(env._execute(path, [1, 3]))
    assert len(actions) == 3
    for i, action in enumerate(actions):
        env._preprocess_action(action)
        expected = path[:, i, [1, 3]]
        torch.testing.assert_close(env._traj_raw_action[:, :2], expected)
        if mode == "position_velocity":
            velocity = (
                torch.tensor([1.0, -2.0]).expand(2, -1) if i < 2 else torch.zeros(2, 2)
            )
            torch.testing.assert_close(env._traj_raw_action[:, 2:], velocity)
            torch.testing.assert_close(
                action.value["qvel"][:, [0, 2]], torch.zeros(2, 2)
            )
        actual = action.value if mode == "position" else action.value["qpos"]
        torch.testing.assert_close(actual[:, [0, 2]], torch.zeros(2, 2))
    # Yielded commands must remain independent when the next frame is emitted.
    first = actions[0].value if mode == "position" else actions[0].value["qpos"]
    torch.testing.assert_close(first[:, 1], torch.full((2,), 0.1))


def test_hold_emits_zero_velocity_and_can_be_recorded():
    env = _command_env("position_velocity")
    env._command.fill_(0.25)
    actions = list(env._hold(0.2))
    assert len(actions) == 2
    for action in actions:
        env._preprocess_action(action)
        torch.testing.assert_close(
            env._traj_raw_action, torch.tensor([[0.25, 0.25, 0, 0]]).expand(2, -1)
        )


def test_grasp_slip_stops_execution_before_mutating_commands():
    env = _command_env("position_velocity")
    env._grasp = ("left", 0.57, torch.zeros(2, 3))
    env._max_slip[1] = 0.013
    assert list(env._execute(torch.ones(2, 3, 4), [1, 3])) == []
    torch.testing.assert_close(env._command, torch.zeros(2, 4))


def test_close_callback_observes_the_executed_frame():
    env = _command_env("position")
    observations = []
    env._point_in_tool = lambda *_: observations.append("observed") or torch.ones(2, 3)
    actions = env._execute(
        torch.ones(2, 2, 4), [1, 3], closed_at=1, grasp_ref=("right", 0.162)
    )
    next(actions)
    assert observations == [] and env._grasp is None
    next(actions)
    assert observations == ["observed"]
    assert env._grasp[:2] == ("right", 0.162)


@pytest.mark.parametrize("invalid", ["motion", "nonfinite", "held", "empty_seat"])
def test_reference_metrics_reject_invalid_or_incomplete_observations(invalid):
    state = _goal_state()
    hands = torch.ones(1, 2, 3)
    if invalid == "motion":
        state[0, 0, 7] = 0.051
    elif invalid == "nonfinite":
        state[0, 0, 0] = float("nan")
    elif invalid == "held":
        hands[0, 0] = state[0, -1, :3]
    else:
        state[0, :2, 0] = 0.42
    assert not measure_route(state, torch.full((50,), 0.012), hands, torch.zeros(1, 3))[
        "reference_routed"
    ].item()


def test_contact_penetration_uses_the_configured_native_back_wall():
    p = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml"
    )
    cfg = config_to_cfg(yaml.safe_load(p.read_text()), source_path=p)
    wall = next(obj for obj in cfg.background if obj.uid == "port_back")
    inner_y = wall.init_pos[1] - wall.shape.size[1] / 2
    state = torch.zeros(1, 1, 13)
    state[..., 6] = 1
    state[..., :3] = torch.tensor(
        [0.586, inner_y - cfg.cable_object[0].radius + 0.002, 0.755]
    )
    metrics = measure_route(
        state, torch.tensor([0.001]), torch.ones(1, 2, 3), torch.zeros(1, 3)
    )
    assert metrics["port_penetration"].item() == pytest.approx(0.002, abs=2e-7)


def test_unknown_cable_fields_are_rejected_by_production_loader(tmp_path):
    p = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml"
    )
    data = yaml.safe_load(p.read_text())
    physical = yaml.safe_load((p.parent / "env.yaml").read_text())
    physical["simulation"]["cable_object"][0]["unknown_cable_option"] = True
    component = tmp_path / "env.yaml"
    component.write_text(yaml.safe_dump(physical))
    data["environment"]["component"] = str(component)
    with pytest.raises((TypeError, ValueError), match="unknown_cable_option"):
        config_to_cfg(data, source_path=p)


@pytest.mark.parametrize(
    "planning_success,tracking_error,object_goal,slip,expected",
    [
        (False, 0.0, False, 0.0, False),
        (False, 0.015, True, 0.0, False),
        (True, 0.015, False, 0.0, False),
        (True, 0.001, False, 0.0, True),
        (True, 0.015, True, 0.0, True),
        (True, 0.001, True, 0.013, False),
    ],
)
@pytest.mark.parametrize("extra_samples", [0, 7])
def test_insertion_validator_requires_planning_and_actual_tracking(
    monkeypatch,
    planning_success,
    tracking_error,
    object_goal,
    slip,
    expected,
    extra_samples,
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    warning = Mock()
    monkeypatch.setattr(
        "embodichain_tasks.manipulation.cable.route_the_cable.logger.log_warning",
        warning,
    )

    actual = torch.eye(4).unsqueeze(0)
    target = actual.clone()
    target[:, 2, 3] += tracking_error

    def generate(states, *, options):
        return SimpleNamespace(
            positions=(
                torch.zeros(1, options.sample_count + extra_samples, 3)
                if planning_success
                else None
            ),
            success=torch.tensor([planning_success]),
        )

    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=1,
        step_dt=1 / 60,
        _tool=lambda side: actual,
        _command=torch.zeros(1, 3),
        _max_slip=torch.tensor([slip]),
        _grasp=None,
        robot=SimpleNamespace(
            get_joint_ids=lambda name: [0, 1, 2], get_qpos=lambda: torch.zeros(1, 3)
        ),
        _motion=SimpleNamespace(generate=generate),
        _execute=MagicMock(return_value=iter(())),
        _hold=MagicMock(return_value=iter(())),
    )
    segment = RouteTheCableEnv._move(
        env,
        "left",
        target,
        "insert",
        tolerance=0.003,
        goal_reached=lambda: torch.tensor([object_goal]),
    )
    assert bool(segment.validator().item()) is expected
    if expected:
        warning.assert_not_called()
    else:
        message = warning.call_args.args[0]
        assert "'insert' rejected" in message
        assert "planning_success=" in message
        assert "position_error_m=" in message
        assert "max_grasp_slip_m=" in message
    if not planning_success:
        assert list(segment.actions) == []
        env._execute.assert_not_called()
    else:
        list(segment.actions)
        trajectory = env._execute.call_args.args[0]
        assert trajectory.shape[1] >= 3 + extra_samples
        assert trajectory.shape[-1] == 3


@pytest.mark.parametrize("settles", [True, False])
def test_insertion_waits_for_tracking_without_bypassing_timeout(settles):
    from types import MethodType

    actual = torch.eye(4).unsqueeze(0)
    target = actual.clone()
    target[:, 0, 3] = 0.01

    def generate(states, *, options):
        return SimpleNamespace(
            positions=torch.zeros(1, options.sample_count, 3),
            success=torch.tensor([True]),
        )

    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=1,
        step_dt=1 / 60,
        cfg=SimpleNamespace(
            expert_trajectory=SimpleNamespace(joint_command_mode="position")
        ),
        _tool=lambda side: actual,
        _command=torch.zeros(1, 3),
        _max_slip=torch.zeros(1),
        _grasp=None,
        robot=SimpleNamespace(
            get_joint_ids=lambda name: [0, 1, 2], get_qpos=lambda: torch.zeros(1, 3)
        ),
        _motion=SimpleNamespace(generate=generate),
        _execute=lambda *args, **kwargs: iter(()),
    )
    env._hold = MethodType(RouteTheCableEnv._hold, env)
    segment = RouteTheCableEnv._move(
        env,
        "left",
        target,
        "insert",
        tolerance=0.003,
        goal_reached=lambda: torch.tensor([False]),
    )
    frames = 0
    for _ in segment.actions:
        frames += 1
        # The controller reaches tolerance after the initial 0.25 s hold.
        if settles and frames == 18:
            actual[:, 0, 3] = 0.009

    assert frames == (18 if settles else 75)
    assert bool(segment.validator().item()) is settles


@pytest.mark.parametrize("success", [True, False])
def test_final_route_validation_reports_physical_metrics(monkeypatch, success):
    warning = Mock()
    monkeypatch.setattr(
        "embodichain_tasks.manipulation.cable.route_the_cable.logger.log_warning",
        warning,
    )
    metrics = {
        "success": torch.tensor([success]),
        "strict_port": torch.tensor([success]),
        "end_position": torch.tensor([[0.586, 0.077, 0.754]]),
        "max_grasp_slip": torch.tensor([0.01]),
    }
    env = SimpleNamespace(route_metrics=lambda: metrics)

    torch.testing.assert_close(
        RouteTheCableEnv._validate_route(env), metrics["success"]
    )
    if success:
        warning.assert_not_called()
    else:
        message = warning.call_args.args[0]
        assert "strict_port" in message
        assert "end_position" in message
        assert "max_grasp_slip" in message
