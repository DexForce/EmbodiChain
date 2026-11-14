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
from embodichain.compute.geometry._cable_route import _measure_cable_route


from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CylinderCfg

pytestmark = pytest.mark.no_sim


def _route_geometry():
    path = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task_program/integration.yaml"
    )
    return yaml.safe_load(path.read_text())["runtime_services"]["cable_material"][
        "geometry"
    ]


def measure_route(state, lengths, hands, origins):
    return _measure_cable_route(
        state, lengths, hands, origins, _route_geometry(), 0.0035
    )


@pytest.fixture(autouse=True)
def local_yam_config_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep configuration tests independent of external model publication."""
    from embodichain_robots.i2rt.yam import cfg as yam_i2rt

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
    assert cfg.robot.root_props.newton_gravity_compensation == 1.0
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


def test_release_requires_seating_depth_even_when_tip_is_inside_strict_channel():
    geometry = _route_geometry()
    geometry["release_region"] = [[0.585, 0.072, 0.753], [0.587, 0.083, 0.772]]
    state = _goal_state().repeat(2, 1, 1)
    # The last capsule points along +Y and has a 6 mm half length. Both tips
    # are in the final channel. The shallow tip must not stop corrections;
    # the deeper tip is seated near a sidewall, outside the hover corridor.
    state[:, -1, :3] = torch.tensor([[0.586, 0.060, 0.755], [0.5846, 0.067, 0.757]])
    metrics = _measure_cable_route(
        state,
        torch.full((50,), 0.012),
        torch.ones(2, 2, 3),
        torch.zeros(2, 3),
        geometry,
        0.0035,
    )
    assert metrics["strict_port"].tolist() == [True, True]
    assert metrics["release_ready"].tolist() == [False, True]


def test_above_channel_release_region_is_independent_of_final_strict_region():
    geometry = _route_geometry()
    # An open-top placement may release from above the channel. This is an
    # action precondition; the final endpoint must still be at or below .759 m.
    geometry["release_region"] = [[0.585, 0.072, 0.753], [0.587, 0.083, 0.772]]
    state = _goal_state().repeat(3, 1, 1)
    state[:, -1, :3] = torch.tensor(
        [[0.586, 0.071, 0.766], [0.586, 0.071, 0.773], [0.5846, 0.071, 0.766]]
    )
    metrics = _measure_cable_route(
        state,
        torch.full((50,), 0.012),
        torch.ones(3, 2, 3),
        torch.zeros(3, 3),
        geometry,
        0.0035,
    )
    assert metrics["release_ready"].tolist() == [True, False, False]
    assert metrics["strict_port"].tolist() == [False, False, False]


@pytest.mark.parametrize("invalid", ["nonfinite", "zero_quaternion"])
def test_strict_seating_does_not_allow_release_with_invalid_cable_state(invalid):
    geometry = _route_geometry()
    geometry["release_region"] = [[0.585, 0.072, 0.753], [0.587, 0.083, 0.772]]
    state = _goal_state()
    state[0, -1, :3] = torch.tensor([0.5846, 0.067, 0.757])
    if invalid == "nonfinite":
        state[0, 0, 0] = float("nan")
    else:
        state[0, 0, 3:7] = 0
    metrics = _measure_cable_route(
        state,
        torch.full((50,), 0.012),
        torch.ones(1, 2, 3),
        torch.zeros(1, 3),
        geometry,
        0.0035,
    )
    assert metrics["strict_port"].item()
    assert not metrics["release_ready"].item()


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
