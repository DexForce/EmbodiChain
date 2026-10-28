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
"""Native task loading and assembly-specific physical success guards."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation
import pytest
import torch

from embodichain.lab.gym.envs.assembly_env import BlindAssemblyEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, register_manager_modules
from embodichain.utils.utility import load_config


def test_grasp_reset_uses_source_mimic_relations_without_joint_name_assumptions() -> (
    None
):
    from embodichain.lab.gym.envs.assembly_env import _grasp_joint_reset

    robot = SimpleNamespace(
        get_joint_ids=lambda **kwargs: [7],
        mimic_ids=[9, 8],
        mimic_parents=[8, 7],
        mimic_multipliers=[-1.0, 2.0],
        mimic_offsets=[0.1, 0.0],
    )
    indices, positions = _grasp_joint_reset(robot, 0.3, 2, torch.device("cpu"))
    assert indices == [7, 8, 9]
    torch.testing.assert_close(
        positions, torch.tensor([[0.3, 0.6, -0.5]]).expand(2, -1)
    )


_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/assembly"
)


def test_formal_and_nominal_profiles_schedule_every_100_updates() -> None:
    paths = sorted(_CONFIG_ROOT.glob("*/agents/ppo.*.yaml"))
    paths = [path for path in paths if ".dev." not in path.name]
    assert len(paths) == 12
    for path in paths:
        trainer = load_config(str(path))["trainer"]
        interval = 100 * trainer["num_envs"] * trainer["buffer_size"]
        assert trainer["eval_freq"] == interval, path
        assert trainer["save_freq"] == interval, path


@pytest.mark.parametrize("task", ["peg_insertion", "gear_meshing", "nut_threading"])
@pytest.mark.parametrize("profile", ["study", "fullrange"])
def test_nominal_task_uses_production_component_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str, profile: str
) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>')
    import importlib

    module = importlib.import_module("embodichain_robots.flexiv.rizon4s.cfg")
    monkeypatch.setattr(module, "get_data_path", lambda path: str(urdf))
    register_manager_modules(
        [
            "embodichain_tasks.manipulation.assembly.obs",
            "embodichain.lab.gym.envs.managers.assembly",
        ]
    )
    path = _CONFIG_ROOT / task / f"task.{profile}.nominal.yaml"
    cfg = config_to_cfg(load_config(str(path)), source_path=path)
    from embodichain_tasks.manipulation.assembly.obs import (
        AssemblyProprioception,
        assembly_contact_labels,
        assembly_privileged_state,
    )

    assert cfg.observations.proprioception.func is AssemblyProprioception
    assert cfg.observations.privileged.func is assembly_privileged_state
    assert cfg.observations.contact_labels.func is assembly_contact_labels
    assert cfg.sim_cfg.physics_dt == pytest.approx(0.001)
    assert cfg.sim_steps_per_control == 20
    assert cfg.control_parts == ["arm"]
    assert cfg.extensions["gripper_model"] == "parallel_jaw"
    assert len(cfg.robot.control_parts["eef"]) == 2
    assert cfg.actions.arm_delta.params["part_name"] == "arm"
    assert cfg.extensions["require_heterogeneous_assets"] is False
    agent = load_config(
        str(_CONFIG_ROOT / task / "agents" / f"ppo.{profile}.nominal.yaml")
    )
    assert agent["policy"]["obs_groups"]["actor"] == ["proprioception"]
    assert agent["policy"]["cfg"]["contact_classes"] == (
        4 if task == "nut_threading" else 3
    )


@pytest.mark.parametrize(
    "kind,moving,anchor",
    [("peg", "peg", "plate"), ("gear", "gear_b", "shaft"), ("nut", "nut", "bolt")],
)
def test_task_assets_keep_real_mating_constraints(
    kind: str, moving: str, anchor: str
) -> None:
    pytest.importorskip("mjfit")
    from embodichain_tasks.manipulation.assembly.assets import build_task_assembly

    asset = build_task_assembly(kind)
    bodies = {b.name: b for b in asset.bodies}
    assert bodies[moving].joint.kind == "free"
    assert bodies[anchor].joint.kind == "fixed"
    assert len(bodies[anchor].geometries) > 1
    if kind == "gear":
        assert bodies["gear_a"].joint.kind == "hinge"
        # The perforated root and boss must be assembled from independent
        # convex pieces; the upstream solid hub cannot remain in collision.
        assert not any(
            g.name in {"gear_b_col_0", "gear_b_hub"} for g in bodies[moving].geometries
        )
        assert any(g.name.startswith("bore_") for g in bodies[moving].geometries)
        assert any(g.name.startswith("boss_") for g in bodies[moving].geometries)
    elif kind == "peg":
        outline = np.asarray(asset.config["mating_outline"])
        assert outline.shape[1] == 2
        assert outline[:, 0].max() - outline[:, 0].min() > 0.05


def test_official_proprioception_uses_world_linear_and_tcp_angular_velocity() -> None:
    """A rotated base distinguishes the two public-reference velocity frames."""
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env._num_envs = 1
    env.assembly_kind = "peg"
    env._assembly_revision = 0
    env._assembly_observation_convention = "official_v1"
    q = torch.arange(1.0, 8.0).unsqueeze(0)
    env.robot = SimpleNamespace(
        get_qpos=lambda **kwargs: q, get_qvel=lambda **kwargs: q
    )
    env._assembly_q_center = q[0]
    env._assembly_q_half_range = torch.full((7,), 2.0)
    env._assembly_velocity_scale = q[0]
    transform = torch.eye(4).unsqueeze(0)
    transform[0, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    jacobian = torch.cat([torch.eye(6), torch.zeros(6, 1)], dim=1).unsqueeze(0)
    env._assembly_chain = SimpleNamespace(
        jacobian=lambda q, ret_eef_pose: (jacobian, transform)
    )
    env._assembly_robot_base = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    env.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(raw_actions=torch.full((1, 7), 0.7))
    )
    actual = env.proprioceptive_frame()
    expected = torch.tensor(
        [
            [0.0] * 7
            + [1.0] * 7
            + [0.8, 2.1, 3.3, 0.0, 0.0, torch.pi / 2]
            + [-2.0, 1.0, 3.0, 4.0, 5.0, 6.0]
            + [0.7] * 7
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["peg", "gear", "nut"])
def test_partial_reset_recomputes_only_selected_proprioceptive_rows(kind: str) -> None:
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env._num_envs = 2
    env.assembly_kind = kind
    env._assembly_revision = 0
    env._assembly_physics_revision = 0
    env._assembly_observation_convention = "legacy_v1"
    q = torch.zeros((2, 7))
    env.robot = SimpleNamespace(
        get_qpos=lambda **kwargs: q, get_qvel=lambda **kwargs: torch.zeros_like(q)
    )
    calls = []

    def jacobian(value, ret_eef_pose):
        calls.append(len(value))
        return torch.eye(6, 7).repeat(len(value), 1, 1), torch.eye(4).repeat(
            len(value), 1, 1
        )

    env._assembly_chain = SimpleNamespace(jacobian=jacobian)
    env.action_manager = None
    before = env.proprioceptive_frame()
    snapshot = before.clone()
    q[1, 0] = 0.5
    env._assembly_revision += 1
    env._assembly_frame_reset_ids = torch.tensor([1])
    after = env.proprioceptive_frame()
    assert calls == [2, 1]
    torch.testing.assert_close(before, snapshot)
    torch.testing.assert_close(after[0], snapshot[0])
    assert after[1, 0].item() == pytest.approx(0.5)


def test_bilateral_grasp_requires_the_part_on_both_pads() -> None:
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env.robot = SimpleNamespace(uid="robot")
    env._moving_uid = "part"
    data = {
        "user_ids": torch.tensor([[[1, 3], [3, 2]], [[1, 3], [1, 2]]]),
        "distance": torch.zeros((2, 2)),
        "is_valid": torch.ones((2, 2), dtype=torch.bool),
    }
    sensor = SimpleNamespace(
        dropped_contacts=0,
        get_data=lambda: data,
        get_actor_ids=lambda uid, links=None: (
            torch.tensor([[1, 2], [1, 2]])
            if uid == "robot"
            else torch.tensor([[3], [3]])
        ),
    )
    env.get_sensor = lambda name: sensor
    assert env._contact_grasp().tolist() == [True, False]


def test_formal_geometry_gate_precedes_world_creation() -> None:
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env.require_heterogeneous_assets = True
    with pytest.raises(RuntimeError, match="per-arena"):
        env._setup_scene()


@pytest.mark.parametrize("capacities", [None, (256, 64)])
def test_assembly_sensor_capacity_defaults_and_development_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capacities: tuple[int, int] | None
) -> None:
    from embodichain.toolkits.assembly import (
        AssemblyAsset,
        AssemblyBody,
        AssemblyGeometry,
        AssemblyJoint,
    )
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    asset = AssemblyAsset(
        bodies=[
            AssemblyBody(
                name=name,
                mass=1.0,
                inertia=np.eye(3).tolist(),
                joint=AssemblyJoint(kind=kind),
                geometries=[AssemblyGeometry(kind="box", size=[0.02, 0.02, 0.01])],
            )
            for name, kind in (("peg", "free"), ("plate", "fixed"))
        ]
    )
    monkeypatch.setattr(EmbodiedEnv, "_prepare_scene", lambda self, **kwargs: None)
    monkeypatch.setattr(AssemblyAsset, "load", lambda manifest: asset)
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env.assembly_kind = "peg"
    env.assembly_manifest = str(tmp_path / "asset.json")
    env.cfg = SimpleNamespace(
        seed=0,
        extensions={},
        sensor=[],
        robot=SimpleNamespace(_pk_urdf_path="robot.urdf", gripper_model="parallel_jaw"),
    )
    env.robot = SimpleNamespace(uid="robot")
    env.sim = SimpleNamespace(
        is_newton_backend=True, add_rigid_object=Mock(), add_articulation=Mock()
    )
    if capacities is not None:
        env.assembly_contact_capacity, env.grasp_contact_capacity = capacities
    try:
        env._prepare_scene()
        expected = (2048, 128) if capacities is None else capacities
        assert (
            tuple(sensor.max_contacts_per_env for sensor in env.cfg.sensor) == expected
        )
    finally:
        if hasattr(env, "_assembly_directory"):
            env._assembly_directory.cleanup()


@pytest.mark.parametrize(
    "name,capacity",
    [
        ("assembly_contact_capacity", 0),
        ("grasp_contact_capacity", 0),
        ("assembly_contact_capacity", True),
        ("assembly_contact_capacity", 2.5),
    ],
)
def test_invalid_assembly_sensor_capacity_precedes_preparation(
    name: str,
    capacity: object,
) -> None:
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    setattr(env, name, capacity)
    with pytest.raises(ValueError, match=name):
        env._prepare_scene()


def _analytic_mating_state(
    kind: str = "peg", grasped: bool = True, penetration: float = 0.0
) -> tuple[BlindAssemblyEnv, torch.Tensor]:
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env.assembly_kind = kind
    env._num_envs = 1
    env._moving_uid, env._fixture_uid = "peg", "plate"
    part = torch.eye(4).unsqueeze(0)
    part[:, 2, 3] = -0.01
    fixture = torch.eye(4).unsqueeze(0)
    objects = {
        "peg": SimpleNamespace(get_local_pose=lambda **kwargs: part),
        "plate": SimpleNamespace(get_local_pose=lambda **kwargs: fixture),
    }
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), get_rigid_object=lambda uid: objects[uid]
    )
    env._assembly_revision = 0
    env._assembly_state_revision = -1
    env._assembly_state_physics_revision = -1
    env._assembly_physics_revision = 0
    env._assembly_edge_points = torch.tensor(
        [[-0.004, -0.004, 0], [0.004, -0.004, 0], [0.004, 0.004, 0], [-0.004, 0.004, 0]]
    )
    env._fixture_zmax = 0.0
    env._moving_zmin = 0.0
    env.assembly_asset = SimpleNamespace(
        config={
            "clearance": 0.001,
            "plate_thickness": 0.01,
            "mating_outline": [
                [-0.005, -0.005],
                [0.005, -0.005],
                [0.005, 0.005],
                [-0.005, 0.005],
            ],
        }
    )
    env._assembly_angle = torch.zeros(1)
    env._assembly_previous_yaw = torch.zeros(1)
    env._assembly_reset_rows = torch.zeros(1, dtype=torch.bool)
    env._assembly_new_milestones = torch.zeros((1, 4), dtype=torch.bool)
    env._assembly_milestones = torch.zeros((1, 4), dtype=torch.bool)
    env._assembly_success_hold = torch.zeros(1, dtype=torch.long)
    env._elapsed_steps = torch.tensor([20])
    data = {
        "is_valid": torch.ones((1, 1), dtype=torch.bool),
        "distance": torch.tensor([[-penetration]]),
    }
    env.get_sensor = lambda name: SimpleNamespace(
        dropped_contacts=0, get_data=lambda: data
    )
    env._contact_grasp = lambda: torch.tensor([grasped])
    env._assembly_capture_angle = torch.full((1,), float("nan"))
    env._assembly_capture_depth = torch.full((1,), float("nan"))
    if kind == "nut":
        env.assembly_asset.config.update(
            pitch=0.004, major_diameter=0.036, nut_height=0.029
        )
    return env, part


@pytest.mark.parametrize("kind", ["peg", "gear", "nut"])
def test_partial_reset_recomputes_selected_assembly_geometry(
    monkeypatch, kind: str
) -> None:
    from embodichain.lab.gym.envs import assembly_env as module

    env, one_part = _analytic_mating_state(kind)
    if kind == "gear":
        env.assembly_asset.config.update(shaft_radius=0.004, bore_clearance=0.001)
    env._num_envs = 2
    part = one_part.repeat(2, 1, 1)
    fixture = torch.eye(4).repeat(2, 1, 1)
    objects = {
        "peg": SimpleNamespace(get_local_pose=lambda **kwargs: part),
        "plate": SimpleNamespace(get_local_pose=lambda **kwargs: fixture),
    }
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), get_rigid_object=lambda uid: objects[uid]
    )
    for field in (
        "_assembly_angle",
        "_assembly_previous_yaw",
        "_assembly_reset_rows",
        "_assembly_success_hold",
        "_elapsed_steps",
        "_assembly_capture_angle",
        "_assembly_capture_depth",
    ):
        setattr(env, field, getattr(env, field).repeat(2))
    env._assembly_new_milestones = env._assembly_new_milestones.repeat(2, 1)
    env._assembly_milestones = env._assembly_milestones.repeat(2, 1)
    data = {
        "is_valid": torch.ones((2, 1), dtype=torch.bool),
        "distance": torch.zeros((2, 1)),
    }
    env.get_sensor = lambda name: SimpleNamespace(
        dropped_contacts=0, get_data=lambda: data
    )
    env._contact_grasp = lambda: torch.ones(2, dtype=torch.bool)
    observed = []
    original = module._signed_polygon_distance

    def measured(samples, polygon):
        observed.append(samples.shape[0])
        return original(samples, polygon)

    monkeypatch.setattr(module, "_signed_polygon_distance", measured)
    env.compute_task_state()
    before = {key: value[0].clone() for key, value in env._assembly_metrics.items()}
    previous_metrics = dict(env._assembly_metrics)
    metric_snapshots = {key: value.clone() for key, value in previous_metrics.items()}
    previous_labels = env._assembly_labels
    label_snapshot = previous_labels.clone()
    labels_before = env._assembly_labels[0].clone()
    hold_before = env._assembly_success_hold[0].clone()
    part[1, 0, 3] = 0.004
    env._assembly_reset_rows = torch.tensor([False, True])
    env._assembly_pending_reset_ids = torch.tensor([1])
    env._assembly_revision += 1
    env.compute_task_state()
    assert observed == ([2, 1] if kind == "peg" else [])
    for key, value in previous_metrics.items():
        torch.testing.assert_close(value, metric_snapshots[key], equal_nan=True)
    torch.testing.assert_close(previous_labels, label_snapshot)
    for key, value in before.items():
        torch.testing.assert_close(env._assembly_metrics[key][0], value)
    torch.testing.assert_close(env._assembly_labels[0], labels_before)
    torch.testing.assert_close(env._assembly_success_hold[0], hold_before)
    assert env._assembly_metrics["radial_error"][1].item() == pytest.approx(0.004)


@pytest.mark.parametrize(
    "grasped,penetration,expected",
    [(True, 0.0, True), (False, 0.0, False), (True, 0.001, False)],
)
def test_seated_peg_requires_grasp_and_penetration_guard(
    grasped: bool, penetration: float, expected: bool
) -> None:
    env, _ = _analytic_mating_state(grasped=grasped, penetration=penetration)
    for revision in range(3):
        env._assembly_revision = revision
        env._assembly_physics_revision = revision
        success, _, metrics = env.compute_task_state()
    assert success.item() is expected
    assert metrics["depth"].item() == pytest.approx(0.01)
    assert env._assembly_labels.tolist() == [[2, 2, 2, 2]]
    previous = env._assembly_success_hold.clone()
    env._assembly_revision += 1
    env.compute_task_state()
    assert torch.equal(env._assembly_success_hold, previous)


def test_peg_contact_labels_follow_signed_rim_and_surface_bands() -> None:
    env, part = _analytic_mating_state()
    part[:, 2, 3] = 0
    # A square aperture has half-width 5 mm. The public task uses a 1 mm
    # Euclidean rim band and a separate 1 mm band about the fixture top.
    env._assembly_edge_points = torch.tensor(
        [
            [0.0, 0.0, 0.0011],  # Above the plane band: free.
            [0.0, 0.0, 0.0005],  # Deep inside the aperture: inside.
            [0.0045, 0.0, 0.0005],  # Inner side of rim near plane: contact.
            [0.0055, 0.0, -0.0005],  # Outer side of rim near plane: contact.
            [0.0045, 0.0, -0.002],  # Rim band below plane: inside.
            [0.0055, 0.0, -0.002],  # Same label on outer half of rim band.
            [0.007, 0.0, -0.002],  # Far outside below plane: free.
            [0.007, 0.0, 0.0005],  # On the plate top: contact.
            # Corner is 0.8 * sqrt(2) mm from the rim, outside its 1 mm
            # Euclidean band even though each infinite edge is closer.
            [0.0058, 0.0058, -0.002],
        ]
    )
    env.compute_task_state()
    assert env._assembly_labels.tolist() == [[0, 2, 1, 1, 2, 2, 0, 1, 0]]


@pytest.mark.parametrize(
    "plane_error,expected",
    [(-0.0011, False), (-0.0005, True), (0.0005, True), (0.0011, False)],
)
def test_peg_seating_has_a_bounded_bottom_plane_interval(
    plane_error: float, expected: bool
) -> None:
    env, part = _analytic_mating_state()
    # Fixture top is z=0 and its thickness is 10 mm, so the seated bottom
    # plane is z=-10 mm. Falling farther through the hole is not seating.
    part[:, 2, 3] = -0.01 + plane_error
    for revision in range(3):
        env._assembly_revision = env._assembly_physics_revision = revision
        success, _, _ = env.compute_task_state()
    assert success.item() is expected


def test_peg_seating_checks_each_bottom_vertex_height() -> None:
    env, part = _analytic_mating_state()
    angle = 0.3
    part[0, :3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    part[:, 2, 3] = -0.0103
    for revision in range(3):
        env._assembly_revision = env._assembly_physics_revision = revision
        success, _, metrics = env.compute_task_state()
    # Even the highest corner passes the old one-sided 9 mm depth threshold;
    # the opposite corner is over 1 mm below the target and must reject it.
    assert metrics["captured"].item()
    assert metrics["depth"].item() > 0.009
    assert not success.item()


def test_nut_rotation_at_fixed_capture_depth_is_not_threading() -> None:
    env, part = _analytic_mating_state(kind="nut")
    part[:, 2, 3] = -0.002
    env.compute_task_state()
    assert env._assembly_metrics["captured"].item()
    for revision in range(1, 5):
        angle = min(revision * torch.pi / 4, torch.pi / 2)
        part[0, :2, :2] = torch.tensor(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        env._assembly_revision = env._assembly_physics_revision = revision
        success, _, _ = env.compute_task_state()
        assert success.item() is False
    # A genuine quarter turn advances one quarter of the 4 mm pitch.
    part[:, 2, 3] = -0.003
    for revision in range(5, 8):
        env._assembly_revision = env._assembly_physics_revision = revision
        success, _, _ = env.compute_task_state()
    assert success.item() is True


@pytest.mark.parametrize("base_pitch", [0.0, 0.4])
def test_encoder_gravity_feedforward_matches_mujoco_bias(
    tmp_path: Path, base_pitch: float
) -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("pytorch_kinematics")
    import xml.etree.ElementTree as ET
    from embodichain_benchmarks.assembly.robot_reference import _append_model

    model = ET.Element("mujoco")
    world = ET.SubElement(model, "worldbody")

    rotation = Rotation.from_euler("y", base_pitch)
    quaternion = np.roll(rotation.as_quat(), 1)
    body = ET.SubElement(
        world, "body", name="base", quat=" ".join(map(str, quaternion))
    )
    ET.SubElement(body, "geom", type="box", size=".02 .02 .02")
    for i in range(1, 8):
        body = ET.SubElement(body, "body", name=f"link{i}", pos="0 0 .1")
        ET.SubElement(body, "joint", name=f"joint{i}", axis="0 1 0", range="-170 170")
        ET.SubElement(
            body, "geom", type="box", size=".01 .01 .05", pos="0 0 .05", mass="1"
        )
    reference = mujoco.MjModel.from_xml_string(ET.tostring(model, encoding="unicode"))
    urdf = ET.Element("robot", name="gravity_reference")
    _append_model(urdf, reference, tmp_path)
    path = tmp_path / "gravity.urdf"
    path.write_text(ET.tostring(urdf, encoding="unicode"))
    q = torch.tensor([[0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3]])
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env._num_envs = 1
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.sim_cfg = SimpleNamespace(physics_cfg=SimpleNamespace(gravity=[0, 0, -9.81]))
    root_pose = np.eye(4)
    root_pose[:3, :3] = rotation.as_matrix()
    env.cfg = SimpleNamespace(
        robot=SimpleNamespace(
            _pk_urdf_path=str(path), init_rot=[0, 0, 0], init_local_pose=root_pose
        )
    )
    env.robot = SimpleNamespace(
        joint_names=[f"joint{i}" for i in range(1, 8)], get_qpos=lambda: q
    )
    env.assembly_fidelity = {}
    env._prepare_gravity_compensation()
    actual = env._gravity_torques().numpy()[0]
    data = mujoco.MjData(reference)
    data.qpos[:] = q.numpy()[0]
    mujoco.mj_forward(reference, data)
    np.testing.assert_allclose(actual, data.qfrc_bias, atol=2e-5, rtol=2e-5)


def test_manager_history_selective_reset_does_not_repeat_other_rows() -> None:
    from embodichain_tasks.manipulation.assembly.obs import AssemblyProprioception
    from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

    frame = torch.zeros((2, 33))
    frame[:, 0] = torch.tensor([1.0, 2.0])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        assembly_kind="peg",
        _assembly_physics_revision=0,
        _assembly_reset_rows=torch.ones(2, dtype=torch.bool),
        refresh_assembly_state=lambda: None,
        proprioceptive_frame=lambda: frame.clone(),
    )
    history = AssemblyProprioception(FunctorCfg(params={"history_length": 30}), env)
    history(env, {})
    env._assembly_physics_revision = 1
    frame[:, 0] = torch.tensor([3.0, 4.0])
    before = history(env, {})
    frame[0, 0] = 5.0
    history.reset(env_ids=torch.tensor([0]))
    after = history(env, {})
    assert torch.equal(before[1], after[1])
    assert after.reshape(2, 31, 33)[0, :, 0].tolist() == [5.0] * 31
    assert before.reshape(2, 31, 33)[0, -1, 0].item() == 3.0


def test_legacy_assembly_observation_imports_remain_available() -> None:
    from embodichain.lab.gym.envs.managers import assembly as legacy
    from embodichain_tasks.manipulation.assembly import obs

    for name in (
        "AssemblyProprioception",
        "assembly_privileged_state",
        "assembly_contact_labels",
    ):
        assert getattr(legacy, name) is getattr(obs, name)


def test_checkpoint_task_resources_relocate_without_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.toolkits.assembly import AssemblyAsset
    import shutil
    import json
    import sys

    source = tmp_path / "source" / "prepared_v4_parallel_jaw"
    source.mkdir(parents=True)
    urdf = source / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>')
    (source / "fidelity.json").write_text(
        json.dumps({"exact_source_dynamics": False, "converter_version": 4})
    )
    for model in ("flexiv_rizon4s", "robotiq_2f85"):
        (source.parent / model).mkdir()
        (source.parent / model / "LICENSE").write_text("fixture license " + model)
    (source.parent / "source_manifest.json").write_text('{"revision":"fixture"}')
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env.cfg = SimpleNamespace(robot=SimpleNamespace(_pk_urdf_path=str(urdf)))
    env.assembly_asset = AssemblyAsset(config={"kind": "peg"})
    env.assembly_fidelity = {"source_gripper_dynamics_preserved": False}
    directory = tmp_path / "run" / "configs"
    frozen = env.export_checkpoint_gym_config(
        {
            "id": "BlindPegInsertionRL-v1",
            "physics": "newton",
            "robot": {
                "class_type": "FlexivRizon4S",
            },
            "env": {"extensions": {}},
        },
        directory,
    )
    target = tmp_path / "relocated" / "configs"
    shutil.copytree(directory, target)
    shutil.rmtree(source.parent)
    monkeypatch.setitem(sys.modules, "mjfit", None)
    path = target / "gym.yaml"
    import yaml

    path.write_text(yaml.safe_dump(frozen))
    cfg = config_to_cfg(load_config(str(path)), source_path=path)
    assert cfg.robot.fpath == str(target / "resources" / "robot" / "robot.urdf")
    asset = AssemblyAsset.load(cfg.extensions["assembly_manifest"])
    assert asset.content_hash == env.assembly_asset.content_hash
    assert (
        target / "resources/robot/licenses/robotiq_2f85.LICENSE"
    ).read_text() == "fixture license robotiq_2f85"
    assert (
        json.loads((target / "resources/robot/fidelity.json").read_text())[
            "converter_version"
        ]
        == 4
    )


def test_default_asset_composition_keeps_materials_joint_names_and_arm_tcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    import xml.etree.ElementTree as ET
    from embodichain_robots import FlexivRizon4SCfg
    from embodichain.lab.gym.envs.assembly_env import _assembly_kinematic_chain
    from embodichain.robot_registry import list_robot_cfgs

    def base_link(name: str, color: str, mass: str) -> str:
        return f"""<link name="{name}"><inertial><mass value="{mass}"/>
          <inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>
          </inertial><visual><geometry><box size=".02 .02 .02"/></geometry>
          <material name=""><color rgba="{color}"/></material></visual>
          <collision><geometry><box size=".01 .01 .01"/></geometry></collision></link>"""

    arm = tmp_path / "arm.urdf"
    pieces = ['<robot name="arm">', base_link("base_link", ".97 .97 .97 1", "3")]
    for i in range(1, 8):
        parent = "base_link" if i == 1 else f"link{i-1}"
        axis = "0 1 0" if i % 2 else "1 0 0"
        pieces.append(
            f"""<link name="link{i}"/><joint name="joint{i}" type="revolute">
          <parent link="{parent}"/><child link="link{i}"/><origin xyz="0 0 .1"/>
          <axis xyz="{axis}"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>"""
        )
    pieces.append(
        '<link name="flange"/><joint name="flange_fixed" type="fixed"><parent link="link7"/><child link="flange"/></joint></robot>'
    )
    arm.write_text("".join(pieces))
    hand = tmp_path / "hand.urdf"
    pieces = [
        '<robot name="hand">',
        base_link("robotiq_85_base_link", ".1 .1 .1 1", ".4"),
    ]
    joints = [
        ("finger_joint", "left_outer_knuckle", None),
        ("left_inner_knuckle_joint", "left_inner_knuckle", 1),
        ("left_inner_finger_joint", "left_inner_finger", -1),
        ("right_inner_knuckle_joint", "right_inner_knuckle", -1),
        ("right_inner_finger_joint", "right_inner_finger", 1),
        ("right_outer_knuckle_joint", "right_outer_knuckle", -1),
    ]
    for name, child, multiplier in joints:
        mimic = (
            ""
            if multiplier is None
            else f'<mimic joint="finger_joint" multiplier="{multiplier}" offset="0"/>'
        )
        pieces.append(
            f'<link name="{child}"/><joint name="{name}" type="revolute"><parent link="robotiq_85_base_link"/><child link="{child}"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="2"/>{mimic}</joint>'
        )
    hand.write_text("".join(pieces) + "</robot>")
    assets = {
        "FlexivRizon4S/rizon4s.urdf": str(arm),
        "Robotiq2F85/Robotiq2F85.urdf": str(hand),
    }
    from embodichain.lab.sim import cfg as cfg_module

    monkeypatch.setattr(
        cfg_module, "get_data_path", lambda path: assets.get(path, path)
    )
    arm_module = importlib.import_module("embodichain_robots.flexiv.rizon4s.cfg")
    monkeypatch.setattr(
        arm_module, "get_data_path", lambda path: assets.get(path, path)
    )
    register_manager_modules(
        [
            "embodichain_tasks.manipulation.assembly.obs",
            "embodichain.lab.gym.envs.managers.assembly",
        ]
    )
    path = _CONFIG_ROOT / "peg_insertion/task.study.default_assets.yaml"
    config = load_config(str(path))
    import yaml

    component = (
        _CONFIG_ROOT.parents[2]
        / "components/embodiments/flexiv_rizon4s_robotiq2f85.yaml"
    )
    declaration = yaml.safe_load(component.read_text())
    declaration["simulation"]["urdf_cfg"]["fpath_prefix"] = str(tmp_path / "assembled")
    local_component = tmp_path / "embodiment.yaml"
    local_component.write_text(yaml.safe_dump(declaration))
    config["embodiment"] = {"component": str(local_component)}
    cfg = config_to_cfg(config, source_path=path).robot
    assert Path(cfg.fpath).is_relative_to(tmp_path)
    root = ET.parse(cfg._pk_urdf_path).getroot()
    assert (
        root.find("link[@name='base_link']/visual/material/color").get("rgba")
        == ".97 .97 .97 1"
    )
    assert (
        root.find("link[@name='robotiq_85_base_link']/visual/material/color").get(
            "rgba"
        )
        == ".1 .1 .1 1"
    )
    assert len(root.findall("./link/collision")) == 2
    assert len(root.findall("./joint/mimic")) == 5
    assert cfg.control_parts["eef"] == ["finger_joint"]
    chain = _assembly_kinematic_chain(cfg, torch.device("cpu"))
    assert chain.get_joint_parameter_names() == [f"joint{i}" for i in range(1, 8)]
    pose = chain.forward_kinematics(torch.zeros(1, 7)).get_matrix()[0]
    torch.testing.assert_close(pose[:3, 3], torch.tensor([0.0, 0.0, 0.845]))
    shoulder = torch.zeros(1, 7)
    shoulder[0, 0] = torch.pi / 2
    moved = chain.forward_kinematics(shoulder).get_matrix()[0, :3, 3]
    torch.testing.assert_close(
        moved, torch.tensor([0.745, 0.0, 0.1]), atol=1e-6, rtol=1e-6
    )
    restored = FlexivRizon4SCfg.from_dict(cfg.to_dict())
    assert restored.control_parts == cfg.control_parts
    assert isinstance(cfg, FlexivRizon4SCfg)
    assert cfg.solver_cfg["arm"].class_type == "PytorchSolver"
    assert not hasattr(cfg, "gripper_model")
    assert "FlexivRizon4SRobotiq" not in list_robot_cfgs()
    assert root.find("link[@name='assembled_base']") is not None
    assert cfg.fpath == cfg.solver_cfg["arm"].urdf_path
    assert tuple(cfg.root_props.newton_mimic_solref) == (0.002, 1.0)
    env_cfg = config_to_cfg(config, source_path=path)
    assert env_cfg.extensions["grasp_link_names"] == [
        "right_inner_finger",
        "left_inner_finger",
    ]
    assert env_cfg.sim_steps_per_control == 20
    q = torch.tensor([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5]])
    jacobian, _ = chain.jacobian(q, ret_eef_pose=True)
    eps = 1e-3
    for joint in range(7):
        perturb = torch.zeros_like(q)
        perturb[0, joint] = eps
        numerical = (
            chain.forward_kinematics(q + perturb).get_matrix()[0, :3, 3]
            - chain.forward_kinematics(q - perturb).get_matrix()[0, :3, 3]
        ) / (2 * eps)
        torch.testing.assert_close(
            jacobian[0, :3, joint], numerical, atol=1e-4, rtol=1e-3
        )


def test_exact_robot_dynamics_gate_belongs_to_task_environment(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>')
    env = BlindAssemblyEnv.__new__(BlindAssemblyEnv)
    env.require_exact_robot_dynamics = True
    env.cfg = SimpleNamespace(robot=SimpleNamespace(_pk_urdf_path=str(urdf)))
    with pytest.raises(ValueError, match="closed loops"):
        env._setup_scene()


def test_peg_task_floor_is_plate_attached_and_preserves_free_part() -> None:
    pytest.importorskip("mjfit")
    from embodichain_tasks.manipulation.assembly.assets import build_task_assembly

    closed = build_task_assembly("peg")
    opened = build_task_assembly("peg", {"bottom_thickness": 0})
    bodies = {body.name: body for body in closed.bodies}
    original = {body.name: body for body in opened.bodies}
    assert set(bodies) == set(original) == {"plate", "peg"}
    plate, peg = bodies["plate"], bodies["peg"]
    assert plate.joint.kind == "fixed" and peg.joint.kind == "free"
    assert len(plate.geometries) == len(original["plate"].geometries) + 1
    floor = next(geometry for geometry in plate.geometries if geometry.name == "bottom")
    assert floor.collision and floor.visible
    vertices = (
        np.asarray(floor.vertices) @ Rotation.from_quat(floor.quaternion).as_matrix().T
        + floor.position
    )
    # The nominal plate is 18 mm deep. Its local underside is -9 mm, and
    # the public generator's 2 mm floor must terminate exactly there.
    np.testing.assert_allclose(vertices[:, 2].max(), -0.009, atol=2e-8)
    np.testing.assert_allclose(np.ptp(vertices[:, 2]), 0.002, atol=2e-8)
    np.testing.assert_allclose(vertices[:, :2].min(axis=0), [-0.07, -0.07], atol=2e-8)
    np.testing.assert_allclose(vertices[:, :2].max(axis=0), [0.07, 0.07], atol=2e-8)
    opening = np.asarray(closed.config["mating_outline"])
    assert np.all(opening >= vertices[:, :2].min(axis=0))
    assert np.all(opening <= vertices[:, :2].max(axis=0))
    assert peg.mass == original["peg"].mass == 0.5
    np.testing.assert_array_equal(peg.inertia, original["peg"].inertia)
    np.testing.assert_array_equal(peg.com, original["peg"].com)
    assert len(peg.geometries) == len(original["peg"].geometries) == 1
    np.testing.assert_array_equal(
        peg.geometries[0].vertices, original["peg"].geometries[0].vertices
    )
    assert not any(
        geometry.name == "bottom" for geometry in original["plate"].geometries
    )


def test_elevated_peg_task_seats_on_local_floor_not_global_ground() -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mjfit")
    import xml.etree.ElementTree as ET
    from embodichain_tasks.manipulation.assembly.assets import build_task_assembly

    asset = build_task_assembly("peg")
    model_xml = ET.fromstring(asset.source_mjcf)
    # Move the whole assembly 25 cm above the unrelated global ground.
    # This is scene preparation, before constructing or stepping the model.
    for name in ("plate", "peg"):
        body = model_xml.find(f"./worldbody/body[@name='{name}']")
        position = np.fromstring(body.attrib["pos"], sep=" ")
        position[2] += 0.25
        body.set("pos", " ".join(map(str, position)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    peg_id, plate_id = model.body("peg").id, model.body("plate").id
    expected_floor_height = model.body_pos[plate_id, 2] - 0.009
    peg = next(body for body in asset.bodies if body.name == "peg")
    geometry = peg.geometries[0]
    vertices = (
        np.asarray(geometry.vertices)
        @ Rotation.from_quat(geometry.quaternion).as_matrix().T
        + geometry.position
    )
    bottom = vertices[np.isclose(vertices[:, 2], vertices[:, 2].min(), atol=1e-8)]
    world_bottom = bottom @ data.xmat[peg_id].reshape(3, 3).T + data.xpos[peg_id]
    assert np.isfinite(data.qpos).all()
    np.testing.assert_allclose(world_bottom[:, 2], expected_floor_height, atol=5e-5)
    floor_id = model.geom("bottom").id
    assert any(floor_id in contact.geom for contact in data.contact[: data.ncon])
