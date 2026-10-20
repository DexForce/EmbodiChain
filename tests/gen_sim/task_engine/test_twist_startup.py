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

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import twist_runtime
from embodichain.gen_sim.task_engine._task_program.press_runtime import (
    PressContactSensor,
)
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    TwistAcceptancePort,
    TwistContactSensor,
    _TwistCandidateState,
)
from embodichain.gen_sim.task_engine._task_program.twist_geometry import (
    LinkMesh,
    TwistGeometry,
)


class _StartupSensor(TwistContactSensor):
    @property
    def dropped_contacts(self) -> int:
        return self.dropped

    def get_data(self) -> dict:
        return self.data

    def update(self, **kwargs) -> None:
        pass

    def get_actor_ids(self, uid, link_names=None) -> torch.Tensor:
        if uid == "table":
            return torch.tensor([[99]])
        if uid == "control":
            return torch.tensor(
                [[10 if name == "panel" else 11 for name in link_names]]
            )
        return torch.tensor([[self.robot_ids[name] for name in link_names]])


@pytest.fixture
def sensor(monkeypatch):
    value = object.__new__(_StartupSensor)
    value.qpos, value.qvel, value.dropped = 0.0, 0.0, 0
    value.data = _contacts()
    value.robot_ids = {"left_finger": 20, "left_other_finger": 21, "left_arm": 22}
    physical = SimpleNamespace(contact_offset=0.006, rest_offset=0.001)
    value.art_object = SimpleNamespace(
        joint_names=["rotation"],
        uid="control",
        cfg=SimpleNamespace(fpath="startup-unit.usda"),
        get_qpos=lambda: torch.tensor([[value.qpos]], dtype=torch.float64),
        get_qvel=lambda: torch.tensor([[value.qvel]], dtype=torch.float64),
        get_link_physical_attr=lambda name: [physical],
    )
    value._sim = SimpleNamespace(
        sim_config=SimpleNamespace(physics_dt=0.01),
        get_articulation=lambda uid: value.art_object,
    )
    route = SimpleNamespace(
        arm="left",
        binding=SimpleNamespace(
            object_id="control", joint="rotation", parent="panel", link="knob"
        ),
        preset=lambda phase: phase,
    )
    robot = SimpleNamespace(
        uid="robot",
        cfg=SimpleNamespace(fpath="startup-unit.urdf", body_scale=(1.0, 1.0, 1.0)),
        link_names=list(value.robot_ids),
        get_link_physical_attr=lambda name: [physical],
        get_qpos=lambda *args, **kwargs: torch.zeros(1, 2),
    )
    vertices = np.frombuffer(
        np.asarray(
            [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=np.float64
        ).tobytes(),
        dtype=np.float64,
    ).reshape(-1, 3)
    faces = np.frombuffer(
        np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
        ).tobytes(),
        dtype=np.int64,
    ).reshape(-1, 3)
    grip = LinkMesh("/Control/knob/grip", "/Control/knob", vertices, faces, "none")
    panel = LinkMesh("/Control/panel/mesh", "/Control/panel", vertices, faces, "none")
    geometry = TwistGeometry(
        grip,
        (grip,),
        (panel,),
        "a" * 64,
        1.0,
        "/Control/knob",
        "/Control/panel",
        (grip, panel),
    )
    # Startup tests keep source parsing separate from qpos/contact acceptance.
    monkeypatch.setattr(
        twist_runtime, "load_twist_geometry", lambda path, binding: geometry
    )
    monkeypatch.setattr(
        twist_runtime,
        "load_robot_link_meshes",
        lambda path, names, scale: {
            name: (LinkMesh(name + "/mesh", name, vertices, faces, "urdfMeshInput"),)
            for name in names
        },
    )
    monkeypatch.setattr(
        PressContactSensor,
        "update_physics_step",
        lambda self, dt: setattr(self, "clock", self.clock + dt),
    )
    monkeypatch.setattr(
        PressContactSensor,
        "reset",
        lambda self, env_ids=None: setattr(self, "armed", False),
    )
    value.configure(route, robot)
    return value


def _contacts(pairs=(), impulses=None) -> dict:
    pairs = list(pairs)
    count = len(pairs)
    if impulses is None:
        impulses = [1.0] * count
    return {
        "is_valid": torch.ones(1, count, dtype=torch.bool),
        "user_ids": torch.tensor(pairs, dtype=torch.int64).reshape(1, count, 2),
        "impulse": torch.tensor(impulses).reshape(1, count),
        "position": torch.zeros(1, count, 3),
        "normal": torch.zeros(1, count, 3),
        "distance": torch.zeros(1, count),
    }


def _finish(sensor) -> dict:
    sensor.arm()
    return sensor.startup_check()


def test_stable_full_startup_observed_until_arm(sensor):
    for _ in range(1500):
        sensor.update_physics_step(0.01)
    result = _finish(sensor)
    assert result["accepted"] is True
    assert result["sample_count"] == 1501
    evidence = sensor.startup_evidence()
    assert evidence["complete"] is True
    assert evidence["table_actor_id"] == 99
    assert evidence["robot_actor_ids"] == [20, 21, 22]
    assert evidence["trace"][0]["timestamp"] == 0.0
    assert evidence["trace"][-1]["timestamp"] == pytest.approx(15.0)
    sensor.update_physics_step(0.01)
    assert len(sensor.startup_evidence()["trace"]) == 1501


@pytest.mark.parametrize("attribute,value", [("qpos", 0.04), ("qvel", 0.04)])
def test_transient_startup_motion_cannot_be_hidden_by_final_settling(
    sensor, attribute, value
):
    setattr(sensor, attribute, value)
    sensor.update_physics_step(0.01)
    sensor.qpos = sensor.qvel = 0.0
    sensor.update_physics_step(0.01)
    result = _finish(sensor)
    assert result["accepted"] is False
    assert result["reason"] == "startup_motion"


@pytest.mark.parametrize(
    "pair,reason",
    [((99, 11), "startup_table_contact"), ((11, 22), "startup_robot_contact")],
)
def test_startup_target_contact_rejected_even_after_contact_disappears(
    sensor, pair, reason
):
    sensor.data = _contacts([pair])
    sensor.update_physics_step(0.01)
    sensor.data = _contacts()
    sensor.update_physics_step(0.01)
    result = _finish(sensor)
    assert result["accepted"] is False
    assert result["reason"] == reason


def test_startup_parent_contact_is_not_knob_contact(sensor):
    sensor.data = _contacts([(99, 10), (10, 22), (99, 11)], [1.0, 1.0, 0.0])
    sensor.update_physics_step(0.01)
    assert _finish(sensor)["accepted"] is True


@pytest.mark.parametrize("corruption", ["dropped", "qpos", "qvel", "impulse"])
def test_invalid_startup_observation_fails_closed(sensor, corruption):
    if corruption == "dropped":
        sensor.dropped = 1
    elif corruption == "impulse":
        sensor.data = _contacts([(99, 11)], [float("nan")])
    else:
        setattr(sensor, corruption, float("nan"))
    sensor.update_physics_step(0.01)
    result = _finish(sensor)
    assert result["accepted"] is False
    assert result["reason"] == "invalid_startup_evidence"


def test_bounded_startup_overflow_fails_closed(sensor):
    for _ in range(3300):
        sensor.update_physics_step(0.01)
    result = _finish(sensor)
    evidence = sensor.startup_evidence()
    assert result["accepted"] is False
    assert result["reason"] == "invalid_startup_evidence"
    assert evidence["overflow"] is True
    assert len(evidence["trace"]) <= evidence["sample_limit"]


def test_reset_starts_new_epoch_but_preserves_last_completed_evidence(sensor):
    sensor.qvel = 0.1
    sensor.update_physics_step(0.01)
    assert _finish(sensor)["accepted"] is False
    original = sensor.startup_evidence()
    sensor.qvel = 0.0
    sensor.reset()
    assert sensor.startup_evidence() == original
    for _ in range(5):
        sensor.update_physics_step(0.01)
    assert _finish(sensor)["accepted"] is True
    fresh = sensor.startup_evidence()
    assert fresh["epoch"] == original["epoch"] + 1
    assert len(fresh["trace"]) == 6
    assert fresh["trace"][0]["timestamp"] == 0.0


def test_early_failure_reset_retains_incomplete_startup_evidence(sensor):
    sensor.qvel = 0.1
    sensor.update_physics_step(0.01)
    sensor.reset()
    evidence = sensor.startup_evidence()
    assert evidence["complete"] is False
    assert evidence["trace"][-1]["qvel"] == 0.1
    assert evidence["summary"]["accepted"] is False


def test_startup_sample_count_has_hard_memory_bound(sensor):
    sensor._sim.sim_config.physics_dt = 1e-6
    sensor.configure(sensor.route, sensor.robot)
    assert sensor._startup_sample_limit == 8192


def test_ready_gate_checks_full_startup_not_only_last_half_second(sensor):
    sensor.data = _contacts([(99, 11)])
    sensor.update_physics_step(0.01)
    sensor.data = _contacts()
    sensor.update_physics_step(0.01)
    port = object.__new__(TwistAcceptancePort)
    port.route, port.sensor, port.dt = sensor.route, sensor, 0.04
    port.robot = sensor.robot
    port._candidate_state = _TwistCandidateState()
    port.results, port.metadata = {}, {}
    policy = SimpleNamespace(
        cfg=SimpleNamespace(preset="ready", kind="wait_stable"),
        entity=SimpleNamespace(entity_id="control"),
    )
    list(
        port.actions(
            policy, segment=SimpleNamespace(calls=[]), active_mask=torch.tensor([True])
        )
    )
    assert sensor.acceptance["accepted"] is False
    assert sensor.acceptance["phase"] == "initial_stable"
    assert sensor.acceptance["reason"] == "startup_table_contact"
    assert sensor.acceptance["startup_check"]["table_contact_samples"] == 1
    assert max(map(abs, sensor.acceptance["joint_positions"])) == 0.0
    assert max(map(abs, sensor.acceptance["joint_velocities"])) == 0.0


def test_original_startup_angle_and_velocity_thresholds_retained(sensor):
    sensor.qpos = math.radians(2.0)
    sensor.qvel = math.radians(2.0)
    sensor.update_physics_step(0.01)
    assert _finish(sensor)["accepted"] is True
