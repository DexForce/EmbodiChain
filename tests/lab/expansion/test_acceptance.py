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

"""Measured acceptance boundaries without native simulation resources."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.expansion.acceptance import _ContactRow

pytestmark = pytest.mark.no_sim


@pytest.fixture
def contact_row(tmp_path):
    def build(duration=0.25, ground=False, frequency=10):
        asset = tmp_path / "robot.urdf"
        asset.write_text(
            '<robot name="test"><link name="root"/><link name="left"/><link name="right"/></robot>'
        )
        actors = {"table": 10, "book": 11}
        if ground:
            actors["default_plane"] = 12

        def actor_ids(uid, links=None):
            if uid == "robot":
                ids = {"root": 1, "left": 2, "right": 3}
                return torch.tensor([[ids[link] for link in links]])
            return torch.tensor([[actors[uid]]])

        sensor = SimpleNamespace(get_actor_ids=actor_ids)
        env = SimpleNamespace(
            control_frequency=frequency,
            robot=SimpleNamespace(
                uid="robot",
                link_names=["root", "left", "right"],
                cfg=SimpleNamespace(
                    fpath=str(asset),
                    root_props=SimpleNamespace(self_collision_enabled=True),
                ),
            ),
            sim=SimpleNamespace(
                sim_config=SimpleNamespace(enable_ground_plane=ground),
                get_rigid_object_uid_list=lambda: ["table", "book"],
                get_rigid_object=lambda uid: SimpleNamespace(
                    body_type="dynamic" if uid == "book" else "static"
                ),
            ),
        )
        cfg = {
            "initial_checks": [],
            "mounting_contacts": [],
            "allowed_contacts": [],
            "events": [],
            "final_checks": [
                {"kind": "joint_range", "joint": "j", "lower": -1, "upper": 1}
            ],
            "contact_hold": {
                "entity": "book",
                "links": ["left", "right"],
                "duration_s": duration,
                "max_translation_drift_m": 0.001,
                "max_rotation_drift_rad": 0.001,
            },
        }
        return _ContactRow(env, cfg, tmp_path, sensor=sensor)

    return build


@pytest.fixture
def sample_contacts():
    """Feed the production observer/sample boundary using explicit physics intervals."""
    from embodichain.lab.expansion.acceptance import MeasuredExpansionValidator

    def build(row):
        row.env.num_envs = 1
        row.env.robot.joint_names = ["j"]
        row.env.robot.get_qpos = lambda: torch.zeros((1, 1))
        row.active = True
        state = SimpleNamespace(
            pose=torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]]),
            contacts=torch.ones((1, 2), dtype=torch.bool),
        )
        row.env.sim.get_rigid_object = lambda uid: SimpleNamespace(
            get_local_pose=lambda: state.pose
        )
        row.env.sim.simulation_time = 0.0
        row.env.sim.physics_step_index = 0
        row.sensor.get_data = lambda: {
            "user_ids": torch.tensor([[[11, 2], [11, 3]]]),
            "is_valid": state.contacts,
            "impulse_valid": torch.ones((1, 2), dtype=torch.bool),
        }
        row.monitor.reset(row.env)
        validator = object.__new__(MeasuredExpansionValidator)
        validator.env = row.env
        validator.rows = {0: row}
        validator.sensor = row.sensor
        validator.checker = SimpleNamespace(
            check=lambda: torch.zeros((1, 3), dtype=torch.int64)
        )
        validator.terminal, validator.results = {}, {}

        def sample(dts=None):
            validator.begin_control_step()
            for dt in ((1 / row.env.control_frequency,) if dts is None else dts):
                row.env.sim.simulation_time += dt
                row.env.sim.physics_step_index += 1
                validator.after_sample(dt, None)
            validator.sample((0,), None)
            return row.terminal[0]

        return sample, state

    return build


@pytest.mark.parametrize(
    "duration,frequency,required",
    [
        (0.25, 10, 4),
        (0.025, 10, 2),
        (0.3, 10, 4),
        (0.14, 50, 8),
        (0.07, 100, 8),
        (0.1400000001, 50, 9),
        (0.0700000001, 100, 9),
    ],
)
def test_contact_hold_requires_full_duration_and_distinct_measured_times(
    contact_row, sample_contacts, duration, frequency, required
):
    row = contact_row(duration, ground=True, frequency=frequency)
    sample, state = sample_contacts(row)
    for _ in range(required - 1):
        assert not sample().accepted
    assert sample().accepted
    report = row._hold_report()
    assert report["elapsed_duration_s"] >= duration - 1e-15
    assert report["required_samples"] == required
    # Repeated observations without physics cannot establish a hold.
    row.reset()
    row.active = True
    for _ in range(required):
        assert not sample(()).accepted
    assert not row._hold_report()["passed"]
    row.reset()
    row.active = True
    for _ in range(required - 1):
        sample()
    state.contacts[0, 0] = False
    assert not sample().accepted
    row.reset()
    row.active = True
    state.contacts.fill_(True)
    for _ in range(required - 1):
        sample()
    state.pose[0, 0] = 0.1
    assert not sample().accepted
    assert row._hold_report()["max_translation_drift_m"] == pytest.approx(0.1)


@pytest.mark.parametrize("ground", [False, True])
def test_native_contacts_only_resolve_present_default_plane(contact_row, ground):
    row = contact_row(ground=ground)
    assert {"table", "book", "robot/root", "robot/left", "robot/right"} <= set(
        row.frames.values()
    )
    assert ("default_plane" in row.frames.values()) is ground


def test_contact_sampling_uses_measured_physics_intervals(contact_row, sample_contacts):
    row = contact_row(ground=True)
    sample, _ = sample_contacts(row)
    for accepted in (False, False, False, True):
        assert sample().accepted is accepted
    assert row._hold_report()["elapsed_duration_s"] == pytest.approx(0.3)


@pytest.mark.parametrize("prior_steps", [500_000, 5_000_000])
@pytest.mark.parametrize("physics_dt,passed", [(0.001, True), (0.0009999, False)])
def test_hold_clock_is_independent_of_long_running_world_roundoff(
    contact_row, sample_contacts, prior_steps, physics_dt, passed
):
    row = contact_row(0.7, ground=True, frequency=100)
    sample, _ = sample_contacts(row)
    clock = 0.0
    for _ in range(prior_steps):
        clock += 0.001
    row.env.sim.simulation_time = clock
    row.env.sim.physics_step_index = prior_steps
    # 71 samples span 700 steps; verify additional rolling windows as well.
    for index in range(80):
        result = sample((physics_dt,) * 10)
        assert result.accepted is (passed and index >= 70)
    report = row._hold_report()
    assert report["elapsed_duration_s"] == pytest.approx(700 * physics_dt, abs=1e-15)
    assert row.env.sim.simulation_time > clock
    assert report["passed"] is passed


def test_hold_clock_follows_variable_physics_intervals(contact_row, sample_contacts):
    row = contact_row(0.25, ground=True, frequency=10)
    sample, _ = sample_contacts(row)
    for intervals, passed in (
        ((0.03, 0.07), False),
        ((0.02, 0.08), False),
        ((0.01, 0.04, 0.05), False),
        ((0.04, 0.06), True),
    ):
        assert sample(intervals).accepted is passed
    assert row._hold_report()["elapsed_duration_s"] == pytest.approx(0.3)
