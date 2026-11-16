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
    def build(duration=0.25, ground=False):
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
            control_frequency=10,
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


@pytest.mark.parametrize("duration,required", [(0.25, 4), (0.025, 2), (0.3, 4)])
def test_contact_hold_requires_full_duration_and_distinct_measured_times(
    contact_row, duration, required
):
    row = contact_row(duration, ground=True)
    pose = [0, 0, 0, 0, 0, 0, 1]
    for i in range(required - 1):
        row.samples.append((pose.copy(), [True, True], i / 10))
    assert not row._hold_report()["passed"]
    row.samples.append((pose.copy(), [True, True], (required - 1) / 10))
    report = row._hold_report()
    assert report["passed"]
    assert report["elapsed_duration_s"] >= duration
    assert row.samples.maxlen == required
    # Enough samples at one physical instant cannot establish a hold duration.
    row.samples.clear()
    row.samples.extend((pose.copy(), [True, True], 0.0) for _ in range(required))
    assert not row._hold_report()["passed"]
    row.samples.clear()
    row.samples.extend((pose.copy(), [True, True], i / 10) for i in range(required))
    row.samples[-1][1][0] = False
    assert not row._hold_report()["passed"]
    row.samples[-1][1][0] = True
    row.samples[-1][0][0] = 0.1
    assert not row._hold_report()["passed"]


@pytest.mark.parametrize("ground", [False, True])
def test_native_contacts_only_resolve_present_default_plane(contact_row, ground):
    row = contact_row(ground=ground)
    assert {"table", "book", "robot/root", "robot/left", "robot/right"} <= set(
        row.frames.values()
    )
    assert ("default_plane" in row.frames.values()) is ground


def test_contact_sampling_uses_measured_world_time(contact_row):
    from unittest.mock import Mock
    from embodichain.lab.expansion.contracts import ValidationCheck, ValidationResult

    row = contact_row(ground=True)
    passed = ValidationResult((ValidationCheck("task.success", "passed"),))
    row.monitor = SimpleNamespace(observe=Mock(), validation=lambda env: passed)
    row.env.sim.get_rigid_object = lambda uid: SimpleNamespace(
        get_local_pose=lambda: torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]])
    )
    row.sensor.get_data = lambda: {
        "user_ids": torch.tensor([[[11, 2], [11, 3]]]),
        "is_valid": torch.ones((1, 2), dtype=torch.bool),
        "impulse_valid": torch.ones((1, 2), dtype=torch.bool),
    }
    for timestamp, accepted in ((0.0, False), (0.1, False), (0.2, False), (0.3, True)):
        row.env.sim.simulation_time = timestamp
        row._interval_samples = 1
        row.sample((0,), None)
        assert row.terminal[0].accepted is accepted
    assert row._hold_report()["elapsed_duration_s"] == 0.3
