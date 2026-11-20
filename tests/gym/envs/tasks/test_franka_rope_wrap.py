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

"""CPU regressions for the Franka task's timed acceptance and contact evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain_tasks.manipulation.cable.franka_rope_wrap import load_config
from embodichain_tasks.manipulation.cable.wrap_evaluation import attach_wrap_monitor
from embodichain_tasks.manipulation.cable.wrap_measurements import evaluate_wrap

pytestmark = pytest.mark.no_sim


def test_scaled_pickup_and_final_hold_are_evaluated_at_their_measured_times(tmp_path):
    path = tmp_path / "72s.json"
    path.write_text(json.dumps({**load_config(), "duration_s": 72.0}))
    config = load_config(path)
    # At 2x duration the grasp finishes at 10 s, the lift at 14 s, and
    # the final two seconds of the measured hold start at 70 s.
    rows = []
    for frame in range(72 * 60 + 1):
        t = frame / 60
        rows.append(
            dict(
                time_s=t,
                finite=True,
                arm_joint_limit_violation_rad=0,
                finger_joint_limit_violation_m=0,
                free_end_m=[0, 0, 0.06 if t >= 14 else 0.006],
                grasp_confirmed=t >= 10,
                bilateral_rope_contact=t >= 10,
                tail_tcp_distance_m=0.025,
                material_slip_m=0,
                winding_deg=420 if t >= 70 else 0,
                contact_arc_deg=330 if t >= 70 else 0,
                rope_max_z_m=0.08,
                rope_min_z_m=0.002,
                post_penetration_m=0,
                robot_post_penetration_m=0,
                anchor_direction_error_deg=0,
                anchor_exit_bend_max_deg=0,
                anchor_error_m=0,
                joint_gap_max_m=0,
                dropped_contacts=0,
            )
        )

    report = evaluate_wrap(rows, config)
    assert len(report["checks"]) == 19 and report["passed"]
    partial = evaluate_wrap(rows[:-1], config)
    assert not partial["checks"]["completed"] and not partial["passed"]

    # A bad measurement at the first hold frame cannot be hidden by recovery.
    rows[70 * 60]["contact_arc_deg"] = 295
    failed = evaluate_wrap(rows, config)
    assert {name for name, ok in failed["checks"].items() if not ok} == {
        "post_contact_arc"
    }


def test_grasp_requires_both_fingers_on_rope_in_the_same_sample(tmp_path):
    config = {**load_config(), "grasp_complete_s": 0, "finger_pad_center_m": [0, 0, 0]}
    env = Mock(num_envs=1, _wrap_monitor=None)
    cable = env.sim.get_cable_object.return_value
    points = np.zeros((9, 3))
    points[:, 2] = np.arange(9) * 0.01
    cable.cfg.points = points
    segments = torch.zeros(1, 8, 13, dtype=torch.float64)
    segments[0, :, :3] = torch.from_numpy((points[1:] + points[:-1]) / 2)
    segments[0, :, 6] = 1  # identity quaternion; capsule axes point up
    cable.data.segment_state_w = segments
    tcp = torch.eye(4, dtype=torch.float64)[None]
    tcp[0, :3, 3] = segments[0, -1, :3]
    env.robot.get_link_pose.return_value = tcp
    env.robot.get_qpos.return_value = torch.zeros(1, 9)
    env.robot.joint_names = [f"j{i}" for i in range(9)]
    urdf = tmp_path / "limits.urdf"
    urdf.write_text(
        "<robot>"
        + "".join(
            f'<joint name="{name}"><limit lower="-3" upper="3"/></joint>'
            for name in env.robot.joint_names
        )
        + "</robot>"
    )
    motion = SimpleNamespace(
        poses=np.tile(np.eye(4), (4, 1, 1)),
        labels=["grasp"] * 4,
        times=np.arange(4) / 60,
    )
    sensor = env.sim.get_sensor.return_value
    sensor.latest_sample_dropped_contacts = 0
    names = ["fr3_leftfinger", "fr3_rightfinger"]
    sensor.get_actor_info.side_effect = lambda i: SimpleNamespace(
        path="/franka", link_name=names[i]
    )
    monitor = attach_wrap_monitor(env, config, motion, urdf)

    def advance(contacts):
        monitor.begin_control_step()
        for substep in range(1, 33):
            fingers = contacts.get(substep, [])
            sensor.get_data.return_value = {
                "user_ids": torch.tensor(
                    [[i, -1] for i in fingers], dtype=torch.long
                ).reshape(-1, 2),
                "distance": torch.zeros(len(fingers)),
                "position": torch.tensor([[0.004, 0.0, 0.075]]).repeat(len(fingers), 1),
                "is_valid": torch.ones(len(fingers), dtype=torch.bool),
            }
            monitor.after_sample(1 / 1920, None)
        return monitor.rows[-1]

    # Between samples both touch, but on the 240 Hz sample grid their
    # contacts alternate. The fingers must not jointly confirm this grasp.
    split = advance({3: [0, 1], 8: [0], 16: [1]})
    assert split["left_rope_contact"] and split["right_rope_contact"]
    assert not split["bilateral_rope_contact"] and not split["grasp_confirmed"]
    together = advance({24: [0, 1]})
    assert together["bilateral_rope_contact"] and together["grasp_confirmed"]
    assert sensor.get_data.call_count == 8
