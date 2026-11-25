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

"""Verify pre-roll, trajectory handoff and measured rope-wrap evidence."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
import yaml

from embodichain.extensions.tasks.configs import get_config_path
from embodichain.lab.evaluation._episode_reduction import (
    _evaluate_scalar_checks,
    _reduce_measurements,
)
from embodichain.lab.gym.envs.managers.events import pre_roll_with_robot_home
from embodichain.lab.gym.envs.task_program._episode_measurement import (
    _EpisodeMeasurements,
)
from embodichain.lab.task_program.integrations._configured_measurements import (
    _decode_measurement_plan,
)
from embodichain.lab.task_program.integrations._trajectory_phase import (
    _Phase,
    _TrajectorySource,
)

pytestmark = pytest.mark.no_sim


def _evaluation() -> dict:
    path = get_config_path(
        "tasks/manipulation/cable/franka_rope_wrap/task_program/evaluation.yaml"
    )
    return yaml.safe_load(path.read_text())


def test_pre_roll_sets_the_robot_at_rest_after_settling_the_rope() -> None:
    env = Mock(num_envs=1, device="cpu")
    env.robot.cfg.init_qpos = [0.0, 0.04]
    pre_roll_with_robot_home(
        env, None, 2, restore_home=True, zero_velocity_target=False
    )
    assert [item[0] for item in env.mock_calls] == [
        "robot.set_qpos",
        "sim.update",
        "sim.update",
        "robot.set_state",
    ]
    assert env.sim.update.call_args_list == [call(step=1, render_final_step=False)] * 2
    state = env.robot.set_state.call_args.kwargs
    home = torch.tensor([[0.0, 0.04]])
    torch.testing.assert_close(state["qpos"], home)
    torch.testing.assert_close(state["target_qpos"], home)
    torch.testing.assert_close(state["qvel"], torch.zeros_like(home))
    torch.testing.assert_close(state["target_qvel"], state["qvel"])


def test_pickup_handoff_keeps_the_authored_start_of_the_orbit() -> None:
    nominal = np.linspace(0, 0.25, 601, dtype=np.float32)[:, None].repeat(9, axis=1)
    loaded = torch.full((1, 9), 0.125)
    robot = Mock(**{"get_qpos.return_value": loaded})
    source = _TrajectorySource(
        robot,
        (_Phase("approach_post", 420, 600, True, False),),
        nominal,
        np.empty(0),
        1 / 60,
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=loaded, timestamp=390 / 60),
        env_ids=torch.tensor([0]),
        batch_size=1,
        is_preview=False,
        require_control_dt=lambda: 1 / 60,
    )
    joined = source.resolve("approach_post", 0, 180, context, torch.tensor([True]))
    assert joined.waypoint_count == 210
    torch.testing.assert_close(
        joined.positions[:, :30], loaded[:, None].expand(-1, 30, -1)
    )
    torch.testing.assert_close(
        joined.positions[0, -1], torch.from_numpy(nominal[-1]), rtol=0, atol=0
    )


def test_recorded_winding_can_be_evaluated_under_a_new_threshold() -> None:
    component = _evaluation()
    reductions = component["measurements"]["reductions"]
    graph = component["graph"]
    reference = dict(
        finite=True,
        arm_joint_limit_violation_rad=0,
        finger_joint_limit_violation_m=0,
        free_end_m=[0, 0, 0.006],
        free_end_z_m=0.006,
        grasp_confirmed=True,
        bilateral_rope_contact=True,
        tail_tcp_distance_m=0.025,
        material_slip_m=0,
        winding_deg=525,
        contact_arc_deg=330,
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
    rows = [dict(reference, frame=i) for i in range(2161)]
    for row in rows[420:451]:
        row["free_end_z_m"] = 0.06
    values = _reduce_measurements(rows, reductions)
    assert values["actual_winding"] == 525
    assert _evaluate_scalar_checks(graph, values)["wrap_success"]
    assert not _evaluate_scalar_checks(
        graph, _reduce_measurements(rows[:-1], reductions)
    )["completed"]
    graph["rules"]["actual_winding"]["params"]["minimum"] = 600
    assert not _evaluate_scalar_checks(graph, values)["wrap_success"]


def test_grasp_requires_both_fingers_in_one_sample_and_preserves_interval_loss(
    tmp_path,
) -> None:
    scene = SimpleNamespace(
        cables=[SimpleNamespace(entity_id="cable", simulation_uid="cable")],
        rigid_objects=[SimpleNamespace(entity_id="post", simulation_uid="post")],
    )
    plan = _decode_measurement_plan(_evaluation()["measurements"], scene)
    plan = replace(
        plan,
        parameters={
            **plan.parameters,
            "grasp_start": 0,
            "finger_pad_center_m": [0, 0, 0],
        },
    )
    env = Mock(num_envs=1, step_dt=1 / 60)
    env.cfg.sim_steps_per_control = 32
    cable = env.sim.get_cable_object.return_value
    points = np.zeros((9, 3))
    points[:, 2] = np.arange(9) * 0.01
    cable.cfg.points = points
    segments = torch.zeros(1, 8, 13, dtype=torch.float64)
    segments[0, :, :3] = torch.from_numpy((points[1:] + points[:-1]) / 2)
    segments[0, :, 6] = 1
    cable.data.segment_state_w = segments
    tcp = torch.eye(4, dtype=torch.float64)[None]
    tcp[0, :3, 3] = segments[0, -1, :3]
    env.robot.get_link_pose.return_value = tcp
    env.robot.get_qpos.return_value = torch.zeros(1, 9)
    env.robot.uid = "robot"
    env.robot.dof = 9
    env.robot.get_joint_ids.return_value = list(range(7))
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
    env.robot.cfg.fpath = str(urdf)
    sensor = env.sim.get_sensor.return_value
    sensor.get_actor_info.side_effect = lambda i: SimpleNamespace(
        path="/robot", link_name=plan.parameters["fingers"][i]
    )
    monitor = _EpisodeMeasurements(env, plan, None)

    def advance(contacts, loss):
        monitor.begin_control_step()
        for step in range(1, 33):
            fingers = contacts.get(step, [])
            sensor.get_data.return_value = {
                "user_ids": torch.tensor(
                    [[i, -1] for i in fingers], dtype=torch.long
                ).reshape(-1, 2),
                "distance": torch.zeros(len(fingers)),
                "position": torch.tensor([[0.004, 0.0, 0.075]]).repeat(len(fingers), 1),
                "is_valid": torch.ones(len(fingers), dtype=torch.bool),
            }
            sensor.evidence_dropped_contacts = loss if step >= 7 else 0
            monitor.after_sample(1 / 1920, None)
        return monitor.rows[-1]

    split = advance({8: [0], 16: [1]}, 1)
    assert not split["bilateral_rope_contact"]
    assert split["dropped_contacts"] == 1
    together = advance({24: [0, 1]}, 0)
    assert together["bilateral_rope_contact"]
    assert together["grasp_confirmed"]
