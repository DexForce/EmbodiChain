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

"""Bounded measured physical evidence, independent of scoring projections."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain_benchmarks.motion_generation.scenarios._replay_diagnostics import (
    ReplayDiagnostics,
    resolve_diagnostics_config,
)


class _Robot:
    device = torch.device("cpu")

    def __init__(self):
        self.qpos = torch.zeros(2, 16)

    def get_joint_ids(self, name):
        return {
            "left_arm": list(range(7)),
            "right_arm": list(range(7, 14)),
            "left_eef": [14],
            "right_eef": [15],
        }[name]

    def get_qpos(self, name=None):
        return self.qpos if name is None else self.qpos[:, self.get_joint_ids(name)]

    def compute_fk(self, qpos, name, to_matrix):
        assert qpos.shape == (2, 7)
        value = torch.eye(4).repeat(2, 1, 1)
        value[:, :3, 3] = qpos[:, :3]
        return value


def _inputs():
    plans = [
        SimpleNamespace(
            skill_id="pick_up",
            segments=[SimpleNamespace(name="grasp", start=0, stop=10)],
        ),
        SimpleNamespace(
            skill_id="pick_up",
            segments=[SimpleNamespace(name="lift", start=0, stop=10)],
        ),
        SimpleNamespace(
            skill_id="coordinated_placement",
            segments=[SimpleNamespace(name="approach", start=0, stop=10)],
        ),
    ]
    compiled = SimpleNamespace(
        action_plans=plans,
        segment=lambda index, name: SimpleNamespace(
            name=name, start=index * 10, stop=(index + 1) * 10
        ),
    )
    case = SimpleNamespace(
        object_id="bread",
        case_parameters={
            "arm_waypoints": {"right_arm": []},
            "scene_object_poses": {"pan": []},
        },
    )
    poses = {name: torch.eye(4).repeat(2, 1, 1) for name in ("bread", "pan")}
    objects = {
        name: SimpleNamespace(
            entity=SimpleNamespace(
                uid=name, get_local_pose=lambda to_matrix, name=name: poses[name]
            )
        )
        for name in poses
    }
    simulation = SimpleNamespace(physics=SimpleNamespace(supports_contact_sensor=False))
    return compiled, case, simulation, _Robot(), objects, poses


def test_trace_covers_prefix_actions_and_preserves_measured_scene_rows():
    compiled, case, simulation, robot, objects, poses = _inputs()
    trace = ReplayDiagnostics(
        {"enabled": True, "samples_per_segment": 3, "max_contacts_per_env": 4},
        compiled,
        case,
        simulation,
        robot,
        objects,
        "left_arm",
        ("left_eef", "right_eef"),
    )
    assert trace.sample_indices == {0, 4, 9, 10, 14, 19, 20, 24, 29}
    command = torch.ones(2, 16)
    robot.qpos[1, 14] = 0.02
    poses["bread"][1, 2, 3] = 0.7
    trace.capture(
        waypoint_index=9, elapsed_s=1.25, command=command, phase="command_end"
    )
    robot.qpos[1, 14] = 0.03
    poses["bread"][1, 2, 3] = 0.8
    result = trace.finish()
    assert len(result["segments"]) == 3
    sample = result["samples"][0]
    assert sample["elapsed_sim_time_s"] == 1.25
    assert sample["gripper_qpos"]["left_eef"][1] == pytest.approx([0.02])
    assert sample["gripper_command_qpos"]["left_eef"][1] == [1.0]
    assert sample["objects"]["bread"][1][2][3] == pytest.approx(0.7)
    assert set(sample["tcp_poses"]) == {"left_arm", "right_arm"}
    assert result["contacts"] == {"available": False, "reason": "backend_unsupported"}
    json.dumps(result, allow_nan=False)


def test_trace_retains_valid_contact_identities_capacity_and_unavailable_impulse(
    monkeypatch,
):
    from embodichain.lab.sim import sensors

    class FakeContactSensor:
        contact_capabilities = SimpleNamespace(
            geometry=True, impulse=False, friction=False
        )
        dropped_contacts = 3

        def __init__(self, cfg, device, owner):
            assert cfg.rigid_uid_list == ["bread", "pan"]
            assert cfg.max_contacts_per_env == 4
            assert cfg.filter_need_both_actor is False

        def update(self):
            pass

        def get_data(self):
            valid = torch.tensor([[True, False], [False, False]])
            return TensorDict(
                {
                    "is_valid": valid,
                    "user_ids": torch.tensor(
                        [[[10, 20], [999, 999]], [[999, 999], [999, 999]]]
                    ),
                    "position": torch.zeros(2, 2, 3),
                    "normal": torch.zeros(2, 2, 3),
                    "distance": torch.zeros(2, 2),
                    "impulse": torch.ones(2, 2),
                },
                batch_size=[2, 2],
            )

        def get_actor_info(self, actor_id):
            return SimpleNamespace(
                path=f"arena0/{actor_id}",
                link_name="finger" if actor_id == 20 else None,
                env_id=0,
            )

    monkeypatch.setattr(sensors, "ContactSensor", FakeContactSensor)
    compiled, case, simulation, robot, objects, _ = _inputs()
    simulation.physics.supports_contact_sensor = True
    trace = ReplayDiagnostics(
        {"enabled": True, "samples_per_segment": 2, "max_contacts_per_env": 4},
        compiled,
        case,
        simulation,
        robot,
        objects,
        "left_arm",
        ("left_eef", "right_eef"),
    )
    trace.capture(
        waypoint_index=9, elapsed_s=1.0, command=robot.qpos, phase="command_end"
    )
    result = trace.finish()
    sample = result["samples"][0]
    assert sample["contacts"][0]["user_ids"] == [[10, 20]]
    assert sample["contacts"][1]["user_ids"] == []
    assert sample["contacts"][0]["impulse"] is None
    assert sample["dropped_contact_rows"] == 3
    assert set(result["contact_actors"]) == {"10", "20"}
    assert result["contact_actors"]["20"]["link_name"] == "finger"
    json.dumps(result, allow_nan=False)


def test_replay_diagnostics_default_off_and_explicit_bounded_configuration():
    assert resolve_diagnostics_config({}) == {"enabled": False}
    assert resolve_diagnostics_config({"replay_diagnostics": {"enabled": True}}) == {
        "enabled": True,
        "samples_per_segment": 8,
        "max_contacts_per_env": 64,
    }
    with pytest.raises(ValueError, match="samples_per_segment"):
        resolve_diagnostics_config(
            {"replay_diagnostics": {"samples_per_segment": 1000}}
        )
    with pytest.raises(ValueError, match="Unknown replay_diagnostics"):
        resolve_diagnostics_config({"replay_diagnostics": {"sample_count": 8}})
