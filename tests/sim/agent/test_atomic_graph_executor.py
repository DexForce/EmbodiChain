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

import torch

from embodichain.lab.sim.agent import atomic_graph_executor as executor
from embodichain.lab.sim.agent.atomic_graph_executor import AtomicGraphAction
from embodichain.lab.sim.agent.edge_action_executor import ActionPlan
from embodichain.lab.sim.atomic_actions import GripperActionCfg


class _Object:
    def get_local_pose(self, to_matrix: bool = True) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        pose[0, :3, 3] = torch.tensor([0.4, 0.2, 0.1])
        return pose


class _Sim:
    def get_rigid_object_uid_list(self):
        return ["cup"]

    def get_rigid_object(self, obj_name: str):
        assert obj_name == "cup"
        return _Object()


class _Robot:
    uid = "robot"
    device = torch.device("cpu")
    dof = 8

    def get_qpos(self, name=None):
        if name == "left_arm":
            return torch.zeros(1, 2)
        if name == "right_arm":
            return torch.zeros(1, 2)
        if name == "left_eef":
            return torch.zeros(1, 2)
        if name == "right_eef":
            return torch.zeros(1, 2)
        return torch.zeros(1, 8)


class _Env:
    def __init__(self) -> None:
        self.robot = _Robot()
        self.sim = _Sim()
        self.left_arm_joints = [0, 1]
        self.left_eef_joints = [2, 3]
        self.right_arm_joints = [4, 5]
        self.right_eef_joints = [6, 7]
        self.left_arm_current_xpos = torch.eye(4, dtype=torch.float32)
        self.right_arm_current_xpos = torch.eye(4, dtype=torch.float32)
        self.left_arm_init_xpos = torch.eye(4, dtype=torch.float32)
        self.right_arm_init_xpos = torch.eye(4, dtype=torch.float32)
        self.left_arm_init_qpos = torch.tensor([0.5, 0.6])
        self.right_arm_init_qpos = torch.tensor([0.7, 0.8])
        self.open_state = torch.tensor([0.05])
        self.close_state = torch.tensor([0.0])


def test_plan_returns_action_plan_for_joint_delta_move() -> None:
    env = _Env()
    action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "right_arm", "sample_interval": 3},
            "target": {"kind": "eef_rotation_delta", "joint_index": 1, "degree": 90},
        }
    )

    plan = action.plan(env=env)

    assert isinstance(plan, ActionPlan)
    assert plan.joint_ids == env.right_arm_joints
    assert plan.trajectory.shape == (1, 3, 2)
    torch.testing.assert_close(
        plan.trajectory[0, -1],
        torch.tensor([0.0, torch.pi / 2], dtype=torch.float32),
    )


def test_resolve_move_targets() -> None:
    env = _Env()

    object_relative = executor._resolve_pose_target(
        {
            "kind": "object_relative_pose",
            "obj_name": "cup",
            "x_offset": 0.1,
            "y_offset": -0.1,
            "z_offset": 0.2,
        },
        env=env,
        robot_name="left_arm",
    )
    torch.testing.assert_close(
        object_relative[:3, 3],
        torch.tensor([0.5, 0.1, 0.3], dtype=torch.float32),
    )

    absolute = executor._resolve_pose_target(
        {"kind": "absolute_position", "x": 0.2, "z": 0.4},
        env=env,
        robot_name="left_arm",
    )
    torch.testing.assert_close(
        absolute[:3, 3],
        torch.tensor([0.2, 0.0, 0.4], dtype=torch.float32),
    )

    orientation = executor._resolve_pose_target(
        {"kind": "eef_orientation", "direction": "down"},
        env=env,
        robot_name="left_arm",
    )
    torch.testing.assert_close(
        orientation[:3, :3],
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        ),
    )


def test_gripper_open_uses_gripper_action_cfg(monkeypatch) -> None:
    env = _Env()
    captured = {}

    class _Engine:
        def __init__(self, cfg_list):
            captured["cfg_list"] = cfg_list

        def execute_static(self, target_list):
            captured["target_list"] = target_list
            trajectory = torch.zeros(1, 2, 8)
            trajectory[:, :, env.left_eef_joints] = torch.tensor([0.05, 0.05])
            return True, trajectory

    monkeypatch.setattr(
        executor,
        "_create_engine",
        lambda env, cfg_list: _Engine(cfg_list),
    )
    action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "gripper_open",
            "cfg": {"control_part": "left_eef", "arm_control_part": "left_arm"},
            "target": {"kind": "gripper_state", "state": "open"},
        }
    )

    plan = action.plan(env=env)

    assert isinstance(captured["cfg_list"][0], GripperActionCfg)
    assert captured["cfg_list"][0].name == "gripper_open"
    assert plan.joint_ids == env.left_eef_joints
    torch.testing.assert_close(captured["target_list"][0], torch.tensor([0.05, 0.05]))
