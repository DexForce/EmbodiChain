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

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from embodichain.lab.sim.agent.edge_action_executor import (
    ActionPlan,
    EdgeActionExecutor,
)


@dataclass
class _Edge:
    left_arm_action = None
    right_arm_action = None
    monitor_sequences = None


class _PlanAction:
    def __init__(self, plan: ActionPlan) -> None:
        self._plan = plan

    def plan(self, env=None, **kwargs) -> ActionPlan:
        return self._plan


class _Robot:
    def __init__(self) -> None:
        self.qpos = torch.zeros(1, 8, dtype=torch.float32)

    def get_qpos(self) -> torch.Tensor:
        return self.qpos.clone()

    def compute_fk(self, qpos, name: str, to_matrix: bool = True) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        pose[0, 0, 3] = float(torch.as_tensor(qpos).sum())
        return pose


class _Env:
    def __init__(self) -> None:
        self.left_arm_joints = [0, 1]
        self.left_eef_joints = [2, 3]
        self.right_arm_joints = [4, 5]
        self.right_eef_joints = [6, 7]
        self.open_state = torch.tensor([0.05], dtype=torch.float32)
        self.robot = _Robot()
        self.step_calls: list[np.ndarray] = []
        self.update_calls = 0
        self.left_arm_current_qpos = torch.zeros(2)
        self.right_arm_current_qpos = torch.zeros(2)
        self.left_arm_current_xpos = torch.eye(4)
        self.right_arm_current_xpos = torch.eye(4)
        self.left_arm_current_gripper_state = torch.tensor([0.05])
        self.right_arm_current_gripper_state = torch.tensor([0.05])

    def step(self, action: torch.Tensor) -> None:
        qpos = action.detach().cpu().squeeze(0)
        self.robot.qpos = qpos.unsqueeze(0)
        self.step_calls.append(qpos.numpy())

    def update_obj_info(self) -> None:
        self.update_calls += 1


def _plan(values, joint_ids, *, name="test") -> ActionPlan:
    return ActionPlan(
        is_success=True,
        trajectory=torch.tensor([values], dtype=torch.float32),
        joint_ids=list(joint_ids),
        action_name=name,
    )


def test_executor_composes_single_arm_plan(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.right_arm_action = _PlanAction(
        _plan([[1.0, 1.1], [2.0, 2.1]], env.right_arm_joints)
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_grasp_after_action",
        lambda env, kwargs: None,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_place_after_action",
        lambda env, kwargs: None,
    )

    result = EdgeActionExecutor().execute(edge=edge, env=env)

    assert result.monitor_index is None
    assert len(result.actions) == 2
    np.testing.assert_allclose(env.step_calls[-1][env.right_arm_joints], [2.0, 2.1])
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [0.0, 0.0])


def test_executor_pads_shorter_dual_arm_plan(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.left_arm_action = _PlanAction(_plan([[1.0, 1.1]], env.left_arm_joints))
    edge.right_arm_action = _PlanAction(
        _plan([[2.0, 2.1], [3.0, 3.1]], env.right_arm_joints)
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_grasp_after_action",
        lambda env, kwargs: None,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_place_after_action",
        lambda env, kwargs: None,
    )

    EdgeActionExecutor().execute(edge=edge, env=env)

    assert len(env.step_calls) == 2
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [1.0, 1.1])
    np.testing.assert_allclose(env.step_calls[-1][env.right_arm_joints], [3.0, 3.1])


def test_executor_writes_gripper_only_plan(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.left_arm_action = _PlanAction(_plan([[0.04, 0.04]], env.left_eef_joints))
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_grasp_after_action",
        lambda env, kwargs: None,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_place_after_action",
        lambda env, kwargs: None,
    )

    EdgeActionExecutor().execute(edge=edge, env=env)

    np.testing.assert_allclose(env.step_calls[-1][env.left_eef_joints], [0.04, 0.04])
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [0.0, 0.0])


def test_executor_returns_monitor_trigger_and_syncs_state(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.right_arm_action = _PlanAction(
        _plan([[1.0, 1.1], [2.0, 2.1]], env.right_arm_joints)
    )
    edge.monitor_sequences = [[partial(lambda: True)]]
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_grasp_after_action",
        lambda env, kwargs: None,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_place_after_action",
        lambda env, kwargs: None,
    )

    result = EdgeActionExecutor().execute(edge=edge, env=env)

    assert result.monitor_index == 0
    assert result.step_index == 0
    assert len(result.actions) == 1
    np.testing.assert_allclose(env.right_arm_current_qpos, np.array([1.0, 1.1]))
