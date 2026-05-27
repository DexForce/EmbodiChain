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
from types import SimpleNamespace

import numpy as np
import torch

from embodichain.lab.sim.agent import atomic_engine_planner as graph_executor
from embodichain.lab.sim.agent.atomic_graph_executor import AtomicGraphAction
from embodichain.lab.sim.agent.edge_action_executor import (
    ActionPlan,
    EdgeActionExecutor,
)


@dataclass
class _Edge:
    left_arm_action = None
    right_arm_action = None
    monitor_sequences = None
    is_recovery = False


class _PlanAction:
    def __init__(self, plan: ActionPlan) -> None:
        self._plan = plan

    def plan(self, env=None, **kwargs) -> ActionPlan:
        return self._plan


class _Robot:
    def __init__(self) -> None:
        self.dof = 8
        self.device = torch.device("cpu")
        self.qpos = torch.zeros(1, 8, dtype=torch.float32)

    def get_qpos(self, name=None) -> torch.Tensor:
        if name == "left_arm":
            return self.qpos[:, [0, 1]].clone()
        if name == "right_arm":
            return self.qpos[:, [4, 5]].clone()
        if name == "left_eef":
            return self.qpos[:, [2, 3]].clone()
        if name == "right_eef":
            return self.qpos[:, [6, 7]].clone()
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
        self.close_state = torch.tensor([0.0], dtype=torch.float32)
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


def _disable_post_action_validators(monkeypatch) -> None:
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_grasp_after_action",
        lambda env, kwargs: None,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.agent.edge_action_executor.validate_pending_public_place_after_action",
        lambda env, kwargs: None,
    )


def _fake_move_engine_factory(env: _Env):
    class _Engine:
        def __init__(self, cfg_list) -> None:
            self.cfg_list = cfg_list if isinstance(cfg_list, list) else [cfg_list]

        def execute_static(self, target_list):
            cfg = self.cfg_list[0]
            steps = int(getattr(cfg, "sample_interval", 2))
            control_part = str(cfg.control_part)
            if control_part.startswith("left"):
                joint_ids = (
                    env.left_eef_joints
                    if control_part.endswith("_eef")
                    else env.left_arm_joints
                )
            else:
                joint_ids = (
                    env.right_eef_joints
                    if control_part.endswith("_eef")
                    else env.right_arm_joints
                )
            target_qpos = target_list[0]
            if target_qpos.ndim == 1:
                target_qpos = target_qpos.unsqueeze(0)
            trajectory = torch.zeros(1, steps, env.robot.dof)
            weights = torch.linspace(0, 1, steps=steps)
            for index, weight in enumerate(weights):
                trajectory[:, index, joint_ids] = target_qpos * weight
            return True, trajectory

    return lambda env, cfg_list: _Engine(cfg_list)


def test_executor_composes_single_arm_plan(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.right_arm_action = _PlanAction(
        _plan([[1.0, 1.1], [2.0, 2.1]], env.right_arm_joints)
    )
    _disable_post_action_validators(monkeypatch)

    result = EdgeActionExecutor().execute(edge=edge, env=env)

    assert result.monitor_index is None
    assert len(result.actions) == 2
    np.testing.assert_allclose(env.step_calls[-1][env.right_arm_joints], [2.0, 2.1])
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [0.0, 0.0])


def test_executor_preserves_cached_gripper_target_for_arm_only_plan(
    monkeypatch,
) -> None:
    env = _Env()
    env.robot.qpos[:, env.right_eef_joints] = torch.tensor([0.03, 0.04])
    env.right_arm_current_gripper_state = env.close_state
    edge = _Edge()
    edge.right_arm_action = _PlanAction(
        _plan([[1.0, 1.1], [2.0, 2.1]], env.right_arm_joints)
    )
    _disable_post_action_validators(monkeypatch)

    EdgeActionExecutor().execute(edge=edge, env=env)

    np.testing.assert_allclose(env.step_calls[-1][env.right_eef_joints], [0.0, 0.0])
    torch.testing.assert_close(env.right_arm_current_gripper_state, env.close_state)


def test_executor_syncs_agent_state_after_success(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.left_arm_action = _PlanAction(
        _plan([[0.2, 0.3], [0.4, 0.5]], env.left_arm_joints)
    )
    _disable_post_action_validators(monkeypatch)

    EdgeActionExecutor().execute(edge=edge, env=env)

    torch.testing.assert_close(env.left_arm_current_qpos, torch.tensor([0.4, 0.5]))
    torch.testing.assert_close(env.left_arm_current_xpos[0, 3], torch.tensor(0.9))


def test_executor_pads_shorter_dual_arm_plan(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.left_arm_action = _PlanAction(_plan([[1.0, 1.1]], env.left_arm_joints))
    edge.right_arm_action = _PlanAction(
        _plan([[2.0, 2.1], [3.0, 3.1]], env.right_arm_joints)
    )
    _disable_post_action_validators(monkeypatch)

    EdgeActionExecutor().execute(edge=edge, env=env)

    assert len(env.step_calls) == 2
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [1.0, 1.1])
    np.testing.assert_allclose(env.step_calls[-1][env.right_arm_joints], [3.0, 3.1])


def test_executor_writes_gripper_only_plan(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.left_arm_action = _PlanAction(_plan([[0.04, 0.04]], env.left_eef_joints))
    _disable_post_action_validators(monkeypatch)

    EdgeActionExecutor().execute(edge=edge, env=env)

    np.testing.assert_allclose(env.step_calls[-1][env.left_eef_joints], [0.04, 0.04])
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [0.0, 0.0])


def test_executor_returns_monitor_trigger_and_syncs_state(monkeypatch) -> None:
    env = _Env()
    env._pending_public_grasp_physical_validations = [{"obj_name": "bottle"}]
    env._pending_public_grasp_physical_validation = {"obj_name": "legacy_bottle"}
    env._pending_public_place_validations = [{"obj_name": "cup"}]
    edge = _Edge()
    edge.right_arm_action = _PlanAction(
        _plan([[1.0, 1.1], [2.0, 2.1]], env.right_arm_joints)
    )
    edge.monitor_sequences = [[partial(lambda: True)]]
    _disable_post_action_validators(monkeypatch)

    result = EdgeActionExecutor().execute(edge=edge, env=env)

    assert result.monitor_index == 0
    assert result.step_index == 0
    assert len(result.actions) == 1
    np.testing.assert_allclose(env.right_arm_current_qpos, np.array([1.0, 1.1]))
    assert env._pending_public_grasp_physical_validations == []
    assert env._pending_public_grasp_physical_validation is None
    assert env._pending_public_place_validations == []


def test_executor_executes_atomic_graph_action_wrapper(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.right_arm_action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "right_arm", "sample_interval": 2},
            "target": {"kind": "eef_rotation_delta", "joint_index": 1, "degree": 90},
        }
    )
    _disable_post_action_validators(monkeypatch)
    monkeypatch.setattr(
        graph_executor,
        "_create_engine",
        _fake_move_engine_factory(env),
    )

    result = EdgeActionExecutor().execute(edge=edge, env=env)

    assert len(result.actions) == 2
    np.testing.assert_allclose(
        env.step_calls[-1][env.right_arm_joints],
        [0.0, np.pi / 2],
        rtol=1e-6,
    )


def test_executor_pads_dual_atomic_graph_action_wrappers(monkeypatch) -> None:
    env = _Env()
    edge = _Edge()
    edge.left_arm_action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "left_arm", "sample_interval": 2},
            "target": {"kind": "joint_delta", "joint_index": 0, "degree": 45},
        }
    )
    edge.right_arm_action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "right_arm", "sample_interval": 3},
            "target": {"kind": "joint_delta", "joint_index": 1, "degree": 90},
        }
    )
    _disable_post_action_validators(monkeypatch)
    monkeypatch.setattr(
        graph_executor,
        "_create_engine",
        _fake_move_engine_factory(env),
    )

    EdgeActionExecutor().execute(edge=edge, env=env)

    assert len(env.step_calls) == 3
    np.testing.assert_allclose(
        env.step_calls[-1][env.left_arm_joints],
        [np.pi / 4, 0.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        env.step_calls[-1][env.right_arm_joints],
        [0.0, np.pi / 2],
        rtol=1e-6,
    )


def test_executor_executes_gripper_atomic_graph_action_wrapper(monkeypatch) -> None:
    env = _Env()

    class _Engine:
        def __init__(self, cfg_list) -> None:
            self.cfg_list = cfg_list if isinstance(cfg_list, list) else [cfg_list]

        def execute_static(self, target_list):
            trajectory = torch.zeros(1, 3, env.robot.dof)
            trajectory[:, :, env.left_eef_joints] = torch.tensor([0.05, 0.05])
            return True, trajectory

    monkeypatch.setattr(
        graph_executor,
        "_create_engine",
        lambda env, cfg_list: _Engine(cfg_list),
    )
    _disable_post_action_validators(monkeypatch)
    edge = _Edge()
    edge.left_arm_action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {
                "control_part": "left_eef",
                "arm_control_part": "left_arm",
                "sample_interval": 3,
            },
            "target": {"kind": "gripper_state", "state": "open"},
        }
    )

    EdgeActionExecutor().execute(edge=edge, env=env)

    np.testing.assert_allclose(env.step_calls[-1][env.left_eef_joints], [0.05, 0.05])
    np.testing.assert_allclose(env.step_calls[-1][env.left_arm_joints], [0.0, 0.0])


def test_executor_passes_recovery_flag_to_atomic_graph_action(monkeypatch) -> None:
    env = _Env()
    captured = {}
    monkeypatch.setattr(
        graph_executor,
        "_resolve_action_target",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        graph_executor,
        "_public_grasp_approach_direction_candidates",
        lambda *args, **kwargs: [("ranked", torch.tensor([0.0, 0.0, -1.0]))],
    )

    def _capture_engine(env, cfg_list):
        cfg = cfg_list[0]
        captured["kwargs"] = cfg.grasp_rank_options

        class _Engine:
            _action_sequence = [("pick_up", SimpleNamespace())]

            def execute_static(self, target_list):
                return True, torch.zeros(1, 1, env.robot.dof)

        return _Engine()

    monkeypatch.setattr(graph_executor, "_create_engine", _capture_engine)
    monkeypatch.setattr(
        graph_executor, "_object_geometry_bounds", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        graph_executor, "_public_grasp_roll_offsets", lambda *args, **kwargs: [0.0]
    )
    monkeypatch.setattr(
        graph_executor,
        "_build_legacy_grasp_pose",
        lambda *args, **kwargs: torch.eye(4),
    )
    _disable_post_action_validators(monkeypatch)
    edge = _Edge()
    edge.is_recovery = True
    edge.right_arm_action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "pick_up",
            "cfg": {"control_part": "right_arm", "hand_control_part": "right_eef"},
            "target": {"kind": "object_semantics", "obj_name": "cup"},
            "runtime_kwargs": {"public_grasp_strategy": "legacy_guided"},
        }
    )

    EdgeActionExecutor().execute(
        edge=edge,
        env=env,
        recovery_public_grasp_strategy="auto_general",
        recovery_public_grasp_candidate_num=64,
    )

    assert captured["kwargs"]["_edge_is_recovery"] is True
    assert captured["kwargs"]["public_grasp_strategy"] == "auto_general"
    assert captured["kwargs"]["public_grasp_candidate_num"] == 64
