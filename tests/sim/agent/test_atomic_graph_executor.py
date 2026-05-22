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

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.agent import atomic_graph_executor as executor
from embodichain.lab.sim.agent.atomic_graph_executor import AtomicGraphAction
from embodichain.lab.sim.agent.edge_action_executor import ActionPlan
from embodichain.lab.sim.atomic_actions import GripperActionCfg, PlaceActionCfg


class _Object:
    def get_local_pose(self, to_matrix: bool = True) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        pose[0, :3, 3] = torch.tensor([0.4, 0.2, 0.1])
        return pose

    def get_vertices(self, env_ids=None, scale: bool = True):
        return [
            torch.tensor(
                [
                    [-0.04, -0.01, 0.0],
                    [0.04, -0.01, 0.0],
                    [0.04, 0.01, 0.0],
                    [-0.04, 0.01, 0.0],
                ],
                dtype=torch.float32,
            )
        ]

    def get_triangles(self, env_ids=None):
        return [torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)]


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


def _joint_group(env: _Env, name: str) -> list[int]:
    groups = {
        "left_arm": env.left_arm_joints,
        "left_eef": env.left_eef_joints,
        "left_full": env.left_arm_joints + env.left_eef_joints,
        "right_arm": env.right_arm_joints,
        "right_eef": env.right_eef_joints,
        "right_full": env.right_arm_joints + env.right_eef_joints,
    }
    return list(groups[name])


def _fake_engine_factory(env: _Env):
    class _Engine:
        def __init__(self, cfg_list) -> None:
            self.cfg_list = cfg_list if isinstance(cfg_list, list) else [cfg_list]

        def execute_static(self, target_list):
            cfg = self.cfg_list[0]
            steps = int(getattr(cfg, "sample_interval", 2))
            trajectory = torch.zeros(1, steps, env.robot.dof)
            if isinstance(cfg, PlaceActionCfg):
                is_left = str(cfg.control_part).startswith("left")
                joint_ids = (
                    env.left_arm_joints + env.left_eef_joints
                    if is_left
                    else env.right_arm_joints + env.right_eef_joints
                )
            elif isinstance(cfg, GripperActionCfg):
                joint_ids = (
                    env.left_eef_joints
                    if str(cfg.control_part).startswith("left")
                    else env.right_eef_joints
                )
            else:
                raise AssertionError(f"Unexpected cfg type: {type(cfg)!r}")
            values = torch.arange(steps * len(joint_ids), dtype=torch.float32).reshape(
                1, steps, len(joint_ids)
            )
            trajectory[:, :, joint_ids] = values
            return True, trajectory

    return lambda env, cfg_list: _Engine(cfg_list)


@pytest.mark.parametrize(
    ("spec", "expected_action_name", "expected_joint_group", "expected_shape"),
    [
        (
            {
                "kind": "atomic_action",
                "name": "move",
                "cfg": {"control_part": "right_arm", "sample_interval": 3},
                "target": {
                    "kind": "eef_rotation_delta",
                    "joint_index": 1,
                    "degree": 90,
                },
            },
            "move",
            "right_arm",
            (1, 3, 2),
        ),
        (
            {
                "kind": "atomic_action",
                "name": "pick_up",
                "cfg": {"control_part": "right_arm", "hand_control_part": "right_eef"},
                "target": {"kind": "object_semantics", "obj_name": "cup"},
            },
            "pick_up",
            "right_full",
            (1, 2, 4),
        ),
        (
            {
                "kind": "atomic_action",
                "name": "place",
                "cfg": {
                    "control_part": "left_arm",
                    "hand_control_part": "left_eef",
                    "sample_interval": 4,
                },
                "target": {"pose": torch.eye(4).tolist()},
            },
            "place",
            "left_full",
            (1, 4, 4),
        ),
        (
            {
                "kind": "atomic_action",
                "name": "gripper_open",
                "cfg": {
                    "control_part": "left_eef",
                    "arm_control_part": "left_arm",
                    "sample_interval": 5,
                },
                "target": {"kind": "gripper_state", "state": "open"},
            },
            "gripper_open",
            "left_eef",
            (1, 5, 2),
        ),
        (
            {
                "kind": "atomic_action",
                "name": "gripper_close",
                "cfg": {
                    "control_part": "right_eef",
                    "arm_control_part": "right_arm",
                    "sample_interval": 6,
                },
                "target": {"kind": "gripper_state", "state": "close"},
            },
            "gripper_close",
            "right_eef",
            (1, 6, 2),
        ),
    ],
)
def test_single_atomic_graph_action_plan_smoke(
    monkeypatch,
    spec,
    expected_action_name: str,
    expected_joint_group: str,
    expected_shape: tuple[int, int, int],
) -> None:
    env = _Env()
    if expected_action_name == "pick_up":
        target = object()
        monkeypatch.setattr(
            executor,
            "_resolve_action_target",
            lambda *args, **kwargs: target,
        )
        monkeypatch.setattr(
            executor,
            "_public_grasp_approach_direction_candidates",
            lambda *args, **kwargs: [("ranked", torch.tensor([0.0, 0.0, -1.0]))],
        )

        def _plan_public_semantic_grasp_action(**kwargs):
            return SimpleNamespace(
                trajectory=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
                joint_ids=env.right_arm_joints + env.right_eef_joints,
            )

        monkeypatch.setattr(
            executor,
            "plan_public_semantic_grasp_action",
            _plan_public_semantic_grasp_action,
        )
    elif expected_action_name in {"place", "gripper_open", "gripper_close"}:
        monkeypatch.setattr(executor, "_create_engine", _fake_engine_factory(env))

    plan = AtomicGraphAction(spec=spec).plan(env=env, require_atomic_action_graph=True)

    assert isinstance(plan, ActionPlan)
    assert plan.action_name == expected_action_name
    assert plan.joint_ids == _joint_group(env, expected_joint_group)
    assert tuple(plan.trajectory.shape) == expected_shape


def test_recovery_public_grasp_candidate_override_caps_strategy_default(
    monkeypatch,
) -> None:
    env = _Env()
    captured_kwargs: dict[str, object] = {}
    spec = {
        "kind": "atomic_action",
        "name": "pick_up",
        "cfg": {"control_part": "right_arm", "hand_control_part": "right_eef"},
        "target": {"kind": "object_semantics", "obj_name": "cup"},
    }
    monkeypatch.setattr(
        executor,
        "_resolve_action_target",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        executor,
        "_public_grasp_approach_direction_candidates",
        lambda *args, **kwargs: [("ranked", torch.tensor([0.0, 0.0, -1.0]))],
    )

    def _plan_public_semantic_grasp_action(**kwargs):
        captured_kwargs.update(kwargs["kwargs"])
        return SimpleNamespace(
            trajectory=torch.zeros(1, 2, 4),
            joint_ids=env.right_arm_joints + env.right_eef_joints,
        )

    monkeypatch.setattr(
        executor,
        "plan_public_semantic_grasp_action",
        _plan_public_semantic_grasp_action,
    )

    AtomicGraphAction(spec=spec).plan(
        env=env,
        require_atomic_action_graph=True,
        _edge_is_recovery=True,
        recovery_public_grasp_strategy="auto_general",
        recovery_public_grasp_candidate_num=16,
    )

    assert captured_kwargs["public_grasp_strategy"] == "auto_general"
    assert captured_kwargs["public_grasp_candidate_num"] == 16


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


def test_eef_orientation_down_retries_yaw_candidates(monkeypatch) -> None:
    env = _Env()
    env.left_arm_current_xpos[:3, :3] = executor._rotation_matrix_xyz_degrees(
        (0.0, 0.0, 45.0),
        device=env.robot.device,
    )
    attempts = []

    class _Engine:
        def __init__(self, cfg_list) -> None:
            self.cfg_list = cfg_list

        def execute_static(self, target_list):
            attempts.append(target_list[0].clone())
            if len(attempts) == 1:
                return False, torch.empty(1, 0, env.robot.dof)
            trajectory = torch.zeros(1, 3, env.robot.dof)
            trajectory[:, :, env.left_arm_joints] = torch.tensor(
                [[[0.0, 0.0], [0.5, 0.6], [1.0, 1.1]]],
                dtype=torch.float32,
            )
            return True, trajectory

    monkeypatch.setattr(
        executor,
        "_create_engine",
        lambda env, cfg_list: _Engine(cfg_list),
    )
    action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "left_arm", "sample_interval": 3},
            "target": {"kind": "eef_orientation", "direction": "down"},
        }
    )

    plan = action.plan(env=env, require_atomic_action_graph=True)

    assert len(attempts) == 2
    torch.testing.assert_close(
        attempts[0][:3, :3],
        executor._orientation_matrix("down", device=env.robot.device),
    )
    torch.testing.assert_close(
        attempts[1][:3, :3],
        executor._rotation_matrix_xyz_degrees(
            (0.0, 0.0, 45.0),
            device=env.robot.device,
        )
        @ executor._orientation_matrix("down", device=env.robot.device),
    )
    assert plan.joint_ids == env.left_arm_joints
    torch.testing.assert_close(plan.trajectory[0, -1], torch.tensor([1.0, 1.1]))


def test_eef_orientation_down_holds_when_current_axis_is_down(monkeypatch) -> None:
    env = _Env()
    env.left_arm_current_xpos[:3, :3] = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def _raise_create_engine(*args, **kwargs):
        raise AssertionError("current down orientation should not call planner")

    monkeypatch.setattr(executor, "_create_engine", _raise_create_engine)
    action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "left_arm", "sample_interval": 4},
            "target": {"kind": "eef_orientation", "direction": "down"},
        }
    )

    plan = action.plan(env=env, require_atomic_action_graph=True)

    assert plan.joint_ids == env.left_arm_joints
    assert tuple(plan.trajectory.shape) == (1, 4, 2)
    torch.testing.assert_close(plan.trajectory, torch.zeros(1, 4, 2))


def test_resolve_object_relative_pose_prefers_cached_obj_info_pose() -> None:
    env = _Env()
    cached_pose = torch.eye(4, dtype=torch.float32)
    cached_pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    env.obj_info = {"cup": {"pose": cached_pose}}

    resolved = executor._resolve_pose_target(
        {
            "kind": "object_relative_pose",
            "obj_name": "cup",
            "offset": [0.1, 0.2, 0.3],
        },
        env=env,
        robot_name="left_arm",
    )

    torch.testing.assert_close(
        resolved[:3, 3],
        torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32),
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


def test_pick_up_plan_uses_ranked_semantic_plan(monkeypatch) -> None:
    env = _Env()
    target = object()
    captured = {}
    trajectory = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)

    monkeypatch.setattr(
        executor,
        "_resolve_action_target",
        lambda *args, **kwargs: target,
    )
    monkeypatch.setattr(
        executor,
        "_public_grasp_approach_direction_candidates",
        lambda *args, **kwargs: [("ranked", torch.tensor([0.0, 0.0, -1.0]))],
    )

    def _plan_public_semantic_grasp_action(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            trajectory=trajectory[:, :, env.right_arm_joints + env.right_eef_joints],
            joint_ids=env.right_arm_joints + env.right_eef_joints,
        )

    monkeypatch.setattr(
        executor,
        "plan_public_semantic_grasp_action",
        _plan_public_semantic_grasp_action,
    )
    action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "pick_up",
            "cfg": {"control_part": "right_arm", "hand_control_part": "right_eef"},
            "target": {"kind": "object_semantics", "obj_name": "cup"},
        }
    )

    plan = action.plan(env=env, require_atomic_action_graph=True)

    assert plan.joint_ids == env.right_arm_joints + env.right_eef_joints
    assert captured["target"] is target
    assert captured["directions"][0][0] == "ranked"
    torch.testing.assert_close(plan.trajectory, trajectory[:, :, [4, 5, 6, 7]])


def test_atomic_graph_failure_needs_explicit_legacy_fallback(monkeypatch) -> None:
    env = _Env()
    action = AtomicGraphAction(
        spec={
            "kind": "atomic_action",
            "name": "move",
            "cfg": {"control_part": "left_arm"},
            "target": {"kind": "current_pose"},
        },
        fallback_action=lambda **kwargs: torch.ones(2, 2),
    )

    def _raise_atomic_graph(*args, **kwargs):
        raise RuntimeError("planner failed")

    monkeypatch.setattr(action, "_run_atomic_graph", _raise_atomic_graph)

    with pytest.raises(RuntimeError, match="planner failed"):
        action.plan(env=env)

    plan = action.plan(env=env, allow_legacy_atomic_action_fallback=True)

    assert plan.joint_ids == env.left_arm_joints
    torch.testing.assert_close(plan.trajectory, torch.ones(1, 2, 2))


def test_recovery_public_grasp_overrides_only_apply_to_recovery_edges() -> None:
    action = {
        "runtime_kwargs": {
            "public_grasp_strategy": "legacy_guided",
            "public_grasp_candidate_num": 32,
        }
    }
    runtime_kwargs = {
        "recovery_public_grasp_strategy": "auto_general",
        "recovery_public_grasp_candidate_num": 64,
    }

    nominal = executor._action_runtime_kwargs(action, runtime_kwargs)
    recovery = executor._action_runtime_kwargs(
        action,
        {**runtime_kwargs, "_edge_is_recovery": True},
    )

    assert nominal["public_grasp_strategy"] == "legacy_guided"
    assert nominal["public_grasp_candidate_num"] == 32
    assert recovery["public_grasp_strategy"] == "auto_general"
    assert recovery["public_grasp_candidate_num"] == 64
