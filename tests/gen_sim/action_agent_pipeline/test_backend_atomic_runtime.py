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

from embodichain.gen_sim.action_agent_pipeline.runtime import atom_actions
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
    build_parallel_action_stream,
    execute_atomic_action,
    execute_parallel_atomic_actions,
    normalize_atomic_action_spec,
    step_env_with_actions,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_action_utils import (
    resolve_arm_side,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coacd_cache_bridge import (
    GraspCollisionCachePreparationError,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.task_graph import AgentTaskGraph
from embodichain.lab.sim.atomic_actions import (
    MoveActionCfg,
    PickUpActionCfg,
    PlaceActionCfg,
)


class _FakeRobot:
    uid = "fake_robot"
    device = torch.device("cpu")
    control_parts = {
        "left_arm": [0, 1],
        "left_eef": [2],
        "right_arm": [3, 4],
        "right_eef": [5],
    }

    def get_qpos(self):
        return torch.zeros(1, 6)


class _FakeObject:
    cfg = SimpleNamespace(shape=SimpleNamespace(fpath="/tmp/fake.obj"))

    def __init__(self, xyz):
        self._pose = torch.eye(4)
        self._pose[:3, 3] = torch.tensor(xyz, dtype=torch.float32)

    def get_local_pose(self, to_matrix: bool = True):
        return self._pose.unsqueeze(0)

    def get_vertices(self, env_ids=None, scale: bool = True):
        return [torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])]

    def get_triangles(self, env_ids=None):
        return [torch.tensor([[0, 1, 2]])]

    def get_body_scale(self, env_ids=None):
        return torch.ones(1, 3)


class _FakeSim:
    def __init__(self):
        self.objects = {"apple": _FakeObject([0.4, -0.2, 0.1])}

    def get_rigid_object(self, uid: str):
        return self.objects.get(uid)


class _FakeEnv:
    def __init__(self):
        self.robot = _FakeRobot()
        self.sim = _FakeSim()
        self.left_arm_joints = [0, 1]
        self.left_eef_joints = [2]
        self.right_arm_joints = [3, 4]
        self.right_eef_joints = [5]
        self.left_arm_current_qpos = torch.tensor([0.1, 0.2])
        self.right_arm_current_qpos = torch.tensor([0.3, 0.4])
        self.left_arm_init_qpos = torch.tensor([-0.1, -0.2])
        self.right_arm_init_qpos = torch.tensor([-0.3, -0.4])
        self.left_arm_current_xpos = torch.eye(4)
        self.right_arm_current_xpos = torch.eye(4)
        self.left_arm_current_gripper_state = torch.tensor([0.0])
        self.right_arm_current_gripper_state = torch.tensor([0.0])
        self.open_state = torch.tensor([0.05])
        self.close_state = torch.tensor([0.0])

    def get_current_qpos_agent(self):
        return self.left_arm_current_qpos, self.right_arm_current_qpos

    def set_current_qpos_agent(self, arm_qpos, is_left):
        if is_left:
            self.left_arm_current_qpos = arm_qpos
        else:
            self.right_arm_current_qpos = arm_qpos

    def get_current_xpos_agent(self):
        return self.left_arm_current_xpos, self.right_arm_current_xpos

    def set_current_xpos_agent(self, arm_xpos, is_left):
        if is_left:
            self.left_arm_current_xpos = arm_xpos
        else:
            self.right_arm_current_xpos = arm_xpos

    def get_current_gripper_state_agent(self):
        return self.left_arm_current_gripper_state, self.right_arm_current_gripper_state

    def set_current_gripper_state_agent(self, arm_gripper_state, is_left):
        if is_left:
            self.left_arm_current_gripper_state = arm_gripper_state
        else:
            self.right_arm_current_gripper_state = arm_gripper_state

    def get_arm_fk(self, qpos, is_left):
        pose = torch.eye(4)
        pose[0, 3] = torch.as_tensor(qpos).flatten()[0]
        return pose


class _FakeBackendAction:
    capture: list | None = None

    def __init__(self, motion_generator, cfg):
        self.motion_generator = motion_generator
        self.cfg = cfg
        if self.capture is not None:
            self.capture.append(
                {
                    "cfg": self.cfg,
                    "motion_generator": self.motion_generator,
                }
            )

    def execute(self, target, start_qpos=None, **kwargs):
        if self.capture is not None:
            self.capture[-1].update({"target": target, "start_qpos": start_qpos})
        if self.cfg.name in {"pick_up", "place"}:
            trajectory = torch.tensor(
                [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]], dtype=torch.float32
            )
            return (
                True,
                trajectory,
                [0, 1, 2] if "left" in self.cfg.control_part else [3, 4, 5],
            )
        if self.cfg.control_part.endswith("eef"):
            trajectory = torch.tensor([[[0.0], [0.05]]], dtype=torch.float32)
            return True, trajectory, [2 if "left" in self.cfg.control_part else 5]
        trajectory = torch.tensor([[[0.1, 0.2], [0.2, 0.3]]], dtype=torch.float32)
        return True, trajectory, [0, 1] if "left" in self.cfg.control_part else [3, 4]


@pytest.fixture(autouse=True)
def _reset_fake_backend_capture():
    _FakeBackendAction.capture = None
    yield
    _FakeBackendAction.capture = None


def test_normalize_atomic_action_spec_rejects_legacy_schema() -> None:
    with pytest.raises(ValueError, match="Legacy action schema"):
        normalize_atomic_action_spec({"action": "move", "robot_name": "left_arm"})


def test_normalize_atomic_action_spec_rejects_legacy_target_kind_schema() -> None:
    with pytest.raises(ValueError, match="Legacy target.kind schema"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target": {"kind": "pose_relative_to_object", "obj_name": "apple"},
                "cfg": {},
            }
        )


def test_normalize_atomic_action_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported atomic action spec fields"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target_qpos": {"source": "initial"},
                "cfg": {},
                "description": "return home",
            }
        )


def test_normalize_atomic_action_spec_rejects_multiple_target_fields() -> None:
    with pytest.raises(ValueError, match="exactly one of target_object"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target_pose": {
                    "reference": "relative",
                    "offset": [0.0, 0.0, 0.1],
                },
                "target_qpos": {"source": "initial"},
                "cfg": {},
            }
        )


def test_normalize_atomic_action_spec_rejects_orientation_field() -> None:
    with pytest.raises(ValueError, match="Unsupported target_pose fields"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target_pose": {
                    "reference": "object",
                    "obj_name": "apple",
                    "offset": [0.0, 0.0, 0.1],
                    "orientation": "current",
                },
                "cfg": {},
            }
        )


def test_normalize_atomic_action_spec_rejects_pickup_pose_target() -> None:
    with pytest.raises(ValueError, match="PickUpAction requires control='arm'"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "PickUpAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target_pose": {
                    "reference": "relative",
                    "offset": [0.0, 0.0, 0.1],
                },
                "cfg": {},
            }
        )


def test_atom_actions_module_exposes_atomic_runtime_entrypoints() -> None:
    assert atom_actions.execute_atomic_action is execute_atomic_action
    assert atom_actions.normalize_atomic_action_spec is normalize_atomic_action_spec
    assert callable(atom_actions.execute_parallel_atomic_actions)


def test_execute_parallel_atomic_actions_requires_env() -> None:
    with pytest.raises(TypeError, match="env"):
        execute_parallel_atomic_actions(left_arm_action=torch.zeros((1, 3)))


def test_execute_parallel_atomic_actions_rejects_none_env() -> None:
    with pytest.raises(ValueError, match="env is required"):
        execute_parallel_atomic_actions(
            left_arm_action=torch.zeros((1, 3)),
            env=None,
        )


def test_build_parallel_action_stream_does_not_step_env() -> None:
    env = _FakeEnv()
    action_stream = build_parallel_action_stream(
        left_arm_action=torch.zeros((2, 3)),
        env=env,
    )

    assert len(action_stream) == 2
    assert not hasattr(env, "stepped_actions")


def test_step_env_with_actions_steps_and_updates_env() -> None:
    class _StepEnv:
        def __init__(self) -> None:
            self.stepped_actions = []
            self.update_count = 0

        def step(self, action):
            self.stepped_actions.append(action)

        def update_obj_info(self) -> None:
            self.update_count += 1

    env = _StepEnv()
    actions = [torch.zeros(1, 1), torch.ones(1, 1)]

    step_env_with_actions(env, actions)

    assert env.stepped_actions == actions
    assert env.update_count == 2


def test_agent_task_graph_run_requires_env() -> None:
    graph = AgentTaskGraph(start="start", goal="goal")
    with pytest.raises(TypeError, match="env"):
        graph.run()


def test_agent_task_graph_run_rejects_none_env() -> None:
    graph = AgentTaskGraph(start="start", goal="goal")
    with pytest.raises(ValueError, match="env is required"):
        graph.run(env=None)


def test_resolve_arm_side_rejects_unavailable_requested_arm() -> None:
    env = _FakeEnv()
    env.right_arm_joints = []
    env.right_eef_joints = []
    env.robot.control_parts = {"left_arm": [0, 1], "left_eef": [2]}

    with pytest.raises(ValueError, match="Requested right_arm"):
        resolve_arm_side(env, "right_arm")


def test_resolve_arm_side_uses_only_available_arm_for_unspecified_name() -> None:
    env = _FakeEnv()
    env.left_arm_joints = []
    env.left_eef_joints = []
    env.robot.control_parts = {"right_arm": [3, 4], "right_eef": [5]}

    assert resolve_arm_side(env, "ur5") == "right"


def test_resolve_arm_side_rejects_env_without_available_arms() -> None:
    env = _FakeEnv()
    env.left_arm_joints = []
    env.left_eef_joints = []
    env.right_arm_joints = []
    env.right_eef_joints = []
    env.robot.control_parts = {}

    with pytest.raises(ValueError, match="No available arm control parts"):
        resolve_arm_side(env, "ur5")


def test_object_referenced_pose_builds_move_cfg_and_pose_target(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture

    monkeypatch.setattr(
        atom_actions,
        "_make_motion_generator",
        lambda env: SimpleNamespace(robot=env.robot, device=env.robot.device),
    )
    monkeypatch.setattr(
        atom_actions,
        "_get_atomic_action_class",
        lambda atomic_action_class: _FakeBackendAction,
    )

    action = execute_atomic_action(
        {
            "atomic_action_class": "MoveAction",
            "robot_name": "left_arm",
            "control": "arm",
            "target_pose": {
                "reference": "object",
                "obj_name": "apple",
                "offset": [0.1, 0.2, 0.3],
            },
            "cfg": {"sample_interval": 12},
        },
        env=env,
    )

    assert action.shape == (2, 3)
    assert isinstance(capture[0]["cfg"], MoveActionCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].sample_interval == 12
    assert capture[0]["target"][:3, 3].tolist() == pytest.approx([0.5, 0.0, 0.4])


def test_gripper_state_qpos_target_interpolates_hand_action(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture

    monkeypatch.setattr(
        atom_actions,
        "_make_motion_generator",
        lambda env: SimpleNamespace(robot=env.robot, device=env.robot.device),
    )
    monkeypatch.setattr(
        atom_actions,
        "_get_atomic_action_class",
        lambda atomic_action_class: _FakeBackendAction,
    )

    action = execute_atomic_action(
        {
            "atomic_action_class": "MoveAction",
            "robot_name": "left_arm",
            "control": "hand",
            "target_qpos": {"source": "gripper_state", "state": "open"},
            "cfg": {"sample_interval": 5, "post_hold_steps": 2},
        },
        env=env,
    )

    assert action.shape == (7, 3)
    assert capture == []
    assert action[0].tolist() == pytest.approx([0.1, 0.2, 0.0])
    assert action[4].tolist() == pytest.approx([0.1, 0.2, 0.05])
    assert action[-1].tolist() == pytest.approx([0.1, 0.2, 0.05])
    assert env.left_arm_current_gripper_state.tolist() == pytest.approx([0.05])


def test_initial_qpos_target_interpolates_arm_action(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture

    monkeypatch.setattr(
        atom_actions,
        "_make_motion_generator",
        lambda env: SimpleNamespace(robot=env.robot, device=env.robot.device),
    )
    monkeypatch.setattr(
        atom_actions,
        "_get_atomic_action_class",
        lambda atomic_action_class: _FakeBackendAction,
    )

    action = execute_atomic_action(
        {
            "atomic_action_class": "MoveAction",
            "robot_name": "right_arm",
            "control": "arm",
            "target_qpos": {"source": "initial"},
            "cfg": {"sample_interval": 4},
        },
        env=env,
    )

    assert action.shape == (4, 3)
    assert capture == []
    assert action[0].tolist() == pytest.approx([0.3, 0.4, 0.0])
    assert action[-1].tolist() == pytest.approx([-0.3, -0.4, 0.0])
    assert env.right_arm_current_qpos.tolist() == pytest.approx([-0.3, -0.4])


def test_target_object_builds_pick_up_cfg(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture

    monkeypatch.setattr(
        atom_actions,
        "_make_motion_generator",
        lambda env: SimpleNamespace(robot=env.robot, device=env.robot.device),
    )
    monkeypatch.setattr(
        atom_actions,
        "_get_atomic_action_class",
        lambda atomic_action_class: _FakeBackendAction,
    )

    execute_atomic_action(
        {
            "atomic_action_class": "PickUpAction",
            "robot_name": "left_arm",
            "control": "arm",
            "target_object": {
                "obj_name": "apple",
                "affordance": "antipodal",
            },
            "cfg": {
                "pre_grasp_distance": 0.07,
                "sample_interval": 11,
            },
        },
        env=env,
        allow_grasp_annotation=True,
    )

    assert isinstance(capture[0]["cfg"], PickUpActionCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].hand_control_part == "left_eef"
    assert capture[0]["cfg"].pre_grasp_distance == pytest.approx(0.07)
    assert capture[0]["target"].label == "apple"


def test_place_action_builds_place_cfg(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture

    monkeypatch.setattr(
        atom_actions,
        "_make_motion_generator",
        lambda env: SimpleNamespace(robot=env.robot, device=env.robot.device),
    )
    monkeypatch.setattr(
        atom_actions,
        "_get_atomic_action_class",
        lambda atomic_action_class: _FakeBackendAction,
    )

    action = execute_atomic_action(
        {
            "atomic_action_class": "PlaceAction",
            "robot_name": "left_arm",
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.1],
                "frame": "world",
            },
            "cfg": {"sample_interval": 19, "lift_height": 0.06},
        },
        env=env,
    )

    assert action.shape == (2, 3)
    assert isinstance(capture[0]["cfg"], PlaceActionCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].lift_height == pytest.approx(0.06)


def test_place_action_rejects_qpos_target(monkeypatch) -> None:
    env = _FakeEnv()
    monkeypatch.setattr(
        atom_actions,
        "_make_motion_generator",
        lambda env: SimpleNamespace(robot=env.robot, device=env.robot.device),
    )
    monkeypatch.setattr(
        atom_actions,
        "_get_atomic_action_class",
        lambda atomic_action_class: _FakeBackendAction,
    )

    with pytest.raises(
        ValueError,
        match="PlaceAction requires control='arm' and target_pose",
    ):
        execute_atomic_action(
            {
                "atomic_action_class": "PlaceAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target_qpos": {"source": "initial"},
                "cfg": {"sample_interval": 20},
            },
            env=env,
        )


def test_grasp_collision_cache_bridge_error_falls_back(monkeypatch) -> None:
    warnings = []

    def raise_cache_error(**kwargs):
        raise GraspCollisionCachePreparationError("cache conversion failed")

    monkeypatch.setattr(
        atom_actions,
        "ensure_grasp_collision_cache_from_env_coacd",
        raise_cache_error,
    )
    monkeypatch.setattr(atom_actions, "log_warning", warnings.append)

    atom_actions._prepare_grasp_collision_cache_from_env_coacd(
        obj_name="apple",
        mesh_vertices=torch.zeros(1, 3),
        mesh_triangles=torch.zeros(1, 3, dtype=torch.int64),
        source_mesh_path="/tmp/fake.obj",
        max_decomposition_hulls=4,
        body_scale=None,
        runtime_kwargs={},
    )

    assert len(warnings) == 1
    assert "falling back to the default grasp collision path" in warnings[0]
    assert "cache conversion failed" in warnings[0]


def test_grasp_collision_cache_unexpected_error_propagates(monkeypatch) -> None:
    def raise_unexpected_error(**kwargs):
        raise AssertionError("unexpected bug")

    monkeypatch.setattr(
        atom_actions,
        "ensure_grasp_collision_cache_from_env_coacd",
        raise_unexpected_error,
    )

    with pytest.raises(AssertionError, match="unexpected bug"):
        atom_actions._prepare_grasp_collision_cache_from_env_coacd(
            obj_name="apple",
            mesh_vertices=torch.zeros(1, 3),
            mesh_triangles=torch.zeros(1, 3, dtype=torch.int64),
            source_mesh_path="/tmp/fake.obj",
            max_decomposition_hulls=4,
            body_scale=None,
            runtime_kwargs={},
        )
