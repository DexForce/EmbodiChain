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

from collections.abc import Sequence
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
from embodichain.gen_sim.action_agent_pipeline.runtime.task_graph import (
    AgentTaskGraph,
    ExecutedActionList,
)
from embodichain.lab.sim.atomic_actions import (
    ActionResult,
    CoordinatedHeldObjectState,
    CoordinatedPickmentCfg,
    CoordinatedPickmentTarget,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
    HeldObjectState,
    JointPositionTarget,
    MoveEndEffectorCfg,
    MoveHeldObjectCfg,
    MoveJointsCfg,
    PickUpCfg,
    PlaceCfg,
    WorldState,
)


class _FakeRobot:
    uid = "fake_robot"
    device = torch.device("cpu")
    dof = 6

    def __init__(self):
        self.control_parts = {
            "left_arm": [0, 1],
            "left_eef": [2],
            "right_arm": [3, 4],
            "right_eef": [5],
        }
        self._joint_ids = {
            name: list(joint_ids) for name, joint_ids in self.control_parts.items()
        }

    def get_qpos(self):
        return torch.zeros(1, 6)

    def get_joint_ids(self, name: str):
        return list(self._joint_ids[name])


class _FakeObject:
    cfg = SimpleNamespace(shape=SimpleNamespace(fpath="/tmp/fake.obj"))

    def __init__(self, xyz, *, yaw_degrees: float = 0.0, extents=(0.3, 0.1, 0.05)):
        self._pose = torch.eye(4)
        self._pose[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
        yaw = torch.deg2rad(torch.tensor(float(yaw_degrees)))
        self._pose[0, 0] = torch.cos(yaw)
        self._pose[0, 1] = -torch.sin(yaw)
        self._pose[1, 0] = torch.sin(yaw)
        self._pose[1, 1] = torch.cos(yaw)
        self._extents = torch.tensor(extents, dtype=torch.float32)

    def get_local_pose(self, to_matrix: bool = True):
        return self._pose.unsqueeze(0)

    def get_vertices(self, env_ids=None, scale: bool = True):
        x, y, z = self._extents.tolist()
        return [
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [x, 0.0, 0.0],
                    [0.0, y, 0.0],
                    [0.0, 0.0, z],
                ]
            )
        ]

    def get_triangles(self, env_ids=None):
        return [torch.tensor([[0, 1, 2], [0, 1, 3]])]

    def get_body_scale(self, env_ids=None):
        return torch.ones(1, 3)


class _FakeSim:
    def __init__(self):
        self.objects = {
            "apple": _FakeObject([0.4, -0.2, 0.1]),
            "pad_x": _FakeObject([0.4, 0.0, 0.0], extents=(0.4, 0.1, 0.02)),
            "pad_y": _FakeObject(
                [0.4, 0.0, 0.0],
                yaw_degrees=90.0,
                extents=(0.4, 0.1, 0.02),
            ),
        }

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
        self.stepped_actions = []
        self.update_count = 0

    def step(self, action):
        self.stepped_actions.append(action)

    def update_obj_info(self) -> None:
        self.update_count += 1

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

    def execute(self, target, state, **kwargs):
        if self.capture is not None:
            self.capture[-1].update({"target": target, "state": state})
        if self.cfg.name == "coordinated_pickment":
            trajectory = torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.0, 0.3, 0.4, 0.0],
                        [0.4, 0.5, 0.04, 0.6, 0.7, 0.04],
                    ]
                ],
                dtype=torch.float32,
            )
            coordinated = CoordinatedHeldObjectState(
                semantics=target.object_semantics,
                left_object_to_eef=target.left_object_to_eef.unsqueeze(0),
                right_object_to_eef=target.right_object_to_eef.unsqueeze(0),
                left_grasp_xpos=torch.eye(4).unsqueeze(0),
                right_grasp_xpos=torch.eye(4).unsqueeze(0),
            )
            return ActionResult(
                success=True,
                trajectory=trajectory,
                next_state=WorldState(
                    last_qpos=trajectory[:, -1, :],
                    coordinated_held_object=coordinated,
                ),
            )
        if self.cfg.name == "move_joints":
            joint_ids = self.motion_generator.robot.get_joint_ids(self.cfg.control_part)
            trajectory = state.last_qpos.unsqueeze(1).repeat(1, 2, 1)
            trajectory[:, -1, joint_ids] = target.qpos.reshape(1, -1)
            return ActionResult(
                success=True,
                trajectory=trajectory,
                next_state=WorldState(
                    last_qpos=trajectory[:, -1, :],
                    held_object=state.held_object,
                ),
            )
        if self.cfg.name == "move_held_object":
            trajectory = torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                        [0.25, 0.35, 0.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
            return ActionResult(
                success=True,
                trajectory=trajectory,
                next_state=WorldState(
                    last_qpos=trajectory[:, -1, :],
                    held_object=state.held_object,
                ),
            )
        if self.cfg.name == "pick_up":
            trajectory = torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                        [0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
            held = HeldObjectState(
                semantics=target.semantics,
                object_to_eef=torch.eye(4).unsqueeze(0),
                grasp_xpos=torch.eye(4).unsqueeze(0),
            )
            return ActionResult(
                success=True,
                trajectory=trajectory,
                next_state=WorldState(last_qpos=trajectory[:, -1, :], held_object=held),
            )
        if self.cfg.name == "place":
            trajectory = torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                        [0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
            return ActionResult(
                success=True,
                trajectory=trajectory,
                next_state=WorldState(last_qpos=trajectory[:, -1, :]),
            )
        if self.cfg.control_part.endswith("eef"):
            trajectory = torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.05, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
            return ActionResult(
                success=True,
                trajectory=trajectory,
                next_state=WorldState(last_qpos=trajectory[:, -1, :]),
            )
        trajectory = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        return ActionResult(
            success=True,
            trajectory=trajectory,
            next_state=WorldState(last_qpos=trajectory[:, -1, :]),
        )


@pytest.fixture(autouse=True)
def _reset_fake_backend_capture():
    _FakeBackendAction.capture = None
    yield
    _FakeBackendAction.capture = None


def test_normalize_atomic_action_spec_rejects_legacy_schema() -> None:
    with pytest.raises(ValueError, match="Legacy action schema"):
        normalize_atomic_action_spec({"action": "move", "robot_name": "left_arm"})


def test_normalize_atomic_action_spec_rejects_old_action_names() -> None:
    with pytest.raises(ValueError, match="Unsupported atomic action class"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveAction",
                "robot_name": "left_arm",
                "control": "arm",
                "target_pose": {
                    "reference": "relative",
                    "offset": [0.0, 0.0, 0.1],
                },
                "cfg": {},
            }
        )


def test_normalize_atomic_action_spec_rejects_legacy_target_kind_schema() -> None:
    with pytest.raises(ValueError, match="Legacy target.kind schema"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveEndEffector",
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
                "atomic_action_class": "MoveJoints",
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
                "atomic_action_class": "MoveJoints",
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


def test_normalize_atomic_action_spec_accepts_coordinated_pickment_targets() -> None:
    normalized = normalize_atomic_action_spec(
        {
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": "dual_arm",
            "control": "arm",
            "target_object": {
                "obj_name": "apple",
                "affordance": "antipodal",
            },
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.16, 0.0, 0.0],
                "frame": "world",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": {"sample_interval": 120, "hand_interp_steps": 10},
        }
    )

    assert normalized["atomic_action_class"] == "CoordinatedPickment"
    assert normalized["target_object"]["obj_name"] == "apple"
    assert normalized["target_object_pose"]["offset"] == [0.16, 0.0, 0.0]


def test_pickup_upright_cfg_is_normalized_for_typed_cfg() -> None:
    rotate_upright = 0.7853981633974483
    normalized = normalize_atomic_action_spec(
        {
            "atomic_action_class": "PickUp",
            "robot_name": "left_arm",
            "control": "arm",
            "target_object": {
                "obj_name": "apple",
                "affordance": "antipodal",
            },
            "cfg": {
                "sample_interval": 45,
                "obj_upright_direction": [1.0, 0.0, 0.0],
                "rotate_upright": rotate_upright,
            },
        }
    )

    spec = atom_actions.AtomicActionSpec.from_normalized(normalized)
    cfg = atom_actions._build_action_cfg(
        _FakeEnv(),
        spec,
        arm_part="left_arm",
        hand_part="left_eef",
        hand_dof=1,
    )

    assert isinstance(cfg, PickUpCfg)
    assert torch.allclose(
        cfg.obj_upright_direction,
        torch.tensor([1.0, 0.0, 0.0]),
    )
    assert cfg.rotate_upright == pytest.approx(rotate_upright)


def test_pickup_upright_cfg_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="obj_upright_direction"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "PickUp",
                "robot_name": "left_arm",
                "control": "arm",
                "target_object": {
                    "obj_name": "apple",
                    "affordance": "antipodal",
                },
                "cfg": {
                    "obj_upright_direction": [1.0, 0.0],
                    "rotate_upright": 0.7853981633974483,
                },
            }
        )


def test_normalize_atomic_action_spec_rejects_orientation_field() -> None:
    with pytest.raises(ValueError, match="Unsupported target_pose fields"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveEndEffector",
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


def test_move_held_object_defaults_orientation_axis_to_none() -> None:
    normalized = normalize_atomic_action_spec(
        {
            "atomic_action_class": "MoveHeldObject",
            "robot_name": "left_arm",
            "control": "arm",
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.1],
                "frame": "world",
            },
            "cfg": {},
        }
    )

    target = normalized["target_object_pose"]
    assert target.get("orientation_goal", "preserve") == "preserve"
    assert target.get("orientation_axis", "none") == "none"


def test_move_held_object_rejects_legacy_horizontal_orientation_goal() -> None:
    with pytest.raises(ValueError, match="orientation_goal"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveHeldObject",
                "robot_name": "left_arm",
                "control": "arm",
                "target_object_pose": {
                    "reference": "relative",
                    "offset": [0.0, 0.0, 0.1],
                    "frame": "world",
                    "orientation_goal": "horizontal",
                },
                "cfg": {},
            }
        )


def test_move_held_object_rejects_invalid_axis_alignment_pairings() -> None:
    base_spec = {
        "atomic_action_class": "MoveHeldObject",
        "robot_name": "left_arm",
        "control": "arm",
        "target_object_pose": {
            "reference": "relative",
            "offset": [0.0, 0.0, 0.1],
            "frame": "world",
            "orientation_goal": "axis_align",
        },
        "cfg": {},
    }

    with pytest.raises(ValueError, match="without align_to"):
        normalize_atomic_action_spec(
            {
                **base_spec,
                "target_object_pose": {
                    **base_spec["target_object_pose"],
                    "orientation_axis": "long_axis",
                },
            }
        )
    with pytest.raises(ValueError, match="with align_to"):
        normalize_atomic_action_spec(
            {
                **base_spec,
                "target_object_pose": {
                    **base_spec["target_object_pose"],
                    "orientation_axis": "x",
                    "align_to": "pad_x",
                },
            }
        )


def test_move_held_object_rejects_axis_for_non_axis_align_goals() -> None:
    with pytest.raises(ValueError, match="orientation_axis='none'"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveHeldObject",
                "robot_name": "left_arm",
                "control": "arm",
                "target_object_pose": {
                    "reference": "relative",
                    "offset": [0.0, 0.0, 0.1],
                    "frame": "world",
                    "orientation_goal": "lay_flat",
                    "orientation_axis": "x",
                },
                "cfg": {},
            }
        )


def test_normalize_atomic_action_spec_rejects_pickup_pose_target() -> None:
    with pytest.raises(ValueError, match="PickUp requires control='arm'"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "PickUp",
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
    assert env.stepped_actions == []


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


def test_executed_action_list_is_sequence() -> None:
    actions = [torch.zeros(1, 2), torch.ones(1, 2)]
    action_list = ExecutedActionList(actions)

    assert isinstance(action_list, Sequence)
    assert action_list.already_executed
    assert len(action_list) == 2
    assert action_list[1] is actions[1]
    assert list(action_list) == actions


def test_agent_task_graph_threads_world_state_between_edges(monkeypatch) -> None:
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

    graph = AgentTaskGraph(start="v0", goal="v3")
    graph.add_node("v0").add_node("v1").add_node("v2").add_node("v3")
    graph.add_edge(
        "e01",
        "v0",
        "v1",
        left_arm_action={
            "atomic_action_class": "PickUp",
            "robot_name": "left_arm",
            "control": "arm",
            "target_object": {"obj_name": "apple", "affordance": "antipodal"},
            "cfg": {},
        },
    )
    graph.add_edge(
        "e12",
        "v1",
        "v2",
        left_arm_action={
            "atomic_action_class": "MoveHeldObject",
            "robot_name": "left_arm",
            "control": "arm",
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.1],
                "frame": "world",
                "orientation_goal": "preserve",
            },
            "cfg": {},
        },
    )
    graph.add_edge(
        "e23",
        "v2",
        "v3",
        left_arm_action={
            "atomic_action_class": "Place",
            "robot_name": "left_arm",
            "control": "arm",
            "target_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, -0.1],
                "frame": "world",
            },
            "cfg": {},
        },
    )

    actions = graph.run(env=env, allow_grasp_annotation=True)

    assert isinstance(actions, ExecutedActionList)
    assert len(capture) == 3
    assert capture[0]["state"].held_object is None
    assert capture[1]["state"].held_object is not None
    assert capture[2]["state"].held_object is not None


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
            "atomic_action_class": "MoveEndEffector",
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
    assert isinstance(capture[0]["cfg"], MoveEndEffectorCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].sample_interval == 12
    assert isinstance(capture[0]["target"], EndEffectorPoseTarget)
    assert capture[0]["target"].xpos[:3, 3].tolist() == pytest.approx([0.5, 0.0, 0.4])


def test_gripper_state_qpos_target_uses_move_joints(monkeypatch) -> None:
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
            "atomic_action_class": "MoveJoints",
            "robot_name": "left_arm",
            "control": "hand",
            "target_qpos": {"source": "gripper_state", "state": "open"},
            "cfg": {"sample_interval": 5, "post_hold_steps": 2},
        },
        env=env,
    )

    assert action.shape == (4, 3)
    assert isinstance(capture[0]["cfg"], MoveJointsCfg)
    assert capture[0]["cfg"].control_part == "left_eef"
    assert isinstance(capture[0]["target"], JointPositionTarget)
    assert capture[0]["target"].qpos.tolist() == pytest.approx([0.05])
    assert action[0].tolist() == pytest.approx([0.1, 0.2, 0.0])
    assert action[1].tolist() == pytest.approx([0.1, 0.2, 0.05])
    assert action[-1].tolist() == pytest.approx([0.1, 0.2, 0.05])
    assert env.left_arm_current_gripper_state.tolist() == pytest.approx([0.05])


def test_initial_qpos_target_uses_move_joints(monkeypatch) -> None:
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
            "atomic_action_class": "MoveJoints",
            "robot_name": "right_arm",
            "control": "arm",
            "target_qpos": {"source": "initial"},
            "cfg": {"sample_interval": 4},
        },
        env=env,
    )

    assert action.shape == (2, 3)
    assert isinstance(capture[0]["cfg"], MoveJointsCfg)
    assert capture[0]["cfg"].control_part == "right_arm"
    assert isinstance(capture[0]["target"], JointPositionTarget)
    assert capture[0]["target"].qpos.tolist() == pytest.approx([-0.3, -0.4])
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
            "atomic_action_class": "PickUp",
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

    assert isinstance(capture[0]["cfg"], PickUpCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].hand_control_part == "left_eef"
    assert capture[0]["cfg"].pre_grasp_distance == pytest.approx(0.07)
    assert isinstance(capture[0]["target"], GraspTarget)
    assert capture[0]["target"].semantics.label == "apple"
    assert capture[0]["target"].semantics.affordance.mesh_vertices is not None
    assert capture[0]["target"].semantics.affordance.mesh_triangles is not None


def test_coordinated_pickment_builds_full_robot_stream(monkeypatch) -> None:
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

    result = execute_parallel_atomic_actions(
        left_arm_action={
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": "dual_arm",
            "control": "arm",
            "target_object": {
                "obj_name": "apple",
                "affordance": "antipodal",
            },
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.16, 0.0, 0.0],
                "frame": "world",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": {"sample_interval": 120, "hand_interp_steps": 10},
        },
        right_arm_action=None,
        env=env,
        return_result=True,
    )

    assert len(result["actions"]) == 2
    assert env.stepped_actions[-1].shape == (1, 6)
    assert isinstance(capture[0]["cfg"], CoordinatedPickmentCfg)
    assert capture[0]["cfg"].control_part == "dual_arm"
    assert env.robot.control_parts["dual_arm"] == [0, 1, 3, 4]
    assert env.robot._joint_ids["dual_arm"] == [0, 1, 3, 4]
    assert capture[0]["cfg"].left_arm_control_part == "left_arm"
    assert capture[0]["cfg"].right_arm_control_part == "right_arm"
    assert isinstance(capture[0]["target"], CoordinatedPickmentTarget)
    assert capture[0]["target"].object_target_pose[:3, 3].tolist() == pytest.approx(
        [0.56, -0.2, 0.1]
    )
    assert capture[0]["target"].left_object_to_eef[:3, 2].tolist() == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    assert capture[0]["target"].right_object_to_eef[:3, 2].tolist() == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    assert env.left_arm_current_qpos.tolist() == pytest.approx([0.4, 0.5])
    assert env.right_arm_current_qpos.tolist() == pytest.approx([0.6, 0.7])
    assert env.left_arm_current_gripper_state.tolist() == pytest.approx([0.04])
    assert env.right_arm_current_gripper_state.tolist() == pytest.approx([0.04])
    assert result["world_states"]["coordinated"].coordinated_held_object is not None


def test_coordinated_pickment_prefers_yawed_top_down_grasps(
    monkeypatch,
) -> None:
    env = _FakeEnv()
    env.left_arm_current_xpos = torch.eye(4)
    env.left_arm_current_xpos[:3, 3] = torch.tensor([0.4, 0.3, 0.04])
    env.right_arm_current_xpos = torch.eye(4)
    env.right_arm_current_xpos[:3, 3] = torch.tensor([0.3, -0.3, 0.04])
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

    execute_parallel_atomic_actions(
        left_arm_action={
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": "dual_arm",
            "control": "arm",
            "target_object": {
                "obj_name": "pad_y",
                "affordance": "antipodal",
            },
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.0],
                "frame": "world",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": {"sample_interval": 120, "hand_interp_steps": 10},
        },
        right_arm_action=None,
        env=env,
        return_result=True,
    )

    target = capture[0]["target"]
    assert isinstance(target, CoordinatedPickmentTarget)
    obj_pose = env.sim.get_rigid_object("pad_y").get_local_pose(to_matrix=True)[0]
    left_world = obj_pose @ target.left_object_to_eef
    right_world = obj_pose @ target.right_object_to_eef
    assert left_world[:3, 2].tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert right_world[:3, 2].tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert abs(left_world[1, 3].item() - right_world[1, 3].item()) > 0.25


def test_coordinated_pickment_skips_ik_failed_grasp_candidate(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture

    def fake_get_arm_ik(target_xpos, is_left, qpos_seed=None):
        # Reject the compact world-Y candidate and force the selector to try
        # the next top-down endpoint candidate.
        if abs(float(target_xpos[0, 3]) - 0.6) < 0.03:
            return False, torch.zeros(2)
        return True, torch.tensor([0.2, 0.3]) if is_left else torch.tensor([0.4, 0.5])

    env.get_arm_ik = fake_get_arm_ik
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

    execute_parallel_atomic_actions(
        left_arm_action={
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": "dual_arm",
            "control": "arm",
            "target_object": {
                "obj_name": "pad_x",
                "affordance": "antipodal",
            },
            "target_object_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.0],
                "frame": "world",
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": {"sample_interval": 120, "hand_interp_steps": 10},
        },
        right_arm_action=None,
        env=env,
        return_result=True,
    )

    target = capture[0]["target"]
    left_local = target.left_object_to_eef[:3, 3]
    right_local = target.right_object_to_eef[:3, 3]
    assert {round(float(left_local[0]), 3), round(float(right_local[0]), 3)} == {
        0.032,
        0.368,
    }


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
            "atomic_action_class": "Place",
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
    assert isinstance(capture[0]["cfg"], PlaceCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].lift_height == pytest.approx(0.06)


def _held_state_with_yaw(
    env: _FakeEnv,
    yaw_degrees: float,
    *,
    mesh_extents=(0.3, 0.1, 0.05),
) -> WorldState:
    yaw = torch.deg2rad(torch.tensor(float(yaw_degrees)))
    current_pose = torch.eye(4)
    current_pose[0, 0] = torch.cos(yaw)
    current_pose[0, 1] = -torch.sin(yaw)
    current_pose[1, 0] = torch.sin(yaw)
    current_pose[1, 1] = torch.cos(yaw)
    semantics = atom_actions._build_object_semantics(
        env,
        {"obj_name": "apple", "affordance": "antipodal"},
        {"allow_grasp_annotation": True},
    )
    x, y, z = mesh_extents
    semantics.geometry["mesh_vertices"] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [x, 0.0, 0.0],
            [0.0, y, 0.0],
            [0.0, 0.0, z],
        ],
        dtype=torch.float32,
    )
    semantics.entity = None
    return WorldState(
        last_qpos=env.robot.get_qpos().clone(),
        held_object=HeldObjectState(
            semantics=semantics,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=current_pose.unsqueeze(0),
        ),
    )


def _resolved_held_object_direction(
    env: _FakeEnv,
    state: WorldState,
    target_object_pose: dict,
) -> torch.Tensor:
    target = atom_actions._resolve_held_object_pose_target(
        env,
        atom_actions.AtomicActionSpec(
            atomic_action_class="MoveHeldObject",
            robot_name="left_arm",
            control="arm",
            target_object_pose=target_object_pose,
            cfg={},
        ),
        state,
    )
    direction = target[:3, 0].clone()
    direction[2] = 0.0
    return direction / torch.linalg.norm(direction)


def test_axis_align_world_axes_preserves_roll_pitch_and_aligns_x_axis() -> None:
    env = _FakeEnv()
    state = _held_state_with_yaw(env, 37.0)

    direction = _resolved_held_object_direction(
        env,
        state,
        {
            "reference": "relative",
            "offset": [0.0, 0.0, 0.1],
            "frame": "world",
            "orientation_goal": "axis_align",
            "orientation_axis": "x",
        },
    )

    assert abs(float(torch.dot(direction, torch.tensor([1.0, 0.0, 0.0])))) == (
        pytest.approx(1.0)
    )


def test_axis_align_world_axes_aligns_y_axis() -> None:
    env = _FakeEnv()
    state = _held_state_with_yaw(env, 12.0)

    direction = _resolved_held_object_direction(
        env,
        state,
        {
            "reference": "relative",
            "offset": [0.0, 0.0, 0.1],
            "frame": "world",
            "orientation_goal": "axis_align",
            "orientation_axis": "y",
        },
    )

    assert abs(float(torch.dot(direction, torch.tensor([0.0, 1.0, 0.0])))) == (
        pytest.approx(1.0)
    )


def test_axis_align_reference_object_long_and_short_axes() -> None:
    env = _FakeEnv()
    state = _held_state_with_yaw(env, 10.0)

    long_direction = _resolved_held_object_direction(
        env,
        state,
        {
            "reference": "object",
            "obj_name": "pad_y",
            "offset": [0.0, 0.0, 0.1],
            "orientation_goal": "axis_align",
            "orientation_axis": "long_axis",
            "align_to": "pad_y",
        },
    )
    short_direction = _resolved_held_object_direction(
        env,
        state,
        {
            "reference": "object",
            "obj_name": "pad_y",
            "offset": [0.0, 0.0, 0.1],
            "orientation_goal": "axis_align",
            "orientation_axis": "short_axis",
            "align_to": "pad_y",
        },
    )

    assert abs(float(torch.dot(long_direction, torch.tensor([0.0, 1.0, 0.0])))) == (
        pytest.approx(1.0)
    )
    assert abs(float(torch.dot(short_direction, torch.tensor([1.0, 0.0, 0.0])))) == (
        pytest.approx(1.0)
    )


def test_axis_align_selects_smallest_equivalent_yaw() -> None:
    env = _FakeEnv()
    state = _held_state_with_yaw(env, 170.0)

    target = atom_actions._resolve_held_object_pose_target(
        env,
        atom_actions.AtomicActionSpec(
            atomic_action_class="MoveHeldObject",
            robot_name="left_arm",
            control="arm",
            target_object_pose={
                "reference": "relative",
                "offset": [0.0, 0.0, 0.1],
                "frame": "world",
                "orientation_goal": "axis_align",
                "orientation_axis": "x",
            },
            cfg={},
        ),
        state,
    )
    final_yaw = torch.rad2deg(torch.atan2(target[1, 0], target[0, 0]))

    assert float(abs(final_yaw)) == pytest.approx(180.0)


def test_move_held_object_builds_cfg_and_object_pose_target(monkeypatch) -> None:
    env = _FakeEnv()
    capture = []
    _FakeBackendAction.capture = capture
    semantics = atom_actions._build_object_semantics(
        env,
        {"obj_name": "apple", "affordance": "antipodal"},
        {"allow_grasp_annotation": True},
    )
    state = WorldState(
        last_qpos=env.robot.get_qpos().clone(),
        held_object=HeldObjectState(
            semantics=semantics,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        ),
    )

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
            "atomic_action_class": "MoveHeldObject",
            "robot_name": "left_arm",
            "control": "arm",
            "target_object_pose": {
                "reference": "object",
                "obj_name": "apple",
                "offset": [0.0, 0.0, 0.2],
                "orientation_goal": "upright",
            },
            "cfg": {"sample_interval": 13},
        },
        env=env,
        state=state,
    )

    assert action.shape == (2, 3)
    assert isinstance(capture[0]["cfg"], MoveHeldObjectCfg)
    assert capture[0]["cfg"].control_part == "left_arm"
    assert capture[0]["cfg"].hand_control_part == "left_eef"
    assert isinstance(capture[0]["target"], HeldObjectPoseTarget)
    assert capture[0]["target"].object_target_pose[:3, 3].tolist() == pytest.approx(
        [0.4, -0.2, 0.3]
    )


def test_move_held_object_requires_prior_pickup() -> None:
    env = _FakeEnv()
    with pytest.raises(ValueError, match="requires a held object"):
        execute_atomic_action(
            {
                "atomic_action_class": "MoveHeldObject",
                "robot_name": "left_arm",
                "control": "arm",
                "target_object_pose": {
                    "reference": "relative",
                    "offset": [0.0, 0.0, 0.1],
                    "frame": "world",
                    "orientation_goal": "preserve",
                },
                "cfg": {},
            },
            env=env,
        )


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
        match="Place requires control='arm' and target_pose",
    ):
        execute_atomic_action(
            {
                "atomic_action_class": "Place",
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
        convex_decomposition_method="coacd",
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
            convex_decomposition_method="coacd",
            body_scale=None,
            runtime_kwargs={},
        )


def test_vhacd_grasp_collision_path_skips_env_coacd_bridge(monkeypatch) -> None:
    def fail_if_called(**kwargs):
        raise AssertionError("env CoACD bridge should not run for VHACD")

    monkeypatch.setattr(
        atom_actions,
        "ensure_grasp_collision_cache_from_env_coacd",
        fail_if_called,
    )

    atom_actions._prepare_grasp_collision_cache_from_env_coacd(
        obj_name="apple",
        mesh_vertices=torch.zeros(1, 3),
        mesh_triangles=torch.zeros(1, 3, dtype=torch.int64),
        source_mesh_path="/tmp/fake.obj",
        max_decomposition_hulls=4,
        convex_decomposition_method="vhacd",
        body_scale=None,
        runtime_kwargs={},
    )


def test_grasp_convex_decomposition_method_uses_vhacd_alias() -> None:
    target_obj = SimpleNamespace(
        cfg=SimpleNamespace(convex_decomposition_method="visacd")
    )

    method = atom_actions._grasp_convex_decomposition_method(target_obj, {})

    assert method == "vhacd"
