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

import ast
from functools import partial
from pathlib import Path

import numpy as np
import torch

RIGHT_ARM_SLICE = slice(4, 8)
LEFT_ARM_SLICE = slice(0, 4)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _finalize_actions(select_qpos_traj, ee_state_list_select):
    return np.concatenate(
        [
            np.array(select_qpos_traj),
            np.array(ee_state_list_select),
            np.array(ee_state_list_select),
        ],
        axis=-1,
    )


def _load_drive_function():
    source_path = (
        REPO_ROOT / "embodichain" / "lab" / "sim" / "agent" / "atom_actions.py"
    )
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    drive_node = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "drive"
    )
    drive_module = ast.Module(body=[drive_node], type_ignores=[])
    namespace = {
        "np": np,
        "torch": torch,
        "tqdm": lambda iterable: iterable,
        "finalize_actions": _finalize_actions,
        "resolve_action": lambda action, env, kwargs: (
            action(env=env, **kwargs) if callable(action) else action
        ),
        "get_error_probability": lambda error_function, default: error_function.keywords.get(
            "probability", default
        ),
        "object_error_types": {"misplaced_object", "fallen_object"},
        "ACTION_ERROR_TRIGGER_PROBABILITY": 1.0,
        "OBJECT_ERROR_TRIGGER_PROBABILITY": 1.0,
        "log_info": lambda *args, **kwargs: None,
        "log_warning": lambda *args, **kwargs: None,
        "log_error": lambda *args, **kwargs: None,
    }
    exec(compile(drive_module, filename=str(source_path), mode="exec"), namespace)
    return namespace["drive"]


drive = _load_drive_function()


def _load_open_gripper_function():
    source_path = (
        REPO_ROOT / "embodichain" / "lab" / "sim" / "agent" / "atom_actions.py"
    )
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    open_gripper_node = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "open_gripper"
    )
    open_gripper_module = ast.Module(body=[open_gripper_node], type_ignores=[])
    plan_calls = {"count": 0}

    def _stub_get_arm_states(env, robot_name):
        if "left" in robot_name:
            return (
                True,
                "left_arm",
                env.left_arm_current_qpos,
                torch.eye(4, dtype=torch.float32),
                env.left_arm_current_gripper_state,
            )
        return (
            False,
            "right_arm",
            env.right_arm_current_qpos,
            torch.eye(4, dtype=torch.float32),
            env.right_arm_current_gripper_state,
        )

    def _stub_plan_gripper_trajectory(*args, **kwargs):
        plan_calls["count"] += 1

    namespace = {
        "np": np,
        "torch": torch,
        "get_arm_states": _stub_get_arm_states,
        "plan_gripper_trajectory": _stub_plan_gripper_trajectory,
        "finalize_actions": _finalize_actions,
        "log_info": lambda *args, **kwargs: None,
    }
    exec(compile(open_gripper_module, filename=str(source_path), mode="exec"), namespace)
    return namespace["open_gripper"], plan_calls


open_gripper, open_gripper_plan_calls = _load_open_gripper_function()


class _DummyRobot:
    def get_qpos(self) -> torch.Tensor:
        return torch.zeros((1, 8), dtype=torch.float32)

    def compute_fk(self, qpos, name: str, to_matrix: bool = True) -> torch.Tensor:
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        pose[0, 0, 3] = float(torch.as_tensor(qpos, dtype=torch.float32).sum())
        return pose


class _DummyEnv:
    def __init__(self) -> None:
        self.left_arm_joints = [0, 1]
        self.left_eef_joints = [2, 3]
        self.right_arm_joints = [4, 5]
        self.right_eef_joints = [6, 7]
        self.init_qpos = np.zeros(8, dtype=np.float32)
        self.left_arm_current_qpos = np.array([0.0, 0.0], dtype=np.float32)
        self.right_arm_current_qpos = np.array([0.0, 0.0], dtype=np.float32)
        self.left_arm_current_gripper_state = np.array([0.05], dtype=np.float32)
        self.right_arm_current_gripper_state = np.array([0.05], dtype=np.float32)
        self.robot = _DummyRobot()
        self.step_calls: list[np.ndarray] = []
        self.update_calls = 0

    def step(self, action: torch.Tensor) -> None:
        self.step_calls.append(action.squeeze(0).detach().cpu().numpy())

    def update_obj_info(self) -> None:
        self.update_calls += 1


def test_open_gripper_skips_when_skip_condition_is_met() -> None:
    env = _DummyEnv()
    env.open_state = torch.tensor([0.05], dtype=torch.float32)
    env.left_arm_current_qpos = np.array([0.0, 0.0], dtype=np.float32)
    env.left_arm_current_gripper_state = np.array([0.0], dtype=np.float32)
    before_calls = open_gripper_plan_calls["count"]

    actions = open_gripper(robot_name="left_arm", env=env)

    assert actions.shape == (1, 4)
    np.testing.assert_allclose(actions[0], np.zeros(4, dtype=np.float32))
    assert open_gripper_plan_calls["count"] == before_calls


def test_drive_stops_failed_trajectory_after_recovery() -> None:
    env = _DummyEnv()
    main_action = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ],
        dtype=np.float32,
    )
    recovery_action = np.array([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32)
    next_action = np.array([[5.0, 5.1, 5.2, 5.3]], dtype=np.float32)

    monitor_calls = {"count": 0}

    def trigger_once() -> bool:
        monitor_calls["count"] += 1
        return monitor_calls["count"] == 1

    monitor = partial(trigger_once)

    drive(
        left_arm_action=None,
        right_arm_action=main_action,
        monitor_sequences=[[monitor]],
        recovery_sequences=[
            [
                partial(
                    drive,
                    left_arm_action=None,
                    right_arm_action=recovery_action,
                )
            ]
        ],
        env=env,
    )

    assert len(env.step_calls) == 2
    np.testing.assert_allclose(env.step_calls[0][RIGHT_ARM_SLICE], main_action[0])
    np.testing.assert_allclose(env.step_calls[1][RIGHT_ARM_SLICE], recovery_action[0])

    drive(
        left_arm_action=None,
        right_arm_action=next_action,
        env=env,
    )

    assert len(env.step_calls) == 3
    np.testing.assert_allclose(env.step_calls[2][RIGHT_ARM_SLICE], next_action[0])


def test_recovery_sequences_are_materialized_lazily() -> None:
    env = _DummyEnv()
    main_action = np.array([[1.0, 1.1, 1.2, 1.3]], dtype=np.float32)
    recovery_calls = {"count": 0}

    def build_recovery_action(env=None, **kwargs):
        recovery_calls["count"] += 1
        return np.array([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32)

    recovery_step = partial(
        drive,
        left_arm_action=None,
        right_arm_action=partial(build_recovery_action),
    )

    assert recovery_calls["count"] == 0

    drive(
        left_arm_action=None,
        right_arm_action=main_action,
        monitor_sequences=[[partial(lambda: True)]],
        recovery_sequences=[[recovery_step]],
        env=env,
    )

    assert recovery_calls["count"] == 1
    assert len(env.step_calls) == 2
    np.testing.assert_allclose(
        env.step_calls[1][RIGHT_ARM_SLICE],
        np.array([9.0, 9.1, 9.2, 9.3], dtype=np.float32),
    )


def test_monitor_sequences_align_with_recovery_sequences_by_outer_index() -> None:
    env = _DummyEnv()
    main_action = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
        ],
        dtype=np.float32,
    )
    first_recovery_action = np.array([[7.0, 7.1, 7.2, 7.3]], dtype=np.float32)
    second_recovery_action = np.array([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32)

    call_state = {"count": 0}

    def first_monitor() -> bool:
        call_state["count"] += 1
        return False

    def second_monitor() -> bool:
        call_state["count"] += 1
        return call_state["count"] == 2

    drive(
        left_arm_action=None,
        right_arm_action=main_action,
        monitor_sequences=[
            [partial(first_monitor)],
            [partial(second_monitor)],
        ],
        recovery_sequences=[
            [
                partial(
                    drive,
                    left_arm_action=None,
                    right_arm_action=first_recovery_action,
                )
            ],
            [
                partial(
                    drive,
                    left_arm_action=None,
                    right_arm_action=second_recovery_action,
                )
            ],
        ],
        env=env,
    )

    assert len(env.step_calls) == 2
    np.testing.assert_allclose(env.step_calls[0][RIGHT_ARM_SLICE], main_action[0])
    np.testing.assert_allclose(
        env.step_calls[1][RIGHT_ARM_SLICE],
        second_recovery_action[0],
    )


def test_single_arm_recovery_keeps_other_arm_remaining_trajectory() -> None:
    env = _DummyEnv()
    left_main_action = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
        ],
        dtype=np.float32,
    )
    right_main_action = np.array(
        [
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
        ],
        dtype=np.float32,
    )
    left_recovery_action = np.array([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32)

    drive(
        left_arm_action=left_main_action,
        right_arm_action=right_main_action,
        monitor_sequences=[[partial(lambda: True)]],
        recovery_sequences=[
            [
                partial(
                    drive,
                    left_arm_action=left_recovery_action,
                    right_arm_action=None,
                )
            ]
        ],
        env=env,
    )

    assert len(env.step_calls) == 2
    np.testing.assert_allclose(env.step_calls[0][LEFT_ARM_SLICE], left_main_action[0])
    np.testing.assert_allclose(env.step_calls[0][RIGHT_ARM_SLICE], right_main_action[0])
    np.testing.assert_allclose(
        env.step_calls[1][LEFT_ARM_SLICE],
        left_recovery_action[0],
    )
    np.testing.assert_allclose(
        env.step_calls[1][RIGHT_ARM_SLICE],
        right_main_action[1],
    )


def test_remaining_other_arm_trajectory_is_consumed_only_once() -> None:
    env = _DummyEnv()
    left_main_action = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ],
        dtype=np.float32,
    )
    right_main_action = np.array(
        [
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
            [6.0, 6.1, 6.2, 6.3],
        ],
        dtype=np.float32,
    )
    left_recovery_first = np.array([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32)
    left_recovery_second = np.array([[8.0, 8.1, 8.2, 8.3]], dtype=np.float32)

    drive(
        left_arm_action=left_main_action,
        right_arm_action=right_main_action,
        monitor_sequences=[[partial(lambda: True)]],
        recovery_sequences=[
            [
                partial(
                    drive,
                    left_arm_action=left_recovery_first,
                    right_arm_action=None,
                ),
                partial(
                    drive,
                    left_arm_action=left_recovery_second,
                    right_arm_action=None,
                ),
            ]
        ],
        env=env,
    )

    assert len(env.step_calls) == 4
    np.testing.assert_allclose(env.step_calls[0][LEFT_ARM_SLICE], left_main_action[0])
    np.testing.assert_allclose(
        env.step_calls[0][RIGHT_ARM_SLICE], right_main_action[0]
    )
    np.testing.assert_allclose(
        env.step_calls[1][RIGHT_ARM_SLICE], right_main_action[1]
    )
    np.testing.assert_allclose(
        env.step_calls[2][RIGHT_ARM_SLICE], right_main_action[2]
    )
    np.testing.assert_allclose(
        env.step_calls[3][RIGHT_ARM_SLICE],
        np.zeros(4, dtype=np.float32),
    )
    np.testing.assert_allclose(
        env.step_calls[3][LEFT_ARM_SLICE],
        left_recovery_second[0],
    )


def test_error_function_probability_defaults_to_global_value() -> None:
    env = _DummyEnv()
    calls = {"count": 0}

    def object_error(actions=None, env=None, **kwargs):
        calls["count"] += 1

    original_rand = np.random.rand
    np.random.rand = lambda: 0.5
    try:
        drive(
            left_arm_action=None,
            right_arm_action=np.array([[1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
            error_functions=[partial(object_error, error_type="misplaced_object")],
            env=env,
        )
    finally:
        np.random.rand = original_rand

    assert calls["count"] == 1


def test_error_function_probability_can_override_default() -> None:
    env = _DummyEnv()
    calls = {"count": 0}

    def object_error(actions=None, env=None, **kwargs):
        calls["count"] += 1

    original_rand = np.random.rand
    np.random.rand = lambda: 0.5
    try:
        drive(
            left_arm_action=None,
            right_arm_action=np.array([[1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
            error_functions=[
                partial(
                    object_error,
                    error_type="misplaced_object",
                    probability=0.25,
                )
            ],
            env=env,
        )
    finally:
        np.random.rand = original_rand

    assert calls["count"] == 0
