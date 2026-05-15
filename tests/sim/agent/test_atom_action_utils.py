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
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_get_qpos():
    source_path = (
        REPO_ROOT / "embodichain" / "lab" / "sim" / "agent" / "atom_action_utils.py"
    )
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    get_qpos_node = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_qpos"
    )
    get_qpos_module = ast.Module(body=[get_qpos_node], type_ignores=[])
    namespace = {
        "np": np,
        "torch": torch,
        "log_error": lambda message, error_type=RuntimeError: (_ for _ in ()).throw(
            error_type(message)
        ),
        "log_warning": lambda *args, **kwargs: None,
        "find_nearest_valid_pose": None,
    }
    exec(compile(get_qpos_module, filename=str(source_path), mode="exec"), namespace)
    return namespace["get_qpos"], namespace


get_qpos, GET_QPOS_NAMESPACE = _load_get_qpos()


class _DummyEnv:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def get_arm_ik(self, target_xpos, is_left, qpos_seed=None):
        self.calls.append(torch.as_tensor(target_xpos).clone())
        if len(self.calls) == 1:
            raise RuntimeError("invalid pose")
        return True, torch.tensor([0.1, 0.2], dtype=torch.float32)


def test_get_qpos_returns_corrected_pose_when_force_valid() -> None:
    env = _DummyEnv()
    requested_pose = torch.eye(4, dtype=torch.float32)
    corrected_pose = torch.eye(4, dtype=torch.float32)
    corrected_pose[0, 3] = 0.42

    GET_QPOS_NAMESPACE["find_nearest_valid_pose"] = (
        lambda env, select_arm, pose: corrected_pose
    )

    solved_pose, solved_qpos = get_qpos(
        env=env,
        is_left=True,
        select_arm="left_arm",
        pose=requested_pose,
        qpos_seed=torch.zeros(2, dtype=torch.float32),
        force_valid=True,
        name="test pose",
    )

    torch.testing.assert_close(solved_pose, corrected_pose)
    torch.testing.assert_close(
        solved_qpos, torch.tensor([0.1, 0.2], dtype=torch.float32)
    )
    torch.testing.assert_close(env.calls[-1], corrected_pose)


def test_get_qpos_uses_corrected_pose_when_ik_returns_false() -> None:
    class _RetFalseEnv:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def get_arm_ik(self, target_xpos, is_left, qpos_seed=None):
            self.calls.append(torch.as_tensor(target_xpos).clone())
            if len(self.calls) == 1:
                return False, torch.zeros(2, dtype=torch.float32)
            return True, torch.tensor([0.3, 0.4], dtype=torch.float32)

    env = _RetFalseEnv()
    requested_pose = torch.eye(4, dtype=torch.float32)
    corrected_pose = torch.eye(4, dtype=torch.float32)
    corrected_pose[1, 3] = -0.25

    GET_QPOS_NAMESPACE["find_nearest_valid_pose"] = (
        lambda env, select_arm, pose: corrected_pose
    )

    solved_pose, solved_qpos = get_qpos(
        env=env,
        is_left=False,
        select_arm="right_arm",
        pose=requested_pose,
        qpos_seed=torch.zeros(2, dtype=torch.float32),
        force_valid=True,
        name="test pose",
    )

    torch.testing.assert_close(solved_pose, corrected_pose)
    torch.testing.assert_close(
        solved_qpos, torch.tensor([0.3, 0.4], dtype=torch.float32)
    )
    torch.testing.assert_close(env.calls[-1], corrected_pose)


def test_get_qpos_raises_runtime_error_when_force_valid_cannot_correct() -> None:
    class _RetFalseEnv:
        def get_arm_ik(self, target_xpos, is_left, qpos_seed=None):
            return False, torch.zeros(2, dtype=torch.float32)

    def _raise_missing_reachability(env, select_arm, pose):
        raise AttributeError("compute_xpos_reachability")

    GET_QPOS_NAMESPACE["find_nearest_valid_pose"] = _raise_missing_reachability

    with pytest.raises(RuntimeError, match="Generate test pose qpos failed"):
        get_qpos(
            env=_RetFalseEnv(),
            is_left=False,
            select_arm="right_arm",
            pose=torch.eye(4, dtype=torch.float32),
            qpos_seed=torch.zeros(2, dtype=torch.float32),
            force_valid=True,
            name="test pose",
        )
