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

import pytest
import torch

from embodichain.lab.gym.envs.managers import ActionManager
from embodichain.lab.gym.envs.managers.actions import (
    DeltaQposTerm,
    EefPoseTerm,
    QposTerm,
    QposDenormalizedTerm,
    QposNormalizedTerm,
    QvelTerm,
    QfTerm,
)
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg


class MockEnv:
    """Minimal mock env for ActionTerm tests."""

    def __init__(self, num_envs: int = 4, action_dim: int = 6):
        self.num_envs = num_envs
        self.active_joint_ids = list(range(action_dim))
        self.device = torch.device("cpu")

    def get_qpos(self):
        return torch.zeros(
            self.num_envs, len(self.active_joint_ids), device=self.device
        )

    @property
    def robot(self):
        """DeltaQposTerm uses env.robot.get_qpos()."""
        return self


class MockEnvWithLimits(MockEnv):
    """Mock env with qpos_limits for QposDenormalizedTerm."""

    def __init__(self, num_envs: int = 4, action_dim: int = 6):
        super().__init__(num_envs, action_dim)
        # qpos_limits shape: (1, dof, 2) for [low, high]
        self._qpos_limits = torch.zeros(1, action_dim, 2)
        self._qpos_limits[..., 0] = -1.0
        self._qpos_limits[..., 1] = 1.0

    @property
    def robot(self):
        return self

    @property
    def body_data(self):
        class BodyData:
            def __init__(_, limits):
                _.qpos_limits = limits

        return BodyData(self._qpos_limits)


class MockEnvForEef(MockEnv):
    """Mock env with compute_ik for EefPoseTerm."""

    def __init__(self, num_envs: int = 2, action_dim: int = 6):
        super().__init__(num_envs, action_dim)

    def compute_ik(self, pose, joint_seed):
        """Return (all success, joint_seed) to simulate IK success."""
        batch_size = joint_seed.shape[0]
        ret = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        return ret, joint_seed.clone()


def test_delta_qpos_term_process_action():
    """DeltaQposTerm: qpos = current_qpos + scale * action."""
    env = MockEnv(num_envs=4, action_dim=6)
    cfg = ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1})
    term = DeltaQposTerm(cfg, env)

    action = torch.ones(4, 6) * 2.0
    result = term.process_action(action)

    # DeltaQposTerm returns tensor directly, not dict
    expected = env.get_qpos() + 0.1 * action
    torch.testing.assert_close(result, expected)
    assert term.action_dim == 6


def test_qpos_term_process_action():
    """QposTerm: qpos = scale * action."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QposTerm, params={"scale": 0.5})
    term = QposTerm(cfg, env)

    action = torch.ones(2, 3)
    result = term.process_action(action)

    # QposTerm returns tensor directly, not dict
    torch.testing.assert_close(result, torch.ones(2, 3) * 0.5)
    assert term.action_dim == 3


def test_qpos_denormalized_term_process_action():
    """QposDenormalizedTerm: [-1,1] -> [low, high] with scale=1."""
    env = MockEnvWithLimits(num_envs=2, action_dim=2)
    cfg = ActionTermCfg(func=QposDenormalizedTerm, params={"scale": 1.0})
    term = QposDenormalizedTerm(cfg, env)

    # action=-1 -> low, action=1 -> high
    action = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    result = term.process_action(action)

    # QposDenormalizedTerm returns tensor directly, not dict
    # low=-1, high=1: qpos = low + (action + 1.0) * 0.5 * (high - low)
    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    torch.testing.assert_close(result, expected)
    assert term.action_dim == 2


def test_eef_pose_term_process_action_6d():
    """EefPoseTerm: 6D pose (x,y,z,euler) -> IK -> qpos."""
    env = MockEnvForEef(num_envs=2, action_dim=6)
    cfg = ActionTermCfg(func=EefPoseTerm, params={"scale": 1.0, "pose_dim": 6})
    term = EefPoseTerm(cfg, env)

    # 6D: position + euler angles
    action = torch.zeros(2, 6)
    action[:, :3] = 0.1  # position
    action[:, 3:6] = 0.0  # euler (identity rotation)
    result = term.process_action(action)

    assert "qpos" in result
    assert "ik_success" in result
    assert result["qpos"].shape == (2, 6)
    assert result["ik_success"].shape == (2,)
    # Mock returns joint_seed (zeros); verify output matches
    torch.testing.assert_close(result["qpos"], env.get_qpos())
    assert term.action_dim == 6


def test_eef_pose_term_process_action_7d():
    """EefPoseTerm: 7D pose (x,y,z,quat) -> IK -> qpos."""
    env = MockEnvForEef(num_envs=2, action_dim=6)
    cfg = ActionTermCfg(func=EefPoseTerm, params={"scale": 1.0, "pose_dim": 7})
    term = EefPoseTerm(cfg, env)

    # 7D: position + quaternion (w,x,y,z)
    action = torch.zeros(2, 7)
    action[:, :3] = 0.1
    action[:, 3] = 1.0  # quat w
    action[:, 4:7] = 0.0  # quat x,y,z (identity)
    result = term.process_action(action)

    assert "qpos" in result
    assert "ik_success" in result
    assert result["qpos"].shape == (2, 6)
    torch.testing.assert_close(result["qpos"], env.get_qpos())
    assert term.action_dim == 7


def test_eef_pose_term_invalid_dim_raises():
    """EefPoseTerm raises ValueError for non-6D/7D action."""
    env = MockEnvForEef(num_envs=2, action_dim=6)
    cfg = ActionTermCfg(func=EefPoseTerm, params={"scale": 1.0, "pose_dim": 5})
    term = EefPoseTerm(cfg, env)

    with pytest.raises(ValueError, match="EEF pose action must be 6D or 7D"):
        term.process_action(torch.zeros(2, 5))


def test_qvel_term_process_action():
    """QvelTerm: qvel = scale * action."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QvelTerm, params={"scale": 0.2})
    term = QvelTerm(cfg, env)

    action = torch.ones(2, 3)
    result = term.process_action(action)

    # QvelTerm returns tensor directly, not dict
    torch.testing.assert_close(result, torch.ones(2, 3) * 0.2)


def test_qf_term_process_action():
    """QfTerm: qf = scale * action."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QfTerm, params={"scale": 10.0})
    term = QfTerm(cfg, env)

    action = torch.ones(2, 3)
    result = term.process_action(action)

    # QfTerm returns tensor directly, not dict
    torch.testing.assert_close(result, torch.ones(2, 3) * 10.0)


def test_action_manager_tensor_input():
    """ActionManager passes dict input to the specified term."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "delta_qpos": ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1}),
    }
    manager = ActionManager(cfg, env)

    # ActionManager expects dict with input_key matching term
    action = torch.ones(2, 3)
    result = manager.process_action(action)

    expected = env.get_qpos() + 0.1 * torch.ones(2, 3)
    torch.testing.assert_close(result, expected)


def test_action_manager_dict_input():
    """ActionManager processes dict input with single term."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "qpos": ActionTermCfg(func=QposTerm, params={"scale": 1.0}),
    }
    manager = ActionManager(cfg, env)

    action_dict = torch.ones(2, 3) * 0.5
    result = manager.process_action(action_dict)

    torch.testing.assert_close(result, torch.ones(2, 3) * 0.5)


# Tests for action term mode (pre/post)


def test_action_term_cfg_default_mode():
    """ActionTermCfg defaults to mode='pre'."""
    cfg = ActionTermCfg(func=DeltaQposTerm, params={})
    assert cfg.mode == "pre"


def test_action_term_cfg_post_mode():
    """ActionTermCfg supports mode='post'."""
    cfg = ActionTermCfg(func=QposNormalizedTerm, params={}, mode="post")
    assert cfg.mode == "post"


def test_action_manager_process_action_pre_mode():
    """ActionManager.process_action defaults to pre mode."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "delta_qpos": ActionTermCfg(func=DeltaQposTerm, params={"scale": 0.1}),
    }
    manager = ActionManager(cfg, env)

    action = torch.ones(2, 3)
    result = manager.process_action(action, mode="pre")

    expected = env.get_qpos() + 0.1 * torch.ones(2, 3)
    torch.testing.assert_close(result, expected)


def test_action_manager_process_action_post_mode():
    """ActionManager.process_action with post mode uses post terms only."""
    env = MockEnvWithLimits(num_envs=2, action_dim=3)
    cfg = {
        "norm": ActionTermCfg(func=QposNormalizedTerm, mode="post"),
    }
    manager = ActionManager(cfg, env)

    # Action values are qpos that will be normalized
    action = torch.ones(2, 3) * 0.5
    result = manager.process_action(action, mode="post")

    # With qpos_limits = [-1, 1], qpos=0.5 normalizes to (0.5-(-1))/(1-(-1)) = 0.75
    torch.testing.assert_close(result, torch.ones(2, 3) * 0.75)


def test_action_manager_mixed_pre_post_terms():
    """ActionManager with both pre and post terms works correctly."""
    env = MockEnvWithLimits(num_envs=2, action_dim=3)
    cfg = {
        "qpos": ActionTermCfg(func=QposTerm, params={"scale": 1.0}, mode="pre"),
        "norm": ActionTermCfg(func=QposNormalizedTerm, mode="post"),
    }
    manager = ActionManager(cfg, env)

    # Pre mode: should return qpos term output
    action = torch.ones(2, 3) * 0.5
    result_pre = manager.process_action(action, mode="pre")
    torch.testing.assert_close(result_pre, torch.ones(2, 3) * 0.5)

    # Post mode: should return normalized output
    result_post = manager.process_action(action, mode="post")
    # Values should be normalized to [0, 1] range
    # With qpos_limits = [-1, 1], normalized qpos = (0.5 - (-1)) / (1 - (-1)) = 0.75
    expected = torch.ones(2, 3) * 0.75
    torch.testing.assert_close(result_post, expected)


def test_action_manager_get_terms_by_mode():
    """ActionManager.get_terms_by_mode returns correct terms."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "qpos": ActionTermCfg(func=QposTerm, params={}, mode="pre"),
        "norm": ActionTermCfg(func=QposNormalizedTerm, params={}, mode="post"),
    }
    manager = ActionManager(cfg, env)

    pre_terms = manager.get_terms_by_mode("pre")
    assert len(pre_terms) == 1
    assert pre_terms[0][0] == "qpos"

    post_terms = manager.get_terms_by_mode("post")
    assert len(post_terms) == 1
    assert post_terms[0][0] == "norm"


def test_action_manager_get_action_dim_by_mode():
    """ActionManager.get_action_dim_by_mode returns correct dimensions."""
    env = MockEnv(num_envs=2, action_dim=3)
    cfg = {
        "qpos": ActionTermCfg(func=QposTerm, params={}, mode="pre"),
        "norm": ActionTermCfg(func=QposNormalizedTerm, params={}, mode="post"),
    }
    manager = ActionManager(cfg, env)

    assert manager.get_action_dim_by_mode("pre") == 3
    assert manager.get_action_dim_by_mode("post") == 3


def test_qpos_normalized_term_from_qpos():
    """QposNormalizedTerm normalizes qpos from limits to [0, 1] range."""
    env = MockEnvWithLimits(num_envs=2, action_dim=3)
    cfg = ActionTermCfg(func=QposNormalizedTerm, params={})
    term = QposNormalizedTerm(cfg, env)

    # qpos at limits: [-1, 1]
    action = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    result = term.process_action(action)

    # [-1, 0, 1] -> [0, 0.5, 1] when normalized to [0, 1]
    expected = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    torch.testing.assert_close(result, expected)
