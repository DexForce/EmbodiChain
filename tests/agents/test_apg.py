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

"""Tests for APG algorithm and differentiable reach environment.

Validates:
  - Reward computation matches the reference APG implementation
  - Observation shape and layout
  - Gradient flow through the differentiable environment step
  - APG algorithm construction and single update step
  - Environment reset and auto-reset on done
  - Robot-agnostic architecture (DiffEnv → ReachDiffEnv)
"""

from __future__ import annotations

import importlib
import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub heavy simulator dependencies that are pulled in transitively when
# importing anything under ``embodichain.lab``.  The diff_reach module
# itself has *no* dexsim dependency — only ``embodichain.utils.configclass``
# — but the parent-package ``__init__`` chain triggers ``dexsim`` via
# ``embodichain.lab.sim``.  We install lightweight stubs so the import
# succeeds in a test environment without the full simulation stack.
# ---------------------------------------------------------------------------
_PACKAGES_TO_STUB = (
    "dexsim",
    "open3d",
    "cv2",
    "trimesh",
    "viser",
    "scipy",
    "casadi",
    "pink",
    "pinocchio",
    "pytorch_kinematics",
    "omegaconf",
    "h5py",
    "lerobot",
    "polars",
    "fvcore",
    "iopath",
    "torchvision",
    "wandb",
    "deepdiff",
    "psutil",
    "prettytable",
    "ortools",
    "matplotlib",
    "mpl_toolkits",
    "PIL",
    "IPython",
    "yacs",
    "toppra",
    "newton",
)


class _FakeModule(ModuleType):
    """A fake module that auto-creates attributes on access."""

    def __getattr__(self, name: str):
        return MagicMock()


class _StubFinder:
    """A *sys.meta_path* finder that intercepts **all** imports starting with
    the given package prefixes and returns lightweight fakes.  This avoids
    having to enumerate every sub-module of ``dexsim`` / ``open3d`` by hand.
    """

    def __init__(self, prefixes: tuple[str, ...]):
        self._prefixes = prefixes

    def find_spec(self, fullname: str, path=None, target=None):
        for pfx in self._prefixes:
            if fullname == pfx or fullname.startswith(pfx + "."):
                from importlib.machinery import ModuleSpec

                return ModuleSpec(fullname, loader=self, origin="stub")
        return None

    def create_module(self, spec):
        mod = _FakeModule(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__spec__ = spec
        return mod

    def exec_module(self, module):
        # Nothing to execute – the module is fully fake.
        pass


# Insert the finder at the front so it takes priority over filesystem finders.
sys.meta_path.insert(0, _StubFinder(_PACKAGES_TO_STUB))

from embodichain.lab.gym.envs.tasks.rl.diff_reach import (  # noqa: E402
    FRANKA_NUM_ARM_JOINTS,
    DEFAULT_ARM_JOINT_Q,
    DEFAULT_JOINT_LIMITS,
    ReachDiffEnv,
    ReachDiffEnvCfg,
    FrankaReachDiffEnvCfg,
    check_reach_success,
    compute_reach_obs,
    compute_reach_reward,
    quat_distance,
    sample_target_pose,
    TARGET_POS_RANGE,
)
from embodichain.lab.gym.envs.diff_env import (  # noqa: E402
    DiffEnv,
    DiffEnvCfg,
    _rotmat_to_quat,
)
from embodichain.agents.rl.algo.apg import (
    APG,
    APGCfg,
    RunningObsNormalizer,
)  # noqa: E402
from embodichain.agents.rl.models import ActorCritic, MLP  # noqa: E402

# ---------------------------------------------------------------------------
# DH-based forward kinematics (kept for test validation)
# ---------------------------------------------------------------------------
# These reproduce the Franka FR3 DH parameters from the reference APG repo
# so we can validate the environment's FK without pytorch_kinematics or Newton.

_FR3_DH_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
_FR3_DH_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
_FR3_DH_ALPHA = [
    0.0,
    -math.pi / 2,
    math.pi / 2,
    math.pi / 2,
    -math.pi / 2,
    math.pi / 2,
    math.pi / 2,
]
_FLANGE_TO_TCP = torch.tensor([0.0, 0.0, 0.1034], dtype=torch.float32)


def _dh_transform(
    theta: torch.Tensor, a: float, d: float, alpha: float
) -> torch.Tensor:
    """Single DH transformation matrix (Modified DH convention)."""
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    batch = theta.shape[0]
    T = torch.zeros(batch, 4, 4, device=theta.device, dtype=theta.dtype)
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st
    T[:, 0, 3] = a
    T[:, 1, 0] = st * ca
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -sa
    T[:, 1, 3] = -sa * d
    T[:, 2, 0] = st * sa
    T[:, 2, 1] = ct * sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = ca * d
    T[:, 3, 3] = 1.0
    return T


def dh_franka_fk(
    joint_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference DH-based FK for Franka FR3 (test-only).

    Args:
        joint_q: Joint positions ``[batch, 7]``.

    Returns:
        ``(eef_pos [batch, 3], eef_quat [batch, 4])``
    """
    batch = joint_q.shape[0]
    device = joint_q.device
    T = (
        torch.eye(4, device=device, dtype=joint_q.dtype)
        .unsqueeze(0)
        .expand(batch, -1, -1)
        .clone()
    )
    for i in range(FRANKA_NUM_ARM_JOINTS):
        Ti = _dh_transform(joint_q[:, i], _FR3_DH_A[i], _FR3_DH_D[i], _FR3_DH_ALPHA[i])
        T = T @ Ti

    tcp_offset = _FLANGE_TO_TCP.to(device=device, dtype=joint_q.dtype)
    ee_pos = T[:, :3, 3] + T[:, :3, :3] @ tcp_offset
    ee_quat = _rotmat_to_quat(T[:, :3, :3])
    return ee_pos, ee_quat


# ---------------------------------------------------------------------------
# Newton mock helper (replaces _init_newton_state in tests)
# ---------------------------------------------------------------------------


def _mock_init_newton_state(self):
    """Lightweight mock for DiffEnv._init_newton_state.

    Sets Newton-related attributes to safe defaults so the env initialises
    without a real Newton installation.  Forces ``_newton_ready=False`` so
    the PyTorch fallback step is used.
    """
    self.state_0 = None
    self._model_joints_per_env = self.cfg.num_joints
    self._newton_ready = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def num_envs():
    return 4


@pytest.fixture
def env_cfg(num_envs):
    return FrankaReachDiffEnvCfg(
        num_envs=num_envs,
        max_episode_steps=30,
        device="cpu",
    )


@pytest.fixture
def env(env_cfg):
    """Create a ReachDiffEnv with mocked Newton (no Newton/dexsim needed).

    The instance-level ``compute_fk`` attribute is set to the pure-PyTorch
    DH FK so all FK calls in the tests produce correct, differentiable results.
    """
    with patch.object(
        DiffEnv,
        "_build_newton_model",
        lambda self: (MagicMock(), 0, 1),
    ):
        with patch.object(DiffEnv, "_init_newton_state", _mock_init_newton_state):
            e = ReachDiffEnv(env_cfg)
    # Override compute_fk with the differentiable DH reference FK.
    e.compute_fk = dh_franka_fk
    yield e
    e.close()


@pytest.fixture
def default_joint_q(device):
    return torch.tensor([DEFAULT_ARM_JOINT_Q], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Forward Kinematics Tests (DH reference)
# ---------------------------------------------------------------------------


class TestFrankaFK:
    """Test differentiable forward kinematics."""

    def test_dh_fk_output_shape(self, default_joint_q):
        """DH-based FK should return (pos[1,3], quat[1,4])."""
        pos, quat = dh_franka_fk(default_joint_q)
        assert pos.shape == (1, 3)
        assert quat.shape == (1, 4)

    def test_dh_fk_batched(self, device):
        """DH FK should handle batched inputs."""
        batch_q = torch.tensor(
            [DEFAULT_ARM_JOINT_Q] * 8, dtype=torch.float32, device=device
        )
        pos, quat = dh_franka_fk(batch_q)
        assert pos.shape == (8, 3)
        assert quat.shape == (8, 4)
        assert torch.allclose(pos[0], pos[7], atol=1e-6)
        assert torch.allclose(quat[0], quat[7], atol=1e-6)

    def test_dh_fk_home_position_reasonable(self, default_joint_q):
        """FK at home position should place EE in front of and above the base."""
        pos, quat = dh_franka_fk(default_joint_q)
        assert pos[0, 0].item() > -0.5
        assert pos[0, 0].item() < 1.0
        assert pos[0, 2].item() > 0.0
        assert pos[0, 2].item() < 1.5
        assert abs(quat[0].norm().item() - 1.0) < 1e-5

    def test_dh_fk_differentiable(self, device):
        """Gradients should flow through DH FK."""
        q = torch.tensor(
            [DEFAULT_ARM_JOINT_Q],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        pos, quat = dh_franka_fk(q)
        loss = pos.sum() + quat.sum()
        loss.backward()
        assert q.grad is not None
        assert q.grad.shape == (1, FRANKA_NUM_ARM_JOINTS)
        assert not torch.all(q.grad == 0)

    def test_dh_fk_different_configs_give_different_ee(self, device):
        """Different joint configurations should yield different EE poses."""
        q1 = torch.tensor(
            [[0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        q2 = torch.tensor(
            [[0.5, -0.5, 0.5, -2.0, 0.5, 1.0, 0.5]],
            dtype=torch.float32,
            device=device,
        )
        pos1, _ = dh_franka_fk(q1)
        pos2, _ = dh_franka_fk(q2)
        assert not torch.allclose(pos1, pos2, atol=1e-3)

    def test_env_compute_fk_matches_dh(self, env, default_joint_q):
        """The env's compute_fk (DH mock) should match DH reference."""
        dh_pos, dh_quat = dh_franka_fk(default_joint_q)
        env_pos, env_quat = env.compute_fk(default_joint_q)
        assert torch.allclose(env_pos, dh_pos, atol=1e-5)
        # Quaternions may differ by sign (double cover); check distance
        d = quat_distance(env_quat, dh_quat)
        assert d.item() < 1e-4

    def test_env_compute_fk_differentiable(self, env, device):
        """Gradients should flow through env.compute_fk (DH mock)."""
        q = torch.tensor(
            [DEFAULT_ARM_JOINT_Q],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        pos, quat = env.compute_fk(q)
        loss = pos.sum() + quat.sum()
        loss.backward()
        assert q.grad is not None
        assert not torch.all(q.grad == 0)


# ---------------------------------------------------------------------------
# Reward and Observation Tests
# ---------------------------------------------------------------------------


class TestRewardAndObs:
    """Test reward computation and observation composition."""

    def test_reward_shape(self, device, num_envs):
        """Reward should be a [num_envs] tensor."""
        eef_pos = torch.randn(num_envs, 3, device=device)
        eef_quat = torch.randn(num_envs, 4, device=device)
        eef_quat = eef_quat / eef_quat.norm(dim=-1, keepdim=True)
        target_pos = torch.randn(num_envs, 3, device=device)
        target_quat = torch.randn(num_envs, 4, device=device)
        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)

        reward = compute_reach_reward(eef_pos, eef_quat, target_pos, target_quat)
        assert reward.shape == (num_envs,)

    def test_reward_perfect_match(self, device):
        """Zero distance should give maximum reward."""
        pos = torch.tensor([[0.3, 0.0, 0.5]], device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        reward_perfect = compute_reach_reward(pos, quat, pos, quat)
        # Reward = -0.2*0 + 0.1*exp(0) - 0.1*0 = 0.1
        assert abs(reward_perfect.item() - 0.1) < 1e-5

    def test_reward_with_action_rate(self, device):
        """Action rate penalty should reduce reward."""
        pos = torch.tensor([[0.3, 0.0, 0.5]], device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        action = torch.ones(1, 7, device=device)
        last_action = torch.zeros(1, 7, device=device)

        r_with_rate = compute_reach_reward(pos, quat, pos, quat, action, last_action)
        r_no_rate = compute_reach_reward(pos, quat, pos, quat)
        assert r_with_rate.item() < r_no_rate.item()

    def test_reward_differentiable(self, device):
        """Reward should be differentiable w.r.t. eef_pos."""
        eef_pos = torch.randn(2, 3, device=device, requires_grad=True)
        eef_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
        target_pos = torch.randn(2, 3, device=device)
        target_quat = eef_quat.clone()

        reward = compute_reach_reward(eef_pos, eef_quat, target_pos, target_quat)
        reward.sum().backward()
        assert eef_pos.grad is not None
        assert not torch.all(eef_pos.grad == 0)

    def test_obs_shape(self, device, num_envs):
        """Observation should be [num_envs, 28]."""
        obs = compute_reach_obs(
            torch.randn(num_envs, 7, device=device),
            torch.randn(num_envs, 3, device=device),
            torch.randn(num_envs, 4, device=device),
            torch.randn(num_envs, 3, device=device),
            torch.randn(num_envs, 4, device=device),
            torch.randn(num_envs, 7, device=device),
        )
        assert obs.shape == (num_envs, 28)

    def test_obs_layout(self, device):
        """Verify observation tensor layout matches documentation."""
        joint_q = torch.ones(1, 7, device=device) * 0.1
        eef_pos = torch.ones(1, 3, device=device) * 0.2
        eef_quat = torch.ones(1, 4, device=device) * 0.3
        target_pos = torch.ones(1, 3, device=device) * 0.4
        target_quat = torch.ones(1, 4, device=device) * 0.5
        last_action = torch.ones(1, 7, device=device) * 0.6

        obs = compute_reach_obs(
            joint_q, eef_pos, eef_quat, target_pos, target_quat, last_action
        )
        assert torch.allclose(obs[0, :7], joint_q[0])
        assert torch.allclose(obs[0, 7:10], eef_pos[0])
        assert torch.allclose(obs[0, 10:14], eef_quat[0])
        assert torch.allclose(obs[0, 14:17], target_pos[0])
        assert torch.allclose(obs[0, 17:21], target_quat[0])
        assert torch.allclose(obs[0, 21:28], last_action[0])

    def test_quat_distance_identical(self, device):
        """Quaternion distance between identical quats should be ~0."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        d = quat_distance(q, q)
        assert d.item() < 1e-6

    def test_quat_distance_double_cover(self, device):
        """q and -q represent the same rotation, distance should be ~0."""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        q2 = -q1
        d = quat_distance(q1, q2)
        assert d.item() < 1e-6

    def test_check_success(self, device):
        """Success check should work correctly."""
        pos = torch.tensor([[0.3, 0.0, 0.5]], device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        assert check_reach_success(pos, quat, pos, quat).item()
        far_pos = torch.tensor([[10.0, 10.0, 10.0]], device=device)
        assert not check_reach_success(pos, quat, far_pos, quat).item()


# ---------------------------------------------------------------------------
# Environment Architecture Tests
# ---------------------------------------------------------------------------


class TestEnvArchitecture:
    """Verify that the env follows the DiffEnv → ReachDiffEnv hierarchy."""

    def test_reach_env_inherits_diff_env_base(self):
        """ReachDiffEnv should inherit from DiffEnv."""
        assert issubclass(ReachDiffEnv, DiffEnv)

    def test_franka_cfg_inherits_reach_cfg(self):
        """FrankaReachDiffEnvCfg should inherit from ReachDiffEnvCfg."""
        assert issubclass(FrankaReachDiffEnvCfg, ReachDiffEnvCfg)
        assert issubclass(FrankaReachDiffEnvCfg, DiffEnvCfg)

    def test_franka_cfg_defaults(self):
        """FrankaReachDiffEnvCfg should have Franka-specific defaults."""
        cfg = FrankaReachDiffEnvCfg()
        assert cfg.num_joints == 7
        assert cfg.end_link_name == "fr3_hand_tcp"
        assert "Franka" in cfg.urdf_path or "FR3" in cfg.urdf_path

    def test_reach_cfg_is_base_for_franka(self):
        """ReachDiffEnvCfg should be the task-level base (no robot-specific defaults)."""
        assert issubclass(ReachDiffEnvCfg, DiffEnvCfg)
        assert ReachDiffEnvCfg is not FrankaReachDiffEnvCfg


# ---------------------------------------------------------------------------
# Environment Tests
# ---------------------------------------------------------------------------


class TestReachDiffEnv:
    """Test the differentiable environment."""

    def test_reset_returns_correct_shape(self, env, num_envs):
        """Reset should return obs [num_envs, 28] and info dict."""
        obs, info = env.reset(seed=42)
        assert obs.shape == (num_envs, 28)
        assert isinstance(info, dict)

    def test_step_returns_correct_shapes(self, env, num_envs):
        """Step should return correctly shaped tensors."""
        env.reset(seed=42)
        action = torch.randn(num_envs, FRANKA_NUM_ARM_JOINTS)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (num_envs, 28)
        assert reward.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert "final_distance" in info
        assert "success" in info

    def test_step_differentiable_reward(self, env, num_envs):
        """Reward should have gradients w.r.t. action."""
        env.reset(seed=42)
        action = torch.randn(num_envs, FRANKA_NUM_ARM_JOINTS, requires_grad=True)
        obs, reward, _, _, _ = env.step(action)

        reward.sum().backward()
        assert action.grad is not None
        assert action.grad.shape == (num_envs, FRANKA_NUM_ARM_JOINTS)
        assert not torch.all(action.grad == 0)

    def test_step_differentiable_obs_jpos(self, env, num_envs):
        """First 7 dims of obs (jpos) should be differentiable w.r.t. action."""
        env.reset(seed=42)
        action = torch.randn(num_envs, FRANKA_NUM_ARM_JOINTS, requires_grad=True)
        obs, reward, _, _, _ = env.step(action)

        obs[:, :7].sum().backward()
        assert action.grad is not None
        assert not torch.all(action.grad == 0)

    def test_multi_step_gradient_chain(self, env, num_envs):
        """Gradients should accumulate across multiple steps."""
        env.reset(seed=42)
        total_reward = torch.tensor(0.0)

        actions = []
        for _ in range(3):
            action = torch.randn(num_envs, FRANKA_NUM_ARM_JOINTS, requires_grad=True)
            actions.append(action)
            obs, reward, _, _, _ = env.step(action)
            total_reward = total_reward + reward.sum()

        total_reward.backward()
        assert actions[-1].grad is not None
        assert not torch.all(actions[-1].grad == 0)

    def test_detach_state(self, env, num_envs):
        """detach_state should break the computation graph."""
        env.reset(seed=42)
        action = torch.randn(num_envs, FRANKA_NUM_ARM_JOINTS)
        env.step(action)
        env.detach_state()
        assert not env.joint_q.requires_grad
        assert not env.last_action.requires_grad
        assert not env.eef_pos.requires_grad
        assert not env.eef_quat.requires_grad

    def test_partial_reset(self, env, num_envs):
        """Resetting a subset of envs should only affect those envs."""
        env.reset(seed=42)
        original_joint_q = env.joint_q.clone()

        reset_ids = torch.tensor([0], device=env.device)
        env.reset(reset_ids)

        for i in range(1, num_envs):
            assert torch.allclose(env.joint_q[i], original_joint_q[i])

    def test_truncation_after_max_steps(self):
        """Environment should truncate after max_episode_steps."""
        cfg = FrankaReachDiffEnvCfg(
            num_envs=1,
            max_episode_steps=5,
            device="cpu",
        )
        with patch.object(
            DiffEnv,
            "_build_newton_model",
            lambda self: (MagicMock(), 0, 1),
        ):
            with patch.object(DiffEnv, "_init_newton_state", _mock_init_newton_state):
                env = ReachDiffEnv(cfg)
        env.compute_fk = dh_franka_fk
        env.reset(seed=42)

        for step in range(5):
            action = torch.zeros(1, FRANKA_NUM_ARM_JOINTS)
            obs, reward, terminated, truncated, info = env.step(action)
            if step < 4:
                assert not truncated.item(), f"Should not truncate at step {step}"

        assert truncated.item() or env.step_count.item() == 0  # auto-reset
        env.close()

    def test_action_clipping(self, env, num_envs):
        """Actions outside [-1, 1] should be clipped."""
        env.reset(seed=42)
        large_action = torch.ones(num_envs, FRANKA_NUM_ARM_JOINTS) * 10.0
        env.step(large_action)
        assert torch.all(env.joint_q >= env.joint_limit_lower)
        assert torch.all(env.joint_q <= env.joint_limit_upper)

    def test_backward_compat_arm_limit_properties(self, env):
        """Backward-compat arm_joint_limit_lower/upper should work."""
        assert torch.equal(env.arm_joint_limit_lower, env.joint_limit_lower)
        assert torch.equal(env.arm_joint_limit_upper, env.joint_limit_upper)


# ---------------------------------------------------------------------------
# Target Sampling Tests
# ---------------------------------------------------------------------------


class TestTargetSampling:
    """Test target pose sampling."""

    def test_sample_shapes(self, device):
        """Sampled targets should have correct shapes."""
        pos, quat = sample_target_pose(8, device=device)
        assert pos.shape == (8, 3)
        assert quat.shape == (8, 4)

    def test_sample_in_workspace(self, device):
        """Sampled positions should be within workspace bounds."""
        pos, quat = sample_target_pose(100, device=device)
        assert torch.all(pos[:, 0] >= TARGET_POS_RANGE["x"][0])
        assert torch.all(pos[:, 0] <= TARGET_POS_RANGE["x"][1])
        assert torch.all(pos[:, 1] >= TARGET_POS_RANGE["y"][0])
        assert torch.all(pos[:, 1] <= TARGET_POS_RANGE["y"][1])
        assert torch.all(pos[:, 2] >= TARGET_POS_RANGE["z"][0])
        assert torch.all(pos[:, 2] <= TARGET_POS_RANGE["z"][1])

    def test_sample_unit_quaternions(self, device):
        """Sampled quaternions should be unit quaternions."""
        _, quat = sample_target_pose(100, device=device)
        norms = quat.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sample_custom_workspace(self, device):
        """Target sampling should respect custom pos_range."""
        custom_range = {"x": (0.1, 0.2), "y": (-0.1, 0.1), "z": (0.5, 0.6)}
        pos, _ = sample_target_pose(50, device=device, pos_range=custom_range)
        assert torch.all(pos[:, 0] >= 0.1)
        assert torch.all(pos[:, 0] <= 0.2)
        assert torch.all(pos[:, 1] >= -0.1)
        assert torch.all(pos[:, 1] <= 0.1)
        assert torch.all(pos[:, 2] >= 0.5)
        assert torch.all(pos[:, 2] <= 0.6)


# ---------------------------------------------------------------------------
# Running Observation Normalizer Tests
# ---------------------------------------------------------------------------


class TestRunningObsNormalizer:
    """Test Welford-style running normalizer."""

    def test_normalize_initial(self, device):
        """Initial normalization with default stats."""
        normalizer = RunningObsNormalizer(7, device)
        obs = torch.ones(4, 7, device=device) * 5.0
        normed = normalizer.normalize(obs)
        assert torch.allclose(normed, obs, atol=0.01)

    def test_normalize_after_update(self, device):
        """After updating with data, normalization should center it."""
        normalizer = RunningObsNormalizer(3, device)
        data = torch.randn(1000, 3, device=device) * 2.0 + 3.0
        normalizer.update(data)
        normed = normalizer.normalize(data)
        assert abs(normed.mean().item()) < 0.5
        assert abs(normed.std().item() - 1.0) < 0.5


# ---------------------------------------------------------------------------
# APG Algorithm Tests
# ---------------------------------------------------------------------------


class TestAPGAlgorithm:
    """Test APG algorithm construction and basic operation."""

    def _build_policy(self, obs_dim=28, action_dim=7, device="cpu"):
        """Build a small actor-critic policy for testing."""
        actor = MLP(obs_dim, action_dim, hidden_dims=[32, 32], activation="tanh")
        critic = MLP(obs_dim, 1, hidden_dims=[32, 32], activation="tanh")
        return ActorCritic(obs_dim, action_dim, torch.device(device), actor, critic)

    def _make_env(self, num_envs=2, max_steps=5):
        """Build a ReachDiffEnv with mocked Newton FK for testing."""
        cfg = FrankaReachDiffEnvCfg(
            num_envs=num_envs, max_episode_steps=max_steps, device="cpu"
        )
        with patch.object(
            DiffEnv,
            "_build_newton_model",
            lambda self: (MagicMock(), 0, 1),
        ):
            with patch.object(DiffEnv, "_init_newton_state", _mock_init_newton_state):
                env = ReachDiffEnv(cfg)
        env.compute_fk = dh_franka_fk
        return env

    def test_apg_construction(self):
        """APG should construct without errors."""
        cfg = APGCfg(
            device="cpu",
            max_episode_steps=10,
            num_grad_steps=2,
            segment_length=0,
        )
        policy = self._build_policy()
        algo = APG(cfg, policy)
        assert algo.cfg is cfg
        assert algo.optimizer is not None

    def test_apg_construction_with_critic_bootstrap(self):
        """APG with critic bootstrap should have separate param groups."""
        cfg = APGCfg(
            device="cpu",
            max_episode_steps=10,
            num_grad_steps=2,
            segment_length=5,
            bootstrap="critic",
        )
        policy = self._build_policy()
        algo = APG(cfg, policy)
        assert len(algo.optimizer.param_groups) == 2

    def test_apg_single_update(self):
        """APG should perform a single update step without errors."""
        num_envs = 2
        cfg = APGCfg(
            device="cpu",
            max_episode_steps=5,
            num_grad_steps=1,
            segment_length=0,
            ent_coef=0.0,
        )
        policy = self._build_policy()
        algo = APG(cfg, policy)
        env = self._make_env(num_envs=num_envs, max_steps=5)
        obs, _ = env.reset(seed=42)

        rollout = {"env": env, "obs": obs, "num_envs": num_envs}
        metrics = algo.update(rollout)

        assert "policy_loss" in metrics
        assert "critic_loss" in metrics
        assert "total_loss" in metrics
        assert "horizon_return" in metrics
        assert isinstance(metrics["policy_loss"], float)
        env.close()

    def test_apg_update_with_entropy(self):
        """APG with entropy bonus should include entropy in metrics."""
        num_envs = 2
        cfg = APGCfg(
            device="cpu",
            max_episode_steps=5,
            num_grad_steps=1,
            segment_length=0,
            ent_coef=0.01,
        )
        policy = self._build_policy()
        algo = APG(cfg, policy)
        env = self._make_env(num_envs=num_envs, max_steps=5)
        obs, _ = env.reset(seed=42)

        rollout = {"env": env, "obs": obs, "num_envs": num_envs}
        metrics = algo.update(rollout)

        assert metrics["entropy"] != 0.0
        env.close()

    def test_apg_update_with_segmented_critic(self):
        """APG with segmented critic bootstrap should train the critic."""
        num_envs = 2
        cfg = APGCfg(
            device="cpu",
            max_episode_steps=10,
            num_grad_steps=1,
            segment_length=5,
            bootstrap="critic",
            critic_coef=0.5,
        )
        policy = self._build_policy()
        algo = APG(cfg, policy)
        env = self._make_env(num_envs=num_envs, max_steps=10)
        obs, _ = env.reset(seed=42)

        rollout = {"env": env, "obs": obs, "num_envs": num_envs}
        metrics = algo.update(rollout)

        assert metrics["critic_loss"] >= 0.0
        env.close()

    def test_apg_registered(self):
        """APG should be registered in the algorithm registry."""
        from embodichain.agents.rl.algo import get_registered_algo_names

        assert "apg" in get_registered_algo_names()

    def test_apg_build_from_registry(self):
        """APG should be buildable from the registry."""
        from embodichain.agents.rl.algo import build_algo

        policy = self._build_policy()
        algo = build_algo(
            "apg",
            {"max_episode_steps": 10, "num_grad_steps": 1},
            policy,
            torch.device("cpu"),
        )
        assert isinstance(algo, APG)


# ---------------------------------------------------------------------------
# Integration: Reward consistency with reference implementation
# ---------------------------------------------------------------------------


class TestRewardConsistencyWithReference:
    """Verify reward matches the reference APG implementation.

    These tests reproduce the exact reward computation from
    analytic_policy_gradients/envs/franka_reach_env.py::_compute_reward
    to ensure our functor produces identical results.
    """

    def test_reward_formula_components(self, device):
        """Test individual reward components match the reference formula."""
        eef_pos = torch.tensor([[0.3, 0.1, 0.5]], device=device)
        target_pos = torch.tensor([[0.4, 0.0, 0.6]], device=device)
        eef_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        pos_dist = (eef_pos - target_pos).norm(dim=-1)
        rot_dist = quat_distance(eef_quat, target_quat)

        expected = (
            -0.2 * pos_dist
            + 0.1 * torch.exp(-(pos_dist**2) / (2 * 0.1**2))
            - 0.1 * rot_dist
        )

        reward = compute_reach_reward(eef_pos, eef_quat, target_pos, target_quat)
        assert torch.allclose(reward, expected, atol=1e-6)

    def test_reward_with_action_rate_formula(self, device):
        """Test action rate penalty matches reference formula."""
        pos = torch.tensor([[0.3, 0.0, 0.5]], device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        action = torch.tensor([[0.1, 0.2, 0.3, 0.0, -0.1, 0.0, 0.1]], device=device)
        last_action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)

        expected_rate = (action**2).sum(dim=-1)
        base_reward = compute_reach_reward(pos, quat, pos, quat)
        full_reward = compute_reach_reward(pos, quat, pos, quat, action, last_action)

        assert torch.allclose(
            full_reward, base_reward - 0.0001 * expected_rate, atol=1e-6
        )
