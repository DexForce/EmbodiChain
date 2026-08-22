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
import pytest

from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    NeuralPlanner,
    NeuralPlannerCfg,
    PlanState,
)
from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions
from embodichain.lab.sim.planners import neural_planner as neural_planner_module
from embodichain.lab.sim.sim_manager import SimulationManager

NUM_ARM_JOINTS = 7
NUM_WAYPOINTS = 8
OBS_DIM = 300


def _create_fake_onnx_model(tmp_path) -> str:
    model_path = tmp_path / "fake_neural_planner.onnx"
    model_path.write_bytes(b"fake-onnx")
    return str(model_path)


class FakeOnnxPolicy:
    obs_dim = OBS_DIM
    fixed_batch_size = 1

    def __init__(self, path, providers=None):
        self.path = path
        self.providers = providers
        self.last_obs = None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        self.last_obs = obs.clone()
        return torch.zeros(obs.shape[0], NUM_ARM_JOINTS, device=obs.device)


@pytest.fixture(autouse=True)
def _mock_onnx_policy(monkeypatch):
    monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", FakeOnnxPolicy)


class FakeRobot:
    uid = "fake_robot"
    device = torch.device("cpu")
    num_instances = 1

    def get_qpos(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        return torch.zeros(1, NUM_ARM_JOINTS)

    def get_qpos_limits(self, name: str | None = None) -> torch.Tensor:
        limits = torch.zeros(1, NUM_ARM_JOINTS, 2)
        limits[..., 0] = -2.0
        limits[..., 1] = 2.0
        return limits

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        to_matrix: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        batch = qpos.shape[0] if qpos.dim() > 1 else 1
        if to_matrix:
            return torch.eye(4).repeat(batch, 1, 1)
        return torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1)


class FakeSimulationManager:
    def __init__(self):
        self.robot = FakeRobot()

    def get_robot(self, uid: str) -> FakeRobot:
        return self.robot


def test_neural_planner_is_registered():
    assert MotionGenerator._support_planner_dict["neural"][0] is NeuralPlanner
    assert MotionGenerator._support_planner_dict["neural"][1] is NeuralPlannerCfg


def test_neural_planner_generate_with_fake_onnx_model(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )
    )

    target_state = PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))
    result = motion_generator.generate(
        target_states=[target_state],
        options=MotionGenOptions(
            plan_opts=NeuralPlanOptions(
                control_part="main_arm",
                start_qpos=torch.zeros(NUM_ARM_JOINTS),
            ),
        ),
    )

    assert result.success.all().item()
    assert result.positions is not None
    assert result.positions.shape[-1] == NUM_ARM_JOINTS
    assert torch.isfinite(result.positions).all()
    assert result.xpos_list is not None
    assert result.xpos_list.shape[-2:] == (4, 4)


def test_neural_planner_uses_plan_opts_start_qpos(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )
    )
    custom_qpos = torch.ones(NUM_ARM_JOINTS)
    result = motion_generator.generate(
        target_states=[
            PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))
        ],
        options=MotionGenOptions(
            plan_opts=NeuralPlanOptions(
                control_part="main_arm",
                start_qpos=custom_qpos,
            ),
        ),
    )

    assert result.success.all().item()
    assert torch.allclose(result.positions[0, 0], custom_qpos)


def test_neural_planner_rejects_short_start_qpos(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )
    )

    with pytest.raises(ValueError, match="policy expects"):
        motion_generator.generate(
            target_states=[
                PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))
            ],
            options=MotionGenOptions(
                plan_opts=NeuralPlanOptions(
                    control_part="main_arm",
                    start_qpos=torch.zeros(NUM_ARM_JOINTS - 1),
                ),
            ),
        )


def test_neural_planner_returns_velocities_and_accelerations(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )
    )

    result = motion_generator.generate(
        target_states=[
            PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))
        ],
        options=MotionGenOptions(
            plan_opts=NeuralPlanOptions(
                control_part="main_arm",
                start_qpos=torch.zeros(NUM_ARM_JOINTS),
            ),
        ),
    )

    assert result.velocities is not None
    assert result.accelerations is not None
    assert result.velocities.shape == result.positions.shape
    assert result.accelerations.shape == result.positions.shape
    assert torch.isfinite(result.velocities).all()
    assert torch.isfinite(result.accelerations).all()


def test_neural_planner_builds_unified_300d_cartesian_observation(
    tmp_path, monkeypatch
):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )
    joint = torch.zeros(1, 7)
    eef = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    waypoint_pos = torch.zeros(1, 8, 3)
    waypoint_quat = torch.zeros(1, 8, 4)
    waypoint_quat[..., 3] = 1.0
    valid = torch.zeros(1, 8)
    valid[:, :2] = 1.0
    obs = planner._build_obs(
        joint,
        eef,
        waypoint_pos,
        waypoint_quat,
        valid,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 7),
    )

    assert obs.shape == (1, 300)
    # Unified layout semantic blocks: active, valid, pos, rot, joint masks.
    semantic_start = 7 + 7 + 8 * (3 + 4 + 7)
    expected_active = torch.zeros_like(valid)
    expected_active[:, 0] = 1.0
    assert torch.equal(obs[:, semantic_start : semantic_start + 8], expected_active)
    assert torch.equal(obs[:, semantic_start + 8 : semantic_start + 16], valid)
    assert torch.equal(obs[:, semantic_start + 16 : semantic_start + 24], valid)
    assert torch.equal(obs[:, semantic_start + 24 : semantic_start + 32], valid)
    assert torch.count_nonzero(obs[:, semantic_start + 32 : semantic_start + 40]) == 0


def test_neural_planner_applies_policy_frame_and_tcp_transforms(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )
    left = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    right = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.0466],
        [0.0, 0.0, 0.0, 1.0],
    ]
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            policy_frame_from_world=left,
            runtime_tcp_from_policy_tcp=right,
        )
    )
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor([[-0.5, 0.1, 0.4]])
    transformed = planner._to_policy_frame(pose)

    assert torch.allclose(transformed[0, :3, 3], torch.tensor([0.5, -0.1, 0.3534]))


def test_neural_planner_finite_diff_helper():
    """Finite-difference estimates match a known polynomial trajectory."""
    b, n, dof = 2, 5, 7
    dt_value = 0.01
    positions = torch.zeros(b, n, dof)
    for t in range(n):
        positions[:, t] = (t * dt_value) ** 2
    dt = torch.full((b, n), dt_value)

    velocities, accelerations = NeuralPlanner._compute_vel_acc_via_finite_diff(
        positions, dt
    )

    # For p(t) = t^2, v(t) = 2t, a(t) = 2. Interior central differences should
    # match exactly; boundary one-sided differences deviate slightly.
    expected_v = torch.zeros(b, n, dof)
    for t in range(n):
        expected_v[:, t] = 2.0 * (t * dt_value)
    expected_a = torch.full((b, n, dof), 2.0)

    assert velocities.shape == (b, n, dof)
    assert accelerations.shape == (b, n, dof)
    assert torch.allclose(velocities[:, 1:-1], expected_v[:, 1:-1], atol=1e-5)
    assert torch.allclose(accelerations[:, 1:-1], expected_a[:, 1:-1], atol=1e-3)
    # Boundary points should still be finite and have the right sign/order.
    assert torch.all(velocities[:, 0] >= 0.0)
    assert torch.all(velocities[:, -1] >= velocities[:, 0])


def test_motion_generator_neural_propagates_motion_gen_options(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
            )
        )
    )
    custom_qpos = torch.ones(NUM_ARM_JOINTS)
    result = motion_generator.generate(
        target_states=[
            PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))
        ],
        options=MotionGenOptions(
            control_part="main_arm",
            start_qpos=custom_qpos,
        ),
    )

    assert result.success.all().item()
    assert torch.allclose(result.positions[0, 0], custom_qpos)


def test_motion_generator_neural_preserves_native_eef_targets(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )
    )

    options = MotionGenOptions(
        is_interpolate=True,
        control_part="main_arm",
        start_qpos=torch.zeros(NUM_ARM_JOINTS),
    )
    result = motion_generator.generate(
        target_states=[
            PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))
        ],
        options=options,
    )

    assert result.success.all().item()
    assert options.is_interpolate is True


def test_neural_planner_rejects_pytorch_checkpoint(tmp_path, monkeypatch):
    pytorch_checkpoint_path = tmp_path / "checkpoint.pt"
    pytorch_checkpoint_path.write_bytes(b"not-an-onnx-model")
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    with pytest.raises(ValueError, match="only accepts standalone .onnx"):
        NeuralPlanner(
            NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=str(pytorch_checkpoint_path),
            )
        )
