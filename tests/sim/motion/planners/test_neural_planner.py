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

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import (
    MoveType,
    NeuralPlanner,
    NeuralPlannerCfg,
    PlanState,
)
from embodichain.lab.sim.motion.planners import neural_planner as neural_planner_module
from embodichain.lab.sim.motion.planners.neural_planner import NeuralPlanOptions
from embodichain.lab.sim.motion.planners.toppra_planner import retime_joint_paths
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.sim_manager import SimulationManager

NUM_ARM_JOINTS = 7
NUM_WAYPOINTS = 5
OBS_DIM = 186
NMG_OBSERVATION_CAPACITY_WIDTHS = {1: 54, 3: 120, 5: 186}
NMG_OBSERVATION_FINGERPRINTS = {
    1: "0b10a1de1989e6b0c447e0a5914181edb48592371785fb2a8d951771a2b0c245",
    3: "84d7eb95ff9f2e389570f299fb161d4bff6b82540b88110e3738f6e347be9f89",
    5: "485fe689029c9802095b01660c2856df15af97eadc5621e07dbfe6fb58b7703b",
}
NMG_OBSERVATION_BLOCKS = (
    "joint",
    "eef",
    "waypoint_pos",
    "waypoint_quat",
    "waypoint_joint",
    "active_onehot",
    "valid_mask",
    "pos_mask",
    "rot_mask",
    "joint_mask",
    "last_action",
    "waypoint_rel_pos",
    "waypoint_rel_quat",
    "waypoint_joint_err",
)
_REAL_ONNX_POLICY = neural_planner_module._OnnxPolicy


def _create_fake_onnx_model(tmp_path) -> str:
    model_path = tmp_path / "fake_neural_planner.onnx"
    model_path.write_bytes(b"fake-onnx")
    return str(model_path)


class FakeOnnxPolicy:
    obs_dim = OBS_DIM
    fixed_batch_size = 1
    observation_metadata = neural_planner_module._WaypointObservationLayout(
        NUM_WAYPOINTS, NUM_ARM_JOINTS, True, True
    ).onnx_metadata

    def __init__(self, path, providers=None, *, cuda_use_tf32=None, device=None):
        self.path = path
        self.providers = providers
        self.cuda_use_tf32 = cuda_use_tf32
        self.device = device
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
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(batch, 1)


def _fake_batch_fk(self, qpos, name=None, env_ids=None, to_matrix=False):
    return torch.eye(4).repeat(qpos.shape[0], qpos.shape[1], 1, 1)


FakeRobot.compute_batch_fk = _fake_batch_fk


class FakeSimulationManager:
    def __init__(self):
        self.robot = FakeRobot()

    def get_robot(self, uid: str) -> FakeRobot:
        return self.robot


def test_neural_planner_is_registered():
    assert MotionGenerator._support_planner_dict["neural"][0] is NeuralPlanner
    assert MotionGenerator._support_planner_dict["neural"][1] is NeuralPlannerCfg
    assert NeuralPlanner.preserve_plan_samples is True
    assert NeuralPlanner.preserve_failed_plan_positions is True
    assert NeuralPlanner.supported_move_types == frozenset(
        {MoveType.EEF_MOVE, MoveType.JOINT_MOVE}
    )


@pytest.mark.parametrize(
    ("capacity", "expected_width"),
    NMG_OBSERVATION_CAPACITY_WIDTHS.items(),
)
def test_neural_planner_observation_width_matches_nmg_contract(
    capacity: int,
    expected_width: int,
):
    layout = neural_planner_module._WaypointObservationLayout(capacity, 7, True)

    assert layout.dim == expected_width
    assert layout.fingerprint == NMG_OBSERVATION_FINGERPRINTS[capacity]
    assert tuple(name for name, _ in layout.block_widths) == NMG_OBSERVATION_BLOCKS


def test_ur5_periodic_observation_matches_nmg_export(tmp_path, monkeypatch):
    """UR5 policy consumes raw joints followed by sin/cos in the ONNX contract."""
    layout = neural_planner_module._WaypointObservationLayout(5, 6, True, True, True)
    assert layout.dim == 186
    assert layout.fingerprint == (
        "5cd10ef20e627bdc17b9fd4f1c038d6615faad192796168695de0afa92c4dd49"
    )
    assert tuple(name for name, _ in layout.block_widths)[-2:] == (
        "joint_sin",
        "joint_cos",
    )
    assert layout.onnx_metadata["nmg.joint_periodic_encoding"] == "sin_cos_radians"

    class FakeUr5Policy(FakeOnnxPolicy):
        output_dim = 6
        observation_metadata = layout.onnx_metadata

    monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", FakeUr5Policy)
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: FakeSimulationManager()),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            control_part="main_arm",
            num_arm_joints=6,
            joint_periodic_features=True,
        )
    )
    joint = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0, 3.0]])
    eef = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    zeros = torch.zeros(1, 5)
    obs = planner._build_obs(
        joint,
        eef,
        torch.zeros(1, 5, 3),
        eef[:, None, 3:].expand(1, 5, 4),
        torch.zeros(1, 5, 6),
        zeros,
        zeros,
        zeros,
        zeros,
        torch.zeros(1, dtype=torch.long),
        torch.zeros_like(joint),
    )
    assert obs.shape == (1, 186)
    torch.testing.assert_close(obs[:, layout.slices["joint"]], joint)
    torch.testing.assert_close(obs[:, layout.slices["joint_sin"]], joint.sin())
    torch.testing.assert_close(obs[:, layout.slices["joint_cos"]], joint.cos())


def test_neural_planner_default_observation_width_matches_k5_nmg_export():
    assert NeuralPlannerCfg().num_waypoints == NUM_WAYPOINTS
    assert neural_planner_module._waypoint_obs_dim(5, use_relative_obs=True) == 186
    assert neural_planner_module._waypoint_obs_dim(5, use_relative_obs=False) == 116


def test_neural_onnx_policy_preserves_distinct_dynamic_batch_rows(tmp_path):
    """The ONNX consumer keeps numerical parity for batch one and batch three."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    weight = torch.zeros(OBS_DIM, NUM_ARM_JOINTS, dtype=torch.float32)
    weight[:NUM_ARM_JOINTS] = torch.eye(NUM_ARM_JOINTS)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["obs", "weight"], ["action"])],
        "nmg-dynamic-batch-contract",
        [
            onnx.helper.make_tensor_value_info(
                "obs", onnx.TensorProto.FLOAT, ["batch", OBS_DIM]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "action", onnx.TensorProto.FLOAT, ["batch", NUM_ARM_JOINTS]
            )
        ],
        [onnx.numpy_helper.from_array(weight.numpy(), name="weight")],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    layout = neural_planner_module._WaypointObservationLayout(5, 7, True)
    onnx.helper.set_model_props(model, layout.onnx_metadata)
    model.ir_version = 10
    model_path = tmp_path / "dynamic_batch_contract.onnx"
    onnx.save(model, model_path)
    policy = _REAL_ONNX_POLICY(model_path)
    observation = torch.zeros(3, OBS_DIM)
    observation[:, :NUM_ARM_JOINTS] = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
        ]
    )

    batch_one = policy(observation[:1])
    batch_three = policy(observation)

    torch.testing.assert_close(batch_one, observation[:1, :NUM_ARM_JOINTS])
    torch.testing.assert_close(batch_three, observation[:, :NUM_ARM_JOINTS])
    torch.testing.assert_close(batch_three[:1], batch_one)
    assert policy.observation_metadata == layout.onnx_metadata


def test_neural_planner_rejects_mismatched_observation_metadata(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )

    class MismatchedMetadataPolicy(FakeOnnxPolicy):
        observation_metadata = {
            **neural_planner_module._WaypointObservationLayout(
                5, 7, True
            ).onnx_metadata,
            "nmg.canonicalize_quat_obs": "false",
        }

    monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", MismatchedMetadataPolicy)

    with pytest.raises(ValueError, match="observation layout metadata mismatch"):
        NeuralPlanner(
            NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )


def test_neural_planner_rejects_missing_observation_metadata(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )

    class MissingMetadataPolicy(FakeOnnxPolicy):
        observation_metadata = {}

    monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", MissingMetadataPolicy)

    with pytest.raises(ValueError, match="observation layout metadata mismatch"):
        NeuralPlanner(
            NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
            )
        )


def test_neural_planner_generate_with_fake_onnx_model(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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


class SaturatingOnnxPolicy(FakeOnnxPolicy):
    """Policy that always saturates, the worst case for implied velocity."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        self.last_obs = obs.clone()
        return torch.ones(obs.shape[0], NUM_ARM_JOINTS, device=obs.device)


def _rollout(tmp_path, monkeypatch, *, policy=None, **cfg_overrides):
    """Roll the neural planner out toward one identity pose target."""
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    if policy is not None:
        monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", policy)
    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
                **cfg_overrides,
            )
        )
    )
    return motion_generator.generate(
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


def test_neural_planner_keeps_nominal_timing_without_constraints(tmp_path, monkeypatch):
    result = _rollout(tmp_path, monkeypatch, policy=SaturatingOnnxPolicy)

    # The default path is unchanged: one nominal interval per rollout step.
    intervals = result.dt[0, 1:]
    assert torch.allclose(intervals, torch.full_like(intervals, 0.01))
    # 0.2 rad per 0.01 s is 20 rad/s, which no arm can execute.
    assert float(result.velocities.abs().max()) == pytest.approx(20.0, rel=1e-3)


def test_neural_planner_constraints_replace_timing_and_bound_dynamics(
    tmp_path, monkeypatch
):
    nominal = _rollout(tmp_path, monkeypatch, policy=SaturatingOnnxPolicy)
    retimed = _rollout(
        tmp_path,
        monkeypatch,
        policy=SaturatingOnnxPolicy,
        constraints={"velocity": 1.0, "acceleration": 2.0},
    )

    assert bool(retimed.success.all())
    assert float(retimed.velocities.abs().max()) <= 1.0 + 1e-6
    assert float(retimed.accelerations.abs().max()) <= 2.0 + 1e-6
    # Respecting the limits costs time; the nominal timing understated it.
    assert float(retimed.dt.sum()) > float(nominal.dt.sum())
    # Retiming resamples the path, so poses must track the returned positions.
    assert retimed.xpos_list.shape[:2] == retimed.positions.shape[:2]


def test_retime_joint_paths_rejects_malformed_input():
    positions = torch.zeros(1, 4, NUM_ARM_JOINTS)
    limits = {"velocity": 1.0, "acceleration": 1.0}
    with pytest.raises(ValueError, match="velocity"):
        retime_joint_paths(positions, constraints={"acceleration": 1.0})
    with pytest.raises(ValueError, match=r"\(B, N, DOF\)"):
        retime_joint_paths(positions[0], constraints=limits)
    # The default interval is a sample count; reading it as seconds would
    # silently collapse a dense path to two samples.
    with pytest.raises(ValueError, match="TIME sampling"):
        retime_joint_paths(
            positions,
            constraints=limits,
            sample_method=TrajectorySampleMethod.TIME,
        )
    with pytest.raises(ValueError, match="path_lengths"):
        retime_joint_paths(
            positions,
            constraints=limits,
            path_lengths=torch.tensor([5]),
        )


def test_retime_joint_paths_slices_a_single_cpu_path_to_its_valid_prefix():
    path = torch.zeros(1, 4, NUM_ARM_JOINTS, dtype=torch.float64)
    path[0, :, 0] = torch.tensor([0.0, 0.05, 0.1, 0.1])
    limits = {"velocity": 1.0, "acceleration": 2.0}

    result = retime_joint_paths(
        path,
        constraints=limits,
        path_lengths=torch.tensor([3]),
        sample_interval=17,
    )
    trimmed = retime_joint_paths(path[:, :3], constraints=limits, sample_interval=17)

    torch.testing.assert_close(result.positions, trimmed.positions)
    torch.testing.assert_close(result.dt, trimmed.dt)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "bfloat16_input", ["positions", "velocity", "acceleration", "all"]
)
def test_retime_joint_paths_accepts_single_cpu_bfloat16_inputs(
    bfloat16_input: str,
) -> None:
    path = torch.zeros(1, 3, NUM_ARM_JOINTS)
    # Binary-exact values isolate dtype conversion from input quantization.
    path[0, :, 0] = torch.tensor([0.0, 0.125, 0.25])
    limits = {
        "velocity": torch.ones(NUM_ARM_JOINTS),
        "acceleration": torch.full((NUM_ARM_JOINTS,), 2.0),
    }
    reference = retime_joint_paths(path, constraints=limits, sample_interval=17)
    if bfloat16_input in {"positions", "all"}:
        path = path.to(torch.bfloat16)
    limits = {
        name: value.to(torch.bfloat16) if bfloat16_input in {name, "all"} else value
        for name, value in limits.items()
    }

    result = retime_joint_paths(path, constraints=limits, sample_interval=17)

    assert result.success.tolist() == reference.success.tolist() == [True]
    for field in ("positions", "velocities", "accelerations", "dt"):
        torch.testing.assert_close(getattr(result, field), getattr(reference, field))


def test_retime_joint_paths_solves_a_cpu_batch_in_one_call(monkeypatch):
    from embodichain.lab.sim.motion.planners import toppra_planner as toppra_module

    paths = torch.zeros(3, 6, NUM_ARM_JOINTS, dtype=torch.float64)
    paths[:, :, 0] = torch.linspace(0.0, 1.0, 6)
    paths[1, :, 1] = torch.linspace(0.0, -0.5, 6)
    paths[2, :, 0] *= 0.3
    limits = {"velocity": 1.0, "acceleration": 2.0}
    rows = [retime_joint_paths(paths[i : i + 1], constraints=limits) for i in range(3)]

    calls = []
    original = toppra_module._retime_toppra_warp
    monkeypatch.setattr(
        toppra_module,
        "_retime_toppra_warp",
        lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs),
    )
    batched = retime_joint_paths(paths, constraints=limits)

    # Rows are no longer solved one after another on the CPU.
    assert len(calls) == 1
    assert bool(batched.success.all())
    for index, row in enumerate(rows):
        assert float(batched.dt[index].sum()) == pytest.approx(
            float(row.dt.sum()), abs=1e-5
        )


def test_retime_joint_paths_promotes_half_precision_before_warp():
    paths = torch.zeros(2, 6, NUM_ARM_JOINTS, dtype=torch.float16)
    paths[:, :, 0] = torch.linspace(0.0, 1.0, 6, dtype=torch.float16)

    result = retime_joint_paths(
        paths,
        constraints={"velocity": 1.0, "acceleration": 2.0},
    )

    assert result.success.tolist() == [True, True]
    assert result.positions.dtype == torch.float64


def test_retiming_samples_on_the_control_period_not_the_rollout_step_count(
    tmp_path, monkeypatch
):
    # The rollout emits however many steps it happened to take. Inheriting that
    # count leaves a stretched trajectory sampled far too coarsely, so a
    # waypoint the rollout stopped on falls between two output samples.
    nominal = _rollout(tmp_path, monkeypatch, policy=SaturatingOnnxPolicy)
    retimed = _rollout(
        tmp_path,
        monkeypatch,
        policy=SaturatingOnnxPolicy,
        constraints={"velocity": 1.0, "acceleration": 2.0},
    )
    duration = float(retimed.dt.sum())
    assert duration > float(nominal.dt.sum())
    # One sample per nominal dt over the solved duration, not one per step.
    assert retimed.positions.shape[1] > nominal.positions.shape[1]
    assert retimed.positions.shape[1] == pytest.approx(duration / 0.01, rel=0.05)
    torch.testing.assert_close(retimed.policy_rollout_positions, nominal.positions)
    torch.testing.assert_close(
        retimed.policy_rollout_lengths, nominal.policy_rollout_lengths
    )


def test_retimed_success_reflects_the_trajectory_actually_returned(
    tmp_path, monkeypatch
):
    # A joint limit the rollout clamps against is also a limit the spline must
    # respect, and the resampled grid must still reach the goal. Success is
    # re-derived from the returned samples rather than inherited.
    retimed = _rollout(
        tmp_path,
        monkeypatch,
        policy=SaturatingOnnxPolicy,
        constraints={"velocity": 1.0, "acceleration": 2.0},
    )
    arm = retimed.positions[..., :NUM_ARM_JOINTS]
    # FakeRobot declares +-2.0 rad on every joint.
    assert float(arm.max()) <= 2.0 + 1e-6
    assert float(arm.min()) >= -2.0 - 1e-6
    assert float(retimed.dt.sum()) > 0.0


def test_retimed_waypoint_check_keeps_order_and_runs_once_per_waypoint(
    tmp_path, monkeypatch
):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )
    # Two joint-space waypoints on joint 0: 0.5 rad, then 1.0 rad.
    batch, samples = 3, 3
    waypoints_joint = torch.zeros(batch, NUM_WAYPOINTS, NUM_ARM_JOINTS)
    waypoints_joint[:, 0, 0] = 0.5
    waypoints_joint[:, 1, 0] = 1.0
    joint_mask = torch.zeros(batch, NUM_WAYPOINTS)
    joint_mask[:, :2] = 1.0
    waypoints_quat = torch.zeros(batch, NUM_WAYPOINTS, 4)
    waypoints_quat[..., 3] = 1.0
    no_mask = torch.zeros(batch, NUM_WAYPOINTS)
    waypoints = (
        torch.zeros(batch, NUM_WAYPOINTS, 3),
        waypoints_quat,
        waypoints_joint,
        no_mask,
        no_mask,
        joint_mask,
    )
    positions = torch.zeros(batch, samples, NUM_ARM_JOINTS)
    positions[0, :, 0] = torch.tensor([0.0, 0.5, 1.0])  # in order
    positions[1, :, 0] = torch.tensor([0.0, 1.0, 0.5])  # second before first
    positions[2, :, 0] = torch.tensor([0.0, 0.0, 0.5])  # never the second
    poses = torch.eye(4).repeat(batch, samples, 1, 1)

    calls = []
    original = planner._is_active_reached
    monkeypatch.setattr(
        planner,
        "_is_active_reached",
        lambda *args: calls.append(1) or original(*args),
    )
    reached = planner._retimed_reaches_waypoints(positions, poses, waypoints, 2)

    assert reached.tolist() == [True, False, False]
    # One batched test per waypoint, independent of the sample count.
    assert len(calls) == 2


def test_retimed_waypoint_check_allows_repeated_stationary_targets(
    tmp_path, monkeypatch
):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )
    waypoints_joint = torch.zeros(1, NUM_WAYPOINTS, NUM_ARM_JOINTS)
    waypoints_joint[:, :3, 0] = 0.5
    joint_mask = torch.zeros(1, NUM_WAYPOINTS)
    joint_mask[:, :3] = 1.0
    waypoints_quat = torch.zeros(1, NUM_WAYPOINTS, 4)
    waypoints_quat[..., 3] = 1.0
    no_mask = torch.zeros(1, NUM_WAYPOINTS)
    waypoints = (
        torch.zeros(1, NUM_WAYPOINTS, 3),
        waypoints_quat,
        waypoints_joint,
        no_mask,
        no_mask,
        joint_mask,
    )
    positions = torch.zeros(1, 2, NUM_ARM_JOINTS)
    positions[:, 1, 0] = 0.5
    poses = torch.eye(4).repeat(1, 2, 1, 1)

    reached = planner._retimed_reaches_waypoints(positions, poses, waypoints, 3)

    assert reached.tolist() == [True]


def test_neural_planner_disables_grad_for_all_fk_calls(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    grad_states = []
    original_compute_fk = fake_sim.robot.compute_fk

    def checked_compute_fk(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return original_compute_fk(*args, **kwargs)

    monkeypatch.setattr(fake_sim.robot, "compute_fk", checked_compute_fk)
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )

    planner.plan(
        [PlanState.single(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
        NeuralPlanOptions(
            control_part="main_arm",
            start_qpos=torch.zeros(NUM_ARM_JOINTS),
            max_steps=1,
        ),
    )

    assert grad_states
    assert not any(grad_states)


def test_neural_planner_builds_unified_k5_cartesian_observation(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )
    joint = torch.arange(7, dtype=torch.float32).unsqueeze(0) / 10.0
    eef = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    waypoint_pos = torch.arange(NUM_WAYPOINTS * 3, dtype=torch.float32).reshape(
        1, NUM_WAYPOINTS, 3
    )
    waypoint_quat = torch.zeros(1, NUM_WAYPOINTS, 4)
    waypoint_quat[..., 3] = 1.0
    waypoint_quat[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.9])
    waypoint_joint = torch.arange(NUM_WAYPOINTS * 7, dtype=torch.float32).reshape(
        1, NUM_WAYPOINTS, 7
    )
    valid = torch.zeros(1, NUM_WAYPOINTS)
    valid[:, :2] = 1.0
    pos_mask = valid.clone()
    rot_mask = valid.clone()
    joint_mask = torch.zeros_like(valid)
    active_idx = torch.zeros(1, dtype=torch.long)
    last_action = torch.arange(7, dtype=torch.float32).unsqueeze(0) / 20.0
    obs = planner._build_obs(
        joint,
        eef,
        waypoint_pos,
        waypoint_quat,
        waypoint_joint,
        valid,
        pos_mask,
        rot_mask,
        joint_mask,
        active_idx,
        last_action,
    )

    expected_active = torch.zeros_like(valid)
    expected_active[:, 0] = 1.0
    pos_block = waypoint_pos * pos_mask.unsqueeze(-1)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4)
    quat_block = torch.where(rot_mask.unsqueeze(-1) > 0.5, waypoint_quat, identity)
    joint_block = torch.zeros_like(waypoint_joint)
    rel_pos = (pos_block - eef[:, None, :3]) * pos_mask.unsqueeze(-1)
    rel_quat = quat_block
    joint_err = torch.zeros_like(waypoint_joint)
    contract_blocks = {
        "joint": joint,
        "eef": eef,
        "waypoint_pos": pos_block.flatten(1),
        "waypoint_quat": quat_block.flatten(1),
        "waypoint_joint": joint_block.flatten(1),
        "active_onehot": expected_active,
        "valid_mask": valid,
        "pos_mask": pos_mask,
        "rot_mask": rot_mask,
        "joint_mask": joint_mask,
        "last_action": last_action,
        "waypoint_rel_pos": rel_pos.flatten(1),
        "waypoint_rel_quat": rel_quat.flatten(1),
        "waypoint_joint_err": joint_err.flatten(1),
    }
    expected = torch.cat(
        [contract_blocks[name] for name in NMG_OBSERVATION_BLOCKS], dim=-1
    )

    assert obs.shape == (1, OBS_DIM)
    torch.testing.assert_close(obs, expected)


def test_neural_planner_preserves_xyzw_for_pose_targets(tmp_path, monkeypatch):
    """Pose targets must not be reordered after ``quat_from_matrix``."""
    from embodichain.utils.math import matrix_from_quat

    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )

    quaternion = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, :3] = matrix_from_quat(quaternion)

    _, waypoint_quat, *_ = planner._parse_waypoints(
        [PlanState.single(move_type=MoveType.EEF_MOVE, xpos=pose)]
    )

    torch.testing.assert_close(waypoint_quat[:, 0], quaternion)


def test_neural_planner_fk_preserves_xyzw_for_asymmetric_rotation(
    tmp_path, monkeypatch
):
    """FK observations use the public ``xyz + xyzw`` pose convention."""
    from embodichain.utils.math import matrix_from_quat

    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )

    quaternion = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, :3] = matrix_from_quat(quaternion)

    fk_pose = planner._policy_pose_xyzw(pose)

    torch.testing.assert_close(fk_pose[:, 3:7], quaternion)


def test_neural_planner_reach_check_passes_xyzw_to_math(monkeypatch):
    """The XYZW math API should receive planner quaternions unchanged."""
    planner = NeuralPlanner.__new__(NeuralPlanner)
    planner.device = torch.device("cpu")
    planner._num_waypoints = 1
    planner._pos_eps = 1.0e-6
    planner._rot_eps = 1.0e-6
    planner._joint_eps = 1.0e-6

    ee_quaternion = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    ee_quaternion = ee_quaternion / torch.linalg.vector_norm(
        ee_quaternion, dim=-1, keepdim=True
    )
    captured: dict[str, torch.Tensor] = {}

    def spy_quat_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        captured["q1"] = q1
        captured["q2"] = q2
        return torch.zeros(q1.shape[:-1], dtype=q1.dtype, device=q1.device)

    monkeypatch.setattr(neural_planner_module, "quat_error_magnitude", spy_quat_error)
    ee_pose = torch.cat((torch.zeros(1, 3), ee_quaternion), dim=-1)
    waypoint_pos = torch.zeros(1, 1, 3)
    waypoint_quat = ee_quaternion.unsqueeze(1).clone()
    waypoint_joint = torch.zeros(1, 1, NUM_ARM_JOINTS)
    mask = torch.ones(1, 1)
    active_idx = torch.zeros(1, dtype=torch.long)

    reached = planner._is_active_reached(
        torch.zeros(1, NUM_ARM_JOINTS),
        ee_pose,
        waypoint_pos,
        waypoint_quat,
        waypoint_joint,
        mask,
        mask,
        torch.zeros_like(mask),
        active_idx,
    )

    assert reached.item()
    torch.testing.assert_close(captured["q1"], ee_quaternion)
    torch.testing.assert_close(captured["q2"], ee_quaternion)


def test_neural_planner_builds_joint_constraint_observation(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )
    target = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.4]])
    parsed = planner._parse_waypoints(
        [PlanState.from_qpos(target, move_type=MoveType.JOINT_MOVE)]
    )
    (
        waypoint_pos,
        waypoint_quat,
        waypoint_joint,
        valid,
        pos_mask,
        rot_mask,
        joint_mask,
        _,
    ) = parsed
    obs = planner._build_obs(
        torch.zeros(1, 7),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        waypoint_pos,
        waypoint_quat,
        waypoint_joint,
        valid,
        pos_mask,
        rot_mask,
        joint_mask,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 7),
    )

    semantic_start = 7 + 7 + NUM_WAYPOINTS * (3 + 4 + 7)
    relative_start = semantic_start + 5 * NUM_WAYPOINTS + 7
    joint_error_start = relative_start + NUM_WAYPOINTS * (3 + 4)
    assert torch.equal(joint_mask[:, 0], torch.ones(1))
    assert torch.count_nonzero(pos_mask) == 0
    assert torch.count_nonzero(rot_mask) == 0
    assert torch.allclose(obs[:, joint_error_start : joint_error_start + 7], target)


def test_neural_planner_parses_ordered_pose_then_joint_sequence(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )
    pose = torch.eye(4).unsqueeze(0)
    joint = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.4]])

    (
        _,
        _,
        parsed_joint,
        valid,
        pos_mask,
        rot_mask,
        joint_mask,
        episode_k,
    ) = planner._parse_waypoints(
        [
            PlanState.from_xpos(pose, move_type=MoveType.EEF_MOVE),
            PlanState.from_qpos(joint, move_type=MoveType.JOINT_MOVE),
        ]
    )

    assert episode_k == 2
    assert torch.equal(valid[0, :2], torch.ones(2))
    assert torch.equal(pos_mask[0, :2], torch.tensor([1.0, 0.0]))
    assert torch.equal(rot_mask[0, :2], torch.tensor([1.0, 0.0]))
    assert torch.equal(joint_mask[0, :2], torch.tensor([0.0, 1.0]))
    assert torch.allclose(parsed_joint[0, 1], joint[0])


def test_neural_planner_accepts_joint_move_goal(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
        )
    )

    result = planner.plan(
        [PlanState.from_qpos(torch.zeros(1, 7), move_type=MoveType.JOINT_MOVE)],
        NeuralPlanOptions(
            control_part="main_arm",
            start_qpos=torch.zeros(NUM_ARM_JOINTS),
            max_steps=1,
        ),
    )

    assert result.success.all().item()


def test_neural_planner_applies_policy_frame_and_tcp_transforms(tmp_path, monkeypatch):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
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


def test_motion_generator_neural_preserves_failed_rollout_positions(
    tmp_path, monkeypatch
):
    class MovingFakeOnnxPolicy(FakeOnnxPolicy):
        def __call__(self, obs: torch.Tensor) -> torch.Tensor:
            self.last_obs = obs.clone()
            action = torch.zeros(obs.shape[0], NUM_ARM_JOINTS, device=obs.device)
            action[:, 0] = 1.0
            return action

    monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", MovingFakeOnnxPolicy)
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=model_path,
                control_part="main_arm",
                max_steps=2,
            )
        )
    )
    target = torch.eye(4)
    target[0, 3] = 10.0

    result = motion_generator.generate(
        target_states=[PlanState.single(move_type=MoveType.EEF_MOVE, xpos=target)],
        options=MotionGenOptions(
            control_part="main_arm",
            start_qpos=torch.zeros(NUM_ARM_JOINTS),
        ),
    )

    assert not result.success.all().item()
    assert result.positions is not None
    assert result.positions[0, 1, 0] > result.positions[0, 0, 0]
    torch.testing.assert_close(result.policy_rollout_positions, result.positions)
    assert result.policy_rollout_lengths.tolist() == [3]


def test_neural_planner_rejects_pytorch_checkpoint(tmp_path, monkeypatch):
    pytorch_checkpoint_path = tmp_path / "checkpoint.pt"
    pytorch_checkpoint_path.write_bytes(b"not-an-onnx-model")
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )

    with pytest.raises(ValueError, match="only accepts standalone .onnx"):
        NeuralPlanner(
            NeuralPlannerCfg(
                robot_uid="fake_robot",
                onnx_model_path=str(pytorch_checkpoint_path),
            )
        )


@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("retimed", [False, True])
def test_native_and_retimed_rollouts_preserve_requested_first_sample(
    tmp_path, monkeypatch, batch_size, retimed
):
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        FakeOnnxPolicy,
        "__call__",
        lambda self, obs: torch.full(
            (obs.shape[0], NUM_ARM_JOINTS), 0.5, device=obs.device
        ),
    )
    fake_sim.robot.num_instances = batch_size
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            control_part="main_arm",
            max_steps=3,
            constraints={"velocity": 1.0, "acceleration": 2.0} if retimed else None,
        )
    )
    # Float64 callers exercise the native float32 conversion as well as retiming.
    start = torch.linspace(
        -0.2, 0.2, batch_size * NUM_ARM_JOINTS, dtype=torch.float64
    ).reshape(batch_size, NUM_ARM_JOINTS)
    target = start.float() + 0.3
    result = planner.plan(
        [PlanState.from_qpos(target, move_type=MoveType.JOINT_MOVE)],
        NeuralPlanOptions(control_part="main_arm", start_qpos=start),
    )
    assert result.positions is not None
    error = (result.positions[:, 0].double() - start).abs().amax(dim=-1)
    assert bool((error < 1e-6).all())
    assert float((result.positions[:, -1] - result.positions[:, 0]).abs().max()) > 1e-6
    assert result.policy_rollout_positions.shape == (batch_size, 4, NUM_ARM_JOINTS)
    torch.testing.assert_close(result.policy_rollout_positions[:, 0], start.float())
    assert result.policy_rollout_lengths.tolist() == [4] * batch_size


@pytest.mark.parametrize("batch_size", [1, 3])
def test_rollout_feedback_and_returned_poses_follow_each_joint_state(
    tmp_path, monkeypatch, batch_size
):
    """Closed-loop observations and returned poses describe the current state."""
    fake_sim = FakeSimulationManager()
    fake_sim.robot.num_instances = batch_size
    offset = torch.tensor([0.2, -0.1, 0.3])
    observations = []

    def analytic_fk(qpos, **kwargs):
        poses = torch.eye(4).repeat(qpos.shape[0], 1, 1)
        poses[:, :3, 3] = qpos[:, :3] + offset
        return poses

    def feedback_action(self, obs):
        observations.append(obs.clone())
        action = torch.zeros(obs.shape[0], NUM_ARM_JOINTS)
        action[:, :3] = 0.5 * (goal_position - obs[:, 7:10]) / 0.1
        return action

    monkeypatch.setattr(fake_sim.robot, "compute_fk", analytic_fk)
    monkeypatch.setattr(FakeOnnxPolicy, "__call__", feedback_action)
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            control_part="main_arm",
            action_scale=0.1,
            pos_eps=0.0001,
            max_steps=20,
        )
    )
    retained = []
    for direction in (1.0, -1.0):
        observations.clear()
        start = torch.zeros(batch_size, NUM_ARM_JOINTS)
        start[:, :3] = direction * 0.01
        distance = torch.linspace(0.03, 0.09, batch_size)
        goal_position = start[:, :3] + offset
        goal_position[:, 0] += direction * distance
        targets = torch.eye(4).repeat(batch_size, 1, 1)
        targets[:, :3, 3] = goal_position
        result = planner.plan(
            [PlanState.from_xpos(targets)],
            NeuralPlanOptions(
                control_part="main_arm",
                start_qpos=start,
                waypoint_modalities=["position"],
            ),
        )
        assert result.success.tolist() == [True] * batch_size
        observed = torch.stack(observations, dim=1)
        torch.testing.assert_close(observed[..., :7], result.positions[:, :-1])
        torch.testing.assert_close(
            observed[..., 7:10], result.positions[:, :-1, :3] + offset
        )
        torch.testing.assert_close(
            result.xpos_list[..., :3, 3], result.positions[..., :3] + offset
        )
        assert (
            float((result.xpos_list[:, -1, :3, 3] - goal_position).norm(dim=-1).max())
            <= 0.0001
        )
        for old, positions, poses in retained:
            torch.testing.assert_close(old.positions, positions)
            torch.testing.assert_close(old.xpos_list, poses)
        retained.append((result, result.positions.clone(), result.xpos_list.clone()))


@pytest.mark.parametrize("intermediate_orientation", [True, False])
def test_explicit_modalities_preserve_all_four_masks(
    tmp_path, monkeypatch, intermediate_orientation
):
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            control_part="main_arm",
            intermediate_orientation=intermediate_orientation,
        )
    )
    poses = [PlanState.from_xpos(torch.eye(4)[None]) for _ in range(3)]
    targets = [
        PlanState.from_qpos(torch.zeros(1, 7), move_type=MoveType.JOINT_MOVE),
        *poses,
    ]
    parsed = planner._parse_waypoints(
        targets, ["joint", "position", "orientation", "pose"]
    )
    assert parsed[4][0, :4].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert parsed[5][0, :4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert parsed[6][0, :4].tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="match the target count"):
        planner._parse_waypoints(targets, ["joint"])
    with pytest.raises(ValueError, match="move_type"):
        planner._parse_waypoints(targets, ["pose"] * 4)


def test_position_only_cannot_implicitly_constrain_an_offset_tcp(tmp_path, monkeypatch):
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    offset = torch.eye(4)
    offset[2, 3] = 0.1
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            control_part="main_arm",
            runtime_tcp_from_policy_tcp=offset.tolist(),
        )
    )
    with pytest.raises(ValueError, match="origins to coincide"):
        planner._parse_waypoints(
            [PlanState.from_xpos(torch.eye(4)[None])], ["position"]
        )


@pytest.mark.parametrize(
    "error,expected", [(0.005, True), (0.01, True), (0.0101, False)]
)
def test_joint_reach_uses_inclusive_maximum_error(error, expected):
    planner = NeuralPlanner.__new__(NeuralPlanner)
    planner.device = torch.device("cpu")
    planner._num_waypoints = 1
    planner._pos_eps, planner._rot_eps, planner._joint_eps = 0.01, 0.1, 0.01
    q = torch.full((1, NUM_ARM_JOINTS), error)
    ee = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    pose_mask = torch.zeros(1, 1)
    reached = planner._is_active_reached(
        q,
        ee,
        torch.zeros(1, 1, 3),
        ee[:, None, 3:],
        torch.zeros(1, 1, NUM_ARM_JOINTS),
        pose_mask,
        pose_mask,
        torch.ones(1, 1),
        torch.zeros(1, dtype=torch.long),
    )
    assert reached.item() is expected


@pytest.mark.no_sim
def test_onnx_runtime_respects_host_thread_budget(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="obs", shape=[None, 32])],
        get_outputs=lambda: [SimpleNamespace(name="action", shape=[None, 7])],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={}),
        get_providers=lambda: ["CPUExecutionProvider"],
    )
    received = {}

    def create_session(path, *, sess_options, providers):
        received["options"] = sess_options
        return session

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            InferenceSession=create_session,
        ),
    )
    messages = []
    monkeypatch.setattr(neural_planner_module.logger, "log_info", messages.append)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 3)
    _REAL_ONNX_POLICY(tmp_path / "policy.onnx")
    assert messages == ["NMG ONNX execution providers: CPUExecutionProvider"]
    assert received["options"].intra_op_num_threads == 3
    assert received["options"].inter_op_num_threads == 1


@pytest.mark.parametrize("retime", [False, True])
def test_shared_policy_uses_each_arms_frame_for_targets_and_rollout(
    tmp_path, monkeypatch, retime
):
    """Equivalent local requests have identical policy inputs after arm switches."""
    bases = {
        "left_arm": torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.5],
                [0.0, -1.0, 0.0, 0.3],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "right_arm": torch.tensor(
            [
                [0.0, -1.0, 0.0, -0.4],
                [1.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.8],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }
    local_pose = torch.eye(4)
    local_pose[:3, 3] = torch.tensor([0.3, 0.1, 0.2])

    class MountedRobot(FakeRobot):
        def compute_fk(self, qpos, name=None, to_matrix=False, **kwargs):
            return (bases[name] @ local_pose).expand(qpos.shape[0], -1, -1).clone()

        def compute_batch_fk(self, qpos, name=None, **kwargs):
            return (bases[name] @ local_pose).expand(*qpos.shape[:2], -1, -1).clone()

    manager = FakeSimulationManager()
    manager.robot = MountedRobot()
    monkeypatch.setattr(SimulationManager, "get_instance", lambda *args: manager)
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            num_waypoints=NUM_WAYPOINTS,
            max_steps=1,
            policy_frame_from_world_by_control_part={
                part: torch.linalg.inv(base).tolist() for part, base in bases.items()
            },
            constraints={"velocity": 0.2, "acceleration": 0.5} if retime else None,
        )
    )
    observations = []
    for part in ("left_arm", "right_arm", "left_arm"):
        target = (bases[part] @ local_pose).unsqueeze(0)
        result = planner.plan(
            [PlanState.from_xpos(target)],
            NeuralPlanOptions(control_part=part, start_qpos=torch.zeros(1, 7)),
        )
        assert result.success.tolist() == [True]
        torch.testing.assert_close(result.xpos_list[:, -1], target)
        observations.append(planner._policy.last_obs)
    torch.testing.assert_close(observations[0], observations[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(observations[0], observations[2], atol=1e-6, rtol=1e-6)
    # EEF translation is expressed in the original single-arm training frame.
    torch.testing.assert_close(observations[0][0, 7:10], local_pose[:3, 3])

    with pytest.raises(ValueError, match="No NMG policy-frame transform"):
        planner.plan(
            [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
            NeuralPlanOptions(
                control_part="unmapped_arm", start_qpos=torch.zeros(1, 7)
            ),
        )


@pytest.mark.parametrize("retime", [False, True])
def test_packed_arms_share_inference_and_match_independent_rollouts(
    tmp_path, monkeypatch, retime
):
    """Two scene rows × two arms preserve different goals, budgets and failures."""
    bases = {
        "left_arm": torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.5],
                [0.0, -1.0, 0.0, 0.3],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "right_arm": torch.tensor(
            [
                [0.0, -1.0, 0.0, -0.4],
                [1.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.8],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    class StepPolicy(FakeOnnxPolicy):
        fixed_batch_size = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observations = []

        def __call__(self, obs):
            self.observations.append(obs.clone())
            return torch.full((obs.shape[0], 7), 0.5)

    class MountedRobot(FakeRobot):
        num_instances = 2

        def compute_fk(self, qpos, name=None, **kwargs):
            # The simulator has two scene rows, not four packed actor rows.
            assert qpos.shape == (2, 7)
            local = torch.eye(4).repeat(2, 1, 1)
            local[:, :3, 3] = qpos[:, :3]
            return bases[name] @ local

        def compute_batch_fk(self, qpos, name=None, **kwargs):
            assert qpos.shape[0] == 2
            local = torch.eye(4).repeat(*qpos.shape[:2], 1, 1)
            local[..., :3, 3] = qpos[..., :3]
            return bases[name] @ local

    monkeypatch.setattr(neural_planner_module, "_OnnxPolicy", StepPolicy)
    manager = FakeSimulationManager()
    manager.robot = MountedRobot()
    monkeypatch.setattr(SimulationManager, "get_instance", lambda *args: manager)
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            batch_control_parts=True,
            num_waypoints=NUM_WAYPOINTS,
            action_scale=0.2,
            policy_frame_from_world_by_control_part={
                part: torch.linalg.inv(base).tolist() for part, base in bases.items()
            },
            constraints={"velocity": 2.0, "acceleration": 4.0} if retime else None,
        )
    )
    pose_goals = torch.eye(4).repeat(2, 1, 1)
    pose_goals[:, :3, 3] = torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
    targets = {
        "left_arm": [PlanState.from_qpos(torch.tensor([[0.1] * 7, [0.3] * 7]))],
        "right_arm": [
            PlanState.from_qpos(torch.tensor([[0.2] * 7, [0.1] * 7])),
            PlanState.from_xpos(bases["right_arm"] @ pose_goals),
        ],
    }
    options = {
        "left_arm": NeuralPlanOptions(
            control_part="left_arm", start_qpos=torch.zeros(2, 7), max_steps=2
        ),
        "right_arm": NeuralPlanOptions(
            control_part="right_arm", start_qpos=torch.zeros(2, 7), max_steps=4
        ),
    }
    packed = planner.plan_multi_part(targets, options)
    observed = planner._policy.observations
    assert len(observed) == 4
    assert all(obs.shape == (4, OBS_DIM) for obs in observed)
    torch.testing.assert_close(
        observed[0][:, 7:10], torch.zeros(4, 3), atol=1e-6, rtol=0
    )
    # The left arm's converged first row holds at .1; its exhausted second row
    # holds at .2 while the right arm continues through its fourth policy step.
    torch.testing.assert_close(
        observed[-1][:, :7], torch.tensor([[0.1] * 7, [0.2] * 7, [0.3] * 7, [0.3] * 7])
    )
    for result in packed.values():
        assert result.policy_batching == {
            "scene_batch_size": 2,
            "control_part_count": 2,
            "actor_batch_size": 4,
            "policy_forward_calls": 4,
        }
        assert result.success.tolist() == [True, False]
    assert packed["left_arm"].policy_rollout_lengths.tolist() == [2, 3]
    assert packed["right_arm"].policy_rollout_lengths.tolist() == [5, 5]
    assert packed["left_arm"].policy_rollout_positions.shape == (2, 3, 7)
    assert packed["right_arm"].policy_rollout_positions.shape == (2, 5, 7)

    planner._policy.observations.clear()
    separate = {part: planner.plan(targets[part], options[part]) for part in targets}
    assert len(planner._policy.observations) == 6
    for part in targets:
        for name in (
            "success",
            "positions",
            "velocities",
            "accelerations",
            "dt",
            "xpos_list",
            "policy_rollout_positions",
            "policy_rollout_lengths",
        ):
            torch.testing.assert_close(
                getattr(packed[part], name), getattr(separate[part], name)
            )


def test_packed_arms_reject_fixed_single_row_policy(tmp_path, monkeypatch):
    manager = FakeSimulationManager()
    monkeypatch.setattr(SimulationManager, "get_instance", lambda *args: manager)
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            batch_control_parts=True,
        )
    )
    targets = {
        part: [PlanState.from_qpos(torch.zeros(1, 7))]
        for part in ("left_arm", "right_arm")
    }
    options = {
        part: NeuralPlanOptions(start_qpos=torch.zeros(1, 7)) for part in targets
    }
    with pytest.raises(ValueError, match="Export a dynamic-batch policy"):
        planner.plan_multi_part(targets, options)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "providers,use_tf32,device,expected_providers,expected_options",
    [
        (None, None, None, None, None),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            None,
            None,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            None,
        ),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            False,
            None,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"use_tf32": "0"}, {}],
        ),
        (
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            True,
            None,
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            [{}, {"use_tf32": "1"}],
        ),
        (
            ["CPUExecutionProvider"],
            False,
            None,
            ["CPUExecutionProvider"],
            [{}],
        ),
        (
            None,
            False,
            None,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"use_tf32": "0"}, {}],
        ),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            False,
            torch.device("cuda:0"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "0", "use_tf32": "0"}, {}],
        ),
        (
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            False,
            torch.device("cuda:3"),
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            [{}, {"device_id": "3", "use_tf32": "0"}],
        ),
        (
            None,
            False,
            torch.device("cuda:2"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "2", "use_tf32": "0"}, {}],
        ),
        (
            None,
            None,
            torch.device("cuda:4"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "4"}, {}],
        ),
        (
            ["CPUExecutionProvider"],
            False,
            torch.device("cuda:2"),
            ["CPUExecutionProvider"],
            [{}],
        ),
        (
            None,
            False,
            torch.device("cuda"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "2", "use_tf32": "0"}, {}],
        ),
        (None, None, torch.device("cpu"), None, None),
    ],
)
def test_onnx_precision_is_explicit_and_provider_scoped(
    monkeypatch,
    tmp_path,
    providers,
    use_tf32,
    device,
    expected_providers,
    expected_options,
):
    """Freeze CUDA arithmetic without changing CPU options or historical defaults."""
    import sys
    from types import SimpleNamespace

    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="obs", shape=[None, 32])],
        get_outputs=lambda: [SimpleNamespace(name="action", shape=[None, 7])],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={}),
        get_providers=lambda: expected_providers or ["CPUExecutionProvider"],
    )
    received = {}

    def create_session(path, **kwargs):
        received.update(kwargs)
        return session

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            InferenceSession=create_session,
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    _REAL_ONNX_POLICY(
        tmp_path / "policy.onnx", providers, cuda_use_tf32=use_tf32, device=device
    )
    assert received["providers"] == expected_providers
    assert received.get("provider_options") == expected_options
    if expected_options is None:
        assert "provider_options" not in received


@pytest.mark.no_sim
@pytest.mark.parametrize("invalid", [0, 1, "false", 1.0])
def test_onnx_precision_rejects_non_boolean_values(tmp_path, invalid):
    with pytest.raises(TypeError, match="onnx_cuda_use_tf32 must be a bool or None"):
        _REAL_ONNX_POLICY(tmp_path / "policy.onnx", cuda_use_tf32=invalid)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda:0", "cuda:3"],
    ids=["host", "default_device", "other_device"],
)
def test_neural_planner_forwards_explicit_precision_and_device(
    tmp_path, monkeypatch, device
):
    manager = FakeSimulationManager()
    manager.robot.device = torch.device(device)
    monkeypatch.setattr(SimulationManager, "get_instance", lambda *args: manager)
    # Device propagation needs no CUDA tensors; frame conversion is covered
    # independently by the planner's CPU geometry tests.
    monkeypatch.setattr(
        NeuralPlanner, "_as_transform", lambda self, value, name: torch.eye(4)
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=_create_fake_onnx_model(tmp_path),
            control_part="main_arm",
            onnx_cuda_use_tf32=False,
        )
    )
    assert planner._policy.cuda_use_tf32 is False
    assert planner._policy.device == torch.device(device)


@pytest.mark.no_sim
def test_retimed_rollout_retains_limit_waypoints_on_a_coarse_clock(
    tmp_path, monkeypatch
):
    model_path = _create_fake_onnx_model(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls, instance_id=0: fake_sim),
    )
    planner = NeuralPlanner(
        NeuralPlannerCfg(
            robot_uid="fake_robot",
            onnx_model_path=model_path,
            control_part="main_arm",
            dt=1.0,
            constraints={"velocity": 2.0, "acceleration": 4.0},
        )
    )
    positions = torch.zeros(1, 3, NUM_ARM_JOINTS, dtype=torch.float64)
    positions[0, :, 0] = torch.tensor([0.8, 1.0, 1.0])
    positions[0, :, 1] = torch.tensor([0.0, 0.02, 0.04])
    # Both a plateau at the position cap and an interior goal must survive.
    joint_targets = torch.zeros(1, NUM_WAYPOINTS, NUM_ARM_JOINTS, dtype=torch.float64)
    joint_targets[:, :2] = positions[:, 1:]
    joint_mask = torch.zeros(1, NUM_WAYPOINTS)
    joint_mask[:, :2] = 1.0
    no_mask = torch.zeros_like(joint_mask)
    quaternions = torch.zeros(1, NUM_WAYPOINTS, 4)
    quaternions[..., 3] = 1.0
    waypoints = (
        torch.zeros(1, NUM_WAYPOINTS, 3),
        quaternions,
        joint_targets,
        no_mask,
        no_mask,
        joint_mask,
    )
    lower, upper = torch.full((NUM_ARM_JOINTS,), -1.0), torch.ones(NUM_ARM_JOINTS)
    result = planner._retimed_result(
        torch.tensor([True]),
        positions,
        "main_arm",
        path_lengths=torch.tensor([3]),
        waypoints=waypoints,
        episode_k=2,
        qpos_limits=(lower, upper),
    )
    assert result.success.tolist() == [True]
    torch.testing.assert_close(result.positions, positions)
    assert result.positions.max() == 1.0
    assert result.dt[0, 1:].min() > 0
    assert result.velocities.abs().max() <= 2.0 + 1e-8
    assert result.accelerations.abs().max() <= 4.0 + 1e-8
