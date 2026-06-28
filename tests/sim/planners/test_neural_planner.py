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
from embodichain.lab.sim.planners.neural_planner import (
    NeuralPlanOptions,
    _WaypointTransformerActor,
)
from embodichain.lab.sim.sim_manager import SimulationManager

NUM_ARM_JOINTS = 7
NUM_WAYPOINTS = 3
OBS_DIM = 28 + 9 * NUM_WAYPOINTS
HIDDEN_DIM = 32


def _create_fake_checkpoint(tmp_path) -> str:
    actor = _WaypointTransformerActor(
        obs_dim=OBS_DIM,
        action_dim=NUM_ARM_JOINTS,
        num_waypoints=NUM_WAYPOINTS,
        use_relative_obs=True,
        hidden_dim=HIDDEN_DIM,
        transformer_nhead=4,
        transformer_num_layers=1,
    )
    checkpoint = {
        "agent": {f"actor_mean.{k}": v for k, v in actor.state_dict().items()},
        "obs_normalizer": {
            "mean": torch.zeros(OBS_DIM),
            "var": torch.ones(OBS_DIM),
            "count": 1.0,
        },
        "args": {
            "policy_arch": "transformer",
            "hidden_dim": HIDDEN_DIM,
            "transformer_nhead": 4,
            "transformer_num_layers": 1,
            "transformer_ff_dim": 0,
            "waypoint_max": NUM_WAYPOINTS,
            "waypoint_use_relative_obs": True,
            "waypoint_intermediate_orientation": True,
            "max_episode_steps": 3,
            "waypoint_pos_threshold": 0.05,
            "waypoint_rot_threshold": 0.3,
        },
    }
    checkpoint_path = tmp_path / "fake_neural_planner.pt"
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


class FakeRobot:
    uid = "fake_robot"
    device = torch.device("cpu")
    num_instances = 1

    def get_qpos(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        return torch.zeros(self.num_instances, NUM_ARM_JOINTS)

    def get_qpos_limits(
        self, name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        batch = len(env_ids) if env_ids is not None else self.num_instances
        limits = torch.zeros(batch, NUM_ARM_JOINTS, 2)
        limits[..., 0] = -2.0
        limits[..., 1] = 2.0
        return limits

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        env_ids: list[int] | None = None,
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


def _patch_sim_manager(monkeypatch) -> None:
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )


def _make_motion_generator(tmp_path, monkeypatch) -> MotionGenerator:
    _patch_sim_manager(monkeypatch)
    checkpoint_path = _create_fake_checkpoint(tmp_path)
    return MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                checkpoint_path=checkpoint_path,
                control_part="main_arm",
            )
        )
    )


def test_neural_planner_is_registered():
    assert MotionGenerator._support_planner_dict["neural"][0] is NeuralPlanner
    assert MotionGenerator._support_planner_dict["neural"][1] is NeuralPlannerCfg
    assert MotionGenerator._support_planner_dict["neural_refine"][0] is NeuralPlanner
    assert MotionGenerator._support_planner_dict["neural_refine"][1] is NeuralPlannerCfg


def test_neural_planner_generate_with_fake_checkpoint(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)
    result = motion_generator.generate(
        target_states=[PlanState(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
        options=MotionGenOptions(
            plan_opts=NeuralPlanOptions(
                control_part="main_arm",
                start_qpos=torch.zeros(NUM_ARM_JOINTS),
            ),
        ),
    )

    assert result.success is True
    assert result.positions is not None
    assert result.positions.shape[-1] == NUM_ARM_JOINTS
    assert torch.isfinite(result.positions).all()
    assert result.xpos_list is not None
    assert result.xpos_list.shape[-2:] == (4, 4)


def test_neural_planner_uses_plan_opts_start_qpos(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)
    custom_qpos = torch.ones(NUM_ARM_JOINTS)
    result = motion_generator.generate(
        target_states=[PlanState(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
        options=MotionGenOptions(
            plan_opts=NeuralPlanOptions(
                control_part="main_arm",
                start_qpos=custom_qpos,
            ),
        ),
    )

    assert result.success is True
    assert torch.allclose(result.positions[0], custom_qpos)


def test_motion_generator_builds_default_neural_plan_options(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)
    custom_qpos = torch.full((NUM_ARM_JOINTS,), 0.25)

    result = motion_generator.generate(
        target_states=[PlanState(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
        options=MotionGenOptions(
            control_part="main_arm",
            start_qpos=custom_qpos,
        ),
    )

    assert result.success is True
    assert torch.allclose(result.positions[0], custom_qpos)


def test_motion_generator_does_not_mutate_user_plan_options(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)
    custom_qpos = torch.full((NUM_ARM_JOINTS,), 0.5)
    plan_opts = NeuralPlanOptions()

    result = motion_generator.generate(
        target_states=[PlanState(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
        options=MotionGenOptions(
            control_part="main_arm",
            start_qpos=custom_qpos,
            plan_opts=plan_opts,
        ),
    )

    assert result.success is True
    assert torch.allclose(result.positions[0], custom_qpos)
    assert plan_opts.control_part is None
    assert plan_opts.start_qpos is None


def test_neural_planner_rejects_short_start_qpos(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="policy expects"):
        motion_generator.generate(
            target_states=[PlanState(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
            options=MotionGenOptions(
                plan_opts=NeuralPlanOptions(
                    control_part="main_arm",
                    start_qpos=torch.zeros(NUM_ARM_JOINTS - 1),
                ),
            ),
        )


def test_neural_planner_rejects_joint_move(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="EEF_MOVE"):
        motion_generator.generate(
            target_states=[
                PlanState(move_type=MoveType.JOINT_MOVE, qpos=torch.zeros(7))
            ],
            options=MotionGenOptions(
                plan_opts=NeuralPlanOptions(
                    control_part="main_arm",
                    start_qpos=torch.zeros(NUM_ARM_JOINTS),
                ),
            ),
        )


def test_neural_planner_requires_env_id_for_multi_instance(tmp_path, monkeypatch):
    motion_generator = _make_motion_generator(tmp_path, monkeypatch)
    motion_generator.robot.num_instances = 2

    with pytest.raises(ValueError, match="env_id is required"):
        motion_generator.generate(
            target_states=[PlanState(move_type=MoveType.EEF_MOVE, xpos=torch.eye(4))],
            options=MotionGenOptions(
                control_part="main_arm",
                start_qpos=torch.zeros(NUM_ARM_JOINTS),
            ),
        )


def test_neural_planner_rejects_checkpoint_dof_mismatch(tmp_path, monkeypatch):
    _patch_sim_manager(monkeypatch)
    checkpoint_path = _create_fake_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="num_arm_joints=6"):
        MotionGenerator(
            cfg=MotionGenCfg(
                planner_cfg=NeuralPlannerCfg(
                    robot_uid="fake_robot",
                    checkpoint_path=checkpoint_path,
                    control_part="main_arm",
                    num_arm_joints=6,
                )
            )
        )
