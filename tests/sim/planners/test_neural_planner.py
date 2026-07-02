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


def test_neural_planner_generate_with_fake_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = _create_fake_checkpoint(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                checkpoint_path=checkpoint_path,
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
    checkpoint_path = _create_fake_checkpoint(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                checkpoint_path=checkpoint_path,
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
    checkpoint_path = _create_fake_checkpoint(tmp_path)
    fake_sim = FakeSimulationManager()
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )

    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(
            planner_cfg=NeuralPlannerCfg(
                robot_uid="fake_robot",
                checkpoint_path=checkpoint_path,
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
