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

"""Contract and numerical tests for the Franka waypoint NMG task."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("newton")

from embodichain.learning.rl.models import (
    waypoint_observation_dim,
    waypoint_observation_normalize_mask,
)
from embodichain_tasks.manipulation.franka_waypoint import (
    FrankaWaypointNMGEnv,
    waypoint_obs_dim,
    waypoint_obs_normalize_mask,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "embodichain_tasks/configs/tasks/manipulation/franka_waypoint/agents/apg.yaml"
)


def test_franka_waypoint_observation_contract_matches_transformer() -> None:
    """The task and generic Transformer must parse the same flat layout."""
    for num_waypoints in (1, 3, 8):
        for use_relative_observations in (False, True):
            assert waypoint_obs_dim(
                num_waypoints,
                use_relative_observations,
            ) == waypoint_observation_dim(
                num_waypoints,
                use_relative_observations,
            )
            assert torch.equal(
                waypoint_obs_normalize_mask(
                    num_waypoints,
                    use_relative_observations,
                ),
                waypoint_observation_normalize_mask(
                    num_waypoints,
                    use_relative_observations,
                ),
            )


def test_franka_waypoint_rollout_schedule_matches_reference() -> None:
    """Variable K cycles uniformly and rotates the next cycle's first stratum."""
    env = object.__new__(FrankaWaypointNMGEnv)
    env._rollout_fixed_num_waypoints = 0
    env.waypoint_min_num_waypoints = 1
    env.num_waypoints = 8
    env.waypoint_steps_per_waypoint = 30

    specs = [env.prepare_differentiable_rollout(index) for index in range(16)]

    assert [spec.metadata["waypoint_count"] for spec in specs] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        1.0,
    ]
    assert [spec.num_steps for spec in specs[:8]] == [
        30,
        60,
        90,
        120,
        150,
        180,
        210,
        240,
    ]
    assert [spec.objective_scale for spec in specs[:8]] == pytest.approx(
        [1.0 / waypoint_count for waypoint_count in range(1, 9)]
    )


def test_franka_waypoint_apg_config_matches_pr6_budget() -> None:
    """The bundled config preserves PR #6 tolerances and optimizer-step budget."""
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    trainer = config["trainer"]
    env_cfg = trainer["learning_env"]["cfg"]
    algorithm_cfg = config["algorithm"]["cfg"]
    network_cfg = config["policy"]["actor"]["network_cfg"]

    cycle_steps = (
        trainer["num_envs"]
        * env_cfg["waypoint_steps_per_waypoint"]
        * sum(
            range(env_cfg["waypoint_min_num_waypoints"], env_cfg["num_waypoints"] + 1)
        )
    )
    num_cycles, remainder = divmod(trainer["total_timesteps"], cycle_steps)

    assert remainder == 0
    assert num_cycles == 544
    assert trainer["iterations"] == num_cycles * 8 == 4352
    assert trainer["rollout_mode"] == "complete"
    assert trainer["torch_deterministic"] is True
    assert trainer["deterministic_actions"] is True
    assert trainer["action_adjoint_max_norm"] == 1.0
    assert trainer["normalize_observations"] is True
    assert env_cfg["waypoint_pos_threshold"] == 0.01
    assert env_cfg["waypoint_rot_threshold"] == 0.1
    assert env_cfg["waypoint_joint_threshold"] == 0.02
    assert network_cfg == {
        "num_waypoints": 8,
        "joint_dim": 7,
        "use_relative_observations": True,
        "hidden_dim": 128,
        "num_attention_heads": 4,
        "num_layers": 2,
        "feedforward_dim": 0,
    }
    assert algorithm_cfg["optimizer"]["learning_rate"] == 2.5e-4
    assert algorithm_cfg["optimizer"]["kwargs"]["eps"] == 1.0e-5
    assert algorithm_cfg["gamma"] == 1.0
    assert algorithm_cfg["max_grad_norm"] == 0.5
    assert algorithm_cfg["max_grad_norm_before_clip"] == 10000.0


@pytest.mark.slow
def test_franka_cartesian_reward_has_finite_action_gradient() -> None:
    """The real Newton/Warp pose path retains its action gradient."""
    env = FrankaWaypointNMGEnv(
        num_envs=2,
        device="cpu",
        num_waypoints=1,
        waypoint_min_num_waypoints=1,
        waypoint_fixed_num_waypoints=1,
        waypoint_steps_per_waypoint=30,
        waypoint_pos_weight=0.0,
        waypoint_rot_precision_weight=0.02,
        waypoint_pose_constraint_weight=0.002,
    )
    try:
        observation, _ = env.reset(seed=11)
        action = torch.zeros((2, 7), requires_grad=True)

        next_observation, reward, _, _, _ = env.step(action)
        reward.sum().backward()

        assert observation.shape == next_observation.shape == (2, 62)
        assert action.grad is not None
        assert torch.isfinite(action.grad).all()
        assert action.grad.abs().sum() > 0.0
    finally:
        env.detach_state()
        env.close()
