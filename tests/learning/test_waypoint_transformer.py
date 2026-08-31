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

"""Tests for unified Cartesian/joint waypoint Transformer models."""

from __future__ import annotations

import torch

from embodichain.learning.rl.models import (
    WaypointTransformerActor,
    WaypointTransformerCritic,
    build_model_from_cfg,
    parse_waypoint_observation,
    waypoint_observation_dim,
    waypoint_observation_normalize_mask,
)

_NUM_WAYPOINTS = 4
_OBSERVATION_DIM = waypoint_observation_dim(_NUM_WAYPOINTS, True)


def _valid_observation(batch_size: int = 3) -> torch.Tensor:
    observation = torch.zeros(batch_size, _OBSERVATION_DIM)
    cursor = 7 + 7 + _NUM_WAYPOINTS * (3 + 4 + 7)
    observation[:, cursor] = 1.0
    cursor += _NUM_WAYPOINTS
    observation[:, cursor : cursor + _NUM_WAYPOINTS] = 1.0
    cursor += _NUM_WAYPOINTS
    observation[:, cursor : cursor + _NUM_WAYPOINTS] = 1.0
    cursor += _NUM_WAYPOINTS
    observation[:, cursor : cursor + _NUM_WAYPOINTS] = 1.0
    return observation


def _actor() -> WaypointTransformerActor:
    return WaypointTransformerActor(
        observation_dim=_OBSERVATION_DIM,
        action_dim=7,
        num_waypoints=_NUM_WAYPOINTS,
        hidden_dim=64,
        num_attention_heads=4,
        num_layers=1,
    )


def test_waypoint_actor_and_critic_have_expected_shapes_and_gradients() -> None:
    torch.manual_seed(7)
    actor = _actor().train()
    critic = WaypointTransformerCritic(
        observation_dim=_OBSERVATION_DIM,
        num_waypoints=_NUM_WAYPOINTS,
        hidden_dim=64,
        num_attention_heads=4,
        num_layers=1,
    ).train()
    observation = _valid_observation()

    action = actor(observation)
    value = critic(observation)
    (action.square().mean() + value.square().mean()).backward()

    assert action.shape == (3, 7)
    assert value.shape == (3, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(value).all()
    assert actor.action_token.grad is not None
    assert critic.action_token.grad is not None


def test_waypoint_actor_uses_future_valid_waypoints() -> None:
    torch.manual_seed(11)
    actor = _actor().eval()
    base = _valid_observation(batch_size=1)
    changed = base.clone()
    # WP3 position is a future token while WP0 remains active.
    position_start = 7 + 7
    changed[:, position_start + 3 * 3 : position_start + 3 * 4] = 5.0

    with torch.no_grad():
        base_action = actor(base)
        changed_action = actor(changed)

    assert not torch.allclose(base_action, changed_action)


def test_waypoint_parser_clamps_active_id_to_last_valid_slot() -> None:
    observation = _valid_observation(batch_size=1)
    cursor = 7 + 7 + _NUM_WAYPOINTS * (3 + 4 + 7)
    observation[:, cursor : cursor + _NUM_WAYPOINTS] = 0.0
    observation[:, cursor + 3] = 1.0
    cursor += _NUM_WAYPOINTS
    observation[:, cursor : cursor + _NUM_WAYPOINTS] = 0.0
    observation[:, cursor : cursor + 2] = 1.0

    fields = parse_waypoint_observation(observation, _NUM_WAYPOINTS, True)

    assert torch.equal(fields["active_id"], torch.tensor([1]))


def test_waypoint_normalization_mask_excludes_all_semantic_fields() -> None:
    mask = waypoint_observation_normalize_mask(_NUM_WAYPOINTS, True)

    assert mask.shape == (_OBSERVATION_DIM,)
    assert int((~mask).sum()) == 6 * _NUM_WAYPOINTS + 7 * _NUM_WAYPOINTS


def test_waypoint_transformer_builds_from_learning_config() -> None:
    module_config = {
        "type": "waypoint_transformer",
        "network_cfg": {
            "num_waypoints": _NUM_WAYPOINTS,
            "hidden_dim": 64,
            "num_attention_heads": 4,
            "num_layers": 1,
        },
    }
    actor = build_model_from_cfg(
        module_config,
        _OBSERVATION_DIM,
        7,
        role="actor",
    )
    critic = build_model_from_cfg(
        module_config,
        _OBSERVATION_DIM,
        1,
        role="critic",
    )

    assert isinstance(actor, WaypointTransformerActor)
    assert isinstance(critic, WaypointTransformerCritic)
