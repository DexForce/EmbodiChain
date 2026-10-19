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

import argparse

import pytest
import torch

from examples.ods.simple_bc_demo.models import (
    ChunkBC,
    ChunkState,
    pose_targets,
    rotation_from_6d,
)
from examples.ods.simple_bc_demo.checkpoint import save_checkpoint, load_policy


def _observations(batch: int, time: int) -> torch.Tensor:
    obs = torch.randn(batch, time, 51) * 0.1
    obs[..., 21:27] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return obs


def _model() -> ChunkBC:
    return ChunkBC(
        torch.zeros(51), torch.ones(51), torch.zeros(10), torch.ones(10), 4, 2, 0
    ).eval()


def test_optimizer_updates_reduce_masked_chunk_loss() -> None:
    torch.manual_seed(5)
    model = _model().train()
    observations = _observations(2, 5)
    actions = torch.zeros(2, 5, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    loss_mask = valid.clone()
    loss_mask[:, 0] = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def loss() -> torch.Tensor:
        return sum(
            model.loss(observations, actions, loss_mask=loss_mask, valid_mask=valid)
        )

    initial = loss().item()
    for _ in range(8):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    assert loss().item() < initial


def test_absolute_targets_reproduce_causal_relative_requests() -> None:
    obs = _observations(5, 1)[:, 0]
    requests = torch.randn(5, 7).clamp(-0.5, 0.5)
    target = pose_targets(obs, requests)
    model = _model()
    recovered, _ = model.action(
        obs, ChunkState(obs, target[:, None].repeat(1, 4, 1), 0)
    )
    torch.testing.assert_close(recovered, requests, atol=2e-5, rtol=2e-5)


def test_degenerate_rotation_predictions_are_finite_proper_rotations() -> None:
    values = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]
    )
    rotations = rotation_from_6d(values)
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations, torch.eye(3).repeat(2, 1, 1)
    )
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2))


def test_chunk_mask_ignores_tail_and_initial_history_padding() -> None:
    model = _model()
    obs = _observations(2, 5)
    actions = torch.rand(2, 5, 7) * 0.1
    valid = torch.tensor(
        [[False, True, True, False, False], [True, True, True, True, False]]
    )
    mask = valid.clone()
    mask[:, 0] = False
    first = model.loss(obs, actions, loss_mask=mask, valid_mask=valid)
    obs[~valid] = 1000
    actions[~valid] = 1000
    second = model.loss(obs, actions, loss_mask=mask, valid_mask=valid)
    torch.testing.assert_close(first, second)
    wrong = mask.clone()
    wrong[:, 0] = True
    with pytest.raises(ValueError, match="real current/future"):
        model.loss(obs, actions, loss_mask=wrong, valid_mask=valid)


def test_chunk_replanning_uses_recent_measurements_and_reset_clears_plan(
    monkeypatch,
) -> None:
    model = _model()
    seen = []

    def predict(history):
        seen.append(history.clone())
        return torch.zeros(len(history), 4, 10)

    monkeypatch.setattr(model, "forward", predict)
    obs = _observations(1, 4)[0]
    _, hidden = model.action(obs[0:1])
    _, hidden = model.action(obs[1:2], hidden)
    assert len(seen) == 1
    _, hidden = model.action(obs[2:3], hidden)
    torch.testing.assert_close(seen[-1], obs[1:3][None])
    model.action(obs[3:4], None)
    torch.testing.assert_close(seen[-1], obs[3:4].repeat(2, 1)[None])


def test_chunk_checkpoint_preserves_rollout_and_training_configuration(
    tmp_path,
) -> None:
    model = _model()
    path = tmp_path / "chunk.pt"
    save_checkpoint(
        path,
        model,
        torch.optim.AdamW(model.parameters()),
        9,
        argparse.Namespace(train_device="cpu"),
        123,
    )
    restored, state = load_policy(str(path), "cpu")
    assert isinstance(restored, ChunkBC)
    assert state["policy_config"]["action_horizon"] == 2
    assert state["step"] == 9
    left = right = None
    for obs in _observations(1, 7).transpose(0, 1):
        a, left = model.action(obs, left)
        b, right = restored.action(obs, right)
        torch.testing.assert_close(a, b)


def test_chunk_normalization_digest_includes_absolute_target_scales() -> None:
    from examples.ods.simple_bc_demo.trainer import (
        _normalization_values,
        _state_digest,
    )

    model = _model()
    original = _state_digest(_normalization_values(model))
    model.target_std[0] *= 2
    assert original != _state_digest(_normalization_values(model))
