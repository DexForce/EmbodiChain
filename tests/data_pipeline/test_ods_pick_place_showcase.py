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

from examples.data_pipeline.ods_pick_place_showcase import (
    RecurrentBC,
    load_policy,
    save_checkpoint,
    _canonical_observations,
    _observation_statistics,
)


def _model() -> RecurrentBC:
    torch.manual_seed(5)
    return RecurrentBC(torch.zeros(28), torch.ones(28), -torch.ones(8), torch.ones(8))


def test_burn_in_has_no_loss_gradient() -> None:
    model = _model()
    obs = torch.randn(2, 32, 28, requires_grad=True)
    arm, hand = model.loss(obs, torch.zeros(2, 32, 8))
    (arm + 5 * hand).backward()
    assert obs.grad[:, :8].count_nonzero() == 0
    assert obs.grad[:, 8:].abs().sum() > 0


def test_optimizer_updates_reduce_deterministic_loss() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    obs, target = torch.zeros(2, 32, 28), torch.full((2, 32, 8), 0.3)
    initial = sum(model.loss(obs, target)).item()
    for _ in range(12):
        optimizer.zero_grad()
        sum(model.loss(obs, target)).backward()
        optimizer.step()
    assert sum(model.loss(obs, target)).item() < initial


def test_checkpoint_round_trip_preserves_recurrent_predictions(tmp_path) -> None:
    model = _model().eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 7, argparse.Namespace(mode="online"), 123)
    loaded, state = load_policy(str(path), "cpu")
    obs = torch.randn(2, 28)
    a, h = model.action(obs, None)
    restored_a, restored_h = loaded.action(obs, None)
    torch.testing.assert_close(a, restored_a)
    torch.testing.assert_close(h, restored_h)
    assert state["step"] == 7 and state["run_id"] == 123
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, optimizer, 8, argparse.Namespace(), 123)


def test_policy_rejects_wrong_width_and_invalid_normalization() -> None:
    with pytest.raises(ValueError):
        RecurrentBC(torch.zeros(27), torch.ones(27), -torch.ones(8), torch.ones(8))
    with pytest.raises(ValueError):
        RecurrentBC(torch.zeros(28), torch.zeros(28), -torch.ones(8), torch.ones(8))


def test_equivalent_quaternions_produce_identical_recurrent_actions() -> None:
    model = _model()
    obs = torch.randn(4, 28)
    equivalent = obs.clone()
    equivalent[:, 21:25] *= -1
    action, hidden = model.action(obs, None)
    other_action, other_hidden = model.action(equivalent, None)
    torch.testing.assert_close(action, other_action)
    torch.testing.assert_close(hidden, other_hidden)


def test_nearly_constant_features_keep_physical_normalization_scales() -> None:
    values = torch.zeros(12, 28)
    values[:, 24] = 1
    mean, std = _observation_statistics(values)
    assert std[9:16].min() >= 0.5
    torch.testing.assert_close(mean[21:25], torch.zeros(4))
    torch.testing.assert_close(std[21:25], torch.ones(4))
    equivalent = values.clone()
    equivalent[:, 24] = -1
    torch.testing.assert_close(
        _canonical_observations(values), _canonical_observations(equivalent)
    )


def test_initial_actions_receive_supervision_with_causal_history() -> None:
    model = _model()
    obs = torch.randn(2, 24, 28, requires_grad=True)
    mask = torch.ones(2, 24, dtype=torch.bool)
    arm, hand = model.loss(obs, torch.zeros(2, 24, 8), loss_mask=mask)
    (arm + 5 * hand).backward()
    assert obs.grad[:, 0].abs().sum() > 0


def test_full_history_prediction_matches_stepwise_inference_and_ignores_padding() -> (
    None
):
    model = _model()
    obs = torch.randn(2, 30, 28)
    full, _ = model(obs)
    hidden = None
    for index in range(obs.shape[1]):
        predicted, hidden = model(obs[:, index : index + 1], hidden)
        torch.testing.assert_close(predicted[:, 0], full[:, index])
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, 4:12], mask[1, 22:] = True, True
    actions = torch.zeros(2, 30, 8)
    original = model.loss(obs, actions, loss_mask=mask)
    obs[0, 12:], actions[0, 12:] = 1000, -1000
    changed = model.loss(obs, actions, loss_mask=mask)
    torch.testing.assert_close(original, changed)


def test_legacy_checkpoint_preserves_original_observation_contract(tmp_path) -> None:
    model = RecurrentBC(
        torch.zeros(28),
        torch.ones(28),
        -torch.ones(8),
        torch.ones(8),
        observation_mode="legacy",
    )
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict()}, path)
    restored, _ = load_policy(str(path), "cpu")
    obs = torch.randn(2, 28)
    torch.testing.assert_close(model.action(obs, None), restored.action(obs, None))
    assert restored.observation_mode == "legacy"


def test_failed_producer_is_not_retried_as_an_empty_sampling_window() -> None:
    from types import SimpleNamespace
    from examples.data_pipeline.ods_pick_place_showcase import _sample
    from embodichain.data_pipeline.engine.data import OnlineDataEngineState

    error = RuntimeError("No unlocked valid chunk: terminal producer failure")

    def fail(*args):
        raise error

    source = SimpleNamespace(state=OnlineDataEngineState.FAILED, sample_batch=fail)
    with pytest.raises(RuntimeError) as caught:
        _sample(source, 2, 32)
    assert caught.value is error
