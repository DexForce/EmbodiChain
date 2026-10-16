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
