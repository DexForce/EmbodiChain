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

"""Pinned acceptance settings reject silent optimizer or recovery changes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest
import torch

from experiments.learning.sl import train_bc
from experiments.learning.sl.pick_place import CAMERAS, TARGET_SEMANTICS


def _config(tmp_path):
    return {
        "model": {
            "observation_dim": 27,
            "target_dim": 10,
            "history_length": 2,
            "prediction_horizon": 32,
            "hidden_dims": [512, 512],
            "cameras": list(CAMERAS),
            "target_semantics": TARGET_SEMANTICS,
        },
        "optimizer": deepcopy(train_bc.FORMAL_OPTIMIZER),
        "objective": {"type": "masked_normalized_target_mse"},
        "evaluation": {
            "development_scenes": 20,
            "development_threshold": 0.9,
            "test_scenes": 100,
            "test_threshold": 0.8,
        },
        "data": {"manifest": str(tmp_path / "manifest.json")},
        "trainer": {
            "total_updates": 20000,
            "batch_size": 64,
            "validation_interval": 2000,
            "output_dir": str(tmp_path / "run/training"),
        },
    }


def test_nonmatching_optimizer_is_rejected_before_corpus_or_model_access(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    config["optimizer"].update(
        type="sgd", learning_rate=0.1, weight_decay=0.5, max_grad_norm=9
    )
    monkeypatch.setattr(
        train_bc,
        "source",
        lambda *args, **kwargs: pytest.fail(
            "Rejected optimizer must not open a corpus"
        ),
    )
    with pytest.raises(ValueError, match="optimizer/schedule"):
        train_bc.build_trainer(config)
    assert not (tmp_path / "run").exists()


def test_checkpoint_binds_pinned_optimizer_and_schedule_before_loading_weights(
    tmp_path, monkeypatch
):
    environment = {"physics": "default"}
    (tmp_path / "environment.json").write_text(json.dumps(environment))

    class Dataset:
        path = tmp_path / "manifest.json"
        episodes = [{"sha256": str(index)} for index in range(1024)]
        source_id = "fixture-qualified-corpus"
        manifest = {
            "config_sha256": hashlib.sha256(
                json.dumps(environment, sort_keys=True).encode()
            ).hexdigest()
        }

        def close(self):
            pass

    monkeypatch.setattr(train_bc, "source", lambda path: Dataset())
    monkeypatch.setattr(
        train_bc,
        "frozen_statistics",
        lambda dataset: {
            "observation_mean": torch.zeros(27),
            "observation_std": torch.ones(27),
            "target_mean": torch.zeros(10),
            "target_std": torch.ones(10),
        },
    )
    trainer = train_bc.build_trainer(_config(tmp_path))
    assert trainer.learner.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 500)
    assert trainer.learner.optimizer.param_groups[0]["weight_decay"] == 1e-6
    assert trainer.learner.max_grad_norm == 1
    assert trainer.signature["optimizer"] == train_bc.FORMAL_OPTIMIZER
    assert trainer.signature["schedule"]["warmup_updates"] == 500
    checkpoint = tmp_path / "original.pt"
    trainer.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    saved["signature"]["optimizer"]["learning_rate"] = 0.1
    changed = tmp_path / "changed.pt"
    torch.save(saved, changed)
    before = {
        name: value.clone()
        for name, value in trainer.learner.model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    with pytest.raises(ValueError, match="recipe"):
        trainer.resume(changed)
    for name, value in before.items():
        torch.testing.assert_close(
            trainer.learner.model.state_dict()[name], value, atol=0, rtol=0
        )
