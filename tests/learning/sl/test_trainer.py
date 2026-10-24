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

"""Offline sampling ownership, semantic identity, and exact run recovery."""

from __future__ import annotations

import copy
import random
import numpy as np
import json
import sys
import subprocess
import torch
import pytest
from embodichain.learning.sl import (
    LossResult,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
    TensorBatchSource,
)


class Regression:
    def compute_loss(self, model, batch):
        return LossResult((model(batch.inputs) - batch.targets).square().mean())


def build(path):
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1)
    )
    x = torch.arange(40, dtype=torch.float32).view(20, 2) / 40
    data = TensorBatchSource(SupervisedBatch(x, x.sum(-1, keepdim=True)))
    learner = SupervisedLearner(
        model, Regression(), torch.optim.Adam(model.parameters(), lr=0.01)
    )
    cfg = SupervisedTrainerCfg(
        total_updates=20, batch_size=4, output_dir=str(path), checkpoint_interval=0
    )
    return SupervisedTrainer(
        learner, data, cfg, signature={"target": "sum", "units": "dimensionless"}
    )


def test_resume_reproduces_dropout_optimizer_and_sampling(tmp_path):
    full = build(tmp_path / "full")
    full.fit()
    expected = copy.deepcopy(full.learner.state_dict())
    split = build(tmp_path / "split")
    split.fit(stop_after=7)
    resumed = build(tmp_path / "resumed")
    resumed.resume(tmp_path / "split/last.pt")
    resumed.fit()
    for key, value in expected["model"].items():
        torch.testing.assert_close(
            value, resumed.learner.model.state_dict()[key], rtol=0, atol=0
        )
    combined = (tmp_path / "split/metrics.jsonl").read_text() + (
        tmp_path / "resumed/metrics.jsonl"
    ).read_text()
    assert combined == (tmp_path / "full/metrics.jsonl").read_text()
    assert resumed.learner.num_updates == 20


def test_dataset_change_rejected_before_state_mutation(tmp_path):
    original = build(tmp_path / "first")
    original.fit(stop_after=2)
    candidate = build(tmp_path / "second")
    candidate.signature["units"] = "metres"
    before = copy.deepcopy(candidate.learner.model.state_dict())
    with pytest.raises(ValueError, match="identity"):
        candidate.resume(tmp_path / "first/last.pt")
    for key, value in before.items():
        torch.testing.assert_close(
            value, candidate.learner.model.state_dict()[key], rtol=0, atol=0
        )


def test_source_snapshots_and_sampled_batches_own_storage():
    x = torch.arange(12, dtype=torch.float32).view(4, 3)
    source = TensorBatchSource(SupervisedBatch({"measured": x}, x + 1))
    same = TensorBatchSource(SupervisedBatch({"other_semantics": x}, x + 1))
    assert source.source_id != same.source_id
    x.fill_(100)
    first = source.sample(3, generator=torch.Generator().manual_seed(2))
    expected = first.inputs["measured"].clone()
    source.sample(3, generator=torch.Generator().manual_seed(7)).inputs[
        "measured"
    ].fill_(200)
    torch.testing.assert_close(first.inputs["measured"], expected)
    assert expected.max() < 100
    assert [batch.inputs["measured"].shape[0] for batch in source.iter_batches(3)] == [
        3,
        1,
    ]


def test_sl_import_does_not_load_simulation():
    code = "import sys; import embodichain.learning.sl; assert not any(k.startswith(('dexsim', 'embodichain.lab', 'embodichain.learning.rl')) for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unified_cli_routes_sl_help():
    result = subprocess.run(
        [sys.executable, "-m", "embodichain", "train-sl", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--resume" in result.stdout


def test_sample_metadata_is_owned():
    source = TensorBatchSource(
        SupervisedBatch(
            torch.ones(2, 1), torch.zeros(2, 1), metadata={"units": {"position": "m"}}
        )
    )
    first = source.sample(1, generator=torch.Generator())
    first.metadata["units"]["position"] = "cm"
    second = source.sample(1, generator=torch.Generator())
    assert second.metadata["units"]["position"] == "m"


def test_optimizer_incompatibility_rejected_before_model_mutation():
    first = torch.nn.Linear(2, 1)
    saved = SupervisedLearner(
        first, Regression(), torch.optim.Adam(first.parameters())
    ).state_dict()
    for optimizer in (torch.optim.SGD, torch.optim.Adam):
        candidate = torch.nn.Linear(2, 1)
        parameters = (
            [{"params": [candidate.weight]}, {"params": [candidate.bias]}]
            if optimizer is torch.optim.Adam
            else candidate.parameters()
        )
        learner = SupervisedLearner(
            candidate, Regression(), optimizer(parameters, lr=0.01)
        )
        before = copy.deepcopy(candidate.state_dict())
        with pytest.raises(ValueError, match="optimizer"):
            learner.load_state_dict(saved)
        for key, value in before.items():
            torch.testing.assert_close(
                candidate.state_dict()[key], value, rtol=0, atol=0
            )


def test_source_identity_retains_key_and_container_types():
    value = torch.ones(2, 1)
    trees = [{1: value}, {"1": value}, (value,), [value]]
    identifiers = [
        TensorBatchSource(SupervisedBatch(tree, value)).source_id for tree in trees
    ]
    assert len(set(identifiers)) == len(trees)


def test_scheduler_type_rejected_before_learner_mutation(tmp_path):
    first = build(tmp_path / "first")
    first.scheduler = torch.optim.lr_scheduler.StepLR(
        first.learner.optimizer, step_size=2, gamma=0.5
    )
    first.fit(stop_after=2)
    second = build(tmp_path / "second")
    second.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        second.learner.optimizer, gamma=0.5
    )
    before = copy.deepcopy(second.learner.model.state_dict())
    with pytest.raises(ValueError, match="scheduler type"):
        second.resume(tmp_path / "first/last.pt")
    for key, value in before.items():
        torch.testing.assert_close(
            second.learner.model.state_dict()[key], value, rtol=0, atol=0
        )


@pytest.mark.parametrize("callback_fails", [False, True])
def test_validation_restores_rng_and_mode_even_on_failure(tmp_path, callback_fails):
    def reset_rng():
        random.seed(17)
        np.random.seed(17)

    reset_rng()
    baseline = build(tmp_path / "baseline")
    count = 1 if callback_fails else 3
    baseline.fit(stop_after=count)
    expected = copy.deepcopy(baseline.learner.model.state_dict())
    expected_random = (random.random(), np.random.rand(), torch.rand(4))
    reset_rng()
    candidate = build(tmp_path / "candidate")
    candidate.cfg.validation_interval = 1

    def evaluate(learner):
        random.random()
        np.random.rand(5)
        torch.rand(9)
        if callback_fails:
            raise RuntimeError("validation failed")
        return {"probe": 0.0}

    candidate.evaluate = evaluate
    if callback_fails:
        with pytest.raises(RuntimeError, match="validation failed"):
            candidate.fit(stop_after=count)
    else:
        candidate.fit(stop_after=count)
    assert candidate.learner.model.training
    for key, value in expected.items():
        torch.testing.assert_close(
            candidate.learner.model.state_dict()[key], value, rtol=0, atol=0
        )
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(4), expected_random[2], rtol=0, atol=0)


def test_fresh_run_cannot_overwrite_existing_artifacts(tmp_path):
    first = build(tmp_path / "run")
    first.fit(stop_after=2)
    log = (tmp_path / "run/metrics.jsonl").read_bytes()
    checkpoint = (tmp_path / "run/last.pt").read_bytes()
    with pytest.raises(FileExistsError, match="fresh output"):
        build(tmp_path / "run").fit()
    assert (tmp_path / "run/metrics.jsonl").read_bytes() == log
    assert (tmp_path / "run/last.pt").read_bytes() == checkpoint
    resumed = build(tmp_path / "run")
    resumed.resume(tmp_path / "run/last.pt")
    resumed.fit(stop_after=3)
    assert resumed.learner.num_updates == 3


def test_older_checkpoint_cannot_append_to_a_later_run_history(tmp_path):
    first = build(tmp_path / "run")
    first.fit(stop_after=2)
    first.save(tmp_path / "step2.pt")
    first.fit(stop_after=3)
    candidate = build(tmp_path / "run")
    before = copy.deepcopy(candidate.learner.model.state_dict())
    with pytest.raises(ValueError, match="history is ahead"):
        candidate.resume(tmp_path / "step2.pt")
    for key, value in before.items():
        torch.testing.assert_close(
            candidate.learner.model.state_dict()[key], value, rtol=0, atol=0
        )
    branch = build(tmp_path / "branch")
    branch.resume(tmp_path / "step2.pt")
    assert branch.fit(stop_after=3)["num_updates"] == 3


def test_same_trainer_may_save_an_initial_policy_before_fitting(tmp_path):
    trainer = build(tmp_path / "run")
    trainer.save(trainer.output / "step_000000.pt")
    assert trainer.fit(stop_after=1)["num_updates"] == 1
