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

"""CPU training, exact resume, train-only statistics and standard exports."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from embodichain.compute.dynamics import NeRDStateCodec
from embodichain.data_pipeline.dynamics.shards import (
    DynamicsShardWriter,
    DynamicsWindowDataset,
)
from embodichain.learning.models.nerd import NeRDModel, NeRDModelCfg, load_model
from embodichain.learning.dynamics.trainer import (
    DynamicsTrainCfg,
    DynamicsTrainer,
    compute_statistics,
    encode_batch,
    evaluate_model,
    load_statistics,
    normalized_mse,
    save_statistics,
)
from embodichain.learning.dynamics.workflow import run_train, run_export, run_eval
from embodichain.learning.dynamics.qualification import (
    paired_noninferiority,
    compare_training_steps,
    capture_training_step,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_data(root, validation_offset=0.0):
    metadata = {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    rng = np.random.default_rng(5)
    with DynamicsShardWriter(root, metadata, trajectories_per_shard=3) as writer:
        for split in ("train", "validation", "test"):
            for index in range(4):
                n = 12
                effort = rng.normal(size=(n, 2)).astype(np.float32)
                q, qd = np.zeros((n + 1, 2), np.float32), np.zeros(
                    (n + 1, 2), np.float32
                )
                q[0] = validation_offset if split != "train" else 0
                for step in range(n):
                    qd[step + 1] = qd[step] + effort[step] * 0.1
                    q[step + 1] = q[step] + qd[step] * 0.1
                writer.add_trajectory(
                    f"{split}-{index}",
                    {"q": q, "qd": qd},
                    {"joint_f": effort},
                    split=split,
                )
    return metadata


def architecture(dropout=0.0):
    return NeRDModelCfg(
        input_dims={"states_embedding": 4, "joint_f": 2},
        output_dim=4,
        n_embd=12,
        n_head=3,
        n_layer=1,
        block_size=4,
        history_length=3,
        head_hidden_dims=(12,),
        dropout=dropout,
    )


def build(
    root,
    out,
    *,
    steps=30,
    dropout=0.0,
    log_interval=1,
    resident_cache_bytes=0,
    numerical_profile="default",
    device="cpu",
):
    torch.manual_seed(2)
    model = NeRDModel(architecture(dropout))
    codec = NeRDStateCodec("cartpole")
    train = DynamicsWindowDataset(root, "train", 3)
    validation = DynamicsWindowDataset(root, "validation", 3)
    cfg = DynamicsTrainCfg(
        total_steps=steps,
        batch_size=8,
        learning_rate=0.005,
        final_learning_rate=0.002,
        validation_interval=10,
        checkpoint_interval=10,
        seed=9,
        log_interval=log_interval,
        resident_cache_bytes=resident_cache_bytes,
        numerical_profile=numerical_profile,
        device=device,
    )
    return DynamicsTrainer(
        model, codec, train, validation, cfg, out, manifest=train.manifest["metadata"]
    )


def test_statistics_only_training_windows(tmp_path):
    make_data(tmp_path / "a", 0.0)
    make_data(tmp_path / "b", 1000.0)
    codec = NeRDStateCodec("cartpole")
    first = compute_statistics(DynamicsWindowDataset(tmp_path / "a", "train", 3), codec)
    second = compute_statistics(
        DynamicsWindowDataset(tmp_path / "b", "train", 3), codec
    )
    for key in first["inputs"]:
        torch.testing.assert_close(
            first["inputs"][key]["mean"], second["inputs"][key]["mean"]
        )
    with pytest.raises(ValueError, match="training split"):
        compute_statistics(
            DynamicsWindowDataset(tmp_path / "a", "validation", 3), codec
        )


@pytest.fixture
def statistics_sources(tmp_path):
    """Provide distinct immutable datasets with the same numeric field schema."""
    metadata = make_data(tmp_path / "own")
    with DynamicsShardWriter(tmp_path / "foreign", metadata) as writer:
        writer.add_trajectory(
            "foreign",
            {
                "q": np.full((6, 2), 17.0, np.float32),
                "qd": np.full((6, 2), 3.0, np.float32),
            },
            {"joint_f": np.zeros((5, 2), np.float32)},
            split="train",
        )
    train = DynamicsWindowDataset(tmp_path / "own", "train", 3)
    validation = DynamicsWindowDataset(tmp_path / "own", "validation", 3)
    foreign = DynamicsWindowDataset(tmp_path / "foreign", "train", 3)
    assert train.manifest["fields"] == foreign.manifest["fields"]
    assert train.fingerprint != foreign.fingerprint
    try:
        yield train, validation, foreign, NeRDStateCodec("cartpole")
    finally:
        for reader in (train, validation, foreign):
            reader.close()


@pytest.mark.parametrize("origin", ["foreign", "missing"])
@pytest.mark.parametrize("initialize_normalization", [True, False])
def test_supplied_statistics_identity_rejected_before_mutation(
    tmp_path, statistics_sources, origin, initialize_normalization
):
    train, validation, foreign, codec = statistics_sources
    statistics = compute_statistics(foreign if origin == "foreign" else train, codec)
    if origin == "missing":
        statistics.pop("dataset_fingerprint")
    path = tmp_path / "statistics.pt"
    save_statistics(path, statistics)
    original_file = path.read_bytes()
    loaded = load_statistics(path)
    original_statistics = deepcopy(loaded)
    model = NeRDModel(architecture())
    original_model = {key: value.clone() for key, value in model.state_dict().items()}
    original_config = deepcopy(model.cfg.to_dict())
    # A rejected constructor must not switch the attention policy or reset any
    # global RNG while admitting an unrelated normalization artifact.
    cfg = DynamicsTrainCfg(
        total_steps=2, batch_size=4, numerical_profile="reference", seed=97
    )
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="statistics/dataset fingerprint mismatch"):
        DynamicsTrainer(
            model,
            codec,
            train,
            validation,
            cfg,
            output,
            manifest=train.manifest["metadata"],
            statistics=loaded,
            initialize_normalization=initialize_normalization,
        )
    assert not output.exists()
    assert model.cfg.to_dict() == original_config
    for key, value in original_model.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    assert random.getstate() == rng[0]
    numpy_rng = np.random.get_state()
    assert numpy_rng[0] == rng[1][0] and numpy_rng[2:] == rng[1][2:]
    np.testing.assert_array_equal(numpy_rng[1], rng[1][1])
    torch.testing.assert_close(torch.get_rng_state(), rng[2], atol=0, rtol=0)
    assert loaded.keys() == original_statistics.keys()
    for group in ("inputs", "output"):
        values = loaded[group] if group == "inputs" else {"output": loaded[group]}
        expected = (
            original_statistics[group]
            if group == "inputs"
            else {"output": original_statistics[group]}
        )
        for field in values:
            for moment in ("mean", "var"):
                torch.testing.assert_close(
                    values[field][moment], expected[field][moment], atol=0, rtol=0
                )
    assert path.read_bytes() == original_file


@pytest.mark.parametrize("initialize_normalization", [True, False])
def test_matching_supplied_statistics_preserve_normalization_and_update(
    tmp_path, statistics_sources, initialize_normalization
):
    train, validation, _, codec = statistics_sources
    baseline = build(tmp_path / "own", tmp_path / "baseline", steps=2)
    path = tmp_path / "matching-statistics.pt"
    save_statistics(path, baseline.statistics)
    loaded = load_statistics(path)
    if not initialize_normalization:
        # Explicitly skipping initialization must retain existing model buffers,
        # even if matching-corpus statistics were supplied for its record.
        baseline.model.input_rms["states_embedding"].mean[0] = 0.5
    model = deepcopy(baseline.model)
    candidate = DynamicsTrainer(
        model,
        codec,
        train,
        validation,
        baseline.cfg.copy(),
        tmp_path / "matching",
        manifest=train.manifest["metadata"],
        statistics=loaded,
        initialize_normalization=initialize_normalization,
    )
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    assert candidate.step() == baseline.step()
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    baseline.train_dataset.close()
    baseline.validation_dataset.close()


@pytest.mark.parametrize("numerical_profile", ["default", "reference"])
def test_shared_recipe_matches_legacy_updates_and_restores_sampler(
    tmp_path, numerical_profile
):
    """The new CLI recipe retains the established NeRD schedule and updates."""
    from embodichain.learning.dynamics.sl_recipe import build_supervised_trainer

    make_data(tmp_path / "data")
    legacy = build(
        tmp_path / "data",
        tmp_path / "legacy",
        steps=4,
        numerical_profile=numerical_profile,
    )
    legacy.cfg.lr_update_interval = 3
    legacy.cfg.lr_offset = 1
    config = {
        "model": architecture().to_dict(),
        "data": {"path": str(tmp_path / "data")},
        "objective": {"numerical_profile": numerical_profile},
        "optimizer": {
            "learning_rate": 0.005,
            "final_learning_rate": 0.002,
            "max_grad_norm": 1.0,
            "lr_update_interval": 3,
            "lr_offset": 1,
        },
        "trainer": {
            "total_updates": 4,
            "batch_size": 8,
            "seed": 9,
            "output_dir": str(tmp_path / "shared"),
        },
    }
    shared = build_supervised_trainer(config)
    shared.learner.model.load_state_dict(legacy.model.state_dict())
    legacy_rates = [legacy.step()["learning_rate"] for _ in range(4)]
    shared_rates = []
    shared.learner.optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: shared_rates.append(
            optimizer.param_groups[0]["lr"]
        )
    )
    shared.fit(stop_after=2)
    reordered = build_supervised_trainer(
        {
            **config,
            "model": {
                **config["model"],
                "input_dims": dict(
                    reversed(list(config["model"]["input_dims"].items()))
                ),
            },
            "trainer": {**config["trainer"], "output_dir": str(tmp_path / "reordered")},
        }
    )
    before = {
        key: value.clone()
        for key, value in reordered.learner.model.state_dict().items()
    }
    with pytest.raises(ValueError, match="identity"):
        reordered.resume(tmp_path / "shared/last.pt")
    assert reordered.learner.num_updates == 0
    for key, value in before.items():
        torch.testing.assert_close(
            reordered.learner.model.state_dict()[key], value, rtol=0, atol=0
        )
    restored = build_supervised_trainer(config)
    restored.resume(tmp_path / "shared/last.pt")
    restored.learner.optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: shared_rates.append(
            optimizer.param_groups[0]["lr"]
        )
    )
    restored.fit()
    assert shared_rates == pytest.approx(legacy_rates, rel=1e-15, abs=0)
    assert shared_rates[0] == shared_rates[1] == shared_rates[2]
    assert shared_rates[3] < shared_rates[2]
    for name, value in legacy.model.state_dict().items():
        torch.testing.assert_close(
            restored.learner.model.state_dict()[name], value, atol=0, rtol=0
        )
    assert torch.equal(restored.sampler.get_state(), legacy.sampler.get_state())
    from embodichain.learning.sl.workflow import train_from_config

    report = train_from_config(
        {
            "factory": "embodichain.learning.dynamics.sl_recipe:build_supervised_trainer",
            **config,
        },
        resume=str(tmp_path / "shared/last.pt"),
    )
    assert report["num_updates"] == legacy.global_step


@pytest.mark.parametrize("numerical_profile", ["default", "reference"])
def test_exact_resume_with_dropout_optimizer_sampler_rng(tmp_path, numerical_profile):
    make_data(tmp_path / "data")
    full = build(
        tmp_path / "data",
        tmp_path / "full",
        steps=20,
        dropout=0.1,
        numerical_profile=numerical_profile,
    )
    full.train()
    expected = {key: value.clone() for key, value in full.model.state_dict().items()}
    interrupted = build(
        tmp_path / "data",
        tmp_path / "resume",
        steps=20,
        dropout=0.1,
        numerical_profile=numerical_profile,
    )
    assert interrupted.train(stop_after=7)["status"] == "interrupted"
    restored = build(
        tmp_path / "data",
        tmp_path / "resume",
        steps=20,
        dropout=0.1,
        numerical_profile=numerical_profile,
    )
    restored.resume(tmp_path / "resume" / "last.pt")
    restored.train()
    for key, value in expected.items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[key], rtol=0, atol=0
        )
    assert torch.equal(full.sampler.get_state(), restored.sampler.get_state())
    full_lines = [
        json.loads(line)
        for line in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()
    ]
    resumed_lines = [
        json.loads(line)
        for line in (tmp_path / "resume" / "metrics.jsonl").read_text().splitlines()
    ]
    assert full_lines == resumed_lines


def test_training_improves_validation_and_export_matches(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run", steps=100)
    initial = evaluate_model(trainer.model, trainer.codec, trainer.validation_dataset)
    trainer.train()
    final = evaluate_model(trainer.model, trainer.codec, trainer.validation_dataset)
    assert final["normalized_mse"] < initial["normalized_mse"] * 0.6
    run_export(tmp_path / "run" / "last.pt", tmp_path / "export")
    exported, _ = load_model(tmp_path / "export")
    inputs, _, _ = encode_batch(
        default_collate([trainer.validation_dataset[0]]), trainer.codec, "cpu"
    )
    trainer.model.eval()
    torch.testing.assert_close(exported(inputs), trainer.model(inputs), rtol=0, atol=0)
    report = run_eval(
        {"dataset": str(tmp_path / "data"), "model": str(tmp_path / "export")}
    )
    assert report["split"] == "test"
    assert report["qualification"] == "offline_only"


@pytest.mark.parametrize("head_dims", [(12,), [12]])
def test_workflow_fine_tune_and_resume_contract(tmp_path, head_dims):
    metadata = make_data(tmp_path / "data")
    model_config = architecture().to_dict()
    model_config["head_hidden_dims"] = head_dims
    cfg = {
        "dataset": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "first"),
        "manifest": metadata,
        "model": model_config,
        "train": {
            "total_steps": 2,
            "batch_size": 4,
            "validation_interval": 1,
            "checkpoint_interval": 1,
        },
    }
    run_train(cfg)
    run_export(tmp_path / "first" / "last.pt", tmp_path / "export")
    existing = {
        path.name: path.read_bytes()
        for path in (tmp_path / "first").iterdir()
        if path.is_file()
    }
    for options in ({}, {"fine_tune": str(tmp_path / "export")}):
        with pytest.raises(FileExistsError, match="another run"):
            run_train({**cfg, **options})
        assert existing == {
            path.name: path.read_bytes()
            for path in (tmp_path / "first").iterdir()
            if path.is_file()
        }
    fine = {
        **cfg,
        "output_dir": str(tmp_path / "fine"),
        "fine_tune": str(tmp_path / "export"),
    }
    assert run_train(fine)["global_step"] == 2
    assert (
        run_train(
            {
                **cfg,
                "output_dir": str(tmp_path / "fine-resume"),
                "resume": str(tmp_path / "fine/last.pt"),
            }
        )["global_step"]
        == 2
    )
    mismatch = {
        **cfg,
        "output_dir": str(tmp_path / "bad"),
        "resume": str(tmp_path / "first" / "last.pt"),
        "train": {**cfg["train"], "total_steps": 3},
    }
    with pytest.raises(ValueError, match="train_cfg mismatch"):
        run_train(mismatch)


@pytest.mark.parametrize("operation", ["step", "train", "save"])
def test_fresh_trainer_rejects_owned_output_before_update(tmp_path, operation):
    make_data(tmp_path / "data")
    original = build(tmp_path / "data", tmp_path / "run", steps=2)
    original.train(stop_after=1)
    before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "run").iterdir()
        if path.is_file()
    }
    candidate = build(tmp_path / "data", tmp_path / "run", steps=2)
    parameters = {
        key: value.clone() for key, value in candidate.model.state_dict().items()
    }
    sampler = candidate.sampler.get_state().clone()
    with pytest.raises(FileExistsError, match="another run"):
        if operation == "save":
            candidate.save(tmp_path / "run/new.pt")
        else:
            getattr(candidate, operation)()
    assert candidate.global_step == candidate.learner.num_updates == 0
    assert not candidate.optimizer.state
    torch.testing.assert_close(candidate.sampler.get_state(), sampler, rtol=0, atol=0)
    for key, value in parameters.items():
        torch.testing.assert_close(
            candidate.model.state_dict()[key], value, rtol=0, atol=0
        )
    assert before == {
        path.name: path.read_bytes()
        for path in (tmp_path / "run").iterdir()
        if path.is_file()
    }


def test_resume_new_directory_preserves_best_export_without_new_improvement(
    tmp_path, monkeypatch
):
    make_data(tmp_path / "data")
    scores = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr(
        "embodichain.learning.dynamics.trainer.evaluate_model",
        lambda *args, **kwargs: {"normalized_mse": next(scores)},
    )
    original = build(tmp_path / "data", tmp_path / "original", steps=4)
    original.cfg.validation_interval = 1
    original.train(stop_after=2)
    saved_best = torch.load(tmp_path / "original/best.pt", weights_only=True)
    assert saved_best["global_step"] == 1
    assert saved_best["resumable"] is False
    assert any(
        not torch.equal(value, original.model.state_dict()[key])
        for key, value in saved_best["model"].items()
    )
    before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "original").iterdir()
        if path.is_file()
    }
    run_export(tmp_path / "original/best.pt", tmp_path / "original-export")
    restored = build(tmp_path / "data", tmp_path / "resumed", steps=4)
    restored.cfg.validation_interval = 1
    restored.resume(tmp_path / "original/last.pt")
    assert list((tmp_path / "resumed").iterdir()) == []
    restored.train(stop_after=3)
    restored_best = torch.load(tmp_path / "resumed/best.pt", weights_only=True)
    assert restored_best["global_step"] == 1
    for key, value in saved_best["model"].items():
        torch.testing.assert_close(restored_best["model"][key], value, rtol=0, atol=0)
    assert before == {
        path.name: path.read_bytes()
        for path in (tmp_path / "original").iterdir()
        if path.is_file()
    }
    run_export(tmp_path / "resumed/best.pt", tmp_path / "resumed-export")
    expected, _ = load_model(tmp_path / "original-export")
    actual, _ = load_model(tmp_path / "resumed-export")
    inputs, _, _ = encode_batch(
        restored.validation_dataset.get_batch([0]), restored.codec, "cpu"
    )
    torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)
    with pytest.raises(ValueError, match="export only"):
        restored.resume(tmp_path / "resumed/best.pt")


def test_resume_preserves_future_best_after_interrupted_publication(
    tmp_path, monkeypatch
):
    make_data(tmp_path / "data")
    scores = iter([1.0, 0.5])
    monkeypatch.setattr(
        "embodichain.learning.dynamics.trainer.evaluate_model",
        lambda *args, **kwargs: {"normalized_mse": next(scores)},
    )
    original = build(tmp_path / "data", tmp_path / "original", steps=4)
    original.cfg.validation_interval = 1
    original.train(stop_after=1)
    first_best = torch.load(tmp_path / "original/best.pt", weights_only=True)
    publish = original._write_best_snapshot

    def interrupt_after_best() -> None:
        publish()
        raise RuntimeError("interrupted after publishing best")

    monkeypatch.setattr(original, "_write_best_snapshot", interrupt_after_best)
    with pytest.raises(RuntimeError, match="publishing best"):
        original.train(stop_after=2)
    assert (
        torch.load(tmp_path / "original/best.pt", weights_only=True)["global_step"] == 2
    )
    assert (
        torch.load(tmp_path / "original/last.pt", weights_only=True)["global_step"] == 1
    )
    before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "original").iterdir()
        if path.is_file()
    }
    candidate = build(tmp_path / "data", tmp_path / "original", steps=4)
    candidate.cfg.validation_interval = 1
    parameters = {
        key: value.clone() for key, value in candidate.model.state_dict().items()
    }
    sampler, rng = candidate.sampler.get_state().clone(), torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with pytest.raises(ValueError, match="best.pt.*fresh output"):
        candidate.resume(tmp_path / "original/last.pt")
    assert candidate.global_step == 0 and not candidate.optimizer.state
    torch.testing.assert_close(candidate.sampler.get_state(), sampler, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert random.getstate() == python_rng
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and actual_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
    for key, value in parameters.items():
        torch.testing.assert_close(
            candidate.model.state_dict()[key], value, rtol=0, atol=0
        )
    restored = build(tmp_path / "data", tmp_path / "fresh", steps=4)
    restored.cfg.validation_interval = 1
    restored.resume(tmp_path / "original/last.pt")
    restored.train(stop_after=1)
    restored_best = torch.load(tmp_path / "fresh/best.pt", weights_only=True)
    assert restored_best["global_step"] == 1
    for key, value in first_best["model"].items():
        torch.testing.assert_close(restored_best["model"][key], value, rtol=0, atol=0)
    assert before == {
        path.name: path.read_bytes()
        for path in (tmp_path / "original").iterdir()
        if path.is_file()
    }


def test_resume_input_order_rejection_preserves_runtime_state(tmp_path):
    make_data(tmp_path / "data")
    original = build(tmp_path / "data", tmp_path / "original", steps=4)
    original.train(stop_after=1)
    candidate = build(tmp_path / "data", tmp_path / "candidate", steps=4)
    candidate.model.cfg.input_dims = dict(
        reversed(list(candidate.model.cfg.input_dims.items()))
    )
    parameters = {
        key: value.clone() for key, value in candidate.model.state_dict().items()
    }
    sampler, rng = candidate.sampler.get_state().clone(), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="input_dims order"):
        candidate.resume(tmp_path / "original/last.pt")
    assert candidate.global_step == 0
    assert not candidate.optimizer.state
    assert list((tmp_path / "candidate").iterdir()) == []
    torch.testing.assert_close(candidate.sampler.get_state(), sampler, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in parameters.items():
        torch.testing.assert_close(
            candidate.model.state_dict()[key], value, rtol=0, atol=0
        )


def test_legacy_best_fallback_requires_matching_source_weights(tmp_path):
    make_data(tmp_path / "data")
    original = build(tmp_path / "data", tmp_path / "legacy", steps=4)
    original.step()
    original.best_validation = 1.0
    original.save(tmp_path / "legacy/best.pt")
    original.step()
    original.save(tmp_path / "legacy/last.pt")
    # Construct the historical v1 layout from actual step-one and step-two
    # checkpoints, retaining each original optimizer/RNG boundary.
    for filename in ("best.pt", "last.pt"):
        path = tmp_path / "legacy" / filename
        state = torch.load(path, weights_only=True)
        for key in ("run_id", "best_model", "best_step"):
            state.pop(key)
        torch.save(state, path)
    expected = torch.load(tmp_path / "legacy/best.pt", weights_only=True)
    restored = build(tmp_path / "data", tmp_path / "restored", steps=4)
    restored.resume(tmp_path / "legacy/last.pt")
    restored.train(stop_after=2)
    best = torch.load(tmp_path / "restored/best.pt", weights_only=True)
    assert best["global_step"] == 1
    assert best["resumable"] is False
    for key, value in expected["model"].items():
        torch.testing.assert_close(best["model"][key], value, rtol=0, atol=0)
    (tmp_path / "legacy/best.pt").unlink()
    missing = build(tmp_path / "data", tmp_path / "missing", steps=4)
    with pytest.raises(ValueError, match="matching best.pt"):
        missing.resume(tmp_path / "legacy/last.pt")
    assert missing.global_step == 0 and not missing.optimizer.state
    assert list((tmp_path / "missing").iterdir()) == []


def test_resume_foreign_directory_or_newer_history_preserves_artifacts(tmp_path):
    make_data(tmp_path / "data")
    source = build(tmp_path / "data", tmp_path / "source", steps=4)
    source.step()
    source.save(tmp_path / "source/step_1.pt")
    source.train(stop_after=3)
    foreign = build(tmp_path / "data", tmp_path / "foreign", steps=4)
    foreign.train(stop_after=1)
    for output, checkpoint, error, message in (
        ("foreign", "source/last.pt", FileExistsError, "another run"),
        ("source", "source/step_1.pt", ValueError, "history is ahead"),
    ):
        before = {
            path.name: path.read_bytes()
            for path in (tmp_path / output).iterdir()
            if path.is_file()
        }
        candidate = build(tmp_path / "data", tmp_path / output, steps=4)
        parameters = {
            key: value.clone() for key, value in candidate.model.state_dict().items()
        }
        with pytest.raises(error, match=message):
            candidate.resume(tmp_path / checkpoint)
        assert candidate.global_step == 0
        assert not candidate.optimizer.state
        for key, value in parameters.items():
            torch.testing.assert_close(
                candidate.model.state_dict()[key], value, rtol=0, atol=0
            )
        assert before == {
            path.name: path.read_bytes()
            for path in (tmp_path / output).iterdir()
            if path.is_file()
        }


def test_resume_rejects_unknown_preprocessing_execution(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run", steps=2)
    trainer.train(stop_after=1)
    state = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=True)
    state.pop("preprocessing_execution")
    path = tmp_path / "old-execution.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="preprocessing execution mismatch"):
        trainer.resume(path)
    trainer.train_dataset.close()
    trainer.validation_dataset.close()


def test_official_loss_order_and_mask():
    prediction = torch.tensor([[[1.0, 2.0], [500.0, 500.0]]], requires_grad=True)
    target = torch.tensor([[[2.0, 5.0], [0.0, 0.0]]])
    variance = torch.tensor([4.0, 9.0])
    weights = 1.0 / torch.sqrt(variance + 1e-5)
    expected = torch.nn.functional.mse_loss(
        prediction[:, :1] * weights, target[:, :1] * weights
    )
    actual = normalized_mse(prediction, target, variance, torch.tensor([[True, False]]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0


@pytest.mark.parametrize("masked", [False, True])
def test_unnormalized_fp64_loss_backward_matches_analytic_oracle(masked):
    prediction = torch.tensor(
        [[[0.25, -0.5], [1.25, 0.125]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[[0.125, -0.25], [0.5, 0.75]]], dtype=torch.float32)
    mask = torch.tensor([[True, False]]) if masked else None
    actual = normalized_mse(prediction, target, None, mask)
    # Binary fractions yield exact independent values and derivatives:
    # residuals=(1/8,-1/4,3/4,-5/8), gradient=2*residual/valid scalar count.
    expected_loss = 5 / 128 if masked else 33 / 128
    expected_gradient = torch.tensor(
        (
            [[[0.125, -0.25], [0.0, 0.0]]]
            if masked
            else [[[0.0625, -0.125], [0.375, -0.3125]]]
        ),
        dtype=torch.float64,
    )
    assert actual.dtype == torch.float64
    assert actual.item() == expected_loss
    actual.backward()
    torch.testing.assert_close(prediction.grad, expected_gradient, atol=0, rtol=0)
    assert target.dtype == torch.float32


def test_bfloat16_predictions_do_not_quantize_float32_targets():
    prediction = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16, requires_grad=True)
    # Both differences disappear if labels are incorrectly converted to BF16.
    target = torch.tensor([[1.0 + 2**-12, 2.0 + 2**-10]], dtype=torch.float32)
    oracle = prediction.detach().float().requires_grad_()
    expected = (oracle - target).square().mean()
    expected.backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = normalized_mse(prediction, target, None)
    assert actual.dtype == torch.float32
    assert actual.item() > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    torch.testing.assert_close(prediction.grad, oracle.grad.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_heterogeneous_target_statistics_preserve_official_loss_and_gradient(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator().manual_seed(42)
    prediction = torch.randn(4, 10, 29, generator=generator).to(device).requires_grad_()
    target = torch.randn(4, 10, 29, generator=generator).to(device)
    variance = torch.logspace(-6, 3, 29, device=device)
    weights = 1.0 / torch.sqrt(variance + 1e-5)
    expected = torch.nn.functional.mse_loss(prediction * weights, target * weights)
    actual = normalized_mse(prediction, target, variance)
    expected_gradient = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, prediction)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


def test_qualification_gate_requires_five_seeds_and_detects_regression():
    assert paired_noninferiority([1.0, 1.0], [1.0, 1.0])["status"] == "pending"
    assert paired_noninferiority([1.0] * 5, [1.01] * 5)["status"] == "passed"
    assert paired_noninferiority([1.0] * 5, [1.1] * 5)["status"] == "failed"
    assert paired_noninferiority([0.0] * 5, [1e-6] * 5)["difference_mode"] == "absolute"


def test_training_capture_detects_altered_update():
    model = NeRDModel(architecture())
    inputs = {"states_embedding": torch.randn(2, 3, 4), "joint_f": torch.randn(2, 3, 2)}
    reference = capture_training_step(model, inputs, torch.randn(2, 3, 4))
    # Self-comparison only tests the checker, not official parity.
    assert compare_training_steps(reference, reference)["passed"]
    candidate = dict(reference, loss=reference["loss"] + 1)
    assert not compare_training_steps(reference, candidate)["passed"]


def test_offline_import_does_not_load_simulation():
    code = "import sys; import embodichain.learning.dynamics.workflow; import embodichain.learning.dynamics.trainer; assert not any(k == 'dexsim' or k.startswith('embodichain.lab.sim') for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_interrupted_workflow_before_validation_is_json_serializable(tmp_path):
    metadata = make_data(tmp_path / "data")
    result = run_train(
        {
            "dataset": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "run"),
            "manifest": metadata,
            "model": architecture().to_dict(),
            "train": {"total_steps": 10, "batch_size": 4, "validation_interval": 10},
            "stop_after": 1,
        }
    )
    assert result["best_validation"] is None
    assert result["status"] == "interrupted"
    assert result["runtime_identity"]["parameter_placements"] == [
        {"device": "cpu", "dtype": "torch.float32"}
    ]
    assert (
        json.loads((tmp_path / "run" / "result.json").read_text())["runtime_identity"]
        == result["runtime_identity"]
    )


def test_ant_offline_training_and_metrics(tmp_path):
    """A floating-base model trains through native-to-legacy targets without sim."""
    metadata = {"codec": {"robot": "ant"}, "dt": 1 / 60}
    with DynamicsShardWriter(tmp_path / "data", metadata) as writer:
        for split in ("train", "validation", "test"):
            q = np.zeros((7, 15), np.float32)
            q[:, 2] = 0.5
            q[:, 6] = 1.0
            q[:, 0] = np.arange(7) * 0.01
            qd = np.zeros((7, 14), np.float32)
            qd[:, 0] = 0.6
            normal = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (6, 1))
            transitions = {
                "joint_f": np.zeros((6, 14), np.float32),
                "contact_normals": normal,
                "contact_points_1": np.zeros((6, 3), np.float32),
                "contact_depths": np.full((6, 1), 0.1, np.float32),
                "contact_thicknesses_0": np.zeros((6, 1), np.float32),
                "contact_thicknesses_1": np.zeros((6, 1), np.float32),
                "gravity_dir": -normal,
            }
            writer.add_trajectory(split, {"q": q, "qd": qd}, transitions, split=split)
    codec = NeRDStateCodec("ant")
    train = DynamicsWindowDataset(tmp_path / "data", "train", 3)
    validation = DynamicsWindowDataset(tmp_path / "data", "validation", 3)
    inputs, _, _ = encode_batch(default_collate([train[0]]), codec, "cpu")
    cfg = architecture()
    cfg.input_dims = {key: value.shape[-1] for key, value in inputs.items()}
    cfg.output_dim = 29
    model = NeRDModel(cfg)
    trainer = DynamicsTrainer(
        model,
        codec,
        train,
        validation,
        DynamicsTrainCfg(total_steps=2, batch_size=2, validation_interval=1),
        tmp_path / "run",
        manifest=metadata,
    )
    trainer.train()
    report = evaluate_model(model, codec, validation)
    assert np.isfinite(report["normalized_mse"])
    assert report["base_orientation_error_rad"] >= 0.0
    assert report["base_position_error"] >= 0.0


def test_memory_mapped_order_resume_and_integrity(tmp_path):
    make_data(tmp_path / "data")
    order = np.zeros((10, 8), np.int64)
    path = tmp_path / "order.npy"
    np.save(path, order)
    trainer = build(tmp_path / "data", tmp_path / "run", steps=10)
    trainer.sample_order = trainer._prepare_order(path)
    assert isinstance(trainer.sample_order, np.memmap)
    trainer.train(stop_after=3)
    snapshot = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    assert isinstance(snapshot["sample_order"], dict)
    assert "sha256" in snapshot["sample_order"]
    restored = build(tmp_path / "data", tmp_path / "restored", steps=10)
    restored.resume(tmp_path / "run" / "last.pt")
    restored.train(stop_after=4)
    order[0, 0] = 1
    np.save(path, order)
    invalid = build(tmp_path / "data", tmp_path / "invalid", steps=10)
    sampler, rng = invalid.sampler.get_state().clone(), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checksum"):
        invalid.resume(tmp_path / "run" / "last.pt")
    assert invalid.sample_order is None and invalid._sample_order_identity is None
    torch.testing.assert_close(invalid.sampler.get_state(), sampler, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_checkpointed_order_resume_is_identical_to_uninterrupted_training(tmp_path):
    from embodichain.learning.dynamics._sample_order import _write_checkpointed_order

    make_data(tmp_path / "data")
    first = build(tmp_path / "data", tmp_path / "full", steps=10)
    order, _ = _write_checkpointed_order(
        tmp_path / "order.json",
        dataset_size=len(first.train_dataset),
        steps=10,
        batch_size=first.cfg.batch_size,
        seed=4,
        block_steps=3,
    )
    first.sample_order = first._prepare_order(order)
    first.train()
    interrupted = build(tmp_path / "data", tmp_path / "partial", steps=10)
    interrupted.sample_order = interrupted._prepare_order(order)
    interrupted.train(stop_after=4)
    resumed = build(tmp_path / "data", tmp_path / "resumed", steps=10)
    resumed.resume(tmp_path / "partial/last.pt")
    resumed.train()
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name], value, atol=0, rtol=0
        )


def test_sparse_logging_retains_validation_and_final_committed_step(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run", steps=13, log_interval=4)
    result = trainer.train()
    rows = [
        json.loads(line)
        for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [4, 8, 10, 12, 13]
    assert "validation" in rows[2] and "validation" in rows[-1]
    assert result["global_step"] == 13


def test_resident_raw_cache_preserves_training_updates_and_validation(tmp_path):
    make_data(tmp_path / "data")
    lazy = build(tmp_path / "data", tmp_path / "lazy", steps=12)
    lazy_result = lazy.train()
    cached = build(
        tmp_path / "data",
        tmp_path / "cached",
        steps=12,
        resident_cache_bytes=1024 * 1024,
    )
    cached_result = cached.train()
    assert lazy_result["last"] == cached_result["last"]
    for name, value in lazy.model.state_dict().items():
        torch.testing.assert_close(
            cached.model.state_dict()[name], value, atol=0, rtol=0
        )


def test_validation_batch_cap_reports_actual_windows(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run")
    report = evaluate_model(
        trainer.model,
        trainer.codec,
        trainer.validation_dataset,
        batch_size=2,
        max_batches=1,
    )
    assert report["windows"] == 2
    assert report["valid_positions"] == 6


def test_grouped_input_batches_preserve_exact_legacy_training_updates(tmp_path):
    make_data(tmp_path / "data")
    old = build(tmp_path / "data", tmp_path / "old", steps=20, dropout=0.1)
    old.train_dataset.get_batch = lambda indices: default_collate(
        [old.train_dataset[int(index)] for index in indices]
    )
    old.validation_dataset.get_batch = lambda indices: default_collate(
        [old.validation_dataset[int(index)] for index in indices]
    )
    old.train()
    expected = {key: value.clone() for key, value in old.model.state_dict().items()}
    grouped = build(tmp_path / "data", tmp_path / "grouped", steps=20, dropout=0.1)
    grouped.train()
    for key, value in expected.items():
        torch.testing.assert_close(
            value, grouped.model.state_dict()[key], rtol=0, atol=0
        )
    assert (tmp_path / "old" / "metrics.jsonl").read_text() == (
        tmp_path / "grouped" / "metrics.jsonl"
    ).read_text()
    old.train_dataset.close()
    old.validation_dataset.close()
    grouped.train_dataset.close()
    grouped.validation_dataset.close()


def _add_physical_metadata(root, signature):
    path = root / "manifest.json"
    data = json.loads(path.read_text())
    data["metadata"].update(
        physics_signature=signature,
        physics_signature_schema=2,
        gravity=[0.0, 0.0, -9.81],
        joint_names=["slider", "pole"],
        asset_sha256="asset-digest",
        contact_schema="fixed-slots",
    )
    path.write_text(json.dumps(data))
    return data["metadata"]


@pytest.mark.parametrize(
    "mutation",
    [
        "changed_signature",
        "dropped_signature",
        "changed_schema",
        "dropped_gravity",
        "changed_asset",
        "empty_override",
    ],
)
def test_training_manifest_cannot_relabel_dataset_physics(tmp_path, mutation):
    make_data(tmp_path / "data")
    metadata = _add_physical_metadata(tmp_path / "data", "old-physics")
    override = dict(metadata)
    if mutation == "changed_signature":
        override["physics_signature"] = "current-physics"
    elif mutation == "dropped_signature":
        override.pop("physics_signature")
    elif mutation == "changed_schema":
        override["physics_signature_schema"] = 3
    elif mutation == "dropped_gravity":
        override.pop("gravity")
    elif mutation == "changed_asset":
        override["asset_sha256"] = "other-asset"
    else:
        override = {}
    with pytest.raises(ValueError, match="manifest/data .* mismatch"):
        run_train(
            {
                "dataset": str(tmp_path / "data"),
                "output_dir": str(tmp_path / "run"),
                "manifest": override,
                "model": architecture().to_dict(),
                "train": {"total_steps": 1, "batch_size": 2},
            }
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physics_signature", "invented"),
        ("legacy_velocity", {"field": "states/legacy_velocity", "version": 1}),
    ],
)
def test_training_annotations_cannot_invent_missing_physics_contract(
    tmp_path, field, value
):
    metadata = make_data(tmp_path / "data")
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        run_train(
            {
                "dataset": str(tmp_path / "data"),
                "output_dir": str(tmp_path / "run"),
                "manifest": {**metadata, field: value},
                "model": architecture().to_dict(),
                "train": {"total_steps": 1, "batch_size": 2},
            }
        )


def test_unrelated_annotations_and_fine_tune_to_new_measured_physics(tmp_path):
    make_data(tmp_path / "old-data")
    make_data(tmp_path / "new-data")
    _add_physical_metadata(tmp_path / "old-data", "old-physics")
    new_metadata = _add_physical_metadata(tmp_path / "new-data", "new-physics")
    config = {
        "dataset": str(tmp_path / "old-data"),
        "output_dir": str(tmp_path / "old-run"),
        "model": architecture().to_dict(),
        "train": {"total_steps": 1, "batch_size": 2},
    }
    run_train(config)
    run_export(tmp_path / "old-run" / "last.pt", tmp_path / "old-export")
    run_train(
        {
            **config,
            "dataset": str(tmp_path / "new-data"),
            "output_dir": str(tmp_path / "new-run"),
            "fine_tune": str(tmp_path / "old-export"),
            "manifest": {**new_metadata, "notes": "adapt to new measured teacher"},
        }
    )
    state = torch.load(tmp_path / "new-run" / "last.pt", weights_only=True)
    assert state["manifest"]["physics_signature"] == "new-physics"
    assert state["manifest"]["notes"] == "adapt to new measured teacher"


def _gradient_parameters():
    generator = torch.Generator().manual_seed(14)
    parameters = [torch.nn.Parameter(torch.zeros(width)) for width in (17, 65, 513)]
    for parameter, scale in zip(parameters, (1e-3, 1.0, 10.0), strict=True):
        parameter.grad = torch.randn(parameter.shape, generator=generator) * scale
    return parameters


@pytest.mark.parametrize("profile", ["default", "reference"])
def test_numerical_clipping_matches_its_independent_cpu_oracle(profile):
    from embodichain.learning.dynamics._numerics import _clip_grad_norm_

    parameters = _gradient_parameters()
    expected = [parameter.grad.clone() for parameter in parameters]
    if profile == "reference":
        # This is the scalar policy validated with the historical CPU trainer;
        # production must preserve its left-to-right FP64 rounding order.
        norms = [float(torch.linalg.vector_norm(gradient)) for gradient in expected]
        total = torch.tensor(math.sqrt(sum(value * value for value in norms)))
        coefficient = (1.0 / (total + 1e-6)).clamp(max=1.0)
        for gradient in expected:
            gradient.mul_(coefficient)
    else:
        originals = [
            torch.nn.Parameter(torch.zeros_like(gradient)) for gradient in expected
        ]
        for parameter, gradient in zip(originals, expected, strict=True):
            parameter.grad = gradient
        total = torch.nn.utils.clip_grad_norm_(originals, 1.0, error_if_nonfinite=True)
    actual = _clip_grad_norm_(parameters, 1.0, profile=profile)
    assert torch.equal(actual, total)
    for parameter, gradient in zip(parameters, expected, strict=True):
        assert torch.equal(parameter.grad, gradient)


def test_reference_attention_context_covers_backward_eval_and_restores_flags(
    tmp_path, monkeypatch
):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run", numerical_profile="reference")
    original = torch.nn.functional.scaled_dot_product_attention
    before = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )
    forward_flags, backward_flags = [], []

    def record(*args, **kwargs):
        forward_flags.append(
            (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
            )
        )
        return original(*args, **kwargs)

    def backward(gradient):
        backward_flags.append(
            (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
            )
        )
        return gradient

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    handle = next(trainer.model.parameters()).register_hook(backward)
    try:
        trainer.step()
        evaluate_model(trainer.model, trainer.codec, trainer.validation_dataset)
    finally:
        handle.remove()
        trainer.train_dataset.close()
        trainer.validation_dataset.close()
    assert forward_flags and backward_flags
    assert all(
        flags == (False, True, False) for flags in forward_flags + backward_flags
    )
    assert before == (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )


def test_attention_context_restores_flags_on_failure():
    from embodichain.learning.dynamics._numerics import _attention_context

    before = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    with pytest.raises(RuntimeError, match="injected"):
        with _attention_context("reference"):
            raise RuntimeError("injected")
    assert before == (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )


def test_reference_nonfinite_gradient_does_not_commit_optimizer_or_cursor(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run", numerical_profile="reference")
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    sampler = trainer.sampler.get_state().clone()
    handle = next(trainer.model.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        with pytest.raises(FloatingPointError, match="gradient norm"):
            trainer.step()
    finally:
        handle.remove()
        trainer.train_dataset.close()
        trainer.validation_dataset.close()
    assert trainer.global_step == 0
    assert not trainer.optimizer.state
    assert torch.equal(trainer.sampler.get_state(), sampler)
    assert all(
        torch.equal(value, trainer.model.state_dict()[key])
        for key, value in before.items()
    )


def test_unknown_numerical_profile_is_rejected_before_output(tmp_path):
    make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="numerical_profile"):
        build(tmp_path / "data", tmp_path / "run", numerical_profile="unknown")
    assert not (tmp_path / "run").exists()


def test_reference_clipping_rejects_unsupported_gradient_precision():
    from embodichain.learning.dynamics._numerics import _clip_grad_norm_

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError, match="FP32 gradients"):
        _clip_grad_norm_([parameter], 1.0, profile="reference")


@pytest.mark.parametrize(
    "old_profile,new_profile", [("default", "reference"), ("reference", "default")]
)
def test_resume_rejects_numerical_profile_changes(tmp_path, old_profile, new_profile):
    make_data(tmp_path / "data")
    original = build(tmp_path / "data", tmp_path / "old", numerical_profile=old_profile)
    original.step()
    original.save(tmp_path / "saved.pt")
    candidate = build(
        tmp_path / "data", tmp_path / "new", numerical_profile=new_profile
    )
    try:
        with pytest.raises(ValueError, match="numerical_profile mismatch"):
            candidate.resume(tmp_path / "saved.pt")
    finally:
        for trainer in (original, candidate):
            trainer.train_dataset.close()
            trainer.validation_dataset.close()


def test_legacy_default_checkpoint_resume_and_export(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(tmp_path / "data", tmp_path / "run")
    trainer.step()
    trainer.save(tmp_path / "new.pt")
    state = torch.load(tmp_path / "new.pt", weights_only=True)
    state.pop("numerical_profile")
    state.pop("attention_backend")
    state["train_cfg"].pop("numerical_profile")
    state["model_cfg"].pop("attention_backend")
    torch.save(state, tmp_path / "legacy.pt")
    restored = build(tmp_path / "data", tmp_path / "restored")
    try:
        restored.resume(tmp_path / "legacy.pt")
        assert restored.global_step == 1
        assert restored.step() == trainer.step()
        run_export(tmp_path / "legacy.pt", tmp_path / "legacy-export")
        model, manifest = load_model(tmp_path / "legacy-export")
        assert model.cfg.attention_backend == manifest["attention_backend"] == "default"
        assert manifest["numerical_profile"] == "default"
    finally:
        for instance in (trainer, restored):
            instance.train_dataset.close()
            instance.validation_dataset.close()


def test_reference_profile_export_and_offline_evaluation_preserve_execution(tmp_path):
    make_data(tmp_path / "data")
    trainer = build(
        tmp_path / "data", tmp_path / "run", steps=2, numerical_profile="reference"
    )
    try:
        trainer.train()
        run_export(tmp_path / "run/last.pt", tmp_path / "export")
        model, manifest = load_model(tmp_path / "export")
        assert model.cfg.attention_backend == manifest["attention_backend"] == "math"
        assert (
            manifest["numerical_profile"]
            == manifest["training"]["numerical_profile"]
            == "reference"
        )
        report = run_eval(
            {"dataset": str(tmp_path / "data"), "model": str(tmp_path / "export")}
        )
        assert (
            report["attention_backend"] == "math"
            and report["numerical_profile"] == "reference"
        )
    finally:
        trainer.train_dataset.close()
        trainer.validation_dataset.close()


@pytest.mark.gpu
def test_reference_cuda_resume_preserves_model_adam_and_rng(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    make_data(tmp_path / "data")
    options = {
        "steps": 8,
        "dropout": 0.1,
        "numerical_profile": "reference",
        "device": "cuda:0",
    }
    full = build(tmp_path / "data", tmp_path / "full", **options)
    full.train()
    partial = build(tmp_path / "data", tmp_path / "partial", **options)
    partial.train(stop_after=3)
    resumed = build(tmp_path / "data", tmp_path / "resume", **options)
    try:
        resumed.resume(tmp_path / "partial/last.pt")
        resumed.train()
        for key, value in full.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[key])
        left, right = full.optimizer.state_dict(), resumed.optimizer.state_dict()
        assert left["param_groups"] == right["param_groups"]
        for index, state in left["state"].items():
            for field, tensor in state.items():
                assert torch.equal(tensor, right["state"][index][field])
        assert torch.equal(full.sampler.get_state(), resumed.sampler.get_state())
    finally:
        for trainer in (full, partial, resumed):
            trainer.train_dataset.close()
            trainer.validation_dataset.close()


@pytest.mark.gpu
def test_reference_cuda_clipping_stays_on_device_and_preserves_ordered_norm():
    from embodichain.learning.dynamics._numerics import _clip_grad_norm_

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # The official six-layer networks have 45 parameter tensors. Exercise both
    # cold and optimized TorchScript execution at that reduction length.
    parameters = [
        torch.nn.Parameter(torch.zeros(17 + index, device="cuda:0"))
        for index in range(45)
    ]
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, (index + 1) * 1e-3)
    original = [parameter.grad.clone() for parameter in parameters]
    norms = torch.stack(torch._foreach_norm(original, 2.0)).cpu().tolist()
    expected = torch.tensor(
        math.sqrt(sum(value * value for value in norms)), device="cuda:0"
    )
    for _ in range(10):
        for parameter, gradient in zip(parameters, original, strict=True):
            parameter.grad.copy_(gradient)
        actual = _clip_grad_norm_(
            parameters, 0.25, profile="reference", error_if_nonfinite=False
        )
        assert torch.equal(actual, expected)
    for parameter, gradient in zip(parameters, original, strict=True):
        parameter.grad.copy_(gradient)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = _clip_grad_norm_(
            parameters, 0.25, profile="reference", error_if_nonfinite=False
        )
    keys = {event.key for event in profile.key_averages()}
    assert actual.device == parameters[0].device
    assert not {"aten::item", "aten::_local_scalar_dense"} & keys
