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
    normalized_mse,
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


def build(root, out, *, steps=30, dropout=0.0):
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


def test_exact_resume_with_dropout_optimizer_sampler_rng(tmp_path):
    make_data(tmp_path / "data")
    full = build(tmp_path / "data", tmp_path / "full", steps=20, dropout=0.1)
    full.train()
    expected = {key: value.clone() for key, value in full.model.state_dict().items()}
    interrupted = build(tmp_path / "data", tmp_path / "resume", steps=20, dropout=0.1)
    assert interrupted.train(stop_after=7)["status"] == "interrupted"
    restored = build(tmp_path / "data", tmp_path / "resume", steps=20, dropout=0.1)
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


def test_workflow_fine_tune_and_resume_contract(tmp_path):
    metadata = make_data(tmp_path / "data")
    cfg = {
        "dataset": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "first"),
        "manifest": metadata,
        "model": architecture().to_dict(),
        "train": {
            "total_steps": 2,
            "batch_size": 4,
            "validation_interval": 1,
            "checkpoint_interval": 1,
        },
    }
    run_train(cfg)
    run_export(tmp_path / "first" / "last.pt", tmp_path / "export")
    fine = {
        **cfg,
        "output_dir": str(tmp_path / "fine"),
        "fine_tune": str(tmp_path / "export"),
    }
    assert run_train(fine)["global_step"] == 2
    mismatch = {
        **cfg,
        "output_dir": str(tmp_path / "bad"),
        "resume": str(tmp_path / "first" / "last.pt"),
        "train": {**cfg["train"], "total_steps": 3},
    }
    with pytest.raises(ValueError, match="train_cfg mismatch"):
        run_train(mismatch)


def test_official_loss_order_and_mask():
    prediction = torch.tensor([[[1.0, 2.0], [500.0, 500.0]]], requires_grad=True)
    target = torch.tensor([[[2.0, 5.0], [0.0, 0.0]]])
    variance = torch.tensor([4.0, 9.0])
    weights = torch.rsqrt(variance + 1e-5)
    expected = torch.nn.functional.mse_loss(
        prediction[:, :1] * weights, target[:, :1] * weights
    )
    actual = normalized_mse(prediction, target, variance, torch.tensor([[True, False]]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0


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
    with pytest.raises(ValueError, match="checksum"):
        invalid.resume(tmp_path / "run" / "last.pt")


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


def test_training_annotations_cannot_invent_missing_physics_contract(tmp_path):
    metadata = make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="physics_signature mismatch"):
        run_train(
            {
                "dataset": str(tmp_path / "data"),
                "output_dir": str(tmp_path / "run"),
                "manifest": {**metadata, "physics_signature": "invented"},
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
