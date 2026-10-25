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

"""Legacy transport and opt-in independent official-trainer execution."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from embodichain_models.dynamics.nerd.data import (
    DynamicsShardWriter,
    DynamicsWindowDataset,
)
from examples.learning.nerd._support._reference import (
    export_legacy_dataset,
    run_reference,
    reference_identity,
    OFFICIAL_COMMIT,
)


def test_external_execution_requires_explicit_trust():
    with pytest.raises(ValueError, match="trust_external_code"):
        run_reference({})


@pytest.mark.parametrize("profile", ["reference", "unknown"])
def test_original_method_adapter_rejects_unsupported_profile_before_source(profile):
    with pytest.raises(ValueError, match="numerical_profile"):
        run_reference(
            {"train": {"numerical_profile": profile}}, trust_external_code=True
        )


def test_original_method_adapter_rejects_math_model_before_data(tmp_path, monkeypatch):
    import examples.learning.nerd._support._reference as reference
    import embodichain_models.dynamics.nerd as models

    for name in ("algorithms", "utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    vanilla = ModuleType("algorithms.vanilla_trainer")
    vanilla.VanillaTrainer = object
    timers = ModuleType("utils.time_report")
    timers.TimeReport = object
    monkeypatch.setitem(sys.modules, vanilla.__name__, vanilla)
    monkeypatch.setitem(sys.modules, timers.__name__, timers)
    monkeypatch.setattr(
        reference, "reference_identity", lambda source: {"root": str(tmp_path)}
    )
    monkeypatch.setattr(
        models,
        "load_model",
        lambda *args, **kwargs: (
            SimpleNamespace(cfg=SimpleNamespace(attention_backend="math")),
            {},
        ),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    try:
        with pytest.raises(ValueError, match="math attention initial_model"):
            run_reference(
                {
                    "source": "unused",
                    "initial_model": "unused",
                    "train": {"device": "cpu"},
                },
                trust_external_code=True,
            )
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


def test_official_transport_uses_the_exact_native_codec_velocity_order():
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from examples.learning.nerd._support._reference import _legacy_states

    with np.load(
        Path(__file__).resolve().parents[3]
        / "models/dynamics/nerd/data/ant-official-frame.npz"
    ) as fixture:
        q = torch.from_numpy(fixture["q"].copy())
        qd = torch.from_numpy(fixture["qd"].copy())
    codec = NeRDStateCodec("ant")
    actual = _legacy_states(q, qd, codec)
    torch.testing.assert_close(actual[..., :15], q, atol=0, rtol=0)
    torch.testing.assert_close(
        actual[..., 15:], codec.native_to_legacy(q, qd), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_training_transport_device_is_independent_of_raw_residency(device):
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from examples.learning.nerd._support._reference import _official_batch

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with np.load(
        Path(__file__).resolve().parents[3]
        / "models/dynamics/nerd/data/ant-official-frame.npz"
    ) as fixture:
        q = torch.from_numpy(fixture["q"].copy())
        qd = torch.from_numpy(fixture["qd"].copy())
    shape = (q.shape[0], q.shape[1] - 1)
    batch = {
        "states": {"q": q, "qd": qd},
        "transitions": {
            "joint_f": torch.zeros(*shape, 14),
            "contact_depths": torch.zeros(*shape, 25),
        },
    }
    resident = {
        key: {name: tensor.to(device) for name, tensor in fields.items()}
        for key, fields in batch.items()
    }
    codec = NeRDStateCodec("ant")
    lazy = _official_batch(batch, codec, device=device)
    cached = _official_batch(resident, codec, device=device)
    for name, actual in lazy.items():
        assert actual.device == torch.device(device)
        torch.testing.assert_close(actual, cached[name], atol=0, rtol=0)


def test_legacy_export_preserves_lengths_splits_and_native_to_twist(tmp_path):
    metadata = {"codec": {"robot": "ant", "base_com": [0.0, 0.0, 0.0]}, "dt": 1 / 60}
    with DynamicsShardWriter(tmp_path / "data", metadata) as writer:
        for index, length in enumerate((3, 5)):
            q = np.zeros((length + 1, 15), np.float32)
            q[:, 0] = 1.0
            q[:, 6] = 1.0
            qd = np.zeros((length + 1, 14), np.float32)
            qd[:, 1] = 2.0
            qd[:, 5] = 1.0
            transitions = {
                "joint_f": np.zeros((length, 14), np.float32),
                "contact_depths": np.zeros((length, 1), np.float32),
                "contact_thicknesses_0": np.full((length, 1), 0.02, np.float32),
                "contact_thicknesses_1": np.full((length, 1), 0.03, np.float32),
                "contact_normals": np.tile([0.0, 0.0, 1.0], (length, 1)).astype(
                    np.float32
                ),
                "contact_points_1": np.zeros((length, 3), np.float32),
            }
            writer.add_trajectory(
                str(index), {"q": q, "qd": qd}, transitions, split="train"
            )
    result = export_legacy_dataset(tmp_path / "data", tmp_path / "legacy.h5", "train")
    with h5py.File(tmp_path / "legacy.h5") as file:
        data = file["data"]
        assert data["states"].shape == (5, 2, 29)
        assert data["traj_lengths"][()].tolist() == [3, 5]
        np.testing.assert_allclose(
            data["states"][0, 0, 15:21], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(data["contact_thicknesses"][0], 0.05)
        assert data["joint_acts"].shape == (5, 2, 8)
    assert result["native_to_legacy_qd"] is True


@pytest.mark.skipif(
    not os.getenv("NERD_REFERENCE_SOURCE"),
    reason="Explicit trusted official checkout/dependency environment required",
)
def test_actual_official_main_two_updates_match_migrated(tmp_path):
    """External reference is opt-in; synthetic data tests migration, not accuracy."""
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd import (
        NeRDModel,
        NeRDModelCfg,
        export_model,
        load_model,
    )
    from examples.learning.nerd._support._trainer import (
        DynamicsTrainer,
        DynamicsTrainCfg,
    )
    from examples.learning.nerd._support._reference import _hash
    from examples.learning.nerd._support._workflow import run_train, run_export

    metadata = {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    with DynamicsShardWriter(tmp_path / "data", metadata) as writer:
        for split in ("train", "validation", "test"):
            q = np.zeros((7, 2), np.float32)
            q[:, 0] = np.arange(7) * 0.02
            qd = np.zeros_like(q)
            qd[:, 0] = 1.2
            writer.add_trajectory(
                split,
                {"q": q, "qd": qd},
                {"joint_f": np.zeros((6, 2), np.float32)},
                split=split,
            )
    source = os.environ["NERD_REFERENCE_SOURCE"]
    assert reference_identity(source)["commit"] == OFFICIAL_COMMIT
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = NeRDModel(
        NeRDModelCfg(
            input_dims={"states_embedding": 4, "joint_f": 2},
            output_dim=4,
            n_embd=12,
            n_head=3,
            n_layer=1,
            history_length=3,
            block_size=3,
            head_hidden_dims=(12,),
        )
    )
    train = DynamicsWindowDataset(tmp_path / "data", "train", 3)
    validation = DynamicsWindowDataset(tmp_path / "data", "validation", 3)
    cfg = DynamicsTrainCfg(total_steps=2, batch_size=2, validation_interval=1, seed=4)
    trainer = DynamicsTrainer(
        model,
        NeRDStateCodec("cartpole"),
        train,
        validation,
        cfg,
        tmp_path / "scratch",
        manifest=metadata,
    )
    export_model(tmp_path / "initial", trainer.model, metadata)
    np.save(tmp_path / "order.npy", np.array([[0, 1], [1, 2]], np.int64))
    spec = {
        "source": source,
        "dataset": str(tmp_path / "data"),
        "dataset_fingerprint": train.fingerprint,
        "initial_model": str(tmp_path / "initial"),
        "initial_model_sha256": _hash(tmp_path / "initial" / "weights.pt"),
        "sample_order": str(tmp_path / "order.npy"),
        "sample_order_sha256": _hash(tmp_path / "order.npy"),
        "train": cfg.to_dict(),
        "output_dir": str(tmp_path / "official"),
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            [
                os.environ.get("NERD_REFERENCE_DEPS", ""),
                source,
                str(Path(__file__).resolve().parents[4]),
                str(
                    Path(__file__).resolve().parents[4]
                    / "extensions/embodichain_models"
                ),
            ]
        ),
    }
    process = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[4]
                / "examples/learning/nerd/_support/_worker.py"
            ),
            "_reference",
            "--spec",
            str(path),
            "--trust-external-code",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    receipt = json.loads((tmp_path / "official" / "result.json").read_text())
    assert receipt["runtime_identity"]["parameter_placements"] == [
        {"device": "cpu", "dtype": "torch.float32"}
    ]
    assert (
        receipt["runtime_identity"]["cuda"]["driver_inventory"]["status"] == "not_used"
    )
    run_train(
        {
            "dataset": spec["dataset"],
            "output_dir": str(tmp_path / "candidate"),
            "fine_tune": spec["initial_model"],
            "sample_order": spec["sample_order"],
            "train": cfg.to_dict(),
        }
    )
    run_export(tmp_path / "candidate" / "last.pt", tmp_path / "candidate-export")
    reference, _ = load_model(tmp_path / "official" / "last-export")
    candidate, _ = load_model(tmp_path / "candidate-export")
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(
            value, reference.state_dict()[name], atol=1e-5, rtol=1e-4
        )
    report = json.loads((tmp_path / "official" / "result.json").read_text())
    assert report["completed_updates"] == 2
    assert report["reference"]["source_patches"] == []
    train.close()
    validation.close()


def test_legacy_roundtrip_preserves_native_qd_and_declared_split(tmp_path):
    from examples.learning.nerd._support._reference import import_legacy_datasets

    metadata = {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    with DynamicsShardWriter(tmp_path / "original", metadata) as writer:
        q = np.arange(12, dtype=np.float32).reshape(6, 2) * 0.01
        writer.add_trajectory(
            "sample",
            {"q": q, "qd": q * 0.5},
            {"joint_f": np.zeros((5, 2), np.float32)},
            split="validation",
        )
    export_legacy_dataset(tmp_path / "original", tmp_path / "legacy.h5", "validation")
    imported = import_legacy_datasets(
        {"validation": tmp_path / "legacy.h5"}, tmp_path / "imported", metadata
    )
    assert imported["trajectories"] == 1
    data = DynamicsWindowDataset(tmp_path / "imported", "validation", 3)
    np.testing.assert_allclose(data[0]["states"]["q"], q[:4])
    np.testing.assert_allclose(data[0]["states"]["qd"], q[:4] * 0.5)
    assert len(DynamicsWindowDataset(tmp_path / "imported", "train", 3)) == 0
    data.close()


def _write_source_ant_velocities(path: Path) -> np.ndarray:
    """Make continuous source states exposing cancellation and signed-zero loss."""
    horizon, trajectories, state_dim = 5, 2, 29
    states = np.zeros((horizon + 1, trajectories, state_dim), np.float32)
    states[..., :3] = [10.0, 20.0, 30.0]
    states[..., 6] = 1.0
    states[..., 15:18] = [5.0, 6.0, 7.0]
    states[..., 18:21] = [0.3, 0.1, 0.2]
    states[:, :, 18] += np.arange(horizon + 1, dtype=np.float32)[:, None] * 0.01
    states[0, 1, 19] = -0.0
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["mode"] = "trajectory"
        data.create_dataset("states", data=states[:-1])
        data.create_dataset("next_states", data=states[1:])
        data.create_dataset("traj_lengths", data=np.array([3, 5], np.int32))
        data.create_dataset("root_body_q", data=states[:-1, :, :7])
        data.create_dataset(
            "joint_acts", data=np.zeros((horizon, trajectories, 8), np.float32)
        )
        data.create_dataset(
            "contact_depths",
            data=np.broadcast_to(
                np.array([0.01, 100.0], np.float32), (horizon, trajectories, 2)
            ),
        )
        data.create_dataset(
            "contact_thicknesses",
            data=np.full((horizon, trajectories, 2), 0.02, np.float32),
        )
        data.create_dataset(
            "contact_points_1", data=np.zeros((horizon, trajectories, 6), np.float32)
        )
        data.create_dataset(
            "contact_normals", data=np.zeros((horizon, trajectories, 6), np.float32)
        )
        data.create_dataset(
            "gravity_dir",
            data=np.broadcast_to(
                np.array([0.0, -1.0, 0.0], np.float32), (horizon, trajectories, 3)
            ),
        )
    return states


def test_original_velocity_import_roundtrip_preserves_fp32_bits_and_defaults(tmp_path):
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd.data import load_manifest
    from examples.learning.nerd._support._reference import (
        import_legacy_datasets,
        _official_batch,
    )

    path = tmp_path / "source.h5"
    original = _write_source_ant_velocities(path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"codec": {"robot": "ant"}, "dt": 1 / 60}
    imported = import_legacy_datasets(
        {"validation": path},
        tmp_path / "preserved",
        metadata,
        preserve_legacy_velocity=True,
        trajectories_per_shard=1,
    )
    import_legacy_datasets({"validation": path}, tmp_path / "default", metadata)
    manifest = load_manifest(tmp_path / "preserved", verify=True)
    baseline = load_manifest(tmp_path / "default", verify=True)
    assert imported["trajectories"] == 2
    assert manifest["fields"]["states"]["legacy_velocity"]["shape"] == [14]
    assert "legacy_velocity" not in baseline["fields"]["states"]
    assert "legacy_velocity" not in baseline["metadata"]
    without_contract = dict(manifest["metadata"])
    contract = without_contract.pop("legacy_velocity")
    assert without_contract == baseline["metadata"]
    assert contract["world_frame"] == "source"
    assert contract["field"] == "states/legacy_velocity"
    assert manifest["metadata"]["legacy_sources"]["validation"]["sha256"] == source_sha
    assert metadata == {"codec": {"robot": "ant"}, "dt": 1 / 60}
    codec = NeRDStateCodec("ant")
    for shard in manifest["shards"]:
        with h5py.File(tmp_path / "preserved" / shard["file"], "r") as file:
            for record in shard["trajectories"]:
                index, length = record["metadata"]["legacy_index"], record["length"]
                group = file[record["group"]]
                stored = group["states/legacy_velocity"][()]
                expected = original[: length + 1, index, 15:]
                np.testing.assert_array_equal(
                    stored.view(np.uint32), expected.view(np.uint32)
                )
                batch = {
                    kind: {
                        name: torch.from_numpy(field[()]).unsqueeze(0)
                        for name, field in group[kind].items()
                    }
                    for kind in ("states", "transitions")
                }
                raw = _official_batch(batch, codec)
                np.testing.assert_array_equal(
                    raw["states"][0].numpy().view(np.uint32),
                    original[:length, index].view(np.uint32),
                )
                np.testing.assert_array_equal(
                    raw["next_states"][0].numpy().view(np.uint32),
                    original[1 : length + 1, index].view(np.uint32),
                )
    exported = export_legacy_dataset(
        tmp_path / "preserved", tmp_path / "export.h5", "validation"
    )
    assert exported["native_to_legacy_qd"] is False
    with h5py.File(tmp_path / "export.h5", "r") as file:
        for index, length in enumerate((3, 5)):
            np.testing.assert_array_equal(
                file["data/states"][:length, index].view(np.uint32),
                original[:length, index].view(np.uint32),
            )
            np.testing.assert_array_equal(
                file["data/next_states"][:length, index].view(np.uint32),
                original[1 : length + 1, index].view(np.uint32),
            )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_sha


def test_preserved_velocity_follows_prefix_masks_and_cpu_residency(tmp_path):
    from torch.utils.data._utils.collate import default_collate
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from examples.learning.nerd._support._reference import (
        import_legacy_datasets,
        _official_batch,
    )

    source = tmp_path / "source.h5"
    original = _write_source_ant_velocities(source)
    import_legacy_datasets(
        {"train": source},
        tmp_path / "data",
        {"codec": {"robot": "ant"}},
        preserve_legacy_velocity=True,
        trajectories_per_shard=1,
    )
    dataset = DynamicsWindowDataset(tmp_path / "data", "train", 3, include_prefix=True)
    indices = [0, 2, 3, 7, 0]
    individually = default_collate([dataset[index] for index in indices])
    grouped = dataset.get_batch(indices)
    for name, value in grouped["states"].items():
        torch.testing.assert_close(value, individually["states"][name], atol=0, rtol=0)
    expected = np.concatenate(
        (original[:2, 0, 15:], np.repeat(original[1:2, 0, 15:], 2, axis=0))
    )
    np.testing.assert_array_equal(
        grouped["states"]["legacy_velocity"][0].numpy().view(np.uint32),
        expected.view(np.uint32),
    )
    assert grouped["valid_mask"][0].tolist() == [True, False, False]
    codec = NeRDStateCodec("ant")
    transported = _official_batch(grouped, codec)
    dataset.enable_device_cache("cpu", max_bytes=1024 * 1024)
    cached = dataset.get_batch(indices)
    for name, value in cached["states"].items():
        torch.testing.assert_close(value, grouped["states"][name], atol=0, rtol=0)
    for name, value in _official_batch(cached, codec).items():
        torch.testing.assert_close(value, transported[name], atol=0, rtol=0)
    cached["states"]["legacy_velocity"][0, 0, 0] = 999
    assert dataset.get_batch(indices)["states"]["legacy_velocity"][0, 0, 0] != 999
    dataset.close()


def test_disabling_velocity_preservation_clears_only_copied_contract(tmp_path):
    from embodichain_models.dynamics.nerd.data import load_manifest
    from examples.learning.nerd._support._reference import import_legacy_datasets

    path = tmp_path / "source.h5"
    _write_source_ant_velocities(path)
    import_legacy_datasets(
        {"train": path},
        tmp_path / "preserved",
        {"codec": {"robot": "ant"}},
        preserve_legacy_velocity=True,
    )
    metadata = load_manifest(tmp_path / "preserved")["metadata"]
    original = json.loads(json.dumps(metadata))
    import_legacy_datasets(
        {"train": path},
        tmp_path / "native",
        metadata,
        preserve_legacy_velocity=False,
    )
    native = load_manifest(tmp_path / "native")
    assert "legacy_velocity" not in native["metadata"]
    assert "legacy_velocity" not in native["fields"]["states"]
    assert metadata == original


@pytest.mark.parametrize("mismatch", ["small_delta", "signed_zero"])
def test_lossless_import_rejects_nonidentical_adjacent_fp32_states(tmp_path, mismatch):
    from examples.learning.nerd._support._reference import import_legacy_datasets

    path = tmp_path / "source.h5"
    _write_source_ant_velocities(path)
    with h5py.File(path, "a") as file:
        if mismatch == "small_delta":
            file["data/next_states"][0, 0, 18] += np.float32(1e-7)
        else:
            file["data/next_states"][0, 0, 21] = -0.0
    # The old tolerance remains available to existing callers.
    import_legacy_datasets(
        {"train": path}, tmp_path / "default", {"codec": {"robot": "ant"}}
    )
    with pytest.raises(ValueError, match="bitwise continuous"):
        import_legacy_datasets(
            {"train": path},
            tmp_path / "preserved",
            {"codec": {"robot": "ant"}},
            preserve_legacy_velocity=True,
        )


def test_lossless_import_rejects_even_explicit_identity_world_rotation(tmp_path):
    from examples.learning.nerd._support._reference import import_legacy_datasets

    with pytest.raises(ValueError, match="source world frame"):
        import_legacy_datasets(
            {"train": tmp_path / "unused.h5"},
            tmp_path / "preserved",
            {"codec": {"robot": "ant"}},
            preserve_legacy_velocity=True,
            world_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    assert not (tmp_path / "preserved").exists()


def test_official_transport_rejects_malformed_preserved_velocity_shape():
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from examples.learning.nerd._support._reference import _official_batch

    q = torch.zeros(1, 2, 15)
    q[..., 6] = 1
    batch = {
        "states": {
            "q": q,
            "qd": torch.zeros(1, 2, 14),
            "legacy_velocity": torch.zeros(1, 2, 13),
        },
        "transitions": {"joint_f": torch.zeros(1, 1, 14)},
    }
    with pytest.raises(ValueError, match="Preserved legacy velocity shape"):
        _official_batch(batch, NeRDStateCodec("ant"))


def test_gravity_repair_is_explicit_preserves_raw_and_other_fields():
    from examples.learning.nerd._support._reference import repair_gravity_transport

    raw = np.array([[[np.nan, -1.0, 8.49e5], [1e-45, -1.0, np.nan]]], np.float32)
    before = raw.copy()
    with pytest.raises(ValueError, match="uninitialized gravity"):
        repair_gravity_transport(raw, (0.0, -1.0, 0.0))
    repaired, evidence = repair_gravity_transport(raw, (0.0, -1.0, 0.0), enabled=True)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(
        repaired, np.array([[[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]])
    )
    assert evidence["nonfinite_components"] == 2
    assert evidence["enabled"] is True


def test_import_explicit_world_rotation_and_gravity_transport(tmp_path):
    from examples.learning.nerd._support._reference import import_legacy_datasets
    import math

    metadata = {"codec": {"robot": "ant", "base_com": [0.0, 0.0, 0.0]}, "dt": 1 / 60}
    q = np.zeros((4, 15), np.float32)
    q[:, 1] = 1.0
    q[:, 3] = -math.sqrt(0.5)
    q[:, 6] = math.sqrt(0.5)
    qd = np.zeros((4, 14), np.float32)
    qd[:, 0] = 1.0
    qd[:, 4] = 1.0
    with DynamicsShardWriter(tmp_path / "source", metadata) as writer:
        writer.add_trajectory(
            "one",
            {"q": q, "qd": qd},
            {
                "joint_f": np.zeros((3, 14), np.float32),
                "contact_depths": np.zeros((3, 1), np.float32),
                "gravity_dir": np.tile([0.0, -1.0, 0.0], (3, 1)).astype(np.float32),
            },
            split="train",
        )
    export_legacy_dataset(tmp_path / "source", tmp_path / "raw.h5", "train")
    with h5py.File(tmp_path / "raw.h5", "a") as file:
        file["data/gravity_dir"][:, :, 0] = np.nan
    import_legacy_datasets(
        {"train": tmp_path / "raw.h5"},
        tmp_path / "canonical",
        metadata,
        world_rotation_xyzw=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        gravity_direction=(0.0, -1.0, 0.0),
    )
    dataset = DynamicsWindowDataset(tmp_path / "canonical", "train", 3)
    item = dataset[0]
    np.testing.assert_allclose(
        item["states"]["q"][0, :7], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        item["states"]["qd"][0, :6], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        item["transitions"]["gravity_dir"][0], [0.0, 0.0, -1.0], atol=1e-6
    )
    assert dataset.manifest["metadata"]["gravity_transport_override"] == [
        0.0,
        -1.0,
        0.0,
    ]
    dataset.close()


@pytest.mark.skipif(
    not os.getenv("NERD_REFERENCE_SOURCE"),
    reason="Explicit trusted official checkout/dependency environment required",
)
def test_real_official_collection_streams_batches_and_preserves_raw(tmp_path):
    spec = {
        "source": os.environ["NERD_REFERENCE_SOURCE"],
        "output_dir": str(tmp_path / "collection"),
        "robot": "cartpole",
        "warp_version": "1.8.0",
        "device": "cpu",
        "num_envs": 2,
        "horizon": 4,
        "splits": {"train": 5, "validation": 2, "test": 2},
        "repair_uninitialized_gravity": True,
        "canonical_world": "z_up",
    }
    spec_path = tmp_path / "generate.json"
    spec_path.write_text(json.dumps(spec))
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            [
                os.environ.get("NERD_REFERENCE_DEPS", ""),
                spec["source"],
                str(Path(__file__).resolve().parents[4]),
                str(
                    Path(__file__).resolve().parents[4]
                    / "extensions/embodichain_models"
                ),
            ]
        ),
    }
    process = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[4]
                / "examples/learning/nerd/_support/_worker.py"
            ),
            "_reference",
            "--spec",
            str(spec_path),
            "--trust-external-code",
            "--generate",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    root = tmp_path / "collection"
    result = json.loads((root / "generation-result.json").read_text())
    assert result["source_trajectory_counts"] == {
        "train": 5,
        "validation": 2,
        "test": 2,
    }
    assert result["gravity_repair"]["train"]["sampler_calls"] == 3
    assert result["source_bug_corrected"] is True
    assert result["teacher_origin"]["integrator"] == "featherstone"
    physics = json.loads((root / "source-physics.json").read_text())
    assert physics["physics_contract_version"] == "official-warp-physical-v2"
    assert (
        result["teacher_origin"]["physics_contract_version"]
        == physics["physics_contract_version"]
    )
    assert result["teacher_origin"]["up_axis"] == 1
    np.testing.assert_allclose(
        physics["source_physical_arrays"]["gravity"], [0.0, -9.81, 0.0], atol=1e-6
    )
    assert (
        physics["source_physical_arrays"]["gravity"]
        == result["teacher_origin"]["gravity"]
    )
    assert result["teacher_origin"]["gravity_magnitude"] == pytest.approx(
        9.81, abs=1e-6
    )
    from examples.learning.nerd._support._reference import _hash
    import hashlib

    assert physics["source_generator_files"]["generate/trajectory_sampler.py"] == _hash(
        Path(spec["source"]) / "generate/trajectory_sampler.py"
    )
    expected_generator_hash = hashlib.sha256(
        json.dumps(
            physics["source_generator_files"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert physics["source_generator_sha256"] == expected_generator_hash
    assert physics["generation_adapter_sha256"] == _hash(
        Path(__file__).resolve().parents[4]
        / "examples/learning/nerd/_support/_reference.py"
    )
    with h5py.File(root / "train.raw.h5") as file:
        assert file["data/states"].shape == (4, 5, 4)
    dataset = DynamicsWindowDataset(root / "canonical", "train", 3)
    assert len(dataset) == 10
    assert (
        dataset.manifest["metadata"]["source_physical_arrays"]["gravity"]
        == physics["source_physical_arrays"]["gravity"]
    )
    assert (
        dataset.manifest["metadata"]["generation_adapter_sha256"]
        == physics["generation_adapter_sha256"]
    )
    np.testing.assert_allclose(
        dataset[0]["transitions"]["gravity_dir"],
        np.tile([0.0, 0.0, -1.0], (3, 1)),
        atol=1e-6,
    )
    dataset.close()


@pytest.mark.skipif(
    not os.getenv("NERD_REFERENCE_SOURCE"),
    reason="Explicit trusted official checkout required",
)
def test_portable_to_official_restoration_preserves_state_dtype_and_ownership():
    code = r"""
import sys,torch
from examples.learning.nerd._support._reference import reference_identity,_restore_official_model
source=sys.argv[1]
reference_identity(source)
sys.path.insert(0,source)
from embodichain_models.dynamics.nerd import NeRDModel,NeRDModelCfg
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(5)
    common=NeRDModel(NeRDModelCfg(input_dims={"states_embedding":4,"joint_f":2},output_dim=4,n_embd=12,n_head=3,n_layer=1,block_size=4,history_length=3,head_hidden_dims=(12,),dropout=.25,output_tanh=True)).to(dtype=dtype).eval()
    common.set_normalization({"states_embedding":(torch.arange(4,dtype=dtype),torch.arange(4,dtype=dtype)+.5),"joint_f":(torch.tensor([.1,-.3],dtype=dtype),torch.tensor([.9,1.2],dtype=dtype))},torch.arange(4,dtype=dtype)*.2,torch.arange(4,dtype=dtype)+.3)
    old=_restore_official_model(common,device="cpu")
    assert not old.training
    for name,value in old.named_parameters():
        assert value.dtype==dtype
        torch.testing.assert_close(value,dict(common.named_parameters())[name],atol=0,rtol=0)
    inputs={"states_embedding":torch.randn(2,3,4,dtype=dtype),"joint_f":torch.randn(2,3,2,dtype=dtype)}
    legacy=lambda:{("joint_acts" if name=="joint_f" else name):value.clone() for name,value in inputs.items()}
    torch.testing.assert_close(old(legacy()),common(inputs),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(old.evaluate(legacy()),common(inputs)[:,-1:,:],atol=1e-6,rtol=1e-5)
    before=common.input_rms["joint_f"].mean.clone()
    old.input_rms["joint_acts"].mean.add_(1.)
    torch.testing.assert_close(common.input_rms["joint_f"].mean,before,atol=0,rtol=0)
assert "dexsim" not in sys.modules
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code, os.environ["NERD_REFERENCE_SOURCE"]],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not os.getenv("NERD_REFERENCE_SOURCE"), reason="Explicit trusted reference required"
)
def test_actual_official_trainer_resume_preserves_optimizer_rng_and_best_model(
    tmp_path,
):
    """Synthetic CPU data verify lifecycle, not learned-dynamics accuracy."""
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd import (
        NeRDModel,
        NeRDModelCfg,
        export_model,
        load_model,
    )
    from examples.learning.nerd._support._trainer import compute_statistics
    from examples.learning.nerd._support._sample_order import _write_checkpointed_order
    from examples.learning.nerd._support._reference import _hash

    metadata = {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    generator = np.random.default_rng(3)
    with DynamicsShardWriter(tmp_path / "data", metadata) as writer:
        for split in ["train", "validation", "test"]:
            for index in range(2):
                q = generator.normal(size=(13, 2)).astype(np.float32)
                qd = generator.normal(size=(13, 2)).astype(np.float32)
                writer.add_trajectory(
                    f"{split}:{index}",
                    {"q": q, "qd": qd},
                    {"joint_f": np.zeros((12, 2), np.float32)},
                    split=split,
                )
    dataset = DynamicsWindowDataset(tmp_path / "data", "train", 3)
    codec = NeRDStateCodec("cartpole")
    stats = compute_statistics(dataset, codec)
    torch.manual_seed(4)
    model = NeRDModel(
        NeRDModelCfg(
            input_dims={"states_embedding": 4, "joint_f": 2},
            output_dim=4,
            n_embd=12,
            n_head=3,
            n_layer=1,
            block_size=4,
            history_length=3,
            dropout=0.1,
        )
    )
    model.set_normalization(
        {k: (v["mean"], v["var"]) for k, v in stats["inputs"].items()},
        stats["output"]["mean"],
        stats["output"]["var"],
    )
    export_model(tmp_path / "initial", model, metadata)
    order, _ = _write_checkpointed_order(
        tmp_path / "order.json",
        dataset_size=len(dataset),
        steps=6,
        batch_size=4,
        seed=7,
        block_steps=2,
    )
    cfg = {
        "total_steps": 6,
        "batch_size": 4,
        "seed": 7,
        "validation_interval": 2,
        "checkpoint_interval": 2,
        "log_interval": 2,
        "resident_cache_bytes": 1024 * 1024,
        "device": "cpu",
    }
    source = os.environ["NERD_REFERENCE_SOURCE"]
    spec = {
        "source": source,
        "dataset": str(tmp_path / "data"),
        "dataset_fingerprint": dataset.fingerprint,
        "initial_model": str(tmp_path / "initial"),
        "initial_model_sha256": _hash(tmp_path / "initial/weights.pt"),
        "sample_order": str(order),
        "sample_order_sha256": _hash(order),
        "train": cfg,
    }
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(
            [
                os.environ.get("NERD_REFERENCE_DEPS", ""),
                source,
                str(Path(__file__).resolve().parents[4]),
                str(
                    Path(__file__).resolve().parents[4]
                    / "extensions/embodichain_models"
                ),
            ]
        ),
    }

    def execute(name, extra):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({**spec, "output_dir": str(tmp_path / name)}))
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[4]
                    / "examples/learning/nerd/_support/_worker.py"
                ),
                "_reference",
                "--spec",
                str(path),
                "--trust-external-code",
                *extra,
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    execute("full", [])
    execute("resumed", ["--stop-after", "3"])
    interrupted = json.loads((tmp_path / "resumed/result.json").read_text())
    assert (
        interrupted["status"] == "interrupted" and interrupted["completed_updates"] == 3
    )
    execute("resumed", ["--resume", str(tmp_path / "resumed/reference-last.pt")])
    for artifact in ["best-export", "last-export"]:
        first, _ = load_model(tmp_path / "full" / artifact)
        second, _ = load_model(tmp_path / "resumed" / artifact)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(second.state_dict()[key], value, atol=0, rtol=0)
    dataset.close()
