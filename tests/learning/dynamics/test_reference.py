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
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from embodichain.data_pipeline.dynamics.shards import (
    DynamicsShardWriter,
    DynamicsWindowDataset,
)
from embodichain.learning.dynamics.reference import (
    export_legacy_dataset,
    run_reference,
    reference_identity,
    OFFICIAL_COMMIT,
)


def test_external_execution_requires_explicit_trust():
    with pytest.raises(ValueError, match="trust_external_code"):
        run_reference({})


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
    from embodichain.compute.dynamics import NeRDStateCodec
    from embodichain.learning.models.nerd import (
        NeRDModel,
        NeRDModelCfg,
        export_model,
        load_model,
    )
    from embodichain.learning.dynamics.trainer import DynamicsTrainer, DynamicsTrainCfg
    from embodichain.learning.dynamics.reference import _hash
    from embodichain.learning.dynamics.workflow import run_train, run_export

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
                str(Path(__file__).resolve().parents[3]),
            ]
        ),
    }
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodichain.learning.dynamics.reference",
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
    from embodichain.learning.dynamics.reference import import_legacy_datasets

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


def test_gravity_repair_is_explicit_preserves_raw_and_other_fields():
    from embodichain.learning.dynamics.reference import repair_gravity_transport

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
    from embodichain.learning.dynamics.reference import import_legacy_datasets
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
                str(Path(__file__).resolve().parents[3]),
            ]
        ),
    }
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodichain.learning.dynamics.reference",
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
    from embodichain.learning.dynamics.reference import _hash
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
        Path(__file__).resolve().parents[3]
        / "embodichain/learning/dynamics/reference.py"
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
from embodichain.learning.dynamics.reference import reference_identity,_restore_official_model
source=sys.argv[1]
reference_identity(source)
sys.path.insert(0,source)
from embodichain.learning.models.nerd import NeRDModel,NeRDModelCfg
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
