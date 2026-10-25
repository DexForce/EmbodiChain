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

"""Historical runtime qualification through explicitly trusted isolated Warp."""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from examples.learning.nerd._support._reference_rollout import (
    evaluate_reference_rollouts,
)

_ROOT = Path(__file__).resolve().parents[4]


def test_external_reference_requires_explicit_trust() -> None:
    with pytest.raises(ValueError, match="trust_external_code"):
        evaluate_reference_rollouts({})


@pytest.mark.parametrize("robot,version", [("cartpole", "1.5.1"), ("ant", "1.8.0")])
def test_wrong_historical_version_rejected_before_source_import(
    robot: str, version: str
) -> None:
    with pytest.raises(ValueError, match="pinned Warp"):
        evaluate_reference_rollouts(
            {"robot": robot, "warp_version": version}, trust_external_code=True
        )


def _check_original_runtime(tmp_path: Path, robot: str, source: str) -> None:
    import numpy as np
    import h5py
    import torch
    import warp as wp
    from examples.learning.nerd._support._reference import generate_official_dataset
    from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg, export_model
    from embodichain_models.dynamics.nerd.data import load_manifest

    # Generate new physical provenance from the actual original teacher. No
    # current-Newton manifest is rewritten to pretend it describes old physics.
    with wp.ScopedDevice("cpu"):
        generation = generate_official_dataset(
            {
                "source": source,
                "robot": robot,
                "output_dir": str(tmp_path / "source-data"),
                "device": "cpu",
                "num_envs": 2,
                "horizon": 4,
                "splits": {"train": 2, "validation": 2, "test": 2},
                "seed": 3,
                "initial_states_source": "env",
                "canonical_world": "z_up",
                "repair_uninitialized_gravity": True,
                "cpu_threads": 2,
            },
            trust_external_code=True,
        )
    metadata = load_manifest(tmp_path / "source-data/canonical")["metadata"]
    assert "gravity" in metadata["source_physical_arrays"]
    dims = (
        {"states_embedding": 4, "joint_f": 2}
        if robot == "cartpole"
        else {
            "states_embedding": 29,
            "contact_normals": 75,
            "contact_points_1": 75,
            "contact_depths": 25,
            "joint_f": 8,
            "gravity_dir": 3,
        }
    )
    torch.manual_seed(7)
    model = NeRDModel(
        NeRDModelCfg(
            input_dims=dims,
            output_dim=4 if robot == "cartpole" else 29,
            n_embd=24,
            n_head=3,
            n_layer=1,
            history_length=3,
            block_size=4,
        )
    )
    with torch.no_grad():
        model.model.output_net.weight.mul_(0.02)
        model.model.output_net.bias.zero_()
        if robot == "ant":
            model.model.output_net.bias[6] = 1.0
    export = tmp_path / "model"
    export_model(export, model, metadata)
    reports = {}
    for mode in ["passive", "random_effort", "fixed_policy"]:
        spec = {
            "source": source,
            "robot": robot,
            "model": str(export),
            "output": str(tmp_path / f"{mode}.json"),
            "device": "cpu",
            "num_envs": 2,
            "trajectories": 3,
            "horizons": [1, 4],
            "seed": 17,
            "initial_states": "default",
            "motion_mode": mode,
            "cpu_threads": 2,
        }
        if mode == "fixed_policy":
            spec["policy"] = {"kind": "pd"}
        reports[mode] = evaluate_reference_rollouts(spec, trust_external_code=True)
        report = reports[mode]
        assert report["teacher_origin"]["kind"] == "official_warp"
        assert report["runtime_identity"]["engine"] == "official_warp"
        assert report["physical_verification"]["complete"]
        assert report["physical_verification"]["missing_recorded_fields"] == []
        assert report["portable_runtime_equivalence"]["status"] == "passed"
        assert report["portable_runtime_equivalence"]["max_abs_error"] <= 1e-5
        assert report["trajectories"] == 3 and report["num_envs"] == 2
        assert len(Path(report["trajectory_evidence"]).read_text().splitlines()) == 3
        with h5py.File(report["trace_evidence"], "r") as evidence:
            assert evidence["0/q"].shape[0] == 2 and evidence["1/q"].shape[0] == 1
            if mode == "fixed_policy":
                assert not np.allclose(
                    evidence["0/teacher_joint_f"][:, 1:],
                    evidence["0/joint_f"][:, 1:],
                    atol=1e-8,
                )
        assert report["selected_teacher_sha256"]
        assert (
            report["trace_evidence_sha256"]
            == hashlib.sha256(Path(report["trace_evidence"]).read_bytes()).hexdigest()
        )
        assert (
            report["horizons"]["4"]["valid_trajectories"]
            + report["horizons"]["4"]["numerically_invalid_trajectories"]
            == 3
        )
    # Run an independently restored equal export: teacher identity and content
    # are independent of the neural artifact's path, and results remain exact.
    second = tmp_path / "equal-model"
    export_model(second, model, metadata)
    again = evaluate_reference_rollouts(
        {
            "source": source,
            "robot": robot,
            "model": str(second),
            "output": str(tmp_path / "equal.json"),
            "device": "cpu",
            "num_envs": 2,
            "trajectories": 3,
            "horizons": [1, 4],
            "seed": 17,
            "initial_states": "default",
            "motion_mode": "passive",
            "cpu_threads": 2,
        },
        trust_external_code=True,
    )
    assert (
        again["selected_teacher_sha256"]
        == reports["passive"]["selected_teacher_sha256"]
    )
    assert again["horizons"] == reports["passive"]["horizons"]
    # Original per-trajectory sampling is deterministic across replay batching.
    sample_reports = []
    for size in [1, 2]:
        sample_reports.append(
            evaluate_reference_rollouts(
                {
                    "source": source,
                    "robot": robot,
                    "model": str(export),
                    "output": str(tmp_path / f"sample-{size}.json"),
                    "device": "cpu",
                    "num_envs": size,
                    "trajectories": 3,
                    "horizons": [1],
                    "seed": 31,
                    "initial_states": "sample",
                    "motion_mode": "passive",
                    "cpu_threads": 2,
                },
                trust_external_code=True,
            )
        )
    assert (
        sample_reports[0]["selected_teacher_sha256"]
        == sample_reports[1]["selected_teacher_sha256"]
    )
    if robot == "cartpole":
        with torch.no_grad():
            model.model.output_net.bias[2:] = torch.finfo(torch.float32).max
        divergent = tmp_path / "divergent-model"
        export_model(divergent, model, metadata)
        failure = evaluate_reference_rollouts(
            {
                "source": source,
                "robot": robot,
                "model": str(divergent),
                "output": str(tmp_path / "nonfinite.json"),
                "num_envs": 2,
                "trajectories": 3,
                "horizons": [1, 3],
                "initial_states": "default",
                "motion_mode": "passive",
                "cpu_threads": 2,
            },
            trust_external_code=True,
        )
        assert failure["nonfinite_rate"] == 1.0 and failure["failure_rate"] == 1.0
        assert failure["horizons"]["3"]["valid_trajectories"] == 0
        assert len(Path(failure["trajectory_evidence"]).read_text().splitlines()) == 3
        assert failure["portable_runtime_equivalence"]["status"] == "matched_nonfinite"
        with torch.no_grad():
            model.model.output_net.bias[2:] = 0.0
    # Missing gravity is a diagnostic run, never complete physical qualification.
    legacy = json.loads(json.dumps(metadata))
    legacy["source_physical_arrays"].pop("gravity")
    legacy["teacher_origin"].pop("gravity", None)
    old = tmp_path / "legacy-model"
    export_model(old, model, legacy)
    old_report = evaluate_reference_rollouts(
        {
            "source": source,
            "robot": robot,
            "model": str(old),
            "output": str(tmp_path / "legacy.json"),
            "num_envs": 2,
            "trajectories": 3,
            "horizons": [1],
            "initial_states": "default",
            "motion_mode": "passive",
            "cpu_threads": 2,
        },
        trust_external_code=True,
    )
    assert old_report["qualification_status"] == "pending_missing_recorded_physics"
    assert old_report["physical_verification"]["missing_recorded_fields"] == ["gravity"]
    # Independent official-version generation records the physical replica
    # count in its source split receipt rather than the old sampling field.
    split_metadata = json.loads(json.dumps(metadata))
    batch = split_metadata["sampling"].pop("batch_size")
    split_metadata["source_split_identities"] = {"train": {"num_envs": batch}}
    split_export = tmp_path / "split-source-model"
    export_model(split_export, model, split_metadata)
    split_spec = {
        "source": source,
        "robot": robot,
        "model": str(split_export),
        "output": str(tmp_path / "split-source.json"),
        "num_envs": 2,
        "trajectories": 3,
        "horizons": [1],
        "initial_states": "default",
        "motion_mode": "passive",
        "cpu_threads": 2,
    }
    split_report = evaluate_reference_rollouts(split_spec, trust_external_code=True)
    assert split_report["physical_verification"]["complete"]
    assert (
        split_report["physical_verification"]["sha256"]
        == reports["passive"]["physical_verification"]["sha256"]
    )
    split_metadata["sampling"]["batch_size"] = batch + 1
    conflicting_export = tmp_path / "conflicting-source-model"
    export_model(conflicting_export, model, split_metadata)
    with pytest.raises(ValueError, match="Conflicting source sampling"):
        evaluate_reference_rollouts(
            {
                **split_spec,
                "model": str(conflicting_export),
                "output": str(tmp_path / "conflicting-source.json"),
            },
            trust_external_code=True,
        )
    # A declared change to gravity must reject, not silently reuse old dynamics.
    changed = json.loads(json.dumps(metadata))
    changed["source_physical_arrays"]["gravity"] = [0.0, -1.0, 0.0]
    bad = tmp_path / "bad-model"
    export_model(bad, model, changed)
    with pytest.raises(ValueError, match="gravity"):
        evaluate_reference_rollouts(
            {
                "source": source,
                "robot": robot,
                "model": str(bad),
                "output": str(tmp_path / "bad.json"),
                "num_envs": 2,
                "trajectories": 3,
                "horizons": [1],
            },
            trust_external_code=True,
        )


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_real_historical_teacher_neural_runtime_and_source_contract(
    tmp_path: Path, robot: str
) -> None:
    source = os.environ.get("NERD_MAIN_REFERENCE_ROOT")
    dependencies = os.environ.get(
        "NERD_WARP18_ROOT" if robot == "cartpole" else "NERD_WARP151_ROOT"
    )
    if not source or not dependencies:
        pytest.skip("Set explicit trusted reference and legacy Warp dependency paths")
    code = "import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('reference_rollout_test',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m._check_original_runtime(Path(sys.argv[2]),sys.argv[3],sys.argv[4])"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve()),
            str(tmp_path),
            robot,
            source,
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (dependencies, str(_ROOT), str(_ROOT / "extensions/embodichain_models"))
            ),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )


def _check_supported_replay(
    tmp_path: Path, robot: str, source: str, data: Path
) -> None:
    import shutil
    import h5py
    import numpy as np
    import torch
    from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg, export_model

    metadata = json.loads((data / f"{robot}-passive/manifest.json").read_text())[
        "metadata"
    ]
    dims = (
        {"states_embedding": 4, "joint_f": 2}
        if robot == "cartpole"
        else {
            "states_embedding": 29,
            "contact_normals": 75,
            "contact_points_1": 75,
            "contact_depths": 25,
            "joint_f": 8,
            "gravity_dir": 3,
        }
    )
    torch.manual_seed(7)
    model = NeRDModel(
        NeRDModelCfg(
            input_dims=dims,
            output_dim=4 if robot == "cartpole" else 29,
            n_embd=24,
            n_layer=1,
            n_head=3,
            history_length=3,
            block_size=4,
        )
    )
    with torch.no_grad():
        model.model.output_net.weight.mul_(0.02)
        model.model.output_net.bias.zero_()
        if robot == "ant":
            model.model.output_net.bias[6] = 1.0
    export = tmp_path / "export"
    export_model(export, model, metadata)
    reports = {}
    specs = {}
    for mode in ["passive", "random_effort", "fixed_policy"]:
        spec = {
            "source": source,
            "robot": robot,
            "model": str(export),
            "output": str(tmp_path / (mode + ".json")),
            "device": "cpu",
            "num_envs": 2,
            "trajectories": 3,
            "horizons": [1, 4, 12],
            "seed": 1000000,
            "initial_states": "sample",
            "motion_mode": mode,
            "cpu_threads": 2,
            "teacher_dataset": str(data / (robot + "-" + mode)),
        }
        if mode == "fixed_policy":
            spec["policy"] = {"kind": "pd"}
        result = evaluate_reference_rollouts(spec, trust_external_code=True)
        reports[mode], specs[mode] = result, spec
        assert result["teacher_generation_environment"]["python"] == "3.8.20"
        assert result["teacher_generation_environment"]["torch"].startswith("2.2.2")
        assert (
            result["teacher_generation_environment"]["runtime_imports_embodichain"]
            is False
        )
        assert result["physical_verification"]["complete"]
        assert result["teacher_physical_verification"]["complete"]
        assert result["teacher_transport_max_abs_error"] <= 1e-5
        assert result["trajectories"] == 3 and set(result["horizons"]) == {
            "1",
            "4",
            "12",
        }
        assert result["portable_runtime_equivalence"]["status"] == "passed"
        if mode == "fixed_policy":
            assert (
                result["teacher_feedback_verification"]
                == "all_selected_actions_verified_against_teacher_states"
            )
            assert result["teacher_feedback_max_abs_error"] <= 1e-5
            with h5py.File(result["trace_evidence"], "r") as file:
                assert np.any(
                    file["0/joint_f"][:, 1:] != file["0/teacher_joint_f"][:, 1:]
                )
    # Different evaluation batching cannot select different teacher contents.
    again = evaluate_reference_rollouts(
        {**specs["passive"], "num_envs": 1, "output": str(tmp_path / "batch1.json")},
        trust_external_code=True,
    )
    assert (
        again["selected_teacher_sha256"]
        == reports["passive"]["selected_teacher_sha256"]
    )
    # A rehashed but inconsistent raw/canonical frame must not become a new oracle.
    changed = tmp_path / "changed-transport"
    shutil.copytree(data / (robot + "-passive"), changed)
    manifest_path = changed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard = changed / manifest["shards"][0]["file"]
    with h5py.File(shard, "a") as file:
        file["0/states/q"][1, 0] += 1.0
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="state transport differs"):
        evaluate_reference_rollouts(
            {
                **specs["passive"],
                "teacher_dataset": str(changed),
                "output": str(tmp_path / "bad-transport.json"),
            },
            trust_external_code=True,
        )
    # PD action verification also rejects consistent raw/canonical effort tampering.
    changed = tmp_path / "changed-feedback"
    shutil.copytree(data / (robot + "-fixed_policy"), changed)
    manifest_path = changed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard = changed / manifest["shards"][0]["file"]
    with h5py.File(shard, "a") as file:
        file["0/transitions/raw_joint_f"][0, 0] += 1.0
        file["0/transitions/joint_f"][0, 6 if robot == "ant" else 0] += 1.0
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="actions differ from declared feedback"):
        evaluate_reference_rollouts(
            {
                **specs["fixed_policy"],
                "teacher_dataset": str(changed),
                "output": str(tmp_path / "bad-feedback.json"),
            },
            trust_external_code=True,
        )
    missing = tmp_path / "missing-teacher-physics"
    shutil.copytree(data / (robot + "-passive"), missing)
    manifest_path = missing / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["source_physical_arrays"].pop("gravity")
    manifest["metadata"]["teacher_origin"].pop("gravity", None)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing measured physical fields"):
        evaluate_reference_rollouts(
            {
                **specs["passive"],
                "teacher_dataset": str(missing),
                "output": str(tmp_path / "missing-physics.json"),
            },
            trust_external_code=True,
        )


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_real_supported_environment_teacher_replay(tmp_path: Path, robot: str) -> None:
    source = os.environ.get("NERD_MAIN_REFERENCE_ROOT")
    dependencies = os.environ.get(
        "NERD_WARP18_ROOT" if robot == "cartpole" else "NERD_WARP151_ROOT"
    )
    data = os.environ.get("NERD_SUPPORTED_EVAL_DATA")
    if not source or not dependencies or not data:
        pytest.skip(
            "Set explicit trusted original source/dependencies and supported-environment evaluation corpus"
        )
    code = "import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('reference_replay_test',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m._check_supported_replay(Path(sys.argv[2]),sys.argv[3],sys.argv[4],Path(sys.argv[5]))"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve()),
            str(tmp_path),
            robot,
            source,
            data,
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (dependencies, str(_ROOT), str(_ROOT / "extensions/embodichain_models"))
            ),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        },
    )
