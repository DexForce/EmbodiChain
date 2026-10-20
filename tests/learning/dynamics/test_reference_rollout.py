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
from embodichain.learning.dynamics.reference_rollout import evaluate_reference_rollouts

_ROOT = Path(__file__).resolve().parents[3]


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
    from embodichain.learning.dynamics.reference import generate_official_dataset
    from embodichain.learning.models.nerd import NeRDModel, NeRDModelCfg, export_model
    from embodichain.data_pipeline.dynamics.shards import load_manifest

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
            "PYTHONPATH": dependencies + os.pathsep + str(_ROOT),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
