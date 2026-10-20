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

"""Actual bounded generation and batched evaluation, isolated from renderer fixtures."""

from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest
import torch
from embodichain.lab.sim.physics.solvers.collection import (
    DynamicsGenerationCfg,
    generate_dataset,
    evaluate_rollouts,
    _sample,
    _task_values,
    _FixedPolicy,
)
from embodichain.data_pipeline.dynamics.shards import load_manifest, trajectory_split

_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_envs", True),
        ("horizon", 2.5),
        ("trajectories", 0),
        ("resume", "false"),
        ("seed", False),
        ("num_substeps", -1),
        ("max_attempts", 1),
        ("effort_scale", float("nan")),
    ],
)
def test_generation_config_rejects_invalid_values_before_runtime(key, value) -> None:
    with pytest.raises((ValueError, TypeError)):
        DynamicsGenerationCfg(**{key: value})


def test_sampling_is_candidate_seed_local_and_passive_zero() -> None:
    first = _sample("ant", 9, 5, "random_effort", "sample", 0.1)
    _sample("cartpole", 73, 2, "random_effort", "sample", 1.0)
    second = _sample("ant", 9, 5, "random_effort", "sample", 0.1)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not first[2][:, :6].any()
    assert not _sample("cartpole", 3, 5, "passive", "default", 1.0)[2].any()


def test_fixed_policy_adapter_uses_own_state_and_records_hash() -> None:
    policy = _FixedPolicy("cartpole", {"kind": "pd"}, torch.device("cpu"))
    q = torch.tensor([[0.0, 0.0], [0.1, 0.2]])
    qd = torch.zeros_like(q)
    force = policy(q, qd)
    assert force[0, 0] == 0 and force[1, 0] != 0
    assert len(policy.identity["sha256"]) == 64
    assert not force[:, 1].any()


def _helper():
    path = Path(__file__).with_name("test_nerd_solver.py")
    spec = importlib.util.spec_from_file_location("_collection_export_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_pipeline(tmp_path: Path, robot: str) -> None:
    from embodichain.lab.sim.physics.solvers.collection import DynamicsSimulation

    cfg = {
        "robot": robot,
        "output_dir": str(tmp_path / "data"),
        "num_envs": 2,
        "trajectories": 3,
        "horizon": 3,
        "seed": 7,
        "split_seed": 11,
        "num_substeps": 2,
        "trajectories_per_shard": 2,
        "device": "cpu",
        "initial_states": "default",
        "motion_mode": "random_effort",
        "effort_scale": 0.01,
    }
    result = generate_dataset(cfg)
    assert result["trajectories"] == 3 and result["transitions"] == 9
    manifest = load_manifest(tmp_path / "data", verify=True)
    original = [row for shard in manifest["shards"] for row in shard["trajectories"]]
    assert [row["id"] for row in original] == ["7:0", "7:1", "7:2"]
    assert [row["split"] for row in original] == [
        trajectory_split(row["id"], 11) for row in original
    ]
    resumed = generate_dataset({**cfg, "trajectories": 5, "resume": True})
    assert resumed["trajectories"] == 5
    manifest = load_manifest(tmp_path / "data", verify=True)
    all_rows = [row for shard in manifest["shards"] for row in shard["trajectories"]]
    assert [row["id"] for row in all_rows] == [f"7:{i}" for i in range(5)]
    assert all_rows[:3] == original
    first_shard = manifest["shards"][0]
    with h5py.File(tmp_path / "data" / first_shard["file"], "r") as stream:
        row = stream[first_shard["trajectories"][0]["group"]]
        q = torch.tensor(row["states/q"][()])
        qd = torch.tensor(row["states/qd"][()])
        force = torch.tensor(row["transitions/joint_f"][()])
        saved_contacts = {
            key: torch.tensor(value[0:1])
            for key, value in row["transitions"].items()
            if key.startswith("contact_")
        }
        assert q.shape[0] == 4 and force.shape[0] == 3
        if robot == "cartpole":
            assert row["transitions/contact_depths"].shape == (3, 0)
    with DynamicsSimulation(robot, device="cpu", num_envs=1, num_substeps=2) as host:
        host.reset(q[:1], qd[:1])
        state, contacts = host.snapshot()
        for key, value in saved_contacts.items():
            torch.testing.assert_close(contacts[key], value)
        host.step(force[:1])
        next_state, _ = host.snapshot()
        torch.testing.assert_close(next_state["q"], q[1:2], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(next_state["qd"], qd[1:2], atol=1e-5, rtol=1e-4)
        checkpoint = _helper()._small_export(
            host.sim.spawn_result, tmp_path / "model", robot
        )
    common = {
        "robot": robot,
        "model": str(checkpoint),
        "device": "cpu",
        "num_envs": 2,
        "trajectories": 3,
        "horizons": [1, 3],
        "num_substeps": 2,
        "initial_states": "default",
    }
    report = evaluate_rollouts(
        {
            **common,
            "teacher_dataset": str(tmp_path / "data"),
            "split": "all",
            "motion_mode": "random_effort",
            "output": str(tmp_path / "replay.json"),
        }
    )
    assert report["num_envs"] == 2 and report["trajectories"] == 3
    assert len(report["error_vs_time"]) == 3
    assert report["teacher_origin"]["kind"] == "current_newton"
    assert report["runtime_identity"]["device"] == "cpu"
    assert report["runtime_identity"]["versions"]["torch"] == torch.__version__
    assert len(report["runtime_identity"]["source_sha256"]["nerd.py"]) == 64
    assert len(report["selected_teacher_sha256"]) == 64
    # Same immutable teacher trajectories have the same selected-content digest
    # even when replay batching and output destination change.
    rebatched = evaluate_rollouts(
        {
            **common,
            "num_envs": 1,
            "teacher_dataset": str(tmp_path / "data"),
            "split": "all",
            "motion_mode": "random_effort",
            "output": str(tmp_path / "rebatched.json"),
        }
    )
    assert rebatched["selected_teacher_sha256"] == report["selected_teacher_sha256"]
    with pytest.raises(ValueError, match="nonzero.*passive"):
        evaluate_rollouts(
            {
                **common,
                "teacher_dataset": str(tmp_path / "data"),
                "split": "all",
                "motion_mode": "passive",
                "output": str(tmp_path / "mislabeled.json"),
            }
        )
    rows = [
        json.loads(line)
        for line in Path(report["trajectory_evidence"]).read_text().splitlines()
    ]
    assert [row["trajectory_id"] for row in rows] == ["7:0", "7:1", "7:2"]
    with h5py.File(report["trace_evidence"], "r") as stream:
        assert stream["0/q"].shape[0] == 2 and stream["1/q"].shape[0] == 1
    fixed = evaluate_rollouts(
        {
            **common,
            "motion_mode": "fixed_policy",
            "policy": {"kind": "pd"},
            "output": str(tmp_path / "closed.json"),
        }
    )
    assert (
        fixed["policy_return"] is not None
        and fixed["teacher_policy_return"] is not None
    )
    assert len(fixed["policy_sha256"]) == 64
    teacher_manifest = load_manifest(fixed["teacher_dataset"], verify=True)
    teacher_file = (
        Path(fixed["teacher_dataset"]) / teacher_manifest["shards"][0]["file"]
    )
    with (
        h5py.File(teacher_file, "r") as teacher,
        h5py.File(fixed["trace_evidence"], "r") as neural,
    ):
        # The same feedback policy recomputes actions from the deviating states.
        assert not np.allclose(
            teacher["0/transitions/joint_f"][1:], neural["0/joint_f"][0, 1:], atol=1e-8
        )
    assert (
        fixed["horizons"]["3"]["valid_trajectories"]
        + fixed["horizons"]["3"]["numerically_invalid_trajectories"]
        == 3
    )


def _check_actor_and_failures(tmp_path: Path) -> None:
    from embodichain.lab.sim.physics.solvers.collection import DynamicsSimulation
    from embodichain.learning.models import build_mlp_from_cfg
    from embodichain.learning.rl.policies import build_policy
    from embodichain.learning.models.nerd import load_model, export_model

    with DynamicsSimulation(
        "cartpole", device="cpu", num_envs=2, num_substeps=2
    ) as host:
        checkpoint = _helper()._small_export(
            host.sim.spawn_result, tmp_path / "model", "cartpole"
        )
    block = {
        "name": "actor_only",
        "actor": {
            "type": "mlp",
            "network_cfg": {
                "hidden_sizes": [8],
                "activation": "tanh",
                "last_activation": "tanh",
            },
        },
    }
    torch.manual_seed(11)
    actor = build_mlp_from_cfg(block["actor"], 4, 1)
    policy = build_policy(block, 4, 1, torch.device("cpu"), actor=actor)
    config_path = tmp_path / "policy.json"
    config_path.write_text(json.dumps({"policy": block}))
    weights = tmp_path / "policy.pt"
    torch.save({"schema_version": 1, "policy": policy.state_dict()}, weights)
    common = {
        "robot": "cartpole",
        "model": str(checkpoint),
        "device": "cpu",
        "num_envs": 2,
        "trajectories": 3,
        "horizons": [1, 3],
        "num_substeps": 2,
        "initial_states": "default",
    }
    result = evaluate_rollouts(
        {
            **common,
            "motion_mode": "fixed_policy",
            "policy": {
                "kind": "actor_only",
                "config": str(config_path),
                "checkpoint": str(weights),
            },
            "output": str(tmp_path / "actor.json"),
        }
    )
    assert result["policy"]["checkpoint_sha256"] and result["policy_return"] is not None
    model, manifest = load_model(checkpoint)
    with torch.no_grad():
        # Finite FP32 weights whose velocity predictions overflow during rollout.
        model.model.output_net.bias[2:] = torch.finfo(torch.float32).max
    divergent = tmp_path / "divergent"
    export_model(divergent, model, manifest)
    report = evaluate_rollouts(
        {
            **common,
            "model": str(divergent),
            "motion_mode": "passive",
            "output": str(tmp_path / "nonfinite.json"),
        }
    )
    assert report["nonfinite_rate"] == 1.0
    assert report["failure_rate"] == 1.0
    assert report["horizons"]["3"]["valid_trajectories"] == 0
    assert report["horizons"]["3"]["q_mae"] is None
    assert len(Path(report["trajectory_evidence"]).read_text().splitlines()) == 3


def _check_rejection_limit(tmp_path: Path) -> None:
    import embodichain.lab.sim.physics.solvers.collection as collection

    original = collection._sample

    def fast_velocity(*args):
        q, qd, effort = original(*args)
        qd[:] = 2e5
        return q, qd, effort

    collection._sample = fast_velocity
    try:
        with pytest.raises(RuntimeError, match="attempt limit"):
            generate_dataset(
                {
                    "robot": "cartpole",
                    "output_dir": str(tmp_path / "rejected"),
                    "num_envs": 2,
                    "trajectories": 1,
                    "horizon": 1,
                    "max_attempts": 2,
                    "device": "cpu",
                    "initial_states": "default",
                    "motion_mode": "passive",
                }
            )
        progress = json.loads(
            (tmp_path / "rejected/generation-progress.json").read_text()
        )
        assert progress["rejected"] == 2 and len(progress["rejections"]) == 2
        assert not load_manifest(tmp_path / "rejected")["shards"]
    finally:
        collection._sample = original


def _subprocess(tmp_path: Path, operation: str, robot: str = "cartpole") -> None:
    code = "import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('collection_test',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);getattr(m,sys.argv[3])(Path(sys.argv[2]),sys.argv[4]) if sys.argv[3]=='_check_pipeline' else getattr(m,sys.argv[3])(Path(sys.argv[2]))"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve()),
            str(tmp_path),
            operation,
            robot,
        ],
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(_ROOT),
            "EMBODICHAIN_SIM_EXIT_PROCESS": "0",
        },
    )


@pytest.mark.subprocess_sim
@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_actual_generation_resume_alignment_and_batched_closed_loop(
    tmp_path: Path, robot: str
) -> None:
    _subprocess(tmp_path, "_check_pipeline", robot)


@pytest.mark.subprocess_sim
def test_actual_actor_artifact_and_nonfinite_evidence(tmp_path: Path) -> None:
    _subprocess(tmp_path, "_check_actor_and_failures")


@pytest.mark.subprocess_sim
def test_actual_rejection_attempt_limit(tmp_path: Path) -> None:
    _subprocess(tmp_path, "_check_rejection_limit")


def test_error_metric_wraps_only_continuous_coordinates() -> None:
    from embodichain.lab.sim.physics.solvers.collection import _error_values

    q = torch.zeros(1, 15)
    q[:, 6] = 1.0
    target = q.clone()
    q[:, 7] = 2 * torch.pi
    error = _error_values("ant", q, torch.zeros(1, 14), target, torch.zeros(1, 14))
    torch.testing.assert_close(
        error[:, 0], torch.tensor([2 * torch.pi / 8], dtype=torch.double)
    )
    cart = torch.tensor([[0.0, 2 * torch.pi]])
    error = _error_values(
        "cartpole", cart, torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)
    )
    assert error[0, 0] < 1e-6


def test_teacher_content_hash_uses_values_and_logical_identity_not_paths() -> None:
    from embodichain.lab.sim.physics.solvers.collection import (
        _teacher_row_sha256,
        _teacher_origin,
    )

    record = {
        "id": "sample:7",
        "file": "shard-a.h5",
        "group": "0",
        "metadata": {"sampling_seed": 7},
    }
    values = [
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
    ]
    expected = _teacher_row_sha256(record, values)
    assert (
        _teacher_row_sha256(
            {**record, "file": "/different/location.h5", "group": "17"}, values
        )
        == expected
    )
    changed = [array.copy() for array in values]
    changed[0][2, 1] = 1.0
    assert _teacher_row_sha256(record, changed) != expected
    assert _teacher_row_sha256({**record, "id": "sample:8"}, values) != expected
    official = {
        "kind": "official_warp",
        "warp_version": "1.5.1",
        "source_commit": "pinned",
    }
    assert _teacher_origin({"teacher_origin": official}) == official
    with pytest.raises(ValueError, match="teacher_origin"):
        _teacher_origin({"robot": "ant"})
