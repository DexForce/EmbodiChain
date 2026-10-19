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
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from examples.data_pipeline.ods_pick_place_showcase import (
    _geometry_statistics,
    policy_observations,
)


def test_direct_entry_resolves_its_worktree_without_pythonpath(tmp_path) -> None:
    """The documented file entry reaches its own runner before any simulation."""
    (tmp_path / "existing.txt").write_text("Do not overwrite")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/data_pipeline/ods_pick_place_showcase.py"),
            "--smoke",
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": "", "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Use a fresh output directory" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_failed_producer_is_not_retried_as_an_empty_sampling_window() -> None:
    from types import SimpleNamespace
    from examples.data_pipeline._ods_pick_place_training import _sample
    from embodichain.data_pipeline.engine.data import OnlineDataEngineState

    error = RuntimeError("No unlocked valid chunk: terminal producer failure")

    def fail(*args, **kwargs):
        raise error

    source = SimpleNamespace(
        state=OnlineDataEngineState.FAILED,
        sample_batch_with_context=fail,
    )
    with pytest.raises(RuntimeError) as caught:
        _sample(source, 2, SimpleNamespace(elapsed_seconds=lambda: 0), 32)
    assert caught.value is error


def test_geometry_features_are_measured_and_quaternion_sign_invariant() -> None:
    obs = {
        "robot": {"qpos": torch.zeros(2, 9), "qvel": torch.zeros(2, 9)},
        "tcp_pose": torch.eye(4).repeat(2, 1, 1),
        "cube_pose": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "goal_position": torch.tensor([[4.0, 6.0, 8.0]]).repeat(2, 1),
        "cube_velocity": torch.zeros(2, 6),
    }
    values = policy_observations(obs)
    assert values.shape == (2, 51)
    torch.testing.assert_close(values[:, 39:42], obs["cube_pose"][:, :3])
    torch.testing.assert_close(
        values[:, 42:45], torch.tensor([[3.0, 4.0, 5.0]]).repeat(2, 1)
    )
    obs["cube_pose"][:, 3:] *= -1
    torch.testing.assert_close(values, policy_observations(obs))
    mean, std = _geometry_statistics(values)
    assert torch.isfinite(std).all() and (std > 0).all()
    torch.testing.assert_close(std[21:27], torch.ones(6))
    obs["cube_pose"].zero_()
    assert torch.isfinite(policy_observations(obs)).all()
    obs["cube_pose"][:, 3] = float("nan")
    assert not torch.isfinite(policy_observations(obs)).all()


def test_budget_deadline_and_baseline_checkpoint_selection() -> None:
    from examples.data_pipeline._ods_pick_place_training import (
        _BudgetExpired,
        _selection,
        _timeout,
    )
    from types import SimpleNamespace

    initial = {"success_rate": 0.9, "lift_rate": 1.0, "goal_error": 0.01}
    worse = {"success_rate": 0.1, "lift_rate": 0.2, "goal_error": 0.3}
    assert _selection(initial) > _selection(worse)
    with pytest.raises(_BudgetExpired):
        _timeout(
            SimpleNamespace(wall_time_budget=5.0),
            SimpleNamespace(elapsed_seconds=lambda: 6.0),
        )
    assert (
        _timeout(
            SimpleNamespace(wall_time_budget=5.0),
            SimpleNamespace(elapsed_seconds=lambda: 3.0),
        )
        == 2.0
    )


def test_qualification_reuse_checks_physics_sources_and_independent_scenes(
    tmp_path,
) -> None:
    import json
    from examples.data_pipeline._ods_pick_place_training import _reuse_qualification

    path = tmp_path / "expert_qualification.json"
    records = [{"seed": i, "success": True} for i in range(40)]
    path.write_text(json.dumps({"records": records, "success_rate": 1.0}))
    config = {
        "device": "cuda:1",
        "gpu_id": 1,
        "seed": 42,
        "physics": "default",
        "env": {"actions": "dls"},
    }
    (tmp_path / "task_config.json").write_text(json.dumps(config))
    hashes = {
        "embodichain/lab/gym/envs/managers/actions.py": "a",
        "extensions/embodichain_tasks/embodichain_tasks/manipulation/pick_place.py": "b",
    }
    (tmp_path / "environment.json").write_text(json.dumps({"source_sha256": hashes}))
    assert (
        _reuse_qualification(path, {**config, "device": "cuda:3", "gpu_id": 3}, hashes)[
            "success_rate"
        ]
        == 1.0
    )
    with pytest.raises(ValueError, match="configuration differs"):
        _reuse_qualification(path, {**config, "physics": "newton"}, hashes)
    with pytest.raises(ValueError, match="source differs"):
        _reuse_qualification(
            path,
            config,
            {**hashes, "embodichain/lab/gym/envs/managers/actions.py": "changed"},
        )
    records[1]["seed"] = records[0]["seed"]
    path.write_text(json.dumps({"records": records, "success_rate": 1.0}))
    with pytest.raises(ValueError, match="independent scenes"):
        _reuse_qualification(path, config, hashes)


def test_offline_reset_diagnostic_uses_original_causal_frame_zero() -> None:
    from examples.data_pipeline._ods_pick_place_training import _offline_reset_positions

    observations = torch.randn(4, 3, 51)
    names = [f"feature_{i}" for i in range(51)]
    names[27:30] = ["cube_0", "cube_1", "cube_2"]
    corpus = SimpleNamespace(
        observations=observations,
        valid=torch.ones(4, 3, dtype=torch.bool),
        frame_ids=torch.zeros(4, 3, 3, dtype=torch.int64),
        manifest={
            "schema": {
                "observation": {"version": "pick_place.geometry@2", "names": names}
            }
        },
    )
    assert _offline_reset_positions(corpus, 4) == observations[:, 0, 27:30].tolist()
    with pytest.raises(ValueError, match="exactly the requested episodes"):
        _offline_reset_positions(corpus, 1)
    corpus.valid[2, 0] = False
    with pytest.raises(ValueError, match="frame zero"):
        _offline_reset_positions(corpus, 4)
    corpus.valid[2, 0] = True
    corpus.frame_ids[1, 0, 2] = 1
    with pytest.raises(ValueError, match="frame zero"):
        _offline_reset_positions(corpus, 4)
