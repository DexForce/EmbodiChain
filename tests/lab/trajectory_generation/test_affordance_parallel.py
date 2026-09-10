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

"""CPU output-contract checks and opt-in real affordance-generation smoke test."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    TrajectoryPhase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.integrations.atomic_candidates import (
    AtomicCandidateRejection,
)
from embodichain.toolkits.graspkit import GraspCandidateBatch
from examples.sim.motion.trajectory_generation.affordance_parallel import (
    _save_result,
    run_affordance_parallel,
)


def _result(*, empty: bool) -> SimpleNamespace:
    count, horizon = (0, 0) if empty else (2, 3)
    qpos = torch.zeros(count, horizon, 2)
    intervals = torch.zeros(count, horizon, dtype=torch.float64)
    lengths = torch.tensor([] if empty else [2, 3], dtype=torch.long)
    if not empty:
        qpos[0, 1:, 0] = 0.1  # Two valid samples, then one held padding sample.
        qpos[1, :, 0] = torch.tensor([0.0, 0.2, 0.3])
        intervals[0, 1] = 0.05
        intervals[1, 1:] = 0.05
    batch = CandidateTrajectoryBatch(
        positions=qpos,
        dt=intervals,
        valid_length=lengths,
        identities=tuple(
            CandidateIdentity(
                "case",
                "initial",
                f"candidate-{row}",
                "geometry",
                "source",
                "revision",
                "template",
            )
            for row in range(count)
        ),
        joint_names=("arm", "hand"),
        phases=tuple(
            (TrajectoryPhase("transit", 0, int(length)),) for length in lengths
        ),
        source_row_indices=torch.zeros(count, dtype=torch.long),
    )
    return SimpleNamespace(
        trajectories=batch,
        branch_metadata=tuple(
            {"grasp_id": f"grasp-{row}", "variant_id": 0} for row in range(count)
        ),
        planning_checks=tuple(
            ValidationResult((ValidationCheck("planning", "passed"),))
            for _ in range(count)
        ),
        rejections=(
            AtomicCandidateRejection(
                0, "failed-grasp", 1, None, "pickup", "ik", "GRASP_IK_FAILED"
            ),
        ),
        summary={
            "status": "empty" if empty else "partial",
            "output_count": count,
            "rejection_counts": {"GRASP_IK_FAILED": 1},
            "physical_validation": False,
        },
    )


@pytest.mark.parametrize("empty", [False, True])
def test_saved_batch_preserves_compact_lengths_padding_and_failure_audit(
    tmp_path: Path, empty: bool
) -> None:
    grasps = GraspCandidateBatch(
        poses=torch.eye(4).repeat(1, 2, 1, 1),
        costs=torch.tensor([[0.0, 0.5]]),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        opening_widths=torch.tensor([[0.05, 0.051]]),
    )
    result = _result(empty=empty)
    report = _save_result(
        tmp_path, result, grasps, seed=13, requested=8, gripper_model_id="test-model"
    )
    saved_report = json.loads((tmp_path / "report.json").read_text())
    assert report == saved_report
    assert report["planning_only"] and report["expert_episodes_committed"] == 0
    assert (
        not report["physical_validation"] and not report["world_collision_validation"]
    )
    assert report["summary"]["rejection_counts"] == {"GRASP_IK_FAILED": 1}
    assert report["rejections"][0]["reason_code"] == "GRASP_IK_FAILED"
    with np.load(tmp_path / "trajectories.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["qpos"], result.trajectories.positions.numpy()
        )
        np.testing.assert_array_equal(
            saved["valid_length"], result.trajectories.valid_length.numpy()
        )
        np.testing.assert_array_equal(
            saved["valid_mask"], result.trajectories.valid_mask.numpy()
        )
        np.testing.assert_array_equal(saved["dt"], result.trajectories.dt.numpy())
        np.testing.assert_array_equal(
            saved["input_grasp_opening_widths"], grasps.opening_widths.numpy()
        )
        assert saved["candidate_ids"].dtype.kind == "U"
        assert list(saved["qpos"].shape) == report["output_shape"]
        if empty:
            assert saved["qpos"].shape == (0, 0, 2)
        else:
            assert saved["qpos"].shape == (2, 3, 2)
            assert not saved["valid_mask"][0, 2]
            assert saved["dt"][0, 2] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trajectories": 0},
        {"trajectories": 65},
        {"seed": -1},
        {"sample_count": 0},
        {"cuda_device": -1},
    ],
)
def test_invalid_cli_values_fail_before_scene_creation(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    output = tmp_path / "new-output"
    with pytest.raises(ValueError):
        run_affordance_parallel(output, **kwargs)
    assert not output.exists()


def test_occupied_output_directory_is_not_overwritten(tmp_path: Path) -> None:
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("user data")
    with pytest.raises(ValueError, match="new or empty"):
        run_affordance_parallel(tmp_path)
    assert sentinel.read_text() == "user data"


def test_native_queue_is_flushed_only_after_scene_owner_scope_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.lab.sim import SimulationManager
    from examples.sim.motion.trajectory_generation import affordance_parallel

    events: list[str] = []
    owners = []

    class Owner:
        pass

    def inner(*args: object, **kwargs: object) -> dict[str, object]:
        owner = Owner()
        owners.append(weakref.ref(owner))
        events.append("scene_opened")
        try:
            return {"planning_only": True}
        finally:
            events.append("destruction_queued")

    def flush() -> None:
        assert owners[0]() is None
        events.append("native_teardown")

    monkeypatch.setattr(affordance_parallel, "_run_affordance_parallel_scene", inner)
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", flush)
    result = run_affordance_parallel(tmp_path)
    assert result == {"planning_only": True}
    assert events == ["scene_opened", "destruction_queued", "native_teardown"]


def test_exception_tracebacks_release_scene_owners_before_native_teardown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.lab.sim import SimulationManager
    from examples.sim.motion.trajectory_generation import affordance_parallel

    owners = []
    events: list[str] = []

    class Owner:
        pass

    def inner(*args: object, **kwargs: object) -> dict[str, object]:
        owner = Owner()
        owners.append(weakref.ref(owner))
        try:
            raise RuntimeError("original backend failure")
        finally:
            events.append("destruction_queued")

    def flush() -> None:
        assert owners[0]() is None
        events.append("native_teardown")

    monkeypatch.setattr(affordance_parallel, "_run_affordance_parallel_scene", inner)
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", flush)
    with pytest.raises(RuntimeError, match="original backend failure"):
        run_affordance_parallel(tmp_path)
    assert events == ["destruction_queued", "native_teardown"]


def test_native_cleanup_error_does_not_mask_original_backend_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from embodichain.lab.sim import SimulationManager
    from examples.sim.motion.trajectory_generation import affordance_parallel

    def inner(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("original backend failure")

    def flush() -> None:
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(affordance_parallel, "_run_affordance_parallel_scene", inner)
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", flush)
    with pytest.raises(ValueError, match="original backend failure"):
        run_affordance_parallel(tmp_path)
    assert "secondary cleanup failure" in capsys.readouterr().err


def test_native_cleanup_error_after_success_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.lab.sim import SimulationManager
    from examples.sim.motion.trajectory_generation import affordance_parallel

    def flush() -> None:
        raise RuntimeError("native cleanup failed")

    monkeypatch.setattr(
        affordance_parallel,
        "_run_affordance_parallel_scene",
        lambda *args, **kwargs: {"planning_only": True},
    )
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", flush)
    with pytest.raises(RuntimeError, match="native cleanup failed"):
        run_affordance_parallel(tmp_path)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_sim
def test_real_antipodal_source_exports_a_filtered_parallel_batch(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "affordance-parallel"
    process = subprocess.run(
        [
            sys.executable,
            "examples/sim/motion/trajectory_generation/affordance_parallel.py",
            "--output",
            str(output),
            "--trajectories",
            "8",
            "--seed",
            "13",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=360,
    )
    assert process.returncode == 0, (process.stdout + process.stderr)[-12000:]
    report = json.loads((output / "report.json").read_text())
    assert report["valid_input_grasps"] >= 8
    assert report["physical_envs"] == 4 and report["canonical_sources"] == 1
    assert report["summary"]["proposed"] >= 8
    assert report["summary"]["rounds"] >= 2
    assert 0 < report["summary"]["output_count"] <= 8
    assert report["planning_only"] and not report["physical_validation"]
    with np.load(output / "trajectories.npz", allow_pickle=False) as saved:
        assert saved["qpos"].shape[0] == report["summary"]["output_count"]
        assert np.isfinite(saved["qpos"]).all()
        assert len(set(saved["candidate_ids"])) == len(saved["candidate_ids"])
        assert np.all(saved["source_row_indices"] == 0)
        assert saved["input_grasp_poses"].shape[0] == 1
