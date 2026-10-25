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

"""Ensure SSIK benchmark acceptance never trusts a solver's success mask alone."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark.robotics.kinematic_solver.ssik_benchmark import (
    SSIKBenchmarkOptions,
    DEFAULT_MODEL_NAMES,
    MODEL_NAMES,
    _gate_notes,
    _measure_streaming_control,
    _memory,
    _evaluate_trajectories,
    _make_solver,
    _parse_reference_artifacts,
    benchmark_ssik_solver,
    evaluate_quality,
)


def _planar_fk(qpos: torch.Tensor) -> torch.Tensor:
    result = torch.eye(4, dtype=torch.float64).repeat(len(qpos), 1, 1)
    result[:, 0, 3] = qpos[:, 0]
    cosine, sine = qpos[:, 1].cos(), qpos[:, 1].sin()
    result[:, 0, 0], result[:, 1, 1] = cosine, cosine
    result[:, 0, 1], result[:, 1, 0] = -sine, sine
    return result


def test_independent_acceptance_rejects_false_successes() -> None:
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
    # Only the identity posture meets the requested pose and limits. The last
    # row meets its own target but exceeds the joint limit.
    qpos = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [float("nan"), 0.0], [0.0, 2.0]]
    )
    target = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    target[-1] = _planar_fk(torch.tensor([[0.0, 2.0]]))[0]
    result = evaluate_quality(
        target, qpos, torch.ones(5, dtype=torch.bool), _planar_fk, limits
    )
    assert result["success_rate"] == 0.2
    assert result["false_successes"] == 4
    assert result["max_position_m"] == 0.0
    assert result["max_rotation_rad"] == 0.0


def test_rotation_acceptance_resolves_angles_below_float32_trace_precision() -> None:
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
    target = torch.eye(4, dtype=torch.float64)[None]
    qpos = torch.tensor([[0.0, 2e-5]], dtype=torch.float32)
    result = evaluate_quality(
        target, qpos, torch.ones(1, dtype=torch.bool), _planar_fk, limits
    )
    assert result["false_successes"] == 1
    assert result["success_rate"] == 0.0


def test_smoke_and_missing_measurements_cannot_certify_gates() -> None:
    for smoke in (False, True):
        notes = _gate_notes(
            [],
            [],
            SSIKBenchmarkOptions(smoke=smoke, models=("aloha_left",), devices=("cpu",)),
        )
        assert notes
        assert all("UNVALIDATED" in note for note in notes)
        assert not any("PASS" in note for note in notes)


def test_fixed_pose_samples_cannot_certify_continuous_control() -> None:
    fixed = {
        "model": "aloha_left",
        "impl": "ssik_warp_cpu",
        "scope": "resident",
        "sample_size": 1,
        "p99_ms": 0.1,
        "repeats": 10000,
    }
    options = SSIKBenchmarkOptions(models=("aloha_left",), devices=("cpu",))
    notes = _gate_notes([fixed], [], options)
    control = [note for note in notes if "continuous-control latency gate" in note]
    assert len(control) == 1
    assert "UNVALIDATED" in control[0]
    streamed = {**fixed, "scope": "streaming_control", "failure_count": 0}
    notes = _gate_notes([fixed, streamed], [], options)
    control = [note for note in notes if "continuous-control latency gate" in note]
    assert "PASS" in control[0]
    notes = _gate_notes([{**streamed, "false_successes": 1}], [], options)
    control = [note for note in notes if "continuous-control latency gate" in note]
    assert "FAIL" in control[0]


@pytest.mark.parametrize("arm_count", [1, 2])
def test_streaming_uses_changing_goals_previous_results_and_path_resets(
    arm_count: int,
) -> None:
    class Solver:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(max_joint_step=None)
            self.seen: list[tuple[float, float]] = []

        def get_ik(
            self, goal: torch.Tensor, seed: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert self.cfg.max_joint_step == 0.05
            self.seen.append((float(goal[0, 0, 3]), float(seed[0, 0])))
            return torch.ones(1, dtype=torch.bool), seed + 0.01

    solvers = tuple(Solver() for _ in range(arm_count))
    target = torch.eye(4).repeat(2, 2, 1, 1)
    target[:, :, 0, 3] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    initial = tuple(
        torch.tensor([[10.0], [20.0]]) + arm * 100 for arm in range(arm_count)
    )
    measured = _measure_streaming_control(
        solvers,
        (target,) * arm_count,
        initial,
        torch.device("cpu"),
        repeats=4,
        warmup=2,
    )
    for arm, solver in enumerate(solvers):
        recorded = solver.seen[2:]
        assert [goal for goal, _ in recorded] == [0.0, 1.0, 2.0, 3.0]
        assert [seed for _, seed in recorded] == pytest.approx(
            [10.0 + arm * 100, 10.01 + arm * 100, 20.0 + arm * 100, 20.01 + arm * 100]
        )
        assert solver.cfg.max_joint_step is None
        assert measured.qpos[arm].flatten().tolist() == pytest.approx(
            [10.01 + arm * 100, 10.02 + arm * 100, 20.01 + arm * 100, 20.02 + arm * 100]
        )
    assert measured.latency["repeats"] == 4
    assert measured.latency["failure_count"] == 0


def test_reference_artifacts_are_explicit_and_model_specific(tmp_path: Path) -> None:
    assert SSIKBenchmarkOptions().models == DEFAULT_MODEL_NAMES
    assert "arx5" not in DEFAULT_MODEL_NAMES
    assert "flexiv_rizon4s" not in DEFAULT_MODEL_NAMES
    assert {"arx5", "flexiv_rizon4s"} <= set(MODEL_NAMES)
    source = tmp_path / "a=b.py"
    assert _parse_reference_artifacts((f"arx5={source}",)) == {"arx5": source}
    for entries in (
        ("arx5",),
        ("arx5=",),
        (f"unknown={source}",),
        (f"arx5={source}", f"arx5={source}"),
    ):
        with pytest.raises(ValueError):
            _parse_reference_artifacts(entries)


@pytest.mark.parametrize("specified_missing_artifact", [False, True])
def test_unsupported_geometry_and_missing_artifacts_are_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    specified_missing_artifact: bool,
) -> None:
    from embodichain.compute.kinematics import _ssik_model
    from scripts.benchmark.robotics.kinematic_solver import ssik_benchmark as module

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(init=lambda: None))
    monkeypatch.setattr(module, "_versions", lambda: {})
    monkeypatch.setattr(module, "_build_workloads", lambda *args: {})
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda name: SimpleNamespace(
            chain=object(),
            cfg=SimpleNamespace(fpath="actual.urdf"),
            manifest={"root": "base", "end": "tcp"},
        ),
    )
    monkeypatch.setattr(
        _ssik_model,
        "_extract_model",
        lambda chain, **kwargs: SimpleNamespace(
            family="unsupported", fingerprint="actual"
        ),
    )

    def make_solver(*args: object) -> None:
        assert not specified_missing_artifact
        raise ValueError(
            "This reference geometry requires an explicitly built reference_artifact"
        )

    monkeypatch.setattr(module, "_make_solver", make_solver)
    options = SSIKBenchmarkOptions(
        smoke=True,
        models=("arx5",),
        devices=("cpu",),
        skip_baseline=True,
        artifact_dir=tmp_path,
        reference_artifacts=(
            (f"arx5={tmp_path / 'missing.py'}",) if specified_missing_artifact else ()
        ),
    )
    previous_logger_level = logging.getLogger("ssik").level
    perf, metrics, notes = benchmark_ssik_solver(options)
    assert logging.getLogger("ssik").level == previous_logger_level
    assert perf == metrics == []
    assert any("actual-chain geometry is unsupported" in note for note in notes)
    assert any("--ssik-reference-artifacts arx5=" in note for note in notes)
    assert not any("PASS" in note for note in notes)
    assert (tmp_path / "measurements.json").is_file()


@pytest.mark.parametrize("backend", ["warp", "reference", "cuda_rr"])
def test_explicit_algorithm_options_reach_only_the_geometric_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    received: dict[str, object] = {}

    def cfg(**kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(init_solver=lambda **kwargs: kwargs)

    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.sim.motion.solvers",
        SimpleNamespace(PytorchSolverCfg=cfg, SSIKSolverCfg=cfg),
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(fpath="actual.urdf"),
        joint_names=["joint"],
        manifest={"root": "root", "end": "end"},
        tcp=torch.eye(4),
        limits=torch.tensor([[-1.0], [1.0]]),
    )
    options = SSIKBenchmarkOptions(
        geometry_mode="near_parallel",
        axis_alignment_tolerance=9e-6,
        srs_search="seed_local",
    )
    _make_solver(model, backend, torch.device("cpu"), options=options)
    assert received["backend"] == backend
    if backend == "warp":
        assert received["geometry_mode"] == "near_parallel"
        assert received["axis_alignment_tolerance"] == 9e-6
        assert received["srs_search"] == "seed_local"
    else:
        assert "geometry_mode" not in received and "srs_search" not in received


@pytest.mark.parametrize(
    "models,devices", [(("arx5",), ("cuda",)), (("flexiv_rizon4s",), ("cpu",))]
)
def test_explicit_cuda_rr_cannot_silently_skip_an_unsupported_workload(
    models: tuple[str, ...], devices: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="requires --ssik-models flexiv_rizon4s"):
        benchmark_ssik_solver(
            SSIKBenchmarkOptions(models=models, devices=devices, cuda_rr=True)
        )


def test_seed_local_continuity_uses_the_public_path_hook() -> None:
    joints = torch.tensor([[0.01, 0.0], [0.02, 0.0], [0.03, 0.0]])
    targets = _planar_fk(joints)
    called = []

    def path_hook(target: torch.Tensor, seed: torch.Tensor):
        assert target.shape == (1, 3, 4, 4) and seed.shape == (1, 2)
        called.append(True)
        return torch.ones((1, 3), dtype=torch.bool), joints[None]

    def no_fixed_seed_candidates(*args: object):
        raise AssertionError("seed-local paths must use the target-aware hook")

    solver = SimpleNamespace(
        cfg=SimpleNamespace(
            max_joint_step=None, backend="warp", srs_search="seed_local"
        ),
        family="srs",
        _get_continuous_ik_path=path_hook,
        _get_continuous_ik_candidates=no_fixed_seed_candidates,
    )
    model = SimpleNamespace(
        fk=_planar_fk, limits=torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    )
    result = _evaluate_trajectories(
        solver, model, targets[None], torch.zeros((1, 2)), torch.device("cpu")
    )
    assert called == [True] and result["success_rate"] == 1.0
    assert result["max_accepted_joint_step_rad"] <= 0.05
    assert solver.cfg.max_joint_step is None


def test_memory_attribution_includes_borrowed_numerical_workspace_once() -> None:
    normal = torch.empty(64, dtype=torch.float64)
    flags = torch.empty(32, dtype=torch.int32)
    solver = SimpleNamespace(
        cfg=SimpleNamespace(class_type="SSIKSolver", backend="warp"),
        dof=6,
        _ik_buffers=SimpleNamespace(arrays={"flags": flags}),
        _near=SimpleNamespace(_workspace={"normal": normal, "alias": normal}),
    )
    result = _memory(
        lambda: (torch.ones(1, dtype=torch.bool), torch.zeros(1, 6)),
        torch.device("cpu"),
        solver,
    )
    # One 64-double numerical workspace plus32 flags, regardless of aliasing.
    assert result["solver_buffer_mb"] == (512 + 128) / 2**20
    assert result["tensor_scratch_gate"] == "PASS"
