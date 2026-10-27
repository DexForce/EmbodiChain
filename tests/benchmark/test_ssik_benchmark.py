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
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np
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
    _implementation_name,
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


@pytest.mark.parametrize("rr_search", ["fixed", "seed_local"])
def test_real_robot_gate_uses_selected_rr_backend_and_quality(
    rr_search: str,
) -> None:
    options = SSIKBenchmarkOptions(
        models=("flexiv_rizon4s",),
        devices=("cuda",),
        cuda_rr=True,
        rr_search=rr_search,
        robot_e2e=True,
    )
    suffix = "_seed_local" if rr_search == "seed_local" else ""
    record = {
        "model": "flexiv_rizon4s",
        "impl": f"ssik_cuda_rr{suffix}_cuda",
        "scope": "robot_streaming_control",
        "sample_size": 1,
        "repeats": 10000,
        "p99_ms": 1.0,
        "validated_success_rate": 0.95,
    }
    standalone = {**record, "scope": "streaming_control"}

    def gate(rows: list[dict]) -> str:
        return next(
            note
            for note in _gate_notes(rows, [], options)
            if "Robot end-to-end control gate" in note
        )

    assert "UNVALIDATED" in gate([])
    assert "UNVALIDATED" in gate([{**record, "repeats": 3}])
    assert "UNVALIDATED" in gate([record])
    assert "PASS" in gate([record, standalone])
    for key in ("false_successes", "step_violations", "failed_hold_violations"):
        assert "FAIL" in gate([{**record, key: 1}, standalone])
    assert "FAIL" in gate([{**record, "p99_ms": 5.01}, standalone])
    assert "FAIL" in gate([{**record, "validated_success_rate": 0.0}, standalone])


def test_paired_robot_gate_requires_coverage_and_uses_eight_ms_bound() -> None:
    options = SSIKBenchmarkOptions(
        models=("aloha_left", "aloha_right"),
        devices=("cpu",),
        robot_e2e=True,
    )
    record = {
        "model": "aloha_pair",
        "impl": "ssik_warp_cpu",
        "scope": "robot_paired_streaming_control",
        "repeats": 10000,
        "p99_ms": 7.9,
        "validated_success_rate": 1.0,
    }
    standalone = {**record, "scope": "paired_streaming_control"}

    def gate(rows: list[dict]) -> str:
        return next(
            note
            for note in _gate_notes(rows, [], options)
            if "Paired Robot end-to-end control gate" in note
        )

    assert "UNVALIDATED" in gate([record])
    assert "PASS" in gate([record, standalone])
    for change in (
        {"p99_ms": 8.01},
        {"validated_success_rate": 0.0},
        {"step_violations": 1},
        {"failed_hold_violations": 1},
        {"false_successes": 1},
    ):
        assert "FAIL" in gate([{**record, **change}, standalone])


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
    assert {"arx5", "flexiv_rizon4s"} <= set(DEFAULT_MODEL_NAMES)
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
        skip_reference=False,
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


@pytest.mark.parametrize(
    ("backend", "family", "srs_search", "rr_search"),
    [
        ("warp", "srs", "seed_local", "fixed"),
        ("warp", "near_parallel", "uniform", "fixed"),
        ("cuda_rr", "unsupported", "uniform", "seed_local"),
        ("warp", "rr", "uniform", "seed_local"),
    ],
)
def test_seed_local_continuity_uses_the_public_path_hook(
    backend: str,
    family: str,
    srs_search: str,
    rr_search: str,
) -> None:
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
            max_joint_step=None,
            backend=backend,
            geometry_mode="rr" if family == "rr" else "strict",
            srs_search=srs_search,
            rr_search=rr_search,
        ),
        family=family,
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


@pytest.mark.parametrize("provider", ["_near", "_rr", "_geometric"])
def test_memory_attribution_includes_borrowed_numerical_workspace_once(
    provider: str,
) -> None:
    normal = torch.empty(64, dtype=torch.float64)
    flags = torch.empty(32, dtype=torch.int32)
    solver = SimpleNamespace(
        cfg=SimpleNamespace(class_type="SSIKSolver", backend="warp"),
        dof=6,
        _ik_buffers=SimpleNamespace(arrays={"flags": flags}),
    )
    setattr(
        solver,
        provider,
        SimpleNamespace(_workspace={"normal": normal, "alias": normal}),
    )
    result = _memory(
        lambda: (torch.ones(1, dtype=torch.bool), torch.zeros(1, 6)),
        torch.device("cpu"),
        solver,
    )
    # One 64-double numerical workspace plus32 flags, regardless of aliasing.
    assert result["solver_buffer_mb"] == (512 + 128) / 2**20
    assert result["tensor_scratch_gate"] == "PASS"


@pytest.mark.parametrize(
    "options",
    [
        SSIKBenchmarkOptions(geometry_mode="strict", rr_search="seed_local"),
        SSIKBenchmarkOptions(robot_e2e=True, quality_only=True),
        SSIKBenchmarkOptions(skip_reference=False, timing_only=True),
        SSIKBenchmarkOptions(reference_artifacts=("arx5=model.py",)),
    ],
)
def test_invalid_optional_scopes_reject_before_runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
    options: SSIKBenchmarkOptions,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(init=lambda: pytest.fail("Runtime must not initialize")),
    )
    with pytest.raises(ValueError, match="requires"):
        benchmark_ssik_solver(options)


@pytest.mark.parametrize("paired", [False, True])
def test_real_robot_worker_serializes_numpy_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paired: bool,
) -> None:
    from scripts.benchmark.robotics.kinematic_solver import ssik_benchmark as module

    captured = {}

    def run_worker(command: list[str], **kwargs: object):
        spec = Path(command[command.index("--spec") + 1])
        captured.update(json.loads(spec.read_text()))
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run_worker)
    cfg = SimpleNamespace(
        backend="warp",
        to_dict=lambda: {
            "tcp": np.eye(4),
            "ik_nearest_weight": np.ones(6, dtype=np.float32),
        },
    )
    result = module._measure_robot_interface(
        SimpleNamespace(name="aloha_left"),
        SimpleNamespace(cfg=cfg),
        torch.device("cpu"),
        tmp_path,
        SSIKBenchmarkOptions(smoke=True),
        partner=(
            (SimpleNamespace(name="aloha_right"), SimpleNamespace(cfg=cfg))
            if paired
            else None
        ),
    )
    assert result["status"] == "completed"
    assert captured["solver_cfg"]["tcp"] == np.eye(4).tolist()
    assert captured["solver_cfg"]["ik_nearest_weight"] == [1.0] * 6
    assert captured["repeats"] == 3 and captured["warmup"] == 20
    assert ("partner" in captured) == paired
    if paired:
        assert captured["partner"]["model"] == "aloha_right"
        assert captured["partner"]["solver_cfg"]["tcp"] == np.eye(4).tolist()


@pytest.mark.parametrize(
    "mode,srs,rr",
    [
        ("strict", "uniform", "fixed"),
        ("strict", "seed_local", "fixed"),
        ("near_parallel", "uniform", "fixed"),
        ("rr", "uniform", "seed_local"),
    ],
)
def test_algorithm_defaults_inherit_presets_and_explicit_values_replace(
    monkeypatch: pytest.MonkeyPatch, mode: str, srs: str, rr: str
) -> None:
    from embodichain.lab.sim.motion.solvers import SSIKSolverCfg

    monkeypatch.setattr(SSIKSolverCfg, "init_solver", lambda self, **kwargs: self)
    model = SimpleNamespace(
        cfg=SimpleNamespace(fpath="actual.urdf"),
        joint_names=["joint"],
        manifest={"root": "root", "end": "end"},
        tcp=torch.eye(4),
        limits=torch.tensor([[-1.0], [1.0]]),
        solver_options={"geometry_mode": mode, "srs_search": srs, "rr_search": rr},
    )
    options = SSIKBenchmarkOptions()
    assert options.skip_reference
    inherited = _make_solver(model, "warp", torch.device("cpu"), options=options)
    assert (inherited.geometry_mode, inherited.srs_search, inherited.rr_search) == (
        mode,
        srs,
        rr,
    )
    overridden = _make_solver(
        model,
        "warp",
        torch.device("cpu"),
        options=SSIKBenchmarkOptions(
            geometry_mode="rr", srs_search="uniform", rr_search="fixed"
        ),
    )
    assert (overridden.geometry_mode, overridden.srs_search, overridden.rr_search) == (
        "rr",
        "uniform",
        "fixed",
    )
    assert model.solver_options == {
        "geometry_mode": mode,
        "srs_search": srs,
        "rr_search": rr,
    }


def test_original_pytorch_baseline_ignores_new_robot_solver_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

    monkeypatch.setattr(PytorchSolverCfg, "init_solver", lambda self, **kwargs: self)
    model = SimpleNamespace(
        cfg=SimpleNamespace(fpath="actual.urdf"),
        joint_names=["joint"],
        manifest={"root": "root", "end": "end"},
        tcp=torch.eye(4),
        limits=torch.tensor([[-1.0], [1.0]]),
        solver_options={
            "geometry_mode": "rr",
            "srs_search": "seed_local",
            "rr_search": "seed_local",
        },
    )
    original = _make_solver(model, "pytorch_original", torch.device("cpu"))
    matched = _make_solver(model, "pytorch_matched", torch.device("cpu"))
    assert original.num_samples == matched.num_samples == 30
    assert original.max_iterations == matched.max_iterations == 500
    assert original.pos_eps == original.rot_eps == 5e-4
    assert matched.pos_eps == matched.rot_eps == 1e-5
    assert original.root_link_name == "root" and original.end_link_name == "end"


@pytest.mark.parametrize("search", ["fixed", "seed_local"])
def test_rr_mode_reaches_cpu_runner_and_actual_mode_gate_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, search: str
) -> None:
    from embodichain.compute.kinematics import _ssik_model
    from scripts.benchmark.robotics.kinematic_solver import ssik_benchmark as module

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(init=lambda: None))
    monkeypatch.setattr(module, "_versions", lambda: {})
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda name: SimpleNamespace(chain=object(), solver_options={}),
    )
    target, seed = torch.eye(4)[None], torch.zeros((1, 7))
    monkeypatch.setattr(
        module, "_build_workloads", lambda *args: {"wide": (target, seed)}
    )
    monkeypatch.setattr(
        _ssik_model,
        "_extract_model",
        lambda *args, **kwargs: SimpleNamespace(
            family="unsupported", fingerprint="actual"
        ),
    )
    seen = []
    monkeypatch.setattr(
        module,
        "_make_solver",
        lambda model, backend, device, artifact, options: seen.append(
            (backend, device.type, options.geometry_mode, options.rr_search)
        )
        or object(),
    )
    monkeypatch.setattr(
        module, "_nearest", lambda *args: (torch.ones(1, dtype=torch.bool), seed)
    )
    monkeypatch.setattr(
        module,
        "_evaluate_workload",
        lambda *args: {"success_rate": 1.0, "false_successes": 0},
    )
    options = SSIKBenchmarkOptions(
        models=("flexiv_rizon4s",),
        devices=("cpu",),
        geometry_mode="rr",
        rr_search=search,
        skip_baseline=True,
        quality_only=True,
        artifact_dir=tmp_path,
    )
    performance, metrics, notes = benchmark_ssik_solver(options)
    assert seen == [("warp", "cpu", "rr", search)]
    impl = _implementation_name("warp", "cpu", options)
    assert (
        impl
        == "ssik_warp_rr" + ("_seed_local" if search == "seed_local" else "") + "_cpu"
    )
    assert (
        performance
        and metrics
        and all(
            row["impl"] == impl and row["geometry_mode"] == "rr"
            for row in performance + metrics
        )
    )
    assert not any("Skipped flexiv_rizon4s/warp/cpu" in note for note in notes)


def test_optional_reference_has_correctness_rows_without_timed_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.compute.kinematics import _ssik_model
    from scripts.benchmark.robotics.kinematic_solver import ssik_benchmark as module

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(init=lambda: None))
    monkeypatch.setattr(module, "_versions", lambda: {})
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda name: SimpleNamespace(chain=object(), solver_options={}),
    )
    target, seed = torch.eye(4)[None], torch.zeros((1, 6))
    monkeypatch.setattr(
        module, "_build_workloads", lambda *args: {"wide": (target, seed)}
    )
    monkeypatch.setattr(
        _ssik_model,
        "_extract_model",
        lambda *args, **kwargs: SimpleNamespace(
            family="three_parallel", fingerprint="actual"
        ),
    )
    called = []
    monkeypatch.setattr(
        module,
        "_make_solver",
        lambda model, backend, *args: called.append(backend) or object(),
    )
    monkeypatch.setattr(
        module,
        "_evaluate_workload",
        lambda *args: {"success_rate": 1.0, "false_successes": 0},
    )

    def no_timing(*args, **kwargs):
        pytest.fail("optional reference alignment must not enter a timing path")

    monkeypatch.setattr(module, "_nearest", no_timing)
    monkeypatch.setattr(module, "_measure", no_timing)
    monkeypatch.setattr(module, "_memory", no_timing)
    performance, metrics, _ = benchmark_ssik_solver(
        SSIKBenchmarkOptions(
            models=("aloha_left",),
            devices=("cpu",),
            skip_reference=False,
            skip_warp=True,
            skip_baseline=True,
            scenarios=("wide",),
            artifact_dir=tmp_path,
        )
    )
    assert called == ["reference"]
    assert (
        not performance
        and len(metrics) == 1
        and metrics[0]["impl"] == "ssik_reference_cpu"
    )


@pytest.mark.parametrize("rr_search", [None, "seed_local"])
def test_legacy_cuda_rr_keeps_fixed_search_unless_explicit(
    monkeypatch: pytest.MonkeyPatch, rr_search: str | None
) -> None:
    received = {}

    def cfg(**kwargs):
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
        solver_options={"geometry_mode": "rr", "rr_search": "seed_local"},
    )
    _make_solver(
        model,
        "cuda_rr",
        torch.device("cuda"),
        options=SSIKBenchmarkOptions(rr_search=rr_search),
    )
    assert received["rr_search"] == (rr_search or "fixed")
    assert "geometry_mode" not in received


@pytest.mark.parametrize("mode", ["strict", "near_parallel", "rr"])
def test_global_rr_search_only_changes_real_rr_configurations(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from embodichain.lab.sim.motion.solvers import SSIKSolverCfg

    monkeypatch.setattr(SSIKSolverCfg, "init_solver", lambda self, **kwargs: self)
    model = SimpleNamespace(
        cfg=SimpleNamespace(fpath="actual.urdf"),
        joint_names=["joint"],
        manifest={"root": "root", "end": "end"},
        tcp=torch.eye(4),
        limits=torch.tensor([[-1.0], [1.0]]),
        solver_options={
            "geometry_mode": mode,
            "srs_search": "uniform",
            "rr_search": "fixed",
        },
    )
    cfg = _make_solver(
        model,
        "warp",
        torch.device("cpu"),
        options=SSIKBenchmarkOptions(rr_search="seed_local"),
    )
    assert isinstance(cfg, SSIKSolverCfg) and cfg.backend == "warp"
    assert cfg.geometry_mode == mode
    assert cfg.rr_search == ("seed_local" if mode == "rr" else "fixed")
