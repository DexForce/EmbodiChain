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

"""Reproducible SSIK model, correctness, latency and acceptance workloads.

Used by the unified kinematic solver benchmark's explicit ``--solvers ssik``
mode. Large fixture arrays and raw timing samples are saved separately from the
single three-table Markdown report. Smoke mode deliberately cannot certify the
production acceptance gates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import psutil
import torch

__all__ = ["SSIKBenchmarkOptions", "benchmark_ssik_solver", "evaluate_quality"]

SSIK_REVISION = "4109147f37a7ca3e2516c83853aca583612783f3"
DEFAULT_MODEL_NAMES = (
    "aloha_left",
    "aloha_right",
    "marvin_bare_left",
    "marvin_bare_right",
    "marvin_gripper_left",
    "marvin_gripper_right",
    "arx5",
    "flexiv_rizon4s",
)
MODEL_NAMES = DEFAULT_MODEL_NAMES


@dataclass(frozen=True)
class SSIKBenchmarkOptions:
    """Explicitly selected benchmark workload and runtime settings."""

    smoke: bool = False
    models: tuple[str, ...] = DEFAULT_MODEL_NAMES
    devices: tuple[str, ...] = ("cpu", "cuda")
    skip_reference: bool = True
    skip_baseline: bool = False
    skip_warp: bool = False
    quality_only: bool = False
    timing_only: bool = False
    batch_sizes: tuple[int, ...] | None = None
    baseline_variants: tuple[str, ...] = ("original", "matched")
    timing_scopes: tuple[str, ...] = ("resident", "transfer", "sustained")
    scenarios: tuple[str, ...] = (
        "wide",
        "boundary",
        "singular",
        "unreachable",
        "trajectory",
    )
    seed: int = 2531
    cpu_threads: int = 1
    artifact_dir: Path | None = None
    reference_artifacts: tuple[str, ...] = ()
    geometry_mode: str | None = None
    axis_alignment_tolerance: float = 1e-5
    srs_search: str | None = None
    rr_search: str | None = None
    cuda_rr: bool = False
    robot_e2e: bool = False


@dataclass
class _Model:
    name: str
    cfg: Any
    chain: Any
    joint_names: list[str]
    limits: torch.Tensor
    tcp: torch.Tensor
    manifest: dict[str, Any]
    solver_options: dict[str, str] | None = None

    def fk(self, qpos: torch.Tensor) -> torch.Tensor:
        return (
            self.chain.forward_kinematics(qpos.to(dtype=torch.float64)).get_matrix()
            @ self.tcp
        )


def _load_model(name: str) -> _Model:
    from embodichain_robots import (
        AlohaMiniCfg,
        ARX5Cfg,
        FlexivRizon4SCfg,
        TianjiMarvinCfg,
    )

    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown SSIK benchmark model: {name}")
    if name == "arx5":
        cfg, part = ARX5Cfg.from_dict({}), "arm"
    elif name == "flexiv_rizon4s":
        cfg, part = FlexivRizon4SCfg.from_dict({}), "arm"
    else:
        cfg = (
            AlohaMiniCfg.from_dict({})
            if name.startswith("aloha")
            else TianjiMarvinCfg.from_dict({"with_gripper": "gripper" in name})
        )
        part = "left_arm" if name.endswith("left") else "right_arm"
    chain = cfg.build_pk_serial_chain(device=torch.device("cpu"))[part].to(
        dtype=torch.float64
    )
    solver_cfg = cfg.solver_cfg[part]
    joint_names = list(chain.get_joint_parameter_names())
    if joint_names != list(cfg.control_parts[part]):
        raise ValueError(f"{name}: PK and control-group joint ordering disagree")
    limits = torch.tensor(chain.get_joint_limits(), dtype=torch.float64)
    tcp = torch.as_tensor(solver_cfg.tcp, dtype=torch.float64)
    payload = {
        "asset_sha256": hashlib.sha256(Path(cfg.fpath).read_bytes()).hexdigest(),
        "asset_path": str(cfg.fpath),
        "root": solver_cfg.root_link_name,
        "end": solver_cfg.end_link_name,
        "joint_names": joint_names,
        "limits": limits.tolist(),
        "tcp": tcp.tolist(),
        "model": name,
    }
    payload["model_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "asset_path"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    preset_file = Path(import_module(type(cfg).__module__).__file__).resolve()
    payload["preset_source_file"] = str(preset_file)
    payload["preset_source_sha256"] = hashlib.sha256(
        preset_file.read_bytes()
    ).hexdigest()
    solver_options = {
        "geometry_mode": getattr(solver_cfg, "geometry_mode", "strict"),
        "srs_search": getattr(solver_cfg, "srs_search", "uniform"),
        "rr_search": getattr(solver_cfg, "rr_search", "fixed"),
    }
    return _Model(name, cfg, chain, joint_names, limits, tcp, payload, solver_options)


def _singular_samples(
    model: _Model, rng: np.random.Generator, count: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Find rank-deficient URDF Jacobian witnesses, then perturb within limits."""
    from scipy.optimize import least_squares

    lower, upper = model.limits.numpy()
    witnesses: list[np.ndarray] = []
    singular_values: list[float] = []

    def residual(qpos: np.ndarray) -> np.ndarray:
        jacobian = model.chain.jacobian(torch.tensor(qpos[None], dtype=torch.float64))[
            0
        ]
        return np.array([torch.linalg.svdvals(jacobian)[-1].item()])

    for _ in range(4):
        initial = rng.uniform(lower + 1e-8, upper - 1e-8)
        result = least_squares(
            residual,
            initial,
            bounds=(lower + 1e-10, upper - 1e-10),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=400,
        )
        sigma = float(residual(result.x)[0])
        if sigma < 1e-7:
            witnesses.append(result.x)
            singular_values.append(sigma)
    if not witnesses:
        raise RuntimeError(
            f"{model.name}: no verified singular witness; cannot label arbitrary samples singular"
        )
    qpos = np.stack([witnesses[index % len(witnesses)] for index in range(count)])
    # Retain exact witnesses; the remaining targets are distinct perturbations.
    qpos[len(witnesses) :] += rng.uniform(
        -1e-6, 1e-6, size=qpos[len(witnesses) :].shape
    )
    qpos = np.clip(qpos, lower, upper)
    return torch.from_numpy(qpos), {
        "method": "bounded least_squares of raw PK Jacobian sigma_min",
        "witness_sigma_min": singular_values,
        "near_singular_perturbation_rad": 1e-6,
    }


def _workspace_bound(model: _Model) -> float:
    """Conservative serial-chain reach bound from URDF fixed translations."""
    root = ET.parse(model.cfg.fpath).getroot()
    parent_joints = {
        joint.find("child").get("link"): joint for joint in root.findall("joint")
    }
    end, base = model.manifest["end"], model.manifest["root"]
    bound = float(torch.linalg.vector_norm(model.tcp[:3, 3]).item())
    while end != base:
        joint = parent_joints[end]
        origin = joint.find("origin")
        if origin is not None:
            bound += float(
                np.linalg.norm(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "))
            )
        end = joint.find("parent").get("link")
    return bound


def _build_workloads(
    model: _Model, options: SSIKBenchmarkOptions, directory: Path
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Save exact float64 URDF/PK targets and independent seed arrays."""
    rng = np.random.default_rng(options.seed)
    lower, upper = model.limits.numpy()
    counts = (32, 12, 8, 8) if options.smoke else (10000, 1000, 200, 500)
    wide = torch.from_numpy(rng.uniform(lower, upper, size=(counts[0], len(lower))))
    boundary = rng.uniform(lower, upper, size=(counts[1], len(lower)))
    for index in range(counts[1]):
        joint = index % len(lower)
        fraction = rng.uniform(0, 1e-3)
        boundary[index, joint] = (
            lower[joint] + fraction * (upper[joint] - lower[joint])
            if index % 2
            else upper[joint] - fraction * (upper[joint] - lower[joint])
        )
    singular, construction = _singular_samples(model, rng, counts[2])
    qpos_groups = {
        "wide": wide,
        "boundary": torch.from_numpy(boundary),
        "singular": singular,
    }
    workloads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    arrays: dict[str, np.ndarray] = {}
    for scenario, qpos in qpos_groups.items():
        target = model.fk(qpos)
        seed = torch.from_numpy(rng.uniform(lower, upper, size=qpos.shape))
        workloads[scenario] = (target, seed)
        arrays.update(
            {
                f"{scenario}_qpos": qpos.numpy(),
                f"{scenario}_target": target.numpy(),
                f"{scenario}_seed": seed.numpy(),
            }
        )
    unreachable = torch.eye(4, dtype=torch.float64).repeat(counts[3], 1, 1)
    directions = torch.from_numpy(rng.normal(size=(counts[3], 3)))
    directions /= directions.norm(dim=-1, keepdim=True)
    bound = _workspace_bound(model)
    unreachable[:, :3, 3] = directions * (bound + 1.0)
    seed = torch.from_numpy(rng.uniform(lower, upper, size=(counts[3], len(lower))))
    workloads["unreachable"] = (unreachable, seed)
    arrays.update(unreachable_target=unreachable.numpy(), unreachable_seed=seed.numpy())
    trajectories, initial_seeds, kinds = [], [], []
    frames, paths = (16, 3) if options.smoke else (1000, 20)
    phase = torch.linspace(0, 1, frames, dtype=torch.float64)
    for index in range(paths):
        start = torch.from_numpy(
            lower + rng.uniform(0.35, 0.65, len(lower)) * (upper - lower)
        )
        if index % 3 == 0:
            finish = torch.from_numpy(
                lower + rng.uniform(0.35, 0.65, len(lower)) * (upper - lower)
            )
            qpos = (
                start
                + (finish - start) * (3 * phase.square() - 2 * phase.pow(3))[:, None]
            )
            target = model.fk(qpos)
            kind = "smooth_joint"
        else:
            target = model.fk(start[None]).repeat(frames, 1, 1)
            radius = min(0.002, bound * 0.002)
            if index % 3 == 1:
                target[:, 0, 3] += radius * phase
                kind = "cartesian_line"
            else:
                target[:, 0, 3] += radius * (torch.cos(2 * torch.pi * phase) - 1)
                target[:, 1, 3] += radius * torch.sin(2 * torch.pi * phase)
                kind = "cartesian_circle"
        trajectories.append(target)
        initial_seeds.append(start)
        kinds.append(kind)
    arrays.update(
        trajectory_target=torch.stack(trajectories).numpy(),
        trajectory_seed=torch.stack(initial_seeds).numpy(),
    )
    np.savez_compressed(directory / f"{model.name}.npz", **arrays)
    manifest = {
        **model.manifest,
        "seed": options.seed,
        "smoke": options.smoke,
        "workload_counts": dict(zip(qpos_groups, counts[:3])),
        "unreachable_count": counts[3],
        "unreachable_translation_radius_m": bound + 1.0,
        "reach_upper_bound_m": bound,
        "singular_construction": construction,
        "trajectory_kinds": kinds,
        "trajectory_frames": frames,
        "npz_sha256": hashlib.sha256(
            (directory / f"{model.name}.npz").read_bytes()
        ).hexdigest(),
    }
    (directory / f"{model.name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    workloads["trajectory"] = (torch.stack(trajectories), torch.stack(initial_seeds))
    return workloads


def evaluate_quality(
    target: torch.Tensor,
    qpos: torch.Tensor,
    reported: torch.Tensor,
    fk: Callable[[torch.Tensor], torch.Tensor],
    limits: torch.Tensor,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Independently reject nonfinite, out-of-limit and inaccurate IK successes."""
    qpos, target, reported = (
        qpos.detach().cpu().double(),
        target.cpu().double(),
        reported.detach().cpu().bool().reshape(-1),
    )
    qpos = qpos.reshape(len(reported), -1)
    finite = torch.isfinite(qpos).all(-1)
    safe = torch.where(finite[:, None], qpos, limits.mean(0))
    actual = fk(safe)
    position = (actual[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    relative = actual[:, :3, :3].transpose(-1, -2) @ target[:, :3, :3]
    skew = torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        -1,
    )
    rotation = torch.atan2(
        0.5 * skew.norm(dim=-1), 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    )
    inside = ((safe >= limits[0] - 1e-7) & (safe <= limits[1] + 1e-7)).all(-1)
    accepted = (
        reported & finite & inside & (position <= tolerance) & (rotation <= tolerance)
    )
    return {
        "sample_size": len(reported),
        "success_rate": accepted.double().mean().item(),
        "reported_success_rate": reported.double().mean().item(),
        "false_successes": int((reported & ~accepted).sum().item()),
        "translation_err_mm": (
            position[accepted].mean().item() * 1000 if accepted.any() else 0.0
        ),
        "rotation_err_deg": (
            rotation[accepted].mean().item() * 180 / np.pi if accepted.any() else 0.0
        ),
        "max_position_m": position[accepted].max().item() if accepted.any() else None,
        "max_rotation_rad": rotation[accepted].max().item() if accepted.any() else None,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        import warp as wp

        wp.synchronize_device(str(device))
        torch.cuda.synchronize(device)


def _measure(
    operation: Callable[[], Any], device: torch.device, repeats: int, warmup: int = 5
) -> tuple[dict[str, Any], list[float]]:
    for _ in range(warmup):
        torch.manual_seed(2531)
        result = operation()
        _sync(device)
        del result
    samples = []
    for _ in range(repeats):
        torch.manual_seed(2531)
        _sync(device)
        started = time.perf_counter()
        result = operation()
        _sync(device)
        samples.append((time.perf_counter() - started) * 1000)
        del result
    percentiles = np.percentile(samples, [50, 95, 99, 100])
    return (
        dict(
            zip(
                ("cost_time_ms", "p95_ms", "p99_ms", "max_ms"), map(float, percentiles)
            ),
            repeats=repeats,
        ),
        samples,
    )


@dataclass
class _ControlMeasurement:
    latency: dict[str, Any]
    samples_ms: list[float]
    valid: tuple[torch.Tensor, ...]
    qpos: tuple[torch.Tensor, ...]


def _measure_streaming_control(
    solvers: tuple[Any, ...],
    targets: tuple[torch.Tensor, ...],
    initial_seeds: tuple[torch.Tensor, ...],
    device: torch.device,
    repeats: int,
    warmup: int = 20,
) -> _ControlMeasurement:
    """Time changing goals using the preceding returned joints as the seed.

    Each target tensor is shaped ``(paths, frames, 4, 4)``. The initial seed
    resets only at path boundaries. Preparation, state updates, output reads
    and later independent FK checks stay outside the completed-call timer.
    """
    if (
        not solvers
        or len(solvers) != len(targets)
        or len(solvers) != len(initial_seeds)
    ):
        raise ValueError("Control timing needs matching solver, target and seed tuples")
    shape = targets[0].shape[:2]
    if repeats < 1 or repeats > shape[0] * shape[1]:
        raise ValueError("Control timing requires enough distinct trajectory frames")
    if any(target.shape[:2] != shape for target in targets):
        raise ValueError("Paired control paths must have matching frame counts")
    goals = tuple(target.to(device=device, dtype=torch.float32) for target in targets)
    starts = tuple(
        seed.to(device=device, dtype=torch.float32) for seed in initial_seeds
    )
    previous_bounds = [solver.cfg.max_joint_step for solver in solvers]
    samples: list[float] = []
    flags: list[list[torch.Tensor]] = [[] for _ in solvers]
    joints: list[list[torch.Tensor]] = [[] for _ in solvers]
    try:
        for solver in solvers:
            solver.cfg.max_joint_step = 0.05
        seeds = tuple(seed[:1].clone() for seed in starts)
        for _ in range(warmup):
            results = tuple(
                solver.get_ik(goal[0, :1], seed)
                for solver, goal, seed in zip(solvers, goals, seeds)
            )
            _sync(device)
            seeds = tuple(result[1] for result in results)
        for path in range(shape[0]):
            seeds = tuple(seed[path : path + 1].clone() for seed in starts)
            for frame in range(shape[1]):
                if len(samples) == repeats:
                    break
                selected = tuple(goal[path, frame : frame + 1] for goal in goals)
                _sync(device)
                started = time.perf_counter()
                results = tuple(
                    solver.get_ik(goal, seed)
                    for solver, goal, seed in zip(solvers, selected, seeds)
                )
                _sync(device)
                samples.append((time.perf_counter() - started) * 1000)
                seeds = tuple(result[1] for result in results)
                for index, (valid, qpos) in enumerate(results):
                    flags[index].append(valid.detach().cpu().clone())
                    joints[index].append(qpos.detach().cpu().clone())
            if len(samples) == repeats:
                break
    finally:
        for solver, bound in zip(solvers, previous_bounds):
            solver.cfg.max_joint_step = bound
    valid = tuple(torch.cat(rows).reshape(repeats) for rows in flags)
    qpos = tuple(torch.cat(rows).reshape(repeats, -1) for rows in joints)
    together = torch.stack(valid).all(dim=0)
    percentile = np.percentile(samples, [50, 95, 99, 100])
    latency = dict(
        zip(("cost_time_ms", "p95_ms", "p99_ms", "max_ms"), map(float, percentile)),
        repeats=repeats,
        failure_count=int((~together).sum()),
        both_reported_success_rate=together.double().mean().item(),
    )
    return _ControlMeasurement(latency, samples, valid, qpos)


def _worker_json_value(value: Any) -> Any:
    """Preserve numerical configuration values when writing a worker spec."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Unsupported Robot worker configuration value: {type(value).__name__}"
    )


def _measure_robot_interface(
    model: _Model,
    solver: Any,
    device: torch.device,
    directory: Path,
    options: SSIKBenchmarkOptions,
    partner: tuple[_Model, Any] | None = None,
) -> dict[str, Any]:
    """Measure a real engine-backed Robot in an isolated worker process."""
    name = model.name if partner is None else model.name.removesuffix("left") + "pair"
    stem = f"{name}_{solver.cfg.backend}_{device.type}_robot_e2e"
    spec_path, output = directory / f"{stem}_spec.json", directory / f"{stem}.json"
    spec = {
        "model": model.name,
        "solver_cfg": solver.cfg.to_dict(),
        "device": str(device),
        "cpu_threads": options.cpu_threads,
        "fixture": str((directory / f"{model.name}.npz").resolve()),
        "repeats": 3 if options.smoke else 10000,
        "warmup": 20,
    }
    if partner is not None:
        other_model, other_solver = partner
        spec["partner"] = {
            "model": other_model.name,
            "solver_cfg": other_solver.cfg.to_dict(),
            "fixture": str((directory / f"{other_model.name}.npz").resolve()),
        }
    spec_path.write_text(json.dumps(spec, indent=2, default=_worker_json_value) + "\n")
    repository = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark.robotics.kinematic_solver._ssik_robot_e2e",
            "--spec",
            str(spec_path.resolve()),
            "--output",
            str(output.resolve()),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=600,
    )
    log = directory / f"{stem}.log"
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode or not output.is_file():
        raise RuntimeError(f"Robot end-to-end worker failed; inspect {log}")
    result = json.loads(output.read_text())
    if result.get("status") not in ("completed", "unavailable"):
        raise RuntimeError(
            f"Robot end-to-end worker returned an invalid status: {output}"
        )
    return result


def _throughput(
    operation: Callable[[], Any], device: torch.device, count: int, repeats: int
) -> float:
    """Measure a repeated-work window with synchronization only at boundaries."""
    torch.manual_seed(2531)
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        output = operation()
        del output
    _sync(device)
    return count * repeats / (time.perf_counter() - started)


def _memory(
    operation: Callable[[], Any], device: torch.device, solver: Any
) -> dict[str, Any]:
    """Separate warmed Torch peaks, Warp pool counters and persistent scratch."""
    import warp as wp

    _sync(device)
    process = psutil.Process()
    cpu_before = process.memory_info().rss
    torch_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    pool = device.type == "cuda" and wp.is_mempool_enabled(str(device))
    warp_before = wp.get_mempool_used_mem_current(str(device)) if pool else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = operation()
    _sync(device)
    torch_peak = (
        torch.cuda.max_memory_allocated(device) - torch_before
        if device.type == "cuda"
        else 0
    )
    warp_after = wp.get_mempool_used_mem_current(str(device)) if pool else None
    # Counters cannot establish a transient Warp allocation peak. Explicit
    # solver scratch tensors are counted separately, including warmed reuse.
    buffers = getattr(solver, "_ik_buffers", getattr(solver, "_buffers", None))
    tensors = getattr(buffers, "arrays", {}) if buffers is not None else {}
    numerical_workspaces = [
        getattr(getattr(solver, name, None), "_workspace", {})
        for name in ("_near", "_rr", "_geometric")
    ]
    seen: set[int] = set()
    scratch = 0
    for value in [
        *tensors.values(),
        *(value for workspace in numerical_workspaces for value in workspace.values()),
    ]:
        if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            scratch += value.untyped_storage().nbytes()
    attributed_scratch: float | str = "unavailable"
    scratch_status = (
        "UNVALIDATED: allocation ownership not established for this backend"
    )
    tensor_scratch_status = scratch_status
    if getattr(
        solver.cfg, "class_type", None
    ) == "SSIKSolver" and solver.cfg.backend in ("warp", "cuda_rr"):
        # Candidate and numerical workspaces use Torch-owned zero-copy views.
        # Driver local-stack reservation is not captured by these counters.
        output_tensors = [value for value in output if isinstance(value, torch.Tensor)]
        output_bytes = sum(
            value.numel() * value.element_size() for value in output_tensors
        )
        if device.type == "cuda":
            transient = max(0, torch_peak - output_bytes)
        else:
            count = output_tensors[0].numel()
            # Float64 target and seed conversions plus effective limits/weights.
            transient = count * (16 + solver.dof) * 8 + solver.dof * 20
        attributed_scratch = (scratch + transient) / 2**20
        tensor_scratch_status = "PASS" if attributed_scratch <= 128 else "FAIL"
        scratch_status = (
            "UNVALIDATED: fresh-process CUDA local-stack attribution required"
            if device.type == "cuda" and tensor_scratch_status == "PASS"
            else tensor_scratch_status
        )
    result = {
        "cpu_rss_delta_mb": (process.memory_info().rss - cpu_before) / 2**20,
        "torch_peak_increment_mb": torch_peak / 2**20,
        "warp_pool_delta_mb": (
            (warp_after - warp_before) / 2**20 if pool else "unavailable"
        ),
        "warp_pool_peak_mb": (
            wp.get_mempool_used_mem_high(str(device)) / 2**20 if pool else "unavailable"
        ),
        "solver_buffer_mb": (
            scratch / 2**20 if tensors or any(numerical_workspaces) else "unavailable"
        ),
        "attributed_scratch_mb": attributed_scratch,
        "tensor_scratch_gate": tensor_scratch_status,
        "scratch_gate": scratch_status,
    }
    del output
    return result


def _parse_reference_artifacts(entries: tuple[str, ...]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for entry in entries:
        name, separator, value = entry.partition("=")
        if not separator or name not in MODEL_NAMES or not value:
            raise ValueError(
                "Reference artifacts require MODEL=PATH with a known SSIK model"
            )
        if name in artifacts:
            raise ValueError(f"Duplicate reference artifact for {name}")
        artifacts[name] = Path(value).expanduser().resolve()
    return artifacts


def _resolved_options(
    model: _Model, options: SSIKBenchmarkOptions
) -> SSIKBenchmarkOptions:
    """Inherit preset algorithms without changing the physical-model fingerprint."""
    defaults = getattr(model, "solver_options", None) or {}
    geometry_mode = (
        options.geometry_mode
        if options.geometry_mode is not None
        else defaults.get("geometry_mode", "strict")
    )
    return replace(
        options,
        geometry_mode=geometry_mode,
        srs_search=(
            options.srs_search
            if options.srs_search is not None
            else defaults.get("srs_search", "uniform")
        ),
        rr_search=(
            (
                options.rr_search
                if options.rr_search is not None
                else defaults.get("rr_search", "fixed")
            )
            if geometry_mode == "rr"
            else "fixed"
        ),
    )


def _make_solver(
    model: _Model,
    backend: str,
    device: torch.device,
    reference_artifact: Path | None = None,
    options: SSIKBenchmarkOptions | None = None,
) -> Any:
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg, SSIKSolverCfg

    common = dict(
        urdf_path=model.cfg.fpath,
        joint_names=model.joint_names,
        root_link_name=model.manifest["root"],
        end_link_name=model.manifest["end"],
        tcp=model.tcp.numpy(),
        user_qpos_limits=model.limits.tolist(),
    )
    if backend.startswith("pytorch"):
        # Reconstruct the original numerical configuration independently of any
        # new robot default; only the matched variant tightens its tolerances.
        matched = (
            {"pos_eps": 1e-5, "rot_eps": 1e-5} if backend == "pytorch_matched" else {}
        )
        cfg = PytorchSolverCfg(**common, **matched)
    else:
        overrides = options or SSIKBenchmarkOptions()
        selected = _resolved_options(model, overrides)
        geometry = (
            {
                "geometry_mode": selected.geometry_mode,
                "axis_alignment_tolerance": selected.axis_alignment_tolerance,
                "srs_search": selected.srs_search,
                "rr_search": selected.rr_search,
            }
            if backend == "warp"
            else {}
        )
        if backend == "cuda_rr":
            # The legacy comparison keeps its historical fixed chart pool.
            geometry["rr_search"] = overrides.rr_search or "fixed"
        cfg = SSIKSolverCfg(
            **common,
            **geometry,
            backend=backend,
            num_samples=128,
            chunk_size=512,
            position_tolerance=1e-5,
            rotation_tolerance=1e-5,
            reference_artifact=(
                str(reference_artifact)
                if backend == "reference" and reference_artifact is not None
                else None
            ),
        )
    return cfg.init_solver(device=device)


def _nearest(
    solver: Any, target: torch.Tensor, seed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reset the baseline's random multi-start seeds outside every timing window
    # through the caller, not inside this operation.
    valid, qpos = solver.get_ik(target, qpos_seed=seed)
    return valid.reshape(target.shape[:-2]), qpos.reshape(*target.shape[:-2], -1)


def _evaluate_workload(
    solver: Any,
    model: _Model,
    target: torch.Tensor,
    seed: torch.Tensor,
    device: torch.device,
    output_path: Path | None = None,
) -> dict[str, Any]:
    valid_rows, qpos_rows = [], []
    for offset in range(0, len(target), 512):
        torch.manual_seed(2531 + offset)
        valid, qpos = _nearest(
            solver,
            target[offset : offset + 512].to(device, dtype=torch.float32),
            seed[offset : offset + 512].to(device, dtype=torch.float32),
        )
        valid_rows.append(valid.cpu())
        qpos_rows.append(qpos.cpu())
    qpos, reported = torch.cat(qpos_rows), torch.cat(valid_rows)
    if output_path is not None:
        np.savez_compressed(output_path, qpos=qpos.numpy(), reported=reported.numpy())
    return evaluate_quality(target, qpos, reported, model.fk, model.limits)


def _evaluate_trajectories(
    solver: Any,
    model: _Model,
    target: torch.Tensor,
    seed: torch.Tensor,
    device: torch.device,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Match public batch semantics: one fixed initial seed per complete path."""
    valid_rows, qpos_rows = [], []
    largest_step = 0.0
    has_step_config = hasattr(solver.cfg, "max_joint_step")
    previous_bound = solver.cfg.max_joint_step if has_step_config else None
    if has_step_config:
        solver.cfg.max_joint_step = 0.05
    try:
        for path, initial in zip(target, seed):
            previous = initial.to(device, dtype=torch.float32)[None]
            # A 1,000-frame SRS candidate output is about30 MiB. Generating
            # the entire path preserves the public API's fixed initial swivel;
            # refreshing seeds in shorter windows would change its coverage.
            for offset in range(0, len(path), len(path)):
                block = path[offset : offset + len(path)].to(
                    device, dtype=torch.float32
                )
                seeds = previous.expand(len(block), -1)
                torch.manual_seed(2531 + offset)
                target_aware = (
                    getattr(solver.cfg, "backend", None) in ("warp", "cuda_rr")
                    and (
                        getattr(solver.cfg, "backend", None) == "cuda_rr"
                        or getattr(solver.cfg, "geometry_mode", None) == "rr"
                    )
                    and getattr(solver.cfg, "rr_search", "fixed") == "seed_local"
                ) or (
                    getattr(solver.cfg, "backend", None) == "warp"
                    and (
                        getattr(solver, "family", None) == "near_parallel"
                        or (
                            getattr(solver.cfg, "srs_search", "uniform") == "seed_local"
                            and getattr(solver, "family", None) == "srs"
                        )
                    )
                )
                if target_aware:
                    valid, chosen = solver._get_continuous_ik_path(
                        block[None], previous
                    )
                    chosen, valid = chosen[0], valid[0]
                else:
                    valid, candidates = solver._get_continuous_ik_candidates(
                        block, seeds
                    )
                    candidates = candidates.reshape(1, len(block), -1, len(initial))
                    valid = valid.reshape(1, len(block), -1)
                if has_step_config and not target_aware:
                    valid, chosen = solver._select_continuous_ik_path(
                        candidates, valid, previous
                    )
                    chosen, valid = chosen[0], valid[0]
                elif not has_step_config:
                    # Apply the requested hard-step constraint before baseline
                    # selection and hold failed postures. Filtering afterwards
                    # would advance from an invalid jump and bias later frames.
                    chosen_frames, valid_frames = [], []
                    rolling = previous
                    for frame in range(len(block)):
                        frame_candidates = candidates[:, frame : frame + 1]
                        inside_step = (
                            frame_candidates - rolling[:, None, None]
                        ).abs().amax(-1) <= 0.05 + 1e-7
                        frame_valid, frame_chosen = solver._select_continuous_ik_path(
                            frame_candidates,
                            valid[:, frame : frame + 1] & inside_step,
                            rolling,
                        )
                        rolling = torch.where(
                            frame_valid[:, 0, None], frame_chosen[:, 0], rolling
                        )
                        chosen_frames.append(rolling)
                        valid_frames.append(frame_valid[:, 0])
                    chosen = torch.cat(chosen_frames)
                    valid = torch.cat(valid_frames)
                differences = (
                    torch.diff(torch.cat((previous, chosen), 0), dim=0).abs().amax(-1)
                )
                step_violations = valid & (differences > 0.05 + 1e-7)
                if step_violations.any():
                    raise RuntimeError(
                        "Continuous selector accepted a posture beyond its hard joint-step bound"
                    )
                largest_step = max(
                    largest_step,
                    differences[valid].max().item() if valid.any() else 0.0,
                )
                valid_rows.append(valid.cpu())
                qpos_rows.append(chosen.cpu())
                previous = chosen[-1:]
    finally:
        if has_step_config:
            solver.cfg.max_joint_step = previous_bound
    if output_path is not None:
        np.savez_compressed(
            output_path,
            qpos=torch.cat(qpos_rows).numpy(),
            reported=torch.cat(valid_rows).numpy(),
        )
    quality = evaluate_quality(
        target.flatten(0, 1),
        torch.cat(qpos_rows),
        torch.cat(valid_rows),
        model.fk,
        model.limits,
    )
    return {**quality, "max_accepted_joint_step_rad": largest_step}


def _versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": (
            next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            )
            if Path("/proc/cpuinfo").is_file()
            else platform.processor()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "ssik_reference_revision": SSIK_REVISION,
    }
    repository = Path(__file__).resolve().parents[4]
    sources = (
        "embodichain/compute/kinematics/_ssik_model.py",
        "embodichain/compute/kinematics/_ssik_geometric.py",
        "embodichain/compute/kinematics/_ssik_near.py",
        "embodichain/compute/kinematics/_ssik_rr.py",
        "embodichain/compute/kinematics/_ssik_rr_parametric.py",
        "embodichain/compute/kinematics/_warp/ssik.py",
        "embodichain/compute/kinematics/_warp/ssik_6r.py",
        "embodichain/compute/kinematics/_warp/ssik_7r.py",
        "embodichain/compute/kinematics/_warp/ssik_near.py",
        "embodichain/compute/kinematics/_warp/ssik_rr.py",
        "embodichain/compute/kinematics/_warp/ssik_rr_cuda.py",
        "embodichain/compute/kinematics/_warp/ssik_rr_parametric.py",
        "embodichain/compute/kinematics/_warp/ssik_rr_parametric_cuda.py",
        "embodichain/compute/kinematics/_warp/ssik_rr_parametric_cpu.py",
        "embodichain/compute/kinematics/_warp/ssik_srs_path.py",
        "embodichain/compute/kinematics/_warp/ssik_srs_search.py",
        "embodichain/compute/kinematics/_warp/ssik_legacy.py",
        "embodichain/compute/kinematics/_warp/ssik_strict_6r.py",
        "embodichain/compute/kinematics/_warp/ssik_srs_geometry.py",
        "embodichain/compute/kinematics/_warp/ssik_srs_optimize.py",
        "embodichain/compute/kinematics/_warp/ssik_srs_parallel.py",
        "embodichain/compute/kinematics/_warp/data/ssik_rr_rizon4.npz",
        "embodichain/compute/kinematics/_warp/data/ssik_rr_rizon4.json",
        "embodichain/compute/kinematics/_warp/data/ssik_rr_rizon4_parametric.npz",
        "embodichain/compute/kinematics/_warp/data/ssik_rr_rizon4_parametric.json",
        "embodichain/lab/sim/motion/solvers/ssik_solver.py",
        "embodichain/lab/sim/motion/solvers/base_solver.py",
        "embodichain/lab/sim/objects/robot.py",
        "embodichain/lab/sim/motion/solvers/_ssik_reference.py",
        "scripts/benchmark/robotics/kinematic_solver/ssik_benchmark.py",
        "scripts/benchmark/robotics/kinematic_solver/run_benchmark.py",
        "scripts/benchmark/robotics/kinematic_solver/prepare_ssik_rr_rizon4.py",
        "scripts/benchmark/robotics/kinematic_solver/_ssik_robot_e2e.py",
    )
    result["source_sha256"] = {
        name: (
            hashlib.sha256((repository / name).read_bytes()).hexdigest()
            if (repository / name).is_file()
            else None
        )
        for name in sources
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    result["git_head"] = head.stdout.strip() if head.returncode == 0 else None
    result["git_dirty"] = (
        bool(status.stdout.strip()) if status.returncode == 0 else None
    )
    result["preset_modules"] = {}
    from embodichain_robots import (
        AlohaMiniCfg,
        ARX5Cfg,
        FlexivRizon4SCfg,
        TianjiMarvinCfg,
    )

    for cfg_type in (AlohaMiniCfg, TianjiMarvinCfg, ARX5Cfg, FlexivRizon4SCfg):
        name = cfg_type.__module__
        source = Path(import_module(name).__file__).resolve()
        result["preset_modules"][name] = {
            "file": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    result["solver_presets"] = {
        "ssik": {
            "num_samples": 128,
            "chunk_size": 512,
            "position_tolerance": 1e-5,
            "rotation_tolerance": 1e-5,
        },
        "pytorch_original": {
            "num_samples": 30,
            "max_iterations": 500,
            "pos_eps": 5e-4,
            "rot_eps": 5e-4,
        },
        "pytorch_matched": {
            "num_samples": 30,
            "max_iterations": 500,
            "pos_eps": 1e-5,
            "rot_eps": 1e-5,
        },
        "continuous_max_joint_step_rad": 0.05,
    }
    for package in ("warp-lang", "pytorch-kinematics", "ssik", "numpy", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _implementation_name(
    backend: str, device: str, options: SSIKBenchmarkOptions
) -> str:
    """Identify the actual algebraic/hybrid mode while retaining legacy names."""
    suffix = ""
    if backend == "warp" and options.geometry_mode in ("near_parallel", "rr"):
        suffix = "_" + options.geometry_mode
    if (
        backend == "cuda_rr" or (backend == "warp" and options.geometry_mode == "rr")
    ) and options.rr_search == "seed_local":
        suffix += "_seed_local"
    return f"ssik_{backend}{suffix}_{device}"


def _selected_implementation(
    rows: list[dict[str, Any]],
    model: str,
    device: str,
    backend: str,
    options: SSIKBenchmarkOptions,
) -> str:
    """Use emitted per-model algorithm metadata rather than global overrides."""
    return next(
        (
            row["impl"]
            for row in rows
            if row.get("model") == model
            and row["impl"].startswith(f"ssik_{backend}_")
            and row["impl"].endswith("_" + device)
        ),
        _implementation_name(backend, device, options),
    )


def _gate_notes(
    perf: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    options: SSIKBenchmarkOptions,
) -> list[str]:
    notes = []
    for model in options.models:
        for device in options.devices:
            selected_backend = (
                "cuda_rr"
                if options.cuda_rr and model == "flexiv_rizon4s" and device == "cuda"
                else "warp"
            )
            backend = _selected_implementation(
                perf + metrics, model, device, selected_backend, options
            )
            quality = {
                row["scenario"]: row
                for row in metrics
                if row.get("model") == model
                and row["impl"] == backend
                and row["scenario"]
                in ("wide", "boundary", "singular", "unreachable", "trajectory")
            }
            baseline = {
                row["scenario"]: row
                for row in metrics
                if row.get("model") == model
                and row["impl"] == f"ssik_pytorch_original_{device}"
            }
            matched = {
                row["scenario"]: row
                for row in metrics
                if row.get("model") == model
                and row["impl"] == f"ssik_pytorch_matched_{device}"
            }
            latency = next(
                (
                    row
                    for row in perf
                    if row.get("model") == model
                    and row["impl"] == backend
                    and row.get("sample_size") == 1
                    and row.get("scope") == "resident"
                ),
                None,
            )
            if options.smoke:
                notes.append(
                    f"Gate {model}/{device}: UNVALIDATED (smoke workload; acceptance requires full workload and repetitions)."
                )
                continue
            if latency is None or latency["repeats"] < 10000:
                latency_status = "UNVALIDATED (requires 10,000 completed-call samples)"
            else:
                latency_status = f"{'PASS' if latency['p99_ms'] <= 5. else 'FAIL'} (p99 {latency['p99_ms']:.4f} ms)"
            notes.append(
                f"Fixed repeated-pose latency diagnostic {model}/{device}: {latency_status}; this does not certify continuous control."
            )
            control = next(
                (
                    row
                    for row in perf
                    if row.get("model") == model
                    and row["impl"] == backend
                    and row.get("scope") == "streaming_control"
                ),
                None,
            )
            if control is None or control["repeats"] < 10000:
                control_status = "UNVALIDATED (requires 10,000 changing-target calls with previous-result seeds)"
            else:
                control_status = f"{'PASS' if control['p99_ms'] <= 5. and not control.get('false_successes', 0) else 'FAIL'} (p99 {control['p99_ms']:.4f} ms; {control.get('failure_count', 'unknown')} reported failures)"
            notes.append(
                f"Single-arm continuous-control latency gate {model}/{device}: {control_status}."
            )
            if options.robot_e2e:
                robot = next(
                    (
                        row
                        for row in perf
                        if row.get("model") == model
                        and row["impl"] == backend
                        and row.get("scope") == "robot_streaming_control"
                    ),
                    None,
                )
                if robot is None or robot["repeats"] < 10000:
                    robot_status = "UNVALIDATED (requires 10,000 real Robot calls)"
                elif (
                    control is None
                    or control["repeats"] < 10000
                    or any(
                        "validated_success_rate" not in row for row in (robot, control)
                    )
                ):
                    robot_status = "UNVALIDATED (requires validated coverage for matching standalone streaming calls)"
                else:
                    passed = robot["p99_ms"] <= 5.0 and not any(
                        robot.get(key, 0)
                        for key in (
                            "false_successes",
                            "step_violations",
                            "failed_hold_violations",
                        )
                    )
                    coverage = (
                        robot["validated_success_rate"]
                        >= control["validated_success_rate"]
                    )
                    robot_status = (
                        f"{'PASS' if passed and coverage else 'FAIL'} "
                        f"(p99 {robot['p99_ms']:.4f} ms; validated coverage "
                        f"{robot['validated_success_rate']:.4%}, standalone "
                        f"{control['validated_success_rate']:.4%})"
                    )
                notes.append(
                    f"Robot end-to-end control gate {model}/{device}: {robot_status}."
                )
            if len(quality) == 5:
                passed = (
                    quality["wide"]["success_rate"] >= 0.999
                    and all(row["false_successes"] == 0 for row in quality.values())
                    and quality["unreachable"]["reported_success_rate"] == 0
                )
                notes.append(
                    f"Reachable coverage and false-accept gate {model}/{device}: {'PASS' if passed else 'FAIL'} (wide {quality['wide']['success_rate']:.4%}; singular coverage {quality['singular']['success_rate']:.4%} recorded separately)."
                )
            else:
                notes.append(
                    f"Reachable coverage and false-accept gate {model}/{device}: UNVALIDATED (missing quality measurements)."
                )
            if len(quality) == 5 and len(baseline) == 5:
                passed = all(
                    quality[key]["success_rate"] >= baseline[key]["success_rate"]
                    for key in ("wide", "boundary", "trajectory")
                )
                notes.append(
                    f"Original-baseline coverage/continuity comparison {model}/{device}: {'PASS' if passed else 'FAIL'}."
                )
            else:
                notes.append(
                    f"Original-baseline coverage/continuity comparison {model}/{device}: UNVALIDATED (missing baseline or Warp quality)."
                )
            if len(quality) == 5 and len(matched) == 5:
                passed = all(
                    quality[key]["success_rate"] >= matched[key]["success_rate"]
                    for key in ("wide", "boundary", "trajectory")
                )
                notes.append(
                    f"Quality-matched baseline coverage/continuity context {model}/{device}: {'PASS' if passed else 'FAIL'} (public batch trajectory Warp {quality['trajectory']['success_rate']:.4%}, matched Pytorch {matched['trajectory']['success_rate']:.4%})."
                )
            if device == "cuda":
                for count in (1000, 10000):
                    warp = next(
                        (
                            row
                            for row in perf
                            if row.get("model") == model
                            and row["impl"] == backend
                            and row.get("sample_size") == count
                            and row.get("scope") == "resident"
                        ),
                        None,
                    )
                    old = next(
                        (
                            row
                            for row in perf
                            if row.get("model") == model
                            and row["impl"] == "ssik_pytorch_original_cuda"
                            and row.get("sample_size") == count
                            and row.get("scope") == "resident"
                        ),
                        None,
                    )
                    if not warp or not old or min(warp["repeats"], old["repeats"]) < 30:
                        status = "UNVALIDATED (requires 30 measurements for both implementations)"
                    else:
                        speedup = old["cost_time_ms"] / warp["cost_time_ms"]
                        status = (
                            f"{'PASS' if speedup >= 2 else 'FAIL'} ({speedup:.3f}x)"
                        )
                    notes.append(f"Throughput gate {model}/cuda n={count}: {status}.")
            scratch_rows = [
                row
                for row in perf
                if row.get("model") == model
                and row["impl"] == backend
                and "scratch_gate" in row
            ]
            scratch_status = (
                "UNVALIDATED (missing allocation measurement)"
                if not scratch_rows
                else (
                    "FAIL"
                    if any(row["scratch_gate"] == "FAIL" for row in scratch_rows)
                    else (
                        "PASS"
                        if all(row["scratch_gate"] == "PASS" for row in scratch_rows)
                        else "UNVALIDATED"
                    )
                )
            )
            notes.append(f"Scratch gate {model}/{device}: {scratch_status}.")
            notes.append(
                f"Default-switch gate {model}/{device}: UNVALIDATED until branch equivalence and Robot end-to-end are validated; numerical singular cases require a documented comparison."
            )
    families = {
        name.removesuffix("left") for name in options.models if name.endswith("left")
    }
    for family in sorted(families):
        if family + "right" not in options.models:
            continue
        for device in options.devices:
            pair = next(
                (
                    row
                    for row in perf
                    if row.get("model") == family + "pair"
                    and row["impl"].startswith("ssik_warp_")
                    and row["impl"].endswith("_" + device)
                    and row.get("scope") == "paired_streaming_control"
                ),
                None,
            )
            if options.smoke or pair is None or pair["repeats"] < 10000:
                status = "UNVALIDATED (requires 10,000 changing-target paired calls)"
            else:
                status = f"{'PASS' if pair['p99_ms'] <= 8. and not pair.get('false_successes', 0) else 'FAIL'} (p99 {pair['p99_ms']:.4f} ms; {pair.get('failure_count', 'unknown')} reported failures)"
            notes.append(
                f"Paired continuous-control latency gate {family}pair/{device}: {status}."
            )
            if options.robot_e2e:
                robot = next(
                    (
                        row
                        for row in perf
                        if row.get("model") == family + "pair"
                        and row["impl"].startswith("ssik_warp_")
                        and row["impl"].endswith("_" + device)
                        and row.get("scope") == "robot_paired_streaming_control"
                    ),
                    None,
                )
                if options.smoke or any(
                    row is None or row["repeats"] < 10000 for row in (robot, pair)
                ):
                    status = "UNVALIDATED (requires matching 10,000 paired Robot and standalone calls)"
                elif any("validated_success_rate" not in row for row in (robot, pair)):
                    status = "UNVALIDATED (missing paired validated coverage)"
                else:
                    passed = robot["p99_ms"] <= 8.0 and not any(
                        robot.get(key, 0)
                        for key in (
                            "false_successes",
                            "step_violations",
                            "failed_hold_violations",
                        )
                    )
                    passed = (
                        passed
                        and robot["validated_success_rate"]
                        >= pair["validated_success_rate"]
                    )
                    status = f"{'PASS' if passed else 'FAIL'} (p99 {robot['p99_ms']:.4f} ms; coverage {robot['validated_success_rate']:.4%}, standalone {pair['validated_success_rate']:.4%})"
                notes.append(
                    f"Paired Robot end-to-end control gate {family}pair/{device}: {status}."
                )
    return notes


def benchmark_ssik_solver(
    options: SSIKBenchmarkOptions | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return rows for the unified report, plus explicit acceptance caveats."""
    logger = logging.getLogger("ssik")
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        return _benchmark_ssik_solver(options)
    finally:
        logger.setLevel(previous_level)


def _benchmark_ssik_solver(
    options: SSIKBenchmarkOptions | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    import warp as wp

    options = options or SSIKBenchmarkOptions()
    if options.geometry_mode not in (None, "strict", "near_parallel", "rr"):
        raise ValueError("SSIK geometry mode must be strict, near_parallel or rr")
    if not math.isfinite(options.axis_alignment_tolerance) or not (
        0 < options.axis_alignment_tolerance <= 1e-3
    ):
        raise ValueError("SSIK axis alignment tolerance must be in (0, 1e-3]")
    if options.srs_search not in (None, "uniform", "seed_local"):
        raise ValueError("SSIK SRS search must be uniform or seed_local")
    if options.rr_search not in (None, "fixed", "seed_local"):
        raise ValueError("SSIK RR search must be fixed or seed_local")
    if options.rr_search == "seed_local" and options.geometry_mode not in (None, "rr"):
        raise ValueError(
            "--ssik-rr-search seed_local requires rr geometry or --ssik-cuda-rr"
        )
    if not options.skip_reference and options.timing_only:
        raise ValueError(
            "--ssik-reference requires correctness measurements; omit --ssik-timing-only"
        )
    if options.reference_artifacts and options.skip_reference:
        raise ValueError(
            "A reference artifact requires explicit --ssik-reference correctness alignment"
        )
    if options.cuda_rr and (
        "flexiv_rizon4s" not in options.models or "cuda" not in options.devices
    ):
        raise ValueError(
            "--ssik-cuda-rr requires --ssik-models flexiv_rizon4s and --ssik-devices cuda; "
            "only that prepared actual model is supported"
        )
    reference_artifacts = _parse_reference_artifacts(options.reference_artifacts)
    if options.cpu_threads < 1 or any(
        name not in MODEL_NAMES for name in options.models
    ):
        raise ValueError("Positive CPU thread count and known model names are required")
    if any(
        variant not in ("original", "matched") for variant in options.baseline_variants
    ):
        raise ValueError("Unknown Pytorch baseline variant")
    if any(
        scenario not in ("wide", "boundary", "singular", "unreachable", "trajectory")
        for scenario in options.scenarios
    ):
        raise ValueError("Unknown SSIK benchmark scenario")
    if "resident" not in options.timing_scopes or any(
        scope not in ("resident", "transfer", "sustained")
        for scope in options.timing_scopes
    ):
        raise ValueError(
            "Timing scopes require resident and may include transfer/sustained"
        )
    if options.quality_only and options.timing_only:
        raise ValueError("quality_only and timing_only are mutually exclusive")
    if options.robot_e2e and options.quality_only:
        raise ValueError("--ssik-robot-e2e requires timing measurements")
    if options.batch_sizes is not None and any(
        count not in (1, 8, 100, 1000, 10000) for count in options.batch_sizes
    ):
        raise ValueError("Unsupported SSIK benchmark batch size")
    if (
        options.smoke
        and options.batch_sizes is not None
        and any(count > 32 for count in options.batch_sizes)
    ):
        raise ValueError("Smoke fixtures contain only 32 targets")
    torch.set_num_threads(options.cpu_threads)
    wp.init()
    directory = options.artifact_dir or Path("outputs/benchmarks") / (
        "ssik_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    directory.mkdir(parents=True, exist_ok=True)
    versions = _versions()
    versions["ssik_algorithm_options"] = {
        "geometry_mode": options.geometry_mode,
        "axis_alignment_tolerance": options.axis_alignment_tolerance,
        "srs_search": options.srs_search,
        "rr_search": options.rr_search,
        "robot_e2e": options.robot_e2e,
        "cuda_rr": options.cuda_rr,
    }
    versions["resolved_solver_options"] = {}
    versions["reference_artifacts"] = {
        name: {
            "path": str(path),
            "source_sha256": (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None
            ),
            "manifest_sha256": (
                hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest()
                if path.with_suffix(".json").is_file()
                else None
            ),
        }
        for name, path in reference_artifacts.items()
    }
    (directory / "runtime.json").write_text(json.dumps(versions, indent=2) + "\n")
    perf: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    notes = [
        f"SSIK artifacts: {directory.resolve()}. Runtime: {json.dumps(versions, sort_keys=True)}.",
        f"Selected timing scopes: {','.join(options.timing_scopes)}; unselected transfer/sustained scopes are UNMEASURED.",
        "SSIK inputs are generated once on CPU using raw float64 URDF/PK FK; timed solver inputs/outputs are float32. Every reported success is independently rechecked at 1e-5 m/rad and against joint limits. Errors describe accepted rows only; no-success rows have zero mean errors and success_rate=0.",
        "Pytorch_original is explicitly reconstructed with original defaults (30 samples, 500 iterations, 5e-4 internal convergence tolerances), independent of robot default changes; acceptance is still rechecked at 1e-5, which can reduce its measured success rate. Pytorch_matched changes only internal convergence tolerances to 1e-5 for a quality-matched comparison. Default-switch throughput gates compare the original baseline.",
        "Resident fixed-workload, transfer-inclusive and initialization timings have distinct scopes. Warp N=1 also measures a streaming_control certificate: changing trajectory frames, preceding result as the next seed, path-boundary resets only, twenty excluded warmups and a 0.05 rad hard step bound. The 5 ms/8 ms control gates require these streaming scopes; repeated fixed targets are diagnostics. Public batch trajectory quality uses each complete 1,000-frame path with its fixed initial seed. Robot-interface end-to-end latency and branch equivalence are validated separately.",
        "Torch peak increments and Warp pool counters are distinct. SSIK candidate/numerical workspaces are Torch-owned zero-copy Warp views. Attributed tensor scratch includes retained workspace and peak transient Torch allocations minus final outputs (CPU uses explicit target/seed/limit conversion sizes); caller inputs and model constants are excluded. These warmed counters cannot capture CUDA driver local-stack reservation: a small attributed tensor workspace does not certify the complete 128 MiB GPU scratch gate, which requires fresh-process driver/NVML attribution. Warp pool high-water is process-global, not an attributable peak. Other backends retain an unvalidated scratch status.",
        "Cartesian lines/circles are small local paths; unlike joint-space trajectories their global reachability is not asserted. Seven-axis finite sampling cannot establish complete reachability.",
        "SSIK leaderboard entries retain model identity, weight standard reachable/trajectory success by attempted count, and exclude deliberately unreachable controls and the separate streaming-control certificate. Timing-only algorithms remain listed as UNMEASURED rather than receiving fabricated success scores. Rankings across different robot models do not establish unconditional algorithm superiority.",
        "Official SSIK reference is opt-in correctness alignment only; its initialization/first-call/steady-state timing is excluded. It is not a core runtime dependency. The documented reference rescue policy is unchanged.",
        "The upstream ssik logger is temporarily set to WARNING during this benchmark and restored afterward; candidate trace logging is excluded from the measured reference scope.",
        "Explicit near_parallel mode always evaluates eight analytical surrogate slots and one original-seed initialization, with bounded original-model polishing on the selected device. These nine candidates are not a pure closed-form solution or complete enumeration of general 6R roots. Fixed Warp RR evaluates16 prepared joint-lock charts (384 slots). RR seed_local reevaluates one parameterized coefficient system at16 global and9 local lock angles (600 slots); two local angles are scalar tangent predictions, while every returned joint solution remains an algebraic RR root. Both RR modes use Warp on the selected CPU/CUDA device without an official SSIK, SciPy QZ or numerical IK fallback.",
    ]
    raw_timings: dict[str, list[float]] = {}
    solvers_for_pairs: dict[
        tuple[str, str], tuple[Any, torch.Tensor, torch.Tensor, _Model]
    ] = {}
    for model_name in options.models:
        model = _load_model(model_name)
        from embodichain.compute.kinematics._ssik_model import _extract_model

        selected = _resolved_options(model, options)
        geometry = _extract_model(
            model.chain,
            geometry_mode=(
                "strict" if selected.geometry_mode == "rr" else selected.geometry_mode
            ),
            axis_alignment_tolerance=selected.axis_alignment_tolerance,
        )
        versions["resolved_solver_options"][model_name] = {
            "geometry_mode": selected.geometry_mode,
            "srs_search": selected.srs_search,
            "rr_search": selected.rr_search,
            "actual_family": geometry.family,
        }
        (directory / "runtime.json").write_text(json.dumps(versions, indent=2) + "\n")
        notes.append(
            f"Geometry {model_name}: {geometry.family}; actual-chain fingerprint {geometry.fingerprint}."
        )
        workloads = _build_workloads(model, options, directory)
        for device_name in options.devices:
            if device_name == "cuda" and not torch.cuda.is_available():
                notes.append(f"Skipped {model_name}/cuda: CUDA unavailable.")
                continue
            device = torch.device(device_name)
            backends = ([] if options.skip_warp else ["warp"]) + (
                []
                if options.skip_baseline
                else [f"pytorch_{variant}" for variant in options.baseline_variants]
            )
            if (
                options.cuda_rr
                and model_name == "flexiv_rizon4s"
                and device.type == "cuda"
            ):
                backends.append("cuda_rr")
            if not options.skip_reference and device.type == "cpu":
                backends.append("reference")
            for backend in backends:
                if (
                    backend == "warp"
                    and geometry.family == "unsupported"
                    and selected.geometry_mode != "rr"
                ):
                    notes.append(
                        f"Skipped {model_name}/warp/{device.type}: actual-chain geometry is unsupported; Warp requires three-parallel 6R or strict SRS 7R. No numerical fallback or acceptance result is substituted."
                    )
                    continue
                artifact = reference_artifacts.get(model_name)
                if (
                    backend == "reference"
                    and artifact is not None
                    and (
                        not artifact.is_file()
                        or not artifact.with_suffix(".json").is_file()
                    )
                ):
                    notes.append(
                        f"Skipped {model_name}/reference: missing artifact source or adjacent manifest at {artifact}. Explicitly build a matching offline artifact, then pass --ssik-reference-artifacts {model_name}=PATH."
                    )
                    continue
                implementation_options = (
                    replace(selected, rr_search=options.rr_search or "fixed")
                    if backend == "cuda_rr"
                    else selected
                )
                impl = _implementation_name(
                    backend, device.type, implementation_options
                )
                common = {
                    "model": model_name,
                    "impl": impl,
                    "component": "ssik_ik",
                    "backend": backend,
                    "geometry_mode": (
                        "rr"
                        if backend == "cuda_rr"
                        else selected.geometry_mode if backend == "warp" else None
                    ),
                    "srs_search": selected.srs_search if backend == "warp" else None,
                    "rr_search": (
                        implementation_options.rr_search
                        if backend in ("warp", "cuda_rr")
                        else None
                    ),
                }
                print(
                    f"SSIK {model_name}/{backend}/{device.type}: building", flush=True
                )
                started = time.perf_counter()
                try:
                    solver = _make_solver(model, backend, device, artifact, options)
                except (ImportError, ModuleNotFoundError) as error:
                    if backend != "reference":
                        raise
                    notes.append(
                        f"Skipped {model_name}/reference: optional dependency unavailable ({error})."
                    )
                    continue
                except ValueError as error:
                    if (
                        backend != "reference"
                        or artifact is not None
                        or "requires an explicitly built reference_artifact"
                        not in str(error)
                    ):
                        raise
                    notes.append(
                        f"Skipped {model_name}/reference: {error}. Build offline with python -m embodichain.lab.sim.motion.solvers._ssik_reference --urdf {model.cfg.fpath} --root {model.manifest['root']} --end {model.manifest['end']} --output MODEL.py; then pass --ssik-reference-artifacts {model_name}=MODEL.py. No derivation runs during this benchmark."
                    )
                    continue
                if backend != "reference":
                    _sync(device)
                    perf.append(
                        {
                            **common,
                            "scope": "initialization",
                            "cost_time_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
                    first_target, first_seed = (
                        value[:1].to(device, dtype=torch.float32)
                        for value in workloads["wide"]
                    )
                    torch.manual_seed(options.seed)
                    _sync(device)
                    started = time.perf_counter()
                    first_output = _nearest(solver, first_target, first_seed)
                    _sync(device)
                    perf.append(
                        {
                            **common,
                            "scope": "first_call_including_jit",
                            "sample_size": 1,
                            "cost_time_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
                    del first_output
                for scenario, (target, seed) in (
                    [] if options.timing_only else workloads.items()
                ):
                    if scenario not in options.scenarios:
                        continue
                    output_path = (
                        directory / f"{model_name}_{impl}_{scenario}_outputs.npz"
                    )
                    print(f"  quality {scenario} n={target.numel() // 16}", flush=True)
                    quality = (
                        _evaluate_trajectories(
                            solver, model, target, seed, device, output_path
                        )
                        if scenario == "trajectory"
                        else _evaluate_workload(
                            solver, model, target, seed, device, output_path
                        )
                    )
                    metrics.append({**common, "scenario": scenario, **quality})
                    (directory / "measurements.partial.json").write_text(
                        json.dumps(
                            {"performance": perf, "quality": metrics, "notes": notes},
                            indent=2,
                        )
                        + "\n"
                    )
                counts = (
                    ()
                    if options.quality_only or backend == "reference"
                    else (
                        options.batch_sizes
                        or ((1, 8) if options.smoke else (1, 8, 100, 1000, 10000))
                    )
                )
                for count in counts:
                    target_cpu, seed_cpu = (
                        value[:count].float() for value in workloads["wide"]
                    )
                    target, seed = target_cpu.to(device), seed_cpu.to(device)
                    repeats = 3 if options.smoke else 10000 if count <= 8 else 30
                    torch.manual_seed(options.seed)
                    operation = lambda: _nearest(solver, target, seed)
                    measured, samples = _measure(operation, device, repeats)
                    memory = _memory(operation, device, solver)
                    resident_median_ms = measured["cost_time_ms"]
                    key = f"{model_name}/{impl}/{count}/resident"
                    raw_timings[key] = samples
                    perf.append(
                        {
                            **common,
                            "sample_size": count,
                            "scope": "resident",
                            **measured,
                            **memory,
                            "queries_per_second": count
                            * 1000
                            / measured["cost_time_ms"],
                            "sustained_queries_per_second": (
                                _throughput(
                                    operation, device, count, 3 if options.smoke else 30
                                )
                                if "sustained" in options.timing_scopes
                                else "UNMEASURED"
                            ),
                        }
                    )
                    if count == 1 and backend in ("warp", "cuda_rr"):
                        path_targets, path_starts = workloads["trajectory"]
                        control = _measure_streaming_control(
                            (solver,),
                            (path_targets,),
                            (path_starts,),
                            device,
                            3 if options.smoke else 10000,
                        )
                        control_quality = evaluate_quality(
                            path_targets.flatten(0, 1)[: control.latency["repeats"]],
                            control.qpos[0],
                            control.valid[0],
                            model.fk,
                            model.limits,
                        )
                        perf.append(
                            {
                                **common,
                                "sample_size": 1,
                                "scope": "streaming_control",
                                **control.latency,
                                "false_successes": control_quality["false_successes"],
                                "validated_success_rate": control_quality[
                                    "success_rate"
                                ],
                            }
                        )
                        metrics.append(
                            {
                                **common,
                                "scenario": "streaming_control",
                                **control_quality,
                            }
                        )
                        raw_timings[f"{model_name}/{impl}/streaming_control"] = (
                            control.samples_ms
                        )
                        np.savez_compressed(
                            directory
                            / f"{model_name}_{impl}_streaming_control_outputs.npz",
                            qpos=control.qpos[0].numpy(),
                            reported=control.valid[0].numpy(),
                        )
                        solvers_for_pairs[model_name, impl] = (
                            solver,
                            path_targets,
                            path_starts,
                            model,
                        )
                    if device.type == "cuda" and "transfer" in options.timing_scopes:

                        def transfer_call() -> tuple[torch.Tensor, torch.Tensor]:
                            valid, joints = _nearest(
                                solver, target_cpu.to(device), seed_cpu.to(device)
                            )
                            return valid.cpu(), joints.cpu()

                        measured, samples = _measure(transfer_call, device, repeats)
                        raw_timings[f"{model_name}/{impl}/{count}/transfer"] = samples
                        perf.append(
                            {
                                **common,
                                "sample_size": count,
                                "scope": "host_transfer_inclusive",
                                **measured,
                            }
                        )
                    (directory / "measurements.partial.json").write_text(
                        json.dumps(
                            {"performance": perf, "quality": metrics, "notes": notes},
                            indent=2,
                        )
                        + "\n"
                    )
                    (directory / "timings.partial.json").write_text(
                        json.dumps(raw_timings) + "\n"
                    )
                    print(
                        f"  n={count}: resident median={resident_median_ms:.3f} ms",
                        flush=True,
                    )
                if options.robot_e2e and backend in ("warp", "cuda_rr"):
                    print(f"  real Robot end-to-end: {model_name}/{impl}", flush=True)
                    robot_result = _measure_robot_interface(
                        model, solver, device, directory, options
                    )
                    if robot_result["status"] == "unavailable":
                        notes.append(
                            f"Robot end-to-end {model_name}/{impl}: UNMEASURED "
                            f"(engine setup unavailable: {robot_result['reason']})."
                        )
                    else:
                        robot_quality = robot_result["quality"]
                        perf.append(
                            {
                                **common,
                                "scope": "robot_streaming_control",
                                "sample_size": 1,
                                **robot_result["latency"],
                                "false_successes": robot_quality["false_successes"],
                                "validated_success_rate": robot_quality["success_rate"],
                                "step_violations": robot_quality["step_violations"],
                                "failed_hold_violations": robot_quality[
                                    "failed_hold_violations"
                                ],
                            }
                        )
                        metrics.append(
                            {
                                **common,
                                "scenario": "robot_streaming_control",
                                **robot_quality,
                            }
                        )
    for model_name, impl in list(solvers_for_pairs):
        if not model_name.endswith("left"):
            continue
        right_name = model_name.removesuffix("left") + "right"
        if (right_name, impl) not in solvers_for_pairs:
            continue
        first, target, seed, left_model = solvers_for_pairs[model_name, impl]
        second, right_target, right_seed, right_model = solvers_for_pairs[
            right_name, impl
        ]
        device = first.device
        control = _measure_streaming_control(
            (first, second),
            (target, right_target),
            (seed, right_seed),
            device,
            3 if options.smoke else 10000,
        )
        pair_quality = [
            evaluate_quality(
                goals.flatten(0, 1)[: control.latency["repeats"]],
                qpos,
                valid,
                arm.fk,
                arm.limits,
            )
            for goals, qpos, valid, arm in zip(
                (target, right_target),
                control.qpos,
                control.valid,
                (left_model, right_model),
            )
        ]
        raw_timings[f"{model_name}/{impl}/paired_streaming_control"] = (
            control.samples_ms
        )
        perf.append(
            {
                "model": model_name.removesuffix("left") + "pair",
                "impl": impl,
                "component": "ssik_ik",
                "sample_size": 2,
                "scope": "paired_streaming_control",
                **control.latency,
                "false_successes": sum(row["false_successes"] for row in pair_quality),
                "validated_success_rate": (
                    control.latency["both_reported_success_rate"]
                    if not any(row["false_successes"] for row in pair_quality)
                    else 0.0
                ),
                "latency_gate": (
                    "unvalidated: smoke"
                    if options.smoke
                    else (
                        "PASS"
                        if control.latency["p99_ms"] <= 8
                        and not any(row["false_successes"] for row in pair_quality)
                        else "FAIL"
                    )
                ),
            }
        )
        np.savez_compressed(
            directory
            / f"{model_name.removesuffix('left')}pair_{impl}_streaming_control_outputs.npz",
            left_qpos=control.qpos[0].numpy(),
            right_qpos=control.qpos[1].numpy(),
            left_reported=control.valid[0].numpy(),
            right_reported=control.valid[1].numpy(),
        )
        if options.robot_e2e:
            robot_result = _measure_robot_interface(
                left_model,
                first,
                device,
                directory,
                options,
                partner=(right_model, second),
            )
            if robot_result["status"] == "unavailable":
                notes.append(
                    f"Paired Robot end-to-end {model_name}/{impl}: UNMEASURED "
                    f"(engine setup unavailable: {robot_result['reason']})."
                )
            else:
                quality = robot_result["quality"]
                perf.append(
                    {
                        "model": model_name.removesuffix("left") + "pair",
                        "impl": impl,
                        "component": "ssik_ik",
                        "sample_size": 2,
                        "scope": "robot_paired_streaming_control",
                        **robot_result["latency"],
                        "validated_success_rate": quality["success_rate"],
                        **{
                            key: quality[key]
                            for key in (
                                "false_successes",
                                "step_violations",
                                "failed_hold_violations",
                            )
                        },
                    }
                )
    notes.extend(_gate_notes(perf, metrics, options))
    (directory / "timings.json").write_text(json.dumps(raw_timings) + "\n")
    (directory / "measurements.json").write_text(
        json.dumps({"performance": perf, "quality": metrics, "notes": notes}, indent=2)
        + "\n"
    )
    return perf, metrics, notes
