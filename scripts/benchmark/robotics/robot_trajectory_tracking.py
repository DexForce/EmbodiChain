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

"""Benchmark default robot joint-trajectory tracking and loaded asset physics.

The benchmark keeps each robot's default drive, body attributes, joint limits,
fixed-base setting, world gravity, and loader-owned articulation-gravity state.
It raises the base by 0.5 m only to remove ground contacts from a free-space
controller check, then tracks a bounded sinusoidal joint reference with 60 Hz
commands and 240 Hz physics. The default command mode sends qpos targets only:
it never writes a velocity target. A position-plus-velocity mode remains an
explicit diagnostic comparison.

Each candidate runs in a dedicated child process.  This is intentional: the
current native simulator owns process-global resources and its documented
cleanup path may terminate the process.  The parent aggregates child results
into exactly one Markdown report.

Run: python scripts/benchmark/robotics/robot_trajectory_tracking.py
Run: python scripts/benchmark/robotics/robot_trajectory_tracking.py --robots franka_panda ur10e
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

_REPO_ROOT = Path(__file__).resolve().parents[3]
# A development virtual environment can have a different worktree installed in
# editable mode. Executing this file directly must still benchmark the source
# tree that contains this script.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot


__all__ = ["add_benchmark_args", "run_all_benchmarks"]


SUPPORTED_ROBOTS = (
    "franka_panda",
    "ur10e",
    "cobotmagic",
    "dexforce_w1",
)
DEFAULT_ROBOTS = SUPPORTED_ROBOTS
_WORKER_RESULT_PREFIX = "ROBOT_TRAJECTORY_TRACKING_RESULT:"
_BASE_HEIGHT_M = 0.5
_MIN_TRAJECTORY_AMPLITUDE = 1.0e-4
_TRACKING_RMSE_LIMIT = 0.10
_TRACKING_P95_LIMIT = 0.20
_STATIC_HOLD_RMSE_LIMIT = 0.10
_QPOS_ONLY_TARGET_QVEL_TOLERANCE = 1.0e-6


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add trajectory-tracking benchmark arguments to ``parser``.

    Args:
        parser: Parser to receive the benchmark options.
    """
    parser.add_argument(
        "--robots",
        nargs="+",
        choices=SUPPORTED_ROBOTS,
        default=list(DEFAULT_ROBOTS),
        help="Robot presets to evaluate. Defaults to all maintained presets.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Physics device. CPU is the reproducible default.",
    )
    parser.add_argument(
        "--renderer",
        default="hybrid",
        choices=("hybrid", "fast-rt", "rt"),
        help="Headless renderer required by the simulator.",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=1.0 / 240.0,
        help="Physics integration step in seconds. Defaults to 1/240.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=60.0,
        help="Joint-reference command rate in Hz. Defaults to 60.",
    )
    parser.add_argument(
        "--command-mode",
        choices=("position", "position_velocity"),
        default="position",
        help=(
            "Reference fields sent to the default drive. "
            "position is qpos-only and never writes a qvel target; "
            "position_velocity is an explicit full-PD diagnostic."
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Static default-pose hold time before tracking. Defaults to 1.0.",
    )
    parser.add_argument(
        "--trajectory-seconds",
        type=float,
        default=3.0,
        help="Measured trajectory duration in seconds. Defaults to 3.0.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.08,
        help="Maximum per-joint sine amplitude in native joint units.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=0.5,
        help="Sine-reference frequency in Hz. Defaults to 0.5.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-robot child-process timeout. Defaults to 180 seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Directory for the single Markdown report.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--robot",
        choices=SUPPORTED_ROBOTS,
        help=argparse.SUPPRESS,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark command-line arguments.

    Args:
        argv: Optional arguments excluding the executable name.

    Returns:
        Parsed benchmark arguments.
    """
    parser = argparse.ArgumentParser(
        description="Measure default robot trajectory tracking and asset physics."
    )
    add_benchmark_args(parser)
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return args


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate numeric arguments that affect simulation correctness.

    Args:
        args: Parsed command-line options.
        parser: Parser used to surface a conventional command-line error.
    """
    positive_fields = (
        "physics_dt",
        "control_hz",
        "settle_seconds",
        "trajectory_seconds",
        "amplitude",
        "frequency_hz",
        "timeout_seconds",
    )
    for field_name in positive_fields:
        if getattr(args, field_name) <= 0.0:
            parser.error(f"--{field_name.replace('_', '-')} must be positive.")
    if args.worker and args.robot is None:
        parser.error("--worker requires --robot.")


def _make_robot_cfg(robot_name: str) -> Any:
    """Build one maintained robot preset without changing its physics defaults.

    Args:
        robot_name: Name from :data:`SUPPORTED_ROBOTS`.

    Returns:
        Robot configuration with only ``init_pos`` overridden for free space.
    """
    from embodichain.lab.sim.robots import (
        CobotMagicCfg,
        DexforceW1Cfg,
        FrankaPandaCfg,
        URRobotCfg,
    )

    common_override = {"init_pos": [0.0, 0.0, _BASE_HEIGHT_M]}
    if robot_name == "franka_panda":
        return FrankaPandaCfg.from_dict(common_override)
    if robot_name == "ur10e":
        return URRobotCfg.from_dict({**common_override, "robot_type": "ur10e"})
    if robot_name == "cobotmagic":
        return CobotMagicCfg.from_dict(common_override)
    if robot_name == "dexforce_w1":
        return DexforceW1Cfg.from_dict(common_override)
    raise ValueError(f"Unsupported robot preset: {robot_name!r}")


def _gpu_memory_used_mb() -> float | None:
    """Return the process-visible GPU's used VRAM, or ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _memory_snapshot(process: psutil.Process) -> dict[str, float | None]:
    """Read CPU RSS and global GPU VRAM for one benchmark worker.

    Args:
        process: Current worker process.

    Returns:
        Memory values in MiB. GPU usage is ``None`` when unavailable.
    """
    return {
        "cpu_rss_mb": process.memory_info().rss / 1024**2,
        "gpu_used_mb": _gpu_memory_used_mb(),
    }


def _update_memory_peaks(
    process: psutil.Process,
    peak_cpu_rss_mb: float,
    peak_gpu_used_mb: float | None,
    *,
    sample_gpu: bool,
) -> tuple[float, float | None]:
    """Update RSS every control tick and GPU VRAM at a low sampling rate.

    Args:
        process: Current worker process.
        peak_cpu_rss_mb: Current peak CPU RSS in MiB.
        peak_gpu_used_mb: Current peak device-wide GPU usage in MiB.
        sample_gpu: Whether this tick should invoke ``nvidia-smi``.

    Returns:
        Updated CPU and GPU peak values.
    """
    peak_cpu_rss_mb = max(peak_cpu_rss_mb, process.memory_info().rss / 1024**2)
    if sample_gpu:
        gpu_used_mb = _gpu_memory_used_mb()
        if gpu_used_mb is not None:
            peak_gpu_used_mb = max(peak_gpu_used_mb or 0.0, gpu_used_mb)
    return peak_cpu_rss_mb, peak_gpu_used_mb


def _source_asset_audit(cfg: Any) -> dict[str, float | int]:
    """Inspect declared inertial properties in the source URDF components.

    This supplements runtime mass readings: a simulator can synthesize a mass
    for a link whose source URDF lacks an inertial block, so the two views are
    intentionally reported separately.

    Args:
        cfg: Robot configuration with a populated ``urdf_cfg``.

    Returns:
        Aggregated source-asset mass and inertial coverage metrics.
    """
    components = getattr(getattr(cfg, "urdf_cfg", None), "components", {})
    component_values = (
        components.values() if isinstance(components, dict) else components or []
    )
    summary: dict[str, float | int] = {
        "source_component_count": 0,
        "source_link_count": 0,
        "source_declared_mass_kg": 0.0,
        "source_inertial_link_count": 0,
        "source_missing_inertial_count": 0,
        "source_invalid_mass_count": 0,
        "source_nonpositive_inertia_count": 0,
        "source_parse_error_count": 0,
    }
    for component in component_values:
        urdf_path = component.get("urdf_path") if isinstance(component, dict) else None
        if not urdf_path:
            summary["source_parse_error_count"] += 1
            continue
        summary["source_component_count"] += 1
        try:
            root = ET.parse(urdf_path).getroot()
        except (ET.ParseError, OSError):
            summary["source_parse_error_count"] += 1
            continue
        for link in root.findall("link"):
            summary["source_link_count"] += 1
            inertial = link.find("inertial")
            if inertial is None:
                summary["source_missing_inertial_count"] += 1
                continue
            summary["source_inertial_link_count"] += 1
            mass_element = inertial.find("mass")
            try:
                mass = float(mass_element.attrib["value"])
            except (AttributeError, KeyError, TypeError, ValueError):
                summary["source_invalid_mass_count"] += 1
                continue
            if not math.isfinite(mass) or mass <= 0.0:
                summary["source_invalid_mass_count"] += 1
            else:
                summary["source_declared_mass_kg"] += mass
            inertia_element = inertial.find("inertia")
            try:
                inertia_values = inertia_element.attrib
                inertia_matrix = np.array(
                    [
                        [
                            float(inertia_values["ixx"]),
                            float(inertia_values["ixy"]),
                            float(inertia_values["ixz"]),
                        ],
                        [
                            float(inertia_values["ixy"]),
                            float(inertia_values["iyy"]),
                            float(inertia_values["iyz"]),
                        ],
                        [
                            float(inertia_values["ixz"]),
                            float(inertia_values["iyz"]),
                            float(inertia_values["izz"]),
                        ],
                    ]
                )
                eigenvalues = np.linalg.eigvalsh(inertia_matrix)
                if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0.0:
                    summary["source_nonpositive_inertia_count"] += 1
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                summary["source_nonpositive_inertia_count"] += 1
    return summary


def _runtime_asset_audit(robot: Robot) -> dict[str, float | int | str]:
    """Read masses and effective drive properties after the robot loads.

    Args:
        robot: Loaded simulation robot.

    Returns:
        Runtime mass and active-drive summary.
    """
    active_joint_ids = list(robot.active_joint_ids)
    masses = robot.get_mass().detach().cpu().numpy().reshape(-1)
    stiffness = robot.default_joint_stiffness[0, active_joint_ids].cpu().numpy()
    damping = robot.default_joint_damping[0, active_joint_ids].cpu().numpy()
    max_effort = robot.default_joint_max_effort[0, active_joint_ids].cpu().numpy()
    attrs = robot.cfg.attrs
    return {
        "runtime_link_count": int(masses.size),
        "runtime_total_mass_kg": float(np.sum(masses)),
        "runtime_min_link_mass_kg": float(np.min(masses)),
        "runtime_nonpositive_mass_count": int(np.count_nonzero(masses <= 0.0)),
        "active_joint_count": len(active_joint_ids),
        "drive_type": str(getattr(robot.cfg.drive_pros, "drive_type", "force")),
        "drive_stiffness_min": float(np.min(stiffness)),
        "drive_stiffness_max": float(np.max(stiffness)),
        "drive_damping_min": float(np.min(damping)),
        "drive_damping_max": float(np.max(damping)),
        "drive_max_effort_min": float(np.min(max_effort)),
        "drive_max_effort_max": float(np.max(max_effort)),
        "static_friction": float(attrs.static_friction),
        "dynamic_friction": float(attrs.dynamic_friction),
        "contact_offset_m": float(attrs.contact_offset),
        "linear_damping": float(attrs.linear_damping),
        "angular_damping": float(attrs.angular_damping),
    }


def _asset_audit_status(
    source_audit: dict[str, float | int], runtime_audit: dict[str, float | int | str]
) -> str:
    """Classify physical-asset audit outcomes without hiding warnings.

    Args:
        source_audit: Parsed URDF inertial-property summary.
        runtime_audit: Loaded simulation mass summary.

    Returns:
        ``pass``, ``review``, or ``fail``.
    """
    if (
        float(runtime_audit["runtime_total_mass_kg"]) <= 0.0
        or int(runtime_audit["runtime_nonpositive_mass_count"]) > 0
    ):
        return "fail"
    if (
        int(source_audit["source_parse_error_count"]) > 0
        or int(source_audit["source_missing_inertial_count"]) > 0
        or int(source_audit["source_invalid_mass_count"]) > 0
        or int(source_audit["source_nonpositive_inertia_count"]) > 0
    ):
        return "review"
    return "pass"


def _trajectory_center_and_amplitudes(
    current: Any,
    limits: Any,
    maximum_amplitude: float,
) -> tuple[Any, Any]:
    """Return an interior reference center and safe per-joint amplitudes.

    A default ready pose can intentionally sit near a joint limit.  Instead of
    treating its tiny remaining margin as an inaccurate controller result, the
    benchmark moves the reference center just inside the limit before measuring
    the sinusoid.  The static hold metric remains measured at the unmodified
    default pose.

    Args:
        current: Settled active-joint positions with shape ``(N,)``.
        limits: Active-joint limits with shape ``(N, 2)``.
        maximum_amplitude: Requested maximum native-unit amplitude.

    Returns:
        Reference center and amplitudes that keep every target inside its limit.
    """
    lower = limits[:, 0]
    upper = limits[:, 1]
    amplitudes = np.minimum(maximum_amplitude, 0.20 * (upper - lower))
    center = np.clip(current, lower + amplitudes, upper - amplitudes)
    return center, amplitudes


def _tracking_status(
    normalized_rmse: float,
    normalized_p95: float,
    static_normalized_rmse: float,
) -> str:
    """Return an interpretable tracking judgment for the default controller.

    Args:
        normalized_rmse: Dynamic RMS error divided by per-joint amplitude.
        normalized_p95: Dynamic P95 absolute error divided by amplitude.
        static_normalized_rmse: Static-hold RMS error divided by amplitude.

    Returns:
        ``pass`` when all high-accuracy gates are met, otherwise ``review``.
    """
    if (
        normalized_rmse <= _TRACKING_RMSE_LIMIT
        and normalized_p95 <= _TRACKING_P95_LIMIT
        and static_normalized_rmse <= _STATIC_HOLD_RMSE_LIMIT
    ):
        return "pass"
    return "review"


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Run one robot's isolated simulation and collect metrics.

    Args:
        args: Parsed worker options. ``args.robot`` must be set.

    Returns:
        Serializable benchmark result for one robot.
    """
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    assert args.robot is not None
    process = psutil.Process(os.getpid())
    memory_before = _memory_snapshot(process)
    peak_cpu_rss_mb = float(memory_before["cpu_rss_mb"])
    peak_gpu_used_mb = memory_before["gpu_used_mb"]
    wall_start = time.perf_counter()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device=args.device,
            physics_dt=args.physics_dt,
            num_envs=1,
            render_cfg=RenderCfg(renderer=args.renderer),
        )
    )
    cfg = _make_robot_cfg(args.robot)
    robot = sim.add_robot(cfg)
    if robot is None:
        raise RuntimeError("SimulationManager.add_robot returned None.")
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    source_audit = _source_asset_audit(cfg)
    runtime_audit = _runtime_asset_audit(robot)
    source_mass = float(source_audit["source_declared_mass_kg"])
    loaded_mass = float(runtime_audit["runtime_total_mass_kg"])
    runtime_source_mass_delta_kg = loaded_mass - source_mass
    runtime_source_mass_delta_ratio = (
        abs(runtime_source_mass_delta_kg) / source_mass if source_mass > 0.0 else None
    )
    active_joint_ids = list(robot.active_joint_ids)
    if not active_joint_ids:
        raise RuntimeError("Robot has no non-mimic active joints to benchmark.")

    physics_steps_per_command = max(
        1, int(round(1.0 / (args.physics_dt * args.control_hz)))
    )
    control_dt = physics_steps_per_command * args.physics_dt
    gpu_sample_interval = max(1, int(round(1.0 / control_dt)))
    hold_target = torch.as_tensor(
        robot.cfg.init_qpos, dtype=torch.float32, device=sim.device
    ).reshape(1, -1)
    hold_velocity = torch.zeros_like(hold_target)
    velocity_target_write_count = 0
    robot.set_qpos(hold_target, target=False)
    robot.set_qpos(hold_target, target=True)
    robot.set_qvel(hold_velocity, target=False)
    if args.command_mode == "position_velocity":
        robot.set_qvel(hold_velocity, target=True)
        velocity_target_write_count += 1
    settle_steps = int(math.ceil(args.settle_seconds / control_dt))
    for settle_step_index in range(settle_steps):
        sim.update(step=physics_steps_per_command)
        peak_cpu_rss_mb, peak_gpu_used_mb = _update_memory_peaks(
            process,
            peak_cpu_rss_mb,
            peak_gpu_used_mb,
            sample_gpu=settle_step_index % gpu_sample_interval == 0,
        )

    hold_actual = robot.get_qpos()[0, active_joint_ids].detach().cpu().numpy()
    hold_reference = hold_target[0, active_joint_ids].detach().cpu().numpy()
    limits = robot.get_qpos_limits()[0, active_joint_ids].detach().cpu().numpy()
    tracked_center, amplitudes = _trajectory_center_and_amplitudes(
        hold_actual, limits, args.amplitude
    )
    valid_joint_mask = amplitudes >= _MIN_TRAJECTORY_AMPLITUDE
    if not np.any(valid_joint_mask):
        raise RuntimeError(
            "No active joint has enough limit margin for the trajectory."
        )
    tracked_joint_ids = [
        joint_id
        for joint_id, is_valid in zip(active_joint_ids, valid_joint_mask, strict=True)
        if is_valid
    ]
    tracked_center = tracked_center[valid_joint_mask]
    tracked_amplitudes = amplitudes[valid_joint_mask]
    static_error = hold_actual[valid_joint_mask] - hold_reference[valid_joint_mask]

    robot.set_qpos(
        torch.as_tensor(
            tracked_center, dtype=torch.float32, device=sim.device
        ).unsqueeze(0),
        joint_ids=tracked_joint_ids,
        target=True,
    )
    if args.command_mode == "position_velocity":
        robot.set_qvel(
            torch.zeros(
                (1, len(tracked_joint_ids)), dtype=torch.float32, device=sim.device
            ),
            joint_ids=tracked_joint_ids,
            target=True,
        )
        velocity_target_write_count += 1
    for center_settle_step_index in range(settle_steps):
        sim.update(step=physics_steps_per_command)
        peak_cpu_rss_mb, peak_gpu_used_mb = _update_memory_peaks(
            process,
            peak_cpu_rss_mb,
            peak_gpu_used_mb,
            sample_gpu=center_settle_step_index % gpu_sample_interval == 0,
        )

    measured_steps = int(math.ceil(args.trajectory_seconds / control_dt))
    actual_samples: list[np.ndarray] = []
    reference_samples: list[np.ndarray] = []
    for step_index in range(measured_steps):
        trajectory_time = (step_index + 1) * control_dt
        reference = tracked_center + tracked_amplitudes * np.sin(
            2.0 * np.pi * args.frequency_hz * trajectory_time
        )
        robot.set_qpos(
            torch.as_tensor(
                reference, dtype=torch.float32, device=sim.device
            ).unsqueeze(0),
            joint_ids=tracked_joint_ids,
            target=True,
        )
        if args.command_mode == "position_velocity":
            reference_velocity = (
                tracked_amplitudes
                * (2.0 * np.pi * args.frequency_hz)
                * np.cos(2.0 * np.pi * args.frequency_hz * trajectory_time)
            )
            robot.set_qvel(
                torch.as_tensor(
                    reference_velocity, dtype=torch.float32, device=sim.device
                ).unsqueeze(0),
                joint_ids=tracked_joint_ids,
                target=True,
            )
            velocity_target_write_count += 1
        sim.update(step=physics_steps_per_command)
        actual = robot.get_qpos()[0, tracked_joint_ids].detach().cpu().numpy()
        actual_samples.append(actual)
        reference_samples.append(reference)
        peak_cpu_rss_mb, peak_gpu_used_mb = _update_memory_peaks(
            process,
            peak_cpu_rss_mb,
            peak_gpu_used_mb,
            sample_gpu=step_index % gpu_sample_interval == 0,
        )

    max_abs_velocity_target = float(
        torch.max(torch.abs(robot.get_qvel(target=True)[0, active_joint_ids])).item()
    )
    if args.command_mode == "position":
        if velocity_target_write_count != 0:
            raise AssertionError("qpos-only mode must not write a velocity target.")
        if max_abs_velocity_target > _QPOS_ONLY_TARGET_QVEL_TOLERANCE:
            raise AssertionError("qpos-only mode must retain a zero velocity target.")

    actual_array = np.stack(actual_samples)
    reference_array = np.stack(reference_samples)
    error_array = actual_array - reference_array
    normalized_error_array = error_array / tracked_amplitudes[np.newaxis, :]
    qpos_rmse = float(np.sqrt(np.mean(np.square(error_array))))
    qpos_p95 = float(np.percentile(np.abs(error_array), 95.0))
    qpos_max = float(np.max(np.abs(error_array)))
    normalized_rmse = float(np.sqrt(np.mean(np.square(normalized_error_array))))
    normalized_p95 = float(np.percentile(np.abs(normalized_error_array), 95.0))
    static_rmse = float(np.sqrt(np.mean(np.square(static_error))))
    static_normalized_rmse = float(
        np.sqrt(np.mean(np.square(static_error / tracked_amplitudes)))
    )
    joint_normalized_rmse = np.sqrt(np.mean(np.square(normalized_error_array), axis=0))
    joint_normalized_p95 = np.percentile(np.abs(normalized_error_array), 95.0, axis=0)
    worst_joint_index = int(np.argmax(joint_normalized_rmse))
    worst_joint_id = tracked_joint_ids[worst_joint_index]
    tracking_status = _tracking_status(
        normalized_rmse, normalized_p95, static_normalized_rmse
    )
    asset_status = _asset_audit_status(source_audit, runtime_audit)
    memory_after = _memory_snapshot(process)
    wall_time_seconds = time.perf_counter() - wall_start
    total_sim_steps = (2 * settle_steps + measured_steps) * physics_steps_per_command
    return {
        "robot": args.robot,
        "status": tracking_status,
        "tracking_success_rate": 1.0 if tracking_status == "pass" else 0.0,
        "asset_status": asset_status,
        "error": "",
        "device": args.device,
        "command_mode": args.command_mode,
        "velocity_target_write_count": velocity_target_write_count,
        "max_abs_velocity_target": max_abs_velocity_target,
        "gravity_m_s2": "0.00,0.00,-9.81",
        "physics_dt_ms": args.physics_dt * 1000.0,
        "control_dt_ms": control_dt * 1000.0,
        "wall_time_seconds": wall_time_seconds,
        "sim_steps_per_second": total_sim_steps / wall_time_seconds,
        "cpu_rss_delta_mb": float(memory_after["cpu_rss_mb"])
        - float(memory_before["cpu_rss_mb"]),
        "peak_cpu_rss_mb": peak_cpu_rss_mb,
        "gpu_used_delta_mb": (
            None
            if memory_before["gpu_used_mb"] is None
            or memory_after["gpu_used_mb"] is None
            else float(memory_after["gpu_used_mb"])
            - float(memory_before["gpu_used_mb"])
        ),
        "peak_gpu_used_mb": peak_gpu_used_mb,
        "qpos_rmse": qpos_rmse,
        "qpos_p95": qpos_p95,
        "qpos_max": qpos_max,
        "normalized_rmse": normalized_rmse,
        "normalized_p95": normalized_p95,
        "static_rmse": static_rmse,
        "static_normalized_rmse": static_normalized_rmse,
        "tracked_joint_count": len(tracked_joint_ids),
        "worst_joint": robot.joint_names[worst_joint_id],
        "worst_joint_normalized_rmse": float(joint_normalized_rmse[worst_joint_index]),
        "worst_joint_normalized_p95": float(joint_normalized_p95[worst_joint_index]),
        "worst_joint_stiffness": float(
            robot.default_joint_stiffness[0, worst_joint_id].item()
        ),
        "worst_joint_damping": float(
            robot.default_joint_damping[0, worst_joint_id].item()
        ),
        "worst_joint_max_effort": float(
            robot.default_joint_max_effort[0, worst_joint_id].item()
        ),
        "fixed_base": bool(robot.cfg.fix_base),
        "min_position_iters": int(robot.cfg.min_position_iters),
        "min_velocity_iters": int(robot.cfg.min_velocity_iters),
        "runtime_source_mass_delta_kg": runtime_source_mass_delta_kg,
        "runtime_source_mass_delta_ratio": runtime_source_mass_delta_ratio,
        **source_audit,
        **runtime_audit,
    }


def _error_result(robot_name: str, error: str) -> dict[str, Any]:
    """Build a reportable result when a robot worker cannot produce metrics.

    Args:
        robot_name: Requested robot preset.
        error: Concise diagnostic for the skipped/failed worker.

    Returns:
        Serializable failure result with stable report fields.
    """
    return {
        "robot": robot_name,
        "status": "error",
        "tracking_success_rate": 0.0,
        "asset_status": "error",
        "error": error,
    }


def _emit_worker_result(result: dict[str, Any]) -> None:
    """Write one machine-readable result record before an intentional child exit.

    Args:
        result: Serializable worker result.
    """
    encoded = (
        _WORKER_RESULT_PREFIX
        + json.dumps(result, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    os.write(sys.stdout.fileno(), encoded)


def _worker_main(args: argparse.Namespace) -> None:
    """Run the private child-worker path and terminate without native teardown.

    Args:
        args: Parsed worker arguments.
    """
    assert args.robot is not None
    try:
        result = _run_worker(args)
    except Exception as exc:  # noqa: BLE001 - benchmark failures are data points.
        result = _error_result(args.robot, f"{type(exc).__name__}: {exc}")
    _emit_worker_result(result)
    # DexSim's standalone cleanup path is process-oriented. Avoid running Python
    # finalizers after a native world has been initialized; the parent remains
    # alive to write the one run-level report.
    os._exit(0)


def _worker_command(args: argparse.Namespace, robot_name: str) -> list[str]:
    """Build the isolated child command for one robot candidate.

    Args:
        args: Parent benchmark options.
        robot_name: Robot preset for the child worker.

    Returns:
        Complete subprocess command.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--robot",
        robot_name,
        "--device",
        args.device,
        "--renderer",
        args.renderer,
        "--physics-dt",
        str(args.physics_dt),
        "--control-hz",
        str(args.control_hz),
        "--command-mode",
        args.command_mode,
        "--settle-seconds",
        str(args.settle_seconds),
        "--trajectory-seconds",
        str(args.trajectory_seconds),
        "--amplitude",
        str(args.amplitude),
        "--frequency-hz",
        str(args.frequency_hz),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]


def _result_from_worker_output(
    robot_name: str,
    output: bytes,
    returncode: int,
) -> dict[str, Any]:
    """Extract a child result while tolerating native simulator log output.

    Args:
        robot_name: Worker robot preset.
        output: Combined child stdout/stderr bytes.
        returncode: Child process exit status.

    Returns:
        Parsed worker result or a concise error result.
    """
    prefix = _WORKER_RESULT_PREFIX.encode("utf-8")
    prefix_index = output.rfind(prefix)
    if prefix_index >= 0:
        payload = output[prefix_index + len(prefix) :].split(b"\n", maxsplit=1)[0]
        try:
            result = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            result = None
        if result is not None:
            if returncode != 0 and not result.get("error"):
                result["error"] = f"worker exited with code {returncode}"
                result["status"] = "error"
                result["tracking_success_rate"] = 0.0
            return result
    decoded = output.decode("utf-8", errors="replace")
    diagnostic_lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    diagnostic = (
        diagnostic_lines[-1] if diagnostic_lines else "no worker result emitted"
    )
    return _error_result(
        robot_name,
        f"worker exited with code {returncode}; {diagnostic[:240]}",
    )


def _run_robot_worker(args: argparse.Namespace, robot_name: str) -> dict[str, Any]:
    """Run one robot worker in a bounded subprocess.

    Args:
        args: Parent benchmark options.
        robot_name: Robot preset to benchmark.

    Returns:
        Aggregatable benchmark result for ``robot_name``.
    """
    environment = os.environ.copy()
    environment["EMBODICHAIN_SIM_EXIT_PROCESS"] = "1"
    try:
        completed = subprocess.run(
            _worker_command(args, robot_name),
            cwd=_REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _error_result(
            robot_name, f"worker timed out after {args.timeout_seconds:.0f} seconds"
        )
    return _result_from_worker_output(
        robot_name, completed.stdout, completed.returncode
    )


def _format_number(value: Any, precision: int = 4) -> str:
    """Format optional numeric report values without implying unavailable data.

    Args:
        value: Numeric or unavailable value.
        precision: Decimal precision for finite values.

    Returns:
        Markdown-safe display value.
    """
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f}"


def _markdown_table(rows: list[dict[str, str]]) -> list[str]:
    """Render one compact Markdown table.

    Args:
        rows: Table rows sharing the same columns.

    Returns:
        Markdown lines for one table.
    """
    if not rows:
        return ["No data."]
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            str(row[header]).replace("|", "/").replace("\n", " ") for header in headers
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _performance_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the required Time & Memory table rows.

    Args:
        results: Worker results in requested-robot order.

    Returns:
        Display rows for time and memory metrics.
    """
    return [
        {
            "robot": result["robot"],
            "status": result["status"],
            "wall_s": _format_number(result.get("wall_time_seconds"), 2),
            "sim_steps_per_s": _format_number(result.get("sim_steps_per_second"), 1),
            "cpu_rss_delta_mb": _format_number(result.get("cpu_rss_delta_mb"), 1),
            "peak_cpu_rss_mb": _format_number(result.get("peak_cpu_rss_mb"), 1),
            "gpu_used_delta_mb": _format_number(result.get("gpu_used_delta_mb"), 1),
            "peak_gpu_used_mb": _format_number(result.get("peak_gpu_used_mb"), 1),
        }
        for result in results
    ]


def _metric_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the required Success & Other Metrics table rows.

    Args:
        results: Worker results in requested-robot order.

    Returns:
        Display rows for tracking and physical-asset metrics.
    """
    return [
        {
            "robot": result["robot"],
            "tracking": result["status"],
            "qpos_rmse": _format_number(result.get("qpos_rmse"), 5),
            "qpos_p95": _format_number(result.get("qpos_p95"), 5),
            "qvel_target_writes": str(result.get("velocity_target_write_count", "n/a")),
            "max_abs_qvel_target": _format_number(
                result.get("max_abs_velocity_target"), 5
            ),
            "norm_rmse": _format_number(result.get("normalized_rmse"), 3),
            "norm_p95": _format_number(result.get("normalized_p95"), 3),
            "static_norm_rmse": _format_number(result.get("static_normalized_rmse"), 3),
            "worst_joint": str(result.get("worst_joint", "n/a")),
            "worst_norm_rmse": _format_number(
                result.get("worst_joint_normalized_rmse"), 3
            ),
            "loaded_mass_kg": _format_number(result.get("runtime_total_mass_kg"), 3),
            "source_mass_kg": _format_number(result.get("source_declared_mass_kg"), 3),
            "loaded-source_mass_kg": _format_number(
                result.get("runtime_source_mass_delta_kg"), 3
            ),
            "source_inertial_coverage": (
                "n/a"
                if result.get("source_link_count") is None
                else f"{result.get('source_inertial_link_count', 0)}/{result['source_link_count']}"
            ),
            "asset_audit": result["asset_status"],
        }
        for result in results
    ]


def _leaderboard_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Rank every evaluated robot by tracking success and normalized RMSE.

    Args:
        results: Worker results to rank.

    Returns:
        Display rows sorted by success rate descending.
    """
    ranked = sorted(
        results,
        key=lambda result: (
            -float(result.get("tracking_success_rate", 0.0)),
            float(result.get("normalized_rmse", math.inf)),
            result["robot"],
        ),
    )
    return [
        {
            "rank": str(index),
            "algorithm": result["robot"],
            "tracking_success_rate": f"{100.0 * result.get('tracking_success_rate', 0.0):.0f}%",
            "normalized_rmse": _format_number(result.get("normalized_rmse"), 3),
            "qpos_rmse": _format_number(result.get("qpos_rmse"), 5),
            "asset_audit": result["asset_status"],
            "status": result["status"],
        }
        for index, result in enumerate(ranked, start=1)
    ]


def _report_notes(results: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """Build non-tabular, reproducible interpretation notes for a run.

    Args:
        results: Completed worker results.
        args: Benchmark options used for the run.

    Returns:
        Markdown bullet text without additional tables.
    """
    notes = [
        "Method: fixed-base free-space check with world gravity (0, 0, -9.81 m/s²); "
        f"the base is raised {_BASE_HEIGHT_M:.1f} m solely to remove plane contact.",
        "No drive gain, damping, effort limit, link mass, friction, inertia, joint limit, "
        "or gravity setting is overridden. Loader-owned articulation gravity is preserved.",
        f"Control: {args.control_hz:.1f} Hz commands, {args.physics_dt * 1000.0:.3f} ms physics dt, "
        f"{args.settle_seconds:.2f} s hold, {args.trajectory_seconds:.2f} s at {args.frequency_hz:.2f} Hz, "
        f"command_mode={args.command_mode}.",
        "qpos-only mode performs zero velocity-target writes and retains a zero "
        "velocity target; setting qvel state to zero during reset is not a target command.",
        "qpos metrics use native URDF joint units (rad for revolute and m for prismatic joints); "
        "normalized metrics divide each joint error by its limit-safe sine amplitude.",
        "Tracking pass gate: dynamic normalized RMSE ≤ 0.10, normalized P95 ≤ 0.20, and static normalized RMSE ≤ 0.10.",
        "Asset audit reads both source URDF inertial declarations and loaded runtime masses. "
        "`review` means missing/invalid source inertial declarations need inspection; it is not silently repaired by the runtime mass fallback.",
        "GPU memory comes from nvidia-smi's device-wide usage; concurrent workloads can affect it.",
    ]
    for result in results:
        if result.get("error"):
            notes.append(f"{result['robot']}: {result['error']}")
            continue
        notes.append(
            f"{result['robot']}: drive={result.get('drive_type')}, fixed_base={result.get('fixed_base')}, "
            f"position/velocity_iters={result.get('min_position_iters')}/{result.get('min_velocity_iters')}, "
            f"stiffness={_format_number(result.get('drive_stiffness_min'), 0)}–{_format_number(result.get('drive_stiffness_max'), 0)}, "
            f"damping={_format_number(result.get('drive_damping_min'), 0)}–{_format_number(result.get('drive_damping_max'), 0)}, "
            f"effort={_format_number(result.get('drive_max_effort_min'), 0)}–{_format_number(result.get('drive_max_effort_max'), 0)}, "
            f"runtime_links={result.get('runtime_link_count')}, "
            f"nonpositive_runtime_masses={result.get('runtime_nonpositive_mass_count')}, "
            f"loaded-source_mass_delta_kg={_format_number(result.get('runtime_source_mass_delta_kg'), 3)}, "
            f"loaded-source_mass_delta_ratio={_format_number(100.0 * result.get('runtime_source_mass_delta_ratio', float('nan')), 2)}%, "
            f"source_missing_inertials={result.get('source_missing_inertial_count')}, "
            f"source_nonpositive_inertias={result.get('source_nonpositive_inertia_count')}, "
            f"friction(static/dynamic)={_format_number(result.get('static_friction'), 2)}/{_format_number(result.get('dynamic_friction'), 2)}, "
            f"contact_offset_m={_format_number(result.get('contact_offset_m'), 4)}, "
            f"velocity_target_writes={result.get('velocity_target_write_count')}."
        )
        notes.append(
            f"{result['robot']}: worst normalized-RMSE joint={result.get('worst_joint')} "
            f"({_format_number(result.get('worst_joint_normalized_rmse'), 3)}; "
            f"P95={_format_number(result.get('worst_joint_normalized_p95'), 3)}), "
            f"drive stiffness/damping/effort={_format_number(result.get('worst_joint_stiffness'), 0)}/"
            f"{_format_number(result.get('worst_joint_damping'), 0)}/"
            f"{_format_number(result.get('worst_joint_max_effort'), 0)}."
        )
    return notes


def _write_markdown_report(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> Path:
    """Write exactly one Markdown report containing the three required tables.

    Args:
        results: Completed worker results.
        args: Benchmark arguments used to produce them.

    Returns:
        Written report path.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"robot_trajectory_tracking_{timestamp}.md"
    lines = [
        "# Robot Trajectory Tracking Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
        *_markdown_table(_performance_rows(results)),
        "",
        "## Success & Other Metrics",
        "",
        *_markdown_table(_metric_rows(results)),
        "",
        "## Leaderboard",
        "",
        *_markdown_table(_leaderboard_rows(results)),
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in _report_notes(results, args)],
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run every selected robot in isolation and write one aggregate report.

    Args:
        args: Parsed options. Arguments are parsed from ``sys.argv`` when omitted.

    Returns:
        Path to the run's single Markdown report.
    """
    if args is None:
        args = _parse_args()
    results = [_run_robot_worker(args, robot_name) for robot_name in args.robots]
    report_path = _write_markdown_report(results, args)
    print(f"Robot trajectory tracking benchmark report: {report_path}")
    return report_path


def main() -> None:
    """Run the parent benchmark or one private isolated worker."""
    args = _parse_args()
    if args.worker:
        _worker_main(args)
        return
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
