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

"""Serial paired-arm call replay on frozen, task-derived motion requests."""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING

import numpy as np
import psutil
import torch

from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners import MoveType, PlanState

from .artifacts import environment_metadata, write_resolved_suite
from .config import stable_hash, suite_to_dict
from .metrics.trajectory import compute_case_outcomes
from .models import _json_value
from .planners.base import PlannerContext
from .registry import create_planner_adapter

if TYPE_CHECKING:
    from .runner import BenchmarkRunner


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)


def _tensor(value: torch.Tensor) -> dict:
    return {"dtype": str(value.dtype).removeprefix("torch."), "values": value.tolist()}


def _restore(value: dict, device: torch.device) -> torch.Tensor:
    dtype = {"float32": torch.float32, "float64": torch.float64}[value["dtype"]]
    return torch.tensor(value["values"], dtype=dtype, device=device)


def _context(runner: "BenchmarkRunner", robot) -> PlannerContext:
    protocol = runner.suite.protocol
    return PlannerContext(
        robot=robot,
        control_part=runner.control_part,
        device=runner.device,
        sample_interval=protocol.sample_interval,
        robot_id=runner.suite.robot.id,
        joint_velocity_limit_rad_s=protocol.joint_velocity_limit_rad_s,
        joint_acceleration_limit_rad_s2=protocol.joint_acceleration_limit_rad_s2,
        joint_jerk_limit_rad_s3=protocol.joint_jerk_limit_rad_s3,
    )


def _adapter(runner, spec, context, cases):
    adapter = create_planner_adapter(spec, context, adapters=runner.adapters or None)
    try:
        available, reason = adapter.availability()
        if not available:
            raise RuntimeError(reason or f"Unavailable planner: {spec.id}")
        adapter.build()
        if adapter.separate_prepare:
            adapter.prepare_cases(deepcopy(cases))
        return adapter
    except BaseException:
        adapter.close()
        raise


def _capture(runner, simulation, robot, provider, cases) -> list[dict]:
    """Freeze ready groups using a declared classical reference, never NMG."""
    references = [s for s in runner.suite.planners if s.adapter == "toppra"]
    if len(references) != 1:
        raise ValueError(
            "Paired request preparation requires one declared TOPPRA reference"
        )
    adapter = _adapter(runner, references[0], _context(runner, robot), cases)
    groups = []
    generator = adapter.require_motion_generator()
    original = generator.generate_multi_part
    current_case = None

    def capture(targets, options):
        if len(targets) != 2 or any(
            state.move_type != MoveType.EEF_MOVE
            for states in targets.values()
            for state in states
        ):
            raise ValueError("Paired replay currently requires two FullPose requests")
        request = {
            "source_case_id": current_case.case_id,
            "group_index": sum(
                g["source_case_id"] == current_case.case_id for g in groups
            ),
            "parts": {},
        }
        for part, states in targets.items():
            opts = options[part]
            if opts.start_qpos is None or opts.cartesian_path_sampler is not None:
                raise ValueError(
                    "Paired requests need explicit starts and serializable targets"
                )
            request["parts"][part] = {
                "start_qpos": _tensor(opts.start_qpos.detach().clone()),
                "targets": [_tensor(state.xpos.detach().clone()) for state in states],
                "options": {
                    key: deepcopy(getattr(opts, key))
                    for key in (
                        "strategy",
                        "sample_count",
                        "is_interpolate",
                        "is_linear",
                        "interpolation_dt",
                        "interpolate_nums",
                        "interpolate_position_step",
                        "interpolate_angle_step",
                        "preserve_cartesian_samples",
                    )
                },
            }
        request["request_id"] = stable_hash(request)
        result = original(targets, options)
        request["reference_success"] = {
            part: result[part].success.tolist() for part in targets
        }
        groups.append(request)
        return result

    try:
        provider.prepare_planner(adapter, cases[0])
        generator.generate_multi_part = capture
        for case in cases:
            current_case = case
            provider.reset_case(simulation, robot, case, runner.control_part)
            artifact = provider.plan_case(adapter, deepcopy(case))
            if error := provider.plan_contract_error(artifact):
                raise RuntimeError(error)
            if not any(g["source_case_id"] == case.case_id for g in groups):
                raise RuntimeError(
                    f"Reference did not reach a ready arm group: {case.case_id}"
                )
    finally:
        generator.generate_multi_part = original
        provider.close_planner(adapter)
        adapter.close()
    return groups


def _inputs(group, adapter, device):
    targets, options = {}, {}
    for part, data in group["parts"].items():
        targets[part] = [
            PlanState.from_xpos(_restore(value, device)) for value in data["targets"]
        ]
        base = dict(data["options"])
        native = adapter.atomic_plan_options(control_dt=base["interpolation_dt"])
        options[part] = MotionGenOptions(
            **base,
            start_qpos=_restore(data["start_qpos"], device),
            control_part=part,
            plan_opts=deepcopy(native),
        )
    return targets, options


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        # Device synchronization also waits for the CUDA execution provider's
        # streams; measuring only the Torch default stream is insufficient.
        torch.cuda.synchronize(device)


def _call(generator, targets, options, dispatch):
    if dispatch == "packed":
        return generator.generate_multi_part(targets, options)
    return {
        part: generator.generate(states, options[part])
        for part, states in targets.items()
    }


def _quality(results, group, source, robot, suite, timing_is_solved, device):
    outcomes = {}
    protocol = suite.protocol
    for part, data in group["parts"].items():
        start = _restore(data["start_qpos"], device)
        poses = torch.stack(
            [_restore(value, device) for value in data["targets"]], dim=1
        )
        case = replace(
            source,
            case_id=group["request_id"] + "/" + part,
            start_qpos=start,
            num_waypoints=poses.shape[1],
            target_waypoints=poses,
            reference_qpos=torch.zeros(
                (*poses.shape[:2], start.shape[-1]), device=device
            ),
            skill_id="N/A",
            primary_success="motion_valid",
            case_parameters={"motion_validity": "ordered_cartesian_waypoints"},
        )
        values = compute_case_outcomes(
            results[part],
            case,
            robot,
            part,
            validation_samples=protocol.validation_samples,
            position_threshold_m=protocol.position_threshold_m,
            rotation_threshold_rad=protocol.rotation_threshold_rad,
            joint_limit_tolerance_rad=protocol.joint_limit_tolerance_rad,
            start_state_tolerance_rad=protocol.start_state_tolerance_rad,
            joint_velocity_limit_rad_s=protocol.joint_velocity_limit_rad_s,
            joint_acceleration_limit_rad_s2=protocol.joint_acceleration_limit_rad_s2,
            joint_jerk_limit_rad_s3=protocol.joint_jerk_limit_rad_s3,
            dynamic_limit_tolerance=protocol.dynamic_limit_tolerance,
            timing_is_solved=timing_is_solved,
        )
        outcomes[part] = [asdict(value) for value in values]
    return outcomes


def _memory(operation, device):
    """Measure a separate warmed call; Torch counters exclude native allocators."""
    _sync(device)
    process = psutil.Process()
    cpu_before = process.memory_info().rss
    gpu_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = operation()
    _sync(device)
    peak = torch.cuda.max_memory_allocated(device) if gpu_before is not None else None
    values = {
        "cpu_delta_mb": (process.memory_info().rss - cpu_before) / 1024**2,
        "gpu_delta_mb": (
            None
            if gpu_before is None
            else (torch.cuda.memory_allocated(device) - gpu_before) / 1024**2
        ),
        "peak_gpu_mb": None if peak is None else peak / 1024**2,
        "peak_gpu_delta_mb": None if peak is None else (peak - gpu_before) / 1024**2,
    }
    del result
    return values


def _parity(left, right):
    """Compare dispatch modes outside timing, retaining differing outcomes."""
    errors = {}
    equivalent = True
    for part in left:
        same = torch.equal(left[part].success, right[part].success)
        fields = {}
        for name, atol in (
            ("positions", 5e-5),
            ("dt", 5e-5),
            ("velocities", 5e-4),
            ("accelerations", 5e-2),
        ):
            a, b = getattr(left[part], name), getattr(right[part], name)
            if a is None or b is None:
                fields[name] = None
                same &= a is None and b is None
            elif a.shape != b.shape:
                fields[name] = {"shape_a": list(a.shape), "shape_b": list(b.shape)}
                same = False
            else:
                fields[name] = float((a - b).abs().max().item())
                same &= bool(torch.allclose(a, b, atol=atol, rtol=0))
        errors[part] = {"equivalent": same, "differences": fields}
        equivalent &= same
    return {"equivalent": equivalent, "arms": errors}


def _measure_group(runner, adapter, robot, group, source, settings, stream):
    generator = adapter.require_motion_generator()
    targets, options = _inputs(group, adapter, runner.device)
    modes = (
        ["sequential", "packed"] if generator.supports_multi_part else ["sequential"]
    )
    for mode in modes:
        for _ in range(settings["warmup_trials"]):
            owned_targets, owned_options = deepcopy((targets, options))
            result = _call(generator, owned_targets, owned_options, mode)
            _sync(runner.device)
            del result
    parity = None
    if len(modes) == 2:
        left = _call(generator, *deepcopy((targets, options)), "sequential")
        right = _call(generator, *deepcopy((targets, options)), "packed")
        parity = _parity(left, right)
        del left, right
    records = []
    for repeat in range(settings["measured_trials"]):
        # Alternate dispatch order to avoid consistently favoring the second
        # mode through warm caches or changing host load.
        for mode in modes[:: 1 if repeat % 2 == 0 else -1]:
            owned_targets, owned_options = deepcopy((targets, options))
            _sync(runner.device)
            started = time.perf_counter()
            result = _call(generator, owned_targets, owned_options, mode)
            _sync(runner.device)
            elapsed = (time.perf_counter() - started) * 1000
            quality = _quality(
                result,
                group,
                source,
                robot,
                runner.suite,
                adapter.timing_is_solved,
                runner.device,
            )
            row = {
                "algorithm": adapter.spec.id,
                "dispatch": mode,
                "track": "paired_arm_requests",
                "group": source.case_id + "/" + str(group["group_index"]),
                "request_id": group["request_id"],
                "batch_size": source.batch_size,
                "control_part_count": len(targets),
                "waypoint_count": ",".join(str(len(v)) for v in targets.values()),
                "target_type": "pose",
                "repeat": repeat,
                "cost_time_ms": elapsed,
                "outcomes": quality,
                "dispatch_parity": parity,
                "policy_batching": {
                    part: value.policy_batching for part, value in result.items()
                },
                "timing_scope": "completed_pair_call_including_fk_transfers_and_selected_retiming",
                "timing_is_solved": adapter.timing_is_solved,
            }
            row = _json_value(row)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            records.append(row)
            del result
    memory = {}
    for mode in modes:
        owned_targets, owned_options = deepcopy((targets, options))
        memory[mode] = _memory(
            lambda mode=mode: _call(generator, owned_targets, owned_options, mode),
            runner.device,
        )
    return records, memory


def _tables(records, memories):
    buckets = defaultdict(list)
    for row in records:
        buckets[(row["algorithm"], row["dispatch"], row["group"])].append(row)
    times, quality = [], []
    for key, rows in buckets.items():
        identity = {
            name: rows[0][name]
            for name in (
                "algorithm",
                "dispatch",
                "track",
                "group",
                "batch_size",
                "control_part_count",
                "waypoint_count",
                "target_type",
            )
        }
        costs = [row["cost_time_ms"] for row in rows]
        times.append(
            {
                **identity,
                "cost_time_ms": statistics.median(costs),
                "p95_ms": float(np.percentile(costs, 95)),
                "repeats": len(rows),
                **memories[key],
            }
        )
        values = [
            value for row in rows for arm in row["outcomes"].values() for value in arm
        ]
        quality.append(
            {
                **identity,
                "success_rate": sum(
                    all(
                        v["geometry_valid"] and v["planning_success"]
                        for arm in row["outcomes"].values()
                        for v in arm
                    )
                    for row in rows
                )
                / len(rows),
                "arm_geometry_rate": sum(v["geometry_valid"] for v in values)
                / len(values),
                "timed_motion_rate": (
                    None
                    if not rows[0]["timing_is_solved"]
                    else sum(v["timed_motion_valid"] is True for v in values)
                    / len(values)
                ),
                "dispatch_equivalent": (
                    None
                    if rows[0]["dispatch_parity"] is None
                    else rows[0]["dispatch_parity"]["equivalent"]
                ),
            }
        )
    leaderboard = sorted(
        quality,
        key=lambda row: (
            -row["success_rate"],
            row["algorithm"],
            row["dispatch"],
            row["group"],
        ),
    )
    return (
        times,
        quality,
        [{"rank": index + 1, **row} for index, row in enumerate(leaderboard)],
    )


def _run(
    runner: "BenchmarkRunner",
    output: Path,
    *,
    request_manifest: Path,
    warmup_trials: int,
    measured_trials: int,
    source_cases_per_suite: int,
) -> dict:
    """Reuse the runner lifecycle while keeping timing replicas out of quality."""
    settings = {
        "warmup_trials": warmup_trials,
        "measured_trials": measured_trials,
        "source_cases_per_suite": source_cases_per_suite,
    }
    if any(type(value) is not int or value < 1 for value in settings.values()):
        raise ValueError("Paired performance settings must be positive integers")
    if runner.replay_cases is None:
        raise ValueError("Paired performance requires a frozen Atomic case manifest")
    if not runner.planner_specs:
        raise ValueError("Paired performance requires at least one selected planner")
    if any(
        track.scenario != "atomic_skill" for track in runner.suite.evaluation_tracks()
    ):
        raise ValueError("Paired performance requires Atomic task suites")
    selected = sorted(runner.replay_cases, key=lambda case: case.case_id)[
        :source_cases_per_suite
    ]
    if len(selected) != source_cases_per_suite:
        raise ValueError("Not enough frozen source cases for paired performance")
    reference = next(
        (spec for spec in runner.suite.planners if spec.adapter == "toppra"), None
    )
    if reference is None:
        raise ValueError("Missing declared reference planner for request preparation")
    from .artifacts import _case_to_dict

    suite_data = suite_to_dict(runner.suite)
    embodiment = runner.embodiment_resolution.to_metadata()
    identity = {
        "schema_version": 1,
        "suite": {key: value for key, value in suite_data.items() if key != "planners"},
        "embodiment": {
            key: value for key, value in embodiment.items() if key != "component"
        },
        "source_cases_sha256": stable_hash([_case_to_dict(case) for case in selected]),
        "reference": {"adapter": reference.adapter, "config": reference.config},
    }
    cached = None
    if request_manifest.exists():
        cached = json.loads(request_manifest.read_text())
        if cached.get("identity") != identity or cached.get(
            "requests_sha256"
        ) != stable_hash(cached.get("requests")):
            raise ValueError(
                "Frozen paired requests conflict with their source identity or hash"
            )
    runner.replay_cases = deepcopy(selected)
    output.mkdir(parents=True, exist_ok=True)
    rows, memory, frozen, planner_metadata = [], {}, [], []
    with (
        (output / "trials.jsonl").open("x") as stream,
        closing(runner._case_batches()) as batches,
    ):
        write_resolved_suite(output / "resolved_suite.yaml", runner.suite)
        for simulation, robot, provider, cases in batches:
            if cached is None:
                groups = _capture(runner, simulation, robot, provider, cases)
            else:
                ids = {case.case_id for case in cases}
                groups = [g for g in cached["requests"] if g["source_case_id"] in ids]
            if not groups:
                raise ValueError("No frozen paired requests for the selected cases")
            frozen.extend(groups)
            # A suite has one declared scene batch in this performance scope.
            # Publish every selected request before any candidate measurement.
            if len(cases) != len(selected):
                raise ValueError(
                    "Paired replay currently requires one track and scene batch"
                )
            payload = cached or {
                "identity": identity,
                "requests": groups,
                "requests_sha256": stable_hash(groups),
                "capture_runtime": environment_metadata(),
            }
            if cached is None:
                _write_new(request_manifest, payload)
            _write_new(output / "requests.json", payload)
            lookup = {case.case_id: case for case in cases}
            for spec in runner.planner_specs:
                adapter = _adapter(runner, spec, _context(runner, robot), cases)
                try:
                    planner_metadata.append(_json_value(asdict(adapter.metadata)))
                    for group in groups:
                        source = lookup[group["source_case_id"]]
                        provider.reset_case(
                            simulation, robot, source, runner.control_part
                        )
                        measured, usage = _measure_group(
                            runner, adapter, robot, group, source, settings, stream
                        )
                        rows.extend(measured)
                        for mode, values in usage.items():
                            memory[
                                (
                                    spec.id,
                                    mode,
                                    source.case_id + "/" + str(group["group_index"]),
                                )
                            ] = values
                finally:
                    adapter.close()
    times, quality, leaderboard = _tables(rows, memory)
    _write_new(
        output / "planners.json",
        {
            "selected": [_json_value(asdict(spec)) for spec in runner.planner_specs],
            "actual": planner_metadata,
            "resolved_embodiment": embodiment,
        },
    )
    aggregates = {
        "paired_arm_time_and_memory": times,
        "paired_arm_success": quality,
        "paired_arm_leaderboard": leaderboard,
    }
    _write_new(output / "aggregates.json", aggregates)
    notes = [
        "Inputs are actual independent ready arm requests captured with a declared TOPPRA reference before candidate measurement, then frozen and reused.",
        "Latency is synchronized wall time for the complete pair call, including FK, ONNX transfers and selected retiming; setup, validation and physical execution are excluded.",
        "Memory uses a separate warmed call. CPU values are retained RSS deltas; GPU counters cover only Torch allocations, excluding ONNX Runtime/cuRobo/native allocations.",
        "Repeated calls do not add independent quality cases. Success here is paired motion geometry, not physical task completion. p95 is descriptive at the recorded repetition count.",
        "Packed/sequential parity is reported per request; a speed comparison cannot imply equal behavior when parity fails.",
    ]
    lines = ["# Paired-arm performance", "", *["- " + note for note in notes], ""]
    for title, table in (
        ("Time & Memory", times),
        ("Success & Other Metrics", quality),
        ("Leaderboard", leaderboard),
    ):
        columns = list(table[0])
        lines += [
            "## " + title,
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        lines += [
            "| "
            + " | ".join(
                "N/A" if row[key] is None else str(row[key]) for key in columns
            )
            + " |"
            for row in table
        ]
        lines.append("")
    report = output / "report.md"
    with report.open("x") as stream:
        stream.write("\n".join(lines))
    execution = {
        "mode": "serial",
        "workers": 1,
        "device": str(runner.device),
        "timing_comparable": runner.execution.get("timing_comparable", True),
        "settings": settings,
        "runtime": environment_metadata(),
        "notes": notes,
    }
    _write_new(output / "execution.json", execution)
    return {
        "status": "completed",
        "run_dir": str(output.resolve()),
        "report": str(report.resolve()),
        "aggregates": str((output / "aggregates.json").resolve()),
        "case_sha256": stable_hash(frozen),
        "case_count": len(frozen),
        "errors": {},
        "execution": execution,
        "performance_tables": {"paired_arm_time_and_memory": times},
        "leaderboards": {"paired_arm_leaderboard": leaderboard},
    }
