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

"""Measure RL transitions, recorded rule evaluation and process guards.

Run: python -m scripts.benchmark.task_program.evaluation_rules --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from time import perf_counter_ns

import torch

from embodichain.lab.gym.envs.managers import ConditionCfg, RLTaskCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.conditions import object_near_target
from embodichain.lab.gym.envs.managers.termination_manager import TerminationManager


def _measure(fn, *, device, iterations, repeats=5):
    synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else lambda: None
    )
    for _ in range(3):
        fn()
    synchronize()
    elapsed = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(iterations):
            fn()
        synchronize()
        elapsed.append((perf_counter_ns() - start) / iterations / 1000)
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) - memory_before
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        fn()
        synchronize()
    allocated = sum(max(0, event.self_cpu_memory_usage) for event in profile.events())
    return {
        "median_us": median(elapsed),
        "max_repeat_mean_us": max(elapsed),
        "cpu_allocated_bytes": allocated,
        "scalar_reads": sum(
            event.name == "aten::_local_scalar_dense" for event in profile.events()
        ),
        "cuda_peak_additional_bytes": peak,
    }


def _near_graph(*, slots=0):
    from embodichain.lab.evaluation import (
        EvaluationSpec,
        PredicateSpec,
        TemporalRule,
        builtin_predicates,
    )

    rules = {
        "near": PredicateSpec(
            "near_target",
            {"position": "position", "target": "target"},
            {"tolerance": 0.5},
        )
    }
    if slots:
        rules["hold"] = TemporalRule("hold_for", "near", slots - 1, steps=3)
    return builtin_predicates().compile(
        EvaluationSpec(rules, ("hold",) if slots else ("near",))
    )


def _measure_conditions(device, rows, iterations, *, legacy_only):
    pose = torch.eye(4, device=device).expand(rows, -1, -1).clone()
    pose[::2, 0, 3] = 0.5
    pose[1::4, 0, 3] = 0.6
    target = torch.zeros(rows, 3, device=device)
    cube = SimpleNamespace(get_local_pose=lambda **kwargs: pose)
    env = SimpleNamespace(
        sim=SimpleNamespace(asset_uids=("cube",), get_rigid_object=lambda uid: cube),
        num_envs=rows,
        device=device,
        target=target,
    )
    legacy = TerminationManager(
        RLTaskCfg(
            success={
                "goal": ConditionCfg(
                    func=object_near_target,
                    params={
                        "entity_cfg": SceneEntityCfg(uid="cube"),
                        "target_pose_key": "target",
                        "tolerance": 0.5,
                        "inclusive": True,
                    },
                )
            }
        ),
        env,
    )
    result = {
        "rows": rows,
        "legacy_manager": _measure(
            legacy.compute, device=device, iterations=iterations
        ),
    }
    if legacy_only:
        return result

    from embodichain.lab.gym.envs.managers.conditions import evaluation_condition

    compiled = TerminationManager(
        RLTaskCfg(
            success={
                "goal": ConditionCfg(
                    func=evaluation_condition,
                    params={
                        "sources": {
                            "position": {"rigid_object": "cube"},
                            "target": {"attribute": "target", "kind": "position"},
                        },
                        "root": "near",
                        "rules": {
                            "near": {
                                "kind": "near_target",
                                "bindings": {
                                    "position": "position",
                                    "target": "target",
                                },
                                "params": {"tolerance": 0.5, "inclusive": True},
                            }
                        },
                    },
                )
            }
        ),
        env,
    )
    compiled.compute()
    for old, new in zip(
        legacy.task_state()[:2], compiled.task_state()[:2], strict=True
    ):
        torch.testing.assert_close(old, new)
    torch.testing.assert_close(legacy.terminated, compiled.terminated)
    torch.testing.assert_close(legacy.truncated, compiled.truncated)
    expected = legacy.snapshot()["conditions"]["success.goal"]["metrics"]["distance"]
    observed = compiled.snapshot()["conditions"]["success.goal"]["rules"]["near"][
        "metrics"
    ]["distance"]
    torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    result["compiled_manager"] = _measure(
        compiled.compute, device=device, iterations=iterations
    )
    return result


def _measure_recording(device, rows, slots, path):
    from embodichain.lab.evaluation import EvaluationStatus, evaluate_recorded_rules

    graph = _near_graph(slots=slots)
    position = torch.full((rows, slots, 3), 0.1)
    position[::2, :, 0] = 0.6
    target = torch.zeros_like(position)
    revisions = torch.arange(slots).expand(rows, -1).clone()
    valid = torch.ones(rows, slots, dtype=torch.bool)
    ids, episodes = tuple(range(rows)), (3,) * rows
    recording = {
        "raw_observations": {
            "schema_version": 1,
            "env_ids": ids,
            "episode_ids": episodes,
            "revisions": revisions,
            "lengths": torch.full((rows,), slots - 1),
            "fields": {
                name: {
                    "descriptor": {
                        "quantity": "xyz",
                        "shape": (3,),
                        "dtype": "torch.float32",
                    },
                    "values": value,
                    "valid": valid,
                }
                for name, value in (("position", position), ("target", target))
            },
        }
    }

    torch.save(recording, path)

    def evaluate(source, chunk):
        return evaluate_recorded_rules(
            graph,
            source,
            env_ids=ids,
            episode_ids=episodes,
            scope="benchmark",
            chunk_steps=chunk,
            device=device,
        ).rules["hold"]

    def full():
        source = torch.load(path, map_location="cpu", weights_only=False)
        return evaluate(source, slots)

    expected = full()
    assert expected.status[::2].eq(EvaluationStatus.FAILED).all()
    assert expected.status[1::2].eq(EvaluationStatus.SATISFIED).all()
    measurements = {"eager_file_full": _measure(full, device=device, iterations=1)}
    for chunk in sorted({1, min(16, slots), slots}):
        observed = evaluate(path, chunk)
        torch.testing.assert_close(expected.status, observed.status)
        torch.testing.assert_close(
            expected.witness_revisions, observed.witness_revisions
        )
        measurements[f"mmap_chunk_{chunk}"] = {
            **_measure(lambda: evaluate(path, chunk), device=device, iterations=1),
            "selected_chunk_bytes_upper_bound": rows * chunk * (2 * (3 * 4 + 1) + 8),
        }
    return {"rows": rows, "slots": slots, **measurements}


def _measure_process(device, rows, iterations):
    from embodichain.lab.evaluation import EvaluationStatus
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge

    graph = _near_graph()
    position = torch.zeros(rows, 3, device=device)
    values = {"position": position, "target": position}
    indices = torch.arange(rows, device=device)
    host = SimpleNamespace(
        _last_segment_lifecycle=SimpleNamespace(
            pending_action=object(),
            abort_started=False,
            observed_step=None,
        ),
        _validator_port=SimpleNamespace(_row_indices=indices),
        _clock=SimpleNamespace(step_index=0),
        _eligible_mask=torch.ones(rows, device=device, dtype=torch.bool),
        env_ids=indices,
    )

    def check():
        status = graph.evaluate(values, batch_size=rows, device=device)["near"].status
        return status == EvaluationStatus.SATISFIED

    def guard():
        host._clock.step_index += 1
        TaskProgramDemoBridge._observe_segment_frame(
            host,
            None,
            episode_owner=None,
            process_check=lambda: (check(), dict),
        )

    return {
        "rows": rows,
        "predicate_only": _measure(check, device=device, iterations=iterations),
        "bridge_guard_no_rejections": _measure(
            guard, device=device, iterations=iterations
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256, 4096])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--recorded-rows", type=int, default=128)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--legacy-rl-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(*args.batches, args.iterations, args.recorded_rows) < 1 or args.slots < 3:
        parser.error(
            "Batch sizes and iterations must be positive and slots must be at least three."
        )
    torch.set_num_threads(1)
    device = torch.empty(0, device=args.device).device
    result = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "dtype": "float32",
        "rl_transitions": [
            _measure_conditions(
                device, rows, args.iterations, legacy_only=args.legacy_rl_only
            )
            for rows in args.batches
        ],
    }
    if not args.legacy_rl_only:
        with TemporaryDirectory() as directory:
            result["recording"] = _measure_recording(
                device,
                args.recorded_rows,
                args.slots,
                Path(directory) / "trajectory.pt",
            )
        result["process"] = _measure_process(
            device, min(256, max(args.batches)), args.iterations
        )
    report = json.dumps(result, indent=2)
    print(report)
    if args.report is not None:
        args.report.write_text(
            "# Evaluation benchmark\n\n"
            "RL calls the termination manager once per transition using in-memory pose providers. "
            "Offline comparisons read the same file using eager loading or memory-mapped chunks. "
            "The process guard runs the bridge callback without rejections. Scalar reads are "
            "counted by the Torch profiler. CPU bytes count positive allocation events.\n\n"
            f"```json\n{report}\n```\n"
        )


if __name__ == "__main__":
    main()
