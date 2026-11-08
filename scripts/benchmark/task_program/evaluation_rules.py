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

"""Compare legacy and compiled condition evaluation on equivalent CPU/GPU batches.

Run: python -m scripts.benchmark.task_program.evaluation_rules --device cpu
"""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter_ns

import torch

from embodichain.lab.evaluation import (
    EvaluationSession,
    EvaluationSpec,
    EvaluationStatus,
    NearTargetHoldSpec,
    PredicateSpec,
    TemporalRule,
    builtin_predicates,
    evaluate_recorded_rules,
)
from embodichain.lab.gym.envs.objectives.conditions import evaluate_near_target


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
        "cuda_peak_additional_bytes": peak,
    }


def _near_graph(*, slots=0):
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


def _measure_conditions(device, rows, iterations):
    graph = _near_graph()
    position = torch.zeros(rows, 3, device=device)
    position[::2, 0] = 0.5
    position[1::4, 0] = 0.6
    target = torch.zeros_like(position)
    values = {"position": position, "target": target}

    def legacy():
        return evaluate_near_target(position, target, 0.5)

    def compiled():
        return graph.evaluate(values, batch_size=rows, device=device)["near"]

    old, new = legacy(), compiled()
    torch.testing.assert_close(old.satisfied, new.status == EvaluationStatus.SATISFIED)
    torch.testing.assert_close(
        old.metrics["distance"], new.metrics["distance"], rtol=0, atol=0
    )
    return {
        "rows": rows,
        "legacy": _measure(legacy, device=device, iterations=iterations),
        "compiled": _measure(compiled, device=device, iterations=iterations),
    }


def _measure_recording(device, rows, slots):
    graph = _near_graph(slots=slots)
    spec = NearTargetHoldSpec("position", "target", 0.5, 3, slots)
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

    def legacy():
        session = EvaluationSession(
            spec, env_ids=ids, episode_ids=episodes, device=device
        )
        p, t, mask, rev = (
            value.to(device) for value in (position, target, valid, revisions)
        )
        for slot in range(slots):
            session.advance(
                slot,
                p[:, slot],
                t[:, slot],
                valid=mask[:, slot],
                revisions=rev[:, slot],
            )
        return session.snapshot(closed=True)

    def compiled(chunk):
        return evaluate_recorded_rules(
            graph,
            recording,
            env_ids=ids,
            episode_ids=episodes,
            scope="benchmark",
            chunk_steps=chunk,
            device=device,
        ).rules["hold"]

    expected = legacy()
    measurements = {"legacy_full": _measure(legacy, device=device, iterations=1)}
    for chunk in sorted({1, min(16, slots), slots}):
        observed = compiled(chunk)
        torch.testing.assert_close(expected.window_status, observed.status)
        torch.testing.assert_close(
            expected.witness_revisions, observed.witness_revisions
        )
        measurements[f"compiled_chunk_{chunk}"] = {
            **_measure(lambda: compiled(chunk), device=device, iterations=1),
            "selected_chunk_bytes_upper_bound": rows * chunk * (2 * (3 * 4 + 1) + 8),
        }
    return {"rows": rows, "slots": slots, **measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256, 4096])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--recorded-rows", type=int, default=128)
    parser.add_argument("--slots", type=int, default=128)
    args = parser.parse_args()
    if min(*args.batches, args.iterations, args.recorded_rows) < 1 or args.slots < 3:
        parser.error(
            "Batch sizes and iterations must be positive and slots must be at least three."
        )
    torch.set_num_threads(1)
    device = torch.empty(0, device=args.device).device
    print(
        json.dumps(
            {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "torch": torch.__version__,
                "cpu_threads": torch.get_num_threads(),
                "dtype": "float32",
                "conditions": [
                    _measure_conditions(device, rows, args.iterations)
                    for rows in args.batches
                ],
                "recording": _measure_recording(device, args.recorded_rows, args.slots),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
