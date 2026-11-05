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

"""Measure CPU stability-policy execution and on-demand report serialization.

Run: python -m scripts.benchmark.task_program.stability --output /tmp/stability
The CPU fixture supplies constant poses and holds, without simulation or native
contact acquisition. This measures policy overhead, not physical task throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

import torch
from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationRigidObjectBinding,
    SimulationSceneBinding,
)
from embodichain.lab.task_program.integrations.simulation.policies import (
    SimulationSegmentPolicyPort,
)
from embodichain.lab.task_program.integrations.simulation.stability import (
    SimulationStabilityConditionCfg,
)
from embodichain.lab.task_program.integrations.simulation import _stability_runtime


def benchmark_policy(
    batch: int, *, held: bool, read_each_step: bool, repeats: int
) -> dict[str, Any]:
    """Measure the public policy port on one fixed three-second CPU hold."""
    pose = torch.eye(4).repeat(batch, 1, 1)
    qpos = torch.zeros(batch, 2)
    body = SimpleNamespace(
        get_local_pose=lambda **kw: pose.clone(), is_non_dynamic=False
    )
    simulation = SimpleNamespace(get_rigid_object=lambda uid: body)
    counts = {"qpos": 0, "fk": 0}

    def get_qpos(**kwargs):
        counts["qpos"] += 1
        return qpos.clone()

    def compute_fk(**kwargs):
        counts["fk"] += 1
        return pose.clone()

    robot = SimpleNamespace(
        get_qpos=get_qpos, get_joint_ids=lambda **kw: [0], compute_fk=compute_fk
    )
    binding = SimulationSceneBinding(
        "benchmark",
        rigid_objects=(SimulationRigidObjectBinding("cube", "native_cube"),),
    )
    port = SimulationSegmentPolicyPort(
        simulation,
        robot,
        binding,
        settle_presets={
            "stable": SimulationStabilityConditionCfg(
                entity="cube",
                kind="hold" if held else "placement",
                motion_parts=("left_arm", "right_arm") if held else (),
                target_position=(0, 0, 0),
                local_axis=(0, 0, 1),
                duration=3.0,
                timeout=12.0,
            )
        },
        step_dt=0.04,
    )
    declaration = dict(
        program_id="stability_benchmark",
        targets={},
        integration=dict(
            scene_registry="benchmark", robot_profile="robot", runtime_preset="default"
        ),
        program=dict(
            kind="segment",
            name="stable",
            steps=dict(kind="invoke", call=dict(kind="pick", object="cube")),
            post=[dict(kind="wait_stable", entity="cube", preset="stable")],
        ),
    )
    segment = next(
        TaskProgramCompiler(binding.declare())
        .compile(decode_task_program(declaration))
        .iter_segments()
    )
    policy = segment.post_policies[0]
    active = torch.ones(batch, dtype=torch.bool)

    def operation():
        count = 0
        for _ in port.actions(policy, segment=segment, active_mask=active):
            count += 1
            if read_each_step:
                port.post_policy_metadata(policy, segment=segment)
        return count, port.post_policy_metadata(policy, segment=segment)

    counts.update(qpos=0, fk=0)
    count, metadata = operation()
    reads = dict(counts)
    assert count == 75 and metadata["accepted_mask"] == [True] * batch
    expected = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    for _ in range(5):
        operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - started) * 1000)
        del result
    tracemalloc.start()
    count, metadata = operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert (
        hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        == expected
    )
    return dict(
        batch=batch,
        condition="hold" if held else "placement",
        fixture_read_counts=reads,
        metadata_reads="every_step" if read_each_step else "terminal_only",
        cost_time_ms=statistics.median(samples),
        p95_ms=sorted(samples)[int(0.95 * (len(samples) - 1))],
        samples_ms=samples,
        peak_python_bytes=peak,
        command_count=count,
        metadata_sha256=expected,
        success_rate=1.0,
    )


def main() -> None:
    """Write machine-readable measurements and the three benchmark report tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("--repeats must be at least five")
    torch.set_num_threads(1)
    rows = [
        benchmark_policy(batch, held=held, read_each_step=reads, repeats=args.repeats)
        for batch in (1, 4)
        for held in (False, True)
        for reads in (False, True)
    ]
    source = Path(_stability_runtime.__file__)
    report = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        runtime_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cpu=next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        python=platform.python_version(),
        torch=torch.__version__,
        dtype="float32",
        device="cpu",
        threads=1,
        warmup=5,
        repeats=args.repeats,
        scope="Completed CPU policy calls over constant fixture poses; construction, simulation, native contacts and GPU excluded. Memory is a separate tracemalloc Python-allocation peak, not native/Torch/RSS peak.",
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CPU stability policy",
        "",
        report["scope"],
        "",
        f"CPU: {report['cpu']}; Torch {torch.__version__}; one thread; five warmups; {args.repeats} samples; 75 holds/call.",
        "",
        "## Time & Memory",
        "",
        "| Batch | Metadata reads | cost_time_ms | p95_ms | peak_python_bytes | peak_gpu_mb |",
        "|---:|---|---:|---:|---:|---|",
        *[
            f"| {r['batch']} | {r['condition']} / {r['metadata_reads']} | {r['cost_time_ms']:.4f} | {r['p95_ms']:.4f} | {r['peak_python_bytes']} | unmeasured |"
            for r in rows
        ],
        "",
        "## Success & Other Metrics",
        "",
        "| Batch | Metadata reads | success_rate | commands |",
        "|---:|---|---:|---:|",
        *[
            f"| {r['batch']} | {r['condition']} / {r['metadata_reads']} | {r['success_rate']} | {r['command_count']} |"
            for r in rows
        ],
        "",
        "## Leaderboard",
        "",
        "| Implementation | overall_success_rate | Meaning |",
        "|---|---:|---|",
        "| Current checkout | 1.0 | Fixture acceptance and deterministic metadata only; no physical qualification |",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output.with_suffix(".md")),
                "rows": [
                    {k: v for k, v in row.items() if k != "samples_ms"} for row in rows
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
