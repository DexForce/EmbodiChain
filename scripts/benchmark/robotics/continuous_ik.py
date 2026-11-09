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

"""Measure CPU fixed-time IK search overhead with an analytic prismatic oracle.

Run: python -m scripts.benchmark.robotics.continuous_ik --output REPORT.md
An optional --reference-source compares the frozen predecessor's numeric routine.
This is solver-adapter overhead, not native IK or end-to-end robot performance.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
import hashlib
import math
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

import torch

from embodichain.compute.kinematics.ik_path import resolve_seeded_ik_path


def _reference(path: Path) -> Callable:
    """Load only the predecessor's numerical function, with measurement disabled."""
    tree = ast.parse(path.read_text())
    function = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "resolve_seeded_ik_path"
    )

    class RemoveObserverImport(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "_search_cost" else node

    function = RemoveObserverImport().visit(function)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "math": math,
        "record_search": lambda *args, **kwargs: None,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["resolve_seeded_ik_path"]


def _workload(count: int, near_limit: bool) -> tuple[dict, torch.Tensor]:
    poses = torch.eye(4).repeat(count, 1, 1)
    poses[:, 0, 3] = 1.99 if near_limit else torch.linspace(0.01, 0.5, count)
    start = torch.zeros(1, 7)

    def solve(pose, seed):
        qpos = torch.zeros_like(seed)
        qpos[:, 0] = pose[:, 0, 3]
        return torch.ones(1, dtype=torch.bool), qpos

    expected = torch.zeros(count + 1, 7)
    expected[1:, 0] = poses[:, 0, 3]
    return (
        dict(
            poses=poses,
            start=start,
            solve=solve,
            lower=torch.full((1, 7), -2.0),
            upper=torch.full((1, 7), 2.0),
            limits=torch.full((1, 7), 10.0),
            dt=1.0,
        ),
        expected,
    )


def _table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def _cpu_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    """Run equivalent workloads and write the three-table benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/continuous_ik.md")
    )
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    algorithms = {"shared": resolve_seeded_ik_path}
    if args.reference_source:
        algorithms["frozen numeric routine"] = _reference(args.reference_source)
    timings, metrics, totals = [], [], {name: [0, 0] for name in algorithms}
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            for count, near in ((32, False), (256, False), (32, True)):
                values, expected = _workload(count, near)
                workload = f"T={count}, D=7, near_limit={near}"
                for name, function in algorithms.items():
                    output = function(**values)
                    correct = output is not None and torch.equal(output, expected)
                    totals[name][0] += int(correct)
                    totals[name][1] += 1
                    if not correct:
                        raise RuntimeError(
                            f"{name} failed the independent prismatic FK oracle"
                        )
                    for _ in range(2):
                        function(**values)
                    samples = []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        result = function(**values)
                        samples.append((time.perf_counter() - started) * 1000)
                        del result
                    tracemalloc.start()
                    result = function(**values)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    timings.append(
                        [
                            name,
                            workload,
                            f"{statistics.median(samples):.3f}",
                            f"{peak / 1024:.2f}",
                            result.numel() * result.element_size(),
                        ]
                    )
                    metrics.append([name, workload, "1/1", "0", "exact"])
    finally:
        torch.set_num_threads(original_threads)
    report = [
        "# Continuous IK search overhead",
        "",
        f"CPU: {_cpu_name()}; Python {platform.python_version()}; Torch {torch.__version__}; float32; B=1; 1 CPU thread.",
        f"Median completed-call wall latency; 2 warmups and {args.repeats} repeats. Inputs and correctness checks are outside timing. Python peak allocation is measured separately and excludes native tensor/allocator memory. Output bytes are reported separately.",
        "This deterministic prismatic solver measures branch-search overhead only. No native robot solver, simulation, CUDA workload, or application speedup is claimed.",
        "",
        _table(
            ["Algorithm", "Workload", "Median ms", "Peak Python KiB", "Output bytes"],
            timings,
        ),
        "",
        _table(
            [
                "Algorithm",
                "Workload",
                "Successful paths",
                "FK position error",
                "Reference equality",
            ],
            metrics,
        ),
        "",
        _table(
            ["Algorithm", "Successful workloads", "Success rate"],
            [
                [name, f"{ok}/{total}", f"{ok / total:.0%}"]
                for name, (ok, total) in sorted(
                    totals.items(), key=lambda item: -item[1][0] / item[1][1]
                )
            ],
        ),
    ]
    if args.reference_source:
        report.extend(
            [
                "",
                f"Reference source SHA256: `{hashlib.sha256(args.reference_source.read_bytes()).hexdigest()}`. Its observer import is disabled to compare numerical work with both measurement scopes inactive.",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
