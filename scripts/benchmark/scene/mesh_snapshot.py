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

"""Compare CPU mesh-envelope kernels with their frozen source implementations.

Run: python -m scripts.benchmark.scene.mesh_snapshot --output report.md
This measures tensor envelope operations only, not native geometry/collision.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
import hashlib
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

import torch

from embodichain.compute.geometry.mesh_bounds import (
    aabb_sphere_cover,
    transform_vertex_bounds,
)


def _load_function(path: Path, name: str) -> Callable:
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
        ]
    )


def main() -> None:
    """Measure equivalent completed CPU calls and write one three-table report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/mesh_snapshot.md")
    )
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    algorithms = {"shared": (transform_vertex_bounds, aabb_sphere_cover)}
    references = []
    if args.reference_checkout:
        root = args.reference_checkout / "embodichain/gen_sim/task_engine/_task_program"
        paths = [root / "scene_geometry.py", root / "drawer_curobo.py"]
        algorithms["frozen source"] = (
            _load_function(paths[0], "world_bounds"),
            _load_function(paths[1], "payload_spheres"),
        )
        references = [
            f"{p.name}: {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in paths
        ]
    timings = []
    metrics = []
    success = {name: [0, 0] for name in algorithms}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            for count in (8, 4096, 100000):
                points = torch.rand(
                    (count, 3), generator=torch.Generator().manual_seed(17)
                )
                pose = torch.eye(4)[None]
                pose[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
                expected_bounds = (
                    points.amin(0)[None] + pose[:, :3, 3],
                    points.amax(0)[None] + pose[:, :3, 3],
                )
                for name, (bounds, spheres) in algorithms.items():
                    for kind, operation in (
                        ("bounds", lambda: bounds(points[None], pose)),
                        ("sphere cover", lambda: spheres(points)),
                    ):
                        result = operation()
                        if kind == "bounds":
                            error = max(
                                float((a - b).abs().max())
                                for a, b in zip(result, expected_bounds)
                            )
                            valid = error <= 1e-6
                            output_bytes = sum(
                                v.numel() * v.element_size() for v in result
                            )
                        else:
                            low, high = points.amin(0), points.amax(0)
                            expected_radius = (
                                torch.linalg.vector_norm((high - low) / 3) / 2
                            )
                            error = float((result[:, 3] - expected_radius).abs().max())
                            valid = result.shape == (27, 4) and error <= 1e-6
                            output_bytes = result.numel() * result.element_size()
                        if not valid:
                            raise RuntimeError(
                                f"{name}/{kind} failed its envelope oracle"
                            )
                        success[name][0] += int(valid)
                        success[name][1] += 1
                        for _ in range(2):
                            operation()
                        samples = []
                        for _ in range(args.repeats):
                            start = time.perf_counter()
                            result = operation()
                            samples.append((time.perf_counter() - start) * 1000)
                            del result
                        tracemalloc.start()
                        result = operation()
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        timings.append(
                            [
                                name,
                                kind,
                                count,
                                f"{statistics.median(samples):.4f}",
                                f"{peak/1024:.2f}",
                                output_bytes,
                            ]
                        )
                        metrics.append([name, kind, count, "1/1", f"{error:.3g}"])
    finally:
        torch.set_num_threads(previous)
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in (cpuinfo.read_text().splitlines() if cpuinfo.is_file() else ())
            if line.startswith("model name")
        ),
        platform.machine(),
    )
    report = [
        "# Mesh-envelope CPU benchmark",
        "",
        f"CPU {cpu}; Python {platform.python_version()}; Torch {torch.__version__}; float32; B=1; one CPU thread.",
        f"Completed-call median after two warmups and {args.repeats} repeats. Inputs/oracles are outside timing. Python allocation peaks are measured separately and exclude native tensor memory; output bytes are reported separately.",
        "The shared sphere helper additionally rejects nonfinite/empty input. Geometry capture is cold setup; projection consumes caller-owned snapshots. No native cooked-shape, CCD, GPU, or end-to-end planning performance is claimed.",
        "",
        _table(
            [
                "Algorithm",
                "Operation",
                "Vertices",
                "Median ms",
                "Peak Python KiB",
                "Output bytes",
            ],
            timings,
        ),
        "",
        _table(
            ["Algorithm", "Operation", "Vertices", "Success", "Max oracle error"],
            metrics,
        ),
        "",
        _table(
            ["Algorithm", "Successful workloads", "Success rate"],
            [
                [name, f"{ok}/{total}", f"{ok/total:.0%}"]
                for name, (ok, total) in success.items()
            ],
        ),
        "",
        *references,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
