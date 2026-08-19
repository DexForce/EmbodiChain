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

"""Measure deterministic E1/E2 scene feasibility and graph compilation.

This CPU benchmark exercises the contract path before simulator motion. It
checks that Scene Engine v1 output adapts successfully, required capabilities
are executable, and E1/E2 compile to action graphs containing pickup,
held-object motion, and placement.

Run this module with ``--iterations 100`` for the default benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
import tracemalloc
from types import SimpleNamespace
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain.task_contracts import TASK_CONTRACTS
from embodichain.gen_sim.action_engine.tasks import TaskFactory, instantiate_seed_graph
from embodichain.gen_sim.task_engine.scene import (
    FeasibilityBroker,
    SceneEngineV1Adapter,
)

__all__ = ["BenchmarkResult", "run_benchmark"]


@dataclass(frozen=True)
class BenchmarkResult:
    """One scenario's contract latency, memory, and correctness metrics."""

    scenario: str
    iterations: int
    elapsed_seconds: float
    peak_bytes: int
    success_count: int
    feasibility_status: str
    action_count: int
    unknown_checks: int
    runtime_probe_checks: int

    @property
    def success_rate(self) -> float:
        """Return successful iterations divided by all iterations."""
        return self.success_count / self.iterations

    @property
    def mean_milliseconds(self) -> float:
        """Return mean contract latency in milliseconds."""
        return self.elapsed_seconds * 1000.0 / self.iterations


def run_benchmark(
    *,
    iterations: int = 100,
    output_dir: str | Path = "outputs/benchmarks",
) -> tuple[tuple[BenchmarkResult, ...], Path]:
    """Run E1/E2 contract regressions and write one Markdown report."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer.")
    results = tuple(
        _benchmark_scenario(task_type, iterations=iterations)
        for task_type in ("E1", "E2")
    )
    report = _write_report(results, output_dir=Path(output_dir))
    return results, report


def _benchmark_scenario(task_type: str, *, iterations: int) -> BenchmarkResult:
    task, requirements = _generated_task(task_type)
    bindings = {
        item["role_id"]: f"{task_type.lower()}_{item['role_id']}"
        for item in requirements["objects"]
    }
    manifest = _static_manifest(task_type, requirements, bindings)
    candidate, reference_bindings = _candidate(
        task_type,
        task,
        requirements,
        bindings,
    )
    registry = build_atomic_capability_registry()
    broker = FeasibilityBroker()
    task_actions = {
        name: contract.core_actions for name, contract in TASK_CONTRACTS.items()
    }
    success_count = 0
    last_report: dict[str, Any] = {}
    last_graph: dict[str, Any] = {}

    tracemalloc.start()
    start = perf_counter()
    try:
        for _ in range(iterations):
            last_report = broker.assess(
                candidate,
                reference_bindings,
                manifest,
                capability_catalog=registry.catalog(),
                task_actions=task_actions,
            )
            last_graph = instantiate_seed_graph(task, bindings, registry=registry)
            actions = {
                str(node.get("atomic_action"))
                for node in last_graph["nodes"]
                if node.get("atomic_action")
            }
            if (
                last_report["status"] != "contradicted"
                and {"PickUp", "MoveHeldObject", "Place"} <= actions
            ):
                success_count += 1
    finally:
        elapsed = perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    action_count = sum(
        bool(node.get("atomic_action")) for node in last_graph.get("nodes", ())
    )
    return BenchmarkResult(
        scenario=task_type,
        iterations=iterations,
        elapsed_seconds=elapsed,
        peak_bytes=peak_bytes,
        success_count=success_count,
        feasibility_status=str(last_report.get("status", "unknown")),
        action_count=action_count,
        unknown_checks=int(last_report.get("summary", {}).get("unknown", 0)),
        runtime_probe_checks=int(
            last_report.get("summary", {}).get("runtime_probe", 0)
        ),
    )


def _generated_task(task_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    factory = TaskFactory(seed=41, executable_only=True)
    for index in range(100):
        task, requirements = factory.generate("L1", index)
        if task["task_instances"][0]["task_type"] == task_type:
            return task, requirements
    raise RuntimeError(f"Could not generate deterministic {task_type} fixture.")


def _static_manifest(
    task_type: str,
    requirements: dict[str, Any],
    bindings: dict[str, str],
) -> dict[str, Any]:
    planner_objects = []
    runtime_objects = []
    for index, requirement in enumerate(requirements["objects"]):
        role_id = str(requirement["role_id"])
        uid = bindings[role_id]
        role = "rigid_object"
        planner_objects.append(
            {
                "uid": uid,
                "source_uid": f"{uid}_0",
                "role": role,
                "name": uid,
                "description": f"Synthetic {task_type} benchmark object.",
                "category": str(requirement["category"]),
                "color": requirement.get("attributes", {}).get("color"),
                "shape": {"shape_type": "Mesh", "fpath": f"/{uid}.glb"},
                "init_pos": [0.15 * index, 0.0, 0.7],
                "init_rot": [90.0, 0.0, 0.0] if task_type == "E2" else [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
                "affordances": list(requirement["affordances"]),
                "initial_state": dict(requirement["initial_state"]),
                "attributes": dict(requirement["attributes"]),
            }
        )
        runtime_objects.append(
            {
                "uid": uid,
                "shape": {"shape_type": "Mesh", "fpath": f"/{uid}.glb"},
                "attrs": {"mass": 0.1},
                "body_type": "dynamic",
            }
        )
    prepared = SimpleNamespace(
        source_config_path=Path("/synthetic/scene_config.json"),
        planner_objects=tuple(planner_objects),
        background=(),
        rigid_objects=tuple(runtime_objects),
        articulations=(),
        asset_hashes={
            uid: uid.encode().hex().ljust(64, "0")[:64] for uid in bindings.values()
        },
    )
    return SceneEngineV1Adapter().adapt_prepared_scene(
        prepared,
        source_format="benchmark",
        robot_profile="dual_franka",
    )


def _candidate(
    task_type: str,
    task: dict[str, Any],
    requirements: dict[str, Any],
    bindings: dict[str, str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    references = []
    reference_bindings = {}
    for index, requirement in enumerate(requirements["objects"]):
        role_id = str(requirement["role_id"])
        role = "object" if index == 0 else "target"
        reference_id = f"task_01.{role}"
        references.append(
            {
                "reference_id": reference_id,
                "role": role,
                "source_structure": "rigid_object",
                "affordances": list(requirement["affordances"]),
                "initial_state": dict(requirement["initial_state"]),
                "attributes": dict(requirement["attributes"]),
            }
        )
        reference_bindings[reference_id] = [bindings[role_id]]
    return (
        {
            "candidate_id": "candidate_01",
            "draft": {
                "task_id": task["task_id"],
                "steps": [{"id": "task_01", "task_type": task_type}],
            },
            "scene_request": {"references": references},
        },
        reference_bindings,
    )


def _write_report(
    results: tuple[BenchmarkResult, ...],
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"e1_e2_scene_action_{timestamp}.md"
    performance_rows = [
        {
            "Scenario": item.scenario,
            "Iterations": item.iterations,
            "Total ms": f"{item.elapsed_seconds * 1000.0:.3f}",
            "Mean ms": f"{item.mean_milliseconds:.3f}",
            "Peak KiB": f"{item.peak_bytes / 1024.0:.1f}",
        }
        for item in results
    ]
    metric_rows = [
        {
            "Scenario": item.scenario,
            "Success rate": f"{item.success_rate:.3f}",
            "Feasibility": item.feasibility_status,
            "Actions": item.action_count,
            "Unknown checks": item.unknown_checks,
            "Runtime probes": item.runtime_probe_checks,
        }
        for item in results
    ]
    leaderboard_rows = [
        {
            "Rank": rank,
            "Scenario": item.scenario,
            "Success rate": f"{item.success_rate:.3f}",
            "Mean ms": f"{item.mean_milliseconds:.3f}",
        }
        for rank, item in enumerate(
            sorted(
                results,
                key=lambda value: (-value.success_rate, value.mean_milliseconds),
            ),
            start=1,
        )
    ]
    lines = [
        "# E1/E2 Scene-Action Contract Benchmark",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
        *_table(performance_rows),
        "",
        "## Success & Other Metrics",
        "",
        *_table(metric_rows),
        "",
        "## Leaderboard",
        "",
        *_table(leaderboard_rows),
        "",
        "## Notes",
        "",
        "- This benchmark covers deterministic contracts and graph compilation, not GPU motion execution.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _table(rows: list[dict[str, object]]) -> list[str]:
    headers = list(rows[0])
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[
            "| " + " | ".join(str(row[header]) for header in headers) + " |"
            for row in rows
        ],
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark E1/E2 scene-action contract stability."
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    return parser


def main() -> int:
    """Run from the command line and print the generated report path."""
    args = _build_parser().parse_args()
    results, report = run_benchmark(
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    for result in results:
        print(
            f"{result.scenario}: success={result.success_rate:.3f}, "
            f"mean={result.mean_milliseconds:.3f} ms, "
            f"peak={result.peak_bytes / 1024.0:.1f} KiB"
        )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
