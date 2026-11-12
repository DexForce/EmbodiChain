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

"""Qualify Newton geometry and measure CPU/CUDA candidate-query costs.

Run: python -m scripts.benchmark.scene.feasibility_collision --output outputs/benchmarks/newton_collision

Uses isolated geometry states, never a physics solver or live scene. Production
Newton CPU queries and captured outcomes form the reference; analytic primitive
cases independently check both implementations. Match rate is collision-classification agreement, not
task success. Contact-band separations are not global minimum distances.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from embodichain.lab.sim.scene.feasibility import CollisionBody, NewtonCollisionChecker
from ._newton_collision_probe import NewtonCollisionProbe


def _pose(x: float = 0, y: float = 0, z: float = 0.4) -> list:
    """Construct an arena transform used by independent geometric fixtures."""
    result = np.eye(4)
    result[:3, 3] = (x, y, z)
    return result.tolist()


def _fixtures() -> list[tuple[str, tuple[CollisionBody, ...], list, list[dict]]]:
    """Provide analytic primitive and transformed-containment cases."""
    fixtures = []
    for kind, dimensions, halfwidth in (
        ("box", (0.2, 0.2, 0.2), 0.1),
        ("sphere", (0.1,), 0.1),
        ("capsule", (0.1, 0.2), 0.1),
        ("cylinder", (0.1, 0.2), 0.1),
    ):
        bodies = (
            CollisionBody("robot", "robot", kind, dimensions, robot=True),
            CollisionBody("obstacle", "obstacle", "box", (0.2, 0.2, 0.2)),
            CollisionBody("floor", "floor", "halfspace", (0, 0, 1, 0)),
        )
        samples = []
        for separation in (-0.02, 0.02, 0.2):
            for allowed in (False, True):
                samples.append(
                    {
                        "frames": {
                            "robot": _pose(),
                            "obstacle": _pose(halfwidth + 0.1 + separation),
                            "floor": _pose(z=0),
                        },
                        "options": {
                            "allowed_contacts": (
                                [("robot", "obstacle")] if allowed else []
                            )
                        },
                        "expected": separation > 0 or allowed,
                    }
                )
        samples.append(
            {
                "frames": {
                    "robot": _pose(z=-0.15),
                    "obstacle": _pose(1),
                    "floor": _pose(z=0),
                },
                "options": {},
                "expected": False,
            }
        )
        fixtures.append(("analytic_" + kind, bodies, [], samples))
    # Non-symmetric local transforms expose root/local composition mistakes.
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler("xyz", (0.3, -0.2, 0.7)).as_matrix()
    rotation[:3, 3] = (0.1, -0.05, 0.07)
    bodies = (
        CollisionBody(
            "robot", "root", "box", (0.2, 0.1, 0.15), local_pose=rotation, robot=True
        ),
        CollisionBody("other", "other", "sphere", (0.04,)),
    )
    samples = []
    for yaw in (-0.8, 0.4, 1.2):
        root = np.eye(4)
        root[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        root[:3, 3] = (0.3, -0.2, 0.4)
        center = root @ rotation
        samples.append(
            {
                "frames": {"root": root.tolist(), "other": center.tolist()},
                "options": {},
                "expected": False,
            }
        )
    fixtures.append(("local_transform_containment", bodies, [], samples))
    return fixtures


def _fixture_name(path: Path) -> str:
    """Identify each input path even when captures share the same filename."""
    name = (
        path.stem if path.stem.startswith("fixture-") else "fixture-" + path.parent.name
    )
    suffix = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    return name + "-" + suffix


def _load_fixture(path: Path) -> tuple:
    """Load canonical physical descriptors and observed frame/permission samples."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) or not data[key]
        for key in ("bodies", "samples")
    ):
        raise ValueError("Fixture requires non-empty physical bodies and query samples")
    return (
        _fixture_name(path),
        tuple(
            (
                replace(CollisionBody(**body), kind="convex")
                if body["kind"] == "mesh"
                and data.get("shape_hints", {}).get(body["collision_id"]) == "CONVEX"
                else CollisionBody(**body)
            )
            for body in data["bodies"]
        ),
        data["ignored_pairs"],
        data["samples"],
        data.get("shape_hints", {}),
    )


def _sync(device: str) -> None:
    """Synchronize the backend that actually owns collision work."""
    if device.startswith("cuda"):
        wp.synchronize_device(device)


def _cpu_model() -> str:
    """Identify the CPU when platform.processor() is empty in a container."""
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _latency(operation: Callable, device: str, warmup: int, repeats: int) -> dict:
    """Measure completed-call wall time, excluding initialization and compilation."""
    for _ in range(warmup):
        value = operation()
        _sync(device)
        del value
    values = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        value = operation()
        _sync(device)
        values.append((time.perf_counter() - start) * 1000)
        del value
    return {
        "cost_time_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _agreement(reference: np.ndarray, separation: np.ndarray) -> dict:
    """Keep float32 boundary ambiguity separate instead of relaxing acceptance."""
    tolerance = 2e-5
    valid = np.isfinite(separation) | np.isposinf(separation)
    ambiguous = np.isfinite(separation) & (np.abs(separation) <= tolerance)
    passed = separation > tolerance
    matched = (reference == passed) & ~ambiguous & valid
    return {
        "success_rate": float(matched.mean()),
        "matched": int(matched.sum()),
        "total": len(reference),
        "ambiguous": int(ambiguous.sum()),
        "invalid_outputs": int((~valid).sum()),
        "false_negative": int((~reference & passed & ~ambiguous).sum()),
        "false_positive": int((reference & ~passed & ~ambiguous).sum()),
        "outside_detection_band": int(np.isinf(separation).sum()),
        "boundary_tolerance_m": tolerance,
    }


def _qualification_summary(
    results: list[dict], qualification: list[dict]
) -> list[dict]:
    """Rank only algorithms qualified on the complete requested fixture set."""
    fixtures = {row["fixture"] for row in (*results, *qualification)}
    algorithms = {row["algorithm"] for row in results}
    grouped = {}
    for row in qualification:
        if row.get("scope") == "full_corpus_qualification":
            fixtures.add(row["fixture"])
            algorithms.add(row["algorithm"])
            completed = grouped.setdefault(row["algorithm"], {})
            if row["fixture"] in completed:
                raise ValueError(
                    "Duplicate fixture qualification for " + row["algorithm"]
                )
            completed[row["fixture"]] = row
        elif row.get("scope", "").startswith("cuda_graph"):
            algorithms.add("newton_cuda_graph")
    summaries = []
    for name in sorted(algorithms):
        completed = grouped.get(name, {})
        matched = sum(row["matched"] for row in completed.values())
        total = sum(row["total"] for row in completed.values())
        complete = bool(total) and set(completed) == fixtures
        summaries.append(
            {
                "algorithm": name,
                "matched": matched,
                "total": total,
                "coverage": f"{len(completed)}/{len(fixtures)}",
                "status": "complete" if complete else "incomplete",
                "success_rate": matched / total if complete else None,
            }
        )
    return sorted(
        summaries,
        key=lambda row: -(
            row["success_rate"] if row["success_rate"] is not None else -1
        ),
    )


def _markdown(
    path: Path, results: list[dict], metadata: dict, qualification: list[dict]
) -> None:
    """Write the required timing, accuracy and complete algorithm tables."""
    lines = [
        "# Newton collision-query evaluation",
        "",
        "## Time & Memory",
        "",
        "| fixture | batch | algorithm / scope | cost_time_ms | setup_s | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        for timing in r["timing"]:
            lines.append(
                f"| {r['fixture']} | {r['batch']} | {r['algorithm']} / {timing['scope']} | {timing['cost_time_ms']:.4f} | {r['setup_s']:.3f} | {r['memory']['cpu_delta_mb']:.2f} | {r['memory']['gpu_delta_mb']} | {r['memory']['peak_gpu_mb']} |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| fixture | batch | algorithm | success_rate | false_negative | false_positive | ambiguous | outside_band |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        q = r["quality"]
        lines.append(
            f"| {r['fixture']} | {r['batch']} | {r['algorithm']} | {q['success_rate']:.6f} | {q['false_negative']} | {q['false_positive']} | {q['ambiguous']} | {q['outside_detection_band']} |"
        )
    for q in qualification:
        if q.get("scope") == "full_corpus_qualification":
            lines.append(
                f"| {q['fixture']} / full corpus | {q['total']} | {q['algorithm']} | {q['success_rate']:.6f} | {q['false_negative']} | {q['false_positive']} | {q['ambiguous']} | {q['outside_detection_band']} |"
            )
    captured = [
        row
        for row in qualification
        if row.get("scope") == "captured_outcome_comparison"
    ]
    if captured:
        lines += [
            "",
            "## Captured outcomes versus current checker",
            "",
            "| fixture | matched / captured | status changes |",
            "| --- | --- | --- |",
        ]
        for row in captured:
            lines.append(
                f"| {row['fixture']} | {row['matched']} / {row['total']} | {json.dumps(row['changes'], sort_keys=True)} |"
            )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | matched / evaluated | fixture coverage | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(_qualification_summary(results, qualification), 1):
        rate = (
            f"{row['success_rate']:.6f}" if row["success_rate"] is not None else "N/A"
        )
        lines.append(
            f"| {rank if rate != 'N/A' else 'N/A'} | {row['algorithm']} | {rate} | {row['matched']} / {row['total']} | {row['coverage']} | {row['status']} |"
        )
    lines += [
        "",
        "## Measurement scope",
        "",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        "- Success rate means geometric classification agreement with the current production Newton checker, including independent analytic cases. It is not task success or a continuous collision certificate.",
        "- Saved capture statuses are historical evidence, reported separately from agreement with the current checker. A changed classification does not abort the report; independent analytic expectations remain assertions.",
        "- Near-zero float32 separations (2e-5 m band) remain ambiguous and count as unmatched; they are never silently accepted.",
        "- Newton's infinity means no active pair in the configured detection band, not exact global clearance. Only declared native convex/primitive shapes are benchmarked; unsupported triangle meshes are covered by the production rejection tests.",
        "- resident: pre-uploaded transforms/masks, collision generation plus device reduction; excludes host packing and result readback.",
        "- host-to-host: frame dictionaries and phase permissions through packing, H2D, collision/reduction and owned D2H results. FK and task execution are outside both backend scopes.",
        "- Each latency row repeats a fixed representative query/batch; it is not the frame-distribution average. Workload profiles separately measure cumulative live-check time, so microbenchmark speedups must not be presented as task speedups.",
        "- CPU memory is retained process RSS growth during setup, not a transient peak. GPU delta is retained Warp pool growth during setup; peak is cumulative process Warp-pool high-water, not per-call peak. Native allocations outside that pool and JIT caches are not covered.",
        "- CUDA Graph setup includes capture, instantiation and first replay in addition to base query setup. CUDA Graph native allocations outside the Warp pool remain unmeasured.",
        "- Isolation checks use independent untouched states; no World, physics solver, reset or simulation stepping is used.",
    ]
    errors = [
        row
        for row in qualification
        if row.get("scope")
        not in {"full_corpus_qualification", "captured_outcome_comparison"}
    ]
    if errors:
        lines += ["", "## Incomplete checks / errors", ""]
        lines.extend(f"- {json.dumps(error, sort_keys=True)}" for error in errors)
    path.write_text("\n".join(lines) + "\n")


def run_all_benchmarks(args: argparse.Namespace) -> dict:
    """Compare equal physical inputs and keep setup/resident/host scopes separate."""
    torch.set_num_threads(args.threads)
    wp.init()
    devices = ["cpu"]
    if wp.is_cuda_available():
        wp.set_mempool_enabled(args.cuda_device, True)
        devices.append(args.cuda_device)
    fixtures = [(*fixture, {}) for fixture in _fixtures()]
    failures = []
    fixture_paths = list(args.fixture)
    fixture_sources = {}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect_config:
        from ._feasibility_profile import capture_workload

        for i, source in enumerate(args.collect_config):
            print(f"Capturing canonical task workload: {source}", flush=True)
            capture_root = args.output / f"native-input-{i}"
            captured = capture_workload(source.resolve(), capture_root)
            if captured is None:
                failures.append(
                    {
                        "fixture": f"capture-{i}:{source.name}",
                        "scope": "workload_capture_error",
                        "source": str(source.resolve()),
                        "reason": "No physical geometry/query samples captured; see "
                        + str(capture_root / "profile.json"),
                    }
                )
            else:
                fixture_paths.append(captured)
    for path in fixture_paths:
        try:
            fixture = _load_fixture(path)
        except ValueError as error:
            failures.append(
                {
                    "fixture": _fixture_name(path),
                    "scope": "fixture_load_error",
                    "source": str(path.resolve()),
                    "reason": str(error),
                }
            )
            continue
        fixtures.append(fixture)
        fixture_sources[fixture[0]] = str(path.resolve())
    if len({fixture[0] for fixture in fixtures}) != len(fixtures):
        raise ValueError("Duplicate fixture input paths are not allowed")
    metadata = {
        "seed": 1701,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batches": args.batches,
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "newton",
                "warp-lang",
                "dexsim-engine",
                "torch",
            )
        },
        "cuda_device": (
            str(wp.get_device(args.cuda_device)) if len(devices) > 1 else "unavailable"
        ),
        "gpu_name": wp.get_device(args.cuda_device).name if len(devices) > 1 else None,
        "cuda_driver_version": (
            wp.get_cuda_driver_version() if len(devices) > 1 else None
        ),
        "dtype": "Newton float32 geometry; same supplied frame inputs",
        "detection_band_m": args.detection_gap,
        "fixture_sources": fixture_sources,
    }
    results = []
    process = psutil.Process()
    for name, bodies, ignored, corpus, hints in fixtures:
        print(
            f"Geometry: {name}; shapes={len(bodies)} samples={len(corpus)}", flush=True
        )
        runtime_rss = process.memory_info().rss
        runtime_start = time.perf_counter()
        baseline = NewtonCollisionChecker(bodies, ignored_pairs=ignored)
        runtime_setup = time.perf_counter() - runtime_start
        runtime_memory = {
            "cpu_delta_mb": (process.memory_info().rss - runtime_rss) / 2**20,
            "gpu_delta_mb": None,
            "peak_gpu_mb": None,
        }
        oracle = []
        captured_statuses = []
        for sample in corpus:
            status = baseline.check(sample["frames"], **sample.get("options", {}))[
                0
            ].status
            accepted = status == "passed"
            if "expected" in sample and accepted != sample["expected"]:
                baseline.close()
                raise AssertionError(f"Independent analytic oracle failed: {name}")
            oracle.append(accepted)
            if "status" in sample:
                captured_statuses.append((sample["status"], status))
        if captured_statuses:
            failures.append(
                {
                    "fixture": name,
                    "scope": "captured_outcome_comparison",
                    "matched": sum(old == new for old, new in captured_statuses),
                    "total": len(captured_statuses),
                    "changes": dict(
                        Counter(
                            f"{old}->{new}"
                            for old, new in captured_statuses
                            if old != new
                        )
                    ),
                }
            )
        batches = args.batches if name.startswith("fixture-") else [1, 8]
        for batch in batches:
            indices = np.arange(batch) % len(corpus)
            samples = [corpus[i] for i in indices]
            expected = np.asarray(oracle)[indices]

            def runtime_call():
                return [
                    baseline.check(s["frames"], **s.get("options", {})) for s in samples
                ]

            # Every correctness case is evaluated even when timing a small prefix.
            for device in devices:
                gc.collect()
                rss = process.memory_info().rss
                before = (
                    wp.get_mempool_used_mem_current(device) if device != "cpu" else None
                )
                start = time.perf_counter()
                probe = NewtonCollisionProbe(
                    bodies,
                    ignored,
                    batch=batch,
                    device=device,
                    detection_gap=args.detection_gap,
                )
                packed = probe.pack(samples)
                probe.upload(packed)
                probe.run()
                _sync(device)
                setup = time.perf_counter() - start
                observed = probe.download()
                probe.verify_isolation()
                probe.verify_buffers()
                quality = _agreement(expected, observed)
                memory = {
                    "cpu_delta_mb": (process.memory_info().rss - rss) / 2**20,
                    "gpu_delta_mb": (
                        (wp.get_mempool_used_mem_current(device) - before) / 2**20
                        if before is not None
                        else None
                    ),
                    "peak_gpu_mb": (
                        wp.get_mempool_used_mem_high(device) / 2**20
                        if before is not None
                        else None
                    ),
                    "owned_array_bytes": probe.persistent_bytes(),
                }
                for scope, call in (
                    ("resident", probe.run),
                    ("host-to-host", lambda: probe.check(samples)),
                ):
                    timing = _latency(call, device, args.warmup, args.repeats)
                    timing["scope"] = scope
                    if scope == "resident":
                        entry = {
                            "fixture": name,
                            "batch": batch,
                            "algorithm": "newton_" + device,
                            "setup_s": setup,
                            "quality": quality,
                            "memory": memory,
                            "timing": [],
                        }
                        results.append(entry)
                    entry["timing"].append(timing)
                print(
                    f"  B={batch} {device}: {entry['timing'][-1]['cost_time_ms']:.3f} ms host-to-host; match={quality['success_rate']:.3f}",
                    flush=True,
                )
                if device != "cpu":
                    try:
                        probe.upload(packed)
                        probe.run()
                        _sync(device)
                        graph_start = time.perf_counter()
                        graph_rss = process.memory_info().rss
                        graph_pool = wp.get_mempool_used_mem_current(device)
                        with wp.ScopedCapture(device=device) as capture:
                            probe.run()
                        graph = capture.graph
                        wp.capture_launch(graph)
                        _sync(device)
                        graph_setup = time.perf_counter() - graph_start
                        graph_memory = {
                            **memory,
                            "cpu_delta_mb": memory["cpu_delta_mb"]
                            + (process.memory_info().rss - graph_rss) / 2**20,
                            "gpu_delta_mb": memory["gpu_delta_mb"]
                            + (wp.get_mempool_used_mem_current(device) - graph_pool)
                            / 2**20,
                            "peak_gpu_mb": wp.get_mempool_used_mem_high(device) / 2**20,
                            "graph_native_allocations": "unmeasured outside Warp pool",
                        }

                        def replay():
                            wp.capture_launch(graph)

                        def graph_host():
                            probe.upload(probe.pack(samples))
                            replay()
                            return probe.download()

                        # Change both geometry and permissions after capture to
                        # reject a graph that accidentally reuses stale inputs.
                        alternate = [corpus[(i + 1) % len(corpus)] for i in indices]
                        probe.upload(probe.pack(alternate))
                        replay()
                        alternate_graph = probe.download()
                        alternate_direct = probe.check(alternate)
                        probe.verify_buffers()
                        if not np.allclose(
                            alternate_graph, alternate_direct, atol=2e-5, rtol=0
                        ):
                            raise AssertionError(
                                "Captured query reused stale geometry or phase masks"
                            )
                        probe.upload(packed)

                        graph_timings = []
                        for scope, call in (
                            ("resident", replay),
                            ("host-to-host", graph_host),
                        ):
                            timing = _latency(call, device, args.warmup, args.repeats)
                            timing["scope"] = scope
                            graph_timings.append(timing)
                        if not np.allclose(
                            graph_host(), observed, atol=2e-5, rtol=0, equal_nan=False
                        ):
                            raise AssertionError(
                                "Captured query changed collision separations"
                            )
                        results.append(
                            {
                                **entry,
                                "algorithm": "newton_cuda_graph",
                                "timing": graph_timings,
                                "setup_s": setup + graph_setup,
                                "graph_capture_s": graph_setup,
                                "memory": graph_memory,
                            }
                        )
                        del replay, graph_host, graph, capture
                    except Exception as error:
                        failures.append(
                            {
                                "fixture": name,
                                "batch": batch,
                                "scope": "cuda_graph",
                                "reason": repr(error),
                            }
                        )
                del probe, packed
                gc.collect()
            timing = _latency(runtime_call, "cpu", args.warmup, args.repeats)
            timing["scope"] = "host-to-host"
            results.append(
                {
                    "fixture": name,
                    "batch": batch,
                    "algorithm": "newton_runtime_cpu",
                    "setup_s": runtime_setup,
                    "quality": {
                        "success_rate": 1.0,
                        "matched": batch,
                        "total": batch,
                        "ambiguous": 0,
                        "false_negative": 0,
                        "false_positive": 0,
                        "outside_detection_band": 0,
                    },
                    "memory": runtime_memory,
                    "timing": [timing],
                }
            )
        # A one-candidate timing must not hide missed geometry later in the corpus.
        for device in devices:
            probe = NewtonCollisionProbe(
                bodies,
                ignored,
                batch=1,
                device=device,
                detection_gap=args.detection_gap,
            )
            observed = []
            for sample in corpus:
                observed.append(probe.check([sample])[0])
                probe.verify_buffers()
            observed = np.asarray(observed)
            q = _agreement(np.asarray(oracle), observed)
            failures.append(
                {
                    "fixture": name,
                    "scope": "full_corpus_qualification",
                    "device": device,
                    "algorithm": "newton_" + device,
                    **q,
                }
            )
            if device != "cpu":
                try:
                    with wp.ScopedCapture(device=device) as qualification_capture:
                        probe.run()
                    qualification_graph = qualification_capture.graph
                    graph_observed = []
                    for sample in corpus:
                        probe.upload(probe.pack([sample]))
                        wp.capture_launch(qualification_graph)
                        graph_observed.append(probe.download()[0])
                        probe.verify_buffers()
                    q = _agreement(np.asarray(oracle), np.asarray(graph_observed))
                    failures.append(
                        {
                            "fixture": name,
                            "scope": "full_corpus_qualification",
                            "device": device,
                            "algorithm": "newton_cuda_graph",
                            **q,
                        }
                    )
                    del qualification_graph, qualification_capture
                except Exception as error:
                    failures.append(
                        {
                            "fixture": name,
                            "scope": "cuda_graph_qualification_error",
                            "reason": repr(error),
                        }
                    )
            del probe
            gc.collect()
        failures.append(
            {
                "fixture": name,
                "scope": "full_corpus_qualification",
                "device": "cpu",
                "algorithm": "newton_runtime_cpu",
                "success_rate": 1.0,
                "matched": len(corpus),
                "total": len(corpus),
                "ambiguous": 0,
                "false_negative": 0,
                "false_positive": 0,
                "outside_detection_band": 0,
            }
        )
        baseline.close()
        payload = {"metadata": metadata, "results": results, "qualification": failures}
        (args.output / "results.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False)
        )
        _markdown(args.output / "report.md", results, metadata, failures)
    print(f"Markdown report saved: {args.output / 'report.md'}", flush=True)
    return payload


def main() -> None:
    """Run the bounded collision experiment with optional captured native geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/newton_collision")
    )
    parser.add_argument("--fixture", action="append", type=Path, default=[])
    parser.add_argument(
        "--collect-config",
        action="append",
        type=Path,
        default=[],
        help="Run a bounded canonical task and capture its physical geometry/query workload",
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 256])
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--detection-gap", type=float, default=0.03)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        any(not 1 <= n <= 256 for n in args.batches)
        or min(args.repeats, args.warmup, args.threads) < 1
        or not 0 < args.detection_gap < 1
    ):
        parser.error(
            "Require B=1..256, positive repeat/warmup/threads and gap in (0,1)"
        )
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
