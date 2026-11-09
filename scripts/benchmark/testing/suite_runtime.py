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

"""Measure complete CI pytest lanes without changing a measured checkout.

Run: python scripts/benchmark/testing/suite_runtime.py --root CHECKOUT
    --python PYTHON --output NEW_DIRECTORY [--cpu-workers 8]

Use the same metric-plugin file for baseline/candidate samples. Each lane runs
through its checkout's real CI runner, including native discovery and teardown.
Independent exporter/frontend commands can be supplied with --extra-plan as a
JSON list of {"name": ..., "argv": [...], "cwd": "relative/to/checkout"} objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

__all__ = ["main"]

_ENTRY = """
import importlib.util, inspect, json, sys
from pathlib import Path
root, plan_path, lane, output, workers, history, freeze_imports = sys.argv[1:]
root=Path(root); output=Path(output)
spec=importlib.util.spec_from_file_location('measured_runner',root/'scripts/ci/run_test_plan.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
build=runner.build_commands
supports_workers='cpu_workers' in inspect.signature(build).parameters
def instrumented(plan, **kwargs):
    if supports_workers: kwargs['cpu_workers']=int(workers)
    elif int(workers)!=4: raise RuntimeError('This runner only supports four CPU workers')
    commands=build(plan,**kwargs)
    for name, command in commands:
        command.extend(['-p','suite_metrics','--ci-metrics-dir',str(output/'metrics'/name)])
        if history and name in {'fast','slow-fast'}:
            command.extend(['--ci-duration-history',history,'--no-loadscope-reorder'])
    return commands
runner.build_commands=instrumented
policy={}
if freeze_imports=='1':
    if 'freeze_native_imports' not in inspect.signature(runner.run_plan).parameters:
        raise RuntimeError('This runner does not support native import freezing')
    policy['freeze_native_imports']=True
code=runner.run_plan(json.loads(Path(plan_path).read_text()),root=root,lanes=[lane],
                     no_finalize_resource_lanes=True,**policy)
(output/(lane+'-exit.json')).write_text(json.dumps({'exitcode':code}))
raise SystemExit(code)
"""


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        info["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        info["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    for key in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / key
        info[key] = path.read_text().strip() if path.exists() else None
    query = shutil.which("nvidia-smi")
    if query:
        result = subprocess.run(
            [
                query,
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        info["gpu_query"] = result.stdout if result.returncode == 0 else result.stderr
    return info


def _source(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True)

    diff = git("diff", "HEAD", "--", ".", ":(exclude)docs/source/**")
    untracked = {}
    for name in git("ls-files", "--others", "--exclude-standard").splitlines():
        if not name.startswith("docs/source/"):
            path = root / name
            if path.is_file():
                untracked[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked": untracked,
    }


def _summarize(directory: Path) -> dict[str, Any]:
    selected: set[str] = set()
    phases: dict[str, list[dict[str, Any]]] = {}
    collection_reports: dict[str, str] = {}
    for path in (directory / "collection").glob("*.json"):
        recorded = json.loads(path.read_text())
        collection_reports.update(
            {
                row["nodeid"]: row["outcome"]
                for row in recorded.get("collection_reports", [])
            }
        )
        for item in recorded["collection"]:
            selected.add(item["nodeid"])
    for path in directory.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            phase = json.loads(line)
            phases.setdefault(phase["nodeid"], []).append(phase)
    counts = dict.fromkeys(("passed", "skipped", "failed", "error", "incomplete"), 0)
    for reports in phases.values():
        if any(r["outcome"] == "failed" and r["phase"] != "call" for r in reports):
            status = "error"
        elif any(r["outcome"] == "failed" for r in reports):
            status = "failed"
        elif any(r["outcome"] == "skipped" for r in reports):
            status = "skipped"
        elif any(r["phase"] == "call" for r in reports) and any(
            r["phase"] == "teardown" for r in reports
        ):
            status = "passed"
        else:
            status = "incomplete"
        counts[status] += 1
    return {
        "collection_skipped": sum(
            value == "skipped" for value in collection_reports.values()
        ),
        "collection_errors": sum(
            value == "failed" for value in collection_reports.values()
        ),
        "selected": len(selected),
        "attempted": len(phases),
        **counts,
        "unattempted": len(selected - phases.keys()),
    }


def _report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Test suite runtime",
        "",
        "## Time & Memory",
        "",
        "| lane | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | ---: | --- | --- | --- |",
    ]
    lines.append(
        f"| complete selected flow | {result['seconds']*1000:.3f} | unmeasured | unmeasured | unmeasured |"
    )
    for row in result["steps"]:
        lines.append(
            f"| {row['name']} | {row['seconds']*1000:.3f} | unmeasured | unmeasured | unmeasured |"
        )
    lines.extend(
        [
            "",
            "## Success & Other Metrics",
            "",
            "| lane | selected | passed | skipped | failed/error | unattempted/incomplete | collection skip/error | exit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    selected = passed = 0
    for row in result["steps"]:
        stats = row.get("tests")
        if stats:
            selected += stats["selected"]
            passed += stats["passed"]
            lines.append(
                f"| {row['name']} | {stats['selected']} | {stats['passed']} | {stats['skipped']} | {stats['failed']+stats['error']} | {stats['unattempted']+stats['incomplete']} | {stats['collection_skipped']}/{stats['collection_errors']} | {row['exitcode']} |"
            )
        else:
            lines.append(
                f"| {row['name']} | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | {row['exitcode']} |"
            )
    rate = f"{passed/selected:.6f}" if selected else "unmeasured"
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| rank | algorithm | overall_success_rate | wall_seconds |",
            "| --- | --- | --- | ---: |",
            f"| 1 | {result['label']} | {rate} | {result['seconds']:.3f} |",
            "",
            "Fresh process wall time includes startup, collection, native discovery, tests, cleanup and exits. "
            "The preexisting CI no-finalize wrapper is identical across variants; test finalizers and pytest hooks run. "
            "Failed lanes are retained and later lanes still execute. Single samples do not establish stable speedups. "
            "Asset/JIT caches are retained. Process RSS high-water marks are in raw metrics; retained deltas, "
            "native-child memory and GPU allocator peaks were not measured. Module-level collection skips/errors are separate from runnable items. In-run metrics aggregation and progress writes are included in outer wall time. Success rate covers selected pytest "
            "cases; additional commands retain their exit status without inventing case counts. "
            "Source identity excludes Sphinx source contents, which this task does not inspect.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Measure requested lanes and preserve all outcomes without retries.

    Args:
        argv: Command arguments, or the process arguments when omitted.

    Returns:
        The first failed step's exit code, or two if measured sources changed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--docs-python", help="SDK-free documentation test interpreter")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory, outside the checkout or ignored by Git",
    )
    parser.add_argument("--label", default="measured")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--freeze-native-imports", action="store_true")
    parser.add_argument(
        "--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    parser.add_argument("--cpu-history", type=Path)
    parser.add_argument(
        "--metrics-plugin",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ci/pytest_metrics.py",
    )
    parser.add_argument(
        "--lane", action="append", choices=["fast", "distributed", "gpu", "sim", "docs"]
    )
    parser.add_argument("--extra-plan", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    if args.cpu_workers < 1:
        parser.error("--cpu-workers must be positive")
    lanes = args.lane or ["fast", "distributed", "gpu", "sim", "docs"]
    extras = json.loads(args.extra_plan.read_text()) if args.extra_plan else []
    names = [*lanes, *(row["name"] for row in extras)]
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in {".", ".."}
        for name in names
    ):
        parser.error("step names must be nonempty file basenames")
    if len(set(names)) != len(names):
        parser.error("step names must be unique across lanes and extra commands")
    if output.is_relative_to(root):
        relative = output.relative_to(root).as_posix() + "/"
        ignored = (
            subprocess.run(
                ["git", "-C", str(root), "check-ignore", "-q", "--", relative]
            ).returncode
            == 0
        )
        if output == root or not ignored:
            parser.error("output must be outside the checkout or ignored by Git")
    output.mkdir(parents=True, exist_ok=False)
    support = output / "support"
    support.mkdir()
    shutil.copyfile(args.metrics_plugin, support / "suite_metrics.py")
    (support / "lane_entry.py").write_text(_ENTRY)
    plan = {
        "version": 1,
        "mode": "full",
        "selectors": ["tests"],
        "lanes": {
            lane: (
                ["tests/docs"]
                if lane == "docs"
                else (
                    ["tests/learning/test_rl_distributed.py"]
                    if lane == "distributed"
                    else ["tests"]
                )
            )
            for lane in lanes
        },
    }
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    before = _source(root)
    env = os.environ.copy()
    env["PATH"] = (
        str(Path(args.python).resolve().parent) + os.pathsep + env.get("PATH", "")
    )
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = os.pathsep.join([str(support), str(root)])
    versions = subprocess.check_output(
        [
            args.python,
            "-c",
            "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':{name:m.version(name) for name in sorted({d.metadata['Name'] for d in m.distributions() if d.metadata['Name']})}}))",
        ],
        text=True,
        env=env,
        cwd=root,
    )
    (output / "runtime.json").write_text(versions)
    if args.docs_python:
        docs_versions = subprocess.check_output(
            [
                args.docs_python,
                "-c",
                "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))",
            ],
            text=True,
            env=env,
            cwd=root,
        )
        (output / "docs-runtime.json").write_text(docs_versions)
    result: dict[str, Any] = {
        "hardware": _hardware(),
        "label": args.label,
        "root": str(root),
        "source_before": before,
        "plugin_sha256": hashlib.sha256(
            (support / "suite_metrics.py").read_bytes()
        ).hexdigest(),
        "execution_policy": {
            "cpu_workers": args.cpu_workers,
            "freeze_native_imports": args.freeze_native_imports,
        },
        "environment": {
            key: env.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "DISPLAY",
                "XDG_RUNTIME_DIR",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CXX",
                "TORCHINDUCTOR_CACHE_DIR",
            ]
        },
        "steps": [],
    }
    commands = [
        (
            lane,
            [
                (
                    args.docs_python
                    if lane == "docs" and args.docs_python
                    else args.python
                ),
                str(support / "lane_entry.py"),
                str(root),
                str(plan_path),
                lane,
                str(output),
                str(args.cpu_workers),
                str(args.cpu_history.resolve()) if args.cpu_history else "",
                "1" if args.freeze_native_imports else "0",
            ],
            root,
        )
        for lane in lanes
    ]
    commands += [
        (row["name"], row["argv"], root / row.get("cwd", ".")) for row in extras
    ]
    start = time.perf_counter()
    for name, command, cwd in commands:
        began = time.perf_counter()
        step_env = env.copy()
        if name in lanes:
            step_env["PATH"] = (
                str(Path(command[0]).resolve().parent) + os.pathsep + env["PATH"]
            )
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=cwd, env=step_env, stdout=log, stderr=subprocess.STDOUT
            )
            (output / "active-process.json").write_text(
                json.dumps({"name": name, "pid": process.pid, "command": command})
            )
            code = process.wait()
        raw_exit = output / f"{name}-exit.json"
        if name in lanes and raw_exit.exists():
            code = json.loads(raw_exit.read_text())["exitcode"]
        row = {
            "name": name,
            "command": command,
            "seconds": time.perf_counter() - began,
            "exitcode": code,
        }
        if name in lanes:
            row["tests"] = _summarize(output / "metrics" / name)
        result["steps"].append(row)
        result["seconds"] = time.perf_counter() - start
        (output / "progress.json").write_text(json.dumps(result, indent=2))
        print(f"{name}: {row['seconds']:.1f}s, exit={code}", flush=True)
    result["seconds"] = time.perf_counter() - start
    result["source_after"] = _source(root)
    result["source_unchanged"] = before == result["source_after"]
    (output / "result.json").write_text(json.dumps(result, indent=2))
    _report(output, result)
    return (
        next((row["exitcode"] for row in result["steps"] if row["exitcode"]), 0)
        if result["source_unchanged"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
