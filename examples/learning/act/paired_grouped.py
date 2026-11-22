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

"""Run and compare the two Push Cube ACT ODS layouts on one GPU pair."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
from statistics import median
import subprocess
import sys
import threading
import time
from typing import Any

from .audit_grouped import _audit, _audit_supply

__all__ = ["main"]

LAYOUTS = {"A": "per-rank", "B": "shared"}


def _smi(kind: str, fields: str) -> list[list[str]]:
    raw = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    return list(csv.reader(raw.splitlines(), skipinitialspace=True))


def _gpus(selection: str) -> list[dict]:
    rows = [
        dict(zip(("index", "uuid", "name", "bus", "memory_mib"), row, strict=True))
        for row in _smi("gpu", "index,uuid,name,pci.bus_id,memory.total")
    ]
    ids = selection.split(",")
    matches = [
        [row for row in rows if device.strip() in (row["index"], row["uuid"])]
        for device in ids
    ]
    if len(matches) != 2 or any(len(match) != 1 for match in matches):
        raise ValueError("Specify two GPU indices or UUIDs")
    cards = [match[0] for match in matches]
    if cards[0] == cards[1] or cards[0]["name"] != cards[1]["name"]:
        raise ValueError("Choose distinct GPUs of the same model")
    active = {uuid for uuid, *_ in _smi("compute-apps", "gpu_uuid,pid")}
    if active.intersection(card["uuid"] for card in cards):
        raise ValueError("The selected GPUs already have compute processes")
    pixels = 20 * 300 * 224 * 224 * 4
    if shutil.disk_usage("/dev/shm").free <= pixels:
        raise ValueError("Available /dev/shm is smaller than the 20-slot camera buffer")
    return cards


def _telemetry(path: Path, uuids: set[str], stop: threading.Event) -> None:
    with path.open("w") as stream:
        while not stop.is_set():
            try:
                gpus = _smi("gpu", "uuid,utilization.gpu,memory.used")
                processes = _smi("compute-apps", "gpu_uuid,pid,used_gpu_memory")
                row = {
                    "monotonic_s": time.monotonic(),
                    "gpus": [gpu for gpu in gpus if gpu[0] in uuids],
                    "processes": [proc for proc in processes if proc[0] in uuids],
                }
            except (OSError, subprocess.SubprocessError, csv.Error) as error:
                row = {"monotonic_s": time.monotonic(), "error": str(error)}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stop.wait(1)


def _execute(
    command: Sequence[str],
    env: dict[str, str],
    log: Path,
    timeout: float,
    telemetry: Path | None = None,
) -> dict:
    start = time.monotonic()
    stop = threading.Event()
    monitor = None
    if telemetry is not None:
        uuids = set(env["CUDA_VISIBLE_DEVICES"].split(","))
        monitor = threading.Thread(
            target=_telemetry, args=(telemetry, uuids, stop), daemon=True
        )
        monitor.start()
    timed_out = False
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                timed_out = isinstance(error, subprocess.TimeoutExpired)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
                if not timed_out:
                    raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "wall_s": time.monotonic() - start,
    }


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _run_seed(
    root: Path,
    group: str,
    seed: int,
    protocol: Path,
    qual: Path,
    env: dict[str, str],
    spec: dict,
    *,
    smoke: bool,
) -> tuple[Path, dict]:
    target = root / group
    target.mkdir(exist_ok=True)
    path = target / str(seed)
    label = f"{group} seed {seed}" + (" smoke" if smoke else "")
    print(f"Starting {label}", flush=True)
    command = [
        sys.executable,
        "-Bu",
        "-m",
        "examples.learning.act.timed_grouped",
        "--nproc-per-node",
        "2",
        "--producer-layout",
        LAYOUTS[group],
        "--seed",
        str(seed),
        "--protocol",
        str(protocol),
        "--qualification",
        str(qual),
        "--output",
        str(path),
    ]
    if smoke:
        command.append("--smoke")
    outcome = _execute(
        command,
        env,
        target / f"{seed}.log",
        spec["deadline_seconds"] + (180 if smoke else 1050),
        target / f"{seed}.gpu.jsonl",
    )
    print(
        f"Finished {label} with status {outcome['returncode']} in {outcome['wall_s']:.1f}s",
        flush=True,
    )
    return path, outcome


def _window(path: Path, recipe: dict, supply: dict) -> dict:
    windows = [rank["window"] for rank in supply["ranks"]]
    window = windows[0]
    if window["steps"] != 500:
        return {"steps": window["steps"]}
    duration = window["end_s"] - window["start_s"]
    if duration <= 0:
        raise ValueError("Invalid steady-state time window")
    episodes = 0
    rank_times = []
    for record in windows:
        timings = {
            key.removesuffix("_s") + "_ms": value * 1000 / record["steps"]
            for key, value in record["phase_totals_s"].items()
        }
        samples = record["nccl_backward_s"]
        timings["nccl_backward_ms"] = 1000 * median(samples) if samples else None
        timings["nccl_samples"] = len(samples)
        if record["source_start"] is not None:
            first, last = record["source_start"], record["source_end"]
            episodes += last["successful_episodes"] - first["successful_episodes"]
            timings["floor_blocked_s"] = last["floor_wait_s"] - first["floor_wait_s"]
            timings["refills_blocked"] = last["floor_waits"] - first["floor_waits"]
        rank_times.append(timings)

    origin = recipe["clock_origin_monotonic"]
    usage = {rank["gpu_uuid"]: [] for rank in recipe["ranks"]}
    with path.with_suffix(".gpu.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if window["start_s"] <= row["monotonic_s"] - origin <= window["end_s"]:
                for uuid, utilization, memory in row.get("gpus", []):
                    if utilization.isdecimal() and memory.isdecimal():
                        usage[uuid].append((int(utilization), int(memory)))
    return {
        "steps": 500,
        "seconds": duration,
        "updates_s": 500 / duration,
        "draws_s": 500 * recipe["protocol"]["batch_size"] / duration,
        "successful_episodes_s": episodes / duration,
        "rank_times": rank_times,
        "gpus": {
            uuid: dict(
                mean_utilization_pct=sum(util for util, _ in rows) / len(rows),
                peak_memory_mib=max(memory for _, memory in rows),
                samples=len(rows),
            )
            for uuid, rows in usage.items()
            if rows
        },
    }


def _row(path: Path, result: dict, spec: dict, seed: int, group: str) -> dict:
    row: dict[str, Any] = dict(seed=seed, group=group, **result, qualified=False)
    try:
        if (path / "supply.json").is_file():
            recipe = _json(path / "recipe.json")
            supply = _json(path / "supply.json")
            if recipe["producer_layout"] != LAYOUTS[group] or recipe["world_size"] != 2:
                raise ValueError("Run used a different paired layout")
            row.update(
                window=_window(path, recipe, supply),
                source_ready_s=supply["source_ready"],
                first_update_s=supply["first_update"],
                global_updates=supply["step"],
            )
        if result["returncode"]:
            row["status"] = "timeout" if result["timed_out"] else "run failed"
            for failure in sorted(path.glob("failure_rank_*.json")):
                row.setdefault("issues", []).append(_json(failure)["error"])
            return row
        audited = _audit(path, spec, seed)
        row.update(
            qualified=audited["qualified"],
            decision_s=audited["published_seconds"],
            published_s=audited["published_seconds"] if audited["chosen"] else None,
            reserved_successes=audited["reserved_successes"],
            minimum_at_release=audited["minimum_at_release"],
            status="passed" if audited["qualified"] else "gate missed",
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        row.update(
            status="timeout" if result["timed_out"] else "audit failed",
            issues=[str(error)],
        )
    return row


def _write_report(root: Path, rows: list[dict], spec: dict) -> None:
    def value(number: float | None) -> str:
        return f"{number:.2f}" if number is not None else "—"

    by_seed = {(row["seed"], row["group"]): row for row in rows}
    ratios = {}
    for seed in spec["seeds"]:
        a = by_seed.get((seed, "A"), {}).get("window", {}).get("updates_s")
        b = by_seed.get((seed, "B"), {}).get("window", {}).get("updates_s")
        if a and b:
            ratios[seed] = b / a
    encoded = dict(runs=rows, paired_updates_ratio_b_over_a=ratios)
    temp = root / "comparison.tmp"
    temp.write_text(json.dumps(encoded, indent=2, allow_nan=False) + "\n")
    temp.replace(root / "comparison.json")
    lines = [
        "# Push Cube ACT ODS comparison",
        "",
        "A uses one producer per GPU. B shares one producer across the two DDP ranks.",
        "The training window covers committed updates 101 through 600.",
        "",
        "## Time & Memory",
        "",
        "| Seed | Layout | Updates/s | Draws/s | Successful episodes/s | B/A updates/s | GPU A/B peak MiB | Publication s |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        window = row.get("window", {})
        memory = (
            "/".join(
                str(card["peak_memory_mib"]) for card in window.get("gpus", {}).values()
            )
            or "—"
        )
        lines.append(
            f"| {row['seed']} | {row['group']} | {value(window.get('updates_s'))} | "
            f"{value(window.get('draws_s'))} | {value(window.get('successful_episodes_s'))} | "
            f"{value(ratios.get(row['seed']))} | {memory} | {value(row.get('published_s'))} |"
        )
    lines.extend(
        [
            "",
            "## Success & Other Metrics",
            "",
            "| Seed | Layout | Reserved successes | Minimum draws at release | Result |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        score = row.get("reserved_successes")
        lines.append(
            f"| {row['seed']} | {row['group']} | {score if score is not None else '—'}/"
            f"{spec['test_scenes']} | {row.get('minimum_at_release') or '—'} | {row['status']} |"
        )
    ranking = sorted(
        LAYOUTS,
        key=lambda group: -sum(
            row["qualified"] for row in rows if row["group"] == group
        ),
    )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Layout | Seeds passing both gates | Median updates/s |",
            "| --- | ---: | ---: |",
        ]
    )
    for group in ranking:
        runs = [row for row in rows if row["group"] == group]
        rates = [
            row["window"]["updates_s"]
            for row in runs
            if row.get("window", {}).get("updates_s")
        ]
        lines.append(
            f"| {group} | {sum(row['qualified'] for row in runs)}/{len(spec['seeds'])} | "
            f"{value(median(rates) if rates else None)} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Qualify and smoke-test the pair, then run every configured seed in both layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--gpus", required=True, help="Ordered physical GPU indices or UUIDs"
    )
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("grouped_elapsed_protocol.json"),
    )
    args = parser.parse_args()
    spec = _json(args.protocol)
    if (spec["producer_envs"], spec["shared_slots"], spec["batch_size"]) != (
        10,
        20,
        32,
    ):
        raise ValueError(
            "The paired study requires 10 environments, 20 slots and a global batch of 32"
        )
    cards = _gpus(args.gpus)
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=",".join(card["uuid"] for card in cards)
    )
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root / "host.json").write_text(
        json.dumps(
            {
                "hostname": os.uname().nodename,
                "gpus": cards,
                "topology": subprocess.check_output(
                    ["nvidia-smi", "topo", "-m"], text=True
                ),
                "versions": {
                    name: version(name)
                    for name in (
                        "torch",
                        "torchvision",
                        "lerobot",
                        "dexsim-engine",
                        "tensordict",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )

    smoke_root = args.root / "smoke"
    smoke_root.mkdir()
    for group in LAYOUTS:
        path, result = _run_seed(
            smoke_root,
            group,
            spec["seeds"][0],
            args.protocol,
            args.qualification,
            env,
            spec,
            smoke=True,
        )
        if result["returncode"]:
            raise RuntimeError(f"{group} smoke failed. See {path.with_suffix('.log')}")
        supply = _json(path / "supply.json")
        recipe = _json(path / "recipe.json")
        _audit_supply(recipe, supply, spec)
        if (
            not supply["smoke_checkpoint"]
            or any(
                producer["engine"]["completed_generation"] < 3
                for producer in supply["producers"]
            )
            or [rank["gpu_uuid"] for rank in recipe["ranks"]]
            != [card["uuid"] for card in cards]
        ):
            raise RuntimeError(
                f"{group} smoke did not verify the assigned cards and recycling"
            )

    rows = []
    for index, seed in enumerate(spec["seeds"]):
        for group in ("A", "B") if index % 2 == 0 else ("B", "A"):
            path, result = _run_seed(
                args.root,
                group,
                seed,
                args.protocol,
                args.qualification,
                env,
                spec,
                smoke=False,
            )
            rows.append(_row(path, result, spec, seed, group))
            _write_report(args.root, rows, spec)
    print(args.root / "comparison.md", flush=True)
    if not all(row["qualified"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
