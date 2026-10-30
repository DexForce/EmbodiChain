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
"""Select a saved peg checkpoint using balanced native validation episodes.

Run after training: python -m scripts.benchmark.assembly.select_peg --run RUN \
    --output OUTPUT --episodes 512 --num-envs 64 --seed 10000 --jobs 1

Each candidate uses the existing strict ``eval_peg`` runner in an isolated
process, with the same saved environment and a fixed per-arena episode quota.
This is validation, not the final independent test. No policy is updated.
Optional bounded concurrency trades additional GPU memory for elapsed time;
per-candidate health, raw episodes and process logs remain independent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from scripts.benchmark.assembly.train_peg import _file_metadata, _write

__all__: list[str] = []


def _select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Require complete candidate coverage and prefer earlier tied updates."""
    if not candidates or any(row.get("status") != "completed" for row in candidates):
        raise ValueError("Selection requires every declared candidate to complete.")
    for row in candidates:
        score = row.get("success_rate")
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Selection requires finite success rates in [0, 1].")
    return min(
        candidates,
        key=lambda row: (-row["success_rate"], row["num_updates"], row["sha256"]),
    )


def _verified_success(
    path: Path, episodes: int, reported_rate: float
) -> tuple[float, int]:
    """Derive success from scored records and cross-check the native metric."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != episodes or any(row.get("success") not in (0, 1) for row in rows):
        raise ValueError(
            "The scored ledger must contain one binary success per episode."
        )
    successes = sum(int(row["success"]) for row in rows)
    rate = successes / episodes
    if not math.isclose(rate, reported_rate, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            "Native success rate disagrees with the scored episode ledger."
        )
    return rate, successes


def _discover(run: Path) -> list[dict[str, Any]]:
    """Keep one copy of identical policy states saved at the same update."""
    import torch

    rows: list[dict[str, Any]] = []
    policies: dict[int, dict[str, Any]] = {}
    for path in sorted((run / "checkpoints").glob("*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        update = int(checkpoint["num_updates"])
        policy = checkpoint["policy"]
        if update in policies:
            previous = policies[update]
            if previous.keys() != policy.keys() or any(
                not torch.equal(previous[key], policy[key]) for key in policy
            ):
                raise ValueError(f"Different policy states share update {update}.")
            continue
        policies[update] = policy
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "num_updates": update,
                "global_step": int(checkpoint["global_step"]),
            }
        )
    if not rows:
        raise ValueError("No complete native checkpoints found.")
    return sorted(rows, key=lambda row: row["num_updates"])


def _evaluate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Invoke the ordinary native evaluator and verify its accounting proof."""
    row = dict(candidate)
    directory = args.output / f"update_{row['num_updates']:05d}"
    log = args.output / f"update_{row['num_updates']:05d}.log"
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark.assembly.eval_peg",
        str(args.run),
        "--checkpoint",
        row["path"],
        "--episodes",
        str(args.episodes),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output",
        str(directory),
        "--raw-log",
        str(log),
    ]
    row["command"] = command
    row["started_at"] = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        environment = dict(os.environ)
        environment["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        with log.open("x") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, env=environment
            )
        row["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                f"Native evaluation returned {result.returncode}; see {log}."
            )
        paths = list(directory.glob("*-strict-peg-health/health.json"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one native health report.")
        path = paths[0]
        health = json.loads(path.read_text())
        raw = health["raw_episodes"]
        native = health["native_health"]
        metrics = health["metrics"]
        if (
            health["status"] != "completed"
            or health["episode_allocation"] != "balanced"
            or health["evaluation_reset_seed"] != args.seed
            or health["episodes_requested"] != args.episodes
            or health["num_envs_requested"] != args.num_envs
            or health["checkpoint"]["sha256"] != row["sha256"]
            or raw["records"] != args.episodes
            or raw["scored_counts_per_env"] != raw["expected_counts_per_env"]
            or len(raw["scored_counts_per_env"]) != args.num_envs
            or sum(raw["scored_counts_per_env"]) != args.episodes
            or native.get("capacity_overflow_seen") is not False
            or native.get("unclassified_overflow_seen") is not False
            or native.get("native_state", {}).get("all_samples_finite") is not True
        ):
            raise ValueError(
                "Native checkpoint, episode accounting or health proof is incomplete."
            )
        episodes = Path(raw["file"]["path"])
        if hashlib.sha256(episodes.read_bytes()).hexdigest() != raw["file"]["sha256"]:
            raise ValueError("Raw episode data changed after native verification.")
        rate, successes = _verified_success(
            episodes, args.episodes, metrics["eval/success_rate"]
        )
        row.update(
            status="completed",
            success_rate=rate,
            successes=successes,
            metrics=metrics,
            health=_file_metadata(path, digest=True),
            raw_episodes=raw["file"],
            convergence_budget_exhausted_seen=native.get(
                "convergence_budget_exhausted_seen"
            ),
        )
    except Exception:
        row.update(status="failed", error=traceback.format_exc())
    row["elapsed_process_wall_seconds"] = time.perf_counter() - started
    row["log"] = _file_metadata(log)
    return row


def _markdown(path: Path, document: dict[str, Any]) -> None:
    """Write the three benchmark tables without upgrading validation to test."""
    rows = document["candidates"]
    lines = [
        "# Balanced checkpoint validation",
        "",
        document["qualification"],
        "",
        "## Time & Memory",
        "",
        "| jobs | elapsed_wall_seconds | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---:|---:|---|---|---|",
        f"| {document['protocol']['jobs']} | {document['elapsed_wall_seconds']:.3f} | unmeasured | unmeasured | unmeasured |",
        "",
        "## Success & Other Metrics",
        "",
        "| update | success_rate | status | convergence_budget_exhausted |",
        "|---:|---:|---|---|",
    ]
    lines += [
        f"| {row['num_updates']} | {row.get('success_rate', 'unmeasured')} | {row['status']} | {row.get('convergence_budget_exhausted_seen', 'unmeasured')} |"
        for row in rows
    ]
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | update | validation_success_rate |",
        "|---:|---:|---:|",
    ]
    complete = [row for row in rows if row.get("status") == "completed"]
    lines += [
        f"| {rank} | {row['num_updates']} | {row['success_rate']} |"
        for rank, row in enumerate(
            sorted(
                complete, key=lambda row: (-row["success_rate"], row["num_updates"])
            ),
            1,
        )
    ]
    lines += [
        "",
        "Elapsed time includes child startup, completed native evaluations and teardown. No independent-test episodes participate in selection. Candidate processes use the same device; concurrency is a resource setting, not a measured speedup claim. Raw native health reports distinguish convergence budgets from capacity loss.",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines))


def main() -> None:
    """Declare a validation protocol before running any candidate evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.episodes, args.num_envs, args.jobs) < 1:
        parser.error("Episode, environment and worker counts must be positive.")
    args.run, args.output = args.run.resolve(), args.output.resolve()
    path = args.output / "selection.json"
    if path.exists():
        parser.error(
            "Preserve existing selection evidence with a new output directory."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = _discover(args.run)
    document = {
        "schema": "embodichain-peg-balanced-selection/v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run),
        "qualification": "Balanced development validation of every saved update; not independent-test or paper-reproduction evidence.",
        "protocol": {
            "episodes": args.episodes,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "episode_allocation": "balanced",
            "jobs": args.jobs,
            "device": args.device,
            "selection_rule": "highest success rate; ties prefer earlier updates",
            "child_environment_overrides": {"TORCHINDUCTOR_COMPILE_THREADS": "1"},
        },
        "declared_candidates": candidates,
        "candidates": [],
        "wrapper_source": _file_metadata(Path(__file__), digest=True),
    }
    _write(path, document)
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(_evaluate, candidate, args) for candidate in candidates
            ]
            for future in as_completed(futures):
                row = future.result()
                document["candidates"].append(row)
                document["candidates"].sort(key=lambda item: item["num_updates"])
                document["elapsed_wall_seconds"] = time.perf_counter() - started
                _write(path, document)
                print(
                    f"update={row['num_updates']} status={row['status']} balanced_validation={row.get('success_rate')}",
                    flush=True,
                )
        document["selected_checkpoint"] = _select_best(document["candidates"])
        document["status"] = "completed"
    except BaseException:
        document["status"] = "failed"
        document["error"] = traceback.format_exc()
        raise
    finally:
        document["elapsed_wall_seconds"] = time.perf_counter() - started
        _write(path, document)
        _markdown(path, document)
    print(f"Balanced validation selection: {path}", flush=True)


if __name__ == "__main__":
    main()
