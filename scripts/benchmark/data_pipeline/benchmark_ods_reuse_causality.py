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

"""Controlled initial-reuse and replacement diagnostics for PickPlace ChunkBC.

Train with a schema-validated two-generation corpus, then evaluate each frozen
checkpoint in a fresh B=1 simulator. These are resident-data diagnostics, not
end-to-end ODS throughput measurements. No formal comparison test scenes are used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.data_pipeline._ods_pick_place_benchmark import load_corpus
from examples.data_pipeline._ods_pick_place_policy import ChunkBC, pose_targets
from examples.data_pipeline.ods_pick_place_showcase import (
    _evaluate_worker,
    _geometry_statistics,
    _receive,
    load_task_config,
    save_checkpoint,
    load_policy,
)

__all__ = ["main"]


def _train(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(str(args.source) + ".manifest.json").read_text())
    corpus = load_corpus(args.source, expected_schema=manifest["schema"])
    if corpus.generations is None:
        raise ValueError("The causal diagnostic requires two recorded generations")
    initial_mask = corpus.valid & (corpus.generations == 1)
    second_mask = corpus.valid & (corpus.generations == 2)
    if not initial_mask.any() or not second_mask.any():
        raise ValueError("Both G1 and G2 must contain accepted data")
    mean, std = _geometry_statistics(corpus.observations[initial_mask])
    target = pose_targets(
        corpus.observations[initial_mask], corpus.actions[initial_mask]
    )
    target_mean, target_std = target.mean(0), target.std(0, correction=0).clamp_min(
        0.05
    )
    target_mean[3:], target_std[3:] = 0, 1
    anchor_count = int(initial_mask.sum())
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    args.train_device = "cuda:0"
    for reuse in args.reuse:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = ChunkBC(mean, std, target_mean, target_std).to(args.train_device)
        model.schema = manifest["schema"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
        # CPU sample RNG is independent from CUDA augmentation RNG.
        torch.random.default_generator.manual_seed(args.seed + 100000)
        first_updates = math.ceil(reuse * anchor_count / args.batch_size)
        branch_updates = math.ceil(
            args.continuation_reuse * anchor_count / args.batch_size
        )

        def train_segment(generation: int, count: int) -> dict[str, Any]:
            model.train()
            begin = time.monotonic()
            targets = 0
            loss = 0.0
            for update in range(count):
                batch = corpus.sample_context(
                    args.batch_size, 2, 64, generations=[generation]
                )
                targets += int(batch["loss_mask"].sum())
                arm, hand = model.loss(
                    batch["observations"].to(args.train_device),
                    batch["actions"].to(args.train_device),
                    loss_mask=batch["loss_mask"].to(args.train_device),
                    valid_mask=batch["valid"].to(args.train_device),
                )
                objective = (9 * arm + hand) / 10
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
                if update == count - 1:
                    loss = float(objective.detach())
            torch.cuda.synchronize()
            return {
                "generation": generation,
                "updates": count,
                "chunks": count * args.batch_size,
                "supervised_targets": targets,
                "last_loss": loss,
                "training_seconds": time.monotonic() - begin,
            }

        before = train_segment(1, first_updates)
        checkpoint = args.output / f"reuse{reuse}_before.pt"
        save_checkpoint(checkpoint, model, optimizer, first_updates, args, 0)
        records.append(
            {
                "reuse": reuse,
                "branch": "before",
                "checkpoint": str(checkpoint),
                "segments": [before],
            }
        )
        weights = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        for branch, generation in (("continue_g1", 1), ("replace_g2", 2)):
            model.load_state_dict(weights)
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            after = train_segment(generation, branch_updates)
            checkpoint = args.output / f"reuse{reuse}_{branch}.pt"
            save_checkpoint(
                checkpoint, model, optimizer, first_updates + branch_updates, args, 0
            )
            records.append(
                {
                    "reuse": reuse,
                    "branch": branch,
                    "checkpoint": str(checkpoint),
                    "segments": [before, after],
                }
            )
            print(json.dumps(records[-1]), flush=True)
        summary = {
            "seed": args.seed,
            "source": str(args.source),
            "content_id": manifest["content_id"],
            "g1_anchors": anchor_count,
            "batch_size": args.batch_size,
            "continuation_reuse": args.continuation_reuse,
            "records": records,
            "resident_diagnostic_seconds": time.monotonic() - started,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "formal_test_used": False,
        }
        (args.output / "training.json").write_text(json.dumps(summary, indent=2))


def _eval_child(
    config: dict[str, Any], requests: Any, results: Any, output: str
) -> None:
    torch.set_num_threads(1)
    _evaluate_worker(config, requests, results, output, False, 8099, False)


def _evaluate(args: argparse.Namespace) -> None:
    state = json.loads((args.output / "training.json").read_text())
    context = mp.get_context("spawn")
    config = load_task_config("cuda:0", "pose", "dls")
    config["seed"] = 810000
    for entry in state["records"]:
        if args.evaluate_branches and entry["branch"] not in args.evaluate_branches:
            continue
        name = Path(entry["checkpoint"]).stem
        output = args.output / (name + "_development")
        if (output / "report.json").exists():
            cleanup_path = output / "cleanup.json"
            if not cleanup_path.exists():
                raise RuntimeError(
                    f"Existing evaluation lacks cleanup evidence: {output}"
                )
            previous_cleanup = json.loads(cleanup_path.read_text())
            if previous_cleanup.get("forced") or previous_cleanup.get("exitcode") != 0:
                raise RuntimeError(
                    f"Existing evaluation did not close cleanly: {output}"
                )
            continue
        output.mkdir(exist_ok=True)
        requests, results = context.Queue(), context.Queue()
        process = context.Process(
            target=_eval_child, args=(config, requests, results, str(output))
        )
        started = time.monotonic()
        cleanup: dict[str, Any] = {}
        try:
            process.start()
            _receive(process, results, timeout=300)
            requests.put(
                {
                    "kind": "student",
                    "name": "development",
                    "checkpoint": entry["checkpoint"],
                    "seeds": list(range(810000, 810000 + args.scenes)),
                    "trace": False,
                }
            )
            report = _receive(process, results, timeout=1800)
            report["elapsed_seconds"] = time.monotonic() - started
            report["checkpoint"] = entry["checkpoint"]
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        "name": name,
                        "success_rate": report["success_rate"],
                        "elapsed_seconds": report["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
        finally:
            if process.pid is not None:
                requests.put(None)
                process.join(30)
                forced = process.is_alive()
                if forced:
                    process.terminate()
                    process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(10)
                cleanup = {
                    "pid": process.pid,
                    "exitcode": process.exitcode,
                    "forced": forced,
                }
            requests.close()
            results.close()
            requests.join_thread()
            results.join_thread()
            (output / "cleanup.json").write_text(json.dumps(cleanup, indent=2))
        if cleanup.get("forced") or cleanup.get("exitcode") != 0:
            raise RuntimeError("Evaluator did not close cleanly")
    _report(args.output)


def _report(output: Path) -> None:
    state = json.loads((output / "training.json").read_text())
    rows = []
    for entry in state["records"]:
        report_path = (
            output / (Path(entry["checkpoint"]).stem + "_development") / "report.json"
        )
        if report_path.exists():
            cleanup_path = report_path.with_name("cleanup.json")
            if not cleanup_path.exists():
                raise RuntimeError(f"Evaluation cleanup missing: {report_path}")
            cleanup = json.loads(cleanup_path.read_text())
            if cleanup.get("forced") or cleanup.get("exitcode") != 0:
                raise RuntimeError(f"Evaluation cleanup failed: {report_path}")
            rows.append((entry, json.loads(report_path.read_text())))
    lines = [
        "# Initial reuse and replacement diagnostic",
        "",
        "Resident-data, paired checkpoint branches; development results only. "
        "No claim about live ODS throughput or population-level significance.",
        "",
        "## Time & Memory",
        "",
        "| Reuse | Branch | Updates | Train s | Memory |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for entry, _ in rows:
        segments = entry["segments"]
        lines.append(
            f"| {entry['reuse']} | {entry['branch']} | "
            f"{sum(s['updates'] for s in segments)} | "
            f"{sum(s['training_seconds'] for s in segments):.2f} | unmeasured |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Reuse | Branch | Success | Lift | Goal error m | Chunks |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry, report in rows:
        lines.append(
            f"| {entry['reuse']} | {entry['branch']} | {report['success_rate']:.3f} | "
            f"{report['lift_rate']:.3f} | {report['goal_error']:.4f} | "
            f"{sum(s['chunks'] for s in entry['segments'])} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Rank | Reuse | Branch | Development success |",
        "| ---: | ---: | --- | ---: |",
    ]
    for rank, (entry, report) in enumerate(
        sorted(rows, key=lambda x: -x[1]["success_rate"]), 1
    ):
        lines.append(
            f"| {rank} | {entry['reuse']} | {entry['branch']} | {report['success_rate']:.3f} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def _audit(args: argparse.Namespace) -> None:
    """Measure fit on identical seeded training contexts, without updating weights."""
    state = json.loads((args.output / "training.json").read_text())
    source = Path(state["source"])
    manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
    corpus = load_corpus(source, expected_schema=manifest["schema"])
    rows = []
    for entry in state["records"]:
        model, _ = load_policy(entry["checkpoint"], "cpu")
        metrics = {}
        for generation in (1, 2):
            torch.manual_seed(990000 + generation)
            arm_total = hand_total = 0.0
            count = 0
            for _ in range(8):
                batch = corpus.sample_context(128, 2, 64, generations=[generation])
                with torch.inference_mode():
                    arm, hand = model.loss(
                        batch["observations"],
                        batch["actions"],
                        loss_mask=batch["loss_mask"],
                        valid_mask=batch["valid"],
                    )
                targets = int(batch["loss_mask"].sum())
                arm_total += float(arm) * targets
                hand_total += float(hand) * targets
                count += targets
            metrics[str(generation)] = {
                "arm_mse": arm_total / count,
                "gripper_mse": hand_total / count,
                "supervised_targets": count,
                "contexts": 1024,
            }
        rows.append(
            {
                "checkpoint": entry["checkpoint"],
                "reuse": entry["reuse"],
                "branch": entry["branch"],
                "metrics": metrics,
            }
        )
    (args.output / "fit_audit.json").write_text(
        json.dumps(
            {
                "scope": "fixed sampled training contexts; no optimizer updates or final scenes",
                "sampler_seeds": [990001, 990002],
                "records": rows,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["train", "evaluate", "report", "audit"], required=True
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Validated two-generation corpus; required for train",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--reuse", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--continuation-reuse", type=int, default=32)
    parser.add_argument(
        "--evaluate-branches",
        nargs="+",
        choices=["before", "continue_g1", "replace_g2"],
    )
    parser.add_argument("--scenes", type=int, default=20)
    args = parser.parse_args()
    if min(args.batch_size, args.scenes, args.continuation_reuse) < 1 or any(
        r < 1 for r in args.reuse
    ):
        parser.error("Batch, scene count and reuse must be positive")
    if args.phase == "train" and args.source is None:
        parser.error("--source is required for training")
    torch.set_num_threads(1)
    if args.phase == "train":
        _train(args)
    elif args.phase == "evaluate":
        _evaluate(args)
    elif args.phase == "audit":
        _audit(args)
    else:
        _report(args.output)


if __name__ == "__main__":
    main()
