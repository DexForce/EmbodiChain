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

"""Budgeted campaign history, routing and independent CPU quality operations.

Run with the reference environment: python -m scripts.benchmark.rendering.campaign_proofs --help
GPU proofs execute in the original frozen worker runtimes. CPU reference work
never contributes to original throughput or resource measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
from typing import Any

from scripts.benchmark.core.artifacts import read_json_object, write_json, stable_hash
from .campaign import (
    ROOT,
    _load,
    storage_gate,
    rebuild_campaign_report,
    selected_case_rows,
    _verify_execution_source,
    _source_files,
    _verify_native_sdk,
)
from scripts.benchmark.core.provenance import git_revision

__all__ = ["proof_population", "run_proofs", "main"]

_MIB = 1024**2


def _reference_reservation(
    experiment: Path,
    row: dict[str, Any],
    cache: Path,
    history: Path | None = None,
) -> int:
    """Bound all distinct precision rungs and diagnostic output before CPU work.

    Cold references reserve raw float32 RGB/repeat, depth and int32 labels,
    metadata, and a possible cache copy. Existing shared entries reserve their
    full actual closure as a possible copy, even when hardlinks will save space.
    Existing local references create no new closure; their checksums remain the
    evaluator's responsibility. Corrupt entries are never regenerated in place.
    """
    from .evaluate import _load_run, _load_history, _reference_identity

    run = _load_run(experiment, row)
    if history is not None:
        run = _load_history(experiment, run, history)
    pixels = run["case"].width * run["case"].height
    reserve = (
        max(512 * _MIB, 100 * pixels * len(run["descriptors"]))
        if row["config"]["temporal_mode"] == "static"
        else 512 * _MIB
    )
    seen = set()
    for descriptor in run["descriptors"]:
        for spp in (4096, 16384):
            identity = _reference_identity(
                run, descriptor, {"spp": spp, "threads": 8, "seed": 0}
            )
            key = stable_hash(identity)
            if key in seen:
                continue
            seen.add(key)
            local = experiment / "references" / key
            shared = cache / "references" / key
            if local.exists():
                if not shared.exists():
                    reserve += (
                        sum(
                            path.stat().st_size
                            for path in local.rglob("*")
                            if path.is_file()
                        )
                        + _MIB
                    )
                continue
            if shared.exists():
                reserve += (
                    sum(
                        path.stat().st_size
                        for path in shared.rglob("*")
                        if path.is_file()
                    )
                    + _MIB
                )
            else:
                reserve += 2 * (32 * pixels + _MIB)
    return reserve


def proof_population(root: Path, kind: str) -> list[dict[str, Any]]:
    """Select completed original runs without changing any trial population.

    Args:
        root: Frozen campaign directory.
        kind: History, routing, quality or measured calibration.

    Returns:
        Explicit source-run requests; publication moving histories are complete,
        while routing uses one representative repeat per selected batch case.
    """
    definition, state = _load(root)
    cases = {c["case_id"]: c for c in definition["cases"]}
    population = []
    for job in state["jobs"]:
        if job["phase"] == "smoke":
            continue
        case = cases[job["case_id"]]["case"]
        if kind == "history" and not (
            job["phase"] == "publication" and case["temporal_mode"] == "moving"
        ):
            continue
        if kind == "routing" and not (
            job["phase"] == "coverage"
            and case["experiment_id"] in ("R-04", "R-05")
            and case["num_envs"] in (16, 64)
        ):
            continue
        if kind == "calibration" and not (
            job["phase"] == "coverage"
            and case["experiment_id"] == "R-01"
            and case["scene_id"] == "calibration"
            and case["num_envs"] == 1
        ):
            continue
        for attempt in job["attempts"]:
            for relative in attempt["experiments"]:
                experiment = root / relative
                for row in selected_case_rows(experiment, job["case_id"]):
                    if row["status"] != "completed":
                        continue
                    if kind == "routing" and row["repeat"] != 0:
                        continue
                    population.append(
                        {
                            "kind": kind,
                            "experiment": relative,
                            "run_id": row["run_id"],
                            "backend": row["backend"],
                            "case": case,
                            "phase": job["phase"],
                            "source_commit": definition["source_commit"],
                        }
                    )
    return population


def _calibrate(experiment: Path, row: dict[str, Any], cache: Path) -> Path:
    from .evaluate import _load_run, _independent_reference
    from .calibration import create_calibration_receipt

    run = _load_run(experiment, row)
    pairs = []
    for item in sorted(
        run["descriptors"], key=lambda item: item["metadata"]["view_id"]
    ):
        reference, alias = _independent_reference(
            experiment,
            run,
            item,
            {"spp": 4096, "threads": 8, "seed": 0},
            shared_cache=cache,
        )
        if reference is not None and not reference["reference_validation"]["passed"]:
            reference, alias = _independent_reference(
                experiment,
                run,
                item,
                {"spp": 16384, "threads": 8, "seed": 0},
                shared_cache=cache,
            )
        if reference is None:
            raise ValueError("Measured calibration requires an independent reference")
        pairs.append(
            {
                "capture": item["metadata_path"],
                "reference": experiment / alias["reference_frame"],
            }
        )
    receipt = experiment / f"calibration-{row['backend']}-{row['run_id']}.json"
    create_calibration_receipt(
        receipt,
        backend=row["backend"],
        profile=run["config"]["profile"],
        training_pairs=pairs[:10],
        heldout_pairs=pairs[10:],
    )
    return receipt


def run_proofs(
    root: Path,
    *,
    kind: str,
    run_ids: tuple[str, ...] = (),
    reevaluate: bool = False,
    retry_failed: bool = False,
    source_worktree: Path | None = None,
) -> None:
    """Preserve a ledger for every untimed proof/evaluation operation.

    Args:
        root: Frozen campaign directory.
        kind: History, routing, independent quality or measured calibration.
        run_ids: Optional explicit original-run filter.
        reevaluate: Append a new quality evaluation after calibration/history updates.
        retry_failed: Append an explicitly selected failed/interrupted proof attempt.
        source_worktree: Clean original source used by history/routing workers.
    """
    root = Path(root).absolute()
    definition, state = _load(root)
    if retry_failed and not run_ids:
        raise ValueError("Proof retries require an explicit original-run selection")
    if reevaluate and kind != "quality":
        raise ValueError("Reevaluation applies only to quality")
    source = (
        _verify_execution_source(definition, source_worktree)
        if kind in ("history", "routing")
        else ROOT
    )
    if kind in ("history", "routing") and definition.get("native_sdk_snapshot"):
        _verify_native_sdk(
            definition["native_sdk_snapshot"],
            Path(definition["interpreters"]["embodichain"]),
        )
    requests = proof_population(root, kind)
    if run_ids:
        requests = [r for r in requests if r["run_id"] in run_ids]
    proofs = state.setdefault("proofs", [])
    for request in requests:
        identity = stable_hash(request)
        previous = [p for p in proofs if p["proof_id"] == identity]
        if previous:
            if not (
                reevaluate
                or retry_failed
                and previous[-1]["status"] in ("failed", "interrupted")
            ):
                continue
        proof = {
            **request,
            "proof_id": identity,
            "evaluation_attempt": len(previous),
            "status": "not_run",
            "execution_source_root": str(source),
            "driver_source_commit": git_revision(ROOT),
            "driver_source_files": _source_files(),
        }
        proofs.append(proof)
    write_json(root / "campaign-state.json", state)
    cache = root / "shared-reference-cache"
    proof_root = Path(definition["proof_root"])
    for proof in proofs:
        if (
            proof["kind"] != kind
            or proof["status"] != "not_run"
            or run_ids
            and proof["run_id"] not in run_ids
        ):
            continue
        case = proof["case"]
        identity = proof["proof_id"]
        destination = (
            proof_root / kind / identity / f"attempt-{proof['evaluation_attempt']:03d}"
        )
        experiment = root / proof["experiment"]
        history = next(
            (
                p
                for p in proofs
                if p["kind"] == "history"
                and p["experiment"] == proof["experiment"]
                and p["run_id"] == proof["run_id"]
                and p["status"] == "completed"
            ),
            None,
        )
        if (
            kind == "quality"
            and proof["phase"] == "publication"
            and case["temporal_mode"] == "moving"
            and history is None
        ):
            proof["reason"] = "complete_publication_history_required"
            write_json(root / "campaign-state.json", state)
            continue
        reserve = 256 * _MIB
        if kind == "history":
            # 4 prefix + 30 warmups + 300 measured captures, with padding.
            # Both acquisition and portable import persist NPZ/PNG evidence.
            pixel_bytes = sum(
                {"rgb": 3, "depth": 4, "seg": 4, "normals": 12}[m]
                for m in case["modalities"]
            )
            reserve += (
                350
                * case["width"]
                * case["height"]
                * case["num_envs"]
                * case["cameras_per_env"]
                * (pixel_bytes + 3)
                * 2
            )
        elif kind in ("quality", "calibration"):
            try:
                row = next(
                    r
                    for r in json.loads((experiment / "runs.json").read_text())
                    if r["run_id"] == proof["run_id"]
                )
                reserve = _reference_reservation(
                    experiment,
                    row,
                    cache,
                    experiment / history["history_evidence"] if history else None,
                )
            except Exception as error:
                proof.update(
                    status="failed",
                    reason=f"proof_admission_input: {type(error).__name__}: {error}",
                )
                write_json(root / "campaign-state.json", state)
                continue
        destinations = (
            [destination, experiment]
            if kind == "history"
            else [destination] if kind == "routing" else [experiment, cache]
        )
        gates = []
        for location in destinations:
            gate = storage_gate(
                definition, reservation_bytes=reserve, destination=location
            )
            gates.append(gate)
            if not gate["passed"]:
                break
        proof["storage_reservation_bytes"] = reserve
        proof["storage_gates"] = gates
        if not gate["passed"]:
            proof["reason"] = gate["reason"]
            proof["storage_gate"] = gate
            write_json(root / "campaign-state.json", state)
            break
        if kind in ("history", "routing"):
            if (
                state["reserved_worker_runs"]
                >= definition["budget"]["maximum_worker_runs"]
            ):
                proof["reason"] = "worker_attempt_budget"
                write_json(root / "campaign-state.json", state)
                break
            state["reserved_worker_runs"] += 1
        proof.setdefault(
            "queued_driver_source_commit", proof.get("driver_source_commit")
        )
        proof.update(
            status="running",
            execution_source_root=str(source),
            driver_source_commit=git_revision(ROOT),
            driver_source_files=_source_files(),
        )
        write_json(root / "campaign-state.json", state)
        print(f"{kind}: {proof['run_id']}", flush=True)
        try:
            if kind == "history":
                from .history_acquisition import acquire_history
                from .history_binding import import_history_proof

                result = acquire_history(
                    experiment,
                    proof["run_id"],
                    source_worktree=source,
                    source_commit=definition["source_commit"],
                    output=destination,
                    gpu_id=0,
                    timeout_s=definition["history_timeout_s"],
                )
                if result["status"] != "completed":
                    raise RuntimeError(f"History acquisition failed: {result}")
                proof["acquisition"] = result
                import_bytes = sum(
                    path.stat().st_size
                    for path in Path(result["root"]).rglob("*")
                    if path.is_file()
                )
                import_gate = storage_gate(
                    definition, reservation_bytes=import_bytes, destination=experiment
                )
                proof["import_storage_gate"] = import_gate
                if not import_gate["passed"]:
                    raise RuntimeError(
                        f"history_import_storage_gate: {import_gate['reason']}"
                    )
                imported = import_history_proof(Path(result["history"]), experiment)
                proof["history_evidence"] = imported["history_evidence"]["path"]
            elif kind == "routing":
                from .routing_diagnostic import prepare_routing_diagnostic
                from scripts.benchmark.core.execution import execute_worker

                request = prepare_routing_diagnostic(
                    source,
                    definition["source_commit"],
                    experiment,
                    proof["run_id"],
                    destination,
                    gpu_id=0,
                )
                output = request["output"]
                result = execute_worker(
                    request["command"],
                    output,
                    backend=proof["backend"],
                    repeat=0,
                    timeout_s=definition["history_timeout_s"],
                    env=request["environment"],
                )
                proof["result"] = result
                if result["status"] != "completed":
                    raise RuntimeError("Routing worker failed; original logs retained")
            elif kind == "quality":
                from .evaluate import evaluate_experiment
                from .calibration_binding import bind_calibration_receipt

                kwargs = {}
                if history:
                    kwargs["history_evidence"] = [
                        experiment / history["history_evidence"]
                    ]
                for calibration in proofs:
                    if (
                        calibration["kind"] == "calibration"
                        and calibration["status"] == "completed"
                        and calibration["backend"] == proof["backend"]
                        and calibration["case"]["profile"] == case["profile"]
                    ):
                        receipt = Path(calibration["receipt"])
                        measured = read_json_object(receipt)
                        if measured.get("status") == "qualified":
                            binding = bind_calibration_receipt(receipt, experiment)
                            if binding.get("evidence_document"):
                                kwargs["calibration_evidence"] = (
                                    experiment / binding["evidence_document"]
                                )
                                break
                proof["report"] = str(
                    evaluate_experiment(
                        experiment,
                        spp=4096,
                        threads=8,
                        seed=0,
                        run_ids=[proof["run_id"]],
                        reference_precision_policy=definition[
                            "reference_precision_policy"
                        ],
                        shared_reference_cache=cache,
                        **kwargs,
                    )
                )
            else:
                rows = json.loads((experiment / "runs.json").read_text())
                row = next(r for r in rows if r["run_id"] == proof["run_id"])
                proof["receipt"] = str(_calibrate(experiment, row, cache))
            proof["status"] = "completed"
        except KeyboardInterrupt:
            proof["status"] = "interrupted"
            write_json(root / "campaign-state.json", state)
            raise
        except Exception as error:
            proof.update(status="failed", reason=f"{type(error).__name__}: {error}")
        write_json(root / "campaign-state.json", state)
    rebuild_campaign_report(root)


def main(argv: list[str] | None = None) -> None:
    """Acquire or evaluate a selected frozen campaign proof population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("history", "routing", "quality", "calibration"),
        required=True,
    )
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--source-worktree", type=Path)
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(proof_population(args.root, args.kind), indent=2))
        return
    run_proofs(
        args.root,
        kind=args.kind,
        run_ids=tuple(args.run_id),
        reevaluate=args.reevaluate,
        retry_failed=args.retry_failed,
        source_worktree=args.source_worktree,
    )


if __name__ == "__main__":
    main()
