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

"""Acquire a fresh untimed proof of an original performance renderer history.

Only the explicit original source checkout executes adapter/capture code.
``--dry-run`` validates the source and complete schedule without creating output
directories or importing any runtime. Quality policy extensions are separate
from, and never rewrite, the original benchmark definition or raw timings.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from scripts.benchmark.core.artifacts import ArtifactStore, create_experiment_directory
from scripts.benchmark.core.contracts import Budget, ExperimentDefinition
from scripts.benchmark.core.execution import run_experiment
from scripts.benchmark.core.records import RunSpec
from .frozen_source import (
    artifact,
    freeze_original_run,
    load_original_run,
    original_environment,
    read_json,
    sha256,
    stable_hash,
    verify_source_worktree,
    write_json,
)
from .history_contract import build_history_schedule, validate_history_proof
from .quality_protocol import protocol_for_modalities, protocol_requirements

__all__ = ["acquire_history", "main"]


def acquire_history(
    experiment_root: Path,
    run_id: str,
    *,
    source_worktree: Path,
    source_commit: str,
    output: Path,
    interpreter: Path | None = None,
    gpu_id: int | None = None,
    timeout_s: float = 3600,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Launch one source-compatible fresh worker, or inspect its exact schedule.

    Args:
        experiment_root: Original frozen formal experiment.
        run_id: Completed original performance run to qualify separately.
        source_worktree: Clean checkout from which original adapters must load.
        source_commit: Explicit original commit, checked against the raw ledger.
        output: Parent directory for a new proof; existing proofs are not reused.
        interpreter: Optional runtime executable; default recorded original argv.
        gpu_id: Optional physical index; default original selected GPU.
        timeout_s: Worker timeout, including deliberately untimed persistence.
        dry_run: Validate/print a plan only, with no mutations or GPU work.

    Returns:
        Resolved plan and, after execution, root/history/result status. A history
        proof never asserts throughput or promotes a target quality status.
    """
    original = load_original_run(experiment_root, run_id)
    source = verify_source_worktree(original, source_worktree, source_commit)
    output = Path(output).resolve()
    if output.is_relative_to(Path(source["worktree"])) or output.is_relative_to(
        original["root"]
    ):
        raise ValueError("History output must be outside the frozen original tree")
    if original["replay"] is None:
        raise ValueError("Original moving performance replay is missing")
    schedule = build_history_schedule(
        original["config"], original["replay"], original["capture"]
    )
    case = original["config"]["case"]
    policy = protocol_for_modalities(
        tuple(case["modalities"]), timed_history=True, temporal_dt_s=1 / 30
    )
    requirements = protocol_requirements(policy["id"], "sequence")
    row = original["row"]
    recorded_python = (
        row.get("command") or [row["software"].get("python_executable")]
    )[0]
    if interpreter is None and not recorded_python:
        raise ValueError("Original interpreter path is missing; select it explicitly")
    python = Path(
        interpreter if interpreter is not None else recorded_python
    ).absolute()
    selected_gpu = row.get("selected_gpu_id") if gpu_id is None else gpu_id
    if type(selected_gpu) is not int or selected_gpu < 0:
        raise ValueError("Original physical GPU index is required")
    budget = Budget(max_runs=1, max_attempts=1, wall_time_s=timeout_s)
    plan = {
        "schema_version": 1,
        "kind": "timed_render_history_request/v1",
        "original_run_id": run_id,
        "original_experiment_root": str(Path(experiment_root).resolve()),
        "source": source,
        "interpreter": str(python),
        "gpu_id": selected_gpu,
        "schedule": schedule,
        "schedule_sha256": stable_hash(schedule),
        "history_quality_protocol": policy,
        "history_quality_requirements": requirements,
        "performance_claim": False,
        "original_input_sha256": original["input_hashes"],
        "filesystem_mutations": not dry_run,
    }
    if dry_run:
        return plan
    definition = ExperimentDefinition(
        experiment_id="timed-history-acquisition",
        definition_version="1.0",
        parameter_matrix={"original_run_id": (run_id,)},
        budget=budget,
        quality_protocol={
            "history_quality_protocol": policy,
            "history_quality_requirements": requirements,
            "target_quality_status": "unchanged",
            "performance_claim": False,
        },
    )
    root = create_experiment_directory(
        output,
        experiment_id="timed-history-acquisition",
        config=plan,
        definition=definition,
    )
    binding, _ = freeze_original_run(original, source, root)
    producer = []
    for name in (
        "history_acquisition.py",
        "history_worker.py",
        "history_contract.py",
        "frozen_source.py",
    ):
        target = root / "producer" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name), target)
        producer.append(
            artifact(
                root, target, "history_producer_source", role="producer_source:" + name
            )
        )
    request = {**plan, "original": binding, "producer_artifacts": producer}
    request_path = write_json(root / "request.json", request)
    environment = original_environment(
        Path(source["worktree"]),
        python,
        Path(source["isaaclab_root"]) if source.get("isaaclab_root") else None,
    )
    # Execute the copied producer, so subsequent development-tree edits cannot
    # change the active acquisition logic. Its bare imports resolve beside it.
    spec = RunSpec(
        backend=row["backend"],
        repeat=row["repeat"],
        case_id=row["case_id"],
        output=root / "capture",
        command=(
            str(python),
            str(root / "producer/history_worker.py"),
            "--request",
            str(request_path),
            "--output",
            str(root / "capture"),
            "--gpu-id",
            str(selected_gpu),
        ),
        timeout_s=timeout_s,
        env=environment,
    )
    status = "failed"
    try:
        runs = run_experiment(
            root, [spec], experiment_id="timed-history-acquisition", budget=budget
        )
        status = runs[0]["status"]
    finally:
        result_path, history_path = root / "capture/result.json", root / "history.json"
        if history_path.is_file() and result_path.is_file():
            history, result = read_json(history_path), read_json(result_path)
            history["acquisition_result"] = artifact(
                root, result_path, "history_acquisition_result"
            )
            if result["status"] != "completed":
                history["status"] = "failed"
            write_json(history_path, history)
            expected = {
                **deepcopy(row),
                "_history_artifact_hashes": original["input_hashes"],
            }
            # The validator follows registered auxiliary evidence as well as
            # direct manifest descriptors, so freeze those hashes first.
            store = ArtifactStore(root)
            indexed = {
                item["path"]: item
                for item in read_json(root / "artifact_index.json").get("artifacts", [])
            }
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name not in {
                    "artifact_index.json",
                    "raw.jsonl",
                    "runs.json",
                    "quality.json",
                    "metrics.json",
                    "budget.json",
                    "manifest.json",
                }:
                    previous = indexed.get(path.relative_to(root).as_posix())
                    if previous is not None:
                        if previous["sha256"] != sha256(path):
                            raise ValueError("Registered history artifact changed")
                    else:
                        store.register_artifact(path, kind="history_proof_evidence")
            checked = validate_history_proof(root, history_path, expected)
            validation_path = write_json(
                root / "history_validation.json",
                {
                    "valid": checked["valid"],
                    "reasons": checked["reasons"],
                    "original_run_id": run_id,
                    "history_sha256": sha256(history_path),
                    "schedule_sha256": plan["schedule_sha256"],
                },
            )
            store.register_artifact(validation_path, kind="history_proof_evidence")
            if not checked["valid"]:
                status = "failed"
    return {
        "root": str(root),
        "history": str(root / "history.json"),
        "status": status,
        "original_run_id": run_id,
        "performance_claim": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit source-pinned acquisition CLI.

    Args:
        argv: Optional command arguments, excluding the module name.

    Returns:
        Zero for a valid inspection/completed proof, one for acquisition failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rendering-history")
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = acquire_history(
        args.experiment_root,
        args.run_id,
        source_worktree=args.source_worktree,
        source_commit=args.source_commit,
        output=args.output,
        interpreter=args.python,
        gpu_id=args.gpu_id,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return int(not args.dry_run and result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
