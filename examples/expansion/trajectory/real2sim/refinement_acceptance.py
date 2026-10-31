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

"""Qualify repair, replay ablation, repeated holding and explicit rejection.

Run: python examples/expansion/trajectory/real2sim/refinement_acceptance.py --output /tmp/refinement-acceptance
Every positive case uses normal run-task and the independent dataset audit.
Negative cases count only when execution evidence records the expected failure
and no episode receipt exists; an import/configuration failure is not a pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

__all__: list[str] = []


def main() -> None:
    """Run the fixed case ledger and retain every log and failed attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        EMBODICHAIN_SIM_EXIT_PROCESS="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    cases = (
        ("clean", "clean", True, False, True, None),
        ("biased", "bias", True, False, True, None),
        ("drifting", "drift", True, False, True, None),
        ("repeated", "bias", True, True, True, None),
        ("same_seed_repeat", "bias", True, False, True, None),
        ("raw_replay", "bias", False, False, False, "external grasp conflicts"),
        ("over_budget", "over_budget", True, False, False, "correction budgets"),
    )
    rows = []
    for name, error, refine, repeated, expected, reason in cases:
        directory = output / name
        command = [
            sys.executable,
            str(root / "prepare_refinement.py"),
            "--output",
            str(directory),
            "--error",
            error,
        ]
        if refine:
            command.append("--refine")
        if repeated:
            command.append("--repeated")
        subprocess.run(
            command, env=environment, check=True, capture_output=True, text=True
        )
        log = directory / "run-task.log"
        start = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "embodichain",
                    "run-task",
                    "--gym-config",
                    str(directory / "task.yaml"),
                    "--real2sim-profile",
                    str(root / "collection.tabletop.single.yaml"),
                    "--dataset-dir",
                    str(directory / "dataset"),
                    "--device",
                    "cpu",
                    "--headless",
                    "--renderer",
                    "no-render",
                    "--disable-sensor",
                    "--max_episodes",
                    "1",
                ],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        receipts = list(directory.glob("dataset/*/meta/real2sim/receipts/*.json"))
        failures = list(
            directory.glob("dataset/*/meta/real2sim/failed-attempts/*.json")
        )
        row = {
            "case": name,
            "expected_accepted": expected,
            "run_task_exit_code": result.returncode,
            "wall_seconds": time.perf_counter() - start,
            "confirmed_receipts": len(receipts),
            "failed_attempts": len(failures),
            "log": str(log),
            "passed": False,
        }
        if expected:
            datasets = list(
                directory.glob("dataset/*/meta/real2sim/collection-report.json")
            )
            if result.returncode == 0 and len(receipts) == 1 and len(datasets) == 1:
                audit = directory / "audit.json"
                with (directory / "audit.log").open("w") as stream:
                    audited = subprocess.run(
                        [
                            sys.executable,
                            str(root / "audit.py"),
                            "--dataset",
                            str(datasets[0].parents[2]),
                            "--profile",
                            str(root / "collection.tabletop.single.yaml"),
                            "--output",
                            str(audit),
                        ],
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                row.update(passed=audited.returncode == 0, audit=str(audit))
        else:
            messages = [json.loads(path.read_text()) for path in failures]
            row["failure_evidence"] = messages
            row["passed"] = (
                result.returncode != 0
                and not receipts
                and len(messages) == 1
                and reason.lower() in json.dumps(messages).lower()
            )
        if name == "same_seed_repeat" and row["passed"]:
            original = next(item for item in rows if item["case"] == "biased")
            dataset_paths = [
                next(
                    (output / case / "dataset").glob(
                        "*/meta/embodichain_episodes.jsonl"
                    )
                )
                for case in ("biased", name)
            ]
            metadata = [
                json.loads(path.read_text().splitlines()[0]) for path in dataset_paths
            ]

            def identities(item: dict[str, object]) -> tuple[object, ...]:
                return (
                    item["candidate_id"],
                    tuple(
                        (
                            call["call_index"],
                            call["prototype_sha256"],
                            call["operator"],
                            call["refinement"]["profile_sha256"],
                        )
                        for call in item["real2sim_lineage"]["calls"]
                    ),
                )

            matched = identities(metadata[0]) == identities(metadata[1])
            positions = []
            for path, item in zip(dataset_paths, metadata):
                with np.load(
                    path.parents[1] / item["real2sim_observed_geometry"]["path"],
                    allow_pickle=False,
                ) as values:
                    positions.append(values["object_pose"][-1, :3].copy())
            difference = float(np.linalg.norm(positions[0] - positions[1]))
            row.update(
                same_seed_identity_matches=matched,
                final_position_difference_m=difference,
            )
            row["passed"] = original["passed"] and matched and difference <= 0.003
        rows.append(row)
        print(f"{name}: {'passed' if row['passed'] else 'FAILED'}", flush=True)
        report = {
            "schema": "real2sim-refinement-acceptance/v1",
            "origin": "synthetic_reference",
            "expected_cases": len(cases),
            "completed_cases": len(rows),
            "passed_cases": sum(item["passed"] for item in rows),
            "real_data_qualified": False,
            "continuous_collision_qualified": False,
            "cases": rows,
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
