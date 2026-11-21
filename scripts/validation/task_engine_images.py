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

"""Run the pinned image cohort with independent semantic-only acceptance.

This explicit network/model qualification is not part of ordinary unit tests.
"""

from __future__ import annotations
import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from urllib.request import urlopen

__all__: list[str] = []


def main(argv: Sequence[str] | None = None) -> None:
    """Run all twenty complete Chinese instructions and retain every outcome.

    Args:
        argv: Optional validation harness arguments.
    """
    repo = Path(__file__).resolve().parents[2]
    expectations = json.loads(
        (
            repo / "tests/gen_sim/task_engine/data/image_task_expectations.json"
        ).read_text()
    )
    revision = expectations["dataset_revision"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/embodichain/task-engine-cases" / revision,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = args.data_dir.expanduser().resolve()
    data.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/{revision}/gen_sim_assets/"

    def fetch(relative, path):
        if not path.exists():
            with urlopen(base + relative, timeout=90) as stream:
                path.write_bytes(stream.read())

    fetch("task_descriptions.json", data / "task_descriptions.json")
    if (
        hashlib.sha256((data / "task_descriptions.json").read_bytes()).hexdigest()
        != expectations["descriptions_sha256"]
    ):
        raise ValueError("Pinned instruction digest mismatch")
    descriptions = json.loads((data / "task_descriptions.json").read_text())
    for case in sorted(expectations["cases"]):
        image = data / (case + ".png")
        fetch("demo_image/" + image.name, image)
        if (
            hashlib.sha256(image.read_bytes()).hexdigest()
            != expectations["image_sha256"][image.name]
        ):
            raise ValueError(f"Pinned image digest mismatch: {image}")
    spec = importlib.util.spec_from_file_location(
        "image_case_acceptance",
        repo / "tests/gen_sim/task_engine/image_case_acceptance.py",
    )
    acceptance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acceptance)
    (output / "logs").mkdir()
    (output / "runs").mkdir()
    metadata = {
        "dataset": "DexForceAI/embodichain_data",
        "revision": revision,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
        ).hexdigest(),
        "expectations_sha256": hashlib.sha256(
            json.dumps(expectations, sort_keys=True).encode()
        ).hexdigest(),
    }
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True
    )
    metadata["source_sha256"] = {
        line[3:]: hashlib.sha256((repo / line[3:]).read_bytes()).hexdigest()
        for line in changed.splitlines()
        if (repo / line[3:]).is_file()
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def run(case):
        start = perf_counter()
        destination = output / "runs" / case
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "embodichain",
                    "task-engine",
                    "--image",
                    str(data / (case + ".png")),
                    "--instruction",
                    descriptions[case]["zh"],
                    "--task-id",
                    case,
                    "--output",
                    str(destination),
                ],
                cwd=repo,
                env={**os.environ, "PYTHONPATH": str(repo)},
                capture_output=True,
                text=True,
                timeout=600,
            )
            log = proc.stdout + proc.stderr
            status = "passed" if proc.returncode == 0 else "generation_failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            log = "Timed out after 600 seconds"
        (output / "logs" / (case + ".log")).write_text(log)
        row = {
            "case": case,
            "status": status,
            "seconds": round(perf_counter() - start, 2),
            "errors": [],
        }
        if status == "passed":
            artifacts = {
                p.stem: json.loads(p.read_text()) for p in destination.glob("*.json")
            }
            errors = acceptance.check_image_case(case, artifacts)
            row.update(
                errors=errors,
                coverage=artifacts["semantic_graph_validation"]["coverage"],
                nodes=len(artifacts["semantic_task_graph"]["nodes"]),
            )
            if errors:
                row["status"] = "acceptance_failed"
        else:
            row["errors"] = [log.strip().splitlines()[-1]]
        return row

    results = []
    # These calls use the configured cloud VLM, not the mutually exclusive local GPU services.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed(
            {pool.submit(run, case): case for case in sorted(expectations["cases"])}
        ):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (output / "results.json").write_text(
                json.dumps(
                    sorted(results, key=lambda x: x["case"]),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "status",
                "seconds",
                "nodes",
                "visual_status",
                "pending_bindings",
                "errors",
            ]
        )
        for row in sorted(results, key=lambda x: x["case"]):
            coverage = row.get("coverage", {})
            writer.writerow(
                [
                    row["case"],
                    row["status"],
                    row["seconds"],
                    row.get("nodes", 0),
                    coverage.get("visual_status", "not_run"),
                    coverage.get("pending_bindings", 0),
                    "; ".join(row["errors"]),
                ]
            )
    passed = sum(r["status"] == "passed" for r in results)
    print(
        f"Semantic-only acceptance: {passed}/20; no physical execution qualification."
    )
    if passed != 20:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
