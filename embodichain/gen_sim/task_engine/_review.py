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

"""Render a plan snapshot with an optional identity-checked native attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..task_spec import canonical_hash
from ._image_run import _image_preview

__all__: list[str] = []


def _attempt(plan: dict[str, Any], root: Path, nodes: set[str]) -> dict[str, Any]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    feedback = json.loads((root / "attempt_feedback.json").read_text())
    report = json.loads((root / "execution_report.json").read_text())
    keys = (
        "spec_digest",
        "graph_digest",
        "plan_digest",
        "bundle_digest",
        "scene_digest",
        "program_digest",
        "policy_sha256",
        "run_id",
        "attempt_id",
    )
    if (
        feedback.get("schema_version") != "embodichain.attempt-feedback/v1"
        or any(not feedback.get(k) or manifest.get(k) != feedback[k] for k in keys)
        or manifest["spec_digest"] != plan["spec_digest"]
        or manifest["graph_digest"] != plan["semantic_graph_sha256"]
        or manifest["plan_digest"] != plan["plan_digest"]
        or feedback["execution_report_digest"] != canonical_hash(report)
        or feedback["execution"] != report["status"]
    ):
        raise ValueError("Execution evidence does not match this plan and attempt")
    calls = {}
    for segment in report.get("segments", []):
        name = segment["name"]
        if name not in nodes or name in calls:
            raise ValueError("Execution segment cannot be paired with one graph call")
        runtime = segment.get("metadata", {}).get("runtime") or {}
        issues = [
            {
                "kind": e["kind"],
                "message": e.get("message"),
                "timestamp": e.get("timestamp"),
                "env_mask": e.get("env_mask"),
            }
            for e in runtime.get("events", [])
            if "failed" in e["kind"]
        ]
        calls[name] = {
            k: segment.get(k)
            for k in (
                "success",
                "failure_reason",
                "start_step",
                "end_step",
                "successes",
                "active",
                "failure_reasons",
                "start_steps",
                "end_steps",
            )
        }
        # DemoSegmentResult row arrays use global environment indices, even
        # when the runtime invocation itself only contains a subset of rows.
        env_ids = (
            list(range(len(segment["active"])))
            if segment.get("active")
            else runtime.get("env_ids", [])
        )
        calls[name].update(issues=issues, env_ids=env_ids)
    return {
        "status": feedback["execution"],
        "physical": feedback["physical_verdict"],
        "verdict": manifest.get("task_spec_verdict"),
        "step": feedback.get("capture_step"),
        "missing": feedback.get("missing_evidence", []),
        "calls": calls,
        "identity": {k: feedback[k] for k in keys},
    }


def write_review(plan_dir: Path, output: Path, run_dir: Path | None = None) -> None:
    from ..workflow import _source_image, load_task_plan

    plan = load_task_plan(plan_dir)
    root = Path(plan["root"])
    destination = output.expanduser().resolve()
    if destination.is_relative_to(root) or any(
        (p / "integration_fingerprint.json").is_file() for p in destination.parents
    ):
        raise ValueError(
            "Select a fresh HTML file outside the published plan or bundle"
        )
    keys = [
        "semantic_task_graph",
        "semantic_graph_validation",
        "task_spec",
        "task_candidates",
        "image_grounding",
    ]
    if "task_document" in plan:
        keys.extend(["task_spec_v1", "plan_coverage"])
    result = {k: json.loads((root / (k + ".json")).read_text()) for k in keys}
    if "qualifier_coverage" in plan:
        result["semantic_graph_validation"]["qualifier_coverage"] = plan[
            "qualifier_coverage"
        ]
    attempt = None
    if run_dir is not None:
        if "task_document" not in plan:
            raise ValueError("Native attempt review requires a version-2 plan")
        try:
            attempt = _attempt(
                plan, run_dir, {n["id"] for n in result["semantic_task_graph"]["nodes"]}
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            attempt = {"error": str(exc)}
    summary = json.loads((root / "run_manifest.json").read_text())
    image = _source_image(root, plan)
    html = _image_preview(result, image, summary, attempt=attempt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(html)


def main(
    argv: list[str] | None = None, *, prog: str = "embodichain gensim task view"
) -> None:
    """Generate a fresh standalone HTML review using the published source plan."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_review(args.plan, args.output, args.run)
    print(args.output.resolve())
