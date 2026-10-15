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

"""Run predeclared E1 cases or sealed prepared cases through the production runner.

This campaign helper selects an expanded member before compilation. It does not
implement motion, physics, effect verification or an alternative executor.
Run: python -m scripts.benchmark.task_program.gen_sim_e1 --help
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile

from embodichain.gen_sim.task_engine.agent import (
    derive_scene_request,
    derive_success_spec,
)
from embodichain.gen_sim.task_engine.contracts import (
    canonical_hash,
    validate_task_candidate,
)
from embodichain.gen_sim.task_engine.orchestration.contracts import (
    validate_role_bindings,
)
from embodichain.gen_sim.task_engine.orchestration.execution_resources import (
    _other_sim_processes,
    run_e8_process,
)
from embodichain.gen_sim.task_engine.orchestration.source_scene import prepare_scene
from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
    load_frozen_physical_input,
)
from embodichain.gen_sim.task_engine.semantic_planner import (
    SemanticTaskPlanner,
    _expand_multi_object_steps,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.gen_sim.task_engine.task_spec import task_spec_from_candidate
from embodichain.gen_sim.task_engine.workflow import _headless_renderer_for_bundle

WORKSPACE = Path(__file__).resolve().parents[3]


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_member(
    candidate: dict, bindings: dict, uid: str, task_id: str
) -> tuple[dict, dict, dict]:
    expanded, expanded_bindings, lineage = _expand_multi_object_steps(
        validate_task_candidate(candidate), validate_role_bindings(bindings)
    )
    matches = [
        member
        for members in lineage.values()
        for member in members
        if member["object_uid"] == uid
    ]
    if len(matches) != 1:
        raise ValueError(
            "The declared developer/holdout UID must select exactly one expanded member."
        )
    member = matches[0]
    selected = deepcopy(expanded)
    step = next(
        step for step in selected["draft"]["steps"] if step["id"] == member["group_id"]
    )
    if step["task_type"] != "E1":
        raise ValueError("This experiment only selects an existing E1 member.")
    step["depends_on"] = []
    step["object"]["reference"] = uid
    selected["draft"].update(
        task_id=task_id,
        instruction=f"Independent single-block E1 diagnostic for {uid}; not original-task credit.",
        steps=[step],
    )
    selected["scene_request"] = derive_scene_request(selected["draft"])
    selected["success_spec"] = derive_success_spec(selected["draft"])
    selected["semantic_hash"] = canonical_hash(selected["draft"]["steps"])
    selected = validate_task_candidate(selected)
    wanted = {row["reference_id"] for row in selected["scene_request"]["references"]}
    expanded_bindings["task_id"] = task_id
    for field in ("reference_bindings", "role_bindings"):
        expanded_bindings[field] = {
            key: value
            for key, value in expanded_bindings[field].items()
            if key in wanted
        }
    return selected, validate_role_bindings(expanded_bindings), member


def _admit(worker: Path, remaining: list[str]) -> None:
    free = list(
        map(
            int,
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines(),
        )
    )
    competitors = _other_sim_processes(os.getpgrp())
    output = Path(remaining[remaining.index("--execution-output") + 1])
    _write(
        output / "post_wait_admission.json",
        {"free_vram_mib": free, "competitors": competitors, "pid": os.getpid()},
    )
    if competitors or min(free) < 8192:
        raise RuntimeError(
            "Native admission rejected: competing simulation or insufficient VRAM."
        )
    os.execv(sys.executable, [sys.executable, "-u", str(worker), *remaining])


def _publish(root: Path, receipt: dict) -> None:
    status = _read(root / "status.json")
    row = {
        key: receipt.get(key)
        for key in (
            "id",
            "case",
            "status",
            "head",
            "source_digest",
            "physical_digest",
            "round",
            "wall_seconds",
            "layers",
            "first_failure",
            "receipt",
            "report",
            "videos",
        )
    }
    status["runs"] = [run for run in status["runs"] if run["id"] != row["id"]] + [row]
    status.update(
        updated_at=datetime.now(timezone.utc).isoformat(),
        head=receipt["head"],
        round=receipt["round"],
        status=receipt["status"],
    )
    _write(root / "status.json", status)


def _native_summary(report: dict, evaluation: dict) -> dict:
    runtime = report.get("runtime_result", {})
    program_success = runtime.get("program_success", [])
    completed = (
        bool(runtime.get("completed"))
        and bool(program_success)
        and all(program_success)
    )
    environments = evaluation.get("environments", [])
    qualified = (
        completed
        and report["status"] == "succeeded"
        and bool(environments)
        and all(
            row["status"] == "satisfied" and row["certificate_ready"]
            for row in environments
        )
    )
    calls = [
        call
        for segment in runtime.get("segments", [])
        for call in segment.get("metadata", {}).get("runtime", {}).get("calls", [])
    ]
    return {
        "status": "qualified" if qualified else "unverified" if completed else "failed",
        "layers": {
            "software": "validated",
            "planning": (
                "calls_completed"
                if calls and all(call["status"] == "completed" for call in calls)
                else "incomplete"
            ),
            "runtime": "program_completed" if completed else "incomplete",
            "task_spec": ",".join(row["status"] for row in environments) or "missing",
        },
        "first_failure": (
            None
            if qualified
            else next(
                (
                    row.get("terminal_reason")
                    for row in report.get("environments", [])
                    if not row.get("success")
                ),
                "Missing complete qualification evidence.",
            )
        ),
    }


def main() -> None:
    """Prepare and execute exactly one predeclared case, without automatic retry."""
    if len(sys.argv) > 1 and sys.argv[1] == "--admit":
        _admit(Path(sys.argv[2]), sys.argv[3:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", choices=("dev", "holdout", "control"), required=True)
    parser.add_argument("--round", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.plan.resolve().parent
    plan = _read(args.plan)
    if (
        plan["schema_version"]
        not in {"gensim.bounded-e-plan/v1", "gensim.prepared-e-plan/v1"}
        or args.round > plan["max_rounds"]
    ):
        raise ValueError("Unsupported or exhausted experiment plan.")
    if (plan["seed"], plan["num_envs"], plan["native_wall_guard_seconds"]) != (
        0,
        1,
        1200,
    ):
        raise ValueError(
            "The existing worker owns the fixed seed/env/wall-guard contract."
        )
    if (
        len(list((root / "attempts").glob("*/admission.json")))
        >= plan["max_native_runs"]
    ):
        raise RuntimeError("Predeclared native run budget exhausted.")
    run_id = f"r{args.round}-{args.case}"
    if "allowed_runs" in plan and run_id not in plan["allowed_runs"]:
        raise ValueError("This case/round is not in the predeclared schedule.")
    output = root / "attempts" / run_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
    ).strip()
    receipt = {
        "id": run_id,
        "case": args.case,
        "round": args.round,
        "status": "preparing",
        "head": head,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "original_task_credit": False,
        "receipt": str(
            (output / "controller_receipt.json").relative_to(
                WORKSPACE / "outputs/gen_sim"
            )
        ),
        "layers": {
            "software": "not_run",
            "planning": "not_run",
            "runtime": "not_run",
            "task_spec": "not_run",
        },
        "videos": [],
        "first_failure": None,
    }
    pinned: dict[str, str] = {}
    try:
        prepared = plan["schema_version"] == "gensim.prepared-e-plan/v1"
        case_input = plan["cases"][args.case] if prepared else plan
        old = WORKSPACE / case_input["reference_bundle"]
        frozen = load_frozen_physical_input(
            prepare_scene(WORKSPACE / case_input["source_config"]), old
        )
        if frozen.physical_digest != case_input["expected_physical_digest"]:
            raise RuntimeError(
                "Physical input changed between declared cases; not admitted."
            )
        receipt["physical_digest"] = frozen.physical_digest
        uid = None if prepared else plan["cases"][args.case]
        bundle = (
            WORKSPACE / case_input["prepared_bundles"][str(args.round)]
            if prepared
            else old
        )
        if (
            prepared
            and bundle != old
            and _read(bundle / "physical_input.json")["physical_digest"]
            != frozen.physical_digest
        ):
            raise ValueError("Prepared bundle does not use the declared fixed model.")
        if uid is not None:
            bindings = _read(old / "role_bindings.json")
            candidate = next(
                row
                for row in _read(old / "task_candidate_set.json")["candidates"]
                if row["candidate_id"] == bindings["candidate_id"]
            )
            candidate, bindings, member = _select_member(
                candidate, bindings, uid, run_id
            )
            spec = task_spec_from_candidate(
                candidate,
                metadata={
                    "qualification_scope": "diagnostic_single_E1",
                    "original_task_credit": False,
                    "parent_bundle": str(old),
                    "parent_candidate_sha256": _sha(old / "task_candidate_set.json"),
                    "selected_member": member,
                },
            )
            planned = SemanticTaskPlanner().plan_spec(
                candidate, bindings, frozen.scene.planner_objects, task_spec=spec
            )
            bundle = output / "bundle"
            generate_task_program_bundle(
                planned.graph,
                frozen,
                bundle,
                robot_profile="dual_franka",
                task_spec=planned.task_spec,
                task_candidate=planned.candidate,
                role_bindings=planned.role_bindings,
                step_lineage=planned.lineage,
                support_contract_revision=plan["support_contract_revision"],
                pick_contact_revision=plan["pick_contact_revision"],
            )
        receipt.update(
            bundle=str(bundle),
            selected_uid=uid,
            preparation_seconds=time.perf_counter() - start,
        )
        receipt["layers"]["software"] = "validated"
        names = (
            subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "embodichain",
                    "embodichain_tasks",
                    "agent_context",
                ],
                cwd=WORKSPACE,
            )
            .decode()
            .split("\0")
        )
        source = {
            name: _sha(WORKSPACE / name)
            for name in names
            if name and (name.endswith(".py") or name.startswith("agent_context/"))
        }
        receipt["source_digest"] = canonical_hash(source)
        if (
            args.round == 0
            and receipt["source_digest"] != plan["baseline_source_digest"]
        ):
            raise RuntimeError(
                "Production source changed within the baseline cohort; not admitted."
            )
        pinned = {str(WORKSPACE / name): digest for name, digest in source.items()}
        pinned.update(frozen.input_files)
        cache_inputs = {
            str(path): _sha(path)
            for directory in plan.get("cache_roots", [])
            for path in Path(directory).glob("*.npy")
        }
        pinned.update(cache_inputs)
        receipt["existing_grasp_cache_files"] = len(cache_inputs)
        for directory in (old, bundle):
            pinned.update(
                {
                    str(path): _sha(path)
                    for path in directory.rglob("*")
                    if path.is_file()
                }
            )
        for path in (Path(__file__), args.plan.resolve(), WORKSPACE / plan["worker"]):
            pinned[str(path)] = _sha(path)
        manifest = output / "drift_inputs.json"
        _write(manifest, pinned)
        with zipfile.ZipFile(
            output / "source.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for name in source:
                archive.write(WORKSPACE / name, name)
            archive.write(Path(__file__), "_experiment/gen_sim_e1.py")
            archive.write(args.plan, "_experiment/plan.json")
            archive.writestr(
                "_git/unstaged.patch",
                subprocess.check_output(["git", "diff", "--binary"], cwd=WORKSPACE),
            )
        receipt["source_archive_sha256"] = _sha(output / "source.zip")
        if args.prepare_only:
            receipt["status"] = "prepared_not_native_qualified"
            return
        command = [
            sys.executable,
            "-u",
            str(Path(__file__)),
            "--admit",
            str(WORKSPACE / plan["worker"]),
            "--drift-manifest",
            str(manifest),
            "--drift-receipt",
            str(output / "drift_receipt.json"),
            "--bundle",
            str(bundle),
            "--execution-output",
            str(output),
            "--num_envs",
            "1",
            "--seed",
            "0",
            "--headless",
            "--renderer",
            _headless_renderer_for_bundle(bundle),
            "--filter_dataset_saving",
            "--failure-policy",
            "stop",
        ]
        receipt.update(status="running", command=command)
        _write(output / "admission.json", receipt)
        _publish(root, receipt)
        result = run_e8_process(command, output / "native.log")
        receipt.update(status="native_returned_unverified", exit_code=result.returncode)
        report_path = output / "execution_report.json"
        if report_path.is_file():
            report = _read(report_path)
            receipt["report"] = str(
                report_path.relative_to(WORKSPACE / "outputs/gen_sim")
            )
            receipt["report_status"] = report["status"]
            evaluation_path = output / "task_spec_evaluation.json"
            receipt.update(
                _native_summary(
                    report, _read(evaluation_path) if evaluation_path.is_file() else {}
                )
            )
            if prepared:
                receipt["qualification_scope"] = plan["qualification_scope"]
                receipt["status"] = "native_returned_unverified"
        receipt["videos"] = [
            {
                "path": str(path.relative_to(WORKSPACE / "outputs/gen_sim")),
                "sha256": _sha(path),
            }
            for path in output.rglob("*.mp4")
        ]
    except Exception as exc:
        receipt.update(status="failed", first_failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.update(
            wall_seconds=time.perf_counter() - start,
            source_input_drift=[
                name
                for name, digest in pinned.items()
                if not Path(name).is_file() or _sha(Path(name)) != digest
            ],
        )
        _write(output / "controller_receipt.json", receipt)
        _publish(root, receipt)
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
