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

"""GenSim CLI and application workflow through configured Task Program execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import uuid
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._run_config import _RunCfg
    from .services.runtime import GenSimServices

__all__: list[str] = []


def main(
    argv: Sequence[str] | None = None, *, prog: str = "embodichain gensim run"
) -> None:
    """Generate, automatically bind, preflight and execute a configured task.

    Args:
        argv: Arguments after the workflow command.
        prog: Display name supplied by the unified GenSim dispatcher.
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Versioned GenSim run YAML."
    )
    parser.add_argument("--image", type=Path, help="Override the input photograph.")
    parser.add_argument("--instruction", help="Override the complete task instruction.")
    parser.add_argument(
        "--output",
        "--output-root",
        "--output_root",
        type=Path,
        help="New run directory.",
    )
    parser.add_argument(
        "--scene", type=Path, help="Reuse a scene workspace instead of generating it."
    )
    parser.add_argument("--task-id", "--task_id", help="Override the task identity.")
    parser.add_argument("--device", help="Execution device, for example cpu or cuda:0.")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args(argv)
    from ._run_config import _load_run_config

    try:
        cfg = _load_run_config(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    report = _run(cfg)
    print(f"Run manifest: {cfg.output / 'run_manifest.json'}")
    if report["status"] != "succeeded":
        raise SystemExit(1)


def _run(cfg: _RunCfg, *, services: GenSimServices | None = None) -> dict[str, Any]:
    from PIL import Image
    from .services.runtime import service_scope
    from .task_engine.orchestration.workflow_artifacts import (
        validate_run_output_directory,
        write_run_json,
    )
    from ._run_binding import _bind_deployment, _load_runtime
    from ._run_execution import _execute_deployment

    # Decode the trusted robot/policy inputs before incurring generation costs.
    runtime, policy, profile = _load_runtime(cfg)
    with Image.open(cfg.image) as image:
        image.verify()
    output = validate_run_output_directory(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / ("source_image" + cfg.image.suffix.lower())
    shutil.copyfile(cfg.image, source)
    manifest: dict[str, Any] = {
        "schema_version": "embodichain.gensim-run/v1",
        "task_id": cfg.task_id,
        "policy_sha256": hashlib.sha256(cfg.execution_policy.read_bytes()).hexdigest(),
        "instruction": cfg.instruction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "running",
        "run_id": uuid.uuid4().hex,
        "stages": {},
        "execution_ready": False,
        "runtime_evidence_status": "not_run",
        "task_spec_verdict": "not_evaluated",
    }
    stage = "task_plan"

    def update(status: str) -> None:
        manifest["stage"] = stage
        manifest["stages"][stage] = status
        write_run_json(output / "run_manifest.json", manifest)

    run = None
    try:
        with service_scope(services) as run:
            from .workflow import (
                load_generated_scene,
                load_task_plan,
                materialize_scene_generation,
                plan_image_task,
                prepare_task_deployment,
                qualify_task_scene,
                submit_scene_generation,
            )
            from .scene_providers import (
                SceneGenerationRequest,
                load_scene_generation_provider,
            )

            update("running")
            task = plan_image_task(
                source,
                cfg.instruction,
                output / "task",
                task_id=cfg.task_id,
                services=run,
            )
            plan = load_task_plan(task)
            manifest["spec_digest"] = plan.get("spec_digest")
            manifest["graph_digest"] = plan["semantic_graph_sha256"]
            update("succeeded")
            stage = "scene"
            update("running")
            scene = cfg.scene
            if scene is None:
                requirements = task / "scene_requirements.json"
                provider_profile = dict(cfg.scene_provider["profile"])
                if cfg.scene_provider["id"] == "legacy_scene_engine":
                    provider_profile["_services"] = run
                provider = load_scene_generation_provider(
                    cfg.scene_provider["id"], provider_profile
                )
                request = SceneGenerationRequest(
                    task_id=cfg.task_id,
                    image_path=source,
                    image_sha256=manifest["image_sha256"],
                    requirements_path=requirements,
                    requirements_sha256=hashlib.sha256(
                        requirements.read_bytes()
                    ).hexdigest(),
                    output=output / "scene-job",
                    options={},
                )
                request_key = hashlib.sha256(
                    (
                        cfg.scene_provider["id"] + request.request_hash + cfg.task_id
                    ).encode("utf-8")
                ).hexdigest()
                job = submit_scene_generation(
                    provider, request, idempotency_key=request_key
                )
                status = provider.inspect(job)
                if status.status != "succeeded":
                    raise RuntimeError(
                        f"Scene generation {status.status}: {status.reason or job.job_id}"
                    )
                scene = materialize_scene_generation(provider, job, output / "scene")
                write_run_json(output / "scene_job.json", status.to_dict())
            manifest["scene"] = str(scene)
            manifest["task"] = str(task)
            update("succeeded")
            stage = "binding"
            update("running")
            if (
                hashlib.sha256(cfg.execution_policy.read_bytes()).hexdigest()
                != manifest["policy_sha256"]
            ):
                raise ValueError("Execution policy changed after task planning")
            if (scene / "generated_scene.json").is_file():
                load_generated_scene(scene)
                qualification = qualify_task_scene(
                    task,
                    scene,
                    cfg.runtime,
                    cfg.execution_policy,
                    output / "qualification",
                    services=run,
                )
                qualification_value = json.loads(
                    (qualification / "scene_qualification.json").read_text()
                )
                if qualification_value["status"] != "accepted":
                    raise ValueError(
                        "Generated scene qualification "
                        + qualification_value["status"]
                        + ": "
                        + json.dumps(qualification_value["issues"], ensure_ascii=False)
                    )
                deployment = prepare_task_deployment(
                    task,
                    scene,
                    qualification,
                    cfg.runtime,
                    cfg.execution_policy,
                    output / "prepared",
                )
            else:
                deployment = _bind_deployment(
                    cfg, scene, task, runtime, policy, profile, services=run
                )
            manifest["deployment"] = str(deployment)
            update("succeeded")
            stage = "execution"
            manifest["execution_ready"] = True
            manifest["runtime_evidence_status"] = "running"
            from .task_spec import canonical_hash

            bundle_files = {}
            for path in deployment.parent.rglob("*"):
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        while block := stream.read(1024 * 1024):
                            digest.update(block)
                    bundle_files[path.relative_to(deployment.parent).as_posix()] = (
                        digest.hexdigest()
                    )
            manifest["attempt_id"] = uuid.uuid4().hex
            manifest["bundle_digest"] = canonical_hash(bundle_files)
            manifest["program_digest"] = bundle_files["program.json"]
            manifest["scene_digest"] = hashlib.sha256(
                (
                    (scene / "generated_scene.json")
                    if (scene / "generated_scene.json").is_file()
                    else (scene / "scene_export" / "scene_config.json")
                ).read_bytes()
            ).hexdigest()
            update("running")
            report = _execute_deployment(
                deployment, device=cfg.device, headless=cfg.headless
            )
            write_run_json(output / "execution_report.json", report)
            write_run_json(
                output / "attempt_feedback.json",
                {
                    "schema_version": "embodichain.attempt-feedback/v1",
                    **{
                        key: manifest[key]
                        for key in (
                            "run_id",
                            "attempt_id",
                            "spec_digest",
                            "graph_digest",
                            "bundle_digest",
                            "scene_digest",
                            "program_digest",
                            "policy_sha256",
                        )
                    },
                    "static_validation": "validated",
                    "planning_feasibility": "see_execution_segments",
                    "execution": report["status"],
                    "physical_verdict": "unavailable",
                    "persistence": "unverified",
                    "capture_step": report.get("length"),
                    "execution_report": "execution_report.json",
                    "execution_report_digest": canonical_hash(report),
                    "issues": [
                        {
                            "node_id": s.get("name"),
                            "code": s.get("failure_reason"),
                            "expected": "segment success",
                            "measured": s.get("success"),
                        }
                        for s in report.get("segments", [])
                        if s.get("success") is False
                    ],
                    "missing_evidence": ["independent_task_spec_evaluation"],
                },
            )
            manifest["runtime_evidence_status"] = report["status"]
            manifest["status"] = (
                "succeeded"
                if report["completed"] and report["all_success"]
                else "failed"
            )
            update(manifest["status"])
            from .task_engine._review import write_review

            try:
                write_review(task, output / "task_review.html", output)
            except (ValueError, OSError, KeyError, TypeError) as error:
                manifest["review_error"] = str(error)
                write_run_json(output / "run_manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        if stage == "execution":
            manifest["runtime_evidence_status"] = "unavailable"
        update(manifest["status"])
        raise
    finally:
        if run is not None:
            run.write_records(output / "service_calls.json")
    return manifest
