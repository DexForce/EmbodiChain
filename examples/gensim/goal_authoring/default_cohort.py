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
"""Run fresh paired production CLIs with a hard five-HTTP-request cap per plan."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def meter(send, calls):
    def metered(request, **kwargs):
        if len(calls) >= 5:
            raise ValueError("Five-request cohort budget exhausted")
        calls.append(json.loads(request.data)["model"])
        return send(request, **kwargs)

    return metered


def trial(args):
    from embodichain.__main__ import main
    from embodichain.gen_sim.services.adapters import chat

    root, dataset, case, route, instruction = args
    folder = root / case / route
    folder.mkdir(parents=True)
    image = dataset / f"{case}.png"
    argv = [
        "embodichain",
        "gensim",
        "task",
        "plan",
        "--image",
        str(image),
        "--instruction",
        instruction,
        "--task-id",
        case,
        "--output",
        str(folder / "plan"),
    ]
    if route == "template":
        argv += ["--route", "template"]
    calls = []
    result = dict(
        case=case,
        route=route,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        instruction=instruction,
        calls=calls,
    )

    try:
        with (
            patch.object(sys, "argv", argv),
            patch.object(chat, "urlopen", meter(chat.urlopen, calls)),
        ):
            code = main()
        if code:
            raise RuntimeError(f"CLI exit {code}")
        from embodichain.gen_sim.workflow import load_task_plan

        load_task_plan(folder / "plan")
        result["published"] = True
    except (Exception, SystemExit) as exc:
        result.update(published=False, error=f"{type(exc).__name__}: {exc}"[:1800])
    (folder / "trial.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(case, route, result["published"], len(calls), flush=True)
    return result


def read(path):
    return json.loads(path.read_text())


def visual_diagnostic(folder):
    """Retain the final response keys alongside the validator's actual diagnostic."""
    (failure,) = folder.glob("plan.failure-*.json")
    attempt = read(failure)["artifacts"]["model_attempts"][-1]
    raw = attempt["response"].strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].removeprefix("json").strip()
    try:
        response = json.loads(raw)
    except ValueError:
        response = None
    return {
        "last_visual_response_keys": (
            sorted(response) if isinstance(response, dict) else None
        ),
        "last_visual_errors": attempt.get("errors", []),
    }


def assess(folder, installation):
    from embodichain.gen_sim.services.adapters import chat

    result = read(folder / "trial.json")
    plan = folder / "plan"
    result.update(
        visual="not_verified",
        qualification="not_verified",
        preflight="not_verified",
        qualification_calls=[],
    )
    if not result["published"]:
        if "Visual grounding invalid" in result["error"]:
            result["visual"] = "failed"
            result.update(visual_diagnostic(folder))
        return result
    manifest = read(plan / "task_plan_manifest.json")
    result["manifest_sha256"] = hashlib.sha256(
        (plan / "task_plan_manifest.json").read_bytes()
    ).hexdigest()
    result["stage"] = manifest.get("stage", manifest.get("visual_status"))
    if manifest.get("stage") == "authoring_only":
        return result
    pending = manifest.get("pending_roles", [])
    result["pending_roles"] = pending
    result["visual"] = (
        "pending"
        if pending or manifest.get("visual_status") == "pending"
        else "complete"
    )
    if installation is None or result["case"] != "task1186":
        result["reason"] = "no matching installation"
        return result
    # Same calibrated native inventory for both routes; no plan input comes from it.
    import shutil
    import trimesh
    import yaml
    from embodichain.gen_sim.workflow import (
        load_generated_scene,
        qualify_task_scene,
        prepare_task_deployment,
    )

    previous = load_generated_scene(installation / "scene")
    if previous["source"]["image_sha256"] != result["image_sha256"]:
        raise ValueError("Installation source mismatch")
    scene = folder / "scene"
    shutil.copytree(installation / "scene", scene)
    previous.pop("root")
    previous["requirements_sha256"] = hashlib.sha256(
        (plan / "scene_requirements.json").read_bytes()
    ).hexdigest()
    for obj in previous["objects"]:
        extent = trimesh.load(scene / obj["mesh_path"], force="mesh").extents
        obj["description"] += " mesh dimensions (metres): " + str(
            extent.round(3).tolist()
        )
    (scene / "generated_scene.json").write_text(
        json.dumps(previous, ensure_ascii=False, indent=2)
    )
    runtime, policy = installation / "runtime.yaml", installation / "policy.yaml"
    integ = yaml.safe_load(
        (installation / "prepared/deployment/integration.yaml").read_text()
    )
    calls = list(result["calls"])
    try:
        with patch.object(chat, "urlopen", meter(chat.urlopen, calls)):
            q = qualify_task_scene(plan, scene, runtime, policy, folder / "qualified")
            report = read(q / "scene_qualification.json")
            result["qualification"] = report["status"]
            result["issues"] = report["issues"]
            if report["status"] == "accepted":
                result["bindings"] = report["bindings"]
                result["preflight"] = "failed"
                prepare_task_deployment(
                    plan,
                    scene,
                    q,
                    runtime,
                    policy,
                    folder / "prepared",
                    skill_options=integ["profile"]["action_options"],
                    handover_pose=integ["runtime_services"]["handover_pose_providers"][
                        0
                    ],
                )
                result["compiled"] = read(folder / "prepared/preflight_report.json")
                result["preflight"] = "passed"
    except Exception as exc:
        if result["qualification"] == "not_verified":
            result["qualification"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["qualification_calls"] = calls[len(result["calls"]) :]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument(
        "--assess",
        action="store_true",
        help="Assess saved trials without reauthoring; use a freshly generated batch.",
    )
    p.add_argument("--fruit-installation", type=Path)
    a = p.parse_args()
    if a.assess:
        rows = [
            assess(a.output / f"task{i}" / r, a.fruit_installation)
            for i in range(1171, 1191)
            for r in ("default", "template")
        ]
        (a.output / "results.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
        )
        sys.exit(0)
    a.output.mkdir(parents=True, exist_ok=False)
    src = json.loads((a.dataset / "task_descriptions.json").read_text())
    with ProcessPoolExecutor(a.workers) as pool:
        list(
            pool.map(
                trial,
                (
                    (a.output, a.dataset, k, r, src[k]["zh"])
                    for k in [f"task{i}" for i in range(1171, 1191)]
                    for r in ("default", "template")
                ),
            )
        )
