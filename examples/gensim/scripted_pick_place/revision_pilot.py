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

"""Run an evidence-driven correction on the prepared cube/platform deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import inspect
from pathlib import Path
from uuid import uuid4

import yaml

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash
from embodichain.gen_sim.task_spec.publication import (
    freeze_task_spec,
    evaluate_attempt_observations,
)


def _attempt(root: Path, revision: int) -> None:
    from examples.expansion.asset_generalization.run import (
        _template,
        _execute_batch,
        _measured,
        _world_bounds,
    )
    from embodichain.lab.sim import SimulationManager

    path = _template(root / f"attempt-{revision}", side=0.04, seed=7, batch_index=0)
    directory = path.parent
    deployment = yaml.safe_load(path.read_text())
    deployment["id"] = f"TaskSpecRevisionPilot-{revision}-v0"
    path.write_text(yaml.safe_dump(deployment))
    program_path = directory / "program.yaml"
    integration_path = directory / "integration.yaml"
    integration = yaml.safe_load(integration_path.read_text())
    support = integration["scene_binding"]["rigid_objects"][1]["affordances"][0]
    from copy import deepcopy

    offset = deepcopy(support)
    offset.update(entity_id="offset_landing", native_name="offset_landing")
    offset["object_target_pose"][3] += 0.15
    integration["scene_binding"]["rigid_objects"][1]["affordances"].append(offset)
    integration_path.write_text(yaml.safe_dump(integration))
    program = yaml.safe_load(program_path.read_text())
    if revision == 1:
        program["program"]["items"][-1]["steps"]["call"]["on"] = "offset_landing"
    program_path.write_text(yaml.safe_dump(program))
    run = json.loads((root / "run.json").read_text())
    spec = TaskSpec.from_dict(run["objective"])
    doc = json.loads((root / "task_spec_v1.json").read_text())
    from embodichain.gen_sim.task_spec.publication import decode_task_document

    decode_task_document(doc, spec)
    if doc["spec_digest"] != run["spec_digest"]:
        raise ValueError("Pilot task changed since creation")
    evaluator_digest = canonical_hash(
        [
            inspect.getsource(f)
            for f in (_world_bounds, _measured, evaluate_attempt_observations)
        ]
    )
    if evaluator_digest != run["evaluator_digest"]:
        raise ValueError("Pilot evaluator changed since creation")
    bundle = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*.yaml")
    }
    result = {
        "episodes": [],
        "run_id": run["run_id"],
        "attempt_id": str(uuid4()),
        "spec_digest": doc["spec_digest"],
        "program_digest": canonical_hash(program),
        "scene_digest": bundle["env.yaml"],
        "bundle_digest": canonical_hash(bundle),
        "evaluator_digest": evaluator_digest,
    }
    try:
        _execute_batch(path, seed=7, episodes=1, report=result)
        episode = result["episodes"][0]["validation"]
        measured = episode.get("measured_validation") or []
        checks = {c["check_id"]: c["status"] for c in measured}
        observations = {
            k: v == "passed" for k, v in checks.items() if v in {"passed", "failed"}
        }
        identity = {
            k: result[k]
            for k in (
                "run_id",
                "attempt_id",
                "spec_digest",
                "scene_digest",
                "program_digest",
                "bundle_digest",
            )
        }
        capture = {
            "identity": identity,
            "step": result["episodes"][0]["capture_step"],
            "observations": observations,
        }
        (directory / "capture.json").write_text(json.dumps(capture, indent=2))
        result["physical_report"] = evaluate_attempt_observations(
            doc, identity, capture
        )
    finally:
        (directory / "feedback.json").write_text(json.dumps(result, indent=2))
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Execute one bounded repair after an independently measured missed landing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2))
    args = parser.parse_args()
    root = args.output.resolve()
    if args.attempt:
        _attempt(root, args.attempt)
        return
    root.mkdir(parents=True, exist_ok=False)
    instruction = "Place the cube on the platform."
    spec = TaskSpec(
        task_id="cube_platform",
        instruction=instruction,
        roles=({"id": "cube"}, {"id": "platform", "kind": "surface"}),
        goal=tuple(
            {
                "predicate": "relation",
                "subject": "cube",
                "reference": "platform",
                "relation": key,
                "observation": key,
            }
            for key in ("cube_footprint_on_platform", "cube_bottom_at_platform_top")
        ),
    )
    from examples.expansion.asset_generalization.run import (
        TASK,
        _world_bounds,
        _measured,
    )

    source = TASK / "env.yaml"
    doc = freeze_task_spec(
        spec,
        image_ref="prepared_scene.yaml",
        image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        role_descriptions={"cube": "cube", "platform": "platform"},
        interpretation=[],
        generalization=[],
        source_kind="prepared_scene",
    )
    (root / "prepared_scene.yaml").write_bytes(source.read_bytes())
    (root / "task_spec_v1.json").write_text(json.dumps(doc, indent=2))
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": str(uuid4()),
                "spec_digest": doc["spec_digest"],
                "evaluator_digest": canonical_hash(
                    [
                        inspect.getsource(f)
                        for f in (
                            _world_bounds,
                            _measured,
                            evaluate_attempt_observations,
                        )
                    ]
                ),
                "objective": spec.to_dict(),
            }
        )
    )
    for revision in (1, 2):
        subprocess.run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--output",
                str(root),
                "--attempt",
                str(revision),
            ],
            check=True,
        )
        report = json.loads(
            (root / f"attempt-{revision}" / "feedback.json").read_text()
        )
        if revision == 1 and not any(
            c["status"] == "failed" and c["observed"] is False
            for c in report["physical_report"]["goal"]
        ):
            raise RuntimeError("No measured missed landing to correct")
        if revision == 2 and not (
            report["physical_report"]["certificate_ready"]
            and report["episodes"][0]["accepted"]
        ):
            raise RuntimeError("Corrected placement was not accepted")


if __name__ == "__main__":
    main()
