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


"""GenSim's bounded TaskSpec binding and evidence adapter (no execution loop)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from embodichain.lab.task_evaluation import UprightTaskEvaluator
from embodichain.task_spec import validate_task_template
from .semantic_graph import semantic_task_graph_hash

__all__: list[str] = []
SCHEMA = "gen_sim.taskspec_e2/v1"


def binding_for_graph(
    template: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any]:
    """Bind one explicit template to an existing E2 recipe, not its action order."""
    evaluator = UprightTaskEvaluator(template)
    if not graph["nodes"] or any(node["task_type"] != "E2" for node in graph["nodes"]):
        raise ValueError(
            "TaskSpec execution currently supports only a single E2 object."
        )
    entities = {
        str(node["call"].get("arguments", node["call"])["object"])
        for node in graph["nodes"]
        if "object" in node["call"].get("arguments", node["call"])
    }
    if len(entities) != 1 or len(graph["task_groups"]) != 1:
        raise ValueError(
            "TaskSpec execution currently supports only a single E2 object."
        )
    return {
        "schema_version": SCHEMA,
        "template": validate_task_template(template),
        "entity_id": next(iter(entities)),
        "graph_hash": semantic_task_graph_hash(graph),
        "template_hash": evaluator.template_hash,
    }


def read_binding(
    root: Path, graph: dict[str, Any], fingerprint: dict[str, Any]
) -> dict[str, Any] | None:
    """Verify the new sidecar before simulator imports; legacy bundles stay v2."""
    path = root / "task_spec_binding.json"
    if fingerprint.get("schema_version") != "semantic_integration_fingerprint/v3":
        if path.exists() or "task_spec_binding" in fingerprint:
            raise ValueError("TaskSpec sidecar requires fingerprint schema v3.")
        return None
    reference = fingerprint.get("task_spec_binding")
    if (
        type(reference) is not dict
        or set(reference) != {"uri", "content_hash"}
        or reference["uri"] != path.name
    ):
        raise ValueError("TaskSpec bundle must declare its binding content reference.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["content_hash"]:
        raise ValueError("TaskSpec binding content fingerprint drifted.")
    binding = json.loads(data)
    if type(binding) is not dict or set(binding) != {
        "schema_version",
        "template",
        "entity_id",
        "graph_hash",
        "template_hash",
    }:
        raise ValueError("Invalid TaskSpec binding fields.")
    expected = binding_for_graph(binding["template"], graph)
    if binding != expected:
        raise ValueError("TaskSpec binding does not match its graph/template.")
    return binding


def write_evidence(path: Path, value: dict[str, Any]) -> dict[str, str]:
    """Write one frozen host artifact and return its exact byte identity."""
    data = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.write_bytes(data)
    return {"uri": path.name, "content_hash": hashlib.sha256(data).hexdigest()}


def observe(
    env: Any, binding: dict[str, Any], *, scope: str, output: Path
) -> dict[str, Any]:
    """Read actual local poses without advancing or resetting the simulation."""
    target = getattr(env, "unwrapped", env)
    obj = target.sim.get_rigid_object(binding["entity_id"])
    if obj is None:
        raise ValueError("TaskSpec bound object has no simulation observation.")
    poses = obj.get_local_pose(to_matrix=True).detach().clone()
    if len(poses) != int(target.num_envs):
        raise ValueError("TaskSpec observation environment count mismatch.")
    result = UprightTaskEvaluator(binding["template"]).evaluate(poses, scope=scope)
    raw = poses.cpu().tolist()

    # Invalid observations remain inspectable without non-standard JSON NaN.
    def finite(value: Any) -> Any:
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value if math.isfinite(value) else None

    state = {
        "schema_version": "gen_sim.observed_pose/v1",
        "scope": scope,
        "entity_id": binding["entity_id"],
        "frame": "scene_z_up",
        "unit": "m",
        "env_ids": list(range(len(poses))),
        "episode_id": 0,
        "poses": finite(raw),
    }
    result["evidence"] = [write_evidence(output / f"{scope}_state.json", state)]
    result["env_ids"] = list(range(len(poses)))
    result["episode_id"] = 0
    return result


def final_acceptance(
    env: Any,
    binding: dict[str, Any],
    initial: dict[str, Any],
    program_result: Any,
    output: Path,
) -> tuple[bool, ...]:
    """Freeze the independent goal check before Gym finalizes episode metadata."""
    final = observe(env, binding, scope="task_goal", output=output)
    task_success = tuple(status == "pass" for status in final["status"])
    accepted = tuple(
        ok
        and initial["status"][index] == "pass"
        and not initial["goal_satisfied"][index]
        and program_result.success[index]
        for index, ok in enumerate(task_success)
    )
    program = write_evidence(
        output / "program_execution.json", program_result.to_metadata()
    )
    write_evidence(
        output / "task_evaluation.json",
        {
            "schema_version": "gen_sim.task_evaluation/v1",
            "template_hash": binding["template_hash"],
            "initial": deepcopy(initial),
            "final": final,
            "program_success": list(program_result.success),
            "task_success": list(task_success),
            "accepted": list(accepted),
            "program_result": program,
            "qualification": "observed_goal_only",
            "certificate_status": "unavailable",
            "qualification_limits": [
                "No asset-content, full-process or robustness certificate.",
                "Instantaneous upright goal; segment stability remains an execution check.",
            ],
        },
    )
    return accepted


def record_failure(
    env: Any,
    binding: dict[str, Any],
    initial: dict[str, Any] | None,
    output: Path,
    failure: dict[str, Any],
    *,
    num_envs: int,
) -> None:
    """Freeze failed-attempt observations before host reset; never imply acceptance."""
    try:
        final = observe(env, binding, scope="task_goal", output=output)
    except Exception as exc:
        final = {
            "status": ["unavailable"] * num_envs,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    write_evidence(
        output / "task_evaluation.json",
        {
            "schema_version": "gen_sim.task_evaluation/v1",
            "template_hash": binding["template_hash"],
            "initial": initial,
            "final": final,
            "execution_failure": failure,
            "accepted": [False] * num_envs,
            "certificate_status": "unavailable",
            "qualification": "failed_attempt",
        },
    )


def require_fresh_evidence_output(output: Path) -> None:
    """Reject reused TaskSpec attempt evidence instead of mixing episode identities."""
    names = (
        "initial_state.json",
        "task_goal_state.json",
        "initial_evaluation.json",
        "program_execution.json",
        "task_evaluation.json",
    )
    if any((output / name).exists() for name in names):
        raise ValueError(
            "TaskSpec execution requires a fresh evidence output directory."
        )
