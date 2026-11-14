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
"""Goal-first authoring with conservative, argument-derived effect checks."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.gen_sim.services.runtime import GenSimServices

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash, validate_task_spec
from embodichain.gen_sim.task_spec.publication import freeze_task_spec
from embodichain.lab.task_program.language.semantic_contracts import (
    semantic_planning_catalog,
    validate_semantic_call_payload,
)
from .semantic_graph import validate_semantic_task_graph
from .orchestration.workflow_artifacts import WorkflowRunArtifacts, write_run_json

__all__: list[str] = []


def _facts(call):
    kind = call.get("call_id", call["kind"])
    args = call.get("arguments", call)
    obj = args.get("object")
    arms = call.get("resources", {})
    if kind == "simulation.open_door":
        yield "articulation_joint_near", dict(
            subject=obj, target_state=args["state"], joint_kind="revolute"
        )
    elif kind == "hand_over":
        yield "handover_complete", dict(
            subject=obj, transfer_arm=arms["source"], receive_arm=arms["destination"]
        )
    elif kind == "simulation.place_relative":
        yield "relation", dict(
            subject=obj,
            reference=args["reference"],
            relation=args["relation"],
            required_arm=arms.get("primary"),
        )
    elif kind == "place":
        for rel in ("on", "inside"):
            if args.get(rel):
                yield "relation", dict(
                    subject=obj,
                    reference=args[rel],
                    relation=rel,
                    required_arm=arms.get("primary"),
                )
    elif (
        kind == "simulation.coordinated_transport"
        and arms.get("left")
        and arms.get("right")
        and arms["left"] != arms["right"]
    ):
        # A transport target is a pose, not a support surface or containment proof.
        yield "semantic_goal", dict(
            subject=obj, target=args["target"], operation="coordinated_transport"
        )


def _coverage(spec, graph):
    roles = {r.name: r for r in spec.roles}
    calls = [n["call"] for n in graph["nodes"]]
    facts = [list(_facts(c)) for c in calls]
    catalog = semantic_planning_catalog()["calls"]
    issues = []
    diagnostics = []
    held = {}
    for node, call in zip(graph["nodes"], calls):
        cid = call.get("call_id", call["kind"])
        args = call.get("arguments", call)
        effect = catalog[cid]["effect"]
        res = call.get("resources", {})
        arm = res.get("primary", "auto")
        obj = args.get("object")
        before = dict(held)
        if effect in {"acquire", "acquire_align"}:
            if arm in held or obj in held.values():
                issues.append(node["id"])
            held[arm] = obj
        elif effect == "transfer":
            if held.get(res["source"]) != obj or res["destination"] in held:
                issues.append(node["id"])
            held.pop(res["source"], None)
            held[res["destination"]] = obj
        elif effect in {"release", "held"}:
            if arm not in held or obj is not None and held[arm] != obj:
                issues.append(node["id"])
            if effect == "release":
                held.pop(arm, None)
        elif (
            (
                effect == "coordinated_release"
                and (
                    held
                    or not res.get("left")
                    or not res.get("right")
                    or res["left"] == res["right"]
                )
            )
            or effect == "control"
            and arm in held
        ):
            issues.append(node["id"])
        if issues and issues[-1] == node["id"]:
            diagnostics.append(
                dict(
                    node_id=node["id"],
                    call=cid,
                    object=obj,
                    requested=res,
                    held_before=before,
                    preconditions=catalog[cid]["preconditions"],
                )
            )
            held = before

    def matches(p, fact):
        name, args = fact
        return (
            p.name == name
            and p.operator == "eq"
            and p.expected is True
            and "subject" in p.arguments
            and p.arguments["subject"] in roles
            and all(
                roles[r].is_single_member
                for r in (p.arguments.get("subject"), p.arguments.get("reference"))
                if r in roles
            )
            and all(k in args and v == args[k] for k, v in p.arguments.items())
            and all(
                k in p.arguments
                for k in args
                if k
                not in {"required_arm", "transfer_arm", "receive_arm", "joint_kind"}
            )
        )

    def indices(p):
        return {i for i, fs in enumerate(facts) if any(matches(p, f) for f in fs)}

    mapped = {}
    missing = []
    for p in spec.goal:
        hits = indices(p)
        # Only the last operation on a subject can witness its terminal state.
        relevant = [
            i
            for i, c in enumerate(calls)
            if c.get("arguments", c).get("object") == p.arguments.get("subject")
        ]
        if relevant and relevant[-1] in hits:
            mapped[p.id] = graph["nodes"][relevant[-1]]["id"]
        else:
            missing.append(p.id)
    remaining = list(spec.milestones)
    done = {}
    while remaining:
        m = next(m for m in remaining if all(k in done for k in m.after))
        hits = set.intersection(*(indices(p) for p in m.achieve))
        start = max((done[k] for k in m.after), default=-1)
        i = min((i for i in hits if i > start), default=len(calls))
        done[m.id] = i
        if i == len(calls):
            missing.append(m.id)
        else:
            mapped[m.id] = graph["nodes"][i]["id"]
        remaining.remove(m)
    for key in ("invariants", "temporal"):
        if getattr(spec, key):
            missing.append(key)
    return {
        "status": "uncovered" if missing or issues else "covered",
        "producers": mapped,
        "uncovered": missing,
        "resource_conflicts": issues,
        "resource_diagnostics": diagnostics,
    }


def author_goal_plan(
    instruction: str,
    task_id: str,
    objective: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate an agent's episode contract and ordered call proposal without a template."""
    if set(objective) - {
        "roles",
        "init",
        "goal",
        "milestones",
        "invariants",
        "temporal",
        "requirements",
    }:
        raise ValueError("Unknown objective declaration")
    spec = TaskSpec.from_dict(
        validate_task_spec(
            {**objective, "instruction": instruction, "task_id": task_id}
        )
    )
    known = {r.name for r in spec.roles}
    for predicate in (
        *spec.init,
        *spec.goal,
        *spec.invariants,
        *(p for m in spec.milestones for p in m.achieve),
        *(t.predicate for t in spec.temporal),
    ):
        for key in ("subject", "reference"):
            if key in predicate.arguments and predicate.arguments[key] not in known:
                raise ValueError(f"Predicate references undeclared {key}")
    nodes = []
    if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
        raise ValueError("calls must be a list of Semantic Call objects")
    for i, call in enumerate(calls):
        validate_semantic_call_payload(call)
        args = call.get("arguments", call)
        if any(
            args.get(k) and args[k] not in known
            for k in ("object", "handle", "reference")
        ):
            raise ValueError("Call references an undeclared logical role")
        nodes.append(
            dict(
                id=f"call_{i+1}",
                task_type="goal",
                task_instance_id="plan",
                role="primary",
                call=call,
                depends_on=[f"call_{i}"] if i else [],
            )
        )
    success = {"goals": [p.id for p in spec.goal]}
    graph = validate_semantic_task_graph(
        dict(
            schema_version="semantic_task_graph/v1",
            task_id=task_id,
            instruction=instruction,
            planner_route="goal_driven",
            integration_fingerprint="0" * 64,
            targets={},
            nodes=nodes,
            task_groups=[
                dict(
                    id="plan",
                    task_type="goal",
                    node_ids=[n["id"] for n in nodes],
                    depends_on=[],
                    success=success,
                )
            ],
            success=success,
        )
    )
    coverage = _coverage(spec, graph)
    blockers = []
    for node in nodes:
        call = node["call"]
        if (
            call.get("call_id") == "simulation.open_door"
            and call["arguments"]["state"] == "closed"
        ):
            blockers.append(
                {
                    "node_id": node["id"],
                    "code": "closing_direction_unsupported",
                    "owner": "OpenDoor",
                }
            )
    document = freeze_task_spec(
        spec,
        image_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        image_ref="instruction.txt",
        source_kind="text",
        role_descriptions={r.name: r.name for r in spec.roles},
        interpretation=[],
        generalization=[],
    )
    return {
        "task_spec": spec.to_dict(),
        "task_spec_v1": document,
        "semantic_task_graph": graph,
        "coverage": coverage,
        "deployment_preflight": {
            "status": (
                "unsupported"
                if blockers
                else (
                    "uncovered"
                    if coverage["status"] != "covered"
                    else "requires_deployment"
                )
            ),
            "issues": blockers,
            "execution_ready": False,
            "physical_verdict": "not_evaluated",
            "spec_digest": document["spec_digest"],
            "graph_digest": canonical_hash(graph),
        },
    }


def _deployment_check(result: dict[str, Any], path: Path) -> None:
    """Compile the actual witness against a trusted configured deployment."""
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.task_program.integrations.composition import (
        load_configured_task_program_deployment,
    )
    from .program_declarations import build_program_declaration

    checks = result["deployment_preflight"]
    raw = load_config(path)
    base = path.resolve().parent
    physical = resolve_gym_components(raw, base_dir=base)
    kwargs = dict(skill_profile=physical.embodiment_skill_profile, base_dir=base)
    try:
        configured = load_configured_task_program_deployment(
            task_program=raw["task_program"], **kwargs
        )
        physical.validate_scene_bindings(configured.scene_binding)
        program = build_program_declaration(
            result["semantic_task_graph"], configured.program_id
        )
        selection = {**raw["task_program"], "program": {"inline": program}}
        resolved = load_configured_task_program_deployment(
            task_program=selection, **kwargs
        )
        resolved.load_program()
        checks["compilation"] = "passed"
        checks["integration_fingerprint"] = resolved.integration.integration_fingerprint
        if checks["status"] == "requires_deployment":
            checks["status"] = "compiled_pending_physical_qualification"
    except (ValueError, KeyError, TypeError) as error:
        checks.update(compilation="rejected", status="unsupported")
        checks["issues"].append(
            {"code": "deployment_contract_mismatch", "detail": str(error)}
        )


def publish(
    instruction: str,
    task_id: str,
    output: Path,
    proposal: dict[str, Any] | None,
    services: GenSimServices | None,
    deployment: Path | None = None,
    image: Path | None = None,
) -> dict[str, Any]:
    """Author from a supplied agent proposal or configured model services."""
    from embodichain.gen_sim.services.compat import structured_call
    from embodichain.gen_sim.task_spec import KNOWN_PREDICATES

    if image is not None:
        image_bytes = image.read_bytes()

    def ask(role, payload):
        if image is not None and role == "task_interpretation":
            from embodichain.gen_sim.services.runtime import service_scope
            from embodichain.gen_sim.services.compat import VisionFacade
            from .image_planning import _json_response

            with service_scope(services) as run:
                return _json_response(
                    VisionFacade(run, "task_image_grounding").complete(
                        image_path=image,
                        user_prompt=json.dumps(payload, ensure_ascii=False),
                        system_prompt="Interpret the instruction with the supplied scene image. Return only the requested JSON objective. Preserve requested properties, identity, relations and ordering. Never infer measured success.",
                    )
                )
        return structured_call(
            role=role,
            prompt=json.dumps(payload, ensure_ascii=False),
            schema={"type": "object"},
            services=services,
        )

    intent_request = {
        "instruction": instruction,
        "predicates": sorted(KNOWN_PREDICATES),
        "request": "Return exactly {roles:[{id,kind}],goal:[predicate],milestones:[{id,after:[milestone IDs],achieve:[predicate]}]}. Each predicate is {id,predicate,subject,arguments:{...}} with optional reference and relation. Use short stable role IDs in subject/reference. Kinds: object, target, surface, container, articulation, button. Default expected true. For hinge endpoints use articulation_joint_near with arguments {target_state:open|closed,joint_kind:revolute}. For placement use relation with subject,reference,relation. For handover_complete use arguments transfer_arm and receive_arm (left or right). State all explicit original obligations even if unachievable. No E-task codes or generated provenance. Empty milestones for a simple single endpoint. Separate final state from ordered history.",
    }
    intent_request["source_contract"] = (
        "Also return sources: {goal_or_milestone_id: exact verbatim substring of instruction}. Cover EVERY goal and milestone. Never add orientation, stability, pose or gripper requirements just because visible in the image. Treat handover as a historical milestone; final goals describe surviving states. For 同样方式 cite the phrase together with its antecedent as one contiguous quote. Preserve all user-requested operations. Do not model arms as objects. Use on or inside, not in. For explicit arm requirements use required_arm:left/right; transfer_arm and receive_arm use left/right. Use one milestone per explicit ordered operation."
    )
    for attempt in range(2):
        try:
            raw = dict(
                proposal["objective"]
                if proposal is not None
                else ask("task_interpretation", intent_request)
            )
            sources = raw.pop("sources", None)
            objective = raw
            frozen = validate_task_spec(
                {**objective, "instruction": instruction, "task_id": task_id}
            )
            clauses = _source_clauses(instruction, frozen, sources)
            break
        except (ValueError, TypeError, KeyError) as error:
            if proposal is not None or attempt:
                raise
            intent_request["repair_error"] = str(error)
    request = {
        "frozen_spec": frozen,
        "catalog": semantic_planning_catalog(),
        "request": "Return only {calls:[...]} in execution order. Each builtin is {kind:pick,object:ROLE,resources:{}} or {kind:place,object:ROLE,on:ROLE,resources:{}}; hand_over uses resources {source:right,destination:left}. EVERY simulation.* call has EXACT shape {kind:registered,call_id:simulation.CALL,arguments:{...},resources:{}}. Example for close: {kind:registered,call_id:simulation.open_door,arguments:{object:DOOR_ROLE,state:closed},resources:{}}. Replace symbolic ROLE with exact frozen IDs. No extra wrappers or node IDs.",
    }
    request["sources"] = sources
    request["rules"] = (
        "Pick with the same primary arm that will give. hand_over.source must hold the object and destination be free. Place using the receiving arm. Use canonical left/right consistently. Keep acquired objects held until release. Never change the frozen specification. If a goal lacks a catalog effect, retain the gap."
    )
    attempts = []
    result = None
    best = None
    for index in range(1 if proposal is not None else 3):
        calls = None
        try:
            calls = (
                proposal["calls"]
                if proposal is not None
                else ask("task_semantic_planning", request)["calls"]
            )
            candidate = author_goal_plan(instruction, task_id, objective, calls)
            coverage = candidate["coverage"]
            feedback = {
                "uncovered": coverage["uncovered"],
                "resource_conflicts": coverage["resource_diagnostics"],
                "clauses": {
                    k: sources[k] for k in coverage["uncovered"] if k in sources
                },
            }
            score = (
                len(coverage["uncovered"]) + len(coverage["resource_conflicts"]),
                len(coverage["uncovered"]),
            )
            if best is None or score < best:
                result, best, selected = candidate, score, index
        except (ValueError, TypeError, KeyError) as error:
            coverage = None
            feedback = {"validation_error": str(error)}
        attempts.append(
            dict(round=index, calls=calls, coverage=coverage, feedback=feedback)
        )
        if coverage is not None and coverage["status"] == "covered":
            break
        request.update(
            previous_calls=calls,
            repair_feedback=feedback,
            revision_instruction="Return the complete revised calls array. Preserve working producers and their order. "
            "For missing holdings insert an acquisition on the correct arm before transfer, not another transfer. "
            "Repair arguments or order only where feedback requires. Do not change goals or add operations absent from the instruction.",
        )
    if result is None:
        with WorkflowRunArtifacts(output) as artifacts:
            write_run_json(
                artifacts.output_dir / "planning_history.json",
                {
                    "frozen_digest": canonical_hash(frozen),
                    "selected_round": None,
                    "attempts": attempts,
                },
            )
            write_run_json(
                artifacts.output_dir / "frozen_objective.json",
                {"spec": frozen, "clauses": clauses},
            )
            write_run_json(
                artifacts.output_dir / "run_manifest.json",
                {
                    "task_id": task_id,
                    "status": "invalid_graph",
                    "execution_ready": False,
                },
            )
            artifacts.commit()
        raise ValueError(
            "No valid graph after bounded planning: "
            + json.dumps(feedback, ensure_ascii=False)
        )
    result["planning_history"] = {
        "frozen_digest": canonical_hash(frozen),
        "source_digest": canonical_hash(sources),
        "selected_round": selected,
        "attempts": attempts,
    }
    spec = TaskSpec.from_dict(frozen)
    source_bytes = image_bytes if image is not None else instruction.encode()
    image_ref = (
        ("source_image" + image.suffix.lower())
        if image is not None
        else "instruction.txt"
    )
    if image is not None:
        if image.read_bytes() != image_bytes:
            raise ValueError("Input image changed during authoring")
    result["task_spec_v1"] = freeze_task_spec(
        spec,
        image_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_kind="image" if image is not None else "text",
        clauses=clauses,
        image_ref=image_ref,
        role_descriptions={r.name: r.name for r in spec.roles},
        interpretation=[],
        generalization=[],
    )
    if deployment is not None:
        _deployment_check(result, deployment)
    result["deployment_preflight"]["spec_digest"] = result["task_spec_v1"][
        "spec_digest"
    ]
    with WorkflowRunArtifacts(output) as artifacts:
        if image is not None:
            (artifacts.output_dir / image_ref).write_bytes(image_bytes)
        (artifacts.output_dir / "instruction.txt").write_text(
            instruction, encoding="utf-8"
        )
        for key, value in result.items():
            write_run_json(artifacts.output_dir / (key + ".json"), value)
        write_run_json(
            artifacts.output_dir / "run_manifest.json",
            dict(
                schema_version="goal_authoring/v1",
                task_id=task_id,
                spec_digest=result["task_spec_v1"]["spec_digest"],
                graph_digest=canonical_hash(result["semantic_task_graph"]),
                coverage=result["coverage"]["status"],
                execution_ready=False,
                physical_verdict="not_evaluated",
            ),
        )
        artifacts.commit()
    return result


def _source_clauses(instruction, frozen, sources):
    ids = [p["id"] for p in frozen["goal"]] + [m["id"] for m in frozen["milestones"]]
    if (
        len(ids) != len(set(ids))
        or not isinstance(sources, dict)
        or set(sources) != set(ids)
    ):
        raise ValueError(
            "sources must identify every goal and milestone exactly once: " + str(ids)
        )
    clauses = []
    for key in ids:
        quote = sources[key]
        if not isinstance(quote, str) or not quote.strip() or quote not in instruction:
            raise ValueError(f"{key}: expected verbatim instruction quote")
        clauses.append(
            dict(
                id="source." + key,
                target=key,
                quote=quote,
                origin="explicit",
                resolution="resolved",
            )
        )
    return clauses
