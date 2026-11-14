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
"""Hybrid authoring pilot for handovers, relative placement and coordinated release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .goal_planning import author_goal_plan, _deployment_check
from .ontology import task_contract
from .semantic_graph import validate_semantic_task_graph
from .orchestration.workflow_artifacts import WorkflowRunArtifacts, write_run_json
from ..task_spec import TaskSpec, canonical_hash
from ..task_spec.publication import freeze_task_spec

__all__: list[str] = []


def _operation_rows(response):
    """Accept a JSON array or its named envelope, never infer missing operations."""
    rows = response.get("operations") if isinstance(response, dict) else response
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty operations array")
    for index, op in enumerate(rows):
        if not isinstance(op, dict) or not all(
            isinstance(op.get(field), str) and op[field].strip()
            for field in ("template", "object", "quote")
        ):
            raise ValueError(
                f"operations[{index}] requires template, object and quote strings"
            )
        for key in ("continue_held", "restore_required"):
            if key in op and type(op[key]) is not bool:
                raise ValueError(f"operations[{index}].{key} requires a boolean")
    return rows


def _check_source_intent(instruction, operations):
    """Reject conflicting or unspecified transfer semantics before freezing goals."""
    source = instruction.casefold()
    pours = [op for op in operations if op["template"] == "E3"]
    motion = r"倾倒动作|倾倒运动|倾倒姿势|倒水姿势|模拟倾倒|pouring motion|pouring gesture|mimic pouring"
    no_fluid = r"(?:不(?:需要|要求|用)?|不要)(?:实际|真的)?(?:倒出|倒入|转移)(?:液体|水)|without (?:transferring|pouring) (?:water|liquid)"
    fluid = r"(?:水|液体|牛奶|饮料)(?:倒入|倒进|注入|进入|转移到)|(?:倒入|倒进|倒出)(?:液体|水)|(?:pour|transfer) (?:the )?(?:liquid|water|milk)"
    if re.search(r"倾倒|倒水|倒入|倒进|pour", source) and not pours:
        raise ValueError("source_omission: missing pour operation")
    for op in pours:
        text = source if len(pours) == 1 else op["quote"].casefold()
        negative = re.search(no_fluid, text)
        movement = bool(negative or re.search(motion, text))
        transfer = bool(re.search(fluid, re.sub(no_fluid, "", text)))
        if not movement and not transfer:
            raise ValueError("ambiguous_pour: clarify motion versus liquid transfer")
        mode = "fluid" if transfer else "motion"
        if op["mode"] != mode:
            raise ValueError(f"pour_mode_mismatch: instruction requires {mode}")
        if (
            re.search(r"放回原位|放回原处|return .*original", text)
            and op.get("restore_required") is not True
        ):
            raise ValueError("source_omission: pour requires restoration and release")
    for pattern, kind, value in [
        (r"扶正|stand .*upright", "E2", "upright"),
        (r"长轴竖直|正倒均可|either end may face up", "E2", "vertical"),
        (
            r"(?:放入|移入|收进|放进|收回)[^，。；;]{0,20}?(?:托盘|抽屉)|place .* inside",
            "E1",
            "inside",
        ),
        (
            r"(?:放到|放在)[^，。；;]{1,16}(?:右侧|右边)(?=[，。；;（(]|$)",
            "E1",
            "right_of",
        ),
    ]:
        for match in re.finditer(pattern, source):
            if not any(
                op["template"] == kind
                and (
                    match.group() in op["quote"].casefold()
                    or op["quote"].casefold() in match.group()
                )
                and op.get(
                    "orientation" if kind == "E2" else "relation",
                    "upright" if kind == "E2" else None,
                )
                == value
                for op in operations
            ):
                raise ValueError(
                    f"source_omission: {match.group()} requires {kind}/{value}"
                )
    if any(op["template"] == "E5" for op in operations) and not re.search(
        r"双臂|双手|两臂|both arms|both hands", source
    ):
        raise ValueError("unsupported_addition: coordinated transport not requested")


def _supported_program(operations, instruction, *, bindings=None):
    roles, goals, milestones, clauses, calls, requirements = {}, {}, [], [], [], set()
    last = None
    holder = {}
    if bindings is None:
        bindings = {}

    def registered(cid, args, side):
        calls.append(
            dict(
                kind="registered",
                call_id="simulation." + cid,
                arguments=args,
                resources={} if side == "auto" else {"primary": side},
            )
        )

    def bind(kind, **values):
        bindings[f"call_{len(calls)}"] = dict(
            kind=kind, call=canonical_hash(calls[-1]), **values
        )

    def role(name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Empty role")
        roles.setdefault(name, dict(id=name, kind="object"))

    def milestone(predicate, quote):
        nonlocal last
        mid = f"m{len(milestones)+1}"
        milestones.append(
            dict(
                id=mid,
                after=[last] if last else [],
                achieve=[dict(predicate, id=mid + ".achieve")],
            )
        )
        clauses.append((mid, quote))
        last = mid

    for index, op in enumerate(operations):
        template = op["template"]
        contract = task_contract(template)
        obj, quote = op["object"], op["quote"]
        if not quote or quote not in instruction:
            raise ValueError("Missing verbatim operation source")
        role(obj)
        side = op.get("primary", "auto")
        if side not in {"auto", "left", "right"}:
            raise ValueError("Invalid primary arm")
        if side == "auto" and obj in holder:
            side = holder[obj]
        resource = {} if side == "auto" else {"primary": side}
        if side != "auto":
            requirements.add(side)
        if template in {"E2", "E3", "E6", "E7", "E8", "E9"}:
            reuse = template == "E3" and obj in holder and holder[obj] == side
            if (obj in holder or side in holder.values()) and not reuse:
                raise ValueError(
                    f"Control arm occupied: {side}, held={holder}, requested={obj}"
                )
            if template == "E2":
                registered("axis_align", {"object": obj}, side)
                orientation = op.get("orientation", "upright")
                if orientation not in {"upright", "vertical"}:
                    raise ValueError("Invalid axis intent")
                pred = dict(
                    predicate=(
                        "object_upright" if orientation == "upright" else "orientation"
                    ),
                    subject=obj,
                    arguments={"orientation_goal": orientation},
                )
                bind("orientation", orientation=orientation)
                # Release onto the original support unless the next operation places this object.
                holder[obj] = side
            elif template == "E3":
                ref = op["reference"]
                role(ref)
                if obj == ref:
                    raise ValueError("Pour needs a distinct receiver")
                mode = op["mode"]
                if mode not in {"motion", "fluid"}:
                    raise ValueError("Pour mode is motion or fluid")
                if not reuse:
                    calls.append(dict(kind="pick", object=obj, resources=resource))
                holder.pop(obj, None)
                registered("move_held_object", {"object": obj, "target": ref}, side)
                registered("pour", {"object": obj}, side)
                bind("pour_motion", reference=ref)
                pred = dict(
                    predicate="pour_motion" if mode == "motion" else "poured",
                    subject=obj,
                    reference=ref,
                )
            elif template == "E6":
                state = op["state"]
                if state not in {"open", "closed"}:
                    raise ValueError("Invalid slide endpoint")
                roles[obj]["kind"] = "articulation"
                registered("articulation_link_slide", {"handle": obj}, side)
                bind("joint_endpoint", state=state, joint_kind="prismatic")
                pred = dict(
                    predicate="articulation_joint_near",
                    subject=obj,
                    arguments=dict(target_state=state, joint_kind="prismatic"),
                )
            elif template == "E7":
                if op["state"] != "open":
                    raise ValueError("E7 only supports opening a hinged door")
                roles[obj]["kind"] = "articulation"
                registered("open_door", {"object": obj, "state": "open"}, side)
                pred = dict(
                    predicate="articulation_joint_near",
                    subject=obj,
                    arguments=dict(target_state="open", joint_kind="revolute"),
                )
            elif template == "E8":
                mode = op["mode"]
                if mode not in {"coarse", "setting"}:
                    raise ValueError("Invalid twist mode")
                args = {"mode": mode}
                if mode == "setting":
                    if type(op.get("setting")) is not int:
                        raise ValueError("Missing calibrated setting")
                    args["setting"] = op["setting"]
                registered("twist", {"object": obj}, side)
                bind("twist", **args)
                pred = dict(
                    predicate="semantic_goal",
                    subject=obj,
                    arguments={"operation": "twist", **args},
                )
            else:
                registered("press", {"object": obj}, side)
                pred = dict(predicate="pressed", subject=obj)
            milestone(pred, quote)
            goals[(obj, template)] = (dict(pred, id=f"final_{obj}_{template}"), quote)
            if template == "E3":
                origin = "original_support_" + obj
                registered("move_held_object", {"object": obj, "target": origin}, side)
                calls.append(
                    dict(kind="place", object=obj, on=origin, resources=resource)
                )
                bind("return_to_origin", subject=obj)
                if op.get("restore_required"):
                    restored = dict(
                        predicate="semantic_goal",
                        subject=obj,
                        arguments={"operation": "return_and_release"},
                    )
                    milestone(restored, quote)
            following = operations[index + 1] if index + 1 < len(operations) else {}
            if template == "E2" and not (
                op.get("continue_held", False)
                and following.get("object") == obj
                and following.get("template") in {"E1", "E3"}
            ):
                calls.append(
                    dict(
                        kind="place",
                        object=obj,
                        on="original_support_" + obj,
                        resources=resource,
                    )
                )
                holder.pop(obj)
        elif template == "E4":
            src, dst = op["source"], op["destination"]
            if (
                {src, dst} != {"left", "right"}
                or obj in holder
                or src in holder.values()
                or dst in holder.values()
            ):
                raise ValueError("Invalid transfer resources")
            requirements.update((src, dst))
            calls.extend(
                [
                    dict(kind="pick", object=obj, resources={"primary": src}),
                    dict(
                        kind="hand_over",
                        object=obj,
                        resources={"source": src, "destination": dst},
                    ),
                ]
            )
            holder[obj] = dst
            milestone(
                dict(
                    predicate="handover_complete",
                    subject=obj,
                    arguments=dict(transfer_arm=src, receive_arm=dst),
                ),
                quote,
            )
        elif template in {"E1", "E5"}:
            ref, rel = op["reference"], op["relation"]
            role(ref)
            if obj == ref or rel not in {
                "on",
                "inside",
                "right_of",
                "left_of",
                "behind",
                "front_of",
                "above",
            }:
                raise ValueError("Invalid support endpoints")
            args = {}
            if template == "E1":
                if holder.get(obj, side) != side:
                    raise ValueError("Wrong placing arm")
                if obj not in holder:
                    if side in holder.values():
                        raise ValueError("Acquisition arm occupied")
                    calls.append(dict(kind="pick", object=obj, resources=resource))
                holder.pop(obj, None)
                if side != "auto":
                    args["required_arm"] = side
                if rel not in {"on", "inside"}:
                    registered(
                        "place_relative",
                        dict(object=obj, reference=ref, relation=rel),
                        side,
                    )
                else:
                    calls.append(
                        {
                            "kind": "place",
                            "object": obj,
                            rel: ref,
                            "resources": resource,
                        }
                    )
            else:
                if holder or rel != "on":
                    raise ValueError(
                        "Coordinated release requires free arms and on destination"
                    )
                requirements.update(("left", "right"))
                milestone(dict(predicate="held_by_both_grippers", subject=obj), quote)
                calls.append(
                    dict(
                        kind="registered",
                        call_id="simulation.coordinated_transport",
                        arguments={"object": obj, "target": ref},
                        resources={"left": "left", "right": "right"},
                    )
                )
            pred = dict(
                predicate="relation",
                subject=obj,
                reference=ref,
                relation=rel,
                arguments=args,
            )
            milestone(pred, quote)
            goals[obj] = (dict(pred, id=f"final_{obj}"), quote)
        else:
            raise ValueError(f"Unsupported hybrid template: {template}")
        roles[obj]["affordances"] = sorted(
            set(roles[obj].get("affordances", ())) | contract.required_affordances
        )
    if not goals or holder:
        raise ValueError("Pilot requires an explicit placement terminal")
    clauses.extend((p["id"], q) for p, q in goals.values())
    objective = dict(
        roles=list(roles.values()),
        milestones=milestones,
        goal=[p for p, _ in goals.values()],
        requirements=[
            dict(consumer="embodiment", key="required_arms", value=sorted(requirements))
        ],
    )
    return objective, clauses, calls


def verify_witness(spec, graph, *, support_targets, bindings=None):
    """Check ordered occurrence effects derived from canonical calls against frozen obligations."""
    from .goal_planning import _facts, _coverage
    from embodichain.lab.task_program.language.semantic_contracts import (
        validate_semantic_call_payload,
    )

    graph = validate_semantic_task_graph(graph)
    facts = []
    roles = {r.name for r in spec.roles}
    bindings = bindings or {}
    for i, n in enumerate(graph["nodes"]):
        call = n["call"]
        validate_semantic_call_payload(call)
        params = call.get("arguments", call)
        if any(
            params.get(k) and params[k] not in roles
            for k in ("object", "handle", "reference")
        ):
            raise ValueError("Undeclared witness role")
        if n["depends_on"] != ([graph["nodes"][i - 1]["id"]] if i else []):
            raise ValueError("Pilot witness requires a dependency chain")
        # This route binds the target symbol to the required support role. Native
        # geometry, contact, release and interval evidence are still pending.
        if call.get("call_id") == "simulation.coordinated_transport":
            args = call["arguments"]
            if (
                args["target"] not in support_targets
                or support_targets[args["target"]] not in roles
                or call["resources"] != {"left": "left", "right": "right"}
            ):
                raise ValueError("Invalid coordinated resource or target binding")
            facts.extend(
                [
                    (
                        n["id"],
                        "acquire",
                        "held_by_both_grippers",
                        {"subject": args["object"]},
                    ),
                    (
                        n["id"],
                        "release",
                        "relation",
                        dict(
                            subject=args["object"],
                            reference=support_targets[args["target"]],
                            relation="on",
                        ),
                    ),
                ]
            )
        else:
            facts.extend((n["id"], "terminal", k, a) for k, a in _facts(call))
            cid = call.get("call_id", call["kind"])
            entity = params.get("object", params.get("handle"))

            def fact(kind, **args):
                facts.append((n["id"], "terminal", kind, dict(subject=entity, **args)))

            if cid == "simulation.press":
                fact("pressed")
            b = bindings.get(n["id"], {})
            if b and canonical_hash(call) == b["call"]:
                if cid == "simulation.axis_align" and b["kind"] == "orientation":
                    fact(
                        (
                            "object_upright"
                            if b["orientation"] == "upright"
                            else "orientation"
                        ),
                        orientation_goal=b["orientation"],
                    )
                elif cid == "simulation.pour" and b["kind"] == "pour_motion":
                    fact("pour_motion", reference=b["reference"])
                elif (
                    cid == "simulation.articulation_link_slide"
                    and b["kind"] == "joint_endpoint"
                ):
                    fact(
                        "articulation_joint_near",
                        target_state=b["state"],
                        joint_kind=b["joint_kind"],
                    )
                elif cid == "simulation.twist" and b["kind"] == "twist":
                    fact(
                        "semantic_goal",
                        operation="twist",
                        **{k: b[k] for k in ("mode", "setting") if k in b},
                    )
                elif cid == "place" and b["kind"] == "return_to_origin":
                    fact("semantic_goal", operation="return_and_release")

    def matches(p, f):
        return (
            p.expected is True
            and p.operator == "eq"
            and p.name == f[2]
            and all(k in f[3] and f[3][k] == v for k, v in p.arguments.items())
        )

    achieved = {}
    links = {}
    missing = []
    remaining = list(spec.milestones)
    while remaining:
        m = next(m for m in remaining if all(k in achieved for k in m.after))
        start = max((achieved[k] + 1 for k in m.after), default=0)
        idx = next(
            (
                i
                for i in range(start, len(facts))
                if all(matches(p, facts[i]) for p in m.achieve)
            ),
            len(facts),
        )
        achieved[m.id] = idx
        if idx < len(facts):
            links[m.id] = facts[idx][:2]
        else:
            missing.append(m.id)
        remaining.remove(m)
    for p in spec.goal:
        subject = p.arguments["subject"]
        last = next(
            (
                n
                for n in reversed(graph["nodes"])
                if n["call"].get("arguments", n["call"]).get("object") == subject
            ),
            None,
        )
        if (
            p.name
            in {
                "pressed",
                "pour_motion",
                "poured",
                "object_upright",
                "orientation",
                "articulation_joint_near",
            }
            or p.arguments.get("operation") == "twist"
        ):
            candidates = [
                f for f in facts if f[2] == p.name and f[3].get("subject") == subject
            ]
            last = {"id": candidates[-1][0]} if candidates else None
        hit = next(
            (
                f
                for f in reversed(facts)
                if last and f[0] == last["id"] and matches(p, f)
            ),
            None,
        )
        if hit:
            links[p.id] = hit[:2]
        else:
            missing.append(p.id)
    conflicts = _coverage(spec, graph)["resource_diagnostics"]
    missing.extend(k for k in ("temporal", "invariants") if getattr(spec, k))
    return dict(
        status="covered" if not missing and not conflicts else "uncovered",
        producers=links,
        uncovered=missing,
        resource_conflicts=conflicts,
        physical_verdict="not_evaluated",
    )


def _scene_projection(doc, graph, packet):
    """Project the rigid objects and destinations of a checked manipulation sequence."""
    from .image_planning import _validate_grounding
    from ..task_spec.publication import project_scene_requirements

    goals = doc["objectives"]["goal"]
    nodes = graph["nodes"]
    destinations = {g["subject"]: g for g in goals if g["predicate"] == "relation"}
    annotations = {}
    if not nodes or len(destinations) != len(goals):
        raise ValueError("Grounded manipulation requires terminal placement relations")
    for n in nodes:
        c = n["call"]
        carry = c.get("call_id") == "simulation.coordinated_transport"
        if not carry and c["kind"] not in {"pick", "hand_over", "place"}:
            raise ValueError("Unsupported grounded manipulation call")
        subject = c.get("arguments", c)["object"]
        goal = destinations.get(subject)
        if not goal or goal["relation"] not in {"on", "inside"}:
            raise ValueError("Missing frozen destination for " + subject)
        relation, reference = goal["relation"], goal["reference"]
        if (carry and (relation != "on" or c["arguments"]["target"] != reference)) or (
            c["kind"] == "place" and c.get(relation) != reference
        ):
            raise ValueError("Placement witness does not match the frozen objective")
        annotations[n["id"]] = dict(
            object=subject, reference=reference, relation=relation
        )
    if any(
        c.get("constraint") or c["resolution"] != "resolved"
        for c in doc["interpretation"]["clauses"]
    ):
        raise ValueError(
            "Hybrid scene qualification cannot discharge additional qualifiers"
        )
    roles = [r["id"] for r in doc["configuration"]["roles"]]
    if packet["image_sha256"] != doc["inputs"]["image"]["sha256"]:
        raise ValueError("Grounding belongs to a different source image")
    grounding = _validate_grounding(
        json.dumps(packet["grounding"]),
        [dict(reference_id=r, reference=r, quantifier="one", count=0) for r in roles],
    )
    if set(grounding) != {"objects", "bindings"} or any(
        grounding["bindings"][r] != [r] for r in roles
    ):
        raise ValueError(
            "Grounding must preserve hybrid role IDs and frozen declarations"
        )
    return {
        "scene_requirements": project_scene_requirements(doc),
        "image_grounding": grounding,
        "plan_coverage": {"role_entities": grounding["bindings"]},
        "semantic_graph_validation": {"source": {"annotations": annotations}},
    }


def publish_hybrid(
    instruction: str,
    task_id: str,
    image: Path,
    output: Path,
    services: Any = None,
    deployment: Path | None = None,
    *,
    scene_grounding: Path | None = None,
) -> dict[str, Any]:
    from ..services.runtime import service_scope
    from ..services.compat import VisionFacade, structured_call
    from .image_planning import _json_response

    image_bytes = image.read_bytes()
    prompt = {
        "instruction": instruction,
        "request": "Return JSON {operations:[...]}. Choose only requested operations from the nine types. Copy task actions exactly in order. Do NOT invent handovers, named arms or extra pick steps. E1 handles both acquisition and placement. E4 ONLY for an explicit handover between two named arms: {template:E4,object,source:left|right,destination:left|right,quote}. E1 single arm placing: {template:E1,object,primary:auto|left|right,reference,relation:on|inside|right_of,quote}. E5 coordinated transport and release: {template:E5,object,reference,relation:on,quote}. Use consistent short English role IDs, retaining the same role when its contents change. E4 must be followed by E1 on the same object using the receiver. Use auto for unspecified arms. Distinguish surface support from volume containment. Every quote must be a verbatim instruction substring.",
    }
    prompt["request"] += (
        " Additionally support E2 align {object,orientation:upright|vertical,continue_held:true only when next E1 places same object}; "
        "E3 pour {object,reference,mode:motion|fluid,restore_required:boolean}; E6 prismatic slide {object:part,state:open|closed}; "
        "E7 hinged door {object:part,state:open}; E8 twist {object,mode:coarse|setting,setting:integer only for setting}; E9 press {object}. "
        "Every operation requires template and quote. For unspecified primary arm use auto, including E1. Relative E1 can use right_of/left_of. "
        "For E5 first explicitly move all payloads. Repeat operations for each named subject. Preserve operation order, qualifiers and original-pose restoration."
    )
    prompt["request"] += (
        " Return only the operations asked for; never demonstrate the catalog. Each affirmative action and each destination needs a matching occurrence. "
        "A pouring motion or gesture means mode=motion. Pouring actual liquid into a receiver means mode=fluid. Never infer liquid transfer from vessel appearance. "
        "E3 owns tilt, restore upright, original-pose return and release; set restore_required=true if requested. Do not add an E2 or E1 for its cleanup. "
        "Use E2 before E1 for upright then place with continue_held=true. No E5 unless explicitly coordinated. Retain on vs inside vs right_of. Use complete verbatim clauses for quote."
    )
    with service_scope(services) as run:
        audit_history = []
        used = 0
        for attempt in range(2):
            try:
                used += 1
                operations = _operation_rows(
                    _json_response(
                        VisionFacade(run, "task_image_grounding").complete(
                            image_path=image,
                            system_prompt="Interpret task goals and resources from the complete instruction and image.",
                            user_prompt=json.dumps(prompt, ensure_ascii=False),
                        )
                    )
                )
                _check_source_intent(instruction, operations)
                bindings = {}
                obj, clauses, calls = _supported_program(
                    operations, instruction, bindings=bindings
                )
                used += 1
                audit = structured_call(
                    services=run,
                    role="task_interpretation",
                    schema={"type": "object"},
                    prompt=json.dumps(
                        {
                            "instruction": instruction,
                            "operations": operations,
                            "ordered_calls": calls,
                            "bound_effects": bindings,
                            "request": "Independently compare the original instruction with these declarative operations in order. Check each source action, subject, recipient, relation, prescribed arm, restoration and orientation, including negative qualifiers. Audit the expanded calls and their bound effects as well as the declarations. One occurrence satisfies once; 分别 means each named subject needs its own occurrence, not separate coordinates unless specified. An object already placed inside a tray remains there unless moved. Do not demand a second operation for an acquisition or restoration already in the expansion. Ignore incidental object appearance unless used to identify an entity. E3 motion+restore_required already includes upright return and release to origin. Reject added acts and ungrounded fluid transfer. If a qualifier cannot be represented, flag it. Return exactly {issues:[{quote:verbatim source clause,reason:specific mismatch}]} or {issues:[]}.",
                        },
                        ensure_ascii=False,
                    ),
                )
                if set(audit) != {"issues"} or not isinstance(audit["issues"], list):
                    raise ValueError("Malformed source audit")
                if any(
                    not isinstance(x, dict)
                    or not x.get("quote")
                    or x["quote"] not in instruction
                    or not x.get("reason")
                    for x in audit["issues"]
                ):
                    raise ValueError("Malformed source audit diagnostic")
                audit_history.append(audit)
                if audit["issues"]:
                    raise ValueError(
                        "source_audit: "
                        + json.dumps(audit["issues"], ensure_ascii=False)
                    )
                break
            except (KeyError, ValueError, TypeError) as error:
                if attempt:
                    raise
                prompt["diagnostic"] = str(error)
                prompt["previous_operations"] = locals().get("operations")
        history = []
        support_targets = {
            op["reference"]: op["reference"]
            for op in operations
            if op["template"] == "E5"
        }
        for attempt in range(3):
            try:
                candidate = author_goal_plan(instruction, task_id, obj, calls)
                spec = TaskSpec.from_dict(candidate["task_spec"])
                candidate_graph = candidate["semantic_task_graph"]
                candidate_graph["planner_route"] = "hybrid"
                feedback = verify_witness(
                    spec,
                    candidate_graph,
                    support_targets=support_targets,
                    bindings=bindings,
                )
                score = len(feedback["uncovered"]) + len(feedback["resource_conflicts"])
                if not history or score < best:
                    best, result, graph, coverage = (
                        score,
                        candidate,
                        candidate_graph,
                        feedback,
                    )
            except (ValueError, TypeError, KeyError) as exc:
                if not history:
                    raise
                feedback = {"status": "invalid", "error": str(exc)}
            history.append(
                dict(calls=calls, coverage=feedback, spec_hash=spec.semantic_hash)
            )
            if feedback["status"] == "covered" or attempt == 2 or used >= 5:
                break
            used += 1
            calls = structured_call(
                services=run,
                role="task_semantic_planning",
                schema={"type": "object"},
                prompt=json.dumps(
                    dict(
                        request="Return corrected canonical calls only as {calls:[...]}. Preserve valid occurrences and their bindings. Registered calls require kind:registered,call_id,arguments,resources. Never remove requested milestones.",
                        frozen=spec.to_dict(),
                        previous=calls,
                        feedback=feedback,
                    ),
                    ensure_ascii=False,
                ),
            )["calls"]
    if image.read_bytes() != image_bytes:
        raise ValueError("Source image changed during authoring")
    doc = freeze_task_spec(
        spec,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        image_ref="source_image" + image.suffix.lower(),
        role_descriptions={r.name: r.name for r in spec.roles},
        interpretation=[],
        generalization=[],
        clauses=[
            dict(
                id="source." + k,
                target=k,
                quote=q,
                origin="explicit",
                resolution="resolved",
            )
            for k, q in clauses
        ],
    )
    result.update(
        task_spec_v1=doc,
        coverage=coverage,
        planning_history=history,
        template_sources=operations,
        pending_support_targets=support_targets,
        pending_operation_bindings=bindings,
        source_audits=audit_history,
    )
    result["deployment_preflight"].update(
        graph_digest=canonical_hash(graph),
        spec_digest=doc["spec_digest"],
        status=(
            "requires_deployment" if coverage["status"] == "covered" else "uncovered"
        ),
    )
    if deployment:
        _deployment_check(result, deployment)
    if scene_grounding is not None:
        if coverage["status"] != "covered":
            raise ValueError("Only a covered witness can enter scene qualification")
        packet = json.loads(scene_grounding.read_text())
        result.update(_scene_projection(doc, graph, packet), scene_grounding=packet)
    with WorkflowRunArtifacts(output) as artifacts:
        (output / doc["inputs"]["image"]["ref"]).write_bytes(image_bytes)
        for k, v in result.items():
            write_run_json(output / (k + ".json"), v)
        write_run_json(
            output / "run_manifest.json",
            dict(
                schema_version="goal_authoring/v1",
                route="hybrid",
                spec_digest=doc["spec_digest"],
                graph_digest=canonical_hash(graph),
                execution_ready=False,
            ),
        )
        if scene_grounding is not None:
            write_run_json(
                output / "task_plan_manifest.json",
                dict(
                    schema_version="embodichain.hybrid-task-plan/v1",
                    task_id=task_id,
                    instruction=instruction,
                    image_sha256=doc["inputs"]["image"]["sha256"],
                    spec_digest=doc["spec_digest"],
                    semantic_graph_sha256=canonical_hash(graph),
                    artifacts={
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in output.iterdir()
                        if p.is_file()
                    },
                ),
            )
        artifacts.commit()
    return result
