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

"""Task-conditioned scene-UID grounding for structured instruction intents."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
from time import perf_counter
from typing import Any

from .scene_inventory import SceneInventory

__all__ = [
    "GroundingCaller",
    "GroundingResult",
    "ground_articulation_parts",
    "ground_scene_references",
]

GroundingCaller = Callable[..., Mapping[str, Any]]

_BINDING_KEYS = frozenset({"reference_id", "status", "uids", "confidence"})
_QUANTIFIERS = frozenset({"one", "all", "count"})
_REDACTED_KEYS = frozenset(
    {
        "absolute_position",
        "bbox",
        "bboxes",
        "bounding_box",
        "camera_matrix",
        "center",
        "centroid",
        "coordinates",
        "depth",
        "dimensions",
        "extrinsics",
        "grasp_pose",
        "init_local_pose",
        "init_pos",
        "init_rot",
        "intrinsics",
        "joint_positions",
        "joints",
        "keypoint",
        "keypoints",
        "location",
        "matrix",
        "pose",
        "position",
        "position_xyz",
        "qpos",
        "quaternion",
        "rotation",
        "scale",
        "target_pose",
        "trajectory",
        "transform",
        "translation",
        "waypoints",
        "world_x",
        "world_y",
        "world_z",
        "x",
        "y",
        "z",
    }
)

_GROUNDING_SCHEMA: dict[str, Any] = {
    "title": "TaskEngineSceneGrounding",
    "type": "object",
    "additionalProperties": False,
    "required": ["bindings"],
    "properties": {
        "bindings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_BINDING_KEYS),
                "properties": {
                    "reference_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["resolved", "ambiguous", "not_found"],
                    },
                    "uids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True)
class GroundingResult:
    """Validated scene bindings and aggregate call statistics.

    Attributes:
        bindings: Mapping from ``<step-id>.<object|target>`` to scene UIDs.
        attempts: Number of grounding-model calls, including one repair call.
        latency_seconds: Total elapsed wall-clock time across the grounding stage.
    """

    bindings: dict[str, tuple[str, ...]]
    attempts: int
    latency_seconds: float


class _UnresolvedSceneReference(ValueError):
    """Insufficient scene evidence cannot be repaired as malformed JSON."""


def ground_articulation_parts(
    instruction: str,
    intent: Mapping[str, Any],
    reference_bindings: Mapping[str, Sequence[str]],
    part_catalogs: Mapping[str, Sequence[Mapping[str, Any]]],
    model: str | None,
    caller: GroundingCaller,
    *,
    force_most_likely: bool = False,
    preselected: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve multi-part articulations to stable catalog IDs.

    This is a GenSim-specific extension of scene grounding.  The model sees
    only part IDs and semantic summaries; native USD paths remain code-owned.
    Single-part articulations and explicit vertical selectors are resolved
    without an additional model call.
    """
    requests = [
        {
            "reference_id": f"{step['id']}.object",
            "reference": step["object"].get("reference", ""),
            "task_type": step["task_type"],
        }
        for step in intent.get("steps", [])
        if step.get("task_type") == "E6"
        and step.get("object", {}).get("kind") == "scene_ref"
    ]
    pending: list[dict[str, Any]] = []
    resolved: dict[str, str] = dict(preselected or {})
    for request in requests:
        uids = tuple(reference_bindings.get(request["reference_id"], ()))
        if len(uids) != 1 or uids[0] not in part_catalogs:
            continue
        parts = tuple(part_catalogs[uids[0]])
        vertical_rank = _requested_vertical_rank(str(request["reference"]))
        if vertical_rank is not None and len(parts) > 1:
            matches = [
                part
                for part in parts
                if str(part.get("vertical_rank", "")) == vertical_rank
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Requested vertical rank {vertical_rank!r} is not uniquely "
                    f"available for {request['reference_id']!r}."
                )
            resolved[request["reference_id"]] = str(matches[0]["part_id"])
        elif request["reference_id"] in resolved:
            continue
        elif len(parts) == 1:
            resolved[request["reference_id"]] = str(parts[0]["part_id"])
        elif parts:
            pending.append({**request, "articulation_id": uids[0], "parts": parts})
    if not pending:
        return resolved
    if not callable(caller):
        raise TypeError("Part grounding caller must be callable.")
    schema = {
        "title": "TaskEngineArticulationPartGrounding",
        "type": "object",
        "additionalProperties": False,
        "required": ["bindings"],
        "properties": {
            "bindings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["reference_id", "status", "part_id", "confidence"],
                    "properties": {
                        "reference_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["resolved", "ambiguous", "not_found"],
                        },
                        "part_id": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                },
            }
        },
    }
    prompt_parts = [
        {
            "reference_id": item["reference_id"],
            "reference": item["reference"],
            "articulation_id": item["articulation_id"],
            "candidates": [
                {
                    "part_id": str(part["part_id"]),
                    "joint": str(part.get("joint", "")),
                    "link": str(part.get("link", "")),
                    "vertical_rank": str(part.get("vertical_rank", "")),
                    "vertical_index": part.get("vertical_index"),
                    "part_count": part.get("part_count"),
                    "semantic_labels": _part_labels(part),
                }
                for part in item["parts"]
            ],
        }
        for item in pending
    ]
    response = caller(
        prompt=(
            "Select the requested GenSim articulation part. Return only one "
            "candidate part_id per reference_id from the supplied candidates. "
            "Never return a USD path or invent an ID. Use ambiguous/not_found "
            "when the instruction cannot be resolved.\n\n"
            f"Instruction:\n{instruction.strip()}\n\n"
            f"Requests:\n{json.dumps(prompt_parts, ensure_ascii=False, sort_keys=True)}"
        ),
        schema=schema,
        model=model,
    )
    raw = response.get("bindings") if isinstance(response, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Part grounding output must contain a bindings list.")
    request_by_id = {item["reference_id"]: item for item in pending}
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Part grounding binding must be a mapping.")
        reference_id = str(item.get("reference_id", ""))
        if reference_id not in request_by_id or reference_id in seen:
            raise ValueError(
                "Part grounding returned an unknown or duplicate reference."
            )
        seen.add(reference_id)
        confidence = item.get("confidence")
        if (
            item.get("status") != "resolved"
            or not isinstance(confidence, (int, float))
            or float(confidence) < 0.5
        ):
            if force_most_likely:
                part_id = str(request_by_id[reference_id]["parts"][0]["part_id"])
            else:
                raise ValueError(f"Part grounding for {reference_id!r} is ambiguous.")
        else:
            part_id = str(item.get("part_id", ""))
        allowed = {
            str(part["part_id"]) for part in request_by_id[reference_id]["parts"]
        }
        if part_id not in allowed:
            raise ValueError(f"Part grounding selected unknown part {part_id!r}.")
        resolved[reference_id] = part_id
    if seen != set(request_by_id):
        raise ValueError("Part grounding omitted one or more requested references.")
    return resolved


_VERTICAL_REFERENCE_TERMS = {
    "top": (
        "最上",
        "上面",
        "上方",
        "上层",
        "顶部",
        "顶层",
        "top",
        "upper",
        "highest",
    ),
    "middle": ("中间", "中部", "中层", "middle", "center", "central"),
    "bottom": (
        "最下",
        "下面",
        "下方",
        "下层",
        "底部",
        "底层",
        "bottom",
        "lower",
        "lowest",
    ),
}


def _requested_vertical_rank(reference: str) -> str | None:
    normalized = reference.strip().lower()
    matches = {
        rank
        for rank, terms in _VERTICAL_REFERENCE_TERMS.items()
        if any(
            (
                term in normalized
                if not term.isascii()
                else re.search(rf"\b{re.escape(term)}\b", normalized) is not None
            )
            for term in terms
        )
    }
    if len(matches) > 1:
        raise ValueError(
            f"Articulation reference {reference!r} contains conflicting vertical ranks."
        )
    return next(iter(matches), None)


def _part_labels(part: Mapping[str, Any]) -> list[str]:
    labels: set[str] = set()
    for key in ("joint", "link", "joint_path", "link_path", "vertical_rank"):
        value = str(part.get(key, "")).strip().lower()
        if not value:
            continue
        labels.add(value)
        labels.update(
            token
            for token in value.replace("/", "_").replace("-", "_").split("_")
            if token
        )
    return sorted(label for label in labels if label)


def ground_scene_references(
    instruction: str,
    intent: Mapping[str, Any],
    inventory: SceneInventory,
    scene_objects: Sequence[Mapping[str, Any]],
    model: str | None,
    caller: GroundingCaller,
) -> GroundingResult:
    """Resolve every ``scene_ref`` selector in one task-conditioned batch.

    The grounding model can only select stable UIDs from a redacted inventory.
    Its output does not add affordances, physical state, coordinates, or poses.
    One failed format validation is repaired with one additional model call.
    Explicitly ambiguous or missing references are rejected without retry.

    Args:
        instruction: Original user instruction for task-level context.
        intent: Validated structured instruction intent.
        inventory: Structural scene inventory defining authoritative candidates.
        scene_objects: Original semantic inventory used to retain open labels.
        model: Model name forwarded unchanged to the injected caller.
        caller: Structured model transport accepting ``prompt``, ``schema``, and
            ``model`` keyword arguments.

    Returns:
        Validated UID bindings together with call-count and latency statistics.

    Raises:
        TypeError: If the intent or response has an invalid container type.
        ValueError: If requests are malformed or grounding remains invalid after
            one repair attempt.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Grounding instruction must be a non-empty string.")
    if not callable(caller):
        raise TypeError("Grounding caller must be callable.")

    requests = _collect_requests(intent)
    prompt_inventory = _grounding_inventory(inventory, scene_objects)
    prompt = _grounding_prompt(instruction.strip(), requests, prompt_inventory)
    started = perf_counter()
    first_error: Exception | None = None

    for attempt in range(2):
        current_prompt = prompt
        if first_error is not None:
            current_prompt += (
                "\n\nREPAIR OVERRIDE: the previous grounding JSON failed local "
                "validation. Return one corrected JSON object only. Preserve the "
                "exact output fields bindings/reference_id/status/uids/confidence, "
                "cover every requested reference exactly once, and select only "
                "UIDs from the supplied candidate inventory. Validation error: "
                f"{first_error}"
            )
        try:
            response = caller(
                prompt=current_prompt,
                schema=deepcopy(_GROUNDING_SCHEMA),
                model=model,
            )
            bindings = _validate_response(
                response,
                requests=requests,
                inventory=inventory,
            )
            return GroundingResult(
                bindings=bindings,
                attempts=attempt + 1,
                latency_seconds=perf_counter() - started,
            )
        except _UnresolvedSceneReference:
            raise
        except (TypeError, ValueError) as error:
            if attempt:
                raise ValueError(
                    "Scene grounding failed validation after one repair: " f"{error}"
                ) from error
            first_error = error
    raise AssertionError("unreachable")


def _collect_requests(intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(intent, Mapping):
        raise TypeError("Instruction intent must be a mapping.")
    steps = intent.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise ValueError("Instruction intent steps must be a list.")

    requests: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for step_index, step in enumerate(steps):
        context = f"InstructionIntent.steps[{step_index}]"
        if not isinstance(step, Mapping):
            raise ValueError(f"{context} must be a mapping.")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"{context}.id must be a non-empty string.")
        task_type = step.get("task_type")
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError(f"{context}.task_type must be a non-empty string.")
        relation = step.get("relation", "none")
        if not isinstance(relation, str):
            raise ValueError(f"{context}.relation must be a string.")

        for slot in ("object", "target"):
            selector = step.get(slot)
            if not isinstance(selector, Mapping):
                raise ValueError(f"{context}.{slot} must be a mapping.")
            if selector.get("kind") != "scene_ref":
                continue
            reference = selector.get("reference")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"{context}.{slot}.reference must be a non-empty string."
                )
            quantifier = selector.get("quantifier")
            if quantifier not in _QUANTIFIERS:
                raise ValueError(
                    f"{context}.{slot}.quantifier must be one of "
                    f"{sorted(_QUANTIFIERS)}."
                )
            count = selector.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{context}.{slot}.count must be an integer >= 0.")
            if quantifier == "count" and count < 1:
                raise ValueError(
                    f"{context}.{slot} quantifier=count requires count>=1."
                )
            if quantifier != "count" and count != 0:
                raise ValueError(
                    f"{context}.{slot} quantifier={quantifier} requires count=0."
                )

            request_id = f"{step_id}.{slot}"
            if request_id in request_ids:
                raise ValueError(f"Duplicate grounding request ID {request_id!r}.")
            request_ids.add(request_id)
            requests.append(
                {
                    "reference_id": request_id,
                    "step_id": step_id,
                    "slot": slot,
                    "task_type": task_type,
                    "relation": relation,
                    "reference": reference.strip(),
                    "quantifier": quantifier,
                    "count": count,
                }
            )
    return requests


def _grounding_inventory(
    inventory: SceneInventory,
    scene_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_by_uid: dict[str, Mapping[str, Any]] = {}
    for item_index, raw in enumerate(scene_objects):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Scene inventory item {item_index} must be a mapping.")
        uid = str(raw.get("runtime_uid", raw.get("uid", ""))).strip()
        if uid:
            raw_by_uid[uid] = raw

    ranked = sorted(
        inventory.entities,
        key=lambda entity: (-inventory.left_score(entity), entity.uid),
    )
    rank_by_uid = {entity.uid: rank for rank, entity in enumerate(ranked, start=1)}
    payload = []
    for entity in sorted(inventory.entities, key=lambda item: item.uid):
        raw = raw_by_uid.get(entity.uid, {})
        score = inventory.left_score(entity)
        side = "left" if score > 0.0 else "right" if score < 0.0 else "center"
        raw_category = raw.get(
            "category",
            raw.get("object_category", entity.category),
        )
        attributes = _redact_semantic_mapping(entity.attributes)
        if entity.color is not None:
            attributes.setdefault("color", entity.color)
        payload.append(
            {
                "uid": entity.uid,
                "role": entity.role,
                "name": str(raw.get("name", entity.name)).strip(),
                "category": str(raw_category).strip() or entity.category,
                "description": entity.description,
                "affordances": sorted(entity.affordances),
                "attributes": attributes,
                "initial_state": _redact_semantic_mapping(entity.initial_state),
                "side": side,
                "rank": rank_by_uid[entity.uid],
            }
        )
    return payload


def _grounding_prompt(
    instruction: str,
    requests: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> str:
    return (
        "Ground the requested natural-language scene references to the supplied "
        "scene inventory. Resolve all requests together using the original task, "
        "step type, relation, quantifier, and reference text as context. Select "
        "only exact inventory UIDs. The inventory's affordances and states are "
        "source evidence only: never infer, add, authorize, or return an "
        "affordance, capability, physical state, coordinate, pose, orientation, "
        "path, or action. The side and rank fields are discrete robot-relative "
        "labels; rank 1 is leftmost. Object requests may select only movable "
        "inventory entities. Target requests may also select support surfaces. "
        "Use status=ambiguous or status=not_found instead of guessing when the "
        "evidence is insufficient. Return exactly one binding per reference_id "
        "with only reference_id, status, uids, and confidence.\n\n"
        f"Instruction:\n{instruction}\n\n"
        "Grounding requests:\n"
        f"{json.dumps(list(requests), ensure_ascii=False, sort_keys=True)}\n\n"
        "Redacted scene inventory:\n"
        f"{json.dumps(list(inventory), ensure_ascii=False, sort_keys=True)}"
    )


def _validate_response(
    value: Mapping[str, Any],
    *,
    requests: Sequence[Mapping[str, Any]],
    inventory: SceneInventory,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError("Scene grounding output must be a mapping.")
    if set(value) != {"bindings"}:
        raise ValueError(
            "Scene grounding output must contain exactly the 'bindings' field."
        )
    raw_bindings = value["bindings"]
    if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
        raise ValueError("Scene grounding bindings must be a list.")

    request_by_id = {str(request["reference_id"]): request for request in requests}
    bindings: dict[str, tuple[str, ...]] = {}
    for binding_index, raw in enumerate(raw_bindings):
        context = f"SceneGrounding.bindings[{binding_index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context} must be a mapping.")
        if set(raw) != _BINDING_KEYS:
            missing = sorted(_BINDING_KEYS - set(raw))
            extra = sorted(set(raw) - _BINDING_KEYS)
            raise ValueError(
                f"{context} fields must be exactly {sorted(_BINDING_KEYS)}; "
                f"missing={missing}, unsupported={extra}."
            )
        reference_id = raw["reference_id"]
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError(f"{context}.reference_id must be a non-empty string.")
        if reference_id not in request_by_id:
            raise ValueError(f"{context} references unknown request {reference_id!r}.")
        if reference_id in bindings:
            raise ValueError(f"Duplicate grounding binding for {reference_id!r}.")

        status = raw["status"]
        if status not in {"resolved", "ambiguous", "not_found"}:
            raise ValueError(
                f"{context}.status must be resolved, ambiguous, or not_found."
            )
        if status != "resolved":
            raise _UnresolvedSceneReference(
                f"Grounding request {reference_id!r} was not resolved: {status}."
            )
        confidence = raw["confidence"]
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"{context}.confidence must be a number in [0, 1].")
        if float(confidence) < 0.5:
            raise ValueError(
                f"Grounding request {reference_id!r} confidence is below 0.5."
            )

        raw_uids = raw["uids"]
        if not isinstance(raw_uids, Sequence) or isinstance(raw_uids, (str, bytes)):
            raise ValueError(f"{context}.uids must be a list.")
        uids = tuple(raw_uids)
        if any(not isinstance(uid, str) or not uid for uid in uids):
            raise ValueError(f"{context}.uids must contain non-empty strings.")
        if len(set(uids)) != len(uids):
            raise ValueError(
                f"Grounding request {reference_id!r} contains duplicate UIDs."
            )
        unknown = sorted(set(uids) - set(inventory.by_uid))
        if unknown:
            raise ValueError(
                f"Grounding request {reference_id!r} selected unknown UIDs {unknown}."
            )

        request = request_by_id[reference_id]
        # An exact runtime identity is stronger evidence than a model's label
        # match. Never silently substitute another object for that identity.
        reference = request["reference"]
        if reference in inventory.by_uid and uids != (reference,):
            raise _UnresolvedSceneReference(
                f"Grounding request {reference_id!r} names exact UID "
                f"{reference!r}, but selected {uids!r}."
            )
        allowed = (
            {entity.uid for entity in inventory.interactive}
            if request["slot"] == "object"
            else {entity.uid for entity in (*inventory.interactive, *inventory.support)}
        )
        disallowed = sorted(set(uids) - allowed)
        if disallowed:
            raise ValueError(
                f"Grounding request {reference_id!r} selected UIDs outside its "
                f"{request['slot']} candidate range: {disallowed}."
            )
        _validate_cardinality(request, uids)
        bindings[reference_id] = uids

    missing = sorted(set(request_by_id) - set(bindings))
    if missing:
        raise ValueError(f"Scene grounding omitted requests {missing}.")
    _reject_self_references(requests, bindings)
    return bindings


def _validate_cardinality(
    request: Mapping[str, Any],
    uids: Sequence[str],
) -> None:
    request_id = str(request["reference_id"])
    quantifier = str(request["quantifier"])
    if quantifier == "one" and len(uids) != 1:
        raise ValueError(
            f"Grounding request {request_id!r} quantifier=one requires exactly one UID."
        )
    if quantifier == "count" and len(uids) != int(request["count"]):
        raise ValueError(
            f"Grounding request {request_id!r} requires exactly "
            f"{request['count']} UIDs."
        )
    if quantifier == "all" and not uids:
        raise ValueError(
            f"Grounding request {request_id!r} quantifier=all requires at "
            "least one UID."
        )


def _reject_self_references(
    requests: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, tuple[str, ...]],
) -> None:
    slots_by_step: dict[str, dict[str, str]] = {}
    for request in requests:
        slots_by_step.setdefault(str(request["step_id"]), {})[str(request["slot"])] = (
            str(request["reference_id"])
        )
    for step_id, slots in slots_by_step.items():
        object_id = slots.get("object")
        target_id = slots.get("target")
        if object_id is None or target_id is None:
            continue
        overlap = sorted(set(bindings[object_id]) & set(bindings[target_id]))
        if overlap:
            raise ValueError(
                f"Grounding step {step_id!r} uses the same UID as object and "
                f"target: {overlap}."
            )


def _redact_semantic_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        name = str(key)
        normalized = name.strip().lower().replace("-", "_")
        if normalized in _REDACTED_KEYS:
            continue
        if isinstance(child, Mapping):
            nested = _redact_semantic_mapping(child)
            if nested:
                result[name] = nested
        elif isinstance(child, (str, int, float, bool)) and not isinstance(
            child, complex
        ):
            result[name] = child
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            semantic_values = [item for item in child if isinstance(item, (str, bool))]
            if semantic_values and len(semantic_values) == len(child):
                result[name] = semantic_values
    return result
