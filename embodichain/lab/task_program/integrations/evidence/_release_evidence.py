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

"""Pure snapshot checks for measured release identity and physical receipts."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

__all__: list[str] = []


def measured_release(
    release_runtime: Mapping[str, Any],
    boundary_runtime: Mapping[str, Any],
    *,
    object_uid: str,
    resource: str,
    invocation_id: str,
    num_envs: int,
) -> tuple[list[bool], list[bool], Mapping[str, Any], Mapping[str, Any]]:
    """Recheck a selected invocation's terminal release and boundary held state.

    Inputs are already-owned JSON snapshots. Selection of workflow nodes and
    role bindings belongs to the caller; invocation/receipt identities are
    checked here before publishing an observation to TaskSpec.
    """
    import torch
    from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

    runtime = release_runtime
    identities = runtime["env_ids"]
    if (
        not isinstance(identities, list)
        or any(type(v) is not int for v in identities)
        or identities != list(range(num_envs))
    ):
        raise ValueError(
            "Placement release evidence has unsupported environment identity."
        )
    result = [False] * num_envs
    available = [False] * num_envs
    found = False
    verified_receipt = verified_call = None
    for call in runtime.get("calls", []):
        if (
            call.get("status") != "completed"
            or call.get("invocation_id") != invocation_id
        ):
            continue
        if (
            type(call.get("invocation_revision")) is not int
            or call["invocation_revision"] < 0
        ):
            continue
        for receipt in call.get("effects", []):
            spec = receipt["effect_spec"]
            if (
                receipt.get("boundary", {}).get("kind") != "terminal"
                or type(receipt.get("timestamp")) not in (int, float)
                or not math.isfinite(receipt["timestamp"])
                or receipt["timestamp"] < 0
                or type(receipt.get("observation_revision")) is not int
                or receipt["observation_revision"] < 0
                or spec.get("effect_kind") != "release"
                or spec.get("invocation_id") != call["invocation_id"]
                or spec.get("invocation_revision") != call.get("invocation_revision")
                or spec.get("env_ids") != identities
            ):
                continue
            expectations = [
                v
                for v in spec["state_expectations"]
                if v.get("object_id") == object_uid
                and v.get("resource_id") == resource
                and v.get("relation") == "detached"
            ]
            if len(expectations) != 1:
                continue
            clauses = [
                v
                for v in spec["clauses"]
                if v["expectation_id"] == expectations[0]["expectation_id"]
            ]
            constraint = next(
                (
                    v
                    for v in clauses
                    if v.get("evidence_kind") == "constraint"
                    and v.get("expected") is False
                ),
                None,
            )
            pose_clause = next(
                (v for v in clauses if v.get("expectation") == "separated"), None
            )
            if constraint is None or pose_clause is None:
                continue
            samples = [
                receipt["evidence"][v["clause_id"]] for v in (constraint, pose_clause)
            ]
            success = receipt["decision"]["success_mask"]
            if (
                len(success) != num_envs
                or any(type(v) is not bool for v in success)
                or any(
                    s.get("env_ids") != identities
                    or s.get("timestamp") != receipt["timestamp"]
                    or s.get("observation_revision") != receipt["observation_revision"]
                    or len(s.get("valid_mask", [])) != num_envs
                    or any(type(v) is not bool for v in s["valid_mask"])
                    for s in samples
                )
                or len(samples[0].get("values", [])) != num_envs
                or any(type(v) is not bool for v in samples[0]["values"])
            ):
                continue
            relative = torch.tensor(samples[1]["object_to_endpoint"])
            validate_rigid_pose_batch(relative, num_envs)
            found = True
            verified_receipt, verified_call = receipt, call
            available = [
                first and second
                for first, second in zip(
                    samples[0]["valid_mask"], samples[1]["valid_mask"], strict=True
                )
            ]
            result = [
                ok and a and b and value is False
                for ok, a, b, value in zip(
                    success,
                    samples[0]["valid_mask"],
                    samples[1]["valid_mask"],
                    samples[0]["values"],
                    strict=True,
                )
            ]
    if not found:
        raise ValueError("No exact measured terminal release receipt is available.")
    current = boundary_runtime
    state = current["task_state"]
    if (
        current["env_ids"] != identities
        or any(type(v) is not int for v in current["env_ids"])
        or type(state["batch_size"]) is not int
        or state["batch_size"] != num_envs
    ):
        raise ValueError(
            "Placement held-state evidence has different environment identity."
        )
    for held in [*state["held_objects"], *state["coordinated_held_objects"]]:
        if held["object_id"] != object_uid:
            continue
        mask = held["active_mask"]
        if len(mask) != num_envs or any(type(v) is not bool for v in mask):
            raise ValueError("Placement held-state mask is invalid.")
        result = [ok and not active for ok, active in zip(result, mask, strict=True)]
    assert verified_receipt is not None and verified_call is not None
    return result, available, verified_receipt, verified_call


def release_window(
    call: Mapping[str, Any], receipt: Mapping[str, Any], num_envs: int
) -> dict[str, Any]:
    """Bind an already verified release receipt to its actual attempt and phase."""
    attempts = call.get("plan_attempts", [])
    if not attempts:
        raise ValueError("Support release lacks an actual planning attempt.")
    last = attempts[-1]
    generation = call.get("active_plan_attempt_generation")
    identity = {key: call[key] for key in ("invocation_id", "invocation_revision")}
    index = last.get("invocation_index")
    if (
        type(generation) is not int
        or generation < 0
        or last.get("attempt_generation") != generation
        or (type(index) is not int)
        or (index < 0)
        or any((last.get(key) != value for (key, value) in identity.items()))
        or (not _time(last.get("planned_at")))
    ):
        raise ValueError("Support release has a foreign or superseded plan.")
    planned = last["planned_at"]
    events = [
        event
        for event in call.get("events", [])
        if all((event.get(key) == value for (key, value) in identity.items()))
        and event.get("invocation_index") == index
        and _time(event.get("timestamp"))
        and (event["timestamp"] >= planned)
    ]
    entries = [
        event
        for event in events
        if event.get("kind") == "trajectory_segment_entered"
        and event.get("segment_name") == "release"
    ]
    completed = [event for event in events if event.get("kind") == "action_completed"]
    if (
        len(entries) != 1
        or len(completed) != 1
        or entries[0].get("env_mask") != [True] * num_envs
        or (completed[0].get("env_mask") != [True] * num_envs)
        or (not _time(receipt.get("timestamp")))
        or (
            not entries[0]["timestamp"]
            <= receipt["timestamp"]
            <= completed[0]["timestamp"]
        )
        or any(
            (
                type(receipt.get(key)) is not int or receipt[key] < 0
                for key in ("verification_id", "observation_revision")
            )
        )
    ):
        raise ValueError("Support release lacks its exact measured phase window.")
    return {
        **identity,
        "attempt_generation": generation,
        "invocation_index": index,
        "planned_at": planned,
        "release_started_at": entries[0]["timestamp"],
        "timestamp": receipt["timestamp"],
        "completed_at": completed[0]["timestamp"],
        "verification_id": receipt["verification_id"],
        "observation_revision": receipt["observation_revision"],
    }


def _time(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0
