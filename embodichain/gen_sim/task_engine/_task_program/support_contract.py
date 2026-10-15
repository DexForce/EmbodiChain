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

"""Time and identity checks for the explicit GenSim support evidence contract."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

__all__: list[str] = []


def _time(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def release_window(
    node: Mapping[str, Any],
    segments: Mapping[str, Any],
    boundary: Mapping[str, Any],
    num_envs: int,
) -> tuple[list[bool], dict[str, Any]]:
    """Bind the verified release to its last actual plan and release phase."""
    from .placement_evidence import measured_release

    raw_call = node["call"]
    uid = raw_call.get("object", raw_call.get("arguments", {}).get("object"))
    released, receipt, call = measured_release(
        node, uid, segments, boundary, num_envs, with_receipt=True
    )
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
        or type(index) is not int
        or index < 0
        or any(last.get(key) != value for key, value in identity.items())
        or not _time(last.get("planned_at"))
    ):
        raise ValueError("Support release has a foreign or superseded plan.")
    planned = last["planned_at"]
    events = [
        event
        for event in call.get("events", [])
        if all(event.get(key) == value for key, value in identity.items())
        and event.get("invocation_index") == index
        and _time(event.get("timestamp"))
        and event["timestamp"] >= planned
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
        or completed[0].get("env_mask") != [True] * num_envs
        or not _time(receipt.get("timestamp"))
        or not entries[0]["timestamp"]
        <= receipt["timestamp"]
        <= completed[0]["timestamp"]
        or any(
            type(receipt.get(key)) is not int or receipt[key] < 0
            for key in ("verification_id", "observation_revision")
        )
    ):
        raise ValueError("Support release lacks its exact measured phase window.")
    return released, {
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


def support_window_flags(
    native: Mapping[str, Any],
    window: Mapping[str, Any],
    num_envs: int,
    *,
    entity: str,
    reference: str,
    native_entity: str,
    native_reference: str,
) -> list[bool]:
    """Bind load to the release phase and require verified release completion."""
    now, step, dt = (
        native.get("observed_at"),
        native.get("observed_step"),
        native.get("step_dt"),
    )
    rows, epochs, pairs = (
        native.get(key) for key in ("rows", "episode_generations", "actor_pairs")
    )
    if (
        type(native.get("support_contract_revision")) is not int
        or native["support_contract_revision"] != 1
        or native.get("clock") != "environment_step"
        or native.get("scope") != "native_contact_support"
        or native.get("entity") != entity
        or native.get("reference") != reference
        or native.get("env_ids") != list(range(num_envs))
        or any(type(v) is not int for v in native.get("env_ids", []))
        or not _time(now)
        or type(step) is not int
        or step < 0
        or not _time(dt)
        or dt == 0
        or not math.isclose(now, step * dt, rel_tol=0, abs_tol=1e-8)
        or not _time(window.get("release_started_at"))
        or not _time(window.get("timestamp"))
        or not _time(window.get("completed_at"))
        or not window["release_started_at"]
        <= window["timestamp"]
        <= window["completed_at"]
        or window["completed_at"] > now + 1e-8
        or any(
            not isinstance(values, list) or len(values) != num_envs
            for values in (rows, epochs, pairs)
        )
    ):
        raise ValueError("Native support lacks its canonical sample/window identity.")
    accepted = []
    for row, (sample_row, epoch, pair) in enumerate(
        zip(rows, epochs, pairs, strict=True)
    ):
        if (
            type(epoch) is not int
            or epoch < 0
            or not isinstance(sample_row, Mapping)
            or not isinstance(pair, Mapping)
        ):
            raise ValueError("Native support epoch or actor pair is invalid.")
        if (
            pair.get("object_uid") != native_entity
            or pair.get("reference_uid") != native_reference
        ):
            raise ValueError("Native support actors differ from frozen bindings.")
        for kind in ("object", "reference"):
            actor = pair.get(kind + "_actor")
            info = pair.get(kind + "_info", {})
            if (
                type(actor) is not int
                or actor < 0
                or type(info.get("actor_id")) is not int
                or info["actor_id"] != actor
                or info.get("link_name") is not None
                or type(info.get("env_id")) is not int
                or info["env_id"] not in {-1, row}
                or not isinstance(info.get("path"), str)
                or not info["path"]
                or not isinstance(info.get("arena_name"), str)
                or not info["arena_name"]
            ):
                raise ValueError("Native support has an unresolved physical actor.")
        if pair["object_actor"] == pair["reference_actor"]:
            raise ValueError("Native support cannot be self-contact.")
        sample = sample_row.get("sample")
        if sample is None:
            accepted.append(False)
            continue
        if (
            not isinstance(sample, Mapping)
            or type(sample_row.get("retained")) is not bool
            or any(
                type(sample.get(key)) is not int or sample[key] < lower
                for key, lower in (
                    ("control_step", 1),
                    ("substep", 1),
                    ("serial", 1),
                    ("native_revision", 0),
                    ("topology_revision", 0),
                )
            )
            or type(sample.get("episode_generation")) is not int
            or sample["episode_generation"] != epoch
            or not _time(sample.get("timestamp"))
            or sample["control_step"] > step
            or not (sample["control_step"] - 1) * dt
            < sample["timestamp"]
            <= sample["control_step"] * dt + 1e-8
        ):
            raise ValueError("Native support sample provenance is invalid.")
        # Sleeping contacts retain their actual load time, not the later read time.
        accepted.append(
            window["release_started_at"] - 1e-8 <= sample["timestamp"] <= now + 1e-8
        )
    return accepted
