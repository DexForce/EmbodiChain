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

"""Pure identity and release-window verification for revision-one support."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

__all__: list[str] = []


def _time(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


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


def release_window(
    call: Mapping[str, Any], receipt: Mapping[str, Any], num_envs: int
) -> dict[str, Any]:
    """Read the source-verified release interval without parsing skill phases."""
    from .canonicalization import json_snapshot

    window = call.get("normalized_release_window")
    if not isinstance(window, Mapping):
        raise ValueError("Support release lacks its normalized observation window.")
    return json_snapshot(window)
