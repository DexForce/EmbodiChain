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

"""Normative comparison of source-bound normalized physical control facts."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .canonicalization import canonical_hash
from .evidence import _digest

__all__: list[str] = []


def control_value(binding: Any, snapshot: Mapping[str, Any], row: int) -> bool:
    """Read one valid control-effect observation without a runtime dependency."""
    if snapshot.get("schema_version") != "physical_evidence.normalized/v1":
        raise ValueError(
            "Control evidence must first be normalized by the Lab adapter."
        )
    if snapshot.get("measurement_hash") != canonical_hash(
        {key: value for key, value in snapshot.items() if key != "measurement_hash"}
    ):
        raise ValueError("Normalized control evidence changed under its identity.")
    refs = snapshot.get("evidence_refs")
    if not isinstance(refs, list) or len(refs) != 2 or refs[0] != binding.binding_hash:
        raise ValueError("Normalized control references have another source.")
    _digest(refs[1], "source_evidence_hash")
    if (
        snapshot.get("source_binding_hash") != binding.binding_hash
        or any(type(v) is not int for v in snapshot.get("env_ids", ()))
        or (
            snapshot.get("episode_ids") is not None
            and any(type(v) is not int for v in snapshot["episode_ids"])
        )
        or snapshot.get("kind") != "control_" + binding.control_kind
        or snapshot.get("evidence_format") != binding.evidence_format
        or snapshot.get("criterion")
        != binding.operation["route_payload"].get(
            "success_criterion", "contact_backed_press"
        )
        or snapshot.get("env_ids") != list(binding.env_ids)
        or snapshot.get("episode_ids")
        != (None if binding.episode_ids is None else list(binding.episode_ids))
    ):
        raise ValueError("Normalized control facts have another declared scope.")
    if "unavailable" in snapshot:
        raise ValueError(str(snapshot["unavailable"]))
    frame = snapshot["frame"]
    timestamp = frame["timestamp"]
    if (
        frame["env_ids"] != list(binding.env_ids)
        or any(type(v) is not int for v in frame["env_ids"])
        or type(frame["observation_revision"]) is not int
        or frame["observation_revision"] < 0
        or type(timestamp) not in (int, float)
        or not math.isfinite(timestamp)
        or timestamp < 0
    ):
        raise ValueError("Normalized control frame has another row or clock identity.")
    boundaries = snapshot.get("boundary_frames")
    if binding.observation_kind == "recorded" or boundaries is not None:
        if (
            not isinstance(boundaries, list)
            or len(boundaries) != len(binding.env_ids)
            or any(type(v) is not int or v <= 0 for v in boundaries)
            or frame["observation_revision"] != max(boundaries)
        ):
            raise ValueError("Control facts have another completed event boundary.")
    batch = frame["batches"]["control_complete"]
    count = len(binding.env_ids)
    if (
        batch["evidence_id"] != "control_complete"
        or batch["value_dtype"] != "torch.bool"
        or ("value_shape" in batch and batch["value_shape"] != [count])
        or batch["env_ids"] != frame["env_ids"]
        or any(type(v) is not int for v in batch["env_ids"])
        or batch["timestamp"] != timestamp
        or batch["observation_revision"] != frame["observation_revision"]
        or any(
            not isinstance(batch[key], list) or len(batch[key]) != count
            for key in ("values", "valid", "acquisition_errors")
        )
        or any(type(value) is not bool for value in (*batch["values"], *batch["valid"]))
    ):
        raise ValueError(
            "Normalized control channel has inconsistent identities or rows."
        )
    if not batch["valid"][row] or batch["acquisition_errors"][row] is not None:
        raise ValueError(
            batch["acquisition_errors"][row] or "Control acquisition unavailable."
        )
    return batch["values"][row]
