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

"""Normalized release facts used by normative placement and Pour predicates."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from .canonicalization import canonical_hash

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
    facts = release_runtime["normalized_release"]
    if (
        facts.get("normalization_hash")
        != canonical_hash({k: v for k, v in facts.items() if k != "normalization_hash"})
        or facts.get("object_id") != object_uid
        or facts.get("resource_id") != resource
        or facts.get("invocation_id") != invocation_id
        or facts.get("env_ids") != list(range(num_envs))
    ):
        raise ValueError("Release facts have another declared semantic identity.")
    for name in ("released", "available"):
        values = facts[name]
        if (
            not isinstance(values, list)
            or len(values) != num_envs
            or any(type(x) is not bool for x in values)
        ):
            raise ValueError("Normalized release measurement rows are malformed.")
    return facts["released"], facts["available"], facts["receipt"], facts["call"]
