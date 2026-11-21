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

"""Pure source-input provenance; no policy, asset resolution or runtime branch."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, Final

from embodichain.gen_sim.task_spec import canonical_json, json_snapshot

__all__ = ["SOURCE_INPUT_PROVENANCE_FILE", "SOURCE_INPUT_PROVENANCE_SCHEMA"]

SOURCE_INPUT_PROVENANCE_FILE: Final = "source_input_provenance.json"
SOURCE_INPUT_PROVENANCE_SCHEMA: Final = "gen_sim.source-input-provenance/v1"


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if type(name) is not str or name in result:
            raise ValueError("Source-input provenance requires unique string keys.")
        result[name] = value
    return result


def _validated_provenance(
    value: Mapping[str, Any] | bytes,
    external_inputs: Mapping[str, str],
) -> dict[str, Any]:
    payload = (
        json.loads(value.decode("utf-8"), object_pairs_hook=_unique)
        if type(value) is bytes
        else json_snapshot(value)
    )
    if (
        type(payload) is not dict
        or set(payload) != {"schema_version", "groups"}
        or payload["schema_version"] != SOURCE_INPUT_PROVENANCE_SCHEMA
        or type(payload["groups"]) is not dict
        or not payload["groups"]
    ):
        raise ValueError("Unsupported or incomplete source-input provenance.")
    combined: dict[str, str] = {}
    for owner, entries in payload["groups"].items():
        if type(owner) is not str or not owner or owner != owner.strip():
            raise ValueError("Source-input owner names must be exact identifiers.")
        if type(entries) is not dict:
            raise TypeError("Source-input groups contain only path/SHA-256 mappings.")
        for name, digest in entries.items():
            if type(name) is not str or not name or name != name.strip():
                raise ValueError("Source-input paths must be exact nonempty strings.")
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("Source-input values must be SHA-256 digests.")
            if name in combined and combined[name] != digest:
                raise ValueError("Source-input groups disagree about the same path.")
            combined[name] = digest
    if combined != dict(external_inputs):
        raise ValueError("Source-input provenance union differs from external inputs.")
    return payload


def _provenance_bytes(
    value: Mapping[str, Any] | bytes,
    external_inputs: Mapping[str, str],
) -> bytes:
    return (
        canonical_json(_validated_provenance(value, external_inputs)) + "\n"
    ).encode()
