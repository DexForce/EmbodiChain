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

"""Shared semantic validation and identity for numeric training-data schemas."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch

__all__ = ["validate_data_schema", "data_schema_fingerprint"]


def validate_data_schema(schema: Mapping[str, Any]) -> None:
    """Validate declarations common to live and resident causal sources.

    Args:
        schema: Observation/action declarations. Actions require explicit
            semantics or representation and finite positive ``step_dt``.
            Optional names are ordered and unique; dtype names use Torch's
            spelling without its prefix. Optional observation ``timing`` must
            be ``pre_action``. Root and feature ``version`` values identify
            schemas and feature transformations when supplied.

    This validates declarations, not their truth or compatibility with a live
    robot. Source adapters also validate stored features and action contracts.
    Units, frames, rotations and normalization remain caller declarations; no
    conversion or missing semantic information is inferred from vector widths.

    Raises:
        ValueError: A required semantic declaration is absent or malformed.
    """
    if not isinstance(schema, Mapping):
        raise ValueError("Schema must be a mapping")
    definitions = [schema]
    for section in ("observation", "action"):
        definition = schema.get(section, {})
        if not isinstance(definition, Mapping):
            raise ValueError(f"Schema {section} must be a mapping")
        definitions.append(definition)
        names = definition.get("names")
        if names is not None and (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"Schema {section}.names must be ordered unique names")
        dtype = definition.get("dtype")
        if dtype is not None and (
            not isinstance(dtype, str)
            or not isinstance(getattr(torch, dtype, None), torch.dtype)
        ):
            raise ValueError(f"Schema {section} dtype must be a Torch dtype name")
    for definition in definitions:
        if "version" in definition:
            value = definition["version"]
            if not (
                isinstance(value, str)
                and value.strip()
                or type(value) is int
                and value > 0
            ):
                raise ValueError(
                    "Schema version must be a nonempty string or positive integer"
                )
    action = schema.get("action", {})
    semantics = action.get("semantics", action.get("representation"))
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("Explicit action semantics or representation is required")
    step_dt = action.get("step_dt")
    if (
        isinstance(step_dt, bool)
        or not isinstance(step_dt, (float, int))
        or not math.isfinite(step_dt)
        or step_dt <= 0
    ):
        raise ValueError("Explicit finite positive action step_dt is required")
    observation = schema.get("observation", {})
    if observation.get("timing", "pre_action") != "pre_action":
        raise ValueError("Causal source observation timing must be pre_action")


def data_schema_fingerprint(schema: Mapping[str, Any]) -> str:
    """Hash a validated finite JSON schema without changing feature order.

    Args:
        schema: Persistable declaration, including any feature-transform,
            normalization, coordinate-frame and action-contract versions.

    Returns:
        SHA-256 of canonical UTF-8 JSON. Mapping key order is ignored; ordered
        lists and caller-declared versions remain significant. This is a schema
        identity, not a hash of trajectory content or callable implementation.

    Raises:
        ValueError: The schema is invalid or not finite JSON data.
    """
    validate_data_schema(schema)
    try:
        payload = json.dumps(
            dict(schema), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Schema fingerprint requires finite JSON metadata") from error
    return hashlib.sha256(payload).hexdigest()
