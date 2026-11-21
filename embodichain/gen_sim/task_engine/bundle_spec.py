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

"""Explicit optional normative-spec declaration for executable Task Engine bundles."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Final
from embodichain.gen_sim.task_spec import TaskSpec, validate_task_spec

__all__ = [
    "ABSENT_TASK_SPEC_SCHEMA",
    "bundle_task_spec_payload",
    "decode_bundle_task_spec",
]

ABSENT_TASK_SPEC_SCHEMA: Final = "task_spec.absent/v1"


def bundle_task_spec_payload(
    value: TaskSpec | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Serialize a verified normative spec or its explicit absence.

    Args:
        value: Existing TaskSpec/data with its valid semantic identity, or None.

    Returns:
        Detached existing TaskSpec format, unchanged for valid specifications;
        None emits only ``schema_version: task_spec.absent/v1``. Absence is not
        an empty task, a successful specification or a physical certificate.
    """
    if value is None:
        return {"schema_version": ABSENT_TASK_SPEC_SCHEMA}
    return validate_task_spec(value)


def decode_bundle_task_spec(value: Mapping[str, Any]) -> TaskSpec | None:
    """Decode the closed absence marker or validate a populated specification.

    Args:
        value: Decoded ``task_spec.json`` declaration mapping.

    Returns:
        Independent validated TaskSpec, or None only for the exact versioned
        absence marker. Missing files and null values are errors; present values
        retain the existing TaskSpec decoder and validation rules.

    Raises:
        ValueError: An absence marker has unsupported fields or version.
    """
    if not isinstance(value, Mapping):
        raise TypeError("Bundle TaskSpec declaration must be a mapping.")
    schema = value.get("schema_version")
    if isinstance(schema, str) and schema.startswith("task_spec.absent/"):
        if dict(value) != {"schema_version": ABSENT_TASK_SPEC_SCHEMA}:
            raise ValueError("Unsupported or extended TaskSpec absence declaration.")
        return None
    return TaskSpec.from_dict(value)
