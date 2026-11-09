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

"""Read-only validation of report and coverage companions for offline display."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from embodichain.gen_sim.task_spec import TaskSpec
from .contracts import canonical_hash
from .bundle_spec import decode_bundle_task_spec
from .semantic_graph import validate_semantic_task_graph
from .semantic_graph_validation import validate_graph_coverage

__all__: list[str] = []
GRAPH_VALIDATION_SCHEMA = "semantic_graph_validation/v1"


def validate_display_coverage(
    value: Mapping[str, Any],
    spec: TaskSpec | None,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute declaration coverage without executing or upgrading a bundle."""
    selected = validate_semantic_task_graph(graph)
    envelope = deepcopy(dict(value))
    if spec is None:
        raise ValueError(
            "HTML coverage companion identity does not match graph/TaskSpec."
        )
    if envelope.get("schema_version") == "semantic_plan_validation/v1":
        from .semantic_plan_validation import validate_semantic_plan
        from embodichain.lab.task_program.language.semantic_contracts import (
            semantic_planning_catalog,
        )

        if (
            envelope.get("graph_digest") != canonical_hash(selected)
            or envelope.get("task_spec_digest") != canonical_hash(spec.to_dict())
            or envelope.get("catalog_digest")
            != canonical_hash(semantic_planning_catalog())
        ):
            raise ValueError("Semantic plan companion identity mismatch")
        source = envelope["source"]
        from .task_spec import task_spec_from_candidate

        if task_spec_from_candidate(source["candidate"]).to_dict() != spec.to_dict():
            raise ValueError("Semantic plan TaskSpec differs from source")
        coverage = validate_semantic_plan(selected, **source)
        if coverage != envelope.get("coverage"):
            raise ValueError("Semantic plan companion is stale")
        return envelope
    legacy = envelope.get("schema_version") == GRAPH_VALIDATION_SCHEMA
    modern = envelope.get("schema_version") == "semantic_graph_coverage/v1"
    if not (legacy or modern):
        raise ValueError("Unsupported HTML coverage companion.")
    if legacy and (
        envelope.get("checker_revision") != 1
        or envelope.get("graph_digest") != canonical_hash(selected)
        or envelope.get("task_spec_digest") != canonical_hash(spec.to_dict())
    ):
        raise ValueError(
            "HTML coverage companion identity does not match graph/TaskSpec."
        )
    source = envelope.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"candidate", "role_bindings"}
        or not isinstance(envelope.get("coverage"), Mapping)
        or "lineage" not in envelope["coverage"]
    ):
        raise ValueError("Malformed HTML coverage companion.")
    coverage = validate_graph_coverage(
        selected,
        spec,
        source["candidate"],
        source["role_bindings"],
        envelope["coverage"]["lineage"],
    )
    if coverage != envelope["coverage"]:
        raise ValueError("HTML coverage companion is stale.")
    return envelope


def verify_display_companions(
    bundle: str | Path, graph: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Read original/current coverage files as evidence, without writing inputs."""
    root = Path(bundle)
    path = root / "semantic_graph_validation.json"
    if not path.is_file():
        return None
    spec = decode_bundle_task_spec(json.loads((root / "task_spec.json").read_text()))
    return validate_display_coverage(json.loads(path.read_text()), spec, graph)


def _validate_runtime_evaluation(
    value: Mapping[str, Any],
    spec: TaskSpec,
    graph: Mapping[str, Any],
    env_ids: list[int],
    *,
    contract: Mapping[str, Any] | None = None,
) -> None:
    """Check verdict identity, complete clause ownership and summary consistency."""
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "task_spec_evaluation/v1"
        or value.get("task_spec_digest") != canonical_hash(spec.to_dict())
        or value.get("graph_digest") != canonical_hash(graph)
    ):
        raise ValueError("TaskSpec runtime evaluation identity differs.")
    if contract is not None and any(
        value.get(key) != contract[key]
        for key in ("program_digest", "constraints_digest")
        if key in contract
    ):
        raise ValueError("TaskSpec runtime evaluation executable identity differs.")
    rows = value.get("environments")
    if (
        not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or any(type(row.get("env_id")) is not int for row in rows)
        or [row.get("env_id") for row in rows] != env_ids
    ):
        raise ValueError("TaskSpec runtime evaluation environment identities differ.")
    expected = {
        "init": {p.id for p in spec.init},
        "goal": {p.id for p in spec.goal},
        "invariants": {p.id for p in spec.invariants},
        "temporal": {p.id for p in spec.temporal},
        "milestones": {f"milestone:{p.id}" for p in spec.milestones},
    }
    priority = ("failed", "unavailable", "unknown", "not_run", "satisfied")
    for row in rows:
        statuses = []
        for section, identities in expected.items():
            records = row.get(section)
            if (
                not isinstance(records, list)
                or any(not isinstance(p, Mapping) for p in records)
                or len(records) != len(identities)
                or {p.get("predicate_id") for p in records} != identities
                or any(p.get("status") not in priority for p in records)
            ):
                raise ValueError("TaskSpec runtime evaluation clauses are incomplete.")
            statuses.extend(p["status"] for p in records)
        aggregate = next(status for status in priority if status in statuses)
        if row.get("status") != aggregate or row.get("certificate_ready") is not (
            aggregate == "satisfied"
        ):
            raise ValueError(
                "TaskSpec runtime evaluation summary contradicts its clauses."
            )
