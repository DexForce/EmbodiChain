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

"""Conservative hierarchy evidence for imported scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Final, TypeAlias

__all__ = [
    "CONSERVATIVE_SCENE_GRAPH_SCHEMA",
    "ConservativeSceneGraph",
    "build_conservative_scene_graph",
    "validate_conservative_scene_graph",
]

CONSERVATIVE_SCENE_GRAPH_SCHEMA: Final = "embodichain.conservative-scene-graph/v1"
ConservativeSceneGraph: TypeAlias = dict[str, Any]


def build_conservative_scene_graph(
    prepared_scene: Any,
    *,
    scene_id: str,
) -> ConservativeSceneGraph:
    """Use exported hierarchy when available and mark every gap as unknown."""
    source_path = Path(getattr(prepared_scene, "source_config_path")).resolve()
    uid_map = dict(getattr(prepared_scene, "uid_map", {}) or {})
    exported = _read_exported_graph(source_path.with_name("scene_graph.json"))
    operational_assumptions = _legacy_operational_assumption_uids(source_path)
    exported_nodes = {
        str(node.get("object_id")): node
        for node in exported.get("nodes", ())
        if isinstance(node, Mapping) and node.get("object_id")
    }

    nodes: list[dict[str, Any]] = []
    for raw in getattr(prepared_scene, "planner_objects"):
        uid = str(raw.get("uid", ""))
        source_uid = str(raw.get("source_uid", uid))
        known = exported_nodes.get(source_uid) or exported_nodes.get(uid)
        if uid == "table":
            node = {
                "uid": uid,
                "parent_uid": None,
                "parent_relation": "root",
                "orientation": "unknown",
                "source": "structural_root",
            }
        elif (
            known is None
            or uid in operational_assumptions
            or source_uid in operational_assumptions
        ):
            node = {
                "uid": uid,
                "parent_uid": "unknown",
                "parent_relation": "unknown",
                "orientation": "unknown",
                "source": "conservative_import",
            }
        else:
            raw_parent = known.get("parent_id")
            parent_uid = (
                uid_map.get(str(raw_parent), str(raw_parent))
                if raw_parent is not None
                else "unknown"
            )
            relation = known.get("parent_relation")
            orientation = known.get("orientation_state")
            node = {
                "uid": uid,
                "parent_uid": parent_uid,
                "parent_relation": relation if relation == "on" else "unknown",
                "orientation": (
                    orientation if orientation in {"standing", "lying"} else "unknown"
                ),
                "source": "scene_graph",
            }
        nodes.append(node)

    relations = []
    for raw in exported.get("relations", ()):
        if not isinstance(raw, Mapping):
            continue
        source_uid = uid_map.get(str(raw.get("source_id")), str(raw.get("source_id")))
        target_uid = uid_map.get(str(raw.get("target_id")), str(raw.get("target_id")))
        relation = str(raw.get("relation", ""))
        if source_uid and target_uid and relation:
            relations.append(
                {
                    "source_uid": source_uid,
                    "relation": relation,
                    "target_uid": target_uid,
                    "source": "scene_graph",
                }
            )
    return validate_conservative_scene_graph(
        {
            "schema_version": CONSERVATIVE_SCENE_GRAPH_SCHEMA,
            "scene_id": str(scene_id),
            "nodes": nodes,
            "relations": relations,
        }
    )


def _legacy_operational_assumption_uids(source_path: Path) -> set[str]:
    manifest_path = source_path.parent.parent / "legacy_conversion.json"
    if not manifest_path.is_file():
        return set()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Legacy conversion manifest is invalid JSON: {manifest_path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("Legacy conversion manifest must contain an object.")
    assumptions = value.get("assumptions", ())
    if not isinstance(assumptions, Sequence) or isinstance(assumptions, (str, bytes)):
        raise ValueError("Legacy conversion assumptions must be a sequence.")
    return {
        str(item["uid"])
        for item in assumptions
        if isinstance(item, Mapping) and isinstance(item.get("uid"), str)
    }


def validate_conservative_scene_graph(
    value: Mapping[str, Any],
) -> ConservativeSceneGraph:
    """Validate and detach one conservative graph."""
    if not isinstance(value, Mapping):
        raise TypeError("ConservativeSceneGraph must be a mapping.")
    result = deepcopy(dict(value))
    expected = {"schema_version", "scene_id", "nodes", "relations"}
    if set(result) != expected:
        raise ValueError("ConservativeSceneGraph fields are invalid.")
    if result.get("schema_version") != CONSERVATIVE_SCENE_GRAPH_SCHEMA:
        raise ValueError("ConservativeSceneGraph schema version is invalid.")
    if not isinstance(result.get("scene_id"), str) or not result["scene_id"]:
        raise ValueError("ConservativeSceneGraph.scene_id must not be empty.")
    nodes = _sequence(result.get("nodes"), "nodes")
    normalized_nodes = []
    for index, raw in enumerate(nodes):
        if not isinstance(raw, Mapping):
            raise TypeError(f"ConservativeSceneGraph.nodes[{index}] must be a mapping.")
        node = dict(raw)
        if set(node) != {
            "uid",
            "parent_uid",
            "parent_relation",
            "orientation",
            "source",
        }:
            raise ValueError(
                f"ConservativeSceneGraph.nodes[{index}] fields are invalid."
            )
        if not isinstance(node["uid"], str) or not node["uid"]:
            raise ValueError(f"ConservativeSceneGraph.nodes[{index}].uid is invalid.")
        if node["parent_uid"] is not None and not isinstance(node["parent_uid"], str):
            raise TypeError(
                f"ConservativeSceneGraph.nodes[{index}].parent_uid is invalid."
            )
        if node["parent_relation"] not in {"root", "on", "unknown"}:
            raise ValueError(
                f"ConservativeSceneGraph.nodes[{index}].parent_relation is invalid."
            )
        if node["orientation"] not in {"standing", "lying", "unknown"}:
            raise ValueError(
                f"ConservativeSceneGraph.nodes[{index}].orientation is invalid."
            )
        if not isinstance(node["source"], str) or not node["source"]:
            raise ValueError(
                f"ConservativeSceneGraph.nodes[{index}].source is invalid."
            )
        normalized_nodes.append(node)
    if len({node["uid"] for node in normalized_nodes}) != len(normalized_nodes):
        raise ValueError("ConservativeSceneGraph node UIDs must be unique.")
    result["nodes"] = normalized_nodes
    result["relations"] = [
        dict(item) for item in _sequence(result.get("relations"), "relations")
    ]
    json.dumps(result, allow_nan=False)
    return result


def _read_exported_graph(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene graph is not valid JSON: {path}") from exc
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"ConservativeSceneGraph.{field_name} must be a sequence.")
    return list(value)
