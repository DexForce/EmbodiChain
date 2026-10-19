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

"""Adapt existing Scene Engine exports without changing their source schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import (
    STATIC_SCENE_MANIFEST_SCHEMA,
    StaticSceneManifest,
    validate_static_scene_manifest,
)

__all__ = ["SceneEngineV1Adapter"]


class SceneEngineV1Adapter:
    """Convert a normalized Scene Engine v1 export to the neutral manifest."""

    def adapt_prepared_scene(
        self,
        prepared_scene: Any,
        *,
        source_format: str,
        robot_profile: str,
    ) -> StaticSceneManifest:
        """Adapt the existing prepared-scene view through a duck-typed boundary."""
        planner_objects = tuple(getattr(prepared_scene, "planner_objects"))
        runtime_objects = (
            tuple(getattr(prepared_scene, "background", ()))
            + tuple(getattr(prepared_scene, "rigid_objects", ()))
            + tuple(getattr(prepared_scene, "articulations", ()))
        )
        runtime_by_uid = {
            str(item.get("uid")): item
            for item in runtime_objects
            if isinstance(item, Mapping) and item.get("uid")
        }
        asset_hashes = dict(getattr(prepared_scene, "asset_hashes", {}) or {})
        objects = [
            self._object_manifest(
                raw,
                runtime=runtime_by_uid.get(str(raw.get("uid")), {}),
                asset_sha256=str(asset_hashes.get(str(raw.get("uid")), "")),
            )
            for raw in planner_objects
        ]
        identity = {
            "source_format": str(source_format),
            "robot_profile": str(robot_profile),
            "objects": [_identity_object(item) for item in objects],
        }
        source_path = Path(getattr(prepared_scene, "source_config_path"))
        return validate_static_scene_manifest(
            {
                "schema_version": STATIC_SCENE_MANIFEST_SCHEMA,
                "scene_id": _canonical_hash(identity),
                "source_format": str(source_format),
                "robot_profile": str(robot_profile),
                "source": {
                    "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
                    "config_path": source_path.expanduser().resolve().as_posix(),
                    "config_sha256": _file_hash(source_path),
                    "asset_hashes": asset_hashes,
                },
                "adapter_capabilities": {
                    "task_conditioned_generation": False,
                    "structured_affordances": any(
                        bool(item["affordances"]) for item in objects
                    ),
                    "articulation_instances": any(
                        item["role"] == "articulation" for item in objects
                    ),
                    "runtime_scene_observation": False,
                },
                "objects": objects,
            }
        )

    def _object_manifest(
        self,
        raw: Mapping[str, Any],
        *,
        runtime: Mapping[str, Any],
        asset_sha256: str,
    ) -> dict[str, Any]:
        uid = str(raw.get("uid", "")).strip()
        role = str(raw.get("role", "")).strip()
        shape = raw.get("shape", runtime.get("shape", {}))
        shape = deepcopy(dict(shape)) if isinstance(shape, Mapping) else {}
        physics_keys = ("attrs", "body_type", "max_convex_hull_num")
        physics = {
            key: deepcopy(runtime[key]) for key in physics_keys if key in runtime
        }
        articulation = deepcopy(dict(runtime)) if role == "articulation" else {}
        affordances = _affordance_evidence(raw.get("affordances", ()))
        if role in {"background", "table", "support_surface"}:
            affordances = _with_structural_evidence(
                affordances,
                "support_surface",
            )
        if role in {"object", "rigid_object"}:
            affordances = _with_structural_evidence(affordances, "rigid")
        return {
            "uid": uid,
            "source_uid": str(raw.get("source_uid", "")),
            "role": role,
            "name": str(raw.get("name", "")),
            "description": str(raw.get("description", "")),
            "category": str(raw.get("category", "")),
            "color": raw.get("color") if isinstance(raw.get("color"), str) else None,
            "geometry": {
                "shape": shape,
                "asset_sha256": asset_sha256,
            },
            "initial_pose": {
                "position": deepcopy(list(raw.get("init_pos", ()))),
                "rotation": deepcopy(list(raw.get("init_rot", ()))),
                "scale": deepcopy(list(raw.get("body_scale", ()))),
            },
            "physics": physics,
            "articulation": articulation,
            "affordances": affordances,
            "initial_state": _mapping_or_empty(raw.get("initial_state")),
            "attributes": _mapping_or_empty(raw.get("attributes")),
            "provenance": {
                "semantic_source": "scene_export",
                "geometry_source": "prepared_scene",
                "physics_source": "prepared_scene_runtime",
            },
        }


def _affordance_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for raw in value:
        if isinstance(raw, str) and raw.strip():
            result.append(_evidence(raw.strip(), status="declared"))
            continue
        if not isinstance(raw, Mapping):
            continue
        affordance_type = str(raw.get("type", raw.get("name", ""))).strip()
        if not affordance_type:
            continue
        status = str(raw.get("status", "declared"))
        result.append(
            {
                "type": affordance_type,
                "status": status,
                "confidence": raw.get("confidence"),
                "source": str(raw.get("source", "scene_export")),
                "link_uid": str(raw.get("link_uid", "")),
                "frame": _mapping_or_empty(raw.get("frame")),
                "parameters": _mapping_or_empty(raw.get("parameters")),
            }
        )
    return sorted(result, key=lambda item: (item["type"], item["source"]))


def _with_structural_evidence(
    evidence: list[dict[str, Any]], affordance_type: str
) -> list[dict[str, Any]]:
    if any(item["type"] == affordance_type for item in evidence):
        return evidence
    return sorted(
        [
            *evidence,
            _evidence(affordance_type, status="verified", source="adapter_structure"),
        ],
        key=lambda item: (item["type"], item["source"]),
    )


def _evidence(
    affordance_type: str,
    *,
    status: str,
    source: str = "scene_export",
) -> dict[str, Any]:
    return {
        "type": affordance_type,
        "status": status,
        "confidence": None,
        "source": source,
        "link_uid": "",
        "frame": {},
        "parameters": {},
    }


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _identity_object(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    geometry = result.get("geometry")
    if isinstance(geometry, dict):
        shape = geometry.get("shape")
        if isinstance(shape, dict) and geometry.get("asset_sha256"):
            shape.pop("fpath", None)
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return ""
    return hashlib.sha256(resolved.read_bytes()).hexdigest()
