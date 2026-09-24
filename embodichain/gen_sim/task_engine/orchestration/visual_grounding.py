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

"""UID-constrained multimodal fallback for unresolved scene references."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

__all__ = ["make_visual_grounding_caller", "visual_grounding_available"]

_VISUAL_SCHEMA: dict[str, Any] = {
    "title": "TaskEngineVisualSceneGrounding",
    "type": "object",
    "additionalProperties": False,
    "required": ["bindings"],
    "properties": {
        "bindings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "reference_id",
                    "status",
                    "uids",
                    "confidence",
                    "evidence_view",
                    "evidence_note",
                    "scene_missing",
                ],
                "properties": {
                    "reference_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["resolved", "ambiguous", "not_found"],
                    },
                    "uids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence_view": {
                        "type": "string",
                        "enum": ["oblique", "top", "catalog", "none"],
                    },
                    "evidence_note": {"type": "string"},
                    "scene_missing": {"type": "boolean"},
                },
            },
        }
    },
}
_CORE_KEYS = ("reference_id", "status", "uids", "confidence")
VisualTransport = Callable[..., Mapping[str, Any]]


def visual_grounding_available() -> bool:
    """Use the same model configuration as text grounding."""
    from embodichain.gen_sim.task_engine.interpretation import _load_llm_settings

    try:
        _load_llm_settings(model=None)
    except ValueError:
        return False
    return True


def _verified_images(evidence: Mapping[str, Any]) -> tuple[Path, ...]:
    from .scene_source import scene_revision_id

    config = Path(str(evidence["source_config_path"]))
    if (
        hashlib.sha256(config.read_bytes()).hexdigest()
        != evidence["source_config_sha256"]
    ):
        raise ValueError("Visual grounding source scene changed after rendering.")
    if scene_revision_id(config) != evidence["scene_revision_id"]:
        raise ValueError("Visual grounding scene revision changed after rendering.")
    for item in evidence.get("objects", ()):
        asset = Path(str(item["asset_path"]))
        if hashlib.sha256(asset.read_bytes()).hexdigest() != item["asset_sha256"]:
            raise ValueError(f"Visual grounding source asset changed: {asset}")
    values = [
        (Path(str(view["annotated_path"])), str(view["annotated_sha256"]))
        for view in evidence["views"]
    ]
    values.append(
        (Path(str(evidence["catalog_path"])), str(evidence["catalog_sha256"]))
    )
    for path, expected in values:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Visual grounding image changed after rendering: {path}")
    return tuple(path for path, _ in values)


def _validate_visual_response(
    raw: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    if set(raw) != {"bindings"} or not isinstance(raw["bindings"], list):
        raise ValueError("Visual grounding must return only a bindings list.")
    visible_views = {
        str(item["uid"]): set(item.get("visible_views", ()))
        for item in evidence.get("objects", ())
    }
    for item in raw["bindings"]:
        if not isinstance(item, Mapping) or set(item) != set(_CORE_KEYS) | {
            "evidence_view",
            "evidence_note",
            "scene_missing",
        }:
            raise ValueError("Visual grounding binding has invalid evidence fields.")
        status = item["status"]
        uids = item["uids"]
        if (
            status not in {"resolved", "ambiguous", "not_found"}
            or not isinstance(uids, list)
            or any(not isinstance(uid, str) for uid in uids)
            or not isinstance(item["scene_missing"], bool)
            or item["evidence_view"] not in {"oblique", "top", "catalog", "none"}
            or not isinstance(item["evidence_note"], str)
            or not item["evidence_note"].strip()
        ):
            raise ValueError("Visual grounding binding has invalid visual evidence.")
        if status == "not_found":
            if uids:
                raise ValueError("Missing scene reference cannot select a UID.")
        elif (
            item["scene_missing"]
            or item["evidence_view"] == "none"
            or not uids
            or any(
                uid not in visible_views
                or not visible_views[uid]
                or (
                    item["evidence_view"] != "catalog"
                    and item["evidence_view"] not in visible_views[uid]
                )
                for uid in uids
            )
        ):
            raise ValueError(
                "Visual grounding selected a UID without visible evidence."
            )


def make_visual_grounding_caller(
    evidence: Mapping[str, Any],
    audit_dir: str | Path,
    *,
    transport: VisualTransport | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """Wrap one VLM so existing UID and compatibility validators stay authoritative."""
    if evidence.get("schema_version") != "gen_sim.visual-grounding-evidence/v1":
        raise ValueError("Visual grounding evidence schema is unsupported.")
    if transport is None and not visual_grounding_available():
        raise ValueError(
            "Configure the Task Engine LLM in gen_sim/.env or the process environment."
        )
    root = Path(audit_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    invoke = _default_visual_transport if transport is None else transport
    call_count = 0

    def caller(
        *, prompt: str, schema: Mapping[str, Any], model: str | None
    ) -> Mapping[str, Any]:
        nonlocal call_count
        images = _verified_images(evidence)
        call_count += 1
        image_hashes = {
            view["name"]: view["annotated_sha256"] for view in evidence["views"]
        }
        image_hashes["catalog"] = evidence["catalog_sha256"]
        visual_prompt = (
            "Use the current scene render and labeled UID catalog only as visual "
            "evidence for the supplied grounding requests. Return only existing "
            "inventory UIDs. If no matching UID exists, whether absent or "
            "visible but unlabeled, return not_found with scene_missing=true; "
            "never substitute another object based only on shape. Articulation "
            "GLBs are editing proxies, not live joint states: "
            "never infer open/closed state from them. Do not infer physical "
            "capability, contact, or coordinates from pixels. Use ambiguous "
            "when identity cannot be established.\n\n"
            f"Evidence render kind: {evidence['render_kind']}\n"
            f"Evidence source hash: {evidence['source_config_sha256']}\n"
            f"Scene revision: {evidence['scene_revision_id']}\n"
            f"Evidence image SHA-256: {json.dumps(image_hashes, sort_keys=True)}\n"
            f"Joint-state-untrusted UIDs: {json.dumps(evidence['articulation_state_untrusted_uids'])}\n\n"
            f"Unrendered UIDs: {json.dumps(evidence.get('unrendered_uids', []))}\n\n"
            f"{prompt}"
        )
        raw = invoke(
            prompt=visual_prompt,
            schema=deepcopy(_VISUAL_SCHEMA),
            model=model,
            image_paths=images,
        )
        if not isinstance(raw, Mapping):
            raise ValueError("Visual grounding must return a mapping.")
        audit = {
            "schema_version": "gen_sim.visual-grounding-call/v1",
            "source_config_sha256": evidence["source_config_sha256"],
            "scene_revision_id": evidence["scene_revision_id"],
            "image_paths": [os.path.relpath(path, start=root) for path in images],
            "image_sha256": image_hashes,
            "prompt_sha256": hashlib.sha256(visual_prompt.encode()).hexdigest(),
            "response": deepcopy(dict(raw)),
        }
        (root / f"call_{call_count:02d}.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _validate_visual_response(raw, evidence)
        return {
            "bindings": [
                {key: item[key] for key in _CORE_KEYS} for item in raw["bindings"]
            ]
        }

    return caller


def _default_visual_transport(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str | None,
    image_paths: tuple[Path, ...],
) -> Mapping[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from embodichain.gen_sim.task_engine.interpretation import (
        _MIMO_MAX_COMPLETION_TOKENS,
        _coerce_instruction_response,
        _is_mimo_compatible,
        _load_llm_settings,
        _structured_output_runnable,
    )

    settings = _load_llm_settings(model=model)
    kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "model": settings["model"],
        "temperature": 0,
        "max_retries": 0,
        "timeout": 90,
        "http_socket_options": (),
    }
    for key in ("base_url", "default_query"):
        if settings[key]:
            kwargs[key] = settings[key]
    if _is_mimo_compatible(settings):
        kwargs.update(
            max_completion_tokens=_MIMO_MAX_COMPLETION_TOKENS,
            extra_body={"thinking": {"type": "disabled"}},
        )
    client = ChatOpenAI(**kwargs)
    runnable = _structured_output_runnable(client, schema, settings=settings)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt
            + "\n\nReturn JSON conforming to this schema:\n"
            + json.dumps(schema, ensure_ascii=False, sort_keys=True),
        }
    ]
    for path in image_paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    response = runnable.invoke(
        [
            SystemMessage(
                content="Return only grounded structured JSON. No reasoning or poses."
            ),
            HumanMessage(content=content),
        ]
    )
    return _coerce_instruction_response(response)
