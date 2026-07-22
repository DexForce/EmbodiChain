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

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import re
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    GYM_CONFIG_PREFERENCE,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_records import (
    resolve_source_gym_config,
)

__all__ = ["resolve_target_replacements"]

_INDEXED_REPLACEMENT_ALIAS_RE = re.compile(
    r"^(?P<keyword>[A-Za-z][A-Za-z0-9 _-]*?)[ _-]?(?P<index>[0-9]+)$"
)


def resolve_target_replacements(
    args: argparse.Namespace,
    target_replacement_spec_cls: Callable[..., object],
    gym_project: Path,
) -> list[object]:
    replacements = []
    alias_config = None
    generic_replacements = list(getattr(args, "target_replacement", []) or [])
    legacy_replacements = [
        (index, values)
        for index, values in (
            (1, getattr(args, "target_replacement1", None)),
            (2, getattr(args, "target_replacement2", None)),
        )
        if values
    ]
    if generic_replacements and legacy_replacements:
        raise ValueError(
            "Use either repeated --target-replacement or legacy "
            "--target-replacement1/2, not both."
        )

    replacement_values = (
        list(enumerate(generic_replacements, start=1))
        if generic_replacements
        else legacy_replacements
    )
    for index, values in replacement_values:
        alias_config = alias_config or _load_replacement_alias_config(gym_project)
        source_uid, prompt = _resolve_target_replacement_arg(
            values,
            alias_config,
            option_name=_replacement_option_name(generic_replacements, index),
            replacement_number=index,
        )
        replacements.append(
            target_replacement_spec_cls(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name=f"new{index}",
            )
        )
    return replacements


def _replacement_option_name(generic_replacements: list[list[str]], index: int) -> str:
    if generic_replacements:
        return f"--target_replacement[{index}]"
    return f"--target_replacement{index}"


def _resolve_target_replacement_arg(
    values: list[str],
    gym_config: dict[str, Any],
    *,
    option_name: str,
    replacement_number: int,
) -> tuple[str, str]:
    if len(values) == 1:
        prompt = str(values[0]).strip()
        if not prompt:
            raise ValueError(f"{option_name} prompt must be non-empty.")
        source_uid = _auto_replacement_source_uid(
            gym_config,
            replacement_number=replacement_number,
            option_name=option_name,
        )
        return source_uid, prompt

    if len(values) == 2:
        source_uid, prompt = values
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError(f"{option_name} prompt must be non-empty.")
        source_uid = _resolve_replacement_source_uid(
            source_uid,
            gym_config,
            option_name=option_name,
        )
        return source_uid, prompt

    raise ValueError(
        f"{option_name} expects either PROMPT or SOURCE_UID PROMPT, got "
        f"{len(values)} values: {values!r}. Quote multi-word prompts."
    )


def _load_replacement_alias_config(gym_project: Path) -> dict[str, Any]:
    config_path = _resolve_replacement_alias_gym_config(gym_project)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Gym config must be a JSON object: {config_path}")
    return data


def _resolve_replacement_alias_gym_config(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        sibling_gym_config = input_path.parent / "gym_config.json"
        if sibling_gym_config.is_file():
            return sibling_gym_config.resolve()
        return _resolve_source_gym_config(input_path)

    direct_gym_config = input_path / "gym_config.json"
    if direct_gym_config.is_file():
        return direct_gym_config.resolve()

    source_config = _resolve_source_gym_config(input_path)
    sibling_gym_config = source_config.parent / "gym_config.json"
    if sibling_gym_config.is_file():
        return sibling_gym_config.resolve()
    return source_config


def _auto_replacement_source_uid(
    gym_config: dict[str, Any],
    *,
    replacement_number: int,
    option_name: str,
) -> str:
    duplicate_groups = _duplicated_numbered_rigid_object_groups(gym_config)
    if len(duplicate_groups) != 1:
        candidates = _format_duplicate_group_candidates(duplicate_groups)
        raise ValueError(
            f"{option_name} was given without an explicit source uid, so the "
            "pipeline expected exactly one duplicated numbered rigid_object "
            f"group in gym_config.json. Found {len(duplicate_groups)} group(s): "
            f"{candidates}. Use SOURCE_UID PROMPT to disambiguate."
        )

    base_name, positioned_objects = duplicate_groups[0]
    if replacement_number > len(positioned_objects):
        candidates = _format_duplicate_group_candidates(duplicate_groups)
        raise ValueError(
            f"{option_name} auto-selection requested replacement #{replacement_number} "
            f"from duplicated group {base_name!r}, but only found "
            f"{len(positioned_objects)} object(s): "
            f"{candidates}. Use SOURCE_UID PROMPT to disambiguate."
        )

    y_values = [float(item["y"]) for item in positioned_objects]
    if len({round(value, 9) for value in y_values}) != len(y_values):
        candidates = _format_duplicate_group_candidates(duplicate_groups)
        raise ValueError(
            f"{option_name} auto-selection requires distinct y coordinates in "
            f"duplicated group {base_name!r}: {candidates}. Use SOURCE_UID PROMPT "
            "to disambiguate."
        )

    selected = positioned_objects[-replacement_number]
    source_uid = selected["object"]["uid"]
    print(
        f"Resolved {option_name} auto source -> {source_uid!r} "
        f"from duplicated rigid_object group {base_name!r} by y={selected['y']}",
        flush=True,
    )
    return source_uid


def _duplicated_numbered_rigid_object_groups(
    gym_config: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in _rigid_objects(gym_config):
        parsed = _parse_numbered_rigid_object_uid(obj["uid"])
        if parsed is None:
            continue
        base_name, number = parsed
        grouped.setdefault(base_name, []).append(
            {
                "number": number,
                "y": _rigid_object_y_coordinate(obj),
                "object": obj,
            }
        )

    duplicate_groups = []
    for base_name, entries in grouped.items():
        if len(entries) < 2:
            continue
        duplicate_groups.append(
            (
                base_name,
                sorted(
                    entries,
                    key=lambda entry: (
                        float(entry["y"]),
                        str(entry["object"]["uid"]),
                    ),
                ),
            )
        )
    return sorted(duplicate_groups, key=lambda item: item[0])


def _parse_numbered_rigid_object_uid(uid: str) -> tuple[str, int] | None:
    match = re.match(r"^(?P<base>.+?)[_-]?(?P<number>[0-9]+)$", uid)
    if match is None:
        return None
    base_name = match.group("base").strip("_-")
    if not base_name:
        return None
    return base_name, int(match.group("number"))


def _rigid_object_y_coordinate(obj: dict[str, Any]) -> float:
    init_pos = obj.get("init_pos")
    if not isinstance(init_pos, (list, tuple)) or len(init_pos) < 2:
        raise ValueError(
            "Auto replacement source selection requires each duplicated "
            f"rigid_object to define init_pos with a y value, got {obj.get('uid')!r}."
        )
    try:
        return float(init_pos[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Auto replacement source selection requires numeric init_pos[1], "
            f"got {obj.get('uid')!r}: {init_pos[1]!r}"
        ) from exc


def _format_duplicate_group_candidates(
    groups: list[tuple[str, list[dict[str, Any]]]],
) -> str:
    if not groups:
        return "<none>"
    parts = []
    for base_name, entries in groups:
        values = ", ".join(
            f"{entry['object']['uid']}#number={entry['number']},y={entry['y']}"
            for entry in entries
        )
        parts.append(f"{base_name}: {values}")
    return "; ".join(parts)


def _resolve_replacement_source_uid(
    source_input: str,
    gym_config: dict[str, Any],
    *,
    option_name: str,
) -> str:
    source_input = str(source_input).strip()
    rigid_objects = _rigid_objects(gym_config)
    by_uid = {obj["uid"]: obj for obj in rigid_objects}
    if source_input in by_uid:
        return source_input

    alias = _parse_indexed_replacement_alias(source_input)
    if alias is None:
        candidates = _format_rigid_object_candidates(rigid_objects)
        raise ValueError(
            f"{option_name} source {source_input!r} is neither a rigid object uid "
            f"nor an indexed alias such as bread1. Rigid object candidates: "
            f"{candidates}"
        )

    keyword, alias_index = alias
    matches = [
        obj for obj in rigid_objects if _rigid_object_matches_keyword(obj, keyword)
    ]
    if alias_index > len(matches):
        candidates = _format_rigid_object_candidates(matches or rigid_objects)
        raise ValueError(
            f"{option_name} alias {source_input!r} requested match #{alias_index} "
            f"for keyword {keyword!r}, but only found {len(matches)} match(es). "
            f"Candidates: {candidates}"
        )

    resolved_uid = matches[alias_index - 1]["uid"]
    print(
        f"Resolved {option_name} source alias {source_input!r} -> {resolved_uid!r}",
        flush=True,
    )
    return resolved_uid


def _rigid_objects(gym_config: dict[str, Any]) -> list[dict[str, Any]]:
    value = gym_config.get("rigid_object", [])
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("gym config rigid_object must be a list or object.")

    rigid_objects = []
    for obj in value:
        if not isinstance(obj, dict):
            continue
        uid = str(obj.get("uid", "")).strip()
        if not uid:
            continue
        copied = dict(obj)
        copied["uid"] = uid
        rigid_objects.append(copied)
    if not rigid_objects:
        raise ValueError("No rigid_object entries found in gym config.")
    return rigid_objects


def _parse_indexed_replacement_alias(alias: str) -> tuple[str, int] | None:
    match = _INDEXED_REPLACEMENT_ALIAS_RE.match(alias.strip())
    if match is None:
        return None
    keyword = match.group("keyword").strip(" _-")
    index = int(match.group("index"))
    if not keyword or index < 1:
        return None
    return keyword, index


def _rigid_object_matches_keyword(obj: dict[str, Any], keyword: str) -> bool:
    keyword_tokens = _search_tokens(keyword)
    if not keyword_tokens:
        return False
    object_tokens = set(_search_tokens(_rigid_object_search_text(obj)))
    return all(token in object_tokens for token in keyword_tokens)


def _rigid_object_search_text(obj: dict[str, Any]) -> str:
    values = [
        obj.get("uid", ""),
        obj.get("source_uid", ""),
        obj.get("category", ""),
        obj.get("semantic_label", ""),
        obj.get("name", ""),
        obj.get("description", ""),
    ]
    shape = obj.get("shape", {})
    if isinstance(shape, dict):
        values.extend(
            [
                shape.get("fpath", ""),
                shape.get("file_path", ""),
                shape.get("category", ""),
            ]
        )
    return " ".join(str(value) for value in values if value)


def _search_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


def _format_rigid_object_candidates(rigid_objects: list[dict[str, Any]]) -> str:
    if not rigid_objects:
        return "<none>"
    parts = []
    for obj in rigid_objects:
        shape = obj.get("shape", {})
        fpath = shape.get("fpath", "") if isinstance(shape, dict) else ""
        parts.append(f"{obj.get('uid')} ({fpath})")
    return ", ".join(parts)


def _resolve_source_gym_config(input_path: Path) -> Path:
    return resolve_source_gym_config(
        input_path,
        gym_config_preference=GYM_CONFIG_PREFERENCE,
    )
