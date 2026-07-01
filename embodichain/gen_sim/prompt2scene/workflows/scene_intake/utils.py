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

import re
from typing import Any

from embodichain.gen_sim.prompt2scene.utils.log import log_warning
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeAsset,
    SceneIntakeInputRecord,
    SceneIntakeSpec,
    SceneIntakeTable,
)

__all__ = ["build_scene_intake_spec", "normalize_asset_name"]


def normalize_asset_name(name: str) -> str:
    """Normalize an object name for stable asset IDs."""
    normalized = name.strip().lower()
    normalized = normalized.replace("-", " ").replace("/", " ")
    normalized = re.sub(r"[^a-z0-9\s_]", "", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "object"


def build_scene_intake_spec(
    *,
    request: Prompt2SceneInput,
    model_output: dict[str, Any],
) -> SceneIntakeSpec:
    """Normalize raw VLM JSON into the stable scene intake schema."""
    _validate_exact_keys(
        model_output,
        allowed_keys={"table", "assets"},
        context="Scene intake model output",
    )
    input_record = SceneIntakeInputRecord.from_request(request)
    table = _parse_table(_require_mapping(model_output.get("table"), "table"))
    assets = _parse_assets(_require_list(model_output.get("assets"), "assets"))
    return SceneIntakeSpec(input=input_record, table=table, assets=assets)


def _parse_table(raw_table: dict[str, Any]) -> SceneIntakeTable:
    _validate_exact_keys(
        raw_table,
        allowed_keys={
            "name",
            "description",
            "complete_table_description",
            "is_complete_visible_table",
            "class_candidate",
            "object_coverage_percent",
        },
        context="Scene intake table",
    )

    if "name" not in raw_table:
        raise ValueError("Scene intake table.name is required.")
    raw_name = str(raw_table["name"]).strip()
    if not raw_name:
        raise ValueError("Scene intake table.name must be non-empty.")
    name = normalize_asset_name(raw_name)

    if "description" not in raw_table:
        raise ValueError("Scene intake table.description is required.")
    description = str(raw_table["description"]).strip()
    if not description:
        raise ValueError("Scene intake table.description must be non-empty.")

    if "complete_table_description" not in raw_table:
        raise ValueError("Scene intake table.complete_table_description is required.")
    complete_table_description = str(
        raw_table["complete_table_description"]
    ).strip()
    if not complete_table_description:
        raise ValueError(
            "Scene intake table.complete_table_description must be non-empty."
        )

    if "is_complete_visible_table" not in raw_table:
        raise ValueError("Scene intake table.is_complete_visible_table is required.")
    is_complete_visible_table = raw_table["is_complete_visible_table"]
    if not isinstance(is_complete_visible_table, bool):
        raise ValueError(
            "Scene intake table.is_complete_visible_table must be a boolean."
        )

    class_candidate = _parse_class_candidate(
        raw_table.get("class_candidate"),
        asset_index="table",
        raw_name=name,
    )

    object_coverage_percent: int | None = None
    raw_percent = raw_table.get("object_coverage_percent")
    if raw_percent is not None:
        if isinstance(raw_percent, bool):
            raise ValueError(
                "Scene intake table.object_coverage_percent must be an integer, "
                "not a boolean."
            )
        try:
            object_coverage_percent = int(raw_percent)
        except (TypeError, ValueError):
            raise ValueError(
                "Scene intake table.object_coverage_percent must be an integer "
                f"between 1 and 100, got {raw_percent!r}."
            )
        if object_coverage_percent not in (10, 30, 50, 70):
            raise ValueError(
                "Scene intake table.object_coverage_percent must be one of "
                f"10, 30, 50, 70, got {object_coverage_percent}."
            )

    return SceneIntakeTable(
        name=name,
        description=description,
        complete_table_description=complete_table_description,
        is_complete_visible_table=is_complete_visible_table,
        class_candidate=class_candidate,
        object_coverage_percent=object_coverage_percent,
    )


def _parse_assets(raw_assets: list[Any]) -> list[SceneIntakeAsset]:
    assets: list[SceneIntakeAsset] = []
    seen_names: set[str] = set()

    for asset_index, raw_asset in enumerate(raw_assets):
        if not isinstance(raw_asset, dict):
            raise ValueError(f"Scene intake asset {asset_index} must be an object.")
        _validate_exact_keys(
            raw_asset,
            allowed_keys={"name", "description", "class_candidate", "count"},
            context=f"Scene intake asset {asset_index}",
        )

        if "name" not in raw_asset:
            raise ValueError(f"Scene intake asset {asset_index}.name is required.")
        raw_name = str(raw_asset["name"]).strip()
        if not raw_name:
            raise ValueError(
                f"Scene intake asset {asset_index}.name must be non-empty."
            )

        if "description" not in raw_asset:
            raise ValueError(
                f"Scene intake asset {asset_index}.description is required."
            )
        description = str(raw_asset["description"]).strip()
        if not description:
            raise ValueError(
                f"Scene intake asset {asset_index}.description must be non-empty."
            )

        class_candidate = _parse_class_candidate(
            raw_asset.get("class_candidate"),
            asset_index=asset_index,
            raw_name=raw_name,
        )
        count = _parse_count(raw_asset.get("count"), asset_index=asset_index)
        base_name = normalize_asset_name(raw_name)
        name = base_name
        suffix = 2
        while name in seen_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        seen_names.add(name)
        assets.append(
            SceneIntakeAsset(
                id=f"interact_{name}",
                name=name,
                count=count,
                description=description,
                class_candidate=class_candidate,
            )
        )
    return assets


def _parse_class_candidate(
    raw_class_candidate: Any,
    *,
    asset_index: int | str,
    raw_name: str,
) -> list[str]:
    if not isinstance(raw_class_candidate, list):
        raise ValueError(
            f"Scene intake asset {asset_index}.class_candidate must be a list."
        )
    class_candidate = [
        normalize_asset_name(str(item))
        for item in raw_class_candidate
        if normalize_asset_name(str(item))
    ]
    expected_name = normalize_asset_name(raw_name)
    normalized_candidates = [expected_name]
    for candidate in class_candidate:
        if candidate != expected_name and candidate not in normalized_candidates:
            normalized_candidates.append(candidate)
    generic_fallbacks = [
        "object",
        "item",
        "container",
        "tableware",
        "household_object",
    ]
    for fallback in generic_fallbacks:
        if len(normalized_candidates) >= 5:
            break
        if fallback != expected_name and fallback not in normalized_candidates:
            normalized_candidates.append(fallback)
    if len(normalized_candidates) != 5:
        raise ValueError(
            f"Scene intake asset {asset_index}.class_candidate must contain exactly five entries."
        )
    if any(not candidate for candidate in normalized_candidates):
        raise ValueError(
            f"Scene intake asset {asset_index}.class_candidate has empty entries."
        )
    return normalized_candidates


def _parse_count(raw_count: Any, *, asset_index: int) -> int:
    if not isinstance(raw_count, int) or isinstance(raw_count, bool):
        raise ValueError(f"Scene intake asset {asset_index}.count must be an integer.")
    if raw_count < 1:
        raise ValueError(f"Scene intake asset {asset_index}.count must be >= 1.")
    return raw_count


def _validate_exact_keys(
    value: dict[str, Any],
    *,
    allowed_keys: set[str],
    context: str,
) -> None:
    extra_keys = sorted(set(value) - allowed_keys)
    if extra_keys:
        log_warning(
            f"{context} has unexpected keys: {extra_keys}. "
            f"These fields will be ignored."
        )


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list.")
    return value
