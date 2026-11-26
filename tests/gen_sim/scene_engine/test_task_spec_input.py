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

import json
from pathlib import Path
from typing import Any

import pytest

from embodichain.gen_sim.scene_engine.task_spec_input import (
    TaskEntity,
    TaskSpecInput,
    load_task_spec_input,
)


def _write_task_spec(tmp_path: Path, references: list[dict[str, Any]]) -> Path:
    path = tmp_path / "task_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "task_spec/v1",
                "inputs": {"instruction": "Move the referenced objects."},
                "interpretation": {"references": references},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_projects_four_references_to_three_entities(tmp_path: Path) -> None:
    brown_block = {
        "id": "evidence_002",
        "description": "Brown wood-grain block",
        "is_articulated": False,
        "bbox": "ignored by the adapter",
    }
    path = _write_task_spec(
        tmp_path,
        [
            {"description": "The block on the left", "entities": [brown_block]},
            {
                "entities": [
                    {
                        "id": "evidence_003",
                        "description": "Purple block",
                        "is_articulated": True,
                    }
                ]
            },
            {"entities": [brown_block]},
            {
                "description": "Pale yellow block",
                "entities": [{"id": "evidence_001", "is_articulated": False}],
            },
        ],
    )
    original_json = path.read_bytes()

    assert load_task_spec_input(str(path)) == TaskSpecInput(
        instruction="Move the referenced objects.",
        entities=(
            TaskEntity(
                "evidence_002",
                "Brown wood-grain block",
                False,
                ("The block on the left",),
            ),
            TaskEntity("evidence_003", "Purple block", True),
            TaskEntity(
                "evidence_001", "Pale yellow block", False, ("Pale yellow block",)
            ),
        ),
    )
    assert path.read_bytes() == original_json


def test_combines_distinct_descriptions_for_one_entity(tmp_path: Path) -> None:
    path = _write_task_spec(
        tmp_path,
        [
            {
                "description": reference,
                "entities": [
                    {
                        "id": "drawer_01",
                        "description": description,
                        "is_articulated": True,
                    }
                ],
            }
            for description, reference in (
                ("Brown cabinet", "Open the left cabinet"),
                ("Cabinet with a silver handle", "Pull its handle"),
                ("Brown cabinet", "Open the left cabinet"),
            )
        ],
    )

    assert load_task_spec_input(path).entities == (
        TaskEntity(
            "drawer_01",
            "Brown cabinet\nCabinet with a silver handle",
            True,
            ("Open the left cabinet", "Pull its handle"),
        ),
    )


def test_accepts_empty_references(tmp_path: Path) -> None:
    assert load_task_spec_input(_write_task_spec(tmp_path, [])).entities == ()


@pytest.mark.parametrize("flag", [None, 0, "true"])
def test_rejects_missing_or_non_boolean_articulation_flag(
    tmp_path: Path, flag: object
) -> None:
    entity = {"id": "evidence_001", "description": "Closed drawer"}
    if flag is not None:
        entity["is_articulated"] = flag
    path = _write_task_spec(tmp_path, [{"entities": [entity]}])

    with pytest.raises(ValueError, match="is_articulated must be an explicit boolean"):
        load_task_spec_input(path)


def test_rejects_conflicting_articulation_flags(tmp_path: Path) -> None:
    path = _write_task_spec(
        tmp_path,
        [
            {
                "entities": [
                    {
                        "id": "evidence_001",
                        "description": "Drawer",
                        "is_articulated": flag,
                    }
                ]
            }
            for flag in (False, True)
        ],
    )

    with pytest.raises(ValueError, match="Conflicting is_articulated"):
        load_task_spec_input(path)


@pytest.mark.parametrize("entity_id", ["", "../block", "nested/block", "a\\b", "table"])
def test_rejects_unsafe_or_reserved_entity_ids(tmp_path: Path, entity_id: str) -> None:
    path = _write_task_spec(
        tmp_path,
        [
            {
                "entities": [
                    {
                        "id": entity_id,
                        "description": "Block",
                        "is_articulated": False,
                    }
                ]
            }
        ],
    )

    with pytest.raises(ValueError, match="safe nonempty asset identifier"):
        load_task_spec_input(path)


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": "task_spec/v2", "interpretation": {"references": []}},
        {
            "schema_version": "task_spec/v1",
            "inputs": {"instruction": "Move the cup."},
            "interpretation": {"references": {}},
        },
        {
            "schema_version": "task_spec/v1",
            "inputs": {"instruction": "Move the cup."},
            "interpretation": {"references": [{"entities": "invalid"}]},
        },
    ],
)
def test_rejects_invalid_consumed_document_structure(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    path = tmp_path / "task_spec.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        load_task_spec_input(path)


@pytest.mark.parametrize(
    "inputs", [{}, {"instruction": None}, {"instruction": ""}, {"instruction": " "}]
)
def test_accepts_missing_or_blank_task_instruction(
    tmp_path: Path, inputs: dict[str, Any]
) -> None:
    path = tmp_path / "task_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "task_spec/v1",
                "inputs": inputs,
                "interpretation": {
                    "references": [
                        {
                            "description": "The cup on the left",
                            "entities": [
                                {
                                    "id": "cup_01",
                                    "description": "Blue cup",
                                    "is_articulated": False,
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_task_spec_input(path) == TaskSpecInput(
        instruction="",
        entities=(TaskEntity("cup_01", "Blue cup", False, ("The cup on the left",)),),
    )


@pytest.mark.parametrize("inputs", [None, {"instruction": 1}])
def test_rejects_invalid_instruction_structure(tmp_path: Path, inputs: object) -> None:
    path = tmp_path / "task_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "task_spec/v1",
                "inputs": inputs,
                "interpretation": {"references": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="TaskSpec.inputs"):
        load_task_spec_input(path)


def _write_scene_objects_task_spec(
    tmp_path: Path, objects: list[dict[str, Any]]
) -> Path:
    path = _write_task_spec(tmp_path, [])
    document = json.loads(path.read_text())
    document["configuration"] = {
        "requirements": [
            {"consumer": "scene", "key": "scene_objects", "value": objects}
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_scene_objects_requirement_loads_official_entities_without_references(
    tmp_path: Path,
) -> None:
    path = _write_scene_objects_task_spec(
        tmp_path,
        [
            {
                "id": "table",
                "description": "The scene support table",
                "asset_kind": "rigid_object",
                "is_articulated": False,
            },
            {
                "id": "cabinet_01",
                "description": "Brown cabinet",
                "asset_kind": "articulation",
                "is_articulated": True,
                "provenance": {"source": "task-image"},
                "context": "Open its drawer.",
            },
            {
                "id": "block_01",
                "description": "Purple block",
                "asset_kind": "rigid_object",
                "is_articulated": False,
            },
            {
                "id": "cabinet_01",
                "description": "Cabinet with a silver handle",
                "asset_kind": "articulation",
                "is_articulated": True,
            },
        ],
    )
    original = path.read_bytes()

    assert load_task_spec_input(path) == TaskSpecInput(
        instruction="Move the referenced objects.",
        entities=(
            TaskEntity(
                "cabinet_01", "Brown cabinet\nCabinet with a silver handle", True
            ),
            TaskEntity("block_01", "Purple block", False),
        ),
    )
    assert path.read_bytes() == original


def test_scene_objects_projects_named_parts_to_their_outermost_owner(
    tmp_path: Path,
) -> None:
    path = _write_scene_objects_task_spec(
        tmp_path,
        [
            {"id": "cabinet", "description": "Brown cabinet", "is_articulated": True},
            {"id": "drawer", "description": "Front drawer", "is_articulated": False},
        ],
    )
    document = json.loads(path.read_text())
    document["configuration"].update(
        roles=[
            {"id": "drawer", "properties": {"part_of": "cabinet"}},
            {"id": "handle", "properties": {"part_of": "drawer"}},
        ],
        descriptions={"handle": "Silver drawer handle"},
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    assert load_task_spec_input(path).entities == (
        TaskEntity(
            "cabinet",
            "Brown cabinet",
            True,
            ("Front drawer", "Silver drawer handle"),
            ("drawer", "handle"),
        ),
    )


@pytest.mark.parametrize("empty", [False, True])
def test_scene_objects_requirement_is_authoritative_over_legacy_references(
    tmp_path: Path, empty: bool
) -> None:
    objects = (
        []
        if empty
        else [
            {
                "id": "cabinet_01",
                "description": "Current rigid cabinet",
                "asset_kind": "articulation",
                "is_articulated": False,
            }
        ]
    )
    path = _write_scene_objects_task_spec(tmp_path, objects)
    document = json.loads(path.read_text())
    document["interpretation"]["references"] = [
        {
            "description": "A superseded role reference",
            "entities": [
                {
                    "id": "cabinet_01",
                    "description": "Legacy cabinet",
                    "is_articulated": True,
                },
                {
                    "id": "legacy_only",
                    "description": "Legacy object",
                    "is_articulated": True,
                },
            ],
        }
    ]
    path.write_text(json.dumps(document), encoding="utf-8")

    assert load_task_spec_input(path).entities == (
        () if empty else (TaskEntity("cabinet_01", "Current rigid cabinet", False),)
    )


@pytest.mark.parametrize("flag", [None, 0, "false"])
def test_scene_objects_requirement_requires_explicit_boolean_flag(
    tmp_path: Path, flag: object
) -> None:
    entity = {
        "id": "cabinet_01",
        "description": "Brown cabinet",
        "asset_kind": "articulation",
    }
    if flag is not None:
        entity["is_articulated"] = flag
    path = _write_scene_objects_task_spec(tmp_path, [entity])

    with pytest.raises(ValueError, match="is_articulated"):
        load_task_spec_input(path)


def test_rejects_duplicate_scene_objects_requirements(tmp_path: Path) -> None:
    path = _write_scene_objects_task_spec(tmp_path, [])
    document = json.loads(path.read_text())
    requirements = document["configuration"]["requirements"]
    requirements.append(dict(requirements[0]))
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="scene_objects"):
        load_task_spec_input(path)
