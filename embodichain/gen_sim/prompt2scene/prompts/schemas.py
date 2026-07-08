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

"""JSON schemas for LLM structured-output calls across all workflows."""

from __future__ import annotations

from typing import Any

from embodichain.gen_sim.prompt2scene.agent_tools.tools.spatial_relations import (
    GRID_VALUE_LIST,
    RELATION_VALUE_LIST,
)

__all__ = [
    "FILTER_EXTRA_INSTANCES_JSON_SCHEMA",
    "IMAGE_METRIC_SCALE_JSON_SCHEMA",
    "SCENE_INTAKE_JSON_SCHEMA",
    "SCENE_EDIT_INTENT_JSON_SCHEMA",
    "SPATIAL_LAYOUT_VERIFIER_JSON_SCHEMA",
    "SPATIAL_LAYOUT_JSON_SCHEMA",
    "TEXT_RELATIONS_JSON_SCHEMA",
    "UP_DOWN_FLIP_CHECK_JSON_SCHEMA",
]


SCENE_INTAKE_JSON_SCHEMA: dict[str, Any] = {
    "title": "SceneIntakeModelOutput",
    "description": (
        "Objects and table information extracted from a text or image input."
    ),
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "table": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Canonical English class name for the visible table "
                        "or tabletop target, such as table, desk, dining_table, "
                        "coffee_table, workbench, or tabletop."
                    ),
                },
                "description": {
                    "type": "string",
                    "minLength": 20,
                    "maxLength": 180,
                    "description": (
                        "One concise standalone appearance description of the "
                        "visible table or tabletop region."
                    ),
                },
                "complete_table_description": {
                    "type": "string",
                    "minLength": 20,
                    "maxLength": 220,
                    "description": (
                        "One concise standalone description of a complete table "
                        "asset for text-to-3D generation, matching the visible "
                        "tabletop color, material, and texture."
                    ),
                },
                "is_complete_visible_table": {
                    "type": "boolean",
                    "description": (
                        "For image input, whether a mostly complete table is "
                        "visible and suitable as the final table geometry source. "
                        "For text input, this should be false."
                    ),
                },
                "class_candidate": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "description": (
                        "Exactly five likely class names for segmenting the "
                        "visible table or tabletop target."
                    ),
                    "items": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "object_coverage_percent": {
                    "type": "integer",
                    "enum": [10, 30, 50, 70],
                    "description": (
                        "For image input with a complete visible table ONLY: "
                        "choose the closest coverage bucket for objects on the "
                        "tabletop: 10 (mostly empty, a few small objects), "
                        "30 (lightly cluttered), 50 (moderately cluttered), "
                        "70 (densely packed). Omit this field entirely for "
                        "text input or when is_complete_visible_table is false."
                    ),
                },
            },
            "required": [
                "name",
                "description",
                "complete_table_description",
                "is_complete_visible_table",
                "class_candidate",
            ],
        },
        "assets": {
            "type": "array",
            "description": (
                "Object category groups on or intended for the tabletop scene."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Canonical English object name, singular, "
                            "snake_case preferred."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 180,
                        "description": (
                            "One concise appearance description of the object for "
                            "image and 3D geometry generation."
                        ),
                    },
                    "class_candidate": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "description": (
                            "Exactly five likely object class names for later "
                            "image detection or segmentation."
                        ),
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            "Number of repeated instances in this object category "
                            "group. Only group objects that can share the same name, "
                            "description, and class_candidate list."
                        ),
                        "minimum": 1,
                    },
                },
                "required": ["name", "description", "class_candidate", "count"],
            },
        },
    },
    "required": ["table", "assets"],
}


FILTER_EXTRA_INSTANCES_JSON_SCHEMA: dict[str, Any] = {
    "title": "FilterExtraImageInstancesOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extra_instance_numbers": {
            "type": "array",
            "description": "1-based mask numbers that should be removed.",
            "items": {"type": "integer", "minimum": 1},
        },
        "reason": {
            "type": "string",
            "description": "Brief reason for the removal decision.",
        },
    },
    "required": ["extra_instance_numbers", "reason"],
}

SPATIAL_LAYOUT_JSON_SCHEMA: dict[str, Any] = {
    "title": "ImageSpatialLayoutOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "anchor": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "asset_id": {"type": "string", "minLength": 1},
                "grid": {
                    "type": "string",
                    "enum": GRID_VALUE_LIST,
                },
                "reason": {"type": "string"},
            },
            "required": ["asset_id", "grid", "reason"],
        },
        "x_order": {
            "type": "array",
            "description": "Asset-id groups ordered from left to right.",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "minItems": 1,
        },
        "y_order": {
            "type": "array",
            "description": "Asset-id groups ordered from front to back.",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "minItems": 1,
        },
        "asset_states": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "asset_id": {"type": "string", "minLength": 1},
                    "is_arbitrary_layout": {"type": "boolean"},
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional short explanation for debugging.",
                    },
                },
                "required": [
                    "asset_id",
                    "is_arbitrary_layout",
                    "reason",
                ],
            },
        },
    },
    "required": ["anchor", "x_order", "y_order", "asset_states"],
}

SPATIAL_LAYOUT_VERIFIER_JSON_SCHEMA: dict[str, Any] = {
    "title": "ImageSpatialLayoutVerifierOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "passed": {
            "type": "boolean",
            "description": "Whether the draft spatial layout is correct.",
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": "Concise verification reason.",
        },
        "corrected_layout": SPATIAL_LAYOUT_JSON_SCHEMA,
    },
    "required": ["passed", "reason", "corrected_layout"],
}


TEXT_RELATIONS_JSON_SCHEMA: dict[str, Any] = {
    "title": "TextRelationsOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "relation": {
                        "type": "string",
                        "enum": RELATION_VALUE_LIST,
                    },
                    "object": {"type": "string", "minLength": 1},
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["subject", "relation", "object", "evidence"],
            },
        },
        "table_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "asset": {"type": "string", "minLength": 1},
                    "grid": {
                        "type": "string",
                        "enum": GRID_VALUE_LIST,
                    },
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["asset", "grid", "evidence"],
            },
        },
        "object_layouts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "asset": {"type": "string", "minLength": 1},
                    "is_arbitrary_layout": {"type": "boolean"},
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional explanation for this unresolved item.",
                    },
                },
                "required": ["asset", "is_arbitrary_layout", "reason"],
            },
        },
    },
    "required": ["object_relations", "table_constraints", "object_layouts"],
}


UP_DOWN_FLIP_CHECK_JSON_SCHEMA: dict[str, Any] = {
    "title": "AlignedUpDownFlipCheckOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_number": {"type": "integer", "enum": [1, 2]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "required": ["selected_number", "confidence", "reason"],
}

IMAGE_METRIC_SCALE_JSON_SCHEMA: dict[str, Any] = {
    "title": "ImageMetricScaleEstimate",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object_scales": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "object_id": {"type": "string"},
                    "bbox_dims_cm": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "number",
                            "minimum": 1.0e-6,
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["object_id", "bbox_dims_cm", "confidence", "reason"],
            },
        },
    },
    "required": ["object_scales"],
}

SCENE_EDIT_INTENT_JSON_SCHEMA: dict[str, Any] = {
    "title": "SceneEditIntentOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "deleted_object_ids": {
            "type": "array",
            "description": (
                "Existing scene object ids that should be removed. This includes "
                "objects removed by delete operations and objects replaced by new "
                "generated objects. Move operations must not appear here."
            ),
            "items": {"type": "string", "minLength": 1},
        },
        "generated_objects": {
            "type": "array",
            "description": (
                "New objects that must be generated by the text-to-geometry "
                "simready pipeline for add or replace operations."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "temp_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Temporary id used by this edit plan, using "
                            "interact_<canonical_name>_<index>, such as "
                            "interact_red_mug_0. Do not use a new_ prefix. "
                            "It must not collide with existing ids."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Canonical English snake_case object name for "
                            "text-to-geometry."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 220,
                        "description": (
                            "Standalone appearance description used for "
                            "text-to-geometry simready generation."
                        ),
                    },
                    "source_operation": {
                        "type": "string",
                        "enum": ["add", "replace"],
                    },
                },
                "required": [
                    "temp_id",
                    "name",
                    "description",
                    "source_operation",
                ],
            },
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                            "enum": ["delete", "replace", "add", "move"],
                    },
                    "target_object_id": {
                        "type": "string",
                        "description": (
                            "Existing object id for delete/replace/move, or empty "
                            "string for pure add."
                        ),
                    },
                    "new_object_temp_id": {
                        "type": "string",
                        "description": (
                            "Generated object temp_id for add/replace using the "
                            "interact_ prefix, or empty string for delete/move."
                        ),
                    },
                    "placement": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "preserve_target",
                                    "random",
                                    "relative_to_object",
                                    "grid",
                                ],
                            },
                            "reference_object_id": {
                                "type": "string",
                                "description": (
                                    "Existing object id used as a spatial "
                                    "reference, or empty string if unused."
                                ),
                            },
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "",
                                    "left_of",
                                    "right_of",
                                    "front_of",
                                    "back_of",
                                ],
                            },
                            "grid": {
                                "type": "string",
                                "enum": [""] + GRID_VALUE_LIST,
                            },
                        },
                        "required": [
                            "type",
                            "reference_object_id",
                            "relation",
                            "grid",
                        ],
                    },
                    "reason": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "type",
                    "target_object_id",
                    "new_object_temp_id",
                    "placement",
                    "reason",
                    "confidence",
                ],
            },
        },
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "candidate_object_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["query", "reason", "candidate_object_ids"],
            },
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": "Brief overall explanation of the edit interpretation.",
        },
    },
    "required": [
        "deleted_object_ids",
        "generated_objects",
        "operations",
        "unresolved",
        "reason",
    ],
}

SCENE_PROMPT_ROUTE_JSON_SCHEMA: dict[str, Any] = {
    "title": "ScenePromptRouteOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "route": {
            "type": "string",
            "enum": ["scene_edit", "scene_randomization"],
            "description": (
                "Workflow route. Use scene_edit for adding, deleting, replacing, "
                "or generating objects. Use scene_randomization for moving "
                "existing objects toward left/right/front/back directions."
            ),
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": "Concise reason for selecting this route.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Routing confidence from 0.0 to 1.0.",
        },
    },
    "required": ["route", "reason", "confidence"],
}

SCENE_RANDOMIZATION_INTENT_JSON_SCHEMA: dict[str, Any] = {
    "title": "SceneRandomizationIntentOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "operations": {
            "type": "array",
            "description": "Existing-object directional movement operations.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_object_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Exact id of the existing object to move.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "front", "back"],
                    },
                    "reason": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "target_object_id",
                    "direction",
                    "reason",
                    "confidence",
                ],
            },
        },
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "candidate_object_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["query", "reason", "candidate_object_ids"],
            },
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["operations", "unresolved", "reason"],
}
