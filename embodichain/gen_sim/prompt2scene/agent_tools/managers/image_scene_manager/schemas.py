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

from typing import Any

__all__ = [
    "IMAGE_METRIC_SCALE_JSON_SCHEMA",
    "UP_DOWN_FLIP_CHECK_JSON_SCHEMA",
]

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
