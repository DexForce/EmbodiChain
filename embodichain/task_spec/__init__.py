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

"""Pure TaskSpec contracts, canonical task identity and evidence references."""

from __future__ import annotations

from .canonicalization import canonical_role_map, canonical_template, semantic_hash
from .contracts import (
    ACTION_WITNESS_SCHEMA,
    EXPANSION_MANIFEST_SCHEMA,
    SCENE_INSTANCE_SCHEMA,
    TASK_TEMPLATE_SCHEMA,
    VALIDATION_CERTIFICATE_SCHEMA,
    ActionWitness,
    ExpansionManifest,
    SceneInstance,
    TaskTemplate,
    ValidationCertificate,
    certificate_passed,
    scene_instance_hash,
    validate_action_witness,
    validate_expansion_manifest,
    validate_scene_instance,
    validate_task_template,
    validate_validation_certificate,
)
from .expressions import validate_expression
from .registry import registry_snapshot
from .validation import validate_template_structure

__all__ = [
    "ACTION_WITNESS_SCHEMA",
    "EXPANSION_MANIFEST_SCHEMA",
    "SCENE_INSTANCE_SCHEMA",
    "TASK_TEMPLATE_SCHEMA",
    "VALIDATION_CERTIFICATE_SCHEMA",
    "ActionWitness",
    "ExpansionManifest",
    "SceneInstance",
    "TaskTemplate",
    "ValidationCertificate",
    "certificate_passed",
    "scene_instance_hash",
    "validate_action_witness",
    "validate_expansion_manifest",
    "validate_scene_instance",
    "validate_task_template",
    "validate_validation_certificate",
    "canonical_template",
    "canonical_role_map",
    "semantic_hash",
    "validate_expression",
    "registry_snapshot",
    "validate_template_structure",
]
