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

"""Replaceable GenSim model capabilities, with run-scoped ownership.

Importing this package does not load configuration or provider dependencies.
"""

from __future__ import annotations
from importlib import import_module
from typing import Any

__all__ = [
    "GenSimServicesCfg",
    "load_services_config",
    "GenSimServices",
    "ServiceRegistry",
    "TextPart",
    "ImagePart",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ImageGenerationRequest",
    "ImageArtifact",
    "SegmentationRequest",
    "SegmentationResult",
    "GeometryRequest",
    "ReconstructedObject",
    "GeometryResult",
    "ArticulationRequest",
    "ArticulationResult",
    "ChatService",
    "ImageGenerationService",
    "SegmentationService",
    "GeometryReconstructionService",
    "ArticulationService",
    "ArticulationJobService",
    "ServiceError",
    "ServiceConfigurationError",
    "CapabilityError",
    "ServiceTimeout",
    "ServiceTransportError",
    "ServiceResponseError",
    "SubmissionUnknown",
]
_EXPORTS = {
    "GenSimServicesCfg": "config",
    "load_services_config": "config",
    "GenSimServices": "runtime",
    "ServiceRegistry": "runtime",
    "TextPart": "contracts",
    "ImagePart": "contracts",
    "ChatMessage": "contracts",
    "ChatRequest": "contracts",
    "ChatResponse": "contracts",
    "ImageGenerationRequest": "contracts",
    "ImageArtifact": "contracts",
    "SegmentationRequest": "contracts",
    "SegmentationResult": "contracts",
    "GeometryRequest": "contracts",
    "ReconstructedObject": "contracts",
    "GeometryResult": "contracts",
    "ArticulationRequest": "contracts",
    "ArticulationResult": "contracts",
    "ChatService": "contracts",
    "ImageGenerationService": "contracts",
    "SegmentationService": "contracts",
    "GeometryReconstructionService": "contracts",
    "ArticulationService": "contracts",
    "ArticulationJobService": "contracts",
    "ServiceError": "errors",
    "ServiceConfigurationError": "errors",
    "CapabilityError": "errors",
    "ServiceTimeout": "errors",
    "ServiceTransportError": "errors",
    "ServiceResponseError": "errors",
    "SubmissionUnknown": "errors",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
