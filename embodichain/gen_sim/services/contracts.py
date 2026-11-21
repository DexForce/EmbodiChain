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

"""Provider-independent synchronous model capability contracts.

Adapters must support concurrent calls on one instance. Artifact paths refer to
caller-owned local output directories; poses use metres, Y-up and xyzw.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Mapping, Protocol

__all__: list[str] = []


@dataclass(frozen=True)
class TextPart:
    """One text part in an ordered multimodal message."""

    text: str


@dataclass(frozen=True)
class ImagePart:
    """One local image or existing data URL; labels are separate text parts."""

    source: Path | str


@dataclass(frozen=True)
class ChatMessage:
    """One system, user or assistant message with ordered content."""

    role: str
    content: tuple[TextPart | ImagePart, ...]


@dataclass(frozen=True)
class ChatRequest:
    """A text/multimodal call with optional structured output guidance."""

    messages: tuple[ChatMessage, ...]
    schema: Mapping[str, Any] | None = None
    model: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class ChatResponse:
    """Model output; domain validation remains with the consuming engine."""

    text: str
    structured: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Generate an image at a caller-selected output path."""

    prompt: str
    output_path: Path


@dataclass(frozen=True)
class ImageArtifact:
    """Generated image artifact."""

    path: Path


@dataclass(frozen=True)
class SegmentationRequest:
    """Segment one concept in a local image."""

    image_path: Path
    prompt: str


@dataclass(frozen=True)
class SegmentationResult:
    """RLE masks, with width/height of the source image."""

    masks: tuple[Mapping[str, Any], ...]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class GeometryRequest:
    """Reconstruct meshes and their coarse scene layout from named masks."""

    image_path: Path
    object_masks: tuple[tuple[str, Path], ...]
    output_root: Path


@dataclass(frozen=True)
class ReconstructedObject:
    """One mesh and pose in the editable scene's Y-up metre frame."""

    object_id: str
    mesh_path: Path
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            not self.object_id
            or len(self.position) != 3
            or len(self.quaternion_xyzw) != 4
            or len(self.scale) != 3
        ):
            raise ValueError("Reconstruction requires an object ID and full pose/scale")
        if not all(
            math.isfinite(value)
            for value in (*self.position, *self.quaternion_xyzw, *self.scale)
        ) or any(value <= 0 for value in self.scale):
            raise ValueError("Reconstruction pose must be finite and scale positive")
        if not math.isclose(
            sum(value * value for value in self.quaternion_xyzw),
            1.0,
            rel_tol=1e-3,
            abs_tol=1e-3,
        ):
            raise ValueError("Reconstruction quaternion must be unit xyzw")


@dataclass(frozen=True)
class GeometryResult:
    """Ordered reconstruction results and optional raw-response audit file."""

    objects: tuple[ReconstructedObject, ...]
    audit_path: Path | None = None


@dataclass(frozen=True)
class ArticulationRequest:
    """Generate a validated movable asset at a local USDC path."""

    prompt: str
    output_path: Path
    image_path: Path | None = None


@dataclass(frozen=True)
class ArticulationResult:
    """Validated USDC and optional request evidence sidecar."""

    path: Path
    evidence_path: Path | None = None


class ChatService(Protocol):
    """Synchronous, concurrent-call-safe text and vision service."""

    def complete(self, request: ChatRequest) -> ChatResponse: ...


class ImageGenerationService(Protocol):
    """Text-to-image capability."""

    def generate(self, request: ImageGenerationRequest) -> ImageArtifact: ...


class SegmentationService(Protocol):
    """Text-prompted segmentation capability."""

    def segment(self, request: SegmentationRequest) -> SegmentationResult: ...


class GeometryReconstructionService(Protocol):
    """Mesh reconstruction including coarse layout; mesh-only is insufficient."""

    def reconstruct(self, request: GeometryRequest) -> GeometryResult: ...


class ArticulationService(Protocol):
    """Articulated asset generation capability."""

    def generate(self, request: ArticulationRequest) -> ArticulationResult: ...


class ArticulationJobService(Protocol):
    """Optional asynchronous UI control capability.

    Job mappings preserve the existing request_id, status, progress, error and
    result.artifacts management contract. Inference pipelines use the separate
    synchronous ArticulationService and do not depend on server job fields.
    """

    def submit(
        self, prompt: str, *, image: str | Path | None = None
    ) -> Mapping[str, Any]: ...
    def status(self, request_id: str) -> Mapping[str, Any]: ...
    def cancel(self, request_id: str) -> Mapping[str, Any]: ...
    def download(
        self, request_id: str, artifact: str, destination: str | Path
    ) -> Path: ...
