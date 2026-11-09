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

"""Existing HTTP services projected into provider-independent results."""

from __future__ import annotations
import json
import requests
from functools import wraps
from ..errors import (
    ServiceError,
    ServiceTimeout,
    ServiceTransportError,
    ServiceResponseError,
)
from pathlib import Path
from typing import Any
from collections.abc import Callable
from ..contracts import (
    ImageGenerationRequest,
    ImageArtifact,
    SegmentationRequest,
    SegmentationResult,
    GeometryRequest,
    GeometryResult,
    ReconstructedObject,
    ArticulationRequest,
    ArticulationResult,
)
from .image_generation import ImageGenerationClient
from .image_segmentation import ImageSegmentationClient
from .geometry_generation import GeometryGenerationClient
from .articulated_generation import ArticulatedGenerationClient

__all__: list[str] = []


def _classify(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            with args[0]._call_lock:
                return function(*args, **kwargs)
        except ServiceError:
            raise
        except requests.Timeout:
            raise ServiceTimeout("Model service timed out") from None
        except requests.RequestException:
            raise ServiceTransportError("Model service transport failed") from None
        except (RuntimeError, TypeError, KeyError):
            raise ServiceResponseError("Model service response is invalid") from None

    return wrapped


def geometry_result(client: Any, request: GeometryRequest) -> GeometryResult:
    raw, objects = client.generate_objects(
        image_path=request.image_path,
        object_masks=list(request.object_masks),
        output_root=request.output_root,
    )
    audit = request.output_root / "geometry_generation_response.json"
    audit.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    return GeometryResult(
        tuple(
            ReconstructedObject(
                object_id=uid,
                mesh_path=request.output_root / f"{uid}.glb",
                position=tuple(obj["translation"]),
                quaternion_xyzw=tuple(
                    obj["rotation_quaternion_wxyz"][1:]
                    + obj["rotation_quaternion_wxyz"][:1]
                ),
                scale=tuple(obj["scale"]),
            )
            for (uid, _), obj in zip(request.object_masks, objects)
        ),
        audit,
    )


class ImageAdapter(ImageGenerationClient):
    """Existing prompt-to-PNG HTTP adapter."""

    @_classify
    def generate(self, request: ImageGenerationRequest) -> ImageArtifact:
        return ImageArtifact(
            self.generate_image_by_prompt(
                prompt=request.prompt, output_path=request.output_path
            )
        )


class SegmentationAdapter(ImageSegmentationClient):
    """Existing prompt-to-RLE HTTP adapter."""

    @_classify
    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        from PIL import Image

        with Image.open(request.image_path) as image:
            size = image.size
        return SegmentationResult(
            tuple(
                self.segment_single_object(
                    image_path=request.image_path, prompt=request.prompt
                )
            ),
            size,
        )


class GeometryAdapter(GeometryGenerationClient):
    """Existing multi-mask reconstruction with canonical xyzw poses."""

    @_classify
    def reconstruct(self, request: GeometryRequest) -> GeometryResult:
        return geometry_result(self, request)


class ArticulationAdapter(ArticulatedGenerationClient):
    """Existing asynchronous generation and validated USDC delivery."""

    @_classify
    def generate(self, request: ArticulationRequest) -> ArticulationResult:
        path = self.generate_articulated_usdc(
            prompt=request.prompt,
            image_path=request.image_path,
            output_path=request.output_path,
        )
        return ArticulationResult(path, path.with_suffix(".request.json"))
