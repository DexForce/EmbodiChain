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
from .geometry_generation import GeometryGenerationClient, _pose_convention
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
    """Normalize a legacy response while preserving its unmodified audit JSON.

    The client's ``generate_objects`` returns the raw document and the objects
    projected by ``_parse_objects_response``. This also supports offline replay
    through that same parser without a network transport.
    """
    raw, objects = client.generate_objects(
        image_path=request.image_path,
        object_masks=list(request.object_masks),
        output_root=request.output_root,
    )
    audit = request.output_root / "geometry_generation_response.json"
    audit.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    result_scope = raw.get("result", {})
    if not isinstance(result_scope, dict):
        raise ServiceResponseError("Geometry response result must be an object.")
    response_objects = result_scope.get("objects", [])
    if not isinstance(response_objects, list):
        raise ServiceResponseError("Geometry response objects must be a list.")
    normalized_objects = []
    for index, ((uid, _), obj) in enumerate(zip(request.object_masks, objects)):
        object_scope = response_objects[index] if index < len(response_objects) else obj
        if not isinstance(object_scope, dict):
            raise ServiceResponseError("Geometry response object must be an object.")
        convention = getattr(client, "pose_convention", "glb_column")
        source = getattr(client, "pose_convention_source", "default")
        coordinate_system = None
        for scope, source_name in (
            (object_scope, "response_object"),
            (result_scope, "response_result"),
        ):
            if "pose_convention" in scope:
                convention = scope["pose_convention"]
                source = source_name
                break
        try:
            convention = _pose_convention(convention)
        except ValueError as exc:
            raise ServiceResponseError(str(exc)) from exc
        for scope in (object_scope, result_scope):
            if "pose_coordinate_system" in scope:
                coordinate_system = scope["pose_coordinate_system"]
                if coordinate_system != "glb_y_up":
                    raise ServiceResponseError(
                        "Geometry generation pose_coordinate_system must be 'glb_y_up'."
                    )
                break
        source_quaternion = tuple(obj["rotation_quaternion_wxyz"])
        w, x, y, z = source_quaternion
        if convention == "sam3d_row_conjugated":
            # Conjugating the unit quaternion transposes only its 3x3 rotation.
            # Translation and scale already use the gateway's GLB coordinates.
            x, y, z = -x, -y, -z
        normalized_objects.append(
            ReconstructedObject(
                object_id=uid,
                mesh_path=request.output_root / f"{uid}.glb",
                position=tuple(obj["translation"]),
                quaternion_xyzw=(x, y, z, w),
                scale=tuple(obj["scale"]),
                source_quaternion_wxyz=source_quaternion,
                pose_coordinate_system=coordinate_system,
                pose_convention=convention,
                pose_convention_source=source,
            )
        )
    return GeometryResult(
        tuple(normalized_objects),
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
